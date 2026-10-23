"""Attention maps of the classifier, for the ROI classes, on sagittal and coronal series.

    python -m roi.extract --tag ens --runs <run dir>,<run dir>,... --studies all --gpus 0,1,2,3

Writes output/roi/attention/<tag>/maps/<SeriesInstanceUID>.npz with
    slices  (F,)          original slice index (output/series_npy) of each frame
    xy      (K, F, G, G)  in-plane attention per frame (sums to 1 over G x G),
                          float16, on the slice zero-padded to a centred square
    z       (K, F)        attention over frames (sums to 1)
    rows, cols, n_slices  original series shape
for the K = 8 ROI classes (`ROI_CLASSES`). A frame sits on every slice of the
series except the outer `z_trim` (training's sampling, dense).

Better maps than a single pass of one model:

  - shift TTA (`--shift_tta`, default on). The attention lives on a stride-8 grid
    of the model input (36 x 36 at 288 px). The input is shifted by +-2 px
    (a quarter cell) in y and x, giving 4 maps whose cells are centred a half
    cell apart; interleaved, they form one map on a grid twice as fine, with
    each value an independent read of the attention at that point. Averaging
    over the passes also smooths the z weights.
  - ensembling (`--runs`). Classifier runs with different input sizes, frame
    counts and backbones attend to the same anatomy with different errors.
    Each run's maps are resampled to the common grid G and averaged.

xy and z attention of a series depend on that series only (the z context runs
per series), so axial series are skipped and other series do not matter.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from params import CACHE_MAX_SLICES, CLASSES, OUTPUT_FOLDER

# One ROI per class in every series: the 7 ligament / meniscus / compartment
# findings, and Baker's -- a cyst is by definition in the popliteal fossa.
ROI_CLASSES = CLASSES[:7] + ["Baker's"]
ROI_IDX = [CLASSES.index(c) for c in ROI_CLASSES]
ROI_PLANES = ("Sagittal", "Coronal")
ATTN_DIR = os.path.join(OUTPUT_FOLDER, "roi", "attention")
# Quarter of the stride-8 attention cell, in model-input pixels.
SHIFT = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--runs", required=True, help="comma-separated classifier run folders (main_cls.py)")
    p.add_argument("--shift_tta", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--grid", type=int, default=72)
    p.add_argument("--studies", default="gt", help="gt | all | comma-separated StudyInstanceUIDs")
    p.add_argument("--gpus", default="0")
    p.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--n_workers", type=int, default=1, help=argparse.SUPPRESS)
    return p.parse_args()


def cache_to_original_slices(n_orig):
    """Original slice index of each slice of the cache (`prepare_cache.py` keeps at most CACHE_MAX_SLICES)."""
    if n_orig <= CACHE_MAX_SLICES:
        return np.arange(n_orig)
    return np.round(np.linspace(0, n_orig - 1, CACHE_MAX_SLICES)).astype(int)


def shift(x, dy, dx, fill):
    """Move the image content by (dy, dx) pixels, filling the uncovered border."""
    _, _, H, W = x.shape
    out = torch.full_like(x, fill)
    out[..., max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        x[..., max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return out


class Extractor:
    def __init__(self, run_dirs, shift_tta, grid, device="cuda"):
        from knee_cls.main import load_run

        self.device = torch.device(device)
        self.models = [load_run(d, self.device) for d in run_dirs]
        self.shift_tta = shift_tta
        self.grid = grid

    @torch.no_grad()
    def _maps(self, model, args, batch):
        """(xy (N, K, G, G), z (S, Z, K)) of one model on one collated study."""
        from knee_cls.data import gpu_augment
        from knee_cls.train import normalize

        images = normalize(gpu_augment(batch, int(args.img_size), aug=None))
        fill = float(normalize(torch.zeros(())))
        shifts = [(dy, dx) for dy in (SHIFT, -SHIFT) for dx in (SHIFT, -SHIFT)] \
            if self.shift_tta else [(0, 0)]
        xy, z = None, 0
        for dy, dx in shifts:
            batch["images"] = shift(images, dy, dx, fill) if dy or dx else images
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                out = model(batch, return_attn=True)
            a = out["xy_attn"][:, ROI_IDX].float()                    # N K h w
            if xy is None:
                n, k, h, w = a.shape
                xy = a.new_zeros(n, k, 2 * h, 2 * w) if self.shift_tta else a.new_zeros(n, k, h, w)
            if self.shift_tta:
                # Content moved by +2 px puts cell j's centre at j + 1/4 cell of the
                # original image (fine index 2j), -2 px at j + 3/4 (fine index 2j + 1).
                xy[..., int(dy < 0)::2, int(dx < 0)::2] = a
            else:
                xy = a
            z = z + out["z_attn"][..., ROI_IDX].float()
        xy = F.interpolate(xy, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        xy = xy.clamp_min(0)
        xy = xy / xy.flatten(2).sum(-1).clamp_min(1e-12)[..., None, None]
        return xy, z / len(shifts)

    def study(self, studies, series, uid):
        """-> list of per-series dicts (see module doc) for one study."""
        from knee_cls.data import KneeStudyDataset, collate_studies, to_device

        sel = series[(series.StudyInstanceUID == uid) & series.Anatomical_Plane.isin(ROI_PLANES)]
        if sel.empty:
            return []
        xy_sum, z_sum, ref = 0, 0, None
        for model, args in self.models:
            ds = KneeStudyDataset(studies, sel, [uid], train=False, max_series=64, n_frames=None,
                                  frame_offset=int(args.frame_offset), z_trim=float(args.z_trim))
            batch = to_device(collate_studies([ds[0]]), self.device)
            if ref is None:
                ref = batch
            # Every run must put its frames on the same slices to be averaged.
            assert torch.equal(batch["slice_centre"], ref["slice_centre"]), "frame mismatch between runs"
            xy, z = self._maps(model, args, batch)
            xy_sum, z_sum = xy_sum + xy, z_sum + z
        xy, z = xy_sum / len(self.models), z_sum / len(self.models)

        shapes = series.set_index("SeriesInstanceUID")
        records = []
        for s, series_uid in enumerate(ref["series_uids"][0]):
            m = ref["slice_series"] == s
            n = int(m.sum())
            info = shapes.loc[series_uid]
            to_orig = cache_to_original_slices(int(info.n_slices))
            records.append(dict(
                series_uid=series_uid,
                slices=to_orig[ref["slice_centre"][m].cpu().numpy()],
                xy=xy[m].permute(1, 0, 2, 3).cpu().numpy().astype(np.float16),
                z=z[s, :n].T.cpu().numpy().astype(np.float32),
                rows=int(info.rows), cols=int(info.cols), n_slices=int(info.n_slices)))
        return records


def study_ids(which, studies):
    if which == "gt":
        return studies.loc[studies.is_gt, "StudyInstanceUID"].tolist()
    if which == "all":
        return studies.StudyInstanceUID.tolist()
    return which.split(",")


def run_worker(args):
    from knee_cls.data import load_tables

    studies, series = load_tables()
    out = os.path.join(ATTN_DIR, args.tag, "maps")
    os.makedirs(out, exist_ok=True)
    todo = study_ids(args.studies, studies)[args.worker::args.n_workers]
    roi_series = series[series.Anatomical_Plane.isin(ROI_PLANES)].groupby("StudyInstanceUID").SeriesInstanceUID.apply(list)
    ex = Extractor(args.runs.split(","), args.shift_tta, args.grid)
    t0 = time.time()
    for i, uid in enumerate(todo):
        if all(os.path.exists(os.path.join(out, f"{s}.npz")) for s in roi_series.get(uid, [])):
            continue
        for r in ex.study(studies, series, uid):
            tmp = os.path.join(out, f"{r['series_uid']}.tmp.npz")
            np.savez(tmp, **{k: v for k, v in r.items() if k != "series_uid"})
            os.replace(tmp, os.path.join(out, f"{r['series_uid']}.npz"))
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            rate = (time.time() - t0) / (i + 1)
            print(f"[w{args.worker}] {i + 1}/{len(todo)}  {rate:.2f} s/study  "
                  f"eta {rate * (len(todo) - i - 1) / 60:.0f} min", flush=True)


def main():
    args = parse_args()
    if args.worker is not None:
        return run_worker(args)
    gpus = args.gpus.split(",")
    procs = []
    for w, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        cmd = [sys.executable, "-m", "roi.extract", *sys.argv[1:], "--worker", str(w),
               "--n_workers", str(len(gpus))]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    print(f"-> {os.path.join(ATTN_DIR, args.tag)}  exit codes {codes}")
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
