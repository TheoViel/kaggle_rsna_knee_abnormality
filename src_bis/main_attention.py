"""Export the per-class attention maps of a trained run, run from src/.

    python main_attention.py --run_dir ../logs/cls_run/2026-10-08/1/ --studies gt
    python main_attention.py --run_dir ... --studies all --dense false

One .npz per series under `<out_dir>/<StudyInstanceUID>/<SeriesInstanceUID>.npz`:

    slices     (F,)          original slice index (in output/series_npy) of each frame
    xy_attn    (K, F, h, w)  softmax over (h, w): where class k is looked for in that frame
                             (mean over heads)
    xy_heads   (K, H, F, h, w) the same per attention head (multi-head runs only)
    z_attn     (K, F)        softmax over frames: which frames class k reads
    z_logits   (K, F)        the same before the softmax, comparable across series
    attn       (K, F, h, w)  z_attn * xy_attn, the 3D attention, sums to 1 per class
    series_attn (K,)         weight of this series in the study-level decision
    probs      (K,)          study-level predicted probabilities
    rows, cols               original in-plane size

Geometry: maps cover the original slice zero-padded (centred) to a square of side
max(rows, cols), at h = w = img_size / attn_stride. `to_original` maps one back
onto the original (rows, cols) grid.

With `--dense` (default) a frame is centred on every slice of the series inside
the same trimmed range as training (the outer `z_trim` slices at each end, never
seen in training, are left out); frame positions are relative, so the model
handles the longer sequences, but the z softmax is then spread over more frames
than in training -- z_logits are the more stable quantity for thresholding.
Probabilities come from a separate pass with the training frame sampling.
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from knee_cls.data import (KneeStudyDataset, collate_studies, gpu_augment,
                           load_tables, make_splits, to_device)
from knee_cls.main import build_model
from knee_cls.train import normalize
from params import CACHE_MAX_SLICES, CLASSES


def to_original(maps, rows, cols):
    """(..., h, w) map over the padded square -> (..., rows, cols)."""
    s = max(rows, cols)
    t = torch.as_tensor(maps, dtype=torch.float32)
    lead = t.shape[:-2]
    t = F.interpolate(t.reshape(-1, 1, *t.shape[-2:]), size=(s, s), mode="bilinear",
                      align_corners=False)
    y0, x0 = (s - rows) // 2, (s - cols) // 2
    return t[:, 0, y0:y0 + rows, x0:x0 + cols].reshape(*lead, rows, cols).numpy()


def cache_to_original_slices(n_orig):
    if n_orig <= CACHE_MAX_SLICES:
        return np.arange(n_orig)
    return np.round(np.linspace(0, n_orig - 1, CACHE_MAX_SLICES)).astype(int)


def load_run(run_dir, device):
    with open(os.path.join(run_dir, "args.json")) as f:
        args = SimpleNamespace(**json.load(f))
    ctx = SimpleNamespace(is_main=True, distributed=False, device=device)
    model = build_model(args, ctx, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "ckpt_final.pt"),
                                     map_location=device))
    return model.eval(), args


class StudyAttention:
    """Run a trained model on single studies and return its attention per series.

    Also what `notebooks/04_attention_maps.ipynb` uses.
    """

    def __init__(self, run_dir, device="cuda"):
        self.device = torch.device(device)
        self.model, self.args = load_run(run_dir, self.device)
        self.studies, self.series = load_tables()
        self.shapes = self.series.set_index("SeriesInstanceUID")[["n_slices", "rows", "cols"]]

    def _dataset(self, ids, dense):
        args = self.args
        return KneeStudyDataset(
            self.studies, self.series, ids, train=False,
            max_series=int(args.val_max_series),
            n_frames=None if dense else int(args.val_n_frames),
            frame_offset=int(args.frame_offset), z_trim=float(args.z_trim))

    @torch.no_grad()
    def _run(self, batch):
        batch = to_device(batch, self.device)
        batch["images"] = normalize(gpu_augment(batch, int(self.args.img_size), aug=None))
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            return batch, self.model(batch, return_attn=True)

    def predict(self, batch):
        """Study probabilities, from a batch built with the training frame sampling."""
        _, out = self._run(batch)
        return torch.sigmoid(out["logits"].float()).cpu().numpy()

    def series_records(self, batch, out, probs):
        """-> list of dicts (the .npz fields, see module doc), one per series of study 0."""
        records = []
        xy = out["xy_attn"].float()
        for s, series_uid in enumerate(batch["series_uids"][0]):
            sel = batch["slice_series"] == s
            n_frames = int(sel.sum())
            xy_s = xy[sel].permute(1, 0, 2, 3)                       # K F h w
            heads = out["xy_attn_heads"][sel].float()                # F K H h w
            z_s = out["z_attn"][s, :n_frames].T                      # K F
            info = self.shapes.loc[series_uid]
            to_orig = cache_to_original_slices(int(info.n_slices))
            rec = dict(
                series_uid=series_uid,
                slices=to_orig[batch["slice_centre"][sel].cpu().numpy()],
                xy_attn=xy_s.cpu().numpy().astype(np.float16),
                z_attn=z_s.cpu().numpy().astype(np.float32),
                z_logits=out["z_logits"][s, :n_frames].T.cpu().numpy().astype(np.float32),
                attn=(xy_s * z_s[..., None, None]).cpu().numpy().astype(np.float32),
                series_attn=out["series_attn"][0, s].float().cpu().numpy(),
                probs=probs, classes=np.array(CLASSES),
                rows=int(info.rows), cols=int(info.cols))
            if heads.shape[2] > 1:
                rec["xy_heads"] = heads.permute(1, 2, 0, 3, 4).cpu().numpy().astype(np.float16)
            records.append(rec)
        return records

    def __call__(self, study_uid, dense=True):
        """-> (probs (K,), list of per-series records) for one study."""
        probs = self.predict(collate_studies([self._dataset([study_uid], False)[0]]))[0]
        batch = collate_studies([self._dataset([study_uid], dense)[0]])
        batch, out = self._run(batch)
        return probs, self.series_records(batch, out, probs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--studies", default="gt", help="gt | holdout | all | comma list")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--dense", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    sa = StudyAttention(a.run_dir)
    args = sa.args
    out_dir = a.out_dir or os.path.join(a.run_dir, "attention")
    os.makedirs(out_dir, exist_ok=True)

    _, gt_ids, hold_ids = make_splits(sa.studies, float(args.holdout_frac), int(args.seed))
    ids = {"gt": gt_ids, "holdout": hold_ids,
           "all": sa.studies.StudyInstanceUID.tolist()}.get(a.studies)
    ids = ids if ids is not None else a.studies.split(",")
    ids = ids[:a.limit] if a.limit else ids

    def loader(dense):
        return DataLoader(sa._dataset(ids, dense), batch_size=1, num_workers=8,
                          collate_fn=collate_studies)

    probs = {}
    for batch in loader(False):
        probs[batch["study_uid"][0]] = sa.predict(batch)[0]
    pd.DataFrame.from_dict(probs, orient="index", columns=CLASSES).rename_axis(
        "StudyInstanceUID").to_csv(os.path.join(out_dir, "preds.csv"))

    for batch in loader(a.dense):
        uid = batch["study_uid"][0]
        batch, out = sa._run(batch)
        os.makedirs(os.path.join(out_dir, uid), exist_ok=True)
        for rec in sa.series_records(batch, out, probs[uid]):
            np.savez_compressed(os.path.join(out_dir, uid, f"{rec.pop('series_uid')}.npz"), **rec)
    print(f"-> {len(ids)} studies to {out_dir}")


if __name__ == "__main__":
    main()
