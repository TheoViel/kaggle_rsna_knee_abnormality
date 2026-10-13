"""Run the nnU-Net knee cascade on every sagittal and coronal series, run from src/.

    python segment_nnunet.py                       # all 4 GPUs, all sagittal + coronal series
    python segment_nnunet.py --limit 8 --gpus 0    # quick test

Output: `input/segs_nnunet/<SeriesInstanceUID>.npy`, uint8, same (slices, rows,
cols) shape and slice order as `output/series_npy/<SeriesInstanceUID>.npy`.
Labels: 0 background, 1 patellar cartilage, 2 femoral cartilage, 3 medial tibial
cartilage, 4 lateral tibial cartilage, 5 medial meniscus, 6 lateral meniscus,
7 femur, 8 tibia, 9 patella.

Same model and settings as `notebooks/02_segmentation_nnunet.ipynb`: the
`aagatti/nnunet_knee` ResEnc-M cascade (3d_lowres, then 3d_cascade_fullres fed
with the low-res prediction), fold 1, step size 0.5. Differences:

  - it runs in memory through nnU-Net's Python API instead of NIfTI files and the
    CLI. The array and spacing are what `SimpleITKIO` (the reader in the model's
    plans) would produce from the notebook's NIfTI: the (z, y, x) volume and the
    spacing reversed to (slice step, row spacing, column spacing), with the slice
    step from `knee_registration.series_geometry`. Direction and origin are not
    used by nnU-Net, so they are not needed;
  - no mirroring TTA, as in the notebook's FAST mode;
  - series larger than `--max_inplane` pixels in-plane or `--max_slices` slices are
    downsized first (antialiased, spacing adjusted) and the prediction is
    resized back to the original grid (nearest neighbour). The caps sit at the
    model's own target spacing (0.31 mm in-plane, 0.8 mm along z over typical
    fields of view), so they cost little resolution, and they bound the memory
    of nnU-Net's resampling and export on the 3D isotropic and 1024 px series.

Parallelism: `--workers_per_gpu` worker processes per GPU (default 2, so CPU
preprocessing of one overlaps GPU prediction of the other), each taking every n-th
series (sorted by size, so the shards are balanced) with its own CPU pools for
nnU-Net's preprocessing and export (resampling dominates). Series go through
both stages in chunks of `--chunk`. Finished series are skipped, so the job can
be stopped and restarted; files are written atomically.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from params import INPUT_FOLDER, OUTPUT_FOLDER, SERIES_INDEX_CSV

MODEL_DIR = os.path.abspath("../models/nnunet_knee/models/Dataset500_KneeMRI")
STAGES = {
    "lowres": os.path.join(MODEL_DIR, "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres"),
    "fullres": os.path.join(MODEL_DIR, "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_cascade_fullres"),
}
OUT_DIR = INPUT_FOLDER + "segs_nnunet/"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planes", default="Sagittal,Coronal")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers_per_gpu", type=int, default=1,
                   help="workers sharing a GPU, so one preprocesses on CPU while another predicts")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--chunk", type=int, default=16, help="series per two-stage round")
    p.add_argument("--num_processes", type=int, default=3, help="CPU preprocessing workers per worker")
    p.add_argument("--num_export", type=int, default=4, help="CPU export workers per worker")
    p.add_argument("--step_size", type=float, default=0.5)
    p.add_argument("--max_inplane", type=int, default=512)
    p.add_argument("--max_slices", type=int, default=160)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--series", default=None, help="comma-separated SeriesInstanceUIDs instead of --planes")
    p.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--todo", default=None, help=argparse.SUPPRESS)
    p.add_argument("--n_workers", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def todo_series(args):
    idx = pd.read_csv(SERIES_INDEX_CSV)
    if args.series:
        idx = idx[idx.SeriesInstanceUID.isin(args.series.split(","))]
    else:
        idx = idx[idx.Anatomical_Plane.isin(args.planes.split(","))]
    idx = idx[[not os.path.exists(os.path.join(args.out_dir, f"{s}.npy"))
               for s in idx.SeriesInstanceUID]]
    # Biggest first and dealt round-robin: shards get similar work, and the
    # slowest series do not end up alone at the tail of one GPU.
    idx = idx.assign(size=idx.n_slices * idx.rows * idx.cols).sort_values("size", ascending=False)
    if args.limit:
        idx = idx.head(args.limit)
    return idx.SeriesInstanceUID.tolist()


def load_case(series_uid, max_inplane, max_slices):
    """-> (image (1, z, y, x) float32, nnU-Net properties, original (z, y, x) shape).

    Spacing is (slice step, row spacing, column spacing): what SimpleITKIO
    returns. Oversized series are downsized and their spacing scaled to match.
    """
    import torch
    import torch.nn.functional as F
    from knee_registration import series_geometry

    vol = np.load(os.path.join(OUTPUT_FOLDER, "series_npy", f"{series_uid}.npy")).astype(np.float32)
    sx, sy, sz = series_geometry(series_uid)["spacing"]
    shape = vol.shape
    z, y, x = shape
    f_xy = min(1.0, max_inplane / max(y, x))
    f_z = min(1.0, max_slices / z)
    if f_xy < 1 or f_z < 1:
        new = (max(1, round(z * f_z)), max(1, round(y * f_xy)), max(1, round(x * f_xy)))
        t = torch.from_numpy(vol)[None, None]
        if f_xy < 1:   # antialiased in-plane, slice by slice
            t = F.interpolate(t[0].transpose(0, 1), size=new[1:], mode="bilinear",
                              antialias=True, align_corners=False).transpose(0, 1)[None]
        if f_z < 1:
            t = F.interpolate(t, size=new, mode="trilinear", align_corners=False)
        vol = t[0, 0].numpy()
        sz, sy, sx = sz * z / new[0], sy * y / new[1], sx * x / new[2]
    return vol[None], {"spacing": [sz, sy, sx]}, shape


def restore_shape(seg, shape):
    """Nearest-neighbour resize of a (z, y, x) label map to `shape`."""
    if seg.shape == tuple(shape):
        return seg
    idx = [np.minimum(((np.arange(n) + 0.5) * s / n).astype(int), s - 1)
           for n, s in zip(shape, seg.shape)]
    return seg[np.ix_(*idx)]


def build_predictor(folder, args):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    pred = nnUNetPredictor(tile_step_size=args.step_size, use_gaussian=True,
                           use_mirroring=False, perform_everything_on_device=True,
                           device=torch.device("cuda", 0), verbose=False,
                           verbose_preprocessing=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(folder, use_folds=(1,),
                                              checkpoint_name="checkpoint_final.pth")
    return pred


def predict(lowres, fullres, images, props, args):
    """Both cascade stages on a list of (1, z, y, x) images -> list of (z, y, x) label maps."""
    seg_low = lowres.predict_from_list_of_npy_arrays(
        images, None, [dict(p) for p in props], None,
        num_processes=args.num_processes, num_processes_segmentation_export=args.num_export)
    # Stage 2 reads the low-res result like a file read would: channel axis,
    # signed (cropping marks the outside with -1).
    return fullres.predict_from_list_of_npy_arrays(
        images, [np.asarray(sl, dtype=np.int16)[None] for sl in seg_low],
        [dict(p) for p in props], None,
        num_processes=args.num_processes, num_processes_segmentation_export=args.num_export)


def run_worker(args):
    os.environ.setdefault("nnUNet_compile", "false")
    import torch

    torch.set_num_threads(4)
    with open(args.todo) as f:
        series = f.read().split()[args.worker::args.n_workers]
    tag = f"[gpu {os.environ.get('CUDA_VISIBLE_DEVICES')} w{args.worker}]"
    print(f"{tag} {len(series)} series", flush=True)
    if not series:
        return
    lowres = build_predictor(STAGES["lowres"], args)
    fullres = build_predictor(STAGES["fullres"], args)

    t0, done = time.time(), 0
    for i in range(0, len(series), args.chunk):
        uids = series[i:i + args.chunk]
        cases, ok = [], []
        for u in uids:
            try:
                cases.append(load_case(u, args.max_inplane, args.max_slices))
                ok.append(u)
            except Exception as e:  # noqa: BLE001 -- one bad series must not stop the shard
                print(f"{tag} FAILED load {u}: {e}", flush=True)
        if not ok:
            continue
        images = [c[0] for c in cases]
        props = [c[1] for c in cases]
        segs = None
        # nnU-Net's spawned worker pools occasionally die at start-up under
        # concurrent load ("Background workers died"), independently of the
        # data: retry the chunk, then go one series at a time.
        for attempt in range(2):
            try:
                segs = predict(lowres, fullres, images, props, args)
                break
            except Exception as e:  # noqa: BLE001
                print(f"{tag} chunk {ok[0][-8:]}.. attempt {attempt + 1} failed: {e}", flush=True)
        if segs is None:
            segs = []
            for u, img, pr in zip(ok, images, props):
                try:
                    segs.append(predict(lowres, fullres, [img], [pr], args)[0])
                except Exception as e:  # noqa: BLE001
                    segs.append(None)
                    print(f"{tag} FAILED {u}: {e}", flush=True)
                    with open(os.path.join(args.out_dir, "_failed.txt"), "a") as f:
                        f.write(f"{u}\t{e}\n")
        for u, img, (_, _, shape), seg in zip(ok, images, cases, segs):
            if seg is None:
                continue
            seg = np.asarray(seg).astype(np.uint8)
            assert seg.shape == img.shape[1:], (u, seg.shape, img.shape)
            seg = restore_shape(seg, shape)
            tmp = os.path.join(args.out_dir, f"{u}.tmp.npy")
            np.save(tmp, seg)
            os.replace(tmp, os.path.join(args.out_dir, f"{u}.npy"))
        done += sum(seg is not None for seg in segs)
        rate = (time.time() - t0) / max(done, 1)
        print(f"{tag} {done}/{len(series)}  {rate:.1f} s/series  "
              f"eta {rate * (len(series) - done) / 3600:.1f} h", flush=True)
    print(f"{tag} finished {done} series in {(time.time() - t0) / 3600:.2f} h", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.worker is not None:
        return run_worker(args)

    gpus = args.gpus.split(",") * args.workers_per_gpu
    # The list is fixed once here: workers recomputing it would see each other's
    # outputs appear and shard a different list.
    series = todo_series(args)
    todo = os.path.join(args.out_dir, f"_todo_{os.getpid()}.txt")
    with open(todo, "w") as f:
        f.write("\n".join(series))
    print(f"-> {len(series)} series to segment, {len(gpus)} workers on GPUs {sorted(set(gpus))} "
          f"-> {args.out_dir}", flush=True)
    procs = []
    for w, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, nnUNet_compile="false",
                   nnUNet_results=os.path.dirname(MODEL_DIR),
                   nnUNet_raw="/tmp/nnUNet_raw", nnUNet_preprocessed="/tmp/nnUNet_preprocessed")
        cmd = [sys.executable, __file__, "--worker", str(w), "--n_workers", str(len(gpus)),
               "--todo", todo]
        cmd += [a for a in sys.argv[1:]]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    os.remove(todo)
    print(f"-> done, exit codes {codes}")
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
