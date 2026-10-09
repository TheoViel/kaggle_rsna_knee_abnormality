"""Train the ROI U-Net on attention-derived (and later self-refined) box labels, run from src/.

    torchrun --standalone --nproc_per_node=4 main_roi.py --labels round0
    torchrun --standalone --nproc_per_node=4 main_roi.py --predict ../logs/roi_run/<date>/<id>/ --pred_out round0_pred

Training: sagittal and coronal series with at least one labelled class, from the
training studies; validation on the radiologist-labelled studies and the
classifier's holdout (see `roi/train.py` for the metrics).

`--predict <run dir>` instead runs a trained model on every sagittal and coronal
series and writes its boxes -- exactly one per class and series, each with its
confidence (`roi.consensus.add_confidence`) -- to output/roi/labels/<pred_out>.csv, which
`python -m roi.labels merge` turns into the next round's labels.
"""

import argparse
import json
import os
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from params import LOG_FOLDER  # noqa: E402
from roi.data import RoiDataset, load_volumes, split_series  # noqa: E402
from roi.labels import LABEL_DIR  # noqa: E402
from roi.train import build_model, fit, predict  # noqa: E402
from training.distributed import (broadcast_object, cleanup_distributed,  # noqa: E402
                                  init_distributed, seed_everything, worker_init_fn)
from utils.logger import create_logger, prepare_log_folder  # noqa: E402

DEFAULTS = {
    "labels": "round0",
    "epochs": 40.0,
    "batch_size": 8,
    "val_batch_size": 8,
    "num_workers": 6,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "threshold": 0.5,
    "augmentations": {
        "mirror_p": 0.0, "affine_p": 0.8, "rot_deg": 12.0, "scale_range": (0.85, 1.15),
        "shift": 0.08, "intensity_p": 0.8, "gamma": 0.3, "contrast": 0.25,
        "brightness": 0.1, "noise_std": 0.02,
    },
    "log_path": LOG_FOLDER + "roi_run/",
    "log_every": 50,
    "eval_every_epochs": 10,
    "seed": 0,
    "predict": "",
    "pred_out": "",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        if isinstance(v, dict):
            p.add_argument(f"--{k}", type=json.loads, default=None)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()
    aug = dict(DEFAULTS["augmentations"])
    aug.update(a.augmentations or {})
    a.augmentations = aug
    return a


def loader(ds, args, train, ctx):
    sampler = DistributedSampler(ds, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=train,
                                 seed=args.seed, drop_last=train) if ctx.distributed else None
    return DataLoader(ds, batch_size=args.batch_size if train else args.val_batch_size,
                      shuffle=train and sampler is None, sampler=sampler, drop_last=train,
                      num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                      worker_init_fn=worker_init_fn), sampler


def main():
    args = parse_args()
    ctx = init_distributed()
    seed_everything(args.seed, ctx.rank)
    vols, index = load_volumes()

    if args.predict:
        sd = torch.load(os.path.join(args.predict, "ckpt_final.pt"), map_location=ctx.device)
        # Models from before an ROI class was added have fewer outputs: the first
        # classes of ROI_CLASSES, in order.
        n_out = sd["decoder.seg_layers.4.weight"].shape[0]
        model = build_model(num_classes=n_out).to(ctx.device)
        model.load_state_dict(sd)
        dl, _ = loader(RoiDataset(vols, index), args, False, ctx)
        pred = predict(model, dl, index, ctx.device, args.threshold, with_labels=False)
        if ctx.is_main:
            # Per-box confidence: U-Net sharpness x support from the study's other series.
            from roi.consensus import add_confidence

            pred = add_confidence(pred).drop(columns="world")
            out = os.path.join(LABEL_DIR, f"{args.pred_out}.csv")
            pred.to_csv(out, index=False)
            print(f"-> {len(pred)} (series, class) predictions, "
                  f"{int((pred.conf >= args.threshold).sum())} confident, to {out}")
        cleanup_distributed()
        return

    out_dir = None
    if ctx.is_main:
        os.makedirs(args.log_path, exist_ok=True)
        out_dir = prepare_log_folder(args.log_path)
    args.output_dir = broadcast_object(out_dir, device=ctx.device)
    if ctx.is_main:
        create_logger(directory=args.output_dir, name="logs.txt")
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump({k: getattr(args, k) for k in DEFAULTS}, f, indent=4, default=str)
        print(f"-> output {args.output_dir}  world {ctx.world_size}")

    labels = pd.read_csv(os.path.join(LABEL_DIR, f"{args.labels}.csv"))
    tr_index, va_index = split_series(index)
    tr_index = tr_index[tr_index.SeriesInstanceUID.isin(set(labels.SeriesInstanceUID))]
    train_ds, val_ds = RoiDataset(vols, tr_index, labels), RoiDataset(vols, va_index, labels)
    if ctx.is_main:
        print(f"- train: {train_ds.summary}\n- val  : {val_ds.summary}")
    train_dl, sampler = loader(train_ds, args, True, ctx)
    val_dl, _ = loader(val_ds, args, False, ctx)

    model = build_model().to(ctx.device)
    if ctx.distributed:
        # The decoder keeps a segmentation head per stage for deep supervision, unused here.
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ctx.device.index],
                                                          find_unused_parameters=True)
    try:
        fit(model, train_dl, val_dl, index, args, ctx, sampler)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
