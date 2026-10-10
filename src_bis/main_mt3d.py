"""Train the multitask 3D model (ROI segmentation + 8 ROI findings) on joint crops, run from src_bis/.

    torchrun --standalone --nproc_per_node=4 main_mt3d.py --crops u10_d32_s256

See `mt3d/model.py` and `mt3d/train.py`. Splits as everywhere else: validation =
the 58 radiologist-labelled studies + the classifier's holdout.
"""

import argparse
import json
import os
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

import torch  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from knee_cls.data import load_tables, make_splits  # noqa: E402
from main_crops import training_targets  # noqa: E402
from mt3d.model import MT3D  # noqa: E402
from mt3d.train import MTDataset, fit, load_crops  # noqa: E402
from params import LOG_FOLDER  # noqa: E402
from roi.extract import ROI_CLASSES  # noqa: E402
from training.distributed import (broadcast_object, cleanup_distributed, init_distributed,  # noqa: E402
                                  seed_everything, worker_init_fn)
from utils.logger import create_logger, prepare_log_folder  # noqa: E402

DEFAULTS = {
    "crops": "u10_d32_s256", "size": 224, "labels": "pl:pl_round2", "planes": "Sagittal,Coronal",
    "head": "masked", "n_dec": 3, "dim": 256, "tf_layers": 2, "dropout": 0.2, "queries": 4,
    "pretrained": True, "detach_masks": False,
    "w_cls": 0.1, "w_seg": 1.0, "w_meta": 0.05,
    "epochs": 15, "batch_size": 8, "lr_unet": 2e-4, "lr": 1e-3, "weight_decay": 0.01, "ema": 0.999,
    "num_workers": 8, "seed": 0, "log_every": 100, "fullfit": False,
    "augmentations": {"mirror_p": 0.0, "affine_p": 0.8, "rot_deg": 12.0, "scale_range": (0.85, 1.15),
                      "shift": 0.08, "intensity_p": 0.8, "gamma": 0.3, "contrast": 0.25, "brightness": 0.1,
                      "noise_std": 0.02},
    "log_path": LOG_FOLDER + "bis/mt3d/",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        if isinstance(v, dict):
            p.add_argument(f"--{k}", type=json.loads, default=None)
        elif isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()
    aug = dict(DEFAULTS["augmentations"])
    aug.update(a.augmentations or {})
    a.augmentations = aug
    return a


def main():
    args = parse_args()
    ctx = init_distributed()
    seed_everything(args.seed, ctx.rank)
    out = None
    if ctx.is_main:
        os.makedirs(args.log_path, exist_ok=True)
        out = prepare_log_folder(args.log_path)
    args.output_dir = broadcast_object(out, device=ctx.device)
    if ctx.is_main:
        create_logger(directory=args.output_dir, name="logs.txt")
        json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=4, default=str)

    arr, boxes, info = load_crops(args.crops)
    info = info[info.Anatomical_Plane.isin(args.planes.split(","))]
    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    val_mask = info.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids))
    targets = training_targets(args.labels, studies, ROI_CLASSES)
    if args.fullfit:
        st_ = studies.set_index("StudyInstanceUID")
        for c in ROI_CLASSES:
            t = targets[c].reindex(st_.index)
            targets[c] = t.where(st_[f"gt_{c}"].isna(), st_[f"gt_{c}"])
        train_ds = MTDataset(arr, boxes, info, targets)
    else:
        train_ds = MTDataset(arr, boxes, info[~val_mask], targets)
    val_ds = MTDataset(arr, boxes, info[val_mask], training_targets("claude", studies, ROI_CLASSES))
    if ctx.is_main:
        print(f"-> {args.output_dir}\n- train {len(train_ds)} series  val {len(val_ds)} series")

    def dl(ds, train):
        s = DistributedSampler(ds, shuffle=train, drop_last=train) if ctx.distributed else None
        return DataLoader(ds, batch_size=args.batch_size, sampler=s, shuffle=train and s is None, drop_last=train,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                          worker_init_fn=worker_init_fn), s

    train_dl, sampler = dl(train_ds, True)
    val_dl, _ = dl(val_ds, False)
    model = MT3D(n_dec=args.n_dec, head=args.head, dim=args.dim, tf_layers=args.tf_layers, dropout=args.dropout,
                 queries=args.queries, pretrained=args.pretrained, detach_masks=args.detach_masks).to(ctx.device)
    if ctx.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ctx.device.index],
                                                          find_unused_parameters=args.head != "masked")
    try:
        tab = fit(model, train_dl, val_dl, val_ds, studies, args, ctx, sampler)
        if ctx.is_main:
            print(f"FINAL gt {tab['gt'].mean():.4f} hold {tab['hold'].mean():.4f}\n{tab.round(4).to_string()}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
