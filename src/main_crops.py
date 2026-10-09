"""Train a 2.5D classifier on ROI crops for one group of findings, run from src/.

    torchrun --standalone --nproc_per_node=4 main_crops.py --group menisci --backbone convnext_tiny.fb_in22k

See `crops/train.py`. Groups: crops.train.GROUPS. Splits as everywhere else:
validation = the 58 radiologist-labelled studies + the classifier's holdout.
"""

import argparse
import json
import os
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from crops.train import GROUPS, CropDataset, Crop25D, fit, load_crops  # noqa: E402
from knee_cls.data import load_tables, make_splits  # noqa: E402
from params import INPUT_FOLDER, LOG_FOLDER, OUTPUT_FOLDER  # noqa: E402
from training.distributed import (broadcast_object, cleanup_distributed, init_distributed,  # noqa: E402
                                  seed_everything, worker_init_fn)
from utils.logger import create_logger, prepare_log_folder  # noqa: E402

DEFAULTS = {
    "group": "menisci", "crops": "r3", "backbone": "convnext_tiny.fb_in22k", "img_size": 0,
    "head": "lstm", "hidden": 256, "dropout": 0.2, "drop_path": 0.1,
    "epochs": 10, "batch_size": 32, "lr": 5e-4, "lr_backbone": 1e-4, "weight_decay": 0.05,
    "planes": "Sagittal,Coronal", "num_workers": 8, "seed": 0,
    "labels": "claude",
    "fullfit": False,
    "augmentations": {"rot_deg": 15.0, "scale_range": (0.85, 1.15), "shift": 0.08,
                      "gamma": 0.3, "contrast": 0.25, "brightness": 0.1},
    "log_path": LOG_FOLDER + "crops_run/",
}


def training_targets(which, studies, classes):
    """Soft training labels per finding: Claude report labels, Codex ones, their mean, or pseudo-labels (`pl:<name>`).

    Codex covers every non-radiologist study (input/codex-labels_final_4349.csv)
    and 52 of the 58 radiologist ones; those are validation-only, so training is
    unaffected. Validation keeps scoring against the Claude labels (`hold`) so
    that runs stay comparable.
    """
    claude = studies.set_index("StudyInstanceUID")
    if which.startswith("pl:"):
        # Pseudo-labels: output/crops/<name>.csv, StudyInstanceUID x findings (crops/predict.py + teacher averaging).
        pl = pd.read_csv(os.path.join(OUTPUT_FOLDER, "crops", f"{which[3:]}.csv")).set_index("StudyInstanceUID")
        return {c: pl[c] for c in classes}
    if which == "claude":
        return {c: claude[f"target_{c}"] for c in classes}
    codex = pd.concat([pd.read_csv(f"{INPUT_FOLDER}codex-labels_final_{n}.csv") for n in (52, 4349)])
    codex = codex.set_index("StudyInstanceUID")
    if which == "codex":
        return {c: codex[c] for c in classes}
    assert which == "mean", which
    return {c: (claude[f"target_{c}"] + codex[c].reindex(claude.index)) / 2 for c in classes}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        # Through-plane sampling of the crop set (crops.prepare), which inference must reproduce.
        from crops.prepare import CROP_DIR, DEPTH
        meta = os.path.join(CROP_DIR, args.crops.split(",")[0] + ".json")
        args.z_sampling = json.load(open(meta)) if os.path.exists(meta) else \
            dict(depth=DEPTH, z_mode="uniform", stride=1)
        json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=4, default=str)

    crops, info = load_crops(args.crops)
    info = info[info.Anatomical_Plane.isin(args.planes.split(","))]
    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    val_mask = info.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids))
    classes = GROUPS[args.group]
    targets = training_targets(args.labels, studies, classes)
    if args.fullfit:
        # Fullfit (submission models): every study is trained on. The 58 radiologist studies
        # take their radiologist labels; every other study keeps the chosen soft labels.
        st_ = studies.set_index("StudyInstanceUID")
        for c in classes:
            t = targets[c].reindex(st_.index)
            gt = st_[f"gt_{c}"]
            targets[c] = t.where(gt.isna(), gt)
        train_ds = CropDataset(crops, info, classes, targets)
    else:
        train_ds = CropDataset(crops, info[~val_mask], classes, targets)
    # Validation is scored from the study tables, not these targets: Claude ones keep all 58 studies in.
    val_ds = CropDataset(crops, info[val_mask], classes, training_targets("claude", studies, classes))
    if ctx.is_main:
        if args.fullfit:
            print("-> FULLFIT: validation studies are trained on, validation scores below are leaky")
        print(f"-> {args.output_dir}  group {args.group} {classes}  backbone {args.backbone}\n"
              f"- train {len(train_ds)} crops  val {len(val_ds)} crops")

    def dl(ds, train):
        s = DistributedSampler(ds, shuffle=train, drop_last=train) if ctx.distributed else None
        return DataLoader(ds, batch_size=args.batch_size, sampler=s, shuffle=train and s is None, drop_last=train,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                          worker_init_fn=worker_init_fn), s

    train_dl, sampler = dl(train_ds, True)
    val_dl, _ = dl(val_ds, False)
    model = Crop25D(args.backbone, len(classes), img_size=args.img_size or None, hidden=args.hidden,
                    dropout=args.dropout, drop_path=args.drop_path, head=args.head).to(ctx.device)
    if ctx.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ctx.device.index],
                                                          find_unused_parameters=len(classes) == 1)
    try:
        tab = fit(model, train_dl, val_dl, val_ds, studies, args, ctx, sampler)
        if ctx.is_main:
            print(f"FINAL gt {tab["gt"].mean():.4f} hold {tab["hold"].mean():.4f}\n{tab.round(4).to_string()}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
