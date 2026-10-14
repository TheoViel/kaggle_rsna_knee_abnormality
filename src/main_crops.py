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

from crops.train import GROUPS, CropDataset, PairDataset, build_model, finding_of, fit, load_crops  # noqa: E402
from knee_cls.data import load_tables, make_folds, make_splits  # noqa: E402
from params import INPUT_FOLDER, LOG_FOLDER, OUTPUT_FOLDER  # noqa: E402
from training.distributed import (broadcast_object, cleanup_distributed, init_distributed,  # noqa: E402
                                  seed_everything, worker_init_fn)
from utils.logger import create_logger, prepare_log_folder  # noqa: E402

DEFAULTS = {
    "group": "menisci", "crops": "r3", "backbone": "convnext_tiny.fb_in22k", "img_size": 0,
    "head": "lstm", "fusion": "late", "hidden": 256, "dropout": 0.2, "drop_path": 0.1,
    "epochs": 10, "batch_size": 32, "lr": 5e-4, "lr_backbone": 1e-4, "weight_decay": 0.05,
    "planes": "Sagittal,Coronal", "num_workers": 8, "seed": 0,
    "labels": "claude",
    "fullfit": False,
    # K-fold (out-of-fold pseudo-labels, knee_cls.data.make_folds): validate on fold `fold` of `n_folds` over every
    # training study, train on the others with the fullfit targets (radiologist labels on the 58). 0: off.
    "n_folds": 0, "fold": 0,
    # Regularisation (crops.train.model_input / mix_batch): context crops (ctx > 1) with random
    # window shifts, random slice windows of depth_out slices, mixup / cutmix within a finding.
    # External data (OAI, see src/oai/): extra crop sets with their own labels; loss weight; OAI only
    # (pretraining); initial weights (fine-tuning); OAI sequences to keep.
    "extra_crops": "", "extra_labels": "", "extra_weight": 1.0, "extra_only": False, "init": "", "extra_seqs": "",
    # Multiple-instance meniscus bags (crops.train.PairDataset): series of a crop set whose exam has a tear
    # of unknown side (--mil_labels csv, StudyInstanceUID + --mil_col == 1); loss on max(medial, lateral).
    "mil_crops": "", "mil_labels": "", "mil_col": "mrnet_meniscus", "mil_weight": 1.0, "mil_bs": 4,
    # Auxiliary per-ROI output (crops.train.Crop25D.fc_aux): e.g. --aux fracture --aux_labels frac_roi reads
    # output/crops/frac_roi.csv columns "<ROI>|fracture"; the study score of the finding = max over ROIs.
    "aux": "", "aux_labels": "", "aux_weight": 1.0,
    "ctx": 1.0, "crop_jitter": 0.0, "depth_out": 0, "slice_jitter": 0, "mixup": 0.0, "cutmix": 0.0, "mix_alpha": 0.4,
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
        return {c: pl[c] if c in pl else pd.Series(dtype=float) for c in classes}
    if which == "claude":
        return {c: claude[f"target_{c}"] if f"target_{c}" in claude else pd.Series(dtype=float) for c in classes}
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
    # Through-plane sampling of the crop set (crops.prepare), which inference must reproduce.
    from crops.prepare import CROP_DIR, DEPTH
    meta = os.path.join(CROP_DIR, args.crops.split(",")[0] + ".json")
    args.z_sampling = json.load(open(meta)) if os.path.exists(meta) else dict(depth=DEPTH, z_mode="uniform", stride=1)
    if ctx.is_main:
        create_logger(directory=args.output_dir, name="logs.txt")
        json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=4, default=str)

    crops, info = load_crops(args.crops)
    extra = None
    if args.extra_crops:
        e_crops, extra = load_crops(args.extra_crops)
        crops.update(e_crops)
        extra = extra[extra.Anatomical_Plane.isin(args.planes.split(","))]
        if args.extra_seqs:
            extra = extra[extra.seq.isin(args.extra_seqs.split(","))]
    info = info[info.Anatomical_Plane.isin(args.planes.split(","))]
    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    val_mask = info.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids))
    if args.n_folds:
        assert not args.fullfit, "k-fold and fullfit are exclusive"
        folds = make_folds(studies, args.n_folds)
        val_mask = info.StudyInstanceUID.map(folds).eq(args.fold)
    classes = GROUPS[args.group]
    targets = training_targets(args.labels, studies, classes)
    if extra is not None:
        lab = pd.read_csv(args.extra_labels).set_index("StudyInstanceUID")
        for c in classes:
            targets[c] = pd.concat([targets[c], lab[c]]) if c in lab else targets[c]
    aux_t = None
    if args.aux:
        assert not args.mixup and not args.cutmix and not args.mil_crops, "aux: no mixup / cutmix / MIL"
        al = pd.read_csv(os.path.join(OUTPUT_FOLDER, "crops", f"{args.aux_labels}.csv")).set_index("StudyInstanceUID")
        aux_t = {c: al[f"{c}|{args.aux}"] for c in classes if f"{c}|{args.aux}" in al}
    if args.fullfit or args.n_folds:
        # Fullfit (submission models): every study is trained on. The 58 radiologist studies
        # take their radiologist labels; every other study keeps the chosen soft labels.
        st_ = studies.set_index("StudyInstanceUID")
        # Per-ROI sub-targets of a study finding ("<ROI>|fracture"): the radiologist study label decides on the 58.
        # Negative -> every ROI 0; positive with no positive ROI target (the report gives no usable location) -> NaN.
        subs = {c: finding_of(c.split("|")[1]) for c in classes if "|" in c}
        subs = {c: F for c, F in subs.items() if F is not None and f"gt_{F}" in st_}
        for F in set(subs.values()):
            gt = st_[f"gt_{F}"].dropna()
            cols = [c for c, f in subs.items() if f == F]
            roi_max = pd.concat([targets[c].reindex(gt.index) for c in cols], axis=1).max(axis=1)
            neg, unk = gt.index[gt < 0.5], gt.index[(gt >= 0.5) & ~(roi_max >= 0.5)]
            for c in cols:
                t = targets[c].reindex(targets[c].index.union(gt.index))
                t.loc[neg], t.loc[unk] = 0.0, np.nan
                targets[c] = t
            if ctx.is_main:
                print(f"- fullfit {F}: radiologist labels on the 58 -> {len(neg)} negatives, {len(unk)} positives without a location (NaN)")
        for c in classes:
            if f"gt_{c}" not in st_:          # no radiologist labels (e.g. MOAKS targets): unchanged
                continue
            idx = targets[c].index.union(st_.index)          # keep extra-data (OAI) studies
            t = targets[c].reindex(idx)
            gt = st_[f"gt_{c}"].reindex(idx)
            targets[c] = t.where(gt.isna(), gt)
        train_ds = CropDataset(crops, info if args.fullfit else info[~val_mask], classes, targets, aux_t)
    else:
        train_ds = CropDataset(crops, info[~val_mask], classes, targets, aux_t)
    if extra is not None:
        tr_info = extra if args.extra_only else pd.concat([info if args.fullfit else info[~val_mask], extra], ignore_index=True)
        train_ds = CropDataset(crops, tr_info, classes, targets, aux_t)
        train_ds.items["w"] = np.where(train_ds.items.src.isin(args.extra_crops.split(",")), args.extra_weight, 1.0)
    # Validation is scored from the study tables, not these targets: Claude ones keep all 58 studies in.
    val_t = training_targets("claude", studies, classes)
    for c in classes:                     # "<ROI>|fracture": scored at study level (crops.train.study_auc)
        F = finding_of(c.split("|")[1]) if "|" in c else None
        if F is not None:
            val_t[c] = training_targets("claude", studies, [F])[F]
    val_ds = CropDataset(crops, info[val_mask], classes, val_t)
    if ctx.is_main:
        if args.n_folds:
            print(f"-> K-FOLD {args.fold} / {args.n_folds}: validation = fold {args.fold} of every training study "
                  f"({info[val_mask].StudyInstanceUID.nunique()} studies); the 58 are scored on their radiologist labels")
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
    mil_dl = None
    if args.mil_crops:
        m_crops, m_info = load_crops(args.mil_crops)
        crops.update(m_crops)
        pos = pd.read_csv(args.mil_labels).set_index("StudyInstanceUID")[args.mil_col]
        m_info = m_info[m_info.Anatomical_Plane.isin(args.planes.split(",")) & m_info.StudyInstanceUID.map(pos).eq(1)]
        mil_ds = PairDataset(crops, m_info, classes)
        s = DistributedSampler(mil_ds, shuffle=True, drop_last=True) if ctx.distributed else None
        mil_dl = DataLoader(mil_ds, batch_size=args.mil_bs, sampler=s, shuffle=s is None, drop_last=True,
                            num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
        if ctx.is_main:
            print(f"- MIL bags: {len(mil_ds)} series ({m_info.StudyInstanceUID.nunique()} exams), {args.mil_bs} per step")
    val_dl, _ = dl(val_ds, False)
    model = build_model(vars(args), len(classes), pretrained=True)
    if args.init:
        sd = torch.load(os.path.join(args.init, "ckpt_final.pt"), map_location="cpu")
        own = model.state_dict()
        skipped = [k for k in sd if k in own and sd[k].shape != own[k].shape]   # e.g. the finding embeddings
        sd = {k: v for k, v in sd.items() if k not in skipped}
        res = model.load_state_dict(sd, strict=False)
        if ctx.is_main:
            print(f"init from {args.init}: skipped (shape) {skipped}, missing {res.missing_keys}")
    model = model.to(ctx.device)
    if ctx.distributed:
        # One ROI embedding row, or modules a variant skips (early: the slice pooling; map3d: the encoder head).
        unused = len(classes) == 1 or args.fusion == "early" or args.head == "map3d"
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ctx.device.index],
                                                          find_unused_parameters=unused)
    try:
        tab = fit(model, train_dl, val_dl, val_ds, studies, args, ctx, sampler, mil_dl=mil_dl)
        if ctx.is_main:
            print(f"FINAL gt {tab["gt"].mean():.4f} hold {tab["hold"].mean():.4f}\n{tab.round(4).to_string()}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
