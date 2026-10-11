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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from knee_cls.data import load_tables, make_splits  # noqa: E402
from main_crops import training_targets  # noqa: E402
from mt3d.model import MT3D  # noqa: E402
from mt3d.anatomy import ANATOMY  # noqa: E402
from mt3d.train import MTDataset, PairDataset, concat_sources, fit, load_anatomy, load_crops  # noqa: E402
from params import CLASSES, LOG_FOLDER, OUTPUT_FOLDER  # noqa: E402
from roi.extract import ROI_CLASSES  # noqa: E402
from training.distributed import (broadcast_object, cleanup_distributed, init_distributed,  # noqa: E402
                                  seed_everything, worker_init_fn)
from utils.logger import create_logger, prepare_log_folder  # noqa: E402

DEFAULTS = {
    "crops": "u10_d32_s256", "size": 224, "labels": "pl:pl_round2", "planes": "Sagittal,Coronal",
    "arch": "unet", "backbone": "convnext_tiny.fb_in22k", "frames": 16, "dec": 192, "drop_path": 0.1, "pool": "pred",
    "levels": "1,2,3", "anat_tokens": False, "zpos": False, "nb": 1,
    "findings": "roi8", "labels_extra": "mean", "zoom": "", "zoom_size": 128, "zoom_frames": 8,
    "pair": False, "slots": 2, "mixup": 0.0,
    # OAI pretraining (`--pretrain oai_u10_d32_s256`): train on the OAI joint crops, with the ROI findings
    # from MOAKS (output/oai/labels_moaks00.csv) and the 25 MOAKS gradings, each tied to its ROI
    # (aux_moaks00.csv, weight w_aux); validation stays on our studies (a transfer test). Fine-tuning:
    # `--init <run dir>/ckpt_final.pt` (heads that do not match are left at their initialisation).
    "pretrain": "", "w_aux": 1.0, "init": "",
    # OAI finding labels for pretraining: the graded convention labels (external_data.md §2, `oai.labels --graded`).
    "pretrain_labels": "../output/oai/labels_moaks00_v4.csv",
    # Report embedding as an auxiliary target of the study token (`mt3d.reports`; pair / set mode).
    "w_rep": 0.0, "rep_dim": 64,
    # Joint training with external data (`external_data.md`): extra joint-crop sets (comma-separated, built by
    # mt3d.prepare), their graded labels (one csv per set, StudyInstanceUID + findings, NaN = unknown), their
    # loss weight; MRNet meniscal tears without a side through MIL (`--w_mil`, labels from `--mil_csv`).
    "extra_crops": "", "extra_labels": "", "extra_weight": "1.0", "w_mil": 0.0,
    "mil_csv": "../output/mrnet/labels.csv",
    "head": "masked", "n_dec": 3, "dim": 256, "tf_layers": 2, "dropout": 0.2, "queries": 4,
    "pretrained": True, "detach_masks": False,
    "w_cls": 1.0, "w_seg": 1.0, "w_meta": 0.05, "w_anat": 0.0,
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
    # roi8: the 8 ROI findings; all12: + effusion, synovitis, contusion, fracture (no ROI; `labels_extra`,
    # since the pseudo-labels cover the ROI findings only), scored as extra tasks of the same model.
    classes = list(ROI_CLASSES) + ([c for c in CLASSES if c not in ROI_CLASSES] if args.findings == "all12" else [])
    targets = training_targets(args.labels, studies, ROI_CLASSES)
    if len(classes) > len(ROI_CLASSES):
        targets |= training_targets(args.labels_extra, studies, classes[len(ROI_CLASSES):])
    anatomy = load_anatomy(args.crops) if args.w_anat > 0 else None
    if args.fullfit:
        st_ = studies.set_index("StudyInstanceUID")
        for c in classes:
            t = targets[c].reindex(st_.index)
            targets[c] = t.where(st_[f"gt_{c}"].isna(), st_[f"gt_{c}"])
        train_ds = MTDataset(arr, boxes, info, targets, classes, anatomy=anatomy)
    else:
        train_ds = MTDataset(arr, boxes, info[~val_mask], targets, classes, anatomy=anatomy)
    val_ds = MTDataset(arr, boxes, info[val_mask], training_targets("claude", studies, classes), classes)
    if args.extra_crops:
        tr_info = info[~val_mask]
        sources = [("main", arr, boxes, tr_info)]
        ext_targets = {c: pd.Series(dtype=float) for c in classes}
        names = args.extra_crops.split(",")
        ws = [float(w) for w in str(args.extra_weight).split(",")]
        ws = ws * len(names) if len(ws) == 1 else ws              # one weight for all, or one per crop set
        for name, lab_csv, w in zip(names, args.extra_labels.split(","), ws):
            e_arr, e_boxes, e_info = load_crops(name)
            lab = pd.read_csv(lab_csv).set_index("StudyInstanceUID")
            e_info = e_info.assign(w=w)
            if args.w_mil > 0 and os.path.exists(args.mil_csv):
                mil = pd.read_csv(args.mil_csv).set_index("StudyInstanceUID").mrnet_meniscus
                e_info = e_info.assign(y_men=e_info.StudyInstanceUID.map(mil))
            sources.append((name, e_arr, e_boxes, e_info))
            for c in classes:
                ext_targets[c] = pd.concat([ext_targets[c], lab[c]]) if c in lab else ext_targets[c]
        all_targets = {c: pd.concat([targets[c], ext_targets[c]]) for c in classes}
        a_arr, a_boxes, a_info = concat_sources(sources)
        train_ds = MTDataset(a_arr, a_boxes, a_info, all_targets, classes, anatomy=anatomy)
        if ctx.is_main:
            print(f"- + external: {', '.join(f'{n} {len(i)} series' for n, _, _, i in sources[1:])}")
    if args.w_rep > 0 and not args.pretrain:
        from mt3d.reports import report_targets
        train_ds.y_rep = report_targets(set(train_ds.info.StudyInstanceUID), args.rep_dim).reindex(
            train_ds.info.StudyInstanceUID).to_numpy(np.float32)
    aux_roi = []
    if args.pretrain:
        oai = os.path.join(OUTPUT_FOLDER, "oai")
        o_arr, o_boxes, o_info = load_crops(args.pretrain)
        lab = pd.read_csv(args.pretrain_labels).set_index("StudyInstanceUID")
        aux = pd.read_csv(os.path.join(oai, "aux_moaks00.csv")).set_index("StudyInstanceUID")
        aux_roi = [ROI_CLASSES.index(c.split("|")[0]) for c in aux.columns]
        train_ds = MTDataset(o_arr, o_boxes, o_info, {c: lab[c] for c in classes}, classes, aux=aux)
    if args.pair:
        # Cross-view study samples (`mt3d.hybrid.PairHybrid`); the batch size counts studies (2 series each).
        train_ds, val_ds = PairDataset(train_ds, True, args.slots), PairDataset(val_ds, False, args.slots)
    if ctx.is_main:
        print(f"-> {args.output_dir}\n- train {len(train_ds)} series  val {len(val_ds)} series")

    def dl(ds, train):
        s = DistributedSampler(ds, shuffle=train, drop_last=train) if ctx.distributed else None
        return DataLoader(ds, batch_size=args.batch_size, sampler=s, shuffle=train and s is None, drop_last=train,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                          worker_init_fn=worker_init_fn), s

    train_dl, sampler = dl(train_ds, True)
    val_dl, _ = dl(val_ds, False)
    n_anat = len(ANATOMY) if args.w_anat > 0 else 0
    if args.arch == "hybrid":
        from mt3d.evaluate import zoom_rois
        from mt3d.hybrid import Hybrid
        model = Hybrid(args.backbone, frames=args.frames, dec=args.dec, dim=args.dim, tf_layers=args.tf_layers,
                       dropout=args.dropout, drop_path=args.drop_path, n_anat=n_anat, pool=args.pool,
                       n_cls=len(classes), levels=[int(v) for v in args.levels.split(",")],
                       anat_tokens=args.anat_tokens, zpos=args.zpos, nb=args.nb,
                       zoom=zoom_rois(args.zoom), zoom_size=args.zoom_size, zoom_frames=args.zoom_frames,
                       aux_roi=aux_roi, rep_dim=args.rep_dim if args.w_rep > 0 else 0).to(ctx.device)
    else:
        model = MT3D(n_dec=args.n_dec, head=args.head, dim=args.dim, tf_layers=args.tf_layers, dropout=args.dropout,
                     queries=args.queries, pretrained=args.pretrained, detach_masks=args.detach_masks,
                     n_anat=n_anat).to(ctx.device)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        sd = {k[len("base."):] if k.startswith("base.") else k: v for k, v in sd.items()}
        own = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        if ctx.is_main:
            print(f"-> init from {args.init}: {len(keep)}/{len(own)} tensors "
                  f"(skipped: {sorted(set(sd) - set(keep))[:8]})")
    if args.pair:
        from mt3d.hybrid import PairHybrid
        model = PairHybrid(model, args.slots).to(ctx.device)
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
