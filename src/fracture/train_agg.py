"""Fracture from the 8 ROI crops of a series, with a learned study-level aggregation head, run from src/.

    torchrun --standalone --nproc_per_node=4 -m fracture.train_agg [--init ../logs/crops_run/2026-10-12/0] [--seed 0]

The per-ROI fracture model (main_crops --group fracture_roi, study score = max over ROIs) scores 0.968 on the 58.
Here one training item is a whole series: its 8 ROI crops go through the same Crop25D (one embedding per crop, ROI
and plane embeddings added), and
  - per-ROI logits (the model's own final layer) learn the per-ROI targets (fracture.roi_labels, v1: frac_roi.csv),
    as before (NaN = no target, masked);
  - an aggregation head, concat(max-pool, attention-pool) of the 8 crop embeddings -> linear, learns a study label:
    max(max of the series' ROI targets, mean of the Claude and Codex fracture labels).
Validation (the 58 + holdout): study score = mean over series of the head's probability ("agg"), compared with the
max over ROIs of the per-ROI probabilities ("max") and their rank average ("agg+max").
--init: start from a fracture_roi run's checkpoint (encoder, embeddings, per-ROI layer); the head starts fresh.
Writes the usual run folder (args.json, logs.txt, ckpt_final.pt, val_study_preds.csv with Fracture_agg / _max).
"""

import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset, DistributedSampler  # noqa: E402

from crops.train import PLANES, build_model, load_crops, model_input  # noqa: E402
from knee_cls.data import load_tables, make_splits  # noqa: E402
from params import CLAUDE_LABELS_CSV, INPUT_FOLDER, LOG_FOLDER, OUTPUT_FOLDER  # noqa: E402
from roi.extract import ROI_CLASSES  # noqa: E402
from training.distributed import (all_gather_lists, all_reduce_mean, broadcast_object, cleanup_distributed,  # noqa: E402
                                  init_distributed, seed_everything, unwrap_model, worker_init_fn)
from utils.logger import create_logger, prepare_log_folder  # noqa: E402

DEFAULTS = dict(crops="f_s1n8", planes="Sagittal,Coronal", roi_labels="frac_roi", backbone="convnext_tiny.fb_in22k",
                img_size=0, head="lstm", hidden=256, dropout=0.2, drop_path=0.1, epochs=10, batch_size=4, lr=5e-4,
                lr_backbone=1e-4, weight_decay=0.05, study_weight=1.0, num_workers=8, seed=0, init="", fullfit=False,
                log_path=LOG_FOLDER + "crops_run/")
AUG = {"rot_deg": 15.0, "scale_range": (0.85, 1.15), "shift": 0.08, "gamma": 0.3, "contrast": 0.25, "brightness": 0.1}


class SeriesDataset(Dataset):
    """One item per series: its 8 ROI crops, the 8 per-ROI targets (NaN = none) and the study target."""

    def __init__(self, crops, info, roi_t, study_t):
        self.crops, self.info = crops, info.reset_index(drop=True)
        self.roi_t = roi_t.reindex(self.info.StudyInstanceUID).to_numpy(np.float32)
        self.study_t = study_t.reindex(self.info.StudyInstanceUID).to_numpy(np.float32)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, i):
        r = self.info.iloc[i]
        return {"x": torch.from_numpy(np.array(self.crops[r.src][r.row])),                 # 8 D S S uint8
                "plane": torch.tensor(PLANES.index(r.Anatomical_Plane)),
                "y_roi": torch.from_numpy(self.roi_t[i]), "y": torch.tensor(self.study_t[i]), "idx": torch.tensor(i)}


class SeriesAgg(nn.Module):
    """Crop25D on the 8 ROI crops of a series -> per-ROI logits and a study logit (max + attention pooling)."""

    def __init__(self, a, pretrained=True):
        super().__init__()
        self.base = build_model(a, len(ROI_CLASSES), pretrained=pretrained)
        d = self.base.fc.in_features
        self.pool_attn = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.drop = nn.Dropout(a.get("dropout", 0.2) if pretrained else 0.0)
        self.fc_study = nn.Linear(2 * d, 1)

    def forward(self, x, plane):
        """x (B, 8, D, S, S) float in [0, 1] (already augmented per crop) -> (roi logits (B, 8), study logit (B,))."""
        b, k = x.shape[:2]
        rid = torch.arange(k, device=x.device).repeat(b)
        e = self.base.embed(x.flatten(0, 1), rid, plane.repeat_interleave(k))           # B*8 d
        roi = self.base.fc(self.base.drop(e)).view(b, k)
        e = e.view(b, k, -1)
        w = self.pool_attn(e).squeeze(-1).softmax(1)
        pooled = torch.cat([e.amax(1), (w[..., None] * e).sum(1)], 1)
        return roi, self.fc_study(self.drop(pooled)).squeeze(-1)


def targets():
    """Per-ROI targets (StudyInstanceUID x 8) and the study target max(ROI max, mean(Claude, Codex))."""
    roi = pd.read_csv(os.path.join(OUTPUT_FOLDER, "crops", f"{DEFAULTS['roi_labels']}.csv")).set_index("StudyInstanceUID")
    roi = roi[[f"{c}|fracture" for c in ROI_CLASSES]]
    claude = pd.read_csv(CLAUDE_LABELS_CSV).set_index("StudyInstanceUID")["Fracture"]
    codex = pd.concat([pd.read_csv(os.path.join(INPUT_FOLDER, f"codex-labels_final_{n}.csv")) for n in (52, 4349)]
                      ).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")["Fracture"]
    avg = pd.concat([claude, codex.reindex(claude.index)], axis=1).mean(axis=1)
    study = pd.concat([roi.max(axis=1).reindex(avg.index), avg], axis=1).max(axis=1)
    return roi, study


@torch.no_grad()
def predict(model, dl, ds, device):
    model.eval()
    rows = []
    for b in dl:
        x = b["x"].to(device)
        x = model_input({"x": x.flatten(0, 1)}, device, None, train=False).view(*x.shape)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            roi, st = model(x, b["plane"].to(device))
        for i, r, s in zip(b["idx"].tolist(), torch.sigmoid(roi.float()).cpu().numpy(), torch.sigmoid(st.float()).cpu().tolist()):
            rows.append((i, list(map(float, r)), s))
    model.train()
    rows = sorted(all_gather_lists(rows))
    info = ds.info.iloc[[r[0] for r in rows]]
    out = pd.DataFrame(np.array([r[1] for r in rows]), columns=[f"{c}|fracture" for c in ROI_CLASSES])
    out.insert(0, "StudyInstanceUID", info.StudyInstanceUID.values)
    out["agg"] = [r[2] for r in rows]
    return out


def evaluate(pred, studies):
    st = studies.set_index("StudyInstanceUID")
    s = pred.groupby("StudyInstanceUID").mean(numeric_only=True)
    tab = pd.DataFrame({"Fracture_agg": s["agg"], "Fracture_max": s[[f"{c}|fracture" for c in ROI_CLASSES]].max(axis=1)})
    tab["Fracture_agg+max"] = (tab.Fracture_agg.rank(pct=True) + tab.Fracture_max.rank(pct=True)) / 2
    gt = st.loc[tab.index, "gt_Fracture"]
    g = gt.notna()
    hold = (st.loc[tab.index[~g], "target_Fracture"] >= 0.5).astype(int)
    res = {c: (roc_auc_score(gt[g], tab.loc[g, c]), roc_auc_score(hold, tab.loc[~g, c])) for c in tab.columns}
    return res, tab


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v) if not isinstance(v, bool) else (lambda s: s.lower() in ("1", "true")), default=v)
    args = p.parse_args()
    ctx = init_distributed()
    seed_everything(args.seed, ctx.rank)
    out = prepare_log_folder(args.log_path) if ctx.is_main else None
    args.output_dir = broadcast_object(out, device=ctx.device)
    args.group = "fracture_roi"                                   # for build_model / inference: 8 ROI rows
    args.model = "fracture_agg"
    args.z_sampling = json.load(open(os.path.join(OUTPUT_FOLDER, "crops", args.crops + ".json")))
    log = print if ctx.is_main else (lambda *a, **k: None)
    if ctx.is_main:
        create_logger(directory=args.output_dir, name="logs.txt")
        json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=4, default=str)

    crops, info = load_crops(args.crops)
    info = info[info.Anatomical_Plane.isin(args.planes.split(","))]
    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    val_mask = info.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids))
    roi_t, study_t = targets()
    if args.fullfit:          # every study; the 58 radiologist studies take their study label as the study target
        st = studies.set_index("StudyInstanceUID")
        study_t = study_t.where(st["gt_Fracture"].reindex(study_t.index).isna(), st["gt_Fracture"].reindex(study_t.index))
    tr_info = info if args.fullfit else info[~val_mask]
    tr_info = tr_info[tr_info.StudyInstanceUID.map(study_t).notna()]
    train_ds = SeriesDataset(crops, tr_info, roi_t, study_t)
    val_ds = SeriesDataset(crops, info[val_mask], roi_t, study_t)
    log(f"-> {args.output_dir}  fracture_agg  backbone {args.backbone}  init {args.init or 'imagenet'}\n"
        f"- train {len(train_ds)} series  val {len(val_ds)} series" + ("  (FULLFIT: validation leaky)" if args.fullfit else ""))

    def dl(ds, train):
        s = DistributedSampler(ds, shuffle=train, drop_last=train) if ctx.distributed else None
        return DataLoader(ds, batch_size=args.batch_size, sampler=s, shuffle=train and s is None, drop_last=train,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                          worker_init_fn=worker_init_fn), s

    train_dl, sampler = dl(train_ds, True)
    val_dl, _ = dl(val_ds, False)
    model = SeriesAgg(vars(args), pretrained=True)
    if args.init:
        sd = torch.load(os.path.join(args.init, "ckpt_final.pt"), map_location="cpu")
        res = model.base.load_state_dict(sd, strict=False)
        log(f"init from {args.init}: missing {res.missing_keys}, unexpected {res.unexpected_keys}")
    model = model.to(ctx.device)
    if ctx.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ctx.device.index])
    m = unwrap_model(model)
    enc = list(m.base.encoder.parameters())
    ids = {id(q) for q in enc}
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr_backbone},
                             {"params": [q for q in m.parameters() if id(q) not in ids], "lr": args.lr}],
                            weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + np.cos(np.pi * (it - warm) / max(1, total - warm))))
    args.augmentations = AUG
    t0 = time.time()
    try:
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            losses = []
            for b in train_dl:
                x = b["x"].to(ctx.device, non_blocking=True)
                n, k = x.shape[:2]
                x = model_input({"x": x.flatten(0, 1)}, ctx.device, args, train=True).view(n, k, *x.shape[2:])
                y_roi, y = b["y_roi"].to(ctx.device), b["y"].to(ctx.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    roi, st_logit = model(x, b["plane"].to(ctx.device))
                mask = (~torch.isnan(y_roi)).float()
                l_roi = (F.binary_cross_entropy_with_logits(roi.float(), torch.nan_to_num(y_roi), reduction="none") * mask
                         ).sum() / mask.sum().clamp(min=1.0)
                l_st = F.binary_cross_entropy_with_logits(st_logit.float(), y)
                loss = l_roi + args.study_weight * l_st
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()
                losses.append(float(loss.detach()))
            pred = predict(m, val_dl, val_ds, ctx.device)
            res, tab = evaluate(pred, studies)
            log(f"epoch {epoch + 1}  loss {all_reduce_mean({'l': float(np.mean(losses))}, ctx.device)['l']:.4f}  "
                + "  ".join(f"{c} {g:.3f}/{h:.3f}" for c, (g, h) in res.items()) + f"  ({time.time() - t0:.0f}s)")
        if ctx.is_main:
            torch.save(m.state_dict(), os.path.join(args.output_dir, "ckpt_final.pt"))
            tab.to_csv(os.path.join(args.output_dir, "val_study_preds.csv"))
            pred.to_csv(os.path.join(args.output_dir, "val_series_preds.csv"), index=False)
            log("FINAL " + "  ".join(f"{c} {g:.4f}/{h:.4f}" for c, (g, h) in res.items()))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
