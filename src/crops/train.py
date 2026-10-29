"""2.5D classifiers on ROI crops: data, model, fit loop, study-level scoring.

One sample = one (series, ROI): its (DEPTH, SIZE, SIZE) crop and the study's
label for the ROI's finding. ROIs can share a model (`GROUPS`): the two menisci,
the three OA compartments, ... -- the crop says which structure, a learnt ROI
embedding says which finding is asked. Sagittal and coronal series share it too,
with a plane embedding.

Model: every slice, with its two neighbours as RGB channels, through a pretrained
timm 2D backbone (global-pooled features), the per-slice vectors (+ ROI and
plane embeddings) through a bidirectional LSTM, attention pooling over slices,
one logit. Trained on the Claude report labels (soft BCE).

Scoring: a study's probability for a finding is the mean over its series of that
ROI's probability. AUC per finding against the radiologist labels of the 58
studies (`gt`) and the Claude labels binarised at 0.5 on the classifier holdout
(`hold`).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from params import OUTPUT_FOLDER
from roi.extract import ROI_CLASSES
from training.distributed import all_gather_lists, all_reduce_mean, unwrap_model

CROP_DIR = os.path.join(OUTPUT_FOLDER, "crops")
GROUPS = {
    "acl": ["ACL"], "mcl": ["MCL"], "ligaments": ["ACL", "MCL"],
    "menisci": ["Medial Meniscus", "Lateral Meniscus"],
    "oa": ["Medial OA", "Lateral OA", "PF OA"],
    "bakers": ["Baker's"], "all": list(ROI_CLASSES),
}
PLANES = ["Sagittal", "Coronal", "Axial"]


# --------------------------------------------------------------------- data


def load_crops(names):
    """Comma-separated crop sets (e.g. r3,r3_axial) -> ({name: memmap}, info with a `src` column)."""
    arrays, infos = {}, []
    for name in names.split(","):
        arrays[name] = np.load(os.path.join(CROP_DIR, f"{name}.npy"), mmap_mode="r")
        infos.append(pd.read_csv(os.path.join(CROP_DIR, f"{name}.csv")).assign(src=name))
    return arrays, pd.concat(infos, ignore_index=True)


class CropDataset(Dataset):
    """(series, ROI) samples of one group, with soft targets."""

    def __init__(self, crops, info, classes, targets):
        self.crops = crops
        rows = []
        for c in classes:
            k = ROI_CLASSES.index(c)
            rows.append(info.assign(roi=k, cls=c, target=info.StudyInstanceUID.map(targets[c])))
        self.items = pd.concat(rows, ignore_index=True).dropna(subset=["target"]).reset_index(drop=True)
        self.roi_ids = {ROI_CLASSES.index(c): i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items.iloc[i]
        return {"x": torch.from_numpy(np.array(self.crops[r.src][r.row, r.roi])),   # D S S uint8
                "roi": torch.tensor(self.roi_ids[r.roi]), "plane": torch.tensor(PLANES.index(r.Anatomical_Plane)),
                "y": torch.tensor(float(r.target), dtype=torch.float32), "idx": torch.tensor(i)}


def gpu_augment(x, aug):
    """(B, D, S, S) float [0, 1]: one in-plane affine and intensity change per sample."""
    b = x.shape[0]
    dev = x.device

    def u(lo, hi):
        return torch.empty(b, device=dev).uniform_(lo, hi)

    ang = u(-aug["rot_deg"], aug["rot_deg"]) * math.pi / 180
    sc = u(*aug["scale_range"])
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, u(-aug["shift"], aug["shift"])], -1),
                         torch.stack([sin, cos, u(-aug["shift"], aug["shift"])], -1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    g = torch.exp(u(-aug["gamma"], aug["gamma"]))[:, None, None, None]
    c = (1 + u(-aug["contrast"], aug["contrast"]))[:, None, None, None]
    x = ((x.clamp(0, 1) ** g - 0.5) * c + 0.5 + u(-aug["brightness"], aug["brightness"])[:, None, None, None])
    return x.clamp(0, 1)


# -------------------------------------------------------------------- model


class Crop25D(nn.Module):
    def __init__(self, backbone, n_rois, pretrained=True, img_size=None, hidden=256, dropout=0.2,
                 drop_path=0.1, head="lstm"):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        if img_size and any(k in backbone for k in ("maxvit", "coatnet", "maxxvit")):
            kw["img_size"] = img_size
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        c = self.encoder.num_features
        self.img_size = img_size
        self.roi_embed = nn.Embedding(n_rois, c)
        self.plane_embed = nn.Embedding(len(PLANES), c)
        self.head_type = head
        if head == "lstm":
            self.rnn = nn.LSTM(c, hidden, batch_first=True, bidirectional=True)
            d = 2 * hidden
        else:
            self.rnn = None
            d = c
        self.attn = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d, 1)
        cfg = self.encoder.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)))[None, :, None, None])
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)))[None, :, None, None])

    def forward(self, x, roi, plane):
        """x (B, D, S, S) float in [0, 1] -> logits (B,)."""
        b, d, s, _ = x.shape
        idx = torch.arange(d, device=x.device)
        trip = torch.stack([idx.sub(1).clamp(min=0), idx, idx.add(1).clamp(max=d - 1)], 1)   # D 3
        frames = x[:, trip].reshape(b * d, 3, s, s)
        if self.img_size and self.img_size != s:
            frames = F.interpolate(frames, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        f = self.encoder((frames - self.mean) / self.std).reshape(b, d, -1)
        f = f + (self.roi_embed(roi) + self.plane_embed(plane))[:, None]
        if self.rnn is not None:
            f, _ = self.rnn(f)
        w = self.attn(f).squeeze(-1).softmax(1)
        return self.fc(self.drop((w[..., None] * f).sum(1))).squeeze(-1)


# ---------------------------------------------------------------- scoring


def study_auc(pred, studies):
    """pred: rows (StudyInstanceUID, cls, p) per sample -> per-finding AUCs (gt, hold) and study table."""
    from sklearn.metrics import roc_auc_score

    st = studies.set_index("StudyInstanceUID")
    tab = pred.groupby(["StudyInstanceUID", "cls"]).p.mean().unstack()
    out = {}
    for c in tab.columns:
        p = tab[c].dropna()
        gt = st.loc[p.index, f"gt_{c}"]
        g = gt.notna()
        hold = ~g
        claude = (st.loc[p.index, f"target_{c}"] >= 0.5).astype(int)
        out[c] = dict(gt=roc_auc_score(gt[g], p[g]) if g.sum() and gt[g].nunique() == 2 else np.nan,
                      hold=roc_auc_score(claude[hold], p[hold]) if claude[hold].nunique() == 2 else np.nan)
    return pd.DataFrame(out).T, tab


@torch.no_grad()
def predict(model, loader, ds, device, amp=True):
    model.eval()
    rows = []
    for b in loader:
        x = b["x"].to(device, non_blocking=True).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            p = torch.sigmoid(model(x, b["roi"].to(device), b["plane"].to(device)).float())
        rows += list(zip(b["idx"].tolist(), p.cpu().tolist()))
    model.train()
    rows = dict(all_gather_lists(rows))
    it = ds.items.iloc[sorted(rows)]
    return pd.DataFrame({"StudyInstanceUID": it.StudyInstanceUID.values, "SeriesInstanceUID": it.SeriesInstanceUID.values,
                         "cls": it.cls.values, "plane": it.Anatomical_Plane.values,
                         "p": [rows[i] for i in sorted(rows)]})


def fit(model, train_dl, val_dl, val_ds, studies, args, ctx, sampler):
    device = ctx.device
    log = print if ctx.is_main else (lambda *a, **k: None)
    m = unwrap_model(model)
    enc = list(m.encoder.parameters())
    ids = {id(p) for p in enc}
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr_backbone},
                             {"params": [p for p in m.parameters() if id(p) not in ids], "lr": args.lr}],
                            weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + np.cos(np.pi * (it - warm) / max(1, total - warm))))
    t0, hist = time.time(), []
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        losses = []
        for b in train_dl:
            x = gpu_augment(b["x"].to(device, non_blocking=True).float() / 255, args.augmentations)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, b["roi"].to(device), b["plane"].to(device))
            loss = F.binary_cross_entropy_with_logits(logit.float(), b["y"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        pred = predict(m, val_dl, val_ds, device)
        tab, study_tab = study_auc(pred, studies)
        loss = all_reduce_mean({"l": float(np.mean(losses))}, device)["l"]
        log(f"epoch {epoch + 1}  loss {loss:.4f}  gt {tab["gt"].mean():.4f}  hold {tab["hold"].mean():.4f}  "
            + "  ".join(f"{c[:10]} {r["gt"]:.3f}/{r["hold"]:.3f}" for c, r in tab.iterrows()) + f"  ({time.time() - t0:.0f}s)")
        hist.append(dict(epoch=epoch + 1, gt=tab["gt"].mean(), hold=tab["hold"].mean()))
        if ctx.is_main:
            pred.to_csv(os.path.join(args.output_dir, "val_series_preds.csv"), index=False)
            study_tab.to_csv(os.path.join(args.output_dir, "val_study_preds.csv"))
            pd.DataFrame(hist).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)
            tab.to_csv(os.path.join(args.output_dir, "val_table.csv"))
    if ctx.is_main:
        torch.save(m.state_dict(), os.path.join(args.output_dir, "ckpt_final.pt"))
    return tab
