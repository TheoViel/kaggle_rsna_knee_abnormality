"""Probes on frozen OrthoFoundation features (`mt3d.ortho`), run from src_bis/.

    python -m mt3d.probe --feats joint --view roi        # each finding from its ROI-pooled tokens, 16 frames
    python -m mt3d.probe --feats roi --view roi          # each finding from its own ROI crop, 8 frames (CLS + mean)
    python -m mt3d.probe --feats joint --view global     # every finding from the slice-level CLS + mean tokens

One sample = one series; per finding, a sequence of per-frame vectors:
  joint / full, view roi     token 2 + k (patches averaged under ROI k's box), with the frame's CLS as context;
  joint / full, view global  CLS and mean patch token;
  roi                        ROI k crop's CLS and mean patch token (concatenated).
Head (shared by the 8 findings): LayerNorm -> Linear(d, h) + finding embedding -> GELU -> 1-layer
transformer over the frames (`--tf`, else none) -> attention pooling -> Linear -> logit. Soft BCE on the
training targets (`--labels`, as main_mt3d), AdamW, cosine. Study probability = mean over its series;
AUC on the 58 (`gt`) and the holdout (`hold`) as everywhere else. Features stay on the GPU.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from crops.train import study_auc
from knee_cls.data import load_tables, make_splits
from main_mt3d import training_targets
from mt3d.ortho import ORTHO_DIR
from mt3d.train import load_crops
from params import OUTPUT_FOLDER
from roi.extract import ROI_CLASSES

from params import CLASSES
EXTRA = [c for c in CLASSES if c not in ROI_CLASSES]      # effusion, synovitis, contusion, fracture
K = len(CLASSES)


class Head(nn.Module):
    def __init__(self, d_in, h=256, tf=1, dropout=0.2, n_planes=3):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, h)
        self.find = nn.Embedding(K, h)
        self.plane = nn.Embedding(n_planes, h)
        self.pos = nn.Parameter(torch.zeros(1, 32, h))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=0.1, batch_first=True,
                                                                   norm_first=True), tf) if tf else None
        self.attn = nn.Sequential(nn.Linear(h, h // 2), nn.Tanh(), nn.Linear(h // 2, 1))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(h, 1)

    def forward(self, x, k, plane):
        """x (B, F, d) frames of one finding's view, k (B,) finding, plane (B,) -> logits (B,)."""
        f = F.gelu(self.proj(self.norm(x)) + self.find(k)[:, None] + self.plane(plane)[:, None] + self.pos[:, :x.shape[1]])
        if self.tf is not None:
            f = self.tf(f)
        w = self.attn(f).softmax(1)
        return self.out(self.drop((w * f).sum(1))).squeeze(-1)


def views(feats, mode, view):
    """Features -> (get, d): get(series idx (B,), finding (B,)) -> (B, frames, d) float, on the GPU."""
    t = torch.from_numpy(np.ascontiguousarray(feats)).cuda()
    if mode == "roi":                                         # N, 8 ROIs, 8 frames, (CLS, mean), 1024
        return (lambda i, k: t[i, k].flatten(-2).float()), 2 * t.shape[-1]
    if view == "global":                                      # N, 16 frames, 10 tokens, 1024
        return (lambda i, k: t[i, :, :2].flatten(-2).float()), 2 * t.shape[-1]
    # ROI k's pooled patches (token 2 + k) and the frame's CLS, per frame; findings without an ROI
    # (k >= 8): the frame's CLS and mean patch token
    def get(i, k):
        roi = torch.cat([t[i, :, 2 + k.clamp(max=7)], t[i, :, 0]], -1)
        glob = t[i, :, :2].flatten(-2)
        return torch.where((k >= 8)[:, None, None], glob, roi).float()
    return get, 2 * t.shape[-1]


def fit_predict(a, info, tr_mask, pr_mask, y, get, d, n_cls, planes, idx):
    """Train a probe on the rows tr_mask, predict the rows pr_mask -> (rows, n_cls) probabilities."""
    yv = torch.from_numpy(y).cuda()
    tr = torch.nonzero(torch.from_numpy(tr_mask)).flatten().cuda()
    pr = torch.nonzero(torch.from_numpy(pr_mask)).flatten().cuda()
    items = torch.nonzero(~torch.isnan(yv[tr]))
    items = torch.stack([tr[items[:, 0]], items[:, 1]], 1)
    model = Head(d, a.h, a.tf).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(items) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    for _ in range(a.epochs):
        perm = items[torch.randperm(len(items), device="cuda")]
        for s_ in range(0, len(perm) - a.bs + 1, a.bs):
            b = perm[s_:s_ + a.bs]
            r, k = b[:, 0], b[:, 1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(get(idx[r], k), k, planes[r])
            loss = F.binary_cross_entropy_with_logits(logit.float(), yv[r, k])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    out = []
    with torch.no_grad():
        for s_ in range(0, len(pr), 512):
            r = pr[s_:s_ + 512]
            ps = []
            for k in range(n_cls):
                kk = torch.full((len(r),), k, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ps.append(torch.sigmoid(model(get(idx[r], kk), kk, planes[r]).float()))
            out.append(torch.stack(ps, 1))
    return torch.cat(out).cpu().numpy()


def oof(a, info, val, studies, yv_np, get, d, classes):
    """k-fold over the training studies (grouped by study): each fold predicted by a probe trained on the
    others; validation studies by a probe on all training studies. Study = mean over its series."""
    planes = torch.tensor(info.Anatomical_Plane.map({"Sagittal": 0, "Coronal": 1}).to_numpy()).cuda()
    idx = torch.tensor(info.i.to_numpy()).cuda()
    train_st = info.StudyInstanceUID[~val].unique()
    rng = np.random.default_rng(a.seed)
    fold = dict(zip(train_st, rng.permutation(len(train_st)) % a.oof))
    f = info.StudyInstanceUID.map(fold).fillna(-1).to_numpy()
    pred = np.full((len(info), len(classes)), np.nan, np.float32)
    for k in range(a.oof):
        m = f == k
        pred[m] = fit_predict(a, info, (f >= 0) & ~m, m, yv_np, get, d, len(classes), planes, idx)
        print(f"  fold {k + 1}/{a.oof}", flush=True)
    vm = val.to_numpy()
    pred[vm] = fit_predict(a, info, f >= 0, vm, yv_np, get, d, len(classes), planes, idx)
    tab = pd.DataFrame(pred, columns=classes).assign(StudyInstanceUID=info.StudyInstanceUID.values)
    tab = tab.groupby("StudyInstanceUID").mean()
    tab.to_csv(a.out)
    st = studies.set_index("StudyInstanceUID")
    from sklearn.metrics import roc_auc_score
    hold = [s_ for s_ in tab.index if s_ in set(info.StudyInstanceUID[val]) and not st.loc[s_, "is_gt"]]
    trn = [s_ for s_ in tab.index if s_ in set(train_st)]
    for name, ids in (("train (out-of-fold)", trn), ("holdout", hold)):
        au = [roc_auc_score((st.loc[ids, f"target_{c}"] >= 0.5).astype(int), tab.loc[ids, c]) for c in classes]
        print(f"{name}: AUC vs Claude labels {np.mean(au):.4f}  " + " ".join(f"{c[:6]} {x:.3f}" for c, x in zip(classes, au)))
    print(f"-> {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feats", default="joint", help="output/bis/ortho/<feats>.npy: joint, roi, full, joint_dinov3")
    p.add_argument("--view", default="roi", choices=["roi", "global"])
    p.add_argument("--labels", default="pl:pl_round2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--h", type=int, default=256)
    p.add_argument("--tf", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--all12", action="store_true", help="also the 4 findings without an ROI (labels: --labels_extra)")
    p.add_argument("--labels_extra", default="mean")
    p.add_argument("--oof", type=int, default=0,
                   help="k-fold out-of-fold predictions for the training studies (pseudo-labels) -> --out")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    feats = np.load(os.path.join(ORTHO_DIR, f"{a.feats}.npy"), mmap_mode="r")
    info = pd.read_csv(os.path.join(OUTPUT_FOLDER, "crops", "f_s1n8.csv")) if a.feats.startswith("roi") else load_crops("u10_d32_s256")[2]
    info = info.reset_index(drop=True).assign(i=np.arange(len(info)))
    info = info[info.Anatomical_Plane.isin(["Sagittal", "Coronal"])]
    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    val = info.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids))
    classes = list(ROI_CLASSES) + (EXTRA if a.all12 else [])
    tg = training_targets(a.labels, studies, ROI_CLASSES)
    if a.all12:
        tg |= training_targets(a.labels_extra, studies, EXTRA)
    y = np.stack([info.StudyInstanceUID.map(tg[c]).to_numpy(float) for c in classes], 1).astype(np.float32)
    get, d = views(feats, "roi" if a.feats.startswith("roi") else "joint", a.view)
    planes = torch.tensor(info.Anatomical_Plane.map({"Sagittal": 0, "Coronal": 1}).to_numpy()).cuda()
    idx = torch.tensor(info.i.to_numpy()).cuda()
    yv = torch.from_numpy(y).cuda()
    if a.oof:
        oof(a, info, val, studies, yv_np=y, get=get, d=d, classes=classes)
        return
    tr = torch.nonzero(torch.from_numpy(~val.to_numpy())).flatten().cuda()
    va = torch.nonzero(torch.from_numpy(val.to_numpy())).flatten().cuda()
    # Training items: (row, finding) with a label.
    items = torch.nonzero(~torch.isnan(yv[tr])).cuda()
    items = torch.stack([tr[items[:, 0]], items[:, 1]], 1)
    model = Head(d, a.h, a.tf).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(items) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()

    def predict():
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(va), 512):
                r = va[s:s + 512]
                ps = []
                for k in range(len(classes)):
                    kk = torch.full((len(r),), k, device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ps.append(torch.sigmoid(model(get(idx[r], kk), kk, planes[r]).float()))
                out.append(torch.stack(ps, 1))
        model.train()
        p_ = torch.cat(out).cpu().numpy()
        it = info.iloc[va.cpu().numpy()]
        wide = pd.DataFrame(p_, columns=classes).assign(StudyInstanceUID=it.StudyInstanceUID.values)
        return wide.melt(id_vars=["StudyInstanceUID"], var_name="cls", value_name="p")

    for ep in range(a.epochs):
        perm = items[torch.randperm(len(items), device="cuda")]
        for s in range(0, len(perm) - a.bs + 1, a.bs):
            b = perm[s:s + a.bs]
            r, k = b[:, 0], b[:, 1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(get(idx[r], k), k, planes[r])
            loss = F.binary_cross_entropy_with_logits(logit.float(), yv[r, k])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if (ep + 1) % 5 == 0 or ep + 1 == a.epochs:
            tab, st_tab = study_auc(predict(), studies)
            roi = tab.loc[list(ROI_CLASSES)]
            x4 = tab.loc[[c for c in EXTRA if c in tab.index]]
            print(f"ep {ep + 1}  loss {loss.item():.4f}  gt8 {roi['gt'].mean():.4f}  hold8 {roi['hold'].mean():.4f}  "
                  + (f"gt4x {x4['gt'].mean():.4f} hold4x {x4['hold'].mean():.4f}  " if len(x4) else "")
                  + "  ".join(f"{c[:6]} {r_['gt']:.3f}/{r_['hold']:.3f}" for c, r_ in tab.iterrows()) + f"  ({time.time() - t0:.0f}s)",
                  flush=True)
    if a.out:
        st_tab.to_csv(a.out)


if __name__ == "__main__":
    main()
