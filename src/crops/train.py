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
    "medial_meniscus": ["Medial Meniscus"], "lateral_meniscus": ["Lateral Meniscus"],
    "oa": ["Medial OA", "Lateral OA", "PF OA"],
    "bakers": ["Baker's"], "all": list(ROI_CLASSES),
    # No box of their own: one "Capsule" crop per series (crops.prepare --capsule), shared.
    "capsule": ["Effusion", "Synovitis"], "effusion": ["Effusion"], "synovitis": ["Synovitis"],
    # Bone findings around the joint, from the same whole-joint crop (experiment).
    "joint4": ["Effusion", "Synovitis", "Contusion", "Fracture"], "bone": ["Contusion", "Fracture"],
    # Both tibiofemoral compartments from one crop that shows both (single-crop sets).
    "tf": ["Medial OA", "Lateral OA"],
    # MOAKS gradings for multi-task pretraining on OAI ("<ROI>|<feature>", see oai.labels.aux_targets).
    "moaks_aux": ['Medial OA|cart_size', 'Medial OA|cart_full', 'Medial OA|cart_sum', 'Medial OA|bml', 'Lateral OA|cart_size', 'Lateral OA|cart_full', 'Lateral OA|cart_sum', 'Lateral OA|bml', 'PF OA|cart_size', 'PF OA|cart_full', 'PF OA|cart_sum', 'PF OA|bml', 'Medial Meniscus|tear', 'Medial Meniscus|maceration', 'Medial Meniscus|signal', 'Medial Meniscus|extrusion', 'Lateral Meniscus|tear', 'Lateral Meniscus|maceration', 'Lateral Meniscus|signal', 'Lateral Meniscus|extrusion', 'ACL|tear', 'Medial OA|osteophyte', 'Lateral OA|osteophyte', 'PF OA|osteophyte', "Baker's|cyst"],
}
GROUPS.update({
    # Multi-task joint training: our findings (our crops) + the MOAKS gradings (OAI crops) in one model.
    "oa_moaks": ["Medial OA", "Lateral OA", "PF OA"] + GROUPS["moaks_aux"],
    "all_moaks": list(ROI_CLASSES) + GROUPS["moaks_aux"],
    # MRNet's exam-level "abnormal" label as an auxiliary target on every ROI crop (mrnet.labels --abn).
    "all_abn": list(ROI_CLASSES) + [f"{c}|abnormal" for c in ROI_CLASSES],
    # Fracture read from the ROI crops, one target per ROI (fracture.roi_labels); study score = max over ROIs.
    "fracture_roi": [f"{c}|fracture" for c in ROI_CLASSES],
    # + the fibular-head crop (crop sets f9_*, fracture.fibula; labels fracture.roi_labels --fibula)
    "fracture_roi9": [f"{c}|fracture" for c in ROI_CLASSES + ["Fibula"]],
    # The "all" model + per-ROI fracture targets (labels pl:pl_round2_fracv1): fracture without a separate model.
    "all_frac": list(ROI_CLASSES) + [f"{c}|fracture" for c in ROI_CLASSES],
})
CAPSULE_FINDINGS = ("Effusion", "Synovitis", "Contusion", "Fracture")      # findings read from the capsule crop


def crop_index(c, single=False):
    """Index of a finding's crop in a crop set: its ROI, or the single capsule crop."""
    if "|" in c:                      # MOAKS auxiliary target: the crop of its ROI
        c = c.split("|")[0]
    if c == "Fibula":                 # ninth crop of the f9_* crop sets (fracture.fibula)
        return len(ROI_CLASSES)
    return 0 if single or c in CAPSULE_FINDINGS else ROI_CLASSES.index(c)
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

    def __init__(self, crops, info, classes, targets, aux_targets=None):
        self.crops = crops
        rows = []
        # Single-crop sets (one union / capsule crop per series): every finding reads crop 0.
        single = all(a.shape[1] == 1 for a in crops.values())
        for i, c in enumerate(classes):
            # roi: which crop of the series; rid: the finding's embedding row (its index in the group).
            rows.append(info.assign(roi=crop_index(c, single), rid=i, cls=c, target=info.StudyInstanceUID.map(targets[c]),
                                    # auxiliary target of the same crop (args.aux), NaN = none
                                    target2=info.StudyInstanceUID.map(aux_targets[c]) if aux_targets and c in aux_targets else np.nan))
        self.items = pd.concat(rows, ignore_index=True).dropna(subset=["target"]).reset_index(drop=True)
        if "w" not in self.items:
            self.items["w"] = 1.0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items.iloc[i]
        return {"x": torch.from_numpy(np.array(self.crops[r.src][r.row, r.roi])),   # D S S uint8
                "roi": torch.tensor(r.rid), "plane": torch.tensor(PLANES.index(r.Anatomical_Plane)),
                "y": torch.tensor(float(r.target), dtype=torch.float32), "idx": torch.tensor(i),
                "w": torch.tensor(float(r.w), dtype=torch.float32),
                "y2": torch.tensor(float(r.target2), dtype=torch.float32)}


class PairDataset(Dataset):
    """Multiple-instance bags: both meniscus crops of a series whose exam has a meniscal tear of unknown
    side (MRNet). Training pushes max(medial, lateral) towards 1 (`fit`, `mil_dl`)."""

    def __init__(self, crops, info, classes):
        self.crops, self.items = crops, info.reset_index(drop=True)
        self.rois = [ROI_CLASSES.index(c) for c in ("Medial Meniscus", "Lateral Meniscus")]
        self.rids = [classes.index(c) for c in ("Medial Meniscus", "Lateral Meniscus")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items.iloc[i]
        a = self.crops[r.src]
        return {"x": torch.from_numpy(np.stack([np.array(a[r.row, k]) for k in self.rois])),     # 2 D S S
                "roi": torch.tensor(self.rids), "plane": torch.tensor(PLANES.index(r.Anatomical_Plane))}


def gpu_augment(x, aug, ctx=1.0, crop_jitter=0.0):
    """(B, D, S, S) float [0, 1]: one in-plane affine and intensity change per sample.

    ctx > 1: x is a context crop (the usual window x ctx, at S = ctx x the output size); the
    affine samples its central 1 / ctx part, so rotations, zoom-outs and shifts see real
    anatomy instead of zero padding. crop_jitter: extra random shift of the window centre,
    as a fraction of the (output) crop side.
    """
    b = x.shape[0]
    dev = x.device

    def u(lo, hi):
        return torch.empty(b, device=dev).uniform_(lo, hi)

    ang = u(-aug["rot_deg"], aug["rot_deg"]) * math.pi / 180
    sc = u(*aug["scale_range"])
    cos, sin = torch.cos(ang) / sc / ctx, torch.sin(ang) / sc / ctx
    jit = 2 * crop_jitter / ctx
    tx = u(-aug["shift"], aug["shift"]) / ctx + (u(-jit, jit) if crop_jitter else 0)
    ty = u(-aug["shift"], aug["shift"]) / ctx + (u(-jit, jit) if crop_jitter else 0)
    theta = torch.stack([torch.stack([cos, -sin, tx], -1), torch.stack([sin, cos, ty], -1)], 1)
    out = round(x.shape[-1] / ctx)
    grid = F.affine_grid(theta, (b, x.shape[1], out, out), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    g = torch.exp(u(-aug["gamma"], aug["gamma"]))[:, None, None, None]
    c = (1 + u(-aug["contrast"], aug["contrast"]))[:, None, None, None]
    x = ((x.clamp(0, 1) ** g - 0.5) * c + 0.5 + u(-aug["brightness"], aug["brightness"])[:, None, None, None])
    return x.clamp(0, 1)


def center_view(x, ctx=1.0):
    """The central 1 / ctx window of context crops, at S / ctx px: the usual crop (validation, inference)."""
    if ctx == 1.0:
        return x
    b = x.shape[0]
    theta = torch.tensor([[1 / ctx, 0, 0], [0, 1 / ctx, 0]], device=x.device, dtype=x.dtype).expand(b, 2, 3)
    out = round(x.shape[-1] / ctx)
    return F.grid_sample(x, F.affine_grid(theta, (b, x.shape[1], out, out), align_corners=False),
                         mode="bilinear", padding_mode="zeros", align_corners=False)


def slice_window(x, depth, jitter=0):
    """(B, D, S, S) -> (B, depth, S, S): the central `depth` slices, shifted by up to +-jitter (train)."""
    if not depth or depth >= x.shape[1]:
        return x
    start = (x.shape[1] - depth) // 2
    if not jitter:
        return x[:, start:start + depth]
    off = torch.randint(-jitter, jitter + 1, (x.shape[0],), device=x.device).clamp(-start, x.shape[1] - depth - start)
    idx = (start + off)[:, None] + torch.arange(depth, device=x.device)
    return torch.gather(x, 1, idx[:, :, None, None].expand(-1, -1, *x.shape[2:]))


def mix_batch(x, y, rid, mixup=0.0, cutmix=0.0, alpha=0.4):
    """Mixup / CutMix between crops of the same finding (rid), soft targets mixed alike."""
    r = np.random.rand()
    if r >= mixup + cutmix:
        return x, y
    perm = torch.arange(len(rid), device=x.device)
    for k in rid.unique():
        i = torch.nonzero(rid == k).flatten()
        perm[i] = i[torch.randperm(len(i), device=x.device)]
    lam = float(np.random.beta(alpha, alpha))
    if r < mixup:
        return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
    s = x.shape[-1]
    h = int(round(s * np.sqrt(1 - lam)))
    cy, cx = np.random.randint(s), np.random.randint(s)
    y0, y1, x0, x1 = max(cy - h // 2, 0), min(cy + h // 2, s), max(cx - h // 2, 0), min(cx + h // 2, s)
    x = x.clone()
    x[..., y0:y1, x0:x1] = x[perm][..., y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (s * s)
    return x, lam * y + (1 - lam) * y[perm]


def model_input(b, device, args, train):
    """Batch crops -> model input: slice window, context window (+ augmentation when training)."""
    x = b["x"].to(device, non_blocking=True).float() / 255
    ctx, depth = getattr(args, "ctx", 1.0) or 1.0, getattr(args, "depth_out", 0)
    x = slice_window(x, depth, getattr(args, "slice_jitter", 0) if train else 0)
    if train:
        return gpu_augment(x, args.augmentations, ctx, getattr(args, "crop_jitter", 0.0))
    return center_view(x, ctx)


# -------------------------------------------------------------------- model


class Crop25D(nn.Module):
    """(B, D, S, S) crop -> one logit. How the D slices are combined:

    fusion  late   every slice (with its two neighbours as RGB) through the 2D encoder alone;
            tsm    as late, plus a temporal shift in the blocks of the last two stages (1/8 of
                   the channels from the previous slice, 1/8 from the next, before the block's
                   depthwise conv): slices exchange information inside the encoder, no extra FLOPs;
            early  all D slices as the input channels of one 2D image (1/D of the FLOPs).
    head    lstm   BiLSTM over the slice vectors, attention pooling (the original);
            attn   attention pooling only;
            transformer  2 transformer layers over the slice vectors, attention pooling;
            map3d  the last feature maps of all slices kept as a (C, D, h, w) volume, two
                   residual 3x3x3 convs, then pooling: 3D context without pooling space first.
    """

    def __init__(self, backbone, n_rois, pretrained=True, img_size=None, hidden=256, dropout=0.2,
                 drop_path=0.1, head="lstm", fusion="late", depth=None, aux=False):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        if img_size and any(k in backbone for k in ("maxvit", "coatnet", "maxxvit", "tiny_vit", "efficientvit_m", "vit_")):
            kw["img_size"] = img_size
        if fusion == "early":
            head = "linear"                  # one 2D image per crop: nothing to aggregate
        self.fusion, self.head_type = fusion, head
        if fusion == "early":
            kw["in_chans"] = depth
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        # Pooled vector size (some heads add a conv after pooling, e.g. MobileNetV4); map channels for map3d.
        c = self.encoder.num_features if head == "map3d" else \
            (getattr(self.encoder, "head_hidden_size", None) or self.encoder.num_features)
        self.img_size = img_size
        self._d = None
        if fusion == "tsm":
            self._add_shifts()
        if head == "map3d":
            m = 256
            self.reduce = nn.Sequential(nn.Conv3d(c, m, 1, bias=False), nn.GroupNorm(16, m), nn.GELU())
            self.conv3d = nn.ModuleList([nn.Sequential(nn.Conv3d(m, m, 3, padding=1, bias=False), nn.GroupNorm(16, m),
                                                       nn.GELU()) for _ in range(2)])
            c = m
        self.roi_embed = nn.Embedding(n_rois, c)
        self.plane_embed = nn.Embedding(len(PLANES), c)
        self.rnn = self.tf = None
        d = c
        if head == "lstm":
            self.rnn = nn.LSTM(c, hidden, batch_first=True, bidirectional=True)
            d = 2 * hidden
        elif head == "transformer":
            self.proj = nn.Linear(c, hidden)
            self.pos = nn.Parameter(torch.zeros(1, 64, hidden))
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout=0.1,
                                                                       batch_first=True, norm_first=True), 2)
            d = hidden
        if head in ("map3d", "linear"):
            d = 2 * c if head == "map3d" else c                  # map3d: avg + max pooled
            self.attn = None
        else:
            self.attn = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d, 1)
        # Auxiliary output (args.aux, e.g. "fracture"): a second logit from the same pass, "is there <aux> in this ROI".
        self.fc_aux = nn.Linear(d, 1) if aux else None
        cfg = self.encoder.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)))[None, :, None, None])
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)))[None, :, None, None])

    def _add_shifts(self):
        """Temporal shift before the depthwise conv of the blocks of the last two stages."""
        n = 0
        last = {f"stages.{i}." for i in range(len(getattr(self.encoder, "stages", [])))[-2:]}
        for name, mod in self.encoder.named_modules():
            if any(name.startswith(k) for k in last) and name.endswith(("conv_dw", "token_mixer")):
                mod.register_forward_pre_hook(self._shift)
                n += 1
        assert n, "tsm: no depthwise conv found in this backbone"

    def _shift(self, mod, inp):
        x = inp[0]
        d = self._d
        nd, ch = x.shape[0], x.shape[1]
        x = x.reshape(nd // d, d, *x.shape[1:])
        f = ch // 8
        prev = torch.cat([torch.zeros_like(x[:, :1, :f]), x[:, :-1, :f]], 1)           # from slice i-1
        nxt = torch.cat([x[:, 1:, f:2 * f], torch.zeros_like(x[:, :1, f:2 * f])], 1)   # from slice i+1
        x = torch.cat([prev, nxt, x[:, :, 2 * f:]], 2).reshape(nd, *x.shape[2:])
        return (x,) + tuple(inp[1:])

    def forward(self, x, roi, plane, both=False):
        """x (B, D, S, S) float in [0, 1] -> logits (B,); both=True: (B, 2) with the auxiliary logit."""
        e = self.drop(self.embed(x, roi, plane))
        main = self.fc(e).squeeze(-1)
        if both:
            return torch.stack([main, self.fc_aux(e).squeeze(-1)], -1)
        return main

    def embed(self, x, roi, plane):
        """x (B, D, S, S) float in [0, 1] -> the crop's vector (B, d), the input of the final layer."""
        b, d, s, _ = x.shape
        emb = self.roi_embed(roi) + self.plane_embed(plane)
        if self.fusion == "early":
            if self.img_size and self.img_size != s:
                x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            f = self.encoder.forward_head(self.encoder.forward_features((x - 0.45) / 0.225), pre_logits=True)
            return f + emb
        idx = torch.arange(d, device=x.device)
        trip = torch.stack([idx.sub(1).clamp(min=0), idx, idx.add(1).clamp(max=d - 1)], 1)   # D 3
        frames = x[:, trip].reshape(b * d, 3, s, s)
        if self.img_size and self.img_size != s:
            frames = F.interpolate(frames, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        self._d = d
        maps = self.encoder.forward_features((frames - self.mean) / self.std)
        if self.head_type == "map3d":
            v = maps.reshape(b, d, *maps.shape[1:]).transpose(1, 2)                         # B C D h w
            v = self.reduce(v)
            for conv in self.conv3d:
                v = v + conv(v)
            v = v + emb[:, :, None, None, None]
            return torch.cat([v.mean((2, 3, 4)), v.amax((2, 3, 4))], 1)
        f = self.encoder.forward_head(maps, pre_logits=True).reshape(b, d, -1)
        f = f + emb[:, None]
        if self.rnn is not None:
            f, _ = self.rnn(f)
        elif self.tf is not None:
            f = self.tf(self.proj(f) + self.pos[:, :d])
        w = self.attn(f).squeeze(-1).softmax(1)
        return (w[..., None] * f).sum(1)


class Crop3D(nn.Module):
    """(B, D, S, S) crop -> one logit, with a 3D CNN: the encoder of the ROI U-Net
    (`roi.train.build_model`), which was trained on these knee volumes (32 x 128 x 128 in
    [0, 1]), then average + max pooling, finding / plane embeddings and a linear layer."""

    UNET = "../logs/roi_run/2026-10-09/1/ckpt_final.pt"

    def __init__(self, n_rois, pretrained=True, dropout=0.2):
        super().__init__()
        from roi.train import build_model as build_unet

        unet = build_unet(num_classes=8)
        if pretrained:
            unet.load_state_dict(torch.load(self.UNET, map_location="cpu"))
        self.encoder = unet.encoder
        c = unet.encoder.output_channels[-1]
        self.roi_embed = nn.Embedding(n_rois, 2 * c)
        self.plane_embed = nn.Embedding(len(PLANES), 2 * c)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * c, 1)

    def forward(self, x, roi, plane):
        return self.fc(self.drop(self.embed(x, roi, plane))).squeeze(-1)

    def embed(self, x, roi, plane):
        v = self.encoder(x[:, None])[-1]                                   # B C d h w
        f = torch.cat([v.mean((2, 3, 4)), v.amax((2, 3, 4))], 1).float()
        return f + self.roi_embed(roi) + self.plane_embed(plane)


def build_model(a, n_rois, pretrained=False):
    """Crop25D (or Crop3D) from a run's args (dict), as trained; older runs lack the newer keys."""
    z = a.get("z_sampling") or {}
    if a["backbone"] == "roi_unet3d":
        return Crop3D(n_rois, pretrained=pretrained, dropout=a.get("dropout", 0.2) if pretrained else 0.0)
    if a["backbone"] == "roi_unet3d_scratch":
        return Crop3D(n_rois, pretrained=False, dropout=a.get("dropout", 0.2) if pretrained else 0.0)
    return Crop25D(a["backbone"], n_rois, pretrained=pretrained, img_size=a.get("img_size") or None,
                   hidden=a.get("hidden", 256), dropout=a.get("dropout", 0.2) if pretrained else 0.0,
                   drop_path=a.get("drop_path", 0.1) if pretrained else 0.0, head=a.get("head", "lstm"),
                   fusion=a.get("fusion", "late"), depth=z.get("depth", 12), aux=bool(a.get("aux")))


# ---------------------------------------------------------------- scoring


def finding_of(sub):
    """'fracture' -> 'Fracture' when a sub-target suffix names one of the 12 findings, else None."""
    from params import CLASSES
    return next((c for c in CLASSES if c.lower() == sub.lower()), None)


def study_auc(pred, studies, aux=None):
    """pred: rows (StudyInstanceUID, cls, p) per sample -> per-finding AUCs (gt, hold) and study table."""
    from sklearn.metrics import roc_auc_score

    st = studies.set_index("StudyInstanceUID")
    tab = pred.groupby(["StudyInstanceUID", "cls"]).p.mean().unstack()
    if aux is not None and "p_aux" in pred:       # auxiliary per-ROI output: study score = max over ROIs
        tab[aux] = pred.groupby(["StudyInstanceUID", "cls"]).p_aux.mean().unstack().max(axis=1)
    # Per-ROI sub-targets of a study finding ("<ROI>|fracture"): the study score is the max over ROIs.
    for f in sorted({c.split("|")[1] for c in tab.columns if "|" in c}):
        F = finding_of(f)
        if F is not None:
            tab[F] = tab[[c for c in tab.columns if c.endswith("|" + f)]].max(axis=1)
    out = {}
    for c in tab.columns:
        if f"gt_{c}" not in st:
            continue
        p = tab[c].dropna()
        gt = st.loc[p.index, f"gt_{c}"]
        g = gt.notna()
        hold = ~g
        claude = (st.loc[p.index, f"target_{c}"] >= 0.5).astype(int)
        out[c] = dict(gt=roc_auc_score(gt[g], p[g]) if g.sum() and gt[g].nunique() == 2 else np.nan,
                      hold=roc_auc_score(claude[hold], p[hold]) if claude[hold].nunique() == 2 else np.nan)
    return pd.DataFrame(out).T, tab


@torch.no_grad()
def predict(model, loader, ds, device, amp=True, args=None):
    model.eval()
    rows = []
    aux = bool(getattr(args, "aux", "")) if args is not None else False
    for b in loader:
        x = model_input(b, device, args, train=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            p = torch.sigmoid(model(x, b["roi"].to(device), b["plane"].to(device), **({"both": True} if aux else {})).float())
        rows += list(zip(b["idx"].tolist(), p.cpu().tolist()))
    model.train()
    rows = dict(all_gather_lists(rows))
    it = ds.items.iloc[sorted(rows)]
    out = pd.DataFrame({"StudyInstanceUID": it.StudyInstanceUID.values, "SeriesInstanceUID": it.SeriesInstanceUID.values,
                        "cls": it.cls.values, "plane": it.Anatomical_Plane.values})
    if aux:
        out["p"] = [rows[i][0] for i in sorted(rows)]
        out["p_aux"] = [rows[i][1] for i in sorted(rows)]
    else:
        out["p"] = [rows[i] for i in sorted(rows)]
    return out


def fit(model, train_dl, val_dl, val_ds, studies, args, ctx, sampler, mil_dl=None):
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
    t0, best, hist = time.time(), None, []
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        losses = []
        mil_it = iter(mil_dl) if mil_dl is not None else None
        for b in train_dl:
            x = model_input(b, device, args, train=True)
            roi, y = b["roi"].to(device), b["y"].to(device)
            x, y = mix_batch(x, y, roi, getattr(args, "mixup", 0.0), getattr(args, "cutmix", 0.0), getattr(args, "mix_alpha", 0.4))
            plane, n = b["plane"].to(device), len(y)
            if mil_it is not None:
                # MIL bags (PairDataset) ride in the same forward pass: medial crops, then lateral crops.
                try:
                    mb = next(mil_it)
                except StopIteration:
                    mil_it = iter(mil_dl)
                    mb = next(mil_it)
                k = mb["x"].shape[0]
                mx = model_input({"x": mb["x"].flatten(0, 1)}, device, args, train=True)      # (2k, D, S, S): pairs interleaved
                order = torch.cat([torch.arange(0, 2 * k, 2), torch.arange(1, 2 * k, 2)]).to(device)
                x = torch.cat([x, mx[order]])
                roi = torch.cat([roi, mb["roi"][:, 0].to(device), mb["roi"][:, 1].to(device)])
                plane = torch.cat([plane, mb["plane"].to(device).repeat(2)])
            aux = bool(getattr(args, "aux", ""))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, roi, plane, **({"both": True} if aux else {}))
            if aux:
                logit, logit2 = logit[:, 0], logit[:, 1]
            w = b["w"].to(device)                                   # extra-data (OAI) items can be down-weighted
            loss = (F.binary_cross_entropy_with_logits(logit[:n].float(), y, reduction="none") * w).sum() / w.sum()
            if aux:
                # Auxiliary target of each crop; items without one (NaN) are masked, and the masked sum keeps the
                # auxiliary head in every rank's graph (DDP).
                y2 = b["y2"].to(device)
                m2 = (~torch.isnan(y2)).float()
                l2 = F.binary_cross_entropy_with_logits(logit2[:n].float(), torch.nan_to_num(y2), reduction="none")
                loss = loss + args.aux_weight * (l2 * m2).sum() / m2.sum().clamp(min=1.0)
            if mil_it is not None:
                bag = torch.maximum(logit[n:n + k], logit[n + k:]).float()        # max-instance MIL
                loss = loss + args.mil_weight * F.binary_cross_entropy_with_logits(bag, torch.ones_like(bag))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        if len(val_ds) == 0:              # pretraining on extra data only: nothing of ours to score
            log(f"epoch {epoch + 1}  loss {all_reduce_mean({'l': float(np.mean(losses))}, device)['l']:.4f}  ({time.time() - t0:.0f}s)")
            continue
        pred = predict(m, val_dl, val_ds, device, args=args)
        tab, study_tab = study_auc(pred, studies, aux=finding_of(args.aux) if getattr(args, "aux", "") else None)
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
    return tab if len(val_ds) else pd.DataFrame({"gt": [np.nan], "hold": [np.nan]})
