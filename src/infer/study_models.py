"""Study-level models (src_study/) as recipe models of infer.pipeline: one prediction per study, plane "Joint".

<weights>/study/<name>/ holds one of:
  - a study model (main_study.py --model study): args.json, ckpt_final.pt. StudyNet: every series' capsule crop
    (sagittal / coronal, then axial; at most val_max_series), ConvNeXt per 3-frame triplet, BiLSTM + frame attention
    per series, per-finding attention over the series;
  - a window-MIL model (main_study.py --model mil, b1_c1's B1 design): args.json, ckpt_final.pt. Every frame of every
    series as a 3-frame window, per-finding attention over the windows;
  - dense heads (c1_train.py, b1_c1's C1 design): c1.json, heads.pt. Frozen features per series frame -- the
    RadImageNet ResNet-50 (<weights>/study/radimagenet/ResNet50.pt), a frozen timm encoder (<weights>/study/<name>/:
    encoder.json {arch, norm}, ckpt_final.pt, crop.json; e.g. convergedmachine's RadImageNet ResNet-50), or the encoder
    of a study model of this folder (c1.json "encoder") -- six slots (plane x fat-suppressed, the series with most slices), one DenseSequenceHead
    per seed; the model's probability is the mean over its seeds.
Crops: capsule crops of the pipeline (process_chunk's crops[z]), z from the crop set the model was trained on
(crop.json, copied from output/crops/<set>.json), so training and test crops match.

The networks are copies of src_study/study.py (StudyNet), src_study/mil.py (WindowNet), jiwei-code/b1_c1_training
dense.py (DenseSequenceHead) and dense_pixels.py (RadEncoder): Kaggle ships src/ only.
"""

import json
import math
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = ["Sagittal", "Coronal", "Axial"]
SLOTS = [("Sagittal", 1), ("Coronal", 1), ("Axial", 1), ("Sagittal", 0), ("Coronal", 0), ("Axial", 0)]
Z_KEYS = ("depth", "z_mode", "stride", "size", "capsule", "capsule_up")
WINDOW_CHUNK = 128         # frames per encoder call
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
MIL_EVAL_WINDOWS = 62      # MIL windows per study (B1's 62; validated: 0.930 / 0.937 vs every frame 0.929 / 0.938)


# ----------------------------------------------------------------------------- networks (copies, see above)

RADIMAGENET_R50 = "radimagenet_r50"     # timm ResNet-50 with the RadImageNet weights (b1_c1's C1 encoder), fine-tuned
RAD_TRUNK = ["conv1", "bn1", "act1", "maxpool", "layer1", "layer2", "layer3", "layer4"]   # RadEncoder's Sequential order


def build_encoder(backbone, pretrained, drop_path):
    """timm encoder (num_classes=0) and its input normalisation (mean, std) on [0, 1] inputs."""
    if backbone == RADIMAGENET_R50:
        enc = timm.create_model("resnet50", pretrained=False, num_classes=0, drop_path_rate=drop_path)
        if pretrained:
            sd = torch.load(None, map_location="cpu", weights_only=True)      # keys backbone.<i>.<rest>
            sd = {RAD_TRUNK[int(k.split(".")[1])] + "." + k.split(".", 2)[2]: v for k, v in sd.items()}
            enc.load_state_dict(sd, strict=True)
        return enc, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)          # RadImageNet input: x / 127.5 - 1
    enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
    cfg = enc.pretrained_cfg
    return enc, cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225))


class StudyNet(nn.Module):
    def __init__(self, backbone="convnext_tiny.fb_in22k", n_classes=12, hidden=256, dropout=0.2, drop_path=0.1,
                 pretrained=True, n_views=1):
        super().__init__()
        self.encoder, mean, std = build_encoder(backbone, pretrained, drop_path)
        c = self.encoder.num_features
        self.register_buffer("mean", torch.tensor(mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(std)[None, :, None, None])
        self.lstm = nn.LSTM(c, hidden, batch_first=True, bidirectional=True)
        self.frame_attn = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.plane_embed = nn.Embedding(len(PLANES), 2 * hidden)
        self.fs_embed = nn.Embedding(2, 2 * hidden)
        self.view_embed = nn.Embedding(n_views, 2 * hidden) if n_views > 1 else None
        self.drop = nn.Dropout(dropout)
        self.series_logit = nn.Linear(2 * hidden, n_classes)
        self.series_attn = nn.Linear(2 * hidden, n_classes)

    def forward(self, x, plane, fs, mask, view=None):
        B, N, D, S, _ = x.shape
        v = x[mask]
        idx = torch.arange(D, device=x.device)
        trip = torch.stack([idx.sub(1).clamp(min=0), idx, idx.add(1).clamp(max=D - 1)], 1)
        frames = v[:, trip].reshape(-1, 3, S, S)
        f = self.encoder((frames - self.mean) / self.std).reshape(len(v), D, -1)
        return self.forward_features(f, plane, fs, mask, view)

    def forward_features(self, f, plane, fs, mask, view=None):
        """f (M, D, c): the encoder's frame features of the M real series (mask order) -> logits (B, n_classes)."""
        B, N = mask.shape
        h, _ = self.lstm(f)
        a = self.frame_attn(h).squeeze(-1).softmax(1)
        s = (a[..., None] * h).sum(1)
        s = s + self.plane_embed(plane[mask]) + self.fs_embed(fs[mask])
        if self.view_embed is not None:
            s = s + self.view_embed(view[mask])
        s = self.drop(s)
        full = s.new_zeros(B, N, s.shape[-1])
        full[mask] = s
        logits = self.series_logit(full)
        w = self.series_attn(full).masked_fill(~mask[..., None], float("-inf")).softmax(1)
        return (w * logits).sum(1)


class WindowNet(nn.Module):
    def __init__(self, backbone="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", n_classes=12, drop=0.2, drop_path=0.1,
                 pretrained=True, img_size=None):
        super().__init__()
        kw = dict(img_size=img_size) if img_size else {}
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path, **kw)
        c = self.encoder.num_features
        cfg = self.encoder.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)))[None, :, None, None])
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)))[None, :, None, None])
        self.plane_embed = nn.Embedding(len(PLANES), c)
        self.fs_embed = nn.Embedding(2, c)
        self.norm = nn.LayerNorm(c)
        self.attention = nn.Sequential(nn.Linear(c, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n_classes))
        self.classifier_weight = nn.Parameter(torch.zeros(n_classes, c))
        self.classifier_bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x, plane, fs, mask, view=None):
        v = x[mask]
        f = torch.cat([self.encoder((c - self.mean) / self.std) for c in v.split(WINDOW_CHUNK)])
        return self.forward_features(f, plane, fs, mask)

    def forward_features(self, f, plane, fs, mask):
        """f (M, c): the encoder's features of the M real windows (mask order) -> logits (B, n_classes)."""
        B, K = mask.shape
        f = f + self.plane_embed(plane[mask]) + self.fs_embed(fs[mask])
        h = f.new_zeros(B, K, f.shape[-1])
        h[mask] = f
        h = self.norm(h)
        a = self.attention(h).masked_fill(~mask[..., None], float("-inf")).softmax(1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.classifier_weight[None]).sum(-1) + self.classifier_bias


class DenseSequenceHead(nn.Module):
    """b1_c1's dense_contiguous_evidence_v1 head; token_dim as a parameter (2048: RadImageNet)."""
    TEMPERATURE, DROPOUT, HIDDEN = 0.5, 0.15, 64

    def __init__(self, token_dim=2048, slots=6, slices=24, targets=12):
        super().__init__()
        self.token_dim, self.slots, self.slices = token_dim, slots, slices
        h = self.HIDDEN
        self.project_weight = nn.Parameter(torch.randn(targets, token_dim, h) * 0.02)
        self.project_bias = nn.Parameter(torch.zeros(targets, h))
        self.slice_weight = nn.Parameter(torch.randn(targets, h) * 0.02)
        self.slice_bias = nn.Parameter(torch.zeros(targets))
        self.position_logits = nn.Parameter(torch.zeros(targets, slices))
        self.sequence3 = nn.Parameter(torch.full((targets, 1, 3), 1.0 / 3.0))
        self.sequence5 = nn.Parameter(torch.full((targets, 1, 5), 1.0 / 5.0))
        self.summary_logits = nn.Parameter(torch.tensor([1.50, 0.75, 0.75, 0.00, -0.25]).repeat(targets, 1))
        self.attention_weight = nn.Parameter(torch.randn(targets, 5) * 0.02)
        self.slot_logits = nn.Parameter(torch.zeros(targets, slots))
        self.output_bias = nn.Parameter(torch.zeros(targets))

    def normalized_lse(self, value):
        t = self.TEMPERATURE
        return t * torch.logsumexp(value / t, dim=-1) - t * math.log(self.slices)

    def forward(self, feature, slot_mask):
        targets = self.output_bias.shape[0]
        feature = F.layer_norm(feature.float(), (self.token_dim,))
        feature = F.dropout(feature, p=self.DROPOUT, training=self.training)
        hidden = torch.einsum("bsld,tdh->bslth", feature, self.project_weight)
        hidden = F.gelu(hidden + self.project_bias[None, None, None])
        hidden = F.dropout(hidden, p=self.DROPOUT, training=self.training)
        raw = torch.einsum("bslth,th->bslt", hidden, self.slice_weight)
        raw = raw + self.slice_bias[None, None, None]
        raw = raw + self.position_logits.T[None, None]
        batch = len(feature)
        sequence = raw.permute(0, 1, 3, 2).reshape(batch * self.slots, targets, self.slices)
        smooth3 = F.conv1d(sequence, self.sequence3, padding=1, groups=targets)
        smooth5 = F.conv1d(sequence, self.sequence5, padding=2, groups=targets)
        raw = sequence.reshape(batch, self.slots, targets, self.slices)
        smooth3 = smooth3.reshape(batch, self.slots, targets, self.slices)
        smooth5 = smooth5.reshape(batch, self.slots, targets, self.slices)
        statistics = torch.stack([
            self.normalized_lse(raw), self.normalized_lse(smooth3), self.normalized_lse(smooth5),
            raw.topk(4, dim=-1, sorted=False).values.mean(dim=-1), raw.mean(dim=-1),
        ], dim=-1)
        series = (statistics * self.summary_logits.softmax(dim=-1)[None, None]).sum(-1)
        attention = self.slot_logits.T[None] + torch.einsum(
            "bstk,tk->bst", statistics, self.attention_weight) / math.sqrt(statistics.shape[-1])
        attention = attention.masked_fill(~slot_mask[:, :, None].bool(), -1e4).softmax(1)
        return (series * attention).sum(1) + self.output_bias[None]


class RadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])

    def forward(self, image):
        return self.backbone(image).mean(dim=(2, 3))


# ----------------------------------------------------------------------------- loading

def crop_z(d):
    """The capsule crop sampling a study model was trained on (crop.json) -> the pipeline's z key (json string)."""
    c = json.load(open(os.path.join(d, "crop.json")))
    return json.dumps({k: c[k] for k in Z_KEYS if c.get(k) is not None}, sort_keys=True)


def load(weights, names, device):
    """Recipe names found under <weights>/study/ -> (models, encoders). Every network runs once per frame: a study /
    MIL model's encoder also serves the dense heads that read its features ("encoder" = its name); encoders used only
    by dense heads (RadImageNet, or a study model outside the recipe) are loaded once."""
    root = os.path.join(weights, "study")
    models, encoders = [], {}

    def add_encoder(name):
        if name in encoders:
            return encoders[name]
        if name == "radimagenet":
            net = RadEncoder()
            net.load_state_dict(torch.load(os.path.join(root, "radimagenet", "ResNet50.pt"), map_location="cpu",
                                           weights_only=True), strict=True)
            e = dict(net=net.to(device).eval(), kind="radimagenet", dim=2048)
        elif os.path.exists(os.path.join(root, name, "encoder.json")):   # a frozen timm encoder (e.g. convergedmachine
            c = json.load(open(os.path.join(root, name, "encoder.json")))  # RadImageNet ResNet-50), c1_features --timm
            net = timm.create_model(c["arch"], pretrained=False, num_classes=0)
            net.load_state_dict(torch.load(os.path.join(root, name, "ckpt_final.pt"), map_location="cpu",
                                           weights_only=True), strict=True)
            e = dict(net=net.to(device).eval(), kind="timm", dim=net.num_features, norm=c["norm"])
        else:
            d = os.path.join(root, name)
            a = json.load(open(os.path.join(d, "args.json")))
            if a.get("model", "study") == "mil":
                net = WindowNet(a["backbone"], pretrained=False)
            else:
                net = StudyNet(a["backbone"], pretrained=False, n_views=2 if a.get("wide_crops") else 1)
            net.load_state_dict(torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu"), strict=True)
            e = dict(net=net.to(device).eval(), kind=a.get("model", "study"), dim=net.encoder.num_features, args=a)
        e["z"] = crop_z(os.path.join(root, name))
        encoders[name] = e
        return e

    for name in sorted(names):
        d = os.path.join(root, name)
        if os.path.exists(os.path.join(d, "c1.json")):
            enc = json.load(open(os.path.join(d, "c1.json")))["encoder"]
            e = add_encoder(enc)
            heads = []
            for sd in torch.load(os.path.join(d, "heads.pt"), map_location="cpu").values():
                h = DenseSequenceHead(token_dim=e["dim"])
                h.load_state_dict(sd, strict=True)
                heads.append(h.to(device).eval())
            models.append(dict(name=name, kind="c1", heads=heads, encoder=enc, z=e["z"], planes=PLANES))
        else:
            e = add_encoder(name)
            a = e["args"]
            models.append(dict(name=name, kind=e["kind"], encoder=name, z=e["z"], planes=a["planes"].split(","),
                               max_series=a["val_max_series"]))
    return models, encoders


# ----------------------------------------------------------------------------- prediction

def study_series(crop_rows, have, planes):
    """Study -> its series indices into crop_rows: sagittal / coronal first, then axial (the crop sets' order in
    training), crop present, plane used by the model."""
    out = {}
    for axial in (False, True):
        for j, s in enumerate(crop_rows):
            if have[j, 0] and s["plane"] in planes and (s["plane"] == "Axial") == axial:
                out.setdefault(s["StudyInstanceUID"], []).append(j)
    return out


def mil_windows(js, depth):
    """A study's MIL windows (series index, frame): every frame of its series in order, MIL_EVAL_WINDOWS evenly
    spaced ones when more (mil.WindowDataset, eval_windows)."""
    pairs = [(j, f) for j in js for f in range(depth)]
    if len(pairs) > MIL_EVAL_WINDOWS:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, MIL_EVAL_WINDOWS).round().astype(int)]
    return pairs


def c1_slots(crop_rows, js):
    """B1-C1's six slots of a study: per (plane, fat-suppressed), the series with most slices, then series UID."""
    out = []
    for plane, fluid in SLOTS:
        cand = [j for j in js if crop_rows[j]["plane"] == plane and int(crop_rows[j].get("fluid") == 1) == fluid]
        out.append(sorted(cand, key=lambda j: (-crop_rows[j]["n_slices"], crop_rows[j]["SeriesInstanceUID"]))[0]
                   if cand else None)
    return out


def _encode(e, x_all, frames, amp):
    """Frame features of one encoder: {(series index, frame): (dim,) float32}, computed in batches of WINDOW_CHUNK.
    Study / MIL encoders read the triplet (f - 1, f, f + 1) with their normalisation (autocast `amp`); RadImageNet
    reads the frame alone, x / 127.5 - 1 on 3 channels, float16 autocast (b1_c1)."""
    frames = sorted(frames)
    D = x_all.shape[2]
    out = {}
    for i in range(0, len(frames), WINDOW_CHUNK):
        part = frames[i:i + WINDOW_CHUNK]
        j = torch.tensor([p[0] for p in part], device=x_all.device)
        f = torch.tensor([p[1] for p in part], device=x_all.device)
        if e["kind"] == "radimagenet":
            img = x_all[j, 0, f].float().div(127.5).sub(1.0)[:, None].expand(-1, 3, -1, -1).contiguous()
            with torch.autocast("cuda", dtype=torch.float16):
                v = e["net"](img).float()
        elif e["kind"] == "timm":           # grey frame x 3; tf: [-1, 1], else ImageNet mean / std (c1_features.prep)
            img = x_all[j, 0, f].float().div(255)[:, None].expand(-1, 3, -1, -1)
            img = img * 2 - 1 if e["norm"] == "tf" else (img - IMAGENET_MEAN.to(img)) / IMAGENET_STD.to(img)
            with torch.autocast(**amp):
                v = e["net"](img.contiguous()).float()
        else:
            trip = torch.stack([(f - 1).clamp(min=0), f, (f + 1).clamp(max=D - 1)], 1)       # (n, 3)
            img = x_all[j[:, None], 0, trip].float().div(255)                              # (n, 3, S, S)
            with torch.autocast(**amp):
                v = e["net"].encoder((img - e["net"].mean) / e["net"].std).float()
        out.update(zip(part, v))
    return out


@torch.no_grad()
def predict_chunk(models, encoders, crop_rows, crops, amp_dtype):
    """-> rows (model, StudyInstanceUID, finding, "Joint", p) for every study with crops."""
    amp = dict(device_type="cuda", dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None)
    plan = []                                       # (model, study, series indices)
    need = {}                                       # encoder -> frames to encode
    for m in models:
        x_all, have = crops[m["z"]]
        D = x_all.shape[2]
        for st, js in study_series(crop_rows, have, m["planes"]).items():
            if m["kind"] == "study":
                js = js[:m["max_series"]]
                fr = [(j, f) for j in js for f in range(D)]
            elif m["kind"] == "mil":
                js = js[:m["max_series"]]
                fr = mil_windows(js, D)
            else:
                js = c1_slots(crop_rows, js)
                fr = [(j, f) for j in js if j is not None for f in range(D)]
            plan.append((m, st, js))
            need.setdefault(m["encoder"], set()).update(fr)
    feats = {name: _encode(encoders[name], crops[encoders[name]["z"]][0], fr, amp) for name, fr in need.items()}
    out = []
    for m, st, js in plan:
        e, f = encoders[m["encoder"]], feats[m["encoder"]]
        dev = crops[m["z"]][0].device
        D = crops[m["z"]][0].shape[2]
        if m["kind"] == "study":
            x = torch.stack([torch.stack([f[(j, k)] for k in range(D)]) for j in js])     # (n, D, c)
            plane, fs = _meta(crop_rows, js, dev)
            mask = torch.ones((1, len(js)), dtype=torch.bool, device=dev)
            with torch.autocast(**amp):
                p = e["net"].forward_features(x, plane[None], fs[None], mask, torch.zeros_like(plane)[None])
        elif m["kind"] == "mil":
            win = mil_windows(js, D)
            x = torch.stack([f[w] for w in win])                                           # (K, c)
            plane, fs = _meta(crop_rows, [j for j, _ in win], dev)
            mask = torch.ones((1, len(win)), dtype=torch.bool, device=dev)
            with torch.autocast(**amp):
                p = e["net"].forward_features(x, plane[None], fs[None], mask)
        else:
            X = torch.zeros((1, len(SLOTS), D, e["dim"]), device=dev)
            M = torch.zeros((1, len(SLOTS)), dtype=torch.bool, device=dev)
            for k, j in enumerate(js):
                if j is not None:                   # features stored as float16 in training (c1_features.py)
                    X[0, k], M[0, k] = torch.stack([f[(j, t)] for t in range(D)]).half().float(), True
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.stack([torch.sigmoid(h(X, M).float()) for h in m["heads"]]).mean(0)
            out += [(m["name"], st, c, "Joint", float(q)) for c, q in zip(CLASSES, p[0].tolist())]
            continue
        out += [(m["name"], st, c, "Joint", float(q)) for c, q in zip(CLASSES, torch.sigmoid(p.float())[0].tolist())]
    return out


def _meta(crop_rows, js, device):
    plane = torch.tensor([PLANES.index(crop_rows[j]["plane"]) for j in js], device=device)
    fs = torch.tensor([int(crop_rows[j].get("fluid") == 1) for j in js], device=device)
    return plane, fs
