"""Per-finding heads on frozen crop-model features (src_study/emb_train.py) as recipe models of infer.pipeline.

<weights>/embheads/<name>/: config.json (emb_train.py's args: finding, head, sources, crop_cls, dims / token_dim, ...),
heads.pt (one state dict per seed). The source crop models (<weights>/crops/<model>/) run as usual; for the crops
the heads read, the pipeline also keeps each crop's vector (Crop25D.embed) and per-frame encoder features
(forward_with_features, the same pass as its logit). Per study:
  set    tokens = the crop vectors of the finding's crops of every source over the study's series (+ plane,
         fat-suppression, source, ROI embeddings), per-token logit + attention -> study logit;
  dense  B1-C1's DenseSequenceHead on the frame features of one source's crops: six plane x fat-suppression slots,
         the series with most slices (then series UID) in each.
The model's probability is the mean over its seeds. Copies of src_study/emb_train.py (SetHead, slot rule).
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PLANES = ["Sagittal", "Coronal", "Axial"]
SLOTS = [("Sagittal", 1), ("Coronal", 1), ("Axial", 1), ("Sagittal", 0), ("Coronal", 0), ("Axial", 0)]


class SetHead(nn.Module):
    def __init__(self, dims, n_roi, hidden=128, dropout=0.2):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden)) for d in dims])
        self.plane = nn.Embedding(3, hidden)
        self.fs = nn.Embedding(2, hidden)
        self.src = nn.Embedding(len(dims), hidden)
        self.roi = nn.Embedding(n_roi, hidden)
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.logit = nn.Linear(hidden, 1)
        self.attn = nn.Linear(hidden, 1)

    def forward(self, xs, meta, mask):
        h = 0
        for k, (x, p) in enumerate(zip(xs, self.proj)):
            h = h + p(x) * (meta[..., 0] == k)[..., None]
        h = h + self.src(meta[..., 0]) + self.plane(meta[..., 1]) + self.fs(meta[..., 2]) + self.roi(meta[..., 3])
        h = self.mlp(h)
        a = self.attn(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(1)
        return (a * self.logit(h).squeeze(-1)).sum(1)


def forward_with_features(net, x, roi, plane):
    """A Crop25D (late fusion, lstm / attn / transformer head) -> (logit, crop vector, frame features): the ops of
    Crop25D.embed + fc in one pass; the frame features are the encoder's pre-logits per triplet, before the ROI /
    plane embeddings (src_study/emb_extract.py)."""
    assert net.fusion != "early" and net.head_type != "map3d", "unsupported crop model for embedding heads"
    b, d, s, _ = x.shape
    emb = net.roi_embed(roi) + net.plane_embed(plane)
    idx = torch.arange(d, device=x.device)
    trip = torch.stack([idx.sub(1).clamp(min=0), idx, idx.add(1).clamp(max=d - 1)], 1)
    frames = x[:, trip].reshape(b * d, 3, s, s)
    if net.img_size and net.img_size != s:
        frames = F.interpolate(frames, size=(net.img_size, net.img_size), mode="bilinear", align_corners=False)
    net._d = d
    maps = net.encoder.forward_features((frames - net.mean) / net.std)
    f0 = net.encoder.forward_head(maps, pre_logits=True).reshape(b, d, -1)
    f = f0 + emb[:, None]
    if net.rnn is not None:
        f, _ = net.rnn(f)
    elif net.tf is not None:
        f = net.tf(net.proj(f) + net.pos[:, :d])
    w = net.attn(f).squeeze(-1).softmax(1)
    e = (w[..., None] * f).sum(1)
    return net.fc(net.drop(e)).squeeze(-1), e, f0


def load(weights, names, device):
    """Embedding heads named in the recipe -> (heads, needs): needs = {source crop model: set of crop classes whose
    features the heads read}."""
    from infer.study_models import DenseSequenceHead

    heads, needs = [], {}
    for name in sorted(names):
        d = os.path.join(weights, "embheads", name)
        c = json.load(open(os.path.join(d, "config.json")))
        assert not c.get("context") and not c.get("rad"), f"{name}: context / RadImageNet tokens are not supported"
        cls = c.get("crop_cls") or c["finding"]
        sources = c["sources"].split(",")
        states = torch.load(os.path.join(d, "heads.pt"), map_location="cpu")
        nets = []
        for sd in states:
            if c["head"] == "dense":
                net = DenseSequenceHead(token_dim=c["token_dim"], slices=c["slices"])
            else:
                net = SetHead(c["dims"], c["n_roi"], hidden=c["hidden"])
            net.load_state_dict(sd, strict=True)
            nets.append(net.to(device).eval())
        heads.append(dict(name=name, finding=c["finding"], head=c["head"], cls=cls, sources=sources, nets=nets,
                          rois=c.get("rois"), target=c.get("target"), dims=c.get("dims")))
        for m in sources:                       # crop classes of the model whose crops give `cls` (e.g. "X|fracture")
            needs.setdefault(m, set()).add(cls)
    return heads, needs


@torch.no_grad()
def predict(heads, store, crop_rows, device):
    """store: {(model, SeriesInstanceUID, cls): (crop vector, frame features)} for the chunk -> rows (name, study,
    finding, "Joint", p)."""
    meta = {s["SeriesInstanceUID"]: s for s in crop_rows}
    out = []
    for h in heads:
        by = {}
        for (m, sid, c), v in store.items():                # the head's crops: its sources, its crop class
            if m in h["sources"] and c == h["cls"]:
                by.setdefault(meta[sid]["StudyInstanceUID"], []).append((h["sources"].index(m), sid, c, v))
        for st, items in by.items():
            if h["head"] == "dense":
                its = sorted(items, key=lambda t: (-meta[t[1]]["n_slices"], t[1]))
                D, c_dim = its[0][3][1].shape
                X = torch.zeros((1, len(SLOTS), D, c_dim), device=device)
                M = torch.zeros((1, len(SLOTS)), dtype=torch.bool, device=device)
                for k, (plane, fluid) in enumerate(SLOTS):
                    cand = [t for t in its if meta[t[1]]["plane"] == plane and int(meta[t[1]].get("fluid") == 1) == fluid]
                    if cand:
                        X[0, k], M[0, k] = cand[0][3][1].half().float(), True       # stored float16 in training
                with torch.autocast("cuda", dtype=torch.float16):
                    p = np.mean([torch.sigmoid(n(X, M)[0, h["target"]].float()).item() for n in h["nets"]])
            else:
                xs = [torch.zeros((1, len(items), d), device=device) for d in h["dims"]]
                mt = torch.zeros((1, len(items), 4), dtype=torch.long, device=device)
                for i, (k, sid, c, v) in enumerate(items):
                    xs[k][0, i] = v[0].half().float()
                    mt[0, i] = torch.tensor([k, PLANES.index(meta[sid]["plane"]), int(meta[sid].get("fluid") == 1),
                                             h["rois"].index(c)])
                mask = torch.ones((1, len(items)), dtype=torch.bool, device=device)
                p = np.mean([torch.sigmoid(n(xs, mt, mask)[0]).item() for n in h["nets"]])
            out.append((h["name"], st, h["finding"], "Joint", float(p)))
    return out
