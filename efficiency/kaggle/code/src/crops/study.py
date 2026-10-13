"""Study-level ROI models: the series-level crop model plus a learnable aggregation over series.

The series model (`crops.train.Crop25D` / `Crop3D`) turns every (series, ROI) crop into a
vector. For one (study, finding), the vectors of all its series -- a variable number,
sagittal / coronal / axial, fluid-sensitive or not -- form a set, and an aggregator turns
the set into one logit:

  mean         average of the vectors (the study-level counterpart of averaging series
               probabilities; the control for the learned aggregators);
  attn         gated attention pooling (Ilse et al., 2018): a learned weight per series, so a
               finding seen on one view only can dominate; starts as the mean;
  transformer  2 transformer layers over the series (they can compare views), then gated
               attention pooling;
  lse          log-sum-exp pooling of the per-series logits with a learned temperature,
               between the mean of logits (r -> 0) and their max (r -> inf).

Training: initialised from a trained series-level run, study-level soft labels, random
series dropout (each series dropped with probability p, at least one kept) so the
aggregator sees incomplete studies.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from crops.train import PLANES, CropDataset


class StudyDataset(Dataset):
    """One sample = every (series, ROI) crop of a study for the group's findings."""

    def __init__(self, crops, info, classes, targets, series_drop=0.0, max_series=None):
        self.crops, self.classes = crops, classes
        items = CropDataset(crops, info, classes, targets).items
        self.items = items
        self.studies = list(items.groupby("StudyInstanceUID", sort=False).indices.items())
        self.targets = {c: targets[c] for c in classes}
        self.series_drop, self.max_series = series_drop, max_series

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        study, idx = self.studies[i]
        it = self.items.iloc[idx]
        series = it.SeriesInstanceUID.unique()
        keep = series
        if self.series_drop:
            m = np.random.rand(len(series)) >= self.series_drop
            if not m.any():
                m[np.random.randint(len(series))] = True
            keep = series[m]
        if self.max_series and len(keep) > self.max_series:
            keep = np.random.choice(keep, self.max_series, replace=False)
        it = it[it.SeriesInstanceUID.isin(keep)]
        x = np.stack([self.crops[r.src][r.row, r.roi] for r in it.itertuples()])
        y = torch.tensor([float(self.targets[c].get(study, np.nan)) for c in self.classes])
        return {"x": torch.from_numpy(x), "rid": torch.tensor(it.rid.to_numpy()),
                "plane": torch.tensor([PLANES.index(p) for p in it.Anatomical_Plane]),
                "y": torch.nan_to_num(y, nan=0.0), "ymask": ~torch.isnan(y), "study": study}


def collate(batch):
    """Concatenate the studies' crops; group = (study in batch, finding) for the aggregation."""
    k = len(batch[0]["y"])
    out = {"x": torch.cat([b["x"] for b in batch]), "rid": torch.cat([b["rid"] for b in batch]),
           "plane": torch.cat([b["plane"] for b in batch]),
           "group": torch.cat([b["rid"] + i * k for i, b in enumerate(batch)]),
           "y": torch.stack([b["y"] for b in batch]).flatten(), "ymask": torch.stack([b["ymask"] for b in batch]).flatten(),
           "study": [b["study"] for b in batch], "n_groups": len(batch) * k}
    return out


def _pad(v, group, n_groups):
    """(N, d) vectors + group ids -> (G, L, d) padded, (G, L) mask (True = real), present (G,)."""
    counts = torch.bincount(group, minlength=n_groups)
    L = int(counts.max())
    order = torch.argsort(group, stable=True)
    g = group[order]
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(g), device=v.device) - start[g]
    out = v.new_zeros(n_groups, L, v.shape[-1])
    out[g, pos] = v[order]
    mask = torch.zeros(n_groups, L, dtype=torch.bool, device=v.device)
    mask[g, pos] = True
    return out, mask, counts > 0


class GatedAttention(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, 1)
        nn.init.zeros_(self.w.weight)                     # uniform weights at start: the mean
        nn.init.zeros_(self.w.bias)

    def forward(self, x, mask):
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf")).softmax(-1)
        return (a[..., None] * x).sum(1), a


class StudyModel(nn.Module):
    def __init__(self, series_model, agg="attn", layers=2):
        super().__init__()
        self.series, self.agg = series_model, agg
        d = series_model.fc.in_features
        if agg in ("attn", "transformer"):
            self.pool = GatedAttention(d)
        if agg == "transformer":
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 2 * d, dropout=0.1, batch_first=True,
                                                                       norm_first=True), layers)
        if agg == "lse":
            self.log_r = nn.Parameter(torch.tensor(-1.0))      # r = exp(log_r), starts near the mean of logits

    def forward(self, x, rid, plane, group, n_groups):
        """-> logits (G,) for the (study, finding) groups, and which groups have crops."""
        v = self.series.embed(x, rid, plane).float()
        if self.agg == "lse":
            l = self.series.fc(self.series.drop(v)).squeeze(-1)
            lp, mask, present = _pad(l[:, None], group, n_groups)
            r = self.log_r.exp()
            z = (r * lp.squeeze(-1)).masked_fill(~mask, float("-inf"))
            logit = (torch.logsumexp(z, -1) - torch.log(mask.sum(-1).clamp(min=1).float())) / r
            return logit, present
        vp, mask, present = _pad(v, group, n_groups)
        if self.agg == "mean":
            f = (vp * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            if self.agg == "transformer":
                safe = mask | ~present[:, None]            # all-padding rows would give NaNs
                vp = self.tf(vp, src_key_padding_mask=~safe)
            f, _ = self.pool(vp, mask | ~present[:, None])
        return self.series.fc(self.series.drop(f)).squeeze(-1), present


def study_predictions(logit, present, batch, classes):
    rows = []
    p = torch.sigmoid(logit.float()).cpu().numpy()
    pres = present.cpu().numpy()
    k = len(classes)
    for i, s in enumerate(batch["study"]):
        for j, c in enumerate(classes):
            if pres[i * k + j]:
                rows.append((s, c, float(p[i * k + j])))
    return rows


# ------------------------------------------------- cross-compartment (series-level sets)


class SeriesSetDataset(Dataset):
    """One sample = one series with its crops for every finding of the group (one per ROI)."""

    def __init__(self, crops, info, classes, targets):
        self.crops, self.classes = crops, classes
        self.items = CropDataset(crops, info, classes, targets).items
        self.series = list(self.items.groupby("SeriesInstanceUID", sort=False).indices.values())

    def __len__(self):
        return len(self.series)

    def __getitem__(self, i):
        it = self.items.iloc[self.series[i]]
        x = np.stack([self.crops[r.src][r.row, r.roi] for r in it.itertuples()])
        return {"x": torch.from_numpy(x), "rid": torch.tensor(it.rid.to_numpy()),
                "plane": torch.tensor([PLANES.index(p) for p in it.Anatomical_Plane]),
                "y": torch.tensor(it.target.to_numpy(), dtype=torch.float32),
                "study": it.StudyInstanceUID.tolist(), "series": it.SeriesInstanceUID.tolist(), "cls": it.cls.tolist()}


def collate_series(batch):
    return {"x": torch.cat([b["x"] for b in batch]), "rid": torch.cat([b["rid"] for b in batch]),
            "plane": torch.cat([b["plane"] for b in batch]), "y": torch.cat([b["y"] for b in batch]),
            "group": torch.cat([torch.full((len(b["rid"]),), i) for i, b in enumerate(batch)]),
            "n_groups": len(batch), "study": sum((b["study"] for b in batch), []),
            "series": sum((b["series"] for b in batch), []), "cls": sum((b["cls"] for b in batch), [])}


class CrossCompartmentModel(nn.Module):
    """Series model + transformer across the ROI crops of one series, one logit per crop.

    The transformer's output is added to the crop vectors through a zero-initialised
    projection, so training starts from the series model's own predictions.
    """

    def __init__(self, series_model, layers=2):
        super().__init__()
        self.series = series_model
        d = series_model.fc.in_features
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 2 * d, dropout=0.1, batch_first=True,
                                                                   norm_first=True), layers)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, rid, plane, group, n_groups):
        v = self.series.embed(x, rid, plane).float()
        vp, mask, _ = _pad(v, group, n_groups)
        h = self.out(self.tf(vp, src_key_padding_mask=~mask))
        order = torch.argsort(group, stable=True)
        counts = torch.bincount(group, minlength=n_groups)
        start = torch.cumsum(counts, 0) - counts
        pos = torch.empty_like(group)
        pos[order] = torch.arange(len(group), device=group.device) - start[group[order]]
        v = v + h[group, pos]
        return self.series.fc(self.series.drop(v)).squeeze(-1)
