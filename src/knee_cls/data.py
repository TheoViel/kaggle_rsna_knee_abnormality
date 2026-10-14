"""Study table, splits, dataset and collate.

One sample is one study: every series it has (up to a cap), each as a few
frames of 3 adjacent slices (see `KneeStudyDataset`). Labels:

  - `target`: the Claude report labels, soft probabilities. What we train on.
  - `gt`:     the radiologist labels from train.csv, only for the 58 studies
              that have them, NaN elsewhere. What we validate on.

The 58 GT studies are never trained on. Out of the remaining studies a random
`holdout_frac` can also be held out: the 58-study AUC has a std of a few points
per class, so a larger, report-labelled holdout is the steadier signal for
comparing runs (scored against the Claude labels binarised at 0.5).

The collate flattens frames across the whole batch -- the encoder only ever sees
real frames -- and keeps the (series, z) and (study, series) indices the model
needs to put them back.

Augmentation runs on the GPU (`gpu_augment`): one random affine per series, so
the slices of a series stay registered with each other, which the z attention
relies on. Resizing from the cache resolution to `img_size` happens in the same
`grid_sample`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from params import (CACHE_DIR, CLASSES, CLAUDE_LABELS_CSV, PLANES,
                    SERIES_INDEX_CSV, TRAIN_CSV)


def load_tables():
    """-> (studies, series). `studies` has target_*/gt_* columns and `is_gt`."""
    claude = pd.read_csv(CLAUDE_LABELS_CSV)
    train = pd.read_csv(TRAIN_CSV).drop(columns=["Report"])
    studies = claude.rename(columns={c: f"target_{c}" for c in CLASSES})
    studies = studies.merge(
        train.rename(columns={c: f"gt_{c}" for c in CLASSES}),
        on="StudyInstanceUID", how="left")
    studies["is_gt"] = studies[[f"gt_{c}" for c in CLASSES]].notna().all(1)

    series = pd.read_csv(SERIES_INDEX_CSV)
    series["cache_path"] = CACHE_DIR + series.SeriesInstanceUID + ".npy"
    return studies.reset_index(drop=True), series


def make_splits(studies, holdout_frac=0.1, seed=0):
    """-> (train_ids, gt_val_ids, holdout_ids)."""
    gt = studies.loc[studies.is_gt, "StudyInstanceUID"].tolist()
    rest = studies.loc[~studies.is_gt, "StudyInstanceUID"].to_numpy()
    rng = np.random.default_rng(seed)
    rest = rest[rng.permutation(len(rest))]
    n_hold = int(round(holdout_frac * len(rest)))
    return rest[n_hold:].tolist(), gt, rest[:n_hold].tolist()


def make_folds(studies, n_folds, seed=42):
    """StudyInstanceUID -> fold (0 .. n_folds - 1) over every training study, for out-of-fold predictions
    (pseudo-labels). The 58 radiologist studies and the rest are spread over the folds separately."""
    rng = np.random.default_rng(seed)
    out = {}
    for part in (studies.is_gt, ~studies.is_gt):
        ids = np.sort(studies.loc[part, "StudyInstanceUID"].to_numpy())
        ids = ids[rng.permutation(len(ids))]
        out.update({s: i % n_folds for i, s in enumerate(ids)})
    return out


def series_meta_vector(plane, fluid, fat):
    v = np.zeros(len(PLANES) + 4, dtype=np.float32)
    v[PLANES.index(plane)] = 1
    v[len(PLANES) + int(fluid)] = 1
    v[len(PLANES) + 2 + int(fat)] = 1
    return v


class KneeStudyDataset(Dataset):
    """One item per study: for each series, `n_frames` frames of 3 adjacent slices.

    Frame sampling, per series of n slices:
      - the outer `z_trim` fraction at each end is dropped: the first and last
        slices are mostly outside the joint;
      - `n_frames` centres are spread evenly over what is left (in training each
        is jittered within its stride);
      - a frame is slices (c - d, c, c + d), d = `frame_offset`, stacked as the 3
        channels of one image. 6 frames thus show the CNN 18 slices.
    The centre indices are returned so attention can be mapped back to slices.
    """

    def __init__(self, studies, series, study_ids, train=False, max_series=8,
                 n_frames=6, frame_offset=1, z_trim=0.1):
        self.train = train
        self.max_series = max_series
        self.n_frames = n_frames
        self.frame_offset = frame_offset
        self.z_trim = z_trim
        st = studies.set_index("StudyInstanceUID").loc[study_ids]
        self.study_ids = list(study_ids)
        self.targets = st[[f"target_{c}" for c in CLASSES]].to_numpy(np.float32)
        self.gts = st[[f"gt_{c}" for c in CLASSES]].to_numpy(np.float32)
        grouped = series.groupby("StudyInstanceUID")
        self.series = []
        for uid in self.study_ids:
            g = grouped.get_group(uid).sort_values(
                ["Anatomical_Plane", "Fluid_Sensitive", "SeriesInstanceUID"])
            self.series.append([
                {"uid": r.SeriesInstanceUID, "path": r.cache_path,
                 "plane": r.Anatomical_Plane,
                 "meta": series_meta_vector(r.Anatomical_Plane, r.Fluid_Sensitive,
                                            r.Fat_Suppression)}
                for r in g.itertuples()])

    @property
    def summary(self):
        n_series = sum(len(s) for s in self.series)
        return f"{len(self)} studies, {n_series} series"

    def __len__(self):
        return len(self.study_ids)

    def frame_centres(self, n):
        """`n_frames=None` puts a frame on every slice (attention export)."""
        d = self.frame_offset
        lo = max(d, int(round(self.z_trim * n)))
        hi = min(n - 1 - d, n - 1 - int(round(self.z_trim * n)))
        if hi < lo:
            lo = hi = n // 2
        if self.n_frames is None:
            return np.arange(lo, hi + 1)
        edges = np.linspace(lo, hi + 1, self.n_frames + 1)
        if self.train:
            c = edges[:-1] + np.random.rand(self.n_frames) * np.diff(edges)
        else:
            c = (edges[:-1] + edges[1:]) / 2
        return np.clip(np.floor(c).astype(int), lo, hi)

    def frame_indices(self, n):
        c = self.frame_centres(n)
        d = self.frame_offset
        idx = np.clip(np.stack([c - d, c, c + d], 1), 0, n - 1)    # F 3
        return c, idx

    def __getitem__(self, i):
        series = self.series[i]
        if len(series) > self.max_series:
            if self.train:
                keep = np.sort(np.random.choice(len(series), self.max_series, replace=False))
            else:
                keep = np.arange(self.max_series)
            series = [series[k] for k in keep]
        frames, centres, rel, metas = [], [], [], []
        for s in series:
            vol = np.load(s["path"], mmap_mode="r")
            c, idx = self.frame_indices(len(vol))
            frames.append(np.ascontiguousarray(vol[idx.ravel()]).reshape(
                len(c), 3, *vol.shape[1:]))
            centres.append(c)
            rel.append(c / max(len(vol) - 1, 1))
            metas.append(s["meta"])
        return {
            "frames": frames,
            "centres": centres,
            "rel": rel,
            "meta": np.stack(metas),
            "series_uids": [s["uid"] for s in series],
            "target": self.targets[i],
            "gt": self.gts[i],
            "study_uid": self.study_ids[i],
        }


def collate_studies(items):
    """Flatten the frames of every series of every study into one image batch."""
    imgs, s_idx, z_idx, z_rel, centres = [], [], [], [], []
    series_study, series_pos, metas, series_uids = [], [], [], []
    s = 0
    max_z = max_series = 0
    for b, it in enumerate(items):
        for m, (fr, c, r) in enumerate(zip(it["frames"], it["centres"], it["rel"])):
            n = len(fr)
            imgs.append(torch.from_numpy(fr))
            s_idx.append(torch.full((n,), s, dtype=torch.long))
            z_idx.append(torch.arange(n))
            z_rel.append(torch.from_numpy(r).float())
            centres.append(torch.from_numpy(c).long())
            series_study.append(b)
            series_pos.append(m)
            max_z = max(max_z, n)
            s += 1
        metas.append(torch.from_numpy(it["meta"]))
        series_uids.append(it["series_uids"])
        max_series = max(max_series, len(it["frames"]))
    return {
        "images": torch.cat(imgs),                           # N 3 H W uint8
        "slice_series": torch.cat(s_idx),
        "slice_z": torch.cat(z_idx),
        "slice_zrel": torch.cat(z_rel),
        "slice_centre": torch.cat(centres),                  # source slice of each frame
        "series_study": torch.tensor(series_study),
        "series_pos": torch.tensor(series_pos),
        "series_meta": torch.cat(metas),
        "n_studies": len(items),
        "max_z": max_z,
        "max_series": max_series,
        "target": torch.from_numpy(np.stack([it["target"] for it in items])),
        "gt": torch.from_numpy(np.stack([it["gt"] for it in items])),
        "study_uid": [it["study_uid"] for it in items],
        "series_uids": series_uids,
    }


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def gpu_augment(batch, img_size, aug=None):
    """uint8 frames -> float (N, C, img_size, img_size), augmented per series.

    `aug=None` only resizes. Otherwise one affine (rotation, isotropic scale,
    translation) and one intensity transform (contrast, brightness, gamma) are
    drawn per *series* and applied to all its slices, plus a little per-slice
    noise.
    """
    x = batch["images"].float() / 255.0
    if aug is None:
        if x.shape[-1] != img_size:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              antialias=True, align_corners=False)
        return x

    dev = x.device
    S = batch["series_study"].shape[0]
    si = batch["slice_series"]

    def u(lo, hi, n=S):
        return torch.empty(n, device=dev).uniform_(lo, hi)

    ang = u(-aug["rot_deg"], aug["rot_deg"]) * math.pi / 180
    on = (torch.rand(S, device=dev) < aug["affine_p"]).float()
    ang = ang * on
    scale = 1 + (u(*aug["scale_range"]) - 1) * on
    tx, ty = u(-aug["shift"], aug["shift"]) * on, u(-aug["shift"], aug["shift"]) * on
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], -1),
                         torch.stack([sin, cos, ty], -1)], 1)          # S 2 3
    grid = F.affine_grid(theta[si], (x.shape[0], x.shape[1], img_size, img_size),
                         align_corners=False)
    # Antialias first when shrinking, grid_sample does not.
    if x.shape[-1] > img_size * 1.25:
        x = F.interpolate(x, scale_factor=img_size * 1.25 / x.shape[-1],
                          mode="bilinear", antialias=True, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                      align_corners=False)

    on = (torch.rand(S, device=dev) < aug["intensity_p"]).float()
    contrast = 1 + u(-aug["contrast"], aug["contrast"]) * on
    bright = u(-aug["brightness"], aug["brightness"]) * on
    gamma = torch.exp(u(-aug["gamma"], aug["gamma"]) * on)
    x = x.clamp(0, 1) ** gamma[si, None, None, None]
    x = (x - 0.5) * contrast[si, None, None, None] + 0.5 + bright[si, None, None, None]
    if aug.get("noise_std", 0):
        x = x + torch.randn_like(x) * aug["noise_std"] * torch.rand(
            x.shape[0], 1, 1, 1, device=dev)
    return x.clamp(0, 1)
