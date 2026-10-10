"""Volumes and partial box labels for the ROI U-Net.

One item = one sagittal or coronal series: its (D, S, S) volume, a (K, D, S, S)
mask with each labelled class's box filled, and which classes are labelled.
Unlabelled classes are not negatives -- the attention was too diffuse to say --
so the loss masks them out (`labelled`).

Canonical orientation: every knee is shown as a right knee. In index space the
left-right axis is the slice axis for sagittal series (slices run towards -x for
every series here) and the column axis for coronal ones (+x), so a left knee is
flipped along that axis (`flip_axis`): medial is then always at the same end,
and the network does not have to work out the side from anatomy. Labels are
flipped with the image, predictions flipped back (`roi.train.predict`). The side
is each series' own, from the side classifier (output/series_laterality.csv,
`side_model`): the DICOM tag is wrong on ~1-2% of tagged series.

Splits follow the classifier's (`knee_cls.data.make_splits`): validation is the
58 radiologist-labelled studies plus the classifier's holdout, training the rest.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from params import OUTPUT_FOLDER
from roi.extract import ROI_CLASSES
from roi.geometry import box_mask, to_volume

ROI_DIR = os.path.join(OUTPUT_FOLDER, "roi")


def load_volumes():
    vols = np.load(os.path.join(ROI_DIR, "volumes.npy"), mmap_mode="r")
    index = pd.read_csv(os.path.join(ROI_DIR, "volumes.csv"))
    return vols, index


def knee_flips(index):
    """Per row of `index`: True for left knees, which are flipped to look like right ones.

    Each series' own side from the side classifier (`roi.laterality.series_side`).
    """
    from roi.laterality import series_side

    return np.array(index.SeriesInstanceUID.map(series_side()).eq("L"), dtype=bool)


def flip_axis(sagittal):
    """Axis of a (D, S, S) volume along which left and right swap."""
    return 0 if sagittal else 2


def split_series(index, holdout_frac=0.1, seed=0):
    from knee_cls.data import load_tables, make_splits

    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=holdout_frac, seed=seed)
    val = index.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids))
    return index[~val], index[val]


class RoiDataset(Dataset):
    def __init__(self, vols, index, labels=None):
        self.vols = vols
        self.index = index.reset_index(drop=True)
        self.flip = knee_flips(self.index)
        _, depth, size, _ = vols.shape
        self.depth, self.size = depth, size
        self.edges = {}                      # row -> {k: volume-grid edges}
        if labels is not None:
            info = self.index.set_index("SeriesInstanceUID")
            for r in labels[labels.SeriesInstanceUID.isin(info.index)].to_dict("records"):
                i = info.loc[r["SeriesInstanceUID"]]
                e = to_volume(r, i.n_slices, i.n_cache, i.rows, i.cols, depth, size)
                self.edges.setdefault(int(i.row), {})[ROI_CLASSES.index(r["class"])] = e

    @property
    def summary(self):
        n = sum(len(v) for v in self.edges.values())
        return f"{len(self)} series, {n} labelled (series, class) pairs"

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        r = self.index.iloc[i]
        row = int(r.row)
        k = len(ROI_CLASSES)
        target = np.zeros((k, self.depth, self.size, self.size), np.uint8)
        labelled = np.zeros(k, bool)
        for c, e in self.edges.get(row, {}).items():
            target[c] = box_mask(e, self.depth, self.size)
            labelled[c] = True
        image = np.array(self.vols[row])
        sagittal = r.Anatomical_Plane == "Sagittal"
        if self.flip[i]:
            ax = flip_axis(sagittal)
            image = np.ascontiguousarray(np.flip(image, ax))
            target = np.ascontiguousarray(np.flip(target, ax + 1))
        return {"image": torch.from_numpy(image)[None],                       # 1 D S S uint8
                "target": torch.from_numpy(target), "labelled": torch.from_numpy(labelled),
                "sagittal": torch.tensor(sagittal), "flipped": torch.tensor(bool(self.flip[i])),
                "row": torch.tensor(row)}


def gpu_augment(image, target, sagittal, aug):
    """Same random transform on image and target, per sample, on the GPU.

    image (N, 1, D, S, S) float in [0, 1], target (N, K, D, S, S) float.
      - left-right mirror (`mirror_p`, 0 by default): reverse the slices of a
        sagittal series, the columns of a coronal one. Off now that volumes are
        in canonical (right-knee) orientation: mirroring would undo it;
      - in-plane affine (rotation, isotropic scale, shift), the same on every
        slice of a volume;
      - intensity: gamma, contrast, brightness, noise.
    """
    n = image.shape[0]
    dev = image.device
    flip = torch.rand(n, device=dev) < aug["mirror_p"]
    for i in torch.nonzero(flip).flatten().tolist():
        # Per-sample tensors are (C, D, y, x): slices for sagittal, columns for coronal.
        dim = 1 if bool(sagittal[i]) else 3
        image[i] = image[i].flip(dim)
        target[i] = target[i].flip(dim)

    def u(lo, hi):
        return torch.empty(n, device=dev).uniform_(lo, hi)

    on = (torch.rand(n, device=dev) < aug["affine_p"]).float()
    ang = u(-aug["rot_deg"], aug["rot_deg"]) * math.pi / 180 * on
    scale = 1 + (u(*aug["scale_range"]) - 1) * on
    tx, ty = u(-aug["shift"], aug["shift"]) * on, u(-aug["shift"], aug["shift"]) * on
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    # affine_grid's 3D matrix acts on (x, y, z); z is left alone.
    theta = torch.stack([torch.stack([cos, -sin, zero, tx], -1),
                         torch.stack([sin, cos, zero, ty], -1),
                         torch.stack([zero, zero, one, zero], -1)], 1)
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    target = (F.grid_sample(target, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False) > 0.5).float()

    on = (torch.rand(n, device=dev) < aug["intensity_p"]).float()[:, None, None, None, None]
    gamma = torch.exp(u(-aug["gamma"], aug["gamma"]))[:, None, None, None, None]
    contrast = (1 + u(-aug["contrast"], aug["contrast"]))[:, None, None, None, None]
    bright = u(-aug["brightness"], aug["brightness"])[:, None, None, None, None]
    out = (image.clamp(0, 1) ** gamma - 0.5) * contrast + 0.5 + bright
    out = out + torch.randn_like(out) * aug["noise_std"]
    image = on * out + (1 - on) * image
    return image.clamp(0, 1), target
