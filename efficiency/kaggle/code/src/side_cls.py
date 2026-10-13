"""Knee-side (left / right) classifier, one prediction per series.

A study usually holds one knee, but not always (bilateral studies exist), so the
side is predicted for every series, sagittal, coronal and axial alike, and the
ROI model flips each series to a right knee on its own prediction.

Input: the (D, S, S) = (32, 128, 128) volume of `roi.prepare` in its *stored*
orientation, plus three constant channels for the plane. The left-right axis is
the slice axis for sagittal series and the column axis for coronal and axial
ones; across all training series it always points the same way (sagittal slices
towards the patient's right, coronal / axial columns towards the left), and
inference flips any test series that does not (`infer.prep`). A left knee is
then the mirror image of a right one along that axis, which the network reads
from anatomy (fibular head lateral, medial condyle larger, ...).

Labels: the series' own DICOM Laterality where it is R / L, else the study's
side inferred from the U-Net's compartment boxes (`roi.laterality`, 99% in
agreement with the tag). Augmentation mirrors along the left-right axis and swaps
the label -- a mirrored right knee is a left knee -- so both sides are seen
equally often.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from params import OUTPUT_FOLDER

PLANES = ["Sagittal", "Coronal", "Axial"]
ROI_DIR = os.path.join(OUTPUT_FOLDER, "roi")


def lr_dim(plane):
    """Dimension of a (C, D, S, S) tensor along which left and right swap."""
    return 1 if plane == "Sagittal" else 3


def load_all_volumes():
    """-> ({name: memmap}, index with columns name, row, SeriesInstanceUID, ...)."""
    vols, parts = {}, []
    for name in ("volumes", "volumes_axial"):
        vols[name] = np.load(os.path.join(ROI_DIR, f"{name}.npy"), mmap_mode="r")
        parts.append(pd.read_csv(os.path.join(ROI_DIR, f"{name}.csv")).assign(name=name))
    return vols, pd.concat(parts, ignore_index=True)


def side_labels(index):
    """1 = right, 0 = left, NaN = unknown; and whether it is the series' own DICOM tag."""
    lat = pd.read_csv(os.path.join(OUTPUT_FOLDER, "series_laterality.csv")).set_index("SeriesInstanceUID")
    tag = index.SeriesInstanceUID.map(lat.dicom_laterality)
    side = index.SeriesInstanceUID.map(lat.side)
    use = tag.where(tag.isin(["R", "L"]), side)
    return use.map({"R": 1.0, "L": 0.0}).to_numpy(), tag.isin(["R", "L"]).to_numpy()


class SideDataset(Dataset):
    def __init__(self, vols, index):
        self.vols = vols
        self.index = index.reset_index(drop=True)
        self.y, self.tagged = side_labels(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        r = self.index.iloc[i]
        return {"image": torch.from_numpy(np.array(self.vols[r["name"]][r.row]))[None],
                "plane": torch.tensor(PLANES.index(r.Anatomical_Plane)),
                "y": torch.tensor(self.y[i], dtype=torch.float32),
                "tagged": torch.tensor(bool(self.tagged[i])), "idx": torch.tensor(i)}


def model_input(image, plane):
    """uint8 (N, 1, D, S, S) + plane ids -> float (N, 4, D, S, S) with plane channels."""
    x = image.float() / 255
    onehot = torch.nn.functional.one_hot(plane, 3).float()[:, :, None, None, None]
    return torch.cat([x, onehot.expand(-1, -1, *x.shape[2:])], 1)


def _block(cin, cout, stride):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


class SideNet(nn.Module):
    """Small anisotropic 3D CNN: in-plane downsampling first, global pooling, one logit (right)."""

    def __init__(self, width=16, dropout=0.2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv3d(4, w, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                                  nn.BatchNorm3d(w), nn.ReLU(inplace=True))
        self.body = nn.Sequential(_block(w, 2 * w, (1, 2, 2)), _block(2 * w, 4 * w, 2),
                                  _block(4 * w, 8 * w, 2), _block(8 * w, 16 * w, 2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(16 * w, 1))

    def forward(self, x):
        return self.head(self.body(self.stem(x)))[:, 0]


def augment(image, y, plane, mirror_p=0.5):
    """Mirror along the left-right axis with the label swapped, plus intensity jitter (GPU, uint8 in)."""
    image = image.clone()
    y = y.clone()
    flip = torch.rand(len(y), device=y.device) < mirror_p
    for i in torch.nonzero(flip).flatten().tolist():
        image[i] = image[i].flip(lr_dim(PLANES[int(plane[i])]))
        y[i] = 1 - y[i]
    x = image.float() / 255
    n = len(y)
    gamma = torch.exp(torch.empty(n, device=x.device).uniform_(-0.3, 0.3))[:, None, None, None, None]
    contrast = torch.empty(n, device=x.device).uniform_(0.8, 1.2)[:, None, None, None, None]
    x = ((x.clamp(0, 1) ** gamma - 0.5) * contrast + 0.5).clamp(0, 1)
    shift = [int(s) for s in torch.randint(-8, 9, (2,))]
    x = torch.roll(x, shifts=shift, dims=(3, 4))
    return (x * 255).to(torch.uint8), y
