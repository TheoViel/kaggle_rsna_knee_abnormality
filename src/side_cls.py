"""Knee-side (left / right) classifier, one prediction per series.

A study usually holds one knee, but not always (bilateral studies exist), so the
side is predicted for every series, sagittal, coronal and axial alike, and the
ROI model flips each series to a right knee on its own prediction.

Input: the (D, S, S) = (32, 128, 128) volume of `roi.prepare` in its *stored*
orientation, plus three constant channels for the plane. The left-right axis is
the slice axis for sagittal series and the column axis for coronal and axial
ones; across all training series it always points the same way (sagittal slices
towards the patient's right, coronal / axial columns towards the left), and
inference flips any test series that does not (`infer.pipeline`). A left knee is
then the mirror image of a right one along that axis, which the network reads
from anatomy (fibular head lateral, medial condyle larger, ...).

Labels: the series' own DICOM Laterality tag, where it is R / L (about half the
series; the others are not trained on). Augmentation mirrors along the left-right
axis and swaps the label -- a mirrored right knee is a left knee -- so both sides
are seen equally often. The tag is wrong on ~1-2% of tagged series, which the
model overrules: its prediction, not the tag, is the side used everywhere else.

    python side_cls.py tags                     # output/series_laterality.csv: the DICOM tag of every series
    torchrun --standalone --nproc_per_node=4 main_side.py
    python side_cls.py predict --run <run dir>  # adds the model's side (p_right_model, side_model)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from params import OUTPUT_FOLDER

PLANES = ["Sagittal", "Coronal", "Axial"]
ROI_DIR = os.path.join(OUTPUT_FOLDER, "roi")
SIDE_CSV = os.path.join(OUTPUT_FOLDER, "series_laterality.csv")


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
    """1 = right, 0 = left, NaN = untagged; and whether the series is tagged."""
    tag = index.SeriesInstanceUID.map(pd.read_csv(SIDE_CSV).set_index("SeriesInstanceUID").dicom_laterality)
    return tag.map({"R": 1.0, "L": 0.0}).to_numpy(), tag.isin(["R", "L"]).to_numpy()


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


def write_tags():
    """output/series_laterality.csv: SeriesInstanceUID, StudyInstanceUID, plane, dicom_laterality (R / L / NaN)."""
    meta = pd.read_csv(os.path.join(OUTPUT_FOLDER, "metadata.csv"), usecols=["SeriesInstanceUID", "Laterality"])
    tag = meta.dropna().groupby("SeriesInstanceUID").Laterality.first().str.upper().str[0]
    idx = pd.read_csv(os.path.join(OUTPUT_FOLDER, "series_index.csv"))
    df = idx[["SeriesInstanceUID", "StudyInstanceUID", "Anatomical_Plane"]].rename(columns={"Anatomical_Plane": "plane"})
    df["dicom_laterality"] = df.SeriesInstanceUID.map(tag)
    df.to_csv(SIDE_CSV, index=False)
    print(f"-> {SIDE_CSV}: {len(df)} series, tags {df.dicom_laterality.value_counts(dropna=False).to_dict()}")


@torch.no_grad()
def predict_sides(volumes, planes, run_dir, device="cuda", batch=64):
    """P(right knee) of (N, D, S, S) uint8 volumes in the training left-right convention."""
    sd = torch.load(os.path.join(run_dir, "ckpt_final.pt"), map_location=device)
    model = SideNet(width=sd["stem.0.weight"].shape[0]).to(device).eval()
    model.load_state_dict(sd)
    plane_ids = torch.tensor([PLANES.index(p) for p in planes])
    out = []
    for i in range(0, len(volumes), batch):
        x = torch.from_numpy(np.asarray(volumes[i:i + batch]))[:, None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(model_input(x, plane_ids[i:i + batch].to(device))).float()).cpu())
    return torch.cat(out).numpy()


def add_predictions(run_dir):
    """Adds the model's side of every series (p_right_model, side_model) to output/series_laterality.csv."""
    vols, index = load_all_volumes()
    p = {}
    for name, g in index.groupby("name", sort=False):
        p.update(zip(g.SeriesInstanceUID, predict_sides(vols[name][g.row.to_numpy()], g.Anatomical_Plane.tolist(),
                                                         run_dir)))
    lat = pd.read_csv(SIDE_CSV)
    lat["p_right_model"] = lat.SeriesInstanceUID.map(p)
    lat["side_model"] = np.where(lat.p_right_model > 0.5, "R", np.where(lat.p_right_model.notna(), "L", None))
    lat.to_csv(SIDE_CSV, index=False)
    t = lat[lat.dicom_laterality.isin(["R", "L"])]
    print(f"-> {SIDE_CSV}: {lat.side_model.notna().sum()} series predicted; agreement with the DICOM tag "
          f"{np.mean(t.side_model == t.dicom_laterality):.4f} on {len(t)} tagged series")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tags", "predict"])
    ap.add_argument("--run", help="side classifier run folder (predict)")
    a = ap.parse_args()
    write_tags() if a.cmd == "tags" else add_predictions(a.run)
