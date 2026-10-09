"""ROI boxes of every sagittal and coronal series, run from src/ after `infer.side`.

    python -m infer.seg --out_dir /tmp/knee/ --weights <ROI U-Net run dir>

Per series: the volume (training left-right convention, see `infer.prep`) is
flipped to a right knee when its predicted side is left, run through the ROI
U-Net, and the probabilities are flipped back -- all the way to the series' own
stored orientation (convention flip XOR side flip, both along the left-right
axis). Then exactly one box per class (`roi.train.prob_to_boxes`), converted to
original voxel indices (`roi.geometry.from_volume`).

Per box confidence and repair, within one knee -- the series of a study with the
same predicted side, so a bilateral study never mixes its knees:
  - confidence = p_mean x (0.5 + 0.5 x support) (`roi.consensus.add_confidence`);
  - failures (peak probability < 0.5) are replaced by the registered box from the
    knee's other series (`roi.consensus.fill_by_registration`).

Writes <out_dir>/rois.csv, one row per (series, class): StudyInstanceUID,
SeriesInstanceUID, plane, side, class, z0, z1, y0, y1, x0, x1 (inclusive, slice
indices in the order of instances.csv), conf, p_mean, support, iou_other_plane,
iou_same_plane, confidence, source ("unet" or "registered").
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from infer.utils import amp_dtype

import knee_registration as kr
from roi.extract import ROI_CLASSES
from roi.geometry import from_volume
from roi.train import build_model, prob_to_boxes

ROI_PLANES = ("Sagittal", "Coronal")
BOX = ["z0", "z1", "y0", "y1", "x0", "x1"]


def _lr_axis(plane):
    return 0 if plane == "Sagittal" else 2          # of a (D, S, S) volume


@torch.no_grad()
def predict_boxes(vols, rows, weights, device="cuda", batch=16, threshold=0.5):
    sd = torch.load(os.path.join(weights, "ckpt_final.pt"), map_location=device)
    model = build_model(num_classes=sd["decoder.seg_layers.4.weight"].shape[0]).to(device).eval()
    model.load_state_dict(sd)
    classes = ROI_CLASSES[:sd["decoder.seg_layers.4.weight"].shape[0]]
    out = []
    for i in range(0, len(rows), batch):
        part = rows.iloc[i:i + batch]
        flips = (part.conv_flip.astype(bool) ^ part.side.eq("L")).to_numpy()
        x = np.stack([np.flip(vols[r], _lr_axis(p)) if f else np.asarray(vols[r])
                      for r, p, f in zip(part.row, part.plane, flips)])
        xt = torch.from_numpy(np.ascontiguousarray(x))[:, None].to(device).float() / 255
        with torch.autocast(device.split(":")[0], dtype=amp_dtype()):
            prob = torch.sigmoid(model(xt).float()).cpu().numpy()
        for j, r in enumerate(part.itertuples()):
            p = np.flip(prob[j], _lr_axis(r.plane) + 1) if flips[j] else prob[j]
            for k, (vb, conf, p_mean) in enumerate(prob_to_boxes(p, threshold)):
                out.append(dict(StudyInstanceUID=r.StudyInstanceUID, SeriesInstanceUID=r.SeriesInstanceUID,
                                plane=r.plane, side=r.side, **{"class": classes[k]}, conf=conf, p_mean=p_mean,
                                **from_volume(vb, r.n_slices, r.n_cache, r.rows, r.cols, *p.shape[1:3])))
    return pd.DataFrame(out)


def run(out_dir, weights, device="cuda"):
    from roi.consensus import _world, add_confidence, fill_by_registration

    t0 = time.time()
    series = pd.read_csv(os.path.join(out_dir, "series.csv"))
    lat = pd.read_csv(os.path.join(out_dir, "series_laterality.csv"))
    rows = series[(series.status == "ok") & series.plane.isin(ROI_PLANES)].merge(
        lat[["SeriesInstanceUID", "side"]], on="SeriesInstanceUID")
    with open(os.path.join(out_dir, "geometry.json")) as f:
        kr.GEOMETRY_OVERRIDE.update(json.load(f))
    vols = np.load(os.path.join(out_dir, "volumes.npy"), mmap_mode="r")
    pred = predict_boxes(vols, rows, weights, device)
    t1 = time.time()

    # Consensus within a knee: the study key carries the side.
    knee = lambda d: d.StudyInstanceUID + "|" + d.side  # noqa: E731
    pred["study"] = pred.StudyInstanceUID
    pred["StudyInstanceUID"] = knee(pred)
    pred = add_confidence(pred)
    weak = pred.conf < 0.5
    knees = rows.assign(StudyInstanceUID=knee(rows), Anatomical_Plane=rows.plane)
    trusted = ~weak
    filled, n_filled = fill_by_registration(pred.assign(source="unet"), knees, trusted, weak)
    filled["StudyInstanceUID"] = filled.StudyInstanceUID.str.split("|").str[0]
    filled["side"] = filled.SeriesInstanceUID.map(rows.set_index("SeriesInstanceUID").side)
    cols = ["StudyInstanceUID", "SeriesInstanceUID", "plane", "side", "class"] + BOX + \
        ["conf", "p_mean", "support", "iou_other_plane", "iou_same_plane", "confidence", "source"]
    filled = filled.reindex(columns=cols).sort_values(["StudyInstanceUID", "SeriesInstanceUID", "class"])
    filled.to_csv(os.path.join(out_dir, "rois.csv"), index=False)
    n_fail = int(weak.sum())
    print(f"-> {len(filled)} ROIs for {filled.SeriesInstanceUID.nunique()} series "
          f"({len(rows)} sagittal / coronal) in {time.time() - t0:.0f}s (U-Net {t1 - t0:.0f}s); "
          f"{n_fail} below the 0.5 threshold, {n_filled} of them replaced by registration; "
          f"mean confidence {filled.confidence.mean():.3f}")
    missing = len(rows) * len(set(filled["class"])) - len(filled)
    if missing:
        print(f"  {missing} (series, class) without a box: no other series of the knee to register from")
    return filled


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--weights", required=True)
    a = p.parse_args()
    run(a.out_dir, a.weights)


if __name__ == "__main__":
    main()
