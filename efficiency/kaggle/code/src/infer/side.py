"""Knee side of every series, run from src/ after `infer.prep`.

    python -m infer.side --out_dir /tmp/knee/ --weights <side run dir>

Writes <out_dir>/series_laterality.csv: SeriesInstanceUID, StudyInstanceUID,
plane, p_right, side (R / L), dicom_laterality (the tag, for reference only).

Every series gets its own prediction -- a study can hold both knees -- and the
model's side is used, not the tag: on the validation studies the classifier and
the tag disagree on ~2% of tagged series, and in every such sagittal / coronal
series the U-Net's own compartment geometry sided with the classifier, i.e. the
tag was wrong.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from infer.utils import amp_dtype

from side_cls import PLANES, SideNet, model_input


@torch.no_grad()
def predict_sides(volumes, planes, weights, device="cuda", batch=64):
    sd = torch.load(os.path.join(weights, "ckpt_final.pt"), map_location=device)
    width = sd["stem.0.weight"].shape[0]
    model = SideNet(width=width).to(device).eval()
    model.load_state_dict(sd)
    plane_ids = torch.tensor([PLANES.index(p) for p in planes])
    out = []
    for i in range(0, len(volumes), batch):
        x = torch.from_numpy(np.asarray(volumes[i:i + batch]))[:, None].to(device)
        with torch.autocast(device.split(":")[0], dtype=amp_dtype()):
            out.append(torch.sigmoid(model(model_input(x, plane_ids[i:i + batch].to(device))).float()).cpu())
    return torch.cat(out).numpy()


def run(out_dir, weights, device="cuda"):
    series = pd.read_csv(os.path.join(out_dir, "series.csv"))
    ok = series[series.status == "ok"]
    vols = np.load(os.path.join(out_dir, "volumes.npy"), mmap_mode="r")
    p = predict_sides(vols[ok.row.to_numpy()], ok.plane.tolist(), weights, device)
    lat = ok[["SeriesInstanceUID", "StudyInstanceUID", "plane"]].copy()
    lat["p_right"] = p
    lat["side"] = np.where(p > 0.5, "R", "L")
    lat["dicom_laterality"] = ok.dicom_laterality.fillna("").to_numpy()
    lat.to_csv(os.path.join(out_dir, "series_laterality.csv"), index=False)
    per_study = lat.groupby("StudyInstanceUID").side.nunique()
    tagged = lat[lat.dicom_laterality.isin(["R", "L"])]
    print(f"-> {len(lat)} series: {lat.side.value_counts().to_dict()}; "
          f"{int((per_study > 1).sum())} studies with both knees; "
          f"agreement with the DICOM tag where present {np.mean(tagged.side == tagged.dicom_laterality):.3f} "
          f"({len(tagged)} series)")
    return lat


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--weights", required=True)
    a = p.parse_args()
    run(a.out_dir, a.weights)


if __name__ == "__main__":
    main()
