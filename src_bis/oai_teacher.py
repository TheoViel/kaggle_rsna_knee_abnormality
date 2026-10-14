"""OAI-only models as teachers: their predictions on our crops -> soft labels mixed with the report labels, run from src/.

    PYTHONPATH=. python ../src_bis/oai_teacher.py --runs 2026-10-13/35,2026-10-13/42 --w 0.5

The OAI-only models (trained on MOAKS image gradings only, never on our studies) predict the medial / lateral
meniscus of every study of the f_s1n8 crop set (sagittal + coronal series, mean over series and runs). Writes
  ../output/bis/oai_teacher_men.csv        teacher probabilities per study
  ../output/bis/pl_oaiteach_men<w>.csv     pl_round2 with both menisci = (1 - w) x pl_round2 + w x teacher
                                           (use as --labels pl:../bis/pl_oaiteach_men<w>)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from crops.train import GROUPS, PLANES, build_model, crop_index
from params import OUTPUT_FOLDER

MEN = ["Medial Meniscus", "Lateral Meniscus"]


@torch.no_grad()
def predict(run, arr, info, dev, bs=256):
    d = os.path.join("..", "logs", "crops_run", run)
    a = json.load(open(os.path.join(d, "args.json")))
    group = GROUPS[a["group"]]
    net = build_model(a, len(group))
    net.load_state_dict(torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu"))
    net = net.to(dev).eval()
    out = {}
    for c in MEN:
        roi, rid = crop_index(c), group.index(c)
        ps = []
        for i in range(0, len(info), bs):
            rows = info.row.values[i:i + bs]
            x = torch.from_numpy(np.ascontiguousarray(arr[rows, roi])).to(dev).float() / 255
            pl = torch.tensor([PLANES.index(p) for p in info.Anatomical_Plane.values[i:i + bs]], device=dev)
            r = torch.full((len(rows),), rid, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ps.append(torch.sigmoid(net(x, r, pl).float()).cpu().numpy())
        out[c] = np.concatenate(ps)
    return pd.DataFrame(out, index=info.StudyInstanceUID.values).groupby(level=0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="2026-10-13/35,2026-10-13/42")
    ap.add_argument("--w", type=float, default=0.5)
    a = ap.parse_args()
    crops = os.path.join(OUTPUT_FOLDER, "crops")
    arr = np.load(os.path.join(crops, "f_s1n8.npy"), mmap_mode="r")
    info = pd.read_csv(os.path.join(crops, "f_s1n8.csv"))
    info = info[info.Anatomical_Plane.isin(["Sagittal", "Coronal"])].reset_index(drop=True)
    dev = torch.device("cuda")
    teacher = sum(predict(r, arr, info, dev) for r in a.runs.split(",")) / len(a.runs.split(","))
    out = os.path.join(OUTPUT_FOLDER, "bis")
    teacher.rename_axis("StudyInstanceUID").to_csv(os.path.join(out, "oai_teacher_men.csv"))
    pl = pd.read_csv(os.path.join(crops, "pl_round2.csv")).set_index("StudyInstanceUID")
    for c in MEN:
        t = teacher[c].reindex(pl.index)
        pl[c] = ((1 - a.w) * pl[c] + a.w * t).fillna(pl[c])
    name = f"pl_oaiteach_men{int(a.w * 100)}"
    pl.reset_index().to_csv(os.path.join(out, f"{name}.csv"), index=False)
    print(f"teacher on {len(teacher)} studies; mean {teacher.mean().round(3).to_dict()} -> {name}.csv "
          f"(--labels pl:../bis/{name})")


if __name__ == "__main__":
    main()
