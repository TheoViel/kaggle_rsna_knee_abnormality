"""Re-score a trained mt3d run on the validation studies, with test-time augmentation, run from src_bis/.

    python -m mt3d.evaluate --run 2026-10-10/7 --tta offsets

TTA (`--tta`, comma-separated, averaged as probabilities):
  none     the run's own validation (centre frame offset);
  offsets  hybrid models: every frame offset (16 frames over 32 slices: even and odd slices);
  scale    in-plane zoom 0.9 and 1.1 around the crop centre (plus 1.0).
Writes val_study_preds_tta.csv next to the run and prints the AUC table (58 / holdout).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crops.train import study_auc
from knee_cls.data import load_tables, make_splits
from main_mt3d import training_targets
from mt3d.anatomy import ANATOMY
from mt3d.train import MTDataset, load_crops, prepare_batch
from params import CLASSES
from roi.extract import ROI_CLASSES

MT = "../logs/bis/mt3d/"


def zoom_rois(z):
    """`--zoom`: "all", or comma-separated ROI names or indices into ROI_CLASSES (e.g. 2,3: the menisci)."""
    if z == "all":
        return list(range(len(ROI_CLASSES)))
    return [int(c) if c.isdigit() else ROI_CLASSES.index(c) for c in z.split(",") if c]


def build(a):
    n_anat = len(ANATOMY) if a.get("w_anat", 0) > 0 else 0
    classes = list(ROI_CLASSES) + ([c for c in CLASSES if c not in ROI_CLASSES] if a.get("findings") == "all12" else [])
    if a.get("arch") == "hybrid":
        from mt3d.hybrid import Hybrid
        m = Hybrid(a["backbone"], frames=a["frames"], dec=a["dec"], dim=a["dim"], tf_layers=a["tf_layers"],
                   n_anat=n_anat, pretrained=False, pool=a.get("pool", "pred"), n_cls=len(classes),
                   levels=[int(v) for v in str(a.get("levels", "1,2,3")).split(",")],
                   anat_tokens=a.get("anat_tokens", False), zpos=a.get("zpos", False), nb=a.get("nb", 1),
                   zoom=zoom_rois(a.get("zoom", "")), zoom_size=a.get("zoom_size", 128),
                   zoom_frames=a.get("zoom_frames", 8))
    else:
        from mt3d.model import MT3D
        m = MT3D(n_dec=a["n_dec"], head=a["head"], dim=a["dim"], tf_layers=a["tf_layers"], queries=a["queries"],
                 pretrained=False, n_anat=n_anat)
    return m, classes


def zoom(x, t, s):
    """In-plane zoom by s around the centre, image and box masks alike."""
    if s == 1:
        return x, t
    n = x.shape[0]
    theta = torch.zeros(n, 3, 4, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1 / s
    theta[:, 2, 2] = 1
    g = F.affine_grid(theta, x.shape, align_corners=False)
    return (F.grid_sample(x, g, align_corners=False),
            F.grid_sample(t, F.affine_grid(theta, t.shape, align_corners=False), align_corners=False))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--tta", default="none,offsets")
    p.add_argument("--batch_size", type=int, default=16)
    a = p.parse_args()
    d = MT + a.run + "/"
    args = json.load(open(d + "args.json"))
    model, classes = build(args)
    model.load_state_dict(torch.load(d + "ckpt_final.pt", map_location="cpu"))
    dev = torch.device("cuda")
    model = model.to(dev).eval()
    arr, boxes, info = load_crops(args["crops"])
    info = info[info.Anatomical_Plane.isin(args["planes"].split(","))]
    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    val = info[info.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids))]
    ds = MTDataset(arr, boxes, val, training_targets("claude", studies, classes), classes)
    dl = DataLoader(ds, batch_size=a.batch_size, num_workers=8)
    tabs = {}
    for mode in a.tta.split(","):
        probs = []
        for b in dl:
            x, t, _ = prepare_batch(b, dev, args["size"])
            pl = b["plane"].to(dev)
            outs = []
            variants = [(1.0, None)]
            if mode == "offsets" and args.get("arch") == "hybrid":
                variants = [(1.0, o) for o in range(max(x.shape[2] // model.frames, 1))]
            elif mode == "scale":
                variants = [(s, None) for s in (0.9, 1.0, 1.1)]
            for s, off in variants:
                xs, ts = zoom(x, t, s)
                kw = {"offset": off} if off is not None else {}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(xs, pl, ts, **kw)
                outs.append(torch.sigmoid(o["cls"].float()))
            probs.append(torch.stack(outs).mean(0).cpu().numpy())
        p_ = np.concatenate(probs)
        wide = pd.DataFrame(p_, columns=classes).assign(StudyInstanceUID=ds.info.StudyInstanceUID.values)
        long = wide.melt(id_vars=["StudyInstanceUID"], var_name="cls", value_name="p")
        tab, study_tab = study_auc(long, studies)
        tabs[mode] = tab
        study_tab.to_csv(d + f"val_study_preds_{mode}.csv")
        roi = tab.loc[[c for c in ROI_CLASSES]]
        print(f"{a.run} tta={mode:8s} gt8 {roi['gt'].mean():.4f} hold8 {roi['hold'].mean():.4f} "
              f"hold7 {roi.drop('Lateral OA')['hold'].mean():.4f}")


if __name__ == "__main__":
    main()
