"""ROI crop classifiers -> study predictions -> submission.csv, run from src/ after `infer.seg`.

    python -m infer.classify --out_dir /tmp/knee/ --weights_root <dir> --recipe <dir>/recipe.json --submission submission.csv

1. Axial boxes. The ROI U-Net only boxes sagittal and coronal series; an axial
   series gets, per ROI, the boxes of the same knee's sagittal and coronal
   series mapped into its grid and intersected in patient space
   (`roi.consensus.registered_box`), the knee being the study and the series'
   predicted side.
2. Crops. Every (series, ROI) crop is cut from the 384 px cache exactly as in
   training (`crops.prepare.crop_box`). It is mirrored to a right knee when
   the knee is left XOR the series' left-right axis runs against the training
   convention (`conv_flip`, see `infer.prep`): the cache is in the series'
   stored orientation, the training crops were in the training convention.
3. Models. Each weights folder holds a crop model (args.json + ckpt_final.pt)
   for one group of findings and a set of planes; it predicts every crop of its
   planes and findings, and a study's probability for a finding is the mean over
   its series.
4. Per-finding combination, from a JSON map {finding: [model folder, ...]}
   (--recipe): one model -> its probability; several -> the mean of their
   ranks over the test studies (models trained on different groups are
   calibrated differently; ranks weight them equally).
5. Submission. One row per study, the 12 findings. Findings the recipe does not
   cover keep the sample-submission value 0.5.

Writes <out_dir>/crops.npy, crops.csv, roi_study_preds.csv and --submission.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import knee_registration as kr
from params import CLASSES

ROI_CLASSES = CLASSES[:7] + ["Baker's"]
BOX = ["z0", "z1", "y0", "y1", "x0", "x1"]


def axial_boxes(series, rois, side):
    """Registered ROI boxes for the axial series (DataFrame like rois.csv)."""
    from roi.consensus import _world, registered_box

    r = rois.copy()
    r["world"] = _world(r)
    r["knee"] = r.StudyInstanceUID + "|" + r.side
    by_knee = dict(tuple(r.groupby("knee")))
    out = []
    ax = series[(series.status == "ok") & (series.plane == "Axial")]
    for t in ax.itertuples():
        src = by_knee.get(f"{t.StudyInstanceUID}|{side.get(t.SeriesInstanceUID)}")
        if src is None:
            continue
        for cls, s in src.groupby("class"):
            box = registered_box(t, s.assign(confidence=s.confidence.fillna(0)))
            if box is not None:
                out.append(dict(StudyInstanceUID=t.StudyInstanceUID, SeriesInstanceUID=t.SeriesInstanceUID,
                                plane="Axial", **{"class": cls}, **box))
    return pd.DataFrame(out)


def _crop_series(args):
    from crops.prepare import crop_box

    torch.set_num_threads(1)
    cache_path, boxes, n_slices, rows, cols, flip, sagittal = args
    cache = np.load(cache_path)
    out = np.zeros((len(ROI_CLASSES), 12, 128, 128), np.uint8)
    have = np.zeros(len(ROI_CLASSES), bool)
    for k, c in enumerate(ROI_CLASSES):
        if c in boxes:
            out[k] = crop_box(cache, boxes[c], n_slices, rows, cols, flip, sagittal)
            have[k] = True
    return out, have


def build_crops(out_dir, series, boxes, side, workers):
    ok = series[series.status == "ok"].reset_index(drop=True)
    by_series = {}
    for r in boxes.to_dict("records"):
        by_series.setdefault(r["SeriesInstanceUID"], {})[r["class"]] = {k: int(r[k]) for k in BOX}
    jobs, rows = [], []
    for r in ok.itertuples():
        if r.SeriesInstanceUID not in by_series:
            continue
        flip = (side.get(r.SeriesInstanceUID) == "L") ^ bool(r.conv_flip)
        jobs.append((os.path.join(out_dir, "cache", f"{r.SeriesInstanceUID}.npy"), by_series[r.SeriesInstanceUID],
                     int(r.n_slices), int(r.rows), int(r.cols), flip, r.plane == "Sagittal"))
        rows.append(dict(row=len(rows), SeriesInstanceUID=r.SeriesInstanceUID, StudyInstanceUID=r.StudyInstanceUID,
                         Anatomical_Plane=r.plane, src="test"))
    arr = np.lib.format.open_memmap(os.path.join(out_dir, "crops.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(jobs), len(ROI_CLASSES), 12, 128, 128))
    have = np.zeros((len(jobs), len(ROI_CLASSES)), bool)
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        for i, (c, h) in enumerate(ex.map(_crop_series, jobs, chunksize=4)):
            arr[i], have[i] = c, h
    arr.flush()
    info = pd.DataFrame(rows)
    info.to_csv(os.path.join(out_dir, "crops.csv"), index=False)
    return np.load(os.path.join(out_dir, "crops.npy"), mmap_mode="r"), info, have


@torch.no_grad()
def predict_model(weights, crops, info, have, device="cuda"):
    """Study x finding probabilities of one crop model."""
    from torch.utils.data import DataLoader

    from crops.train import GROUPS, CropDataset, Crop25D
    from infer.utils import amp_dtype

    a = json.load(open(os.path.join(weights, "args.json")))
    classes = GROUPS[a["group"]]
    sel = info[info.Anatomical_Plane.isin(a["planes"].split(","))]
    dummy = {c: pd.Series(0.0, index=sel.StudyInstanceUID.unique()) for c in classes}
    ds = CropDataset({"test": crops}, sel, classes, dummy)
    # Only (series, ROI) pairs that actually got a box.
    keep = have[ds.items.row.to_numpy(), ds.items.roi.to_numpy()]
    ds.items = ds.items[keep].reset_index(drop=True)
    sd = torch.load(os.path.join(weights, "ckpt_final.pt"), map_location="cpu")
    model = Crop25D(a["backbone"], len(classes), pretrained=False, img_size=a.get("img_size") or None,
                    hidden=a.get("hidden", 256), dropout=0.0, drop_path=0.0, head=a.get("head", "lstm"))
    n_planes = sd["plane_embed.weight"].shape[0]
    if n_planes != model.plane_embed.num_embeddings:
        model.plane_embed = nn.Embedding(n_planes, model.plane_embed.embedding_dim)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    ps = []
    for b in DataLoader(ds, batch_size=64, num_workers=8, pin_memory=True):
        x = b["x"].to(device, non_blocking=True).float() / 255
        with torch.autocast("cuda", dtype=amp_dtype()):
            ps.append(torch.sigmoid(model(x, b["roi"].to(device), b["plane"].to(device)).float()).cpu().numpy())
    it = ds.items.assign(p=np.concatenate(ps) if ps else [])
    return it.groupby(["StudyInstanceUID", "cls"]).p.mean().unstack()


def run(out_dir, weights_root, recipe, submission, workers=None):
    t0 = time.time()
    series = pd.read_csv(os.path.join(out_dir, "series.csv"))
    lat = pd.read_csv(os.path.join(out_dir, "series_laterality.csv"))
    side = dict(zip(lat.SeriesInstanceUID, lat.side))
    rois = pd.read_csv(os.path.join(out_dir, "rois.csv"))
    with open(os.path.join(out_dir, "geometry.json")) as f:
        kr.GEOMETRY_OVERRIDE.update(json.load(f))
    ax = axial_boxes(series, rois, side)
    boxes = pd.concat([rois[["StudyInstanceUID", "SeriesInstanceUID", "plane", "class"] + BOX], ax], ignore_index=True)
    crops, info, have = build_crops(out_dir, series, boxes, side, workers)
    t1 = time.time()
    models = sorted({m for ms in recipe.values() for m in ms})
    preds = {m: predict_model(os.path.join(weights_root, m), crops, info, have) for m in models}
    t2 = time.time()
    study = pd.DataFrame(index=series.StudyInstanceUID.drop_duplicates().values)
    for c, ms in recipe.items():
        cols = [preds[m][c].reindex(study.index) for m in ms]
        if len(cols) == 1:
            study[c] = cols[0]
        else:
            study[c] = sum(col.rank(pct=True) for col in cols) / len(cols)
    study.rename_axis("StudyInstanceUID").to_csv(os.path.join(out_dir, "roi_study_preds.csv"))
    sub = pd.DataFrame({"StudyInstanceUID": study.index})
    for c in CLASSES:
        sub[c] = sub.StudyInstanceUID.map(study[c]) if c in study else np.nan
    sub[CLASSES] = sub[CLASSES].fillna(0.5)                                  # uncovered findings / studies
    sub.to_csv(submission, index=False)
    print(f"-> {submission}: {len(sub)} studies; recipe covers {list(recipe)}; 0.5 elsewhere. "
          f"{len(ax)} axial boxes, {len(info)} series cropped in {t1 - t0:.0f}s, {len(models)} models in {t2 - t1:.0f}s")
    return sub


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--weights_root", required=True, help="folder holding the crop-model run folders")
    p.add_argument("--recipe", required=True, help="JSON {finding: [model folder, ...]}")
    p.add_argument("--submission", default="submission.csv")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()
    with open(a.recipe) as f:
        recipe = json.load(f)
    run(a.out_dir, a.weights_root, recipe, a.submission, a.workers)


if __name__ == "__main__":
    main()
