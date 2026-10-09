"""Does an ROI contain the evidence for its finding? Crop classifiers, run from src/.

    python -m roi.evidence crops --pred round2_pred     # crops of every candidate region
    python -m roi.evidence train --target MCL           # one classifier per region, AUCs

Cross-view agreement cannot test the MCL ROI: it comes from an anatomical prior
that is consistent across views by construction. What can: a classifier that
only sees the inside of the ROI should predict MCL injury (report labels) better
than the same classifier on control regions. Regions, per sagittal / coronal series:

  MCL           the predicted MCL box
  MCL_in6/out6  the same box moved 6 mm towards the joint / away from it
  LCL           its mirror on the lateral side (lateral collateral ligament)
  MedialOA      the medial compartment box (bone included: bone-bruise evidence)
  global        the whole volume (baseline)

Crops come from the 384 px cache (3x the U-Net's resolution: the MCL is a few mm
thick), in canonical right-knee orientation, resized to (16, 64, 64). Each
region's classifier is a small 3D CNN trained on the training studies and scored
by study-level AUC on the validation studies (series averaged): against the
Claude labels binarised at 0.5, and against the radiologist labels of the 58.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from params import CACHE_DIR, OUTPUT_FOLDER
from roi.boxes import _geometry, world_box
from roi.labels import LABEL_DIR
from roi.mcl_prior import world_to_voxel_box

EVID_DIR = os.path.join(OUTPUT_FOLDER, "roi", "evidence")
REGIONS = ["MCL", "MCL_in6", "MCL_out6", "LCL", "MedialOA", "global"]
CROP = (16, 64, 64)


def _shifted(series_uid, box, shape, mm, medial_sign):
    """The box moved `mm` along patient x towards medial (+) in patient space."""
    lo, hi = world_box(series_uid, box["z0"], box["z1"], box["y0"], box["y1"], box["x0"], box["x1"])
    d = np.array([medial_sign * mm, 0, 0])
    return world_to_voxel_box(series_uid, lo + d, hi + d, shape)


def _lcl(series_uid, mcl, lat_oa, med_oa, shape, medial_sign):
    """Mirror of the MCL box about the knee's centre line (between the compartments)."""
    lo, hi = world_box(series_uid, mcl["z0"], mcl["z1"], mcl["y0"], mcl["y1"], mcl["x0"], mcl["x1"])
    ml, mh = world_box(series_uid, med_oa["z0"], med_oa["z1"], med_oa["y0"], med_oa["y1"], med_oa["x0"], med_oa["x1"])
    ll, lh = world_box(series_uid, lat_oa["z0"], lat_oa["z1"], lat_oa["y0"], lat_oa["y1"], lat_oa["x0"], lat_oa["x1"])
    centre = ((ml[0] + mh[0]) / 2 + (ll[0] + lh[0]) / 2) / 2
    x = sorted([2 * centre - lo[0], 2 * centre - hi[0]])
    return world_to_voxel_box(series_uid, np.array([x[0], lo[1], lo[2]]), np.array([x[1], hi[1], hi[2]]), shape)


def _crop(cache, box, n_slices, rows, cols, flip_axis):
    """Box (original voxels) -> (16, 64, 64) uint8 crop of the 384 cache, right-knee orientation."""
    import torch
    import torch.nn.functional as F

    n_c, size = cache.shape[0], cache.shape[1]
    s = max(rows, cols)
    oy, ox = (s - rows) // 2, (s - cols) // 2
    f = size / s
    zc = (lambda z: z * (n_c - 1) / max(n_slices - 1, 1)) if n_c != n_slices else (lambda z: z)
    z0, z1 = int(np.floor(zc(box["z0"]))), int(np.ceil(zc(box["z1"])))
    y0, y1 = int((box["y0"] + oy) * f), int(np.ceil((box["y1"] + 1 + oy) * f))
    x0, x1 = int((box["x0"] + ox) * f), int(np.ceil((box["x1"] + 1 + ox) * f))
    c = cache[max(z0, 0):min(z1, n_c - 1) + 1, max(y0, 0):min(y1, size), max(x0, 0):min(x1, size)]
    if c.size == 0:
        return np.zeros(CROP, np.uint8)
    t = torch.from_numpy(np.ascontiguousarray(c).astype(np.float32))[None, None]
    t = F.interpolate(t, size=CROP, mode="trilinear", align_corners=False)[0, 0].numpy()
    if flip_axis is not None:
        t = np.flip(t, flip_axis)
    return np.ascontiguousarray(t.round().clip(0, 255).astype(np.uint8))


def _series_crops(args):
    import torch

    torch.set_num_threads(1)
    r, boxes, side = args
    cache = np.load(os.path.join(CACHE_DIR, f"{r['SeriesInstanceUID']}.npy"))
    shape = (r["n_slices"], r["rows"], r["cols"])
    medial_sign = 1.0 if side == "R" else -1.0
    flip = (0 if r["Anatomical_Plane"] == "Sagittal" else 2) if side == "L" else None
    regions = {"MCL": boxes["MCL"], "MedialOA": boxes["Medial OA"],
               "global": dict(z0=0, z1=r["n_slices"] - 1, y0=0, y1=r["rows"] - 1, x0=0, x1=r["cols"] - 1)}
    uid = r["SeriesInstanceUID"]
    regions["MCL_in6"] = _shifted(uid, boxes["MCL"], shape, -6.0, medial_sign)    # towards the joint
    regions["MCL_out6"] = _shifted(uid, boxes["MCL"], shape, 6.0, medial_sign)    # away from it
    regions["LCL"] = _lcl(uid, boxes["MCL"], boxes["Lateral OA"], boxes["Medial OA"], shape, medial_sign)
    out = np.zeros((len(REGIONS),) + CROP, np.uint8)
    for i, k in enumerate(REGIONS):
        if regions.get(k) is not None:
            out[i] = _crop(cache, regions[k], *shape, flip)
    return out


def build_crops(pred_name):
    from roi.laterality import series_side

    os.makedirs(EVID_DIR, exist_ok=True)
    vols = pd.read_csv(os.path.join(OUTPUT_FOLDER, "roi", "volumes.csv"))
    pred = pd.read_csv(os.path.join(LABEL_DIR, f"{pred_name}.csv"))
    side = series_side()
    need = ["MCL", "Medial OA", "Lateral OA"]
    b = {(r["SeriesInstanceUID"], r["class"]): {k: int(r[k]) for k in ("z0", "z1", "y0", "y1", "x0", "x1")}
         for r in pred[pred["class"].isin(need)].to_dict("records")}
    jobs, keep = [], []
    for r in vols.to_dict("records"):
        u = r["SeriesInstanceUID"]
        if all((u, c) in b for c in need) and u in side:
            jobs.append((r, {c: b[(u, c)] for c in need}, side[u]))
            keep.append(r)
    arr = np.lib.format.open_memmap(os.path.join(EVID_DIR, "crops.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(jobs), len(REGIONS)) + CROP)
    with ProcessPoolExecutor(64) as ex:
        for i, c in enumerate(ex.map(_series_crops, jobs, chunksize=16)):
            arr[i] = c
    arr.flush()
    pd.DataFrame(keep).to_csv(os.path.join(EVID_DIR, "crops.csv"), index=False)
    print(f"-> {len(jobs)} series x {len(REGIONS)} regions {CROP} in {EVID_DIR}")


def train_region(region, target, device, epochs=12, seed=0):
    """Train a crop classifier for one region; -> dict of validation AUCs."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    from knee_cls.data import load_tables, make_splits

    torch.manual_seed(seed)
    np.random.seed(seed)
    crops = np.load(os.path.join(EVID_DIR, "crops.npy"), mmap_mode="r")
    info = pd.read_csv(os.path.join(EVID_DIR, "crops.csv"))
    studies, _ = load_tables()
    st = studies.set_index("StudyInstanceUID")
    y = info.StudyInstanceUID.map(st[f"target_{target}"]).to_numpy(np.float32)
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    val = info.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids)).to_numpy()
    k = REGIONS.index(region)
    x_all = torch.from_numpy(np.ascontiguousarray(crops[:, k]))           # N 16 64 64 uint8, in RAM

    def block(ci, co, st_):
        return nn.Sequential(nn.Conv3d(ci, co, 3, stride=st_, padding=1, bias=False), nn.BatchNorm3d(co),
                             nn.ReLU(inplace=True), nn.Conv3d(co, co, 3, padding=1, bias=False),
                             nn.BatchNorm3d(co), nn.ReLU(inplace=True))

    model = nn.Sequential(block(1, 16, 1), block(16, 32, 2), block(32, 64, 2), block(64, 128, 2),
                          nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    tr_idx = np.flatnonzero(~val)
    steps = epochs * (len(tr_idx) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    yt = torch.from_numpy(y)
    for _ in range(epochs):
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm) - 63, 64):
            b = perm[i:i + 64]
            x = x_all[b].to(device).float()[:, None] / 255
            x = x * torch.empty(len(b), 1, 1, 1, 1, device=device).uniform_(0.85, 1.15)   # intensity jitter
            loss = F.binary_cross_entropy_with_logits(model(x)[:, 0], yt[b].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    va_idx = np.flatnonzero(val)
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(model(x_all[va_idx[i:i + 256]].to(device).float()[:, None] / 255)[:, 0]).cpu().numpy()
                            for i in range(0, len(va_idx), 256)])
    v = info.iloc[va_idx].assign(p=p).groupby("StudyInstanceUID").p.mean()
    claude = (st.loc[v.index, f"target_{target}"] >= 0.5).astype(int)
    gt = st.loc[v.index, f"gt_{target}"]
    res = {"region": region, "auc_claude": roc_auc_score(claude, v), "n_pos_claude": int(claude.sum())}
    g = gt.notna()
    if g.sum() and gt[g].nunique() == 2:
        res.update(auc_gt58=roc_auc_score(gt[g], v[g]), n_pos_gt=int(gt[g].sum()))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--pred", default="round2_pred")
    t = sub.add_parser("train")
    t.add_argument("--target", default="MCL")
    t.add_argument("--regions", default=",".join(REGIONS))
    t.add_argument("--seeds", type=int, default=3)
    t.add_argument("--gpu", default="0")
    a = p.parse_args()
    if a.cmd == "crops":
        return build_crops(a.pred)
    import torch

    device = torch.device(f"cuda:{a.gpu}")
    rows = [train_region(r, a.target, device, seed=s) for r in a.regions.split(",") for s in range(a.seeds)]
    df = pd.DataFrame(rows)
    out = df.groupby("region", sort=False).agg(["mean", "std"]).round(3)
    print(out.to_string())
    df.to_csv(os.path.join(EVID_DIR, f"auc_{a.target}.csv"), index=False)


if __name__ == "__main__":
    main()
