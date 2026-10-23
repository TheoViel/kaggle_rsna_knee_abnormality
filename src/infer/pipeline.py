"""The whole test-time pipeline in one streaming pass per GPU, run from src/.

    python -m infer.pipeline --data_dir <dir with test_series/> --weights <dir> --out_dir /tmp/knee/ \
        --submission submission.csv --gpus 0,1

<weights> (rsna-knee-weights-1) holds side/ and roi/ (run folders), crops/<run>/ (args.json + ckpt_final.pt) and
recipe.json ({finding: [crop run, ...]}).

Steps, organised for a small machine (Kaggle: 2 x T4, 4 CPU cores):
  0. input normalisation (`infer.sanitize`): plane names, and unusual DICOMs (compressed, multi-frame, RGB, several
     files per slice position, no PixelSpacing) rewritten as plain files;
  1. the studies are split between the GPUs, one process each, balanced on file count. CPU work is only reading the
     DICOMs, in DataLoader workers: the headers of every file (slice order, geometry) and the pixels of the <= 64
     slices the cache keeps;
  2. everything else runs on the GPU, chunk by chunk of studies, nothing goes through disk:
       - preprocessing as in training: percentile normalisation, padding, the 384 px cache (`prepare_cache.py`) and
         the 32 x 128 x 128 volume (`roi.prepare`), in the training left-right convention;
       - side classifier, per series (`side_cls.py`);
       - ROI U-Net on sagittal / coronal series flipped to a right knee: one box per ROI and series, with a
         confidence; weak boxes replaced by registration from the same knee's other series; axial boxes registered
         from the sagittal / coronal ones (`roi.consensus`);
       - crops (`crops.prepare.crop_window`), and the crop models, each on the findings the recipe takes from it.
     float16 autocast, fixed batch shapes (zero padded).
The shards' per-model study probabilities are then combined per the recipe (mean rank over all test studies) into the
submission. Studies without a prediction for a finding (e.g. no sagittal / coronal series) get the finding's median;
findings outside the recipe stay at 0.5.

Writes to --out_dir: shard<i>/series.csv (series, side, status), shard<i>/rois.csv (U-Net and axial boxes),
shard<i>/preds.csv (model, StudyInstanceUID, cls, p), roi_study_preds.csv (NaN where nothing was predicted); and
--submission.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import knee_registration as kr
from params import CACHE_MAX_SLICES, CACHE_SIZE, CLASSES
from roi.extract import ROI_CLASSES

ROI_PLANES = ("Sagittal", "Coronal")
VOL_DEPTH, VOL_SIZE = 32, 128            # roi.prepare.cache_to_volume
SIDE_BATCH, UNET_BATCH, CROP_BATCH = 64, 16, 64
CHUNK = 24                               # studies per GPU chunk
BOX = ["z0", "z1", "y0", "y1", "x0", "x1"]


def list_series(data_dir, series_dir, series_csv):
    if series_csv and os.path.exists(series_csv):
        df = pd.read_csv(series_csv)
    else:                                  # no series csv: walk the folders, plane from geometry
        rows = [(os.path.basename(os.path.dirname(p)), os.path.basename(p))
                for p in sorted(glob.glob(os.path.join(data_dir, series_dir, "*", "*")))]
        df = pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID"])
    if "Anatomical_Plane" not in df:
        df["Anatomical_Plane"] = ""
    df["folder"] = [os.path.join(data_dir, series_dir, r.StudyInstanceUID, r.SeriesInstanceUID)
                    for r in df.itertuples()]
    return df[["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "folder"]]


def shard_studies(df, n):
    """Study -> shard, greedy on the number of files so the GPUs finish together."""
    size = df.assign(n=[len(os.listdir(f)) if os.path.isdir(f) else 0 for f in df.folder]) \
        .groupby("StudyInstanceUID").n.sum().sort_values(ascending=False)
    load, out = [0] * n, {}
    for s, k in size.items():
        i = int(np.argmin(load))
        out[s] = i
        load[i] += k
    return out


# ------------------------------------------------------------------ reading (CPU)


def read_series(folder, plane):
    """Headers of every slice, pixels of the slices the cache keeps -> raw volume + metadata.

    Same slice order as the training arrays (`prepare_data.py`) and the same slice subsampling as
    `prepare_cache.preprocess_volume`.
    """
    import pydicom

    from dicom_io import pixel_spacing, slice_order, to_monochrome2

    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    # Short series (most): one full read. Longer ones: headers first, pixels of the kept slices only.
    full = len(files) <= CACHE_MAX_SLICES
    dss = [pydicom.dcmread(f, stop_before_pixels=not full) for f in files]
    order, major = slice_order(dss)
    files, dss = [files[i] for i in order], [dss[i] for i in order]
    key = [(int(d.Rows), int(d.Columns), int(d.BitsAllocated), int(d.PixelRepresentation)) for d in dss]
    if len(set(key)) > 1:                 # keep the majority shape rather than fail
        keep = max(set(key), key=key.count)
        files = [f for f, k in zip(files, key) if k == keep]
        dss = [d for d, k in zip(dss, key) if k == keep]
    n = len(dss)
    kept = np.round(np.linspace(0, n - 1, CACHE_MAX_SLICES)).astype(int) if n > CACHE_MAX_SLICES else range(n)
    arrs = [(dss[i] if full else pydicom.dcmread(files[i])).pixel_array for i in kept]
    vol = np.stack([a[0] if a.ndim == 3 and a.shape[0] == 1 else a for a in arrs])
    vol = to_monochrome2(vol, [dss[i] for i in kept])
    plane = plane if isinstance(plane, str) and plane else (major or "Sagittal")
    geom = kr.geometry_from_slices([d.ImageOrientationPatient for d in dss], [d.ImagePositionPatient for d in dss],
                                   pixel_spacing(dss[0]), dss[0].get("SliceThickness"))
    d = np.array(geom["direction"]).reshape(3, 3)
    # Training convention: sagittal slices towards patient right (-x), columns towards left (+x).
    conv_flip = bool(d[0, 2] > 0) if plane == "Sagittal" else bool(d[0, 0] < 0)
    tag = next((str(x.get("Laterality", "")).upper()[:1] for x in dss if x.get("Laterality", "")), "")
    # 16-bit pixels travel as int16 (shared memory), widened on the GPU.
    unsigned = vol.dtype == np.uint16
    raw = torch.from_numpy(vol.view(np.int16) if vol.dtype in (np.uint16, np.int16) else vol.astype(np.int32))
    meta = dict(plane=plane, n_slices=n, n_cache=min(n, CACHE_MAX_SLICES), rows=int(vol.shape[1]),
                cols=int(vol.shape[2]), conv_flip=conv_flip, dicom_laterality=tag)
    return raw, unsigned, meta, geom


class StudyDataset(Dataset):
    def __init__(self, df):
        self.studies = list(df.groupby("StudyInstanceUID", sort=False))

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        torch.set_num_threads(1)
        study, g = self.studies[i]
        out = []
        for r in g.itertuples():
            base = dict(StudyInstanceUID=study, SeriesInstanceUID=r.SeriesInstanceUID)
            try:
                raw, unsigned, meta, geom = read_series(r.folder, r.Anatomical_Plane)
                out.append(dict(**base, **meta, status="ok", raw=raw, unsigned=unsigned, geom=geom))
            except Exception as e:  # noqa: BLE001 -- one bad series must not stop the run
                out.append(dict(**base, plane=r.Anatomical_Plane, status=f"failed: {e!r}"[:300]))
        return out


def _first(batch):
    return batch[0]


# ------------------------------------------------------------- GPU preprocessing


def gpu_cache(raw, unsigned):
    """`prepare_cache.preprocess_volume` on the GPU (slices already subsampled)."""
    v = raw.int()
    if unsigned:
        v = v & 0xFFFF
    v = v.float()
    sample = v[:, ::4, ::4].reshape(-1)
    if sample.numel() <= 1 << 24:          # torch.quantile's size limit
        lo, hi = torch.quantile(sample, torch.tensor([0.005, 0.995], device=v.device)).tolist()
    else:
        lo, hi = np.percentile(sample.cpu().numpy(), [0.5, 99.5])
    v = ((v - lo) / max(hi - lo, 1e-6)).clamp(0, 1)
    z, h, w = v.shape
    if h != w:
        s = max(h, w)
        y0, x0 = (s - h) // 2, (s - w) // 2
        v = F.pad(v, (x0, s - w - x0, y0, s - h - y0))
    t = v[:, None]
    if t.shape[-1] != CACHE_SIZE:
        t = F.interpolate(t, size=(CACHE_SIZE, CACHE_SIZE), mode="bilinear", antialias=True, align_corners=False)
    return (t[:, 0].clamp(0, 1) * 255).round().to(torch.uint8)


def gpu_volume(cache):
    """`roi.prepare.cache_to_volume` on the GPU."""
    v = cache.float()[None, None]
    v = F.interpolate(v, size=(v.shape[2], VOL_SIZE, VOL_SIZE), mode="area")
    v = F.interpolate(v, size=(VOL_DEPTH, VOL_SIZE, VOL_SIZE), mode="trilinear", align_corners=False)
    return v[0, 0].round().clamp(0, 255).to(torch.uint8)


def _lr_axis(plane):
    return 0 if plane == "Sagittal" else 2          # of a (D, S, S) volume


def gpu_crops(cache, boxes, n_slices, rows, cols, flip, sagittal):
    """`crops.prepare.crop_box` of every ROI with a box -> (K, DEPTH, SIZE, SIZE) uint8, have (K,)."""
    from crops.prepare import DEPTH, SIZE, crop_window

    n_c, cs = cache.shape[0], cache.shape[1]
    out = torch.zeros((len(ROI_CLASSES), DEPTH, SIZE, SIZE), dtype=torch.uint8, device=cache.device)
    have = np.zeros(len(ROI_CLASSES), bool)
    for k, c in enumerate(ROI_CLASSES):
        if c not in boxes:
            continue
        zs, y0, x0, n = crop_window(n_c, cs, boxes[c], n_slices, rows, cols)
        win = torch.zeros((DEPTH, n, n), dtype=torch.float32, device=cache.device)
        ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y0 + n, cs), min(x0 + n, cs)
        if ye > ys and xe > xs:
            win[:, ys - y0:ye - y0, xs - x0:xe - x0] = cache[torch.as_tensor(zs, device=cache.device)][:, ys:ye, xs:xe]
        t = F.interpolate(win[:, None], size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)[:, 0]
        t = t.round().clamp(0, 255).to(torch.uint8)
        if flip:
            t = t.flip(0) if sagittal else t.flip(2)
        out[k], have[k] = t, True
    return out, have


# --------------------------------------------------------------------- boxes


def consensus(pred, rows):
    """U-Net boxes of whole studies -> rois rows: per-box confidence, failures filled by registration.

    rows: the studies' sagittal / coronal series (+ side). Works knee by knee -- the series of a study with the same
    predicted side, so a bilateral study never mixes its knees -- so any set of whole studies gives the same result.
    """
    from roi.consensus import add_confidence, fill_by_registration

    knee = lambda d: d.StudyInstanceUID + "|" + d.side  # noqa: E731
    pred["study"] = pred.StudyInstanceUID
    pred["StudyInstanceUID"] = knee(pred)
    pred = add_confidence(pred)
    weak = pred.conf < 0.5
    knees = rows.assign(StudyInstanceUID=knee(rows), Anatomical_Plane=rows.plane)
    filled, _ = fill_by_registration(pred.assign(source="unet"), knees, ~weak, weak)
    filled["StudyInstanceUID"] = filled.StudyInstanceUID.str.split("|").str[0]
    filled["side"] = filled.SeriesInstanceUID.map(rows.set_index("SeriesInstanceUID").side)
    cols = ["StudyInstanceUID", "SeriesInstanceUID", "plane", "side", "class"] + BOX + \
        ["conf", "p_mean", "support", "iou_other_plane", "iou_same_plane", "confidence", "source"]
    return filled.reindex(columns=cols).sort_values(["StudyInstanceUID", "SeriesInstanceUID", "class"])


def axial_boxes(series, rois, side):
    """ROI boxes of the axial series, registered from the same knee's sagittal / coronal boxes (`crops.axial_boxes`)."""
    from roi.consensus import _world, registered_box

    r = rois.copy()
    r["world"] = _world(r)
    r["knee"] = r.StudyInstanceUID + "|" + r.side
    by_knee = dict(tuple(r.groupby("knee")))
    out = []
    for t in series[(series.status == "ok") & (series.plane == "Axial")].itertuples():
        src = by_knee.get(f"{t.StudyInstanceUID}|{side.get(t.SeriesInstanceUID)}")
        if src is None:
            continue
        for cls, s in src.groupby("class"):
            box = registered_box(t, s.assign(confidence=s.confidence.fillna(0)))
            if box is not None:
                out.append(dict(StudyInstanceUID=t.StudyInstanceUID, SeriesInstanceUID=t.SeriesInstanceUID,
                                plane="Axial", **{"class": cls}, **box))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------ models


def _padded(x, n):
    """Zero-pad the batch dimension to n (one shape per model: no re-planning per batch size)."""
    return x if len(x) == n else torch.cat([x, x.new_zeros((n - len(x),) + x.shape[1:])])


class Models:
    def __init__(self, weights, recipe, device):
        from crops.train import GROUPS, PLANES, Crop25D
        from roi.train import build_model
        from side_cls import SideNet

        self.device = device
        sd = torch.load(os.path.join(weights, "side", "ckpt_final.pt"), map_location="cpu")
        self.side = SideNet(width=sd["stem.0.weight"].shape[0])
        self.side.load_state_dict(sd)
        self.side = self.side.to(device).eval()
        sd = torch.load(os.path.join(weights, "roi", "ckpt_final.pt"), map_location="cpu")
        self.unet = build_model(num_classes=len(ROI_CLASSES))
        self.unet.load_state_dict(sd)
        self.unet = self.unet.to(device).eval()
        self.crops = []
        for m in sorted({m for ms in recipe.values() for m in ms}):
            d = os.path.join(weights, "crops", m)
            a = json.load(open(os.path.join(d, "args.json")))
            group = GROUPS[a["group"]]
            sd = torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu")
            net = Crop25D(a["backbone"], len(group), pretrained=False, img_size=a.get("img_size") or None,
                          hidden=a.get("hidden", 256), dropout=0.0, drop_path=0.0, head=a.get("head", "lstm"))
            net.load_state_dict(sd)
            net = net.to(device).eval().to(memory_format=torch.channels_last)
            used = [c for c in group if m in recipe.get(c, [])]     # findings the recipe takes from it
            self.crops.append(dict(name=m, net=net, planes=a["planes"].split(","), planes_all=PLANES,
                                   rois={c: (ROI_CLASSES.index(c), group.index(c)) for c in used}))

    @torch.no_grad()
    def warmup(self):
        """One dummy batch through every network: CUDA / cuDNN first-call costs, out of the timings."""
        from crops.prepare import DEPTH, SIZE

        z = torch.zeros
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            self.side(z((SIDE_BATCH, 4, VOL_DEPTH, VOL_SIZE, VOL_SIZE), device=self.device))
            self.unet(z((UNET_BATCH, 1, VOL_DEPTH, VOL_SIZE, VOL_SIZE), device=self.device))
            for m in self.crops:
                i = z(CROP_BATCH, dtype=torch.long, device=self.device)
                m["net"](z((CROP_BATCH, DEPTH, SIZE, SIZE), device=self.device), i, i)
        torch.cuda.synchronize()


@torch.no_grad()
def process_chunk(studies, models, timer):
    """A list of studies (StudyDataset items, caches already on the GPU) -> series rows, rois, preds."""
    from roi.geometry import from_volume
    from roi.train import prob_to_boxes
    from side_cls import PLANES as SIDE_PLANES, model_input

    dev, amp = models.device, dict(device_type="cuda", dtype=torch.float16)
    ser = [s for st in studies for s in st if s["status"] == "ok"]
    for s in ser:
        kr.GEOMETRY_OVERRIDE[s["SeriesInstanceUID"]] = s["geom"]

    # Side, per series (volumes in the training left-right convention).
    t = time.time()
    p_right = []
    for i in range(0, len(ser), SIDE_BATCH):
        part = ser[i:i + SIDE_BATCH]
        x = _padded(torch.stack([s["volume"] for s in part])[:, None], SIDE_BATCH)
        pl = _padded(torch.tensor([SIDE_PLANES.index(s["plane"]) for s in part], device=dev), SIDE_BATCH)
        with torch.autocast(**amp):
            p_right.append(torch.sigmoid(models.side(model_input(x, pl)).float())[:len(part)].cpu())
    p_right = torch.cat(p_right).numpy() if ser else np.zeros(0)
    for s, p in zip(ser, p_right):
        s["p_right"], s["side"] = float(p), "R" if p > 0.5 else "L"
    timer["side"] += time.time() - t

    # ROI U-Net on sagittal / coronal series, flipped to a right knee; probabilities back
    # to the stored orientation (side flip XOR convention flip), then one box per class.
    t = time.time()
    seg = [s for s in ser if s["plane"] in ROI_PLANES]
    pred = []
    for i in range(0, len(seg), UNET_BATCH):
        part = seg[i:i + UNET_BATCH]
        x = torch.stack([s["volume"].flip(_lr_axis(s["plane"])) if s["side"] == "L" else s["volume"] for s in part])
        with torch.autocast(**amp):
            prob = torch.sigmoid(models.unet(_padded(x[:, None], UNET_BATCH).float() / 255).float())
        prob = torch.stack([prob[j].flip(_lr_axis(s["plane"]) + 1) if (s["side"] == "L") ^ s["conv_flip"]
                            else prob[j] for j, s in enumerate(part)]).cpu().numpy()
        for s, p in zip(part, prob):
            for k, (vb, conf, p_mean) in enumerate(prob_to_boxes(p, 0.5)):
                pred.append(dict(StudyInstanceUID=s["StudyInstanceUID"], SeriesInstanceUID=s["SeriesInstanceUID"],
                                 plane=s["plane"], side=s["side"], **{"class": ROI_CLASSES[k]}, conf=conf,
                                 p_mean=p_mean, **from_volume(vb, s["n_slices"], s["n_cache"], s["rows"], s["cols"],
                                                              *p.shape[1:3])))
    timer["unet"] += time.time() - t

    # Consensus and registration fill within a knee, then the axial series' boxes.
    t = time.time()
    meta_cols = ["StudyInstanceUID", "SeriesInstanceUID", "plane", "n_slices", "n_cache", "rows", "cols",
                 "conv_flip", "side", "status"]
    series = pd.DataFrame([{k: s.get(k) for k in meta_cols} for s in ser], columns=meta_cols)
    rois = consensus(pd.DataFrame(pred), series[series.plane.isin(ROI_PLANES)]) if pred else pd.DataFrame()
    side = dict(zip(series.SeriesInstanceUID, series.side))
    ax = axial_boxes(series, rois, side) if len(rois) else pd.DataFrame()
    boxes = pd.concat([rois, ax], ignore_index=True)
    timer["consensus"] += time.time() - t

    # Crops from the GPU caches, mirrored to a right knee in the training convention.
    t = time.time()
    by_series = {}
    for r in boxes.to_dict("records"):
        by_series.setdefault(r["SeriesInstanceUID"], {})[r["class"]] = {k: int(r[k]) for k in BOX}
    crop_rows = [s for s in ser if s["SeriesInstanceUID"] in by_series]
    cs, have = [], []
    for s in crop_rows:
        c, h = gpu_crops(s["cache"], by_series[s["SeriesInstanceUID"]], s["n_slices"], s["rows"], s["cols"],
                         (s["side"] == "L") ^ s["conv_flip"], s["plane"] == "Sagittal")
        cs.append(c)
        have.append(h)
    torch.cuda.synchronize()
    timer["crops"] += time.time() - t

    # Crop models: every (series, ROI) of the planes and findings each one is used for.
    t = time.time()
    out = []
    if crop_rows:
        x_all, have = torch.stack(cs), np.stack(have)
        for m in models.crops:
            items = [(j, roi, rid, m["planes_all"].index(s["plane"]), s["StudyInstanceUID"], c)
                     for j, s in enumerate(crop_rows) if s["plane"] in m["planes"]
                     for c, (roi, rid) in m["rois"].items() if have[j, roi]]
            for i in range(0, len(items), CROP_BATCH):
                part = items[i:i + CROP_BATCH]
                j, roi, rid, pl = (torch.tensor(v, device=dev) for v in list(zip(*part))[:4])
                x = _padded(x_all[j, roi], CROP_BATCH).float() / 255
                with torch.autocast(**amp):
                    p = torch.sigmoid(m["net"](x, _padded(rid, CROP_BATCH), _padded(pl, CROP_BATCH)).float())
                out += [(m["name"], st, c, float(v)) for (_, _, _, _, st, c), v in zip(part, p[:len(part)].tolist())]
    timer["models"] += time.time() - t
    preds = pd.DataFrame(out, columns=["model", "StudyInstanceUID", "cls", "p"])
    preds = preds.groupby(["model", "StudyInstanceUID", "cls"], as_index=False).p.mean()
    series = series.assign(p_right=[s["p_right"] for s in ser],
                           dicom_laterality=[s["dicom_laterality"] for s in ser])
    failed = pd.DataFrame([{k: s.get(k) for k in ["StudyInstanceUID", "SeriesInstanceUID", "plane", "status"]}
                           for st in studies for s in st if s["status"] != "ok"])
    return pd.concat([series, failed], ignore_index=True), boxes, preds


def run_shard(df, weights, recipe, out_dir, workers):
    # No cudnn.benchmark: it buys nothing at these shapes, and timing every algorithm of every
    # new conv shape cost ~9 s on an H100 and ~2 min on a T4.
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    dev = torch.device("cuda")
    t0 = time.time()
    models = Models(weights, recipe, dev)
    timer = dict.fromkeys(["load", "warmup", "wait", "prep", "side", "unet", "consensus", "crops", "models"], 0.0)
    timer["load"] = time.time() - t0
    t = time.time()
    models.warmup()
    timer["warmup"] = time.time() - t
    loader = DataLoader(StudyDataset(df), batch_size=1, collate_fn=_first, num_workers=workers,
                        prefetch_factor=4 if workers else None, persistent_workers=False)
    all_series, all_boxes, all_preds, chunk = [], [], [], []

    def flush():
        s, b, p = process_chunk(chunk, models, timer)
        all_series.append(s)
        all_boxes.append(b)
        all_preds.append(p)
        chunk.clear()

    t = time.time()
    for n, study in enumerate(loader):
        timer["wait"] += time.time() - t
        t = time.time()
        for s in study:
            if s["status"] == "ok":
                try:
                    s["cache"] = gpu_cache(s.pop("raw").to(dev, non_blocking=True), s.pop("unsigned"))
                    v = gpu_volume(s["cache"])
                    s["volume"] = v.flip(_lr_axis(s["plane"])) if s["conv_flip"] else v
                except Exception as e:  # noqa: BLE001
                    s["status"] = f"failed: {e!r}"[:300]
        chunk.append(study)
        torch.cuda.synchronize()
        timer["prep"] += time.time() - t
        if len(chunk) == CHUNK:
            flush()
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(loader)} studies, {time.time() - t0:.0f}s", flush=True)
        t = time.time()
    if chunk:
        flush()
    os.makedirs(out_dir, exist_ok=True)
    pd.concat(all_series, ignore_index=True).to_csv(os.path.join(out_dir, "series.csv"), index=False)
    pd.concat(all_boxes, ignore_index=True).to_csv(os.path.join(out_dir, "rois.csv"), index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(os.path.join(out_dir, "preds.csv"), index=False)
    print(f"shard done: {len(loader)} studies in {time.time() - t0:.0f}s -- " +
          ", ".join(f"{k} {v:.0f}s" for k, v in timer.items()), flush=True)


def combine(preds, recipe, studies):
    """Per-model study probabilities -> one column per finding: a single model's probability, or the mean of several
    models' ranks over the test studies. A study missing from some models (e.g. no axial series for an axial model)
    gets the mean of the others; NaN only if no model predicts it."""
    tab = {m: g.pivot(index="StudyInstanceUID", columns="cls", values="p") for m, g in preds.groupby("model")}
    study = pd.DataFrame(index=pd.Index(studies, name="StudyInstanceUID"))
    for c, ms in recipe.items():
        cols = [tab[m][c].reindex(study.index) if m in tab and c in tab[m] else
                pd.Series(np.nan, index=study.index) for m in ms]
        study[c] = cols[0] if len(cols) == 1 else pd.concat([col.rank(pct=True) for col in cols], axis=1).mean(axis=1)
    return study


def run(data_dir, weights, out_dir, submission, gpus, workers, series_dir="test_series", series_csv=None,
        limit=None):
    from infer.sanitize import sanitize

    t0 = time.time()
    with open(os.path.join(weights, "recipe.json")) as f:
        recipe = json.load(f)
    df = list_series(data_dir, series_dir, series_csv)
    if limit:
        df = df[df.StudyInstanceUID.isin(df.StudyInstanceUID.drop_duplicates()[:limit])]
    os.makedirs(out_dir, exist_ok=True)
    for d in glob.glob(os.path.join(out_dir, "shard*")) + [os.path.join(out_dir, "dicom")]:   # an earlier run
        shutil.rmtree(d, ignore_errors=True)
    df = sanitize(df, os.path.join(out_dir, "dicom"), max(1, len(gpus) * workers))
    df["shard"] = df.StudyInstanceUID.map(shard_studies(df, len(gpus)))
    df.to_csv(os.path.join(out_dir, "series_list.csv"), index=False)
    shards = sorted(df["shard"].unique())                  # fewer studies than GPUs: some idle
    procs = [subprocess.Popen([sys.executable, "-m", "infer.pipeline", "--shard", str(i), "--weights", weights,
                               "--out_dir", out_dir, "--workers", str(workers)],
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[i]))) for i in shards]
    if any(p.wait() for p in procs):
        raise RuntimeError("a shard failed")
    preds = pd.concat([pd.read_csv(os.path.join(out_dir, f"shard{i}", "preds.csv"), dtype={"model": str})
                       for i in shards])
    studies = df.StudyInstanceUID.drop_duplicates().tolist()
    study = combine(preds, recipe, studies)
    study.to_csv(os.path.join(out_dir, "roi_study_preds.csv"))
    sub = pd.DataFrame({"StudyInstanceUID": studies})
    for c in CLASSES:
        sub[c] = sub.StudyInstanceUID.map(study[c]) if c in study else np.nan
        n = int(sub[c].isna().sum())
        if c in study and n and sub[c].notna().any():     # studies without a prediction: the finding's median
            print(f"  {c}: {n} studies without a prediction -> median")
            sub[c] = sub[c].fillna(sub[c].median())
    sub[CLASSES] = sub[CLASSES].fillna(0.5)                                  # findings outside the recipe
    sub.to_csv(submission, index=False)
    print(f"-> {submission}: {len(sub)} studies, recipe covers {list(recipe)} (0.5 elsewhere), "
          f"{time.time() - t0:.0f}s in total")
    return sub


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir")
    p.add_argument("--weights", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--submission", default="submission.csv")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--workers", type=int, default=2, help="DICOM reading processes per GPU")
    p.add_argument("--series_dir", default="test_series")
    p.add_argument("--series_csv", default=None, help="series list (default: walk <data_dir>/<series_dir>)")
    p.add_argument("--limit", type=int, default=None, help="first N studies (debugging)")
    p.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.shard is not None:               # worker process, spawned by run()
        with open(os.path.join(a.weights, "recipe.json")) as f:
            recipe = json.load(f)
        df = pd.read_csv(os.path.join(a.out_dir, "series_list.csv"))
        df = df[df["shard"] == a.shard]
        run_shard(df, a.weights, recipe, os.path.join(a.out_dir, f"shard{a.shard}"), a.workers)
        return
    run(a.data_dir, a.weights, a.out_dir, a.submission, a.gpus.split(","), a.workers, a.series_dir,
        a.series_csv, a.limit)


if __name__ == "__main__":
    main()
