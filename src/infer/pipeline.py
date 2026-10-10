"""The whole test-time pipeline in one streaming pass per GPU, run from src/.

    python -m infer.pipeline --data_dir <dir with test_series/> --weights <dir> --out_dir /tmp/knee/ \
        --submission submission.csv --gpus 0,1

<weights> holds side/ and roi/ (run folders), crops/<run>/ (args.json + ckpt_final.pt)
and recipe.json ({finding: [crop run, ...]}, or {finding: {"models": [...], "minus": {"finding": f,
"beta": b}}} to subtract b x finding f's combined score: the medial / lateral OA cross-talk
correction, see experiments.md section 14).

The operations of `infer.prep` -> `infer.side` -> `infer.seg` -> `infer.classify`,
reorganised for a small machine (Kaggle: 2 x T4, 4 CPU cores):
  - the studies are split between the GPUs, one process each, balanced on file count;
  - CPU work is only reading the DICOMs, in DataLoader workers: the headers of every
    file (slice order, geometry) and the pixels of the <= 64 slices the cache keeps;
  - everything else runs on the GPU: percentile normalisation, padding and resizing
    (the 384 px cache and the 32 x 128 x 128 volume), side classifier, ROI U-Net,
    crop cutting, crop models;
  - nothing goes through disk: studies are processed by chunks, and a chunk's caches
    stay on the GPU until its crops are cut. Box consensus, registration fill and
    axial boxes work within a knee, so chunking changes nothing;
  - each crop model only predicts the findings the recipe takes from it;
  - float16 autocast, fixed batch shapes (zero padded); crop sampling (frames, stride)
    read from each crop model's args.json, crops cut once per distinct sampling.
The shards' per-model study probabilities are then gathered and combined per the
recipe (rank average over all test studies) into the submission; findings the
recipe does not cover stay at 0.5.

Writes to --out_dir: shard<i>/series.csv (series.csv of `infer.prep` + side,
p_right), shard<i>/rois.csv (U-Net and axial boxes), shard<i>/preds.csv (model,
StudyInstanceUID, cls, p), roi_study_preds.csv; and --submission.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import knee_registration as kr
from params import CACHE_MAX_SLICES, CACHE_SIZE, CLASSES
from roi.extract import ROI_CLASSES

ROI_PLANES = ("Sagittal", "Coronal")
VOL_DEPTH, VOL_SIZE = 32, 128            # roi.prepare.cache_to_volume
SIDE_BATCH, UNET_BATCH, CROP_BATCH = 64, 16, 64
CHUNK = 24                               # studies per GPU chunk


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

    Same slices, order and majority-shape rule as `infer.prep.read_series` +
    `prepare_cache.preprocess_volume`'s slice subsampling.
    """
    import pydicom

    from infer.prep import slice_order
    from knee_registration import geometry_from_slices

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
    plane = plane if isinstance(plane, str) and plane else (major or "Sagittal")
    geom = geometry_from_slices([d.ImageOrientationPatient for d in dss], [d.ImagePositionPatient for d in dss],
                                dss[0].PixelSpacing, dss[0].get("SliceThickness"))
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


def gpu_crops(cache, boxes, n_slices, rows, cols, flip, sagittal, depth=12, z_mode="uniform", stride=1,
              size=128, classes=ROI_CLASSES, margin=None):
    """`crops.prepare.crop_box` of every class with a box -> (K, depth, size, size) uint8, have (K,)."""
    from crops.prepare import MARGIN, crop_window

    SIZE = size
    margin = MARGIN if margin is None else margin
    n_c, cs = cache.shape[0], cache.shape[1]
    out = torch.zeros((len(classes), depth, SIZE, SIZE), dtype=torch.uint8, device=cache.device)
    have = np.zeros(len(classes), bool)
    for k, c in enumerate(classes):
        if c not in boxes:
            continue
        zs, y0, x0, n = crop_window(n_c, cs, boxes[c], n_slices, rows, cols, depth, margin=margin, z_mode=z_mode,
                                    stride=stride)
        win = torch.zeros((depth, n, n), dtype=torch.float32, device=cache.device)
        ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y0 + n, cs), min(x0 + n, cs)
        if ye > ys and xe > xs:
            if zs.dtype.kind == "f":          # resize: linear between the two nearest slices
                lo = np.floor(zs).astype(int)
                hi = np.minimum(lo + 1, n_c - 1)
                w = torch.as_tensor(zs - lo, dtype=torch.float32, device=cache.device)[:, None, None]
                sub = (1 - w) * cache[torch.as_tensor(lo, device=cache.device)][:, ys:ye, xs:xe].float() + \
                    w * cache[torch.as_tensor(hi, device=cache.device)][:, ys:ye, xs:xe].float()
            else:
                sub = cache[torch.as_tensor(zs, device=cache.device)][:, ys:ye, xs:xe]
            win[:, ys - y0:ye - y0, xs - x0:xe - x0] = sub
        t = F.interpolate(win[:, None], size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)[:, 0]
        t = t.round().clamp(0, 255).to(torch.uint8)
        if flip:
            t = t.flip(0) if sagittal else t.flip(2)
        out[k], have[k] = t, True
    return out, have


# ------------------------------------------------------------------------ models


def _padded(x, n):
    """Zero-pad the batch dimension to n (one shape per model: no re-planning per batch size)."""
    return x if len(x) == n else torch.cat([x, x.new_zeros((n - len(x),) + x.shape[1:])])


class Models:
    def __init__(self, weights, recipe, device):
        from crops.train import GROUPS, PLANES, build_model as build_crop_model, crop_index
        from roi.train import build_model
        from side_cls import SideNet

        self.device = device
        sd = torch.load(os.path.join(weights, "side", "ckpt_final.pt"), map_location="cpu")
        self.side = SideNet(width=sd["stem.0.weight"].shape[0])
        self.side.load_state_dict(sd)
        self.side = self.side.to(device).eval()
        sd = torch.load(os.path.join(weights, "roi", "ckpt_final.pt"), map_location="cpu")
        k = sd["decoder.seg_layers.4.weight"].shape[0]
        self.roi_classes = ROI_CLASSES[:k]
        self.unet = build_model(num_classes=k)
        self.unet.load_state_dict(sd)
        self.unet = self.unet.to(device).eval()
        self.crops = []
        for m in sorted({m for r in recipe.values() for m in r["models"]}):
            d = os.path.join(weights, "crops", m)
            a = json.load(open(os.path.join(d, "args.json")))
            group = GROUPS[a["group"]]
            sd = torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu")
            net = build_crop_model(a, len(group))
            n_planes = sd["plane_embed.weight"].shape[0]
            if n_planes != net.plane_embed.num_embeddings:     # runs from before axial crops: 2 planes
                net.plane_embed = nn.Embedding(n_planes, net.plane_embed.embedding_dim)
            net.load_state_dict(sd)
            net = net.to(device).eval()
            if a["backbone"] not in ("roi_unet3d", "roi_unet3d_scratch"):    # 2D encoders only
                net = net.to(memory_format=torch.channels_last)
            used = [c for c in group if c in recipe and m in recipe[c]["models"]]     # findings taken from it
            z = a.get("z_sampling") or dict(depth=12, z_mode="uniform", stride=1)    # older runs: uniform 12
            z = {k: z[k] for k in ("depth", "z_mode", "stride", "size", "capsule", "capsule_up", "margin",
                                   "compartment", "oa_midline", "union") if k in z}
            self.crops.append(dict(name=m, net=net, z=json.dumps(z, sort_keys=True), planes=a["planes"].split(","),
                                   planes_all=PLANES,
                                   rois={c: (crop_index(c, "capsule" in z), group.index(c)) for c in used}))


    @torch.no_grad()
    def warmup(self):
        """One dummy batch through every network: CUDA / cuDNN first-call costs, out of the timings."""
        amp = dict(device_type="cuda", dtype=torch.float16)
        z = torch.zeros
        with torch.autocast(**amp):
            self.side(z((SIDE_BATCH, 4, VOL_DEPTH, VOL_SIZE, VOL_SIZE), device=self.device))
            self.unet(z((UNET_BATCH, 1, VOL_DEPTH, VOL_SIZE, VOL_SIZE), device=self.device))
            for m in self.crops:
                i = z(CROP_BATCH, dtype=torch.long, device=self.device)
                k = json.loads(m["z"])
                m["net"](z((CROP_BATCH, k["depth"], k.get("size", 128), k.get("size", 128)), device=self.device), i, i)
        torch.cuda.synchronize()


@torch.no_grad()
def process_chunk(studies, models, timer):
    """A list of studies (StudyDataset items, caches already on the GPU) -> series rows, rois, preds."""
    from infer.classify import axial_boxes
    from infer.seg import consensus
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
        timer["unet"] += time.time() - t
        t = time.time()
        for s, p in zip(part, prob):
            for k, (vb, conf, p_mean) in enumerate(prob_to_boxes(p, 0.5)):
                pred.append(dict(StudyInstanceUID=s["StudyInstanceUID"], SeriesInstanceUID=s["SeriesInstanceUID"],
                                 plane=s["plane"], side=s["side"], **{"class": models.roi_classes[k]}, conf=conf,
                                 p_mean=p_mean, **from_volume(vb, s["n_slices"], s["n_cache"], s["rows"], s["cols"],
                                                              *p.shape[1:3])))
        timer["boxes"] += time.time() - t
        t = time.time()

    # Consensus and registration fill within a knee, then the axial series' boxes.
    t = time.time()
    meta_cols = ["StudyInstanceUID", "SeriesInstanceUID", "plane", "n_slices", "n_cache", "rows", "cols",
                 "conv_flip", "side", "status"]
    series = pd.DataFrame([{k: s.get(k) for k in meta_cols} for s in ser], columns=meta_cols)
    rois = pd.DataFrame()
    if pred:
        rois, _, _ = consensus(pd.DataFrame(pred), series[series.plane.isin(ROI_PLANES)])
    side = dict(zip(series.SeriesInstanceUID, series.side))
    ax = axial_boxes(series, rois, side) if len(rois) else pd.DataFrame()
    boxes = pd.concat([rois, ax], ignore_index=True)
    timer["consensus"] += time.time() - t

    # Crops from the GPU caches, mirrored to a right knee in the training convention.
    t = time.time()
    by_series = {}
    for r in boxes.to_dict("records"):
        by_series.setdefault(r["SeriesInstanceUID"], {})[r["class"]] = {k: int(r[k]) for k in
                                                                        ["z0", "z1", "y0", "y1", "x0", "x1"]}
    crop_rows = [s for s in ser if s["SeriesInstanceUID"] in by_series]
    crops = {}                            # sampling -> (crops (N, K, D, S, S), have (N, K))
    for z in {m["z"] for m in models.crops}:
        kw = json.loads(z)
        cap, cap_up = kw.pop("capsule", None), kw.pop("capsule_up", None)
        compartment, midline = kw.pop("compartment", None), kw.pop("oa_midline", "")
        union = [c for c in kw.pop("union", "").split(",") if c]
        if cap is not None:               # effusion / synovitis: one capsule crop around the ROI union
            from crops.prepare import capsule_boxes

            shape = {s["SeriesInstanceUID"]: (s["rows"], s["cols"]) for s in crop_rows}
            src = boxes[boxes.SeriesInstanceUID.isin(shape)]      # axial: union of its registered boxes
            series_boxes = capsule_boxes(src, cap, cap_up, shape, union or None)
            kw.update(classes=["Capsule"], margin=0.0)
        else:
            series_boxes = by_series
            if compartment is not None or midline:
                # OA compartment crops, exactly as crops.prepare --compartment / --oa_midline built them.
                from crops.prepare import compartment_box, oa_midline

                series_boxes = {}
                for s in crop_rows:
                    uid = s["SeriesInstanceUID"]
                    bx = dict(by_series.get(uid, {}))
                    if compartment is not None and s["plane"] != "Axial":
                        for oa, men in (("Medial OA", "Medial Meniscus"), ("Lateral OA", "Lateral Meniscus")):
                            if oa in bx and men in bx:
                                bx[oa] = compartment_box(bx[oa], bx[men], compartment, (s["rows"], s["cols"]))
                    if midline and s["plane"] in ("Coronal", "Axial"):
                        bx = oa_midline(bx, clip=midline == "clip")
                    series_boxes[uid] = bx
        cs, have = [], []
        for s in crop_rows:
            c, h = gpu_crops(s["cache"], series_boxes.get(s["SeriesInstanceUID"], {}), s["n_slices"], s["rows"],
                             s["cols"], (s["side"] == "L") ^ s["conv_flip"], s["plane"] == "Sagittal", **kw)
            cs.append(c)
            have.append(h)
        if cs:
            crops[z] = torch.stack(cs), np.stack(have)
    torch.cuda.synchronize()
    timer["crops"] += time.time() - t

    # Crop models: every (series, ROI) of the planes and findings each one is used for.
    out = []
    for m in models.crops if crop_rows else []:
        t = time.time()
        x_all, have = crops[m["z"]]
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
        dt = time.time() - t
        timer["models"] += dt
        timer["model " + m["name"]] = timer.get("model " + m["name"], 0.0) + dt
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
    timer = dict.fromkeys(["load", "warmup", "wait", "prep", "side", "unet", "boxes", "consensus", "crops", "models"], 0.0)
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
    total = time.time() - t0
    print(f"shard done: {len(loader)} studies in {total:.0f}s -- " +
          ", ".join(f"{k} {v:.0f}s" for k, v in timer.items()), flush=True)


def load_recipe(path):
    """recipe.json -> {finding: {"models": [...], "minus": {"finding": f, "beta": b} or None}}."""
    with open(path) as f:
        r = json.load(f)
    return {c: (dict(models=v, minus=None) if isinstance(v, list) else dict(models=v["models"], minus=v.get("minus")))
            for c, v in r.items()}


def combine(preds, recipe, studies):
    """Per-model study probabilities -> one column per finding: a single model's probability,
    or the mean of several models' ranks over the test studies; then the cross-talk
    corrections (score - beta x another finding's combined score)."""
    tab = {m: g.pivot(index="StudyInstanceUID", columns="cls", values="p") for m, g in preds.groupby("model")}
    study = pd.DataFrame(index=pd.Index(studies, name="StudyInstanceUID"))
    for c, r in recipe.items():
        cols = [tab[m][c].reindex(study.index) if m in tab and c in tab[m] else
                pd.Series(np.nan, index=study.index) for m in r["models"]]
        study[c] = cols[0] if len(cols) == 1 else sum(col.rank(pct=True) for col in cols) / len(cols)
    base = study.copy()
    for c, r in recipe.items():
        if r["minus"]:
            # Back to [0, 1] as percentile ranks over the test studies (same order, valid probabilities).
            study[c] = (base[c] - r["minus"]["beta"] * base[r["minus"]["finding"]]).rank(pct=True)
    return study


def run(data_dir, weights, out_dir, submission, gpus, workers, series_dir="test_series", series_csv=None,
        limit=None):
    t0 = time.time()
    recipe = load_recipe(os.path.join(weights, "recipe.json"))
    df = list_series(data_dir, series_dir, series_csv)
    if limit:
        df = df[df.StudyInstanceUID.isin(df.StudyInstanceUID.drop_duplicates()[:limit])]
    os.makedirs(out_dir, exist_ok=True)
    for d in glob.glob(os.path.join(out_dir, "shard*")):    # outputs of an earlier run
        shutil.rmtree(d)
    df["shard"] = df.StudyInstanceUID.map(shard_studies(df, len(gpus)))
    df.to_csv(os.path.join(out_dir, "series_list.csv"), index=False)
    shards = sorted(df["shard"].unique())                  # fewer studies than GPUs: some idle
    procs = []
    for i in shards:
        g = gpus[i]
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "infer.pipeline", "--shard", str(i),
             "--weights", weights, "--out_dir", out_dir, "--workers", str(workers)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))))
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
    sub[CLASSES] = sub[CLASSES].fillna(0.5)                                  # uncovered findings / studies
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
        recipe = load_recipe(os.path.join(a.weights, "recipe.json"))
        df = pd.read_csv(os.path.join(a.out_dir, "series_list.csv"))
        df = df[df["shard"] == a.shard]
        run_shard(df, a.weights, recipe, os.path.join(a.out_dir, f"shard{a.shard}"), a.workers)
        return
    run(a.data_dir, a.weights, a.out_dir, a.submission, a.gpus.split(","), a.workers, a.series_dir,
        a.series_csv, a.limit)


if __name__ == "__main__":
    main()
