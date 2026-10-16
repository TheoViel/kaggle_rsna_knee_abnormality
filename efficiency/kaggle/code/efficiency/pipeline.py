"""Fast test-time pipeline (efficiency track), run from efficiency/ with src/ importable.

    python pipeline.py --data_dir <dir> --weights <dir> --out_dir /tmp/knee/ --submission submission.csv --gpus 0,1

Same models and operations as src/infer/pipeline.py (load -> side / flip -> ROI U-Net -> crops -> crop models ->
rank-average recipe), made cheaper:
  - series: one sagittal + one coronal per study (fluid-sensitive first, then the lowest SeriesInstanceUID);
    no axial series, so no axial boxes / crops;
  - reading: a thread pool per GPU process (no DataLoader processes); one `read()` per file, headers parsed
    directly and pixels from the PixelData offset (fast_dicom, bit-identical to pydicom; pydicom fallback);
  - GPU: weights in float16 with half inputs (no autocast), uint8 to the GPU, normalisation on device;
  - one process per GPU, each with its shard of studies (no DataParallel); reading overlaps the GPU work.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("KNEE_SRC", os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # vendored packages next to efficiency/ (Kaggle code dataset)
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import fast_dicom  # noqa: E402
import knee_registration as kr  # noqa: E402
from infer.pipeline import _lr_axis, _padded, combine, gpu_crops, gpu_volume, load_recipe, shard_studies  # noqa: E402
from params import CACHE_SIZE  # noqa: E402
from params import CACHE_MAX_SLICES, CLASSES  # noqa: E402

ROI_PLANES = ("Sagittal", "Coronal")
VOL_DEPTH, VOL_SIZE = 32, 128
SIDE_BATCH, UNET_BATCH, CROP_BATCH = 32, 16, 64
CHUNK = 32                                # studies per chunk: 64 series -> full batches of 64 crops (no padding)
THREADS = int(os.environ.get("KNEE_THREADS", 16))          # file reads in flight per process
PREFETCH = int(os.environ.get("KNEE_PREFETCH", 48))        # series read ahead of the GPU
UNET_PLANES = os.environ.get("KNEE_UNET_PLANES", "Sagittal,Coronal").split(",")   # others: registered boxes
READ_PROCS = int(os.environ.get("KNEE_READ_PROCS", 2))     # reader processes per GPU process (0: threads only)


def list_series(data_dir, series_dir, series_csv, rule="fs"):
    """Candidate series per study and slot (Sagittal / Coronal), ranked; read_study keeps the first usable one per
    slot, so each study normally reads exactly one sagittal + one coronal series.

    Per slot, in order: the series the csv labels with that plane (fluid-sensitive first, then SeriesInstanceUID);
    the series without a plane label; then the study's other series (e.g. axial-labelled), as a last resort: the
    DICOM geometry decides (a candidate whose orientation is not the slot's plane is skipped). rule="all": every
    series, as labelled (no slots).
    -> rows (StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane, folder, slot, rank); rank 0 = first choice.
    """
    df = pd.read_csv(series_csv) if series_csv and os.path.exists(series_csv) else None
    if df is None:
        rows = [(os.path.basename(os.path.dirname(p)), os.path.basename(p))
                for p in sorted(glob.glob(os.path.join(data_dir, series_dir, "*", "*")))]
        df = pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID"])
    if "Anatomical_Plane" not in df:
        df["Anatomical_Plane"] = ""
    df["Anatomical_Plane"] = df.Anatomical_Plane.fillna("").astype(str)
    df["folder"] = [os.path.join(data_dir, series_dir, r.StudyInstanceUID, r.SeriesInstanceUID)
                    for r in df.itertuples()]
    cols = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "folder", "slot", "rank"]
    if rule == "all":
        return df.assign(slot="all", rank=0)[cols].reset_index(drop=True)
    fs = pd.to_numeric(df.Fluid_Sensitive, errors="coerce").fillna(0) if "Fluid_Sensitive" in df else 0
    df = df.assign(fs=-fs if rule == "fs" else 0)
    out = []
    for slot in ROI_PLANES:
        tier = np.where(df.Anatomical_Plane == slot, 0, np.where(~df.Anatomical_Plane.isin(ROI_PLANES + ("Axial",)), 1,
                        np.where(df.Anatomical_Plane == "Axial", 3, 2)))
        d = df.assign(slot=slot, tier=tier).sort_values(["StudyInstanceUID", "tier", "fs", "SeriesInstanceUID"])
        d["rank"] = d.groupby("StudyInstanceUID").cumcount()
        out.append(d)
    return pd.concat(out).sort_values(["StudyInstanceUID", "slot", "rank"])[cols].reset_index(drop=True)


# ------------------------------------------------------------------ reading (CPU threads)


class _Hdr(dict):
    """fast_dicom fields with the pydicom-style access infer.prep.slice_order uses."""
    __getattr__ = dict.__getitem__


def _install_decoders():
    """JPEG decoders for pydicom, from the wheels dataset ($KNEE_WHEELS), only when a compressed file shows up
    (the training DICOMs are all uncompressed: the normal run never pays for pip)."""
    import fcntl

    wheels = os.environ.get("KNEE_WHEELS")
    if not wheels:
        return
    with open("/tmp/knee_decoders.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists("/tmp/knee_decoders.done"):
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--find-links", wheels,
                            "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"], check=False)
            open("/tmp/knee_decoders.done", "w").close()


def _read_file(f, pixels=True):
    try:
        return _Hdr(fast_dicom.read(f) if pixels else fast_dicom.read_header(f))
    except Exception:  # noqa: BLE001 -- compressed or unusual: pydicom
        import pydicom

        d = pydicom.dcmread(f, stop_before_pixels=not pixels)
        try:
            a = d.pixel_array if pixels else np.zeros((1, 1))
        except Exception:  # noqa: BLE001 -- no decoder for this transfer syntax: install them once, retry
            _install_decoders()
            a = pydicom.dcmread(f).pixel_array
        return _Hdr(InstanceNumber=int(d.get("InstanceNumber", 0) or 0),
                    ImageOrientationPatient=[float(x) for x in d.ImageOrientationPatient],
                    ImagePositionPatient=[float(x) for x in d.ImagePositionPatient],
                    PixelSpacing=[float(x) for x in d.PixelSpacing],
                    SliceThickness=float(d.SliceThickness) if d.get("SliceThickness") is not None else None,
                    Rows=int(d.Rows), Columns=int(d.Columns), BitsAllocated=int(d.BitsAllocated),
                    PixelRepresentation=int(d.PixelRepresentation), Laterality=str(d.get("Laterality", "")),
                    pixels=a[0] if a.ndim == 3 and a.shape[0] == 1 else a)


def read_series(folder, plane, pool=None):
    """src/infer/pipeline.read_series with fast_dicom: same slices, order, majority shape and subsampling."""
    from infer.prep import slice_order
    from knee_registration import geometry_from_slices

    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    # Long series (> 64 slices, ~2%): headers of every slice first, pixels of the kept slices only.
    full = len(files) <= CACHE_MAX_SLICES
    fn = _read_file if full else (lambda f: _read_file(f, pixels=False))
    dss = list(pool.map(fn, files)) if pool else [fn(f) for f in files]
    for d, f in zip(dss, files):
        d.setdefault("InstanceNumber", 0)
        d["_file"] = f
    order, major = slice_order(dss)
    dss = [dss[i] for i in order]
    key = [(d["Rows"], d["Columns"], d["BitsAllocated"], d["PixelRepresentation"]) for d in dss]
    if len(set(key)) > 1:
        keep = max(set(key), key=key.count)
        dss = [d for d, k in zip(dss, key) if k == keep]
    n = len(dss)
    kept = np.round(np.linspace(0, n - 1, CACHE_MAX_SLICES)).astype(int) if n > CACHE_MAX_SLICES else range(n)
    if not full:
        px = list(pool.map(_read_file, [dss[i]["_file"] for i in kept])) if pool else \
            [_read_file(dss[i]["_file"]) for i in kept]
        vol = np.stack([d["pixels"] for d in px])
    else:
        vol = np.stack([dss[i]["pixels"] for i in kept])
    plane = plane if isinstance(plane, str) and plane and plane != "?" else (major or "Sagittal")
    geom = geometry_from_slices([d["ImageOrientationPatient"] for d in dss], [d["ImagePositionPatient"] for d in dss],
                                dss[0]["PixelSpacing"], dss[0].get("SliceThickness"))
    dm = np.array(geom["direction"]).reshape(3, 3)
    conv_flip = bool(dm[0, 2] > 0) if plane == "Sagittal" else bool(dm[0, 0] < 0)
    tag = next((d["Laterality"].upper()[:1] for d in dss if d.get("Laterality")), "")
    unsigned = vol.dtype == np.uint16
    raw = torch.from_numpy(vol.view(np.int16) if vol.dtype in (np.uint16, np.int16) else vol.astype(np.int32))
    meta = dict(plane=plane, geom_plane=major, n_slices=n, n_cache=min(n, CACHE_MAX_SLICES), rows=int(vol.shape[1]),
                cols=int(vol.shape[2]), conv_flip=conv_flip, dicom_laterality=tag)
    return raw, unsigned, meta, geom


def _read_job(r, pool, plane=None):
    base = dict(StudyInstanceUID=r.StudyInstanceUID, SeriesInstanceUID=r.SeriesInstanceUID)
    try:
        raw, unsigned, meta, geom = read_series(r.folder, plane or r.Anatomical_Plane, pool)
        return dict(**base, **meta, status="ok", raw=raw, unsigned=unsigned, geom=geom)
    except Exception as e:  # noqa: BLE001 -- one bad series must not stop the run
        return dict(**base, plane=r.Anatomical_Plane, status=f"failed: {e!r}"[:300])


MIN_SLICES = 5                             # fewer (localizer, broken series): try the slot's next candidate


def read_study(rows, pool):
    """One study's candidate rows (list_series) -> list of series dicts: per slot, the first candidate that reads,
    has >= MIN_SLICES slices and whose geometry is the slot's plane (unknown geometry: the slot's plane is assumed).
    A series read for one slot and rejected there can fill the other slot. Slots left empty are skipped (worst case:
    one series in one view; nothing usable: the study's findings fall back to 0.5). Failed reads are reported."""
    if rows and rows[0].slot == "all":
        return [_read_job(r, pool) for r in rows]
    cache, used, out, failed = {}, set(), [], []
    for slot in ROI_PLANES:
        for r in (r for r in rows if r.slot == slot):
            if r.SeriesInstanceUID in used:
                continue
            s = cache.get(r.SeriesInstanceUID)
            if s is None:
                s = cache[r.SeriesInstanceUID] = _read_job(r, pool, plane="?")
            ok = s["status"] == "ok" and s["n_slices"] >= MIN_SLICES and s.get("geom_plane") in (slot, None)
            if ok:
                dm = np.array(s["geom"]["direction"]).reshape(3, 3)       # training left-right convention of the slot
                s = dict(s, plane=slot, conv_flip=bool(dm[0, 2] > 0) if slot == "Sagittal" else bool(dm[0, 0] < 0))
                if s["plane"] != r.Anatomical_Plane or r.rank:
                    s["fallback"] = f"{slot} rank {r.rank} ({r.Anatomical_Plane or 'unlabelled'})"
                if s["SeriesInstanceUID"] not in used:
                    used.add(s["SeriesInstanceUID"])
                    out.append(s)
                break
    for uid, s in cache.items():
        if uid not in used and s["status"] != "ok":
            failed.append(s)
    return out + failed


class StudyReader:
    """Studies in order (list of series dicts), read ahead by a thread pool (one job per study, its files on
    `file_pool`); reading starts at construction, so it overlaps the model loading."""

    def __init__(self, df, threads=THREADS, prefetch=PREFETCH):
        self.studies = [list(g.itertuples()) for _, g in df.groupby("StudyInstanceUID", sort=False)]
        self.prefetch = max(1, prefetch // 2)
        self.file_pool = ThreadPoolExecutor(threads)
        self.study_pool = ThreadPoolExecutor(max(2, threads // 4))
        self.futs, self.nxt = {}, 0
        self._fill()

    def _fill(self):
        while self.nxt < len(self.studies) and len(self.futs) < self.prefetch:
            self.futs[self.nxt] = self.study_pool.submit(read_study, self.studies[self.nxt], self.file_pool)
            self.nxt += 1

    def __iter__(self):
        for i in range(len(self.studies)):
            self._fill()
            yield self.futs.pop(i).result()
        self.file_pool.shutdown()
        self.study_pool.shutdown()


class _StudyDataset(torch.utils.data.Dataset):
    """One study per item, read in a DataLoader worker process with its own file-read thread pool."""

    def __init__(self, df, threads):
        self.studies = [list(g.itertuples()) for _, g in df.groupby("StudyInstanceUID", sort=False)]
        self.threads, self.pool = threads, None

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        torch.set_num_threads(1)
        if self.pool is None:
            self.pool = ThreadPoolExecutor(self.threads)
        return read_study(self.studies[i], self.pool)


def process_reader(df, threads=THREADS, workers=2):
    """Reading in `workers` processes (the main process keeps its GIL for the GPU work); iteration starts now."""
    loader = torch.utils.data.DataLoader(_StudyDataset(df, threads), batch_size=1, collate_fn=lambda b: b[0],
                                         num_workers=workers, prefetch_factor=8)
    return iter(loader)


# ------------------------------------------------------------- GPU preprocessing


def gpu_cache(raw, unsigned):
    """infer.pipeline.gpu_cache without host syncs: the percentiles stay on the GPU."""
    import torch.nn.functional as F

    v = raw.int()
    if unsigned:
        v = v & 0xFFFF
    v = v.float()
    sample = v[:, ::4, ::4].reshape(-1)
    if sample.numel() <= 1 << 24:
        q = torch.quantile(sample, torch.tensor([0.005, 0.995], device=v.device))
        lo, hi = q[0], q[1]
    else:
        lo, hi = (torch.tensor(x, device=v.device) for x in np.percentile(sample.cpu().numpy(), [0.5, 99.5]))
    v = ((v - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)
    z, h, w = v.shape
    if h != w:
        s = max(h, w)
        y0, x0 = (s - h) // 2, (s - w) // 2
        v = F.pad(v, (x0, s - w - x0, y0, s - h - y0))
    t = v[:, None]
    if t.shape[-1] != CACHE_SIZE:
        t = F.interpolate(t, size=(CACHE_SIZE, CACHE_SIZE), mode="bilinear", antialias=True, align_corners=False)
    return (t[:, 0].clamp(0, 1) * 255).round().to(torch.uint8)


# ------------------------------------------------------------------------ models


def shareable(net):
    """Crop25D, late fusion, pooled slice vectors: the finding / plane embeddings enter after the encoder, so one
    encoder pass serves every finding read from the same crop."""
    return getattr(net, "fusion", None) == "late" and getattr(net, "head_type", None) != "map3d"


def encode(net, x, engine=None):
    """Crop25D.embed up to the slice vectors, without the finding / plane embedding -> (B, D, C).
    engine: TensorRT FrameEncoder (trt/export_onnx.py) of a fixed frame batch, used instead of the PyTorch encoder."""
    import torch.nn.functional as F

    b, d, s, _ = x.shape
    idx = torch.arange(d, device=x.device)
    trip = torch.stack([idx.sub(1).clamp(min=0), idx, idx.add(1).clamp(max=d - 1)], 1)
    frames = x[:, trip].reshape(b * d, 3, s, s)
    if engine is not None:
        n = engine.in_shape[0]
        assert len(frames) % n == 0, (len(frames), n)
        return torch.cat([engine(frames[i:i + n]) for i in range(0, len(frames), n)]).reshape(b, d, -1)
    if net.img_size and net.img_size != s:
        frames = F.interpolate(frames, size=(net.img_size, net.img_size), mode="bilinear", align_corners=False)
    net._d = d
    maps = net.encoder.forward_features((frames - net.mean) / net.std)
    return net.encoder.forward_head(maps, pre_logits=True).reshape(b, d, -1)


def head(net, f, roi, plane, both=False):
    """The rest of Crop25D.forward from the slice vectors (B, D, C)."""
    f = f + (net.roi_embed(roi) + net.plane_embed(plane))[:, None]
    if net.rnn is not None:
        f, _ = net.rnn(f)
    elif net.tf is not None:
        f = net.tf(net.proj(f) + net.pos[:, :f.shape[1]])
    w = net.attn(f).squeeze(-1).softmax(1)
    e = (w[..., None] * f).sum(1)
    main = net.fc(e).squeeze(-1)
    return torch.stack([main, net.fc_aux(e).squeeze(-1)], -1) if both else main


class Models:
    def __init__(self, weights, recipe, device, half=True, engines=None):
        """engines: folder of TensorRT engines (unet.engine, enc_<model>.engine, built on this GPU model by
        trt/build_benchmark.py); the networks without an engine stay in PyTorch."""
        from crops.train import GROUPS, PLANES, build_model as build_crop_model, crop_index, finding_of
        from roi.extract import ROI_CLASSES
        from roi.train import build_model
        from side_cls import SideNet

        self.device, self.dtype = device, torch.float16 if half else torch.float32
        sd = torch.load(os.path.join(weights, "side", "ckpt_final.pt"), map_location="cpu")
        self.side = SideNet(width=sd["stem.0.weight"].shape[0])
        self.side.load_state_dict(sd)
        self.side = self.side.to(device, self.dtype).eval()
        sd = torch.load(os.path.join(weights, "roi", "ckpt_final.pt"), map_location="cpu")
        k = sd["decoder.seg_layers.4.weight"].shape[0]
        self.roi_classes = ROI_CLASSES[:k]
        self.unet = build_model(num_classes=k)
        self.unet.load_state_dict(sd)
        self.unet = self.unet.to(device, self.dtype).eval()
        self.trt = {}
        if engines:
            sys.path.insert(0, os.path.join(HERE, "trt"))
            from trt_runtime import TRTModule

            for f in sorted(os.listdir(engines)):
                if f.endswith(".engine"):
                    self.trt[f[:-len(".engine")]] = TRTModule(os.path.join(engines, f), device)
        self.unet_trt = self.trt.get("unet")
        self.crops = []
        refs = {m for r in recipe.values() for m in r["models"]}
        self.restricted = [(x, x.split("@")[0], x.split("@")[1].split(",")) for x in sorted(refs) if "@" in x]
        for m in sorted({x.split("@")[0] for x in refs}):
            d = os.path.join(weights, "crops", m)
            a = json.load(open(os.path.join(d, "args.json")))
            group = GROUPS[a["group"]]
            sd = torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu")
            net = build_crop_model(a, len(group))
            n_planes = sd["plane_embed.weight"].shape[0]
            if n_planes != net.plane_embed.num_embeddings:
                net.plane_embed = torch.nn.Embedding(n_planes, net.plane_embed.embedding_dim)
            net.load_state_dict(sd)
            net = net.to(device, self.dtype).eval().to(memory_format=torch.channels_last)
            takes = {f for f, r in recipe.items() if any(x.split("@")[0] == m for x in r["models"])}
            aux = finding_of(a["aux"]) if a.get("aux") else None
            if aux in takes:
                used = list(group)
            else:
                used = [c for c in group if c in takes or ("|" in c and finding_of(c.split("|")[1]) in takes)]
            z = a.get("z_sampling") or dict(depth=12, z_mode="uniform", stride=1)
            z = {k: z[k] for k in ("depth", "z_mode", "stride", "size", "capsule", "capsule_up", "margin",
                                   "compartment", "oa_midline", "union", "fibula") if k in z}
            self.crops.append(dict(name=m, net=net, trt=self.trt.get(f"enc_{m}") if shareable(net) else None,
                                   z=json.dumps(z, sort_keys=True), planes=a["planes"].split(","),
                                   aux=a["aux"] if a.get("aux") else None, planes_all=PLANES,
                                   rois={c: (crop_index(c, "capsule" in z), group.index(c)) for c in used}))

    @torch.no_grad()
    def warmup(self):
        z, dt = torch.zeros, self.dtype
        from side_cls import model_input

        pl = z(SIDE_BATCH, dtype=torch.long, device=self.device)
        self.side(model_input(z((SIDE_BATCH, 1, VOL_DEPTH, VOL_SIZE, VOL_SIZE), device=self.device), pl).to(dt))
        (self.unet_trt or self.unet)(z((UNET_BATCH, 1, VOL_DEPTH, VOL_SIZE, VOL_SIZE), device=self.device, dtype=dt))
        for m in self.crops:
            i = z(CROP_BATCH, dtype=torch.long, device=self.device)
            k = json.loads(m["z"])
            x = z((CROP_BATCH, k["depth"], k.get("size", 128), k.get("size", 128)), device=self.device, dtype=dt)
            if m["trt"] is not None:
                head(m["net"], encode(m["net"], x, m["trt"]), i, i)
            else:
                m["net"](x, i, i)
        torch.cuda.synchronize()


def gpu_regions(prob, threshold=0.5):
    """roi.train.prob_to_boxes, GPU part: per (series, class) the peak, and the probabilities inside the bounding
    box of the thresholded region (the only voxels the CPU labelling looks at) -> [[(sub, offset, conf), ...], ...].
    One device-to-host copy of the sub-volumes instead of the whole (B, K, D, S, S) float32 maps."""
    b, k = prob.shape[:2]
    conf = prob.flatten(2).amax(2)                                           # B K
    thr = torch.where(conf >= threshold, torch.full_like(conf, threshold), 0.5 * conf)
    region = prob >= thr[..., None, None, None]
    bounds = []
    for axes in ((3, 4), (2, 4), (2, 3)):                                   # z, y, x
        a = region.any(axes[1]).any(axes[0])                                 # B K L
        idx = torch.arange(a.shape[-1], device=a.device)
        lo = torch.where(a, idx, a.shape[-1]).amin(-1)
        hi = torch.where(a, idx, -1).amax(-1)
        bounds += [lo, hi]
    bounds = torch.stack(bounds, -1).cpu().numpy()                           # B K 6
    flat, meta = [], []
    for i in range(b):
        for j in range(k):
            z0, z1, y0, y1, x0, x1 = bounds[i, j]
            flat.append(prob[i, j, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1].reshape(-1))
            meta.append(((z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1), (z0, y0, x0)))
    host = torch.cat(flat).cpu().numpy()
    conf = conf.cpu().numpy()
    out, o = [], 0
    for n, (shape, off) in enumerate(meta):
        size = int(np.prod(shape))
        out.append((host[o:o + size].reshape(shape), off, float(conf.flat[n])))
        o += size
    return [out[i * k:(i + 1) * k] for i in range(b)]


def boxes_from_regions(subs, threshold=0.5):
    """roi.train.prob_to_boxes on the bounding-box sub-volumes of gpu_regions (same components, same result)."""
    from scipy import ndimage as ndi

    out = []
    for p, (oz, oy, ox), conf in subs:
        region = p >= (threshold if conf >= threshold else 0.5 * conf)
        lab, n = ndi.label(region, structure=np.ones((3, 3, 3)))
        mass = ndi.sum(p, lab, index=np.arange(1, n + 1))
        comp = lab == int(np.argmax(mass)) + 1
        zs, ys, xs = np.nonzero(comp)
        out.append((dict(z0=zs.min() + oz, z1=zs.max() + oz, y0=ys.min() + oy, y1=ys.max() + oy,
                         x0=xs.min() + ox, x1=xs.max() + ox), conf, float(p[comp].mean())))
    return out


@torch.no_grad()
def process_gpu(studies, models, timer):
    """src/infer/pipeline.process_chunk without axial series, models in half precision: side model, U-Net and the
    GPU part of the box extraction (-> s["regions"] per series)."""
    from side_cls import PLANES as SIDE_PLANES, model_input

    dev, dt = models.device, models.dtype
    ser = [s for st in studies for s in st if s["status"] == "ok"]
    for s in ser:
        kr.GEOMETRY_OVERRIDE[s["SeriesInstanceUID"]] = s["geom"]

    t = time.time()
    p_right = []
    for i in range(0, len(ser), SIDE_BATCH):
        part = ser[i:i + SIDE_BATCH]
        x = _padded(torch.stack([s["volume"] for s in part])[:, None], SIDE_BATCH)
        pl = _padded(torch.tensor([SIDE_PLANES.index(s["plane"]) for s in part], device=dev), SIDE_BATCH)
        p_right.append(torch.sigmoid(models.side(model_input(x, pl).to(dt)).float())[:len(part)].cpu())
    p_right = torch.cat(p_right).numpy() if ser else np.zeros(0)
    for s, p in zip(ser, p_right):
        s["p_right"], s["side"] = float(p), "R" if p > 0.5 else "L"
    timer["side"] += time.time() - t

    t = time.time()
    seg = [s for s in ser if s["plane"] in UNET_PLANES]
    for i in range(0, len(seg), UNET_BATCH):
        part = seg[i:i + UNET_BATCH]
        x = torch.stack([s["volume"].flip(_lr_axis(s["plane"])) if s["side"] == "L" else s["volume"] for s in part])
        prob = torch.sigmoid((models.unet_trt or models.unet)(_padded(x[:, None], UNET_BATCH).to(dt) / 255).float())
        prob = prob[:len(part)]
        prob = torch.stack([prob[j].flip(_lr_axis(s["plane"]) + 1) if (s["side"] == "L") ^ s["conv_flip"]
                            else prob[j] for j, s in enumerate(part)])
        for s, sub in zip(part, gpu_regions(prob)):
            s["regions"] = sub
        timer["unet"] += time.time() - t
        t = time.time()
    return ser


@torch.no_grad()
def process_post(studies, ser, models, timer):
    """CPU (boxes, consensus) then GPU (crops, crop models) part of a chunk: runs in a worker thread while the main
    thread prepares the next chunk and runs its side model and U-Net."""
    from fast_consensus import consensus
    from roi.geometry import from_volume

    dev, dt = models.device, models.dtype
    t = time.time()
    pred = []
    for s in ser:
        if "regions" not in s:
            continue
        for k, (vb, conf, p_mean) in enumerate(boxes_from_regions(s.pop("regions"))):
            pred.append(dict(StudyInstanceUID=s["StudyInstanceUID"], SeriesInstanceUID=s["SeriesInstanceUID"],
                             plane=s["plane"], side=s["side"], **{"class": models.roi_classes[k]}, conf=conf,
                             p_mean=p_mean, **from_volume(vb, s["n_slices"], s["n_cache"], s["rows"], s["cols"],
                                                          VOL_DEPTH, VOL_SIZE)))
    timer["boxes"] += time.time() - t

    t = time.time()
    meta_cols = ["StudyInstanceUID", "SeriesInstanceUID", "plane", "n_slices", "n_cache", "rows", "cols",
                 "conv_flip", "side", "status"]
    series = pd.DataFrame([{k: s.get(k) for k in meta_cols} for s in ser], columns=meta_cols)
    boxes = pd.DataFrame()
    if pred:
        boxes, _, _ = consensus(pd.DataFrame(pred), series[series.plane.isin(ROI_PLANES)])
    timer["consensus"] += time.time() - t

    t = time.time()
    by_series = {}
    for r in boxes.to_dict("records"):
        by_series.setdefault(r["SeriesInstanceUID"], {})[r["class"]] = {k: int(r[k]) for k in
                                                                        ["z0", "z1", "y0", "y1", "x0", "x1"]}
    crop_rows = [s for s in ser if s["SeriesInstanceUID"] in by_series]
    crops = {}
    for z in {m["z"] for m in models.crops} if crop_rows else ():          # no usable series: nothing to cut
        kw = json.loads(z)
        cap, cap_up = kw.pop("capsule", None), kw.pop("capsule_up", None)
        compartment, midline = kw.pop("compartment", None), kw.pop("oa_midline", "")
        union = [c for c in kw.pop("union", "").split(",") if c]
        fibula = kw.pop("fibula", False)
        if fibula:
            from fracture.fibula import fibula_box
            from roi.extract import ROI_CLASSES

            series_boxes = {}
            for s in crop_rows:
                bx = dict(by_series.get(s["SeriesInstanceUID"], {}))
                fb = fibula_box(bx, s["plane"], s["rows"], s["cols"], s["n_slices"])
                if fb is not None:
                    bx["Fibula"] = fb
                series_boxes[s["SeriesInstanceUID"]] = bx
            kw.update(classes=ROI_CLASSES + ["Fibula"])
        elif cap is not None:
            from crops.prepare import capsule_boxes

            shape = {s["SeriesInstanceUID"]: (s["rows"], s["cols"]) for s in crop_rows}
            src = boxes[boxes.SeriesInstanceUID.isin(shape)]
            series_boxes = capsule_boxes(src, cap, cap_up, shape, union or None)
            kw.update(classes=["Capsule"], margin=0.0)
        else:
            series_boxes = by_series
            if compartment is not None or midline:
                from crops.prepare import compartment_box, oa_midline

                series_boxes = {}
                for s in crop_rows:
                    uid = s["SeriesInstanceUID"]
                    bx = dict(by_series.get(uid, {}))
                    if compartment is not None:
                        for oa, men in (("Medial OA", "Medial Meniscus"), ("Lateral OA", "Lateral Meniscus")):
                            if oa in bx and men in bx:
                                bx[oa] = compartment_box(bx[oa], bx[men], compartment, (s["rows"], s["cols"]))
                    if midline and s["plane"] == "Coronal":
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

    out = []
    for m in models.crops if crop_rows else []:
        t = time.time()
        x_all, have = crops[m["z"]]
        items = [(j, roi, rid, m["planes_all"].index(s["plane"]), s["StudyInstanceUID"], c)
                 for j, s in enumerate(crop_rows) if s["plane"] in m["planes"]
                 for c, (roi, rid) in m["rois"].items() if have[j, roi]]
        res = []
        if m["trt"] is not None or (shareable(m["net"]) and len({r for r, _ in m["rois"].values()}) < len(m["rois"])):
            # Encoder (TensorRT engine, or PyTorch when several findings share a crop) once per (series, crop),
            # then the heads per finding.
            uniq = sorted({(j, roi) for j, roi, *_ in items})
            pos = {u: k for k, u in enumerate(uniq)}
            feats = []
            for i in range(0, len(uniq), CROP_BATCH):
                part = uniq[i:i + CROP_BATCH]
                j, roi = (torch.tensor(v, device=dev) for v in zip(*part))
                feats.append(encode(m["net"], _padded(x_all[j, roi], CROP_BATCH).to(dt) / 255, m["trt"])[:len(part)])
            feats = torch.cat(feats)
            u = torch.tensor([pos[(j, roi)] for j, roi, *_ in items], device=dev)
            rid = torch.tensor([it[2] for it in items], device=dev)
            pl = torch.tensor([it[3] for it in items], device=dev)
            res.append(torch.sigmoid(head(m["net"], feats[u], rid, pl, bool(m["aux"])).float()))
        else:
            for i in range(0, len(items), CROP_BATCH):
                part = items[i:i + CROP_BATCH]
                j, roi, rid, pl = (torch.tensor(v, device=dev) for v in list(zip(*part))[:4])
                x = _padded(x_all[j, roi], CROP_BATCH).to(dt) / 255
                p = torch.sigmoid(m["net"](x, _padded(rid, CROP_BATCH), _padded(pl, CROP_BATCH),
                                           **({"both": True} if m["aux"] else {})).float())
                res.append(p[:len(part)])
        if res:
            p = torch.cat(res).cpu().numpy()
            for k, (_, _, _, pl_, st, c) in enumerate(items):
                if m["aux"]:
                    out.append((m["name"], st, f"{c}|{m['aux']}", m["planes_all"][pl_], float(p[k, 1])))
                    out.append((m["name"], st, c, m["planes_all"][pl_], float(p[k, 0])))
                else:
                    out.append((m["name"], st, c, m["planes_all"][pl_], float(p[k])))
        dt_ = time.time() - t
        timer["models"] += dt_
        timer["model " + m["name"]] = timer.get("model " + m["name"], 0.0) + dt_
    rows = pd.DataFrame(out, columns=["model", "StudyInstanceUID", "cls", "plane", "p"])
    preds = rows.groupby(["model", "StudyInstanceUID", "cls"], as_index=False).p.mean()
    from crops.train import finding_of
    sub = preds[preds.cls.str.contains("|", regex=False)]
    if len(sub):
        sub = sub.assign(cls=sub.cls.str.split("|").str[1].map(finding_of)).dropna(subset=["cls"])
        preds = pd.concat([preds, sub.groupby(["model", "StudyInstanceUID", "cls"], as_index=False).p.max()],
                          ignore_index=True)
    for ref, base, planes in models.restricted:
        sub = rows[(rows.model == base) & rows.plane.isin(planes)]
        preds = pd.concat([preds, sub.groupby(["StudyInstanceUID", "cls"], as_index=False).p.mean().assign(model=ref)],
                          ignore_index=True)
    series = series.assign(p_right=[s["p_right"] for s in ser], dicom_laterality=[s["dicom_laterality"] for s in ser])
    failed = pd.DataFrame([{k: s.get(k) for k in ["StudyInstanceUID", "SeriesInstanceUID", "plane", "status"]}
                           for st in studies for s in st if s["status"] != "ok"])
    return pd.concat([series, failed], ignore_index=True), boxes, preds


def run_shard(df, weights, recipe, out_dir, threads, half=True):
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    dev = torch.device("cuda")
    t0 = time.time()
    reader = process_reader(df, threads, READ_PROCS) if READ_PROCS else StudyReader(df, threads)
    models = Models(weights, recipe, dev, half, os.environ.get("KNEE_ENGINES") or None)
    print(f"TensorRT engines: {sorted(models.trt) or 'none'}", flush=True)
    timer = dict.fromkeys(["load", "warmup", "wait", "prep", "side", "unet", "boxes", "consensus", "crops",
                           "models"], 0.0)
    timer["load"] = time.time() - t0
    t = time.time()
    models.warmup()
    timer["warmup"] = time.time() - t
    all_series, all_boxes, all_preds, chunk = [], [], [], []
    n_studies = df.StudyInstanceUID.nunique()

    post = ThreadPoolExecutor(1)
    pending = []

    def collect():
        s, b, p_ = pending.pop(0).result()
        all_series.append(s)
        all_boxes.append(b)
        all_preds.append(p_)

    def flush():
        studies = list(chunk)
        ser = process_gpu(studies, models, timer)
        while pending:                      # one chunk in the worker at a time (GPU memory)
            collect()
        pending.append(post.submit(process_post, studies, ser, models, timer))
        chunk.clear()

    t = time.time()
    for n, study in enumerate(reader):
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
            print(f"  {n + 1}/{n_studies} studies, {time.time() - t0:.0f}s", flush=True)
        t = time.time()
    if chunk:
        flush()
    while pending:
        collect()
    post.shutdown()
    os.makedirs(out_dir, exist_ok=True)
    pd.concat(all_series, ignore_index=True).to_csv(os.path.join(out_dir, "series.csv"), index=False)
    pd.concat(all_boxes, ignore_index=True).to_csv(os.path.join(out_dir, "rois.csv"), index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(os.path.join(out_dir, "preds.csv"), index=False)
    total = time.time() - t0
    print(f"shard done: {n_studies} studies in {total:.0f}s -- " +
          ", ".join(f"{k} {v:.1f}s" for k, v in timer.items()) +
          f" | peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def run(data_dir, weights, out_dir, submission, gpus, threads, series_dir="test_series", series_csv=None,
        limit=None, rule="fs", recipe_path=None, half=True, studies=None):
    t0 = time.time()
    recipe_path = recipe_path or os.path.join(weights, "recipe.json")
    recipe = load_recipe(recipe_path)
    df = list_series(data_dir, series_dir, series_csv, rule)
    if studies is not None:
        df = df[df.StudyInstanceUID.isin(set(studies))]
    if limit:
        df = df[df.StudyInstanceUID.isin(df.StudyInstanceUID.drop_duplicates()[:limit])]
    os.makedirs(out_dir, exist_ok=True)
    for d in glob.glob(os.path.join(out_dir, "shard*")):
        shutil.rmtree(d)
    df["shard"] = df.StudyInstanceUID.map(shard_studies(df[df["rank"] == 0], len(gpus)))   # first choices
    df.to_csv(os.path.join(out_dir, "series_list.csv"), index=False)
    shards = sorted(df["shard"].unique())
    procs = []
    for i in shards:
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--shard", str(i), "--weights", weights, "--recipe",
             recipe_path, "--out_dir", out_dir, "--threads", str(threads)] + ([] if half else ["--fp32"]),
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[i]))))
    if any(p.wait() for p in procs):
        raise RuntimeError("a shard failed")
    preds = pd.concat([pd.read_csv(os.path.join(out_dir, f"shard{i}", "preds.csv"), dtype={"model": str})
                       for i in shards])
    ids = df.StudyInstanceUID.drop_duplicates().tolist()
    study = combine(preds, recipe, ids)
    study.to_csv(os.path.join(out_dir, "roi_study_preds.csv"))
    sub = pd.DataFrame({"StudyInstanceUID": ids})
    for c in CLASSES:
        sub[c] = sub.StudyInstanceUID.map(study[c]) if c in study else np.nan
    sub[CLASSES] = sub[CLASSES].fillna(0.5)
    sub.to_csv(submission, index=False)
    print(f"-> {submission}: {len(sub)} studies, {time.time() - t0:.0f}s in total", flush=True)
    return sub


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir")
    p.add_argument("--weights", required=True)
    p.add_argument("--recipe", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--submission", default="submission.csv")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--series_dir", default="test_series")
    p.add_argument("--series_csv", default=None)
    p.add_argument("--rule", default="fs", help="series selection: fs (1 sag + 1 cor, fluid-sensitive first) / all")
    p.add_argument("--studies", default=None, help="csv with a StudyInstanceUID column: only these studies")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--engines", default=None, help="TensorRT engines folder (trt/build_benchmark.py, same GPU model)")
    p.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.shard is not None:
        recipe = load_recipe(a.recipe)
        df = pd.read_csv(os.path.join(a.out_dir, "series_list.csv"))
        df = df[df["shard"] == a.shard]
        run_shard(df, a.weights, recipe, os.path.join(a.out_dir, f"shard{a.shard}"), a.threads, not a.fp32)
        return
    if a.engines:
        os.environ["KNEE_ENGINES"] = os.path.abspath(a.engines)
    studies = pd.read_csv(a.studies).StudyInstanceUID.tolist() if a.studies else None
    run(a.data_dir, a.weights, a.out_dir, a.submission, a.gpus.split(","), a.threads, a.series_dir, a.series_csv,
        a.limit, a.rule, a.recipe, not a.fp32, studies)


if __name__ == "__main__":
    main()
