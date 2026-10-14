"""OrthoFoundation (DINOv3 ViT-L/16, knee MRI + X-ray continued pretraining) features, run from src_bis/.

    python -m mt3d.ortho --mode roi                 # the main agent's 8-frame ROI crops (output/crops/f_s1n8)
    python -m mt3d.ortho --mode joint               # the joint crops (output/bis/mt3d/u10_d32_s256), ROI-pooled
    python -m mt3d.ortho --mode full                # full slices from the 384 px cache, ROI-pooled

Model: github.com/ytrsk/OrthoFoundation, `OrthoFoudation-L.pth` (Meta DINOv3 key names), loaded into timm's
`vit_large_patch16_dinov3` (`load_ortho`; checked against Meta's dinov3 code: max |diff| 2.6e-3 on patch
tokens of mean magnitude 0.21, fp32). Preprocessing as in its pretraining: a grayscale slice replicated to
3 channels, ImageNet mean / std, 224 px. Output tokens: 1 CLS + 4 registers + 14 x 14 patches, layer-normed.

Saved per frame (float16, output/bis/ortho/<mode>.npy):
  roi    (N series, 8 ROIs, 8 frames, 2, 1024): CLS and mean patch token of each ROI crop frame;
  joint  (N, 16 frames, 10, 1024): CLS, mean patch token, and the patch tokens averaged under each of the
         8 ROI boxes (the crop's box fractions; the box's own slices only, else its nearest frame);
  full   as joint, on the whole slice (the square 384 px cache, 16 frames over the ROI union's slices).
Sharded over the GPUs (one process each), written into one memmap.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from params import CACHE_DIR, OUTPUT_FOLDER

ORTHO_DIR = os.path.join(OUTPUT_FOLDER, "bis", "ortho")
WEIGHTS = os.path.join(ORTHO_DIR, "OrthoFoudation-L.pth")
MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
DIM, GRID = 1024, 14


def load_ortho(weights=WEIGHTS, img_size=224, generic=False):
    """OrthoFoundation; generic=True: the DINOv3-L it started from (timm lvd1689m weights, the control)."""
    import timm
    from timm.models.eva import checkpoint_filter_fn

    if generic:
        return timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=img_size)

    sd = torch.load(weights, map_location="cpu")
    sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items()}
    m = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0, img_size=img_size)
    m.load_state_dict(checkpoint_filter_fn(sd, m), strict=True)
    return m


def encode(model, gray, size=224):
    """gray (n, H, W) float in [0, 1] on the GPU -> tokens (n, 1 + 4 + 196, 1024), autocast fp16."""
    x = F.interpolate(gray[:, None], size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    x = (x.expand(-1, 3, -1, -1) - MEAN.to(x)) / STD.to(x)
    with torch.autocast("cuda", dtype=torch.float16):
        return model.forward_features(x)


def box_weights(boxes, z_pos, grid=GRID):
    """boxes (K, 6) fractions (z0, z1, y0, y1, x0, x1), z_pos (F,) frame centres as fractions ->
    (F, K, grid * grid) pooling weights: the patches whose centre is inside the box, on frames inside its
    slices (a box with no frame inside: its nearest frame)."""
    c = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    iny = (c[None] >= boxes[:, 2:3]) & (c[None] < boxes[:, 3:4])                    # K g
    inx = (c[None] >= boxes[:, 4:5]) & (c[None] < boxes[:, 5:6])
    # at least the patch row / column holding the box centre
    iny |= torch.arange(grid)[None] == ((boxes[:, 2] + boxes[:, 3]) / 2 * grid).long().clamp(0, grid - 1)[:, None]
    inx |= torch.arange(grid)[None] == ((boxes[:, 4] + boxes[:, 5]) / 2 * grid).long().clamp(0, grid - 1)[:, None]
    sp = (iny[:, :, None] & inx[:, None, :]).flatten(1).float()                       # K g*g
    inz = (z_pos[:, None] >= boxes[None, :, 0]) & (z_pos[:, None] < boxes[None, :, 1])  # F K
    near = (z_pos[:, None] - (boxes[None, :, 0] + boxes[None, :, 1]) / 2).abs().argmin(0)
    inz[near, torch.arange(len(boxes))] = True
    w = inz.float()[:, :, None] * sp[None]
    return w / w.sum(-1, keepdim=True).clamp(min=1)


def _worker(rank, n_gpu, mode, out_path, shape, generic=False, crops="u10_d32_s256"):
    dev = torch.device("cuda", 0)                  # each process sees only its own GPU (CUDA_VISIBLE_DEVICES)
    model = load_ortho(generic=generic).to(dev).eval()
    out = np.load(out_path, mmap_mode="r+")
    if mode == "roi":
        arr = np.load(os.path.join(OUTPUT_FOLDER, "crops", "f_s1n8.npy"), mmap_mode="r")      # N 8 8 128 128
        idx = np.arange(rank, len(arr), n_gpu)
        for k, i in enumerate(idx):
            x = torch.from_numpy(np.array(arr[i])).to(dev).float().flatten(0, 1) / 255             # 64 128 128
            with torch.no_grad():
                t = encode(model, x).float()
            f = torch.stack([t[:, 0], t[:, 5:].mean(1)], 1).reshape(8, 8, 2, DIM)
            out[i] = f.half().cpu().numpy()
            if rank == 0 and k % 500 == 0:
                print(f"  [{mode}] {k}/{len(idx)}", flush=True)
    else:
        from mt3d.train import load_crops
        arr, boxes, info = load_crops(crops)
        frames = np.arange(16) * 2 + 1                         # 16 of the 32 slices (the hybrids' eval offset)
        z_pos = torch.tensor((frames + 0.5) / 32, dtype=torch.float32)
        idx = np.arange(rank, len(arr), n_gpu)
        for k, i in enumerate(idx):
            bx = torch.from_numpy(boxes[i]).float()
            if mode == "joint":
                x = torch.from_numpy(np.array(arr[i][frames])).to(dev).float() / 255             # 16 256 256
                w = box_weights(bx, z_pos)
            else:
                r = info.iloc[i]
                cache = np.load(os.path.join(CACHE_DIR, f"{r.SeriesInstanceUID}.npy"))           # n_c 384 384
                n_c, cs = cache.shape[0], cache.shape[1]
                # The same slices as the joint crop (its z-range), but the whole square slice in-plane;
                # boxes from crop fractions to whole-slice fractions.
                zs, y0, x0, n = _union_window(r, n_c, cs)
                zf = np.interp(frames, np.arange(32), np.linspace(zs[0], zs[-1], 32))
                lo = np.floor(zf).astype(int).clip(0, n_c - 1); hi = (lo + 1).clip(max=n_c - 1); wz = (zf - lo)[:, None, None]
                sl = (1 - wz) * cache[lo] + wz * cache[hi]
                if r.side == "L":
                    sl = sl[::-1] if r.Anatomical_Plane == "Sagittal" else sl[:, :, ::-1]
                x = torch.from_numpy(np.ascontiguousarray(sl)).to(dev).float() / 255
                fb = bx.clone()
                fb[:, 2:4] = (bx[:, 2:4] * n + y0) / cs
                if r.side == "L" and r.Anatomical_Plane == "Coronal":
                    x0 = cs - x0 - n                             # the crop window, mirrored with the slice
                fb[:, 4:6] = (bx[:, 4:6] * n + x0) / cs
                w = box_weights(fb.clamp(0, 1), z_pos)
            with torch.no_grad():
                t = encode(model, x).float()                                                     # 16 201 1024
            p = t[:, 5:]
            roi = torch.einsum("fkn,fnc->fkc", w.to(dev), p)
            f = torch.cat([t[:, :1], p.mean(1, keepdim=True), roi], 1)                          # 16 10 1024
            out[i] = f.half().cpu().numpy()
            if rank == 0 and k % 500 == 0:
                print(f"  [{mode}] {k}/{len(idx)}", flush=True)
    out.flush()


_WINDOWS = {}


def _union_window(r, n_c, cs):
    """The joint crop's slice positions and in-plane window (mt3d.prepare), from its union box."""
    from crops.prepare import capsule_boxes, crop_window
    from roi.labels import LABEL_DIR

    if not _WINDOWS:
        pred = pd.read_csv(os.path.join(LABEL_DIR, "round3_pred.csv"))
        vols = pd.read_csv(os.path.join(OUTPUT_FOLDER, "roi", "volumes.csv"))
        shape = {v.SeriesInstanceUID: (v.rows, v.cols) for v in vols.itertuples()}
        _WINDOWS.update(capsule_boxes(pred[pred.SeriesInstanceUID.isin(shape)], 0.10, None, shape))
    u = _WINDOWS[r.SeriesInstanceUID]["Capsule"]
    return crop_window(n_c, cs, u, r.n_slices, r.rows, r.cols, 32, 0.0, 0, "resize")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["roi", "joint", "full"])
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--crops", default="u10_d32_s256", help="joint mode: the joint-crop set (ours, or external)")
    p.add_argument("--model", default="ortho", choices=["ortho", "dinov3"], help="dinov3: the generic DINOv3-L control")
    a = p.parse_args()
    os.makedirs(ORTHO_DIR, exist_ok=True)
    if a.mode == "roi":
        n = len(np.load(os.path.join(OUTPUT_FOLDER, "crops", "f_s1n8.npy"), mmap_mode="r"))
        shape = (n, 8, 8, 2, DIM)
    else:
        from mt3d.train import load_crops
        n = len(load_crops(a.crops)[2])
        shape = (n, 16, 10, DIM)
    tag = "" if a.crops == "u10_d32_s256" else f"_{a.crops}"            # ours: joint.npy; external: joint_<crops>.npy
    path = os.path.join(ORTHO_DIR, f"{a.mode}{tag}{'_dinov3' if a.model == 'dinov3' else ''}.npy")
    np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape).flush()
    gpus = [int(g) for g in a.gpus.split(",")]
    # One process per GPU (each sees only its device).
    ctx = mp.get_context("spawn")
    procs = []
    for rank, g in enumerate(gpus):
        pr = ctx.Process(target=_worker, args=(rank, len(gpus), a.mode, path, shape, a.model == "dinov3", a.crops))
        procs.append(pr)
    for rank, pr in enumerate(procs):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[rank])
        pr.start()
    for pr in procs:
        pr.join()
    print(f"-> {path} {shape}")


if __name__ == "__main__":
    main()
