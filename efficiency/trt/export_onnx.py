"""ONNX exports of the efficiency pipeline's networks, for TensorRT engines built on the target GPU (T4), run from
efficiency/ with the onnx / onnxruntime packages importable.

    python trt/export_onnx.py --weights ../output/dataset_efficiency --out ../output/dataset_efficiency_onnx

Exported (fixed shapes, float16 graphs: TensorRT builds them strongly typed, i.e. fp16 as the PyTorch pipeline; TensorRT
11 has no fp16 builder flag any more):
  unet.onnx           (16, 1, 32, 128, 128) in [0, 1]       -> U-Net logits (16, K, 32, 128, 128)
  enc_<model>.onnx    (512, 3, S, S) frames in [0, 1]       -> slice vectors (512, C)
                      = Crop25D.encode's per-frame part: resize to img_size, mean / std, encoder, pooled head.
                      The finding / plane embeddings, LSTM and attention pooling stay in PyTorch (pipeline.head).
Parity: onnxruntime (CUDA, fp16 graph) vs PyTorch fp16 on random inputs, max abs difference and correlation printed.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

FRAMES = 512          # frames per encoder call (64 crops x 8 slices; the 24-slice joint crops in 3 calls)
UNET_BATCH = 16


class FrameEncoder(nn.Module):
    """(N, 3, S, S) frames in [0, 1] -> (N, C) slice vectors (Crop25D.encode without the slice triplets)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, frames):
        n = self.net
        if n.img_size and n.img_size != frames.shape[-1]:
            frames = F.interpolate(frames, size=(n.img_size, n.img_size), mode="bilinear", align_corners=False)
        maps = n.encoder.forward_features((frames - n.mean) / n.std)
        return n.encoder.forward_head(maps, pre_logits=True)


def parity(path, torch_fn, x):
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
    with torch.no_grad():
        ref = torch_fn(x).float().cpu().numpy()
    out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    d = np.abs(out - ref)
    c = np.corrcoef(out.ravel()[:2_000_000], ref.ravel()[:2_000_000])[0, 1]
    return float(d.max()), float(d.mean()), float(c)


@torch.no_grad()
def export(module, x, path):
    torch.onnx.export(module, (x,), path, input_names=["x"], output_names=["y"], opset_version=17, dynamo=False,
                      do_constant_folding=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.join(HERE, "..", "output", "dataset_efficiency"))
    p.add_argument("--out", default=os.path.join(HERE, "..", "output", "dataset_efficiency_onnx"))
    p.add_argument("--models", default="", help="comma-separated crop models (default: every crops/ folder)")
    p.add_argument("--no_unet", action="store_true")
    a = p.parse_args()
    from crops.train import GROUPS, build_model as build_crop_model
    from roi.train import build_model

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    meta = {}
    if not a.no_unet:
        sd = torch.load(os.path.join(a.weights, "roi", "ckpt_final.pt"), map_location="cpu")
        k = sd["decoder.seg_layers.4.weight"].shape[0]
        unet = build_model(num_classes=k)
        unet.load_state_dict({key: v.float() for key, v in sd.items()})
        unet = unet.to(dev).half().eval()
        x = torch.rand(UNET_BATCH, 1, 32, 128, 128, device=dev).half()
        path = os.path.join(a.out, "unet.onnx")
        export(unet, x, path)
        meta["unet"] = dict(input=list(x.shape), classes=k)
        print("unet", "max / mean abs diff, corr: %.2e %.2e %.6f" % parity(path, unet, x), flush=True)
    names = a.models.split(",") if a.models else sorted(os.listdir(os.path.join(a.weights, "crops")))
    for m in names:
        d = os.path.join(a.weights, "crops", m)
        args = json.load(open(os.path.join(d, "args.json")))
        sd = torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu")
        net = build_crop_model(args, len(GROUPS[args["group"]]))
        n_planes = sd["plane_embed.weight"].shape[0]
        if n_planes != net.plane_embed.num_embeddings:
            net.plane_embed = nn.Embedding(n_planes, net.plane_embed.embedding_dim)
        net.load_state_dict({key: v.float() for key, v in sd.items()})
        net = net.to(dev).half().eval()
        assert net.fusion == "late" and net.head_type != "map3d", (m, net.fusion, net.head_type)
        size = (args.get("z_sampling") or {}).get("size", 128)
        enc = FrameEncoder(net).eval()
        x = torch.rand(FRAMES, 3, size, size, device=dev).half()
        path = os.path.join(a.out, f"enc_{m}.onnx")
        export(enc, x, path)
        with torch.no_grad():
            dim = enc(x[:2]).shape[1]
        meta[m] = dict(input=list(x.shape), dim=dim, backbone=args["backbone"], img_size=args.get("img_size") or 0)
        print(m, args["backbone"], "max / mean abs diff, corr: %.2e %.2e %.6f" % parity(path, enc, x), flush=True)
        del net, enc, x, sd
        torch.cuda.empty_cache()
    json.dump(dict(frames=FRAMES, unet_batch=UNET_BATCH, nets=meta), open(os.path.join(a.out, "onnx_meta.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
