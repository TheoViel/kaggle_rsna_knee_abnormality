"""Build TensorRT engines from the ONNX exports and benchmark them against PyTorch fp16, on the current GPU.

    python trt/build_benchmark.py --onnx <dataset_efficiency_onnx> --weights <dataset_efficiency> --out <engines dir>

Per network: build time, PyTorch fp16 vs TensorRT latency (same fixed batch), speedup, and output parity (max abs
difference, correlation). Writes <out>/<name>.engine, <out>/benchmark.csv and <out>/trt_meta.json (GPU, versions).
Run on the target GPU (Kaggle T4): engines only load on the GPU model / TensorRT version that built them.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EFF = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, EFF)
sys.path.insert(0, os.path.dirname(EFF))                 # vendored packages (Kaggle code dataset)
sys.path.insert(0, os.environ.get("KNEE_SRC", os.path.join(EFF, "..", "src")))

from export_onnx import FrameEncoder  # noqa: E402
from trt_runtime import TRTModule, build_engine, engine_name  # noqa: E402


def bench(fn, x, n=10):
    with torch.no_grad():
        for _ in range(2):
            fn(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            fn(x)
        torch.cuda.synchronize()
    return (time.time() - t) / n * 1e3


def torch_net(name, weights, dev):
    import torch.nn as nn

    from crops.train import GROUPS, build_model as build_crop_model
    from roi.train import build_model

    if name == "unet":
        sd = torch.load(os.path.join(weights, "roi", "ckpt_final.pt"), map_location="cpu")
        net = build_model(num_classes=sd["decoder.seg_layers.4.weight"].shape[0])
        net.load_state_dict({k: v.float() for k, v in sd.items()})
        return net.to(dev).half().eval()
    d = os.path.join(weights, "crops", name)
    args = json.load(open(os.path.join(d, "args.json")))
    sd = torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu")
    net = build_crop_model(args, len(GROUPS[args["group"]]))
    n_planes = sd["plane_embed.weight"].shape[0]
    if n_planes != net.plane_embed.num_embeddings:
        net.plane_embed = nn.Embedding(n_planes, net.plane_embed.embedding_dim)
    net.load_state_dict({k: v.float() for k, v in sd.items()})
    net = net.to(dev).half().eval().to(memory_format=torch.channels_last)
    return FrameEncoder(net).eval()


def main():
    import tensorrt as trt

    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nets", default="", help="comma-separated (default: all in onnx_meta.json)")
    p.add_argument("--opt_level", type=int, default=3)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.onnx, "onnx_meta.json")))
    nets = a.nets.split(",") if a.nets else list(meta["nets"])
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    rows = []
    for name in nets:
        onnx_name = "unet.onnx" if name == "unet" else f"enc_{name}.onnx"
        eng = os.path.join(a.out, engine_name(onnx_name))
        t = time.time()
        build_engine(os.path.join(a.onnx, onnx_name), eng, opt_level=a.opt_level)
        t_build = time.time() - t
        m = TRTModule(eng)
        net = torch_net(name, a.weights, dev)
        torch.manual_seed(0)
        x = torch.rand(meta["nets"][name]["input"], device=dev).half()
        if name != "unet":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            ref = net(x).float()
            out = m(x).float()
        d = (out - ref).abs()
        corr = float(np.corrcoef(out.flatten()[:2_000_000].cpu(), ref.flatten()[:2_000_000].cpu())[0, 1])
        t_torch, t_trt = bench(net, x), bench(m, x)
        rows.append(dict(net=name, build_s=round(t_build, 1), torch_ms=round(t_torch, 2), trt_ms=round(t_trt, 2),
                         speedup=round(t_torch / t_trt, 2), max_abs=float(d.max()), corr=round(corr, 6),
                         engine_mb=round(os.path.getsize(eng) / 1e6)))
        print(rows[-1], flush=True)
        del net, m, x, ref, out
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "benchmark.csv"), index=False)
    print(df.to_string(index=False))
    json.dump(dict(gpu=torch.cuda.get_device_name(0), tensorrt=trt.__version__, torch=torch.__version__,
                   cuda=torch.version.cuda, nets=nets), open(os.path.join(a.out, "trt_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
