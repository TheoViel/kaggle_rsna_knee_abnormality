"""Train the 12-class knee classifier with xy / z / series attention, run from src/.

    python main_cls.py --epochs 1 --num_workers 2           # smoke test, 1 GPU
    torchrun --standalone --nproc_per_node=4 main_cls.py    # the real thing

Trains on the Claude report labels (soft), validates on the 58 studies with
radiologist labels in train.csv, which are never trained on. `--holdout_frac`
additionally holds out report-labelled studies for a steadier comparison
between runs; set it to 0 for the final fit.

See `knee_cls/model.py` for the architecture and how its attention maps become
per-class 3D localisations, and `roi/extract.py` to export them.
"""

import argparse
import json
import os
import warnings

# Batches are ragged (studies have 3-14 series of 11-64 slices), so without this
# the caching allocator fragments and stalls on OOM retries near the memory cap.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# DDP complains once per step about the 1x1-conv grad layout; harmless.
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

from knee_cls.main import run
from params import LOG_FOLDER
from training.distributed import (broadcast_object, cleanup_distributed,
                                  init_distributed, seed_everything)
from utils.logger import create_logger, prepare_log_folder


DEFAULTS = {
    "holdout_frac": 0.1,

    # Data. The CNN sees `img_size` frames resampled from the 384 cache; a
    # frame is 3 slices `frame_offset` apart, as channels.
    "img_size": 224,
    "max_series": 8,            # train: random subset when a study has more
    "val_max_series": 12,
    "n_frames": 6,              # frames per series, strided over the stack
    "val_n_frames": 6,
    "frame_offset": 1,
    "z_trim": 0.1,              # fraction of slices dropped at each end
    "augmentations": {
        "affine_p": 0.8, "rot_deg": 15.0, "scale_range": (0.85, 1.15), "shift": 0.08,
        "intensity_p": 0.8, "contrast": 0.2, "brightness": 0.1, "gamma": 0.25,
        "noise_std": 0.02,
    },

    # Model
    "backbone": "convnext_tiny.fb_in22k",
    "pretrained": True,
    "dim": 256,
    "z_layers": 1,
    "z_heads": 4,
    "dropout": 0.1,
    "drop_path": 0.1,
    "xy_heads": 1,              # in-plane attention maps per class
    "attn_stride": 8,           # attention-map resolution: 8 (3-level FPN) or 16
    "grad_checkpointing": False,

    # Optimisation. batch_size is studies per GPU.
    "epochs": 20.0,
    "batch_size": 8,
    "val_batch_size": 8,
    "num_workers": 8,
    "lr": 3e-4,
    "lr_backbone": 2e-4,
    "weight_decay": 0.05,
    "warmup_prop": 0.05,
    "max_grad_norm": 1.0,
    "aux_weight": 0.25,
    "ema_decay": 0.999,
    "amp": True,

    "log_path": LOG_FOLDER + "cls_run/",
    "log_every": 25,
    "eval_every_epochs": 1,
    "seed": 0,
    "dist_backend": None,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        if isinstance(v, dict):
            p.add_argument(f"--{k}", type=json.loads, default=None,
                           help="JSON, merged into the defaults")
        elif isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: str(s).lower() != "false", default=v)
        elif v is None:
            p.add_argument(f"--{k}", default=None)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    aug = dict(DEFAULTS["augmentations"])
    aug.update(args.augmentations or {})
    args.augmentations = aug
    return args


def main():
    args = parse_args()
    ctx = init_distributed(args.dist_backend)
    seed_everything(int(args.seed), ctx.rank)

    output_dir = None
    if ctx.is_main:
        os.makedirs(args.log_path, exist_ok=True)
        output_dir = prepare_log_folder(args.log_path)
    output_dir = broadcast_object(output_dir, device=ctx.device)
    args.output_dir = output_dir

    if ctx.is_main:
        create_logger(directory=output_dir, name="logs.txt")
        print(f"-> output {output_dir}  world {ctx.world_size}")
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump({k: getattr(args, k, None) for k in DEFAULTS}, f, indent=4,
                      default=str)
    try:
        run(args, ctx)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
