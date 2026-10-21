"""A crops_run's training command, rebuilt from its args.json (non-default values), with overrides; run from src/.

    python tools_cmd.py 2026-10-12/58 fs_only=True seed=1 n_folds=0 fold=0
"""
import json
import sys

from main_crops import DEFAULTS

SKIP = {"output_dir", "augmentations", "z_sampling", "local_rank", "world_size", "rank", "distributed"}
HEAD = ("set -o pipefail; cd /data/rsna_knee/src && ../.venv/bin/torchrun --standalone --nproc_per_node=$NP "
        "--master_port $PORT main_crops.py")


def q(v):
    v = str(v)
    return "'" + v + "'" if any(ch in v for ch in " '\"") else v


def command(run, **over):
    a = json.load(open(f"../logs/crops_run/{run}/args.json"))
    a.update(over)
    parts = [f"--{k} {q(v)}" for k, v in a.items() if k in DEFAULTS and k not in SKIP and v != DEFAULTS[k]
             and not (isinstance(v, list) and list(DEFAULTS[k] or []) == v)]
    return HEAD + " " + " ".join(parts)


if __name__ == "__main__":
    over = {}
    for kv in sys.argv[2:]:
        k, v = kv.split("=", 1)
        over[k] = {"True": True, "False": False}.get(v, int(v) if v.lstrip("-").isdigit() else v)
    print(command(sys.argv[1], **over))
