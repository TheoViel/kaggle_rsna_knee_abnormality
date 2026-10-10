"""Train a cross-compartment model (`crops.study.CrossCompartmentModel`) from a series-level run, run from src/.

    torchrun --standalone --nproc_per_node=4 main_cross.py --init ../logs/crops_run/2026-10-10/<oa run>

One sample = one series with its crops for every finding of the group (for `oa`: the
medial, lateral and patellofemoral compartments); a transformer across them lets each
compartment's prediction depend on the others (which one is worse). Targets per crop as
in series-level training; a study's probability is the mean over its series.
"""

import argparse
import json
import os
import time
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from crops.study import CrossCompartmentModel, SeriesSetDataset, collate_series  # noqa: E402
from crops.train import GROUPS, build_model, gpu_augment, load_crops, study_auc  # noqa: E402
from knee_cls.data import load_tables, make_splits  # noqa: E402
from main_crops import training_targets  # noqa: E402
from params import LOG_FOLDER  # noqa: E402
from training.distributed import (all_gather_lists, all_reduce_mean, broadcast_object,  # noqa: E402
                                  cleanup_distributed, init_distributed, seed_everything, unwrap_model,
                                  worker_init_fn)
from utils.logger import create_logger, prepare_log_folder  # noqa: E402

DEFAULTS = {
    "init": "", "layers": 2, "epochs": 4, "batch_size": 16, "lr": 5e-4, "lr_series": 1e-4, "lr_backbone": 2e-5,
    "weight_decay": 0.05, "labels": "", "seed": 0, "fullfit": False, "num_workers": 8,
    "log_path": LOG_FOLDER + "cross_run/",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()
    a.series = json.load(open(os.path.join(a.init, "args.json")))
    a.labels = a.labels or a.series["labels"]
    return a


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    for b in loader:
        x = b["x"].to(device, non_blocking=True).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(x, b["rid"].to(device), b["plane"].to(device), b["group"].to(device), b["n_groups"])
        p = torch.sigmoid(logit.float()).cpu().tolist()
        rows += list(zip(b["series"], b["study"], b["cls"], p))
    model.train()
    return pd.DataFrame(all_gather_lists(rows), columns=["SeriesInstanceUID", "StudyInstanceUID", "cls", "p"]
                        ).drop_duplicates(["SeriesInstanceUID", "cls"])


def main():
    args = parse_args()
    ctx = init_distributed()
    seed_everything(args.seed, ctx.rank)
    log = print if ctx.is_main else (lambda *a, **k: None)
    out = None
    if ctx.is_main:
        os.makedirs(args.log_path, exist_ok=True)
        out = prepare_log_folder(args.log_path)
    args.output_dir = broadcast_object(out, device=ctx.device)
    if ctx.is_main:
        create_logger(directory=args.output_dir, name="logs.txt")
        json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=4, default=str)
    sa = args.series
    classes = GROUPS[sa["group"]]
    crops, info = load_crops(sa["crops"])
    info = info[info.Anatomical_Plane.isin(sa["planes"].split(","))]
    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    val_mask = info.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids))
    targets = training_targets(args.labels, studies, classes)
    if args.fullfit:
        st_ = studies.set_index("StudyInstanceUID")
        for c in classes:
            targets[c] = targets[c].reindex(st_.index).where(st_[f"gt_{c}"].isna(), st_[f"gt_{c}"])
    train_ds = SeriesSetDataset(crops, info if args.fullfit else info[~val_mask], classes, targets)
    val_ds = SeriesSetDataset(crops, info[val_mask], classes, training_targets("claude", studies, classes))
    log(f"-> {args.output_dir}  {sa['group']} {classes}  cross-compartment from {args.init}\n"
        f"- train {len(train_ds)} series  val {len(val_ds)} series" + ("  (FULLFIT)" if args.fullfit else ""))

    def dl(ds, train):
        s = DistributedSampler(ds, shuffle=train, drop_last=train) if ctx.distributed else None
        return DataLoader(ds, batch_size=args.batch_size, sampler=s, shuffle=train and s is None, drop_last=train,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                          collate_fn=collate_series, worker_init_fn=worker_init_fn), s

    train_dl, sampler = dl(train_ds, True)
    val_dl, _ = dl(val_ds, False)
    series = build_model(sa, len(classes), pretrained=False)
    series.load_state_dict(torch.load(os.path.join(args.init, "ckpt_final.pt"), map_location="cpu"))
    model = CrossCompartmentModel(series, args.layers).to(ctx.device)
    if ctx.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ctx.device.index],
                                                          find_unused_parameters=True)
    m = unwrap_model(model)
    enc = {id(p) for p in m.series.encoder.parameters()}
    ser = {id(p) for p in m.series.parameters()} - enc
    groups = [{"params": [p for p in m.parameters() if id(p) in enc], "lr": args.lr_backbone},
              {"params": [p for p in m.parameters() if id(p) in ser], "lr": args.lr_series},
              {"params": [p for p in m.parameters() if id(p) not in enc | ser], "lr": args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + np.cos(np.pi * (it - warm) / max(1, total - warm))))
    aug = sa.get("augmentations")
    t0, hist = time.time(), []
    try:
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            losses = []
            for b in train_dl:
                x = gpu_augment(b["x"].to(ctx.device, non_blocking=True).float() / 255, aug)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(x, b["rid"].to(ctx.device), b["plane"].to(ctx.device), b["group"].to(ctx.device),
                                  b["n_groups"])
                loss = F.binary_cross_entropy_with_logits(logit.float(), b["y"].to(ctx.device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()
                losses.append(float(loss.detach()))
            pred = predict(m, val_dl, ctx.device)
            tab, study_tab = study_auc(pred, studies)
            loss = all_reduce_mean({"l": float(np.mean(losses))}, ctx.device)["l"]
            log(f"epoch {epoch + 1}  loss {loss:.4f}  gt {tab['gt'].mean():.4f}  hold {tab['hold'].mean():.4f}  "
                + "  ".join(f"{c[:10]} {r['gt']:.3f}/{r['hold']:.3f}" for c, r in tab.iterrows())
                + f"  ({time.time() - t0:.0f}s)")
            hist.append(dict(epoch=epoch + 1, gt=tab["gt"].mean(), hold=tab["hold"].mean()))
            if ctx.is_main:
                pred.to_csv(os.path.join(args.output_dir, "val_series_preds.csv"), index=False)
                study_tab.to_csv(os.path.join(args.output_dir, "val_study_preds.csv"))
                tab.to_csv(os.path.join(args.output_dir, "val_table.csv"))
                pd.DataFrame(hist).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)
        if ctx.is_main:
            torch.save(m.state_dict(), os.path.join(args.output_dir, "ckpt_final.pt"))
            print(f"FINAL gt {tab['gt'].mean():.4f} hold {tab['hold'].mean():.4f}\n{tab.round(4).to_string()}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
