"""Train the per-series knee-side classifier, run from src/.

    torchrun --standalone --nproc_per_node=4 main_side.py

See `side_cls.py`. Validation: the series of the validation studies (radiologist-
labelled + classifier holdout) whose own DICOM tag is R / L -- the one label not
inferred -- scored per series and per study (vote over its series).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from params import LOG_FOLDER
from roi.data import split_series
from side_cls import SideDataset, SideNet, augment, load_all_volumes, model_input
from training.distributed import (all_gather_lists, broadcast_object, cleanup_distributed,
                                  init_distributed, seed_everything, unwrap_model, worker_init_fn)
from utils.logger import create_logger, prepare_log_folder

DEFAULTS = {"epochs": 8, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4, "width": 16,
            "num_workers": 6, "seed": 0, "log_path": LOG_FOLDER + "side_run/"}


@torch.no_grad()
def evaluate(model, loader, ds, device):
    model.eval()
    rows = []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(model_input(b["image"].to(device), b["plane"].to(device))).float())
        rows += [(int(i), float(v)) for i, v in zip(b["idx"], p.cpu())]
    model.train()
    rows = dict(all_gather_lists(rows))
    idx = ds.index.iloc[sorted(rows)].copy()
    idx["p_right"] = [rows[i] for i in sorted(rows)]
    idx["y"] = ds.y[sorted(rows)]
    idx["tagged"] = ds.tagged[sorted(rows)]
    t = idx[idx.tagged]
    series_acc = float(((t.p_right > 0.5) == (t.y == 1)).mean())
    study = t.groupby("StudyInstanceUID").agg(p=("p_right", "mean"), y=("y", "mean"))
    study = study[study.y.isin([0, 1])]
    return {"series_acc": series_acc, "study_acc": float(((study.p > 0.5) == (study.y == 1)).mean()),
            "n_series": len(t), "n_studies": len(study)}, idx


def main():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    ctx = init_distributed()
    seed_everything(args.seed, ctx.rank)
    out = None
    if ctx.is_main:
        os.makedirs(args.log_path, exist_ok=True)
        out = prepare_log_folder(args.log_path)
    args.output_dir = broadcast_object(out, device=ctx.device)
    if ctx.is_main:
        create_logger(directory=args.output_dir, name="logs.txt")
        json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=4)

    vols, index = load_all_volumes()
    tr, va = split_series(index)
    train_ds, val_ds = SideDataset(vols, tr), SideDataset(vols, va)
    keep = ~np.isnan(train_ds.y)
    train_ds.index, train_ds.y, train_ds.tagged = (train_ds.index[keep].reset_index(drop=True),
                                                   train_ds.y[keep], train_ds.tagged[keep])
    if ctx.is_main:
        print(f"-> {args.output_dir}\n- train {len(train_ds)} series ({int(train_ds.tagged.sum())} tagged)"
              f"\n- val   {len(val_ds)} series ({int(val_ds.tagged.sum())} tagged)")

    def dl(ds, train):
        s = DistributedSampler(ds, shuffle=train, drop_last=train) if ctx.distributed else None
        return DataLoader(ds, batch_size=args.batch_size, sampler=s, shuffle=train and s is None,
                          num_workers=args.num_workers, pin_memory=True, drop_last=train,
                          persistent_workers=True, worker_init_fn=worker_init_fn), s

    train_dl, sampler = dl(train_ds, True)
    val_dl, _ = dl(val_ds, False)
    model = SideNet(width=args.width).to(ctx.device)
    if ctx.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ctx.device.index])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    t0 = time.time()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        losses = []
        for b in train_dl:
            image, y = augment(b["image"].to(ctx.device), b["y"].to(ctx.device), b["plane"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(model_input(image, b["plane"].to(ctx.device)))
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        m, pred = evaluate(unwrap_model(model), val_dl, val_ds, ctx.device)
        if ctx.is_main:
            print(f"epoch {epoch + 1}  loss {np.mean(losses):.4f}  val series acc {m['series_acc']:.4f} "
                  f"({m['n_series']})  study acc {m['study_acc']:.4f} ({m['n_studies']})  {time.time() - t0:.0f}s")
    if ctx.is_main:
        torch.save(unwrap_model(model).state_dict(), os.path.join(args.output_dir, "ckpt_final.pt"))
        pred.to_csv(os.path.join(args.output_dir, "val_pred.csv"), index=False)
    cleanup_distributed()


if __name__ == "__main__":
    main()
