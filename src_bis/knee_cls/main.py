"""Loaders, model and the train/validate run."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from knee_cls.data import KneeStudyDataset, collate_studies, load_tables, make_splits
from knee_cls.model import KneeAttnNet
from knee_cls.train import fit
from params import NUM_CLASSES
from training.distributed import worker_init_fn


def build_loader(args, studies, series, ids, train, ctx=None):
    ds = KneeStudyDataset(
        studies, series, ids, train=train,
        max_series=int(args.max_series if train else args.val_max_series),
        n_frames=int(args.n_frames if train else args.val_n_frames),
        frame_offset=int(args.frame_offset), z_trim=float(args.z_trim))
    sampler = None
    if ctx is not None and ctx.distributed:
        sampler = DistributedSampler(ds, num_replicas=ctx.world_size, rank=ctx.rank,
                                     shuffle=train, seed=int(args.seed), drop_last=train)
    loader = DataLoader(
        ds, batch_size=int(args.batch_size if train else args.val_batch_size),
        shuffle=train and sampler is None, sampler=sampler, drop_last=train,
        num_workers=int(args.num_workers), pin_memory=True, collate_fn=collate_studies,
        persistent_workers=int(args.num_workers) > 0, worker_init_fn=worker_init_fn,
        prefetch_factor=4 if int(args.num_workers) > 0 else None)
    return loader, sampler, ds


def build_model(args, ctx, pretrained=None):
    model = KneeAttnNet(
        backbone=args.backbone,
        pretrained=bool(args.pretrained if pretrained is None else pretrained),
        num_classes=NUM_CLASSES, dim=int(args.dim), z_layers=int(args.z_layers),
        z_heads=int(args.z_heads), dropout=float(args.dropout),
        drop_path=float(args.drop_path), attn_stride=int(args.attn_stride),
        in_chans=3, xy_heads=int(getattr(args, "xy_heads", 1)),
        grad_checkpointing=bool(args.grad_checkpointing),
        verbose=int(ctx.is_main) if ctx is not None else 1)
    device = ctx.device if ctx is not None else torch.device("cuda")
    model = model.to(device)
    if ctx is not None and ctx.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=False)
    return model


def run(args, ctx):
    studies, series = load_tables()
    train_ids, gt_ids, hold_ids = make_splits(
        studies, holdout_frac=float(args.holdout_frac), seed=int(args.seed))
    if ctx.is_main:
        print(f"-> train {len(train_ids)}  gt-val {len(gt_ids)}  holdout {len(hold_ids)}")

    train_loader, sampler, train_ds = build_loader(args, studies, series, train_ids,
                                                   train=True, ctx=ctx)
    val_loader, _, val_ds = build_loader(args, studies, series, gt_ids + hold_ids,
                                         train=False, ctx=ctx)
    if ctx.is_main:
        print(f"- train : {train_ds.summary}")
        print(f"- val   : {val_ds.summary}\n")

    model = build_model(args, ctx)
    metrics = fit(model, train_loader, args, ctx, train_sampler=sampler,
                  val_loader=val_loader)
    del model
    torch.cuda.empty_cache()
    if ctx.is_main and metrics:
        print(f"\nFINAL  " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()
                                      if isinstance(v, float)))
    return metrics
