"""Data, losses and fit loop for the multitask 3D model (`mt3d.model.MT3D`).

One sample = one sagittal / coronal series: its joint crop (`mt3d.prepare`), its 8
ROI boxes (segmentation targets) and the study's soft labels for the 8 ROI
findings. Study scores are the mean over its series, as for the crop models,
scored by `crops.train.study_auc` (58 radiologist studies `gt`, holdout `hold`).
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from crops.train import study_auc
from mt3d.model import PLANES, box_masks
from mt3d.prepare import MT_DIR
from roi.data import gpu_augment
from roi.extract import ROI_CLASSES
from training.distributed import all_gather_lists, all_reduce_mean, unwrap_model


def load_crops(name):
    arr = np.load(os.path.join(MT_DIR, f"{name}.npy"), mmap_mode="r")
    boxes = np.load(os.path.join(MT_DIR, f"{name}_boxes.npy"))
    info = pd.read_csv(os.path.join(MT_DIR, f"{name}.csv"))
    return arr, boxes, info


def load_anatomy(name):
    """(N, D, 128, 128) uint8 label maps and (N, 9) labelled structures (`mt3d.anatomy`)."""
    return (np.load(os.path.join(MT_DIR, f"{name}_anat.npy"), mmap_mode="r"),
            np.load(os.path.join(MT_DIR, f"{name}_anat_lab.npy")))


def concat_sources(sources):
    """[(name, arr, boxes, info), ...] -> ({name: arr}, {name: boxes}, info with a `src` column): one training
    set over several crop sets (ours + external data, e.g. MRNet)."""
    return ({n: a for n, a, _, _ in sources}, {n: b for n, _, b, _ in sources},
            pd.concat([i.assign(src=n) for n, _, _, i in sources], ignore_index=True))


class MTDataset(Dataset):
    def __init__(self, arr, boxes, info, targets, classes=ROI_CLASSES, anatomy=None, aux=None, rep=None):
        """aux: optional DataFrame (StudyInstanceUID index, one column per auxiliary target, NaN = unknown).
        arr / boxes may be dicts keyed by `info.src` (`concat_sources`). Optional info columns: `w` (loss weight
        of the study, default 1) and `y_men` (any meniscal tear, side unknown: the MIL target, default NaN).
        Anatomy targets exist for our crop set only (src "main")."""
        self.arr, self.boxes, self.anatomy, self.classes = arr, boxes, anatomy, list(classes)
        self.info = info.reset_index(drop=True)
        self.y_aux = None if aux is None else \
            aux.reindex(self.info.StudyInstanceUID).to_numpy(np.float32)
        # rep: optional DataFrame (StudyInstanceUID index) of report embeddings (`mt3d.reports`), NaN = none.
        self.y_rep = None if rep is None else rep.reindex(self.info.StudyInstanceUID).to_numpy(np.float32)
        y = np.stack([self.info.StudyInstanceUID.map(targets[c]).to_numpy(float) for c in classes], 1)
        self.y = y.astype(np.float32)                       # NaN: no label (masked out)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, i):
        r = self.info.iloc[i]
        src = r.get("src", "main")
        arr = self.arr[src] if isinstance(self.arr, dict) else self.arr
        boxes = self.boxes[src] if isinstance(self.boxes, dict) else self.boxes
        w = float(r.get("w", 1.0))
        out = {"w": torch.tensor(1.0 if np.isnan(w) else w), "y_men": torch.tensor(float(r.get("y_men", np.nan)))}
        if self.y_aux is not None:
            out["y_aux"] = torch.from_numpy(self.y_aux[i])
        if self.y_rep is not None:
            out["y_rep"] = torch.from_numpy(self.y_rep[i])
        if self.anatomy is not None:
            maps, lab = self.anatomy
            ok = src == "main" and lab[r.row].any()
            out["anat_lab"] = torch.from_numpy(lab[r.row] if src == "main" else np.zeros(lab.shape[1], bool))
            # Series without anatomy: an empty map, nothing labelled (the map is never read then).
            out["anat"] = torch.from_numpy(np.array(maps[r.row]) if ok else np.zeros(maps.shape[1:], np.uint8))
        return out | {"x": torch.from_numpy(np.array(arr[r.row])), "boxes": torch.from_numpy(boxes[r.row]),
                "plane": torch.tensor(PLANES.index(r.Anatomical_Plane)),
                "meta": torch.tensor([r.Fluid_Sensitive, r.Fat_Suppression], dtype=torch.float32),
                "y": torch.from_numpy(self.y[i]), "idx": torch.tensor(i)}


class PairDataset(Dataset):
    """Study samples for `mt3d.hybrid.PairHybrid`: `slots` series of one study (`MTDataset` items stacked
    on a slot axis).
      slots = 2  one sagittal and one coronal series, random ones in training; in validation every series
                 once, each paired cyclically with the other plane's series. A study with one plane only
                 repeats that series in the second slot, marked invalid;
      slots > 2  a set: in training a random subset of the study's series (sagittal and coronal alternated,
                 so both planes are in when they exist); in validation all of them, in chunks of `slots`.
                 Missing slots repeat a series, marked invalid."""

    def __init__(self, base, train, slots=2):
        self.base, self.train, self.slots = base, train, slots
        info = base.info.assign(i=np.arange(len(base.info)))
        self.studies = []
        for uid, g in info.groupby("StudyInstanceUID", sort=False):
            self.studies.append((uid, g[g.Anatomical_Plane == "Sagittal"].i.tolist(),
                                 g[g.Anatomical_Plane == "Coronal"].i.tolist()))
        self.pairs = []                                   # validation groups: (study, [series or None] * slots)
        for k, (_, sag, cor) in enumerate(self.studies):
            if slots == 2:
                for j in range(max(len(sag), len(cor))):
                    self.pairs.append((k, [sag[j % len(sag)] if sag else None, cor[j % len(cor)] if cor else None]))
            else:
                order = self._interleave(sag, cor)
                for j in range(0, len(order), slots):
                    chunk = order[j:j + slots]
                    self.pairs.append((k, chunk + [None] * (slots - len(chunk))))
        self.info = base.info
        self.classes = base.classes

    @staticmethod
    def _interleave(a, b):
        out = []
        for j in range(max(len(a), len(b))):
            out += ([a[j]] if j < len(a) else []) + ([b[j]] if j < len(b) else [])
        return out

    def __len__(self):
        return len(self.studies) if self.train else len(self.pairs)

    def __getitem__(self, i):
        if self.train:
            uid, sag, cor = self.studies[i]
            k = i
            if self.slots == 2:
                group = [sag[np.random.randint(len(sag))] if sag else None, cor[np.random.randint(len(cor))] if cor else None]
            else:
                order = self._interleave(list(np.random.permutation(sag)), list(np.random.permutation(cor)))
                group = order[:self.slots] + [None] * max(0, self.slots - len(order))
        else:
            k, group = self.pairs[i]
        valid = torch.tensor([g is not None for g in group])
        first = next(g for g in group if g is not None)
        items = [self.base[g if g is not None else first] for g in group]
        study_level = ("y", "idx", "y_aux", "y_rep", "w", "y_men")
        out = {key: torch.stack([s[key] for s in items]) for key in items[0] if key not in study_level}
        out |= {key: items[0][key] for key in ("y_aux", "y_rep", "w", "y_men") if key in items[0]}
        return out | {"valid": valid, "y": items[0]["y"], "idx": torch.tensor(i), "study": torch.tensor(k)}


def flatten_pairs(b):
    """A PairDataset batch -> an MTDataset-like batch of B * 2 series (slot-major per study)."""
    return {k: (v.flatten(0, 1) if k not in ("valid", "y", "idx", "study", "y_aux", "y_rep", "w", "y_men") else v) for k, v in b.items()}


def seg_loss(logits, target):
    """BCE + soft Dice, target (B, K, D, S, S) downsampled to the logits' grid (soft, area)."""
    t = F.adaptive_avg_pool3d(target, logits.shape[2:])
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, t)
    p = torch.sigmoid(logits)
    inter = (p * t).sum((2, 3, 4))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3, 4)) + t.sum((2, 3, 4)) + 1)
    return bce + dice.mean(), (1 - dice.detach()).mean(0)


def masked_seg_loss(logits, target, labelled):
    """`seg_loss` over the labelled (sample, structure) pairs only."""
    m = labelled.float()
    if m.sum() == 0:
        return logits.sum() * 0
    t = F.adaptive_avg_pool3d(target, logits.shape[2:])
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none").mean((2, 3, 4))
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * t).sum((2, 3, 4)) + 1) / (p.sum((2, 3, 4)) + t.sum((2, 3, 4)) + 1)
    return ((bce + dice) * m).sum() / m.sum()


def prepare_batch(b, device, size, aug=None):
    """-> image (B, 1, D, S, S), ROI box masks (B, 8, D, S, S), anatomy masks (B, 9, D, S, S) or None."""
    x = b["x"].to(device, non_blocking=True).float()[:, None] / 255          # B 1 D S S
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(x.shape[2], size, size), mode="trilinear", align_corners=False)
    target = box_masks(b["boxes"].to(device), x.shape[2:])
    k = target.shape[1]
    if "anat" in b:
        a = b["anat"].to(device, non_blocking=True).long()
        n = b["anat_lab"].shape[1]
        a = F.one_hot(a, n + 1)[..., 1:].permute(0, 4, 1, 2, 3).float()      # B 9 D 128 128
        target = torch.cat([target, F.interpolate(a, size=x.shape[2:], mode="nearest")], 1)
    if aug is not None:
        x, target = gpu_augment(x, target, b["plane"] == 0, aug)
    return x, target[:, :k], (target[:, k:] if target.shape[1] > k else None)


@torch.no_grad()
def predict(model, loader, ds, device, size):
    model.eval()
    rows, dice = [], []
    pair = isinstance(ds, PairDataset)
    for b in loader:
        if pair:
            b = flatten_pairs(b)
        x, target, _ = prepare_batch(b, device, size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, b["plane"].to(device), target, *([b["valid"].to(device)] if pair else []))
        p = torch.sigmoid(out["cls"].float()).cpu().numpy()
        dice.append(seg_loss(out["seg"][-1], target)[1].cpu().numpy())
        rows += [(i, p[j].tolist()) for j, i in enumerate(b["idx"].tolist())]
    model.train()
    rows = dict(all_gather_lists(rows))
    idx = sorted(rows)
    p = np.array([rows[i] for i in idx])
    if pair:                                   # one row per (sagittal, coronal) pair
        it = ds.info.iloc[[next(g for g in ds.pairs[i][1] if g is not None) for i in idx]]
        wide = pd.DataFrame(p, columns=ds.classes).assign(StudyInstanceUID=it.StudyInstanceUID.values,
                                                           SeriesInstanceUID=it.SeriesInstanceUID.values,
                                                           plane="pair")
    else:
        it = ds.info.iloc[idx]
        wide = pd.DataFrame(p, columns=ds.classes).assign(StudyInstanceUID=it.StudyInstanceUID.values,
                                                           SeriesInstanceUID=it.SeriesInstanceUID.values,
                                                           plane=it.Anatomical_Plane.values)
    long = wide.melt(id_vars=["StudyInstanceUID", "SeriesInstanceUID", "plane"], var_name="cls", value_name="p")
    dice = all_reduce_mean({c: float(v) for c, v in zip(ROI_CLASSES, np.mean(dice, 0))}, device)
    return long, wide, dice


def mixup(x, target, anat, y, b, alpha, pair):
    """Mixup between samples (studies in pair / set mode: series slot by slot, so planes mostly match):
    images, ROI box targets and soft labels mixed with one weight per batch; the anatomy targets too, a
    structure staying labelled only when it is in both; the series flags (validity, meta) mixed or joined."""
    lam = float(np.random.beta(alpha, alpha))
    n = y.shape[0]
    perm = torch.randperm(n, device=x.device)
    m = x.shape[0] // n                                  # series per sample (slots)
    sp = (perm[:, None] * m + torch.arange(m, device=x.device)).flatten()
    x = lam * x + (1 - lam) * x[sp]
    target = lam * target + (1 - lam) * target[sp]
    y = torch.where(torch.isnan(y) | torch.isnan(y[perm]), y, lam * y + (1 - lam) * y[perm])
    b = dict(b)
    b["meta"] = lam * b["meta"] + (1 - lam) * b["meta"][sp.cpu()]
    if anat is not None:
        anat = lam * anat + (1 - lam) * anat[sp]
        b["anat_lab"] = b["anat_lab"] & b["anat_lab"][sp.cpu()]
    if pair:
        b["valid"] = b["valid"] | b["valid"][perm.cpu()]
    return x, target, anat, y, b


def self_classes_index(args):
    return ROI_CLASSES.index("Medial Meniscus"), ROI_CLASSES.index("Lateral Meniscus")


def fit(model, train_dl, val_dl, val_ds, studies, args, ctx, sampler):
    device = ctx.device
    log = print if ctx.is_main else (lambda *a, **k: None)
    m = unwrap_model(model)
    unet = [p for n, p in m.named_parameters() if n.startswith(("encoder", "transpconvs", "stages"))]
    ids = {id(p) for p in unet}
    opt = torch.optim.AdamW([{"params": unet, "lr": args.lr_unet},
                             {"params": [p for p in m.parameters() if id(p) not in ids], "lr": args.lr}],
                            weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + np.cos(np.pi * (it - warm) / max(1, total - warm))))
    ema = copy.deepcopy(m).eval() if args.ema else None
    t0, hist, it = time.time(), [], 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        run = {"cls": [], "seg": [], "anat": [], "meta": []}
        for b in train_dl:
            pair = "valid" in b
            if pair:
                b = flatten_pairs(b)
            x, target, anat = prepare_batch(b, device, args.size, args.augmentations)
            y = b["y"].to(device)
            if getattr(args, "mixup", 0) > 0:
                x, target, anat, y, b = mixup(x, target, anat, y, b, args.mixup, pair)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, b["plane"].to(device), target, *([b["valid"].to(device)] if pair else []))
            lab = ~torch.isnan(y)
            # Per-study loss weights (external data can be down- or up-weighted).
            wl = b["w"].to(device).float()[:, None].expand_as(y)[lab]
            l_cls = (F.binary_cross_entropy_with_logits(out["cls"].float()[lab], y[lab], reduction="none") * wl).sum() \
                / wl.sum().clamp(min=1e-6)
            l_seg = sum(seg_loss(s, target)[0] for s in out["seg"]) / len(out["seg"])
            l_meta = F.binary_cross_entropy_with_logits(out["meta"].float(), b["meta"].to(device))
            l_anat = sum(masked_seg_loss(s, anat, b["anat_lab"].to(device)) for s in out["anat"]) / len(out["anat"]) \
                if out["anat"] else torch.zeros((), device=device)
            loss = args.w_cls * l_cls + args.w_seg * l_seg + args.w_meta * l_meta + args.w_anat * l_anat
            # MIL (MRNet): "a meniscal tear, side unknown" supervises max(medial, lateral meniscus logits).
            ym = b["y_men"].to(device).float()
            if getattr(args, "w_mil", 0) > 0 and (~torch.isnan(ym)).any():
                k = ~torch.isnan(ym)
                mm, lm = self_classes_index(args)
                any_men = torch.maximum(out["cls"].float()[:, mm], out["cls"].float()[:, lm])
                loss = loss + args.w_mil * F.binary_cross_entropy_with_logits(any_men[k], ym[k])
            if "y_aux" in b and "aux" in out:
                ya = b["y_aux"].to(device)
                la = ~torch.isnan(ya)
                l_aux = F.binary_cross_entropy_with_logits(out["aux"].float()[la], ya[la]) if la.any() \
                    else out["aux"].sum() * 0
                loss = loss + args.w_aux * l_aux
            if "y_rep" in b and "rep" in out:
                yr = b["y_rep"].to(device)
                ok = ~torch.isnan(yr).any(1)
                l_rep = (1 - F.cosine_similarity(out["rep"].float()[ok], yr[ok], dim=1)).mean() if ok.any() \
                    else out["rep"].sum() * 0
                loss = loss + args.w_rep * l_rep
                run.setdefault("rep", []).append(float(l_rep.detach()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step()
            sched.step()
            it += 1
            if ema is not None:
                d = min(args.ema, (1 + it) / (10 + it))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), m.parameters()):
                        pe.lerp_(pm, 1 - d)
                    # Buffers (BatchNorm running statistics, e.g. CoAtNet) follow the live model.
                    for be, bm in zip(ema.buffers(), m.buffers()):
                        be.copy_(bm)
            run["cls"].append(float(l_cls.detach()))
            run["seg"].append(float(l_seg.detach()))
            run["meta"].append(float(l_meta.detach()))
            run["anat"].append(float(l_anat.detach()))
            if it % args.log_every == 0:
                r = all_reduce_mean({k: float(np.mean(v[-args.log_every:])) for k, v in run.items()}, device)
                log(f"  [{it}/{total}] ep {epoch}  cls {r['cls']:.4f}  seg {r['seg']:.4f}  anat {r['anat']:.4f}  rep {r.get('rep', 0):.4f}  meta {r['meta']:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.1e}  {time.time() - t0:.0f}s")
        long, wide, dice = predict(ema if ema is not None else m, val_dl, val_ds, device, args.size)
        tab, study_tab = study_auc(long, studies)
        log(f"epoch {epoch + 1}  gt {tab['gt'].mean():.4f}  hold {tab['hold'].mean():.4f}  "
            + "  ".join(f"{c[:10]} {r['gt']:.3f}/{r['hold']:.3f}" for c, r in tab.iterrows())
            + f"  | dice {np.mean(list(dice.values())):.3f}  ({time.time() - t0:.0f}s)")
        hist.append(dict(epoch=epoch + 1, gt=tab["gt"].mean(), hold=tab["hold"].mean(),
                         dice=np.mean(list(dice.values()))))
        if ctx.is_main:
            wide.to_csv(os.path.join(args.output_dir, "val_series_preds.csv"), index=False)
            study_tab.to_csv(os.path.join(args.output_dir, "val_study_preds.csv"))
            tab.to_csv(os.path.join(args.output_dir, "val_table.csv"))
            pd.DataFrame(hist).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)
    if ctx.is_main:
        torch.save((ema if ema is not None else m).state_dict(), os.path.join(args.output_dir, "ckpt_final.pt"))
    return tab
