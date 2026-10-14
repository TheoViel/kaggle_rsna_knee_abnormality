"""OA crop model trained on series triplets (medial, lateral, PF crops of the same series) with a pairwise
compartment-ranking loss, single GPU; run from src/.

    PYTHONPATH=. python ../src_bis/main_oa_pair.py --pair_weight 1.0 [--crops f_s1n8] [--img_size 0] \
        [--init ../logs/crops_run/2026-10-11/8] [--labels pl:pl_round2] [--seed 0]

Why: our OA models score "how arthritic is the knee" more than "which compartment" (medial-lateral prediction
correlation 0.84 vs 0.32 in the radiologist labels). Each training item holds the three OA crops of one series, so the
loss can also ask which compartment is worse:
    loss = sum_c BCE(z_c, y_c) / 3 + pair_weight x sum_{a<b} |y_a - y_b| BCE(z_a - z_b, (y_a - y_b + 1) / 2) / sum |y_a - y_b|
(NaN targets masked). pair_weight 0 is the plain OA model (same data, model and schedule), the A/B reference.
Model: crops.train.Crop25D (`oa` group, finding embeddings), as main_crops.py builds it; augmentation, optimizer and
schedule as crops.train.fit. Writes logs/bis/crops_run/<date>/<n>/ (args.json, val_series_preds.csv,
val_study_preds.csv, ckpt_final.pt) in the main_crops layout, so src_bis/oa_eval.py reads it as "bis:<date>/<n>".
"""

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))   # src/ before src_bis/

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from crops.train import GROUPS, PLANES, build_model, crop_index, load_crops, model_input
from knee_cls.data import load_tables, make_splits
from params import OUTPUT_FOLDER

OA = GROUPS["oa"]                      # ["Medial OA", "Lateral OA", "PF OA"]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def targets(which, studies):
    if which.startswith("pl:"):
        name = which[3:]
        path = name if name.endswith(".csv") else os.path.join(OUTPUT_FOLDER, "crops", f"{name}.csv")
        return pd.read_csv(path).set_index("StudyInstanceUID")[OA]
    if which == "claude":
        return studies.set_index("StudyInstanceUID")[[f"target_{c}" for c in OA]].set_axis(OA, axis=1)
    raise ValueError(which)


class Triplets(Dataset):
    def __init__(self, arrays, info, y):
        self.arrays, self.info = arrays, info.reset_index(drop=True)
        self.y = y.reindex(self.info.StudyInstanceUID).to_numpy(np.float32)
        self.rois = [crop_index(c) for c in OA]

    def __len__(self):
        return len(self.info)

    def __getitem__(self, i):
        r = self.info.iloc[i]
        x = np.stack([np.asarray(self.arrays[r.src][r.row, k]) for k in self.rois])       # 3 D S S uint8
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(self.y[i]),
                "plane": torch.tensor(PLANES.index(r.Anatomical_Plane)), "idx": torch.tensor(i)}


def forward(model, b, device, args, train):
    bsz = b["x"].shape[0]
    x = model_input({"x": b["x"].flatten(0, 1)}, device, args, train)                # (3B, D, S, S)
    rid = torch.arange(3, device=device).repeat(bsz)
    plane = b["plane"].to(device).repeat_interleave(3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = model(x, rid, plane).float()
    return z.view(bsz, 3)


def loss_fn(z, y, pair_weight):
    m = ~torch.isnan(y)
    yy = torch.nan_to_num(y)
    l = (F.binary_cross_entropy_with_logits(z, yy, reduction="none") * m).sum() / m.sum().clamp(min=1)
    if pair_weight > 0:
        num, den = 0.0, 0.0
        for a, c in PAIRS:
            w = (yy[:, a] - yy[:, c]).abs() * (m[:, a] & m[:, c])
            t = ((yy[:, a] - yy[:, c] + 1) / 2).clamp(0, 1)
            num = num + (F.binary_cross_entropy_with_logits(z[:, a] - z[:, c], t, reduction="none") * w).sum()
            den = den + w.sum()
        l = l + pair_weight * num / max(float(den), 1e-6)
    return l


@torch.no_grad()
def predict(model, dl, ds, device, args):
    model.eval()
    out = []
    for b in dl:
        p = torch.sigmoid(forward(model, b, device, args, False)).cpu().numpy()
        out.append((b["idx"].numpy(), p))
    model.train()
    idx = np.concatenate([o[0] for o in out])
    p = np.concatenate([o[1] for o in out])
    info = ds.info.iloc[idx]
    rows = [pd.DataFrame({"StudyInstanceUID": info.StudyInstanceUID.values, "SeriesInstanceUID": info.SeriesInstanceUID.values,
                          "cls": c, "plane": info.Anatomical_Plane.values, "p": p[:, k]}) for k, c in enumerate(OA)]
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", default="f_s1n8")
    ap.add_argument("--planes", default="Sagittal,Coronal")
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k")
    ap.add_argument("--img_size", type=int, default=0)
    ap.add_argument("--init", default="../logs/crops_run/2026-10-11/8")
    ap.add_argument("--labels", default="pl:pl_round2")
    ap.add_argument("--pair_weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32, help="series per step (x3 crops)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--fullfit", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    ref = json.load(open("../logs/crops_run/2026-10-11/33/args.json"))         # augmentation / model settings of mk89
    args = argparse.Namespace(**{**ref, **vars(a), "group": "oa", "z_sampling": None})
    day = datetime.date.today().isoformat()
    base = os.path.join("..", "logs", "bis", "crops_run", day)
    os.makedirs(base, exist_ok=True)
    n = 0
    while os.path.exists(os.path.join(base, str(n))):
        n += 1
    args.output_dir = os.path.join(base, str(n))
    os.makedirs(args.output_dir)
    json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=2, default=str)
    print(f"-> {args.output_dir} {a.name}  pair_weight {a.pair_weight}  labels {a.labels}  crops {a.crops}", flush=True)

    arrays, info = load_crops(a.crops)
    info = info[info.Anatomical_Plane.isin(a.planes.split(","))]
    studies, _ = load_tables()
    _, gt, hold = make_splits(studies, 0.1, 0)
    val = set(gt) | set(hold)
    y = targets(a.labels, studies)
    if a.fullfit:     # radiologist labels for the 58
        st = studies.set_index("StudyInstanceUID")
        for c in OA:
            g = st[f"gt_{c}"].reindex(y.index)
            y[c] = y[c].where(g.isna(), g)
    tr = info if a.fullfit else info[~info.StudyInstanceUID.isin(val)]
    va = info[info.StudyInstanceUID.isin(val)]
    tr = tr[tr.StudyInstanceUID.isin(y.index)]
    train_ds, val_ds = Triplets(arrays, tr, y), Triplets(arrays, va, y)
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=a.num_workers,
                          pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=a.batch_size * 2, num_workers=a.num_workers, pin_memory=True)
    print(f"- train {len(train_ds)} series  val {len(val_ds)} series", flush=True)

    device = torch.device("cuda")
    model = build_model(vars(args), len(OA), pretrained=True)
    if a.init:
        sd = torch.load(os.path.join(a.init, "ckpt_final.pt"), map_location="cpu")
        own = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        res = model.load_state_dict(sd, strict=False)
        print(f"init from {a.init}: missing {res.missing_keys}", flush=True)
    model = model.to(device)
    enc = list(model.encoder.parameters())
    ids = {id(p) for p in enc}
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr_backbone},
                             {"params": [p for p in model.parameters() if id(p) not in ids], "lr": a.lr}],
                            weight_decay=a.weight_decay)
    total = a.epochs * len(train_dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + np.cos(np.pi * (it - warm) / max(1, total - warm))))
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import oa_eval
    t0 = time.time()
    for epoch in range(a.epochs):
        losses = []
        for b in train_dl:
            z = forward(model, b, device, args, True)
            loss = loss_fn(z, b["y"].to(device), a.pair_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            losses.append(float(loss))
        pred = predict(model, val_dl, val_ds, device, args)
        study = pred.groupby(["StudyInstanceUID", "cls"]).p.mean().unstack()[OA]
        pred.to_csv(os.path.join(args.output_dir, "val_series_preds.csv"), index=False)
        study.to_csv(os.path.join(args.output_dir, "val_study_preds.csv"))
        s = {c: oa_eval.auc(study[c], c) for c in OA}
        ct = oa_eval.crosstalk(study["Medial OA"], study["Lateral OA"])
        print(f"epoch {epoch + 1}  loss {np.mean(losses):.4f}  " + "  ".join(f"{c[:6]} {v[0]:.3f}/{v[1]:.3f}" for c, v in s.items())
              + f"  MOA-LOA corr {ct:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "ckpt_final.pt"))
    print("FINAL", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
