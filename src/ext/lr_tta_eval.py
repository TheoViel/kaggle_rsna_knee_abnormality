"""Left / right mirror test-time augmentation on validation runs (single GPU); run from src/.

    python -m ext.lr_tta_eval --runs 2026-10-09/75,2026-10-11/33 [--out ../output/locations/lr_tta_eval.csv]

Every crop is a canonical right knee (left knees are mirrored by the side classifier: crops.prepare.crop_box) and
training never flips. The mirror pass applies that same flip once more: reversed slice order for sagittal crops,
flipped columns for coronal / axial crops, i.e. the knee as it would look with the opposite side call. Per run,
on its validation crops (the 58 + the holdout, ext.tta_eval.val_dataset): plain, mirror, and the mean of the two
probabilities, aggregated per study as the pipeline does. Prints AUC on the 58 / holdout per finding.
"""
import argparse
import json
from types import SimpleNamespace

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from crops.train import build_model, center_view, slice_window
from ext.tta_eval import R, study_scores, val_dataset
from knee_cls.data import load_tables, make_splits
from main_crops import DEFAULTS


def mirror(x, plane):
    """(B, D, S, S) crops, plane index (0 sagittal, 1 coronal, 2 axial) -> the left-right mirror of each crop."""
    out = x.flip(3)                                   # coronal / axial: columns
    sag = plane == 0
    out[sag] = x[sag].flip(1)                         # sagittal: slice order
    return out


@torch.no_grad()
def run_passes(model, ds, a, device, bs=128):
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8)
    plain, mir = [], []
    for b in loader:
        x = b["x"].to(device).float() / 255
        x = slice_window(x, getattr(a, "depth_out", 0), 0)
        roi, plane = b["roi"].to(device), b["plane"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            plain.append(torch.sigmoid(model(center_view(x, a.ctx or 1.0), roi, plane).float()).cpu())
            mir.append(torch.sigmoid(model(center_view(mirror(x, plane), a.ctx or 1.0), roi, plane).float()).cpu())
    return torch.cat(plain).numpy(), torch.cat(mir).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="../output/locations/lr_tta_eval.csv")
    args = ap.parse_args()
    device = torch.device("cuda")
    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    S = studies.set_index("StudyInstanceUID")
    rows = []
    for run in args.runs.split(","):
        a = SimpleNamespace(**{**DEFAULTS, **json.load(open(R + run + "/args.json"))})
        ds, classes = val_dataset(a, studies)
        model = build_model(vars(a), len(classes), pretrained=False)
        model.load_state_dict(torch.load(R + run + "/ckpt_final.pt", map_location="cpu"))
        model = model.to(device).eval()
        plain, mir = run_passes(model, ds, a, device)
        sp, sm, sb = study_scores(ds.items, plain), study_scores(ds.items, mir), study_scores(ds.items, (plain + mir) / 2)
        sp.rename_axis("StudyInstanceUID").to_csv(R + run + "/val_study_preds_plain.csv")      # recipe-level scoring
        sb.rename_axis("StudyInstanceUID").to_csv(R + run + "/val_study_preds_lrtta.csv")
        for F in sp.columns:
            if f"gt_{F}" not in S:
                continue
            G = S.loc[gt_ids, f"gt_{F}"].dropna().round()
            H = (S.loc[hold_ids, f"target_{F}"].dropna() >= 0.5).astype(int)
            r = dict(run=run, group=a.group, finding=F)
            for k, p in (("plain", sp[F]), ("mirror", sm[F]), ("plain+mirror", sb[F])):
                r[k + "_58"] = roc_auc_score(G, p.reindex(G.index))
                r[k + "_hold"] = roc_auc_score(H, p.reindex(H.index))
            rows.append(r)
            print(f"{run} {F:18s} " + "  ".join(f"{k} {r[k + '_58']:.3f} / {r[k + '_hold']:.3f}"
                                                 for k in ("plain", "mirror", "plain+mirror")), flush=True)
        del model
        torch.cuda.empty_cache()
    d = pd.DataFrame(rows)
    d.to_csv(args.out, index=False)
    for k in ("plain", "mirror", "plain+mirror"):
        print(f"mean over {len(d)} (run, finding): {k:13s} 58 {d[k + '_58'].mean():.4f} / holdout {d[k + '_hold'].mean():.4f}")


if __name__ == "__main__":
    main()
