"""Exactness check of the inference code for the hybrid models, run from src_bis/.

    python -m mt3d.check_pipeline --runs 2026-10-11/30,2026-10-11/32,2026-10-11/51,2026-10-11/52 [--amp bf16]

Feeds the stored validation joint crops (output/bis/mt3d/<crops>.npy, exactly what validation used)
through the pipeline's own hybrid path (`infer.pipeline.load_hybrid`, `hybrid_predict`: grouping,
batching, padding, per-study mean) and compares with each run's validation output
(val_study_preds.csv): max |difference| per study and the AUCs (58 / holdout), per run and for the
rank-averaged ensemble. With --amp bf16 (the validation dtype) the predictions must match.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import infer.pipeline as P
from crops.train import study_auc
from knee_cls.data import load_tables, make_splits
from mt3d.blend import rank_avg
from mt3d.train import load_crops
from roi.extract import ROI_CLASSES

MT = "../logs/bis/mt3d/"


def validation_fp32(h, studies, val):
    """The training-time validation path (`mt3d.train.predict` on a `PairDataset`), autocast off."""
    import contextlib
    import types

    import mt3d.train as T
    from main_mt3d import training_targets
    from torch.utils.data import DataLoader

    arr, boxes, _ = load_crops("u10_d32_s256")
    ds = T.PairDataset(T.MTDataset(arr, boxes, val, training_targets("claude", studies, ROI_CLASSES)), False, h["slots"])
    T.all_gather_lists, T.all_reduce_mean = (lambda r: r), (lambda d, dev: d)
    real = T.torch
    T.torch = types.SimpleNamespace(**{k: getattr(real, k) for k in dir(real) if not k.startswith("__")})
    T.torch.autocast = lambda *a, **k: contextlib.nullcontext()
    try:
        long, _, _ = T.predict(h["net"], DataLoader(ds, batch_size=4, num_workers=8), ds, torch.device("cuda"), h["size"])
    finally:
        T.torch = real
    return long.groupby(["StudyInstanceUID", "cls"]).p.mean().unstack()[list(ROI_CLASSES)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--ref_fp32", action="store_true",
                   help="reference = the validation code path (mt3d.train.predict) re-run in float32, not the saved csv")
    a = p.parse_args()
    P.AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[a.amp]
    dev = torch.device("cuda")
    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    ours, refs = {}, {}
    for run in a.runs.split(","):
        h = P.load_hybrid(os.path.join(MT, run), run, dev)
        arr, _, info = load_crops("u10_d32_s256")
        val = info[info.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids)) & info.Anatomical_Plane.isin(P.ROI_PLANES)]
        rows = [dict(StudyInstanceUID=r.StudyInstanceUID, plane=r.Anatomical_Plane) for r in val.itertuples()]
        crops = torch.from_numpy(np.ascontiguousarray(arr[val.row.to_numpy()])).to(dev)
        out, i = [], 0
        while i < len(rows):                                     # blocks of whole studies (rows are study-ordered)
            j = min(i + 400, len(rows))
            while j < len(rows) and rows[j]["StudyInstanceUID"] == rows[j - 1]["StudyInstanceUID"]:
                j += 1
            out += P.hybrid_predict(h, rows[i:j], crops[i:j])
            i = j
        pred = pd.DataFrame(out, columns=["model", "StudyInstanceUID", "cls", "p"])
        mine = pred.pivot(index="StudyInstanceUID", columns="cls", values="p")[list(ROI_CLASSES)]
        ref = pd.read_csv(os.path.join(MT, run, "val_study_preds.csv"), index_col=0)[list(ROI_CLASSES)] \
            if not a.ref_fp32 else validation_fp32(h, studies, val)
        mine = mine.reindex(ref.index)
        ours[run], refs[run] = mine, ref
        t_m, _ = study_auc(pred.drop(columns="model"), studies)
        print(f"{run}: max |p - validation p| {np.nanmax(np.abs(mine.values - ref.values)):.2e}, "
              f"studies {mine.notna().all(1).sum()}/{len(ref)}; AUC pipeline {t_m['gt'].mean():.4f} / {t_m['hold'].mean():.4f}")
    idx = refs[next(iter(refs))].index
    st = studies.set_index("StudyInstanceUID")
    gt_m = st.loc[idx, [f"gt_{c}" for c in ROI_CLASSES]].notna().all(1)
    from sklearn.metrics import roc_auc_score
    for name, tabs in (("pipeline", ours), ("validation", refs)):
        e = {c: rank_avg(list(tabs.values()), c, idx) for c in ROI_CLASSES}
        g = np.mean([roc_auc_score(st.loc[idx, f"gt_{c}"][gt_m], e[c][gt_m]) for c in ROI_CLASSES])
        h = np.mean([roc_auc_score((st.loc[idx, f"target_{c}"] >= 0.5).astype(int)[~gt_m], e[c][~gt_m]) for c in ROI_CLASSES])
        print(f"ensemble ({name} predictions): 58 {g:.4f}  holdout {h:.4f}")


if __name__ == "__main__":
    main()
