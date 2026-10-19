"""pl_round3 training targets, run from src/ (proposal of 2026-10-17, reviewed by the user).

    python -m pl.round3 --oof ../output/pl_dataset_13c,../output/pl_dataset_8c --oai ../output/dataset_20/oai_runs.json

Sources, per study and finding:
  M   clean out-of-fold prediction of the dataset_13 recipe (pl.assemble on kfold_clean runs: the 58 and the holdout are
      never trained on); effusion / contusion from the dataset_8 recipe's ef0 / cc0 (dataset_13 has no such model).
      The first --oof directory that has a finding wins.
  R   mean(Claude, Codex) report labels; Claude alone where Codex is missing.
  O   the OAI-only ensemble (models trained only on OAI MOAKS gradings, output/dataset_20): rank mean of its models'
      predictions on every study (crops.predict -> <run>/all_study_preds.csv). Out-of-sample everywhere.
M and O are quantile-mapped onto D, the finding's current training label (pl_round2 for the 8 ROI findings, the OAI
noisy-OR target for synovitis, mean(Claude, Codex) for effusion / contusion): x* = quantile of D at rank(x). That keeps
each finding's prevalence (the chain-D noisy-OR pushed every target to a mean of ~0.55).

    pl_round3 = 0.5 M* + 0.5 R                      every finding
              = 0.5 M* + 0.25 R + 0.25 O*           lateral OA, effusion, synovitis
Variants: --w_m, --w_o, --oai_findings, --report best (per-finding best report source, BEST_REPORT),
--base current: (1 - w_o) D + w_o O* on the OAI findings, D elsewhere (e.g. pl2_oai50: the current labels + 50% OAI).
Fracture is not written: its models train on per-ROI location labels (frac_roi_fib_bones), unchanged.
Writes output/crops/pl_round3.csv (StudyInstanceUID x 11 findings) and prints each target's AUC on the 58.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee_cls.data import load_tables, make_splits
from params import INPUT_FOLDER, OUTPUT_FOLDER

C = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion"]
# Per finding, the report source that labels the 58 better (label AUC on the 52 with Codex, 2026-10-17)
BEST_REPORT = {"ACL": "claude", "MCL": "claude", "Medial Meniscus": "codex", "Lateral Meniscus": "claude",
               "Medial OA": "codex", "Lateral OA": "claude", "PF OA": "claude", "Effusion": "codex",
               "Synovitis": "claude", "Baker's": "codex", "Contusion": "codex"}
OAI_SETUPS = {"ACL": ["acl_mr85", "all_88"], "Medial Meniscus": ["all_88", "men_mk87", "men_cafax87"],
              "Lateral Meniscus": ["all_88", "men_mk87", "men_cafax87"], "Medial OA": ["all_88", "oa_mk89", "oa_mk90"],
              "Lateral OA": ["all_88", "oa_mk89", "oa_mk90"], "PF OA": ["all_88", "oa_mk89", "oa_mk90", "pf_ax0"],
              "Baker's": ["all_88", "bakers_91"], "Synovitis": ["syn_sy224", "syn_sc"]}
RUNS = os.path.join(OUTPUT_FOLDER, "..", "logs", "crops_run")
O_SETUPS_ALL = list(OAI_SETUPS) + ["Effusion"]


def qmap(x, ref):
    """Values of x replaced by the quantile of ref's distribution at x's percentile rank."""
    r = x.rank(pct=True)
    vals = np.quantile(ref.dropna().to_numpy(), r.dropna().clip(0, 1).to_numpy())
    return pd.Series(vals, index=r.dropna().index).reindex(x.index)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oof", required=True, help="comma-separated pl.assemble output dirs (pl_mean.csv), first wins")
    p.add_argument("--oai", default=os.path.join(OUTPUT_FOLDER, "dataset_20", "oai_runs.json"))
    p.add_argument("--name", default="pl_round3")
    # Variants (defaults = the reviewed pl_round3 recipe):
    p.add_argument("--w_m", type=float, default=0.5, help="weight of M*; R gets 1 - w_m (- w_o where OAI is used)")
    p.add_argument("--w_o", type=float, default=0.25, help="weight of O* on the --oai_findings")
    p.add_argument("--oai_findings", default="Lateral OA,Effusion,Synovitis", help="'' = no OAI; 'all' = every OAI-graded finding")
    p.add_argument("--base", default="m_r", choices=["m_r", "current"],
                   help="m_r: w_m M* + (1 - w_m) R (pl_round3); current: the current training label D itself "
                        "(pl_round2 for the ROI findings), mixed with O* only: (1 - w_o) D + w_o O*")
    p.add_argument("--w_o_map", default="", help="with --base current: JSON {finding: OAI weight} (e.g. pl4_pick.json); "
                                                 "findings not in it keep the current label")
    p.add_argument("--report", default="mean", choices=["mean", "best"],
                   help="R: mean(Claude, Codex), or per finding the source that labels the 58 better (BEST_REPORT)")
    a = p.parse_args()
    st, _ = load_tables()
    ids = st.StudyInstanceUID
    _, g, _ = make_splits(st, 0.1, 0)
    S = st.set_index("StudyInstanceUID")
    rd = lambda f: pd.read_csv(f).set_index("StudyInstanceUID")
    claude = rd(os.path.join(INPUT_FOLDER, "full_claude_context_v2_labels.csv")).reindex(ids)
    codex = pd.concat([rd(os.path.join(INPUT_FOLDER, f"codex-labels_final_{n}.csv")) for n in (52, 4349)]).reindex(ids)
    R = (claude[C] + codex[C].fillna(claude[C])) / 2
    Rmean = R.copy()                                          # the current labels use the mean, whatever the variant
    if a.report == "best":
        for c in C:
            R[c] = codex[c].fillna(claude[c]) if BEST_REPORT[c] == "codex" else claude[c]
    with_oai = set(O_SETUPS_ALL) if a.oai_findings == "all" else {x for x in a.oai_findings.split(",") if x}
    pl2 = rd(os.path.join(OUTPUT_FOLDER, "crops", "pl_round2.csv")).reindex(ids)
    D = Rmean.copy()                                          # current training labels (quantile-mapping targets)
    for c in pl2.columns:
        D[c] = pl2[c].fillna(Rmean[c])
    D["Synovitis"] = rd(os.path.join(OUTPUT_FOLDER, "crops", "syn_noisyor_rank.csv")).Synovitis.reindex(ids)
    M = {}
    for d in a.oof.split(","):
        pm = rd(os.path.join(d, "pl_mean.csv")).reindex(ids)
        for c in C:
            if c not in M and c in pm and pm[c].notna().any():
                M[c] = pm[c]
                print(f"M {c}: {d} ({int(pm[c].notna().sum())} studies)")
    runs = json.load(open(a.oai))
    O = {}
    for c in C:
        names = ["oai_cap_teacher"] if c == "Effusion" else [f"oai_{s}_s{k}" for s in OAI_SETUPS.get(c, []) for k in (0, 1)]
        ranks = [rd(os.path.join(RUNS, runs[n], "all_study_preds.csv"))[c].reindex(ids).rank(pct=True) for n in names]
        if ranks:
            O[c] = sum(ranks) / len(ranks)
    out = pd.DataFrame(index=ids)
    for c in C:
        if a.base == "current":                                # pl2_oai50-style: the current label + OAI only
            wmap = json.load(open(a.w_o_map)) if a.w_o_map else None
            w = wmap.get(c, 0.0) if wmap is not None else (a.w_o if c in with_oai else 0.0)
            out[c] = (1 - w) * D[c] + w * qmap(O[c], D[c]) if w else D[c]
            out[c] = out[c].fillna(D[c])
            continue
        assert c in M, f"no out-of-fold prediction for {c}"
        m = qmap(M[c], D[c])
        if c in with_oai:
            out[c] = a.w_m * m + (1 - a.w_m - a.w_o) * R[c] + a.w_o * qmap(O[c], D[c])
        else:
            out[c] = a.w_m * m + (1 - a.w_m) * R[c]
        out[c] = out[c].fillna(D[c])                          # no M / O (should not happen): the current label
    out.to_csv(os.path.join(OUTPUT_FOLDER, "crops", f"{a.name}.csv"))
    rows = []
    for c in C:
        y = S.loc[g, f"gt_{c}"].dropna().round()
        f = lambda x: roc_auc_score(y, x.reindex(y.index))
        rows.append(dict(finding=c, current=f(D[c].fillna(0)), M=f(M[c]), R=f(R[c]), pl_round3=f(out[c]),
                         mean_before=round(D[c].mean(), 3), mean_after=round(out[c].mean(), 3)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"-> output/crops/{a.name}.csv ({len(out)} studies)")


if __name__ == "__main__":
    main()
