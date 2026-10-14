"""Candidate OA recipes from validation runs, scored as the inference pipeline combines them; run from src/.

    PYTHONPATH=. python ../src_bis/oa_final.py

A recipe entry is a list of members; a member is a run ref or a list of refs (a seed group: mean rank, one vote),
as src_bis/infer/pipeline.py combines them. "minus" subtracts beta x another finding's combined score, then re-ranks
(the pipeline's cross-talk correction). Prints 58 / holdout per finding, the OA mean, and a paired bootstrap of each
finding against the current recipe.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oa_eval as E  # noqa: E402

_orig = E.run_dir
E.run_dir = lambda r: os.path.join("..", "logs", "bis", "mt3d", r[5:]) if r.startswith("mt3d:") else _orig(r)

d, n = "2026-10-11/", "2026-10-14/"
HYB = ["mt3d:2026-10-11/" + r for r in ("30", "32", "51", "52")]            # dataset_2_moaks_hybrid counterparts
CMP = [d + r for r in ("9", "13", "14")]                                   # oa_cmp30_mid, MOAKS init
MK89 = [d + r for r in ("33", "115", "117")]                               # f_s1n8 224 px, MOAKS init
MK90 = [d + r for r in ("17", "116", "118")]                               # f_s1n8 + axial, MOAKS init
R16 = [d + "16", n + "13", n + "17"]                                       # oa_mid_slide + oa_axial_mid, MOAKS init
S2 = [n + "26", n + "29", n + "30"]                                        # lateral-OA-only on oa_cmp30_mid, MOAKS init
S3 = [n + "33", n + "35", n + "37"]                                        # lateral-OA-only on f_s1n8 224 px, MOAKS init
RECIPE = {"Medial OA": {"models": ["2026-10-09/75", d + "33", d + "17"]}, "Lateral OA": {"models": [d + "33"]},
          "PF OA": {"models": [d + "17"]}}
MED_A = ["2026-10-09/75", d + "33", d + "17", HYB]                         # recipe + the hybrid vote
CANDIDATES = {
    "A  MOA recipe+hyb | LOA cmp+mk89 | PF recipe": {"Medial OA": {"models": MED_A}, "Lateral OA": {"models": [CMP, MK89]},
                                                     "PF OA": {"models": [d + "17"]}},
    "B  A + LOA - 0.3 x MOA": {"Medial OA": {"models": MED_A},
                               "Lateral OA": {"models": [CMP, MK89], "minus": ("Medial OA", 0.3)},
                               "PF OA": {"models": [d + "17"]}},
    "C  A, PF mk90+cmp+r16": {"Medial OA": {"models": MED_A}, "Lateral OA": {"models": [CMP, MK89]},
                              "PF OA": {"models": [MK90, CMP, R16]}},
    "C2 A, PF mk90+r16": {"Medial OA": {"models": MED_A}, "Lateral OA": {"models": [CMP, MK89]},
                          "PF OA": {"models": [MK90, R16]}},
    "E  B, MOA = recipe vote + hybrid vote": {"Medial OA": {"models": [[*MED_A[:3]], HYB]},
                                               "Lateral OA": {"models": [CMP, MK89], "minus": ("Medial OA", 0.3)},
                                               "PF OA": {"models": [d + "17"]}},
    "F  B, MOA = hybrid vote only": {"Medial OA": {"models": [HYB]},
                                     "Lateral OA": {"models": [CMP, MK89], "minus": ("Medial OA", 0.3)},
                                     "PF OA": {"models": [d + "17"]}},
    "G  E, LOA S2+mk89 - 0.3": {"Medial OA": {"models": [[*MED_A[:3]], HYB]},
                                 "Lateral OA": {"models": [S2, MK89], "minus": ("Medial OA", 0.3)}, "PF OA": {"models": [d + "17"]}},
    "H  E, LOA S2+cmp+mk89 - 0.3": {"Medial OA": {"models": [[*MED_A[:3]], HYB]},
                                    "Lateral OA": {"models": [S2, CMP, MK89], "minus": ("Medial OA", 0.3)}, "PF OA": {"models": [d + "17"]}},
    "I  E, LOA S2 - 0.3": {"Medial OA": {"models": [[*MED_A[:3]], HYB]},
                           "Lateral OA": {"models": [S2], "minus": ("Medial OA", 0.3)}, "PF OA": {"models": [d + "17"]}},
    "J  E, LOA S2+cmp+mk89 (no minus)": {"Medial OA": {"models": [[*MED_A[:3]], HYB]},
                                         "Lateral OA": {"models": [S2, CMP, MK89]}, "PF OA": {"models": [d + "17"]}},
    "K  E, LOA S2+S3 - 0.3 (pre-defined)": {"Medial OA": {"models": [[*MED_A[:3]], HYB]},
                                            "Lateral OA": {"models": [S2, S3], "minus": ("Medial OA", 0.3)}, "PF OA": {"models": [d + "17"]}},
    "D  B + PF mk90+r16": {"Medial OA": {"models": MED_A},
                           "Lateral OA": {"models": [CMP, MK89], "minus": ("Medial OA", 0.3)},
                           "PF OA": {"models": [MK90, R16]}},
}


def member(m, c):
    if isinstance(m, list):
        return E.rank_mean([member(x, c) for x in m])
    return E.study(m, c)


def combine(recipe):
    base = {c: E.rank_mean([member(m, c) for m in r["models"]]) for c, r in recipe.items()}
    out = dict(base)
    for c, r in recipe.items():
        if r.get("minus"):
            f, beta = r["minus"]
            out[c] = (base[c] - beta * base[f]).rank(pct=True)
    return out


def main():
    ref = combine(RECIPE)
    rows = {"recipe": ref, **{k: combine(v) for k, v in CANDIDATES.items()}}
    for name, p in rows.items():
        s = {c: E.auc(p[c], c) for c in E.OA}
        print(f"{name:46s}", "  ".join(f"{c[:3]} {E.fmt(v)}" for c, v in s.items()),
              f"  mean {E.fmt(np.mean(list(s.values()), 0))}")
    print("\npaired bootstrap vs recipe (delta, P(delta > 0)):")
    for name, p in rows.items():
        if name == "recipe":
            continue
        cells = []
        for c in E.OA:
            b = E.bootstrap_delta(p[c], ref[c], c, n=1000)
            cells.append(f"{c[:3]} " + " ".join(f"{k} {m:+.3f} ({q:.2f})" for k, (m, lo, hi, q) in b.items()))
        print(f"  {name:44s}", " | ".join(cells))


if __name__ == "__main__":
    main()
