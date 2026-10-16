"""Recipe R + an OAI-only vote, and PF OA as seed groups; run from src/.

    PYTHONPATH=. python ../src_bis/oa_oai_vote.py

OAI-only: `all`-group ConvNeXt trained on OAI baseline crops with MOAKS v4 labels only (extra_only, runs 2026-10-13/35
and /42; CAFormer /70), so its validation predictions are leak-free and its checkpoint is deployable as is. It is the
medial-meniscus vote of the menisci work (0.983 -> 0.990 on the 58). Here it is tried as one extra vote per OA finding
(added before the lateral OA minus correction), and PF OA is tried as seed groups instead of the single mk90 seed 17.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oa_eval as E  # noqa: E402
import oa_final as F  # noqa: E402

d, n, o = "2026-10-11/", "2026-10-14/", "2026-10-13/"
OAI = [o + "35", o + "42"]                       # OAI-only ConvNeXt, seeds 0 / 1
OAI3 = [o + "35", o + "42", o + "70"]            # + CAFormer
MED_R = [[*F.MED_A[:3]], F.HYB]                  # R: [75, 33, 17] + [hybrids]
S9 = [n + "70", n + "72", n + "73"]              # lateral-only, Claude labels, oa_cmp30_mid
S10 = [n + "74", n + "77", n + "79"]             # PF-only, Claude labels
MK90 = F.MK90                                    # 17, 116, 118


def recipe(med=(), lat=(), pf=(d + "17",)):
    return {"Medial OA": {"models": MED_R + list(med)},
            "Lateral OA": {"models": [S9, *lat], "minus": ("Medial OA", 0.3)},
            "PF OA": {"models": list(pf)}}


CANDIDATES = {
    "R": recipe(),
    "R + OAI vote (all 3)": recipe([OAI], [OAI], [d + "17", OAI]),
    "R + OAI3 vote (all 3)": recipe([OAI3], [OAI3], [d + "17", OAI3]),
    "R, MOA + OAI": recipe(med=[OAI]),
    "R, LOA + OAI": recipe(lat=[OAI]),
    "R, PF 17 + OAI": recipe(pf=[d + "17", OAI]),
    "R, PF [mk90 x3]": recipe(pf=[MK90]),
    "R, PF [S10]": recipe(pf=[S10]),
    "R, PF [mk90 x3] + [S10]": recipe(pf=[MK90, S10]),
    "R, PF [mk90 x3] + OAI": recipe(pf=[MK90, OAI]),
    "R, PF [S10] + OAI": recipe(pf=[S10, OAI]),
    "R, PF [mk90 x3] + [S10] + OAI": recipe(pf=[MK90, S10, OAI]),
}


def main():
    ref = F.combine(CANDIDATES["R"])
    print("alone:", "  ".join(f"{c[:3]} {E.fmt(E.auc(F.member(OAI, c), c))}" for c in E.OA))
    for name, r in CANDIDATES.items():
        p = F.combine(r)
        s = {c: E.auc(p[c], c) for c in E.OA}
        line = f"{name:34s} " + "  ".join(f"{c[:3]} {E.fmt(v)}" for c, v in s.items()) + \
            f"  mean {E.fmt(np.mean(list(s.values()), 0))}"
        if name != "R":
            cells = []
            for c in E.OA:
                if p[c].equals(ref[c]):
                    continue
                b = E.bootstrap_delta(p[c], ref[c], c, n=1000)
                cells.append(f"{c[:3]} " + " ".join(f"{m:+.3f} ({q:.2f})" for m, lo, hi, q in b.values()))
            line += "  | vs R: " + "; ".join(cells)
        print(line)


if __name__ == "__main__":
    main()
