"""Recipe R2 with the pl_oadec students (decorrelated OA pseudo-labels) swapped in or added; run from src/.

    PYTHONPATH=. python ../src_bis/pl_dec_r2.py S16_RUNS S18_RUNS        (comma-separated seed runs)

Offline, as the pipeline combines recipes (oa_final.combine), with a paired bootstrap of each changed finding
against R2. S16 = lateral-only on oa_cmp30_mid (S9's setup), S18 = the mk90 setup (3 findings), both on pl_oadec.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oa_eval as E  # noqa: E402
import oa_final as F  # noqa: E402

n, d = "2026-10-14/", "2026-10-11/"
S9 = [n + "70", n + "72", n + "73"]
S10 = [n + "74", n + "77", n + "79"]
MED1 = ["2026-10-09/75", d + "33", d + "17"]         # R's first medial vote: 75, mk89, mk90
MINUS = ("Medial OA", 0.3)


def recipe(med, lat, pf):
    return {"Medial OA": {"models": med}, "Lateral OA": {"models": lat, "minus": MINUS},
            "PF OA": {"models": pf, "minus": MINUS}}


def main():
    s16, s18 = (a.split(",") for a in sys.argv[1:3])
    R2 = recipe([MED1, F.HYB], [S9], [F.MK90, S10])
    med_s18 = ["2026-10-09/75", d + "33", s18]           # mk90 (17) replaced by the S18 group in the first vote
    cands = {
        "R2": R2,
        "LOA [S16]": recipe([MED1, F.HYB], [s16], [F.MK90, S10]),
        "LOA [S9] + [S16]": recipe([MED1, F.HYB], [S9, s16], [F.MK90, S10]),
        "PF [S18] + [S10]": recipe([MED1, F.HYB], [S9], [s18, S10]),
        "PF [mk90] + [S10] + [S18]": recipe([MED1, F.HYB], [S9], [F.MK90, S10, s18]),
        "MOA + [S18] vote": recipe([MED1, F.HYB, s18], [S9], [F.MK90, S10]),
        "MOA mk90 -> [S18]": recipe([med_s18, F.HYB], [S9], [F.MK90, S10]),
        "all: LOA [S9]+[S16], PF [S18]+[S10], MOA + [S18]": recipe([MED1, F.HYB, s18], [S9, s16], [s18, S10]),
    }
    ref = F.combine(R2)
    print("seed groups alone: " + "  ".join(f"S16 Lat {E.fmt(E.auc(F.member(s16, 'Lateral OA'), 'Lateral OA'))}".split("\n"))
          + "  " + "  ".join(f"S18 {c[:3]} {E.fmt(E.auc(F.member(s18, c), c))}" for c in E.OA))
    for name, r in cands.items():
        p = F.combine(r)
        s = {c: E.auc(p[c], c) for c in E.OA}
        line = f"{name:48s} " + "  ".join(f"{c[:3]} {E.fmt(v)}" for c, v in s.items()) + \
            f"  mean {E.fmt(np.mean(list(s.values()), 0))}"
        cells = []
        for c in E.OA:
            if name == "R2" or p[c].equals(ref[c]):
                continue
            b = E.bootstrap_delta(p[c], ref[c], c, n=1000)
            cells.append(f"{c[:3]} " + " ".join(f"{m:+.3f} ({q:.2f})" for m, lo, hi, q in b.values()))
        print(line + ("  | vs R2: " + "; ".join(cells) if cells else ""))


if __name__ == "__main__":
    main()
