"""Students on pl_oadec vs their baselines: single seeds, cross-talk, and inside R2; run from src/."""
import sys
sys.path.insert(0, '../src_bis')
import oa_eval as E  # noqa: E402
import oa_final as F  # noqa: E402
import oa_oai_vote as V  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

n, d = "2026-10-14/", "2026-10-11/"
med = F.member(F.MED_A[:3], "Medial OA")


def ct(p):
    i = p.index.intersection(med.index)
    return spearmanr(p[i], med[i])[0]


def show(name, refs, c):
    p = F.member(refs, c)
    print(f"  {name:28s} {c[:3]} {E.fmt(E.auc(p, c))}   corr with medial {ct(p):.2f}")


S = {a: [n + r for r in b.split()] for a, b in
     dict(S9="70 72 73", S2="26 29 30", S10="74 77 79", S1="23").items()}
print("lateral OA"); show("S9 seed 0 (Claude)", [S["S9"][0]], "Lateral OA"); show("S9 group (Claude)", [S["S9"]], "Lateral OA")
show("S2 seed 0 (pl_round2)", [S["S2"][0]], "Lateral OA")
for r in sys.argv[1:2]:
    show(f"S16 {r} (pl_oadec)", [r], "Lateral OA")
print("PF OA"); show("S10 seed 0 (Claude)", [S["S10"][0]], "PF OA"); show("S10 group (Claude)", [S["S10"]], "PF OA")
show("S1 seed 0 (pl_round2)", S["S1"], "PF OA")
for r in sys.argv[2:3]:
    show(f"S17 {r} (pl_oadec)", [r], "PF OA")
if len(sys.argv) > 3:
    print("mk90 setup (3 findings)")
    for c in E.OA:
        show("mk90 seed 0 (17, pl_round2)", [d + "17"], c); show(f"S18 {sys.argv[3]} (pl_oadec)", [sys.argv[3]], c)
