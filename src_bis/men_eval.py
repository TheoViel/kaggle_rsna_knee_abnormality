"""Menisci evaluation: seed groups from logs/men_runs.txt, alone and in the current menisci recipe; run from src/.

    PYTHONPATH=. python ../src_bis/men_eval.py [TAG ...]       (tags of logs/men_runs.txt lines, e.g. B N1 N2)

Recipe = validation counterparts of the submitted menisci entries (ext.finding_eval.RECIPE): Medial Meniscus
[mk87 2026-10-11/30, cafax87 2026-10-10/93, 88 2026-10-09/75]; Lateral Meniscus [30, 93@Sagittal,Coronal, 75].
Per group: each seed and the group (mean rank) on the 58 / holdout, the medial-lateral prediction correlation, and
the recipe with the group added as one more vote (paired bootstrap vs the recipe).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oa_eval as E  # noqa: E402

M = ["Medial Meniscus", "Lateral Meniscus"]
RECIPE = {"Medial Meniscus": ["2026-10-11/30", "2026-10-10/93", "2026-10-09/75"],
          "Lateral Meniscus": ["2026-10-11/30", "2026-10-10/93@Sagittal,Coronal", "2026-10-09/75"]}
RUNS = os.path.join("..", "logs", "men_runs.txt")


def group(tag):
    """Validation runs of a tag (first word of the logs/men_runs.txt line after the time)."""
    out = []
    for line in open(RUNS):
        parts = line.split()
        if len(parts) > 2 and parts[1] == tag and "->" in line:
            out.append(line.split("-> ")[1].strip())
    return out


def recipe(c):
    return E.rank_mean([E.study(r, c) for r in RECIPE[c]])


def report(tags):
    for c in M:
        print(f"== {c}: recipe {E.fmt(E.auc(recipe(c), c))}")
        for tag in tags:
            runs = group(tag)
            have = [r for r in runs if len(E.study(r, c))]
            if not have:
                continue
            g = E.rank_mean([E.study(r, c) for r in have])
            add = E.rank_mean([E.study(r, c) for r in RECIPE[c]] + [g])
            b = E.bootstrap_delta(add, recipe(c), c, n=1000)
            print(f"   {tag:6s} seeds {[E.fmt(E.auc(E.study(r, c), c)) for r in have]}  group {E.fmt(E.auc(g, c))}"
                  f"  | recipe + group {E.fmt(E.auc(add, c))} " +
                  " ".join(f"{k} {m:+.3f} ({q:.2f})" for k, (m, lo, hi, q) in b.items()))
    for tag in tags:
        runs = group(tag)
        try:
            ct = E.crosstalk(E.rank_mean([E.study(r, M[0]) for r in runs]), E.rank_mean([E.study(r, M[1]) for r in runs]))
            print(f"   {tag}: medial-lateral prediction corr {ct:.2f}")
        except Exception:  # noqa: BLE001 -- single-finding groups
            pass


if __name__ == "__main__":
    report(sys.argv[1:])
