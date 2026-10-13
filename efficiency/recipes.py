"""A few pre-defined simple recipes under the 1 sag + 1 cor rule: 58 / holdout and GPU cost."""
from evaluate import Scorer, combine, SUBMITTED
MS = {"88": .262, "91": .262, "mr85": .262, "pl86": .262, "mk90": .262, "fr9a": .262 * 9, "fr9b": .262 * 9,
      "cafax87": .425, "mk87": 1.087, "mk89": .777, "jc1": 1.18, "j27": 1.18}
J = {c: ["jc1"] for c in ["Effusion", "Synovitis", "Contusion", "Fracture"]}
R = {
    "A submitted 8 ROI + fr9a,b + jc1": {**SUBMITTED, **{c: ["jc1"] for c in ["Effusion", "Synovitis", "Contusion"]}},
    "B one specialist/finding, mk87, jc1x4": {"ACL": ["mr85"], "MCL": ["pl86"], "Medial Meniscus": ["mk87"], "Lateral Meniscus": ["mk87"],
        "Medial OA": ["mk90"], "Lateral OA": ["mk89"], "PF OA": ["mk90"], "Baker's": ["91"], **J},
    "C as B, cafax87 menisci": {"ACL": ["mr85"], "MCL": ["pl86"], "Medial Meniscus": ["cafax87"], "Lateral Meniscus": ["cafax87"],
        "Medial OA": ["mk90"], "Lateral OA": ["mk89"], "PF OA": ["mk90"], "Baker's": ["91"], **J},
    "D 88 x8 + jc1x4": {**{c: ["88"] for c in SUBMITTED if c != "Fracture"}, **J},
    "E as B, fracture fr9a": {"ACL": ["mr85"], "MCL": ["pl86"], "Medial Meniscus": ["mk87"], "Lateral Meniscus": ["mk87"],
        "Medial OA": ["mk90"], "Lateral OA": ["mk89"], "PF OA": ["mk90"], "Baker's": ["91"], **J, "Fracture": ["fr9a"]},
    "F as B, mk87+cafax87 menisci": {"ACL": ["mr85"], "MCL": ["pl86"], "Medial Meniscus": ["mk87", "cafax87"], "Lateral Meniscus": ["mk87", "cafax87"],
        "Medial OA": ["mk90"], "Lateral OA": ["mk89"], "PF OA": ["mk90"], "Baker's": ["91"], **J},
}
def cost(r):
    used = {(m.split("@")[0], f) for f, ms in r.items() for m in ms}
    c = sum(2 * MS[m] for m, f in used if m not in ("jc1", "j27"))
    c += sum(2 * MS[m] for m in {m for m, _ in used if m in ("jc1", "j27")})   # one shared encoder pass
    return c
S = Scorer()
for rule in ("fs", "all_sc"):
    keep = S.pick(rule)
    print("== rule", rule)
    for n, r in R.items():
        res, a58, ah = S.auc(combine(r, keep, S.index))
        c = cost(r) * (1 if rule == "fs" else 2.1)
        print(f"  {n:40s} 58 {a58:.4f} hold {ah:.4f}  models {c:5.1f} ms/study H100 ~ {c * 6.5:4.0f} s T4")
