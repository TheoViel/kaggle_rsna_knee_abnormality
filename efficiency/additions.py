"""Recipe eff + one cheap addition at a time (1 sag + 1 cor rule)."""
from evaluate import Scorer, combine
EFF = {"ACL": ["mr85"], "MCL": ["pl86"], "Medial Meniscus": ["cafax87"], "Lateral Meniscus": ["cafax87"],
       "Medial OA": ["mk90"], "Lateral OA": ["mk90"], "PF OA": ["mk90"], "Baker's": ["91"],
       "Effusion": ["jc1"], "Synovitis": ["jc1"], "Contusion": ["jc1"], "Fracture": ["jc1"]}
J4 = ["Effusion", "Synovitis", "Contusion", "Fracture"]
ADD = {"base": ({}, 0), "+jc2 (x4)": ({c: ["jc2"] for c in J4}, 2.36), "+j27 (x4)": ({c: ["j27"] for c in J4}, 2.36),
       "+88 (8 ROI)": ({c: ["88"] for c in EFF if c not in J4}, 4.2), "+mk87 menisci": ({"Medial Meniscus": ["mk87"], "Lateral Meniscus": ["mk87"]}, 4.35),
       "+fr9a": ({"Fracture": ["fr9a"]}, 4.7), "+mk89 OA": ({c: ["mk89"] for c in ["Medial OA", "Lateral OA", "PF OA"]}, 4.66)}
S = Scorer(); keep = S.pick("fs")
for n, (add, ms) in ADD.items():
    r = {c: v + add.get(c, []) for c, v in EFF.items()}
    res, a58, ah = S.auc(combine(r, keep, S.index))
    print(f"{n:16s} 58 {a58:.4f} hold {ah:.4f}  +{ms:.1f} ms/study H100  " + " ".join(f"{c[:5]} {res[c][0]:.3f}/{res[c][1]:.3f}" for c in add))
