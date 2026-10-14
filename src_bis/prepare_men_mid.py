"""crops.prepare with the intercondylar midline applied to the menisci crops too, run from src/ (src/ unchanged).

    python ../src_bis/prepare_men_mid.py --boxes round3_pred --name men_mid_slide --depth 8 --z_mode strided \
        --stride 1 --oa_midline slide

crops.prepare --oa_midline keeps the coronal / axial medial and lateral OA crop windows on their own side of the
midline (halfway between the two menisci boxes' inner edges). This wrapper extends the same limit to the medial and
lateral meniscus boxes, so a meniscus crop cannot show the other meniscus. The crop set's json gets
"men_midline": "<mode>" (inference must apply the same limit).
"""

import json
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

import crops.prepare as cp  # noqa: E402

_oa_midline = cp.oa_midline


def oa_and_menisci_midline(bx, clip=False):
    out = _oa_midline(bx, clip)
    if not {"Medial Meniscus", "Lateral Meniscus"} <= set(bx):
        return out
    mm, lm = bx["Medial Meniscus"], bx["Lateral Meniscus"]
    med_left = (mm["x0"] + mm["x1"]) < (lm["x0"] + lm["x1"])
    mid = (mm["x1"] + lm["x0"]) / 2 if med_left else (lm["x1"] + mm["x0"]) / 2
    out = dict(out)
    for c, left in (("Medial Meniscus", med_left), ("Lateral Meniscus", not med_left)):
        out[c] = {**out[c], **({"x_hi": mid} if left else {"x_lo": mid})}
    return out


cp.oa_midline = oa_and_menisci_midline
mode = sys.argv[sys.argv.index("--oa_midline") + 1] if "--oa_midline" in sys.argv else ""
name = sys.argv[sys.argv.index("--name") + 1]
sys.argv[0] = os.path.join(SRC, "crops", "prepare.py")
cp.main()                     # main() reads the module-level oa_midline: the patched one (runpy would redefine it)
path = os.path.join(cp.CROP_DIR, f"{name}.json")
z = json.load(open(path))
z["men_midline"] = mode
json.dump(z, open(path, "w"))
print("men_midline ->", path, z)
