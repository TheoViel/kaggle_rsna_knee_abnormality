"""crops.prepare with the sagittal PF OA crop kept anterior to the tibiofemoral joint; run from src/ (src/ unchanged).

    python ../src_bis/prepare_pf_mid.py --boxes round3_pred --name f_s1n8_pfmask --depth 8 --z_mode strided --stride 1 \
        [--pf_mode mask|y|x]

On sagittal series the PF OA box (patella + trochlea) touches the tibiofemoral boxes, and its crop window (box + 25%
margin) takes in the anterior femoral condyle and the front of the joint. Models then score PF OA like "an arthritic
knee": their PF predictions correlate 0.74-0.77 with medial / lateral OA, the radiologists' labels 0.28-0.37. As the
intercondylar midline did for lateral OA (oa_midline), the crop is kept off the other joint:
  mask (default)  mk90's window, with its tibiofemoral corner zeroed: rows from the joint line (menisci top minus a
                  quarter of their height) and columns posterior to the menisci's anterior edge;
  y / x           the window ends at the joint line / at the menisci's anterior edge and slides (keeps its size):
                  both lose PF anatomy (patella cut when it sits low; half the window in the air in front of the knee).
Coronal and axial crops are unchanged. The crop set's json gets "pf_limit": <mode>; inference must apply pf_limit()
and the same crop_window / crop_box changes to sagittal series.
"""

import json
import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

import crops.prepare as cp  # noqa: E402

MEN = ("Medial Meniscus", "Lateral Meniscus")


def pf_limit(bx, mode="mask"):
    """Sagittal series boxes -> the PF OA box with a limit that keeps its window off the tibiofemoral joint.

    mode "mask": the window is unchanged, its tibiofemoral corner is zeroed (crop_box). mode "x": the window stops at
    the menisci's anterior edge (too strong: half the window becomes air in front of the knee and the trochlea is
    cut). mode "y": the window ends at the joint line and slides up (cuts the patella when it sits low)."""
    men = [bx[m] for m in MEN if m in bx]
    if "PF OA" not in bx or not men:
        return bx
    pf = bx["PF OA"]
    if mode == "mask":
        # the tibiofemoral corner: rows from the joint line (menisci top - 1/4 of their height: the condyles'
        # weight-bearing cartilage), columns posterior to the menisci's anterior edge
        y0 = min(m["y0"] for m in men)
        h = max(m["y1"] for m in men) - y0 + 1
        anterior_left = (pf["x0"] + pf["x1"]) / 2 < sum((m["x0"] + m["x1"]) / 2 for m in men) / len(men)
        lim = {"mask_y": y0 - 0.25 * h,
               **({"mask_x_lo": min(m["x0"] for m in men)} if anterior_left else {"mask_x_hi": max(m["x1"] for m in men)})}
    elif mode == "y":
        lim = {"y_hi": min(m["y0"] for m in men)}
    else:
        anterior_left = (pf["x0"] + pf["x1"]) / 2 < sum((m["x0"] + m["x1"]) / 2 for m in men) / len(men)
        lim = {"x_hi": min(m["x0"] for m in men)} if anterior_left else {"x_lo": max(m["x1"] + 1 for m in men)}
    return {**bx, "PF OA": {**pf, **lim}}


_crop_window = cp.crop_window


def crop_window(n_c, cs, box, n_slices, rows, cols, *args, **kw):
    """crops.prepare.crop_window + a row limit (y_hi): the window slides up so it ends at the limit row."""
    zs, y0, x0, n = _crop_window(n_c, cs, box, n_slices, rows, cols, *args, **kw)
    if "y_hi" in box:
        s = max(rows, cols)
        hi = (box["y_hi"] + 1 + (s - rows) // 2) * cs / s
        if y0 + n > hi:
            y0 = int(round(hi - n))
    return zs, y0, x0, n


_crop_box = cp.crop_box


def crop_box(cache, box, n_slices, rows, cols, flip, sagittal, depth=cp.DEPTH, size=cp.SIZE, margin=cp.MARGIN,
             min_side=cp.MIN_SIDE, z_mode="uniform", stride=1):
    """crops.prepare.crop_box + mode "mask": zero the output pixels whose source lies in the box's masked corner
    (rows >= mask_y and columns >= mask_x_lo / <= mask_x_hi, original pixels). Sagittal flips reverse the slices
    only, so the in-plane mask is unaffected."""
    out = _crop_box(cache, box, n_slices, rows, cols, flip, sagittal, depth, size, margin, min_side, z_mode, stride)
    if "mask_y" not in box:
        return out
    cs = cache.shape[1]
    _, y0, x0, n = crop_window(cache.shape[0], cs, box, n_slices, rows, cols, depth, margin, min_side, z_mode, stride)
    s = max(rows, cols)
    f = cs / s
    oy, ox = (s - rows) // 2, (s - cols) // 2
    src_y = (y0 + (np.arange(size) + 0.5) * n / size) / f - oy           # output pixel centres -> original pixels
    src_x = (x0 + (np.arange(size) + 0.5) * n / size) / f - ox
    my = src_y >= box["mask_y"]
    mx = src_x >= box["mask_x_lo"] if "mask_x_lo" in box else src_x <= box["mask_x_hi"]
    out = out.copy()
    out[:, my[:, None] & mx[None, :]] = 0
    return out


MODE = "mask"


def hook(bx, plane, shape, n_slices, union=False):
    """Stands in for crops.prepare.notch_box (called on sagittal and coronal series with --acl_notch)."""
    return pf_limit(bx, MODE) if plane == "Sagittal" else bx


if __name__ == "__main__":
    if "--pf_mode" in sys.argv:
        i = sys.argv.index("--pf_mode")
        MODE = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    cp.notch_box = hook
    cp.crop_window = crop_window
    cp.crop_box = crop_box
    name = sys.argv[sys.argv.index("--name") + 1]
    sys.argv += ["--acl_notch", "notch"]           # only to route sagittal boxes through the hook; ACL unchanged
    sys.argv[0] = os.path.join(SRC, "crops", "prepare.py")
    cp.main()
    path = os.path.join(cp.CROP_DIR, f"{name}.json")
    z = json.load(open(path))
    z.pop("acl_notch", None)
    z["pf_limit"] = MODE
    json.dump(z, open(path, "w"))
    print("pf_limit ->", path, z)
