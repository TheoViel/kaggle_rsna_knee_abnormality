"""Overlay exported attention maps on the original slices, run from src/.

    python -m utils.plot_attention --attn_dir ../logs/cls_run/<date>/<id>/attention \
        --study <uid> --classes ACL,MCL --out ../logs/attn.png

For each requested class, the series the study attention weights most for that
class is shown on its top-3 frames by z attention, with the in-plane attention
(normalised per frame) overlaid.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from main_attention import to_original  # noqa: E402
from params import CLASSES, OUTPUT_FOLDER, SERIES_INDEX_CSV  # noqa: E402


def plot_study(attn_dir, study, classes, out, n_top=3):
    idx = pd.read_csv(SERIES_INDEX_CSV).set_index("SeriesInstanceUID")
    files = sorted(glob.glob(os.path.join(attn_dir, study, "*.npz")))
    data = {os.path.basename(f)[:-4]: np.load(f) for f in files}
    fig, axes = plt.subplots(len(classes), n_top, figsize=(3.2 * n_top, 3.4 * len(classes)),
                             squeeze=False)
    for r, c in enumerate(classes):
        k = CLASSES.index(c)
        uid = max(data, key=lambda u: data[u]["series_attn"][k])
        d = data[uid]
        vol = np.load(os.path.join(OUTPUT_FOLDER, "series_npy", f"{uid}.npy"), mmap_mode="r")
        top = np.argsort(-d["z_attn"][k])[:n_top]
        for j, f in enumerate(sorted(top)):
            img = np.asarray(vol[d["slices"][f]], dtype=np.float32)
            lo, hi = np.percentile(img, [1, 99.5])
            att = to_original(d["xy_attn"][k, f].astype(np.float32), int(d["rows"]), int(d["cols"]))
            ax = axes[r, j]
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            ax.imshow(att / att.max(), cmap="jet", alpha=0.35, vmin=0, vmax=1)
            ax.set_title(f"{c} p={d['probs'][k]:.2f}  {idx.loc[uid, 'Anatomical_Plane'][:3]}\n"
                         f"slice {d['slices'][f]}  z={d['z_attn'][k, f]:.2f}  "
                         f"series w={d['series_attn'][k]:.2f}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_dir", required=True)
    p.add_argument("--study", required=True)
    p.add_argument("--classes", default=",".join(CLASSES))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plot_study(a.attn_dir, a.study, a.classes.split(","), a.out)


if __name__ == "__main__":
    main()
