"""Build the combined weights folder: the main agent's dataset_2_moaks + the hybrid ensemble, run from src_bis/.

    python -m mt3d.package --out ../output/bis/dataset_2_moaks_hybrid \\
        --hybrids h30=2026-10-11/67,h32=2026-10-11/68,h51=2026-10-11/69,h52=2026-10-11/70

Copies <base> (side/, roi/, crops/<run>/, recipe.json) and each hybrid run's args.json + ckpt_final.pt into
mt3d/<name>/, with the joint-crop settings it was trained on (crop.json), then writes four recipes:
  recipe.json          (default) the base recipe + one member per hybrid backbone (the mean rank of its 2
                       seeds) on all 8 ROI findings: the hybrid weighs 2 / (n + 2) of each finding;
  recipe_flat4.json    each of the 4 hybrid models as one member (4 / (n + 4): the hybrid dominates);
  recipe_all8.json     the hybrid ensemble's mean rank as one member, all 8 ROI findings;
  recipe_acl_moa.json  the hybrid ensemble as one member on ACL and medial OA only.
"""

import argparse
import json
import os
import shutil

from mt3d.prepare import MT_DIR

MT = "../logs/bis/mt3d/"
ACL_MOA = ["ACL", "Medial OA"]
README = """| Finding           | Models (recipe.json)                                                        | 58    | Holdout |
| ----------------- | --------------------------------------------------------------------------- | ----- | ------- |
| ACL               | `acl` + `all` + hybrid ConvNeXt + hybrid HGNetV2                            | 0.977 | 0.994   |
| MCL               | `mcl` + `all` + hybrid ConvNeXt + hybrid HGNetV2                            | 0.984 | 0.953   |
| Medial meniscus   | `menisci` + `all` + hybrid ConvNeXt + hybrid HGNetV2                        | 0.975 | 0.954   |
| Lateral meniscus  | `menisci` + `all` + hybrid ConvNeXt + hybrid HGNetV2                        | 0.923 | 0.924   |
| Medial OA         | `oa` 224 px (MOAKS) + `oa` + axial (MOAKS) + `all` + hybrid ConvNeXt + HGNetV2 | 0.984 | 0.950   |
| Lateral OA        | `oa` 224 px (MOAKS) + hybrid ConvNeXt + hybrid HGNetV2                      | 0.849 | 0.925   |
| PF OA             | `oa` + axial (MOAKS) + hybrid ConvNeXt + hybrid HGNetV2                     | 0.912 | 0.936   |
| Baker's           | `bakers` + `all` + hybrid ConvNeXt + hybrid HGNetV2                         | 0.986 | 0.973   |
| **Macro (8 ROI)** |                                                                             | **0.949** | **0.951** |

Each hybrid member is the mean rank of its backbone's 2 seeds, so the hybrids weigh 2 / (n + 2) of a
finding instead of dominating it.

dataset_2_moaks (unchanged: side/, roi/, crops/) + 4 hybrid multitask models (mt3d/), from src_bis/mt3d
(the auxiliary agent's line of work, see src_bis/mt3d/README.md). For comparison, dataset_2_moaks alone
scores 0.957 / 0.945 (58 / holdout).

**Hybrid models** (mt3d/h30, h32: ConvNeXt-tiny, seeds 0 / 1; h51, h52: HGNetV2-B2, seeds 0 / 1):
- Input: one joint crop per sagittal / coronal series, the union of its 8 ROI boxes + 10%, 32 x 256 x 256
  resized to 224; 16 frames, each with its two neighbouring slices as RGB.
- Model: an ImageNet 2D backbone per frame, a 3D FPN over the stacked maps; auxiliary ROI box
  segmentation (weight 0.3); per-ROI masked pooling; a transformer over the tokens of up to 4 series of
  the study, so ROI tokens see the same ROI in the other views.
- Training: MOAKS pretraining on OAI (25 gradings, 10 epochs), then 15 epochs on our data with the
  pl_round2 targets; fullfit (fullfit runs 2026-10-11/67, 68, 69, 70; the 58 radiologist studies trained on
  their radiologist labels). Validation scores above come from the non-fullfit counterparts (runs
  2026-10-11/30, 32, 51, 52).

**Recipes** (each finding = mean of its members' percentile ranks over the test studies):
- recipe.json (default): one member per hybrid backbone (mean rank of its 2 seeds), every ROI finding.
- recipe_flat4.json: each of the 4 hybrid models as one member: 0.945 / 0.952 (the hybrid dominates).
- recipe_all8.json: the 4 hybrids' mean rank as one member (nested list): 0.950 / 0.950.
- recipe_acl_moa.json: the same, on ACL and medial OA only: 0.958 / 0.945.

**Inference**: needs src_bis/infer/pipeline.py (src/infer/pipeline.py + the hybrid models + nested recipe
lists) and notebooks/rsna-knee-inf-bis.ipynb; src/ cannot read mt3d/. The hybrids reuse the shared
reading, side, U-Net and box steps; float16 autocast (--amp fp16) on T4s.
Checked end to end from the DICOMs of the 493 validation studies with the non-fullfit twins:
recipe_flat4.json 0.9454 / 0.9519 (offline 0.945 / 0.952); the 4 hybrids alone 0.9412 / 0.9517 vs 0.9410 / 0.9515
in validation. 493 studies in 160 s on 2 x H100 (110 s for the hybrids alone).
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="../output/dataset_2_moaks")
    p.add_argument("--out", required=True)
    p.add_argument("--hybrids", required=True, help="name=run,...")
    a = p.parse_args()
    meta = os.path.join(a.out, "dataset-metadata.json")         # this package's own Kaggle metadata: kept
    meta = open(meta).read() if os.path.exists(meta) else None
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(a.base, a.out, ignore=shutil.ignore_patterns("dataset-metadata.json", "readme.md"))
    names = []
    for spec in a.hybrids.split(","):
        name, run = spec.split("=")
        d = os.path.join(a.out, "mt3d", name)
        os.makedirs(d)
        for f in ("args.json", "ckpt_final.pt"):
            shutil.copy(os.path.join(MT, run, f), d)
        args = json.load(open(os.path.join(d, "args.json")))
        shutil.copy(os.path.join(MT_DIR, f"{args['crops']}.json"), os.path.join(d, "crop.json"))
        names.append(name)
    base = json.load(open(os.path.join(a.base, "recipe.json")))
    by_backbone = {}
    for n in names:                                   # one member per backbone, its seeds averaged
        bb = json.load(open(os.path.join(a.out, "mt3d", n, "args.json")))["backbone"]
        by_backbone.setdefault(bb, []).append(n)
    for fname, use, members in (("recipe.json", list(base), list(by_backbone.values())),
                                ("recipe_flat4.json", list(base), names), ("recipe_all8.json", list(base), [names]),
                                ("recipe_acl_moa.json", ACL_MOA, [names])):
        r = {}
        for c, v in base.items():
            if c in use:
                v = dict(v, models=v["models"] + members) if isinstance(v, dict) else v + members
            r[c] = v
        json.dump(r, open(os.path.join(a.out, fname), "w"), indent=1)
    with open(os.path.join(a.out, "readme.md"), "w") as f:
        f.write(README)
    if meta:
        with open(os.path.join(a.out, "dataset-metadata.json"), "w") as f:
            f.write(meta)
    print(f"-> {a.out}: base {a.base} + hybrids {by_backbone}; recipe.json (1 member per backbone, all 8), "
          f"recipe_flat4.json, recipe_all8.json, recipe_acl_moa.json")


if __name__ == "__main__":
    main()
