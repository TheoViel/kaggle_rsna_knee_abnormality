"""Build the combined weights folder: the main agent's dataset_2_moaks + the hybrid ensemble, run from src_bis/.

    python -m mt3d.package --out ../output/bis/dataset_2_moaks_hybrid \\
        --hybrids h30=2026-10-11/67,h32=2026-10-11/68,h51=2026-10-11/69,h52=2026-10-11/70

Copies <base> (side/, roi/, crops/<run>/, recipe.json) and each hybrid run's args.json + ckpt_final.pt into
mt3d/<name>/, with the joint-crop settings it was trained on (crop.json), then writes three recipes:
  recipe.json          (default) the base recipe + each hybrid model as one more member on all 8 ROI findings
                       (micro-level rank average: 4 / (n + 4) of each finding);
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="../output/dataset_2_moaks")
    p.add_argument("--out", required=True)
    p.add_argument("--hybrids", required=True, help="name=run,...")
    a = p.parse_args()
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
    for fname, use, members in (("recipe.json", list(base), names), ("recipe_all8.json", list(base), [names]),
                                ("recipe_acl_moa.json", ACL_MOA, [names])):
        r = {}
        for c, v in base.items():
            if c in use:
                v = dict(v, models=v["models"] + members) if isinstance(v, dict) else v + members
            r[c] = v
        json.dump(r, open(os.path.join(a.out, fname), "w"), indent=1)
    print(f"-> {a.out}: base {a.base} + hybrids {names}; recipe.json (4 members, all 8), recipe_all8.json, "
          f"recipe_acl_moa.json")


if __name__ == "__main__":
    main()
