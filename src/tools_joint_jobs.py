"""Joint OAI + competition retrains of dataset_23's crop models (oai_ratio 0.5), run from src/.

    python tools_joint_jobs.py > ../logs/sched/joint/jobs.txt

Per setup (a dataset_23 model, or its seed pair): the model's own arguments (args.json, non-default values), plus the
OAI crop sets matching its crops (3 visits) with the MOAKS v4 labels (+ MRNet's for models already using MRNet), and
--oai_ratio 0.5. Validation run seed 0 (jv_<setup>), then the fullfits of the seeds dataset_23 uses (jf_<setup>_s<k>).
Skipped: fracture (fb0/1) and MCL (pl86): OAI grades neither; the OAI-only models.
"""
import json
import os

from main_crops import DEFAULTS

D = "../output/dataset_23/crops"
OAI = {"f_s1n8": "s1n8", "f_s1n8_axial": "s1n8_axial", "cap_e15_d24": "cap_e15_d24", "cap_e15_d24_s224": "cap_e15_d24_s224"}
SETUPS = {"p2o50_mr85": ["p2o50_mr85_s0", "p2o50_mr85_s1"], "p2o50_88": ["p2o50_88_s0", "p2o50_88_s1"],
          "p2o50_91": ["p2o50_91_s0", "p2o50_91_s1"], "p2o50_mk87": ["p2o50_mk87_s0", "p2o50_mk87_s1"],
          "p2o50_cafax87": ["p2o50_cafax87_s0", "p2o50_cafax87_s1"], "p2o50_mk89": ["p2o50_mk89_s0", "p2o50_mk89_s1"],
          "88": ["88"], "ax0": ["ax0"], "mk89": ["mk89"], "mk90": ["mk90"], "sy224": ["sy224a", "sy224b"], "sc": ["sc0", "sc1"]}
SKIP = {"output_dir", "seed", "fullfit", "n_folds", "fold", "augmentations", "z_sampling", "extra_crops", "extra_labels",
        "local_rank", "world_size", "rank", "distributed"}
HEAD = ("set -o pipefail; cd /data/rsna_knee/src && ../.venv/bin/torchrun --standalone --nproc_per_node=$NP "
        "--master_port $PORT main_crops.py")


def q(v):
    v = str(v)
    return "'" + v + "'" if any(ch in v for ch in " '\"") else v


def command(a, seed, fullfit):
    parts = [f"--{k} {q(v)}" for k, v in a.items() if k in DEFAULTS and k not in SKIP and v != DEFAULTS[k]
             and not (isinstance(v, list) and list(DEFAULTS[k] or []) == v)]
    # OAI has no capsule axial crops: capsule models get sagittal / coronal OAI crops (as the OAI-only synovitis models)
    oai = [f"oai{v}_{OAI[c]}" for c in a["crops"].split(",") if c in OAI for v in ("", "_m12", "_m24")]
    assert oai and all(os.path.exists(f"../output/crops/{x}.csv") for x in oai), oai
    extra = [x for x in a.get("extra_crops", "").split(",") if x]
    assert all(x.startswith("mrnet") for x in extra), extra
    labels = "../output/oai/labels_moaks_all_v4_plus_mrnet.csv" if extra else "../output/oai/labels_moaks_all_v4.csv"
    parts += [f"--extra_crops {','.join(extra + oai)}", f"--extra_labels {labels}", "--oai_ratio 0.5", f"--seed {seed}"]
    parts += ["--fullfit True"] if fullfit else ["--n_folds 0 --fold 0"]
    return HEAD + " " + " ".join(parts)


def main():
    val, ff = [], []
    for name, models in SETUPS.items():
        args = [json.load(open(os.path.join(D, m, "args.json"))) for m in models]
        assert all(x["fullfit"] for x in args), name
        val.append(f"jv_{name}\t{command(args[0], 0, False)}")
        for m, a in zip(models, args):
            ff.append(f"jf_{name}_s{a['seed']}\t{command(a, a['seed'], True)}")
    print("\n".join(val + ff))


if __name__ == "__main__":
    main()
