# OA overnight (2026-10-13/14): medial, lateral and PF OA

All scores are AUC on the 58 radiologist studies / the holdout (Claude labels ≥ 0.5). Unless stated, they come from validation runs. Tools, run from `src/`:

| Tool | Purpose |
|---|---|
| `src_bis/oa_eval.py` | a run alone, in the recipe, and the cross-talk correlation; paired bootstrap |
| `src_bis/oa_final.py` | candidate recipes, combined as the pipeline combines them, with bootstrap against the recipe |

Run logs: `logs/oa_runs.txt`, `logs/oa_*.log`, `logs/oa_eval_*.txt`.

## Starting point

The submitted OA recipe (validation counterparts):

| Finding | Models | Score |
|---|---|---|
| Medial OA | 75, `mk89` (33), `mk90` (17) | 0.977 / 0.947 |
| Lateral OA | `mk89` (33) | 0.878 / 0.903 |
| PF OA | `mk90` (17) | 0.923 / 0.930 |

**Two caveats found first:**
- **The recipe's single models are the luckiest of their 3 seeds on the 58.**
  - `mk89` (33/115/117), lateral OA on the 58: 0.878 / 0.863 / 0.874.
  - `mk90` (17/116/118), PF OA on the 58: 0.923 / 0.897 / 0.897.

  So candidates are compared with seed groups, not single seeds. A seed group is a mean rank counted as one vote.
- **For lateral OA, the two evaluation sets disagree systematically.**
  - The holdout's report-derived labels rank the radiologists' lateral OA at only 0.856.
  - 4 of the top-10 false positives on the 58 have moderate lateral OA in their report.
  - Image-graded models (hybrids, fastMRI, the OAI stacker) gain on the holdout and lose on the 58.
  - Only changes that help on both sets are kept.

## Recommended recipe (R)

| Finding | Members (seed groups, one vote each) | Score | Current recipe |
|---|---|---|---|
| Medial OA | [75, 33, 17] (the current recipe) + [hybrids `mt3d` 2026-10-11/30, 32, 51, 52] | **0.988 / 0.950** | 0.977 / 0.947 |
| Lateral OA | [lateral-only, **Claude labels**, `oa_cmp30_mid`: 2026-10-14/70, 72, 73], minus 0.3 × medial OA | **0.938 / 0.933** | 0.878 / 0.903 |
| PF OA | `mk90` (17), unchanged | 0.921 / 0.931 | 0.921 / 0.931 |
| **OA mean** | | **0.949 / 0.938** | 0.925 / 0.927 |

- **How these were measured:** through `src_bis/infer/pipeline.py` with the validation twins, on 493 studies (`output/bis/weights_val_oa/recipe_R.json`). Offline (`oa_final.py`) gives the same within ±0.002.
- **Lateral OA alone (S9):** 5 seeds, 0.903–0.930 on the 58 (group 0.919 / 0.929). The pseudo-label version (S2) gives 0.903 / 0.916, and the 3-finding `cmp30` gives 0.890 / 0.913.
- **Bootstrap against the current recipe:** lateral OA is positive in 93% / 100% of resamples without the correction (+0.041 / +0.024), and in 89% / 99% with it (+0.061 / +0.030). Medial OA: +0.011 (0.91) / +0.003 (0.89).
- **PF option (S10, PF-only on Claude labels, `mk90` crops, 2026-10-14/74, 77, 79):** 0.915 / 0.937, the most consistent PF group (seeds 0.912–0.921). It is 0.008 below `mk90` seed 17 on the 58 and 0.006 above it on the holdout. With it the OA mean is 0.947 / 0.940.
- **Superseded recipe H:** lateral OA = [lateral-only `cmp30`, pseudo-labels] + [`cmp30`] + [`mk89`] − 0.3 × medial, scoring 0.903 / 0.926 (pipeline mean 0.937 / 0.936).

Fullfit package: `output/bis/dataset_oa`.

| File | Recipe |
|---|---|
| `recipe.json` | R |
| `recipe_pf_claude.json` | R + the PF option |
| `recipe_H.json` | H |

- **Fullfit runs:** lateral-only Claude 2026-10-14/75, 76, 78; PF-only Claude /85, 88, 90. The rest are listed in the package readme.
- **Fullfit vs validation twins** (493 studies, Spearman): medial 0.989, lateral 0.938, PF 0.970.
- **Loading the new models:** `src/crops/train.py` needs the single-finding groups (`lateral_oa`, `pf_oa`, `medial_oa`); `src_bis/crops/train.py` has them.

## What makes the lateral OA gain

All three pieces are needed (3 seeds each, lateral OA on the 58 / holdout):

| Model | Crop | Targets | Labels | Lateral OA |
|---|---|---|---|---|
| `mk89` setup | standard | 3 findings | pseudo-labels | 0.872 / 0.908 |
| S3 | standard | lateral only | pseudo-labels | 0.865 / 0.915 |
| `cmp30` | midline-limited | 3 findings | pseudo-labels | 0.890 / 0.913 |
| S2 | midline-limited | lateral only | pseudo-labels | 0.903 / 0.916 |
| S11 | midline-limited | lateral only | Claude + Codex mean | 0.901 / 0.920 |
| **S9** | **midline-limited** | **lateral only** | **Claude** | **0.919 / 0.928** |

Crop ablation, lateral-only on Claude labels, 3 seeds each (with −0.3 × medial OA in brackets):

| Model | Crop set | Compartment box | Midline | Lateral OA |
|---|---|---|---|---|
| S9 | `oa_cmp30_mid` | +30% | yes | **0.919 / 0.928** (0.940 / 0.933) |
| S17 | `oa_cmp30` | +30% | no | 0.907 / 0.921 (0.926 / 0.928) |
| S19 | `oa_m50_mid` (margin 0.5 instead of the box) | no | yes | 0.899 / 0.926 (0.919 / 0.931) |
| S18 | `oa_cmp50` | +50% | no | 0.875 / 0.908 (0.880 / 0.911) |
| S16 | `oa_mid_slide` | no | yes | 0.867 / 0.928 (0.870 / 0.933) |
| S14 | `oa_cmp30_mid` at 224 px | +30% | yes | 0.865 / 0.929 |

- **The compartment box** is the joined OA + meniscus box, extended 30% into the femoral condyle and the tibial plateau. It is what matters most; at +50% it reaches too far.
- **The midline** adds about +0.01.
- **At 224 px** the gain is lost.

1. **The crop stops at the intercondylar midline,** so the model cannot see the medial compartment.
2. **One finding per model,** so the encoder cannot share "arthritic knee" features learned from the other compartments' targets.
3. **Claude labels.** The pseudo-labels (`pl_round2`) amplify the medial–lateral label correlation (0.54, against 0.44 for Claude and 0.32 for the radiologists). The Claude labels also rank the radiologists' lateral OA better (0.856, against 0.845 for `pl_round2`).

PF-only on Claude labels at 224 px with axial crops (S15, 2026-10-14) scores 0.902 / 0.937, below S10. The same recipe does not help medial OA: medial-only S12 on Claude labels scores 0.986 / 0.948, and adding it to R's medial OA gives +0.001. PF OA moves only slightly: PF-only on Claude labels (S10) is the most consistent PF group but does not beat the lucky seed 17 on the 58.

## What worked

1. **One finding per model on the midline-limited compartment crop: lateral-only `cmp30`.**
   - 3 seeds: 0.911, 0.901, 0.894 on the 58; group 0.903 / 0.916.
   - The 3-finding `cmp30` group scores 0.890 / 0.913.
   - Both changes are needed. The crop stops at the intercondylar midline, so the input cannot show the medial compartment, and the single target stops the encoder from sharing "arthritic knee" features.
   - The same lateral-only training on the standard crop (`mk89` setup, 224 px) does not help: 0.865 / 0.915 against 0.872 / 0.908.
   - PF-only (`mk90`-like) does not help either: 0.901 / 0.933.
2. **The hybrid models for medial OA.**
   - They are my multitask 3D models on the whole-joint crop, from the `dataset_2_moaks_hybrid` package.
   - The 4-model vote scores 0.994 / 0.952 alone.
3. **The midline-limited compartment crops (`oa_cmp30_mid`, MOAKS init, main agent's runs 9/13/14).** On average they beat the recipe setups on medial and lateral OA (0.986 / 0.946, 0.888 / 0.912), but were not in the recipe.
4. **fastMRI+ OA-only models.**
   - These are clinical coronal series with compartment-specific cartilage boxes (runs 2026-10-13/92, 100).
   - Medial OA alone: 0.984 and 0.992 on the 58.
   - Recipe + fastMRI vote: +0.008 on the 58 for medial OA.
   - Subsumed by the hybrids.

## What did not work

Single seed unless stated.

| Idea | Run | Result |
|---|---|---|
| Pair loss over medial / lateral / PF crops of the same series | `bis:2026-10-13/2` vs `/3` | Cross-talk 0.84 → 0.91; lateral OA 0.892 → 0.855 |
| Per-compartment report extraction: 3,538 reports read by 25 Claude agents | `output/bis/oa_reports`, `src_bis/oa_labels.py` | As targets, not better than the Claude labels against the radiologists (lateral 0.836–0.841 vs 0.856; PF 0.86–0.88 vs 0.945); does not lower the label correlation |
| Extracted severity as an auxiliary target (`mk89` setup) | 2026-10-13/101 | Medial −0.005, lateral −0.021, PF −0.009 on the 58 |
| MOAKS pretraining on 3 OAI visits (+m12, m24) | CoAtNet 91; `mk89` 2026-10-14/7; `mk90` /10 | Lateral OA −0.038 on the 58, holdout +0.006; same failure as for the menisci |
| Joint training with fastMRI (`mk89` setup) | 2026-10-13/93 | Lateral 0.847 on the 58 |
| One model on OAI + fastMRI image-graded labels only | 2026-10-14/3 | Cross-talk 0.35 (radiologists 0.32), but medial 0.938, lateral 0.799, PF 0.772 |
| Stacker over all OA model groups, trained on the holdout | – | Learns the report convention; lateral 0.861–0.870 on the 58 |
| PF variants | Q1 (224 px + axial) 2026-10-14/8, Q2 (sagittal + axial only) /12, run-16 seeds 13/17, S1 PF-only /23 | Each within ±0.01 of the `mk90` seed mean (0.906 / 0.932). PF ensembles gain +0.003 to +0.006 on the holdout but stay 0.006–0.016 below the lucky `mk90` seed 17 on the 58, so PF is unchanged |
| Single tibiofemoral crop (`tf` group, main agent's runs 2026-10-10/54–56) | – | Lateral 0.81–0.88, cross-talk 0.78–0.89 |
| Lateral-only on the standard crop (S3, `mk89` setup, 224 px) | 2026-10-14/33, 35, 37 | Group 0.865 / 0.915 vs 3-finding `mk89` 0.872 / 0.908 |
| Medial-only on `oa_cmp30_mid` (S4) | 2026-10-14/43, 45, 46 | Group 0.984 / 0.947 vs 3-finding 0.986 / 0.946; recipe + hybrid + S4: 0.986 / 0.950 (H's medial: 0.988 / 0.950) |
| Lateral-only on `oa_cmp30_mid` at 224 px (S5) | 2026-10-14/48 + 2 seeds | Group 0.886 / 0.916; H + S5: 0.905 / 0.928 (H: 0.905 / 0.927) |
| Lateral-only on `oa_mid_slide` (midline crop without the compartment box, S6) | 3 seeds, `logs/oa_runs.txt` | Group 0.890 / 0.916; H + S6: 0.905 / 0.927 |

## Robustness

- **Correction strength β, R's lateral OA (S9 group minus β × the combined medial OA):**

  | β | 58 / holdout |
  |---|---|
  | 0 | 0.919 / 0.928 |
  | 0.1 | 0.936 / 0.931 |
  | 0.2 | 0.932 / 0.933 |
  | 0.3 | 0.940 / 0.933 |
  | 0.4 | 0.934 / 0.928 |
  | 0.5 | 0.938 / 0.918 |

  It is a plateau from 0.1 to 0.4.

- **Correction strength β, H's lateral OA:**

  | β | 58 / holdout |
  |---|---|
  | 0 | 0.897 / 0.914 |
  | 0.2 | 0.903 / 0.924 |
  | 0.3 | 0.905 / 0.927 |
  | 0.4 | 0.911 / 0.924 |
  | 0.5 | 0.911 / 0.917 |

  It is a broad plateau from 0.2 to 0.4.
- **Fullfit package vs its validation twins:** on the 493 validation studies, through the pipeline, the Spearman correlation is 0.989 (medial), 0.977 (lateral) and 0.970 (PF).

## Running R with `src/` (ported 2026-10-14)

`src/infer/pipeline.py` now supports what R needs:
- **Nested seed groups:** `flat()`, plus `member()` in `combine`.
- **Hybrid models** from `<weights>/mt3d/<name>/` (`load_hybrid`, `hybrid_crops`, `study_groups`, `hybrid_predict`). The inference code is in `src/mt3d/` (`hybrid.py`, `model.py` and `ortho.py`, copied from `src_bis/mt3d`).
- **The `pf_oa` / `lateral_oa` / `medial_oa` groups** in `src/crops/train.py`.
- **`--recipe`** (a recipe file) and **`--amp`** (hybrid autocast, default fp16).

Checks:
- **No change for previous recipes:** with `dataset_2_moaks_mrnet_men_frac` and `dataset_7`, 100 validation studies, original and ported code run back to back give bit-identical per-model predictions and submissions. The hybrid warm-up only runs when a hybrid is loaded, because an extra autocast context alone shifts the CAFormer models by about 1e-4. Note that the original code itself varies by up to about 1.6e-4 between sessions (`sc0` / `sc1`, CAFormer), so compare runs back to back.
- **Recipe R** (`output/bis/weights_val_oa/recipe_R.json`, 493 studies): identical to the `src_bis` pipeline (Spearman 1.0; 0.988 / 0.950, 0.938 / 0.933, 0.921 / 0.931). 147 s on one H100.

The inference notebook's weight check (`notebooks/rsna-knee-inf.ipynb`, parameters cell) assumes flat entries under `crops/`. For recipes with nested lists or hybrids it needs:

```python
def flat(v):
    return [m for x in v for m in (flat(x) if isinstance(x, list) else [x])]
models = {m.split("@")[0] for v in recipe.values() for m in flat(v if isinstance(v, list) else v["models"])}
for w in ("side", "roi") + tuple(f"mt3d/{m}" if os.path.isdir(os.path.join(WEIGHTS, "mt3d", m)) else f"crops/{m}" for m in models):
    assert os.path.exists(os.path.join(WEIGHTS, w, "ckpt_final.pt")), f"missing weights: {w}"
```

## Replicating the lateral OA recipe on the menisci (2026-10-14): it does not transfer

- **Tools and logs:** `src_bis/men_eval.py`; runs in `logs/men_runs.txt`.
- **Midline crop set:** `men_mid_slide`, built with `src_bis/prepare_men_mid.py`. The OA midline is extended to the meniscus boxes and changes 68% / 87% of coronal medial / lateral meniscus crops.
- **Setup:** MOAKS-initialised ConvNeXt at 128 px (init 2026-10-13/82) unless stated; 3 seeds each; 58 / holdout.

| Group | Crop | Medial meniscus | Lateral meniscus | Cross-talk |
|---|---|---|---|---|
| B: 2-finding | standard | 0.972 / 0.954 | 0.885 / 0.927 | 0.53 |
| B2: 2-finding | midline | 0.972 / 0.952 | 0.893 / 0.928 | 0.54 |
| N1 / N2: one finding | standard | 0.969 / 0.955 | 0.894 / 0.917 | – |
| N3 / N4: one finding | midline | 0.966 / 0.955 | 0.903 / 0.919 | – |
| `mk87` setup (CoAtNet 224, MOAKS init 29), 2-finding | standard | 0.977 / 0.956 | 0.933 / 0.925 | – |
| N7: `mk87` setup, lateral only | standard | – | 0.925 / 0.928 | – |
| N6: `mk87` setup, lateral only | midline | – | 0.916 / 0.924 | – |

The current menisci recipe scores 0.983 / 0.955 (medial) and 0.959 / 0.921 (lateral). In it, swapping `mk87` for N7 gives lateral 0.957 / 0.923; for N6, 0.954 / 0.923.

**Why it does not transfer:**
- **The cross-talk does not come from the crop.** The predictions' medial–lateral correlation is about 0.53 against 0.12 in the labels, but the midline crop leaves it unchanged (0.53 → 0.54). The shared signal is more likely general meniscal degeneration in the same knee, which is real image content, not leakage from the other compartment.
- **The label lever does not apply.** For the lateral meniscus the pseudo-labels are already the best source against the radiologists (`pl_round2` 0.911, Claude 0.888, Codex 0.850).
- **Single-finding training gives ±0.01 on the 128 px models, and nothing on the strong CoAtNet setup.**

The only menisci gain found so far remains the OAI-only `all` model as an extra medial meniscus vote: 0.990 / 0.959 against 0.983 / 0.955 (see `oai-menisci` in memory and the earlier OAI report).

## PF OA and the OAI-only vote (2026-10-14 afternoon): recipe R2

R's PF OA was the single `mk90` model (MOAKS-pretrained, init 2026-10-11/8), the luckiest of its 3 seeds on the 58. Scripts: `oa_oai_vote.py` (offline), `prepare_pf_mid.py` (PF crop limits).

- **PF cross-talk, from the labels.** Spearman PF ~ medial / lateral OA is 0.37 / 0.28 for the radiologists, but 0.69 / 0.69 for the report labels we train on, and 0.74–0.77 for every PF model. The models learn "PF ≈ any OA".
- **PF-limited sagittal crops do not fix it** (unlike the lateral OA midline):
  - Moving the window (stop at the menisci's anterior edge, or at the joint line) loses PF anatomy: half the window is air, or the patella is cut.
  - Zeroing the tibiofemoral corner of mk90's window (`f_s1n8_pfmask`, S15, PF-only, Claude labels, run 2026-10-14/228): 0.914 / 0.936, the same as S10 seed 0 (0.914 / 0.937). Cross-talk unchanged (0.74). Discarded after the first seed.
- **The minus correction does fix it.** PF − β × combined medial OA, with a plateau at β 0.2–0.4 for every PF base, on both sets. At β = 0.3:

  | PF base | no correction | − 0.3 × medial |
  |---|---|---|
  | mk90 seed 17 | 0.923 / 0.930 | 0.936 / 0.939 |
  | [mk90 ×3] | 0.907 / 0.932 | 0.921 / 0.941 |
  | [S10] | 0.915 / 0.937 | 0.918 / 0.943 |
  | [mk90 ×3] + [S10] | 0.910 / 0.936 | 0.923 / 0.944 |

- **R2's PF:** [mk90 ×3] + [S10] − 0.3 × medial OA. Two seed groups, all MOAKS-pretrained, no lucky seed.
  - Pipeline with validation twins (`weights_val_oa/recipe_R2.json`): **0.921 / 0.945** vs R 0.921 / 0.931. Bootstrap (offline): 58 −0.001 (P 0.47), holdout +0.014 (P 0.99).
  - Fullfits: mk90 seeds 1 and 2 = 2026-10-14/229, /232 (`mk90s1`, `mk90s2`); S10 = pfc0–2.
  - Package: `output/bis/dataset_oa/recipe_R2.json`. Fullfit vs twins Spearman on PF: 0.949.
- **OAI-only models (2026-10-13/35, /42) as an OA vote: worse everywhere.** Alone: medial 0.930, lateral 0.803, PF 0.812. As one extra vote: medial −0.006, lateral −0.045 / −0.023, PF −0.037 / −0.015.
- **OAI-only as the medial meniscus vote: confirmed** (pipeline, validation twins): 0.983 / 0.955 → **0.990 / 0.959** (bootstrap P 0.91 / 0.97). Copied to `dataset_oa/crops/oai0`, `oai1`. To use it, see the package readme.
- **Fullfit of the OAI vote: OAI + the 58 radiologist studies** (`src_bis/train_oai58.py`; runs 2026-10-14/235, /237 = seeds 0, 1; package `oai58s0`, `oai58s1`).
  - `--extra_only --fullfit` alone would drop the 58; the wrapper appends their 255 series (2.6% of the items, radiologist labels).
  - Holdout (the only clean set): alone 0.948 and as the medial meniscus vote 0.958, against 0.948 / 0.959 for OAI-only.
  - Spearman with OAI-only on the 493 validation studies: 0.949.
  - `--gt_repeat k` weights the 58 up; not tried, as it can't be validated.

## Decorrelated OA pseudo-labels (2026-10-14 evening): no gain

**Idea:** pseudo-labels from teachers without the cross-talk, so students get pseudo-labels' smoothing without `pl_round2`'s amplified medial–lateral correlation.

**Teachers:** out of fold. `src_bis/train_oof.py` uses 5 folds over the training studies only; the 58 and the holdout are predicted by every fold and trained by none. `main_crops`' own k-fold mode would train on them. Seed 0, runs in `logs/oa_runs.txt` ("OOF"):
- T1: lateral-only on `oa_cmp30_mid`, Claude labels (S9's setup);
- T2: PF-only, Claude labels (S10's setup);
- T3: the mk90 setup.

**Pseudo-labels** (`src_bis/make_pl_dec.py` → `output/crops/pl_oadec.csv`, recipe fixed in advance):
- Teacher scores, as ranks:
  - lateral = T1 − 0.3 × T3's medial OA;
  - PF = mean(T2, T3) − 0.3 × T3's medial OA.
- Each corrected rank is mapped onto the teachers' own probabilities (quantile mapping).
- Target = 0.5 × teacher + 0.5 × Claude label. Only the lateral and PF OA columns of `pl_round2` are replaced, and only on the training studies.

**The labels are decorrelated.** Spearman over the 3,914 training studies:

| Labels | medial–lateral | medial–PF |
|---|---|---|
| Claude | 0.76 | 0.65 |
| `pl_round2` | 0.80 | 0.70 |
| `pl_oadec` | 0.67 | 0.61 |

The teachers on the 58 / holdout: corrected lateral 0.932 / 0.931, corrected PF 0.916 / 0.947.

**The students don't gain.** Baselines are each setup's 3-seed group or mean:

| Student | pl_oadec | Baseline |
|---|---|---|
| S16 lateral-only cmp30 (291, 296, 301) | group 0.913 / 0.928 | S9 0.919 / 0.928 |
| S17 PF-only (292) | 0.901 / 0.939 | S10 seed mean 0.916 / 0.936 (discarded after seed 0) |
| S18 mk90 setup (293, 297, 303), medial | 0.981 / 0.947 | seed mean 0.975 / 0.947 |
| S18 mk90 setup, lateral | 0.867 / 0.931 | seed mean 0.872 / 0.912 |
| S18 mk90 setup, PF | 0.906 / 0.939 | [mk90 ×3] 0.907 / 0.932 |

The students' prediction cross-talk drops only a little (0.78 → 0.76 lateral, 0.77 → 0.72 for the mk90 setup).

**Inside R2** (`src_bis/pl_dec_r2.py`, offline, against R2 at 0.988 / 0.950, 0.938 / 0.934, 0.923 / 0.944):
- Lateral [S16] instead of [S9]: −0.016 / −0.009. [S9] + [S16]: −0.005 / −0.004.
- PF [S18] instead of [mk90 ×3]: −0.005 / −0.001.
- An extra medial vote [S18]: −0.002 / 0.000.
- mk90 → [S18] inside the first medial vote: ±0.000.
- All together: −0.002 to −0.005.

**Why:** R2 already removes the cross-talk at the recipe level, with the minus correction on lateral and PF. Students that learn the corrected ordering bring nothing the correction doesn't. Their holdout gains (S18 lateral +0.019) are the report-label convention: pl_oadec is half Claude, which are the holdout's labels.

**R2 is unchanged.**
