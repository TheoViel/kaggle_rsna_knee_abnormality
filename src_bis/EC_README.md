# Effusion and contusion (night of 2026-10-14/15)

**Goal:** beat the main agent's effusion (`ef0`/`ef1`, joint4 capsule models) and contusion (`cc0`–`cc2`, `contusion_cap` on `pl:rep_study_bone`) models. Rank-blended 50/50 with b1_c1, they lowered the public LB. The suspicion is overfitting to the 58.

## Result

Package: `output/bis/dataset_ec`. All runs are in `logs/ec_runs.txt`; the queue and worker are in `logs/bis/ec/`.

```json
"Effusion": [["e2ff0", "e2ff1", "e2ff2"], ["e5ff0", "e5ff1"]],
"Contusion": [["c2ff0@FS", "c2ff1@FS", "c2ff2@FS"]]
```

Scores from the validation twins.

| | 58 / holdout | fastMRI+ half B | OAI, 300 knees (3 cuts) | OAI with b1_c1 |
|---|---|---|---|---|
| **Effusion, main ef0+ef1** | 0.983 / 0.943 | 0.957 | 0.990 / 0.918 / 0.903 | 0.993 / 0.921 / 0.910 |
| **Effusion, new** | 0.986 / 0.940 | 0.958 | 0.993 / 0.931 / 0.918 | 0.994 / 0.929 / 0.918 |
| b1_c1 alone | 0.989 / – | – | 0.994 / 0.915 / 0.912 | |
| **Contusion, main cc0–2** | 0.950 / 0.956 | 0.931 | – | – |
| **Contusion, new (@FS)** | 0.943 / 0.961 | 0.949 | – | – |

- **Contusion:** a clear external gain (+0.018 on fastMRI+, P 0.99), from dropping the 58-calibrated labels and averaging over fluid-sensitive series. On the 58 it is −0.007 (19 positives). The holdout is +0.005.
- **Effusion:** a smaller gain. It ties or beats the main agent's models on every set except the report-label holdout (−0.003). With b1_c1, it gains on two of the three OAI cuts against b1_c1 alone (+0.014, +0.006). The main agent's blend did not (+0.006, −0.002).
- **Selection caveat:** these benchmarks were used to choose among about 10 effusion and 10 contusion variants, so they are slightly optimistic for the chosen ones.
- **What decides:** the LB. If one probe is possible, try contusion first: it has the larger and cleaner gain.

## Why the 58 cannot decide

- **Effusion:** b1_c1 scores 0.989 on the 58, which is about 9 misordered pairs out of 805.
- **Contusion:** the 58 has only 19 positives.
- **Diversity:** every candidate correlates 0.94–0.96 with b1_c1 on the 58, so blends barely move.
- **Track record:** the main agent's blend looked +0.004 on the 58 and still lowered the LB.

## External benchmarks labelled from the images

The 58 and presumably the LB are labelled from the images too.

- **fastMRI+** (`src_bis/ext_bench.py`): 1,172 knees, one coronal PD series each, capsule crops `fm_cap_e15_d24`.
  - Effusion: 142 positives. Contusion: 119 ("Bone-Fracture/Contusion/dislocation" boxes).
  - Split into two stratified halves (`output/fastmri/halves.csv`). Candidates use half A at most; "fmB" is half B (72 / 60 positives).
- **OAI effusion** (MOAKS grades): 2,400 knees, crops `oai_cap_e15_d24`.
- **OAI head-to-head with b1_c1** (`src_bis/make_oai_root.py`, `src_bis/oai_h2h.py`):
  - 300 knees, 100 each at grade ≥ 2, mild and 0, extracted into a competition-style root.
  - The team runtime runs on them, giving b1_c1's own predictions.
  - The 300 knees are excluded from all OAI training here (`labels_moaks00_v4_nobench.csv`).

## Effusion

| Model | 58 / holdout | fmB | OAI ≥2 vs 0 / ≥2 vs mild / any vs 0 (300) |
|---|---|---|---|
| b1_c1 | 0.989 / – | – | **0.994 / 0.915 / 0.912** |
| main ef0+ef1 (joint4) | 0.983 / 0.943 | 0.957 | 0.990 / 0.918 / 0.903 |
| main + b1_c1 (50/50) | 0.987 (58) | – | 0.993 / 0.921 / 0.910 |
| E1 effusion-only, mean labels (333) | 0.985 / 0.942 | 0.950 | 0.987 / 0.902 / 0.899 |
| E6 effusion + OAI joint, without the 300 (344) | 0.980 / 0.941 | 0.940 | 0.989 / 0.917 / 0.937 * |
| **E2, effusion-only, noisy-OR targets with OAI pseudo-labels** (2026-10-15/39) | 0.976 / 0.934 | 0.956 | **0.993 / 0.927 / 0.915** (+ b1_c1: 0.994 / 0.925 / 0.919) |
| E2 seed 1 (2026-10-15/54) | 0.984 / 0.935 | 0.947 | 0.993 / 0.928 / 0.921 |
| E5, effusion-only, CAFormer-S18, mean labels (2026-10-15/44) | 0.984 / 0.942 | 0.953 | 0.993 / 0.920 / 0.911 |
| **[E2 s0, s1] + [E5]** (2 votes) | **0.985** / 0.941 (pipeline 0.984 / 0.941) | **0.958** | **0.993 / 0.930 / 0.919** (+ b1_c1: 0.994 / 0.928 / 0.918) |
| OAI-only capsule (2026-10-13/28) | 0.961 / 0.881 | 0.927 (fm) | trained on these knees |

\* E6 trained on 2,100 other OAI knees, so it is OAI-domain-adapted. Its OAI edge does not show on our data (58 / holdout) or on fastMRI.

- **E2's targets:** `1 − (1 − mean label) × (1 − rank(OAI-only PL))`, the synovitis recipe (`output/crops/eff_noisyor_rank.csv`; PL = run 28's effusion on every study). E2 sees no OAI image, so the OAI comparison isn't domain-biased as E6's is. But its targets carry the MOAKS grading convention, which may favour it on OAI's labels.
  - It beats the main agent's effusion on all three OAI cuts and ties on fmB.
  - The 58 (0.976 vs 0.983, ~6 pairs) and the report-label holdout (by construction) are lower.

**So far:**
- On OAI, b1_c1 alone is as good as any blend of the main agent's models, at every weight from 0 to 0.5. That is consistent with the LB.
- No effusion candidate beats b1_c1 without OAI-domain help.

## Contusion

| Model | 58 / holdout | holdout @FS | fmB |
|---|---|---|---|
| b1_c1 | 0.955 / – | – | – |
| main cc0–2 (3 seeds, `rep_study_bone`) | 0.950 / 0.956 | 0.964 | 0.931 |
| main joint4 contusion output (10/27) | 0.942 / 0.960 | 0.961 | 0.938 |
| **C2, contusion-only, plain mean labels, 3 seeds** (335, 346, 2026-10-15/0) | 0.949 / 0.953 | 0.961 | **0.949** |
| C7, C2 on Claude labels only (2026-10-15/6) | 0.941 / 0.949 | 0.955 | 0.938 |
| C8, C2 on Codex labels only (2026-10-15/16) | 0.949 / 0.956 | 0.965 | 0.932 |
| C1, C2 + fastMRI half A as extra data (2026-10-15/10) | 0.941 / 0.958 | 0.960 | 0.954 * |
| C9, C2 with CAFormer-S18 (2026-10-15/9) | 0.943 / 0.951 | 0.954 | 0.935 (C2 + C9: 0.946) |
| C3, C2 trained on fluid-sensitive series only (341) | 0.930 / 0.952 | – | 0.895 |
| fastMRI-only, half A (334) | 0.672 / 0.679 | – | 0.745 |

\* C1 trained on fastMRI half A, so it is fastMRI-domain-adapted. It loses on the 58 and with @FS on the holdout.

- **Label source matters most externally.** Seed 0 on fmB: Claude/Codex mean 0.949, Claude 0.938, Codex 0.932, the main agent's 58-calibrated `rep_study_bone` 0.931 (3 seeds). The average of two independent report readings transfers best to image-labelled knees.
- **@FS** = the study score averaged over its fluid-sensitive series only. Bone-marrow edema is bright only there; the other series score 0.914 / 0.928 alone.
  - In the pipeline: `"name@FS"` recipe entries in `src/infer/pipeline.py`, falling back to all series.
  - Verified against the offline aggregation (Spearman 0.999). `dataset_6` is bit-identical except the known CAFormer session drift.
- **fmB, C2 vs cc:** with 2 seeds, +0.015, 90% interval [+0.004, +0.029], P > 0 = 0.99. With 3 seeds: 0.949 vs 0.931.
  - The cc labels were calibrated on the 58 ("radiologists count traumatic edema positive 2/3 of the time ...").
  - Plain report labels generalize better to image-labelled knees.
- **Through the pipeline** (validation twins, `output/bis/weights_val_ec/recipe_c2fs.json`, `"Contusion": [["c2s0@FS", "c2s1@FS", "c2s2@FS"]]`): 0.943 / 0.961, the same as offline.
