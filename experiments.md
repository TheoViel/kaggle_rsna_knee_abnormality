# Experiments

All scores are **AUC against the radiologist labels of the 58 studies** in `train.csv`, which are never trained on. Every table was generated from the run folders (`logs/cls_run/`, `logs/crops_run/`), not copied by hand. **Bold marks the best value of each column.**

- **Noise:** with 8–35 positives per finding, a single finding's AUC on 58 studies moves by several points between equivalent models.
- **Selection rule:** model choices are therefore never made on the 58. For each finding, the run kept for the final comparison (marked **★**) is the one with the best score on a separate holdout of 435 report-labelled studies. That score is not reported here. The ★ run is not always the bold one: bold is the best on the 58, chosen after the fact.
- **Crop models:** unless stated, the ROI crop models use ConvNeXt-tiny, 128 px crops of 12 slices, a BiLSTM over slices, sagittal and coronal series, and 10 epochs. "Train labels" is the report-label set the model was trained on (Claude, Codex, or their mean).

---

## 1. Baseline sequence classifier

`src/main_cls.py`: a 2.5D attention model on whole series, all 12 findings. It is a reference only; it is not combined with the ROI models.

| Run | Configuration (convnext_tiny unless stated) | 58 radiologist (macro) |
|---|---|---|
| `cls_run/2026-10-08/2` | 224 px, 6 frames (reference) | 0.881 |
| `cls_run/2026-10-08/3` | 224 px, 10 frames | 0.878 |
| `cls_run/2026-10-08/4` | 288 px, 6 frames | 0.884 |
| `cls_run/2026-10-08/5` | convnext_small, 224 px | 0.882 |
| `cls_run/2026-10-08/6` | 224 px, 8 frames, frame offset 2 | 0.878 |
| runs 2–6 | Ensemble of the 5 | **0.890** |

**Per finding:**

| Run | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture | Macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 224 px, 6 frames (reference) | 0.958 | 0.762 | 0.930 | 0.824 | 0.974 | 0.832 | 0.846 | 0.948 | 0.792 | 0.944 | **0.934** | 0.831 | 0.881 |
| 224 px, 10 frames | 0.962 | 0.748 | 0.933 | 0.837 | 0.971 | 0.805 | 0.840 | 0.957 | 0.778 | 0.933 | 0.919 | 0.850 | 0.878 |
| 288 px, 6 frames | 0.972 | 0.732 | **0.941** | 0.827 | 0.971 | 0.826 | **0.875** | **0.961** | 0.778 | 0.906 | 0.930 | 0.887 | 0.884 |
| convnext_small, 224 px | **0.984** | **0.764** | 0.894 | 0.809 | **0.988** | **0.834** | 0.866 | 0.952 | 0.760 | 0.940 | 0.920 | 0.872 | 0.882 |
| 224 px, 8 frames, frame offset 2 | 0.967 | 0.717 | 0.915 | 0.827 | 0.983 | 0.789 | 0.811 | 0.929 | **0.810** | **0.971** | 0.923 | **0.893** | 0.878 |
| Ensemble of the 5 | 0.974 | 0.753 | 0.936 | **0.845** | 0.986 | 0.820 | 0.855 | 0.959 | 0.791 | 0.955 | 0.933 | 0.876 | **0.890** |

- **Best: the ensemble of the 5 runs, 0.890.** No single change beats the reference beyond noise.
- The weakest findings are MCL, synovitis, lateral OA and the lateral meniscus.

---

## 2. Menisci

ROI crop models on the medial and lateral meniscus boxes. The `all` group is one model for all 8 ROIs.

| Run | Group | Backbone | px | Head | Planes | Train labels | Ep. | Med. men. | Lat. men. | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | menisci | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | **0.971** | 0.906 | 0.938 |
| 3 | medial_meniscus | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.966 | — | — |
| 4 | lateral_meniscus | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | — | 0.870 | — |
| 10 | all | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.954 | 0.889 ★ | 0.922 |
| 11 | menisci | convnext_tiny | 128 | lstm | sag | Claude | 10 | 0.959 | 0.880 | 0.919 |
| 13 | menisci | convnext_tiny | 128 | lstm | cor | Claude | 10 | 0.933 | 0.856 | 0.894 |
| 15 | menisci | convnext_tiny | 224 | lstm | sag+cor | Claude | 10 | 0.957 | 0.912 | 0.934 |
| 16 | menisci | convnext_tiny | 128 | mean | sag+cor | Claude | 10 | 0.964 | 0.906 | 0.935 |
| 17 | menisci | convnext_small | 128 | lstm | sag+cor | Claude | 10 | 0.947 | **0.923** | 0.935 |
| 18 | menisci | tf_efficientnetv2_s | 128 | lstm | sag+cor | Claude | 10 | 0.956 | 0.911 | 0.933 |
| 19 | menisci | maxvit_tiny | 224 | lstm | sag+cor | Claude | 10 | 0.906 | 0.922 | 0.914 |
| 20 | menisci | coatnet_0 | 224 | lstm | sag+cor | Claude | 10 | 0.962 ★ | 0.919 | **0.940** |
| 21 | menisci | convnext_tiny | 128 | lstm | sag+cor | Claude | 15 | 0.936 | 0.894 | 0.915 |
| 35 | menisci | convnext_tiny | 128 | lstm | axial | Claude | 10 | 0.964 | 0.881 | 0.923 |
| 36 | menisci | convnext_tiny | 128 | lstm | sag+cor+axial | Claude | 10 | 0.969 | 0.893 | 0.931 |
| 41 | menisci | convnext_tiny | 128 | lstm | sag+cor | mean | 10 | 0.962 | 0.914 | 0.938 |
| 46 | menisci | convnext_tiny | 128 | lstm | sag+cor | Codex | 10 | 0.962 | 0.908 | 0.935 |

- **Medial Meniscus**: selected run (★) **20**, 0.962 on the 58; best on the 58: run **2**, 0.971.
- **Lateral Meniscus**: selected run (★) **10**, 0.889 on the 58; best on the 58: run **17**, 0.923.
- **Grouping:** one shared model for both menisci beats separate models on the lateral meniscus (run 2 vs run 4).
- **Planes:** sagittal + coronal beat either plane alone. Axial does not help the menisci.
- **Resolution and backbone:** 224 px (runs 15, 19, 20) and CoAtNet-0 / ConvNeXt are the most consistent.
- **Training length:** 15 epochs overfits (run 21).

---

## 3. OA (medial, lateral, patellofemoral)

ROI crop models on the three compartment boxes. The `oa` group is one model for the three findings.

| Run | Group | Backbone | px | Head | Planes | Train labels | Ep. | Med. OA | Lat. OA | PF OA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | oa | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.971 | 0.845 | 0.864 | 0.893 |
| 10 | all | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.958 ★ | 0.843 | 0.898 | 0.900 |
| 12 | oa | convnext_tiny | 128 | lstm | sag | Claude | 10 | 0.958 | 0.791 | 0.857 | 0.869 |
| 14 | oa | convnext_tiny | 128 | lstm | cor | Claude | 10 | 0.963 | 0.841 | 0.857 | 0.887 |
| 22 | oa | convnext_tiny | 224 | lstm | sag+cor | Claude | 10 | 0.975 | 0.880 ★ | 0.876 | 0.911 |
| 23 | oa | convnext_tiny | 128 | mean | sag+cor | Claude | 10 | 0.971 | 0.855 | 0.856 | 0.894 |
| 24 | oa | convnext_small | 128 | lstm | sag+cor | Claude | 10 | 0.950 | 0.824 | 0.879 | 0.884 |
| 25 | oa | tf_efficientnetv2_s | 128 | lstm | sag+cor | Claude | 10 | 0.975 | 0.814 | 0.894 | 0.895 |
| 26 | oa | maxvit_tiny | 224 | lstm | sag+cor | Claude | 10 | 0.963 | 0.865 | **0.916** | 0.915 |
| 27 | oa | coatnet_0 | 224 | lstm | sag+cor | Claude | 10 | **0.977** | **0.882** | 0.900 | **0.919** |
| 28 | oa | convnext_tiny | 128 | lstm | sag+cor | Claude | 15 | 0.964 | 0.839 | 0.898 | 0.901 |
| 29 | oa | convnext_tiny | 128 | lstm | axial | Claude | 10 | 0.904 | 0.843 | 0.914 | 0.887 |
| 30 | oa | convnext_tiny | 128 | lstm | sag+cor+axial | Claude | 10 | 0.958 | 0.834 | 0.900 ★ | 0.897 |
| 42 | oa | convnext_tiny | 128 | lstm | sag+cor | mean | 10 | 0.969 | 0.865 | 0.873 | 0.902 |
| 47 | oa | convnext_tiny | 128 | lstm | sag+cor | Codex | 10 | 0.974 | 0.876 | 0.853 | 0.901 |

- **Medial OA**: selected run (★) **10**, 0.958 on the 58; best on the 58: run **27**, 0.977.
- **Lateral OA**: selected run (★) **22**, 0.880 on the 58; best on the 58: run **27**, 0.882.
- **PF OA**: selected run (★) **30**, 0.900 on the 58; best on the 58: run **26**, 0.916.
- **Resolution:** 224 px helps lateral OA most (runs 22 and 27).
- **PF OA:** the patellofemoral joint is classically read on axial images. Axial alone (run 29) and MaxViT at 224 px (run 26) give the best PF OA.
- **Medial OA** is between 0.95 and 0.98 for nearly every variant.

---

## 4. ACL and MCL

ROI crop models on the ligament boxes. The `ligaments` group is one model for both.

**ACL**

| Run | Group | Backbone | px | Head | Planes | Train labels | Ep. | ACL |
|---|---|---|---|---|---|---|---|---|
| 6 | ligaments | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.978 |
| 7 | acl | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.987 |
| 10 | all | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.973 |
| 37 | acl | convnext_tiny | 128 | lstm | axial | Claude | 10 | 0.961 |
| 38 | acl | convnext_tiny | 128 | lstm | sag+cor+axial | Claude | 10 | **0.991** |
| 39 | acl | convnext_tiny | 128 | lstm | sag+cor | mean | 10 | 0.969 ★ |
| 44 | acl | convnext_tiny | 128 | lstm | sag+cor | Codex | 10 | 0.967 |

- **ACL**: selected run (★) **39**, 0.969 on the 58; best on the 58: run **38**, 0.991.
- **Grouping:** separate models beat the shared `ligaments` model.
- **Axial:** adding axial to sagittal + coronal gives the best ACL (run 38). Axial alone is poor.

**MCL**

| Run | Group | Backbone | px | Head | Planes | Train labels | Ep. | MCL |
|---|---|---|---|---|---|---|---|---|
| 6 | ligaments | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.955 |
| 8 | mcl | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.982 |
| 10 | all | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.966 |
| 33 | mcl | convnext_tiny | 128 | lstm | axial | Claude | 10 | **0.986** |
| 34 | mcl | convnext_tiny | 128 | lstm | sag+cor+axial | Claude | 10 | 0.977 ★ |
| 40 | mcl | convnext_tiny | 128 | lstm | sag+cor | mean | 10 | 0.973 |
| 45 | mcl | convnext_tiny | 128 | lstm | sag+cor | Codex | 10 | 0.977 |

- **MCL**: selected run (★) **34**, 0.977 on the 58; best on the 58: run **33**, 0.986.
- **Every crop model is far above the baseline** (0.95–0.99 vs 0.753). There are only 9 MCL positives among the 58.
- **Axial:** it matters for MCL too (runs 33 and 34).

---

## Baker's cyst (for completeness)

| Run | Group | Backbone | px | Head | Planes | Train labels | Ep. | Baker's |
|---|---|---|---|---|---|---|---|---|
| 9 | bakers | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.935 |
| 10 | all | convnext_tiny | 128 | lstm | sag+cor | Claude | 10 | 0.964 |
| 31 | bakers | convnext_tiny | 128 | lstm | axial | Claude | 10 | 0.978 |
| 32 | bakers | convnext_tiny | 128 | lstm | sag+cor+axial | Claude | 10 | 0.955 |
| 43 | bakers | convnext_tiny | 128 | lstm | sag+cor | mean | 10 | 0.986 ★ |
| 48 | bakers | convnext_tiny | 128 | lstm | sag+cor | Codex | 10 | **0.991** |

- **Baker's**: selected run (★) **43**, 0.986 on the 58; best on the 58: run **48**, 0.991.
- **Training labels:** the mean and Codex labels clearly help Baker's (0.986 / 0.991, vs 0.935).

---

## 5. b1_c1 (`input/b1_c1_sealed_58_predictions.csv`)

Predictions for the 58 studies. How they were produced is not documented here.

- Macro: **0.936**.
- Per-finding scores are in the comparison below.

---

## 6. Pure ROI models vs b1_c1

**Pure ROI.** For each ROI finding, the prediction of the ★ crop model alone, with no other model involved. Predictions: `output/roi_pure_val_preds.csv`.

| Finding | Selected run | Configuration | 58 |
|---|---|---|---|
| ACL | `crops_run/2026-10-09/39` | acl, convnext_tiny, 128 px, sag+cor, mean labels | 0.969 |
| MCL | `crops_run/2026-10-09/34` | mcl, convnext_tiny, 128 px, sag+cor+axial, Claude labels | 0.977 |
| Medial Meniscus | `crops_run/2026-10-09/20` | menisci, coatnet_0, 224 px, sag+cor, Claude labels | 0.962 |
| Lateral Meniscus | `crops_run/2026-10-09/10` | all, convnext_tiny, 128 px, sag+cor, Claude labels | 0.889 |
| Medial OA | `crops_run/2026-10-09/10` | all, convnext_tiny, 128 px, sag+cor, Claude labels | 0.958 |
| Lateral OA | `crops_run/2026-10-09/22` | oa, convnext_tiny, 224 px, sag+cor, Claude labels | 0.880 |
| PF OA | `crops_run/2026-10-09/30` | oa, convnext_tiny, 128 px, sag+cor+axial, Claude labels | 0.900 |
| Baker's | `crops_run/2026-10-09/43` | bakers, convnext_tiny, 128 px, sag+cor, mean labels | 0.986 |

**Comparison on the 8 ROI findings:**

| Model | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | **Macro (8 ROI)** |
|---|---|---|---|---|---|---|---|---|---|
| b1_c1 | **0.977** | 0.964 | 0.966 | 0.865 | **0.986** | 0.876 | 0.870 | 0.976 | 0.935 |
| **Pure ROI** | 0.969 | 0.977 | 0.962 | 0.889 | 0.958 | 0.880 | **0.900** | **0.986** | 0.940 |
| ROI + b1_c1 (rank average, 50/50) | 0.974 | **0.983** | **0.969** | **0.890** | 0.974 | **0.890** | 0.890 | 0.983 | **0.944** |

| | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's |
|---|---|---|---|---|---|---|---|---|
| Pure ROI − b1_c1 | -0.007 | +0.014 | -0.004 | +0.025 | -0.028 | +0.004 | +0.030 | +0.009 |

- **Pure ROI vs b1_c1: 0.940 vs 0.935.**
  - Pure ROI wins on PF OA (+0.030), the lateral meniscus (+0.024), MCL (+0.013), Baker's (+0.010) and lateral OA.
  - b1_c1 wins on medial OA (−0.028), ACL and the medial meniscus.
- **Blending with b1_c1 helps both: 0.944** (rank average, 50/50).
  - The weight is fixed in advance: b1_c1 only covers the 58, so it cannot be tuned elsewhere.
  - Over the weights below, the blend is never worse than either model alone, so the gain does not depend on the weight.

| ROI weight in the rank average | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| 8-ROI macro on the 58 | 0.935 | 0.940 | 0.944 | 0.943 | 0.940 |

**All 12 findings.** The 4 findings without an ROI (effusion, synovitis, contusion, fracture) have no ROI model, so they take b1_c1's prediction:

| Model | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | Effusion | Synovitis | Contusion | Fracture | **Macro (12)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1_c1 | **0.977** | 0.964 | 0.966 | 0.865 | **0.986** | 0.876 | 0.870 | 0.976 | **0.989** | **0.851** | **0.955** | **0.953** | 0.936 |
| ROI + b1_c1 (rank average, 50/50) | 0.974 | **0.983** | **0.969** | **0.890** | 0.974 | **0.890** | 0.890 | 0.983 | **0.989** | **0.851** | **0.955** | **0.953** | **0.942** |

- **ROI + b1_c1 reaches 0.942 on all 12 findings, vs 0.936 for b1_c1 alone.**

**For reference only: the best single run per finding on the 58.** This is chosen on the 58, so it is optimistic and not used above:

| Finding | Best run on the 58 | 58 |
|---|---|---|
| ACL | `crops_run/2026-10-09/38` | 0.991 |
| MCL | `crops_run/2026-10-09/33` | 0.986 |
| Medial Meniscus | `crops_run/2026-10-09/2` | 0.971 |
| Lateral Meniscus | `crops_run/2026-10-09/17` | 0.923 |
| Medial OA | `crops_run/2026-10-09/27` | 0.977 |
| Lateral OA | `crops_run/2026-10-09/27` | 0.882 |
| PF OA | `crops_run/2026-10-09/26` | 0.916 |
| Baker's | `crops_run/2026-10-09/48` | 0.991 |

## 7. Pseudo-labels

**Method.**
- **Teachers:** for each ROI finding, the 5 runs with the best holdout score (the ★ selection rule).
- **Teacher predictions:** each teacher predicts every study (`python -m crops.predict`). On the training studies these are in-sample predictions, so this round works partly as self-distillation.
- **Round 1 targets:** the average of the 5 teachers' probabilities, per study and finding (`output/crops/pl_round1.csv`).
- **Round 2 targets:** 0.5 × the round-1 pseudo-labels + 0.5 × the mean of the Claude and Codex labels (`output/crops/pl_round2.csv`). The 50/50 weight was fixed in advance.
- **Retraining:** each ★ configuration, with exactly its original arguments and only the labels changed (`main_crops.py --labels pl:<name>`). Run numbers are in `logs/crops_run/2026-10-09/`; run 10 and its retrains (52, 59) cover both the lateral meniscus and medial OA.

| Finding | Teachers (top 5 on the holdout) | Original ★ | Round 1 | Round 2 |
|---|---|---|---|---|
| ACL | 39, 6, 44, 10, 7 | 39 | 49 | 56 |
| MCL | 34, 10, 8, 45, 40 | 34 | 50 | 57 |
| Medial Meniscus | 20, 15, 41, 19, 10 | 20 | 51 | 58 |
| Lateral Meniscus | 10, 15, 46, 19, 41 | 10 | 52 | 59 |
| Medial OA | 10, 42, 27, 47, 30 | 10 | 52 | 59 |
| Lateral OA | 22, 26, 30, 27, 28 | 22 | 53 | 60 |
| PF OA | 30, 42, 23, 47, 22 | 30 | 54 | 61 |
| Baker's | 43, 32, 48, 9, 31 | 43 | 55 | 62 |

**The training targets themselves, scored against the radiologists** (52 studies: Codex has no labels for 6 of the 58):

| Target | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | **Macro (8 ROI)** |
|---|---|---|---|---|---|---|---|---|---|
| Pseudo-labels (round 1) | 0.972 | 0.974 | 0.948 | 0.908 | **0.962** | 0.842 | 0.861 | **0.988** | 0.932 |
| Claude labels | **0.992** | 0.976 | 0.959 | 0.874 | 0.941 | **0.860** | **0.939** | 0.899 | 0.930 |
| Codex labels | 0.965 | 0.976 | **0.986** | 0.850 | 0.943 | 0.820 | 0.902 | **0.988** | 0.929 |
| Mean of Claude + Codex | 0.991 | 0.980 | 0.979 | 0.865 | 0.944 | 0.849 | 0.933 | 0.975 | 0.940 |
| 0.25 pseudo-labels + 0.75 mean labels | **0.992** | 0.986 | 0.975 | 0.898 | 0.950 | 0.837 | 0.931 | 0.981 | 0.944 |
| **0.5 pseudo-labels + 0.5 mean labels (round-2 targets)** | 0.991 | **0.991** | 0.970 | **0.911** | 0.960 | 0.845 | 0.929 | **0.988** | **0.948** |
| 0.75 pseudo-labels + 0.25 mean labels | 0.985 | **0.991** | 0.967 | **0.911** | **0.962** | 0.842 | 0.904 | **0.988** | 0.944 |

- **The pseudo-labels and the report labels err differently.**
  - The pseudo-labels win on the lateral meniscus, medial OA and Baker's.
  - The report labels win on PF OA (0.939 Claude vs 0.861), ACL and the medial meniscus.
- **Mixing them beats every single source at all three weights.** The a-priori 50/50 mix scores 0.948, vs 0.939 for the mean of the report labels and 0.932 for the pseudo-labels.

**Retrained models, 58 radiologist studies:**

| Model | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | **Macro (8 ROI)** |
|---|---|---|---|---|---|---|---|---|---|
| Original ★ runs | 0.969 | 0.977 | 0.962 | 0.889 | 0.958 | 0.880 | **0.900** | 0.986 | 0.940 |
| Round 1: retrained on pseudo-labels | **0.977** | 0.989 | **0.974** | 0.902 | 0.969 | 0.867 | 0.896 | 0.984 | 0.945 |
| **Round 2: retrained on 0.5 pseudo-labels + 0.5 mean labels** | 0.969 | **0.991** | 0.969 | **0.925** | 0.967 | 0.872 | 0.896 | 0.984 | 0.947 |
| 5-teacher average (the round-1 pseudo-labels) | 0.974 | 0.977 | 0.952 | 0.916 | 0.971 | 0.863 | 0.873 | **0.987** | 0.939 |
| Round 2 + b1_c1 (rank average, 50/50) | 0.972 | 0.990 | 0.972 | 0.904 | 0.981 | **0.886** | 0.889 | 0.985 | **0.947** |
| b1_c1 | **0.977** | 0.964 | 0.966 | 0.865 | **0.986** | 0.876 | 0.870 | 0.976 | 0.935 |

- **Round 2 is the best single-model result: 0.947**, vs 0.940 for the original ★ runs, 0.944 for round 1 and 0.935 for b1_c1.
  - The lateral meniscus gains most: 0.889 → 0.925.
  - MCL reaches 0.991.
  - Lateral OA stays slightly below the original.
- **The students beat their teachers.** The round-1 models (single models) score above the 5-teacher average they were trained on (0.944 vs 0.939).
- **Blending with b1_c1 no longer adds on the ROI findings:** 0.947, the same as round 2 alone. It helps medial and lateral OA, where b1_c1 is strong, and costs the lateral meniscus.
- **All 12 findings,** with b1_c1 on the 4 findings without an ROI: round 2 + b1_c1 scores **0.944**, vs 0.936 for b1_c1 alone.
- **Noise:** the gains are within the ~±0.01–0.02 noise of 58 studies, but their direction is consistent across both rounds and most findings.

## 8. Through-plane frame sampling

All runs use run 59's configuration: `all` group, ConvNeXt-tiny 128 px, sagittal + coronal, PL round-2 labels, seed 0, 10 epochs. Only the slices that make up the crop change (`crops.prepare --depth --z_mode --stride`):

- **uniform** (used so far): n frames spread evenly over the box's slices widened by one slice each side, nearest slice. The median box spans ~9 slices including that margin, so 12 frames repeat slices.
- **strided**: n real slices, `stride` apart, centred on the box centre, whatever the box's extent.
- **resize**: the uniform positions, interpolated linearly between slices.

Mean AUC over the 8 ROI findings on the 58 radiologist studies. ★ marks the holdout pick.

| Run | Sampling | Frames | Stride | ACL | MCL | Medial Meniscus | Lateral Meniscus | Medial OA | Lateral OA | PF OA | Baker's | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `59` (reference) | uniform | 12 | – | 0.971 | 0.977 | 0.978 | 0.925 | 0.967 | 0.818 | 0.874 | 0.987 | 0.937 |
| `70` (reference, seed 1) | uniform | 12 | – | 0.968 | 0.977 | 0.962 | 0.906 | **0.971** | 0.851 | 0.860 | 0.989 | 0.935 |
| `71` | uniform | 8 | – | 0.973 | 0.984 | **0.990** | 0.914 | 0.969 | 0.816 | 0.870 | 0.989 | 0.938 |
| `74` | uniform | 16 | – | 0.972 | 0.975 | 0.975 | 0.922 | 0.967 | 0.839 | 0.875 | 0.991 | 0.940 |
| `75` | strided | 8 | 1 | 0.973 | 0.984 | 0.976 | 0.935 | 0.969 | **0.859** | 0.880 | **0.993** | **0.946** |
| `72` ★ | strided | 12 | 1 | 0.971 | 0.984 | 0.977 | 0.908 | 0.969 | **0.859** | 0.883 | 0.984 | 0.942 |
| `77` | strided | 16 | 1 | 0.972 | **1.000** | 0.972 | 0.918 | 0.966 | 0.816 | 0.888 | 0.984 | 0.940 |
| `76` | strided | 8 | 2 | **0.974** | 0.973 | 0.976 | 0.884 | 0.961 | 0.803 | **0.892** | 0.967 | 0.929 |
| `73` | resize | 12 | – | 0.971 | 0.973 | 0.971 | **0.944** | 0.967 | 0.836 | 0.874 | 0.989 | 0.941 |
| `78` | resize | 16 | – | 0.971 | 0.991 | 0.966 | 0.940 | 0.967 | 0.845 | 0.878 | 0.991 | 0.944 |

- **Noise:** the two seeds of the reference differ by 0.002 on the mean. Single findings (Lateral OA, Lateral Meniscus) move by up to 0.03 between seeds.
- **Stride 1, 8–12 frames is best:** 0.946 and 0.942, against 0.935–0.937 for the reference. The holdout prefers 12 frames, 8 frames is level with the reference there. Spending the frames on adjacent real slices helps more than covering the box exactly.
- **Stride 2 is the only clear loss** (0.929): 8 frames spread over ~16 slices take in too much outside the ROI.
- **Frame count matters less than sampling.** Uniform 8 and 16 frames and resize 12 sit within noise of the reference. Resize 16 is up on the 58 (0.944) but level on the holdout.

### 8b. Stride 1, 8 frames on every submission model

The round-2 runs behind the submission recipe, retrained on stride-1 8-frame crops, everything else unchanged. Axial crops for the MCL and the axial OA model use the same sampling. AUC on the 58 radiologist studies:

| Model (12 → 8 frames) | Finding | 12 frames | 8 frames |
|---|---|---|---|
| acl `56` → `79` | ACL | 0.969 | **0.973** |
| mcl `57` → `80` | MCL | 0.991 | **1.000** |
| menisci `58` → `81` | Medial Meniscus | 0.969 | **0.977** |
| | Lateral Meniscus | 0.892 | **0.913** |
| all `59` → `75` | 8 findings, mean | 0.937 | **0.946** |
| oa 224 `60` → `82` | Medial OA | **0.981** | 0.975 |
| | Lateral OA | **0.872** | 0.868 |
| | PF OA | 0.889 | **0.894** |
| oa + axial `61` → `83` | Medial OA | 0.967 | **0.972** |
| | Lateral OA | 0.834 | **0.863** |
| | PF OA | 0.896 | **0.906** |
| bakers `62` → `84` | Baker's | 0.984 | **0.989** |

Submission recipe (C except PF OA and Lateral OA, rank average), same model slots:

| | ACL | MCL | Medial Meniscus | Lateral Meniscus | Medial OA | Lateral OA | PF OA | Baker's | Mean |
|---|---|---|---|---|---|---|---|---|---|
| 12 frames (56–62) | 0.972 | 0.994 | 0.975 | 0.921 | **0.974** | **0.872** | 0.896 | 0.986 | 0.949 |
| 8 frames (79–84, 75) | **0.973** | **0.996** | **0.978** | **0.938** | **0.974** | 0.869 | **0.906** | **0.991** | **0.953** |

- **Wins:** 8 frames scores higher on 17 of 19 model × finding pairs (mean +0.008), and on the recipe (+0.004).
- **Holdout:** flat (mean −0.001), so the 8-frame models are at least as good there.
- **Speed:** 8 frames is a third fewer frames per crop than 12, so the crop models run about 33% faster at inference.

## 9. Slice aggregation

How the 8 slices of a crop are combined. All runs use run 75's setup: `all` group, ConvNeXt-tiny, stride-1 8-frame crops, PL round-2 labels. GFLOP is per crop.

- **Late fusion** (the default) runs each slice, with its two neighbours as RGB, through the 2D encoder on its own, then combines the slice vectors.
- **Temporal shift** moves 1/8 of the channels from the previous slice and 1/8 from the next, before the depthwise conv of every block in the last two stages. Slices exchange information inside the encoder at no FLOP cost, for ~25% more latency from the memory shuffles.
- **map3d** keeps the last feature maps of all slices as a (C, D, h, w) volume, applies two residual 3×3×3 convs, then pools. Space is not pooled before the slices are combined.
- **Early fusion** stacks the 8 slices as the input channels of a single 2D image.

AUC on the 58 radiologist studies. ★ marks the holdout pick.

| Variant | Run | GFLOP | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LSTM + attention pooling (reference) ★ | `09/75` | 23.3 | 0.973 | **0.984** | 0.976 | 0.935 | 0.969 | 0.859 | 0.880 | **0.993** | 0.946 |
| attention pooling only | `09/92` | 23.3 | 0.974 | 0.977 | 0.972 | 0.907 | 0.971 | 0.845 | 0.876 | 0.989 | 0.939 |
| transformer (2 layers) | `09/93` | 23.3 | 0.968 | 0.973 | 0.972 | **0.942** | 0.974 | 0.872 | **0.892** | 0.986 | **0.947** |
| map3d: 3D convs on the last feature maps | `09/94` | 24.2 | **0.990** | 0.977 | 0.981 | 0.906 | 0.977 | **0.874** | 0.870 | 0.989 | 0.946 |
| temporal shift + LSTM | `09/95` | 23.3 | 0.968 | 0.982 | 0.972 | 0.927 | 0.972 | 0.853 | 0.874 | 0.989 | 0.942 |
| temporal shift + map3d | `09/96` | 24.2 | 0.985 | 0.977 | **0.988** | 0.893 | 0.971 | 0.863 | 0.879 | 0.991 | 0.943 |
| early fusion (8 slices as channels) | `09/97` | 2.9 | 0.980 | 0.959 | 0.964 | 0.887 | **0.978** | 0.861 | 0.861 | 0.982 | 0.934 |

- **No aggregation beats the BiLSTM.** The transformer (0.947), map3d (0.946) and temporal shift (0.942) are within the ~0.002–0.005 seed noise, both on the 58 and on the holdout.
- **Mixing slices earlier does not help here.** The 2.5D input (neighbour slices as RGB) and 8 adjacent slices already carry the local 3D context.
- **Early fusion is 8× cheaper but costs 0.012** on the 58 (−0.014 on the holdout).
- **Ensembles:** LSTM + transformer + map3d reach 0.949 as a rank average, within noise of the single LSTM model.

## 10. Backbones

Same setup (run 75, LSTM head). Small ImageNet backbones from timm, chosen from current small-model comparisons and common Kaggle medical-imaging choices. The ConvNeXt family uses lr_backbone 1e-4; the mobile CNNs (EfficientNet, MobileNetV4, HGNet, RegNet, ResNet) use 3e-4 and EdgeNeXt 2e-4. EfficientViT-B1 could not run (no drop-path support in timm). Params are the encoder's; GFLOP is per 8-frame 128 px crop.

| Variant | Run | params | GFLOP | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny.fb_in22k (reference) | `09/75` | 27.8M | 23.3 | 0.973 | 0.984 | 0.976 | 0.935 | 0.969 | 0.859 | 0.880 | 0.993 | 0.946 |
| caformer_s18.sail_in22k_ft_in1k | `10/5` | 23.2M | 20.6 | 0.977 | **0.991** | 0.966 | **0.937** | **0.974** | 0.868 | 0.887 | 0.991 | **0.949** |
| hgnetv2_b2.ssld_stage2_ft_in1k | `10/6` | 9.2M | 6.0 | 0.987 | 0.984 | 0.977 | 0.918 | 0.967 | 0.839 | **0.900** | 0.993 | 0.946 |
| convnext_nano.in12k_ft_in1k | `09/98` | 15.0M | 12.8 | **0.990** | 0.982 | 0.970 | 0.925 | 0.966 | 0.845 | 0.876 | 0.991 | 0.943 |
| edgenext_small.usi_in1k | `10/4` | 5.3M | 5.0 | 0.972 | 0.982 | 0.988 | 0.911 | 0.964 | 0.857 | 0.866 | **0.995** | 0.942 |
| resnet34d.ra2_in1k | `10/7` | 21.3M | 20.4 | 0.978 | 0.982 | 0.980 | 0.909 | 0.971 | 0.832 | 0.884 | 0.991 | 0.941 |
| tf_efficientnetv2_b3.in21k_ft_in1k ★ | `10/0` | 12.8M | 8.4 | 0.982 | 0.982 | 0.971 | 0.897 | 0.971 | 0.849 | 0.885 | 0.989 | 0.941 |
| convnextv2_nano.fcmae_ft_in22k_in1k | `09/99` | 15.0M | 12.8 | 0.975 | 0.977 | 0.975 | 0.899 | 0.967 | **0.892** | 0.852 | 0.986 | 0.940 |
| regnety_016.tv2_in1k | `10/8` | 10.3M | 8.4 | 0.979 | 0.971 | 0.982 | 0.912 | 0.972 | 0.845 | 0.866 | 0.989 | 0.940 |
| mobilenetv4_conv_medium.e500_r256_in1k | `10/1` | 8.4M | 4.3 | 0.974 | 0.975 | **0.992** | 0.903 | 0.967 | 0.826 | 0.885 | 0.986 | 0.939 |
| mobilenetv4_hybrid_medium.ix_e550_r256_in1k | `10/2` | 9.8M | 5.0 | 0.984 | 0.968 | 0.981 | 0.847 | 0.964 | 0.843 | 0.888 | 0.989 | 0.933 |
| efficientnet_b3.ra2_in1k | `09/100` | 10.7M | 5.0 | 0.982 | 0.977 | 0.958 | 0.882 | 0.960 | 0.838 | 0.865 | 0.993 | 0.932 |

- **ConvNeXt-tiny stays the reference.** Only CAFormer-S18 scores higher on the 58 (0.949), within noise and level on the holdout.
- **HGNetV2-B2 is the efficient option:** 0.946 with a third of the parameters and a quarter of the FLOPs.
- **The mobile CNNs trail by 0.005–0.015:** EfficientNet-B3, MobileNetV4 and RegNetY.
- **Ensembles across backbones add little:** ConvNeXt-tiny + CAFormer 0.950, + HGNetV2 0.948, + ConvNeXt-nano 0.948.

## 11. Effusion and synovitis: capsule crops

Effusion and synovitis have no box of their own, but both sit inside the joint capsule. The eight ROIs (menisci, the three compartments, the ligaments, the patellofemoral joint, Baker's cyst) outline the capsule, so no new segmentation model is needed.

**Capsule box.** Per sagittal or coronal series, the capsule box is the union of the 8 ROI boxes, widened in-plane by a fraction of its height and width on each side (`crops.prepare --capsule`).
- The union already spans ~87% of the slices and ~2/3 of the image in-plane.
- On effusion-positive studies, the bright suprapatellar fluid above the patella often reaches past the plain union on sagittal series, so expansion was the main thing to validate.
- Superior-only extra expansion was tested too.

**Crops.** One square crop per series around the box, 160 px, with uniform frames over the box's slices. The box covers most slices, so 8 strided frames as for the small ROIs would not cover the capsule.

**Models.** One `capsule` group: effusion and synovitis share the crop and have separate finding embeddings. Trained on the Claude + Codex label mean (the pseudo-labels cover only the 8 ROI findings), sagittal + coronal series, 10 epochs.

The 3D models use the same box, 32 slices interpolated over it at 128 px (the ROI U-Net's input size), and the U-Net's 6-stage residual encoder with average + max pooling. It starts either from the ROI U-Net weights, which were trained on these knee volumes, or from scratch.

AUC on the 58 radiologist studies. ★ marks the holdout pick.

| Run | Capsule box | Frames | Model | Train labels | Effusion | Synovitis | Mean |
|---|---|---|---|---|---|---|---|
| `10/10` | +0% | 16 | 2.5D ConvNeXt-tiny | mean | 0.976 | 0.824 | **0.900** |
| `10/9` | +15% | 16 | 2.5D ConvNeXt-tiny | mean | 0.978 | 0.797 | 0.887 |
| `10/11` | +30% | 16 | 2.5D ConvNeXt-tiny | mean | 0.969 | 0.783 | 0.876 |
| `10/12` | +10%, +30% superior | 16 | 2.5D ConvNeXt-tiny | mean | 0.970 | 0.803 | 0.887 |
| `10/13` | +15% | 8 | 2.5D ConvNeXt-tiny | mean | 0.978 | 0.804 | 0.891 |
| `10/14` ★ | +15% | 24 | 2.5D ConvNeXt-tiny | mean | 0.980 | 0.784 | 0.882 |
| `10/22` | +15% | 24 | 2.5D ConvNeXt-tiny, seed 1 | mean | 0.969 | 0.786 | 0.878 |
| `10/25` | +15% (+ axial capsule boxes) | 24 | 2.5D ConvNeXt-tiny, sagittal + coronal + axial | mean | **0.985** | 0.796 | 0.890 |
| `10/15` | +15% | 32 (3D) | 3D, ROI U-Net encoder (pretrained) | mean | 0.955 | 0.821 | 0.888 |
| `10/16` | +15% | 32 (3D) | 3D, same encoder from scratch | mean | 0.940 | **0.845** | 0.893 |
| `10/17` | +15% | 16 | 2.5D, map3d head | mean | 0.949 | 0.810 | 0.880 |
| `10/18` | +15% | 16 | 2.5D, temporal shift | mean | 0.975 | 0.791 | 0.883 |
| `10/19` | +15% | 16 | 2.5D, early fusion | mean | 0.924 | 0.806 | 0.865 |
| `10/20` | +15% | 24 | 2.5D ConvNeXt-tiny | PL (capsule) | 0.975 | 0.772 | 0.873 |
| `10/21` | +15% | 32 (3D) | 3D from scratch | PL (capsule) | 0.952 | 0.789 | 0.870 |

References on the 58:

| | Effusion | Synovitis |
|---|---|---|
| b1_c1 | 0.989 | 0.851 |
| Baseline sequence classifier, 5-run ensemble | 0.959 | 0.791 |
| Claude labels | 0.869 | 0.824 |
| Codex labels (52 studies) | 0.920 | 0.742 |

- **Effusion works well from the capsule crop:** 0.97–0.98. That is above our whole-series baseline (0.959) and above the labels it was trained on (0.869 / 0.920), but below b1_c1 (0.989).
- **Synovitis stays hard:** 0.78–0.82 for the 2.5D models, around the baseline's 0.791, and below b1_c1's 0.851. With 27 positives among 58 studies, the spread between configurations is mostly noise. The two seeds of the selected configuration score 0.784 and 0.786.
- **Expansion:** +15% or more is needed on the holdout (+0%: 0.935, +15%: 0.940, +30%: 0.940, superior +30%: 0.941). On the 58 the tight box scores best on synovitis (0.824), which is consistent with noise. +15% is kept: it contains the suprapatellar recess and is best on effusion.
- **Frames:** more frames help a little now that the ROI is large. On the holdout, 8 frames score 0.936, 16 score 0.940 and 24 score 0.944.
- **3D models are not better here.** The 3D encoder is level or better on synovitis on the 58 (0.82–0.85) but worse on effusion (0.94–0.955) and clearly worse on the holdout (0.917).
  - Pretraining on the ROI task does not help: the pretrained and from-scratch encoders score the same.
  - Rank-averaging 2.5D + 3D lifts synovitis on the 58 (0.83) but lowers the holdout (0.937 vs 0.943), so it is not selected.
- **Aggregation:** as for the ROIs, the LSTM is as good as anything else. The temporal shift ties, while map3d and early fusion lose.
- **Axial series:** capsule boxes on axial series come from the union of the registered axial ROI boxes. Adding them scores 0.890 on the 58 (effusion 0.985, synovitis 0.796), vs 0.882 without, but 0.941 on the holdout vs 0.944. That is within seed noise, so it is not selected. It is the first option to retry, since axial series show the suprapatellar and patellar recesses well.
- **Pseudo-labels don't help here** (0.874 vs 0.882). The teacher mean agrees better with the radiologists than the label mean (effusion 0.973 vs 0.916), but the student trained on it does not improve.
- **Selected:** run `10/14`, +15%, 24 frames, 2.5D ConvNeXt-tiny, label mean. The submission uses fullfit versions of both seeds, rank-averaged.
  - Through the inference pipeline with the non-fullfit seeds: effusion 0.975, synovitis 0.786.
  - The 12-finding macro rises from 0.799 to **0.865**, with contusion and fracture still at 0.5.

## 12. Contusion and fracture from the joint crop

Bone bruises and fractures around the knee also lie inside the capsule box, which covers the femoral condyles, tibial plateau and patella. The same crop (+15%, 24 frames, 160 px, 2.5D ConvNeXt-tiny, Claude + Codex label mean) was trained on all four non-ROI findings at once (`joint4` group), with and without axial capsule crops.

| Run | Planes | Effusion | Synovitis | Contusion | Fracture |
|---|---|---|---|---|---|
| `10/14` (capsule: effusion + synovitis only, §11 ★) | sag + cor | 0.980 | 0.784 | – | – |
| `10/26` joint4 | sag + cor | 0.976 | **0.804** | 0.931 | 0.936 |
| `10/27` joint4 ★ | sag + cor + axial | **0.991** | 0.781 | **0.942** | **0.946** |
| b1_c1 | | 0.989 | 0.851 | 0.955 | 0.953 |

- **Contusion and fracture read well from the joint crop:** 0.94–0.95, close to b1_c1. The axial version (★ on the holdout: contusion 0.960, fracture 0.953) is better on both findings, on the 58 and on the holdout.
- **Sharing hurts synovitis on the holdout:** 0.930 in the shared model vs 0.946 for the capsule-only model. Effusion and synovitis therefore stay with the §11 capsule models, and contusion and fracture come from `10/27`.

**All 12 findings, pure ROI models**, through the inference pipeline on the 58 (non-fullfit models):
- 8 ROI findings: 8-frame recipe.
- Effusion and synovitis: capsule seeds `10/14` + `10/22`.
- Contusion and fracture: `10/27`.

| | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture | **Macro (12)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1_c1 | **0.977** | 0.964 | 0.966 | 0.865 | **0.986** | **0.876** | 0.870 | **0.989** | **0.851** | 0.976 | **0.955** | **0.953** | 0.936 |
| **Pure ROI** | 0.972 | **0.994** | **0.977** | **0.934** | 0.974 | 0.868 | **0.905** | 0.975 | 0.786 | **0.991** | 0.942 | 0.947 | **0.939** |

## 13. Study-level models: learned aggregation over series

The series-level models are scored per study by averaging their series' probabilities (§ series → study analysis: max, top-k, plane / sequence / box-confidence weighting, all within noise of the mean). Here the averaging is replaced by a learned layer, trained end to end (`main_study.py`, `crops/study.py`):

- **What a sample is:** one study, with every (series, ROI) crop of the group's findings. The crop encoder is the series model: 2.5D ConvNeXt + BiLSTM over slices, which gives one vector per crop.
- **The set to aggregate:** for each (study, finding), the vectors of that study's series. The number of series varies, so the set is padded and masked.
- **Aggregators:**
  - mean pooling (the control);
  - gated attention pooling (MIL), which starts as the mean;
  - 2 transformer layers, then attention pooling;
  - log-sum-exp pooling of the per-series logits, with a learned temperature between mean and max.
- **Training:**
  - initialised from the series-level run, 4 epochs;
  - learning rates: encoder 2e-5, series head 1e-4, aggregator 5e-4;
  - each series dropped with probability 0.25 (at least one kept);
  - at most 8 series per study;
  - the series run's labels.
- **Validation:** every series kept, one prediction per (study, finding).

AUC on the 58; holdout = mean over the findings; ★ = holdout pick.

**`all` model (8 ROI findings), from `09/75`**

| Aggregation | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | Mean (58) | Holdout |
|---|---|---|---|---|---|---|---|---|---|---|
| series level (`09/75`, mean of series probabilities) | **0.973** | **0.984** | **0.976** | 0.935 | 0.969 | **0.859** | 0.880 | **0.993** | 0.946 | 0.942 |
| study, mean pooling | **0.973** | **0.984** | 0.964 | 0.939 | 0.974 | 0.850 | 0.889 | **0.993** | 0.946 | 0.943 |
| study, gated attention | **0.973** | **0.984** | 0.973 | **0.945** | 0.972 | 0.856 | **0.891** | 0.984 | **0.947** | 0.943 |
| study, gated attention, no series dropout | **0.973** | **0.984** | 0.969 | 0.940 | 0.971 | 0.853 | 0.889 | 0.986 | 0.946 | 0.944 |
| study, transformer + attention ★ | **0.973** | **0.984** | 0.966 | 0.929 | 0.972 | **0.859** | 0.889 | 0.986 | 0.945 | 0.944 |
| study, LSE (learned r = 0.37) | **0.973** | **0.984** | 0.970 | 0.939 | **0.975** | 0.845 | 0.891 | 0.991 | 0.946 | 0.943 |

**Capsule model (effusion, synovitis), from `10/14`**

| Aggregation | Effusion | Synovitis | Mean (58) | Holdout |
|---|---|---|---|---|
| series level (`10/14`) | **0.980** | **0.784** | **0.882** | 0.943 |
| study, mean pooling | 0.978 | 0.782 | 0.880 | 0.944 |
| study, gated attention | 0.979 | 0.780 | 0.880 | 0.944 |
| study, transformer + attention | 0.975 | 0.766 | 0.870 | 0.944 |
| study, LSE ★ | 0.974 | 0.781 | 0.878 | 0.944 |

**Joint model with axial (4 non-ROI findings), from `10/27`**

| Aggregation | Effusion | Synovitis | Contusion | Fracture | Mean (58) | Holdout |
|---|---|---|---|---|---|---|
| series level (`10/27`) | 0.991 | 0.781 | 0.942 | **0.946** | **0.915** | 0.947 |
| study, mean pooling ★ | **0.996** | 0.774 | **0.943** | 0.942 | 0.914 | 0.949 |
| study, gated attention | **0.996** | **0.787** | 0.931 | 0.939 | 0.913 | 0.947 |
| study, transformer + attention | 0.990 | 0.779 | 0.937 | 0.935 | 0.910 | 0.947 |

- **No learned aggregation beats averaging.**
  - Every variant lands within ±0.002 (`all`), ±0.012 (capsule) and ±0.005 (joint) of the series-level model on the 58.
  - On the holdout they score 0.943–0.949, against 0.942–0.947 at series level.
  - The mean-pooling control scores the same, so the 4 extra epochs at study level explain the small holdout drift, not the aggregation.
- **The aggregators learn to average.**
  - The attention weights stay almost uniform: the largest series weight in a study is typically only 1.04–1.19 × the uniform weight (90th percentile 1.17–1.58).
  - The only plane preference is a mild upweighting of axial series in the joint model (×1.12–1.31).
  - LSE settles at r = 0.37, close to a mean of logits; it does not move towards the max.
- **Why:** the series models are trained with the *study* label on every series. Each series therefore already learns to predict the study's label from its own view, as far as that view allows, including "not visible here, but likely given the rest". There is little left for a set-level layer to fix. Anomalies visible on one view only exist, but they are not frequent or consistent enough in these labels to teach a re-weighting.
- **Not adopted.** The series-level models with averaging stay. To make the aggregation matter, series-level training would need series-level labels (which view shows the finding), which we don't have.

## 14. Medial and lateral OA: what was wrong

**Symptom.** On the 58, medial OA scored 0.974 and lateral OA 0.868, against b1_c1's 0.986 and 0.876. Almost all errors are false positives. Lateral OA is ranked high in studies with medial OA but no lateral OA, and vice versa:

| | MOA–LOA correlation |
|---|---|
| Radiologist labels (58) | 0.32 |
| Claude training labels | 0.44 |
| Our predictions (ranks) | 0.84 |
| b1_c1 predictions (ranks) | 0.81 |

Within studies without medial OA, lateral OA scores only 0.826. Both models score "how arthritic is the knee" more than "which compartment".

**Cause 1: the ROI crop crosses into the other compartment.**
- On coronal (and axial) series, left–right runs along the image columns.
- The crop is a square of 1.5 × the box's larger side, centred on the box.
- Lateral OA boxes are wide (0.36–0.39 of the image, vs 0.27–0.29 for medial), so the lateral crop is about half the image wide and reaches past the intercondylar notch into the medial compartment.
- The boxes themselves rarely overlap the other side: only 2% of coronal lateral-OA boxes overlap the medial meniscus box by more than 20%. It's the crop that crosses over.
- Sagittal crops are not affected: there left–right is the slice axis, and 8 strided slices stay within the box.

**Cause 2: the labels.** The training labels are correlated, and the pseudo-labels amplify it (MOA–LOA correlation: Claude 0.44, Claude + Codex mean 0.47, PL round 2 0.54, PL round 1 0.60).

**Fixes tried** in the OA-only model (`oa` group, ConvNeXt-tiny 128 px, sagittal + coronal, 8 strided frames, PL round 2 unless stated):
- **Midline crop** (`crops.prepare --oa_midline slide|clip`). The intercondylar midline is placed halfway between the medial and lateral menisci boxes' inner edges. Each medial / lateral OA crop window on coronal and axial series is shifted outward so it never crosses it, keeping its size and gaining periphery, where osteophytes are. `clip` also cuts the box at the midline.
- **Compartment box** (`--compartment f`): the OA box joined with the same side's meniscus box, extended by f × its height up and down into the femoral condyle and the tibial plateau.
- **Context and sampling:** wider margins, and 12 uniform frames over the box.
- **Labels:** Claude only, and the Claude + Codex mean.
- **Cross-compartment model** (`main_cross.py`): a transformer across the three compartment crops of a series, so the model can see which compartment is worse.

AUC on the 58:

| Crop / labels | Run | Medial OA | Lateral OA | PF OA | Holdout (3 OA) |
|---|---|---|---|---|---|
| Reference crop (8 strided frames, 25% margin) | `10/29` | 0.972 | 0.870 | 0.893 | 0.926 |
| 12 uniform frames | `10/30` | 0.972 | 0.853 | 0.876 | 0.925 |
| margin 0.5 | `10/31` | 0.980 | 0.849 | 0.874 | 0.923 |
| margin 0.75 | `10/32` | **0.984** | 0.855 | 0.867 | 0.915 |
| compartment box +30% (femur / tibia) | `10/33` | **0.984** | 0.865 | 0.888 | 0.923 |
| compartment box +50% | `10/34` | 0.980 | 0.843 | 0.896 | 0.921 |
| Claude labels (not PL) | `10/35` | 0.969 | 0.851 | 0.900 | 0.929 |
| Claude + Codex mean labels | `10/36` | 0.969 | 0.870 | 0.891 | 0.928 |
| **midline crop (slide)** | `10/37` | 0.971 | **0.884** | 0.889 | 0.927 |
| midline crop (clip box) | `10/38` | 0.972 | 0.880 | 0.893 | 0.927 |
| midline crop (slide), seed 1 | `10/39` | 0.977 | 0.870 | 0.894 | 0.926 |
| **compartment +30% + midline** | `10/40` | 0.978 | 0.882 | 0.889 | 0.928 |
| margin 0.5 + midline | `10/41` | 0.977 | 0.878 | 0.883 | 0.925 |
| compartment +30% + midline, 224 px | `10/42` | **0.984** | 0.870 | 0.893 | 0.928 |
| midline + axial (midline) crops | `10/43` | 0.974 | 0.874 | **0.903** | 0.927 |
| b1_c1 | | 0.986 | 0.876 | 0.870 | |

- **Wider crops help medial OA and hurt lateral OA** (margin 0.75: 0.985 / 0.855). That is the cross-over effect: a wider lateral crop sees more of the medial compartment.
- **The midline crop helps lateral OA:** 0.884 and 0.870 over two seeds, against 0.870 for the reference. Its prediction correlation drops a little (0.83 → 0.79–0.80).
- **Compartment box + midline** is the best all-round crop: 0.978 / 0.882 / 0.889, and the best holdout among the 128 px runs. At 224 px it reaches 0.985 on medial OA but drops lateral OA to 0.870.
- **Labels without PL** do not fix the cross-talk (Claude labels: lateral OA 0.851).
- **The cross-compartment model** does not help (0.978 / 0.855 / 0.893).
- **Holdout caveat:** for lateral OA the holdout scores against Claude's labels, so it favours Claude-trained models (holdout 0.914 for a model that scores 0.851 on the 58). It is a weak selection rule for this finding.

**Post-hoc cross-talk correction.** Lateral OA score = rank(lateral OA) − β × rank(medial OA), with β chosen on the holdout.
- On the 7 midline-crop runs averaged, the holdout picks β = 0.4 (lateral OA holdout 0.907 → 0.923).
- On the 58 it raises lateral OA from 0.876 to **0.899**.
- The same correction does not help medial OA, which is the dominant compartment.

**Ensembles** (rank average per finding, AUC on the 58):

| | Medial OA | Lateral OA | PF OA | MOA + LOA |
|---|---|---|---|---|
| Current submission recipe | 0.974 | 0.868 | **0.906** | 0.921 |
| 7 midline-crop runs, averaged | 0.980 | 0.876 | 0.896 | 0.928 |
| Greedy selection on the holdout | 0.978 | 0.870 | 0.903 | 0.924 |
| **7 midline-crop runs + lateral OA correction (β = 0.4)** | 0.980 | **0.899** | 0.896 | **0.940** |
| b1_c1 | **0.986** | 0.876 | 0.870 | 0.931 |

Lateral OA now beats b1_c1. Medial OA is 0.006 short, about 4 misordered positive/negative pairs out of 645. More 224 px seeds are training.

## 15. ACL

ACL scored 0.972 on the 58 (b1_c1: 0.977) with 24 positives and 34 negatives (816 positive/negative pairs).

**The ACL crop.**
- On sagittal series the U-Net's "ACL" box mostly covers the posterior tibial plateau, not the ligament. The box was learned from attention maps that favoured the classic pivot-shift bone bruise there.
- Coronal-only crops (0.979) beat sagittal-only ones (0.967) in the series-level predictions.
- To test whether the ligament was being cropped out, an anatomical **notch box** was built from the two meniscus boxes (`crops.prepare --acl_notch`):
  - sagittal: the slices between the menisci;
  - coronal: the columns between their inner edges;
  - vertically: from half a meniscus height above the joint line, the notch roof, to a quarter below, the tibial spines.
  - Its crops show the notch and the ligament.

**Sweep** (ACL-only model, PL round 2, one change at a time). The holdout is saturated for ACL (0.98–0.995 for every variant) and cannot rank them, so the 58 is the main signal here. Selecting on it is optimistic.

| Variant | Run | ACL (58) | Holdout |
|---|---|---|---|
| Reference: ConvNeXt 128, sag + cor, 8 strided frames (`09/79`) | `09/79` | 0.973 | 0.992 |
| reference, seed 1 | `10/57` | 0.974 | 0.992 |
| reference, seed 2 | `10/58` | 0.972 | 0.993 |
| margin 0.5 | `10/59` | 0.972 | 0.993 |
| margin 0.75 | `10/60` | 0.960 | 0.993 |
| 12 strided frames | `10/61` | 0.974 | 0.993 |
| 16 strided frames | `10/62` | 0.974 | 0.993 |
| sagittal only | `10/63` | 0.967 | 0.992 |
| coronal only | `10/64` | 0.973 | 0.981 |
| **+ axial crops** | `10/65` | **0.989** | 0.993 |
| + axial crops, seed 1 | `10/76` | 0.983 | 0.992 |
| whole-joint capsule crop | `10/66` | 0.977 | 0.994 |
| ACL ∪ lateral compartment crop (192 px) | `10/72` | 0.967 | 0.993 |
| notch crop | `10/73` | 0.969 | 0.994 |
| notch crop, seed 1 | `10/75` | 0.967 | 0.993 |
| notch ∪ U-Net ACL box | `10/74` | 0.968 | 0.995 |
| **CAFormer-S18** | `10/67` | **0.989** | 0.993 |
| **HGNetV2-B2** | `10/68` | 0.984 | 0.994 |
| ConvNeXt 224 px | `10/69` | 0.975 | 0.993 |
| **transformer head** | `10/70` | 0.987 | 0.993 |
| **map3d head** | `10/71` | 0.987 | 0.993 |
| CAFormer + axial | `10/77` | 0.987 | 0.994 |
| transformer head + axial | `10/78` | **0.989** | 0.993 |
| HGNetV2 + axial | `10/79` | **0.989** | 0.993 |
| CAFormer + transformer + axial | `10/80` | 0.984 | 0.993 |

- **The notch crop does not help** (0.967–0.969), nor does joining it with the old box. The bruise-site crop carries much of the signal: secondary signs matter, and the zoomed-out notch crop loses detail on the ligament itself.
- **Wider context does not help** either: margin 0.75 scores 0.960, and the lateral-compartment union 0.967.
- **What helps:** axial crops (0.983–0.989 over 2 seeds), CAFormer-S18 (0.989), a transformer or map3d head over the slices (0.987), and HGNetV2-B2 (0.984). Combining these factors in one model does not add up: 0.984–0.989.
- **The `all` model drags ACL down** in the recipe: axial alone scores 0.989, and adding `all` brings it to 0.979.

**Ensembles** (rank average):

| ACL ensemble | 58 | Holdout |
|---|---|---|
| Current recipe (`09/79` + `all` `09/75`) | 0.975 | 0.992 |
| All 6 axial runs | 0.988 | 0.993 |
| **4 distinct axial models: ConvNeXt `10/65`, CAFormer `10/77`, ConvNeXt + transformer head `10/78`, HGNetV2 `10/79`** | **0.993** | **0.994** |
| b1_c1 | 0.977 | |

The ACL recipe is now the 4 distinct axial models, one per architecture (the same "one run per idea" rule as for OA), without the `all` model.

## 16. External data: OAI (Osteoarthritis Initiative)

**Data.** OAI baseline MRI (NDA package 1250166) for the 2,400 knees (2,030 participants) with a baseline MOAKS reading (`kMRI_SQ_MOAKS_BICL00`).
- **Series:** 5 per knee, 12,053 in total (132 GB): sagittal IW-TSE with fat suppression, coronal IW-TSE, sagittal 3D DESS, and coronal and axial DESS reformats.
- **Scanner:** 3 T Siemens only, research protocol. Our data is mixed 1.5/3 T clinical.

**Labels** (`src/oai/labels.py`), from MOAKS:
- **Compartment OA**, with a Hunter-2011-style definition that leans on cartilage, because osteophytes are scored for only ~20% of knees:
  - 1: full-thickness loss on a lesion covering ≥ 10% of a subregion, or an osteophyte of grade ≥ 2, or a ≥ 10% lesion with a large bone marrow lesion or maceration;
  - 0.5: borderline;
  - 0 otherwise.
- **Meniscal tear:** any tear or maceration, or a root tear.
- **ACL:** complete tear 1, partial 0.5.
- **Effusion and Hoffa synovitis:** grade ≥ 2 → 1, mild → 0.25.
- **Baker's:** popliteal cyst.
- **No counterpart:** MCL, contusion, fracture.
- **Prevalence:** medial / lateral / PF OA 0.34 / 0.26 / 0.55 (Claude on our data: 0.23 / 0.14 / 0.27); medial / lateral meniscus 0.41 / 0.20; effusion 0.29; synovitis 0.23; ACL 0.06.

**Pipeline** (`src/oai/prep.py`, `src/oai/boxes.py`): the test-time path, run straight from the series archives.
- **Thin 3D series** (DESS 0.7 mm, reformats 1.5 mm) are averaged into ~3 mm slabs, so that 8-slice crops cover the same anatomy as on clinical series.
- **Side classifier:** agrees with the known knee side on 98–100% of the OAI series.
- **ROI U-Net:** mean box confidence 0.968 on every sequence, DESS included; 2 weak boxes out of 77,208.

**Strategies tried** (each against the same model without OAI; AUC on the 58 / holdout):

| Strategy | Result |
|---|---|
| **Joint training** (our data + OAI crops, OAI labels), OA model | lateral OA 0.882 → 0.913 on the 58 for seed 0, but 0.865 for seed 1. The 4-model OA ensemble with OAI is level on the 58 (0.978 / 0.896 corrected) and lower on the holdout (−0.007 to −0.018). |
| Joint training, TSE sequences only | worse than all sequences |
| Joint training, OAI weight 0.5 / 2 | between / no better |
| OAI only (transfer test) | medial OA 0.913, lateral 0.772, PF 0.829 on our 58, without any of our data. Medial–lateral prediction correlation **0.14** (ours: 0.84). |
| OAI pretraining (same labels) → fine-tuning | 0.922 / 0.920 vs 0.917 / 0.928 |
| OAI-only model as a teacher (0.25 mixed into PL round 2) | the labels improve on the 58 (lateral OA 0.845 → 0.876), the models trained on them do not |
| Joint training, "all" model | PF OA +0.032 and medial meniscus +0.014 on the 58; MCL −0.009 on the holdout; holdout mean −0.002 |
| Joint training, menisci (ConvNeXt), 2 seeds | lateral meniscus holdout +0.004 / +0.017, the 58 mixed; in the menisci recipe +0.002 on the holdout only |
| Joint training: capsule (effusion, synovitis), ACL, Baker's | no gain (synovitis holdout −0.018) |
| Joint training, OA with axial (the PF OA model) | **PF OA −0.014 on both**, also with PF labels removed or made stricter: the OAI axial (DESS reformat) crops are the problem, not the label |
| **MOAKS multi-task pretraining → fine-tuning** | see below |

**MOAKS multi-task pretraining.** The crop model is first trained on OAI alone, with 25 fine-grained MOAKS gradings as targets, each tied to its ROI crop:
- per compartment: cartilage lesion size, full-thickness loss and extent, bone marrow lesions, osteophytes;
- per meniscus: tear, maceration, signal, extrusion;
- ACL grade and popliteal cyst.

It is then fine-tuned on our data with the usual recipe (`main_crops.py --init`). This is the first strategy that helps on both the 58 and the holdout for OA, in two different models:

| Model | Finding | ImageNet init | MOAKS-pretrained |
|---|---|---|---|
| OA (run 40 setup) | Medial OA | 0.978 / 0.947 | 0.984 / 0.947 |
| | Lateral OA | 0.882 / 0.905 | 0.897 / 0.910 |
| | PF OA | 0.889 / 0.931 | 0.915 / 0.931 |
| All (run 75 setup) | Lateral OA | 0.859 / 0.898 | 0.880 / 0.904 |
| | PF OA | 0.880 / 0.924 | 0.906 / 0.931 |
| | MCL | 0.984 / 0.953 | 0.980 / 0.927 |
| Menisci (ConvNeXt) | mean | 0.947 / 0.936 | 0.943 / 0.940 |

- **A shorter, lower-learning-rate fine-tuning gains less.**
- **MCL:** the first MOAKS-pretrained "all" seed dropped on the MCL holdout (0.953 → 0.927). A second seed did not (0.957), so it was seed noise.

**Seeds and other models** (MOAKS-pretrained vs ImageNet init, 58 / holdout):
- **OA model, 3 seeds:**
  - medial OA 0.984–0.986 vs 0.978;
  - lateral OA 0.878–0.897 / 0.910–0.915 vs 0.882 / 0.905;
  - PF OA 0.909–0.915 vs 0.889.
- **The other OA ensemble members (37, 42, 43) and the PF OA model (83):**
  - PF OA +0.014 to +0.026 on the 58 in all six models;
  - lateral OA +0.003 to +0.013 on the holdout in all six;
  - medial OA +0.002 to +0.017 on the 58 in all six.
- **"All" model, 2 seeds:** 0.952 / 0.943 and 0.952 / 0.945, vs 0.946 / 0.942.

**Multi-task joint training.** One model learns our findings on our crops and the 25 MOAKS gradings on OAI crops at the same time (groups `oa_moaks`, `all_moaks`).
- OA, OAI weight 0.25: 0.921 / 0.932 and 0.910 / 0.928 over 2 seeds (ref 0.917 / 0.928).
- OA, OAI weight 1: 0.924 / 0.926.
- "All": 0.952 / 0.946 (ref 0.946 / 0.942).

This is about the same as pretraining followed by fine-tuning.

**Recipe level** (`dataset_6` recipe; 8 ROI findings, 58 / holdout):

| Recipe | 8 ROI |
|---|---|
| `dataset_6` | 0.9610 / 0.9498 |
| OA models replaced by their MOAKS-pretrained versions | 0.9601 / 0.9504 |
| **MOAKS-pretrained OA models added**: 8 OA models; PF OA = 83 + its MOAKS version | **0.9624 / 0.9501** |
| + a MOAKS-pretrained "all" model added for MCL, menisci and Baker's | 0.9609 / 0.9495 |
| + a multi-task "all" model added instead | 0.9614 / 0.9505 |

Through the inference pipeline, the "added" recipe scores 0.961 on the 8 ROI findings vs 0.960 for `dataset_6`: medial OA 0.983, lateral OA 0.901, PF OA 0.914. It is packaged as `output/dataset_7` with fullfit fine-tunes.

**Last two checks.**
- **MOAKS pretraining for the menisci CoAtNet** (224 px, the backbone of the menisci recipe): lateral meniscus 0.938 / 0.923 and medial 0.974 / 0.957. That is within the range of the two existing CoAtNet seeds (0.913–0.942 / 0.925–0.926 and 0.968–0.977 / 0.952–0.955).
- **20-epoch MOAKS pretraining** instead of 10, then the OA fine-tune: 0.985 / 0.947, 0.863 / 0.912, 0.914 / 0.931. The same as with 10 epochs, so longer pretraining does not help.

**Conclusions.**
- **What helps:** fine-grained MOAKS supervision (pretraining or multi-task) gives slightly better single models, about +0.005. Mostly for OA, PF OA in particular.
- **What doesn't:** joint training on derived OAI labels, OAI-teacher pseudo-labels, OAI for effusion, synovitis, ACL and Baker's, and OAI axial (DESS reformat) data for PF OA.
- **At the ensemble level** the gain shrinks to +0.001 to +0.002, within noise. OAI is not a big lever here: the domain (3 T research sequences, older OA-enriched cohort) and the label definitions differ from ours. The ensembles already capture most of what the extra supervision adds.
- **The clearest qualitative effect:** OAI labels teach compartment separation. The medial–lateral OA prediction correlation of an OAI-only model is 0.14, vs 0.84 for models trained on our labels.

## 17. DINOv2 ViT-S

`vit_small_patch14_dinov2.lvd142m` (21.6M params; also the `reg4` registers variant) as the crop encoder. The setup is run 75's: `all` model, 8-frame crops, BiLSTM head, PL round 2, no external data. The 128 px crops are resized to the patch grid: 224 px (16 × 16 patches) or 126 px (9 × 9). The backbone learning rate is 3–10× below ConvNeXt's 1e-4. AUC on the 58 / holdout:

| Encoder | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | Mean |
|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt-tiny 128 (run 75) | 0.973 / 0.991 | **0.984 / 0.953** | 0.976 / 0.950 | **0.935** / 0.911 | 0.969 / **0.947** | 0.859 / 0.898 | 0.880 / **0.924** | **0.993** / 0.966 | **0.946** / 0.942 |
| DINOv2 ViT-S 224, lr 3e-5 | 0.968 / 0.992 | 0.971 / 0.934 | 0.986 / 0.957 | 0.923 / 0.917 | 0.972 / 0.940 | **0.870** / 0.898 | 0.894 / 0.920 | 0.987 / 0.965 | **0.946** / 0.940 |
| DINOv2 ViT-S 224, lr 1e-5 | **0.974 / 0.993** | 0.964 / 0.932 | **0.995** / 0.956 | 0.906 / **0.920** | **0.975** / 0.942 | 0.861 / 0.892 | **0.898** / 0.921 | 0.982 / **0.972** | 0.944 / 0.941 |
| DINOv2 ViT-S + registers 224, lr 3e-5 | 0.971 / **0.993** | 0.966 / 0.939 | 0.990 / **0.959** | 0.903 / 0.916 | 0.974 / 0.943 | 0.849 / **0.904** | 0.897 / **0.924** | 0.987 / 0.966 | 0.942 / **0.943** |
| DINOv2 ViT-S 126, lr 3e-5 | **0.974** / 0.992 | 0.971 / 0.933 | 0.975 / 0.943 | 0.886 / 0.902 | 0.960 / 0.939 | 0.839 / 0.889 | 0.879 / 0.921 | 0.987 / 0.965 | 0.934 / 0.936 |

- **Overall level with ConvNeXt-tiny at 224 px.** It is clearly worse at 126 px: the patch grid is too coarse.
- **Medial meniscus: consistently better** on both metrics in all three 224 px runs (0.986–0.995 / 0.956–0.959 vs 0.976 / 0.950).
- **MCL: consistently worse** (−0.015 on the 58, −0.015 to −0.021 on the holdout).
- **Learning rate:** 1e-5 and 3e-5 end up the same; the registers variant adds nothing.
- **Ensembles:** ConvNeXt + DINOv2 scores 0.947–0.948 / 0.943–0.944. In the menisci recipe, DINOv2 adds at most +0.003 to the medial meniscus and costs the lateral up to 0.012. Not adopted.

## Next steps

- ROIs and crop models for the 4 remaining findings: capsule / joint recesses for effusion and synovitis, bone for contusion and fracture.
- Combine the best variants found here per finding (224 px; axial for PF OA, ACL and MCL), trained on the round-2 targets. Then ensemble seeds and backbones of the ROI models.
- Out-of-fold teachers (k-fold) for a cleaner pseudo-label round, since the current teachers' predictions on training studies are in-sample.
