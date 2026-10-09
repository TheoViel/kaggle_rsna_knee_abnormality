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

### Next steps

- ROIs and crop models for the 4 remaining findings: capsule / joint recesses for effusion and synovitis, bone for contusion and fracture.
- Combine the best variants found here per finding (224 px; axial for PF OA, ACL and MCL), trained on the round-2 targets. Then ensemble seeds and backbones of the ROI models.
- Out-of-fold teachers (k-fold) for a cleaner pseudo-label round, since the current teachers' predictions on training studies are in-sample.
