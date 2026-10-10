# RSNA Knee Abnormality Detection

Twelve findings per knee MRI study (ACL, MCL, medial / lateral meniscus, medial / lateral / patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture), scored by macro AUC.

The work so far has four parts:

1. **Baseline sequence classifier**: a 2.5D attention model on whole series.
2. **Left / right classifier**: the knee side of every series.
3. **ROI localization**: one box per finding-related structure in every sagittal and coronal series, learned from the classifier's attention maps.
4. **ROI classification**: 2.5D classifiers on the ROI crops.

**Best results on the 58 radiologist-labelled studies:**
- **ROI models: 0.940 macro AUC on the 8 ROI findings**, vs 0.935 for b1_c1.
- **Rank-averaged with b1_c1: 0.944 on the 8 ROI findings, and 0.942 on all 12** (b1_c1 alone: 0.936; our baseline: 0.890).

All scores are in `experiments.md`.

---

## Data and evaluation

| | |
|---|---|
| Training labels | `input/full_claude_context_v2_labels.csv`: soft labels extracted from the radiology reports, all 4,407 studies |
| **Radiologist score** | AUC against the radiologist labels of the **58 studies** in `train.csv` that have them. These studies are never trained on. This is the number reported throughout. |
| Model selection | A fixed 10% holdout of the other, report-labelled studies (~430) is used only to choose between models. Its scores are not reported. |
| Ceiling reference | The Claude labels themselves score **0.911** against the radiologists. |

With 9–35 positives per finding, a single finding's radiologist AUC on 58 studies can move by several points between equivalent models. Model choices are therefore made on the holdout, never on the 58.

Preprocessing (`notebooks/01_preparation.ipynb`, `src/prepare_cache.py`): DICOM → one array per series, slices ordered along the slice normal → per-series percentile normalisation, padded to a square, 384 px, at most 64 slices (`output/cache_384/`).

---

## Soft labels vs radiologists

Two sets of soft labels have been extracted from the radiology reports:

| Label set | Files | Studies covered |
|---|---|---|
| **Claude** | `input/full_claude_context_v2_labels.csv` | all 4,407 studies; the current training labels |
| **Codex** | `input/codex-labels_final_52.csv` + `input/codex-labels_final_4349.csv` | 52 of the 58 radiologist studies + all 4,349 others |

Six radiologist studies have no Codex label, so both sets are compared on the same **52 studies**. Positives among the 52: ACL 21, MCL 8, medial meniscus 23, lateral meniscus 20, medial OA 12, lateral OA 9, PF OA 17, effusion 32, synovitis 23, Baker's 10, contusion 17, fracture 15.

The models are shown on the same 52 studies for reference: what the labels score is roughly the most a model trained on them can be expected to reach.

| AUC vs radiologists | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture | **Macro** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Claude labels (all 58) | 0.994 | 0.980 | 0.965 | 0.888 | 0.942 | 0.856 | 0.945 | 0.869 | 0.824 | 0.902 | 0.852 | 0.920 | **0.911** |
| Claude labels (52) | 0.992 | 0.976 | 0.959 | 0.874 | 0.941 | 0.860 | 0.939 | 0.850 | 0.831 | 0.899 | 0.823 | 0.936 | **0.907** |
| Codex labels (52) | 0.965 | 0.976 | 0.986 | 0.850 | 0.943 | 0.820 | 0.902 | 0.920 | 0.742 | 0.988 | 0.893 | 0.921 | **0.909** |
| **Mean of Claude and Codex (52)** | 0.991 | 0.980 | 0.979 | 0.865 | 0.944 | 0.849 | 0.933 | 0.898 | 0.822 | 0.975 | 0.882 | 0.931 | **0.921** |
| *Baseline classifier ensemble (52)* | 0.971 | 0.722 | 0.924 | 0.816 | 0.983 | 0.791 | 0.859 | 0.948 | 0.786 | 0.948 | 0.928 | 0.888 | *0.880* |
| *ROI crop models (52), 8 ROI findings only* | 0.985 | 0.980 | 0.966 | 0.888 | 0.962 | 0.827 | 0.850 | — | — | 0.924 | — | — | *0.923 (8 ROI)* |

The last row uses the crop models for the 8 ROI findings and the baseline for the other 4.

- **The two label sets score the same overall but err differently.**
  - Codex is clearly better on Baker's (+0.09), effusion (+0.07), contusion (+0.07) and the medial meniscus.
  - Claude is better on synovitis (+0.09), PF OA, lateral OA and ACL.
- **Their average is the best label set: 0.921**, against 0.907 / 0.909 for either alone.
- **Codex calls more findings positive** than Claude, on the 4,349 other studies (share with label ≥ 0.5):

  | | Effusion | Baker's | MCL | ACL |
  |---|---|---|---|---|
  | Codex | 47% | 31% | 8.5% | 17% |
  | Claude | 26% | 13% | 1.9% | 11% |

  The rankings still agree: per finding, Codex's scores recover Claude's binary labels with AUC 0.955–0.997 (Spearman 0.67–0.88).
- **The models are already at label level on some findings and above it on others.**
  - On the 8 ROI findings, the ROI crop models reach 0.923 on these 52 studies, vs 0.930 / 0.929 for the Claude / Codex labels.
  - The baseline classifier beats both label sets on effusion (0.948), contusion (0.928) and medial OA (0.983).
  - Training on many studies averages out the reports' noise, and the images carry information the reports do not.
- **Next step:** train on the mean of the Claude and Codex labels.

---

## 1. Baseline sequence classifier

`src/main_cls.py`, `src/knee_cls/`

- **Per series:**
  - a 2D ConvNeXt sees one *frame* at a time: 3 adjacent slices as channels, 224–288 px, 6 frames strided over the series;
  - per finding, an attention map over the image (stride 8) and an attention over frames, giving a 3D attention volume per finding;
  - an attention over the series of the study.
- **Training:** soft BCE on the Claude labels, 4-GPU DDP, ~12 min per run.

| AUC on the 58 | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture | **Macro** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Best single run (288 px) | 0.972 | 0.732 | 0.941 | 0.827 | 0.971 | 0.826 | 0.875 | 0.961 | 0.778 | 0.906 | 0.930 | 0.887 | **0.884** |
| Ensemble of 5 runs | 0.974 | 0.753 | 0.936 | 0.845 | 0.986 | 0.820 | 0.855 | 0.959 | 0.791 | 0.955 | 0.933 | 0.876 | **0.890** |

Runs: `logs/cls_run/2026-10-08/{2,3,4,5,6}`. Its validation predictions are in `output/global_ens_val_preds.csv`.

Sweep findings:
- 224 px with 6 frames of 3 slices gave the best quality for the compute.
- 10 frames, 288 px and ConvNeXt-small did not beat it beyond noise.

---

## 2. Left / right (knee side) classifier

`src/side_cls.py`, `src/main_side.py`. Weights: `logs/side_run/2026-10-09/0`.

- **Model:** a small 3D CNN on the 32 × 128 × 128 series volume in stored orientation, plus the plane as input channels.
- **One prediction per series**, sagittal, coronal and axial alike, because a study can contain both knees.
- **Augmentation:** mirroring along the left-right axis, with the label swapped.

Results:
- **98.2%** agreement with the DICOM `Laterality` tag (0020,0060) on the tagged held-out series, and 98.4% per study.
- The disagreements are tag errors, not model errors. In all 21 disputed sagittal/coronal series, the ROI model's own medial/lateral compartment geometry, computed without the tag, sides with the classifier.
- Over all training series, the tag disagrees with the model on 1.2% (144 of 12,000 tagged series). The tag is missing on about half the series anyway.
- **The model, not the tag, is the source of truth.**

Output: `output/series_laterality.csv`, one row per series:
- `dicom_laterality`;
- `p_right_model`, `side_model`;
- `lr_axis`, `lr_axis_points_to`, `flip_to_right_knee`.

In stored orientation the left-right axis runs the same way across all training series: sagittal slices towards the patient's right, coronal and axial columns towards the left. A left knee is therefore an exact mirror of a right one along that axis.

---

## 3. ROI localization

`src/roi/`, `src/main_roi.py`. Weights: `logs/roi_run/2026-10-09/1` (round 3).

**Output:** exactly one box per ROI in every sagittal and coronal series, for 8 ROIs:
- ACL, MCL;
- medial meniscus, lateral meniscus;
- medial, lateral and patellofemoral compartments (the 3 OA findings);
- Baker's (popliteal fossa).

A missing box counts as a model failure. Axial series get their boxes by registration only.

**Model:** a 3D residual U-Net (`dynamic_network_architectures`, 1 × 32 × 128 × 128 input), one channel per ROI, read back as the bounding box of the main component. Volumes are first flipped so that every knee is a right knee, using the side classifier.

**Labels.** There are no ground-truth boxes, so the labels are built and then self-refined:

1. **Round 0, from attention:** an ensemble of 4 classifier runs, with ±2 px shift TTA (finer attention grid). A box is kept only when the attention is compact. A box is dropped when the other view of the study disagrees in patient space.
2. **MCL:** the classifier never attends to the ligament itself (its compact MCL maps sit in the tibia). MCL boxes therefore come from an anatomical prior: a band along the medial edge of the medial compartment, from −45 to +35 mm around the joint line.
3. **Self-training rounds:** a predicted box becomes a label when another series of the study confirms it (patient-space IoU ≥ 0.3, other plane or same plane). Then:
   - medial / lateral pairs on the wrong side of the knee are removed;
   - a meniscus outside its own compartment is removed;
   - gaps are filled by registration from the study's other series.

**Per-box confidence** = U-Net mean probability in the box × (0.5 + 0.5 × support), where support is the share of the study's other series whose box agrees.

Quality on 2,058 held-out series. No ground-truth boxes exist, so the checks are label-free:

| | Round 0 | Round 1 | Round 3 (current) |
|---|---|---|---|
| Cross-view IoU (sagittal vs coronal, patient space) | 0.335 | 0.411 | **0.464** |
| Same-plane IoU (two series of one plane) | 0.553 | 0.659 | **0.698** |
| Medial meniscus on the medial side | 0.948 | 0.959 | **1.000** |
| Meniscus inside its own compartment (medial / lateral) | 0.64 / 0.78 | 0.84 / 0.78 | **1.00 / 1.00** |
| Boxes below the confidence threshold | 0 | 0 | 0 |

**MCL evidence test** (`src/roi/evidence.py`). A small classifier that sees only the crop inside a region must predict MCL injury.

| Region | Radiologist AUC (58) |
|---|---|
| **MCL box** | **0.91** |
| Moved 6 mm inward | 0.82 |
| Moved 6 mm outward | 0.80 |
| Lateral mirror (LCL, control) | 0.70 |
| Whole volume | 0.71 |

The box sits where the evidence is. These AUCs rest on only 9 positive studies.

Labels and predictions are in `output/roi/labels/` (`roundN.csv`, `roundN_pred.csv`). Evaluate with `python -m roi.evaluate round3_pred`.

---

## 4. ROI classification

`src/crops/`, `src/main_crops.py`, runs in `logs/crops_run/`

**Crops** (`python -m crops.prepare`), for every series and every ROI:
- a square around the box with 25% margin, cut from the 384 px cache and resized to 128 × 128;
- 12 slices spanning the box's slice range;
- right-knee orientation.

**2.5D model:**
- each slice, with its two neighbours as RGB channels, goes through a pretrained timm backbone (ConvNeXt-tiny);
- a BiLSTM runs over the per-slice feature vectors, followed by attention pooling and one logit;
- ROI and plane embeddings let several findings share a model.

**Training and scoring:** soft BCE on the Claude labels. A study's probability is the mean over its series.

Best model per finding on the 58 radiologist studies (ConvNeXt-tiny, sagittal + coronal, 10 epochs):

| Finding | Model group | ROI model (58) | Baseline ensemble (58) |
|---|---|---|---|
| ACL | `acl` (alone) | **0.987** | 0.974 |
| MCL | `mcl` (alone) | **0.982** | 0.753 |
| Medial meniscus | `menisci` (shared) | **0.971** | 0.936 |
| Lateral meniscus | `menisci` (shared) | **0.906** | 0.845 |
| Medial OA | `oa` (shared) | 0.971 | **0.986** |
| Lateral OA | `oa` (shared) | **0.845** | 0.820 |
| PF OA | `oa` (shared) | **0.864** | 0.855 |
| Baker's | `bakers` | 0.935 | **0.955** |

Grouping findings:
- **Menisci:** one shared model beats two separate ones (lateral 0.906 vs 0.870).
- **Ligaments:** separate models beat a shared one (ACL 0.987 vs 0.978).
- **OA:** the 3 compartments share one model.
- **One model for all 8 ROIs** is close: 0.931 macro on the 8 findings (radiologists), against 0.938 for the per-group menisci model on its two findings.
- **Planes:** sagittal + coronal together beat either plane alone (menisci 0.938 vs 0.919 sagittal-only / 0.894 coronal-only).

**ROI models vs b1_c1 on the 58.** For each finding, the ROI crop model is the run with the best holdout score, never chosen on the 58. All tables are in `experiments.md`, section 6.

| On the 58 | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | **8 ROI** |
|---|---|---|---|---|---|---|---|---|---|
| b1_c1 | 0.977 | 0.964 | 0.966 | 0.865 | 0.986 | 0.876 | 0.870 | 0.976 | 0.935 |
| **Pure ROI** | 0.969 | 0.977 | 0.962 | 0.889 | 0.958 | 0.880 | 0.900 | 0.986 | **0.940** |
| ROI + b1_c1 (rank average, 50/50) | 0.974 | 0.983 | 0.969 | 0.890 | 0.974 | 0.890 | 0.890 | 0.983 | **0.944** |

The 50/50 weight is fixed a priori: b1_c1 has no holdout predictions to tune it on. With b1_c1 on the 4 findings without an ROI, the blend scores **0.942** on all 12 findings, vs 0.936 for b1_c1 alone.

**ROI models vs the soft labels.** These are the best round-1 ROI models, trained on the Claude labels only. They are compared with both label sets on the 52 radiologist studies that have Codex labels:

| AUC vs radiologists (52) | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | **Macro (8 ROI)** |
|---|---|---|---|---|---|---|---|---|---|
| Claude labels | **0.992** | 0.976 | 0.959 | 0.874 | 0.941 | **0.860** | **0.939** | 0.899 | 0.930 |
| Codex labels | 0.965 | 0.976 | **0.986** | 0.850 | 0.943 | 0.820 | 0.902 | **0.988** | 0.929 |
| Mean of both labels | 0.991 | **0.980** | 0.979 | 0.865 | 0.944 | 0.849 | 0.933 | 0.975 | **0.939** |
| **ROI models** | 0.985 | **0.980** | 0.966 | **0.888** | **0.962** | 0.827 | 0.850 | 0.924 | 0.923 |
| Baseline ensemble | 0.971 | 0.722 | 0.924 | 0.816 | 0.983 | 0.791 | 0.859 | 0.948 | 0.877 |

- **Overall:** the ROI models are within 0.007 of either label set on the 8 ROI findings, up from 0.877 for the baseline.
- **Where they beat both label sets:** lateral meniscus and medial OA. They tie the best on MCL.
- **Where they fall short:** PF OA (0.850 vs 0.939 for Claude) is the largest gap, then Baker's (vs Codex) and lateral OA.

The two label sets disagree most on what counts as a Baker's cyst: on the 4,349 non-radiologist studies Codex calls 31% positive, Claude 13%.

Models trained on the mean labels and on Codex alone (`main_crops.py --labels mean|codex`) are in `experiments.md`.

The rows above use the round-1 ROI models. The full sweep (backbones, 224 px, axial crops, label sources) is in `experiments.md`.

**Sweep findings** (details in `experiments.md`):
- 224 px helps the menisci and lateral OA.
- Axial crops (boxes registered from sagittal + coronal, `python -m crops.axial_boxes`) help PF OA, ACL and MCL.
- The mean / Codex labels help Baker's and slightly help the lateral meniscus.

### Since then: 8-frame crops, aggregation, backbones, effusion / synovitis

Details are in `experiments.md` sections 7–11. Scores are on the 58 radiologist studies.

- **Pseudo-labels (section 7):** two rounds. Round 2 targets are 0.5 × the mean of the top-5 holdout models + 0.5 × the Claude/Codex mean. 8 ROI findings: 0.940 → 0.947.
- **Frame sampling (section 8):** 8 adjacent slices centred on the ROI (stride 1) replace 12 uniform frames over the box.
  - Submission recipe: 0.949 → **0.953**, with a third fewer frames to run.
  - Each model records its sampling in `args.json`, and inference reproduces it.
- **Slice aggregation (section 9):** the BiLSTM over slice vectors stays.
  - A transformer, 3D convs over the last feature maps, and a temporal shift inside the encoder all tie with it.
  - Early fusion (slices as channels) is 8× cheaper and costs 0.012.
- **Backbones (section 10):** ConvNeXt-tiny stays. CAFormer-S18 (+0.003) is within noise. HGNetV2-B2 matches it with ¼ of the FLOPs. The mobile CNNs trail.
- **Effusion and synovitis (section 11):** one capsule crop per series.
  - The crop is the union of the 8 ROI boxes, widened 15% per side to include the suprapatellar recess, with 24 uniform frames at 160 px, through the same 2.5D model.
  - Effusion reaches **0.980** (b1_c1 0.989, our baseline 0.959).
  - Synovitis reaches 0.784 (b1_c1 0.851, baseline 0.791), and stays the weak finding.
  - 3D models (the ROI U-Net's encoder, pretrained or from scratch) are worse on effusion and on the holdout.

---

## Inference

`notebooks/rsna-knee-inf.ipynb` (cells: imports, parameters, run, outputs) runs `src/infer/pipeline.py`, one streaming pass per GPU:

- **Sharding.** The studies are split between the GPUs, balanced on file count, with one process per GPU.
- **CPU work.** DataLoader workers only read the DICOMs: headers of every slice, and pixels of the ≤ 64 slices the cache keeps.
- **GPU work.** Everything else runs on the GPU, in fp16: normalisation and resizing (384 px cache, 32×128×128 volume), side classifier, ROI U-Net, crop cutting and the 7 crop models.
- **CPU between GPU steps.** Box extraction, per-knee consensus and registration (including axial boxes) run per chunk of 24 studies. They work knee by knee, so chunking changes nothing.
- **No intermediate files.** A chunk's caches stay on the GPU. The old staged path wrote ~32 GB of 384 px caches for the test set.
- **Fewer predictions.** Each crop model only predicts the findings `recipe.json` takes from it, which is 21% fewer FLOPs.
- **Output.** Per finding, the recipe's models are combined by rank average over the test studies. The 4 findings without an ROI stay at 0.5.

The staged modules (`infer.prep` → `infer.side` → `infer.seg` → `infer.classify`) still exist and do the same operations.

Weight folders (side classifier, ROI U-Net, fullfit crop models, `recipe.json`):

| Folder | Crop models | Findings predicted (others at 0.5) | 58 AUC, same recipe with non-fullfit models |
|---|---|---|---|
| `output/dataset_1/` | 12 uniform frames, runs 63–69 | 8 ROI findings | 0.949 (8 ROI) |
| `output/kaggle_weights/` | 8 strided frames, runs 85–91 | 8 ROI findings | 0.952 (8 ROI) |
| `output/dataset_2/` | 85–91 + capsule fullfit `10/23`, `10/24` (2 seeds) | 8 ROI + effusion + synovitis | 0.938 (10), 0.865 (12) |
| `output/dataset_3/` | dataset_2 + joint-crop fullfit `10/28` (sag + cor + axial) | all 12 | **0.939 (12)** |

The two capsule models (24 frames × 160 px, one crop per sagittal or coronal series) cost ~0.9 TFLOP per study, about 40% of the 8-frame ROI crop models: roughly +1.5 min on 2×T4.

Parity on the 58 radiologist studies (round-2 models, recipe C): **0.9486** fused fp16 vs 0.9488 staged. Per-finding correlation is ≥ 0.9986, and 95% of the U-Net boxes are identical.

### Runtime, ~1300 test studies on Kaggle (2×T4, 4 CPU cores)

Workload: ~7,200 series (~5,450 sagittal/coronal through the U-Net) and ~180 GB of DICOMs. The training set averages 25 MB per series.

T4 times are extrapolated: isolated H100 fp16 latencies × 10, the rough fp16 gap between the two cards for these convnets. CPU times are measured on 1–4 cores.

| Stage | Staged path (before) | Fused pipeline |
|---|---|---|
| Read DICOMs (disk) | ~180 GB | ~165 GB (slices past the 64 kept are not decoded): **5–14 min at 200–500 MB/s** |
| Decode + preprocess (CPU) | 0.245 s/series → 7.4 min on 4 cores | decode only, 0.054 s/series → 1.6 min, preprocessing moved to GPU |
| 384 px caches to disk | ~32 GB written + read back | none |
| ROI U-Net | 1 GPU: ~6.6 min | 2 GPUs: ~3.3 min |
| Box extraction (CPU) | 0.074 s/series, serial: 6.7 min | 0.008 s/series (exact, label within the region's bbox): 0.4 min |
| Crop models | 1 GPU, every ROI: ~15 min | 2 GPUs, used findings only: ~6 min |
| **Total** | **~40–50 min** | **~12–16 min**: max(disk, ~10 min GPU per T4) + model loading and warm-up (seconds; `cudnn.benchmark` is off, it cost ~2 min of first-call autotuning on a T4) |

Measured here with 4 cores and 2 H100s: 300 training studies in 139 s. That run was bound by cold NFS reads at ~300 MB/s; its GPU stages took 35 s per shard.

Bottlenecks now:

1. **Disk reads.** These are hardware-bound and overlap with the GPU work.
2. **The two 224 px crop models**, 65 (CoAtNet, menisci) and 67 (ConvNeXt, OA). They are about half the GPU time, because 128 px crops are upsampled to 224 px.
3. **The ROI U-Net**, about a third of the GPU time.

---

## Tools

| | |
|---|---|
| `segmentation_viewer.py` (`bash app.sh`) | Streamlit viewer: ROI boxes per series and slice, attention heatmaps, localizer lines between planes |
| `notebooks/04_attention_maps.ipynb` | Baseline classifier attention maps, as rough segmentations |
| `notebooks/05_registration.ipynb` | Registration between views (sagittal / coronal → axial) |

---

## Next steps

- Contusion and fracture come from the joint crop (`experiments.md` section 12): 0.942 / 0.946 on the 58, vs b1_c1's 0.955 / 0.953. They need a decision before going into a submission (`output/dataset_3/`).
- Synovitis: axial capsule crops (+0.012 on the 58, level on the holdout), and better labels. Even the soft labels score only 0.74–0.82 against the radiologists.
- Retrain the side classifier on its own (cleaner) labels, and the ROI U-Net on the round-3 labels (round 4).
- The 4 findings without an ROI (synovitis 0.79 and fracture 0.88 are the weakest) need their own approach.
- Train on the mean of the Claude and Codex labels: on the 52 radiologist studies it beats either set alone (0.921 vs 0.907 / 0.909).
