# Efficiency pipeline

This is a fast version of the main pipeline, `src/infer/pipeline.py` with `dataset_2_moaks_mrnet_men_frac`, built for the efficiency track (2×T4, 4 CPU cores). It produces all 12 findings in a single pass and uses no b1_c1. The notebook is `rsna-knee-inf-eff.ipynb`.

Efficiency = AUC / (0.5 − max AUC) + runtime / 32400. We use these surrogates:
- AUC: the score on the 58 radiologist studies.
- max AUC: the current full ensemble's 58 score plus 0.005, which is 0.9544 + 0.005 = 0.9594.

At those values, 0.001 of AUC is worth about 70 s of runtime.

## What changed from `src/infer/pipeline.py`

| | Main pipeline | Efficiency pipeline |
|---|---|---|
| Series | every series (~5.5 per study, ~129 MB) | one sagittal and one coronal per study, fluid-sensitive first (~39 MB); no axial series |
| DICOM reading | pydicom in DataLoader processes, one file at a time | `fast_dicom.py`: one `read()` per file, header parsed directly, pixels by `np.frombuffer` at the PixelData offset (bit-identical to pydicom on 2,541 files, 13× faster); 2 reader processes per GPU with 16 threads each; headers only for slices past the 64 kept; reading starts before the models load |
| Precision | fp16 autocast | fp16 weights with fp16 inputs (no autocast); weights stored in fp16 (561 MB instead of 1.1 GB) |
| Box extraction | full (8, 32, 128, 128) float32 maps copied to the CPU | peak and region bounds computed on the GPU, only each class's region sub-volume copied back (identical boxes) |
| Consensus | per-row pandas indexing | `fast_consensus.py` (same values, 3.5× faster) |
| CPU / GPU overlap | sequential per chunk | boxes, consensus, crops and models of chunk *i* run in a worker thread while chunk *i+1* goes through prep, side model and U-Net |
| Models | 10 crop models, rank-averaged | one model per finding; the joint model's encoder runs once per crop for its 4 findings (embeddings enter after the encoder) |
| pip | 4 wheels installed | none: the U-Net architecture is vendored; JPEG decoders are installed from the wheels dataset only if a compressed DICOM appears |

## Recipe (`recipe.json` of the weights)

| Finding | Model |
|---|---|
| ACL | `mr85` |
| MCL | `pl86` |
| Medial / lateral meniscus | `cafax87` |
| Medial / lateral / PF OA | `mk90` |
| Baker's | `91` |
| Effusion, synovitis, contusion, fracture | `jc1`: joint4 capsule crop, Codex labels, fullfit `logs/bis/crops_run/2026-10-12/3` (validation: `2026-10-12/1`) |

The scores below are from validation counterparts on the 58 radiologist studies and the holdout, through this pipeline (`score_sub.py`):

| | 58 | Holdout | Crop-model GPU (H100 ms per study) |
|---|---|---|---|
| Submitted ensemble (all series, b1_c1 for 3 findings + fracture blend) | 0.9544 | – | – |
| Submitted 8-ROI recipe + fr9a/b + jc1, 1 sag + 1 cor | 0.9422 | 0.9382 | 26.7 |
| **This recipe, 1 sag + 1 cor** | **0.9447** | **0.9391** | **6.6** |

Per finding on the 58, compared with the latest submission and its parts:

| Finding (58) | Efficiency recipe | `dataset_2_moaks_mrnet_men_frac` | b1_c1 | Latest sub (blend) | Positives |
|---|---|---|---|---|---|
| ACL | **0.993** | 0.983 | 0.977 | 0.983 | 24 |
| MCL | 0.977 | **0.995** | 0.964 | **0.995** | 9 |
| Medial meniscus | 0.971 | **0.986** | 0.966 | **0.986** | 26 |
| Lateral meniscus | 0.939 | **0.958** | 0.865 | **0.958** | 23 |
| Medial OA | **0.991** | 0.977 | 0.986 | 0.977 | 15 |
| Lateral OA | **0.892** | 0.878 | 0.876 | 0.878 | 11 |
| PF OA | **0.925** | 0.923 | 0.870 | 0.923 | 21 |
| Baker's | 0.989 | **0.991** | 0.976 | **0.991** | 12 |
| Effusion | 0.983 | – | **0.989** | **0.989** | 35 |
| Synovitis | 0.805 | – | **0.851** | **0.851** | 27 |
| Contusion | 0.941 | – | **0.955** | **0.955** | 19 |
| Fracture | 0.931 | 0.961 | 0.953 | **0.967** | 18 |
| **Mean, 8 ROI findings** | 0.960 | **0.961** | 0.935 | **0.961** | |
| **Mean, 12 findings** | 0.945 | – | 0.936 | **0.954** | |

- **Latest sub (blend):** `dataset_2_moaks_mrnet_men_frac` on the 8 ROI findings, b1_c1 on effusion, synovitis and contusion, and the rank average of both on fracture.
- **`dataset_2_moaks_mrnet_men_frac`:** predicts only the 8 ROI findings and fracture; its mean over those 9 findings is 0.961.
- **Efficiency recipe:** validation counterparts of its models, run through this pipeline.
- **Where the 0.009 gap on the 12-finding mean comes from:**
  - Mostly synovitis (−0.046 against b1_c1) and fracture (−0.036 against the blend).
  - The rest comes from effusion and contusion (−0.006 and −0.014), and from the menisci and MCL (−0.015 to −0.019), where fewer series and one model per finding cost a little.
  - ACL and the medial and lateral OA gain 0.010 to 0.014.
- **Noise:** with 9 to 35 positives per finding, per-finding gaps under about 0.01–0.02 are noise.

Choices were made on both sets, from a few pre-defined recipes (`recipes.py`, `additions.py`) rather than per-finding argmax (`select_models.py` shows how noisy that is).

Tested and discarded:
- **All sagittal and coronal series instead of one each:** at best +0.002 on the holdout, −0.001 on the 58, for about 2× the I/O and GPU work.
- **U-Net on sagittal only, coronal boxes from registration:** −0.008 on the 58 and −0.003 on the holdout, to save about 25 s.
- **Adding `mk87`, `j27`, `fr9a`, `88` or `mk89`:** each changes the score by ±0.002 (noise), for +2 to +5 ms per study.
- **channels_last_3d for the U-Net:** slower.
- **OA models trained for fluid-sensitive sagittal and coronal series** (`train_fs.py`, a wrapper of `src/main_crops.py` that keeps only `Fluid_Sensitive == 1` crops). Scores are the mean of the 3 OA findings, 58 / holdout, under the 1 sag + 1 cor rule; `mk90` scores 0.937 / 0.927.
  - **Trained from MOAKS (`bis/crops_run/2026-10-13/0`):** 0.926 / 0.924. Each finding drops by 0.003 to 0.03, PF OA the most. It sees about half the training crops.
  - **`mk90` fine-tuned 3 epochs on fluid-sensitive series (`2026-10-13/1`):** 0.937 / 0.927, a tie.

## Runtime

Measured with 493 validation studies, 2 H100 GPUs and 4 CPU cores (warm NFS cache). Each shard takes about 20 s, 7 s of which is model load and imports. The original pipeline with the submitted recipe and all series takes about 4× longer per study.

Estimate for about 1,300 test studies on 2×T4 (per GPU, 650 studies), from H100 timings and FLOP counts:

| Stage | Estimate |
|---|---|
| I/O: 51 GB at about 372 MB/s (public tip: 16 threads) | about 140 s, overlapped with the GPU |
| ROI U-Net: 378 GFLOPs per volume, 1,300 volumes | 30–50 s |
| Crop models: 6.6 ms per study on H100, ×10–12 on a T4 | about 55 s |
| Prep, crop cutting, side model | about 35 s |
| Startup: imports, 561 MB of weights, warm-up | about 20–30 s |

The expected total is about 3–4 min, likely bound by I/O. This needs a Kaggle run to confirm.

## Files

| | |
|---|---|
| `pipeline.py` | the pipeline (`python pipeline.py --data_dir … --weights … --out_dir … --gpus 0,1`); `--rule all` reads every series; env: `KNEE_THREADS`, `KNEE_READ_PROCS`, `KNEE_UNET_PLANES` |
| `fast_dicom.py`, `fast_consensus.py` | the reader and the consensus |
| `build_package.py` | `kaggle/code` (src/ + efficiency/ + dynamic_network_architectures; Kaggle `theoviel/rsna-knee-code-eff`) and `output/dataset_efficiency` (fp16, recipe models only; Kaggle `theoviel/rsna-knee-weights-efficiency`) |
| `evaluate.py`, `recipes.py`, `additions.py`, `select_models.py`, `explore.py` | offline scoring from the validation runs' per-series predictions |
| `score_sub.py` | score a pipeline submission on the validation studies (`val_studies.csv`, with the `weights_val/` symlinks) |
| `bench.py` | GPU cost per network |
