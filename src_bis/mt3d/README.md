# Multitask 3D / hybrid models on the joint crop (`src_bis/mt3d/`)

One model for the 8 ROI findings, trained on one crop per series: the union of the
8 ROI boxes. An auxiliary segmentation task (the ROI boxes) tells the model where
each finding lives, and the features are pooled under each ROI's own predicted
mask. This was inspired by the 1st and 2nd place RSNA 2025 aneurysm solutions.

All scores are the mean AUC over the 8 ROI findings:
- **58**: radiologist labels.
- **holdout**: Claude labels at 0.5, about 430 studies.
- **hold7**: the holdout without lateral OA. Its Claude labels reward medial / lateral cross-talk (`experiments.md` §14).

Everything is trained on `pl:pl_round2`, so it compares directly with the main agent's `09/75`.

**Selection is on the holdout only.** Seed-to-seed noise is ±0.005 on the 58 and about ±0.001 on the holdout.

## Results in one table

| Model | Runs | 58 | Holdout |
|---|---|---|---|
| 3D U-Net (ROI U-Net init), box seg + masked pooling | 10/1 | 0.928 | 0.929 |
| Hybrid: ImageNet ConvNeXt per frame + 3D FPN, box + anatomy seg | 10/7, 10/11 | 0.936–0.940 | 0.940 |
| Hybrid, sagittal + coronal pair of a study through one transformer | 10/15 | 0.940 | 0.944 |
| Hybrid set (4 series per study), seg weight 1.0 | 11/1, 11/3 | 0.939–0.941 | 0.945–0.946 |
| **Hybrid set, seg weight 0.1–0.5** (box ± anatomy) | 11/7, 11/11, 11/15, 11/18, 11/19, 11/23–25 | 0.932–0.941 | **0.947–0.950** |
| Ensemble of 5 light-weight set models | 11/7, 11/11, 11/13–15 | 0.937 | **0.951** |
| Ensemble of all 12 light-weight set models (saturates) | 11/7–26 | 0.940 | **0.951** |
| `09/75`, the 2.5D "all" model | | 0.946 | 0.942 |
| `dataset_5` recipe, 13 models, offline rebuild | | **0.960** | 0.948 |
| recipe + 0.5 × the 5-model ensemble (rank average) | | 0.950 | **0.953** |

## What mattered, in order

1. **Backbone: +0.010–0.012.** The 3D U-Net pretrained on box locations plateaus at 0.93. An ImageNet ConvNeXt-tiny per frame (neighbours as RGB), with its stride-8/16/32 maps stacked into 3D and a small 3D FPN decoder, reaches 0.940 and trains 3× faster.
2. **Cross-view fusion: +0.005–0.006 on the holdout.** Several series of a study go through the same model, and their tokens (global + 8 ROI) go through one transformer. Each ROI token can attend to the same ROI in the other views.
   - It helps PF OA (holdout 0.925 → 0.94), lateral OA and MCL.
   - 2, 4 or 6 series per study perform about the same (6: holdout 0.949–0.950); 4 is the default.
3. **The auxiliary box segmentation is essential, at a light weight.** Set model, holdout:

   | Box seg weight | Anatomy seg weight | 58 | Holdout |
   |---|---|---|---|
   | 0 | 0 | 0.922 | 0.939 |
   | 0 | 0.3 | 0.925 | 0.937 |
   | 0.1 | 0.1 | 0.935 | 0.950 |
   | 0.1 | 0 | 0.935 | 0.948 |
   | 0.3 | 0 | 0.937–0.940 (2 seeds) | 0.948 (mean) |
   | 0.5 | 0 | 0.934 | 0.948 |
   | 0.3 | 0.3 | 0.932–0.941 (3 seeds) | 0.948 (mean) |
   | 1.0 | 1.0 | 0.939–0.941 (2 seeds) | 0.946 |

   - Without the box task, the pooling masks are unsupervised attention and the model loses about 0.01 on both sets.
   - The anatomy task (nnU-Net masks) doesn't add anything on top of the box task.
   - Any weight from 0.1 to 0.5 gives the same holdout (0.948); 1.0 is slightly too strong (0.946). The 1st place's classification weight of 0.1 (with segmentation at 1) underfits our dense soft labels.

## What didn't help (holdout within noise, or worse)

- **Resolution:** 3D U-Net at 256 or 320 px (320 was stopped early); hybrid with 32 frames; pooling on the stride-4 level.
- **Heads:** cross-attention pooling of the bottleneck (2nd place), worse; a 4-layer transformer; anatomy tokens; a learned frame-position embedding.
- **Pooling under the U-Net boxes** instead of the predicted masks: slightly worse.
- **Zoom branch**, a 2.5D crop of each ROI through the shared encoder:
  - single-series models: lateral meniscus holdout +0.01;
  - set models: no gain, at 128 or 224 px, menisci only or all ROIs.
- **Backbones:** CAFormer-S18 no gain; ConvNeXt-small +0.001 for 2× the cost.
- **Training:** mixup between studies; 25 epochs; backbone lr 2e-4; drop path 0.2.
- **Labels:** the Claude + Codex mean, worse than `pl_round2` on both sets.
- **Test-time augmentation:** frame-offset and zoom TTA.
- **All 12 findings in one model:**
  - the 8 ROI findings keep their score;
  - the 4 extra ones are near the main agent's dedicated models in single-series mode (58: 0.914 vs 0.9125);
  - they're worse in set mode (0.879).

## Per finding: where the hybrid stands

AUC, 58 / holdout:

| | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's |
|---|---|---|---|---|---|---|---|---|
| Hybrid set, seg weight 0.3 (11/7) | 0.991 / 0.992 | 0.971 / 0.969 | 0.960 / 0.948 | 0.866 / 0.920 | **0.994** / 0.949 | 0.826 / 0.913 | 0.894 / 0.936 | 0.982 / 0.971 |
| `dataset_5` recipe | 0.993 / 0.994 | 0.991 / 0.942 | 0.978 / 0.953 | **0.945** / 0.922 | 0.980 / 0.946 | **0.899** / 0.922 | 0.906 / 0.937 | 0.991 / 0.968 |

- **Lateral meniscus:** every hybrid run is at the recipe's level on the holdout (0.91–0.92) but 0.03–0.07 lower on the 58. The hybrid fits the report-derived labels but transfers less well to the radiologists for this finding. I didn't tune on the 58 to chase it.
- **Lateral OA:** the hybrid has the same cross-talk problem as the main agent's crop models had. The joint crop sees both compartments. The midline crops of `dataset_5` fix it there.
- **Medial OA:** the hybrid is the best model we have, 0.99 on the 58, above b1_c1's 0.986.

## Files and commands (run from `src_bis/`)

| | |
|---|---|
| `mt3d/prepare.py` | joint crops (union of the 8 ROI boxes + 10%, 32 × 256 × 256) and the ROI boxes in crop coordinates → `output/bis/mt3d/` |
| `mt3d/anatomy.py` | nnU-Net anatomy targets (fluid-sensitive series only, partial labels) |
| `mt3d/model.py` | the 3D U-Net multitask model (MT3D) |
| `mt3d/hybrid.py` | `Hybrid` (2D backbone + 3D FPN + masked pooling + zoom branch) and `PairHybrid` (pairs / sets of series) |
| `mt3d/train.py`, `main_mt3d.py` | data, losses, fit loop; series, pair or set datasets |
| `mt3d/summary.py` | table of all runs: `python -m mt3d.summary` |
| `mt3d/blend.py` | runs alone, vs `09/75` and the `dataset_5` recipe, and blended into them: `python -m mt3d.blend --runs 2026-10-11/7,...` |
| `mt3d/evaluate.py` | re-score a run, with optional TTA |
| `mt3d/queue.sh` | experiment queue: workers read `logs/bis/queue.txt` |

Best configuration:

    torchrun --standalone --nproc_per_node=2 main_mt3d.py --arch hybrid --pair true --slots 4 --batch_size 4 \
        --epochs 15 --lr_unet 1e-4 --lr 5e-4 --w_seg 0.3 --w_anat 0.3

This is about 22 minutes on 2 H100s. `--w_anat 0` drops the anatomy task (same score, no anatomy targets needed).

## Follow-ups (2026-10-11)

**MOAKS pretraining** (as in `experiments.md` §16, adapted to the hybrid set model):
- **Recipe:** `--pretrain oai_u10_d32_s256` trains on the OAI joint crops of 2,400 knees, using the 25 MOAKS gradings, each predicted from its ROI's token. Fine-tuning from that checkpoint uses `--init`.
- **Gradings only (run 28):** fine-tunes score 0.949 / 0.950 on the holdout (runs 30, 32), against 0.947–0.949 for ImageNet init. Lateral OA holdout rises from 0.918–0.923 to 0.931–0.933. A small but consistent gain.
- **Gradings plus the MOAKS-derived findings (run 27):** no gain.

**Report embeddings as an auxiliary target** (`--w_rep`, `mt3d/reports.py`):
- **Setup:** each training report is embedded with multilingual-e5-base. The model predicts its 64-d PCA projection from the study token, with a cosine loss.
- **Result: it hurts.** Holdout 0.935 at weight 1.0 (2 seeds) and 0.945 at 0.3, against 0.948.
- **Where it hurts most:** the OA findings, with lateral OA at 0.86–0.89 vs 0.92. A whole-report embedding probably encodes the language and institution style and overall severity more than the compartment-level facts, so it pulls the medial and lateral compartments together.
- **Not adopted.**

**Other backbones with MOAKS pretraining** (the `11/28` → `11/32` recipe, 2 fine-tune seeds each, holdout / 58):
- **CoAtNet-0, `--levels 2,3,4`** (pretraining 43, fine-tunes 45 and 46):
  - holdout 0.950 / 0.948, level with ConvNeXt;
  - the 58 is lower, 0.927 for both seeds (ConvNeXt: 0.937–0.941).
  - The first attempt (37, 39, 41) was invalid: the EMA didn't copy BatchNorm buffers, which CoAtNet is the only backbone to have. Fixed in `mt3d/train.py`.
- **DINOv2 ViT-S, `--levels 5,8,11`** (pretraining 38; fine-tunes 40 and 42, plus 44 with the backbone lr at 3e-5):
  - blocks 5, 8 and 11 are resampled to strides 8, 16 and 32 (ViTDet style);
  - holdout 0.912–0.916, 58 0.908–0.914, clearly weaker on every finding;
  - a lower learning rate doesn't help.
- **Ensembles:**

  | Ensemble | Holdout | 58 |
  |---|---|---|
  | ConvNeXt × 2 | 0.950 | 0.939 |
  | CoAtNet × 2 | 0.950 | 0.928 |
  | **ConvNeXt + CoAtNet** | **0.952** | 0.935 |
  | + DINOv2 | 0.946 | 0.933 |
  | `dataset_2` | 0.946 | 0.953 |
- **CAFormer-S18 and HGNetV2-B2, `--levels 1,2,3`** (pretraining 47 / 48; fine-tunes 49–50 / 51–52):

  | 2-seed ensemble | Holdout | 58 |
  |---|---|---|
  | CAFormer | 0.947 | 0.934 |
  | HGNetV2 | 0.950 | 0.937 |
  | **ConvNeXt + HGNetV2** | **0.952** | **0.941** (best hybrid pair on both sets) |
  | ConvNeXt + CoAtNet | 0.952 | 0.935 |
  | ConvNeXt + CAFormer | 0.950 | 0.938 |
  | All four backbones | 0.951 | 0.937 |

**MRNet joint training** (`--extra_crops mrnet_u10_d32_s256 --extra_labels ../output/mrnet/labels_graded.csv`, on top of the MOAKS ConvNeXt recipe, 2 seeds each):

| Variant | Runs | Holdout | 58 |
|---|---|---|---|
| Reference: MOAKS ConvNeXt × 2 | 30, 32 | 0.950 | 0.939 |
| + MRNet | 53, 55 | 0.949 | 0.937 |
| + MRNet + MIL (`--w_mil 1`) | 54, 56 | 0.924 | 0.915 |

- **MRNet alone is neutral.** The lateral meniscus gains +0.01 on the 58, which is within noise; PF OA loses 0.02.
- **The MIL loss is harmful.** It supervises max(medial, lateral) with MRNet's "meniscal tear" flag. The lateral meniscus collapses to 0.805 / 0.851.
  - MRNet's report-derived tear flag includes lower-grade tears that our convention calls negative.
  - Pushing the larger of the two logits up then mostly teaches the model false positives.

**External data with the graded labels** (`external_data.md` §2; MOAKS ConvNeXt set recipe; holdout / 58; reference runs 30 and 32: 0.950 / 0.939):

| Variant | Runs | Holdout | 58 |
|---|---|---|---|
| Pretraining with the 25 gradings + OAI graded v4 findings | 57 → 62, 63 | 0.947, 0.947 | 0.938, 0.942 |
| + OAI v4 joint | 58, 61 | 0.946, 0.950 | 0.935, 0.940 |
| + MRNet graded, weight 2 | 59 | 0.946 | 0.929 |
| + MRNet soft side labels (`labels_mensoft`) | 64 | 0.949 | 0.937 |
| + MRNet pseudo-labels (`labels_pl`) | 65 | 0.949 | 0.935 |
| + OAI v4 + MRNet + SKM-TEA (weights 1, 1, 1) | 60 | 0.947 | 0.933 |
| + OAI v4 + MRNet soft + SKM-TEA (weights 1, 1, 4) | 66 | 0.947 | 0.933 |

- **Every variant is within seed noise of the reference.** None helps.
- **As extra ensemble members** they add nothing: ConvNeXt + HGNetV2 stays at 0.952 / 0.941.
- **The only external-data gain for the hybrid** remains the 25-grading MOAKS pretraining.
- **Crops:** joint crops for SKM-TEA in `skm_u10_d32_s256`. Per-source loss weights via `--extra_weight 1,1,4`.

## Inference (`src_bis/infer/pipeline.py`, `notebooks/rsna-knee-inf-bis.ipynb`)

`src/infer/pipeline.py` copied to `src_bis/` and extended. Nothing in `src/` or `rsna-knee-inf.ipynb` is touched.

**Hybrid models** (`<weights>/mt3d/<name>/`: args.json, ckpt_final.pt, crop.json) share all preprocessing with the crop models: DICOM reading, GPU cache, side, U-Net, consensus.
- **Crop:** one joint crop per sagittal / coronal series (union of the 8 consensus boxes + 10%, 32 × 256 × 256), cut once from the GPU cache and shared by all hybrids.
- **Grouping:** a study's series are grouped 4 at a time as in validation (`study_groups` = `PairDataset`).
- **Study prediction:** the mean over groups.

**Recipe:** a nested list is one member, the mean rank of its models.

**Autocast** (`--amp`):
- fp16, for T4s;
- bf16, as validated;
- fp32, for exactness checks.

Two fp16 overflows fixed in `mt3d/hybrid.py`: the masked pooling and the token projection now run in fp32. bf16 and fp32 outputs are unchanged.

**Reproduction** (493 validation studies; 58 / holdout):

| Check | Result |
|---|---|
| `python -m mt3d.check_pipeline --amp fp32 --ref_fp32` (pipeline code vs validation code, same crops) | max \|Δp\| 2e-6 (ConvNeXt) / 6e-5 (HGNetV2), identical AUCs |
| Same, bf16, against the saved validation predictions | ensemble 0.9410 / 0.9515, identical |
| End to end from the DICOMs, 4 non-fullfit hybrids (`output/bis/weights_val`) | bf16 0.9408 / 0.9517; **fp16 0.9412 / 0.9517**; validation 0.9410 / 0.9515 |
| End to end, `dataset_2_moaks` non-fullfit twin + hybrids on ACL and medial OA (`output/bis/weights_val_combined`) | 0.9580 / 0.9453 vs offline 0.9578 / 0.9451 |
| Same, hybrids on all 8 findings | 0.9505 / 0.9501 vs offline 0.9503 / 0.9500 |

**Runtime:** 493 studies on 2 H100s in 160 s for the combined recipe, against 110 s for the hybrids alone.

**Submission package:** `python -m mt3d.package --out ../output/bis/dataset_2_moaks_hybrid --hybrids h30=2026-10-11/67,h32=2026-10-11/68,h51=2026-10-11/69,h52=2026-10-11/70`.
- **Contents:** `dataset_2_moaks` plus the 4 fullfit hybrids (MOAKS-pretrained ConvNeXt × 2 and HGNetV2 × 2; the 58 trained on their radiologist labels).
- **`recipe.json` (default, the user's choice):** each of the 4 hybrids is one member of every ROI finding's rank average (0.945 / 0.952 offline).
- **`recipe_all8.json`:** the hybrid ensemble as one member (0.950 / 0.950).
- **`recipe_acl_moa.json`:** the same, on ACL and medial OA only (0.958 / 0.945).
