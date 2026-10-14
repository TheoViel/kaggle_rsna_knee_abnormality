# ROI models vs b1_c1 on the 58 radiologist studies

AUC per finding on the 58 studies of `train.csv` with radiologist labels. The ROI models are never trained on these studies.

- **What the scores use.** Every ensemble is scored with the non-fullfit versions of its models. The submission folders hold fullfit versions, which were trained on the 58 and can't be scored there.
- **How ensembles combine.** Findings with several models in the recipe use a rank average of their predictions.
- **Noise.** With 58 studies, a 0.01–0.02 gap on a single finding can be noise.

Bold marks the best in each column.

## Submissions

Each row runs the full inference pipeline (`src/infer/pipeline.py`) from the DICOMs. Findings an ensemble does not predict are set to 0.5.

The last block replaces the 4 non-ROI findings by b1_c1's predictions, the configuration that was submitted. **LB** is the public leaderboard score where one exists.

| On the 58 | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | Effusion | Synovitis | Contusion | Fracture | **8 ROI** | **12** | **LB** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1_c1 | 0.977 | 0.964 | 0.966 | 0.865 | **0.986** | **0.876** | 0.870 | 0.976 | **0.989** | **0.851** | **0.955** | **0.953** | 0.935 | 0.936 | |
| *ROI models only (0.5 where no model)* | | | | | | | | | | | | | | | |
| 12 frames (`output/dataset_1`) | 0.972 | **0.994** | 0.975 | 0.920 | 0.974 | 0.872 | 0.897 | 0.986 | 0.5 | 0.5 | 0.5 | 0.5 | 0.949 | 0.799 | |
| 8 frames (`output/kaggle_weights`) | 0.972 | **0.994** | **0.977** | **0.934** | 0.974 | 0.868 | **0.905** | **0.991** | 0.5 | 0.5 | 0.5 | 0.5 | 0.952 | 0.801 | |
| 8 frames + capsule (`output/dataset_2`) | 0.972 | **0.994** | **0.977** | **0.934** | 0.974 | 0.868 | **0.905** | **0.991** | 0.975 | 0.786 | 0.5 | 0.5 | 0.952 | 0.865 | |
| All 12 (`output/dataset_3`) | 0.972 | **0.994** | **0.977** | **0.934** | 0.974 | 0.868 | **0.905** | **0.991** | 0.975 | 0.786 | 0.942 | 0.947 | 0.952 | 0.939 | |
| All 12 + new OA (`output/dataset_4`) | 0.972 | **0.994** | **0.977** | **0.934** | 0.978 | **0.903** | **0.905** | **0.991** | 0.975 | 0.786 | 0.942 | 0.947 | 0.957 | 0.942 | |
| All 12 + new OA + new ACL (`output/dataset_5`) | **0.993** | **0.994** | **0.977** | **0.934** | 0.978 | **0.903** | **0.905** | **0.991** | 0.975 | 0.786 | 0.942 | 0.947 | **0.959** | **0.944** | |
| + MCL HGNetV2, menisci 2 seeds (`output/dataset_6`) | **0.993** | **0.994** | 0.975 | **0.937** | 0.978 | **0.903** | **0.905** | **0.991** | 0.975 | 0.786 | 0.942 | 0.947 | **0.960** | **0.944** | |
| + MOAKS-pretrained OA models (`output/dataset_7`) | **0.993** | **0.994** | 0.975 | **0.937** | 0.983 | 0.901 | **0.914** | **0.991** | 0.975 | 0.786 | 0.942 | 0.947 | **0.961** | **0.945** | |
| *8 ROI models + b1_c1 for effusion, synovitis, contusion, fracture* | | | | | | | | | | | | | | | |
| `dataset_1` + b1_c1 (submitted) | 0.972 | **0.994** | 0.975 | 0.920 | 0.974 | 0.872 | 0.897 | 0.986 | **0.989** | **0.851** | **0.955** | **0.953** | 0.949 | 0.945 | **0.962** |
| `kaggle_weights` + b1_c1 | 0.972 | **0.994** | **0.977** | **0.934** | 0.974 | 0.868 | **0.905** | **0.991** | **0.989** | **0.851** | **0.955** | **0.953** | 0.952 | 0.947 |  |
| `dataset_4` + b1_c1 | 0.972 | **0.994** | **0.977** | **0.934** | 0.978 | **0.903** | **0.905** | **0.991** | **0.989** | **0.851** | **0.955** | **0.953** | 0.957 | 0.950 |  |
| `dataset_5` + b1_c1 | **0.993** | **0.994** | **0.977** | **0.934** | 0.978 | **0.903** | **0.905** | **0.991** | **0.989** | **0.851** | **0.955** | **0.953** | **0.959** | **0.952** |  |
| `dataset_6` + b1_c1 | **0.993** | **0.994** | 0.975 | **0.937** | 0.978 | **0.903** | **0.905** | **0.991** | **0.989** | **0.851** | **0.955** | **0.953** | **0.960** | **0.952** |  |
| `dataset_7` + b1_c1 | **0.993** | **0.994** | 0.975 | **0.937** | 0.983 | 0.901 | **0.914** | **0.991** | **0.989** | **0.851** | **0.955** | **0.953** | **0.961** | **0.953** |  |
| `dataset_2_moaks` + b1_c1 (submitted; `dataset_2` with MOAKS-pretrained OA models) | 0.972 | **0.994** | **0.977** | **0.934** | 0.977 | 0.878 | **0.923** | **0.991** | **0.989** | **0.851** | **0.955** | **0.953** | 0.956 | 0.950 | `dataset_2` + ~0.001 |
| `dataset_2_moaks_mrnet` + b1_c1 (submitted; `dataset_2_moaks` + ACL with MRNet + MCL with MRNet pseudo-labels) | 0.983 | **0.996** | 0.978 | 0.945 | 0.977 | 0.878 | 0.923 | **0.991** | **0.989** | **0.851** | **0.955** | **0.953** | 0.959 | | `dataset_2_moaks` + 0.0001 |
| `dataset_2_moaks_mrnet_ext` + b1_c1 (+ OAI/MRNet `all` for med. men. / med. OA) | 0.983 | **0.996** | **0.982** | 0.945 | **0.980** | 0.878 | 0.923 | **0.991** | **0.989** | **0.851** | **0.955** | **0.953** | **0.960** | | |
| `dataset_2_moaks_mrnet_men` + b1_c1 (submitted; + MOAKS menisci + CAFormer menisci, lat. men. on sag + cor) | 0.983 | **0.996** | **0.983** | **0.959** | 0.977 | 0.878 | 0.923 | **0.991** | **0.989** | **0.851** | **0.955** | **0.953** | **0.961** | | `dataset_2_moaks_mrnet` + 0.0001 |
| `dataset_2_moaks_mrnet_men_frac` (+ 9-ROI fracture models; b1_c1 for effusion, synovitis, contusion) | 0.983 | **0.996** | **0.983** | **0.959** | 0.977 | 0.878 | 0.923 | **0.991** | **0.989** | **0.851** | **0.955** | 0.961 | **0.961** | | `dataset_2_moaks_mrnet_men` - 0.0001 |
| `dataset_2_moaks_mrnet_men_frac_cont` (+ capsule contusion models, report labels; contusion and fracture rank-blended with b1_c1) | 0.983 | **0.996** | **0.983** | **0.959** | 0.977 | 0.878 | 0.923 | **0.991** | **0.989** | **0.851** | **0.960** | **0.967** | **0.961** | | |

Note: the `dataset_2_moaks_mrnet*` rows are on a different scale from the rows above: recipe-level evaluation of the validation (non-fullfit) runs (`src/ext/recipe_eval.py`, external_data.md §6.4). On that scale `dataset_2_moaks` scores 0.957 (ACL 0.975, MCL 0.991, med. men. 0.978, lat. men. 0.945, med. OA 0.977, lat. OA 0.878, PF OA 0.923, Baker's 0.991); compare against that. The 12-finding column is left empty.

What each folder contains:

- **`dataset_1`:** the round-2 crop models on 12 uniform frames over each ROI box.
- **`kaggle_weights`:** the same model setups on 8 adjacent slices centred on each ROI. Runs 79–84 and 75, fullfit as 85–91.
- **`dataset_2`:** adds one capsule crop per series for effusion and synovitis.
  - The crop is the union of the 8 ROI boxes, widened 15% per side, with 24 frames at 160 px.
  - Models: runs 10/14 and 10/22, fullfit as 10/23 and 10/24.
- **`dataset_3`:** adds contusion and fracture from the same joint crop, including axial series. Run 10/27, fullfit as 10/28.
- **`dataset_4`:** replaces the medial and lateral OA models.
  - Both findings use 4 OA models whose crops stop at the intercondylar midline: runs 10/37, 10/40, 10/42 and 10/43, fullfit as 10/46, 10/48, 10/49 and 10/50.
  - Lateral OA subtracts 0.4 × the medial OA score (β chosen on the holdout), then is converted back to [0, 1] as percentile ranks. See `experiments.md` §14.
  - PF OA is unchanged.
- **`dataset_5`:** replaces the ACL models.
  - ACL now uses 4 ACL-only models with axial crops, one per architecture: ConvNeXt, CAFormer-S18, ConvNeXt with a transformer head, and HGNetV2-B2.
  - Runs 10/65, 10/77, 10/78 and 10/79, fullfit as 10/81–10/84.
  - The `all` model is no longer used for ACL. See `experiments.md` §15.
- **`dataset_6`:** two changes, both chosen on the holdout.
  - MCL: an HGNetV2-B2 specialist with axial crops (run 10/87, fullfit 10/98) replaces the ConvNeXt one. MCL holdout 0.942 → 0.954.
  - Menisci: a second seed of the CoAtNet specialist (run 10/97, fullfit 10/99) is added. Seeds differ by ±0.03 on the lateral meniscus on the 58, so averaging two makes it more robust.
  - On the 58 it scores the same as `dataset_5`.
- **`dataset_7`:** adds MOAKS-pretrained versions of the OA models.
  - Each crop model was first trained on OAI (2,400 knees, 25 MOAKS gradings), then fine-tuned on our data.
  - Medial/lateral OA use 8 models (the 4 current + 4 MOAKS-pretrained); PF OA uses 2 (90 + its MOAKS version).
  - Fullfits: 2026-10-11/24–28.
  - On the 58: +0.001 (8 ROI), mostly PF OA 0.905 → 0.914 and medial OA 0.978 → 0.983. Holdout flat. See `experiments.md` §16.

Findings:

- **8 ROI findings:** the ROI models beat b1_c1, 0.952 vs 0.935. The largest gaps are lateral meniscus (+0.069), PF OA (+0.035) and MCL (+0.030). b1_c1 stays ahead on ACL, medial OA and lateral OA.
- **Effusion:** 0.975, close to b1_c1's 0.989.
- **Synovitis:** the clear gap, 0.786 vs 0.851.
- **Contusion and fracture:** 0.942 and 0.947, slightly below b1_c1's 0.955 and 0.953.
- **All 12 findings:** the pure ROI ensemble passes b1_c1: 0.939 with `dataset_3`, **0.942** with `dataset_4`, vs 0.936.
- **Leaderboard:** `dataset_1` + b1_c1 fallback scored **0.962**, against 0.945 for the same configuration on the 58. With the same fallback, `dataset_4` scores 0.950 on the 58, +0.005 over `dataset_1`. That gain comes from the 8-frame crops (lateral meniscus, PF OA, Baker's) and the new OA models (lateral OA).
- **Lateral OA:** the `dataset_4` OA models beat b1_c1, 0.903 vs 0.876. Medial OA is still 0.008 short (0.978 vs 0.986).
- **ACL:** the `dataset_5` models score 0.993, vs 0.977 for b1_c1. The ROI models now beat b1_c1 on 7 of the 8 ROI findings; medial OA is the exception.

## CAFormer-S18 and HGNetV2-B2

These two backbones were trained as single "all" models (8 ROI findings, 8 frames). Each row puts them alone, in place of the ConvNeXt "all" model in the recipe, or blended into the 8-frame recipe.

- **Source of the scores.** They come offline from each run's validation predictions, which is why the reference recipe reads 0.953 here vs 0.952 through the pipeline above.
- **The four non-ROI findings.** In every row, effusion and synovitis come from the capsule models (10/14 + 10/22), and contusion and fracture from the joint model (10/27). The 12-finding column is therefore comparable across rows.
- **Blends.** The extra model is added to every ROI finding's list in the per-finding rank average, with the same weight as each recipe model.

| On the 58 | ACL | MCL | Med. men. | Lat. men. | Med. OA | Lat. OA | PF OA | Baker's | Effusion | Synovitis | Contusion | Fracture | **8 ROI** | **12** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1_c1 | 0.977 | 0.964 | 0.966 | 0.865 | **0.986** | **0.876** | 0.870 | 0.976 | **0.989** | **0.851** | **0.955** | **0.953** | 0.935 | 0.936 |
| *Single "all" models (8 frames)* | | | | | | | | | | | | | | |
| ConvNeXt-tiny (`09/75`) | 0.973 | 0.984 | 0.976 | 0.935 | 0.969 | 0.859 | 0.880 | **0.993** | 0.975 | 0.781 | 0.942 | 0.946 | 0.946 | 0.934 |
| CAFormer-S18 (`10/5`) | 0.977 | 0.991 | 0.966 | 0.937 | 0.974 | 0.868 | 0.887 | 0.991 | 0.975 | 0.781 | 0.942 | 0.946 | 0.949 | 0.936 |
| HGNetV2-B2 (`10/6`) | **0.987** | 0.984 | 0.977 | 0.918 | 0.967 | 0.839 | 0.900 | **0.993** | 0.975 | 0.781 | 0.942 | 0.946 | 0.946 | 0.934 |
| *Recipe variants* | | | | | | | | | | | | | | |
| 8-frame recipe (reference, `kaggle_weights`) | 0.973 | 0.995 | 0.978 | 0.938 | 0.974 | 0.868 | **0.906** | 0.991 | 0.975 | 0.781 | 0.942 | 0.946 | **0.953** | **0.939** |
| Recipe, CAFormer replaces the ConvNeXt "all" model | 0.974 | **0.997** | 0.975 | 0.934 | 0.974 | 0.868 | **0.906** | 0.990 | 0.975 | 0.781 | 0.942 | 0.946 | 0.952 | 0.938 |
| Recipe, HGNetV2 replaces the ConvNeXt "all" model | 0.980 | 0.994 | 0.980 | 0.916 | 0.972 | 0.868 | **0.906** | **0.993** | 0.975 | 0.781 | 0.942 | 0.946 | 0.951 | 0.938 |
| *Blends* | | | | | | | | | | | | | | |
| Recipe + CAFormer | 0.974 | 0.995 | 0.976 | **0.941** | 0.974 | 0.871 | 0.898 | **0.993** | 0.975 | 0.781 | 0.942 | 0.946 | **0.953** | **0.939** |
| Recipe + HGNetV2 | 0.979 | 0.992 | **0.984** | 0.935 | 0.974 | 0.852 | 0.900 | **0.993** | 0.975 | 0.781 | 0.942 | 0.946 | 0.951 | 0.938 |
| Recipe + CAFormer + HGNetV2 | 0.978 | 0.995 | 0.981 | 0.940 | 0.974 | 0.859 | 0.892 | **0.993** | 0.975 | 0.781 | 0.942 | 0.946 | 0.951 | 0.938 |

Findings:

- **As single models,** CAFormer-S18 is the best "all" model (0.949 vs 0.946 for ConvNeXt). HGNetV2-B2 ties ConvNeXt with about a quarter of the FLOPs.
- **Inside the ensemble, the gains vanish.** Every swap and blend lands within 0.002 of the 8-frame recipe, which is noise on 58 studies.
- **The per-finding trade-offs cancel out.**
  - HGNetV2 helps ACL and the medial meniscus but costs lateral OA and the lateral meniscus.
  - CAFormer helps the lateral meniscus but costs PF OA.
- **Recommendation:** keep the current 8-frame recipe. If inference time becomes a concern, swapping in HGNetV2 is a cheaper option at essentially the same score (0.951 vs 0.953).
