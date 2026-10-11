# External data: OAI, SKM-TEA, MRNet

Status as of 2026-10-11:
- External data is allowed in the competition.
- The only external data in a submission so far is MOAKS (OAI) pretraining, in the OA models of `output/dataset_2_moaks`. It was submitted: **public LB up by almost 0.001 over `dataset_2`**, the first external-data change confirmed on the leaderboard.

All commands run from `src/` with `PYTHONPATH=.` and `../.venv/bin/python`; `torchrun` means `../.venv/bin/torchrun --standalone --nproc_per_node=4`. Scores are AUC on the 58 radiologist-labelled studies / holdout (Claude labels), from the non-fullfit validation runs.

## 1. Summary

| Source | What it is | Best use found | Effect |
|---|---|---|---|
| **OAI** (MOAKS readings) | 2,400 knees, 5 series each (3 T research protocol), semi-quantitative MOAKS gradings | Pretraining on 25 fine-grained MOAKS gradings, then fine-tuning on our data | About +0.005 on single OA models (PF OA most). +0.001 to +0.002 at recipe level. `dataset_2_moaks`: public LB +~0.001 over `dataset_2` |
| **SKM-TEA** | 155 knees, 3D sagittal qDESS, expert 3D pathology boxes | none | Neutral for ACL, menisci and OA at weights 1, 4 and 8 (weight 4 menisci looked +0.010 on one seed; it did not hold over 3) |
| **MRNet** | 1,250 clinical exams (sagittal, coronal and axial `.npy` volumes), binary labels: abnormal / ACL tear / meniscal tear | Joint training for **ACL**; MRNet pseudo-labels for **MCL**; with OAI in a second "all" model for the medial meniscus and medial OA | ACL: mean of 3 seeds 0.984 with MRNet loss weight 2 (0.978 at weight 1) vs 0.973 for the reference; MCL holdout +0.006 to +0.014 (3 seeds); both in `output/dataset_2_moaks_mrnet`. Medial meniscus / OA +0.002 to +0.004 from the second "all" model: `output/dataset_2_moaks_mrnet_ext` |

What doesn't work: joint training on labels derived from OAI, OAI as a pseudo-label teacher, OAI axial (DESS reformat) crops for PF OA, SKM-TEA labels, and pretraining on MRNet only.

**Selection with external data.** External labels follow a different convention from the report-based weak labels the holdout is scored against, so a holdout drop is expected. Select on the 58 (with seeds) and the public leaderboard; don't reject a model for a holdout drop alone.

**Lesson from the leaderboard.** Small gains on the 58 or the holdout have not transferred to the leaderboard. The lateral OA correction and the ACL ensemble regressed there. Judge with several seeds and prefer robust, untuned changes.

## 2. Label convention for external data (mandatory)

External labels must follow the competition grading convention in `rsna-knee-abnormality-detection_competition_overview.md`, with **graded intermediate targets**:
- **1:** convention-positive (high-grade or complete ACL tear; surface-reaching meniscal tear; ≥ ~1 cm area of > 50% thickness cartilage loss; moderate or large effusion or Baker's cyst, …).
- **0:** truly healthy.
- **0.25 / 0.5:** intermediate grades (low-grade sprain, mild effusion, partial-thickness cartilage loss, meniscal signal without a tear, …).

Do **not** exclude intermediate grades and do **not** map them to 0; the user rejected both. With several readers, take the mean. A finding the source cannot express is NaN, and NaN targets are dropped from training.

## 3. Shared pipeline

External series go through the same steps as test-time series, so their crops match ours:
1. **Cache and volume.**
   - Slices are ordered.
   - `prepare_cache.preprocess_volume` builds the 384 px uint8 cache: `output/<data>/cache_384/<SeriesInstanceUID>.npy`.
   - `roi.prepare.cache_to_volume` builds the 32 × 128 × 128 volume, in the training left-right convention (`conv_flip`), stored in `output/<data>/volumes.npy`.
   - **Thin 3D series** (OAI DESS 0.7 mm and its reformats, SKM-TEA 0.8 mm) are first averaged into ~3 mm slabs (`oai.prep.slab_average`), so that 8 strided slices cover the same anatomy as on clinical series.
2. **Side.** The side classifier (`logs/side_run/2026-10-09/0`), or the known side for OAI.
3. **ROI boxes.** The ROI U-Net (`logs/roi_run/2026-10-09/1`) on sagittal and coronal series gives 8 boxes per series.
4. **Crop sets** (`crops.prepare`), with the same sampling as the training crop set they extend: `f_s1n8` = 8 strided frames (stride 1), 128 px, margin 0.25. Each crop set writes `output/crops/<name>.{npy,csv,json}`.
5. **Labels.** One CSV with columns `StudyInstanceUID` + finding names. StudyInstanceUIDs are `OAI_<id>_<RIGHT|LEFT>`, `SKM_MTR_xxx` and `MRNET_xxxx`.

Training with external data (`main_crops.py`):

| Flag | Effect |
|---|---|
| `--extra_crops A,B` | Extra crop sets |
| `--extra_labels file.csv` | Their labels |
| `--extra_weight w` | Loss weight of the external samples |
| `--extra_only True` | External data only (pretraining) |
| `--init <run dir>` | Initial weights. Only the weights whose shapes match are loaded, so the finding embeddings can differ |
| `--extra_seqs` | OAI sequences to keep |

- Validation always stays on our 58 + holdout.
- With `--fullfit True`, external studies keep their own labels: there are no radiologist labels for them.

## 4. OAI (Osteoarthritis Initiative)

### 4.1 Reproduction

**Prerequisites:**
- An NDA account with access to the OAI collection.
- The NDA download tool: `python -m venv ../.nda-venv && ../.nda-venv/bin/pip install nda-tools`.
- Credentials: the NDA username is `theo_viel_enpc`; `downloadcmd` keeps the password in the Python keyring.
- Never print the keyring contents.

**Inputs:**
- **Tabular data** (supplied by the user in `input/oai/`): `Semi-Quant Scoring_ASCII/kMRI_SQ_MOAKS_BICL00.txt` holds the baseline MOAKS readings. Also present: `MR Image Assessment_ASCII`, `OAI Complete Data_ASCII`, `Quant_ASCII`.
- **Image package:** NDA package **1250166**, the baseline. Package 1250165 was the screening package, which is not needed.

**Steps:**

```bash
# 1. Package manifest (image03.txt), then the 12,053 series of the 2,400 MOAKS knees (~132 GB)
../.nda-venv/bin/downloadcmd -dp 1250166 -u theo_viel_enpc -d ../input/oai/images/baseline --file-regex '.*\.(txt|pdf)$'
python -m oai.select_series        # -> input/oai/images/baseline_moaks_series{.csv,_s3.txt}
../.nda-venv/bin/downloadcmd -dp 1250166 -u theo_viel_enpc -t ../input/oai/images/baseline_moaks_series_s3.txt \
    -d ../input/oai/images/baseline -wt 16
# (re-run with --verify to fetch missing files)

# 2. Caches / volumes (slab averaging of the thin series), then sides, boxes and crops.prepare metadata
python -m oai.prep --workers 64    # -> output/oai/{cache_384, volumes.npy, series.csv, geometry.json}
python -m oai.boxes                # -> output/oai/{rois.csv, rois_axial.csv, side_check.csv, sides.csv, volumes.csv, volumes_axial.csv}

# 3. Labels
python -m oai.labels --graded      # graded convention labels -> output/oai/labels_moaks00_v4.csv (the ones to use)
python -m oai.labels --aux         # 25 MOAKS pretraining targets -> output/oai/aux_moaks00.csv

# 4. Crop sets (match the competition crop sets they are trained with)
P="python -m crops.prepare --cache_dir ../output/oai/cache_384 --side_csv ../output/oai/sides.csv"
$P --name oai_s1n8        --volumes ../output/oai/volumes.csv       --boxes ../output/oai/rois.csv       --depth 8 --z_mode strided --stride 1
$P --name oai_s1n8_axial  --volumes ../output/oai/volumes_axial.csv --boxes ../output/oai/rois_axial.csv --depth 8 --z_mode strided --stride 1
$P --name oai_cmp30_mid   --volumes ../output/oai/volumes.csv       --boxes ../output/oai/rois.csv       --depth 8 --z_mode strided --stride 1 --compartment 0.3 --oa_midline slide
$P --name oai_mid_slide   --volumes ../output/oai/volumes.csv       --boxes ../output/oai/rois.csv       --depth 8 --z_mode strided --stride 1 --oa_midline slide
$P --name oai_axial_mid   --volumes ../output/oai/volumes_axial.csv --boxes ../output/oai/rois_axial.csv --depth 8 --z_mode strided --stride 1 --oa_midline slide
```

**Pipeline checks:**
- The side classifier agrees with the known side on 98–100% of series.
- Mean ROI U-Net confidence is 0.968 on every sequence (DESS included); 2 weak boxes out of 77,208.
- Axial boxes come from registration onto the AX_MPR reformats (`infer.classify.axial_boxes`).

### 4.2 Labels

**Series and knees:**
- 5 series per knee: SAG_IW_TSE, COR_IW_TSE, SAG_3D_DESS, COR_MPR, AX_MPR.
- One "study" = one knee, `OAI_<id>_<side>`.

**Graded v4 rules** (`oai.labels --graded`):

| Finding | Rule |
|---|---|
| Medial / lateral / PF OA | 1 if a cartilage lesion covers ≥ 10% of a subregion (MOAKS size ≥ 2) with full-thickness loss; 0.5 if a large partial-thickness lesion or a small full-thickness one; 0.25 if a small partial-thickness lesion; 0 if no lesion. MOAKS has no ">50% depth" grade, only full thickness |
| Menisci | tear / maceration / root tear 1; intrasubstance signal only 0.25; normal 0 |
| ACL | complete tear 1; partial tear 0.5; intact 0 |
| Effusion (whole-knee effusion-synovitis), synovitis (Hoffa) | grade ≥ 2 → 1; mild → 0.25; 0 → 0 |
| Baker's | popliteal cyst present 1 |
| MCL, contusion, fracture | NaN (no MOAKS counterpart) |

- **Compartments:** medial = FMC FMP TMA TMC TMP; lateral = FLC FLP TLA TLC TLP; PF = FMA FLA PM PL.
- **Multiple reading projects:** mean.
- **Superseded files** (don't use): `labels_moaks00.csv` / `_v2` (a liberal OA definition with osteophytes and bone marrow lesions, PF OA prevalence 0.55 vs ~0.27 in our data), `_noPF`, `_strictPF`. The v3 exclusion labels were deleted.

**Auxiliary targets** (`aux_moaks00.csv`, group `moaks_aux` in `crops/train.py`):
- 25 gradings scaled to [0, 1], each tied to the ROI crop it describes (`"<ROI>|<feature>"`).
- Per compartment: cartilage size / full thickness / sum, bone marrow lesions, osteophytes.
- Per meniscus: tear, maceration, signal, extrusion.
- ACL tear; Baker's cyst.

### 4.3 What was tried

| Strategy | Result (58 / holdout) |
|---|---|
| Joint training, OA model, liberal v2 labels | lateral OA +0.03 for one seed, −0.02 for another; the 4-model OA ensemble is level on the 58 and −0.007 to −0.018 on the holdout |
| Joint training, OA, graded v4 labels | neutral: run-40 setup 0.919 / 0.925 vs 0.917 / 0.928; run-83 setup 0.907 / 0.928 vs 0.914 / 0.926. Better than v2 on the holdout |
| Joint, TSE sequences only; OAI weight 0.5 / 2 | no better |
| OAI-only model (transfer test) | medial OA 0.913, lateral 0.772, PF 0.829 on the 58 without any of our data. Medial–lateral prediction correlation **0.14** (models trained on our labels: 0.84) |
| OAI pretraining on the same labels → fine-tuning | 0.922 / 0.920 vs 0.917 / 0.928 |
| OAI-only model as a pseudo-label teacher | the labels improve, the models trained on them do not |
| Joint training: "all" model, menisci, capsule (effusion / synovitis), ACL, Baker's | no consistent gain; synovitis holdout −0.018 |
| Joint training with OAI axial (DESS reformat) crops for PF OA | **PF OA −0.014 on both**, whatever the PF label. The axial reformats are the problem |
| **MOAKS multi-task pretraining → fine-tuning** | OA models: medial OA +0.002 to +0.017, PF OA +0.014 to +0.026 on the 58 in all 6 OA models; lateral OA +0.003 to +0.013 on the holdout. "All" model 0.952 / 0.943–0.945 vs 0.946 / 0.942 (2 seeds). Menisci CoAtNet: within seed range. 20-epoch pretraining = 10 epochs |
| Multi-task joint training (`oa_moaks`, `all_moaks`, OAI weight 0.25) | about the same as pretraining then fine-tuning: OA 0.921 / 0.932 and 0.910 / 0.928 (2 seeds) vs 0.917 / 0.928 |
| Recipe level (`dataset_6` recipe) | adding MOAKS-pretrained OA models gives 0.9624 / 0.9501 vs 0.9610 / 0.9498: noise level |

**MOAKS pretraining recipe** (run `logs/crops_run/2026-10-11/8`, used by `dataset_2_moaks`):

```bash
torchrun main_crops.py --group moaks_aux --crops oa_cmp30_mid --extra_crops oai_cmp30_mid \
    --extra_labels ../output/oai/aux_moaks00.csv --extra_only True --planes Sagittal,Coronal \
    --backbone convnext_tiny.fb_in22k --epochs 10 --lr 0.0005 --lr_backbone 0.0001 --batch_size 32 --labels pl:pl_round2
# then any model with the reference setup + --init ../logs/crops_run/2026-10-11/8
# (dataset_2_moaks: mk89 = run 82 setup, oa 224 px; mk90 = run 83 setup, oa + axial; both --fullfit True)
```

## 5. SKM-TEA

### 5.1 Reproduction

**Data:**
- SKM-TEA on Redivis (Stanford AIMI): `aimi.skm_tea:5r8z:v1_0.dicom:efwx`, 155 `MTR_*.tar.gz` archives, 19 GB.
- Pathology annotations (`train.json`, `val.json`, `test.json`, COCO-style 3D boxes; `scan_labels.csv`) go into `input/skm_tea/`. The user put them there.
- **Redivis token:** a token with the AIMI data use agreements accepted, passed via the environment. The current one is in `/root/.redivis_token` (mode 600); load it with `REDIVIS_API_TOKEN=$(cat /root/.redivis_token)` and never echo it.

```bash
REDIVIS_API_TOKEN=$(cat /root/.redivis_token) python -m ext.download --data skm   # -> input/skm_tea/dicom/
python -m skm.prep --workers 32        # DESS echo 2, slab average k=4 (160 -> 40 slices) -> output/skm/{cache_384, volumes.npy, series.csv, geometry.json}
python -m ext.boxes --data skm         # side classifier + U-Net -> output/skm/{sides.csv, volumes.csv, rois.csv}
python -m skm.labels                   # meniscus positions (mm_z / lm_z into series.csv) + graded labels -> output/skm/labels_graded.csv
python -m crops.prepare --name skm_s1n8 --cache_dir ../output/skm/cache_384 --side_csv ../output/skm/sides.csv \
    --volumes ../output/skm/volumes.csv --boxes ../output/skm/rois.csv --depth 8 --z_mode strided --stride 1
```

**Details:**
- **Series and side:** one sagittal series per knee (81 left, 74 right). Echo 2 is T2-weighted with bright fluid, the closest to a clinical fluid-sensitive sagittal.
- **ROI boxes:** no registration fill, since there is a single series. All 8 ROIs are found on every scan, with mean confidence 0.955–0.979.
- **Medial vs lateral placement** of annotation boxes uses the nearest U-Net meniscus centre. It agrees with side + DICOM geometry on 155 / 155 scans. Raw slice indices increase toward patient right on every scan.

### 5.2 Labels (`skm.labels`)

| Finding | Rule |
|---|---|
| ACL | low-grade sprain 0.25, moderate (partial) tear 0.5, complete tear 1 |
| Menisci (side by position) | tear 1; myxoid degeneration or extrusion only 0.25 |
| OA | cartilage lesions; patellar → PF, femoral / tibial → medial / lateral by position. Grade 2B / 3 over ≥ 10 mm → 1; 2B / 3 smaller, or 2A over ≥ 10 mm → 0.5; otherwise 0.25 |
| Effusion | annotated → 1 |
| MCL, synovitis, Baker's, contusion, fracture | NaN |

- Scans not inspected are NaN; 145 / 155 are labelled.
- Box format: `[SI, AP, LR start, sizes]`; reversed boxes are fixed with `min` / `abs`.
- Mean targets: ACL 0.14, medial / lateral meniscus 0.31 / 0.22, medial / lateral / PF OA 0.22 / 0.23 / 0.17.

### 5.3 Results (joint training, `--extra_crops skm_s1n8 --extra_labels ../output/skm/labels_graded.csv`)

| Model | Reference | + SKM-TEA |
|---|---|---|
| ACL (run 79 setup) | 0.973 / 0.992 | 0.971 / 0.992 |
| Menisci CoAtNet (run 81 setup) | 0.945 / 0.939 | 0.944 / 0.940 |
| OA 224 px (run 82 setup) | 0.913 / 0.929 | 0.911 / 0.929 |

- Neutral everywhere at weight 1. Adding SKM-TEA to MRNet also cancels part of MRNet's ACL gain (see 6.3).
- With 155 scans of one research sequence, weight 1 is too little to matter. At weight 4 (`--extra_weight 4.0`), menisci seed 0 reached 0.957 / 0.939 (lateral meniscus 0.955 in the recipe), but seeds 1 and 2 gave 0.948 and 0.941: seed luck. Weight 8: 0.954, flat in the recipe. OA at weight 4 stays neutral (0.914 / 0.928).
- **Not tried:** using SKM-TEA's expert boxes to check or train the ROI localisation.

## 6. MRNet

### 6.1 Reproduction

**Data:**
- MRNet on Redivis: `aimi.mrnet_knee_mri_s:4a2c:v1_0`; image tables `train:5wjf` and `valid:zcmk`, plus 6 label tables.
- 1,250 exams (1,130 train + 120 valid; both are used for training here). Each has sagittal T2 fat-suppressed, coronal T1 and axial PD fat-suppressed series: uint8 arrays of slices × 256 × 256, about 30–35 slices.
- No DICOM geometry.

```bash
REDIVIS_API_TOKEN=$(cat /root/.redivis_token) python -m ext.download --data mrnet  # -> input/mrnet/
python -m mrnet.prep --workers 32      # -> output/mrnet/{cache_384, volumes.npy, series.csv, labels.csv}
python -m ext.boxes --data mrnet       # -> output/mrnet/{sides.csv, volumes.csv, rois.csv}
python -m mrnet.labels                 # -> output/mrnet/labels_graded.csv + output/ext/{skm_mrnet_labels_graded, oai4_mrnet_labels}.csv
python -m crops.prepare --name mrnet_s1n8 --cache_dir ../output/mrnet/cache_384 --side_csv ../output/mrnet/sides.csv \
    --volumes ../output/mrnet/volumes.csv --boxes ../output/mrnet/rois.csv --depth 8 --z_mode strided --stride 1

# Label variants (6.2). The model-based ones need our reference runs' predictions on the MRNet crops first:
python -m crops.predict --runs 75,79,80,81,82,84 --gpus 0 --crops mrnet_s1n8   # -> logs/crops_run/2026-10-09/<run>/mrnet_s1n8_study_preds.csv
python -m mrnet.labels --men_side ../logs/crops_run/2026-10-09/81/mrnet_s1n8_study_preds.csv --name labels_menpl
python -m mrnet.labels --men_soft 0.75,0.4 --name labels_mensoft
python -m mrnet.labels --pl --name labels_pl
```

**Orientation** (no geometry):
- No `conv_flip` is applied. The side classifier canonicalises each series to a right-knee appearance whatever its stored orientation, which is all the crops need: MRNet labels have no side.
- **Side** = mean classifier probability over the 3 planes of an exam (the exam vote).
- Planes agree pairwise on 80–90% of exams, so MRNet stores all 3 planes in the same convention.
- The coronal T1 is the least confident: only 53% of coronal series are called with confidence, vs 96% of sagittal ones.

**Boxes:**
- U-Net on sagittal and coronal; no registration fill, so no axial crops.
- All 8 ROIs on every series. Mean confidence 0.94–0.97, except coronal Baker's 0.84 and coronal PF 0.90.
- A visual check of the crops showed correct anatomy.

**Zero-shot check** (`python -m crops.predict --runs <run> --crops mrnet_s1n8`):
- Our menisci model (run 81) scores AUC **0.891** on MRNet meniscal tears, using max(medial, lateral). The published MRNet result is about 0.85.
- Our ACL model (run 79) scores **0.929** on all exams and 0.963 on MRNet's validation split (published: 0.965).
- This confirms the pipeline transfers.
- The weaker train-split ACL score suggests that MRNet's "tear" includes lower-grade tears our convention calls negative.

### 6.2 Labels (`mrnet.labels`)

MRNet labels are binary and report-derived, so they cannot be graded:

| Finding | Rule |
|---|---|
| ACL | the ACL label (0 / 1) |
| Menisci | no meniscal tear → both 0; tear → side unknown, NaN |
| Everything else | `abnormal = 0` (242 normal exams) → every finding 0; otherwise NaN |

- Result: ACL 1,250 labels (21% positive); menisci 801 negatives; 242 all-negative normal knees.

Label files (all from `mrnet.labels`; regenerating them reproduces the files used in training exactly):

| File | Content |
|---|---|
| `output/mrnet/labels_graded.csv` | The rules above (the default) |
| `output/mrnet/labels_menpl.csv` | `--men_side`: meniscus-positive exams take side from a model's medial / lateral predictions (run 81); the likelier side gets 1, the other keeps its predicted probability |
| `output/mrnet/labels_mensoft.csv` | `--men_soft 0.75,0.4`: meniscus-positive exams get fixed soft targets, medial 0.75 and lateral 0.4 (about the side prior: 73% of positives look medial to our menisci model) |
| `output/mrnet/labels_pl.csv` | `--pl`: **pseudo-labels** for every ROI finding. Unlabelled findings (MCL, OA, Baker's) = mean prediction of the recipe's reference runs on the MRNet crops (`PL_RUNS`: ACL 79 + 75, MCL 80 + 75, menisci 81 + 75, medial OA 82 + 75, lateral and PF OA 82, Baker's 84 + 75). MRNet labels override: ACL = label; menisci as `--men_side` with the same predictions; normal exams all 0. Mean targets: MCL 0.09, medial / lateral / PF OA 0.20 / 0.14 / 0.22, Baker's 0.14 |
| `output/ext/skm_mrnet_labels_graded.csv` | SKM-TEA graded + MRNet graded, for `--extra_crops skm_s1n8,mrnet_s1n8` |
| `output/ext/oai4_mrnet_labels.csv` | OAI graded v4 + MRNet graded, for `--extra_crops oai_s1n8,mrnet_s1n8` |

### 6.3 Results (joint training, `--extra_crops mrnet_s1n8 --extra_labels ../output/mrnet/labels_graded.csv`)

**ACL** (run 79 setup: `--group acl --crops f_s1n8 --labels pl:pl_round2`, ConvNeXt-tiny):

| Run | Seed 0 | Seed 1 | Seed 2 | Mean on the 58 |
|---|---|---|---|---|
| Reference | 0.973 / 0.992 | 0.974 / 0.992 | 0.972 / 0.993 | 0.973 |
| **+ MRNet** | 0.985 / 0.993 | 0.978 / 0.993 | 0.972 / 0.994 | **0.978** |
| + MRNet, weight 0.5 | 0.987 / 0.992 | | | |
| **+ MRNet, weight 2** (adopted) | **0.988** / 0.993 | 0.987 / 0.993 | 0.977 / 0.994 | **0.984** |
| + SKM-TEA | 0.971 / 0.992 | | | |
| + SKM-TEA + MRNet | 0.979 / 0.993 | 0.980 / 0.994 | | 0.980 |
| MRNet only (pretraining, no our data) | 0.896 / 0.905 | | | |
| MRNet pretraining → fine-tune on our data | 0.967 / 0.992 | | | |
| MRNet pretraining → fine-tune, joint with MRNet | 0.979 / 0.993 | | | |

- **MRNet helps ACL on average** (seed 2 only ties the reference); every joint variant is at or above the reference on the 58, and none hurts the holdout.
- In the recipe (rank average with the "all" model, as in `dataset_2`), the weight-2 runs score 0.983 / 0.981 / 0.978 on the 58 vs 0.975 for the original pair; the holdout is 0.994 for both.
- **Packaged** in `output/dataset_2_moaks_mrnet` (with the MCL model of 6.5): the ACL model `85` replaced by `mr85` (fullfit 2026-10-11/76, weight 2). Not uploaded.
- Pretraining then fine-tuning on our data alone loses the gain: MRNet has to stay in the training mix.
- ACL scores are near ceiling (the 58 have few ACL tears), so expect noise.
- Runs: `logs/crops_run/2026-10-11/50–54, 61–68`; logs in `logs/ext_*.log`.

**Menisci** (run 81 setup: CoAtNet-0, 224 px, batch 16):

| Run | Seed 0 | Seed 1 |
|---|---|---|
| Reference | 0.945 / 0.939 (medial 0.977, lateral 0.913) | 0.949 / 0.938 (0.975, 0.923) |
| **+ MRNet** | **0.951** / 0.941 (medial 0.972, lateral **0.929**) | 0.948 / 0.937 (0.972, 0.923) |
| + MRNet, side from model (`labels_menpl`) | 0.942 / 0.938 | 0.941 / 0.941 |
| + MRNet, soft side labels (`labels_mensoft`) | 0.953 / 0.941 (0.977, 0.929) | 0.940 / 0.938 (0.971, 0.909) |
| + MRNet, weight 2 | 0.951 / 0.938 (0.972, 0.929) | |
| + SKM-TEA | 0.944 / 0.940 | |
| **+ SKM-TEA, weight 4** | **0.957** / 0.939 (0.976, **0.938**) | *queued* (seeds 1, 2; weight 8) |
| + SKM-TEA + MRNet | 0.946 / 0.940 | |

- In the recipe (+ "all"), MRNet menisci models lose ~0.005 on the medial meniscus and gain ~0.003 on the lateral: net zero, so they are not used. (A menisci + MRNet fullfit was started and stopped at epoch 3: there is no such model.)
- **No real gain:** MRNet averages 0.950 / 0.939 over 2 seeds vs 0.947 / 0.939 for the reference, within seed noise (the reference alone spans 0.945–0.949).
- Assigning the side with our model does not help: its errors become labels.

- ACL + MRNet at weight 4: 0.985 / 0.992 and 0.983 / 0.994 (2 seeds), the same as weight 2.
- OA 224 px + SKM-TEA: weight 1 0.911 / 0.929, weight 4 0.914 / 0.928, vs 0.913 / 0.929. Neutral.
- OA 224 px + SKM-TEA + MRNet (normal knees as OA negatives): 0.909 / 0.930. Neutral.

### 6.4 Recipe-level evaluation and the "all" model

`python -m ext.recipe_eval` rebuilds `dataset_2_moaks`'s recipe from the non-fullfit validation runs (each finding = mean of its models' percentile ranks, as at inference) and swaps candidates in. This is the comparison that counts: a model only matters through the findings the recipe takes from it.

| Swap (58) | ACL | MCL | Med. men. | Lat. men. | Med. OA | Baker's | Macro 58 / hold |
|---|---|---|---|---|---|---|---|
| Base (`dataset_2_moaks`) | 0.975 | **0.991** | 0.978 | 0.945 | 0.977 | **0.991** | 0.9572 / 0.9460 |
| ACL + MRNet w2 (3 seeds) | **0.983** / 0.981 / 0.978 | | | | | | 0.9582 / 0.9580 / 0.9576 |
| Menisci + SKM-TEA w4 (1 seed) | | | 0.978 | **0.955** | | | **0.9585** / 0.9462 |
| "all" + OAI v4 (2 seeds) | 0.974 | 0.982 / 0.980 | 0.978 | 0.935 / 0.942 | **0.981** / 0.978 | 0.984 / 0.985 | 0.9545 / 0.9547 |
| "all" + MRNet w2 | 0.977 | 0.986 | 0.981 | 0.925 | 0.978 | 0.989 | 0.9545 / 0.947 |
| "all" + OAI v4 + MRNet | 0.974 | 0.977 | **0.982** | 0.936 | 0.980 | 0.989 | 0.9549 / **0.9476** |

- **"all" model with external data, as a replacement: rejected.** On its own (validation runs), "all" + OAI v4 gains about +0.03 on PF OA (0.913 / 0.907 vs 0.880) and "all" + OAI v4 + MRNet scores 0.953 / 0.946 vs 0.946 / 0.942. But the recipe takes PF and lateral OA from the OA models. Among the findings it does feed, MCL (−0.005 to −0.014), the lateral meniscus and Baker's drop: external data has no MCL labels and few Baker's ones.
- **Where it helps:** the medial meniscus (0.981–0.982 vs 0.978) and medial OA (0.978–0.981 vs 0.977). A per-finding recipe could take the external "all" model only for those two (about +0.001 on the macro, at the cost of one more ConvNeXt-tiny at inference), if seed 1 of "all" + OAI v4 + MRNet confirms (queued).
- Averaging both "all" models everywhere is weaker (MCL 0.984).

### 6.5 MRNet pseudo-labels

Pseudo-labelled MRNet (`labels_pl.csv`, see 6.2) turns the 1,250 clinical exams into training data for every ROI finding. Joint training (`--extra_crops mrnet_s1n8 --extra_labels ../output/mrnet/labels_pl.csv`), recipe level (58 / holdout):

| Model | Seeds | Recipe finding(s) | Base |
|---|---|---|---|
| **MCL** (run 80 setup) | 0.995 / 0.959, 0.993 / 0.958, 0.985 / 0.955 | MCL | 0.991 / 0.945, seed 1: 0.989 / 0.952 |
| "all" | macro 0.9570 / 0.9468, 0.9556 / 0.9473 | | 0.9572 / 0.9460 |
| OA 224 px | lateral OA 0.872 / 0.904 | | 0.878 / 0.900 |
| Baker's | 0.989 / 0.970 | | 0.991 / 0.971 |
| Menisci CoAtNet | 0.955 / 0.936 alone; medial meniscus 0.975 in the recipe | | 0.978 |

- **MCL is the one that works**: flat on the 58, where MCL is near ceiling (mean 0.991 vs 0.990), and +0.006 to +0.014 on the holdout in all 3 seeds. Pseudo-labels share the holdout's convention, so here the holdout gain is meaningful.
- The others are neutral or worse.
- Also checked with more seeds: "all" + OAI v4 + MRNet seed 1 confirms the medial meniscus (0.980 vs 0.978) and medial OA (0.980 vs 0.977) gains (6.4). Menisci + SKM-TEA weight 4 does **not** hold (seeds 0.957 / 0.948 / 0.941, recipe macro 0.9585 / 0.9563 / 0.9548), and weight 8 is flat (0.9573).

**Packaged** (not uploaded): `output/dataset_2_moaks_mrnet` = `dataset_2_moaks` + ACL with MRNet (`mr85`) + MCL with MRNet pseudo-labels (`pl86`, fullfit 2026-10-11/102), recipe level 0.9588 / 0.9477. `output/dataset_2_moaks_mrnet_ext` = the same + an "all" + OAI v4 + MRNet model (`ext88`, fullfit 2026-10-11/103) used for the medial meniscus and medial OA only. Recipe level: 0.9580–0.9596 / 0.9474–0.9478 depending on the seed pairing, vs 0.9572 / 0.9460. Smoke-tested through `infer.pipeline`.

### 6.6 Next steps

- **Candidate packages** (new folders only; submit one at a time to read the leaderboard):
  - `output/dataset_2_moaks_mrnet` = `dataset_2_moaks` + ACL trained with MRNet + MCL trained with MRNet pseudo-labels;
  - `output/dataset_2_moaks_mrnet_ext` = `dataset_2_moaks_mrnet` + external "all" model for the medial meniscus and medial OA (one extra ConvNeXt-tiny at inference).
  - Both are to be submitted together; the difference between them isolates the second "all" model.
  - Never modify submitted folders (`dataset_1`, `2`, `4`, `5`, `2_moaks`). Do not upload or submit without the user's go-ahead.
- **Untried ideas:**
  - MRNet "abnormal" as an auxiliary target;
  - MIL on meniscus-positive exams: loss on max(medial, lateral) instead of a side guess;
  - MRNet axial crops: they need boxes without geometry, e.g. a U-Net run on axial, or a rule from the sagittal / coronal boxes.

## 7. Files

| Path | Content |
|---|---|
| `src/oai/{select_series,prep,boxes,labels}.py` | OAI pipeline |
| `src/skm/{prep,labels}.py` | SKM-TEA pipeline |
| `src/mrnet/{prep,labels}.py` | MRNet pipeline |
| `src/ext/download.py` | Redivis downloads (SKM-TEA, MRNet) |
| `src/ext/boxes.py` | Side + U-Net boxes without consensus (SKM-TEA, MRNet) |
| `src/ext/recipe_eval.py` | Recipe-level 58 / holdout scores of `dataset_2_moaks` with swapped models (6.4) |
| `src/crops/predict.py --crops <set>` | Predictions of trained runs on another crop set (zero-shot checks, pseudo-labels) |
| `output/{oai,skm,mrnet}/` | Caches, volumes, metadata, boxes, labels |
| `output/crops/{oai_*,skm_s1n8,mrnet_s1n8}.*` | External crop sets |
| `output/mrnet/labels_{graded,menpl,mensoft,pl}.csv` | MRNet label variants (6.2) |
| `output/ext/{skm_mrnet_labels_graded,oai4_mrnet_labels}.csv` | Combined label files for training on two external sets at once |
| `experiments.md` §16 | OAI experiment log, in more detail |
