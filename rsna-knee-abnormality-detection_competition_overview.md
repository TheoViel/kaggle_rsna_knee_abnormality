# RSNA Knee Abnormality Detection — Competition Overview

Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview

## abstract

A single knee scan can reveal a dozen different problems. In this competition, you are tasked to build machine learning models that detect a defined set of clinically important abnormalities on knee MRI examinations.

## Description

The knee is the most commonly injured and imaged joint in the body. Osteoarthritis alone affects an estimated 654 million people worldwide, while acute knee injuries account for 15 to 40 percent of all sports-related trauma. MRIs show clinicians ligaments, cartilage, menisci, and bone in detail, without exposing patients to radiation.

Reading those scans isn’t always straightforward. ACL and MCL tears, meniscal damage, cartilage loss, fractures, and other abnormalities can be subtle, and radiologists don’t always interpret them the same way. Access to musculoskeletal radiologists is also limited, especially outside major medical centers, leading to delays and inconsistent diagnoses.

In this competition, you will develop multimodal machine learning models to detect twelve clinically important knee abnormalities. You'll work with the first RSNA AI Challenge dataset that pairs every imaging study with its original radiology report, enabling your models to learn from both visual scans and written diagnostic text.

High-performing models can act as robust decision support tools, delivering the accuracy, consistency, and speed needed to elevate expert-level knee MRI interpretation and improve care across disparate clinic settings.

## Evaluation

Submissions are evaluated by the average [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted confidence scores and the observed targets across the twelve targets:

$$
\text{Final Score} = \frac{1}{12}\sum_{i=0}^{11} AUC_i
$$

The final score is, in other words, the macro-averaged AUC ROC.

## Submission File
For each row in the test set, you must predict a confidence score for each of the twelve target labels. The file should contain a header and have the following format:

```
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
<uid_2>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
...
```

**Efficiency Track** <br/>
First Efficiency Prize: $7,000 <br/>
Second Efficiency Prize: $6,000 <br/>
Third Efficiency Prize: $5,000 <br/>

Because this competition is being hosted in coordination with the Radiological Society of North America (RSNA) Annual Meeting, winners will be invited and strongly encouraged to attend the AI Challenge Recognition Event with waived fee, contingent on review of solution and fulfillment of winners' obligations.

Note that, per the competition rules, in addition to the standard Kaggle Winners' Obligations (open-source licensing requirements, solution packaging/delivery, presentation to host), the host team also asks that you:

(i) create a short video presenting your approach and solution, and

(ii) publish a link to your open sourced code and the weights on the competition forum

(iii) Share final version of model as publicly available for open distribution and validation. Please see https://www.kaggle.com/models/tom99763/9th-place-models-rsna-iad/PyTorch/default as an example.

## Code Requirements

<img src="https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png" style="float: right; width: 70px; padding: 10px">

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met: 
- CPU Notebook <= 9 hours run-time 
- GPU Notebook <= 9 hours run-time 
- Internet access disabled 
- Freely & publicly available external data is allowed, including pre-trained models 
- Submission file must be named `submission.csv` 

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## Efficiency Prize Evaluation

## Efficiency Prize

We are hosting a second track that focuses on model efficiency, because highly accurate models are often computationally heavy.

For the Efficiency Prize, we will evaluate submissions on both runtime and predictive performance.

To be eligible for an Efficiency Prize, a submission:
- Must be among the submissions selected by a team for the Leaderboard Prize, or else among those submissions automatically selected under the conditions described in the [**My Submissions**](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection/submissions) tab.
- Must be ranked on the Private Leaderboard higher than the `sample_submission.csv` benchmark.

All submissions meeting these conditions will be considered for the Efficiency Prize. A submission may be eligible for both the Leaderboard Prize and the Efficiency Prize.

An Efficiency Prize will be awarded to eligible submissions according to how they are ranked by the following evaluation metric on the private test data. See the [**Prizes**](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection/overview/prizes) tab for the prize awarded to each rank. More details may be posted via discussion forum updates.

## Efficiency Score

We compute a submission's **efficiency score** by:
\\[ \text{Efficiency} = \frac{ \text{AUC} }{ \text{Benchmark} - \max\text{AUC} } + \frac{ \text{RuntimeSeconds} }{ 32400 } \\]

where \\(\text{AUC}\\) is the submission's score on the [main competition metric](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection/overview/evaluation), \\(\text{Benchmark}\\) is the score of the benchmark `sample_submission.csv`, \\(\max\text{AUC}\\) is the maximum \\(\text{AUC}\\) of all submissions on the Private Leaderboard, and \\(\text{RuntimeSeconds}\\) is the number of seconds it takes for the submission to be evaluated. The objective is to minimize the efficiency score.

During the training period of the competition, you may see a leaderboard for the *public* test data in the following notebook, updated daily: **[Efficiency
Leaderboard](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)**. After the competition ends, we will update this leaderboard with efficiency scores on the *private* data. During the training period, this leaderboard will show only the rank of each team, but not the complete score.

## data-description

## Dataset Description

This dataset contains knee MRI studies annotated for twelve common findings: ligament and meniscus injuries, three compartments of osteoarthritis, joint effusion, synovitis, Baker's cyst, bone contusion, and fracture. Each study comprises a collection of individual MRI sequences from a single scanning session formatted as DICOM series. Your task is to predict the per-study probability of each of the twelve findings.

Studies come from a diverse international mix of imaging sites and span a wide range of scanners, protocols, and populations. Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report from which you may wish to derive the labels for the remaining studies.

## Files

**train.csv** One row per training study.

*   `StudyInstanceUID` - unique identifier for the study; matches the folder name under `train_series/`.
*   `Report` - the free-text radiology report. May be in any of several languages, depending on the reporting institution.

Twelve binary labels:
*   `ACL` - anterior cruciate ligament injury (0/1).
*   `MCL` - medial collateral ligament injury (0/1).
*   `Medial Meniscus` - medial meniscus tear (0/1).
*   `Lateral Meniscus` - lateral meniscus tear (0/1).
*   `Medial OA` - osteoarthritis of the medial tibiofemoral compartment (0/1).
*   `Lateral OA` - osteoarthritis of the lateral tibiofemoral compartment (0/1).
*   `PF OA` - patellofemoral osteoarthritis (0/1).
*   `Effusion` - joint effusion / excess fluid (0/1).
*   `Synovitis` - inflammation of the joint lining (0/1).
*   `Baker's` - Baker's cyst (0/1).
*   `Contusion` - bone contusion / bone bruise (0/1).
*   `Fracture` - fracture (0/1).

**train_series.csv** One row per training series. Each series is a single MRI acquisition and each study comprises several series.

*   `StudyInstanceUID` - study this series belongs to.
*   `SeriesInstanceUID` - unique identifier for the series; matches the folder name under `train_series/<StudyInstanceUID>/`.
*   `Fluid_Sensitive` - `1` if the sequence emphasizes fluid signal (T2, PD, STIR, and similar), `0` otherwise.
*   `Fat_Suppression` - `1` if the sequence applies fat suppression, `0` otherwise. Note that although `Fluid_Sensitive` and `Fat_Suppression` are often correlated, as observed in the training set, they are not necessarily equivalent for every case.
*   `Anatomical_Plane` - imaging plane: `Sagittal`, `Coronal`, or `Axial`.

**train_series/** Training DICOMs, organized as `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`. Each `.dcm` is a single image slice. Series typically contain 20–45 slices (median 30), with a long tail out to a few hundred.

**test.csv** Example test file with three study IDs from the public test set. During scoring, this example data will be replaced with the actual test data. There are about 1300 studies in the test set. The `Report` field will not be provided at the testing stage.

*   `StudyInstanceUID` - unique identifier for a test study.

**test_series.csv** Same schema as `train_series.csv`, for the example test studies. Replaced with the real test-series descriptors during scoring.

**test_series/** Example test DICOMs, same layout as `train_series/`. Replaced with the real test DICOMs during scoring.

**sample_submission.csv** A valid submission with all label columns set to `0.5`.

## Dataset Distribution Notice

Although efforts have been made to ensure each abnormality is represented in each dataset, the prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets. 

## DICOM Notes

Intensities, orientations, and resolutions vary across series and studies. Series come in a mix of transfer syntaxes (uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR Little Endian). Every DICOM has been stripped to an allowlisted set of 86 metadata tags.

## Clinical Background & Label Definitions (host discussion)

Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343 — "Courtesy of Dr. Jacob Kazam", 91 votes, posted 2026-08-06. The post also contains anatomy illustrations and 13 annotated positive/negative example images, not reproduced here.

### Clinical problem

- First RSNA AI Challenge to pair every study with its original radiology report, on a large, multilingual, international dataset. Models may use images and report text.
- The knee is the most common site of osteoarthritis (nearly a quarter of adults 40+, ~654M people worldwide). Knee injuries are 15–40% of sports injuries.
- MRI is the reference standard (ligaments, menisci, cartilage, bone, no ionizing radiation). Reader variability and limited access to MSK radiologists motivate automated tools.

### Anatomy relevant to the labels

- Bones: femur, tibia, patella. Cartilage covers their ends. The synovium lines the joint capsule: excess fluid is **effusion**, inflammation is **synovitis**.
- Three compartments, each assessed separately for osteoarthritis: **medial** (inner, medial femoral condyle / tibial plateau), **lateral** (outer), **patellofemoral** (patella / femoral trochlea).
- Ligaments: ACL and PCL (intra-articular, front-to-back stability), MCL and LCL (sides). **Only ACL and MCL tears are labeled.**
- Medial and lateral menisci: C-shaped fibrocartilage wedges, evaluated separately.
- Bone marrow: bruise = contusion (marrow edema), break = fracture. A fluid-filled outpouching behind the knee is a Baker (popliteal) cyst.

### Imaging notes

- A study is a set of series, each with a given pulse sequence and plane (axial, coronal, sagittal). Interpretation integrates multiple series, planes and slices, so it is inherently 3D.
- Cruciate ligaments and menisci are best seen on sagittal/coronal images. Patellofemoral cartilage is best seen on axial images.
- "Fluid sensitive" here means edema, hemorrhage and fluid appear bright and fat is suppressed in some way (PD / T2, often fat-suppressed). Most abnormalities are detected on these sequences.

### Label criteria

Twelve binary labels, one study = one knee. Ground truth is consensus from two subspecialty MSK radiologists with a third adjudicating disagreements. **Borderline ("on the fence") findings were graded negative** to favor specificity.

| Label | Positive when | Negative when |
|---|---|---|
| ACL | High-grade partial or full-thickness tear: complete discontinuity or >50% of fibers disrupted, with or without secondary signs (e.g. pivot-shift contusions) | Mild signal change, degeneration, thickening without discontinuity |
| MCL | High-grade partial or complete *acute* tear: disrupted fibers with edema in/adjacent to the ligament | Low-grade sprain, chronic or remote stress changes |
| Medial Meniscus | Abnormal signal that definitely contacts the meniscal surface on ≥2 images, or a morphologic abnormality (truncated, diminutive, displaced fragment) | Intrasubstance degeneration not reaching the surface |
| Lateral Meniscus | Same criteria as medial meniscus | Same |
| Medial OA | Moderate/large area (~≥1 cm) of high-grade cartilage loss (>50% thickness) in the medial compartment, with or without subchondral marrow changes | — |
| Lateral OA | Same criteria, lateral compartment | — |
| PF OA | Same criteria, patellofemoral compartment | — |
| Effusion | Moderate or large amount of fluid distending the joint | — |
| Synovitis | Inflammation and thickening of the synovial lining | — |
| Baker's | Moderate or large fluid collection in the characteristic location behind the knee | — |
| Contusion | Bone-marrow edema-like signal from impact, without a discrete fracture line | — |
| Fracture | Acute cortical break or fracture line | — |

### Community Q&A from the thread

- **Labels vs. reports:** A participant reports that `train.csv` has only 58 labeled rows (the rest NaN) and that those labels don't always match the report text, and asks whether the reports are the ground truth. No host answer in the thread.
- **Bilateral knee MRIs:** A participant asks how to label a study containing both knees (e.g. left ACL torn, right normal) when there is one label per `StudyInstanceUID`. No host answer in the thread.
- **Typo:** The meniscus text appears to be copy-pasted. A participant asks whether the "medial meniscus tear" definition should read "lateral" where the lateral meniscus is meant. No host answer in the thread.
- **Report text value:** A participant asks whether anyone has found report text to give a clear advantage over image-only models. Unanswered.
- **Framing:** Multi-label classification, with each of the 12 findings predicted as an independent probability.
