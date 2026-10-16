# Submissions

## Leaderboard

The public LB uses about 30% of the roughly 1,300 test studies (about 390). Scores are rounded to 0.001, and the list is ordered by the exact score, so the order inside a tie is real.

**The 58 column** is the macro AUC over all 12 findings as submitted, on the 58 radiologist studies. It uses the validation (non-fullfit) counterparts of each package's models, combined as `infer.pipeline` does, then the notebook's merges: b1_c1 alone for findings without models, 50/50 rank blends for effusion, contusion and fracture, rank(models, rank(family, b1_c1)) for synovitis. `dataset_8` has no b1_c1. Computed by `src_study/tools/sub58.py` (per-finding scores in `logs/study_run/sub58.csv`).

| # | LB | LB − 13 (implied) | Weights | What it is | 58 |
|---|---|---|---|---|---|
| 1 | **0.965** | 0 | `rsna-knee-weights-13` | `dataset_12` with lateral OA back to `mk89`, synovitis models at 224 px (`sy224a` / `b`) | 0.9585 |
| 2 | 0.964 | −0.0001 | weights-13 + 2× OA correction | Notebook `OA_CORRECTION` (assumed `{"Lateral OA": 0.3, "PF OA": 0.3}`) | 0.9593 |
| 3 | 0.964 | −0.0002 | `rsna-knee-weights-6` | `2-moaks-mrnet-men` + synovitis models (`sy0`, `sy1`, `sc0`, `sc1`) | 0.9563 |
| 4 | 0.964 | −0.0003 | `rsna-knee-weights-7` | `dataset_6` + contusion (`cc0`–`2`) and effusion (`ef0`–`1`) models | 0.9564 |
| 5 | 0.963 | −0.0013 | `rsna-knee-weights-2-moaks-mrnet-men` | `2-moaks-mrnet` + MOAKS-pretrained menisci model `mk87`, CAFormer `cafax87` | 0.9531 |
| 6 | 0.963 | −0.0014 | `rsna-knee-weights-2-moaks-mrnet-men-frac` | `2-moaks-mrnet-men` + fracture models `fr9a` / `fr9b` | 0.9542 |
| 7 | 0.963 | −0.0015 | `rsna-knee-weights-17` | `dataset_13` + 50% rank blends with C1 RadImageNet heads (`c1rad`, `c1cm`) on medial / lateral / PF OA, Baker's, fracture, effusion | 0.9626 |
| 8 | 0.963 | −0.0016 | `rsna-knee-weights-2-moaks-mrnet` | `2-moaks` + MRNet-trained ACL `mr85` and MCL `pl86` | 0.9515 |
| 9 | 0.963 | −0.0017 | `rsna-knee-weights-2-moaks` | `dataset_2` with MOAKS-pretrained OA models (`mk89`, `mk90`) | 0.9504 |
| 10 | 0.963 | −0.0018 | `rsna-knee-weights-12` | `dataset_11` + PF OA blend (`mk90`, `mk89`, `ax0`) | 0.9615 |
| 11 | 0.963 | −0.0019 | `rsna-knee-weights-11` | `dataset_10` with lateral OA `lc0`–`2` (no correction); no contusion / effusion models | 0.9615 |
| 12 | 0.962 | −0.0021 to −0.0029 | `rsna-knee-weights-2-moaks-hybrid` | `2-moaks` + 4 mt3d hybrid models on the 8 ROI findings (`output/bis/dataset_2_moaks_hybrid`) | 0.9448 |
| 13 | 0.962 | −0.0022 to −0.0030 | `rsna-knee-weights-2` | Retrained ROI models (`85`–`91`) | 0.9480 |
| 14 | 0.962 | −0.0023 to −0.0031 | `rsna-knee-weights-1` | First package (`63`–`69`) | 0.9420 |
| 15 | 0.962 | −0.0024 to −0.0032 | `rsna-knee-weights-4` | `dataset_2` with midline OA models (`o37`–`o43`), lateral OA − 0.4 × medial OA | 0.9511 |
| 16 | 0.962 | −0.0025 to −0.0033 | `rsna-knee-weights-5` | `dataset_4` + ACL `a65` / `a77`–`79`, 8-frame retrains | 0.9526 |
| 17 | 0.959 | −0.0050 to −0.0063 | `rsna-knee-weights-8` | `dataset_7` + fracture models `fr9a` / `fr9b`; submitted **without b1_c1**: effusion, synovitis, contusion and fracture from our models alone | 0.9557 |

Not submitted: `dataset_18` (58: 0.9629), `dataset_19` (0.9658).

## Changes between submissions: 58 vs LB

The 58 Δ is the change of the leaderboard table's 58 score (12 findings) (`src_study/tools/sub58.py`); classes with a change under 0.0005 are not listed. `dataset_1`'s validation counterparts are approximate: its fullfits trained on `pl_round2`, the matching validation runs on `pl_round1`. Before `dataset_6` the notebook is taken to use b1_c1 alone for synovitis (the family blend's start is not recorded).

The LB change uses two assumptions on top of the rounded scores and the exact order:

- two neighbouring submissions with the same rounded score are 0.0001 apart;
- `dataset_13` (0.965) is 0.0001 above the best 0.964 (13 + OA correction).

With `dataset_17` at −0.0015 from `dataset_13` (the measured drop), this fixes every submission from 0.965 down to `dataset_11` relative to `dataset_13` (third column above). The gap between `dataset_11` (the last 0.963) and `2-moaks-hybrid` (the first 0.962) is only bounded, at 0.0002 to 0.0010, so changes that cross it are ranges. Paired public noise for small changes is about ±0.0008 (`src_study/tools/shakeup.py`), so changes of 0.0001 to 0.0003 are within noise.

| From → to | Change | Classes changed (58 Δ) | 58 Δ | LB change |
|---|---|---|---|---|
| 1 → 2 | Retrained ROI models | LM +0.042, PF OA +0.010, MCL +0.007, Baker's +0.004, MOA +0.003, ACL +0.003, LOA +0.002, MM +0.001 | +0.0060 | +0.0001 |
| 2 → 4 | Midline OA models, lateral OA − 0.4 × medial OA | LOA +0.031, MOA +0.006 | +0.0031 | −0.0002 |
| 4 → 5 | ACL models, 8-frame retrains | ACL +0.018 | +0.0015 | −0.0001 |
| 2 → 2-moaks | MOAKS-pretrained OA models | PF OA +0.017, LOA +0.010, MOA +0.003 | +0.0025 | +0.0005 to +0.0013 |
| 2-moaks → 2-moaks-mrnet | MRNet ACL / MCL models | ACL +0.008, MCL +0.005 | +0.0010 | +0.0001 |
| 2-moaks-mrnet → men | MOAKS menisci model, CAFormer menisci | LM +0.014, MM +0.005 | +0.0016 | +0.0003 |
| men → men-frac | + fracture models `fr9a` / `fr9b` (blended with b1_c1) | Fracture +0.014 | +0.0012 | −0.0001 |
| men → 6 | + synovitis models (blended with b1_c1 and the family model) | Synovitis +0.038 | +0.0032 | **+0.0011** |
| 6 → 7 | + contusion and effusion models (blended with b1_c1) | Contusion +0.004, Effusion −0.002 | +0.0002 | −0.0001 |
| 7 → 8 | + fracture models `fr9a` / `fr9b`, and b1_c1 dropped: effusion, synovitis, contusion and fracture from our models alone | Contusion −0.009, Fracture +0.008, Effusion −0.004, Synovitis −0.004 | −0.0008 | **−0.0047 to −0.0060** |
| 11 → 12 | PF OA blend (holdout gain) | – | 0.0000 | +0.0001 |
| 12 → 13 | Lateral OA `lc0`–`2` → `mk89`, synovitis 224 px | LOA −0.041, Synovitis +0.004 | −0.0030 | **+0.0018** |
| 6 → 13 | Synovitis 224 px, bone-crop fracture `fb0` / `fb1`, PF OA blend | Fracture +0.022, Synovitis +0.004 | +0.0022 | +0.0002 |
| 13 → 13 + OA correction | Lateral OA and PF OA − 0.3 × medial OA | LOA +0.006, PF OA +0.004 | +0.0008 | −0.0001 |
| 13 → 17 | 6 blends with RadImageNet C1 heads | LOA +0.023, PF OA +0.011, MOA +0.006, Effusion +0.005, Fracture +0.003, Baker's +0.002 | +0.0042 | **−0.0015** |

### What the comparisons show

- **The 58 doesn't rank submissions.** `dataset_13` has the lowest 58 score among 11, 12, 13 and 17, and the best LB.
- **Every change that moved lateral OA away from `mk89` lost on the LB,** despite gains on the 58 (and twice on the holdout):
  - midline models (2 → 4);
  - `lc0`–`2` (in 11 and 12; reverting in 12 → 13 gained +0.0018);
  - the OA correction (13 → 13 + correction);
  - the RadImageNet heads (13 → 17).
- **Changes that left lateral OA alone were neutral to positive** (+0.0001 to +0.0011 each): MOAKS pretraining, the MRNet ACL / MCL models, the menisci models, the synovitis models, the PF OA blend.
- **Blends with b1_c1 on contusion, effusion and fracture were neutral to negative** (men → men-frac, 6 → 7, part of 13 → 17).
- **Dropping b1_c1 cost about 0.005 (7 → 8),** while the 58 says −0.0008. On the 58 our own effusion, synovitis and contusion models are each a little below b1_c1 (and our fracture a little above it); on the LB they are much worse. For these four findings b1_c1 is clearly the stronger model on the test set, and the 58 underrates the gap by about 6×.
- **The bone-crop fracture models barely moved the LB:** 6 → 13 is +0.0002, despite fracture +0.022 and synovitis +0.004 on the 58.
- **Apart from dropping b1_c1 (7 → 8), only three changes exceed the paired noise (±0.0008):** lateral OA back to `mk89` (+0.0018), the synovitis models (+0.0011) and the RadImageNet blends (−0.0015). Two of the three are lateral OA.
