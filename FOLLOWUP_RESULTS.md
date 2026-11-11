# Follow-up experiments: extra seeds, ℓ_min = 1 and validation-selected thresholds

This report covers the three parts of the follow-up to the Protocol-A ablation study of the HybridSBD shot boundary detection models (leak-free SHOT training set of 448 videos): **Part 1** trains A0 and A1 with two more seeds; **Part 2** re-evaluates every model with ℓ_min = 1 on all three test sets (Protocol A with ℓ_min = 1); **Part 3** evaluates every model with thresholds selected on the SHOT validation split (Protocol B). A short additional check (validation F1 per epoch) follows. Evaluation uses the existing best checkpoints (EMA weights selected by SHOT validation F1). Windows (T = 128, stride 64), window averaging, threshold grids (τ_c 0.10–0.90 step 0.05, τ_g ∈ {0.3, 0.4, 0.5, 0.6}), the decision rule and ±2-frame scene matching are unchanged. Seed 42 is the main-study run.

| Part | Content |
|---|---|
| Part 1 | Extra seeds for A0 and A1 (training) |
| Part 2 | Protocol A with ℓ_min = 1 on all test sets (evaluation only) |
| Part 3 | Protocol B, validation-selected thresholds (evaluation only) |
| Part 4 | Additional check: validation F1 for every epoch |

---

# Part 1: Extra seeds for A0 and A1 (training)

A0 (full model, dual-label) and A1 (full model, midpoint only) were trained with seeds 43 and 44, giving four new runs (A0-s43, A0-s44, A1-s43, A1-s44). Recipe, augmentation, synthetic transitions, EMA (0.999), early stopping (patience 10 on SHOT validation F1 with the variant's own decision rule) and the 80-epoch maximum are identical to the seed-42 runs. Only the training seed changes: model initialisation, training-data order, the ClipShots clip drawn each epoch, augmentation and synthetic transitions.

## 1.1 Train/validation split is identical for all seeds

In the training code the SHOT and ClipShots train/validation split was shuffled with the training seed, so seeds 43 and 44 would have used different validation videos. I fixed the split to the seed-42 split for every training seed (the option defaults to 42, so the seed-42 runs are unchanged). SHA-256 of the sorted SHOT validation video names:

| Training seed | Old rule (split follows the seed) | Fixed split (used) |
|---|---|---|
| 42 | f0070e4a… | f0070e4a… |
| 43 | 88ddfd45… | f0070e4a… |
| 44 | 792acaf5… | f0070e4a… |

All four new runs printed the same checksums before training: SHOT validation 67 videos (f0070e4a…), SHOT train 381 videos (ed9cb6ec…), ClipShots train 3,108 videos (f623f785…). The pre-extracted training clips use the stable per-video MD5 seed and are unchanged.

## 1.2 The four new runs

| Run | Seed | Best epoch | Best SHOT val F1 | Epochs run | Stopped early | Training hours |
|---|---|---|---|---|---|---|
| A0 | 42 | 21 | 0.8029 | 31 | True | 5.40 |
| A0-s43 | 43 | 13 | 0.7953 | 23 | True | 6.97 |
| A0-s44 | 44 | 16 | 0.7935 | 26 | True | 7.26 |
| A1 | 42 | 7 | 0.7863 | 17 | True | 2.92 |
| A1-s43 | 43 | 10 | 0.7915 | 20 | True | 6.46 |
| A1-s44 | 44 | 9 | 0.7967 | 19 | True | 6.18 |

Seed-42 rows are the main-study runs, shown for reference.

## 1.3 SHOT validation F1 history of every epoch

SHOT validation F1 after every epoch; best epoch in bold. Seed 42 runs are A1 and A0.

| Epoch | A1 | A0 | A1-s43 | A1-s44 | A0-s43 | A0-s44 |
|---|---|---|---|---|---|---|
| 1 | 0.7029 | 0.6988 | 0.7257 | 0.7280 | 0.7223 | 0.7087 |
| 2 | 0.7499 | 0.7514 | 0.7659 | 0.7515 | 0.7586 | 0.7446 |
| 3 | 0.7535 | 0.7543 | 0.7715 | 0.7664 | 0.7615 | 0.7487 |
| 4 | 0.7628 | 0.7774 | 0.7845 | 0.7902 | 0.7893 | 0.7827 |
| 5 | 0.7807 | 0.7758 | 0.7891 | 0.7910 | 0.7881 | 0.7885 |
| 6 | 0.7796 | 0.7722 | 0.7877 | 0.7909 | 0.7850 | 0.7888 |
| 7 | **0.7863** | 0.7725 | 0.7891 | 0.7940 | 0.7862 | 0.7867 |
| 8 | 0.7785 | 0.7748 | 0.7863 | 0.7927 | 0.7902 | 0.7880 |
| 9 | 0.7695 | 0.7654 | 0.7909 | **0.7967** | 0.7801 | 0.7782 |
| 10 | 0.7852 | 0.7838 | **0.7915** | 0.7949 | 0.7929 | 0.7857 |
| 11 | 0.7843 | 0.7793 | 0.7822 | 0.7935 | 0.7835 | 0.7801 |
| 12 | 0.7783 | 0.7843 | 0.7818 | 0.7864 | 0.7889 | 0.7818 |
| 13 | 0.7824 | 0.7821 | 0.7901 | 0.7812 | **0.7953** | 0.7706 |
| 14 | 0.7704 | 0.7681 | 0.7835 | 0.7941 | 0.7853 | 0.7825 |
| 15 | 0.7840 | 0.7885 | 0.7865 | 0.7859 | 0.7930 | 0.7847 |
| 16 | 0.7735 | 0.7926 | 0.7782 | 0.7909 | 0.7864 | **0.7935** |
| 17 | 0.7846 | 0.7810 | 0.7774 | 0.7874 | 0.7781 | 0.7888 |
| 18 |  | 0.7932 | 0.7882 | 0.7914 | 0.7858 | 0.7912 |
| 19 |  | 0.7946 | 0.7733 | 0.7898 | 0.7806 | 0.7825 |
| 20 |  | 0.7960 | 0.7830 |  | 0.7926 | 0.7866 |
| 21 |  | **0.8029** |  |  | 0.7832 | 0.7828 |
| 22 |  | 0.7935 |  |  | 0.7883 | 0.7882 |
| 23 |  | 0.7907 |  |  | 0.7933 | 0.7908 |
| 24 |  | 0.7953 |  |  |  | 0.7870 |
| 25 |  | 0.7854 |  |  |  | 0.7820 |
| 26 |  | 0.7847 |  |  |  | 0.7850 |
| 27 |  | 0.7913 |  |  |  |  |
| 28 |  | 0.7951 |  |  |  |  |
| 29 |  | 0.7810 |  |  |  |  |
| 30 |  | 0.7886 |  |  |  |  |
| 31 |  | 0.7870 |  |  |  |  |

## 1.4 Does training several runs at once on one GPU change the results?

A seed-42 A1 run trained alongside the four new runs (five jobs on one GPU) was compared with the original seed-42 A1 run over its first two epochs. Train loss 0.03501 vs 0.03502 (epoch 1) and 0.01847 vs 0.01849 (epoch 2); SHOT validation F1 0.7046 vs 0.7029 (epoch 1) and 0.7505 vs 0.7499 (epoch 2). The two runs are **not bit-identical**, so a repeated training run, with or without concurrent jobs, gives slightly different numbers; I could not separate concurrency from other run-to-run non-determinism. The differences are small (at most 0.002 validation F1 and 0.0001 train loss) and far below the spread between seeds at epoch 2 (validation F1 0.7446–0.7659 across the four new runs). Data-loader seeding is per run and the split is fixed, so concurrency does not change the data each run sees.

---

# Part 2: Protocol A with ℓ_min = 1 (evaluation only)

ℓ_min is fixed to 1 (no minimum-shot-length merging) on SHOT, ClipShots and BBC and is not searched. Thresholds are still selected on each test set (Protocol A).

## 2.1 Checks

**Previous evaluation reproduced for A0** (ClipShots with the ℓ_min grid; SHOT and BBC with ℓ_min = 1):

| Test set | Expected F1 | Reproduced F1 | Expected thresholds | Reproduced thresholds |
|---|---|---|---|---|
| SHOT | 0.8449 | 0.8449 | τ_c 0.4, τ_g 0.3 | τ_c 0.4, τ_g 0.3 |
| ClipShots | 0.8052 | 0.8052 | τ_c 0.65, τ_g 0.6, ℓ_min 4 | τ_c 0.65, τ_g 0.6, ℓ_min 4 |
| BBC | 0.9719 | 0.9719 | τ_c 0.45, τ_g 0.6 | τ_c 0.45, τ_g 0.6 |

All 17 seed-42 models reproduce their stored results exactly (51 of 51 model-test-set results). For SHOT and BBC the ℓ_min = 1 results are identical to the main study for all 17 models (F1, thresholds, TP / FP / FN); only ClipShots changes.

**Saved probabilities vs a fresh forward pass** (A0 and A1, all three test sets): the maximum absolute difference of p_y and p_z over all videos is 0.0, and F1, thresholds and TP / FP / FN are identical. The results in Parts 2 and 3 therefore use the saved probabilities.

## Results

### 2.2 Supervision (Table 7a, seed 42)

| ID | Description | SHOT F1 | ClipShots F1 | BBC F1 |
|---|---|---|---|---|
| A0 | Full, dual-label (primary) | 0.8449 | 0.8001 | 0.9719 |
| A1 | Full, midpoint only | 0.8478 | 0.7609 | 0.9698 |
| A2 | Full, unified single head | **0.8509** | 0.7709 | 0.9473 |
| A3 | Reduced, midpoint only | 0.8312 | 0.7606 | 0.9714 |
| A4 | Reduced, dual-label | 0.8370 | **0.8045** | **0.9721** |

Differences:

| Delta | SHOT | ClipShots | BBC |
|---|---|---|---|
| A0 − A1 (full) | -0.0029 | +0.0392 | +0.0021 |
| A4 − A3 (reduced) | +0.0058 | +0.0439 | +0.0007 |
| A0 − A2 (dual vs unified) | -0.0060 | +0.0292 | +0.0245 |

### 2.3 Components, leave-one-out (Table 7b)

| ID | Description | SHOT F1 | ClipShots F1 | BBC F1 |
|---|---|---|---|---|
| A0 | Full model | 0.8449 | 0.8001 | 0.9719 |
| C1 | w/o difference-based branch (B) | **0.8536** | 0.7942 | 0.9701 |
| C2a | w/o FrameSimilarity | 0.8457 | 0.8009 | 0.9681 |
| C2b | w/o ColorHistogramSimilarity | 0.8361 | **0.8077** | 0.9730 |
| C3 | w/o temporal delta | 0.8363 | 0.7956 | 0.9728 |
| C4 | w/o per-frame pathway (relations from Branch A) | 0.8420 | 0.8068 | 0.9699 |
| C5a | w/o pixel-level motion descriptor | 0.8487 | 0.8021 | 0.9715 |
| C5b | w/o edge-density head | 0.8454 | 0.7967 | **0.9733** |
| C6 | w/o self-attention and gate | 0.8364 | 0.7969 | 0.9667 |

Change in F1 relative to A0:

| ID | ΔSHOT | ΔClipShots | ΔBBC |
|---|---|---|---|
| C1 | +0.0087 | -0.0059 | -0.0017 |
| C2a | +0.0008 | +0.0008 | -0.0037 |
| C2b | -0.0089 | +0.0076 | +0.0012 |
| C3 | -0.0086 | -0.0045 | +0.0009 |
| C4 | -0.0029 | +0.0067 | -0.0020 |
| C5a | +0.0037 | +0.0020 | -0.0003 |
| C5b | +0.0004 | -0.0034 | +0.0015 |
| C6 | -0.0085 | -0.0032 | -0.0052 |

Largest single-component change on any dataset: 0.0089.

### 2.4 Build-up ladder

| Step | Run | What is added | SHOT F1 | ClipShots F1 | BBC F1 |
|---|---|---|---|---|---|
| 0 | M0_min | RGB encoder + MLP + dual head only | 0.8169 | 0.7555 | 0.9543 |
| 1 | M1_edge | + edge-density head | 0.8261 | 0.7672 | 0.9584 |
| 2 | B_noatt | + 4 fusion blocks, no attention | 0.8351 | 0.7984 | 0.9682 |
| 3 | A4 | + attention and gate (reduced model) | 0.8370 | 0.8045 | **0.9721** |
| 4 | S1_rel | + FrameSimilarity, ColorHist, delta | 0.8392 | **0.8054** | 0.9706 |
| 5 | A0 | + diff branch, motion, per-frame pathway (full model) | **0.8449** | 0.8001 | 0.9719 |

Change at each step:

| Step | SHOT | ClipShots | BBC |
|---|---|---|---|
| M0_min → M1_edge | +0.0091 | +0.0117 | +0.0041 |
| M1_edge → B_noatt | +0.0091 | +0.0312 | +0.0098 |
| B_noatt → A4 | +0.0018 | +0.0061 | +0.0039 |
| A4 → S1_rel | +0.0023 | +0.0008 | -0.0015 |
| S1_rel → A0 | +0.0057 | -0.0053 | +0.0013 |

### 2.5 Precision, recall and selected thresholds

| ID | SHOT P / R | ClipShots P / R | BBC P / R | SHOT τ_c, τ_g | ClipShots τ_c, τ_g | BBC τ_c, τ_g |
|---|---|---|---|---|---|---|
| A0 | 0.842 / 0.848 | 0.773 / 0.829 | 0.963 / 0.980 | 0.4, 0.3 | 0.65, 0.6 | 0.45, 0.6 |
| A1 | 0.858 / 0.838 | 0.702 / 0.830 | 0.975 / 0.964 | 0.45, – | 0.45, – | 0.45, – |
| A2 | 0.856 / 0.846 | 0.741 / 0.803 | 0.926 / 0.969 | 0.45, – | 0.65, – | 0.5, – |
| A3 | 0.869 / 0.797 | 0.701 / 0.832 | 0.966 / 0.977 | 0.45, – | 0.45, – | 0.45, – |
| A4 | 0.839 / 0.835 | 0.786 / 0.824 | 0.971 / 0.973 | 0.4, 0.4 | 0.7, 0.6 | 0.5, 0.6 |
| C1 | 0.864 / 0.844 | 0.775 / 0.815 | 0.968 / 0.972 | 0.4, 0.3 | 0.7, 0.5 | 0.45, 0.6 |
| C2a | 0.854 / 0.837 | 0.776 / 0.828 | 0.970 / 0.966 | 0.45, 0.5 | 0.65, 0.6 | 0.5, 0.6 |
| C2b | 0.839 / 0.833 | 0.776 / 0.842 | 0.967 / 0.979 | 0.3, 0.3 | 0.65, 0.6 | 0.45, 0.6 |
| C3 | 0.813 / 0.861 | 0.758 / 0.838 | 0.961 / 0.985 | 0.4, 0.4 | 0.65, 0.6 | 0.45, 0.6 |
| C4 | 0.857 / 0.827 | 0.792 / 0.822 | 0.957 / 0.983 | 0.5, 0.3 | 0.7, 0.6 | 0.45, 0.6 |
| C5a | 0.841 / 0.857 | 0.772 / 0.834 | 0.960 / 0.983 | 0.4, 0.3 | 0.65, 0.6 | 0.45, 0.6 |
| C5b | 0.839 / 0.852 | 0.762 / 0.835 | 0.964 / 0.983 | 0.35, 0.4 | 0.65, 0.6 | 0.45, 0.6 |
| C6 | 0.839 / 0.834 | 0.773 / 0.823 | 0.954 / 0.979 | 0.45, 0.4 | 0.7, 0.6 | 0.45, 0.6 |
| M0_min | 0.795 / 0.840 | 0.745 / 0.766 | 0.957 / 0.952 | 0.45, 0.5 | 0.6, 0.6 | 0.45, 0.6 |
| M1_edge | 0.839 / 0.814 | 0.741 / 0.795 | 0.964 / 0.953 | 0.4, 0.6 | 0.55, 0.6 | 0.45, 0.6 |
| B_noatt | 0.845 / 0.826 | 0.773 / 0.826 | 0.960 / 0.976 | 0.45, 0.5 | 0.7, 0.6 | 0.45, 0.6 |
| S1_rel | 0.846 / 0.833 | 0.782 / 0.830 | 0.960 / 0.981 | 0.4, 0.4 | 0.65, 0.5 | 0.45, 0.6 |
| A0-s43 | 0.881 / 0.814 | 0.765 / 0.825 | 0.960 / 0.971 | 0.4, 0.4 | 0.55, 0.6 | 0.4, 0.6 |
| A0-s44 | 0.853 / 0.844 | 0.800 / 0.807 | 0.964 / 0.977 | 0.4, 0.4 | 0.65, 0.6 | 0.45, 0.6 |
| A1-s43 | 0.863 / 0.830 | 0.702 / 0.822 | 0.964 / 0.970 | 0.45, – | 0.45, – | 0.4, – |
| A1-s44 | 0.850 / 0.837 | 0.712 / 0.797 | 0.967 / 0.971 | 0.45, – | 0.5, – | 0.45, – |

A1, A2, A3 and their seeds use τ_c only.

### 2.6 Table 8 — Per-type analysis on ClipShots

| ID | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |
|---|---|---|---|---|
| A0 | 0.8484 | 0.7866 | 0.7415 | 0.6232 |
| A1 | 0.9093 | 0.6623 | n/a | n/a |
| A2 | 0.8011 | 0.8070 | n/a | 0.6965 |

### 2.7 Table 9 — Recall by gradual-transition duration on ClipShots

| ID | ≤ 10 frames | 11–30 frames | > 30 frames |
|---|---|---|---|
| A0 | 0.7740 (n = 832) | 0.8018 (n = 1241) | 0.7500 (n = 228) |
| C6 | 0.7500 (n = 832) | 0.8251 (n = 1241) | 0.7368 (n = 228) |

C6 − A0: ≤ 10 -0.0240, 11–30 +0.0234, > 30 -0.0132.

### 2.8 Seeds table

F1 of A0 and A1 at seeds 42, 43 and 44 (seed 42 = the main-study run). Mean and sample standard deviation (n − 1) over the three seeds.

| Model | Test set | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|---|---|---|---|---|---|---|
| A0 | SHOT | 0.8449 | 0.8463 | 0.8486 | 0.8466 | 0.0018 |
| A0 | ClipShots | 0.8001 | 0.7938 | 0.8031 | 0.7990 | 0.0047 |
| A0 | BBC | 0.9719 | 0.9655 | 0.9706 | 0.9693 | 0.0034 |
| A1 | SHOT | 0.8478 | 0.8460 | 0.8436 | 0.8458 | 0.0021 |
| A1 | ClipShots | 0.7609 | 0.7573 | 0.7524 | 0.7569 | 0.0043 |
| A1 | BBC | 0.9698 | 0.9670 | 0.9691 | 0.9686 | 0.0015 |

**Paired difference A0 − A1** (same seed for both):

| Test set | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|---|---|---|---|---|---|
| SHOT | -0.0029 | +0.0003 | +0.0049 | +0.0008 | 0.0039 |
| ClipShots | +0.0392 | +0.0366 | +0.0507 | +0.0421 | 0.0075 |
| BBC | +0.0021 | -0.0014 | +0.0015 | +0.0007 | 0.0019 |

**Cut recall and gradual recall on ClipShots:**

| Model | Metric | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|---|---|---|---|---|---|---|
| A0 | Cut recall | 0.8484 | 0.8423 | 0.8341 | 0.8416 | 0.0072 |
| A0 | Gradual recall | 0.7866 | 0.7870 | 0.7479 | 0.7739 | 0.0225 |
| A1 | Cut recall | 0.9093 | 0.9048 | 0.8914 | 0.9019 | 0.0093 |
| A1 | Gradual recall | 0.6623 | 0.6454 | 0.5958 | 0.6345 | 0.0346 |

### 2.9 ClipShots comparison, previous vs new

| ID | Previous F1 (best ℓ_min) | ℓ_min picked before | F1 with ℓ_min = 1 | Difference |
|---|---|---|---|---|
| A0 | 0.8052 | 4 | 0.8001 | -0.0051 |
| A1 | 0.7625 | 4 | 0.7609 | -0.0016 |
| A2 | 0.7752 | 3 | 0.7709 | -0.0042 |
| A3 | 0.7640 | 4 | 0.7606 | -0.0034 |
| A4 | 0.8095 | 4 | 0.8045 | -0.0050 |
| C1 | 0.7980 | 3 | 0.7942 | -0.0038 |
| C2a | 0.8069 | 4 | 0.8009 | -0.0060 |
| C2b | 0.8111 | 3 | 0.8077 | -0.0034 |
| C3 | 0.8013 | 3 | 0.7956 | -0.0057 |
| C4 | 0.8113 | 4 | 0.8068 | -0.0045 |
| C5a | 0.8063 | 4 | 0.8021 | -0.0042 |
| C5b | 0.7999 | 4 | 0.7967 | -0.0033 |
| C6 | 0.8032 | 4 | 0.7969 | -0.0063 |
| M0_min | 0.7678 | 4 | 0.7555 | -0.0123 |
| M1_edge | 0.7774 | 4 | 0.7672 | -0.0102 |
| B_noatt | 0.8065 | 4 | 0.7984 | -0.0081 |
| S1_rel | 0.8084 | 4 | 0.8054 | -0.0030 |

### 2.10 ℓ_min sensitivity for A0 on ClipShots

Best F1 for each fixed ℓ_min, thresholds re-searched for each value:

| ℓ_min | τ_c | τ_g | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | 0.65 | 0.6 | 0.7734 | 0.8287 | 0.8001 |
| 2 | 0.65 | 0.6 | 0.7800 | 0.8274 | 0.8030 |
| 3 | 0.65 | 0.6 | 0.7848 | 0.8256 | 0.8047 |
| 4 | 0.65 | 0.6 | 0.7871 | 0.8241 | **0.8052** |

### 2.11 Interpretation check (0.01 band)

| Statement | Value | Holds? |
|---|---|---|
| A0 − A1 on ClipShots ≥ 0.01, seed 42 | +0.0392 | **yes** |
| A0 − A1 on ClipShots ≥ 0.01, mean over the three seeds | +0.0421 | **yes** |
| A0 − A1 on ClipShots positive for every seed | +0.0392 / +0.0366 / +0.0507 | **yes** |
| A4 − A3 on ClipShots ≥ 0.01 | +0.0439 | **yes** |
| A0 − A2 on ClipShots ≥ 0.01 | +0.0292 | **yes** |
| A0 − A2 on BBC ≥ 0.01 | +0.0245 | **yes** |
| No leave-one-out change (C1–C6) reaches 0.01 on any test set | largest 0.0089 | **yes** |

---

# Part 3: Protocol B, validation-selected thresholds (evaluation only)

For each model τ_c and τ_g are selected once on the SHOT validation videos and applied unchanged to SHOT, ClipShots and BBC (ℓ_min = 1). Nothing is tuned on a test set. Midpoint-only (A1, A3) and unified (A2) models select τ_c only.

## 3.1 Validation videos as full videos (prerequisite)

Before this work the 67 SHOT validation videos existed only as pre-cut training clips, so I **added a data path**: the 67 videos are now decoded completely with the same decoder, resizing (48 × 64), window layout (T = 128, stride 64) and window averaging as the SHOT test set (1,270 windows), and scored with the same scene matching. Ground truth comes from the SHOT label file with the same transition-to-scene conversion as ClipShots; 720 of 775 transitions lie within the decoded video length and are used. The label file is the only ground truth available for these videos and is not the authors' final annotation, so validation F1 is noisier than test F1 and the selected thresholds are noisier too.

## 3.2 Threshold selection and application

τ_c ∈ {0.10, …, 0.90} and τ_g ∈ {0.3, 0.4, 0.5, 0.6} were searched once per model by maximising F1 on the full validation videos with ℓ_min = 1. The selected values and the validation F1 are in the thresholds table (3.6). They were applied unchanged to the three test sets for every model, including the new seeds. Numbers are expected to be lower than under Protocol A, especially on ClipShots and BBC.

## Results

### 3.3 Supervision (Table 7a, seed 42)

| ID | Description | SHOT F1 | ClipShots F1 | BBC F1 |
|---|---|---|---|---|
| A0 | Full, dual-label (primary) | 0.8084 | **0.7658** | 0.9439 |
| A1 | Full, midpoint only | 0.8313 | 0.7352 | 0.9597 |
| A2 | Full, unified single head | 0.8350 | 0.7589 | 0.9473 |
| A3 | Reduced, midpoint only | 0.8223 | 0.7562 | **0.9713** |
| A4 | Reduced, dual-label | **0.8364** | 0.7627 | 0.9660 |

Differences:

| Delta | SHOT | ClipShots | BBC |
|---|---|---|---|
| A0 − A1 (full) | -0.0230 | +0.0305 | -0.0158 |
| A4 − A3 (reduced) | +0.0141 | +0.0065 | -0.0054 |
| A0 − A2 (dual vs unified) | -0.0267 | +0.0069 | -0.0034 |

### 3.4 Components, leave-one-out (Table 7b)

| ID | Description | SHOT F1 | ClipShots F1 | BBC F1 |
|---|---|---|---|---|
| A0 | Full model | 0.8084 | 0.7658 | 0.9439 |
| C1 | w/o difference-based branch (B) | 0.8357 | 0.7758 | 0.9444 |
| C2a | w/o FrameSimilarity | **0.8408** | 0.7265 | 0.9449 |
| C2b | w/o ColorHistogramSimilarity | 0.8260 | 0.7513 | 0.9546 |
| C3 | w/o temporal delta | 0.8208 | 0.7709 | 0.9583 |
| C4 | w/o per-frame pathway (relations from Branch A) | 0.8310 | 0.7920 | 0.9605 |
| C5a | w/o pixel-level motion descriptor | 0.8154 | **0.7951** | 0.9663 |
| C5b | w/o edge-density head | 0.8340 | 0.7638 | **0.9680** |
| C6 | w/o self-attention and gate | 0.8147 | 0.7708 | 0.9534 |

Change in F1 relative to A0:

| ID | ΔSHOT | ΔClipShots | ΔBBC |
|---|---|---|---|
| C1 | +0.0274 | +0.0101 | +0.0005 |
| C2a | +0.0325 | -0.0393 | +0.0009 |
| C2b | +0.0176 | -0.0144 | +0.0107 |
| C3 | +0.0124 | +0.0052 | +0.0144 |
| C4 | +0.0226 | +0.0262 | +0.0165 |
| C5a | +0.0070 | +0.0294 | +0.0224 |
| C5b | +0.0257 | -0.0020 | +0.0241 |
| C6 | +0.0063 | +0.0050 | +0.0095 |

Largest single-component change on any dataset: 0.0393.

### 3.5 Build-up ladder

| Step | Run | What is added | SHOT F1 | ClipShots F1 | BBC F1 |
|---|---|---|---|---|---|
| 0 | M0_min | RGB encoder + MLP + dual head only | 0.8124 | 0.7396 | 0.9502 |
| 1 | M1_edge | + edge-density head | 0.8082 | 0.7633 | 0.9513 |
| 2 | B_noatt | + 4 fusion blocks, no attention | 0.8259 | 0.7791 | 0.9525 |
| 3 | A4 | + attention and gate (reduced model) | **0.8364** | 0.7627 | **0.9660** |
| 4 | S1_rel | + FrameSimilarity, ColorHist, delta | 0.8275 | **0.7862** | 0.9534 |
| 5 | A0 | + diff branch, motion, per-frame pathway (full model) | 0.8084 | 0.7658 | 0.9439 |

Change at each step:

| Step | SHOT | ClipShots | BBC |
|---|---|---|---|
| M0_min → M1_edge | -0.0042 | +0.0236 | +0.0011 |
| M1_edge → B_noatt | +0.0177 | +0.0159 | +0.0012 |
| B_noatt → A4 | +0.0105 | -0.0164 | +0.0135 |
| A4 → S1_rel | -0.0089 | +0.0234 | -0.0125 |
| S1_rel → A0 | -0.0191 | -0.0204 | -0.0095 |

### 3.6 Precision, recall and selected thresholds

| ID | τ_c, τ_g (from validation) | Validation F1 | SHOT P / R | ClipShots P / R | BBC P / R |
|---|---|---|---|---|---|
| A0 | 0.6, 0.4 | 0.7749 | 0.883 / 0.745 | 0.684 / 0.869 | 0.929 / 0.959 |
| A1 | 0.55, – | 0.7482 | 0.909 / 0.766 | 0.744 / 0.726 | 0.983 / 0.938 |
| A2 | 0.55, – | 0.7583 | 0.892 / 0.785 | 0.679 / 0.859 | 0.939 / 0.956 |
| A3 | 0.5, – | 0.7413 | 0.888 / 0.766 | 0.730 / 0.784 | 0.977 / 0.966 |
| A4 | 0.45, 0.5 | 0.7405 | 0.865 / 0.810 | 0.665 / 0.894 | 0.948 / 0.985 |
| C1 | 0.55, 0.3 | 0.7712 | 0.899 / 0.781 | 0.700 / 0.870 | 0.933 / 0.956 |
| C2a | 0.5, 0.4 | 0.7618 | 0.863 / 0.820 | 0.613 / 0.891 | 0.917 / 0.974 |
| C2b | 0.45, 0.3 | 0.7632 | 0.878 / 0.780 | 0.648 / 0.893 | 0.927 / 0.983 |
| C3 | 0.55, 0.5 | 0.7588 | 0.863 / 0.783 | 0.685 / 0.882 | 0.950 / 0.966 |
| C4 | 0.55, 0.5 | 0.7664 | 0.887 / 0.782 | 0.723 / 0.876 | 0.953 / 0.968 |
| C5a | 0.55, 0.6 | 0.7581 | 0.898 / 0.747 | 0.739 / 0.861 | 0.968 / 0.964 |
| C5b | 0.45, 0.5 | 0.7672 | 0.863 / 0.807 | 0.665 / 0.898 | 0.952 / 0.985 |
| C6 | 0.55, 0.5 | 0.7539 | 0.862 / 0.772 | 0.686 / 0.880 | 0.946 / 0.961 |
| M0_min | 0.5, 0.6 | 0.7235 | 0.834 / 0.792 | 0.686 / 0.803 | 0.972 / 0.930 |
| M1_edge | 0.5, 0.6 | 0.7363 | 0.879 / 0.748 | 0.720 / 0.813 | 0.973 / 0.931 |
| B_noatt | 0.55, 0.5 | 0.7387 | 0.872 / 0.784 | 0.704 / 0.872 | 0.949 / 0.956 |
| S1_rel | 0.5, 0.4 | 0.7598 | 0.894 / 0.770 | 0.717 / 0.871 | 0.931 / 0.977 |
| A0-s43 | 0.5, 0.4 | 0.7751 | 0.907 / 0.754 | 0.677 / 0.863 | 0.929 / 0.948 |
| A0-s44 | 0.45, 0.4 | 0.7625 | 0.870 / 0.828 | 0.675 / 0.883 | 0.924 / 0.981 |
| A1-s43 | 0.55, – | 0.7588 | 0.909 / 0.764 | 0.763 / 0.711 | 0.984 / 0.926 |
| A1-s44 | 0.55, – | 0.7602 | 0.902 / 0.757 | 0.736 / 0.746 | 0.981 / 0.939 |

A1, A2, A3 and their seeds use τ_c only.

### 3.7 Table 8 — Per-type analysis on ClipShots

| ID | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |
|---|---|---|---|---|
| A0 | 0.8712 | 0.8640 | 0.6488 | 0.7338 |
| A1 | 0.8810 | 0.3968 | n/a | n/a |
| A2 | 0.8718 | 0.8331 | n/a | 0.7218 |

### 3.8 Table 9 — Recall by gradual-transition duration on ClipShots

| ID | ≤ 10 frames | 11–30 frames | > 30 frames |
|---|---|---|---|
| A0 | 0.8341 (n = 832) | 0.8759 (n = 1241) | 0.9079 (n = 228) |
| C6 | 0.8257 (n = 832) | 0.8767 (n = 1241) | 0.8421 (n = 228) |

C6 − A0: ≤ 10 -0.0084, 11–30 +0.0008, > 30 -0.0658.

### 3.9 Seeds table

F1 of A0 and A1 at seeds 42, 43 and 44 (seed 42 = the main-study run). Mean and sample standard deviation (n − 1) over the three seeds.

| Model | Test set | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|---|---|---|---|---|---|---|
| A0 | SHOT | 0.8084 | 0.8236 | 0.8483 | 0.8268 | 0.0202 |
| A0 | ClipShots | 0.7658 | 0.7588 | 0.7653 | 0.7633 | 0.0039 |
| A0 | BBC | 0.9439 | 0.9383 | 0.9514 | 0.9446 | 0.0066 |
| A1 | SHOT | 0.8313 | 0.8303 | 0.8228 | 0.8281 | 0.0046 |
| A1 | ClipShots | 0.7352 | 0.7360 | 0.7410 | 0.7374 | 0.0032 |
| A1 | BBC | 0.9597 | 0.9545 | 0.9594 | 0.9579 | 0.0029 |

**Paired difference A0 − A1** (same seed for both):

| Test set | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|---|---|---|---|---|---|
| SHOT | -0.0230 | -0.0067 | +0.0255 | -0.0014 | 0.0247 |
| ClipShots | +0.0305 | +0.0228 | +0.0242 | +0.0259 | 0.0041 |
| BBC | -0.0158 | -0.0162 | -0.0080 | -0.0133 | 0.0046 |

**Cut recall and gradual recall on ClipShots:**

| Model | Metric | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|---|---|---|---|---|---|---|
| A0 | Cut recall | 0.8712 | 0.8663 | 0.9067 | 0.8814 | 0.0220 |
| A0 | Gradual recall | 0.8640 | 0.8561 | 0.8340 | 0.8514 | 0.0156 |
| A1 | Cut recall | 0.8810 | 0.8725 | 0.8775 | 0.8770 | 0.0043 |
| A1 | Gradual recall | 0.3968 | 0.3659 | 0.4654 | 0.4094 | 0.0509 |

### 3.10 Interpretation check (0.01 band)

| Statement | Value | Holds? |
|---|---|---|
| A0 − A1 on ClipShots ≥ 0.01, seed 42 | +0.0305 | **yes** |
| A0 − A1 on ClipShots ≥ 0.01, mean over the three seeds | +0.0259 | **yes** |
| A0 − A1 on ClipShots positive for every seed | +0.0305 / +0.0228 / +0.0242 | **yes** |
| A4 − A3 on ClipShots ≥ 0.01 | +0.0065 | **no** |
| A0 − A2 on ClipShots ≥ 0.01 | +0.0069 | **no** |
| A0 − A2 on BBC ≥ 0.01 | -0.0034 | **no** |
| No leave-one-out change (C1–C6) reaches 0.01 on any test set | largest 0.0393 | **no** |

Differences below 0.01 F1 are treated as inconclusive. Under Protocol B the thresholds themselves add noise, so single-model differences are less reliable than under Protocol A.

---

# Part 4: Additional check — SHOT validation F1 for every epoch

The validation F1 of every epoch for A0 and A1 at seed 42 and for the four new runs is in Table 1.3. What happens after each run's best epoch:

## 4.1 Behaviour after the best epoch

| Run | Best epoch | Best val F1 | Stopped at epoch | Range after best | Last | Trend per epoch |
|---|---|---|---|---|---|---|
| A1 | 7 | 0.7863 | 17 | 0.7695–0.7852 | 0.7846 | +0.00024 |
| A0 | 21 | 0.8029 | 31 | 0.7810–0.7953 | 0.7870 | -0.00066 |
| A1-s43 | 10 | 0.7915 | 20 | 0.7733–0.7901 | 0.7830 | -0.00053 |
| A1-s44 | 9 | 0.7967 | 19 | 0.7812–0.7949 | 0.7898 | -0.00021 |
| A0-s43 | 13 | 0.7953 | 23 | 0.7781–0.7933 | 0.7933 | +0.00037 |
| A0-s44 | 16 | 0.7935 | 26 | 0.7820–0.7912 | 0.7850 | -0.00036 |

## 4.2 Did validation F1 level off or fall after the best epoch?

In every one of the six runs, including A1 at seed 42, early stopping ended training after 10 epochs without improvement, and validation F1 had **levelled off** rather than still rising or collapsing: after its best epoch each run moves up and down inside a band about 0.01–0.02 wide just below its best value, with a linear trend close to zero (between −0.0007 and +0.0004 F1 per epoch), and no run was still climbing when it stopped. A1 at seed 42 (best epoch 7, 0.7863) fluctuated between 0.7695 and 0.7852 afterwards and ended at 0.7846, so it was levelled off and unstable at the 0.01 level, not falling steadily. The new A1 runs behave the same way (A1-s43 drifts slightly down, −0.0005 per epoch; A1-s44 is flat, −0.0002). The epoch-to-epoch swings are about as large as the gaps between runs, so the best-epoch choice itself is noisy.
