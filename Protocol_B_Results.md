# Protocol B results: thresholds selected on validation

This report gives the Protocol B evaluation of the trained HybridSBD ablation models (leak-free SHOT training set of 448 videos). In Protocol B the decision thresholds are **not** tuned on the test sets: for each model, τ_c and τ_g are selected once, by maximising F1 on the 67 SHOT validation videos, and then applied unchanged to the SHOT test set (200 videos), the ClipShots test set (500 videos) and BBC (11 episodes). The minimum shot length is ℓ_min = 1 (no merging) everywhere. Everything else is as in the main study: existing best checkpoints (EMA weights selected by SHOT validation F1), sliding windows (T = 128, stride 64) averaged back to a per-video track, threshold grids τ_c ∈ {0.10, 0.15, …, 0.90} and τ_g ∈ {0.3, 0.4, 0.5, 0.6}, the decision rule (dual-label models: boundary if p_y > τ_c, gradual if p_z > τ_g, track = gradual OR cut; midpoint-only and unified models: τ_c only) and scoring with ±2-frame scene matching, with precision, recall and F1 pooled over all videos. Numbers are expected to be lower than with per-test-set tuning (Protocol A), especially on ClipShots and BBC. Single training seed (42) per model.

## 1. Validation data and threshold selection

The SHOT validation split (15 % of the 448 training videos, 67 videos) was previously available only as pre-cut training clips. For Protocol B these 67 videos were decoded completely, with the same decoder, resizing (48 × 64), window layout and window averaging as the SHOT test set (1,270 windows). Ground truth comes from the SHOT label file with the same transition-to-scene conversion as ClipShots; 720 of 775 transitions lie within the decoded video length and are used. The label file is the only ground truth available for these videos and is not the authors' final annotation, so validation F1 is noisier than test F1, and the thresholds it selects are noisier too. The validation videos were also used for early stopping and checkpoint selection, so validation F1 is slightly optimistic.

Validation-selected thresholds and validation F1 are listed in section 2.4.

## 2. Protocol B (thresholds selected once on the 67 SHOT validation videos, applied unchanged)

Each model's τ_c and τ_g were selected once, by maximising F1 on the 67 SHOT validation videos evaluated as full videos (same windows, averaging and scoring as the test sets, ℓ_min = 1), and then applied unchanged to SHOT, ClipShots and BBC. Nothing was tuned on a test set. Validation F1 and the selected thresholds are in the thresholds table.

### 2.1 Supervision (Table 7a, seed 42)

| ID | Description | SHOT F1 | ClipShots F1 | BBC F1 |
|---|---|---|---|---|
| A0 | Full, dual-label (primary) | 0.8084 | **0.7658** | 0.9439 |
| A1 | Full, midpoint only | 0.8313 | 0.7352 | 0.9597 |
| A2 | Full, unified single head | 0.8350 | 0.7589 | 0.9473 |
| A3 | Reduced, midpoint only | 0.8223 | 0.7562 | **0.9713** |
| A4 | Reduced, dual-label | **0.8364** | 0.7627 | 0.9660 |

| Delta | SHOT | ClipShots | BBC |
|---|---|---|---|
| A0 − A1 (full) | -0.0230 | +0.0305 | -0.0158 |
| A4 − A3 (reduced) | +0.0141 | +0.0065 | -0.0054 |
| A0 − A2 (dual vs unified) | -0.0267 | +0.0069 | -0.0034 |

### 2.2 Components, leave-one-out (Table 7b)

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

### 2.3 Build-up ladder

| Step | Run | What is added | SHOT F1 | ClipShots F1 | BBC F1 |
|---|---|---|---|---|---|
| 0 | M0_min | RGB encoder + MLP + dual head only | 0.8124 | 0.7396 | 0.9502 |
| 1 | M1_edge | + edge-density head | 0.8082 | 0.7633 | 0.9513 |
| 2 | B_noatt | + 4 fusion blocks, no attention | 0.8259 | 0.7791 | 0.9525 |
| 3 | A4 | + attention and gate (reduced model) | **0.8364** | 0.7627 | **0.9660** |
| 4 | S1_rel | + FrameSimilarity, ColorHist, delta | 0.8275 | **0.7862** | 0.9534 |
| 5 | A0 | + diff branch, motion, per-frame pathway (full model) | 0.8084 | 0.7658 | 0.9439 |

| Step | SHOT | ClipShots | BBC |
|---|---|---|---|
| M0_min → M1_edge | -0.0042 | +0.0236 | +0.0011 |
| M1_edge → B_noatt | +0.0177 | +0.0159 | +0.0012 |
| B_noatt → A4 | +0.0105 | -0.0164 | +0.0135 |
| A4 → S1_rel | -0.0089 | +0.0234 | -0.0125 |
| S1_rel → A0 | -0.0191 | -0.0204 | -0.0095 |

### 2.4 Precision, recall and selected thresholds

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

### 2.5 Table 8 — Per-type analysis on ClipShots

| ID | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |
|---|---|---|---|---|
| A0 | 0.8712 | 0.8640 | 0.6488 | 0.7338 |
| A1 | 0.8810 | 0.3968 | n/a | n/a |
| A2 | 0.8718 | 0.8331 | n/a | 0.7218 |

### 2.6 Table 9 — Recall by gradual-transition duration on ClipShots

| ID | ≤ 10 frames | 11–30 frames | > 30 frames |
|---|---|---|---|
| A0 | 0.8341 (n = 832) | 0.8759 (n = 1241) | 0.9079 (n = 228) |
| C6 | 0.8257 (n = 832) | 0.8767 (n = 1241) | 0.8421 (n = 228) |

C6 − A0: ≤ 10 -0.0084, 11–30 +0.0008, > 30 -0.0658.

## 3. Protocol B compared with Protocol A (ℓ_min = 1)

Protocol A selects thresholds on each test set (optimistic); Protocol B does not. F1 under each protocol and the difference:

| ID | SHOT A | SHOT B | Δ | ClipShots A | ClipShots B | Δ | BBC A | BBC B | Δ |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.8449 | 0.8084 | -0.0366 | 0.8001 | 0.7658 | -0.0344 | 0.9719 | 0.9439 | -0.0279 |
| A1 | 0.8478 | 0.8313 | -0.0165 | 0.7609 | 0.7352 | -0.0257 | 0.9698 | 0.9597 | -0.0101 |
| A2 | 0.8509 | 0.8350 | -0.0159 | 0.7709 | 0.7589 | -0.0120 | 0.9473 | 0.9473 | -0.0000 |
| A3 | 0.8312 | 0.8223 | -0.0089 | 0.7606 | 0.7562 | -0.0044 | 0.9714 | 0.9713 | -0.0000 |
| A4 | 0.8370 | 0.8364 | -0.0006 | 0.8045 | 0.7627 | -0.0418 | 0.9721 | 0.9660 | -0.0061 |
| C1 | 0.8536 | 0.8357 | -0.0179 | 0.7942 | 0.7758 | -0.0184 | 0.9701 | 0.9444 | -0.0257 |
| C2a | 0.8457 | 0.8408 | -0.0048 | 0.8009 | 0.7265 | -0.0744 | 0.9681 | 0.9449 | -0.0233 |
| C2b | 0.8361 | 0.8260 | -0.0101 | 0.8077 | 0.7513 | -0.0564 | 0.9730 | 0.9546 | -0.0184 |
| C3 | 0.8363 | 0.8208 | -0.0155 | 0.7956 | 0.7709 | -0.0247 | 0.9728 | 0.9583 | -0.0144 |
| C4 | 0.8420 | 0.8310 | -0.0110 | 0.8068 | 0.7920 | -0.0148 | 0.9699 | 0.9605 | -0.0094 |
| C5a | 0.8487 | 0.8154 | -0.0333 | 0.8021 | 0.7951 | -0.0070 | 0.9715 | 0.9663 | -0.0053 |
| C5b | 0.8454 | 0.8340 | -0.0113 | 0.7967 | 0.7638 | -0.0329 | 0.9733 | 0.9680 | -0.0053 |
| C6 | 0.8364 | 0.8147 | -0.0217 | 0.7969 | 0.7708 | -0.0262 | 0.9667 | 0.9534 | -0.0133 |
| M0_min | 0.8169 | 0.8124 | -0.0045 | 0.7555 | 0.7396 | -0.0159 | 0.9543 | 0.9502 | -0.0041 |
| M1_edge | 0.8261 | 0.8082 | -0.0179 | 0.7672 | 0.7633 | -0.0039 | 0.9584 | 0.9513 | -0.0071 |
| B_noatt | 0.8351 | 0.8259 | -0.0093 | 0.7984 | 0.7791 | -0.0193 | 0.9682 | 0.9525 | -0.0157 |
| S1_rel | 0.8392 | 0.8275 | -0.0117 | 0.8054 | 0.7862 | -0.0192 | 0.9706 | 0.9534 | -0.0171 |

Mean change over the 17 models: SHOT -0.0146 (range -0.0366 to -0.0006), ClipShots -0.0254 (range -0.0744 to -0.0039), BBC -0.0120 (range -0.0279 to -0.0000). No model scores higher under Protocol B than under Protocol A on any test set, as expected, since Protocol A picks the best thresholds on the test set itself.

## 4. Interpretation check (0.01 band)

| Statement | Value | Holds? |
|---|---|---|
| A0 − A1 on ClipShots ≥ 0.01, seed 42 | +0.0305 | **yes** |
| A0 − A1 on ClipShots ≥ 0.01, mean over the three seeds | +0.0259 | **yes** |
| A0 − A1 on ClipShots positive for every seed | +0.0305 / +0.0228 / +0.0242 | **yes** |
| A4 − A3 on ClipShots ≥ 0.01 | +0.0065 | **no** |
| A0 − A2 on ClipShots ≥ 0.01 | +0.0069 | **no** |
| A0 − A2 on BBC ≥ 0.01 | -0.0034 | **no** |
| No leave-one-out change (C1–C6) reaches 0.01 on any test set | largest 0.0393 | **no** |

Differences below 0.01 F1 are treated as inconclusive. Under Protocol B the thresholds themselves add noise: a model's validation-selected τ_c / τ_g can sit far from its best test thresholds, so single-model differences (for example in the leave-one-out table) are less reliable than under Protocol A.

## 5. SHOT validation F1 per epoch (training histories)

| Epoch | A1 | A0 |
|---|---|---|
| 1 | 0.7029 | 0.6988 |
| 2 | 0.7499 | 0.7514 |
| 3 | 0.7535 | 0.7543 |
| 4 | 0.7628 | 0.7774 |
| 5 | 0.7807 | 0.7758 |
| 6 | 0.7796 | 0.7722 |
| 7 | **0.7863** | 0.7725 |
| 8 | 0.7785 | 0.7748 |
| 9 | 0.7695 | 0.7654 |
| 10 | 0.7852 | 0.7838 |
| 11 | 0.7843 | 0.7793 |
| 12 | 0.7783 | 0.7843 |
| 13 | 0.7824 | 0.7821 |
| 14 | 0.7704 | 0.7681 |
| 15 | 0.7840 | 0.7885 |
| 16 | 0.7735 | 0.7926 |
| 17 | 0.7846 | 0.7810 |
| 18 |  | 0.7932 |
| 19 |  | 0.7946 |
| 20 |  | 0.7960 |
| 21 |  | **0.8029** |
| 22 |  | 0.7935 |
| 23 |  | 0.7907 |
| 24 |  | 0.7953 |
| 25 |  | 0.7854 |
| 26 |  | 0.7847 |
| 27 |  | 0.7913 |
| 28 |  | 0.7951 |
| 29 |  | 0.7810 |
| 30 |  | 0.7886 |
| 31 |  | 0.7870 |

Best epoch in bold. This is the validation F1 used for early stopping (pre-cut validation clips, fixed thresholds), not the full-video validation F1 of section 1.
