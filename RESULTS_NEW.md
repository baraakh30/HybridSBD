# RESULTS_NEW — HybridSBD Protocol-A ablation (leak-free)

All numbers in this document come from the saved experiment results (training histories, per-dataset threshold-search results and speed measurements). One seed (42) per variant; per the spec, F1 differences below 0.01 are treated as inconclusive. Thresholds are tuned on each test set (Protocol A), so absolute F1 is optimistic by construction and comparable only to other Protocol-A numbers.

## 1. Data and training approach

### 1.1 Datasets

| Role | Dataset | Videos | Notes |
|---|---|---|---|
| Train | ClipShots train | 3,656 | 3,657 listed; one source file (`5186782609.mp4`) is unreadable and is explicitly allow-listed as missing |
| Train / validation | SHOT (AutoShot, `kuaishou_v2.txt`) | 448 | 648 labelled videos minus the 200 SHOT test videos |
| Test | SHOT test | 200 | the authors' released SHOT test annotation |
| Test | ClipShots test | 500 | official ClipShots test annotations |
| Zero-shot test | BBC Planet Earth | 11 episodes | Zenodo record 14865504; never used for training or selection |

**SHOT label coverage.** The SHOT download holds 878 video files in three folders (854 distinct videos), but only **648 have labels** in `kuaishou_v2.txt`; the other **206 videos have no annotations at all** and cannot be used for supervised training, so they are not used. All 648 labelled videos are present on disk. The label file is also irregular, and a tolerant parser is used: 738 header entries for 648 videos (90 videos appear twice; the 45 non-test duplicates are identical), mixed `a,b` / `a b` line formats, one header ending in `.mp4.png`, and 2 lines with an odd number of integers that are skipped. A strict parser would silently drop 42 of 9,274 transition lines (0.5 %).

For the 200 test videos the file contains an earlier version of the labels: 2,593 labels against 2,516 transitions in the final test annotation, 2,398 of them within ±2 frames of the final annotation. Training on these videos therefore means training on almost the exact test answers.

**Limitation of the released SHOT labels.** The released label file mixes the training labels and the labels of the 200 test videos in one file, so the test videos can only be told apart with the separate test annotation. It also covers far fewer videos than the dataset description: the AutoShot paper describes 853 videos, of which 200 are test videos, which would leave 653 for training. The label file, however, covers only 648 videos and these include all 200 test videos; roughly 206 videos have no labels at all. After excluding the test videos, 448 labelled videos remain for training, about 205 fewer than the paper's split implies. Our SHOT results therefore use less training data than the authors' setup could, which is a limitation of the released data, and more labelled training videos could improve them.

**Test videos are excluded from training.** The SHOT label file contains the 200 test videos, so training on all 648 would train on the test set. Training uses the 448 other videos, and an overlap audit across all train/test sets (SHOT, ClipShots, BBC) finds zero shared videos.

### 1.2 Preparation

1. Archives are downloaded, extracted and checked, then every video is decoded once and each frame is bilinearly resized to **48 × 64** (uint8, RGB). Decoding is chunked to bound memory.
2. Videos are cut into **128-frame clips** stored as tensors with per-frame labels (0 none, 1 cut, 2 gradual) and a transition-length value.
3. **Training clips.** ClipShots: 20 clips per video (60 % centred on a transition with ±15-frame jitter, the rest random), 73,120 clips from 3,656 videos. SHOT: 70 clips per video; every transition gets 3 clips centred on it (jitter −20 / 0 / +20 frames) and the remaining quota is random, 31,360 clips from 448 videos.
4. **Test clips.** Sliding windows of 128 frames with stride 64 cover each full video; overlapping predictions are averaged back into one probability track per video.
5. A verification step checks that every clip loads, has the right shape and label range, and that every expected video is present, before any training starts.

### 1.3 Training

| Setting | Value |
|---|---|
| Targets | y = midpoint of every transition (cuts and graduals); z = every frame inside a gradual (many-hot) |
| Loss | 5.0 · focal BCE(y; α 0.8, γ 2) + 0.3 · focal BCE(z; α 0.5, γ 2, label smoothing 0.1) |
| Optimiser | AdamW, lr 5e-5, weight decay 0.05, batch 32, AMP, gradient clip 1.0 |
| Schedule | up to 80 epochs; 5 warm-up epochs (lr × 0.25 → 1), then cosine to 5e-6 |
| Early stopping | patience 10 on SHOT validation F1 (15 % of the SHOT training videos: 381 train / 67 validation) |
| Epoch mix | all SHOT training clips + a fresh random 70 % of the ClipShots training clips each epoch (ClipShots split 3,108 train / 548 validation videos) |
| Basic augmentation | horizontal flip, per-frame brightness and contrast jitter, noise, brightness shift |
| Strong augmentation | per-clip zoom/shift (up to 25 %), colour gain, occasional grayscale |
| Synthetic transitions | probability 0.25 per clip: hard cut 40 %, dissolve 40 %, fade through black 20 % (3–30 frames), spliced from two clips of the same batch with a 3-frame clear margin from real transitions; labels follow the real-label convention |
| Weight averaging | EMA, decay 0.999; validation, checkpoints and test use the EMA weights |
| Seed | 42 (fixed hash seed); every variant uses the same recipe, data and seed |
| Hardware / software | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, Python 3.12.15, PyTorch 2.14.1+cu132, CUDA 13.2 |

**Synthetic transitions in detail.** Each clip of a batch is replaced, with probability 0.25, by a splice of itself (clip i) and another clip j from the same batch: a **hard cut** (40 %: frames before a random point come from i, after it from j), a **dissolve** (40 %: a linear cross-fade from i to j over 3–30 frames) or a **fade through black** (20 %: i fades to black, then j fades in, over 3–30 frames). Labels before the splice come from i and after it from j; a cut is labelled on the two frames at the splice, and a dissolve or fade is labelled gradual on every frame of the transition, exactly like real transitions. The splice is kept at least 3 frames away from every real transition in either clip. This adds many extra transitions with exactly known boundaries, especially gradual ones, which are rare and hard to annotate, and it uses only frames and labels from the training videos.

### 1.4 Evaluation (Protocol A)

- Per test set, the model's best checkpoint (by SHOT validation F1) is evaluated once over a threshold grid and the best setting is reported: boundary threshold τ_c ∈ {0.10, 0.15, …, 0.90}, gradual threshold τ_g ∈ {0.3, 0.4, 0.5, 0.6}, and minimum scene length ℓ_min ∈ {1, 2, 3, 4} for ClipShots only (ℓ_min = 1 for SHOT and BBC).
- Decision rule (dual-label models): boundary if p_y > τ_c; gradual if p_z > τ_g; cut = boundary and not gradual; the predicted many-hot track is gradual OR cut. Midpoint-only (A1, A3) and unified (A2) variants use τ_c only.
- Scoring: AutoShot's scene-based matching with **±2 frame tolerance**; precision, recall and F1 over transitions, pooled over all videos.

### 1.5 Protocol details (spec Section 0, items 3–8)

- **Minimum shot length ℓ_min** is applied in the evaluation code, shared by all three test sets: after the boundary track is turned into scenes, every predicted scene shorter than ℓ_min frames is merged into the previous scene. The grid {1, 2, 3, 4} is searched for ClipShots only; SHOT and BBC use ℓ_min = 1 (no merging).
- **Overlapping windows.** Probabilities of all windows that cover a frame are summed and divided by the number of windows covering it (plain average, separately for p_y and p_z). Frames near the start or end are covered by one window only and use that window's value.
- **Final window.** ClipShots test: window starts 0, 64, 128, … plus one extra window starting at (frames − 128), so the last window is shifted back and ends on the final frame. SHOT test: a window that would run past the end is shifted back to start at (frames − 128); a video shorter than 128 frames is padded by repeating its last frame. BBC: windows start every 64 frames and the last window is padded by repeating the final frame. The same window and averaging code is used for every run.
- **Seeds.** Clip sampling during data preparation uses an MD5-based stable per-video seed (not Python's `hash`). Training seeds the hash seed, Python, NumPy, PyTorch (CPU and CUDA) and every data-loader worker to the same seed (42) and runs cuDNN in deterministic mode.
- **Benchmark hygiene.** `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.empty_cache()` are called before every latency and VRAM measurement 
- **Data overlap.** No video appears in more than one of SHOT train / SHOT validation / SHOT test / ClipShots train / ClipShots test / BBC.

## 2. Compliance with the specification

### 2.1 A0 reproduction checks (spec Section 1)

**Check 1 — architecture, on the trained A0 checkpoint: PASS.**

| Item | Required | Found |
|---|---|---|
| Fusion blocks | 4 | 4 |
| Positional encoding | none | none |
| Delta projection weight | (64, 640) | (64, 640) |
| D | 384 | 384 |
| Fusion input / embedding dim | 655 / 640 | 655 / 640 |
| Exact parameter count | rounds to 10.9M | 10,938,880 (10.9M) |

**Check 2 — performance (Protocol A, paper's acceptable ranges):**

| Test set | Paper | Acceptable range | Our A0 | Result |
|---|---|---|---|---|
| SHOT F1 | 0.924 | 0.915 – 0.93 | 0.8449 | **below range** |
| ClipShots F1 | 0.793 | 0.785 – 0.8 | 0.8052 | above range |
| BBC F1 | 0.971 | 0.965 – 0.975 | 0.9719 | within range |

Check 2 does **not** pass on SHOT, and this is expected: **it is caused by a deliberate change in the training data, not by the code.** The SHOT label file `kuaishou_v2.txt` lists 648 videos, and the 200 SHOT *test* videos (the authors' released test annotation) are among them. The paper-era setup therefore trained on 648 videos that include the entire SHOT test set — a data leak that inflates SHOT F1. We removed the 200 test videos from training (448 videos remain), so the SHOT numbers here are leak-free and are lower than the paper's 0.924. As a one-off control, the same A0 recipe trained on all 648 videos reaches SHOT 0.9155, which is inside the required 0.915–0.930 range, so the gap is explained by the leak and not by an architecture or pipeline difference (Check 1 passes). Per the spec, this is reported to the team: the ablations proceed on the same code and data policy for all runs, and the main-paper SHOT number needs discussion. ClipShots (0.8052) is slightly above its range and BBC (0.9719) is inside it; neither benchmark shares videos with the SHOT test set.

### 2.2 Deviations from the specification

| Spec says | What was done | Why |
|---|---|---|
| SHOT: 648 training videos (§3) | **448 videos**: the 200 SHOT test videos are excluded | The 648 include the SHOT test set (data leak), see 2.1 |
| Early stopping patience 15 (§3) | patience 10 for all 13 runs and the build-up runs | Shortens training time: the validation optimum appeared early in the first runs, so the same patience was used for every variant |
| Augmentation: flip, brightness, contrast, noise, clip brightness (§3) | all of these, **plus** strong per-clip augmentation, synthetic transitions (p = 0.25) and weight averaging (EMA 0.999) | Added to recover accuracy once the leaked videos were removed; identical for every run, so the comparison between variants stays fair |
| All runs on the same code version as A0 (§1) | A0 and the other 12 runs were trained from two code revisions | A0 was trained first as the recipe experiment; only the launcher scripts differ, the training code is identical |
| 13 runs, nothing else (§5, prompt) | **extra** build-up ladder (4 runs, section 5), speed table with GFLOPs, B = 1 / 32 timings and an AutoShot comparison, one extra C4 cut-recall analysis | Requested additions to identify which components do the work; kept in separate sections |
| Training speed | channels-last memory format | Memory-layout change only; values are mathematically identical |

## 3. Main ablation (Protocol A, 13 runs)

### 3.1 Supervision (Table 7a)

| ID | Variant | Heads | Params | SHOT F1 | ClipShots F1 | BBC F1 | Latency B=16 (ms) |
|---|---|---|---|---|---|---|---|
| A0 | Full, dual-label (primary) | 2 | 10,938,880 | 0.8449 | 0.8052 | 0.9719 | 30.5 |
| A1 | Full, midpoint only | 2 | 10,938,880 | 0.8478 | 0.7625 | 0.9698 | 30.5 |
| A2 | Full, unified single head | 1 | 10,715,775 | **0.8509** | 0.7752 | 0.9473 | 30.5 |
| A3 | Reduced, midpoint only | 2 | 8,983,570 | 0.8312 | 0.7640 | 0.9714 | **10.7** |
| A4 | Reduced, dual-label | 2 | 8,983,570 | 0.8370 | **0.8095** | **0.9721** | **10.7** |

| Delta | SHOT | ClipShots | BBC |
|---|---|---|---|
| A0 − A1 (full) | -0.0029 | +0.0427 | +0.0021 |
| A4 − A3 (reduced) | +0.0058 | +0.0455 | +0.0007 |
| A0 − A2 (dual vs unified) | -0.0060 | +0.0300 | +0.0245 |

### 3.2 Components, leave-one-out (Table 7b)

| ID | Description | Params | SHOT F1 | ClipShots F1 | BBC F1 | Latency B=16 (ms) |
|---|---|---|---|---|---|---|
| A0 | Full model | 10,938,880 | 0.8449 | 0.8052 | 0.9719 | 30.5 |
| C1 | w/o difference-based branch (B) | 9,293,744 | **0.8536** | 0.7980 | 0.9701 | **18.2** |
| C2a | w/o FrameSimilarity | 10,833,504 | 0.8457 | 0.8069 | 0.9681 | 30.4 |
| C2b | w/o ColorHistogramSimilarity | 10,835,936 | 0.8361 | 0.8111 | 0.9730 | 29.9 |
| C3 | w/o temporal delta | 10,848,576 | 0.8363 | 0.8013 | 0.9728 | 30.6 |
| C4 | w/o per-frame pathway (relations from Branch A) | 10,938,880 | 0.8420 | **0.8113** | 0.9699 | 25.3 |
| C5a | w/o pixel-level motion descriptor | 10,927,330 | 0.8487 | 0.8063 | 0.9715 | 29.5 |
| C5b | w/o edge-density head | 10,716,864 | 0.8454 | 0.7999 | **0.9733** | 30.1 |
| C6 | w/o self-attention and gate | 7,393,792 | 0.8364 | 0.8032 | 0.9667 | 29.9 |

Change in F1 relative to A0:

| ID | ΔSHOT | ΔClipShots | ΔBBC |
|---|---|---|---|
| C1 | +0.0087 | -0.0072 | -0.0017 |
| C2a | +0.0008 | +0.0017 | -0.0037 |
| C2b | -0.0089 | +0.0059 | +0.0012 |
| C3 | -0.0086 | -0.0039 | +0.0009 |
| C4 | -0.0029 | +0.0062 | -0.0020 |
| C5a | +0.0037 | +0.0012 | -0.0003 |
| C5b | +0.0004 | -0.0052 | +0.0015 |
| C6 | -0.0085 | -0.0020 | -0.0052 |

Largest single-component change on any dataset: 0.0089. Every leave-one-out change is below 0.01 F1, so none is individually conclusive.

### 3.3 Precision / recall and chosen thresholds

| ID | SHOT P / R | ClipShots P / R | BBC P / R |
|---|---|---|---|
| A0 | 0.842 / 0.848 | 0.787 / 0.824 | 0.963 / 0.980 |
| A1 | 0.858 / 0.838 | 0.708 / 0.826 | 0.975 / 0.964 |
| A2 | 0.856 / 0.846 | 0.753 / 0.799 | 0.926 / 0.969 |
| A3 | 0.869 / 0.797 | 0.710 / 0.827 | 0.966 / 0.977 |
| A4 | 0.839 / 0.835 | 0.800 / 0.819 | 0.971 / 0.973 |
| C1 | 0.864 / 0.844 | 0.785 / 0.811 | 0.968 / 0.972 |
| C2a | 0.854 / 0.837 | 0.793 / 0.821 | 0.970 / 0.966 |
| C2b | 0.839 / 0.833 | 0.786 / 0.837 | 0.967 / 0.979 |
| C3 | 0.813 / 0.861 | 0.771 / 0.834 | 0.961 / 0.985 |
| C4 | 0.857 / 0.827 | 0.805 / 0.817 | 0.957 / 0.983 |
| C5a | 0.841 / 0.857 | 0.786 / 0.828 | 0.960 / 0.983 |
| C5b | 0.839 / 0.852 | 0.773 / 0.828 | 0.964 / 0.983 |
| C6 | 0.839 / 0.834 | 0.789 / 0.817 | 0.954 / 0.979 |

Selected thresholds (τ_c boundary, τ_g gradual, ℓ_min minimum scene length) and training length:

| ID | SHOT τ_c, τ_g | ClipShots τ_c, τ_g, ℓ_min | BBC τ_c, τ_g | Best epoch / epochs run | Train hours |
|---|---|---|---|---|---|
| A0 | 0.4, 0.3 | 0.65, 0.6, 4 | 0.45, 0.6 | 21 / 31 | 5.40 |
| A1 | 0.45, – | 0.45, –, 4 | 0.45, – | 7 / 17 | 2.92 |
| A2 | 0.45, – | 0.65, –, 3 | 0.5, – | 18 / 28 | 5.73 |
| A3 | 0.45, – | 0.45, –, 4 | 0.45, – | 30 / 40 | 5.09 |
| A4 | 0.4, 0.4 | 0.7, 0.6, 4 | 0.5, 0.6 | 30 / 40 | 5.09 |
| C1 | 0.4, 0.3 | 0.7, 0.5, 3 | 0.45, 0.6 | 16 / 26 | 5.19 |
| C2a | 0.45, 0.5 | 0.65, 0.6, 4 | 0.5, 0.6 | 26 / 36 | 8.14 |
| C2b | 0.3, 0.3 | 0.65, 0.6, 3 | 0.45, 0.6 | 32 / 42 | 9.44 |
| C3 | 0.4, 0.4 | 0.65, 0.6, 3 | 0.45, 0.6 | 22 / 32 | 6.81 |
| C4 | 0.5, 0.3 | 0.7, 0.6, 4 | 0.45, 0.6 | 19 / 29 | 6.22 |
| C5a | 0.4, 0.3 | 0.65, 0.6, 4 | 0.45, 0.6 | 19 / 29 | 5.14 |
| C5b | 0.35, 0.4 | 0.65, 0.6, 4 | 0.45, 0.6 | 21 / 31 | 4.53 |
| C6 | 0.45, 0.4 | 0.7, 0.6, 4 | 0.45, 0.6 | 27 / 37 | 7.74 |

### 3.4 Per-type analysis on ClipShots (Tables 8 and 9)

#### Table 8 — Per-type analysis on ClipShots

| Model | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |
| --- | --- | --- | --- | --- |
| A0 | 0.8462 | 0.7771 | 0.7562 | 0.6205 |
| A1 | 0.9032 | 0.6601 | N/A | N/A |
| A2 | 0.7987 | 0.8001 | N/A | 0.6941 |

#### Table 9 — Recall by gradual-transition duration

| Model | ≤ 10 (n = 832) | 11–30 (n = 1241) | > 30 (n = 228) |
| --- | --- | --- | --- |
| A0 | 0.7608 | 0.7937 | 0.7456 |
| C6 | 0.7368 | 0.8147 | 0.7281 |

## 4. Interpretation (spec Section 9)

Single seed per variant; differences below 0.01 F1 are inconclusive.

### 4.1 Supervision

| Dataset | A0 − A1 (full) | A4 − A3 (reduced) | Case (spec §9) |
|---|---|---|---|
| SHOT | -0.0029 | +0.0058 | 3 — neither ≥ 0.01: no accuracy claim |
| ClipShots | +0.0427 | +0.0455 | 1 — both clearly positive: dual-label helps independently of architectural complexity |
| BBC | +0.0021 | +0.0007 | 3 — neither ≥ 0.01: no accuracy claim |

Dual-label supervision therefore helps independently of architectural complexity on ClipShots (the benchmark with many gradual transitions) and has no measurable accuracy effect on SHOT or BBC. It also changes the error profile: cut recall / gradual recall on ClipShots is 0.846 / 0.777 for A0 (dual) against 0.903 / 0.660 for A1 (midpoint only).

### 4.2 Hypotheses

| Comparison | Hypothesis | Outcome |
|---|---|---|
| A0 vs A2 | A0 higher | Confirmed on ClipShots (+0.0300) and BBC (+0.0245); A2 is higher on SHOT (A0 − A2 = -0.0060). Wording that a unified target is harmful is supported on ClipShots and BBC, not on SHOT. |
| C2a / C2b | Largest drops | Not confirmed. The largest drop per dataset is C2b on SHOT (-0.0089), C1 on ClipShots (-0.0072), C6 on BBC (-0.0052); C2a is +0.0008 / +0.0017 / -0.0037 and C2b -0.0089 / +0.0059 / +0.0012 (SHOT / ClipShots / BBC). No single stream stands out; the streams look redundant. |
| C4 | Lower cut recall than A0 | Direction confirmed but small: cut recall 0.8409 vs 0.8462 for A0 (-0.0053); overall F1 -0.0029 SHOT, +0.0062 ClipShots, -0.0020 BBC. The Branch C argument should be weakened. (extra analysis) |
| C6 | Lower recall on > 30-frame transitions | Partly: > 30 frames -0.0175 (n = 228), but the larger drop is ≤ 10 frames (-0.0240) and 11–30 frames improves (+0.0210). Convolutions alone are largely sufficient; C6 has fewer parameters (7.4M vs 10.9M) but no latency benefit on this GPU. |
| Any Cx | Drop | No single component drop reaches 0.01 on any dataset (largest 0.0089); each component is individually not needed. |

The spec's follow-up for a small A0 − A1 difference (two extra seeds of A0 and A1) is not needed for the ClipShots claim, since that difference is far above 0.01; for SHOT and BBC it would be needed to claim any effect.

## 5. Build-up ladder: what does the work?

The leave-one-out runs cannot show which parts matter if parts can substitute for each other, so we also build the model up from a minimal core. Same recipe, data and seed as section 3.

| Step | Run | What is added | Params | SHOT F1 | ClipShots F1 | BBC F1 | Latency B=16 (ms) |
|---|---|---|---|---|---|---|---|
| 0 | M0_min | RGB encoder + MLP + dual head only | 2,281,074 | 0.8169 | 0.7678 | 0.9543 | **8.8** |
| 1 | M1_edge | + edge-density head | 2,387,986 | 0.8261 | 0.7774 | 0.9584 | 9.1 |
| 2 | B_noatt | + 4 fusion blocks, no attention | 5,438,482 | 0.8351 | 0.8065 | 0.9682 | 10.0 |
| 3 | A4 | + attention and gate (reduced model) | 8,983,570 | 0.8370 | **0.8095** | **0.9721** | 10.7 |
| 4 | S1_rel | + FrameSimilarity, ColorHist, delta | 9,282,194 | 0.8392 | 0.8084 | 0.9706 | 11.9 |
| 5 | A0 | + diff branch, motion, per-frame pathway (full model) | 10,938,880 | **0.8449** | 0.8052 | 0.9719 | 30.4 |

**Reading it** (against the 0.01 inconclusive band, single seed):

- The bare core already reaches SHOT 0.8169 / ClipShots 0.7678 / BBC 0.9543, within 0.028 / 0.037 / 0.018 of the full model on SHOT / ClipShots / BBC, with 21 % of its parameters.
- The **temporal fusion blocks** are the largest single step and the only one that clears the 0.01 band (on ClipShots): +0.0091 SHOT, +0.0291 ClipShots, +0.0098 BBC. The edge head adds +0.0091 / +0.0096 / +0.0041 (SHOT / ClipShots / BBC): small, but positive on all three datasets.
- Self-attention (+0.0018 / +0.0030 / +0.0039), the similarity and delta streams (+0.0023 / -0.0011 / -0.0015) and the remaining branches (+0.0057 / -0.0032 / +0.0013) each change every dataset by under 0.01.
- The full model's SHOT advantage over the reduced model is +0.0079, while ClipShots is -0.0043 and BBC -0.0002.
- Dual-label vs a single unified head (A0 − A2): -0.0060 SHOT, +0.0300 ClipShots, +0.0245 BBC.
- Dual-label supervision is the one effect clearly above the band: ClipShots +0.0427 (full) and +0.0455 (reduced), with SHOT and BBC unchanged within 0.006.

## 6. Speed, memory and size

FP32, synthetic input `(B, 128, 3, 48, 64)`, forward pass only, 10 warm-up + 80 timed runs, otherwise idle GPU (NVIDIA RTX PRO 6000 Blackwell Workstation Edition). Peak VRAM is the maximum allocated during the forward pass. GFLOPs are per 128-frame clip and count convolutions and matrix multiplies only. AutoShot uses its native 27 × 48 input; our models use 48 × 64.

**Size, compute and peak VRAM**

| Model | Params (M) | FP32 (MB) | GFLOPs / clip | VRAM B=1 (MB) | VRAM B=16 (MB) | VRAM B=32 (MB) |
|---|---|---|---|---|---|---|
| A0 | 10.94 | 41.7 | 29.43 | 216 | 1707 | 3298 |
| A1 | 10.94 | 41.7 | 29.43 | 216 | 1707 | 3298 |
| A2 | 10.72 | 40.9 | 29.37 | 214 | 1706 | 3296 |
| A3 | 8.98 | 34.3 | 10.79 | 154 | 941 | 1781 |
| A4 | 8.98 | 34.3 | 10.79 | 154 | 941 | 1781 |
| C1 | 9.29 | 35.5 | 19.24 | 158 | 951 | 1798 |
| C2a | 10.83 | 41.3 | 29.34 | 216 | 1707 | 3298 |
| C2b | 10.84 | 41.3 | 29.40 | 216 | 1707 | 3298 |
| C3 | 10.85 | 41.4 | 29.41 | 216 | 1707 | 3298 |
| C4 | 10.94 | 41.7 | 21.13 | 216 | 1707 | 3298 |
| C5a | 10.93 | 41.7 | 29.43 | 216 | 1707 | 3298 |
| C5b | 10.72 | 40.9 | 28.72 | 214 | 1705 | 3294 |
| C6 | 7.39 | 28.2 | 28.42 | 188 | 1679 | 3271 |
| M0_min | **2.28** | **8.7** | **8.78** | **102** | **889** | **1729** |
| M1_edge | 2.39 | 9.1 | 9.01 | 103 | 890 | 1730 |
| B_noatt | 5.44 | 20.7 | 9.79 | 128 | 915 | 1755 |
| S1_rel | 9.28 | 35.4 | 10.94 | 157 | 945 | 1785 |
| AutoShot (TransNetV2Supernet) | 14.30 | 54.5 | 96.83 | 1094 | 16205 | 32321 |

**Latency and throughput**

| Model | ms B=1 | ms B=16 | ms B=32 | frames/s B=1 | frames/s B=16 | frames/s B=32 |
|---|---|---|---|---|---|---|
| A0 | 5.0 | 30.4 | 65.9 | 25729 | 67261 | 62149 |
| A1 | 3.9 | 30.5 | 66.1 | 32454 | 67050 | 61970 |
| A2 | 3.9 | 30.5 | 66.1 | 32889 | 67135 | 61985 |
| A3 | 2.1 | 10.7 | 22.6 | 61737 | 190879 | 180931 |
| A4 | 2.1 | 10.7 | 22.7 | 61085 | 190815 | 180806 |
| C1 | 3.3 | 18.3 | 39.4 | 38443 | 111861 | 103896 |
| C2a | 3.7 | 30.6 | 66.2 | 34587 | 67026 | 61834 |
| C2b | 3.8 | 30.1 | 65.1 | 33370 | 68046 | 62905 |
| C3 | 3.9 | 30.8 | 66.6 | 32507 | 66487 | 61462 |
| C4 | 3.7 | 25.4 | 54.6 | 34714 | 80553 | 75005 |
| C5a | 3.8 | 29.8 | 64.0 | 33481 | 68748 | 64009 |
| C5b | 3.7 | 30.4 | 65.7 | 34191 | 67427 | 62370 |
| C6 | 3.3 | 30.1 | 65.5 | 38709 | 68005 | 62504 |
| M0_min | **0.6** | **8.8** | **19.6** | **208153** | **231509** | **208935** |
| M1_edge | 0.7 | 9.1 | 20.0 | 184985 | 225810 | 204543 |
| B_noatt | 1.4 | 10.0 | 21.6 | 90046 | 205255 | 190013 |
| S1_rel | 2.6 | 11.9 | 24.8 | 49568 | 172295 | 165449 |
| AutoShot (TransNetV2Supernet) | 10.4 | 151.2 | 307.4 | 12364 | 13542 | 13324 |

| Model | B=16 speedup vs AutoShot | B=16 VRAM vs AutoShot | GFLOPs vs AutoShot |
|---|---|---|---|
| M0_min | 17.1× faster | 18.2× less | 11.0× fewer |
| M1_edge | 16.7× faster | 18.2× less | 10.7× fewer |
| B_noatt | 15.2× faster | 17.7× less | 9.9× fewer |
| A4 | 14.1× faster | 17.2× less | 9.0× fewer |
| S1_rel | 12.7× faster | 17.2× less | 8.9× fewer |
| C1 | 8.3× faster | 17.0× less | 5.0× fewer |
| A0 | 5.0× faster | 9.5× less | 3.3× fewer |

The diff branch (a second encoder) accounts for most of the full model's cost: A0 30.4 ms vs C1 (no diff branch) 18.3 ms vs A4 10.7 ms at B=16.

### 6.1 Re-measurement of the previously reported speed table

The original checkpoints are not available, but latency, memory and parameter count depend only on the architecture, so the six earlier model variants (different width D and number of fusion blocks) were rebuilt with the same settings and random weights and timed with the same protocol (batch 8 / 16 / 32). The old table was measured on an RTX 4090 and this one on the NVIDIA RTX PRO 6000 Blackwell Workstation Edition.

| Model | Params old → new (M) | B=8 ms old → new | B=16 ms old → new | B=32 ms old → new | B=16 VRAM MB old → new |
|---|---|---|---|---|---|
| D = 512, 3 fusion blocks | 14.13 → 14.13 | 35.6 → 14.8 | 76.9 → 30.4 | 158.5 → 66.2 | 1437 → 1731 |
| D = 256, 3 fusion blocks | 5.79 → 5.79 | 34.2 → 14.4 | 74.2 → 29.5 | 152.8 → 64.3 | 1309 → 1665 |
| D = 384, 4 fusion blocks (A0 architecture) | 10.94 → 10.94 | 35.8 → 15.0 | 76.4 → 30.5 | 158.5 → 65.9 | 1388 → 1706 |
| D = 256, 4 fusion blocks | 6.53 → 6.53 | 33.8 → 14.8 | 73.0 → 30.0 | 155.4 → 65.1 | 1321 → 1671 |
| D = 512, 4 fusion blocks | 17.05 → 17.05 | 37.0 → 15.4 | 79.3 → 31.3 | 161.1 → 67.8 | 1482 → 1754 |
| D = 384, 3 fusion blocks | 9.29 → 9.29 | 35.3 → 14.8 | 76.5 → 30.3 | 157.9 → 65.7 | 1364 → 1692 |
| AutoShot (TransNetV2Supernet) | 14.30 → 14.30 | 121.1 → 74.5 | 1351.5 → 155.3 | 2835.6 → 307.3 | 16183 → 16205 |

- Parameter counts reproduce exactly, and AutoShot's VRAM reproduces to within about 22 MB.
- Latency does not reproduce in absolute terms because the GPU is different (the new one is much faster): our models run at about 30 ms instead of 73–79 ms at B=16, and AutoShot at 155 ms instead of 1,351 ms.
- The old note that AutoShot's latency jumps ~11× from B=8 to B=16 (121 → 1352 ms) is not seen here (74 → 155 → 307 ms at B=8/16/32: roughly linear). With this GPU the B=16 speedup of the old dim-384 4-block model over AutoShot is 5.1×, not the 17.7× reported before.
- Peak memory of our models measures 272–356 MB higher than in the old table at B=16 (peak allocation differs by environment).
- TransNetV2 could not be re-measured (no PyTorch weights or code in this environment), so its old numbers are not reproduced here.
- The measurements on the new, faster GPU are the ones used in this study; the old table should be read as RTX 4090 numbers.

## 7. Caveats

- Single seed per variant. The spec treats differences below 0.01 F1 as inconclusive, which covers every leave-one-out difference. The larger effects (dual-label supervision on ClipShots, the fusion blocks on ClipShots) exceed the band but are also single-seed.
- Thresholds are tuned per test set (Protocol A), so absolute F1 is optimistic. These numbers are leak-free (SHOT test videos excluded from training) and should not be compared with numbers from models trained on all 648 SHOT videos.
- Runs of this study used two code revisions (A0 and the other 12 runs). The training code is identical across them; only the launcher scripts differ.
- `B_noatt` was stopped manually after epoch 43 (best validation F1 at epoch 41) and evaluated from its best checkpoint, so it did not use the patience-10 stop of the other runs.
- Latency, VRAM and GFLOPs are measured on synthetic input; GFLOPs exclude non-matmul/conv operations.

