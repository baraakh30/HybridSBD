# Results — HybridSBD ablation study (leak-free)

All numbers in this document come from the saved experiment results (training histories, per-test-set threshold-search results, validation-selected thresholds and speed measurements). The evaluation uses the minimum scene length ℓ_min = 1 (no merging) on SHOT, ClipShots and BBC, and is reported under two threshold rules: **Protocol A** selects τ_c and τ_g on each test set (optimistic by construction, comparable only to other Protocol-A numbers) and **Protocol B** selects them once on the 67 SHOT validation videos and applies them unchanged to all three test sets. Every variant has one training run (seed 42); A0 and A1 also have seeds 43 and 44. Per the spec, F1 differences below 0.01 are treated as inconclusive.

| Section | Content |
|---|---|
| 1 | Data and training approach |
| 2 | Compliance with the specification |
| 3 | Extra seeds for A0 and A1 (training) |
| 4 | Protocol A, ℓ_min = 1 (Tables 7a, 7b, build-up ladder, precision/recall, Tables 8 and 9, seeds table, ℓ_min sensitivity) |
| 5 | Protocol B, validation-selected thresholds (same tables) |
| 6 | Validation F1 for every epoch |
| 7 | Interpretation |
| 8 | Speed, memory and size |
| 9 | Caveats |

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

### 1.4 Evaluation (Protocols A and B)

- Each model's best checkpoint (by SHOT validation F1) is evaluated over a threshold grid: boundary threshold τ_c ∈ {0.10, 0.15, …, 0.90} and gradual threshold τ_g ∈ {0.3, 0.4, 0.5, 0.6}; the minimum scene length is fixed to ℓ_min = 1 (no merging) on all three test sets. **Protocol A** reports the best setting per test set. **Protocol B** selects the setting once, on the 67 SHOT validation videos evaluated as full videos, and applies it unchanged to SHOT, ClipShots and BBC.
- Decision rule (dual-label models): boundary if p_y > τ_c; gradual if p_z > τ_g; cut = boundary and not gradual; the predicted many-hot track is gradual OR cut. Midpoint-only (A1, A3) and unified (A2) variants use τ_c only.
- Scoring: AutoShot's scene-based matching with **±2 frame tolerance**; precision, recall and F1 over transitions, pooled over all videos.

### 1.5 Protocol details (spec Section 0, items 3–8)

- **Minimum shot length ℓ_min** is applied in the evaluation code, shared by all three test sets: after the boundary track is turned into scenes, every predicted scene shorter than ℓ_min frames is merged into the previous scene. All results use ℓ_min = 1 (no merging); the sensitivity of A0 on ClipShots to ℓ_min ∈ {1, 2, 3, 4} is in section 4.9.
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

**Check 2 — performance (Protocol A, ℓ_min = 1, paper's acceptable ranges):**

| Test set | Paper | Acceptable range | Our A0 | Result |
|---|---|---|---|---|
| SHOT F1 | 0.924 | 0.915 – 0.93 | 0.8449 | **below range** |
| ClipShots F1 | 0.793 | 0.785 – 0.8 | 0.8001 | above range |
| BBC F1 | 0.971 | 0.965 – 0.975 | 0.9719 | within range |

Check 2 does **not** pass on SHOT, and this is expected: **it is caused by a deliberate change in the training data, not by the code.** The SHOT label file `kuaishou_v2.txt` lists 648 videos, and the 200 SHOT *test* videos (the authors' released test annotation) are among them. The paper-era setup therefore trained on 648 videos that include the entire SHOT test set — a data leak that inflates SHOT F1. We removed the 200 test videos from training (448 videos remain), so the SHOT numbers here are leak-free and are lower than the paper's 0.924. As a one-off control, the same A0 recipe trained on all 648 videos reaches SHOT 0.9155, which is inside the required 0.915–0.930 range, so the gap is explained by the leak and not by an architecture or pipeline difference (Check 1 passes). Per the spec, this is reported to the team: the ablations proceed on the same code and data policy for all runs, and the main-paper SHOT number needs discussion. ClipShots (0.8001) is above its range and BBC (0.9719) is inside it; neither benchmark shares videos with the SHOT test set.

### 2.2 Deviations from the specification

| Spec says | What was done | Why |
|---|---|---|
| SHOT: 648 training videos (§3) | **448 videos**: the 200 SHOT test videos are excluded | The 648 include the SHOT test set (data leak), see 2.1 |
| Early stopping patience 15 (§3) | patience 10 for all 13 runs and the build-up runs | Shortens training time: the validation optimum appeared early in the first runs, so the same patience was used for every variant |
| Augmentation: flip, brightness, contrast, noise, clip brightness (§3) | all of these, **plus** strong per-clip augmentation, synthetic transitions (p = 0.25) and weight averaging (EMA 0.999) | Added to recover accuracy once the leaked videos were removed; identical for every run, so the comparison between variants stays fair |
| All runs on the same code version as A0 (§1) | A0 and the other 12 runs were trained from two code revisions | A0 was trained first as the recipe experiment; only the launcher scripts differ, the training code is identical |
| 13 runs, nothing else (§5, prompt) | **extra** build-up ladder (4 runs, section 5), speed table with GFLOPs, B = 1 / 32 timings and an AutoShot comparison, one extra C4 cut-recall analysis | Requested additions to identify which components do the work; kept in separate sections |
| Training speed | channels-last memory format | Memory-layout change only; values are mathematically identical |

## 3. Extra seeds for A0 and A1 (training)

A0 (full model, dual-label) and A1 (full model, midpoint only) were trained with seeds 43 and 44, giving four new runs (A0-s43, A0-s44, A1-s43, A1-s44). Recipe, augmentation, synthetic transitions, EMA (0.999), early stopping (patience 10 on SHOT validation F1 with the variant's own decision rule) and the 80-epoch maximum are identical to the seed-42 runs. Only the training seed changes: model initialisation, training-data order, the ClipShots clip drawn each epoch, augmentation and synthetic transitions.

### 3.1 Train/validation split is identical for all seeds

In the training code the SHOT and ClipShots train/validation split was shuffled with the training seed, so seeds 43 and 44 would have used different validation videos. I fixed the split to the seed-42 split for every training seed (the option defaults to 42, so the seed-42 runs are unchanged). SHA-256 of the sorted SHOT validation video names:

| Training seed | Old rule (split follows the seed) | Fixed split (used) |
|---|---|---|
| 42 | f0070e4a… | f0070e4a… |
| 43 | 88ddfd45… | f0070e4a… |
| 44 | 792acaf5… | f0070e4a… |

All four new runs printed the same checksums before training: SHOT validation 67 videos (f0070e4a…), SHOT train 381 videos (ed9cb6ec…), ClipShots train 3,108 videos (f623f785…). The pre-extracted training clips use the stable per-video MD5 seed and are unchanged.

### 3.2 The four new runs

| Run | Seed | Best epoch | Best SHOT val F1 | Epochs run | Stopped early | Training hours |
|---|---|---|---|---|---|---|
| A0 | 42 | 21 | 0.8029 | 31 | True | 5.40 |
| A0-s43 | 43 | 13 | 0.7953 | 23 | True | 6.97 |
| A0-s44 | 44 | 16 | 0.7935 | 26 | True | 7.26 |
| A1 | 42 | 7 | 0.7863 | 17 | True | 2.92 |
| A1-s43 | 43 | 10 | 0.7915 | 20 | True | 6.46 |
| A1-s44 | 44 | 9 | 0.7967 | 19 | True | 6.18 |

Seed-42 rows are the main-study runs, shown for reference.

### 3.3 SHOT validation F1 history of every epoch

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

### 3.4 Does training several runs at once on one GPU change the results?

A seed-42 A1 run trained alongside the four new runs (five jobs on one GPU) was compared with the original seed-42 A1 run over its first two epochs. Train loss 0.03501 vs 0.03502 (epoch 1) and 0.01847 vs 0.01849 (epoch 2); SHOT validation F1 0.7046 vs 0.7029 (epoch 1) and 0.7505 vs 0.7499 (epoch 2). The two runs are **not bit-identical**, so a repeated training run, with or without concurrent jobs, gives slightly different numbers; I could not separate concurrency from other run-to-run non-determinism. The differences are small (at most 0.002 validation F1 and 0.0001 train loss) and far below the spread between seeds at epoch 2 (validation F1 0.7446–0.7659 across the four new runs). Data-loader seeding is per run and the split is fixed, so concurrency does not change the data each run sees.


## 4. Protocol A with ℓ_min = 1 (evaluation only)

ℓ_min is fixed to 1 (no minimum-shot-length merging) on SHOT, ClipShots and BBC and is not searched. Thresholds are still selected on each test set (Protocol A).

### 4.1 Checks

The evaluation pipeline was checked against the earlier evaluation of the same checkpoints and reproduces it exactly for all 17 seed-42 models (51 of 51 model-test-set results: F1, thresholds, TP / FP / FN).

**Saved probabilities vs a fresh forward pass** (A0 and A1, all three test sets): the maximum absolute difference of p_y and p_z over all videos is 0.0, and F1, thresholds and TP / FP / FN are identical. The results in sections 4 and 5 therefore use the saved probabilities.

### Results

#### 4.2 Supervision (Table 7a, seed 42)

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

#### 4.3 Components, leave-one-out (Table 7b)

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

#### 4.4 Build-up ladder

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

#### 4.5 Precision, recall and selected thresholds

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

#### 4.6 Table 8 — Per-type analysis on ClipShots

| ID | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |
|---|---|---|---|---|
| A0 | 0.8484 | 0.7866 | 0.7415 | 0.6232 |
| A1 | 0.9093 | 0.6623 | n/a | n/a |
| A2 | 0.8011 | 0.8070 | n/a | 0.6965 |

#### 4.7 Table 9 — Recall by gradual-transition duration on ClipShots

| ID | ≤ 10 frames | 11–30 frames | > 30 frames |
|---|---|---|---|
| A0 | 0.7740 (n = 832) | 0.8018 (n = 1241) | 0.7500 (n = 228) |
| C6 | 0.7500 (n = 832) | 0.8251 (n = 1241) | 0.7368 (n = 228) |

C6 − A0: ≤ 10 -0.0240, 11–30 +0.0234, > 30 -0.0132.

#### 4.8 Seeds table

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

#### 4.9 ℓ_min sensitivity for A0 on ClipShots

Best F1 for each fixed ℓ_min, thresholds re-searched for each value:

| ℓ_min | τ_c | τ_g | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | 0.65 | 0.6 | 0.7734 | 0.8287 | 0.8001 |
| 2 | 0.65 | 0.6 | 0.7800 | 0.8274 | 0.8030 |
| 3 | 0.65 | 0.6 | 0.7848 | 0.8256 | 0.8047 |
| 4 | 0.65 | 0.6 | 0.7871 | 0.8241 | **0.8052** |

#### 4.10 Interpretation check (0.01 band)

| Statement | Value | Holds? |
|---|---|---|
| A0 − A1 on ClipShots ≥ 0.01, seed 42 | +0.0392 | **yes** |
| A0 − A1 on ClipShots ≥ 0.01, mean over the three seeds | +0.0421 | **yes** |
| A0 − A1 on ClipShots positive for every seed | +0.0392 / +0.0366 / +0.0507 | **yes** |
| A4 − A3 on ClipShots ≥ 0.01 | +0.0439 | **yes** |
| A0 − A2 on ClipShots ≥ 0.01 | +0.0292 | **yes** |
| A0 − A2 on BBC ≥ 0.01 | +0.0245 | **yes** |
| No leave-one-out change (C1–C6) reaches 0.01 on any test set | largest 0.0089 | **yes** |


## 5. Protocol B, validation-selected thresholds (evaluation only)

For each model τ_c and τ_g are selected once on the SHOT validation videos and applied unchanged to SHOT, ClipShots and BBC (ℓ_min = 1). Nothing is tuned on a test set. Midpoint-only (A1, A3) and unified (A2) models select τ_c only.

### 5.1 Validation videos as full videos (prerequisite)

Before this work the 67 SHOT validation videos existed only as pre-cut training clips, so I **added a data path**: the 67 videos are now decoded completely with the same decoder, resizing (48 × 64), window layout (T = 128, stride 64) and window averaging as the SHOT test set (1,270 windows), and scored with the same scene matching. Ground truth comes from the SHOT label file with the same transition-to-scene conversion as ClipShots; 720 of 775 transitions lie within the decoded video length and are used. The label file is the only ground truth available for these videos and is not the authors' final annotation, so validation F1 is noisier than test F1 and the selected thresholds are noisier too.

### 5.2 Threshold selection and application

τ_c ∈ {0.10, …, 0.90} and τ_g ∈ {0.3, 0.4, 0.5, 0.6} were searched once per model by maximising F1 on the full validation videos with ℓ_min = 1. The selected values and the validation F1 are in the thresholds table (5.6). They were applied unchanged to the three test sets for every model, including the new seeds. Numbers are expected to be lower than under Protocol A, especially on ClipShots and BBC.

### Results

#### 5.3 Supervision (Table 7a, seed 42)

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

#### 5.4 Components, leave-one-out (Table 7b)

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

#### 5.5 Build-up ladder

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

#### 5.6 Precision, recall and selected thresholds

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

#### 5.7 Table 8 — Per-type analysis on ClipShots

| ID | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |
|---|---|---|---|---|
| A0 | 0.8712 | 0.8640 | 0.6488 | 0.7338 |
| A1 | 0.8810 | 0.3968 | n/a | n/a |
| A2 | 0.8718 | 0.8331 | n/a | 0.7218 |

#### 5.8 Table 9 — Recall by gradual-transition duration on ClipShots

| ID | ≤ 10 frames | 11–30 frames | > 30 frames |
|---|---|---|---|
| A0 | 0.8341 (n = 832) | 0.8759 (n = 1241) | 0.9079 (n = 228) |
| C6 | 0.8257 (n = 832) | 0.8767 (n = 1241) | 0.8421 (n = 228) |

C6 − A0: ≤ 10 -0.0084, 11–30 +0.0008, > 30 -0.0658.

#### 5.9 Seeds table

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

#### 5.10 Interpretation check (0.01 band)

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


## 6. Additional check — SHOT validation F1 for every epoch

The validation F1 of every epoch for A0 and A1 at seed 42 and for the four new runs is in Table 3.3. What happens after each run's best epoch:

### 6.1 Behaviour after the best epoch

| Run | Best epoch | Best val F1 | Stopped at epoch | Range after best | Last | Trend per epoch |
|---|---|---|---|---|---|---|
| A1 | 7 | 0.7863 | 17 | 0.7695–0.7852 | 0.7846 | +0.00024 |
| A0 | 21 | 0.8029 | 31 | 0.7810–0.7953 | 0.7870 | -0.00066 |
| A1-s43 | 10 | 0.7915 | 20 | 0.7733–0.7901 | 0.7830 | -0.00053 |
| A1-s44 | 9 | 0.7967 | 19 | 0.7812–0.7949 | 0.7898 | -0.00021 |
| A0-s43 | 13 | 0.7953 | 23 | 0.7781–0.7933 | 0.7933 | +0.00037 |
| A0-s44 | 16 | 0.7935 | 26 | 0.7820–0.7912 | 0.7850 | -0.00036 |

### 6.2 Did validation F1 level off or fall after the best epoch?

In every one of the six runs, including A1 at seed 42, early stopping ended training after 10 epochs without improvement, and validation F1 had **levelled off** rather than still rising or collapsing: after its best epoch each run moves up and down inside a band about 0.01–0.02 wide just below its best value, with a linear trend close to zero (between −0.0007 and +0.0004 F1 per epoch), and no run was still climbing when it stopped. A1 at seed 42 (best epoch 7, 0.7863) fluctuated between 0.7695 and 0.7852 afterwards and ended at 0.7846, so it was levelled off and unstable at the 0.01 level, not falling steadily. The new A1 runs behave the same way (A1-s43 drifts slightly down, −0.0005 per epoch; A1-s44 is flat, −0.0002). The epoch-to-epoch swings are about as large as the gaps between runs, so the best-epoch choice itself is noisy.

## 7. Interpretation (0.01 band)

Differences below 0.01 F1 are inconclusive. Each cell shows Protocol A / Protocol B. Seed 42 unless stated; "3-seed mean" is the mean over seeds 42, 43 and 44 of the paired A0 − A1 difference.

### 7.1 Supervision

| Dataset | A0 − A1 (seed 42) | A0 − A1 (3-seed mean) | A4 − A3 (reduced) | Outcome |
|---|---|---|---|---|
| SHOT | -0.0029 / -0.0230 | +0.0008 / -0.0014 | +0.0058 / +0.0141 | A: neither ≥ 0.01: no accuracy claim; B: only one of the two ≥ 0.01: mixed |
| ClipShots | +0.0392 / +0.0305 | +0.0421 / +0.0259 | +0.0439 / +0.0065 | A: dual-label helps for both the full and the reduced model; B: only one of the two ≥ 0.01: mixed |
| BBC | +0.0021 / -0.0158 | +0.0007 / -0.0133 | +0.0007 / -0.0054 | A: neither ≥ 0.01: no accuracy claim; B: neither ≥ 0.01: no accuracy claim |

On ClipShots the paired A0 − A1 difference is positive at every seed under both protocols (Protocol A: +0.0392, +0.0366, +0.0507; Protocol B: +0.0305, +0.0228, +0.0242), so dual-label supervision helps on the benchmark with many gradual transitions. On SHOT and BBC the paired A0 − A1 differences (seeds 42, 43, 44) are: SHOT Protocol A: -0.0029, +0.0003, +0.0049 (sign changes across seeds); SHOT Protocol B: -0.0230, -0.0067, +0.0255 (sign changes across seeds); BBC Protocol A: +0.0021, -0.0014, +0.0015 (sign changes across seeds); BBC Protocol B: -0.0158, -0.0162, -0.0080 (same sign at all seeds).
The error profile changes as well: on ClipShots (3-seed mean, Protocol A) cut recall / gradual recall is 0.842 / 0.774 for A0 (dual) against 0.902 / 0.635 for A1 (midpoint only); under Protocol B 0.881 / 0.851 against 0.877 / 0.409.

### 7.2 Hypotheses

| Comparison | Hypothesis | Protocol A / Protocol B |
|---|---|---|
| A0 vs A2 | A0 higher | SHOT -0.0060 / -0.0267; ClipShots +0.0292 / +0.0069; BBC +0.0245 / -0.0034 |
| C2a / C2b | Largest drops | Largest drop per dataset: SHOT C2b -0.0089 / no drop; ClipShots C1 -0.0059 / C2a -0.0393; BBC C6 -0.0052 / no drop. Not confirmed (the largest drop is not always C2a or C2b, or it is below 0.01) |
| C6 | Lower recall on > 30-frame transitions | C6 − A0 recall: ≤ 10 -0.0240 / -0.0084; 11–30 +0.0234 / +0.0008; > 30 -0.0132 / -0.0658 (n = 228 for > 30) |
| Any Cx | Drop | Largest single-component change on any dataset: 0.0089 / 0.0393 |

### 7.3 Build-up ladder: what does the work?

Change in F1 at each step (Protocol A / Protocol B); a step clears the band if it is ≥ 0.01 on that dataset under that protocol.

| Step | SHOT | ClipShots | BBC |
|---|---|---|---|
| M0_min → M1_edge (+ edge-density head) | +0.0091 / -0.0042 | +0.0117 / +0.0236 ✔ | +0.0041 / +0.0011 |
| M1_edge → B_noatt (+ 4 fusion blocks, no attention) | +0.0091 / +0.0177 (B only) | +0.0312 / +0.0159 ✔ | +0.0098 / +0.0012 |
| B_noatt → A4 (+ attention and gate) | +0.0018 / +0.0105 (B only) | +0.0061 / -0.0164 | +0.0039 / +0.0135 (B only) |
| A4 → S1_rel (+ FrameSimilarity, ColorHist, delta) | +0.0023 / -0.0089 | +0.0008 / +0.0234 (B only) | -0.0015 / -0.0125 |
| S1_rel → A0 (+ diff branch, motion, per-frame pathway) | +0.0057 / -0.0191 | -0.0053 / -0.0204 | +0.0013 / -0.0095 |

The bare core (M0_min) is within 0.028 / 0.045 / 0.018 F1 of the full model on SHOT / ClipShots / BBC under Protocol A with 21 % of its parameters. ✔ = clears the band under both protocols.

## 8. Speed, memory and size

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

## 9. Caveats

- One training run per variant, except A0 and A1 (three seeds). The spec treats differences below 0.01 F1 as inconclusive. The dual-label effect on ClipShots holds at every seed under both protocols; the effect of the fusion blocks and all other ladder steps are single-seed.
- Protocol A tunes thresholds on each test set, so its absolute F1 is optimistic and comparable only to other Protocol-A numbers. Protocol B selects thresholds on the SHOT validation videos, which are labelled with the SHOT label file rather than the authors' final annotation, so its selected thresholds are noisy and its numbers are lower; under Protocol B, A0 − A1 on ClipShots is the only comparison of section 7 that is positive and ≥ 0.01, A0 is lower than A1 on BBC at all three seeds, and leave-one-out differences are less reliable.
- These numbers are leak-free (SHOT test videos excluded from training) and should not be compared with numbers from models trained on all 648 SHOT videos.
- Runs of this study used two code revisions (A0 and the other 12 runs). The training code is identical across them; only the launcher scripts differ.
- `B_noatt` was stopped manually after epoch 43 (best validation F1 at epoch 41) and evaluated from its best checkpoint, so it did not use the patience-10 stop of the other runs.
- Latency, VRAM and GFLOPs are measured on synthetic input; GFLOPs exclude non-matmul/conv operations.
- The per-type analysis (Tables 8 and 9) was computed for A0, A1, A2, C6 and the extra seeds of A0 and A1; the cut-recall analysis of C4 was not repeated under ℓ_min = 1 and Protocol B, so the hypothesis about C4's cut recall is not evaluated here.

