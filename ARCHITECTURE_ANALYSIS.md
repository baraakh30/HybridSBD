# Architecture Analysis: Why the Hybrid Boundary Detector Fails on Scene Detection

**Date:** February 16, 2026  
**Status:** Analysis only — no code changes  

---

## Executive Summary

The Hybrid Boundary Detector achieves strong results on **ClipShots** (Cut F1=0.84, Gradual F1=0.66, Overall F1=0.82) but collapses on **scene boundary detection** tasks:

| Dataset | Precision | Recall | F1 | Problem |
|---------|-----------|--------|----|---------|
| ClipShots (val) | ~0.85 | ~0.80 | 0.82 | ✓ Works |
| RAI (scenes) | 0.087 | 0.767 | 0.157 | Massive FP explosion |
| BBC (shots) | 0.225 | 0.148 | 0.179 | Low recall + high FP |

**Root cause:** The entire architecture — model, loss, training data, evaluation — is designed to detect **which individual frames are boundary frames** (frame-level boundary detection), not **where scene transitions happen** (event-level scene detection). These are fundamentally different tasks.

---

## 1. THE CORE PROBLEM: Frame-Level vs. Event-Level Detection

### 1.1 What the Model Learns (Frame-Level Boundary Detection)

The model is trained to answer: **"Is this frame part of a transition?"**

- Input: 128-frame clips at 48×64 resolution
- Output: Per-frame boundary probability `[B, T, 1]` for each of the 128 frames
- Training signal: Per-frame labels (0=none, 1=cut, 2=gradual) from ClipShots
- Evaluation during training: Per-frame TP/FP/FN with ±3 frame tolerance

This means the model learns to fire a **high probability on EVERY frame** that looks like it could be near a visual change. In ClipShots, this works because:
- Transitions ARE local visual events (cuts = instant change, graduals = dissolves/wipes)
- The training data has dense labels marking which frames are transitional
- Validation uses `extract_transitions_from_predictions()` which groups consecutive boundary frames into single transition events

### 1.2 What Scene Detection Requires (Event-Level Detection)

Scene detection asks: **"Where do the content/topic changes happen?"**

- A "scene" in broadcast video lasts 2-60+ seconds (50-1500+ frames at 25fps)
- Scene boundaries often coincide with shot boundaries, but NOT all shot boundaries are scene boundaries
- Within a scene, there can be dozens of shot cuts (camera angle changes, B-roll inserts) that are NOT scene transitions
- The model needs to distinguish **"this cut changes the topic"** from **"this cut is just a camera angle change within the same topic"**

### 1.3 The Mismatch

| Aspect | What Model Does | What Scene Detection Needs |
|--------|----------------|---------------------------|
| Granularity | Frame-level ("is this frame a boundary?") | Event-level ("is there a scene change here?") |
| What triggers detection | ANY visual change (cut, dissolve, motion) | SEMANTIC content changes (topic shifts) |
| Positive rate | ~4.2% of frames are boundaries | ~0.05-0.1% of frames are scene boundaries |
| Context window | 128 frames (~5 sec at 25fps) | Need 10-60+ seconds of context |
| Feature level | Low-level visual (RGB diffs, edges, motion) | High-level semantic (what is the content about?) |

---

## 2. ARCHITECTURAL ISSUES IN DETAIL

### 2.1 Spatial Encoder: Optimized for Visual Change, Not Semantic Content

**File:** `detection/hybrid_boundary_detector.py`, class `EfficientSpatialEncoder`

```
Architecture: 16→64→128→256 channels, input 48×64 resolution
Output: 640 dims = 512 (attention pooling) + 128 (edge density features)
```

**Problems for scene detection:**

1. **Input resolution too low (48×64):** This is fine for detecting visual discontinuities (cuts produce massive pixel differences at any resolution), but it destroys semantic content information. At 48×64, you can't tell if the scene shows a forest vs. an ocean — you can only tell if the pixels changed.

2. **Edge density extractor is a CUT detector, not a scene detector:** The `EdgeDensityExtractor` (lines 184-216) computes edge variance across spatial regions — this separates "gradual transitions" from "smooth pans" but says nothing about whether a cut represents a scene change.

3. **No semantic features:** The encoder is a shallow 4-layer CNN (16→64→128→256 channels). This is enough for texture/edge features but lacks the depth and capacity for semantic understanding. Scene detection needs features like "this is a cooking show segment" vs. "this is a news anchor segment."

### 2.2 Frame Difference Encoding: Detects ALL Visual Changes

**File:** `detection/hybrid_boundary_detector.py`, `HybridBoundaryDetector.forward()` (lines 1230-1260)

```python
diff_distances = [1, 4, 8, 16]
all_diffs[:, dist:, i*C:(i+1)*C] = x[:, dist:] - x[:, :-dist]
```

**Problem:** This is the PRIMARY detection signal. Multi-distance RGB differences fire on EVERY visual discontinuity:
- Camera cuts within a scene (angle changes) → HIGH diff
- Scene transitions → HIGH diff
- Motion, lighting changes → MODERATE diff

There is NO mechanism to distinguish "meaningful" changes (scene transitions) from "irrelevant" changes (within-scene camera work). The model has no way to learn this distinction because it wasn't trained with that label structure.

### 2.3 Temporal Context: Too Short for Scene Detection

**File:** `detection/hybrid_boundary_detector.py`

The model processes **128-frame clips** (about 5 seconds at 25fps). This is the core design constraint:

1. **MultiScaleDilatedConv** (lines 404-475): Dilated convolutions with rates [1,2,4,8,16,32,64] — max effective receptive field is ~128 frames. This covers the full clip but ONLY the clip.

2. **MultiLevelTemporalAggregation** (lines 478-545): Long-range dilations [64, 96] with global pooling — but still bounded by the 128-frame input.

3. **LightweightTemporalAttention** (lines 548-600): Full self-attention over T=128 frames — sees the whole clip, but that's only ~5 seconds.

**Why this fails for scenes:** To know if a cut is a scene boundary, you need to compare content BEFORE the cut (seconds to minutes) with content AFTER the cut (seconds to minutes). A 5-second window cannot capture "this is a new topic/setting/activity."

Example: In BBC Planet Earth, a scene might show underwater footage for 30 seconds, then cut to aerial footage. The model sees a 5-second window around the cut and only knows "the pixels changed abruptly" — same signal as an underwater camera angle change.

### 2.4 Pixel Motion Features: Irrelevant for Scene Detection

**File:** `detection/hybrid_boundary_detector.py`, class `PixelMotionExtractor` (lines 817-880)

```python
distances = [1, 2, 4, 8, 16]
# Computes: magnitude mean/std, direction coherence at each distance
```

**Problem:** These are low-level motion features designed to separate gradual transitions from camera pans. For scene detection, motion features are almost entirely uninformative — a scene boundary can happen during static shots (anchor → anchor), dynamic shots (action → action), or any combination.

### 2.5 Reasoning Module: Designed for Transition TYPES, Not Scene SEMANTICS

**File:** `detection/hybrid_boundary_detector.py`, class `ReasoningModule` (lines 940-1000)

The reasoning module computes:
- Pixel motion magnitude/std/coherence (from `PixelMotionExtractor`)
- Motion acceleration and jerk (from `MotionDynamicsExtractor`)
- Spatial activity (from RGB encoder features)

Total output: 160 dims = 64 (pixel motion) + 32 (dynamics) + 64 (spatial activity)

**Problem:** All of these features describe HOW pixels change, not WHAT the content is. They cannot distinguish:
- Cut from Scene A to Scene B (scene change)
- Cut from Camera 1 to Camera 2 within Scene A (not a scene change)

Both produce identical pixel-level signatures.

### 2.6 Detection Head: Two-Stage but Still Frame-Level

**File:** `detection/hybrid_boundary_detector.py`, class `BoundaryDetectionHead` (lines 1010-1105)

```
Stage 1: Binary boundary detection → [B, T, 1]
Stage 2: Cut vs. Gradual classification → [B, T, 2] (converted to 3-class)
```

**Problem:** The two-stage design is clever for reducing boundary/type confusion but fundamentally still makes a per-frame prediction. There is no event-level aggregation — the model doesn't predict "there is a scene transition between frame X and Y" but rather "frame X has high boundary probability."

### 2.7 GradualTransitionDetector: Niche Component, Irrelevant for Scenes

**File:** `detection/hybrid_boundary_detector.py`, class `GradualTransitionDetector` (lines 672-730)

Uses long-window trend detection (windows of 41 and 81 frames) with smoothness gating. This is specialized for detecting dissolves/wipes in ClipShots — completely irrelevant for scene detection where boundaries are predominantly hard cuts.

---

## 3. LOSS FUNCTION ISSUES

### 3.1 Focal Loss: Calibrated for Wrong Imbalance Ratio

**File:** `core/binary_boundary_loss.py`, class `BinaryBoundaryLoss`

```python
focal_alpha=0.85  # Calibrated for ~4.2% boundary rate in ClipShots
focal_gamma=2.0   # Standard focusing parameter
```

**Problem for scene detection:**
- ClipShots: ~4.2% of frames are boundaries → alpha=0.85 is reasonable
- RAI/BBC scenes: ~0.05-0.1% of frames are scene boundaries → the model is calibrated for 40-80x more positives than actually exist in scene detection

The result is exactly what we see: **massive false positive explosion** (RAI: 932 FPs for 89 TPs). The model's threshold was trained to expect boundaries ~4% of the time, but in scene detection they occur ~0.05% of the time.

### 3.2 Type Classification Loss: Irrelevant for Scene Detection

```python
type_weight=0.15
type_class_weights=[0.05, 10.0, 5.0]  # [none, cut, gradual]
```

The auxiliary task of classifying cut vs. gradual doesn't help scene detection at all. Scene boundaries are almost all hard cuts — the distinction between cut and gradual is orthogonal to the scene detection problem.

### 3.3 Consistency Loss: Reinforces Frame-Level Thinking

```python
consistency_weight=0.30
# Teaches: type=none → boundary=0, type=cut/gradual → boundary=1
```

This loss explicitly couples boundary detection to transition type classification. It reinforces the assumption that "if there's a visual transition of any type, it's a boundary" — exactly wrong for scene detection where most visual transitions are NOT scene boundaries.

### 3.4 No Scene-Level Loss

The loss function has NO component that:
- Penalizes predicting too many transitions in a temporal window
- Rewards sparse predictions matching scene-level granularity
- Considers content similarity before/after a predicted boundary
- Operates at the event level rather than frame level

---

## 4. TRAINING DATA ISSUES

### 4.1 ClipShots Dataset: Boundary-Level Annotations

**File:** `predecode_dataset.py`, `train_hybrid_detector.py`

ClipShots provides **per-frame transition labels**:
- Frame-level: 0=none, 1=cut, 2=gradual
- Every visual transition (cut, dissolve, wipe, fade) is labeled
- This includes both scene changes AND within-scene shot transitions

The model is trained on ALL visual transitions equally. It has no way to learn that some transitions are more "important" (scene-level) than others (shot-level).

### 4.2 Sampling Strategy: Transition-Focused

```python
# Training: 60% of clips are centered on transitions
if has_transition and random.random() < 0.6:
    t_start, t_end = random.choice(transitions)
    center_frame = (t_start + t_end) // 2
    start = center_frame - (seq_len // 2)
```

This biases training heavily toward frames near transitions. The model learns to fire whenever it sees visual change patterns, regardless of whether they represent scene boundaries.

### 4.3 128-Frame Clips: Insufficient Context

```python
seq_len = 128  # ~5 seconds at 25fps
frame_size = (48, 64)  # Low resolution
```

Scene detection needs context about what happens BEFORE and AFTER a potential boundary over seconds to minutes. The 128-frame window only provides ~5 seconds, making it impossible to learn "the content topic changed."

---

## 5. EVALUATION/POST-PROCESSING ISSUES

### 5.1 Training Validation: Transition-Level but Short-Window

**File:** `train_hybrid_detector.py`, `validate()` function

During training, validation uses `extract_transitions_from_predictions()` with `min_gap=3` and `tolerance=3`. This is designed for ClipShots clips where transitions are close together and tolerance is tight.

### 5.2 RAI Test: NMS Cannot Fix the Fundamental Problem

**File:** `test_rai_dataset.py`

```python
nms_window=75   # 3 seconds
min_gap=50      # 2 seconds  
tolerance=3     # 3 frames
```

The RAI test applies aggressive temporal NMS (Non-Maximum Suppression) to reduce false positives. But when the model fires high probabilities on EVERY cut/transition it sees (not just scene boundaries), NMS can reduce the count but cannot select WHICH peaks correspond to scene changes.

**Evidence from results:**
- Video 1: 6 GT boundaries → 107 predictions → After NMS, still too many
- Video 8: 11 GT boundaries → 189 predictions → 17x over-prediction

The model has high boundary probability on every visual discontinuity. NMS reduces to the top peaks but can't distinguish scene boundaries from within-scene cuts.

### 5.3 BBC Test: Tolerance Issue

**File:** `test_bbc.py`

BBC uses `tolerance=15` frames and `min_distance=10`. The BBC dataset has 466 shot annotations in a single video — these are dense shot-level boundaries, not sparse scene boundaries.

The model gets P=0.225, R=0.148, F1=0.179 — it detects some shots but misses most (low recall). This is because:
1. Without overlapping clips (stride = clip length), predictions are noisy at clip boundaries
2. The model's probability peaks don't always align exactly with BBC's shot boundaries
3. BBC's annotation density (466 transitions in 72K frames ≈ 1 every 154 frames) is different from ClipShots training distribution

---

## 6. WHY IT WORKS ON CLIPSHOTS BUT NOT RAI/BBC

### ClipShots Success Explained

ClipShots is a **shot boundary detection** dataset:
- Every visual transition has a label
- Transitions are the unit of evaluation
- The model just needs to detect visual changes → the architecture is PERFECT for this
- 128-frame clips contain enough context (most ClipShots videos are short clips)
- 48×64 resolution is enough for detecting visual discontinuities

### RAI Failure Explained

RAI provides **scene-level annotations** (content segments):
- A "scene" may contain many shot cuts (none of which are scene boundaries)
- The model fires on ALL shot cuts → massive false positives
- The model has no semantic understanding to distinguish scene-level content changes
- 5-second context window can't capture "the topic changed"

The precision is 0.087 because for every 1 correct scene boundary, the model predicts ~11 false positives (within-scene cuts that look identical to the model).

### BBC Failure Explained

BBC has **dense shot annotations** (every camera cut in a 2-hour documentary):
- 466 transitions in a single video (one every ~154 frames)
- The model has lower precision because of clip boundary artifacts
- Low recall because non-overlapping clips miss some transitions
- The model was never calibrated for this annotation density

---

## 7. WHAT WOULD BE NEEDED FOR A MODEL GOOD AT EVERYTHING

### 7.1 For Scene Detection (RAI-like)

#### A. Semantic Features (Critical)
- Use a **pretrained visual backbone** (e.g., CLIP, DINOv2, ResNet-50 pretrained on ImageNet)
- Extract features that encode WHAT is in the frame, not just HOW pixels changed
- Scene detection = detecting when the visual SEMANTICS change, not when pixels change

#### B. Long-Range Context (Critical)
- Increase temporal context to 500-2000 frames (20-80 seconds)
- Use hierarchical temporal modeling: frame → shot → scene
- Scene boundaries need comparison of content over 10+ seconds before/after

#### C. Scene-Level Training Signal
- Train on scene-annotated data (RAI, MovieNet, BBC with scene labels)
- Use event-level loss: predict scene boundary EVENTS, not frame-level probabilities
- Penalize density of predictions (scenes are rare events)

#### D. Two-Stage Architecture
- **Stage 1:** Shot boundary detection (current model, works well)
- **Stage 2:** Scene boundary classification — for each detected shot boundary, classify if it's a scene change using semantic features from long temporal windows

### 7.2 For Universal Boundary Detection (All Datasets)

#### A. Multi-Scale Temporal Hierarchy
```
Frame level → Shot level → Scene level
(128 frames)  (1-5K frames)  (5-50K frames)
```

#### B. Semantic Feature Integration
- Current: Raw RGB + frame diffs + edge density + pixel motion = ALL low-level
- Needed: Add pretrained semantic encoder (even frozen CLIP/DINOv2 features would help enormously)
- Semantic similarity between adjacent segments is the KEY signal for scene detection

#### C. Configurable Post-Processing Per Task
- **Shot detection (ClipShots):** Low NMS, tight tolerance (current approach)
- **Scene detection (RAI):** Very high NMS, large min_gap, possibly learned threshold
- The model could output the same raw predictions, but the post-processing determines the detection granularity

#### D. Multi-Task Training
- Train jointly on shot-level AND scene-level datasets
- Use task-specific heads sharing a common backbone
- The shared backbone learns general visual change representations, task heads learn task-specific decisions

#### E. Contrastive Scene Embedding
- Learn embeddings where frames from the same scene are close, different scenes are far
- Use these embeddings to identify scene boundaries: sharp jumps in embedding space = scene change
- This naturally handles the "is this a meaningful change?" question

### 7.3 Minimal Changes for Quick Improvement on RAI/BBC

If re-architecture is not an option, these post-processing changes could help:

1. **Adaptive thresholding:** Set threshold based on the distribution of probabilities in the video, not a fixed value
2. **Much more aggressive NMS:** Use `nms_window=200-500` frames (8-20 seconds) and `min_gap=200+` to match expected scene lengths
3. **Confidence calibration:** Train a small calibration network on RAI-like data that converts frame-level boundary probs into scene-level transition likelihood
4. **Feature clustering:** After getting boundary probs, cluster consecutive high-prob regions and only keep the single highest peak per cluster — where clusters are defined by content similarity

---

## 8. SUMMARY OF ALL IDENTIFIED ISSUES

| # | Issue | Component | Impact |
|---|-------|-----------|--------|
| 1 | Frame-level detection, not event-level | Entire architecture | **CRITICAL** — fundamental mismatch |
| 2 | No semantic features (only pixel-level) | Spatial encoder | **CRITICAL** — can't distinguish scene vs. shot |
| 3 | 128-frame context window (5 sec) | Model input/training | **CRITICAL** — too short for scene context |
| 4 | Trained on ALL visual transitions equally | Training data (ClipShots) | **HIGH** — no scene-level signal |
| 5 | Low resolution input (48×64) | Spatial encoder | **HIGH** — destroys semantic content |
| 6 | Focal loss calibrated for 4.2% boundary rate | Loss function | **HIGH** — 40-80x mismatch for scene detection |
| 7 | Frame difference is primary signal | Feature extraction | **MEDIUM** — fires on ALL visual changes |
| 8 | Pixel motion features irrelevant for scenes | ReasoningModule | **MEDIUM** — adds nothing for scene task |
| 9 | No density regularization in loss | Loss function | **MEDIUM** — no penalty for over-prediction |
| 10 | Type classification (cut/gradual) irrelevant | Detection head | **LOW** — orthogonal to scene detection |
| 11 | GradualTransitionDetector unnecessary | Detection head | **LOW** — niche component for ClipShots |
| 12 | NMS cannot fix semantic gap | Post-processing | **LOW** — treating symptoms, not cause |

---

## 9. COMPARISON WITH LITERATURE

| Method | Approach | Shot F1 | Scene F1 | Key Difference |
|--------|----------|---------|----------|----------------|
| TransNetV2 | Frame-level + dilated convs | 0.77 | - | Frame-level only |
| AutoShot | Boundary detection network | 0.82 | - | Frame-level only |
| **Ours** | Hybrid frame-level | **0.82** | **0.16** | Frame-level ✓, Scene ✗ |
| LGSS (2020) | Shot + semantic sim | - | 0.53 | Two-stage with semantics |
| BaSSL (2022) | Contrastive shot-level SSL | - | 0.59 | Self-supervised shot embeddings |
| MovieNet/ShotCoL | Scene-level grouping | - | 0.52 | Groups shots into scenes |

The literature confirms: **scene detection requires semantic features and long-range temporal context.** Pure frame-level visual change detection (our approach) cannot solve scene detection.

---

## 10. CONCLUSION

The architecture is **correctly and effectively designed for shot boundary detection** (ClipShots F1=0.82 proves this). The failure on scene detection is not a bug — it's a fundamental architectural limitation. The model detects EVERY visual change (which is exactly what shot boundary detection requires), but scene detection requires understanding WHICH visual changes are semantically meaningful (topic/content shifts).

**The model is not broken — it's solving a different problem than scene detection.**

To make it "good for everything," the architecture must evolve from pure frame-level visual change detection to include:
1. Semantic understanding (pretrained visual features)
2. Long-range temporal context (hierarchical temporal modeling)  
3. Scene-level training signal (scene-annotated datasets)
4. Task-adaptive prediction heads or post-processing
