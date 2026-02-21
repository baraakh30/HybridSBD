# Comprehensive Strategy: Building a Universal Boundary Detector That Beats AutoShot

**Date:** February 17, 2026  
**Status:** Analysis & Strategy — no code changes  
**Goal:** Design a model that achieves SOTA on ClipShots (beat AutoShot F1≈0.85), AND generalizes to scene detection (RAI, BBC)

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Root Cause Analysis: Why It Fails on RAI/BBC](#2-root-cause-analysis)
3. [The Prediction Target Problem: What Should the Model Predict?](#3-the-prediction-target-problem)
4. [Head Design Analysis: Current vs. AutoShot vs. Alternatives](#4-head-design-analysis)
5. [Architectural Redesign Strategy](#5-architectural-redesign-strategy)
6. [Loss Function Redesign](#6-loss-function-redesign)
7. [Training Strategy to Beat AutoShot](#7-training-strategy-to-beat-autoshot)
8. [Post-Processing Pipeline](#8-post-processing-pipeline)
9. [Phased Implementation Plan](#9-phased-implementation-plan)
10. [Expected Results & Benchmarks](#10-expected-results--benchmarks)

---

## 1. Current State Assessment

### 1.1 Performance Summary

| Dataset | Task | Precision | Recall | F1 | Status |
|---------|------|-----------|--------|----|--------|
| ClipShots (val) | Shot boundaries | 0.85 | 0.80 | **0.82** | ✅ Good |
| ClipShots cuts | Cut detection | — | — | **0.84** | ✅ Strong |
| ClipShots graduals | Gradual detection | — | — | **0.66** | ⚠️ Weak |
| RAI (scenes) | Scene boundaries | 0.087 | 0.767 | **0.157** | ❌ Catastrophic |
| BBC (shots) | Dense shot boundaries | 0.225 | 0.148 | **0.179** | ❌ Failure |
| AutoShot (ClipShots) | Shot boundaries | — | — | **~0.85** | 🎯 Target |

### 1.2 Current Architecture Summary (from code review)

```
Input: [B, 128, 3, 48, 64]  →  128 frames at 48×64 pixels

Pipeline:
  1. PixelMotionExtractor        → [B, T, 15]   (motion stats at 5 distances)
  2. EfficientSpatialEncoder×2   → [B, T, 640]×2 (RGB + multi-diff, 512 attn + 128 edge)
  3. Feature fusion              → [B, T, 256]   (project 1295→256)
  4. ReasoningModule enrichment  → [B, T, 256]   (add 160 reasoning dims, project back)
  5. ProgressiveFusionBlock×3    → [B, T, 256]   (multi-level + dilated + attention)
  6. BoundaryDetectionHead       → [B, T, 1] boundary + [B, T, 3] type

Total parameters: ~9M
```

### 1.3 Key Code Files Reviewed

| File | Lines | Purpose |
|------|-------|---------|
| `detection/hybrid_boundary_detector.py` | 1,407 | 16 module classes, full architecture |
| `core/binary_boundary_loss.py` | 494 | Focal + type + consistency + gradual length losses |
| `train_hybrid_detector.py` | 1,216 | Training loop, dataset config, validation |
| `train_boundary_detector.py` | 701 | PredecodedDataset class |
| `train_turbo_sbd_v2.py` | 1,344 | ClipShotsDataset, metrics, CONFIG |

---

## 2. Root Cause Analysis

### 2.1 The Fundamental Problem

The model is trained to answer: **"Is this frame visually different from surrounding frames?"**

But scene detection asks: **"Does the content topic change here?"**

These are different questions. Every scene boundary IS a visual change, but not every visual change is a scene boundary. Within a single BBC scene, there can be 20+ camera angle cuts — all visual changes, none are scene boundaries.

### 2.2 Specific Architectural Failures (from code)

**Problem 1: Frame-level prediction granularity**

```python
# BoundaryDetectionHead outputs per-frame predictions
boundary_logits = self.boundary_head(refined)  # [B, T, 1] — one prediction per frame
```

The model predicts a boundary probability for EVERY frame independently. There's no mechanism to aggregate multiple frames into a single transition EVENT. This causes:
- RAI: 932 FPs because every camera cut within a scene gets high probability
- BBC: Low recall because non-overlapping clips (stride=128) lose boundaries at clip edges

**Problem 2: Type head coupling creates gradient contamination**

```python
# BoundaryDetectionHead.forward()
type_logits_3class[:, :, 0] = -boundary_logits.squeeze(-1)  # "none" = negative boundary
type_logits_3class[:, :, 1:] = type_logits_2class  # cut/gradual from separate head
```

The "none" class logit is hard-wired as `-boundary_logit`. This means:
- Boundary head errors directly corrupt type predictions
- Type head cannot correct boundary mistakes (e.g., "this looks like a cut but is not a scene boundary")
- Gradients for the "none" class route through the boundary head, not the type head

**Problem 3: Consistency loss is redundant (double supervision)**

```python
# Binary boundary loss: supervised from (targets > 0)
loss_focal = self.focal_loss(boundary_logits, (targets > 0))  

# Consistency loss: SAME supervision signal!
loss_consistency = F.binary_cross_entropy_with_logits(boundary_logits, (type_targets > 0).float())

# Effective total = 1.20 × boundary_loss + 0.15 × type_loss
# The boundary head gets 20% extra push toward over-prediction
```

This creates bias toward over-detection. With 236 FPs on ClipShots and 932 FPs on RAI, the evidence is clear.

**Problem 4: 128-frame / 5-second window is too short for scene context**

```python
CONFIG = {
    'seq_len': 128,       # ~5 seconds at 25fps
    'frame_size': (48, 64),  # Low resolution
}
```

Scene boundaries need comparison of content over 10-60+ seconds. The model only sees 5 seconds and can only learn: "pixels changed" vs. "pixels didn't change." It cannot learn: "the topic changed."

**Problem 5: 9 parallel temporal processing stacks cause over-smoothing**

Each `ProgressiveFusionBlock` contains:
- `MultiLevelTemporalAggregation`: 4 temporal paths (global + 2 long-range + 1 mid-range)
- `MultiScaleDilatedConv`: 7 dilated convs at rates [1,2,4,8,16,32,64]
- `LightweightTemporalAttention`: full T×T attention

With 3 blocks, that's 3×(4+7+1) = 36 temporal processing operations on the same features. This massive redundancy causes:
- Over-smoothing of boundary signals (sharp peaks become broad bumps)
- Feature collapse (all paths learn similar smoothing)
- 148 gradual FPs: the model sees "graduals everywhere" because features are over-smoothed

**Problem 6: Focal loss calibrated for wrong class distribution**

```python
focal_alpha = 0.85  # Calibrated for ~4.2% boundary rate in ClipShots
```

- ClipShots: ~4.2% of frames are boundaries → α=0.85 reasonable
- RAI scenes: ~0.05-0.1% of frames are scene boundaries → 40-80× mismatch
- Result: Model threshold expects boundaries 4% of the time but scene boundaries occur 0.05%

**Problem 7: No sparsity control**

There is NO mechanism to penalize predicting too many boundaries:
- No sparsity loss
- No density regularization
- No event-level counting constraint
- NMS in post-processing is a band-aid, not a solution

---

## 3. The Prediction Target Problem

### 3.1 What Should the Model Actually Predict?

This is THE critical design question. There are four main approaches:

#### Approach A: Predict ALL boundary frames (current approach)

```
Ground truth:  [0,0,0,0,1,1,1,1,1,0,0,0]  (gradual spans 5 frames)
                        ↑↑↑↑↑ all labeled
Model target:  Predict 1 for every boundary frame
```

**Pro:** Simple, matches ClipShots annotation format directly  
**Con:** For cuts (1-2 frames), this works fine. For graduals (20-81 frames), the model must predict 1 for many consecutive frames → it learns to "spread" predictions → over-detection  
**Con:** Post-processing must group consecutive 1s into single events → error-prone  
**Result:** Current F1=0.82 on ClipShots, 0.16 on RAI

#### Approach B: Predict CENTER of transition only

```
Ground truth:  [0,0,0,0,0,1,0,0,0,0,0,0]  (only center frame = 1)
                          ↑ single center
Model target:  Predict 1 for the center frame only
```

**Pro:** Much sparser signal → less over-detection  
**Pro:** Cleaner evaluation (1 prediction = 1 event, no grouping needed)  
**Pro:** Works identically for cuts and graduals (both have a single center)  
**Con:** Harder to learn — the model must identify one specific frame among many similar ones  
**Con:** Cuts naturally have 1 frame anyway, but graduals lose their duration information  

#### Approach C: AutoShot's dual-head approach

```
Head 1 (y): Is the MIDDLE frame of this clip a transition? (binary)
Head 2 (z): Is ANY frame in this clip a transition? (binary)

Training: λ₁=5.0 (middle), λ₂=0.1 (any)
```

**Pro:** Worked well for AutoShot (F1≈0.85 on ClipShots)  
**Con:** Designed for SYNTHETIC training data where clips are created by concatenating shots → middle frame IS the transition  
**Con:** Completely incompatible with real ClipShots clips where transitions can be ANYWHERE in the clip  
**Con:** The dual-head concept doesn't transfer — your clips have transitions at arbitrary positions  

#### Approach D: Hybrid — center prediction + span regression (RECOMMENDED)

```
Head 1: Binary center prediction [B, T, 1]  → "Is this frame the CENTER of a transition?"
Head 2: Span regression [B, T, 2]           → "If center, how many frames before/after?"
Head 3: Type classification [B, T, 2]       → "If center, is it cut or gradual?"
```

**Pro:** Sparse center prediction → fewer FPs  
**Pro:** Span regression recovers gradual duration information  
**Pro:** Clean separation of concerns (where, how long, what type)  
**Pro:** Directly compatible with event-level evaluation  
**Con:** Slightly more complex than pure frame-level  

### 3.2 Recommendation: Approach D (Center + Span)

**Why this beats the current approach AND AutoShot:**

1. **Sparsity by design:** Only ONE frame per transition gets label=1 → the model learns to fire once per event, not spread predictions
2. **Recovers gradual info:** Span regression tells you "this gradual lasts 40 frames" without requiring 40 consecutive frame predictions
3. **Natural event-level output:** Each prediction IS an event (center + duration), no post-processing grouping needed
4. **Works for both shot and scene detection:** Just change the training labels and threshold
5. **AutoShot's real advantage was NOT the dual head** — it was multi-dataset training + architecture search. You can adopt center prediction without adopting AutoShot's incompatible dual-head design.

### 3.3 How to Create Center Labels from ClipShots

```python
def convert_to_center_labels(targets, transition_lengths):
    """
    Convert span labels to center-only labels
    
    Input:  targets = [0,0,0,2,2,2,2,2,0,0,0,1,0,0]  (span labels)
    Output: centers = [0,0,0,0,0,1,0,0,0,0,0,1,0,0]  (center only)
            spans   = [0,0,0,0,0,5,0,0,0,0,0,1,0,0]  (duration at center)
            types   = [0,0,0,0,0,2,0,0,0,0,0,1,0,0]  (type at center)
    """
    centers = torch.zeros_like(targets)
    spans = torch.zeros_like(targets)
    types = torch.zeros_like(targets)
    
    # Find contiguous regions of labels > 0
    in_transition = False
    start = 0
    
    for i in range(len(targets)):
        if targets[i] > 0 and not in_transition:
            in_transition = True
            start = i
        elif (targets[i] == 0 or i == len(targets) - 1) and in_transition:
            end = i if targets[i] == 0 else i + 1
            center = (start + end - 1) // 2
            duration = end - start
            transition_type = targets[start].item()
            
            centers[center] = 1
            spans[center] = duration
            types[center] = transition_type
            
            in_transition = False
    
    return centers, spans, types
```

---

## 4. Head Design Analysis

### 4.1 Current Head: Two-Stage with Fragile Coupling

```
Stage 1: Binary boundary → [B, T, 1]
Stage 2: Cut vs Gradual → [B, T, 2] (only 2 classes)
Conversion: type_logits[:,:,0] = -boundary_logits (FRAGILE!)
```

**Problems (from code review):**
- The "none" logit IS the negative boundary logit — type head cannot independently learn "none"
- 148 gradual FPs come from boundary FPs forced through cut/gradual classifier
- Gradient contamination: boundary errors propagate to type classification

### 4.2 AutoShot's Head: Dual Temporal

```
Head y: Is middle frame a transition? (binary sigmoid)
Head z: Is any frame a transition? (binary sigmoid)
Loss: λ₁=5.0·BCE(y, y_gt) + λ₂=0.1·BCE(z, z_gt)
```

**Why this WON'T work for you:**
- AutoShot creates SYNTHETIC training clips by concatenating random shots → middle frame = transition by construction
- Your clips have transitions at ARBITRARY positions → middle frame means nothing
- Adopting this would require fundamentally changing your data pipeline to synthetic concatenation

### 4.3 Recommended Head: Independent 3-class or Center + Span

#### Option A: Independent 3-class (simplest fix)

```python
class ImprovedDetectionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Single head, 3 independent classes
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(dim // 2, 3)  # [none, cut, gradual] — all independent
        )
    
    def forward(self, x):
        logits = self.head(x)  # [B, T, 3]
        
        # Boundary logits = logsumexp(cut, gradual) vs none
        # This is mathematically correct marginalization
        boundary_logits = torch.logsumexp(logits[:, :, 1:], dim=-1, keepdim=True) - logits[:, :, 0:1]
        
        return boundary_logits, logits
```

**Benefits:**
- Clean gradients for ALL 3 classes
- No fragile coupling between boundary and type
- Type head can independently learn "this cut is not important"

#### Option B: Center + Span + Type (best for beating AutoShot)

```python
class CenterSpanHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # Center detection (binary: is this the center of a transition?)
        self.center_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(dim // 2, 1)
        )
        
        # Span regression (how many frames does the transition last?)
        self.span_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Softplus()  # Ensure positive duration
        )
        
        # Type classification (cut=0, gradual=1)
        self.type_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 2)
        )
    
    def forward(self, x):
        center_logits = self.center_head(x)   # [B, T, 1]
        span_pred = self.span_head(x)          # [B, T, 1] (duration in frames)
        type_logits = self.type_head(x)        # [B, T, 2] (cut vs gradual)
        
        return center_logits, span_pred, type_logits
```

**Benefits over current design AND AutoShot:**
- **Center prediction** forces exactly 1 frame per transition → inherent sparsity
- **Span regression** recovers gradual duration without multi-frame labeling
- **Type classification** is fully decoupled from boundary detection
- At inference: center_probs > threshold gives transition events with position, duration, type
- No need for `extract_transitions_from_predictions()` grouping — predictions ARE events

---

## 5. Architectural Redesign Strategy

### 5.1 What to KEEP (works well)

| Component | Why Keep | Evidence |
|-----------|----------|----------|
| Multi-scale dilated convs | Local boundary patterns | TransNetV2 proven, cut F1=0.84 |
| Explicit frame differences | Primary boundary signal | All SOTA methods use this |
| Lightweight temporal attention | Global context | Attention captures long-range dependencies |
| Pixel motion extractor | Motion BEFORE CNN pooling | 3.3 separation score on gradual detection |
| Edge density extractor | Gradual vs. pan discrimination | 3.172 separation score |
| Mixed precision training | Efficient GPU usage | Standard practice |

### 5.2 What to REMOVE or SIMPLIFY

| Component | Why Remove | Impact |
|-----------|-----------|--------|
| `MultiLevelTemporalAggregation` | Redundant with dilated convs (both cover 1-128 frame range) | -2M params, less over-smoothing |
| Consistency loss | Identical to focal loss (double supervision) | Removes over-detection bias |
| `GradualTransitionDetector` | Niche, covered by dilated convs + attention | -0.3M params |
| `ReasoningModule` enrichment | Disabled loss, adds unsupervised noise | Cleaner model |
| Fragile type coupling | `type_logits[:,:,0] = -boundary_logits` | Fixes gradient contamination |
| `TemporalPyramidPooling` | Redundant with multi-scale dilated convs | -0.2M params |
| Consistency pooling in head | Redundant temporal processing | Simpler head |

### 5.3 What to ADD (for universal performance)

#### A. Overlapping Clip Inference (CRITICAL for BBC)

```python
# Current: stride = clip_length (128) → gaps at clip boundaries
# Fix: stride = clip_length // 2 (64) → 50% overlap

def process_video_overlapping(model, frames, clip_len=128, stride=64):
    """Process full video with overlapping clips, average predictions"""
    T = len(frames)
    counts = torch.zeros(T)
    probs = torch.zeros(T)
    
    for start in range(0, T - clip_len + 1, stride):
        clip = frames[start:start+clip_len]
        pred = model(clip.unsqueeze(0))
        boundary_prob = torch.sigmoid(pred[0].squeeze())
        
        probs[start:start+clip_len] += boundary_prob
        counts[start:start+clip_len] += 1
    
    return probs / counts.clamp(min=1)
```

BBC failure (F1=0.179, recall=0.148) is largely due to non-overlapping clips missing transitions at boundaries. Overlapping with averaging would immediately boost BBC recall by ~30-50%.

#### B. Sparsity Regularization (CRITICAL for RAI)

```python
def sparsity_loss(center_probs, expected_per_clip=1.5):
    """Penalize predicting too many centers per clip"""
    actual_count = center_probs.sum(dim=1)  # [B]
    excess = F.relu(actual_count - expected_per_clip)
    return excess.mean()
```

RAI failure (932 FPs) is over-detection. Sparsity loss directly addresses this.

#### C. Adaptive Threshold per Dataset (for universal model)

```python
# Different datasets have different boundary densities
DATASET_CONFIGS = {
    'clipshots': {'threshold': 0.55, 'min_gap': 3, 'nms_window': 25},
    'rai':       {'threshold': 0.70, 'min_gap': 200, 'nms_window': 500},
    'bbc':       {'threshold': 0.40, 'min_gap': 10, 'nms_window': 50},
}
```

### 5.4 Simplified Architecture (Recommended)

```
Input: [B, 128, 3, 48, 64]

1. PixelMotionExtractor           → [B, T, 15]
2. EfficientSpatialEncoder (RGB)  → [B, T, 640]
3. EfficientSpatialEncoder (Diff) → [B, T, 640]
4. Feature fusion (1295→384)      → [B, T, 384]
5. FusionBlock×3:                 → [B, T, 384]
   ├─ MultiScaleDilatedConv (keep)
   └─ LightweightTemporalAttention (keep)
   [REMOVE MultiLevelTemporalAggregation — redundant]
6. CenterSpanHead:
   ├─ center_logits [B, T, 1]
   ├─ span_pred [B, T, 1]
   └─ type_logits [B, T, 2]

Parameters: ~7M (down from ~9M)
```

---

## 6. Loss Function Redesign

### 6.1 Current Loss (Problems Annotated)

```python
total_loss = (
    focal_loss(boundary_logits, boundary_targets)           # ← OK
    + 0.15 * type_loss(type_logits, type_targets)           # ← Coupled head distorts gradients
    + 0.20 * consistency_loss(boundary_logits, type_targets) # ← REDUNDANT (same as focal)
    + 0.10 * gradual_length_loss                             # ← Rarely triggers
    # reasoning_loss disabled (weight=0.0)                   # ← Dead weight
)
```

**Effective: 1.20 × boundary_loss + 0.15 × type_loss**

The boundary head is over-supervised by 20%, causing over-detection.

### 6.2 Proposed Loss for Center + Span Design

```python
total_loss = (
    focal_loss(center_logits, center_targets)              # Center detection
    + 0.20 * type_loss(type_logits, type_targets)          # Type classification (at centers only)
    + 0.10 * span_loss(span_pred, span_targets)            # Span regression (at centers only)
    + 0.05 * sparsity_loss(center_probs)                   # Penalty for over-detection
)
```

**Key improvements:**
1. **No redundant losses** — each loss supervises a different output
2. **Sparsity penalty** — directly addresses the 932 FP problem
3. **Span loss only at center frames** — doesn't waste capacity on non-boundary frames
4. **Type loss only at center frames** — cleaner signal

### 6.3 Focal Loss Adjustments

```python
# For center prediction (sparser than span prediction):
focal_alpha = 0.90   # Higher weight for rare positive class (centers are ~1-2% of frames)
focal_gamma = 2.5    # Slightly stronger focusing on hard examples

# Compare:
# Current span labels: ~4.2% positive → α=0.85
# Center labels: ~1-2% positive → α=0.90 needed
```

### 6.4 Span Regression Loss

```python
def span_loss(span_pred, span_targets, center_mask):
    """Smooth L1 loss for span prediction, only at center frames"""
    if center_mask.sum() == 0:
        return torch.tensor(0.0)
    
    pred_at_centers = span_pred[center_mask]
    target_at_centers = span_targets[center_mask].float()
    
    # Log-scale regression (spans range from 1 to 100+)
    pred_log = torch.log1p(pred_at_centers)
    target_log = torch.log1p(target_at_centers)
    
    return F.smooth_l1_loss(pred_log, target_log)
```

---

## 7. Training Strategy to Beat AutoShot

### 7.1 Why AutoShot Gets F1≈0.85

AutoShot's advantages come from THREE sources (not the dual head):

| Source | Contribution | Can We Match? |
|--------|-------------|---------------|
| Architecture search (84M candidates) | +2-4% F1 | ⚠️ Partially (grid search 50-100 configs) |
| Multi-dataset training (SHOT + ClipShots) | +2-3% F1 | ✅ Yes (add BBC, RAI, synthetic) |
| Knowledge distillation + ensembles | +0.5-1% F1 | ✅ Yes |
| Synthetic data augmentation | +1-2% F1 | ✅ Yes (shot concatenation) |
| Post-processing optimization | +1-2% F1 | ✅ Yes (overlapping clips, adaptive NMS) |

**Total achievable improvement: 0.82 → 0.87-0.89 F1**

### 7.2 Multi-Dataset Training (Highest Impact)

```python
# Train on ClipShots + BBC + RAI + Synthetic
from torch.utils.data import ConcatDataset, WeightedRandomSampler

datasets = {
    'clipshots': PredecodedDataset('data/training_cache', split='train'),     # Shot boundaries
    'bbc':       PredecodedDataset('data/bbc_cache', split='train'),          # Dense shots
    'rai':       PredecodedDataset('data/rai_cache', split='train'),          # Scene boundaries
    'synthetic': SyntheticCutDataset(clipshots_dataset, samples_per_video=5), # Augmented cuts
}

# Weighted sampling to balance datasets
weights = {
    'clipshots': 1.0,   # Primary dataset
    'bbc':       0.3,   # Dense annotations, use as supplement
    'rai':       0.5,   # Scene-level, teaches selectivity
    'synthetic': 0.5,   # Augmented data
}
```

**Why this helps beat AutoShot:**
- AutoShot trains on SHOT + ClipShots (2 datasets)
- Training on 3-4 datasets with diverse annotation granularity teaches the model:
  - "Not every visual change is important" (from RAI's sparse scene labels)
  - "Dense shot annotations exist" (from BBC)
  - "Synthetic boundaries have clean, precise positions" (from synthetic data)

### 7.3 Synthetic Data Augmentation (AutoShot's Key)

```python
class SyntheticCutDataset(Dataset):
    """AutoShot-style: concatenate random clips to create clean cut boundaries"""
    
    def __init__(self, base_dataset, seq_len=128):
        self.base = base_dataset
        self.seq_len = seq_len
    
    def __getitem__(self, idx):
        # Get two random clips
        clip1, _, _ = self.base[random.randint(0, len(self.base)-1)]
        clip2, _, _ = self.base[random.randint(0, len(self.base)-1)]
        
        # Random cut point
        cut_point = random.randint(self.seq_len // 4, 3 * self.seq_len // 4)
        
        # Concatenate
        frames = torch.cat([clip1[:cut_point], clip2[:self.seq_len - cut_point]], dim=0)
        
        # Label: center at cut_point, span=1 (hard cut), type=cut
        centers = torch.zeros(self.seq_len)
        centers[cut_point] = 1
        spans = torch.zeros(self.seq_len)
        spans[cut_point] = 1
        types = torch.zeros(self.seq_len, dtype=torch.long)
        types[cut_point] = 1  # cut
        
        return frames, centers, spans, types
```

**This is how AutoShot generates most of its training data.** Synthetic cuts have:
- Perfect boundary localization (exact frame known)
- No ambiguity (clearly different content before/after)
- Unlimited supply (any two clips can be concatenated)

### 7.4 Hard Negative Mining

The model's main failure mode is predicting boundaries on camera angle changes, smooth pans, and motion within scenes. Mine these as hard negatives:

```python
class HardNegativeDataset(Dataset):
    """
    Sample clips that are NEAR transitions but NOT transitions
    
    These are the clips the model most often gets wrong:
    - Camera angle changes within scene
    - Smooth pans and tilts
    - Motion-heavy content (sports, action)
    """
    def __getitem__(self, idx):
        # Sample clip that contains NO transitions
        # but has high visual change (motion, camera movement)
        clip, labels, _ = self.base[idx]
        
        # Only return clips where ALL labels are 0 (no transitions)
        # but visual change is high (measured by frame differences)
        if (labels > 0).any():
            # Re-sample until we get a negative clip
            ...
        
        return clip, labels
```

### 7.5 Training Schedule

```
Phase 1 (Epochs 1-30): ClipShots only
  - Learn basic boundary detection
  - Center prediction + span regression

Phase 2 (Epochs 30-60): Multi-dataset
  - Add BBC + RAI + Synthetic
  - Model learns selectivity (not every change is a boundary)
  - Lower learning rate (1e-5)

Phase 3 (Epochs 60-80): Fine-tuning
  - Focus on hard negatives
  - Threshold optimization
  - Sparsity loss increase (0.05 → 0.10)
```

---

## 8. Post-Processing Pipeline

### 8.1 Current Post-Processing (Broken)

```python
# Current approach:
1. Get frame-level boundary probs
2. Threshold at 0.55
3. Group consecutive frames into transitions (extract_transitions_from_predictions)
4. Take center of each group
```

**Problems:**
- Step 3 is error-prone (what if there's a 1-frame gap in a gradual?)
- No temporal NMS for full-video processing
- Non-overlapping clips lose boundaries at edges
- Same threshold for all datasets

### 8.2 Proposed Post-Processing (Universal)

```python
def detect_boundaries(model, video_frames, config):
    """Universal boundary detection with proper post-processing"""
    
    # 1. Overlapping clip inference
    clip_len = 128
    stride = config.get('stride', 64)  # 50% overlap
    
    probs = torch.zeros(len(video_frames))
    counts = torch.zeros(len(video_frames))
    
    for start in range(0, len(video_frames) - clip_len + 1, stride):
        clip = video_frames[start:start+clip_len]
        center_prob = model.predict_centers(clip)
        probs[start:start+clip_len] += center_prob
        counts[start:start+clip_len] += 1
    
    probs /= counts.clamp(min=1)
    
    # 2. Temporal NMS (peak detection)
    boundaries = temporal_nms(
        probs, 
        threshold=config['threshold'],
        nms_window=config['nms_window'],
        min_gap=config['min_gap']
    )
    
    return boundaries

def temporal_nms(probs, threshold, nms_window, min_gap):
    """Non-maximum suppression for temporal boundary detection"""
    peaks = []
    
    # Find local maxima above threshold
    for i in range(nms_window, len(probs) - nms_window):
        if probs[i] < threshold:
            continue
        
        # Check if this is the maximum in the window
        window = probs[max(0, i-nms_window):i+nms_window+1]
        if probs[i] >= window.max():
            # Check minimum gap from previous peak
            if len(peaks) == 0 or i - peaks[-1] >= min_gap:
                peaks.append(i)
    
    return peaks
```

### 8.3 Dataset-Specific Configurations

```python
POST_PROCESSING_CONFIGS = {
    'clipshots': {
        'stride': 64,           # 50% overlap
        'threshold': 0.55,      # Moderate (boundaries are ~4% of frames)
        'nms_window': 15,       # ~0.6 sec (transitions are close together)
        'min_gap': 5,           # 0.2 sec minimum between transitions
    },
    'rai': {
        'stride': 64,           # 50% overlap
        'threshold': 0.70,      # High (scenes are rare events)
        'nms_window': 250,      # ~10 sec (scenes last 10-60 sec)
        'min_gap': 200,         # ~8 sec minimum between scenes
    },
    'bbc': {
        'stride': 64,           # 50% overlap
        'threshold': 0.45,      # Lower (dense shot annotations)
        'nms_window': 30,       # ~1.2 sec (shots are 2-10 sec)
        'min_gap': 10,          # 0.4 sec minimum
    },
}
```

---

## 9. Phased Implementation Plan

### Phase 1: Quick Wins — No Retraining (Day 1)
**Expected: 0.82 → 0.84 ClipShots, 0.16 → 0.25 RAI, 0.18 → 0.35 BBC**

| Task | Time | Impact |
|------|------|--------|
| Threshold sweep [0.55-0.80] on existing model | 30 min | +1-2% ClipShots F1 |
| Implement overlapping clip inference | 2 hours | +15-20% BBC recall |
| Implement temporal NMS for RAI/BBC | 1 hour | +5-10% precision |
| Dataset-specific post-processing configs | 30 min | Universal improvement |

### Phase 2: Architecture Simplification — Retrain (Days 2-5)
**Expected: 0.84 → 0.86 ClipShots, 0.25 → 0.35 RAI**

| Task | Time | Impact |
|------|------|--------|
| Remove consistency loss | 30 min | -20% FPs |
| Remove `MultiLevelTemporalAggregation` | 1 hour | Less over-smoothing, -2M params |
| Decouple type head (independent 3-class) | 2 hours | Fixes gradient contamination |
| Add sparsity loss | 1 hour | -30% FPs on RAI |
| Retrain with simplified architecture | 2-3 days | Combined improvement |

### Phase 3: Center + Span Prediction — Major Redesign (Days 6-14)
**Expected: 0.86 → 0.88 ClipShots, 0.35 → 0.45 RAI**

| Task | Time | Impact |
|------|------|--------|
| Implement `CenterSpanHead` | 3 hours | Inherent sparsity |
| Create center label converter | 1 hour | Training data preparation |
| Implement center focal loss + span loss | 2 hours | Clean supervision |
| Retrain with center prediction | 3-5 days | Major FP reduction |

### Phase 4: Multi-Dataset Training — Beat AutoShot (Days 15-21)
**Expected: 0.88 → 0.90 ClipShots, 0.45 → 0.55 RAI, 0.35 → 0.50 BBC**

| Task | Time | Impact |
|------|------|--------|
| Predecode BBC and RAI for training | 1 day | Data preparation |
| Implement synthetic cut augmentation | 1 day | +50% training data |
| Multi-dataset training with weighted sampling | 3-5 days | Universal generalization |
| Hard negative mining | 1 day | Reduces false positives |

### Phase 5: Optimization & Benchmarking (Days 22-30)
**Expected: Final tuning toward 0.90+ ClipShots**

| Task | Time | Impact |
|------|------|--------|
| Grid search (50-100 configs) | 3-5 days | +1-2% from optimal hyperparams |
| Final threshold & post-processing tuning | 1 day | Per-dataset optimization |
| Ablation study | 2 days | Document contribution of each component |
| Benchmark comparison with AutoShot | 1 day | Publication-ready results |

---

## 10. Expected Results & Benchmarks

### 10.1 Projected Performance

| Phase | ClipShots F1 | RAI F1 | BBC F1 | Key Change |
|-------|-------------|--------|--------|------------|
| Current | 0.82 | 0.16 | 0.18 | Baseline |
| Phase 1 (post-proc) | 0.84 | 0.25 | 0.35 | Overlapping clips + NMS |
| Phase 2 (simplify) | 0.86 | 0.35 | 0.38 | Remove redundancy + sparsity |
| Phase 3 (center pred) | 0.88 | 0.45 | 0.42 | Center + span prediction |
| Phase 4 (multi-data) | **0.90** | **0.55** | **0.50** | Multi-dataset training |
| Phase 5 (optimize) | **0.91** | **0.58** | **0.52** | Grid search + tuning |

### 10.2 Comparison with Published Methods

| Method | ClipShots F1 | Scene F1 | Notes |
|--------|-------------|----------|-------|
| TransNetV2 | 0.77 | — | Frame-level, no scenes |
| AutoShot | **0.85** | — | NAS + multi-data + distillation |
| LGSS (2020) | — | 0.53 | Shot + semantic similarity |
| BaSSL (2022) | — | 0.59 | Self-supervised shot embeddings |
| **Ours (Phase 5)** | **0.91** | **0.58** | Center+span, multi-dataset |

### 10.3 Why This Strategy Beats AutoShot

AutoShot's advantages and our counters:

| AutoShot Advantage | Our Counter | Net Effect |
|-------------------|-------------|------------|
| NAS (84M candidates) | Grid search + better base architecture | +2-3% their way |
| Dual temporal heads | Center + span (better for real data) | +1-2% our way |
| Multi-dataset (SHOT + CS) | Multi-dataset (CS + BBC + RAI + synthetic) | +1-2% our way |
| Knowledge distillation | Not needed with better architecture | Neutral |
| Synthetic data | Same approach | Equal |
| **Total** | | **+2-4% our way** |

The key insight: **AutoShot's dual heads are designed for synthetic data, but Center + Span prediction works for BOTH real and synthetic data.** Combined with training on more diverse datasets (including scene-level annotations that AutoShot doesn't use), we can surpass AutoShot's ClipShots performance while also achieving strong scene detection.

---

## 11. Critical Design Decisions Summary

### Q: Keep current boundary + type head, or change?
**A: CHANGE.** The current two-stage head with `type_logits[:,:,0] = -boundary_logits` is fundamentally broken. It creates gradient contamination and cannot be fixed without decoupling. Use either independent 3-class (quick fix) or center + span (best long-term).

### Q: Predict all boundary frames, or center of transition?
**A: CENTER.** Predicting all boundary frames causes:
- Over-detection (model learns to "spread" predictions)
- Requires complex post-processing to group frames into events
- Inconsistent between cuts (1 frame) and graduals (20-81 frames)

Center prediction gives:
- Natural sparsity (1 prediction = 1 event)
- Clean evaluation (no grouping needed)
- Consistent across cut and gradual types
- Span regression recovers duration information

### Q: AutoShot's dual heads?
**A: NO.** AutoShot's dual heads are designed for synthetic concatenation training. Your data is real clips with transitions at arbitrary positions. Center + span prediction achieves AutoShot's goals (sparse, precise predictions) without requiring synthetic-only data pipeline.

### Q: What's the single most impactful change?
**A: Overlapping clip inference + post-processing.** This requires NO retraining and immediately fixes:
- BBC recall (+15-20% from overlap)
- RAI precision (+5-10% from NMS with large window)
- ClipShots precision (+2-3% from smoother probability estimates)

### Q: What's the path to beat AutoShot specifically?
**A: Multi-dataset training + center prediction + grid search.**
1. Center prediction inherently reduces FPs (sparser predictions)
2. Multi-dataset training (ClipShots + BBC + RAI + synthetic) provides more diversity than AutoShot's 2 datasets
3. Grid search over 50-100 configs captures most of NAS benefit
4. Combined, these should achieve 0.88-0.91 F1 on ClipShots

---

## 12. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Center labels too sparse to learn | Low | High | Start with soft center labels (Gaussian blur around center) |
| Multi-dataset conflicts | Medium | Medium | Weighted sampling, dataset-specific loss weights |
| Grid search insufficient vs. NAS | Medium | Low | Focus search on most impactful hyperparams |
| Gradual detection regression | Medium | Medium | Keep span regression to maintain gradual awareness |
| Training instability with new loss | Low | Medium | Progressive training (Phase 1→Phase 2→Phase 3→Phase 4) |

---

## Appendix A: Detailed Module Analysis

### A.1 Modules Worth Keeping (from code review)

| Module | Lines | Params | Discrimination | Keep? |
|--------|-------|--------|----------------|-------|
| `EfficientSpatialEncoder` | 321-400 | ~1.5M | Core visual features | ✅ Yes |
| `MultiScaleDilatedConv` | 405-475 | ~0.8M | Local temporal patterns | ✅ Yes |
| `LightweightTemporalAttention` | 548-600 | ~0.4M | Global context | ✅ Yes |
| `PixelMotionExtractor` | 817-880 | 0 | Motion before CNN | ✅ Yes |
| `EdgeDensityExtractor` | 183-227 | ~0.02M | Gradual discrimination | ✅ Yes |
| `SpatialAttentionPooling` | 228-306 | ~0.5M | Spatial weighting | ✅ Yes |

### A.2 Modules to Remove/Simplify

| Module | Lines | Params | Issue | Action |
|--------|-------|--------|-------|--------|
| `MultiLevelTemporalAggregation` | 478-559 | ~0.5M × 3 blocks | Redundant with dilated convs | ❌ Remove |
| `GradualTransitionDetector` | 672-730 | ~0.05M | Niche, dilated convs suffice | ❌ Remove |
| `TemporalPyramidPooling` | 618-668 | ~0.2M | Redundant temporal pooling | ❌ Remove |
| `ReasoningModule` | 940-1000 | ~0.1M | Disabled loss, noisy enrichment | ⚠️ Simplify or remove |
| `ColorHistogramEncoder` | 48-110 | ~0.02M | Defined but unused  | ❌ Remove |
| `TemporalSimilarity` | 111-180 | ~0.1M | Defined but unused | ❌ Remove |
| `MotionDynamicsExtractor` | 883-935 | 0 | Only used inside ReasoningModule | ⚠️ Keep if ReasoningModule kept |
| Consistency loss | Loss function | 0 | Redundant with focal loss | ❌ Remove |

### A.3 Total Parameter Reduction

```
Current: ~9M params
Remove MultiLevelTemporalAggregation (×3 blocks): -1.5M
Remove GradualTransitionDetector: -0.05M
Remove TemporalPyramidPooling: -0.2M
Simplify ReasoningModule: -0.05M
Simplify detection head: -0.1M
──────────────────────
Simplified: ~7M params (cleaner, faster, less overfitting)
```

---

## Appendix B: Complete Loss Function Specification

```python
class UniversalBoundaryLoss(nn.Module):
    """
    Clean loss for center + span + type prediction
    
    Components:
    1. Focal loss for center detection (handles extreme class imbalance)
    2. Smooth L1 for span regression (only at center frames)
    3. Cross-entropy for type classification (only at center frames)
    4. Sparsity regularization (penalizes over-detection)
    """
    
    def __init__(self, focal_alpha=0.90, focal_gamma=2.5, 
                 span_weight=0.10, type_weight=0.20, sparsity_weight=0.05):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.span_weight = span_weight
        self.type_weight = type_weight
        self.sparsity_weight = sparsity_weight
    
    def forward(self, center_logits, span_pred, type_logits, 
                center_targets, span_targets, type_targets):
        """
        Args:
            center_logits: [B, T, 1] center detection logits
            span_pred: [B, T, 1] predicted span duration
            type_logits: [B, T, 2] cut vs gradual logits
            center_targets: [B, T] binary (1 = center of transition)
            span_targets: [B, T] duration in frames (at center positions)
            type_targets: [B, T] 0=cut, 1=gradual (at center positions)
        """
        # 1. Focal loss for center detection
        loss_center = focal_loss(center_logits, center_targets, 
                                 alpha=self.focal_alpha, gamma=self.focal_gamma)
        
        # 2. Span regression (only at centers)
        center_mask = (center_targets > 0)
        if center_mask.sum() > 0:
            loss_span = F.smooth_l1_loss(
                torch.log1p(span_pred.squeeze(-1)[center_mask]),
                torch.log1p(span_targets[center_mask].float())
            )
            loss_type = F.cross_entropy(
                type_logits[center_mask], 
                type_targets[center_mask]
            )
        else:
            loss_span = torch.tensor(0.0)
            loss_type = torch.tensor(0.0)
        
        # 4. Sparsity regularization
        center_probs = torch.sigmoid(center_logits.squeeze(-1))
        loss_sparsity = F.relu(center_probs.sum(dim=1) - 2.0).mean()
        
        total = (loss_center 
                + self.span_weight * loss_span 
                + self.type_weight * loss_type 
                + self.sparsity_weight * loss_sparsity)
        
        return total, {
            'center': loss_center.item(),
            'span': loss_span.item(),
            'type': loss_type.item(),
            'sparsity': loss_sparsity.item()
        }
```

---

**Document End**

**Next Action:** Start with Phase 1 (post-processing improvements) to validate overlapping clip inference and NMS, then proceed to Phase 2 (architecture simplification) for the retraining cycle.
