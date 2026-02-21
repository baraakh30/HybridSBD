# Hybrid Boundary Detector - Comprehensive Analysis
**Date:** February 17, 2026  
**Model:** hybrid_detector4 (main_dim=256, 3 fusion blocks, ~9M params)  
**Status:** F1=0.731 (0.846 cuts, 0.573 graduals) - **Gradual detection is the bottleneck**

---

## Executive Summary

**The Problem:** Gradual transition detection has low precision (0.488) with 148 false positives, despite decent recall (0.695). The model over-detects boundaries, especially confusing smooth camera pans and similar content as gradual transitions.

**Root Cause:** Architecture over-engineering (9 parallel temporal conv stacks) + redundant loss supervision (consistency loss duplicates focal loss) + fragile two-stage type prediction that hard-couples to boundary logits.

**Solution:** Simplify architecture, remove redundant loss, decouple type prediction. **Estimated improvement: 0.73 → 0.80-0.83 F1**

---

## Current Performance

### Test Set Results (tolerance=3, transition-level)

| Metric | Cuts | Graduals | Overall |
|--------|------|----------|---------|
| **Precision** | 0.764 | **0.488** ⚠️ | 0.644 |
| **Recall** | 0.947 | 0.695 | 0.845 |
| **F1** | 0.846 | **0.573** ⚠️ | **0.731** |
| **TP / FP / FN** | 285 / 88 / 16 | 141 / **148** / 62 | 426 / 236 / 78 |

**Key Issue:** 148 gradual FPs vs 141 TPs → More false alarms than correct detections!

### Frame-Level Metrics
- Boundary F1: 0.632 (Precision: 0.550, Recall: 0.742)
- Gradual mIoU: 0.417 (very low)

---

## Architecture Deep Dive

### Model Pipeline
```
Input [B,T,3,H,W] (128 frames, 48×64)
  ↓
PixelMotionExtractor (15 dims: magnitude/std/coherence at 5 distances)
  ↓
Dual Spatial Encoders (RGB + 4-distance diffs)
  ├─ RGB: 640 dims (512 attention + 128 edge)
  └─ Diff: 640 dims (512 attention + 128 edge)
  ↓
Feature Fusion (1295 → main_dim=256)
  ↓
ReasoningModule Enrichment (+160 dims: pixel motion + dynamics + spatial)
  ↓
3× ProgressiveFusionBlock (each contains):
  ├─ MultiLevelTemporalAggregation [global + dilation 64,96,16]
  ├─ MultiScaleDilatedConv [dilations 1,2,4,8,16,32,64]
  └─ LightweightTemporalAttention [4 heads, 256d]
  ↓
BoundaryDetectionHead:
  ├─ TemporalPyramidPooling [scales 1,4,8]
  ├─ GradualTransitionDetector [windows 41,81]
  ├─ Consistency features [windows 11,25]
  ├─ Stage 1: Binary boundary → logits [B,T,1]
  └─ Stage 2: Cut vs Gradual → coupled to stage 1
       type_logits[:,:,0] = -boundary_logits  ⚠️ FRAGILE COUPLING
  ↓
Output: boundary_logits [B,T,1], type_logits [B,T,3]
```

### Architecture Problems

#### **Problem 1: Massive Temporal Redundancy**
Each `ProgressiveFusionBlock` runs 3 parallel temporal processors:
- `MultiLevelTemporalAggregation`: dilations [64, 96, 16]
- `MultiScaleDilatedConv`: dilations [1, 2, 4, 8, 16, 32, 64]
- `LightweightTemporalAttention`: full sequence context

With 3 fusion blocks, that's **9 temporal conv stacks** + 3 attention layers processing the same features. The dilated conv alone already covers 1→128 frame receptive fields. This redundancy causes:
- **Over-smoothing** of boundary signals (why model sees "graduals everywhere")
- **Wasted parameters** (~3-4M on redundant paths)
- **Feature collapse** (all 3 paths learn similar smoothing patterns)

#### **Problem 2: Fragile Two-Stage Type Prediction**
```python
# In BoundaryDetectionHead forward():
type_logits_3class[:, :, 0] = -boundary_logits.squeeze(-1)  # "none" = negative boundary
type_logits_3class[:, :, 1:] = type_logits_2class  # cut/gradual from separate head
```

This hard-wires the "none" class logit as `-boundary_logit`. Consequences:
- If boundary head over-predicts (which it does: 236 FPs), every FP feeds wrong signal to type classifier
- Type head cannot learn to correct boundary mistakes
- Gradient flow is blocked for "none" class (comes from detached boundary head)
- 148 gradual FPs likely come from boundary FPs being forced through cut/gradual classifier

#### **Problem 3: Reasoning Module Adds Noise, Not Signal**
- **160 dims of enrichment** from pixel motion stats + acceleration/jerk
- Reasoning loss is **disabled** (weight=0.0) because margin constraints were too strict
- Acceleration/jerk are just diff(diff(motion_magnitude)) → extremely noisy
- The 160-dim "enrichment" is unsupervised noise the model must learn to ignore
- **Solution:** Remove or re-enable with softer MSE-based supervision

---

## Loss Function Analysis

### Current Loss Components
```python
total_loss = (
    focal_loss                                    # Binary boundary detection
    + type_weight * type_loss                     # 3-class type (none/cut/gradual)
    + consistency_weight * consistency_loss       # ⚠️ REDUNDANT!
    + 0.1 * gradual_length_loss                   # Short gradual penalty
)
```

### Configuration
```python
focal_alpha = 0.85        # Weight for positive class (boundaries)
focal_gamma = 2.0         # Hard example focusing
type_weight = 0.15        # Type loss weight
type_class_weights = [1.0, 20.0, 10.0]  # [none, cut, gradual]
consistency_weight = 0.20  # ⚠️ PROBLEM!
reasoning_weight = 0.00    # Disabled (was causing instability)
boundary_threshold = 0.55  # Inference threshold
```

### Problem: Consistency Loss is Redundant

```python
# Focal loss already supervises boundary_logits:
loss_focal = focal_loss(boundary_logits, (targets > 0))

# Consistency loss does THE SAME THING:
def consistency_loss(boundary_logits, type_targets):
    is_boundary = (type_targets > 0).float()  # Same as focal loss target!
    return F.binary_cross_entropy_with_logits(boundary_logits, is_boundary)
```

**Both losses supervise `boundary_logits` against the same binary targets!**

This is effectively:
```python
total_loss = (1.0 + 0.20) * boundary_loss + 0.15 * type_loss
            = 1.20 * boundary_loss + 0.15 * type_loss
```

The boundary head gets **20% extra weight** pushing it to over-predict. With 236 FPs, this is clearly happening.

---

## Training Configuration Analysis

### Sampling Strategy
```python
# From train_turbo_sbd_v2.py ClipShotsDataset
samples_per_video_train = 20  # 20 clips × 128 frames = 2,560 frames per video
deterministic = False         # Random sampling with smart focus

# Smart sampling:
if has_transition and random.random() < 0.6:  # 60% transition-focused
    center_frame = (t_start + t_end) // 2
    target_frame_start = center_frame - (seq_len // 2)
    target_frame_start += random.randint(-15, 15)  # ±15 frame jitter
else:  # 40% random location
    target_frame_start = random.randint(0, total_frames - seq_len)
```

**Analysis:**
- ✅ **Good:** 60% transition focus ensures all transitions seen multiple times
- ✅ **Good:** ±15 jitter prevents overfitting to centered boundaries
- ⚠️ **Concern:** Boundaries still only ~4% of frames. Model compensates by over-predicting.
- 💡 **Opportunity:** Could add more challenging negatives (smooth pans, motion) to training

### Hyperparameters
```python
batch_size = 32
num_epochs = 80
learning_rate = 1e-4
weight_decay = 0.01
patience = 15
num_workers = 6        # CPU workers for data loading
prefetch_factor = 4    # Batches to pre-load
```

All reasonable. No obvious issues here.

---

## Prioritized Recommendations

### 🔥 **TIER 1: Quick Wins (No Retraining Required)**

#### **1.1 Raise Boundary Threshold** ⏱️ 5 minutes, 📈 +1-2% F1
**Current:** `boundary_threshold = 0.55`  
**Problem:** 236 FPs vs 78 FNs → Model is over-predicting  
**Fix:** Test thresholds [0.60, 0.65, 0.70] on existing model

```bash
# Test immediately without retraining
python test_unified_hybrid.py --checkpoint outputs/hybrid_detector4/best_model.pt --boundary_threshold 0.60
python test_unified_hybrid.py --checkpoint outputs/hybrid_detector4/best_model.pt --boundary_threshold 0.65
```

**Expected:** Drop ~30-50 FPs → +3-5% precision → +1-2% F1

---

#### **1.2 Test Post-Processing: Top-K Boundaries per Clip** ⏱️ 30 minutes, 📈 +2-3% precision

Add simple filtering to keep only the most confident boundaries per clip:

```python
def filter_top_k_boundaries(boundary_probs, k=3):
    """Keep only top-K boundaries per clip"""
    B, T = boundary_probs.shape
    for b in range(B):
        topk_values, topk_indices = torch.topk(boundary_probs[b], k=min(k, T))
        threshold = topk_values[-1].item()
        boundary_probs[b][boundary_probs[b] < threshold] = 0
    return boundary_probs
```

**Expected:** Enforce sparsity → reduce scattered FPs → +2-3% precision

---

### 🚀 **TIER 2: Architecture Simplification (Requires Retraining)**

#### **2.1 Remove Consistency Loss** ⏱️ 30 minutes, 📈 +2-4% F1

**File:** `core/binary_boundary_loss.py`

```python
# OLD:
total_loss = (loss_focal + 
             self.type_weight * loss_type + 
             self.consistency_weight * loss_consistency)

# NEW:
total_loss = (loss_focal + 
             self.type_weight * loss_type)
# Consistency loss removed - it's redundant with focal loss
```

**Reasoning:** Eliminates double supervision of boundary head → reduces over-prediction bias

---

#### **2.2 Decouple Type Head from Boundary Logits** ⏱️ 1-2 hours, 📈 +2-3% gradual F1

**File:** `detection/hybrid_boundary_detector.py` → `BoundaryDetectionHead`

**Current fragile coupling:**
```python
# In BoundaryDetectionHead.forward():
type_logits_3class[:, :, 0] = -boundary_logits.squeeze(-1)  # ❌ Hard-coupled
type_logits_3class[:, :, 1:] = type_logits_2class
```

**Fix: Independent 3-class prediction:**
```python
# Replace the 2-class type head with 3-class
self.type_head = nn.Sequential(
    nn.Linear(dim + 32, dim // 2),
    nn.LayerNorm(dim // 2),
    nn.GELU(),
    nn.Dropout(0.15),
    nn.Linear(dim // 2, 3)  # Direct 3-class: none, cut, gradual
)

# In forward():
type_logits = self.type_head(type_input)  # [B, T, 3] - clean gradients!
return boundary_logits, type_logits
```

**Benefits:**
- Type head gets clean gradients for all 3 classes during training
- Can still gate with boundary at inference: `type_preds = torch.where(is_boundary, type_preds, 0)`
- Removes fragile dependency on boundary head quality

---

#### **2.3 Simplify ProgressiveFusionBlock** ⏱️ 2-3 hours, 📈 +2-3% F1, -2M params

**Problem:** Each fusion block has 3 redundant temporal processors

**File:** `detection/hybrid_boundary_detector.py` → `ProgressiveFusionBlock`

**REMOVE:** `MultiLevelTemporalAggregation` (dilations [64, 96, 16])  
**KEEP:** `MultiScaleDilatedConv` [1,2,4,8,16,32,64] (already covers full range)  
**KEEP:** `LightweightTemporalAttention` (global context)

**Simplified architecture:**
```python
class ProgressiveFusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # Local patterns (multi-scale dilated convs)
        self.local_conv = MultiScaleDilatedConv(dim)
        
        # Global context (attention)
        self.global_attn = LightweightTemporalAttention(dim)
        
        # Gated fusion (2 sources instead of 3)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # Local + Global (no redundant multi-level)
        local_feat = self.local_conv(x)
        global_feat = self.global_attn(x)
        
        # Gated fusion
        combined = torch.cat([local_feat, global_feat], dim=-1)
        gate_weight = self.gate(combined)
        fused = self.fusion(combined)
        fused = gate_weight * (local_feat + global_feat) / 2 + fused
        
        return fused
```

**Benefits:**
- Removes redundant temporal processing
- -2M parameters → faster training, less overfitting
- Cleaner feature learning (no over-smoothing)

---

#### **2.4 Remove or Simplify Reasoning Enrichment** ⏱️ 1 hour, 📈 Cleaner model

**Option A: Remove entirely** (simplest)
```python
# In HybridBoundaryDetector.forward():
# Skip reasoning module and enrichment projection
x = self.feature_proj(combined)  # [B, T, main_dim]

# Jump directly to fusion blocks
for fusion_block in self.fusion_blocks:
    x = fusion_block(x)
```

**Option B: Re-enable with MSE supervision** (if you want to keep it)
```python
# Replace margin-based reasoning loss with MSE to target ranges
def reasoning_loss_mse(reasoning_outputs, labels):
    # Softer supervision with MSE instead of hard margins
    acceleration = reasoning_outputs['acceleration'].squeeze(-1)
    
    # Target ranges (use means instead of strict bounds)
    accel_targets = torch.zeros_like(acceleration)
    accel_targets[labels == 1] = 0.5  # Cuts: moderate acceleration
    accel_targets[labels == 2] = 0.3  # Graduals: smooth acceleration
    accel_targets[labels == 0] = 0.0  # None: no acceleration
    
    return F.mse_loss(acceleration, accel_targets)
```

---

#### **2.5 Add Sparsity Regularization to Loss** ⏱️ 1 hour, 📈 +1-2% precision

**File:** `core/binary_boundary_loss.py`

```python
def sparsity_loss(boundary_probs, expected_per_clip=2.5):
    """
    Penalize predicting too many boundaries per clip
    
    ClipShots average: ~2-3 transitions per 128-frame clip
    Current model: predicting ~4-5 (over-detection)
    """
    B, T = boundary_probs.shape
    actual_count = boundary_probs.sum(dim=1)  # [B]
    penalty = F.relu(actual_count - expected_per_clip)
    return penalty.mean()

# In forward():
boundary_probs = torch.sigmoid(boundary_logits.squeeze(-1))
loss_sparsity = sparsity_loss(boundary_probs, expected_per_clip=2.5)

total_loss = (loss_focal + 
             self.type_weight * loss_type +
             0.1 * loss_sparsity)  # Add sparsity penalty
```

**Expected:** Direct penalty on over-detection → fewer FPs

---

### 🎯 **TIER 3: Data & Training Improvements**

#### **3.1 Multi-Dataset Training** ⏱️ 1-2 days, 📈 +2-3% F1

Add BBC and RAI datasets (already have predecoding scripts):

```python
# In train_hybrid_detector.py
from torch.utils.data import ConcatDataset

datasets = [
    PredecodedDataset('data/training_cache', split='train'),     # ClipShots
    PredecodedDataset('data/bbc_cache', split='train'),          # BBC
    PredecodedDataset('data/rai_cache', split='train')           # RAI
]

combined_train = ConcatDataset(datasets)
train_loader = DataLoader(combined_train, batch_size=32, shuffle=True)
```

**Prerequisite:**
```bash
python predecode_bbc.py --split train --num_workers 8
python predecode_rai.py --split train --num_workers 8
```

**Benefits:** More diversity → better generalization → fewer smooth-pan FPs

---

#### **3.2 Synthetic Data Augmentation** ⏱️ 1 day, 📈 +1-2% F1

AutoShot-style approach: concatenate random shots to create synthetic cuts

```python
class SyntheticCutDataset(Dataset):
    def __init__(self, base_dataset, samples_per_video=5):
        self.base_dataset = base_dataset
        self.samples_per_video = samples_per_video
        
    def __getitem__(self, idx):
        # Get two random clips
        clip1, labels1, _ = self.base_dataset[random.randint(0, len(self.base_dataset)-1)]
        clip2, labels2, _ = self.base_dataset[random.randint(0, len(self.base_dataset)-1)]
        
        # Concatenate: last 64 frames of clip1 + first 64 of clip2
        frames = torch.cat([clip1[-64:], clip2[:64]], dim=0)  # [128, C, H, W]
        
        # Label: hard cut at frame 64
        labels = torch.zeros(128, dtype=torch.long)
        labels[64] = 1  # Cut
        
        return frames, labels, torch.zeros(128)
```

**Benefits:** Doubles training data, forces precise boundary localization

---

#### **3.3 Hyperparameter Grid Search** ⏱️ 3-5 days, 📈 +1-2% F1

After fixing architecture, tune:

```python
search_grid = {
    'main_dim': [256, 384],
    'num_fusion_blocks': [2, 3],
    'focal_alpha': [0.80, 0.85, 0.90],
    'type_weight': [0.10, 0.15, 0.20],
    'boundary_threshold': [0.55, 0.60, 0.65]
}
# ~2×2×3×3×3 = 108 runs (with early stopping, ~50-60 actual runs)
```

---

## What NOT to Do

### ❌ **Don't Copy AutoShot's Dual Temporal Heads**
AutoShot's dual heads (middle_frame, any_frame) are designed for **synthetic training data** (concatenated shots where middle frame = transition). Your data is real clips with transitions at arbitrary positions. Your two-stage design (boundary → type) is more appropriate.

### ❌ **Don't Pursue Full NAS (84M Candidates)**
Diminishing returns after ~100 trials. Focus on fixing structural issues first.

### ❌ **Don't Retrain from Scratch Without Fixes**
Current F1=0.731 is decent. The main issue is precision tuning and architectural redundancy, not fundamental inability to learn.

### ❌ **Don't Add More Complexity**
The model is already over-engineered with 16 module classes and 9 parallel temporal conv stacks. Simplify first.

---

## Implementation Priority & Timeline

### **Phase 1: Immediate Testing (No Retraining)**
**Time:** 1 hour  
**Expected:** 0.731 → 0.74-0.75 F1

- [ ] Test threshold sweep [0.60, 0.65, 0.70]
- [ ] Implement and test top-K boundary filtering
- [ ] Document best threshold/K values

### **Phase 2: Core Fixes (Retrain Required)**
**Time:** 1 week  
**Expected:** 0.73 → 0.80-0.82 F1

- [ ] **Day 1-2:** Implement fixes 2.1-2.5 (architecture simplification)
  - Remove consistency loss
  - Decouple type head
  - Simplify fusion blocks
  - Remove reasoning enrichment
  - Add sparsity loss
- [ ] **Day 3-4:** Retrain with fixed architecture
- [ ] **Day 5-6:** Validation and threshold tuning
- [ ] **Day 7:** Documentation and checkpointing

### **Phase 3: Data & Scaling (Optional)**
**Time:** 2-3 weeks  
**Expected:** 0.82 → 0.84-0.86 F1

- [ ] Multi-dataset training (BBC + RAI)
- [ ] Synthetic data augmentation
- [ ] Hyperparameter grid search
- [ ] Final model selection and benchmarking

---

## Expected Results After Fixes

### Conservative Estimate
| Metric | Current | After Phase 1 | After Phase 2 | Target |
|--------|---------|---------------|---------------|--------|
| **Overall F1** | 0.731 | 0.745 | 0.815 | 0.850 |
| **Cut F1** | 0.846 | 0.850 | 0.870 | 0.900 |
| **Gradual F1** | 0.573 | 0.600 | 0.720 | 0.780 |
| **Precision** | 0.644 | 0.690 | 0.770 | 0.820 |
| **Recall** | 0.845 | 0.840 | 0.860 | 0.880 |

### Key Improvements
- **Gradual precision:** 0.488 → 0.65-0.72 (fix main issue)
- **False positives:** 236 → ~100-120 (50% reduction)
- **Parameters:** ~9M → ~7M (cleaner model)

---

## References & Context

### Related Files
- **Architecture:** [detection/hybrid_boundary_detector.py](detection/hybrid_boundary_detector.py) (1,407 lines, 16 modules)
- **Training:** [train_hybrid_detector.py](train_hybrid_detector.py) (1,216 lines)
- **Loss:** [core/binary_boundary_loss.py](core/binary_boundary_loss.py) (494 lines)
- **Dataset:** [train_boundary_detector.py](train_boundary_detector.py) (PredecodedDataset)
- **Results:** [outputs/hybrid_detector4/test_results_test/](outputs/hybrid_detector4/test_results_test/)

### Previous Analysis Documents
- [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) - AutoShot comparison & improvement strategies
- [ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md) - Layer-by-layer architecture breakdown
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command reference for training/testing

### Key Insights from Code Review
1. **Pixel-level motion extraction** (before CNN) is a good design choice
2. **Multi-scale dilated convs** [1→64] already cover full 128-frame receptive field
3. **Two-stage type prediction** is conceptually correct but implementation is fragile
4. **Consistency loss** creates double supervision bias toward over-detection
5. **Reasoning module** contributes minimal signal with disabled supervision

---

## Conclusion

The hybrid detector has a **solid foundation** but is held back by:
1. **Over-engineering:** 9 parallel temporal conv stacks cause over-smoothing
2. **Redundant supervision:** Consistency loss doubles boundary loss weight
3. **Fragile coupling:** Type head hard-wired to boundary logits

**Simple fixes (remove redundancy, decouple type head, add sparsity) can achieve 0.80+ F1 without changing the core design.**

The architecture's strength (multi-scale processing, pixel-level motion) is being undermined by structural inefficiencies. **Simplify first, then scale.**

---

**Next Action:** Start with Phase 1 (threshold testing) to validate the over-detection hypothesis, then proceed to Phase 2 (architecture fixes) for maximum impact.
