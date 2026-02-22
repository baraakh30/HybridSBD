# Video Shot Boundary Detection

A dual-head deep learning system for detecting shot boundaries in video — handling both **hard cuts** and **gradual transitions** (dissolves, fades). Trained on AutoShot + ClipShots; evaluated zero-shot on BBC Planet Earth.

## Results

All models trained exclusively on SHOT + ClipShots. **Bold** = best per column.

| Model | Params | SHOT F1 | ClipShots F1 | BBC zero-shot F1 |
|---|---|---|---|---|
| AutoShot (paper) | ~7M | 0.841 | 0.787 | — |
| Ours dim=256, 3-block | 5.8M | 0.9160 | 0.7804 | 0.9563 |
| Ours dim=512, 3-block | 14.1M | 0.9103 | 0.7844 | 0.9654 |
| Ours dim=256, 4-block | 6.5M | 0.9201 | 0.7916 | 0.9669 |
| **Ours dim=384, 4-block** | **10.9M** | **0.9238** | **0.7934** | **0.9706** |
| Ours dim=512, 4-block | 17.1M | 0.9066 | 0.7729 | 0.9651 |

See [RESULTS.md](RESULTS.md) for full methodology, threshold sweeps, and per-video breakdowns.

---

## Architecture

`HybridBoundaryDetector` — dual-head design following AutoShot's evaluation paradigm:

- **Head y (one_hot):** Predicts the midpoint frame of each transition (primary loss weight 5.0).
- **Head z (many_hot):** Predicts every frame inside gradual transitions (auxiliary loss weight 0.3).

**Inference:**
```
pred_is_boundary = sigmoid(y) > boundary_threshold
pred_is_gradual  = sigmoid(z) > gradual_threshold
pred_many_hot    = pred_is_gradual | (pred_is_boundary & ~pred_is_gradual)
```

### Feature streams (main_dim=512)

| Stream | Extractor | Output dim |
|---|---|---|
| RGB spatial | EfficientSpatialEncoder (3D stem ±7fr) | 640 → main_dim/2 |
| Diff spatial | EfficientSpatialEncoder (3D stem on multi-dist diffs) | 640 → main_dim/2 |
| Feature delta | 2D per-frame, distances [1,2,4,8] | 640×4 → main_dim/2 |
| Frame similarity | FrameSimilarity ×3 window scales | 192 → main_dim/4 |
| Color hist similarity | ColorHistogramSimilarity | 128 → main_dim/4 |
| Pixel motion | PixelMotionExtractor (5 distances) | 15 (raw) |

Fused features pass through N × `ProgressiveFusionBlock` (MultiScaleDilatedConv with dilations 1,4,16,64 + LightweightTemporalAttention with learned positional encoding + gated residual).

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.x, and `torchcodec` for video decoding.

---

## Workflow

### 1. Pre-decode training data

```bash
# AutoShot training clips (jitter + random fill, 70 clips/video)
python predecode_autoshot_train.py

# ClipShots training clips (transition-focused sampling)
python predecode_dataset.py

# BBC Planet Earth (sliding window, for fine-tuning / zero-shot eval)
python predecode_bbc.py
```

### 2. Train

```bash
# Train from scratch
python train_hybrid_detector.py

# Resume from checkpoint
python train_hybrid_detector.py --resume outputs/hybrid_detector/best_model.pt

# Key options
python train_hybrid_detector.py --main_dim 384 --num_fusion_blocks 4 --epochs 80
```

### 3. Evaluate

```bash
# AutoShot test set (SHOT benchmark)
python predecode_autoshot.py  # pre-decode test clips first
python test_autoshot.py --checkpoint outputs/hybrid_detector/best_model.pt

# BBC Planet Earth (zero-shot)
python test_bbc.py --checkpoint outputs/hybrid_detector/best_model.pt

# ClipShots
python test_bbc.py --checkpoint outputs/hybrid_detector/best_model.pt  # analogous
```

### 4. Fine-tune on BBC

```bash
python finetune_bbc.py --checkpoint outputs/hybrid_detector/best_model.pt
```

---

## Project Structure

```
detection/
  hybrid_boundary_detector.py   # Model definition: HybridBoundaryDetector
core/
  binary_boundary_loss.py       # Focal BCE dual-head loss
  context_aware_loss.py
data/
  autoshot_cache/               # Pre-decoded AutoShot test clips
  autoshot_cache_train/         # Pre-decoded AutoShot training clips
  bbc_cache/                    # Pre-decoded BBC clips
  training_cache/               # Pre-decoded ClipShots clips

predecode_autoshot_train.py     # Pre-decode AutoShot train set
predecode_autoshot.py           # Pre-decode AutoShot test set
predecode_dataset.py            # Pre-decode ClipShots
predecode_bbc.py                # Pre-decode BBC Planet Earth
train_hybrid_detector.py        # Main training script
finetune_bbc.py                 # BBC fine-tuning script
test_autoshot.py                # Evaluate on SHOT benchmark
test_bbc.py                     # Evaluate on BBC Planet Earth
RESULTS.md                      # Full results and methodology
```

---

## Citation

```bibtex
@misc{hybrid_sbd_2026,
  title  = {Dual-Head Hybrid Shot Boundary Detector},
  year   = {2026}
}
```
