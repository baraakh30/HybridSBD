# Shot Boundary Detection Models Comparison (V2 - V9)

This document provides a comprehensive comparison of the different model architectures for shot boundary detection, from V2 to V9 (excluding V8).

---

## Summary Table

| Version | Architecture | Key Innovation | Parameters | Input Features | Output Heads |
|---------|--------------|----------------|------------|----------------|--------------|
| V2 | Multi-Head Boundary Detector | TCN + BiLSTM + Transformer stack | ~22M | HSV (768) + OF (40) | Boundary, Type, Extent |
| V3 | Lightweight Difference-Based | Temporal differences as primary signal | ~2M | HSV (768) + OF (40) | Boundary, Type, Extent |
| V4 | Stable & Simple | Raw + Difference features combined | ~2M | HSV (768) + OF (40) | Boundary, Type, Extent |
| V5 | U-Net Transformer | Hierarchical encoder-decoder with transformer bottleneck | Medium | HSV (768) + OF (40) | Boundary, Type, Extent |
| V6 | Multi-Scale Cross-Modal | Cross-modal attention + explicit change detection | ~500K | HSV (768) + OF (40) + PCA (256) | Boundary only |
| V7 | Adaptive Feature Fusion | Flexible feature input + HSV compression | ~500K-1M | Configurable: HSV/OF/PCA | Boundary only |
| V9 | CLIP-Based Contrastive | Similarity-based + semantic features | ~1M | CLIP (512) | Boundary only |

---

## V2: Multi-Head Boundary Detector

### Philosophy
Full-featured architecture with separate feature encoders and multiple output heads for comprehensive boundary analysis.

### Architecture
```
HSV (768) ──> Encoder ──┐
                        ├──> Feature Fusion ──> Multi-Scale Diff ──> TCN ──> BiLSTM ──> Transformer ──> Heads
OF (40) ───> Encoder ───┘
```

### Key Components

1. **Feature Encoders**: Separate gated encoders for HSV and Optical Flow
   - Two-layer projection with gating mechanism
   - LayerNorm + GELU activation

2. **Multi-Scale Difference Layer**: Computes frame differences at scales [1, 3, 5, 10]

3. **Feature Fusion Module**: Attention-based adaptive fusion with learnable weights

4. **Temporal Encoding Stack**:
   - TCN (multi-scale kernels [3, 7, 15])
   - BiLSTM (2 layers, bidirectional)
   - Transformer (4 layers, 8 heads, Pre-LN, DropPath)

5. **Three Output Heads**:
   - **Boundary Head**: Binary boundary probability
   - **Type Head**: Transition type (cut, dissolve, fade, wipe, non-boundary)
   - **Extent Head**: Transition length regression

### Configuration
```python
hidden_dim: 512
num_heads: 8
num_transformer_layers: 4
tcn_channels: 256
lstm_hidden: 256
dropout: 0.25
drop_path: 0.2
```

### Strengths
- Comprehensive multi-task learning
- Rich temporal encoding with TCN + LSTM + Transformer
- Flexible: supports optional CNN and CLIP features

### Weaknesses
- Large model (~22M parameters) - prone to overfitting
- Complex architecture may be overkill for the task

---

## V3: Lightweight Difference-Based Architecture

### Philosophy
> "Shot boundaries are about CHANGE detection, not feature learning."

Key insight: We don't need a model to learn what features look like—we need it to learn what CHANGES in features indicate boundaries.

### Architecture
```
HSV (768) ──> Project (64) ──┐
                              ├──> Concat (128) ──> Temporal Differences ──> Multi-scale TCN ──> Bidirectional Context ──> Heads
OF (40) ───> Project (64) ───┘
```

### Key Components

1. **Feature Projection**: Simple projection to 64D per modality

2. **Temporal Difference Module**: Multi-scale [1, 2, 4, 8, 16]
   - Computes `diff[t] = features[t] - features[t-s]`
   - Outputs both difference features AND magnitude of change

3. **Magnitude Embedding**: Learns to weight different scales via sigmoid gating

4. **Multi-Scale TCN**:
   - Uses **depthwise separable convolutions** for efficiency
   - Parallel branches with different kernel sizes [3, 5, 7]
   - Two layers per branch with dilation

5. **Bidirectional Context**: Lightweight bidirectional using 1D convolutions (faster than LSTM)

### Configuration
```python
hidden_dim: 128  # Much smaller!
diff_scales: [1, 2, 4, 8, 16]
tcn_kernel_sizes: [3, 5, 7]
dropout: 0.3
```

### Strengths
- 10x fewer parameters than V2
- Difference-focused = directly models what matters
- Efficient depthwise separable convolutions

### Weaknesses
- May lose scene context by only using differences

---

## V4: Stable and Simple

### Philosophy
> "Keep BOTH raw features AND difference features (don't discard raw). A boundary is 'change relative to context' - we need both."

### Architecture
```
        ┌─────────────────────────────────────────────┐
HSV ────┤                                             │
        ├──> Concat (808) ──> Project (128) ──────────┤
OF ─────┤                                             ├──> Concat ──> Temporal Encoder ──> Heads
        └──> Differences (scales) ──> Project (128) ──┘
```

### Key Components

1. **Dual Pathway**:
   - Raw features projected to hidden dim
   - Difference features projected to hidden dim
   - Combined via concatenation

2. **Spatial Dropout**: Drops entire feature channels (more aggressive regularization)

3. **Simple Conv-Based Temporal Encoder**:
   - Stacked ConvBlocks with kernels [3, 5, 7]
   - No attention, no LSTM—just convolutions

4. **Conservative Initialization**: Boundary head bias initialized to -1.0 (predict fewer boundaries initially)

### Configuration
```python
hidden_dim: 128
diff_scales: [1, 2, 4, 8]  # Fewer scales
dropout: 0.4  # Higher dropout
```

### Strengths
- More stable training than V3
- Preserves context from raw features
- Very simple architecture

### Weaknesses
- May underfit complex patterns

---

## V5: U-Net Transformer (Context-Aware)

### Philosophy
> "WHERE happens at fine scales (high resolution). WHY happens at coarse scales (low resolution, high context). Transitions are objects that have a duration, not just points."

### Architecture
```
Fusion (HSV + OF) ──> Down1 ──> Down2 ──> Transformer Bottleneck ──> Up1 ──> Up2 ──> Heads
                  ↓           ↓                                   ↑      ↑
                  └─────────────────── Skip Connections ──────────┴──────┘
```

### Key Components

1. **Gated Multimodal Fusion**:
   - Learns to weight Color vs Motion
   - Can ignore unreliable modalities (e.g., camera shake)
   ```python
   z = sigmoid(gate([h1, h2]))
   out = z * h1 + (1-z) * h2
   ```

2. **U-Net Encoder-Decoder**:
   - **DownBlocks**: Conv1d stride 2 + residual blocks
   - **UpBlocks**: Interpolate + concat with skip + residual

3. **Transformer Bottleneck**: Global context at lowest resolution
   - Learnable positional encoding
   - 2 transformer layers, 8 heads

4. **Skip Connections**: Preserve precise boundary locations

### Configuration
```python
base_dim: 128
dim_mults: (1, 2, 4)  # [128, 256, 512]
dropout: 0.2
drop_path: 0.2
```

### Strengths
- Hierarchical processing captures both local precision and global context
- Skip connections preserve boundary localization
- Elegant encoder-decoder design

### Weaknesses
- More complex than simple approaches
- May be overkill for simpler boundary types

---

## V6: Multi-Scale Cross-Modal Temporal Boundary Detector

### Philosophy
1. **EXPLICIT CHANGE DETECTION**: Boundaries are about CHANGE, not absolute features
2. **CROSS-MODAL ATTENTION**: Different modalities detect different boundary types
3. **LIGHTWEIGHT DESIGN**: ~500K parameters instead of millions
4. **AGGRESSIVE REGULARIZATION**: Combat overfitting

### Architecture
```
HSV ──> ModalityEncoder ──┐
                          │
Flow ─> ModalityEncoder ──┼──> Cross-Modal Fusion Layers ──> Temporal Aggregation ──> Boundary Head
                          │
PCA ──> ModalityEncoder ──┘
```

### Key Components

1. **Modality Encoders**: Each modality gets its own encoder
   - Raw projection + multi-scale temporal differences
   - **Symmetric differences**: backward (x[t] - x[t-k]) AND forward (x[t+k] - x[t])

2. **Cross-Modal Fusion**:
   - Pairwise cross-attention between all modalities
   - HSV→Flow, HSV→PCA, Flow→HSV, Flow→PCA, PCA→HSV, PCA→Flow

3. **Efficient Cross-Modal Attention**: Standard multi-head attention

4. **Temporal Aggregation**: Dilated convolutions [1, 2, 4]

5. **Regularization**:
   - Feature Dropout (drops entire feature channels)
   - Stochastic Depth (randomly skip layers)

### Configuration
```python
hsv_dim: 768, flow_dim: 40, pca_dim: 256
hidden_dim: 64
diff_scales: (1, 2, 4, 8)
dropout: 0.3, feature_dropout: 0.2
stochastic_depth_rate: 0.1
num_attention_heads: 4
```

### Strengths
- Rich cross-modal interaction
- Explicit change detection
- Strong regularization for small datasets

### Weaknesses
- Requires PCA features (extra preprocessing)
- Only predicts boundaries (no type/extent)

---

## V7: Adaptive Feature Fusion with Change Focus

### Philosophy
1. **FLEXIBLE FEATURE INPUT**: Can use any combination of HSV, optical flow, and PCA
2. **HSV COMPRESSION**: Compress HSV from 768 to 128/256 to prevent overfitting
3. **CHANGE-CENTRIC**: Focus on temporal changes, not absolute features
4. **DUAL-SCALE PROCESSING**: Captures both instant cuts and gradual transitions

### Architecture
```
HSV ──> [Compressor] ──> DiffEncoder ──┐
                                        │
Flow ────────────────> DiffEncoder ──> AdaptiveFusion ──> LocalTCN ──> GlobalAttention ──> Head
                                        │                      ↑
PCA ─────────────────> DiffEncoder ──┘                        │
                                                               │
        Magnitudes ─────────────────────────────────────────────┘
```

### Key Components

1. **HSV Compressor** (optional): Two-stage compression 768 → 128
   - Reduces overfitting from high-dimensional HSV

2. **Temporal Difference Encoders**: One per modality
   - Multi-scale [1, 2, 4, 8, 16]
   - Outputs both difference features AND change magnitude

3. **Adaptive Fusion**:
   - Learns to weight different features based on informativeness
   - Softmax-gated combination

4. **Dual-Scale Processing**:
   - **Local Context Encoder**: TCN with increasing dilation [1, 2, 4]
   - **Global Context Encoder**: Lightweight attention with learnable PE

5. **Magnitude Processing**: Separate pathway for change magnitudes

### Configuration
```python
hsv_dim: 768 → compressed_hsv_dim: 128
flow_dim: 40, pca_dim: 256
hidden_dim: 128
diff_scales: (1, 2, 4, 8, 16)
dropout: 0.4, feature_dropout: 0.15
num_local_layers: 3, num_global_layers: 2
```

### Model Variants
- `trimodal`: HSV + Flow + PCA (compressed HSV)
- `dual`/`flow_pca`: Flow + PCA only
- `hsv_flow`: HSV + Flow (compressed HSV)
- `flow_only`: Only optical flow
- `pca_only`: Only PCA

### Strengths
- Maximum flexibility in feature selection
- HSV compression prevents overfitting
- Magnitude signal as separate pathway

### Weaknesses
- Configuration complexity
- Only boundary output

---

## V9: CLIP-Based Contrastive Boundary Detector

### Philosophy
> "CLIP already 'understands' scenes semantically. We just need to detect when the semantics CHANGE significantly."

Key insight: CLIP features encode high-level semantic content. Boundaries are where similarity drops.

### Architecture
```
CLIP (512) ──┬──> Feature Projection ────────────────────────────────┐
             │                                                        │
             ├──> Multi-Scale Similarity ──> Discontinuity Encoding ──┼──> Combine ──> LocalConv ──> Attention ──> Head
             │                                                        │
             └──> Temporal Differences ───────────────────────────────┘
```

### Key Components

1. **Multi-Scale Similarity Computation**:
   - Core insight: boundaries are where similarity drops
   - Computes cosine similarity at scales [1, 2, 4, 8, 16]
   - Both forward and backward similarity
   - **Discontinuity = 1 - similarity** (high at boundaries)

2. **Temporal Difference Module**:
   - Standard multi-scale differences [1, 2, 4, 8]
   - Captures magnitude of change

3. **Local Temporal Convolutions**:
   - Multi-branch with kernels [3, 5, 7]
   - BatchNorm + ReLU

4. **Lightweight Attention**:
   - Single layer of self-attention
   - Pre-norm design

5. **Calibrated Boundary Head**: Multi-layer MLP

### Configuration
```python
clip_dim: 512
hidden_dim: 128
bottleneck_dim: 64
similarity_scales: (1, 2, 4, 8, 16)
kernel_sizes: (3, 5, 7)
dropout: 0.4, feature_dropout: 0.2
num_attention_heads: 4
attention_layers: 1
```

### Model Variants
- `base`: ~1M parameters
- `large`: Larger hidden_dim=256, more attention layers

### Strengths
- Leverages powerful CLIP semantic features
- Similarity-based detection is intuitive
- Returns similarity matrix for analysis

### Weaknesses
- Requires CLIP features (expensive to extract)
- May miss low-level visual changes that CLIP ignores

---

## Evolution of Design Decisions

### 1. Model Size Trend
```
V2 (~22M) → V3 (~2M) → V4 (~2M) → V6 (~500K) → V7 (~500K-1M) → V9 (~1M)
```
Clear trend toward smaller, more regularized models.

### 2. Feature Processing Evolution

| Version | Raw Features | Differences | Fusion Method |
|---------|--------------|-------------|---------------|
| V2 | Yes | Multi-scale | Attention-weighted |
| V3 | No | Multi-scale + magnitude | Magnitude gating |
| V4 | Yes | Multi-scale | Concatenation |
| V5 | Yes (fused) | No | Gated fusion |
| V6 | No | Symmetric multi-scale | Cross-modal attention |
| V7 | Compressed | Multi-scale + magnitude | Adaptive gating |
| V9 | Yes | Multi-scale + similarity | Concatenation |

### 3. Temporal Modeling Evolution

| Version | Local | Global |
|---------|-------|--------|
| V2 | TCN + BiLSTM | Transformer (4 layers) |
| V3 | Multi-scale TCN | Bidirectional Conv |
| V4 | Simple Conv stack | None |
| V5 | U-Net Conv | Transformer bottleneck |
| V6 | Dilated Conv | Cross-modal attention |
| V7 | TCN (dilated) | Lightweight attention |
| V9 | Multi-branch Conv | Single attention layer |

### 4. Regularization Evolution
- V2: Basic dropout + drop path
- V3: Higher dropout (0.3)
- V4: Higher dropout (0.4) + spatial dropout
- V6: Feature dropout + stochastic depth
- V7: HSV compression + feature dropout (0.15)
- V9: Feature dropout + high dropout (0.4)

---

## Recommendations by Use Case

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Full analysis (type + extent) | V2 or V4 | Multi-head outputs |
| Small dataset | V6 or V7 | Strong regularization |
| CLIP features available | V9 | Designed for semantic features |
| Fast inference | V4 | Simplest architecture |
| Flexibility in features | V7 | Configurable inputs |
| Maximum accuracy | V6 | Cross-modal attention |
| Gradual transitions focus | V5 | U-Net preserves duration |

---

## Code Example

```python
# V6 - Cross-Modal (recommended for ClipShots)
from detection.models_v6 import create_model_v6, ModelConfigV6
config = ModelConfigV6(dropout=0.4)
model = create_model_v6('full', config)

# V7 - Flexible Features
from detection.models_v7 import create_model_v7, ModelConfigV7
model = create_model_v7('trimodal', hidden_dim=128, dropout=0.4)

# V9 - CLIP-Based
from detection.models_v9 import create_model_v9, ModelConfigV9
model = create_model_v9('base')
```
