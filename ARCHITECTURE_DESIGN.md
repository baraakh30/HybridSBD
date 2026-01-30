# SBDNet: Efficient Frame-Based Shot Boundary Detection Architecture

## Executive Summary

This document outlines a new state-of-the-art architecture for shot boundary detection that:
1. Works directly on raw video frames (no pre-extracted features)
2. Uses efficient data loading with high workers and large prefetching
3. Combines the best ideas from DSM, TransNetV2, and modern architectures
4. Targets beating DSM (0.848 Cut F1, 0.870 Gradual F1) on ClipShots

---

## Analysis of SOTA Methods

### 1. DSM (Dilated Segment Modeling) - Tang et al.
**Results**: Cut F1=0.848, Gradual F1=0.870

**Architecture**:
- **Backbone**: ResNet-18 (3D convolutions)
- **Input**: 16 frames at 128×128 resolution
- **Approach**: Sliding window classification (3 classes: no transition, cut, gradual)
- **Key features**: 
  - 3D convolutions for spatiotemporal modeling
  - Kinetics pretraining
  - Simple classification head

**Strengths**: Solid performance, pre-training helps  
**Weaknesses**: Heavy 3D ResNet, sliding window is slow, coarse temporal resolution

### 2. TransNetV2 - Souček et al.
**Results**: ClipShots F1=77.9%, BBC=96.2%, RAI=93.9%

**Architecture**:
- **Core**: Stacked Dilated DCNNs (DDCNNs)
  - Parallel dilated 3D convolutions with rates [1, 2, 4, 8]
  - Multiple stacked blocks (L=3 layers, S=2 blocks each)
- **Input**: 27×48 resolution (very small!), 100 frames at a time
- **Key features**:
  - **Frame Similarity Layer**: Computes cosine similarity between frames
  - **Color Histograms**: 128-dim color histogram features
  - **Two-head output**: Single-frame + many-hot predictions
  - Stochastic depth regularization

**Strengths**: Very fast, efficient small resolution, frame similarity is clever  
**Weaknesses**: Low resolution may miss fine details, no semantic understanding

### 3. DeepSBD (AlexNet-like baseline)
- Simple 3D CNN with pooling
- Classification approach (3 classes)
- Moderate performance

---

## Design Philosophy for SBDNet

### Core Insights

1. **Shot boundaries are about CHANGE detection**:
   - A cut = instant visual discontinuity
   - Gradual = progressive visual change over frames
   - We need to model CHANGES explicitly, not just features

2. **Multi-scale temporal reasoning is critical**:
   - Cuts happen in 1-2 frames
   - Gradual transitions span 5-50+ frames
   - Need receptive fields at multiple scales

3. **Frame similarity is a strong signal**:
   - TransNetV2's frame similarity layer is brilliant
   - Boundaries = low similarity between consecutive regions

4. **Efficiency matters**:
   - Small spatial resolution is OK (TransNetV2 uses 27×48!)
   - Temporal resolution is more important than spatial
   - Prefetching and parallel loading are essential

5. **Don't overfit**:
   - Your previous models overfit
   - Need strong regularization: dropout, stochastic depth, data augmentation
   - Keep model relatively small (~2-5M params)

---

## SBDNet Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SBDNet Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: [B, T, H, W, 3]  (T=100 frames, H=48, W=64, RGB)                   │
│                                                                             │
│   ┌─────────────────────┐                                                   │
│   │   Stem: Conv3D 7×7  │  → Spatial downsampling to 24×32                  │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│   ┌──────────▼──────────┐                                                   │
│   │  Stage 1: MDDCNN    │  → Filters=32, Spatial: 24×32                     │
│   │  (2 blocks)         │                                                   │
│   └──────────┬──────────┘                                                   │
│              │ + AvgPool (1,2,2)                                            │
│   ┌──────────▼──────────┐                                                   │
│   │  Stage 2: MDDCNN    │  → Filters=64, Spatial: 12×16                     │
│   │  (2 blocks)         │                                                   │
│   └──────────┬──────────┘                                                   │
│              │ + AvgPool (1,2,2)                                            │
│   ┌──────────▼──────────┐                                                   │
│   │  Stage 3: MDDCNN    │  → Filters=128, Spatial: 6×8                      │
│   │  (2 blocks)         │                                                   │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│   ┌──────────▼──────────────────────────────────────────────────────────┐   │
│   │                     Feature Aggregation                              │   │
│   │  ┌────────────────┬────────────────┬────────────────┐               │   │
│   │  │ Flatten 3D     │ Frame          │ Temporal       │               │   │
│   │  │ (128×6×8=6144) │ Similarity     │ Difference     │               │   │
│   │  │                │ (128-dim)      │ (128-dim)      │               │   │
│   │  └────────┬───────┴───────┬────────┴───────┬────────┘               │   │
│   │           └───────────────┼────────────────┘                        │   │
│   │                           │ Concat                                  │   │
│   └───────────────────────────┼─────────────────────────────────────────┘   │
│                               │                                             │
│   ┌───────────────────────────▼─────────────────────────────────────────┐   │
│   │                    Temporal Transformer                              │   │
│   │  • 2 layers, 4 heads                                                │   │
│   │  • Learned positional encoding                                      │   │
│   │  • Pre-norm (LayerNorm before attention)                            │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                             │
│   ┌───────────────────────────▼─────────────────────────────────────────┐   │
│   │                    Output Heads                                      │   │
│   │  ┌─────────────────────┐  ┌─────────────────────┐                   │   │
│   │  │ Boundary Head       │  │ Many-Hot Head       │                   │   │
│   │  │ (single-frame)      │  │ (all frames in      │                   │   │
│   │  │                     │  │  transition)        │                   │   │
│   │  └─────────────────────┘  └─────────────────────┘                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Output: [B, T, 1] per head                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Input Pipeline (Critical for Speed!)

```python
class VideoDataLoader:
    """
    Efficient video frame loading with:
    - Multiple workers (8-16)
    - Large prefetch buffer (2-4 batches)
    - Memory-mapped video reading
    - On-GPU preprocessing
    """
    
    def __init__(self):
        self.num_workers = 8
        self.prefetch_factor = 4
        self.input_size = (48, 64)  # H, W - small but effective
        self.sequence_length = 100  # frames per sample
        self.stride = 50  # overlap for training
```

**Key Implementation Details**:
- Use `decord` library for fast GPU-accelerated video decoding
- Memory-map video files to avoid loading entire videos
- Use `torch.utils.data.DataLoader` with `pin_memory=True`
- Prefetch to GPU with `prefetch_factor=4`

#### 2. Stem Layer

```python
class Stem(nn.Module):
    """Initial convolution for spatial downsampling"""
    def __init__(self, out_channels=32):
        super().__init__()
        # 7×7 spatial, 3 temporal kernel
        self.conv = nn.Conv3d(3, out_channels, 
                              kernel_size=(3, 7, 7),
                              stride=(1, 2, 2),  # Downsample spatial only
                              padding=(1, 3, 3))
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
```

**Output**: [B, 32, T, 24, 32]

#### 3. Multi-scale Dilated DCNN (MDDCNN) Block

This is the core building block, inspired by TransNetV2 but enhanced:

```python
class MDDCNN(nn.Module):
    """
    Multi-scale Dilated Dense CNN Block
    
    Key innovation: Parallel dilated convolutions capture
    multiple temporal scales simultaneously
    """
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4, 8]):
        super().__init__()
        
        # Parallel branches with different dilation rates
        self.branches = nn.ModuleList([
            nn.Sequential(
                # Spatial conv: 3×3
                nn.Conv3d(in_channels, out_channels // len(dilation_rates),
                         kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(out_channels // len(dilation_rates)),
                nn.ReLU(inplace=True),
                # Temporal conv with dilation
                nn.Conv3d(out_channels // len(dilation_rates), 
                         out_channels // len(dilation_rates),
                         kernel_size=(3, 1, 1), 
                         padding=(d, 0, 0),
                         dilation=(d, 1, 1))
            )
            for d in dilation_rates
        ])
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Squeeze-and-Excitation for channel attention
        self.se = SEBlock(out_channels, reduction=16)
    
    def forward(self, x):
        # Process all branches
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate along channel dimension
        out = torch.cat(branch_outputs, dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        # Channel attention
        out = self.se(out)
        
        return out
```

#### 4. Stacked MDDCNN Stage

```python
class StackedMDDCNN(nn.Module):
    """
    Stack of MDDCNN blocks with residual connection
    and optional stochastic depth
    """
    def __init__(self, in_channels, out_channels, num_blocks=2, 
                 stochastic_depth_prob=0.1):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            MDDCNN(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_blocks)
        ])
        
        # Residual projection if channels change
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) \
                        if in_channels != out_channels else nn.Identity()
        
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.drop_prob = stochastic_depth_prob
    
    def forward(self, x, training=True):
        shortcut = self.shortcut(x)
        
        for block in self.blocks:
            x = block(x)
        
        # Stochastic depth
        if training and random.random() < self.drop_prob:
            x = shortcut
        else:
            x = x + shortcut
        
        x = self.pool(x)
        return x
```

#### 5. Frame Similarity Module (Key Innovation from TransNetV2)

```python
class FrameSimilarity(nn.Module):
    """
    Computes similarity between frames to detect discontinuities.
    
    Key insight: At shot boundaries, frame similarity drops sharply.
    This provides a strong signal for both cuts and gradual transitions.
    """
    def __init__(self, in_channels, similarity_dim=128, output_dim=128, 
                 lookup_window=101):
        super().__init__()
        
        self.lookup_window = lookup_window
        self.similarity_dim = similarity_dim
        
        # Project features to similarity space
        self.projection = nn.Linear(in_channels, similarity_dim)
        
        # Output projection
        self.output_proj = nn.Linear(lookup_window, output_dim)
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Global average pool spatial dimensions
        x = x.mean(dim=[3, 4])  # [B, C, T]
        x = x.permute(0, 2, 1)   # [B, T, C]
        
        # Project to similarity space
        x = self.projection(x)  # [B, T, similarity_dim]
        
        # Normalize for cosine similarity
        x = F.normalize(x, p=2, dim=-1)
        
        # Compute pairwise similarity matrix
        # For each frame, compute similarity to nearby frames
        half_window = self.lookup_window // 2
        similarities = []
        
        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            
            # Similarity to window
            sim = torch.einsum('bc,bnc->bn', x[:, t], x[:, start:end])
            
            # Pad to fixed size
            pad_left = t - start if t < half_window else half_window
            pad_right = self.lookup_window - (end - start) - pad_left
            
            if pad_left > 0 or pad_right > 0:
                sim = F.pad(sim, (pad_left, pad_right), value=0)
            
            similarities.append(sim)
        
        # Stack: [B, T, lookup_window]
        similarities = torch.stack(similarities, dim=1)
        
        # Project to output dim
        output = self.output_proj(similarities)  # [B, T, output_dim]
        
        return output
```

#### 6. Temporal Difference Module (Explicit Change Detection)

```python
class TemporalDifference(nn.Module):
    """
    Computes multi-scale temporal differences.
    
    Key insight: Shot boundaries show high difference magnitude
    at various temporal scales.
    """
    def __init__(self, in_channels, diff_scales=[1, 2, 4, 8], output_dim=128):
        super().__init__()
        
        self.scales = diff_scales
        
        # Project differences
        self.proj = nn.Sequential(
            nn.Linear(in_channels * len(diff_scales) * 2, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Global average pool
        x = x.mean(dim=[3, 4])  # [B, C, T]
        x = x.permute(0, 2, 1)   # [B, T, C]
        
        differences = []
        for scale in self.scales:
            # Backward difference: x[t] - x[t-scale]
            diff_back = torch.zeros_like(x)
            diff_back[:, scale:] = x[:, scale:] - x[:, :-scale]
            
            # Forward difference: x[t+scale] - x[t]
            diff_fwd = torch.zeros_like(x)
            diff_fwd[:, :-scale] = x[:, scale:] - x[:, :-scale]
            
            differences.extend([diff_back, diff_fwd])
        
        # Concatenate all differences
        diff_features = torch.cat(differences, dim=-1)  # [B, T, C*scales*2]
        
        # Project
        output = self.proj(diff_features)  # [B, T, output_dim]
        
        return output
```

#### 7. Lightweight Temporal Transformer

```python
class TemporalTransformer(nn.Module):
    """
    Lightweight transformer for global temporal context.
    Uses pre-norm and learned positional encoding.
    """
    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 300, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]
        
        # Transform
        x = self.transformer(x)
        x = self.norm(x)
        
        return x
```

#### 8. Output Heads

```python
class OutputHeads(nn.Module):
    """
    Two prediction heads:
    1. Single-frame: Predicts if each frame is a boundary
    2. Many-hot: Predicts all frames that belong to a transition
    
    This dual supervision helps with gradual transitions.
    """
    def __init__(self, d_model):
        super().__init__()
        
        self.head_single = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 1)
        )
        
        self.head_many = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        # x: [B, T, D]
        single_frame = self.head_single(x)  # [B, T, 1]
        many_hot = self.head_many(x)        # [B, T, 1]
        
        return single_frame, many_hot
```

### Complete Model

```python
class SBDNet(nn.Module):
    """
    Shot Boundary Detection Network
    
    Key innovations:
    1. Multi-scale dilated convolutions for temporal modeling
    2. Frame similarity for explicit discontinuity detection
    3. Temporal differences for change magnitude
    4. Dual-head output for cut and gradual transitions
    """
    
    def __init__(self, 
                 base_filters=32,
                 num_stages=3,
                 blocks_per_stage=2,
                 transformer_layers=2,
                 transformer_heads=4,
                 dropout=0.3,
                 stochastic_depth=0.1):
        super().__init__()
        
        # Stem
        self.stem = Stem(base_filters)
        
        # Stacked MDDCNN stages
        self.stages = nn.ModuleList()
        in_ch = base_filters
        for i in range(num_stages):
            out_ch = base_filters * (2 ** i)
            self.stages.append(
                StackedMDDCNN(in_ch, out_ch, blocks_per_stage, 
                             stochastic_depth_prob=stochastic_depth)
            )
            in_ch = out_ch
        
        # Feature dimension after stages
        # 128 channels × 6 × 8 spatial = 6144 after flatten
        final_channels = base_filters * (2 ** (num_stages - 1))
        spatial_size = 48 // (2 ** (num_stages + 1)) * 64 // (2 ** (num_stages + 1))
        flatten_dim = final_channels * spatial_size
        
        # Auxiliary modules
        self.frame_similarity = FrameSimilarity(final_channels, output_dim=128)
        self.temporal_diff = TemporalDifference(final_channels, output_dim=128)
        
        # Aggregation
        total_dim = flatten_dim + 128 + 128  # flatten + similarity + diff
        self.aggregation = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal transformer
        self.transformer = TemporalTransformer(
            d_model=512,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # Output heads
        self.heads = OutputHeads(512)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, training=True):
        # x: [B, T, H, W, C] -> [B, C, T, H, W]
        x = x.permute(0, 4, 1, 2, 3).float() / 255.0
        
        B, C, T, H, W = x.shape
        
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x, training=training)
        
        # x: [B, final_channels, T, H', W']
        
        # Frame similarity
        sim_features = self.frame_similarity(x)  # [B, T, 128]
        
        # Temporal differences
        diff_features = self.temporal_diff(x)  # [B, T, 128]
        
        # Flatten spatial
        _, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(B, T, -1)  # [B, T, C*H*W]
        
        # Aggregate
        x = torch.cat([x, sim_features, diff_features], dim=-1)
        x = self.aggregation(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Heads
        single_frame, many_hot = self.heads(x)
        
        return {
            'single_frame': single_frame.squeeze(-1),
            'many_hot': many_hot.squeeze(-1),
            'similarity': sim_features
        }
```

---

## Training Strategy

### 1. Data Augmentation

```python
class VideoAugmentation:
    """
    Strong augmentation to prevent overfitting
    """
    def __init__(self):
        self.transforms = [
            RandomTemporalCrop(length=100),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            RandomGrayscale(p=0.1),
            GaussianNoise(std=0.02),
            Cutout(n_holes=2, length=8),  # Spatial cutout
            TemporalDropout(p=0.1),  # Randomly drop frames
        ]
```

### 2. Loss Function

```python
class SBDLoss(nn.Module):
    """
    Combined loss for shot boundary detection
    
    Components:
    1. Focal loss for single-frame prediction (handles class imbalance)
    2. BCE loss for many-hot prediction
    3. Temporal smoothness regularization
    """
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, 
                 single_weight=1.0, many_weight=0.5, smooth_weight=0.1):
        super().__init__()
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.single_weight = single_weight
        self.many_weight = many_weight
        self.smooth_weight = smooth_weight
    
    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        pt = target * p + (1 - target) * (1 - p)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        return (alpha_weight * focal_weight * bce).mean()
    
    def forward(self, outputs, targets):
        single_pred = outputs['single_frame']
        many_pred = outputs['many_hot']
        
        single_target = targets['single_frame']
        many_target = targets['many_hot']
        
        # Main losses
        loss_single = self.focal_loss(single_pred, single_target)
        loss_many = F.binary_cross_entropy_with_logits(many_pred, many_target)
        
        # Temporal smoothness (predictions shouldn't change too abruptly except at boundaries)
        pred_diff = torch.abs(single_pred[:, 1:] - single_pred[:, :-1])
        target_diff = torch.abs(single_target[:, 1:] - single_target[:, :-1])
        loss_smooth = F.mse_loss(pred_diff, target_diff)
        
        total_loss = (self.single_weight * loss_single + 
                      self.many_weight * loss_many +
                      self.smooth_weight * loss_smooth)
        
        return {
            'total': total_loss,
            'single': loss_single,
            'many': loss_many,
            'smooth': loss_smooth
        }
```

### 3. Training Configuration

```python
training_config = {
    # Data
    'input_size': (48, 64),  # H, W
    'sequence_length': 100,
    'stride': 50,
    'batch_size': 16,
    'num_workers': 8,
    'prefetch_factor': 4,
    
    # Model
    'base_filters': 32,
    'num_stages': 3,
    'blocks_per_stage': 2,
    'transformer_layers': 2,
    'transformer_heads': 4,
    'dropout': 0.3,
    'stochastic_depth': 0.1,
    
    # Optimization
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'scheduler': 'OneCycleLR',
    'max_epochs': 100,
    'warmup_epochs': 5,
    
    # Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # Regularization
    'label_smoothing': 0.1,
    'mixup_alpha': 0.2,
    'gradient_clip': 1.0,
    
    # Early stopping
    'patience': 15,
    'min_delta': 1e-4,
}
```

---

## Efficient Data Loading Pipeline

### Video Dataset

```python
import decord
from decord import VideoReader, cpu, gpu

class ClipShotsDataset(torch.utils.data.Dataset):
    """
    Efficient video dataset with:
    - Decord for fast GPU-accelerated video decoding
    - Caching of video metadata
    - On-the-fly sampling and augmentation
    """
    
    def __init__(self, video_list, annotations, 
                 input_size=(48, 64), 
                 sequence_length=100,
                 stride=50,
                 training=True,
                 use_gpu_decode=True):
        
        self.videos = video_list
        self.annotations = annotations
        self.input_size = input_size
        self.seq_len = sequence_length
        self.stride = stride
        self.training = training
        
        # Use GPU decoding if available
        self.ctx = gpu(0) if use_gpu_decode and torch.cuda.is_available() else cpu(0)
        
        # Pre-compute all valid samples
        self.samples = self._build_sample_index()
        
        # Augmentation
        self.augment = VideoAugmentation() if training else None
    
    def _build_sample_index(self):
        """Pre-compute all valid (video, start_frame) pairs"""
        samples = []
        for video_path, video_info in self.annotations.items():
            num_frames = video_info['num_frames']
            
            # Create overlapping windows
            for start in range(0, num_frames - self.seq_len + 1, self.stride):
                samples.append({
                    'video_path': video_path,
                    'start_frame': start,
                    'end_frame': start + self.seq_len,
                    'transitions': self._get_transitions_in_range(
                        video_info['transitions'], start, start + self.seq_len
                    )
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load frames using decord
        vr = VideoReader(sample['video_path'], ctx=self.ctx, 
                        width=self.input_size[1], height=self.input_size[0])
        
        frames = vr.get_batch(
            list(range(sample['start_frame'], sample['end_frame']))
        ).asnumpy()  # [T, H, W, C]
        
        # Create labels
        single_frame_labels = self._create_single_frame_labels(
            sample['transitions'], self.seq_len
        )
        many_hot_labels = self._create_many_hot_labels(
            sample['transitions'], self.seq_len
        )
        
        # Augmentation
        if self.augment:
            frames, single_frame_labels, many_hot_labels = \
                self.augment(frames, single_frame_labels, many_hot_labels)
        
        return {
            'frames': torch.from_numpy(frames).byte(),
            'single_frame': torch.from_numpy(single_frame_labels).float(),
            'many_hot': torch.from_numpy(many_hot_labels).float()
        }
```

### DataLoader Configuration

```python
def create_dataloader(dataset, batch_size=16, num_workers=8, training=True):
    """
    Create highly efficient dataloader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=training
    )
```

---

## Expected Performance

Based on the architecture analysis and design:

| Metric | DSM | TransNetV2 | SBDNet (Projected) |
|--------|-----|------------|-------------------|
| Cut F1 | 0.848 | 0.779* | **0.86-0.88** |
| Gradual F1 | 0.870 | - | **0.88-0.90** |
| Speed (FPS) | ~30 | ~300 | ~200 |
| Parameters | ~11M | ~3M | **~5M** |

*TransNetV2 reports combined F1

### Why SBDNet Should Beat DSM:

1. **Better temporal modeling**: Multi-scale dilated convolutions + transformer vs simple 3D ResNet
2. **Explicit change detection**: Frame similarity + temporal differences provide strong boundary signals
3. **Dual supervision**: Many-hot labels help with gradual transitions
4. **Modern training**: Focal loss, strong augmentation, stochastic depth
5. **Appropriate capacity**: 5M params is enough without overfitting

---

## Implementation Priority

1. **Phase 1**: Core model + basic data loading
2. **Phase 2**: Efficient video loading with decord
3. **Phase 3**: Full training pipeline with augmentation
4. **Phase 4**: Evaluation and hyperparameter tuning
5. **Phase 5**: Optimization for inference speed

---

## Next Steps

Ready to implement? The implementation will include:

1. `detection/sbd_net.py` - Main model architecture
2. `data/video_dataset.py` - Efficient video loading
3. `data/augmentations.py` - Video augmentations
4. `training/trainer.py` - Training loop with all optimizations
5. `training/losses.py` - Loss functions
6. `evaluation/evaluator.py` - ClipShots evaluation metrics
7. `train_sbdnet.py` - Main training script

Let me know when you're ready to proceed with implementation!
