"""
Hybrid Boundary Detector - AutoShot-Style Dual Head Architecture

DESIGN PHILOSOPHY:
Adopts AutoShot's proven dual-head approach:
  Head y (one_hot):  Predicts the MIDPOINT frame of each transition (primary)
  Head z (many_hot): Predicts ALL frames within gradual transitions (auxiliary)

Training: L = lambda1 * FocalBCE(y_hat, y) + lambda2 * FocalBCE(z_hat, z)
  - lambda1 = 5.0 (heavily weighted: precise boundary localization)
  - lambda2 = 0.1 (lightly weighted: gradual extent awareness)

WHY THIS BEATS THE OLD APPROACH:
1. one_hot forces SPARSE predictions (1 frame per transition) -> fewer FPs
2. many_hot teaches gradual awareness WITHOUT over-predicting boundaries
3. No separate type head -> no gradient contamination between heads
4. Type inference: high z around a y-peak -> gradual, else -> cut
5. Clean BCE loss -> no complex class weighting schemes

ARCHITECTURE (kept from proven components):
  Input [B, T, 3, H, W]
    |
  PixelMotionExtractor -> [B, T, 15] (motion before CNN)
  ColorHistogramEncoder -> [B, T, 128] (color palette shifts for graduals)
  EfficientSpatialEncoder (RGB) -> [B, T, 640] (512 attention + 128 edge)
  EfficientSpatialEncoder (Diff) -> [B, T, 640] (multi-distance diffs)
    |
  Feature fusion (1423 -> main_dim)
    |
  ProgressiveFusionBlock x N (dilated convs + attention)
    |
  DualBoundaryHead:
    head_y -> [B, T, 1] one_hot (midpoint of transitions)
    head_z -> [B, T, 1] many_hot (all gradual frames)

REMOVED (dead weight / counterproductive):
  - ReasoningModule, MotionDynamicsExtractor (acceleration/jerk not discriminative)
  - TemporalSimilarity (unused, weak discriminator)
  - MultiLevelTemporalAggregation (redundant with dilated convs)
  - TemporalPyramidPooling (redundant temporal pooling)
  - GradualTransitionDetector (covered by dilated convs at long dilations)
  - Old BoundaryDetectionHead with type coupling (gradient contamination)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ColorHistogramSimilarity(nn.Module):
    """
    AutoShot-style pairwise color histogram similarity features.

    Instead of encoding per-frame histograms as feature vectors, this computes
    the cosine similarity between frame t's histogram and every frame in a
    ±50-frame window. This gives an explicit "palette distance" signal:
      - Cut:     similarity drops to near 0 in a single step
      - Gradual: smooth decay over many frames
      - Scene:   similarity stays near 1.0

    This is the key information ColorHistogramEncoder was missing.
    Output: [B, T, 128]
    """
    def __init__(self, num_bins=64, lookup_window=101, output_dim=128):
        super().__init__()
        self.num_bins = num_bins
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1

        self.fc = nn.Sequential(
            nn.Linear(lookup_window, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def _compute_histograms(self, x):
        """
        x: [B, T, C, H, W]  values in [0, 1]
        returns: [B, T, num_bins*3] L2-normalised histograms
        """
        B, T, C, H, W = x.shape
        x_q = (x * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)  # [B,T,C,H,W]
        x_q = x_q.reshape(B * T * C, H * W)

        hist = torch.zeros(B * T * C, self.num_bins, device=x.device, dtype=torch.float32)
        ones = torch.ones_like(x_q, dtype=torch.float32)
        hist.scatter_add_(1, x_q, ones)
        hist = hist / (H * W + 1e-8)                              # normalise counts
        hist = hist.reshape(B, T, C * self.num_bins)              # [B, T, bins*3]
        hist = F.normalize(hist, p=2, dim=2)                      # L2 normalise
        return hist

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        returns: [B, T, output_dim]
        """
        B, T = x.shape[:2]
        hist = self._compute_histograms(x)                         # [B, T, D]

        # Pairwise cosine similarity: [B, T, T]
        sim = torch.bmm(hist, hist.transpose(1, 2))

        # Replicate-pad (same reason as FrameSimilarity): zero-pad makes the first
        # and last ~50 frames look like scene cuts due to cos-sim dropping to 0.
        pad = (self.lookup_window - 1) // 2
        sim_padded = F.pad(sim, (pad, pad), mode='replicate')      # [B, T, T+2*pad]

        t_idx = torch.arange(T, device=x.device)
        col_idx = (t_idx.unsqueeze(1) +
                   torch.arange(self.lookup_window, device=x.device).unsqueeze(0))  # [T, W]
        col_idx = col_idx.unsqueeze(0).expand(B, -1, -1)          # [B, T, W]
        sim_windowed = torch.gather(sim_padded, 2, col_idx)        # [B, T, W]

        return self.fc(sim_windowed)                               # [B, T, output_dim]


class FrameSimilarity(nn.Module):
    """
    AutoShot-style CNN-level frame similarity features.

    Projects CNN features to a normalised embedding space and computes
    the cosine similarity between frame t and every frame in a ±50-frame
    window.  The pattern is unambiguous:
      - Cut:     similarity to future frames collapses to ~0 at the cut frame
      - Gradual: smooth V-shaped dip over tens of frames
      - Scene:   similarity stays high

    This is the component AutoShot identifies as the dominant discriminator
    between cuts and graduals that a plain CNN cannot capture.
    Output: [B, T, output_dim]
    """
    def __init__(self, in_channels, lookup_window=101, output_dim=128):
        super().__init__()
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1

        # No learnable projection — just L2-normalise the input features directly.
        # A trainable Linear here would overfit: it learns to map training-set frames
        # into a space that looks like training labels, then fails on val.
        # Raw L2-normalised deep features already have strong inter-scene distance.
        self.fc = nn.Sequential(
            nn.Linear(lookup_window, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: [B, T, in_channels]  CNN features (spatially pooled)
        returns: [B, T, output_dim]
        """
        B, T, C = x.shape

        # L2-normalise only — no learned projection
        proj = F.normalize(x.reshape(B * T, C), p=2, dim=1)      # [B*T, C]
        proj = proj.reshape(B, T, C)                               # [B, T, C]

        # Pairwise cosine similarity: [B, T, T]
        sim = torch.bmm(proj, proj.transpose(1, 2))

        # Replicate-pad: boundary frames copy their nearest real neighbour.
        # Zero-pad creates a fake "similarity drop" at every clip edge because
        # cos-sim(real_frame, zero_vector) -> 0, which is indistinguishable from
        # a true scene cut -> systematic FP at every clip boundary during inference.
        pad = (self.lookup_window - 1) // 2
        sim_padded = F.pad(sim, (pad, pad), mode='replicate')      # [B, T, T+2*pad]

        t_idx = torch.arange(T, device=x.device)
        col_idx = (t_idx.unsqueeze(1) +
                   torch.arange(self.lookup_window, device=x.device).unsqueeze(0))  # [T, W]
        col_idx = col_idx.unsqueeze(0).expand(B, -1, -1)          # [B, T, W]
        sim_windowed = torch.gather(sim_padded, 2, col_idx)        # [B, T, W]

        return self.fc(sim_windowed)                               # [B, T, output_dim]


class EdgeDensityExtractor(nn.Module):
    """
    Extract edge density features - TOP discriminator (3.172 separation score).
    
    Graduals have HIGH edge variance (content emergence)
    vs smooth pans have LOW edge variance (just translation).
    """
    def __init__(self, in_channels=256):
        super().__init__()
        self.edge_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, groups=1)
        
    def forward(self, x):
        """
        Args:
            x: [B, 256, 6, 8] spatial CNN features
        Returns:
            [B, 64] edge density features
        """
        B, C, H, W = x.shape
        edges = torch.abs(self.edge_conv(x))
        edge_per_region = edges.reshape(B, 32, -1)
        edge_variance = edge_per_region.var(dim=2)
        edge_mean = edge_per_region.mean(dim=2)
        stats = torch.cat([edge_variance, edge_mean], dim=1)
        return stats


class EfficientSpatialEncoder(nn.Module):
    """
    CNN encoder with stacked dilated 3D stem + shared 2D body.

    Two forward paths share the same conv2/3/4 body, GAP, and edge extractor:

      forward()           — temporal path (3D stem → body)
        stem3d sees ±7 frames at pixel level → detects dissolve blending/motion blur.
        Used for: rgb_features (temporal context), diff_features.

      forward_per_frame() — 2D-only path (entry2d → body)
        NO temporal mixing: each frame processed independently.
        Used for: FrameSimilarity, feat_delta (need sharp cut signatures).
        Reuses the identical body weights — zero extra computation.

    Input:  [B, T, C, H, W]
    Output: [B, T, output_dim + 128]  (both paths)
    """
    def __init__(self, input_channels=3, output_dim=512):
        super().__init__()

        # --- Temporal path: pseudo-3D stacked-channel Conv2d stem ---
        # Strategy: spatial downsample FIRST (cheap, 4x fewer activations), then
        # stack [x[t-d], x[t], x[t+d]] as 3C channels, use 1x1 Conv2d for temporal
        # mixing (no spatial processing needed — body handles that). This is ~36x
        # fewer FLOPs than the naive (3C, 3x3) @ full resolution approach.
        # RF over time: 1 + 2*1 + 2*2 + 2*4 = 15 frames (±7 temporal receptive field).
        # DO NOT use for FrameSimilarity/feat_delta — temporal blending blurs cut spikes.
        self.stem_spatial = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2 — do spatial reduction BEFORE temporal stack
        )
        self.stem_t1 = nn.Sequential(
            nn.Conv2d(16 * 3, 16, kernel_size=1, bias=False),  # 1x1: pure temporal mixing
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.stem_t2 = nn.Sequential(
            nn.Conv2d(16 * 3, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.stem_t4 = nn.Sequential(
            nn.Conv2d(16 * 3, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # additional H/4, W/4 — matches body input stride
        )

        # --- Shared 2D body (used by BOTH paths) ---
        # Running conv2/3/4 once via the temporal path gives rgb_features.
        # forward_per_frame() reuses these same weights — no duplicate computation.
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_proj = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.edge_extractor = EdgeDensityExtractor(in_channels=256)
        self.edge_proj = nn.Sequential(
            nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU()
        )

    def forward_per_frame(self, x):
        """
        2D-only path: NO temporal mixing. Each frame is processed independently.
        Reuses the same conv2/3/4 / gap / edge body weights — zero extra parameters.

        NOTE: This DOES run the body a second time (separate from forward()).
        That is intentional: forward() produces 3D-blended features (best for gradual
        detection), forward_per_frame() produces sharp features (required for
        FrameSimilarity and feat_delta, where temporal blending destroys the signal).
        The compute overhead is ~50% of the rgb_encoder cost.

        Input:  [B, T, C, H, W]
        Output: [B, T, output_dim + 128]  (same 640 dims as forward())
        """
        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W)
        x2d = self.stem_spatial(x_bt)    # [B*T, 16, H/2, W/2]
        x2d = F.max_pool2d(x2d, 2)      # [B*T, 16, H/4, W/4] — matches stem_t4's MaxPool2d
        return self._body(x2d).reshape(B, T, -1)

    def _temporal_stack(self, x_bt, T, dilation):
        """Stack [x[t-d], x[t], x[t+d]] as channels. Boundary frames get zeros."""
        B_T, C, H, W = x_bt.shape
        x = x_bt.reshape(-1, T, C, H, W)        # [B, T, C, H, W]
        B = x.shape[0]
        d = dilation
        # Zero-pad boundaries — reflect padding creates false gradual signals at clip edges
        prev = torch.cat([torch.zeros_like(x[:, :d]), x[:, :-d]], dim=1) if d < T else torch.zeros_like(x)
        next_ = torch.cat([x[:, d:], torch.zeros_like(x[:, -d:])], dim=1) if d < T else torch.zeros_like(x)
        stacked = torch.cat([prev, x, next_], dim=2)   # [B, T, 3C, H, W]
        return stacked.reshape(B * T, -1, H, W)         # [B*T, 3C, H, W]

    def _body(self, x2d):
        """Shared 2D body: [B*T, 16, H/4, W/4] → [B*T, output_dim+128]"""
        x2d = self.conv2(x2d)
        x2d = self.conv3(x2d)
        x2d = self.conv4(x2d)
        pooled = self.gap_proj(self.gap(x2d))
        edge   = self.edge_proj(self.edge_extractor(x2d))
        return torch.cat([pooled, edge], dim=-1)

    def forward(self, x):
        """
        Temporal path (pseudo-3D stem → shared body). Use for rgb_features, diff_features.
        Input:  [B, T, C, H, W]
        Output: [B, T, output_dim + 128]  = [B, T, 640]
        """
        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W)

        # Spatial reduction first (cheap), then pseudo-3D temporal mixing at H/2 resolution
        s0 = self.stem_spatial(x_bt)                                   # [B*T, 16, H/2, W/2]
        s1 = self.stem_t1(self._temporal_stack(s0,  T, dilation=1))   # [B*T, 16, H/2, W/2]
        s2 = self.stem_t2(self._temporal_stack(s1,  T, dilation=2))   # [B*T, 16, H/2, W/2]
        x2d = self.stem_t4(self._temporal_stack(s2, T, dilation=4))   # [B*T, 16, H/4, W/4]
        return self._body(x2d).reshape(B, T, -1)



class MultiScaleDilatedConv(nn.Module):
    """
    Multi-scale dilated temporal convolutions - TransNetV2's core mechanism.
    
    Captures patterns at multiple temporal scales WITHOUT pooling:
    - Dilation 1:  Instant cuts (1 frame)
    - Dilation 4:  Quick fades (4-7 frames)  
    - Dilation 16: Medium graduals (16-31 frames)
    - Dilation 64: Long graduals (64-127 frames)
    
    4 scales cover the full 1-127 frame range with exponential growth.
    """
    def __init__(self, dim, dilation_rates=[1, 4, 16, 64]):
        super().__init__()
        
        self.dilation_rates = dilation_rates
        
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(
                dim, dim,
                kernel_size=3,
                dilation=rate,
                padding=rate,
                groups=dim // 4
            )
            for rate in dilation_rates
        ])
        
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(dim) for _ in dilation_rates
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv1d(dim * len(dilation_rates), dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),  # 3.15M params with no regularization was primary overfit source
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C] temporal features
        Returns:
            [B, T, C] multi-scale temporal features
        """
        x_t = x.transpose(1, 2)
        
        scale_features = []
        for conv, norm in zip(self.dilated_convs, self.norms):
            feat = conv(x_t)
            feat = norm(feat)
            feat = F.relu(feat, inplace=True)
            scale_features.append(feat)
        
        multi_scale = torch.cat(scale_features, dim=1)
        fused = self.fusion(multi_scale)
        output = fused + x_t
        
        return output.transpose(1, 2)


class LightweightTemporalAttention(nn.Module):
    """
    Efficient temporal self-attention for global sequence context.
    
    Pre-norm transformer style with residual connection.
    Attention as refinement, not primary path.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(0.15)
        self.proj_drop = nn.Dropout(0.15)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C] temporal features
        Returns:
            [B, T, C] attention-refined features
        """
        B, T, C = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return x + out


class PixelMotionExtractor(nn.Module):
    """
    Extract motion features from RAW PIXELS before CNN encoding.
    
    Features at 5 temporal distances [1, 2, 4, 8, 16]:
    - Magnitude statistics (mean, std)
    - Direction coherence (uniform = pan, non-uniform = content change)
    
    Output: [B, T, 15] (5 distances x 3 stats)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] raw video frames
        Returns:
            motion_features: [B, T, 15]
        """
        B, T, C, H, W = x.shape

        distances = [1, 2, 4, 8, 16]
        all_stats = []

        for dist in distances:
            if dist < T:
                # Backward difference: diff[t] = x[t] - x[t-dist]
                # Fires AT the cut frame (high magnitude exactly at frame t=cut),
                # not dist frames before it.
                diff = x[:, dist:] - x[:, :-dist]   # [B, T-dist, C, H, W]
                diff_gray = (0.299 * diff[:, :, 0]
                           + 0.587 * diff[:, :, 1]
                           + 0.114 * diff[:, :, 2])  # [B, T-dist, H, W]

                magnitude = diff_gray.abs()
                mag_mean = magnitude.mean(dim=[2, 3])   # [B, T-dist]
                mag_std  = magnitude.std(dim=[2, 3])    # [B, T-dist]

                # Spatial uniformity: fraction of pixels with near-zero motion.
                # High uniformity = pan/zoom (content unchanged).
                # Low uniformity  = scene change (spatially irregular diff).
                threshold = mag_mean.unsqueeze(-1).unsqueeze(-1) * 0.5
                # +1e-6: prevents threshold=0 on static frames where mag_mean=0.
                # Without it: 0 < 0 is always False → uniformity=0 for static scenes
                # (inverted signal — a perfectly static scene should score uniformity=1.0).
                threshold = threshold + 1e-6
                uniformity = (magnitude < threshold).float().mean(dim=[2, 3])  # [B, T-dist]

                stats = torch.stack([mag_mean, mag_std, uniformity], dim=-1)  # [B, T-dist, 3]
                # Zero-pad at the START: frames 0..dist-1 have no prior context
                stats = F.pad(stats, (0, 0, dist, 0), value=0)   # [B, T, 3]
                all_stats.append(stats)
            else:
                all_stats.append(torch.zeros(B, T, 3, device=x.device))
        
        motion_features = torch.cat(all_stats, dim=-1)
        return motion_features


class ProgressiveFusionBlock(nn.Module):
    """
    Progressive fusion with local + global patterns.
    
    1. Local patterns (dilated convs) - primary signal, covers 1-127 frame range
    2. Global context (attention) - refinement
    3. Gated fusion - learn what to keep from each
    """
    def __init__(self, dim):
        super().__init__()
        
        self.local_conv = MultiScaleDilatedConv(dim)
        self.global_attn = LightweightTemporalAttention(dim)
        
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )

        # fusion now takes dim input (the gated-weighted blend), not dim*2
        self.fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C] temporal features
        Returns:
            [B, T, C] fused local + global features
        """
        local_feat  = self.local_conv(x)
        global_feat = self.global_attn(local_feat)  # attention refines local conv output, not raw x

        combined     = torch.cat([local_feat, global_feat], dim=-1)  # [B, T, dim*2]
        gate_weight  = self.gate(combined)                            # [B, T, dim] in (0,1)
        # Soft gate: weight local vs global rather than adding both unconditionally.
        # gate≈1 → trust local (dilated conv); gate≈0 → trust global (attention).
        weighted = gate_weight * local_feat + (1 - gate_weight) * global_feat
        return x + self.fusion(weighted)  # residual connection around the whole block


class DualBoundaryHead(nn.Module):
    """
    AutoShot-style dual-head boundary detection.
    
    Head y (one_hot):  Predicts whether each frame is the MIDPOINT of a transition.
                       Cuts: midpoint = the cut frame itself.
                       Graduals: midpoint = center of the gradual span.
                       
    Head z (many_hot): Predicts whether each frame is part of a gradual transition.
                       Cuts: all zeros (instantaneous, no span).
                       Graduals: 1 for every frame in the gradual span.
    
    Training:
        L = lambda1 * FocalBCE(y_hat, y_one_hot) + lambda2 * FocalBCE(z_hat, z_many_hot)
        lambda1 = 5.0 (primary), lambda2 = 0.1 (auxiliary)
    
    Inference:
        1. Detect peaks in sigmoid(y_logits) > threshold -> transition locations
        2. For each peak, check sigmoid(z_logits) around it:
           - High z -> gradual transition
           - Low z  -> hard cut
    
    WHY THIS WORKS:
    - one_hot forces exactly 1 prediction per transition -> inherent sparsity
    - many_hot captures gradual extent without contributing to boundary count
    - No gradient contamination between detection and classification
    - Natural type inference from the relationship between y and z
    """
    def __init__(self, dim):
        super().__init__()
        
        # SEPARATE refine blocks per head -- y and z have incompatible target densities:
        #   y-head (one_hot):  ~0.8% positive (1 midpoint per 128-frame clip)
        #   z-head (many_hot): ~0-20% positive (0 for cut clips, 10-20 frames for graduals)
        # A shared refine block averages contradictory gradients, causing y-head to fire
        # on every gradual frame (many_hot leakage) -> FP burst on all gradual transitions.
        self.refine_y = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.refine_z = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Head y: one_hot midpoint (PRIMARY)
        # Prior ~0.8% positive (1 transition per 128-frame clip).
        # Init bias = log(p/(1-p)) = log(0.008/0.992) ~= -4.82 so sigmoid(logit) ~= 0.008
        # at init. Without this, bias~0 -> sigmoid~0.5 -> ~64 FPs/clip before any training.
        # This is the RetinaNet focal-loss bias initialisation trick.
        self.head_y = nn.Linear(dim // 2, 1)
        nn.init.constant_(self.head_y.bias, -4.82)   # RetinaNet prior: sigmoid(-4.82) ~= 0.008

        # Head z: many_hot gradual extent (AUXILIARY)
        # Prior ~2.1% positive: 79% cut-clips have all-zero z-targets;
        # gradual-clips average ~13 positive frames / 128 total.
        # bias = log(0.021/0.979) ~= -3.85
        self.head_z = nn.Linear(dim // 2, 1)
        nn.init.constant_(self.head_z.bias, -3.85)   # RetinaNet prior: sigmoid(-3.85) ~= 0.021

    def forward(self, x):
        """
        Args:
            x: [B, T, C] temporal features from fusion blocks
        Returns:
            y_logits: [B, T, 1] - one_hot midpoint predictions
            z_logits: [B, T, 1] - many_hot gradual extent predictions
        """
        y_logits = self.head_y(self.refine_y(x))
        z_logits = self.head_z(self.refine_z(x))
        return y_logits, z_logits


class HybridBoundaryDetector(nn.Module):
    """
    Hybrid Boundary Detector with AutoShot-Style Dual Heads.
    
    Combines proven feature extraction with AutoShot's dual-head training:
    - Dual spatial encoders (RGB + multi-distance diffs) with edge extraction
    - Pixel-level motion features (before CNN encoding)
    - Color histogram features (gradual detection via palette shifts)
    - Progressive fusion blocks (dilated convs + attention)
    - Dual boundary heads (one_hot midpoint + many_hot extent)
    
    Feature pipeline (proportional projection → no gradient dilution or over-compression):
      RGB 640 → main_dim//2 | Diff 640 → main_dim//2 | FrameSim 128 → main_dim//4
      ColorHistSim 128 → main_dim//4 | Motion 15 → 32
      concat → 2-step MLP → main_dim
    """
    def __init__(self, main_dim=512, num_fusion_blocks=3):
        super().__init__()
        
        # Pixel-level motion extractor (BEFORE CNN encoding)
        self.pixel_motion_extractor = PixelMotionExtractor()

        # Dual spatial encoders: 640 dims each (output_dim=512 + 128 edge)
        # The 3D stem blends ±7 temporal neighbours at pixel level.
        # Use rgb_encoder.forward_per_frame() for FrameSimilarity and feat_delta.
        self.rgb_encoder  = EfficientSpatialEncoder(input_channels=3,  output_dim=512)
        self.diff_encoder = EfficientSpatialEncoder(input_channels=15, output_dim=512)
        # Note: per_frame_features reuse rgb_encoder's body via forward_per_frame().
        # No separate encoder needed — zero duplicate conv computation.

        # AutoShot-style similarity features (the dominant discriminators)
        # Three temporal scales give explicit short / medium / long-range context:
        #   window  15 — cuts:      similarity collapses in a single step
        #   window  51 — quick fades (10-25 frames): partial overlap visible
        #   window 127 — slow dissolves (50-100 frames): broad V-shaped dip
        # Each outputs 64-d; concat → 192-d → projected to sim_dim in frame_sim_proj.
        # A single 127-frame window cannot distinguish a 1-frame cut from a 100-frame
        # dissolve — both look like a dip, just at different widths.
        self.frame_sim_short  = FrameSimilarity(
            in_channels=640, lookup_window=15,  output_dim=64
        )
        self.frame_sim_medium = FrameSimilarity(
            in_channels=640, lookup_window=51,  output_dim=64
        )
        self.frame_sim_long   = FrameSimilarity(
            in_channels=640, lookup_window=127, output_dim=64
        )
        # ColorHistogramSimilarity: pairwise palette distance -> [B,T,128]
        self.color_hist_similarity = ColorHistogramSimilarity(
            num_bins=64, lookup_window=127, output_dim=128
        )

        # Feature-space dropout applied to rgb_features BEFORE FrameSimilarity.
        # Spatial augmentations (brightness/contrast/noise) are invariant to L2-norm —
        # they change feature magnitude, which cosine similarity ignores entirely.
        # Dropout changes the feature DIRECTION after L2-norm, giving a different
        # [T,T] similarity matrix each forward pass for the same clip.
        # This is the only effective augmentation against FrameSimilarity memorization.
        self.feat_dropout = nn.Dropout(p=0.15)

        # Per-stream projections: proportional to information content.
        #   RGB   640 → main_dim//2  (primary, most info-rich — 3D stem features)
        #   Diff  640 → main_dim//2  (primary — 3D stem on multi-dist diffs)
        #   Feat delta 640 → 64      (2D per-frame features — sharp cut spike)
        #   FSim  128 → main_dim//4  (key discriminator — 2D per-frame cosine sims)
        #   CSim  128 → main_dim//4  (key discriminator — palette sims)
        #   Motion 15               (raw — already compact, no compression needed)
        rgb_dim    = main_dim // 2   # 256 for main_dim=512
        diff_dim   = main_dim // 2   # 256
        delta_dim  = 64              # feature-level delta: fixed small dim
        sim_dim    = main_dim // 4   # 128
        motion_out = 15              # pass raw motion features — no projection loss
        fuse_in    = rgb_dim + diff_dim + delta_dim + sim_dim + sim_dim + motion_out
        # = 256 + 256 + 64 + 128 + 128 + 15 = 847 for main_dim=512

        self.rgb_proj = nn.Sequential(
            nn.Linear(640, rgb_dim), nn.LayerNorm(rgb_dim), nn.GELU()
        )
        self.diff_proj = nn.Sequential(
            nn.Linear(640, diff_dim), nn.LayerNorm(diff_dim), nn.GELU()
        )
        # Feature-level temporal delta on 2D per-frame features (sharp — no 3D blur).
        self.feat_delta_proj = nn.Sequential(
            nn.Linear(640, delta_dim), nn.LayerNorm(delta_dim), nn.GELU()
        )
        self.frame_sim_proj = nn.Sequential(
            nn.Linear(192, sim_dim), nn.LayerNorm(sim_dim), nn.GELU()  # 3×64 → sim_dim
        )
        self.color_sim_proj = nn.Sequential(
            nn.Linear(128, sim_dim), nn.LayerNorm(sim_dim), nn.GELU()
        )
        # No motion_proj: PixelMotionExtractor's 15 features are already compact
        # hard-engineered statistics — compressing to 10 only discards information.
        # BUT they are raw pixel-scale values (mag_mean ~0.01-0.05, uniformity 0-1),
        # i.e. 10-100x smaller than LayerNorm-normalized streams.
        # A shared Linear(847->1024) cannot weight these fairly without scaling them.
        # LayerNorm (no affine compression) re-centres + unit-std without losing the
        # relative magnitudes between motion dimensions.
        self.motion_norm = nn.LayerNorm(motion_out)

        # Two-step mixer: fuse_in → 2×main_dim → main_dim
        # Expansion step lets streams interact before final compression.
        self.feature_proj = nn.Sequential(
            nn.Linear(fuse_in, main_dim * 2),
            nn.LayerNorm(main_dim * 2),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(main_dim * 2, main_dim),
            nn.LayerNorm(main_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        
        # Progressive fusion blocks (dilated convs + attention)
        self.fusion_blocks = nn.ModuleList([
            ProgressiveFusionBlock(main_dim)
            for _ in range(num_fusion_blocks)
        ])
        
        # Dual boundary head (AutoShot-style)
        self.detection_head = DualBoundaryHead(main_dim)
        
        self.main_dim = main_dim
        
    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, T, C, H, W] video frames (typically T=128, C=3, H=48, W=64)
            return_features: If True, return intermediate features dict
            
        Returns:
            y_logits: [B, T, 1] - one_hot midpoint predictions
            z_logits: [B, T, 1] - many_hot gradual extent predictions
        """
        B, T, C, H, W = x.shape
        
        # 1. Extract pixel-level motion features BEFORE CNN encoding
        pixel_motion_features = self.pixel_motion_extractor(x)  # [B, T, 15]
        
        # 2. Compute multi-distance differences as stacked channels
        # [1,4,8,16,32]: distance 32 explicitly encodes mid-dissolve blend ratios.
        diff_distances = [1, 4, 8, 16, 32]
        all_diffs = torch.zeros(B, T, C * len(diff_distances), H, W, device=x.device, dtype=x.dtype)
        for i, dist in enumerate(diff_distances):
            if dist < T:
                all_diffs[:, dist:, i*C:(i+1)*C] = x[:, dist:] - x[:, :-dist]
        
        # 3. Spatial encoding
        # Two paths through EfficientSpatialEncoder — same body weights, different stems:
        #   forward()           — 3D stem (±7 frame RF): blends temporal context.
        #                         Good for rgb_features, diff_features (gradual  detection).
        #   forward_per_frame() — 2D only (no temporal mix): preserves sharp per-frame.
        #                         Required for FrameSimilarity + feat_delta so cut spikes
        #                         are not smeared across neighbours before cosine sim.
        rgb_features      = self.rgb_encoder(x)               # [B, T, 640] — 3D stem
        diff_features     = self.diff_encoder(all_diffs)       # [B, T, 640] — 3D stem on diffs
        per_frame_feats   = self.rgb_encoder.forward_per_frame(x)  # [B, T, 640] — 2D only, sharp

        # 4. Feature-level temporal delta on SHARP per-frame features.
        # Backward difference: feat_delta[t] = pf[t] - pf[t-1].
        # Fires AT the cut frame; using 3D features would spread the spike over ±7 frames.
        feat_delta = torch.zeros_like(per_frame_feats)             # [B, T, 640]
        feat_delta[:, 1:] = per_frame_feats[:, 1:] - per_frame_feats[:, :-1]
        # Normalize by per-clip mean absolute delta (a single scalar per clip).
        # This preserves the temporal contrast: cut frames have large ||delta||,
        # scene frames have small ||delta|| — the ratio is the discriminative signal.
        # Per-dimension temporal std (previous approach) destroyed this: on a
        # scene-only clip all deltas are noise -> std is tiny -> noise gets amplified
        # to the same O(1) magnitude as a real cut frame after division.
        clip_scale = feat_delta.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)  # [B,1,1]
        feat_delta = feat_delta / clip_scale

        # 5. AutoShot-style similarity features — use SHARP per-frame features.
        # Using 3D features here would pre-smooth the [T,T] similarity matrix and mask
        # the single-frame dip that distinguishes cuts from graduals.
        # Three scales: short=cuts, medium=quick fades, long=slow dissolves.
        rgb_for_sim             = self.feat_dropout(per_frame_feats)           # direction noise
        frame_sim_features      = torch.cat([
            self.frame_sim_short(rgb_for_sim),   # [B, T, 64]  — cut-scale
            self.frame_sim_medium(rgb_for_sim),  # [B, T, 64]  — quick-fade scale
            self.frame_sim_long(rgb_for_sim),    # [B, T, 64]  — dissolve scale
        ], dim=-1)                               # [B, T, 192]
        color_hist_sim_features = self.color_hist_similarity(x)                # [B, T, 128]

        # 6. Per-stream projection (proportional gradient budget)
        rgb_proj   = self.rgb_proj(rgb_features)
        diff_proj  = self.diff_proj(diff_features)
        delta_proj = self.feat_delta_proj(feat_delta)
        fsim_proj  = self.frame_sim_proj(frame_sim_features)
        csim_proj  = self.color_sim_proj(color_hist_sim_features)
        mot_proj   = self.motion_norm(pixel_motion_features)   # scale to O(1), same as other streams

        combined = torch.cat([rgb_proj, diff_proj, delta_proj, fsim_proj, csim_proj, mot_proj], dim=-1)
        x_proj = self.feature_proj(combined)  # [B, T, main_dim]
        
        # 7. Progressive fusion
        for fusion_block in self.fusion_blocks:
            x_proj = fusion_block(x_proj)
        
        # 8. Dual head detection
        y_logits, z_logits = self.detection_head(x_proj)
        
        if return_features:
            return y_logits, z_logits, {
                'rgb': rgb_features,
                'diff': diff_features,
                'pixel_motion': pixel_motion_features,
                'frame_sim': frame_sim_features,
                'color_hist_sim': color_hist_sim_features,
                'fused': x_proj
            }
        
        return y_logits, z_logits
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_hybrid_detector(main_dim=512, num_fusion_blocks=3):
    """
    Factory function for creating Hybrid Boundary Detector.
    
    Args:
        main_dim: Main processing dimension (default 512)
        num_fusion_blocks: Number of progressive fusion blocks (default 3)
    
    Returns:
        HybridBoundaryDetector model
    """
    print("=" * 80)
    print("Creating Hybrid Boundary Detector - AutoShot-Style Dual Heads")
    print("=" * 80)
    
    print("\nDUAL HEAD DESIGN (AutoShot approach):")
    print("  Head y (one_hot):  Midpoint of each transition (primary, lambda=5.0)")
    print("  Head z (many_hot): All frames in gradual transitions (auxiliary, lambda=0.1)")
    print("  Loss: L = 5.0 * FocalBCE(y, one_hot) + 0.1 * FocalBCE(z, many_hot)")
    
    print("\nCORE COMPONENTS (proven effective):")
    print("  - Pixel-level motion features (before CNN encoding)")
    print("  - FrameSimilarity: CNN-level cosine sim in ±63 frame window (AutoShot key)")
    print("  - ColorHistogramSimilarity: palette cosine sim in ±63 frame window (AutoShot key)")
    print("  - Dual spatial encoders (RGB + multi-distance diffs)")
    print("  - Edge density extractor (3.172 separation score)")
    print("  - Multi-scale dilated convs (dilations 1, 4, 16, 64)")
    print("  - Lightweight temporal attention (global context)")
    print("  - Progressive fusion blocks (gated local + global)")
    
    print("\nREMOVED (dead weight):")
    print("  - ReasoningModule + MotionDynamicsExtractor (accel/jerk not discriminative)")
    print("  - TemporalSimilarity, MultiLevelTemporalAggregation (redundant)")
    print("  - TemporalPyramidPooling, GradualTransitionDetector (covered by dilated convs)")
    print("  - Old type head with gradient contamination")
    
    model = HybridBoundaryDetector(
        main_dim=main_dim,
        num_fusion_blocks=num_fusion_blocks
    )
    
    total_params, trainable_params = model.count_parameters()
    
    print(f"\nMODEL STATISTICS:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory footprint: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    print(f"  Main processing dim: {main_dim}")
    print(f"  Fusion blocks: {num_fusion_blocks}")
    
    print(f"\nTARGET: Beat AutoShot F1~0.85 on ClipShots")
    print("=" * 80)
    
    return model


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("TESTING HYBRID BOUNDARY DETECTOR (Dual Head)")
    print("=" * 80)
    
    model = create_hybrid_detector(main_dim=256, num_fusion_blocks=3)
    
    print("\nTesting forward pass...")
    batch = torch.randn(2, 128, 3, 48, 64)
    print(f"Input shape: {batch.shape}")
    
    with torch.no_grad():
        y_logits, z_logits = model(batch)
    
    print(f"\nOutput shapes:")
    print(f"  y_logits (one_hot):  {y_logits.shape} (expected: [2, 128, 1])")
    print(f"  z_logits (many_hot): {z_logits.shape} (expected: [2, 128, 1])")
    
    y_probs = torch.sigmoid(y_logits)
    z_probs = torch.sigmoid(z_logits)
    
    print(f"\nPrediction analysis:")
    print(f"  y_probs range: [{y_probs.min():.4f}, {y_probs.max():.4f}]")
    print(f"  z_probs range: [{z_probs.min():.4f}, {z_probs.max():.4f}]")
    
    for threshold in [0.5, 0.6, 0.7]:
        n_boundaries = (y_probs.squeeze(-1) > threshold).sum().item()
        total_frames = batch.shape[0] * batch.shape[1]
        sparsity = 100 * (1 - n_boundaries / total_frames)
        print(f"  Transitions (y>{threshold}): {n_boundaries}/{total_frames} ({sparsity:.1f}% non-transition)")
    
    # Test return_features mode
    with torch.no_grad():
        y_logits, z_logits, features = model(batch, return_features=True)
    print(f"\nFeature shapes:")
    for k, v in features.items():
        print(f"  {k}: {v.shape}")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
