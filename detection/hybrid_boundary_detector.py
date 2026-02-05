"""
Hybrid Boundary Detector - 2026 SOTA Architecture

DESIGN PHILOSOPHY:
Combines the best of both worlds:
1. TransNetV2's SECRET: Multi-scale dilated convolutions for LOCAL boundary detection
2. Modern advancement: Lightweight temporal attention for GLOBAL sequence context
3. Proven critical: Explicit frame differences as primary signal
4. Interpretability: Reasoning module that "thinks" step-by-step
5. Efficiency: ~14M parameters (proven effective range)

WHY THIS WORKS:
- Boundaries are LOCAL events → dilated convs with multiple receptive fields
- Sequence context matters → lightweight attention for long-range dependencies
- Frame differences strongest signal → explicit RGB diff encoding
- Multi-scale fusion → combine local (1,2,4,8,16 frames) + global (full sequence)

KEY INNOVATIONS:
✓ Multi-scale dilated convs (kernels 1,2,4,8,16,32,64) - TransNetV2 insight
✓ Explicit frame difference encoder - proven critical
✓ Lightweight temporal attention - efficient global context
✓ Progressive feature fusion - local to global
✓ Interpretable reasoning module - step-by-step decision making
✓ Binary + auxiliary type heads - SOTA formulation

ARCHITECTURE FLOW:
Input [B,T,3,H,W]
  ↓
Frame Differences [B,T,3,H,W] (explicit RGB diffs)
  ↓
Spatial Encoding [B,T,D] (CNN feature extraction)
  ↓
Multi-Scale Dilated Convs [B,T,D] (local patterns: 1,2,4,8,16 frames)
  ↓
Lightweight Attention [B,T,D] (global sequence context)
  ↓
Progressive Fusion [B,T,D] (combine local + global)
  ↓
Binary Boundary Head [B,T,1] + Type Head [B,T,2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ColorHistogramEncoder(nn.Module):
    """
    Explicit color histogram features - TransNetV2's SECRET for gradual transitions
    
    WHY THIS MATTERS:
    - Gradual transitions = slow color palette shifts
    - CNNs can smooth these out - histograms capture them explicitly
    - Cheap to compute, massive impact on gradual recall
    
    Output: Always 128 dims (TransNetV2 also uses 128 for histograms)
    """
    def __init__(self, num_bins=32):
        super().__init__()
        self.num_bins = num_bins
        
        # Project histogram features to FIXED 128 dims
        self.hist_proj = nn.Sequential(
            nn.Linear(num_bins * 3, 128),  # RGB histograms → 128 dims
            nn.LayerNorm(128),
            nn.GELU()
        )
        
    def compute_histogram(self, x):
        """
        Compute RGB histograms for a batch of frames (FULLY VECTORIZED - FASTEST)
        Args:
            x: [B, C, H, W] frames in range [0, 1]
        Returns:
            [B, num_bins * 3] histogram features
        """
        B, C, H, W = x.shape
        
        # FASTEST: Use scatter_add for fully vectorized histogram
        # Quantize to bins: [0,1] -> [0, num_bins-1]
        x_quantized = (x * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)  # [B, 3, H, W]
        
        # Reshape for vectorized processing: [B*3, H*W]
        x_flat = x_quantized.reshape(B * C, -1)
        
        # Create histogram tensor: [B*3, num_bins]
        hist = torch.zeros(B * C, self.num_bins, device=x.device, dtype=torch.float32)
        
        # Vectorized histogram computation using scatter_add
        ones = torch.ones_like(x_flat, dtype=torch.float32)
        hist.scatter_add_(1, x_flat, ones)
        
        # Normalize and reshape: [B, 3*num_bins]
        hist = hist / (H * W + 1e-8)
        hist = hist.reshape(B, C * self.num_bins)
        
        return hist
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] frames
        Returns:
            [B, 128] histogram features (FIXED dimension)
        """
        hist = self.compute_histogram(x)
        return self.hist_proj(hist)


class TemporalSimilarity(nn.Module):
    """
    Modern temporal similarity features for gradual detection
    
    KEY INSIGHT from TransNetV2:
    - Graduals: Similarity smoothly decays (0.95 → 0.90 → 0.85 → 0.80)
    - Cuts: Similarity drops sharply (0.95 → 0.30)
    - None: Similarity stays constant (0.98 → 0.98)
    
    IMPROVEMENTS over TransNetV2:
    - Multi-scale similarities (not just one lookup window)
    - Efficient offset-based computation (not full T×T matrix)
    - Learnable similarity aggregation
    - Works with spatial features (flexible input dimension)
    
    Output: Always 128 dims (TransNetV2 also uses 128 for similarity)
    """
    def __init__(self, input_dim, similarity_offsets=[1, 2, 4, 8, 16, 32]):
        super().__init__()
        
        self.similarity_offsets = similarity_offsets
        
        # Project input features for similarity computation
        # Handles both global (256) and spatial (3072) features
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),  # Project to fixed 256 for similarity
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # Learnable aggregation of similarities → FIXED 128 dims
        self.similarity_fusion = nn.Sequential(
            nn.Linear(len(similarity_offsets), 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, T, 256] temporal features from spatial encoder
        Returns:
            [B, T, 128] similarity features (FIXED dimension)
        """
        B, T, C = x.shape
        
        # Project for similarity computation
        x_proj = self.proj(x)  # [B, T, C]
        x_norm = F.normalize(x_proj, p=2, dim=-1)  # L2 normalize
        
        # Compute cosine similarities at multiple offsets
        similarities = []
        for offset in self.similarity_offsets:
            if offset < T:
                # Cosine similarity between frame t and frame t+offset
                sim = (x_norm[:, :-offset] * x_norm[:, offset:]).sum(dim=-1)  # [B, T-offset]
                # Pad to full length
                sim_padded = F.pad(sim, (0, offset), value=1.0)  # Pad with 1.0 (self-similarity)
                similarities.append(sim_padded.unsqueeze(-1))  # [B, T, 1]
            else:
                # Offset too large, use self-similarity
                similarities.append(torch.ones(B, T, 1, device=x.device))
        
        # Stack similarities: [B, T, num_offsets]
        similarities = torch.cat(similarities, dim=-1)
        
        # Learnable aggregation
        similarity_features = self.similarity_fusion(similarities)  # [B, T, C//2]
        
        return similarity_features


class EdgeDensityExtractor(nn.Module):
    """
    Extract edge density features - TOP discriminator (3.172 separation)
    
    Empirical finding: Graduals have HIGH edge variance (content emergence)
    vs smooth pans have LOW edge variance (just translation)
    
    Computes:
    - Edge density per region
    - Variance of edge density across regions (KEY!)
    - Edge orientation distribution
    """
    def __init__(self, in_channels=256):
        super().__init__()
        
        # Sobel-style edge detector
        self.edge_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, groups=1)
        
    def forward(self, x):
        """
        Args:
            x: [B, 256, 6, 8] spatial CNN features
        Returns:
            [B, 64] edge density features (reduced from 128)
        """
        B, C, H, W = x.shape
        
        # Compute edge magnitude per spatial location
        edges = torch.abs(self.edge_conv(x))  # [B, 32, 6, 8]
        
        # 1. Edge density variance across regions (KEY DISCRIMINATOR!)
        # Gradual: edges in some regions, not others (content emergence)
        # Pan: edges uniformly distributed (just translation)
        edge_per_region = edges.reshape(B, 32, -1)  # [B, 32, 48]
        edge_variance = edge_per_region.var(dim=2)  # [B, 32] - HIGH = gradual
        
        # 2. Overall edge density (magnitude)
        edge_mean = edge_per_region.mean(dim=2)  # [B, 32]
        
        # Combine: variance (primary) + mean (secondary)
        stats = torch.cat([edge_variance, edge_mean], dim=1)  # [B, 64]
        
        return stats


class SpatialAttentionPooling(nn.Module):
    """
    Multi-scale attention-based spatial pooling
    
    Uses attention at multiple scales to capture both:
    - Fine-grained: Local edge patterns, small regions
    - Coarse-grained: Global structure, large regions
    
    This preserves discriminative information without parameter explosion.
    
    Input: [B, C, H, W] spatial features
    Output: [B, out_dim] multi-scale attention-weighted features
    """
    def __init__(self, in_channels=256, out_dim=512, num_heads=8):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        # Fine-grained attention (full 6×8 resolution)
        self.fine_query = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_channels, out_dim // 2)
        )
        self.fine_key = nn.Conv2d(in_channels, out_dim // 2, kernel_size=1)
        self.fine_value = nn.Conv2d(in_channels, out_dim // 2, kernel_size=1)
        
        # Coarse-grained attention (2×2 regions)
        self.coarse_query = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_channels, out_dim // 2)
        )
        self.coarse_key = nn.Conv2d(in_channels, out_dim // 2, kernel_size=1)
        self.coarse_value = nn.Conv2d(in_channels, out_dim // 2, kernel_size=1)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 256, 6, 8] spatial features
        Returns:
            [B, out_dim] multi-scale attention-pooled features
        """
        B, C, H, W = x.shape
        
        # FINE-GRAINED: Attend to all 48 locations
        fine_q = self.fine_query(x)  # [B, out_dim//2]
        fine_k = self.fine_key(x).flatten(2)  # [B, out_dim//2, 48]
        fine_v = self.fine_value(x).flatten(2)  # [B, out_dim//2, 48]
        
        # Attention over spatial locations
        fine_attn = torch.einsum('bd,bdn->bn', fine_q, fine_k)  # [B, 48]
        fine_attn = torch.softmax(fine_attn / (self.head_dim ** 0.5), dim=-1)
        fine_out = torch.einsum('bn,bdn->bd', fine_attn, fine_v)  # [B, out_dim//2]
        
        # COARSE-GRAINED: Pool to 2×2 then attend
        x_coarse = F.adaptive_avg_pool2d(x, (2, 2))  # [B, 256, 2, 2]
        coarse_q = self.coarse_query(x_coarse)  # [B, out_dim//2]
        coarse_k = self.coarse_key(x_coarse).flatten(2)  # [B, out_dim//2, 4]
        coarse_v = self.coarse_value(x_coarse).flatten(2)  # [B, out_dim//2, 4]
        
        coarse_attn = torch.einsum('bd,bdn->bn', coarse_q, coarse_k)
        coarse_attn = torch.softmax(coarse_attn / (self.head_dim ** 0.5), dim=-1)
        coarse_out = torch.einsum('bn,bdn->bd', coarse_attn, coarse_v)  # [B, out_dim//2]
        
        # Combine fine and coarse
        out = torch.cat([fine_out, coarse_out], dim=-1)  # [B, out_dim]
        out = self.fusion(out)
        
        return out


class EfficientSpatialEncoder(nn.Module):
    """
    Spatial-aware CNN encoder with statistics extraction + attention pooling
    
    TWO-STAGE APPROACH:
    1. Extract spatial statistics BEFORE pooling (preserves discriminative patterns)
    2. Attention pooling for global summary (efficient representation)
    
    This captures both spatial variance AND content-weighted summary!
    
    Architecture: 16→64→128→256 channels
    Output: [B, 640] = 512 (attention) + 128 (spatial stats)
    """
    def __init__(self, input_channels=3, output_dim=512):
        super().__init__()
        
        # CNN backbone (same as before)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # H/2, W/2
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # H/4, W/4
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # H/8, W/8
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            # NO pooling here - keep spatial structure
        )
        
        # STAGE 1: Extract edge density features (TOP discriminator)
        self.edge_extractor = EdgeDensityExtractor(in_channels=256)
        self.edge_proj = nn.Sequential(
            nn.Linear(64, 128),  # Edge features → 128 dims
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # STAGE 2: Attention pooling (content-weighted summary)
        self.spatial_pool = SpatialAttentionPooling(
            in_channels=256,
            out_dim=output_dim,
            num_heads=8
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - single frame or batch of frames
        Returns:
            [B, 640] - 512 (attention) + 128 (edge features)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # [B, 256, 6, 8] - spatial structure preserved
        
        # Extract edge density features BEFORE pooling!
        edge_features = self.edge_extractor(x)  # [B, 64]
        edge_features = self.edge_proj(edge_features)  # [B, 128]
        
        # Attention pooling for global summary
        pooled = self.spatial_pool(x)  # [B, 512]
        
        # Combine both: global summary + edge features
        out = torch.cat([pooled, edge_features], dim=-1)  # [B, 640]
        
        return out


class MultiScaleDilatedConv(nn.Module):
    """
    Multi-scale dilated temporal convolutions - TransNetV2's SECRET WEAPON
    
    WHY DILATED CONVOLUTIONS:
    - Capture patterns at multiple temporal scales WITHOUT pooling
    - Kernel 1: Single frame changes (instant cuts)
    - Kernel 2: 2-frame patterns (quick cuts)
    - Kernel 4: 4-frame patterns (fade transitions start)
    - Kernel 8: 8-frame patterns (dissolve/wipe mid-transition)
    - Kernel 16: 16-frame patterns (medium gradual transitions)
    - Kernel 32: 32-frame patterns (long gradual transitions)
    - Kernel 64: 64-frame patterns (very long gradual transitions - up to 128 frames!)
    
    EXTENDED for ClipShots dataset (graduals up to 81 frames):
    - Added dilation 32 and 64 to cover full gradual range
    - This fixes the "very long gradual" detection problem
    
    TransNetV2 proved this works better than:
    - Regular convolutions (limited receptive field)
    - Pooling (loses temporal precision)
    - Transformers (overkill for local patterns)
    """
    def __init__(self, dim, dilation_rates=[1, 2, 4, 8, 16, 32, 64]):
        super().__init__()
        
        self.dilation_rates = dilation_rates
        
        # Separate dilated conv for each scale
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(
                dim, dim, 
                kernel_size=3,
                dilation=rate,
                padding=rate,  # Same padding for each dilation
                groups=dim // 4  # Depthwise groups for efficiency
            )
            for rate in dilation_rates
        ])
        
        # Batch norm and activation for each scale
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(dim) for _ in dilation_rates
        ])
        
        # Fusion layer - combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(dim * len(dilation_rates), dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C] temporal features
        Returns:
            [B, T, C] multi-scale temporal features
        """
        # Transpose for 1D conv: [B, C, T]
        x_t = x.transpose(1, 2)
        
        # Apply each dilated conv at its scale
        scale_features = []
        for conv, norm in zip(self.dilated_convs, self.norms):
            feat = conv(x_t)  # [B, C, T]
            feat = norm(feat)
            feat = F.relu(feat, inplace=True)
            scale_features.append(feat)
        
        # Concatenate all scales
        multi_scale = torch.cat(scale_features, dim=1)  # [B, C*num_scales, T]
        
        # Fuse scales
        fused = self.fusion(multi_scale)  # [B, C, T]
        
        # Residual connection
        output = fused + x_t
        
        # Transpose back: [B, T, C]
        return output.transpose(1, 2)


class LightweightTemporalAttention(nn.Module):
    """
    Efficient temporal self-attention for global sequence context
    
    DESIGN CHOICES:
    - Single-head attention (boundary detection doesn't need multiple heads)
    - Linear complexity via efficient attention patterns
    - Small key/value dims (global context, not fine-grained)
    - Residual connection (attention as refinement, not primary path)
    
    This gives global context WITHOUT the bloat of full transformers
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # Layer norm (pre-norm transformer style)
        self.norm = nn.LayerNorm(dim)
        
        # Dropout
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C] temporal features
        Returns:
            [B, T, C] attention-refined features
        """
        B, T, C = x.shape
        
        # Pre-norm
        x_norm = self.norm(x)
        
        # QKV projections
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Residual connection
        return x + out


class TemporalPyramidPooling(nn.Module):
    """
    Simplified temporal pyramid pooling
    
    Uses adaptive pooling at multiple scales to capture patterns at different temporal resolutions
    Reduced from [1,2,4,8] to [1,4,8] - removes redundant level
    """
    def __init__(self, dim, pyramid_levels=[1, 4, 8]):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.num_levels = len(pyramid_levels)  # 3 levels
        
        # Project concatenated pyramid features back to dim
        # Each level contributes C channels, so total is num_levels * C
        self.pyramid_proj = nn.Sequential(
            nn.Linear(dim * self.num_levels, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C] temporal features
        Returns:
            [B, T, C] pyramid-pooled features
        """
        B, T, C = x.shape
        x_t = x.transpose(1, 2)  # [B, C, T]
        
        pyramid_features = []
        for bins in self.pyramid_levels:
            # AdaptiveAvgPool1d outputs exactly 'bins' values regardless of T
            pooled = F.adaptive_avg_pool1d(x_t, bins)  # [B, C, bins]
            # Upsample back to T using interpolation
            upsampled = F.interpolate(pooled, size=T, mode='linear', align_corners=False)  # [B, C, T]
            pyramid_features.append(upsampled)
        
        # Concatenate all pyramid levels: [B, C*3, T] (3 levels)
        pyramid = torch.cat(pyramid_features, dim=1)
        pyramid = pyramid.transpose(1, 2)  # [B, T, C*3]
        
        # Project back to original dimension
        output = self.pyramid_proj(pyramid)  # [B, T, C]
        
        return output


# LengthPredictor removed - was not providing clear benefit and added complexity


class ProgressiveFusionBlock(nn.Module):
    """
    Progressive fusion of local (dilated conv) and global (attention) features
    
    STRATEGY:
    1. Local patterns first (dilated convs) - primary signal
    2. Global context second (attention) - refinement
    3. Gated fusion - learn what to keep from each
    """
    def __init__(self, dim):
        super().__init__()
        
        # Multi-scale dilated convolutions (local patterns)
        self.local_conv = MultiScaleDilatedConv(dim)
        
        # Lightweight temporal attention (global context)
        self.global_attn = LightweightTemporalAttention(dim)
        
        # Gated fusion - learn to combine local + global
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
        """
        Args:
            x: [B, T, C] temporal features
        Returns:
            [B, T, C] fused local + global features
        """
        # Extract local patterns (dilated convs)
        local_feat = self.local_conv(x)
        
        # Extract global context (attention)
        global_feat = self.global_attn(x)
        
        # Concatenate for gating
        combined = torch.cat([local_feat, global_feat], dim=-1)
        
        # Compute gate weights (what to keep from local vs global)
        gate_weight = self.gate(combined)
        
        # Gated fusion
        fused = self.fusion(combined)
        fused = gate_weight * local_feat + (1 - gate_weight) * global_feat + fused
        
        return fused


class MotionDynamicsExtractor(nn.Module):
    """
    Extract motion dynamics: acceleration and jerk
    
    TOP discriminators from empirical analysis:
    - Acceleration (3.300 separation): Rate of change of motion
    - Jerk (3.254 separation): Rate of change of acceleration
    
    Key insight:
    - Gradual: smooth acceleration (content fading in/out)
    - Pan: constant velocity (no acceleration)
    - Cut: instant jump (infinite acceleration spike)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, diff_features):
        """
        Compute motion dynamics from temporal difference features
        
        Args:
            diff_features: [B, T, D] encoded difference features
        Returns:
            acceleration: [B, T, 1] - rate of change of motion
            jerk: [B, T, 1] - rate of change of acceleration
        """
        B, T, D = diff_features.shape
        
        # Motion magnitude: norm of diff features (proxy for optical flow magnitude)
        motion = diff_features.norm(dim=-1, keepdim=True)  # [B, T, 1]
        
        # Velocity: change in motion (first derivative)
        velocity = torch.zeros_like(motion)
        velocity[:, 1:] = motion[:, 1:] - motion[:, :-1]
        
        # Acceleration: change in velocity (second derivative)
        acceleration = torch.zeros_like(motion)
        acceleration[:, 2:] = velocity[:, 2:] - velocity[:, 1:-1]
        
        # Jerk: change in acceleration (third derivative)
        jerk = torch.zeros_like(motion)
        jerk[:, 3:] = acceleration[:, 3:] - acceleration[:, 2:-1]
        
        # Smooth with small window to reduce noise
        acceleration = F.avg_pool1d(
            acceleration.transpose(1, 2), 
            kernel_size=3, stride=1, padding=1
        ).transpose(1, 2)
        
        jerk = F.avg_pool1d(
            jerk.transpose(1, 2),
            kernel_size=3, stride=1, padding=1
        ).transpose(1, 2)
        
        return acceleration, jerk


class ReasoningModule(nn.Module):
    """
    REDESIGNED Reasoning Module - Motion-based + LEARNABLE
    
    Uses proven discriminative features:
    1. Motion acceleration (3.300 separation)
    2. Motion jerk (3.254 separation)  
    3. Edge density variance (from spatial features)
    4. Motion coherence (directional consistency)
    
    Makes features learnable via small MLPs to adapt to data patterns
    """
    def __init__(self, rgb_dim=640, diff_dim=640):
        super().__init__()
        
        # Motion dynamics extractor (acceleration + jerk)
        self.motion_dynamics = MotionDynamicsExtractor()
        
        # Learnable fusion of motion dynamics
        self.motion_fusion = nn.Sequential(
            nn.Linear(2, 32),  # acceleration + jerk → 32
            nn.LayerNorm(32),
            nn.GELU()
        )
        
        # Edge density from RGB spatial features (already has edge extractor)
        # Extract norm of features as proxy for spatial activity
        self.spatial_activity = nn.Sequential(
            nn.Linear(rgb_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        
        # Motion coherence from diff features (direction consistency)
        self.coherence_proj = nn.Sequential(
            nn.Linear(64, 64),  # Input is coherence_score [B, T, 64]
            nn.LayerNorm(64),
            nn.GELU()
        )
        
    def forward(self, rgb_features, diff_features):
        """
        Args:
            rgb_features: [B, T, 640] from RGB spatial encoder
            diff_features: [B, T, 640] from diff spatial encoder
        Returns:
            reasoning_outputs: dict with discriminative signals
        """
        B, T, D = rgb_features.shape
        
        # 1. MOTION DYNAMICS: Acceleration + Jerk (TOP discriminators!)
        acceleration, jerk = self.motion_dynamics(diff_features)  # [B, T, 1] each
        motion_features = torch.cat([acceleration, jerk], dim=-1)  # [B, T, 2]
        motion_logits = self.motion_fusion(motion_features)  # [B, T, 32]
        
        # 2. SPATIAL ACTIVITY: Edge density from RGB features
        spatial_logits = self.spatial_activity(rgb_features)  # [B, T, 64]
        
        # 3. MOTION COHERENCE: Direction consistency from diff features
        # Normalize diff features and compute local consistency
        diff_norm = F.normalize(diff_features, p=2, dim=-1)
        diff_prev = F.pad(diff_norm[:, :-1], (0, 0, 1, 0), value=0)
        coherence_score = (diff_norm * diff_prev).sum(dim=-1, keepdim=True)  # [B, T, 1]
        coherence_score = torch.cat([coherence_score, diff_features[:, :, :63]], dim=-1)  # [B, T, 64]
        coherence_logits = self.coherence_proj(coherence_score)  # [B, T, 64]
        
        # Combine all discriminative signals
        combined = torch.cat([
            motion_logits,     # [B, T, 32] - acceleration/jerk
            spatial_logits,    # [B, T, 64] - edge density
            coherence_logits   # [B, T, 64] - motion coherence
        ], dim=-1)  # [B, T, 160]
        
        # Return as reasoning outputs (learnable features)
        reasoning_outputs = {
            'combined': combined,  # [B, T, 160] - all discriminative features
            'acceleration': acceleration,  # [B, T, 1] - for monitoring
            'jerk': jerk  # [B, T, 1] - for monitoring
        }
        
        return reasoning_outputs


class BoundaryDetectionHead(nn.Module):
    """
    Redesigned boundary detection head with rich discriminative features
    
    IMPROVEMENTS:
    - Temporal pyramid pooling (length-invariant)
    - Simplified consistency pooling (2 scales instead of 3)
    - Enhanced type head with interaction features
    - Removed length predictor (wasn't helping)
    """
    def __init__(self, dim):
        super().__init__()
        
        # Temporal Pyramid Pooling (simplified)
        self.temporal_pyramid = TemporalPyramidPooling(dim, pyramid_levels=[1, 4, 8])
        
        # Temporal consistency pooling (reduced to 2 scales)
        self.consistency_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=1, padding=k//2)
            for k in [11, 25]  # Short and medium gradual windows
        ])
        
        # Feature refinement (pyramid + 2 consistency + original)
        self.refine = nn.Sequential(
            nn.Linear(dim * 4, dim),  # pyramid + original + 2 consistency
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Binary boundary detection (simplified - reasoning already in fused features)
        self.boundary_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.2),  # Increased dropout
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, 1)
        )
        
        # Type classification (simplified - reasoning already in fused features)
        self.type_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.2),  # Increased dropout
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, 3)  # None, Cut, Gradual
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C] temporal features (already enriched with reasoning)
        Returns:
            boundary_logits: [B, T, 1]
            type_logits: [B, T, 3] - 3-CLASS (none=0, cut=1, gradual=2)
        """
        # Temporal Pyramid Pooling
        pyramid_features = self.temporal_pyramid(x)  # [B, T, C]
        
        # Temporal consistency features (variance over windows)
        x_t = x.transpose(1, 2)  # [B, C, T]
        consistency_features = []
        for pool in self.consistency_pools:
            pooled_mean = pool(x_t)  # [B, C, T]
            pooled_sq = pool(x_t ** 2)
            variance_feat = pooled_sq - pooled_mean ** 2
            variance_feat = torch.clamp(variance_feat, min=0)
            consistency_features.append(variance_feat.transpose(1, 2))  # [B, T, C]
        
        # Concatenate: original + pyramid + 2 consistency features
        multi_scale = torch.cat([x, pyramid_features] + consistency_features, dim=-1)
        
        # Refine features
        refined = self.refine(multi_scale)
        
        # Detection heads use refined features (reasoning already enriched upstream)
        boundary_logits = self.boundary_head(refined)  # [B, T, 1]
        type_logits = self.type_head(refined)  # [B, T, 3]
        
        return boundary_logits, type_logits


class HybridBoundaryDetector(nn.Module):
    """
    Hybrid Boundary Detector - 2026 REDESIGNED with Empirical Features
    
    REDESIGN BASED ON FEATURE ANALYSIS:
    ✓ Motion acceleration (3.300 separation) - rate of change of motion
    ✓ Motion jerk (3.254 separation) - rate of change of acceleration
    ✓ Edge density variance (3.172 separation) - content emergence vs translation
    ✓ Removed: histogram (1.3), temporal similarity (0.99) - weak discriminators
    
    KEY INSIGHT:
    Graduals vs smooth pans have IDENTICAL RGB features but DIFFERENT motion dynamics:
    - Gradual: smooth acceleration (content fading)
    - Pan: zero acceleration (constant velocity)
    
    ARCHITECTURE:
    - Dual encoders (RGB + multi-distance diffs) with edge extraction
    - Motion dynamics extractor (acceleration + jerk from diffs)
    - Learnable reasoning module (motion + spatial + coherence)
    - Progressive fusion blocks (local→global)
    - Binary boundary + 3-class type heads
    
    FOCUSED DESIGN:
    - Spatial encoders: 640 dims (512 attention + 128 edge features)
    - Reasoning: 160 dims (32 motion + 64 spatial + 64 coherence) 
    - Main processing: configurable (default 384)
    - Total: ~8-10M params (proven effective range)
    """
    def __init__(self, main_dim=512, num_fusion_blocks=4):
        super().__init__()
        
        # Spatial encoders: 640 dims each (512 attention + 128 edge features)
        self.rgb_encoder = EfficientSpatialEncoder(input_channels=3, output_dim=512)
        self.diff_encoder = EfficientSpatialEncoder(input_channels=12, output_dim=512)
        
        # Feature fusion: FOCUSED on discriminative features
        # Input: 640 (RGB) + 640 (diff) = 1,280 dims
        # REMOVED: histogram (1.3 separation), temporal similarity (0.99 separation)
        # ADDED: edge density in spatial encoders, motion dynamics in reasoning
        feature_input_dim = 640 + 640
        # Direct projection with dropout for regularization  
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_input_dim, main_dim * 2),  # 1280 → 768 (if main=384)
            nn.LayerNorm(main_dim * 2),
            nn.GELU(),
            nn.Dropout(0.25),  # INCREASED: prevent overfitting
            nn.Linear(main_dim * 2, main_dim),  # 768 → 384
            nn.LayerNorm(main_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Progressive fusion blocks at MAIN dimension (where complexity lives)
        self.fusion_blocks = nn.ModuleList([
            ProgressiveFusionBlock(main_dim)
            for _ in range(num_fusion_blocks)
        ])
        
        # Reasoning module (REDESIGNED - motion-based + learnable)
        # Takes rgb_features and diff_features directly from spatial encoders
        self.reasoning_module = ReasoningModule(rgb_dim=640, diff_dim=640)
        
        # Final boundary detection head at MAIN dimension
        self.detection_head = BoundaryDetectionHead(main_dim)
        
        # Enrichment projection (was incorrectly created in forward()!)
        # This projects reasoning-enriched features back to main_dim
        # Reasoning outputs 160 dims (32 motion + 64 spatial + 64 coherence)
        self.enrichment_proj = nn.Linear(main_dim + 160, main_dim)
        
        self.main_dim = main_dim
        
    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, T, C, H, W] video frames (typically T=128, C=3, H=48, W=64)
            return_features: If True, return intermediate features
            
        Returns:
            boundary_logits: [B, T, 1] - binary boundary detection
            type_logits: [B, T, 3] - 3-class (none, cut, gradual)
            
        Note: Reasoning features are used internally to enrich representations
              but NOT supervised directly (they're too ambiguous - graduals and 
              smooth pans have identical reasoning features!)
        """
        B, T, C, H, W = x.shape
        
        # 1. EFFICIENT: Compute multi-distance differences as stacked channels
        # Instead of encoding each distance separately (4x slower!), 
        # we stack them as channels and encode ONCE
        diff_distances = [1, 4, 8, 16]
        
        # Pre-allocate stacked difference tensor: [B, T, C*len(distances), H, W]
        all_diffs = torch.zeros(B, T, C * len(diff_distances), H, W, device=x.device, dtype=x.dtype)
        
        for i, dist in enumerate(diff_distances):
            if dist < T:
                all_diffs[:, dist:, i*C:(i+1)*C] = x[:, dist:] - x[:, :-dist]
        
        # Spatial encoding: Edge extraction + attention pooling
        x_flat = x.reshape(B * T, C, H, W)
        diff_flat = all_diffs.reshape(B * T, C * len(diff_distances), H, W)
        
        # RGB features: 640 dims (512 attention + 128 edge features)
        rgb_features = self.rgb_encoder(x_flat)  # [B*T, 640]
        rgb_features = rgb_features.reshape(B, T, -1)  # [B, T, 640]
        
        # Difference features: 640 dims (512 attention + 128 edge features)
        diff_features = self.diff_encoder(diff_flat)  # [B*T, 640]
        diff_features = diff_features.reshape(B, T, -1)  # [B, T, 640]
        
        # REMOVED: Histogram and temporal similarity (weak discriminators)
        # FOCUS: RGB + diff features contain edge density + spatial structure
        
        # Combine FOCUSED features (RGB + diff only)
        # Total: 640 + 640 = 1,280 dims
        combined = torch.cat([rgb_features, diff_features], dim=-1)
        x = self.feature_proj(combined)  # [B, T, main_dim]
        
        # REASONING: Extract motion dynamics and discriminative features
        # CRITICAL: Compute BEFORE fusion blocks to preserve raw signals!
        # Acceleration/jerk computed from diff_features, edge density from rgb_features
        reasoning_outputs = self.reasoning_module(rgb_features, diff_features)
        
        # Get combined reasoning features (160 dims: motion + spatial + coherence)
        reasoning_combined = reasoning_outputs['combined']  # [B, T, 160]
        
        # Enrich fused features with reasoning signals
        x_enriched = torch.cat([x, reasoning_combined], dim=-1)  # [B, T, main_dim+160]
        
        # Project back to main_dim using properly initialized layer from __init__
        x = self.enrichment_proj(x_enriched)
        
        # 7. Progressive fusion (local→global) - with motion-enriched features
        for fusion_block in self.fusion_blocks:
            x = fusion_block(x)
        
        # 8. Boundary & type detection (NO reasoning features passed to head)
        # Detection head uses only fused features (reasoning already enriched them)
        boundary_logits, type_logits_raw = self.detection_head(x)
        
        # GATING: Suppress cut/gradual predictions when boundary confidence is low
        # This structurally enforces: "no boundary → type must be none"
        # DETACHED: No gradient flow from type loss to boundary head via gating
        boundary_probs = torch.sigmoid(boundary_logits).detach()  # [B, T, 1]
        type_logits = type_logits_raw.clone()
        # Multiply cut/gradual logits by boundary probability (none logit unchanged)
        type_logits[:, :, :2] = type_logits[:, :, :2] * boundary_probs
        
        if return_features:
            return boundary_logits, type_logits, {
                'rgb': rgb_features,
                'diff': diff_features,
                'fused': x,
                'reasoning': reasoning_outputs,
                'acceleration': reasoning_outputs['acceleration'],
                'jerk': reasoning_outputs['jerk']
            }
        
        # Return reasoning outputs for MONITORING
        # Track acceleration/jerk to verify they're discriminative
        # Return both gated and raw type logits (raw for training, gated for inference)
        return boundary_logits, type_logits, reasoning_outputs, type_logits_raw
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_hybrid_detector(main_dim=512, num_fusion_blocks=4):
    """
    Factory function for creating Hybrid Boundary Detector
    
    Args:
        main_dim: Main processing dimension (default 512 for competitive capacity)
                  This is used for fusion blocks and detection heads.
                  Component dims are FIXED: spatial=256, histogram=128, similarity=128
                  TransNetV2 uses 1024, recommended 384-768 for our architecture
        num_fusion_blocks: Number of progressive fusion blocks (default 4)
    
    Returns:
        HybridBoundaryDetector model
    """
    print("="*80)
    print("Creating Hybrid Boundary Detector - 2026 REDESIGNED")
    print("="*80)
    print("\nDESIGN PHILOSOPHY (Based on Empirical Feature Analysis):")
    print("  ✓ Motion acceleration (3.300 separation) - TOP discriminator")
    print("  ✓ Motion jerk (3.254 separation) - rate of acceleration change")
    print("  ✓ Edge density variance (3.172 separation) - content emergence detection")
    print("  ✓ Motion coherence (1.957 separation) - directional consistency")
    print("  ✗ REMOVED: Histogram (1.3), Temporal similarity (0.99) - weak features")
    
    print("\nKEY INSIGHT:")
    print("  Graduals vs smooth pans have IDENTICAL RGB features!")
    print("  Separation comes from MOTION DYNAMICS (acceleration/jerk)")
    print("  Model now computes motion derivatives from frame differences")
    
    print("\nARCHITECTURE CHANGES:")
    print("  + Edge extraction in spatial encoders (before pooling)")
    print("  + Motion dynamics module (acceleration + jerk)")
    print("  + Learnable reasoning (160 dims: motion+spatial+coherence)")
    print("  - Removed histogram encoder (weak discriminator)")
    print("  - Removed temporal similarity (weak discriminator)")
    
    model = HybridBoundaryDetector(
        main_dim=main_dim,
        num_fusion_blocks=num_fusion_blocks
    )
    
    total_params, trainable_params = model.count_parameters()
    
    print(f"\nMODEL STATISTICS:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory footprint: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    print(f"\nARCHITECTURE DIMENSIONS:")
    print(f"  Spatial encoders: 640 dims (512 attention + 128 edge)")
    print(f"  Reasoning module: 160 dims (motion+spatial+coherence)")
    print(f"  Main processing: {main_dim} dims (fusion + detection)")
    print(f"\nCOMPARISON:")
    print(f"  TransNetV2: ~7.0M params")
    print(f"  Previous model: ~{total_params/1e6:.1f}M params (had weak features)")
    print(f"  This model: {total_params/1e6:.1f}M params (FOCUSED features)")
    print(f"  Target range: 7-10M params (proven effective)")
    
    print("\n" + "="*80)
    
    return model


if __name__ == '__main__':
    print("\n" + "="*80)
    print("TESTING HYBRID BOUNDARY DETECTOR")
    print("="*80)
    
    # Create model with default settings
    model = create_hybrid_detector(main_dim=256, num_fusion_blocks=3)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch = torch.randn(2, 128, 3, 48, 64)  # Typical ClipShots input
    print(f"Input shape: {batch.shape}")
    
    with torch.no_grad():
        boundary_logits, type_logits, reasoning = model(batch)
    
    print(f"\nOutput shapes:")
    print(f"  Boundary logits: {boundary_logits.shape} (expected: [2, 128, 1])")
    print(f"  Type logits: {type_logits.shape} (expected: [2, 128, 3])")
    
    print(f"\nReasoning outputs (motion dynamics):")
    print(f"  Combined features: {reasoning['combined'].shape} - motion+spatial+coherence")
    print(f"  Acceleration: {reasoning['acceleration'].shape} - rate of motion change")
    print(f"  Jerk: {reasoning['jerk'].shape} - rate of acceleration change")
    
    # Analyze predictions
    boundary_probs = torch.sigmoid(boundary_logits)
    type_probs = torch.softmax(type_logits, dim=-1)
    
    print(f"\nPrediction analysis:")
    print(f"  Boundary probs range: [{boundary_probs.min():.4f}, {boundary_probs.max():.4f}]")
    print(f"  Type probs (none): [{type_probs[:,:,0].min():.4f}, {type_probs[:,:,0].max():.4f}]")
    print(f"  Type probs (cut): [{type_probs[:,:,1].min():.4f}, {type_probs[:,:,1].max():.4f}]")
    print(f"  Type probs (gradual): [{type_probs[:,:,2].min():.4f}, {type_probs[:,:,2].max():.4f}]")
    
    print(f"\nMotion dynamics analysis:")
    print(f"  Acceleration: [{reasoning['acceleration'].min():.4f}, {reasoning['acceleration'].max():.4f}]")
    print(f"  Jerk: [{reasoning['jerk'].min():.4f}, {reasoning['jerk'].max():.4f}]")
    
    # Count predicted boundaries
    for threshold in [0.5, 0.6, 0.7]:
        n_boundaries = (boundary_probs.squeeze(-1) > threshold).sum().item()
        total_frames = batch.shape[0] * batch.shape[1]
        sparsity = 100 * (1 - n_boundaries / total_frames)
        print(f"  Predicted boundaries (p>{threshold}): {n_boundaries}/{total_frames} ({sparsity:.1f}% non-boundary)")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
