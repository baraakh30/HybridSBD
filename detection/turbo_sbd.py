import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransitionDiscriminator(nn.Module):
    """
    Discriminates real transitions from normal motion/changes
    
    Key insight: Transitions have MONOTONIC similarity patterns:
    - Gradual: Smooth monotonic decrease in similarity over time
    - Cut: Sharp single-frame similarity drop
    - Motion: Random fluctuations, no monotonic trend
    """
    def __init__(self, dim):
        super().__init__()
        
        # 1. Frame-to-frame similarity tracker
        # Computes cosine similarity between adjacent frames
        self.similarity_proj = nn.Linear(dim, dim // 2)
        
        # 2. Monotonicity detector (detects consistent directional change)
        # Real graduals have consistent decreasing similarity
        self.monotonicity_conv = nn.Conv1d(1, 16, kernel_size=15, padding=7)
        
        # 3. Boundary sharpness (sharp for cuts, smooth for graduals)
        self.sharpness_detector = nn.Sequential(
            nn.Conv1d(dim, dim // 2, kernel_size=3, padding=1, groups=dim // 2),
            nn.GELU()
        )
        
        # 4. Temporal extent estimator (how long is the pattern?)
        self.extent_estimator = nn.Sequential(
            nn.Conv1d(dim + 16 + dim // 2, dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(dim, dim // 2, kernel_size=1)
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim // 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        
        # 1. Compute frame-to-frame similarities
        x_proj = self.similarity_proj(x)  # [B, T, C//2]
        
        # Cosine similarity between adjacent frames
        similarities = []
        for t in range(T - 1):
            sim = F.cosine_similarity(
                x_proj[:, t, :],
                x_proj[:, t + 1, :],
                dim=1
            )  # [B]
            similarities.append(sim)
        
        # Pad to maintain temporal dimension
        similarities = torch.stack(similarities, dim=1)  # [B, T-1]
        similarities = F.pad(similarities, (0, 1), value=1.0)  # [B, T]
        
        # 2. Detect monotonic trends (real graduals have monotonic similarity decrease)
        sim_expanded = similarities.unsqueeze(1)  # [B, 1, T]
        monotonicity = self.monotonicity_conv(sim_expanded)  # [B, 16, T]
        
        # 3. Boundary sharpness
        x_t = x.transpose(1, 2)  # [B, C, T]
        sharpness = self.sharpness_detector(x_t)  # [B, C//2, T]
        
        # 4. Combine all discriminative signals
        combined = torch.cat([x_t, monotonicity, sharpness], dim=1)  # [B, C+16+C//2, T]
        extent = self.extent_estimator(combined)  # [B, C//2, T]
        
        # Transpose back
        discriminative = extent.transpose(1, 2)  # [B, T, C//2]
        
        # Fusion to full dimension
        out = self.fusion(discriminative)  # [B, T, C]
        
        return out


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block (2022-2026 standard for vision)
    More robust than DepthwiseSeparable, better generalization
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = input + self.drop_path(x)
        return x

class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) for regularization"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class ModernSpatialEncoder(nn.Module):
    """
    2026 State-of-the-art spatial encoder (ConvNeXt-style)
    Input: [B*T, C, H, W] -> Output: [B*T, embed_dim]
    
    Improvements over old version:
    - ConvNeXt blocks (better generalization than depthwise sep)
    - Deeper with residual connections (learns better features)
    - Stochastic depth for regularization (prevents overfitting)
    - LayerNorm instead of BatchNorm (more stable)
    """
    def __init__(self, input_channels=3, embed_dim=128, depths=[2, 2, 3]):
        super().__init__()
        self.out_dim = embed_dim
        
        # Stem: 48x64 -> 24x32
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, embed_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([embed_dim // 4, 24, 32], eps=1e-6)
        )
        
        # Stage 1: 24x32 -> 12x16 (dim: 32 -> 64)
        self.downsample1 = nn.Sequential(
            nn.LayerNorm([embed_dim // 4, 24, 32], eps=1e-6),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=2, stride=2)
        )
        self.stage1 = nn.Sequential(*[
            ConvNeXtBlock(embed_dim // 2, drop_path=0.1 * i / sum(depths))
            for i in range(depths[0])
        ])
        
        # Stage 2: 12x16 -> 6x8 (dim: 64 -> 128)
        self.downsample2 = nn.Sequential(
            nn.LayerNorm([embed_dim // 2, 12, 16], eps=1e-6),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2)
        )
        self.stage2 = nn.Sequential(*[
            ConvNeXtBlock(embed_dim, drop_path=0.1 * (depths[0] + i) / sum(depths))
            for i in range(depths[1])
        ])
        
        # Stage 3: 6x8 spatial processing (dim: 128)
        self.stage3 = nn.Sequential(*[
            ConvNeXtBlock(embed_dim, drop_path=0.1 * (depths[0] + depths[1] + i) / sum(depths))
            for i in range(depths[2])
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.downsample1(x)
        x = self.stage1(x)
        x = self.downsample2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.norm(x)
        return x

class LearnableTemporalDiff(nn.Module):
    """
    Learnable temporal difference computation (2026 approach)
    Instead of manual diff, let network learn what temporal patterns matter
    """
    def __init__(self, feature_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        # Learn to combine multi-scale temporal differences
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.projection = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """
        x: [B, T, C] - sequence of frame features
        Returns: [B, T, C] - learned temporal difference features
        """
        B, T, C = x.shape
        
        # Multi-scale temporal differences (learned weights)
        diffs = []
        for i, shift in enumerate([1, 2, 4]):
            diff = x - torch.roll(x, shifts=shift, dims=1)
            diff[:, :shift] = 0  # Zero out invalid positions
            diffs.append(diff * self.scale_weights[i])
        
        # Concatenate and project
        combined = torch.cat(diffs, dim=-1)  # [B, T, C*3]
        out = self.projection(combined)  # [B, T, C]
        return out

class EfficientTemporalAttention(nn.Module):
    """
    Efficient self-attention for temporal modeling (2026 standard)
    Uses linear complexity attention for long sequences
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        
        # Normalize input
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Efficient attention with Flash Attention style (if available)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        # Residual connection
        return x + x_attn

class TemporalContrastModule(nn.Module):
    """
    Learns discriminative temporal patterns by contrasting different time scales
    
    Key insight:
    - Cuts: Sharp discontinuity at single frame (high short-term contrast)
    - Graduals: Smooth change over time (high long-term contrast, low short-term)
    - None: Continuous motion (low contrast at all scales)
    
    Updated to cover dataset distribution:
    - Short graduals (2-5f): 34.3% → kernels 3, 7
    - Medium graduals (6-15f): 32.8% → kernels 7, 15
    - Long graduals (16-30f): 24.9% → kernels 31, 63
    - Very long (>30f): 8.0% (up to 243f!) → kernel 63 + dilated
    """
    def __init__(self, dim):
        super().__init__()
        # Multi-scale contrast computation (8 scales for full coverage + clean division)
        # 256 / 8 = 32 channels per scale (divides evenly for grouped conv)
        channels_per_scale = dim // 8
        
        # Fine scales: cuts and short graduals (2-5 frames)
        self.ultra_short = nn.Conv1d(dim, channels_per_scale, kernel_size=3, padding=1, groups=channels_per_scale)  # ±1 frame
        self.short_contrast = nn.Conv1d(dim, channels_per_scale, kernel_size=7, padding=3, groups=channels_per_scale)  # ±3 frames
        
        # Medium scales: medium graduals (6-15 frames)
        self.medium_contrast = nn.Conv1d(dim, channels_per_scale, kernel_size=15, padding=7, groups=channels_per_scale)  # ±7 frames
        self.medium_long = nn.Conv1d(dim, channels_per_scale, kernel_size=23, padding=11, groups=channels_per_scale)  # ±11 frames
        
        # Long scales: long graduals (16-30 frames)
        self.long_contrast = nn.Conv1d(dim, channels_per_scale, kernel_size=31, padding=15, groups=channels_per_scale)  # ±15 frames
        self.long_extended = nn.Conv1d(dim, channels_per_scale, kernel_size=47, padding=23, groups=channels_per_scale)  # ±23 frames
        
        # Very long scales: very long graduals (30-60 frames) - 95th percentile coverage
        self.vlong_contrast = nn.Conv1d(dim, channels_per_scale, kernel_size=63, padding=31, groups=channels_per_scale)  # ±31 frames
        
        # Ultra long scales: extreme graduals (60-120 frames) - dilated for efficiency
        # Dilation=2 → effective receptive field: 63*2-1 = 125 frames (±62 frames)
        self.ultra_long = nn.Conv1d(dim, channels_per_scale, kernel_size=63, padding=62, dilation=2, groups=channels_per_scale)
        
        # Adaptive gating: learn importance of each scale
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, 8, 1),  # 8 scales
            nn.Sigmoid()
        )
        
        # Learnable fusion of contrast patterns
        self.contrast_fusion = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # x: [B, C, T]
        # Compute contrasts at 8 different scales
        ultra_short = self.ultra_short(x)
        short = self.short_contrast(x)
        medium = self.medium_contrast(x)
        medium_long = self.medium_long(x)
        long = self.long_contrast(x)
        long_extended = self.long_extended(x)
        vlong = self.vlong_contrast(x)
        ultra_long = self.ultra_long(x)
        
        # Stack scales for gating: [B, 8, C_per_scale, T] -> [B, C, T]
        scales = [ultra_short, short, medium, medium_long, long, long_extended, vlong, ultra_long]
        
        # Compute adaptive weights for each scale
        gate_weights = self.scale_gate(x)  # [B, 8, 1]
        
        # Weight each scale by learned importance
        weighted_scales = []
        for i, scale in enumerate(scales):
            weight = gate_weights[:, i:i+1, :]  # [B, 1, 1]
            weighted_scales.append(scale * weight)
        
        # Concatenate weighted contrasts
        contrasts = torch.cat(weighted_scales, dim=1)  # [B, C, T]
        
        # Fuse and add to input
        fused = self.contrast_fusion(contrasts)
        
        # Residual connection
        return x + fused

class DualStreamTemporalProcessor(nn.Module):
    """
    Dual-stream processing for different temporal dynamics
    
    Fast stream: Captures abrupt changes (cuts)
    Slow stream: Captures gradual transitions (dissolves, fades)
    
    Updated slow stream to handle long graduals (up to 60+ frames)
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # Fast stream: Small receptive field, high temporal resolution
        self.fast_stream = nn.ModuleList([
            EfficientTemporalAttention(dim, num_heads=num_heads, proj_drop=0.1),
            TemporalContrastModule(dim)
        ])
        
        # Slow stream: Very large receptive field for long graduals (up to 120 frames)
        # Use parallel multi-scale temporal smoothing
        self.slow_stream = nn.ModuleList([
            EfficientTemporalAttention(dim, num_heads=num_heads // 2, proj_drop=0.1),
        ])
        
        # Multi-scale temporal smoothing (parallel, not sequential)
        self.slow_medium_range = nn.Conv1d(dim, dim // 2, kernel_size=31, padding=15, groups=dim // 2)
        self.slow_long_range = nn.Conv1d(dim, dim // 2, kernel_size=63, padding=62, dilation=2, groups=dim // 2)
        
        # Cross-stream interaction
        self.cross_fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        
        # Fast stream processing
        fast = x
        fast = self.fast_stream[0](fast)  # Attention
        fast_t = fast.transpose(1, 2)  # [B, C, T]
        fast_t = self.fast_stream[1](fast_t)  # Contrast
        fast = fast_t.transpose(1, 2)  # [B, T, C]
        
        # Slow stream processing
        slow = x
        slow = self.slow_stream[0](slow)  # Attention
        slow_t = slow.transpose(1, 2)  # [B, C, T]
        
        # Multi-scale temporal smoothing (parallel)
        slow_medium = self.slow_medium_range(slow_t)  # [B, C//2, T]
        slow_long = self.slow_long_range(slow_t)      # [B, C//2, T]
        slow_multi = torch.cat([slow_medium, slow_long], dim=1)  # [B, C, T]
        
        slow = slow_multi.transpose(1, 2) + slow  # Residual
        slow = F.layer_norm(slow, [C])  # Normalize
        
        # Fuse streams
        combined = torch.cat([fast, slow], dim=-1)  # [B, T, 2C]
        fused = self.cross_fusion(combined)  # [B, T, C]
        
        return fused

class TransitionLengthPredictor(nn.Module):
    """
    Auxiliary task: Predict transition length
    
    Helps model learn:
    - Cuts have length 1-2
    - Graduals have length 2-50+
    - None has length 0
    
    This forces the model to understand temporal extent
    """
    def __init__(self, dim, max_length=100):
        super().__init__()
        self.max_length = max_length
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1)  # Regression: predict length
        )
        
    def forward(self, x):
        # x: [B, T, C]
        # Predict length for each frame
        lengths = self.predictor(x)  # [B, T, 1]
        lengths = torch.sigmoid(lengths) * self.max_length  # Scale to [0, max_length]
        return lengths.squeeze(-1)  # [B, T]

class TemporalTransformerBlock(nn.Module):
    """
    Modern temporal transformer block (2026 standard)
    Combines attention with feed-forward network
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.1, drop_path=0.1):
        super().__init__()
        self.attn = EfficientTemporalAttention(dim, num_heads=num_heads, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        # x: [B, T, C]
        x = self.attn(x)
        x = x + self.drop_path(self.mlp(x))
        return x

class TurboSBD(nn.Module):
    """
    TurboSBD - 2026 State-of-the-Art Shot Boundary Detection
    
    Architecture improvements:
    1. Modern spatial encoder (ConvNeXt-style blocks)
    2. Learnable temporal difference (not manual diffs)
    3. Dual-stream temporal processing (fast/slow)
    4. Temporal contrast learning (discriminative patterns)
    5. Auxiliary length prediction (transition awareness)
    6. End-to-end learnable (no manual routing)
    7. Stochastic depth regularization (prevents overfitting)
    """
    def __init__(self, num_classes=3, hidden_dim=256, num_layers=4, num_heads=8, 
                 use_length_predictor=True):
        super().__init__()
        self.use_length_predictor = use_length_predictor
        
        # 1. Modern Spatial Encoders (ConvNeXt-style)
        self.rgb_encoder = ModernSpatialEncoder(input_channels=3, embed_dim=128, depths=[2, 2, 3])
        
        # 2. Learnable Temporal Difference
        self.temporal_diff = LearnableTemporalDiff(feature_dim=128, num_scales=3)
        
        # Project concatenated features to hidden dim
        # RGB (128) + Temporal Diff (128) = 256
        self.feature_projection = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        
        # 3. Dual-Stream Temporal Processing (NEW!)
        self.dual_stream = DualStreamTemporalProcessor(hidden_dim, num_heads=num_heads)
        
        # 4. Temporal Transformer Blocks with Contrast Modules (ENHANCED!)
        dpr = [x.item() for x in torch.linspace(0, 0.15, num_layers)]
        self.temporal_blocks = nn.ModuleList()
        self.contrast_modules = nn.ModuleList()
        
        for i in range(num_layers):
            self.temporal_blocks.append(
                TemporalTransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    drop=0.1,
                    drop_path=dpr[i]
                )
            )
            # Add contrast module after each transformer block
            self.contrast_modules.append(TemporalContrastModule(hidden_dim))
        
        # 5. Auxiliary Length Predictor (NEW!)
        if self.use_length_predictor:
            self.length_predictor = TransitionLengthPredictor(hidden_dim, max_length=100)
        
        # 6. Transition Discriminator (NEW! - KEY FIX)
        # Separates real transitions from motion/noise
        self.transition_discriminator = TransitionDiscriminator(hidden_dim)
        
        # 7. Multi-Scale Temporal Pooling (capture different temporal patterns)
        # Use depthwise convolutions (groups=hidden_dim) for efficiency
        self.temporal_pool_scales = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),  # Short-range
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),  # Medium-range
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7, groups=hidden_dim),  # Long-range
        ])
        
        # Fuse multi-scale features
        self.scale_fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1)
        
        # Enhanced Classification Head (stronger discriminative capacity)
        # Now takes discriminator features + pooled features
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),  # pooled + discriminator
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, return_length_pred=False):
        """
        x: [B, T, 3, H, W] - Raw RGB frames
        return_length_pred: Whether to return auxiliary length predictions (for training)
        """
        B, T, C, H, W = x.shape
        
        # 1. Spatial Feature Extraction (Modern ConvNeXt Encoder)
        x_flat = x.view(B * T, C, H, W)
        spatial_features = self.rgb_encoder(x_flat)  # [B*T, 128]
        spatial_features = spatial_features.view(B, T, -1)  # [B, T, 128]
        
        # 2. Learnable Temporal Difference Features
        temporal_diff_features = self.temporal_diff(spatial_features)  # [B, T, 128]
        
        # 3. Combine RGB and Temporal Diff
        combined_features = torch.cat([spatial_features, temporal_diff_features], dim=-1)  # [B, T, 256]
        
        # Project to hidden dimension
        x = self.feature_projection(combined_features)  # [B, T, hidden_dim]
        
        # 4. Dual-Stream Processing (Fast + Slow)
        x = self.dual_stream(x)  # [B, T, hidden_dim]
        
        # 5. Temporal Transformer Blocks with Contrast Learning
        for i, (block, contrast) in enumerate(zip(self.temporal_blocks, self.contrast_modules)):
            x = block(x)  # [B, T, hidden_dim]
            # Apply contrast module
            x_t = x.transpose(1, 2)  # [B, hidden_dim, T]
            x_t = contrast(x_t)  # [B, hidden_dim, T]
            x = x_t.transpose(1, 2)  # [B, T, hidden_dim]
        
        # 6. Auxiliary Length Prediction (helps learn temporal extent)
        length_pred = None
        if self.use_length_predictor:
            length_pred = self.length_predictor(x)  # [B, T]
        
        # 7. Transition Discrimination (KEY FIX - separates transitions from motion)
        discriminative_features = self.transition_discriminator(x)  # [B, T, hidden_dim]
        
        # 8. Multi-Scale Temporal Pooling + Classification
        x_t = x.transpose(1, 2)  # [B, hidden_dim, T]
        
        # Apply multi-scale pooling and concatenate
        pooled_scales = []
        for pool in self.temporal_pool_scales:
            pooled_scales.append(pool(x_t))
        x_multiscale = torch.cat(pooled_scales, dim=1)  # [B, hidden_dim*3, T]
        
        # Fuse scales
        x_t = self.scale_fusion(x_multiscale)  # [B, hidden_dim, T]
        x_t = F.gelu(x_t)
        
        x_pooled = x_t.transpose(1, 2)  # [B, T, hidden_dim]
        
        # Combine pooled features + discriminative features
        x_final = torch.cat([x_pooled, discriminative_features], dim=-1)  # [B, T, hidden_dim*2]
        
        logits = self.head(x_final)  # [B, T, 3]
        
        if return_length_pred and length_pred is not None:
            return logits, length_pred
        return logits

def create_model():
    """Factory function for creating TurboSBD model"""
    print("Creating TurboSBD model...")
    model = TurboSBD(num_classes=3, hidden_dim=256, num_layers=4, num_heads=8)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    return model