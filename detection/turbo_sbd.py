import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SpatialEncoder(nn.Module):
    """
    Minimal Spatial Encoder for tiny frames (48x64).
    Input: [B*T, C, H, W] -> Output: [B*T, C]
    """
    def __init__(self, input_channels=3, base_dim=20):
        super().__init__()
        # Input: 48x64 → Already tiny, don't need deep processing
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_dim, 3, stride=2, padding=1, bias=False), # 24x32
            nn.BatchNorm2d(base_dim),
            nn.GELU()
        )
        
        self.layer1 = DepthwiseSeparableConv2d(base_dim, base_dim*2, 3, stride=2, padding=1) # 12x16
        self.layer2 = DepthwiseSeparableConv2d(base_dim*2, base_dim*4, 3, stride=2, padding=1) # 6x8
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = base_dim*4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        return x.flatten(1)

class FrameSimilarity(nn.Module):
    """
    Compute frame-to-frame similarity (TransNetV2 style).
    High similarity = static content, Low similarity = transition
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
    def forward(self, x):
        """
        x: [B, T, C] - sequence of frame features
        Returns: [B, T, 1] - similarity scores
        """
        B, T, C = x.shape
        
        # Compute cosine similarity between consecutive frames
        # x[t] vs x[t+1]
        x_curr = x[:, :-1, :]  # [B, T-1, C]
        x_next = x[:, 1:, :]   # [B, T-1, C]
        
        # Cosine similarity
        sim = F.cosine_similarity(x_curr, x_next, dim=2)  # [B, T-1]
        
        # Pad first frame with 1.0 (max similarity)
        sim = torch.cat([torch.ones(B, 1, device=x.device), sim], dim=1)  # [B, T]
        
        return sim.unsqueeze(2)  # [B, T, 1]

class BidirectionalContext(nn.Module):
    """
    Bidirectional temporal context using forward and backward Conv1D.
    """
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.forward_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.backward_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.fusion = nn.Conv1d(dim*2, dim, 1)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x: [B, C, T]
        # Forward pass
        x_fwd = self.forward_conv(x)
        # Backward pass (reverse time dimension)
        x_bwd = self.backward_conv(x.flip(dims=[2])).flip(dims=[2])
        
        # Combine
        x_bi = torch.cat([x_fwd, x_bwd], dim=1)  # [B, 2C, T]
        x_fused = self.fusion(x_bi)  # [B, C, T]
        
        # Residual + Norm
        x_out = x + x_fused
        x_out = x_out.transpose(1, 2)  # [B, T, C]
        x_out = self.norm(x_out)
        x_out = x_out.transpose(1, 2)  # [B, C, T]
        
        return x_out

class CutExpert(nn.Module):
    """
    Expert specialized for detecting CUTS (abrupt changes).
    Overlapping context [1,2,4] with Gradual Expert for better differentiation.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Overlapping temporal scales with Gradual Expert
        # Differentiation via diff ratios: cuts have low smoothness ratios
        self.layers = nn.ModuleList([
            GatedTemporalBlock(hidden_dim, kernel_size=3, dilation=1),  # Instant change
            GatedTemporalBlock(hidden_dim, kernel_size=5, dilation=2),  # Early context
            GatedTemporalBlock(hidden_dim, kernel_size=7, dilation=4),  # Medium context (overlap)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GradualExpert(nn.Module):
    """
    Expert specialized for detecting GRADUAL transitions (dissolves, fades).
    Needs FULL multi-scale context: detect onset (dilations 1,2) AND track progression (4,8,16).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Multi-scale: Short-term to detect onset + Long-term to track progression
        self.layers = nn.ModuleList([
            GatedTemporalBlock(hidden_dim, kernel_size=3, dilation=1),   # Detect onset/start
            GatedTemporalBlock(hidden_dim, kernel_size=5, dilation=2),   # Early progression
            GatedTemporalBlock(hidden_dim, kernel_size=7, dilation=4),   # Medium dissolves
            GatedTemporalBlock(hidden_dim, kernel_size=11, dilation=8),  # Long dissolves
            GatedTemporalBlock(hidden_dim, kernel_size=13, dilation=16), # Very long fades
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GatedTemporalBlock(nn.Module):
    """
    A minimal 'Mamba-style' block using Gated Linear Units and Depthwise Convs.
    Stable, fast, and captures long-range dependencies via large kernel or dilation.
    """
    def __init__(self, dim, expansion=2, kernel_size=31, dilation=1):
        super().__init__()
        inner_dim = dim * expansion
        self.conv_in = nn.Conv1d(dim, inner_dim * 2, 1) # Project to 2x (for GLU)
        
        # Depthwise temporal convolution (long context)
        # Padding for SAME output (assuming k is odd): P = dilation * (k - 1) / 2
        p = (dilation * (kernel_size - 1)) // 2
        self.conv_temporal = nn.Conv1d(
            inner_dim, inner_dim, kernel_size, 
            padding=p, dilation=dilation, groups=inner_dim
        )
        
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Conv1d(inner_dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        
        # LayerNorm (needs permute for 1D)
        x_norm = x.transpose(1, 2) # [B, T, C]
        x_norm = self.norm(x_norm)
        x_norm = x_norm.transpose(1, 2) # [B, C, T]
        
        # Project & Split
        x_proj = self.conv_in(x_norm)
        x_a, x_b = x_proj.chunk(2, dim=1)
        
        # Temporal Mixing on branch A
        x_a = self.conv_temporal(x_a)
        x_a = self.act(x_a)
        
        # Gating (Mamba-like: signal * gate)
        x_gate = x_a * x_b
        
        # Output project
        out = self.out_proj(x_gate)
        return residual + out

class GatedTemporalBlock(nn.Module):
    """
    A minimal 'Mamba-style' block using Gated Linear Units and Depthwise Convs.
    Stable, fast, and captures long-range dependencies via large kernel or dilation.
    """
    def __init__(self, dim, expansion=2, kernel_size=31, dilation=1):
        super().__init__()
        inner_dim = dim * expansion
        self.conv_in = nn.Conv1d(dim, inner_dim * 2, 1) # Project to 2x (for GLU)
        
        # Depthwise temporal convolution (long context)
        # Padding for SAME output (assuming k is odd): P = dilation * (k - 1) / 2
        p = (dilation * (kernel_size - 1)) // 2
        self.conv_temporal = nn.Conv1d(
            inner_dim, inner_dim, kernel_size, 
            padding=p, dilation=dilation, groups=inner_dim
        )
        
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Conv1d(inner_dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        
        # LayerNorm (needs permute for 1D)
        x_norm = x.transpose(1, 2) # [B, T, C]
        x_norm = self.norm(x_norm)
        x_norm = x_norm.transpose(1, 2) # [B, C, T]
        
        # Project & Split
        x_proj = self.conv_in(x_norm)
        x_a, x_b = x_proj.chunk(2, dim=1)
        
        # Temporal Mixing on branch A
        x_a = self.conv_temporal(x_a)
        x_a = self.act(x_a)
        
        # Gating (Mamba-like: signal * gate)
        x_gate = x_a * x_b
        
        # Output project
        out = self.out_proj(x_gate)
        return residual + out

class TurboSBD(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=192, num_layers=4):
        super().__init__()
        
        # 1. Spatial Encoders (Boosted capacity: base_dim=20)
        # RGB Stream
        self.rgb_encoder = SpatialEncoder(input_channels=3, base_dim=20)
        # Diff Stream (Explicit change detection - Multi-Scale)
        # 3 channels for diff1, 3 for diff4, 3 for diff16 = 9 input channels
        self.diff_encoder = SpatialEncoder(input_channels=9, base_dim=20)
        
        feature_dim = self.rgb_encoder.out_dim + self.diff_encoder.out_dim # 80 + 80 = 160
        
        # Frame Similarity
        self.frame_similarity = FrameSimilarity(feature_dim)
        
        # Project to hidden dim (160 features + 1 similarity)
        self.project_features = nn.Linear(feature_dim + 1, hidden_dim)
        self.drop = nn.Dropout(0.35)
        
        # 2. Bidirectional Context
        self.bidir_context = BidirectionalContext(hidden_dim, kernel_size=7)
        
        # 3. Hierarchical Expert System with Two-Stage Routing
        # 3. Parallel Independent Experts (No routing - all experts process all frames)
        # Cut Expert: Fast temporal patterns
        self.cut_expert = CutExpert(hidden_dim)
        
        # Gradual Expert: Slow temporal patterns  
        self.gradual_expert = GradualExpert(hidden_dim)
        
        # 4. Multi-Scale Temporal Aggregation for Classification (ENHANCED!)
        # Problem: Long graduals (avg 32 frames) need wider context than 5 frames
        # Solution: Multiple pooling scales combined
        
        # Cut Expert: Short-range only (cuts are instant)
        self.cut_temporal_pool = nn.Conv1d(hidden_dim + 5, hidden_dim + 5, 
                                           kernel_size=5, padding=2, groups=hidden_dim + 5)
        
        # Gradual Expert: Multi-scale (short, medium, long, very long graduals)
        # Debug analysis: 28% WRONG_EXPERT failures have avg length 21.2 frames
        # Need 4th scale to cover ±12 frames = 25-frame kernel
        self.gradual_pool_short = nn.Conv1d(hidden_dim + 5, hidden_dim + 5,
                                            kernel_size=5, padding=2, groups=hidden_dim + 5)   # ±2 frames
        self.gradual_pool_mid = nn.Conv1d(hidden_dim + 5, hidden_dim + 5,
                                          kernel_size=9, padding=4, groups=hidden_dim + 5)     # ±4 frames  
        self.gradual_pool_long = nn.Conv1d(hidden_dim + 5, hidden_dim + 5,
                                           kernel_size=17, padding=8, groups=hidden_dim + 5)   # ±8 frames
        self.gradual_pool_vlong = nn.Conv1d(hidden_dim + 5, hidden_dim + 5,
                                            kernel_size=25, padding=12, groups=hidden_dim + 5)  # ±12 frames (NEW!)
        
        # Combine multi-scale gradual features: 4x scales -> project back to original dim
        self.gradual_scale_fusion = nn.Conv1d((hidden_dim + 5) * 4, hidden_dim + 5, 1)
        
        # 5. UNIFIED 3-CLASS CLASSIFIER (Replaces broken binary expert combination)
        # Concatenates both expert outputs and learns to balance all 3 classes together
        # This fixes WRONG_EXPERT failures where binary heads couldn't fire
        self.unified_head = nn.Sequential(
            nn.Conv1d((hidden_dim + 5) * 2, hidden_dim, 1),  # Concat both experts + diff signals
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, 3, 1)  # Direct 3-class output [None, Cut, Gradual]
        )
        
    def forward(self, x):
        """
        x: [B, T, 3, H, W] (Raw Frames)
        """
        B, T, C, H, W = x.shape
        
        # 1. Compute Differences
        # diff1[t] = x[t] - x[t-1]
        x_diff1 = x - torch.roll(x, shifts=1, dims=1)
        x_diff1[:, 0] = 0
        
        # diff4[t] = x[t] - x[t-4] (Capture slower changes)
        x_diff4 = x - torch.roll(x, shifts=4, dims=1)
        x_diff4[:, :4] = 0
        
        # diff16[t] = x[t] - x[t-16] (Capture deep dissolves)
        x_diff16 = x - torch.roll(x, shifts=16, dims=1)
        x_diff16[:, :16] = 0
        
        # Concatenate differences: [B, T, 9, H, W]
        x_diff_combined = torch.cat([x_diff1, x_diff4, x_diff16], dim=2)
        
        # Flatten for spatial encoder
        x_flat = x.view(B*T, C, H, W)
        diff_flat = x_diff_combined.view(B*T, 9, H, W)
        
        # 2. Extract Features
        feat_rgb = self.rgb_encoder(x_flat)   # [B*T, 80]
        feat_diff = self.diff_encoder(diff_flat) # [B*T, 80]
        
        # Combine
        combined = torch.cat([feat_rgb, feat_diff], dim=1) # [B*T, 160]
        combined = combined.view(B, T, -1)  # [B, T, 160]
        
        # 3. Frame Similarity
        similarity = self.frame_similarity(combined)  # [B, T, 1]
        
        # Concatenate similarity as extra feature
        combined = torch.cat([combined, similarity], dim=2)  # [B, T, 161]
        
        # Project to hidden
        combined = self.project_features(combined) # [B, T, 192]
        combined = self.drop(combined)
        
        # [B, Hidden, T] for Conv1D
        temporal_feat = combined.transpose(1, 2)
        
        # 4. Bidirectional Context
        temporal_feat = self.bidir_context(temporal_feat)  # [B, 192, T]
        
        # 4.5. Compute multi-scale difference signals for Router 1
        # KEY: Ratios distinguish transition types better than raw magnitudes
        # - Cuts: High diff1, low diff4/diff16 → Low ratios
        # - Graduals: Sustained change → High diff4/diff1 and diff16/diff4 ratios  
        # - Normal motion: Random or spatial change → Inconsistent ratios
        diff1_mag = torch.mean(torch.abs(x_diff1), dim=(2, 3, 4), keepdim=False)  # [B, T]
        diff4_mag = torch.mean(torch.abs(x_diff4), dim=(2, 3, 4), keepdim=False)  # [B, T]
        diff16_mag = torch.mean(torch.abs(x_diff16), dim=(2, 3, 4), keepdim=False)  # [B, T]
        
        # Compute smoothness ratios with clamping to prevent NaN
        # Clamp denominators to avoid division by tiny numbers
        eps = 1e-3  # Larger epsilon for stability
        ratio_4_1 = diff4_mag / torch.clamp(diff1_mag, min=eps)   # [B, T]
        ratio_16_4 = diff16_mag / torch.clamp(diff4_mag, min=eps)  # [B, T]
        
        # Clamp ratios to reasonable range [0, 10] to prevent explosions
        ratio_4_1 = torch.clamp(ratio_4_1, 0, 10)
        ratio_16_4 = torch.clamp(ratio_16_4, 0, 10)
        
        # Stack: [diff1, diff4, diff16, ratio_4/1, ratio_16/4]
        diff_signals = torch.stack([diff1_mag, diff4_mag, diff16_mag, ratio_4_1, ratio_16_4], dim=1)  # [B, 5, T]
        
        # 5. Parallel Independent Experts - All process every frame
        cut_output = self.cut_expert(temporal_feat)        # [B, 192, T]
        gradual_output = self.gradual_expert(temporal_feat)  # [B, 192, T]
        
        # Concatenate diff signals with expert outputs for heads
        cut_input = torch.cat([cut_output, diff_signals], dim=1)  # [B, 192+5, T]
        gradual_input = torch.cat([gradual_output, diff_signals], dim=1)  # [B, 192+5, T]
        
        # 5.5 MULTI-SCALE TEMPORAL AGGREGATION (ENHANCED!)
        # Cut: Single scale (cuts are instant)
        cut_input = self.cut_temporal_pool(cut_input)  # [B, 192+5, T] - sees ±2 frames
        
        # Gradual: Multi-scale fusion (short, medium, long, very long graduals)
        # - Short (5 frames, ±2):  For 2-5 frame graduals
        # - Mid (9 frames, ±4):    For 6-15 frame graduals  
        # - Long (17 frames, ±8):  For 16-24 frame graduals
        # - VLong (25 frames, ±12): For 25-40+ frame graduals (fixes WRONG_EXPERT failures)
        grad_short = torch.nn.functional.gelu(self.gradual_pool_short(gradual_input))   # [B, 197, T]
        grad_mid = torch.nn.functional.gelu(self.gradual_pool_mid(gradual_input))       # [B, 197, T]
        grad_long = torch.nn.functional.gelu(self.gradual_pool_long(gradual_input))     # [B, 197, T]
        grad_vlong = torch.nn.functional.gelu(self.gradual_pool_vlong(gradual_input))   # [B, 197, T] NEW!
        
        # Concatenate all 4 scales and fuse with residual connection for stability
        grad_multiscale = torch.cat([grad_short, grad_mid, grad_long, grad_vlong], dim=1)  # [B, 197*4, T]
        grad_fused = self.gradual_scale_fusion(grad_multiscale)  # [B, 197, T]
        gradual_input = gradual_input + grad_fused  # Residual connection - prevents oscillation!
        
        # 6. UNIFIED 3-CLASS PREDICTION
        # Concatenate outputs from both experts (each has expert features + diff signals)
        combined_features = torch.cat([cut_input, gradual_input], dim=1)  # [B, (192+5)*2, T]
        
        # Single unified head sees features from BOTH experts simultaneously
        # This allows learning complex decision boundaries that binary heads couldn't
        logits = self.unified_head(combined_features)  # [B, 3, T]
        
        # Transpose to [B, T, 3] for loss computation
        logits = logits.transpose(1, 2)
        
        return logits

def create_model():
    return TurboSBD()
