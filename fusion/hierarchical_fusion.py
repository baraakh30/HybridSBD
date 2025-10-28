"""
Hierarchical fusion strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from .base import BaseFusionStrategy, FusionStrategyRegistry


class HierarchicalFusion(BaseFusionStrategy):
    """
    Strategy 5: Three-Level Hierarchical Fusion
    
    Implements progressive fusion through three levels:
    1. Low-level: HSV + Optical Flow → 256-dim
    2. Mid-level: CNN + Low-level → 512-dim  
    3. High-level: CLIP + Mid-level → 512-dim
    """
    
    def __init__(self, feature_dims: Dict[str, int],
                 conv_kernel: int = 3,
                 attention_heads: int = 4,
                 device: str = "auto"):
        """
        Initialize hierarchical fusion.
        
        Args:
            feature_dims: Dictionary mapping feature names to dimensions
            conv_kernel: Kernel size for 1D convolution
            attention_heads: Number of attention heads
            device: Device to run on
        """
        super().__init__(feature_dims, device)
        self.conv_kernel = conv_kernel
        self.attention_heads = attention_heads
        
        # Expected feature names and their roles
        self.low_level_features = ["hsv_histogram", "optical_flow"]
        self.mid_level_feature = "cnn"
        self.high_level_feature = "clip"
        
        # Model components
        self.level1_fusion = None  # Low-level fusion (HSV + Optical Flow)
        self.level2_fusion = None  # Mid-level fusion (CNN + Level1)
        self.level3_fusion = None  # High-level fusion (CLIP + Level2)
        
        self._output_dim = 512  # Final output dimension
        
    def initialize(self) -> None:
        """Initialize the hierarchical fusion model."""
        # Level 1: Low-level fusion (HSV + Optical Flow)
        level1_input_dim = sum(
            self.feature_dims[name] for name in self.low_level_features 
            if name in self.feature_dims
        )
        self.level1_fusion = LowLevelFusion(
            input_dim=level1_input_dim,
            output_dim=256,
            conv_kernel=self.conv_kernel
        )
        
        # Level 2: Mid-level fusion (CNN + Level1)
        cnn_dim = self.feature_dims.get(self.mid_level_feature, 2048)
        self.level2_fusion = MidLevelFusion(
            cnn_dim=cnn_dim,
            low_level_dim=256,
            output_dim=512,
            num_heads=self.attention_heads
        )
        
        # Level 3: High-level fusion (CLIP + Level2)
        clip_dim = self.feature_dims.get(self.high_level_feature, 512)
        self.level3_fusion = HighLevelFusion(
            clip_dim=clip_dim,
            mid_level_dim=512,
            output_dim=512
        )
        
        # Move to device
        self.level1_fusion = self.level1_fusion.to(self.device)
        self.level2_fusion = self.level2_fusion.to(self.device)
        self.level3_fusion = self.level3_fusion.to(self.device)
        
        self.is_initialized = True
    
    def fuse_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse features using hierarchical fusion.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Fused feature array of shape (N, 512)
        """
        if not self.is_initialized:
            self.initialize()
        
        self._validate_features(features)
        
        with torch.no_grad():
            # Level 1: Low-level fusion (HSV + Optical Flow)
            low_level_inputs = []
            for name in self.low_level_features:
                if name in features:
                    feat_tensor = torch.from_numpy(features[name]).float().to(self.device)
                    low_level_inputs.append(feat_tensor)
            
            if low_level_inputs:
                low_level_concat = torch.cat(low_level_inputs, dim=1)
                level1_output = self.level1_fusion(low_level_concat)
            else:
                # Fallback if low-level features not available
                level1_output = torch.zeros((len(list(features.values())[0]), 256)).to(self.device)
            
            # Level 2: Mid-level fusion (CNN + Level1)
            if self.mid_level_feature in features:
                cnn_tensor = torch.from_numpy(features[self.mid_level_feature]).float().to(self.device)
                level2_output = self.level2_fusion(cnn_tensor, level1_output)
            else:
                # Fallback if CNN features not available
                level2_output = level1_output
                if level2_output.shape[1] != 512:
                    # Adjust dimension if needed
                    level2_output = F.adaptive_avg_pool1d(
                        level2_output.unsqueeze(1), 512
                    ).squeeze(1)
            
            # Level 3: High-level fusion (CLIP + Level2)
            if self.high_level_feature in features:
                clip_tensor = torch.from_numpy(features[self.high_level_feature]).float().to(self.device)
                final_output = self.level3_fusion(clip_tensor, level2_output)
            else:
                # Fallback if CLIP features not available
                final_output = level2_output
            
            # Normalize final output
            final_output = F.normalize(final_output, p=2, dim=1)
            
            return final_output.cpu().numpy()
    
    def compute_distances(self, fused_features: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between consecutive frames.
        
        Args:
            fused_features: Fused features of shape (N, 512)
            
        Returns:
            Distance array of shape (N-1,)
        """
        if len(fused_features) < 2:
            return np.array([0.0])
        
        # Compute frame-to-frame differences
        diff = fused_features[1:] - fused_features[:-1]
        
        # Compute Euclidean distances
        distances = np.linalg.norm(diff, axis=1)
        
        return distances
    
    @property
    def name(self) -> str:
        """Return the name of the fusion strategy."""
        return "hierarchical"
    
    @property
    def output_dim(self) -> int:
        """Return the dimensionality of the fused features."""
        return self._output_dim


class LowLevelFusion(nn.Module):
    """Low-level fusion module (HSV + Optical Flow)."""
    
    def __init__(self, input_dim: int, output_dim: int, conv_kernel: int = 3):
        super().__init__()
        
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2
        )
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for low-level fusion.
        
        Args:
            x: Input tensor of shape (N, input_dim)
            
        Returns:
            Output tensor of shape (N, output_dim)
        """
        # Reshape for 1D convolution: (N, input_dim) -> (N, input_dim, 1)
        x = x.unsqueeze(2)
        
        # Apply 1D convolution
        x = self.conv1d(x)
        
        # Remove extra dimension and apply activation
        x = x.squeeze(2)  # (N, output_dim)
        x = self.activation(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class MidLevelFusion(nn.Module):
    """Mid-level fusion module (CNN + Low-level)."""
    
    def __init__(self, cnn_dim: int, low_level_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.cnn_projection = nn.Linear(cnn_dim, output_dim)
        self.low_level_projection = nn.Linear(low_level_dim, output_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, cnn_features: torch.Tensor, low_level_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for mid-level fusion.
        
        Args:
            cnn_features: CNN features of shape (N, cnn_dim)
            low_level_features: Low-level features of shape (N, low_level_dim)
            
        Returns:
            Output tensor of shape (N, output_dim)
        """
        # Project to same dimension
        cnn_proj = self.cnn_projection(cnn_features)  # (N, output_dim)
        low_proj = self.low_level_projection(low_level_features)  # (N, output_dim)
        
        # Cross-attention: CNN as query, low-level as key/value
        query = cnn_proj.unsqueeze(1)  # (N, 1, output_dim)
        key_value = low_proj.unsqueeze(1)  # (N, 1, output_dim)
        
        attended, _ = self.cross_attention(query, key_value, key_value)
        attended = attended.squeeze(1)  # (N, output_dim)
        
        # Residual connection and normalization
        output = self.layer_norm(cnn_proj + attended)
        
        return output


class HighLevelFusion(nn.Module):
    """High-level fusion module (CLIP + Mid-level)."""
    
    def __init__(self, clip_dim: int, mid_level_dim: int, output_dim: int):
        super().__init__()
        
        self.clip_projection = nn.Linear(clip_dim, output_dim)
        self.mid_level_projection = nn.Linear(mid_level_dim, output_dim)
        
        # Gated fusion mechanism
        self.gate_network = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, clip_features: torch.Tensor, mid_level_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for high-level fusion.
        
        Args:
            clip_features: CLIP features of shape (N, clip_dim)
            mid_level_features: Mid-level features of shape (N, mid_level_dim)
            
        Returns:
            Output tensor of shape (N, output_dim)
        """
        # Project to same dimension
        clip_proj = self.clip_projection(clip_features)  # (N, output_dim)
        mid_proj = self.mid_level_projection(mid_level_features)  # (N, output_dim)
        
        # Gated fusion
        combined = torch.cat([clip_proj, mid_proj], dim=1)  # (N, output_dim * 2)
        gate_weights = self.gate_network(combined)  # (N, output_dim)
        
        # Apply gating: G ⊙ CLIP + (1-G) ⊙ MidLevel
        gated_output = gate_weights * clip_proj + (1 - gate_weights) * mid_proj
        
        # Layer normalization
        output = self.layer_norm(gated_output)
        
        return output


# Register the strategy
FusionStrategyRegistry.register("hierarchical", HierarchicalFusion)