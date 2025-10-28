"""
Cross-Transformer fusion strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from .base import BaseFusionStrategy, FusionStrategyRegistry


class CrossTransformerFusion(BaseFusionStrategy):
    """
    Strategy 4: Cross-Transformer Fusion
    
    Uses cross-modal attention layers where each modality attends to others,
    enabling context-dependent adaptive fusion.
    """
    
    def __init__(self, feature_dims: Dict[str, int],
                 hidden_dim: int = 512,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 device: str = "auto"):
        """
        Initialize cross-transformer fusion.
        
        Args:
            feature_dims: Dictionary mapping feature names to dimensions
            hidden_dim: Hidden dimension for transformers
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            device: Device to run on
        """
        super().__init__(feature_dims, device)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Model components
        self.input_projections = None
        self.cross_attention_layers = None
        self.output_projection = None
        
        self._output_dim = hidden_dim
        
    def initialize(self) -> None:
        """Initialize the cross-transformer model."""
        # Input projections for each modality
        self.input_projections = nn.ModuleDict()
        for name, dim in self.feature_dims.items():
            self.input_projections[name] = nn.Linear(dim, self.hidden_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = CrossAttentionLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                num_modalities=len(self.feature_dims)
            )
            self.cross_attention_layers.append(layer)
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Move to device
        self.input_projections = self.input_projections.to(self.device)
        self.cross_attention_layers = self.cross_attention_layers.to(self.device)
        self.output_projection = self.output_projection.to(self.device)
        
        self.is_initialized = True
    
    def fuse_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse features using cross-transformer attention.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Fused feature array of shape (N, hidden_dim)
        """
        if not self.is_initialized:
            self.initialize()
        
        self._validate_features(features)
        
        with torch.no_grad():
            # Project each modality to hidden dimension
            projected_features = {}
            
            for name, feat in features.items():
                if name in self.input_projections:
                    feat_tensor = torch.from_numpy(feat).float().to(self.device)
                    projected = self.input_projections[name](feat_tensor)
                    projected_features[name] = projected
            
            if not projected_features:
                # Fallback if no features available
                batch_size = len(list(features.values())[0])
                return np.zeros((batch_size, self.hidden_dim))
            
            # Stack modalities for cross-attention
            # Shape: (batch_size, num_modalities, hidden_dim)
            modality_stack = torch.stack(list(projected_features.values()), dim=1)
            
            # Apply cross-attention layers
            attended_features = modality_stack
            for layer in self.cross_attention_layers:
                attended_features = layer(attended_features)
            
            # Global pooling across modalities
            fused_features = torch.mean(attended_features, dim=1)  # (batch_size, hidden_dim)
            
            # Final projection
            output = self.output_projection(fused_features)
            
            # Normalize
            output = F.normalize(output, p=2, dim=1)
            
            return output.cpu().numpy()
    
    def compute_distances(self, fused_features: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between consecutive frames.
        
        Args:
            fused_features: Fused features of shape (N, hidden_dim)
            
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
        return "cross_transformer"
    
    @property
    def output_dim(self) -> int:
        """Return the dimensionality of the fused features."""
        return self._output_dim


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer where each modality attends to all others.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, num_modalities: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        
        # Multi-head attention for cross-modal interactions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cross-attention.
        
        Args:
            x: Input tensor of shape (batch_size, num_modalities, hidden_dim)
            
        Returns:
            Output tensor of same shape
        """
        batch_size, num_modalities, hidden_dim = x.shape
        
        # Reshape for attention: (batch_size * num_modalities, 1, hidden_dim)
        x_reshaped = x.view(batch_size * num_modalities, 1, hidden_dim)
        
        # Create queries, keys, and values
        # Each modality queries all modalities (including itself)
        queries = x_reshaped  # (batch_size * num_modalities, 1, hidden_dim)
        
        # Keys and values are all modalities for each batch
        keys_values = x.unsqueeze(1).repeat(1, num_modalities, 1, 1)  # (batch_size, num_modalities, num_modalities, hidden_dim)
        keys_values = keys_values.view(batch_size * num_modalities, num_modalities, hidden_dim)
        
        # Apply cross-attention
        attended, _ = self.cross_attention(queries, keys_values, keys_values)
        attended = attended.squeeze(1)  # (batch_size * num_modalities, hidden_dim)
        
        # Reshape back
        attended = attended.view(batch_size, num_modalities, hidden_dim)
        
        # Residual connection and layer norm
        x = self.layer_norm1(x + self.dropout(attended))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        
        # Second residual connection and layer norm
        x = self.layer_norm2(x + ffn_output)
        
        return x


# Register the strategy
FusionStrategyRegistry.register("cross_transformer", CrossTransformerFusion)