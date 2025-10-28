"""
Concatenation-based fusion strategy.
"""

import numpy as np
from typing import Dict

from .base import BaseFusionStrategy, FusionStrategyRegistry


class ConcatenationFusion(BaseFusionStrategy):
    """
    Strategy 1: Concatenation-Based Fusion
    
    Concatenates all feature representations and computes Euclidean distance.
    Simple baseline approach with feature-wise normalization.
    """
    
    def __init__(self, feature_dims: Dict[str, int], device: str = "auto"):
        """
        Initialize concatenation fusion.
        
        Args:
            feature_dims: Dictionary mapping feature names to dimensions
            device: Device to run on
        """
        super().__init__(feature_dims, device)
        self._output_dim = sum(feature_dims.values())
        
    def initialize(self) -> None:
        """Initialize the fusion strategy."""
        self.is_initialized = True
    
    def fuse_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse features by concatenation.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Concatenated feature array of shape (N, total_dim)
        """
        self._validate_features(features)
        
        # Normalize each feature type
        normalized_features = []
        
        for name in sorted(features.keys()):  # Sort for consistent ordering
            feat = features[name]
            
            # L2 normalization per feature vector
            feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
            normalized_features.append(feat_norm)
        
        # Concatenate all features
        fused_features = np.concatenate(normalized_features, axis=1)
        
        return fused_features
    
    def compute_distances(self, fused_features: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between consecutive frames.
        
        Args:
            fused_features: Fused features of shape (N, fused_dim)
            
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
        return "concatenation"
    
    @property
    def output_dim(self) -> int:
        """Return the dimensionality of the fused features."""
        return self._output_dim


# Register the strategy
FusionStrategyRegistry.register("concatenation", ConcatenationFusion)