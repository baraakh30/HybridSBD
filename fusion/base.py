"""
Base fusion strategy interface.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict, List, Union, Optional


class BaseFusionStrategy(ABC):
    """Base class for all fusion strategies."""
    
    def __init__(self, feature_dims: Dict[str, int], device: str = "auto"):
        """
        Initialize the fusion strategy.
        
        Args:
            feature_dims: Dictionary mapping feature names to their dimensions
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.feature_dims = feature_dims
        self.device = self._get_device(device)
        self.is_initialized = False
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the fusion strategy."""
        pass
    
    @abstractmethod
    def fuse_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse multiple feature representations.
        
        Args:
            features: Dictionary mapping feature names to feature arrays
                     Each array has shape (N, feature_dim)
        
        Returns:
            Fused feature array of shape (N, fused_dim)
        """
        pass
    
    @abstractmethod
    def compute_distances(self, fused_features: np.ndarray) -> np.ndarray:
        """
        Compute frame-to-frame distances from fused features.
        
        Args:
            fused_features: Fused features of shape (N, fused_dim)
            
        Returns:
            Distance array of shape (N-1,)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the fusion strategy."""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the dimensionality of the fused features."""
        pass
    
    def __call__(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Convenience method to fuse features and compute distances."""
        if not self.is_initialized:
            self.initialize()
        
        fused_features = self.fuse_features(features)
        distances = self.compute_distances(fused_features)
        return distances
    
    def _validate_features(self, features: Dict[str, np.ndarray]) -> None:
        """Validate input features."""
        if not features:
            raise ValueError("No features provided")
        
        # Check feature dimensions and shapes
        feature_lengths = {}
        for name, feat_array in features.items():
            if name not in self.feature_dims:
                raise ValueError(f"Unknown feature type: {name}")
            
            expected_dim = self.feature_dims[name]
            actual_dim = feat_array.shape[1] if len(feat_array.shape) > 1 else feat_array.shape[0]
            
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Feature {name} has dimension {actual_dim}, expected {expected_dim}"
                )
            
            # Track feature lengths for alignment check
            feature_lengths[name] = feat_array.shape[0]
        
        # Check that all features have the same number of frames
        lengths = list(feature_lengths.values())
        if len(set(lengths)) > 1:
            feature_info = ", ".join([f"{name}: {length}" for name, length in feature_lengths.items()])
            raise ValueError(
                f"Feature arrays have mismatched lengths: {feature_info}. "
                f"All feature extractors must produce the same number of features per video."
            )


class FusionStrategyRegistry:
    """Registry for fusion strategies."""
    
    _strategies = {}
    
    @classmethod
    def register(cls, name: str, strategy_class):
        """Register a fusion strategy."""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def get(cls, name: str, **kwargs):
        """Get a fusion strategy by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown fusion strategy: {name}")
        return cls._strategies[name](**kwargs)
    
    @classmethod
    def list_strategies(cls):
        """List available fusion strategies."""
        return list(cls._strategies.keys())