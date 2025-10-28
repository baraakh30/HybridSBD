"""
Base feature extractor interface.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Union, Tuple, Optional


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the feature extractor.
        
        Args:
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
        """
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
        """Initialize the feature extractor (load models, etc.)."""
        pass
    
    @abstractmethod
    def extract_features(self, frames: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract features from frames.
        
        Args:
            frames: Input frames of shape (N, H, W, C) or (N, C, H, W)
            
        Returns:
            Feature array of shape (N, feature_dim)
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimensionality of the extracted features."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the feature extractor."""
        pass
    
    def __call__(self, frames: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convenience method to extract features."""
        if not self.is_initialized:
            self.initialize()
        return self.extract_features(frames)
    
    def extract_batch(self, frames: Union[np.ndarray, torch.Tensor], 
                     batch_size: int = 32) -> np.ndarray:
        """
        Extract features in batches to handle memory constraints.
        
        Args:
            frames: Input frames
            batch_size: Batch size for processing
            
        Returns:
            Feature array
        """
        if not self.is_initialized:
            self.initialize()
            
        features_list = []
        num_frames = len(frames)
        
        for i in range(0, num_frames, batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_features = self.extract_features(batch_frames)
            features_list.append(batch_features)
        
        return np.concatenate(features_list, axis=0)


class FeatureExtractorRegistry:
    """Registry for feature extractors."""
    
    _extractors = {}
    
    @classmethod
    def register(cls, name: str, extractor_class):
        """Register a feature extractor."""
        cls._extractors[name] = extractor_class
    
    @classmethod
    def get(cls, name: str, **kwargs):
        """Get a feature extractor by name."""
        if name not in cls._extractors:
            raise ValueError(f"Unknown feature extractor: {name}")
        return cls._extractors[name](**kwargs)
    
    @classmethod
    def list_extractors(cls):
        """List available feature extractors."""
        return list(cls._extractors.keys())