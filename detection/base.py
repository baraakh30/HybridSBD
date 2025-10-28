"""
Base detection method interface.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any


class BaseDetectionMethod(ABC):
    """Base class for all detection methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detection method.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the detection method."""
        pass
    
    @abstractmethod
    def detect_boundaries(self, distance_signal: np.ndarray) -> List[int]:
        """
        Detect shot boundaries from distance signal.
        
        Args:
            distance_signal: Frame-to-frame distance signal
            
        Returns:
            List of boundary frame indices
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the detection method."""
        pass
    
    def __call__(self, distance_signal: np.ndarray) -> List[int]:
        """Convenience method to detect boundaries."""
        if not self.is_initialized:
            self.initialize()
        return self.detect_boundaries(distance_signal)
    
    def _post_process_boundaries(self, boundaries: List[int], 
                               signal_length: int,
                               min_shot_duration: int = 30) -> List[int]:
        """
        Post-process detected boundaries.
        
        Args:
            boundaries: Raw boundary detections
            signal_length: Length of the original signal
            min_shot_duration: Minimum shot duration in frames
            
        Returns:
            Filtered boundary list
        """
        if not boundaries:
            return boundaries
        
        # Sort boundaries
        boundaries = sorted(set(boundaries))
        
        # Filter boundaries that are too close together
        filtered_boundaries = []
        last_boundary = -min_shot_duration
        
        for boundary in boundaries:
            if boundary - last_boundary >= min_shot_duration:
                if boundary < signal_length:  # Ensure boundary is within signal
                    filtered_boundaries.append(boundary)
                    last_boundary = boundary
        
        return filtered_boundaries


class DetectionMethodRegistry:
    """Registry for detection methods."""
    
    _methods = {}
    
    @classmethod
    def register(cls, name: str, method_class):
        """Register a detection method."""
        cls._methods[name] = method_class
    
    @classmethod
    def get(cls, name: str, **kwargs):
        """Get a detection method by name."""
        if name not in cls._methods:
            raise ValueError(f"Unknown detection method: {name}")
        return cls._methods[name](**kwargs)
    
    @classmethod
    def list_methods(cls):
        """List available detection methods."""
        return list(cls._methods.keys())