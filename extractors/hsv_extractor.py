"""
HSV Histogram Feature Extractor.
"""

# Configure OpenCV environment variables
import os
os.environ.setdefault('OPENCV_FFMPEG_READ_ATTEMPTS', '16384')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

import cv2
import numpy as np
import torch
from scipy import ndimage
from typing import Union

from .base import BaseFeatureExtractor, FeatureExtractorRegistry


class HSVHistogramExtractor(BaseFeatureExtractor):
    """
    HSV Histogram Feature Extractor.
    
    Extracts 768-dimensional HSV histogram features:
    - H channel: 256 bins
    - S channel: 256 bins  
    - V channel: 256 bins
    """
    
    def __init__(self, bins: int = 256, smoothing: bool = True, 
                 device: str = "auto"):
        """
        Initialize HSV histogram extractor.
        
        Args:
            bins: Number of bins per channel
            smoothing: Whether to apply Gaussian smoothing
            device: Device to run on
        """
        super().__init__(device)
        self.bins = bins
        self.smoothing = smoothing
        self._feature_dim = bins * 3  # H, S, V channels
        
    def initialize(self) -> None:
        """Initialize the extractor."""
        self.is_initialized = True
    
    def extract_features(self, frames: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract HSV histogram features from frames.
        
        Args:
            frames: Input frames of shape (N, H, W, C) in RGB format
            
        Returns:
            Feature array of shape (N, 768)
        """
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        
        # Ensure frames are in uint8 format
        if frames.dtype != np.uint8:
            frames = (frames * 255).astype(np.uint8)
        
        features = []
        
        for frame in frames:
            # Convert RGB to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Extract histograms for each channel
            h_hist = cv2.calcHist([hsv_frame], [0], None, [self.bins], [0, 256])
            s_hist = cv2.calcHist([hsv_frame], [1], None, [self.bins], [0, 256])
            v_hist = cv2.calcHist([hsv_frame], [2], None, [self.bins], [0, 256])
            
            # Flatten histograms
            h_hist = h_hist.flatten()
            s_hist = s_hist.flatten()
            v_hist = v_hist.flatten()
            
            # Apply Gaussian smoothing if enabled
            if self.smoothing:
                h_hist = ndimage.gaussian_filter1d(h_hist, sigma=1.0)
                s_hist = ndimage.gaussian_filter1d(s_hist, sigma=1.0)
                v_hist = ndimage.gaussian_filter1d(v_hist, sigma=1.0)
            
            # Normalize histograms
            h_hist = h_hist / (np.sum(h_hist) + 1e-8)
            s_hist = s_hist / (np.sum(s_hist) + 1e-8)
            v_hist = v_hist / (np.sum(v_hist) + 1e-8)
            
            # Concatenate all histograms
            feature_vector = np.concatenate([h_hist, s_hist, v_hist])
            features.append(feature_vector)
        
        return np.array(features)
    
    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of the extracted features."""
        return self._feature_dim
    
    @property
    def name(self) -> str:
        """Return the name of the feature extractor."""
        return "hsv_histogram"


# Register the extractor
FeatureExtractorRegistry.register("hsv_histogram", HSVHistogramExtractor)