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
                 device: str = "auto", frame_sampling_rate: int = 1,
                 interpolation_method: str = "linear"):
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
        # Sampling and interpolation
        self.frame_sampling_rate = max(1, int(frame_sampling_rate))
        self.interpolation_method = interpolation_method
        
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
        # Delegate to batch-aware implementation which supports sampling
        return self.extract_batch(frames)

    def extract_batch(self, frames: Union[np.ndarray, torch.Tensor], batch_size: int = 32) -> np.ndarray:
        """
        Extract features with optional frame sampling and interpolation to match original frame count.
        """
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        # Ensure frames are in uint8 format
        if frames.dtype != np.uint8:
            frames = (frames * 255).astype(np.uint8)

        total_frames = len(frames)

        # If sampling rate == 1, compute histograms for all frames
        if self.frame_sampling_rate <= 1 or total_frames <= 1:
            return self._compute_histograms(frames)

        # Sample frames
        sampled_indices = np.arange(0, total_frames, self.frame_sampling_rate)
        sampled_frames = frames[sampled_indices]

        # Compute histograms for sampled frames
        sampled_features = self._compute_histograms(sampled_frames)

        # If only one sampled feature, replicate
        if sampled_features.shape[0] == 1:
            return np.tile(sampled_features[0], (total_frames, 1))

        # Interpolate to match original frame count
        interpolated = self._interpolate_features(sampled_features, sampled_indices, total_frames)
        return interpolated

    def _compute_histograms(self, frames: np.ndarray) -> np.ndarray:
        """Compute normalized HSV histograms for given frames (no sampling logic here)."""
        features = []
        for frame in frames:
            # Convert RGB to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            # Extract histograms for each channel
            h_hist = cv2.calcHist([hsv_frame], [0], None, [self.bins], [0, 256]).flatten()
            s_hist = cv2.calcHist([hsv_frame], [1], None, [self.bins], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv_frame], [2], None, [self.bins], [0, 256]).flatten()

            # Apply Gaussian smoothing if enabled
            if self.smoothing:
                h_hist = ndimage.gaussian_filter1d(h_hist, sigma=1.0)
                s_hist = ndimage.gaussian_filter1d(s_hist, sigma=1.0)
                v_hist = ndimage.gaussian_filter1d(v_hist, sigma=1.0)

            # Normalize histograms
            h_hist = h_hist / (np.sum(h_hist) + 1e-8)
            s_hist = s_hist / (np.sum(s_hist) + 1e-8)
            v_hist = v_hist / (np.sum(v_hist) + 1e-8)

            feature_vector = np.concatenate([h_hist, s_hist, v_hist])
            features.append(feature_vector.astype(np.float32))

        return np.array(features)

    def _interpolate_features(self, sampled_features: np.ndarray, sampled_indices: np.ndarray,
                              total_frames: int) -> np.ndarray:
        """Interpolate sampled histogram features to produce per-frame features."""
        # Ensure shapes
        M, D = sampled_features.shape
        target_indices = np.arange(total_frames)
        interpolated = np.zeros((total_frames, D), dtype=np.float32)

        try:
            from scipy.interpolate import interp1d
            for dim in range(D):
                if self.interpolation_method == 'linear':
                    f = interp1d(sampled_indices, sampled_features[:, dim], kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                elif self.interpolation_method == 'cubic' and len(sampled_indices) >= 4:
                    f = interp1d(sampled_indices, sampled_features[:, dim], kind='cubic',
                                 bounds_error=False, fill_value='extrapolate')
                else:
                    f = interp1d(sampled_indices, sampled_features[:, dim], kind='nearest',
                                 bounds_error=False, fill_value='extrapolate')
                interpolated[:, dim] = f(target_indices)
        except Exception:
            # Fallback to numpy.interp per-dimension
            for dim in range(D):
                interpolated[:, dim] = np.interp(target_indices, sampled_indices, sampled_features[:, dim])

        return interpolated
    
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