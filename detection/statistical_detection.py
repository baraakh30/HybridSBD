"""
Statistical detection pipeline with dynamic thresholding.
"""

import numpy as np
from scipy import signal, ndimage
from scipy.signal import savgol_filter, find_peaks
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseDetectionMethod, DetectionMethodRegistry


class StatisticalDetection(BaseDetectionMethod):
    """
    Advanced statistical detection pipeline with:
    - Dynamic thresholding system
    - Multi-scale analysis
    - Adaptive window management
    - Content-aware filtering
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize statistical detection.
        
        Args:
            config: Configuration dictionary containing:
                - smoothing_method: 'gaussian' or 'savgol'
                - smoothing_window: Window size for smoothing
                - normalization_type: 'none', 'minmax', 'zscore', 'rolling_zscore', 'robust'
                - z_score_window: Window size for rolling z-score normalization
                - short_scale, medium_scale, long_scale: Multi-scale windows
                - threshold_method: 'fixed', 'adaptive', 'percentile'
                - fixed_threshold: Fixed threshold value
                - adaptive_beta: Adaptive threshold multiplier
                - min_shot_duration: Minimum shot duration
                - temporal_tolerance: Temporal tolerance for boundaries
        """
        super().__init__(config)
        
        # Smoothing parameters
        self.smoothing_method = config.get("smoothing_method", "gaussian")
        self.smoothing_window = config.get("smoothing_window", 5)
        
        # Normalization parameters
        self.normalization_type = config.get("normalization_type", "rolling_zscore")
        self.z_score_window = config.get("z_score_window", 30)
        
        # Multi-scale analysis
        self.short_scale = config.get("short_scale", 15)
        self.medium_scale = config.get("medium_scale", 45)
        self.long_scale = config.get("long_scale", 75)
        
        # Thresholding parameters
        self.threshold_method = config.get("threshold_method", "adaptive")
        self.fixed_threshold = config.get("fixed_threshold", 0.5)
        self.adaptive_beta = config.get("adaptive_beta", 1.5)
        
        # Post-processing parameters
        self.min_shot_duration = config.get("min_shot_duration", 30)
        self.temporal_tolerance = config.get("temporal_tolerance", 5)
        
    def initialize(self) -> None:
        """Initialize the detection method."""
        self.is_initialized = True
    
    def detect_boundaries(self, distance_signal: np.ndarray) -> List[int]:
        """
        Detect shot boundaries using statistical analysis.
        
        Args:
            distance_signal: Frame-to-frame distance signal
            
        Returns:
            List of boundary frame indices
        """
        if len(distance_signal) == 0:
            return []
        
        # Step 1: Signal preprocessing
        processed_signal = self._preprocess_signal(distance_signal)
        
        # Step 2: Multi-scale analysis
        scale_results = self._multi_scale_analysis(processed_signal)
        
        # Step 3: Dynamic thresholding
        boundaries = self._dynamic_thresholding(processed_signal, scale_results)
        
        # Step 4: Post-processing
        boundaries = self._post_process_boundaries(
            boundaries, len(distance_signal), self.min_shot_duration
        )
        
        return boundaries
    
    def _preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess the distance signal.
        
        Args:
            signal: Raw distance signal
            
        Returns:
            Preprocessed signal
        """
        # Apply smoothing
        if self.smoothing_method == "gaussian":
            smoothed = ndimage.gaussian_filter1d(signal, sigma=self.smoothing_window/3)
        elif self.smoothing_method == "savgol":
            window_length = min(self.smoothing_window, len(signal))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                smoothed = savgol_filter(signal, window_length, 2)
            else:
                smoothed = signal.copy()
        else:
            smoothed = signal.copy()
        
        # Apply normalization based on type
        normalized = self._normalize_signal(smoothed)
        
        return normalized
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal based on configured normalization type.
        
        Args:
            signal: Input signal (after smoothing)
            
        Returns:
            Normalized signal
        """
        if self.normalization_type == "none":
            # No normalization - use raw signal
            return signal.copy()
        
        elif self.normalization_type == "minmax":
            # Min-max normalization to [0, 1]
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val - min_val > 1e-8:
                return (signal - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(signal)
        
        elif self.normalization_type == "zscore":
            # Global z-score normalization
            mean = np.mean(signal)
            std = np.std(signal)
            if std > 1e-8:
                return (signal - mean) / std
            else:
                return np.zeros_like(signal)
        
        elif self.normalization_type == "rolling_zscore":
            # Rolling z-score (original behavior)
            return self._rolling_z_score(signal, self.z_score_window)
        
        elif self.normalization_type == "robust":
            # Robust normalization using median and IQR
            median = np.median(signal)
            q75, q25 = np.percentile(signal, [75, 25])
            iqr = q75 - q25
            if iqr > 1e-8:
                return (signal - median) / iqr
            else:
                return np.zeros_like(signal)
        
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")
    
    def _rolling_z_score(self, signal: np.ndarray, window: int) -> np.ndarray:
        """
        Apply rolling z-score normalization.
        
        Args:
            signal: Input signal
            window: Window size for rolling statistics
            
        Returns:
            Z-score normalized signal
        """
        if len(signal) <= window:
            # Fallback to global z-score for short signals
            mean = np.mean(signal)
            std = np.std(signal)
            return (signal - mean) / (std + 1e-8)
        
        z_scores = np.zeros_like(signal)
        
        for i in range(len(signal)):
            start = max(0, i - window // 2)
            end = min(len(signal), i + window // 2 + 1)
            
            window_data = signal[start:end]
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            z_scores[i] = (signal[i] - mean) / (std + 1e-8)
        
        return z_scores
    
    def _multi_scale_analysis(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform multi-scale analysis on the signal.
        
        Args:
            signal: Preprocessed signal
            
        Returns:
            Dictionary with analysis results for each scale
        """
        scales = {
            "short": self.short_scale,
            "medium": self.medium_scale,
            "long": self.long_scale
        }
        
        results = {}
        
        for scale_name, scale_size in scales.items():
            if scale_size < len(signal):
                # Compute local statistics
                local_means = self._compute_local_stats(signal, scale_size, np.mean)
                local_stds = self._compute_local_stats(signal, scale_size, np.std)
                local_maxs = self._compute_local_stats(signal, scale_size, np.max)
                
                results[scale_name] = {
                    "means": local_means,
                    "stds": local_stds,
                    "maxs": local_maxs,
                    "scale": scale_size
                }
            else:
                # Handle case where scale is larger than signal
                results[scale_name] = {
                    "means": np.array([np.mean(signal)]),
                    "stds": np.array([np.std(signal)]),
                    "maxs": np.array([np.max(signal)]),
                    "scale": scale_size
                }
        
        return results
    
    def _compute_local_stats(self, signal: np.ndarray, window: int, 
                           stat_func) -> np.ndarray:
        """
        Compute local statistics using a sliding window.
        
        Args:
            signal: Input signal
            window: Window size
            stat_func: Statistical function to apply
            
        Returns:
            Array of local statistics
        """
        stats = []
        
        for i in range(len(signal)):
            start = max(0, i - window // 2)
            end = min(len(signal), i + window // 2 + 1)
            
            window_data = signal[start:end]
            stat_value = stat_func(window_data)
            stats.append(stat_value)
        
        return np.array(stats)
    
    def _dynamic_thresholding(self, signal: np.ndarray, 
                            scale_results: Dict[str, np.ndarray]) -> List[int]:
        """
        Apply dynamic thresholding to detect boundaries.
        
        Args:
            signal: Preprocessed signal
            scale_results: Multi-scale analysis results
            
        Returns:
            List of boundary indices
        """
        if self.threshold_method == "fixed":
            threshold = self.fixed_threshold
            boundaries = np.where(signal > threshold)[0].tolist()
            
        elif self.threshold_method == "adaptive":
            boundaries = self._adaptive_thresholding(signal, scale_results)
            
        elif self.threshold_method == "percentile":
            threshold = np.percentile(signal, 95)
            boundaries = np.where(signal > threshold)[0].tolist()
            
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        
        return boundaries
    
    def _adaptive_thresholding(self, signal: np.ndarray, 
                             scale_results: Dict[str, np.ndarray]) -> List[int]:
        """
        Adaptive thresholding using multi-scale information.
        
        Args:
            signal: Preprocessed signal
            scale_results: Multi-scale analysis results
            
        Returns:
            List of boundary indices
        """
        # Compute adaptive threshold based on local statistics
        adaptive_threshold = np.zeros_like(signal)
        
        for i in range(len(signal)):
            # Use medium scale for threshold computation
            if "medium" in scale_results:
                local_mean = scale_results["medium"]["means"][min(i, len(scale_results["medium"]["means"]) - 1)]
                local_std = scale_results["medium"]["stds"][min(i, len(scale_results["medium"]["stds"]) - 1)]
                adaptive_threshold[i] = local_mean + self.adaptive_beta * local_std
            else:
                # Fallback to global statistics
                adaptive_threshold[i] = np.mean(signal) + self.adaptive_beta * np.std(signal)
        
        # Find peaks that exceed adaptive threshold
        peaks, properties = find_peaks(
            signal,
            height=adaptive_threshold,
            distance=self.min_shot_duration // 2,
            prominence=0.1
        )
        
        return peaks.tolist()
    
    def _content_aware_filtering(self, boundaries: List[int], 
                               signal: np.ndarray) -> List[int]:
        """
        Apply content-aware filtering to remove false positives.
        
        Args:
            boundaries: Raw boundary detections
            signal: Distance signal
            
        Returns:
            Filtered boundaries
        """
        if not boundaries:
            return boundaries
        
        filtered = []
        
        for boundary in boundaries:
            # Check if boundary has sufficient prominence
            if self._check_boundary_prominence(boundary, signal):
                # Check temporal consistency
                if self._check_temporal_consistency(boundary, boundaries):
                    filtered.append(boundary)
        
        return filtered
    
    def _check_boundary_prominence(self, boundary: int, signal: np.ndarray, 
                                  min_prominence: float = 0.2) -> bool:
        """
        Check if a boundary has sufficient prominence.
        
        Args:
            boundary: Boundary index
            signal: Distance signal
            min_prominence: Minimum required prominence
            
        Returns:
            True if boundary has sufficient prominence
        """
        if boundary >= len(signal):
            return False
        
        boundary_value = signal[boundary]
        
        # Check local neighborhood
        start = max(0, boundary - 10)
        end = min(len(signal), boundary + 11)
        
        local_signal = signal[start:end]
        local_max = np.max(local_signal)
        local_min = np.min(local_signal)
        
        if local_max == local_min:
            return False
        
        prominence = (boundary_value - local_min) / (local_max - local_min)
        
        return prominence >= min_prominence
    
    def _check_temporal_consistency(self, boundary: int, 
                                  all_boundaries: List[int]) -> bool:
        """
        Check temporal consistency of a boundary.
        
        Args:
            boundary: Boundary to check
            all_boundaries: All detected boundaries
            
        Returns:
            True if boundary is temporally consistent
        """
        # For now, just check minimum distance constraint
        # In a full implementation, this could include more sophisticated checks
        
        nearby_boundaries = [
            b for b in all_boundaries 
            if b != boundary and abs(b - boundary) < self.min_shot_duration
        ]
        
        return len(nearby_boundaries) == 0
    
    @property
    def name(self) -> str:
        """Return the name of the detection method."""
        return "statistical"


# Register the method
DetectionMethodRegistry.register("statistical", StatisticalDetection)