"""
Optical Flow Feature Extractor.
"""

# Configure OpenCV environment variables
import os
os.environ.setdefault('OPENCV_FFMPEG_READ_ATTEMPTS', '16384')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

import cv2
import numpy as np
import torch
from typing import Union, Optional
import logging

from .base import BaseFeatureExtractor, FeatureExtractorRegistry


class OpticalFlowExtractor(BaseFeatureExtractor):
    """
    Optical Flow Feature Extractor.
    
    Extracts 40-dimensional optical flow features using histogram of optical flow (HOF):
    - 8 direction bins (45° each)
    - 5 magnitude bins (logarithmic scale)
    Supports frame sampling to reduce computational cost for long videos.
    """
    
    def __init__(self, method: str = "farneback", device: str = "auto",
                 frame_sampling_rate: int = 1, interpolation_method: str = "linear"):
        """
        Initialize optical flow extractor.
        
        Args:
            method: Optical flow method ('farneback', 'lucas_kanade')
            device: Device to run on
            frame_sampling_rate: Process every Nth frame pair (1 = all frames, 3 = every 3rd frame)
            interpolation_method: Method to interpolate features ('linear', 'nearest', 'cubic')
        """
        super().__init__(device)
        self.method = method
        self._feature_dim = 40  # 8 directions × 5 magnitudes
        self.prev_frame = None
        self.frame_sampling_rate = max(1, frame_sampling_rate)
        self.interpolation_method = interpolation_method
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self) -> None:
        """Initialize the optical flow extractor."""
        
        self.is_initialized = True
    
    def extract_batch(self, frames: Union[np.ndarray, torch.Tensor], 
                     batch_size: int = 32) -> np.ndarray:
        """
        Override batch processing for optical flow to maintain temporal continuity with sampling.
        
        Args:
            frames: Input frames
            batch_size: Ignored for optical flow (must process sequentially)
            
        Returns:
            Feature array aligned with input frame count (total_frames, feature_dim)
        """
        if not self.is_initialized:
            self.initialize()
        
        total_frames = len(frames)
        
        # Sample frame pairs if sampling rate > 1
        if self.frame_sampling_rate > 1:
            # Sample frames but maintain temporal relationships
            sampled_indices = np.arange(0, total_frames, self.frame_sampling_rate)
            # Ensure we have at least 2 frames for optical flow
            if len(sampled_indices) < 2:
                sampled_indices = np.array([0, total_frames - 1]) if total_frames > 1 else np.array([0])
            
            sampled_frames = frames[sampled_indices]
            self.logger.info(f"Optical Flow: Sampling {len(sampled_frames)} out of {total_frames} frames (rate: {self.frame_sampling_rate})")
            
            # Extract features from sampled frames
            sampled_features = self.extract_features(sampled_frames)
            
            # Interpolate features to match original frame count
            if len(sampled_features) > 0:
                features = self._interpolate_features(sampled_features, sampled_indices, total_frames)
            else:
                features = np.zeros((total_frames, self._feature_dim))
        else:
            # Process all frames sequentially  
            raw_features = self.extract_features(frames)
            
            # Pad optical flow features to match frame count
            # Optical flow produces N-1 features for N frames, so we pad with zeros at the beginning
            if len(raw_features) == total_frames - 1:
                zero_feature = np.zeros((1, self._feature_dim))
                features = np.concatenate([zero_feature, raw_features], axis=0)
            elif len(raw_features) == total_frames:
                features = raw_features
            else:
                # Handle edge case by padding/truncating
                features = np.zeros((total_frames, self._feature_dim))
                copy_len = min(len(raw_features), total_frames)
                features[:copy_len] = raw_features[:copy_len]
        
        # Final check to ensure correct dimensions
        assert features.shape[0] == total_frames, \
            f"Optical flow: Expected {total_frames} features, got {features.shape[0]}"
        
        return features

    def extract_features(self, frames: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract optical flow features from frame sequences.
        
        Args:
            frames: Input frames of shape (N, H, W, C) in RGB format
            
        Returns:
            Feature array of shape (N-1, 40) for N input frames
        """
        # Input validation
        if frames is None:
            raise ValueError("Input frames cannot be None")
        
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        
        if len(frames.shape) != 4:
            raise ValueError(f"Expected 4D input (N, H, W, C), got shape {frames.shape}")
        
        if frames.shape[0] < 2:
            # Not enough frames for optical flow, return zeros
            return np.zeros((max(1, frames.shape[0] - 1), self._feature_dim))
        
        # Ensure frames are in uint8 format
        if frames.dtype != np.uint8:
            if frames.dtype == np.float32 or frames.dtype == np.float64:
                # Assume normalized [0,1] values
                frames = np.clip(frames, 0, 1)
                frames = (frames * 255).astype(np.uint8)
            else:
                # Try to convert safely
                frames = frames.astype(np.uint8)
        
        features = []
        
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            # Handle frame dimension mismatches by resizing if needed
            if current_frame.shape != next_frame.shape:
                # Determine target size (use the larger dimensions for each axis)
                target_shape = tuple(max(dim1, dim2) for dim1, dim2 in zip(current_frame.shape, next_frame.shape))
                
                # Resize frames to match target shape
                if current_frame.shape != target_shape:
                    if len(current_frame.shape) == 3:
                        current_frame = cv2.resize(current_frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                    else:  # 2D grayscale
                        current_frame = cv2.resize(current_frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                
                if next_frame.shape != target_shape:
                    if len(next_frame.shape) == 3:
                        next_frame = cv2.resize(next_frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                    else:  # 2D grayscale
                        next_frame = cv2.resize(next_frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Convert to grayscale
            if len(current_frame.shape) == 3:
                if current_frame.shape[2] == 3:  # RGB
                    gray1 = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
                elif current_frame.shape[2] == 4:  # RGBA
                    gray1 = cv2.cvtColor(current_frame, cv2.COLOR_RGBA2GRAY)
                    gray2 = cv2.cvtColor(next_frame, cv2.COLOR_RGBA2GRAY)
                else:
                    raise ValueError(f"Unsupported number of channels: {current_frame.shape[2]}")
            else:
                gray1 = current_frame
                gray2 = next_frame
            
            # Handle shape mismatches by resizing to match the larger frame
            if gray1.shape != gray2.shape:
                # Determine target size (use the larger dimensions)
                target_height = max(gray1.shape[0], gray2.shape[0])
                target_width = max(gray1.shape[1], gray2.shape[1])
                target_size = (target_width, target_height)
                
                # Resize frames to match target size
                if gray1.shape[:2] != (target_height, target_width):
                    gray1 = cv2.resize(gray1, target_size, interpolation=cv2.INTER_LINEAR)
                if gray2.shape[:2] != (target_height, target_width):
                    gray2 = cv2.resize(gray2, target_size, interpolation=cv2.INTER_LINEAR)
                    
            if self.method == "farneback":
                # Use Farneback dense optical flow
                flow_dense = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
            elif self.method == "lucas_kanade":
                # Use Lucas-Kanade sparse optical flow with corner detection
                corners = cv2.goodFeaturesToTrack(
                    gray1,
                    maxCorners=100,
                    qualityLevel=0.3,
                    minDistance=7,
                    blockSize=7
                )
                
                if corners is not None:
                    # Track features using Lucas-Kanade
                    next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                        gray1, gray2, corners, None,
                        winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )
                    
                    # Convert sparse to dense approximation
                    flow_dense = self._sparse_to_dense_flow((corners, next_corners, status), gray1.shape)
                else:
                    # No corners found, use zero flow
                    flow_dense = np.zeros((gray1.shape[0], gray1.shape[1], 2), dtype=np.float32)
            else:
                # Use Farneback method as default
                flow_dense = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
            
            # Validate flow output
            if flow_dense is None:
                flow_dense = np.zeros((gray1.shape[0], gray1.shape[1], 2), dtype=np.float32)
            
            # Extract HOF features
            hof_features = self._compute_hof(flow_dense)
            features.append(hof_features)
        
        # Handle case where we have only one frame
        if len(features) == 0:
            features.append(np.zeros(self._feature_dim))
        
        return np.array(features)
    
    def _sparse_to_dense_flow(self, sparse_flow_data, shape):
        """Convert sparse optical flow to dense approximation with better interpolation."""
        corners, next_corners, status = sparse_flow_data
        flow_dense = np.zeros((shape[0], shape[1], 2), dtype=np.float32)
        
        if corners is not None and next_corners is not None:
            # Get valid tracked points
            good_old = corners[status == 1]
            good_new = next_corners[status == 1]
            
            if len(good_old) > 0:
                # Create flow vectors
                flow_vectors = good_new - good_old
                
                # Create coordinate grids
                y_coords, x_coords = np.mgrid[0:shape[0], 0:shape[1]]
                
                # Get coordinates of valid points
                old_points = good_old.reshape(-1, 2)
                
                if len(old_points) >= 3:
                    # Use scipy for proper interpolation if enough points
                    try:
                        from scipy.interpolate import griddata
                        
                        # Interpolate flow components separately
                        flow_u = griddata(
                            old_points, flow_vectors[:, 0, 0], 
                            (x_coords, y_coords), method='linear', fill_value=0
                        )
                        flow_v = griddata(
                            old_points, flow_vectors[:, 0, 1], 
                            (x_coords, y_coords), method='linear', fill_value=0
                        )
                        
                        flow_dense[:, :, 0] = flow_u
                        flow_dense[:, :, 1] = flow_v
                        
                    except ImportError:
                        # Fallback to simple nearest neighbor if scipy not available
                        for i, (old, flow_vec) in enumerate(zip(old_points, flow_vectors)):
                            x, y = int(old[0]), int(old[1])
                            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                                # Apply flow in a small neighborhood
                                for dy in range(-2, 3):
                                    for dx in range(-2, 3):
                                        ny, nx = y + dy, x + dx
                                        if 0 <= nx < shape[1] and 0 <= ny < shape[0]:
                                            # Weight by distance
                                            weight = 1.0 / (1.0 + np.sqrt(dx*dx + dy*dy))
                                            flow_dense[ny, nx, 0] += weight * flow_vec[0, 0]
                                            flow_dense[ny, nx, 1] += weight * flow_vec[0, 1]
                else:
                    # Too few points, use simple assignment
                    for i, (old, flow_vec) in enumerate(zip(old_points, flow_vectors)):
                        x, y = int(old[0]), int(old[1])
                        if 0 <= x < shape[1] and 0 <= y < shape[0]:
                            flow_dense[y, x, 0] = flow_vec[0, 0]
                            flow_dense[y, x, 1] = flow_vec[0, 1]
        
        return flow_dense
    
    def _compute_hof(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute Histogram of Optical Flow (HOF) features.
        
        Args:
            flow: Dense optical flow of shape (H, W, 2)
            
        Returns:
            HOF feature vector of shape (40,)
        """
        if flow is None or len(flow.shape) != 3:
            return np.zeros(self._feature_dim)
        
        # Extract flow components
        u = flow[:, :, 0]  # Horizontal component
        v = flow[:, :, 1]  # Vertical component
        
        # Compute magnitude and angle
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u)
        
        # Convert angle from [-π, π] to [0, 2π]
        angle = np.where(angle < 0, angle + 2 * np.pi, angle)
        
        # Define bins
        angle_bins = 8  # 8 directions (45° each)
        magnitude_bins = 5  # 5 magnitude levels
        
        # Flatten arrays for easier processing
        magnitude_flat = magnitude.flatten()
        angle_flat = angle.flatten()
        
        # Apply threshold for noise
        valid_mask = magnitude_flat > 0.1
        if not np.any(valid_mask):
            return np.zeros(self._feature_dim)
            
        magnitude_valid = magnitude_flat[valid_mask]
        angle_valid = angle_flat[valid_mask]
        
        # Define magnitude thresholds with better handling
        mag_max = np.max(magnitude_valid)
        if mag_max < 1e-6:  # Handle very small flows
            return np.zeros(self._feature_dim)
        
        # Use percentile-based binning for more robust thresholds
        mag_percentiles = np.percentile(magnitude_valid, [0, 20, 40, 60, 80, 100])
        mag_thresholds = mag_percentiles
        
        # Ensure unique thresholds
        mag_thresholds = np.unique(mag_thresholds)
        if len(mag_thresholds) < magnitude_bins + 1:
            # Fall back to linear spacing if not enough unique values
            mag_thresholds = np.linspace(0, mag_max, magnitude_bins + 1)
        
        # Use np.histogram2d for efficient computation
        angle_edges = np.linspace(0, 2 * np.pi, angle_bins + 1)
        
        # Create 2D histogram
        hof, _, _ = np.histogram2d(
            angle_valid, magnitude_valid,
            bins=[angle_edges, mag_thresholds[:magnitude_bins + 1]]
        )
        
        # Flatten and normalize histogram
        hof_flat = hof.flatten()
        hof_norm = hof_flat / (np.sum(hof_flat) + 1e-8)
        
        return hof_norm
    
    def _interpolate_features(self, sampled_features: np.ndarray, sampled_indices: np.ndarray, 
                             total_frames: int) -> np.ndarray:
        """
        Interpolate optical flow features to match the original frame count.
        
        Args:
            sampled_features: Features extracted from sampled frames (M, feature_dim)
            sampled_indices: Indices of sampled frames (frame pairs start indices)
            total_frames: Total number of original frames
            
        Returns:
            Interpolated features (total_frames, feature_dim) - same as other extractors
        """
        if len(sampled_features) == 1:
            # Only one sample, replicate it for all frames
            return np.tile(sampled_features[0], (total_frames, 1))
        
        # For optical flow, we need to create features that align with frames
        # Optical flow produces features for frame pairs, but we want frame-aligned features
        target_indices = np.arange(total_frames)
        interpolated_features = np.zeros((total_frames, sampled_features.shape[1]))
        
        try:
            from scipy.interpolate import interp1d
            
            # Map sampled optical flow features to frame indices
            # If we sampled every Nth frame, the optical flow features correspond to those frame pairs
            flow_frame_indices = sampled_indices[:-1] if len(sampled_indices) > 1 else sampled_indices
            
            # Handle edge case where we only have one flow feature
            if len(sampled_features) == 1 or len(flow_frame_indices) == 1:
                return np.tile(sampled_features[0], (total_frames, 1))
            
            # Create interpolation function for each feature dimension
            for dim in range(sampled_features.shape[1]):
                if self.interpolation_method == 'linear':
                    f = interp1d(flow_frame_indices, sampled_features[:, dim], 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
                elif self.interpolation_method == 'cubic':
                    if len(flow_frame_indices) >= 4:
                        f = interp1d(flow_frame_indices, sampled_features[:, dim], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')
                    else:
                        f = interp1d(flow_frame_indices, sampled_features[:, dim], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
                else:  # nearest
                    f = interp1d(flow_frame_indices, sampled_features[:, dim], 
                               kind='nearest', bounds_error=False, fill_value='extrapolate')
                
                interpolated_features[:, dim] = f(target_indices)
                
        except ImportError:
            # Fallback: simple linear interpolation without scipy
            self.logger.warning("Scipy not available, using simple interpolation")
            flow_frame_indices = sampled_indices[:-1] if len(sampled_indices) > 1 else sampled_indices
            
            for dim in range(sampled_features.shape[1]):
                interpolated_features[:, dim] = np.interp(
                    target_indices, flow_frame_indices, sampled_features[:, dim]
                )
        
        # Ensure we return exactly the right number of features
        assert interpolated_features.shape[0] == total_frames, \
            f"Expected {total_frames} features, got {interpolated_features.shape[0]}"
        
        return interpolated_features
    
    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of the extracted features."""
        return self._feature_dim
    
    @property
    def name(self) -> str:
        """Return the name of the feature extractor."""
        return f"optical_flow_{self.method}"


# Register the extractor
FeatureExtractorRegistry.register("optical_flow", OpticalFlowExtractor)