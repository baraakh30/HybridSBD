"""
Transformer-based boundary detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import math
from tqdm import tqdm
import sys
import os

# Add parent directory to path for focal_loss import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from focal_loss import FocalLoss, TverskyLoss, FocalTverskyLoss, SoftF1Loss, ComboLossV2

from .base import BaseDetectionMethod, DetectionMethodRegistry

# Try to import scipy for peak detection
try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample - SOTA regularization for transformers.
    Randomly drops entire residual branches during training.
    """
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with batches of any shape
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class TransformerDetection(BaseDetectionMethod):
    """
    Transformer-based boundary detection using sequence modeling.
    
    SOTA Features:
    - Frame difference features for clearer boundary signal
    - Causal OR Bidirectional attention (configurable)
    - NMS Peak Detection for clean output
    - Batch Inference for speed
    - Center-Frame Prediction for reliability
    - Magnitude skip connection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize transformer detection.
        
        Args:
            config: Configuration dictionary containing:
                - input_dim: Input feature dimension
                - hidden_dim: Hidden dimension (default: 512)
                - num_layers: Number of transformer layers (default: 6)
                - num_heads: Number of attention heads (default: 8)
                - ff_dim: Feed-forward dimension (default: 2048)
                - dropout: Dropout rate (default: 0.1)
                - sequence_length: Input sequence length (default: 100)
                - confidence_threshold: Classification threshold (default: 0.3)
                - use_causal_attention: Use causal (True) or bidirectional (False) attention
                - use_nms: Use Non-Maximum Suppression for peak detection
                - nms_distance: Minimum distance between peaks for NMS
                - use_center_frame: Only trust center region predictions
                - center_ratio: Ratio of center to trust (default: 0.2 = 20%)
                - batch_size_inference: Batch size for inference (default: 32)
        """
        super().__init__(config)
        
        # Model architecture parameters
        self.input_dim = config.get("input_dim", 512)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_layers = config.get("num_layers", 4)  # Reduced from 6: TCN+LSTM provide local context
        self.num_heads = config.get("num_heads", 8)
        self.ff_dim = config.get("ff_dim", 1024)  # Reduced from 2048 for efficiency
        self.dropout = config.get("dropout", 0.25)  # INCREASED from 0.15 to prevent overfitting
        
        # Processing parameters
        self.sequence_length = config.get("sequence_length", 100)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)  # INCREASED from 0.3
        
        # SOTA Enhancement: Attention type (causal vs bidirectional)
        self.use_causal_attention = config.get("use_causal_attention", False)  # Default: bidirectional for better context
        
        # SOTA Enhancement: NMS Peak Detection
        self.use_nms = config.get("use_nms", True)
        self.nms_distance = config.get("nms_distance", 15)  # Minimum frames between detected boundaries
        
        # SOTA Enhancement: Center-Frame Prediction
        self.use_center_frame = config.get("use_center_frame", True)
        self.center_ratio = config.get("center_ratio", 0.2)  # Trust middle 20% of window
        
        # SOTA Enhancement: Batch Inference
        self.batch_size_inference = config.get("batch_size_inference", 32)
        
        # SOTA Enhancement: DropPath (Stochastic Depth) for better generalization
        self.drop_path_rate = config.get("drop_path_rate", 0.2)  # INCREASED from 0.1
        
        # Model components
        self.model = None
        self.device = self._get_device(config.get("device", "auto"))
        
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
    
    def initialize(self) -> None:
        """Initialize the transformer model."""
        self.model = BoundaryDetectionTransformer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            max_sequence_length=self.sequence_length,
            use_causal_attention=self.use_causal_attention,  # Configurable!
            use_difference_features=True,
            use_magnitude_skip=True,
            drop_path_rate=self.drop_path_rate,  # Stochastic depth for generalization
            use_multi_scale=True  # CRITICAL: Multi-scale for gradual transition detection
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.is_initialized = True
    
    def detect_boundaries(self, features: np.ndarray) -> List[int]:
        """
        Detect boundaries from feature sequences with SOTA enhancements.
        
        Uses:
        - Batch inference for speed
        - FULL COVERAGE prediction (no center-only restriction) 
        - NMS peak detection for clean output
        
        Args:
            features: Feature array of shape (N, feature_dim)
            
        Returns:
            List of boundary frame indices
        """
        if len(features) < 2:
            return []
        
        # Step 1: Collect all windows for batch processing
        # CRITICAL FIX: Use overlapping windows to ensure FULL COVERAGE
        # Previous issue: center_ratio=0.2 with stride=20 left edge frames uncovered
        windows = []
        window_starts = []
        
        # Use smaller stride for better coverage (50% overlap)
        stride = max(1, self.sequence_length // 2)
        
        for start_idx in range(0, len(features) - self.sequence_length + 1, stride):
            end_idx = start_idx + self.sequence_length
            windows.append(features[start_idx:end_idx])
            window_starts.append(start_idx)
        
        # Handle edge case: ensure we cover the end of the video
        if len(features) >= self.sequence_length:
            last_start = len(features) - self.sequence_length
            if not window_starts or window_starts[-1] != last_start:
                windows.append(features[last_start:])
                window_starts.append(last_start)
        
        # Handle very short videos
        if not windows and len(features) >= 2:
            # Pad to sequence_length
            padded = np.zeros((self.sequence_length, features.shape[1]))
            padded[:len(features)] = features
            windows.append(padded)
            window_starts.append(0)
        
        if not windows:
            return []
        
        # Step 2: Batch inference
        all_probs = self._batch_predict(windows)
        
        # Step 3: Aggregate probabilities with FULL COVERAGE
        # CRITICAL FIX: Use ALL predictions, not just center 20%
        # This ensures edge frames (0-10, last 10) are covered
        full_probs = np.zeros(len(features) - 1)
        prob_counts = np.zeros(len(features) - 1)
        
        for i, (start_idx, probs) in enumerate(zip(window_starts, all_probs)):
            # FULL COVERAGE: Use all predictions from each window
            # The overlapping windows will average predictions for middle frames
            # and edge frames will get predictions from their respective edge windows
            for j in range(len(probs)):
                global_idx = start_idx + j
                if 0 <= global_idx < len(full_probs):
                    full_probs[global_idx] += probs[j]
                    prob_counts[global_idx] += 1
        
        # Average overlapping predictions
        # CRITICAL: Check for uncovered frames (should not happen with new strategy)
        uncovered = np.where(prob_counts == 0)[0]
        if len(uncovered) > 0:
            # Fallback: interpolate from neighbors for any uncovered frames
            for idx in uncovered:
                left_idx = idx - 1
                right_idx = idx + 1
                vals = []
                if left_idx >= 0 and prob_counts[left_idx] > 0:
                    vals.append(full_probs[left_idx] / prob_counts[left_idx])
                if right_idx < len(full_probs) and prob_counts[right_idx] > 0:
                    vals.append(full_probs[right_idx] / prob_counts[right_idx])
                if vals:
                    full_probs[idx] = np.mean(vals)
                    prob_counts[idx] = 1
        
        prob_counts[prob_counts == 0] = 1  # Avoid division by zero
        full_probs = full_probs / prob_counts
        
        # Step 4: Apply NMS peak detection or threshold
        if self.use_nms and HAS_SCIPY:
            boundaries = self._nms_peak_detection(full_probs)
        else:
            # Fallback: simple thresholding
            boundaries = list(np.where(full_probs > self.confidence_threshold)[0])
        
        # Step 5: Post-process boundaries
        boundaries = self._post_process_boundaries(
            boundaries, len(features), self.config.get("min_shot_duration", 30)
        )
        
        return boundaries
    
    def _batch_predict(self, windows: List[np.ndarray], return_duration: bool = False) -> List[np.ndarray]:
        """
        Batch inference for multiple windows.
        
        Args:
            windows: List of feature windows
            return_duration: If True, also return duration predictions
            
        Returns:
            List of probability arrays (and optionally duration arrays)
        """
        all_probs = []
        all_durations = [] if return_duration else None
        
        with torch.no_grad():
            # Process in batches
            for batch_start in range(0, len(windows), self.batch_size_inference):
                batch_end = min(batch_start + self.batch_size_inference, len(windows))
                batch_windows = windows[batch_start:batch_end]
                
                # Stack into batch tensor
                batch_tensor = torch.from_numpy(np.stack(batch_windows)).float()
                batch_tensor = batch_tensor.to(self.device)
                
                # Forward pass - use multi-task mode if returning duration
                if return_duration and hasattr(self.model, 'use_multi_task') and self.model.use_multi_task:
                    outputs = self.model(batch_tensor, mode='frame_multi_task')
                    if isinstance(outputs, dict):
                        logits = outputs['class_logits']
                        duration_pred = outputs['duration_pred']
                        probs = torch.sigmoid(logits).cpu().numpy()
                        durations = torch.expm1(duration_pred).cpu().numpy()  # Convert from log scale
                    else:
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        durations = None
                else:
                    logits = self.model(batch_tensor, mode='frame')
                    probs = torch.sigmoid(logits).cpu().numpy()
                    durations = None
                
                # Extract each window's predictions
                for i in range(len(batch_windows)):
                    all_probs.append(probs[i, :, 0])
                    if return_duration and durations is not None:
                        all_durations.append(durations[i, :, 0])
        
        if return_duration:
            return all_probs, all_durations
        return all_probs
    
    def detect_boundaries_with_types(self, features: np.ndarray) -> List[dict]:
        """
        Detect boundaries with transition type information (cut vs gradual).
        
        Uses multi-task predictions to classify each detected boundary as:
        - 'cut': duration ≈ 1 (sharp, instant transition)
        - 'gradual': duration > 1 (dissolve, fade, wipe)
        
        Args:
            features: Feature array of shape (N, feature_dim)
            
        Returns:
            List of dictionaries with:
            - 'frame': boundary frame index
            - 'type': 'cut' or 'gradual'
            - 'duration': predicted transition duration in frames
            - 'confidence': boundary confidence score
        """
        if len(features) < 2:
            return []
        
        # Collect windows (same as detect_boundaries)
        windows = []
        window_starts = []
        stride = max(1, self.sequence_length // 2)
        
        for start_idx in range(0, len(features) - self.sequence_length + 1, stride):
            end_idx = start_idx + self.sequence_length
            windows.append(features[start_idx:end_idx])
            window_starts.append(start_idx)
        
        if len(features) >= self.sequence_length:
            last_start = len(features) - self.sequence_length
            if not window_starts or window_starts[-1] != last_start:
                windows.append(features[last_start:])
                window_starts.append(last_start)
        
        if not windows and len(features) >= 2:
            padded = np.zeros((self.sequence_length, features.shape[1]))
            padded[:len(features)] = features
            windows.append(padded)
            window_starts.append(0)
        
        if not windows:
            return []
        
        # Batch inference with duration predictions
        result = self._batch_predict(windows, return_duration=True)
        if isinstance(result, tuple):
            all_probs, all_durations = result
        else:
            all_probs = result
            all_durations = None
        
        # Aggregate probabilities and durations
        full_probs = np.zeros(len(features) - 1)
        full_durations = np.zeros(len(features) - 1)
        prob_counts = np.zeros(len(features) - 1)
        
        for i, start_idx in enumerate(window_starts):
            probs = all_probs[i]
            durations = all_durations[i] if all_durations else None
            
            for j in range(len(probs)):
                global_idx = start_idx + j
                if 0 <= global_idx < len(full_probs):
                    full_probs[global_idx] += probs[j]
                    if durations is not None:
                        full_durations[global_idx] += durations[j]
                    prob_counts[global_idx] += 1
        
        # Average
        prob_counts[prob_counts == 0] = 1
        full_probs = full_probs / prob_counts
        full_durations = full_durations / prob_counts
        
        # Find boundaries using NMS
        if self.use_nms and HAS_SCIPY:
            boundary_indices = self._nms_peak_detection(full_probs)
        else:
            boundary_indices = list(np.where(full_probs > self.confidence_threshold)[0])
        
        # Post-process
        boundary_indices = self._post_process_boundaries(
            boundary_indices, len(features), self.config.get("min_shot_duration", 30)
        )
        
        # Build result with type information
        results = []
        for idx in boundary_indices:
            duration = full_durations[idx] if all_durations else 1.0
            transition_type = 'gradual' if duration > 1.5 else 'cut'
            
            results.append({
                'frame': idx,
                'type': transition_type,
                'duration': float(duration),
                'confidence': float(full_probs[idx])
            })
        
        return results
    
    def _nms_peak_detection(self, probs: np.ndarray) -> List[int]:
        """
        Apply Non-Maximum Suppression to find clean boundary peaks.
        
        Args:
            probs: Probability array for all frames
            
        Returns:
            List of boundary indices (peaks)
        """
        # Find peaks that are:
        # 1. Above threshold
        # 2. Local maxima
        # 3. Separated by at least nms_distance frames
        peaks, properties = find_peaks(
            probs,
            height=self.confidence_threshold,
            distance=self.nms_distance,
            prominence=0.1  # Peak must stand out from neighbors
        )
        
        return peaks.tolist()
    
    def _predict_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict boundary probabilities for a single sequence.
        (Kept for backward compatibility)
        """
        with torch.no_grad():
            sequence_tensor = torch.from_numpy(sequence).float()
            sequence_tensor = sequence_tensor.unsqueeze(0).to(self.device)
            logits = self.model(sequence_tensor, mode='frame')
            probs = torch.sigmoid(logits).squeeze()
            return probs.cpu().numpy()
    
    def load_model(self, model_path: str) -> None:
        """Load a pre-trained model."""
        if not self.is_initialized:
            self.initialize()
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def save_model(self, model_path: str, optimizer=None, epoch=None, loss=None) -> None:
        """Save the current model."""
        if not self.is_initialized:
            raise RuntimeError("Model must be initialized before saving")
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "loss": loss
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, model_path)
        print(f"Saved model to {model_path}")
    
    @property
    def name(self) -> str:
        """Return the name of the detection method."""
        return "transformer"


class MultiScaleDifferenceLayer(nn.Module):
    """
    Multi-scale difference layer for detecting BOTH hard cuts AND gradual transitions.
    
    CRITICAL FOR GRADUAL TRANSITIONS:
    - Hard cuts: Single-frame spike in difference magnitude
    - Dissolves/Fades: Gradual increase in cumulative difference over 5-30 frames
    
    This layer computes differences at multiple scales:
    - Scale 1: Frame-to-frame (catches hard cuts)
    - Scale 3: Every 3rd frame (catches fast dissolves)
    - Scale 5: Every 5th frame (catches medium dissolves)
    - Scale 10: Every 10th frame (catches slow fades)
    """
    
    def __init__(self, input_dim: int, scales: List[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.scales = scales or [1, 3, 5, 10]  # Multi-scale time steps
        
        # Each scale produces: difference + magnitude + normalized_magnitude
        features_per_scale = input_dim + 2
        self.output_dim = features_per_scale * len(self.scales) + 4  # +4 for combined stats
        
        # Learnable scale weights (learn which scales are most important)
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))
        
        # Fusion MLP to combine multi-scale features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.output_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        all_scale_features = []
        all_magnitudes = []
        
        for scale in self.scales:
            if scale >= seq_len:
                # Scale too large, use zeros
                diff = torch.zeros(batch_size, seq_len - 1, self.input_dim, device=x.device, dtype=x.dtype)
                magnitude = torch.zeros(batch_size, seq_len - 1, 1, device=x.device, dtype=x.dtype)
            else:
                # Compute difference at this scale: F[t+scale] - F[t]
                diff = x[:, scale:, :] - x[:, :-scale, :]  # (batch, seq_len-scale, input_dim)
                
                # Pad to consistent length (seq_len - 1)
                if diff.size(1) < seq_len - 1:
                    pad_size = seq_len - 1 - diff.size(1)
                    # Pad at the beginning (early frames don't have long-range history)
                    diff = F.pad(diff, (0, 0, pad_size, 0), mode='constant', value=0)
                elif diff.size(1) > seq_len - 1:
                    diff = diff[:, :seq_len - 1, :]
                
                magnitude = torch.norm(diff, dim=-1, keepdim=True)
            
            # Normalize magnitude relative to local context
            local_mean = F.avg_pool1d(
                magnitude.transpose(1, 2),
                kernel_size=min(7, seq_len - 1),
                stride=1,
                padding=min(3, (seq_len - 2) // 2)
            ).transpose(1, 2)
            if local_mean.size(1) != magnitude.size(1):
                local_mean = local_mean[:, :magnitude.size(1), :]
            normalized_mag = magnitude / (local_mean + 1e-8)
            
            # Combine: [diff, magnitude, normalized_magnitude]
            scale_features = torch.cat([diff, magnitude, normalized_mag], dim=-1)
            all_scale_features.append(scale_features)
            all_magnitudes.append(magnitude)
        
        # Combine all scales
        multi_scale = torch.cat(all_scale_features, dim=-1)
        
        # Compute combined statistics across scales
        stacked_mags = torch.stack(all_magnitudes, dim=-1)  # (batch, seq_len-1, 1, num_scales)
        mag_mean = stacked_mags.mean(dim=-1)
        mag_max = stacked_mags.max(dim=-1)[0]
        mag_std = stacked_mags.std(dim=-1)
        
        # Weighted magnitude (learned scale importance)
        weights = F.softmax(self.scale_weights, dim=0)
        weighted_mag = (stacked_mags.squeeze(-2) * weights).sum(dim=-1, keepdim=True)
        
        combined_stats = torch.cat([mag_mean, mag_max, mag_std, weighted_mag], dim=-1)
        
        # Final multi-scale output
        output = torch.cat([multi_scale, combined_stats], dim=-1)
        
        # Apply fusion MLP
        output = self.fusion_mlp(output)
        
        # Return weighted magnitude for skip connection
        return output, seq_len - 1, weighted_mag


class DifferenceLayer(nn.Module):
    """
    Computes frame differences and multi-scale features for boundary detection.
    
    Enhanced with:
    - First and second-order differences
    - Magnitude with adaptive thresholding
    - Neighbor context (local window around each difference)
    - Spike detection features
    """
    
    def __init__(self, input_dim: int, include_magnitude: bool = True, 
                 include_second_order: bool = True, include_neighbor_context: bool = True,
                 neighbor_window: int = 2):
        """
        Initialize the difference layer.
        
        Args:
            input_dim: Dimension of input features
            include_magnitude: Whether to include L2 magnitude of differences
            include_second_order: Whether to include second-order differences (acceleration)
            include_neighbor_context: Whether to include context from neighboring differences
            neighbor_window: Number of neighbors on each side (e.g., 2 = 5 total positions)
        """
        super().__init__()
        self.input_dim = input_dim
        self.include_magnitude = include_magnitude
        self.include_second_order = include_second_order
        self.include_neighbor_context = include_neighbor_context
        self.neighbor_window = neighbor_window
        
        # Calculate output dimension
        self.output_dim = input_dim  # First-order difference
        if include_magnitude:
            self.output_dim += 1  # Add magnitude
            self.output_dim += 1  # Add normalized magnitude (relative to neighbors)
            self.output_dim += 1  # Add spike score
        if include_second_order:
            self.output_dim += input_dim  # Add second-order difference
        if include_neighbor_context:
            # Add compressed neighbor context (not full features, just statistics)
            self.output_dim += 4  # mean, max, min, std of neighbor magnitudes
        
        # Learnable scaling for magnitude (helps with gradient flow)
        self.magnitude_scale = nn.Parameter(torch.ones(1))
        
        # Adaptive magnitude threshold (learned)
        self.magnitude_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Learnable spike sensitivity
        self.spike_sensitivity = nn.Parameter(torch.tensor(2.0))
        
        # Small MLP to compress neighbor context
        if include_neighbor_context:
            self.neighbor_compressor = nn.Sequential(
                nn.Linear(2 * neighbor_window + 1, 16),
                nn.ReLU(),
                nn.Linear(16, 4)
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Compute frame differences with enhanced features.
        
        Args:
            x: Input features of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (differences, output_sequence_length, magnitude):
            - differences: Tensor of shape (batch_size, seq_len-1, output_dim)
            - output_sequence_length: Length of output sequence (seq_len - 1)
            - magnitude: Raw magnitude tensor for skip connection (batch, seq_len-1, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # First-order difference: D_t = F_{t+1} - F_t
        first_order = x[:, 1:, :] - x[:, :-1, :]  # (batch, seq_len-1, input_dim)
        
        components = [first_order]
        
        # Magnitude: ||D_t||
        magnitude = torch.norm(first_order, dim=-1, keepdim=True)  # (batch, seq_len-1, 1)
        raw_magnitude = magnitude.clone()  # Save for skip connection
        
        if self.include_magnitude:
            scaled_magnitude = magnitude * self.magnitude_scale
            components.append(scaled_magnitude)
            
            # Normalized magnitude (relative to local mean)
            # This helps detect spikes relative to surrounding context
            if seq_len > 3:
                # Compute local mean magnitude using sliding window
                local_mean = F.avg_pool1d(
                    magnitude.transpose(1, 2),  # (batch, 1, seq_len-1)
                    kernel_size=min(5, seq_len - 1),
                    stride=1,
                    padding=min(2, (seq_len - 2) // 2)
                ).transpose(1, 2)  # (batch, seq_len-1, 1)
                # Trim to match size
                if local_mean.size(1) > magnitude.size(1):
                    local_mean = local_mean[:, :magnitude.size(1), :]
                elif local_mean.size(1) < magnitude.size(1):
                    padding = torch.zeros(batch_size, magnitude.size(1) - local_mean.size(1), 1, 
                                         device=x.device, dtype=x.dtype)
                    local_mean = torch.cat([local_mean, padding], dim=1)
                normalized_magnitude = magnitude / (local_mean + 1e-8)
            else:
                normalized_magnitude = magnitude
            components.append(normalized_magnitude)
            
            # Spike score: how much this differs from adaptive threshold
            # Higher score = more likely to be a boundary
            spike_score = torch.sigmoid(
                self.spike_sensitivity * (normalized_magnitude - self.magnitude_threshold)
            )
            components.append(spike_score)
        
        # Second-order difference: D2_t = D_{t+1} - D_t (acceleration of change)
        if self.include_second_order:
            if seq_len > 2:
                second_order = first_order[:, 1:, :] - first_order[:, :-1, :]  # (batch, seq_len-2, input_dim)
                # Pad to match first_order length
                padding = torch.zeros(batch_size, 1, self.input_dim, device=x.device, dtype=x.dtype)
                second_order = torch.cat([second_order, padding], dim=1)  # (batch, seq_len-1, input_dim)
            else:
                second_order = torch.zeros_like(first_order)
            components.append(second_order)
        
        # Neighbor context: statistics from surrounding differences
        if self.include_neighbor_context and seq_len > 2 * self.neighbor_window + 1:
            # Pad magnitude for neighbor extraction
            padded_mag = F.pad(
                magnitude.transpose(1, 2),  # (batch, 1, seq_len-1)
                (self.neighbor_window, self.neighbor_window),
                mode='replicate'
            ).transpose(1, 2)  # (batch, seq_len-1+2*window, 1)
            
            # Extract neighbor windows
            neighbor_features = []
            for i in range(seq_len - 1):
                window = padded_mag[:, i:i + 2 * self.neighbor_window + 1, 0]  # (batch, window_size)
                neighbor_features.append(window)
            neighbor_windows = torch.stack(neighbor_features, dim=1)  # (batch, seq_len-1, window_size)
            
            # Compress neighbor context using learned MLP
            neighbor_context = self.neighbor_compressor(neighbor_windows)  # (batch, seq_len-1, 4)
            components.append(neighbor_context)
        elif self.include_neighbor_context:
            # Fallback for short sequences
            neighbor_context = torch.zeros(batch_size, seq_len - 1, 4, device=x.device, dtype=x.dtype)
            components.append(neighbor_context)
        
        # Concatenate all components
        output = torch.cat(components, dim=-1)  # (batch, seq_len-1, output_dim)
        
        return output, seq_len - 1, raw_magnitude


class BoundaryDetectionTransformer(nn.Module):
    """
    Transformer model for boundary detection with:
    - Frame difference features (learns from changes, not raw content)
    - Causal attention (each position only sees past frames)
    - Multi-scale difference features for GRADUAL TRANSITION detection
    - Magnitude skip connection (direct path from difference magnitude to output)
    - Neighbor context for better spike detection
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, 
                 num_layers: int = 6, num_heads: int = 8,
                 ff_dim: int = 2048, dropout: float = 0.15,
                 max_sequence_length: int = 1000,
                 use_causal_attention: bool = True,
                 use_difference_features: bool = True,
                 use_magnitude_skip: bool = True,
                 drop_path_rate: float = 0.1,
                 use_multi_scale: bool = True):  # NEW: Multi-scale for gradual transitions
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.use_causal_attention = use_causal_attention
        self.use_difference_features = use_difference_features
        self.use_magnitude_skip = use_magnitude_skip
        self.drop_path_rate = drop_path_rate
        self.use_multi_scale = use_multi_scale
        
        # CRITICAL FOR GRADUAL TRANSITIONS: Multi-scale difference layer
        # Detects both hard cuts (scale=1) and gradual transitions (scales 3, 5, 10)
        if use_difference_features:
            if use_multi_scale:
                # Multi-scale for gradual transitions
                self.difference_layer = MultiScaleDifferenceLayer(
                    input_dim=input_dim,
                    scales=[1, 3, 5, 10]  # Detect changes at multiple time scales
                )
                projection_input_dim = input_dim  # MultiScale outputs input_dim
            else:
                # Original single-scale (for backward compat)
                self.difference_layer = DifferenceLayer(
                    input_dim=input_dim,
                    include_magnitude=True,
                    include_second_order=True,
                    include_neighbor_context=True,
                    neighbor_window=2
                )
                projection_input_dim = self.difference_layer.output_dim
        else:
            self.difference_layer = None
            projection_input_dim = input_dim
        
        # Multi-scale input projection (handles variable input dimensions better)
        # For large inputs (>1000), use gradual compression to preserve information
        if projection_input_dim > hidden_dim * 2:
            self.input_projection = nn.Sequential(
                nn.Linear(projection_input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.input_projection = nn.Sequential(
                nn.Linear(projection_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        
        # Temporal Convolutional Network (TCN) with MULTI-SCALE kernels
        # Different kernel sizes capture patterns at different temporal scales
        # This is CRITICAL for gradual transitions which have patterns over 5-30 frames
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=1),  # Larger kernel for gradual
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7, groups=1),  # Even larger for slow fades
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tcn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Bidirectional LSTM for local sequential context
        # Complements transformer's global attention with local recurrence
        self.local_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)  # Project LSTM output
        self.lstm_gate = nn.Parameter(torch.tensor(0.3))  # Learnable gate for LSTM contribution
        
        # Positional encoding (sinusoidal)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout, max_sequence_length)
        
        # Relative position bias for attention (learned)
        self.relative_pos_bias = nn.Parameter(torch.zeros(num_heads, max_sequence_length, max_sequence_length))
        nn.init.trunc_normal_(self.relative_pos_bias, std=0.02)
        
        # Transformer encoder with Pre-LN and DropPath for stability + generalization
        # DropPath rates increase linearly from 0 to drop_path_rate (stochastic depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.transformer_layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True  # Pre-LN for better training stability
                )
            )
            self.drop_paths.append(DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity())
        
        # Output head for sequence-level classification
        self.sequence_classifier = nn.Sequential(
            nn.Linear(hidden_dim + 1 if use_magnitude_skip else hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Binary classification: has_boundary or not
        )
        
        # Frame classifier with magnitude skip connection and residual
        # Takes encoded features + raw magnitude for direct boundary signal
        frame_input_dim = hidden_dim + 1 if use_magnitude_skip else hidden_dim
        
        # First block with residual
        self.frame_fc1 = nn.Linear(frame_input_dim, hidden_dim // 2)
        self.frame_ln1 = nn.LayerNorm(hidden_dim // 2)
        self.frame_fc1_res = nn.Linear(frame_input_dim, hidden_dim // 2) if frame_input_dim != hidden_dim // 2 else nn.Identity()
        
        # Second block with residual  
        self.frame_fc2 = nn.Linear(hidden_dim // 2, 128)
        self.frame_ln2 = nn.LayerNorm(128)
        self.frame_fc2_res = nn.Linear(hidden_dim // 2, 128)
        
        # Output layer - CLASSIFICATION HEAD (binary: is this a boundary?)
        self.frame_output = nn.Linear(128, 1)
        self.frame_dropout = nn.Dropout(dropout)
        
        # =====================================================================
        # MULTI-TASK HEAD: Duration Regression with TEMPORAL CONTEXT
        # 
        # Key insight: To distinguish cuts from graduals, the model needs to see
        # HOW features change over time, not just the features at a single frame.
        # - Cuts: Sharp, sudden change at ONE frame
        # - Graduals: Smooth, gradual change over MULTIPLE frames
        #
        # Solution: Add temporal convolution to capture neighboring frame patterns
        # before predicting duration. This lets the head learn:
        # - "Sharp spike in features" → Cut → duration ≈ 1
        # - "Gradual ramp in features" → Gradual → duration > 1
        # =====================================================================
        self.use_multi_task = True  # Can be disabled for backward compat
        
        # Temporal context aggregation (1D conv over sequence dimension)
        # IMPORTANT: We use MULTI-SCALE convolutions to capture:
        # - Local patterns (k=3): sharp cut transitions
        # - Medium patterns (k=7): short graduals
        # - Wide patterns (k=15): long dissolves/fades
        self.duration_temporal_conv = nn.Sequential(
            # Conv1d expects (batch, channels, length)
            nn.Conv1d(128, 64, kernel_size=3, padding=1),  # Local: ±1 frames
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),   # Medium: ±3 frames  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=15, padding=7),  # Wide: ±7 frames
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        # Magnitude temporal analysis - critical for distinguishing cuts vs graduals!
        # Cuts: single HIGH spike in magnitude
        # Graduals: spread of MODERATE values over time
        self.magnitude_temporal_conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),   # Local magnitude pattern
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=11, padding=5),  # Wider context
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        
        # Duration regression head (combines hidden features + magnitude patterns)
        # Input: 32 (from h2_temporal) + 8 (from magnitude patterns) = 40
        self.duration_head = nn.Sequential(
            nn.Linear(40, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Less dropout for regression
            nn.Linear(32, 1)  # Output: log(1 + duration)
        )
        
        # Learnable weight for magnitude skip connection
        if use_magnitude_skip:
            self.magnitude_skip_weight = nn.Parameter(torch.tensor(1.0))
            
            # Additional: Magnitude-based gating (learned attention on magnitude)
            self.magnitude_gate = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Cache for causal mask
        self._causal_mask_cache = {}
        
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal attention mask (lower triangular).
        
        Each position can only attend to positions before it (and itself).
        This ensures the model learns temporal patterns progressively.
        """
        cache_key = (seq_len, device)
        if cache_key not in self._causal_mask_cache:
            # Create upper triangular mask (True = masked out = cannot attend)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self._causal_mask_cache[cache_key] = mask
        return self._causal_mask_cache[cache_key]
        
    def forward(self, x: torch.Tensor, mode: str = 'frame') -> torch.Tensor:
        """
        Forward pass with difference features, causal attention, and magnitude skip.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            mode: 'sequence' for sequence-level classification,
                  'frame' for frame-level classification (default, for boundary localization)
            
        Returns:
            - If mode='sequence': Binary logits of shape (batch_size, 1) indicating if sequence has boundary
            - If mode='frame': Boundary logits of shape (batch_size, sequence_length-1, 1) for each frame transition
        """
        batch_size, sequence_length, _ = x.shape
        
        # Step 1: Compute frame differences (if enabled)
        magnitude = None
        if self.use_difference_features:
            x, diff_seq_len, magnitude = self.difference_layer(x)  # (batch, seq_len-1, diff_dim), magnitude
            sequence_length = diff_seq_len
        
        # Step 2: Input projection (multi-scale for large inputs)
        x = self.input_projection(x)  # (batch_size, seq_len(-1), hidden_dim)
        
        # Step 3: TCN for local pattern capture (before global attention)
        # TCN captures sharp transitions that attention might smooth over
        x_tcn = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # (batch, seq, hidden)
        tcn_gate = self.tcn_gate(x)  # (batch, seq, hidden)
        x = x + tcn_gate * x_tcn  # Gated residual connection
        
        # Step 4: Bidirectional LSTM for local sequential patterns
        lstm_out, _ = self.local_lstm(x)  # (batch, seq, hidden)
        lstm_out = self.lstm_proj(lstm_out)
        x = x + self.lstm_gate * lstm_out  # Gated residual
        
        # Step 5: Add positional encoding
        x = self.positional_encoding(x)
        
        # Step 6: Apply layer normalization
        x = self.layer_norm(x)
        
        # Step 7: Get causal attention mask (if enabled)
        if self.use_causal_attention:
            causal_mask = self._get_causal_mask(sequence_length, x.device)
        else:
            causal_mask = None
        
        # Step 8: Transformer encoding with causal mask and DropPath
        encoded = x
        for layer, drop_path in zip(self.transformer_layers, self.drop_paths):
            # Apply transformer layer with residual DropPath
            encoded = encoded + drop_path(layer(encoded, src_mask=causal_mask) - encoded)
        
        if mode == 'sequence':
            # SEQUENCE-LEVEL CLASSIFICATION
            # Use global average pooling over the sequence
            sequence_repr = encoded.mean(dim=1)  # (batch_size, hidden_dim)
            
            # Add magnitude skip connection (max magnitude as boundary signal)
            if self.use_magnitude_skip and magnitude is not None:
                max_magnitude = magnitude.max(dim=1)[0]  # (batch_size, 1)
                max_magnitude = max_magnitude * self.magnitude_skip_weight
                sequence_repr = torch.cat([sequence_repr, max_magnitude], dim=1)
            
            logits = self.sequence_classifier(sequence_repr)  # (batch_size, 1)
            return logits
        
        else:  # mode == 'frame' or mode == 'frame_multi_task'
            # FRAME-LEVEL CLASSIFICATION with magnitude skip and residual connections
            # Each position in encoded represents a frame transition (difference)
            
            if self.use_magnitude_skip and magnitude is not None:
                # Apply magnitude gating: learn to weight the magnitude contribution
                magnitude_gated = self.magnitude_gate(magnitude) * magnitude * self.magnitude_skip_weight
                
                # Concatenate encoded features with gated magnitude
                frame_input = torch.cat([encoded, magnitude_gated], dim=-1)  # (batch, seq_len-1, hidden_dim+1)
            else:
                frame_input = encoded
            
            # Frame classifier with residual connections
            # Block 1: frame_input -> hidden_dim // 2
            h1 = self.frame_fc1(frame_input)
            h1 = self.frame_ln1(h1)
            h1 = F.gelu(h1)
            h1 = self.frame_dropout(h1)
            h1 = h1 + self.frame_fc1_res(frame_input)  # Residual
            
            # Block 2: hidden_dim // 2 -> 128
            h2 = self.frame_fc2(h1)
            h2 = self.frame_ln2(h2)
            h2 = F.gelu(h2)
            h2 = self.frame_dropout(h2)
            h2 = h2 + self.frame_fc2_res(h1)  # Residual
            
            # Classification output
            class_logits = self.frame_output(h2)  # (batch_size, seq_len-1, 1)
            
            # =====================================================================
            # MULTI-TASK: Duration Regression Head with TEMPORAL CONTEXT
            # Predicts log(1 + duration) for transition type awareness
            # Uses temporal convolution to see neighboring frame patterns
            # CRITICAL: Also uses MAGNITUDE patterns to distinguish cuts/graduals!
            # =====================================================================
            if mode == 'frame_multi_task' and hasattr(self, 'use_multi_task') and self.use_multi_task:
                # h2 shape: (batch_size, seq_len-1, 128)
                # Conv1d needs: (batch, channels, length) -> (batch, 128, seq_len-1)
                h2_conv = h2.transpose(1, 2)  # (batch, 128, seq_len-1)
                
                # Apply temporal convolution to capture transition shape from hidden features
                h2_temporal = self.duration_temporal_conv(h2_conv)  # (batch, 32, seq_len-1)
                h2_temporal = h2_temporal.transpose(1, 2)  # (batch, seq_len-1, 32)
                
                # Apply temporal convolution to magnitude signal
                # This is KEY for cuts vs graduals: cuts have sharp spike, graduals have spread
                if magnitude is not None:
                    # magnitude: (batch, seq_len-1, 1)
                    mag_conv = magnitude.transpose(1, 2)  # (batch, 1, seq_len-1)
                    mag_temporal = self.magnitude_temporal_conv(mag_conv)  # (batch, 8, seq_len-1)
                    mag_temporal = mag_temporal.transpose(1, 2)  # (batch, seq_len-1, 8)
                    
                    # Concatenate hidden features with magnitude patterns
                    duration_input = torch.cat([h2_temporal, mag_temporal], dim=-1)  # (batch, seq_len-1, 40)
                else:
                    # Fallback: pad with zeros if no magnitude
                    pad = torch.zeros(h2_temporal.shape[0], h2_temporal.shape[1], 8, device=h2.device)
                    duration_input = torch.cat([h2_temporal, pad], dim=-1)
                
                # Duration prediction from combined features
                duration_pred = self.duration_head(duration_input)  # (batch_size, seq_len-1, 1)
                
                # Return both outputs as a dictionary
                return {
                    'class_logits': class_logits,
                    'duration_pred': duration_pred
                }
            
            # Standard mode: return only classification logits (backward compatible)
            return class_logits


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TransformerTrainer:
    """
    Trainer for the transformer boundary detection model.
    Uses FRAME-LEVEL classification for precise boundary localization.
    
    Supports MULTI-TASK LEARNING:
    - Binary classification: Is this frame a boundary?
    - Duration regression: How long is the transition?
    
    The duration task teaches the model to recognize gradual transitions!
    """
    
    def __init__(self, model: BoundaryDetectionTransformer, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Multi-task learning flag - ONLY enabled when loss_type == "multi_task"
        # Will be set to True below if loss_type is "multi_task"
        self.use_multi_task = False
        
        # Training parameters
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 16)
        self.num_epochs = config.get("num_epochs", 100)
        self.early_stopping_patience = config.get("early_stopping_patience", 20)  # Increased patience for better convergence
        
        # Classification threshold (can be tuned for precision/recall trade-off)
        # Higher threshold = better precision, lower recall (fewer false positives)
        # Lower threshold = better recall, lower precision (catch more boundaries)
        # 
        # With 0.8 threshold: Precision=0.81, Recall=0.71, F1=0.77
        # Lowering to 0.5 should improve recall → better F1
        self.classification_threshold = config.get("classification_threshold", 0.5)
        
        # Temporal tolerance for evaluation (standard in SBD benchmarks)
        # Prediction within ±tolerance frames of ground truth is considered correct
        self.temporal_tolerance = config.get("temporal_tolerance", 3)
        
        # CRITICAL: Loss configuration for FRAME-LEVEL classification
        # Frame-level imbalance: ~1.7% boundary frames, ~98.3% non-boundary
        # Balance ratio: 1:56.6 (boundary:non-boundary) - EXTREME IMBALANCE!
        # 
        # Strategy: Use Soft F1 Loss as primary for direct F1 optimization
        
        loss_type = config.get("loss_type", "multi_task")  # "bce", "focal", "combo", "soft_f1", "combo_v2" - DEFAULT: combo_v2
        
        if loss_type == "focal":
            # Focal Loss for FRAME-LEVEL extreme imbalance (1:56.6)
            focal_alpha = config.get("focal_alpha", 0.85)
            focal_gamma = config.get("focal_gamma", 2.0)
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"Using Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
            
        elif loss_type == "combo":
            # Combination of Focal Loss and Dice Loss
            focal_alpha = config.get("focal_alpha", 0.75)
            focal_gamma = config.get("focal_gamma", 2.0)
            self.focal_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.dice_weight = config.get("dice_weight", 0.5)
            self.criterion = self._combo_loss
            print(f"Using Combo Loss (Focal + Dice):")
            print(f"  Focal: alpha={focal_alpha}, gamma={focal_gamma}")
            print(f"  Dice weight: {self.dice_weight}")
            
        elif loss_type == "soft_f1":
            # BEST FOR F1 OPTIMIZATION - Directly optimizes F1 score!
            self.criterion = SoftF1Loss()
            print(f"Using Soft F1 Loss - DIRECTLY OPTIMIZES F1 SCORE!")
            print(f"  This is the most direct way to maximize F1!")
            
        elif loss_type == "tversky":
            # Tversky Loss - Control precision vs recall
            tversky_alpha = config.get("tversky_alpha", 0.3)  # FP weight (lower = tolerate FP)
            tversky_beta = config.get("tversky_beta", 0.7)    # FN weight (higher = penalize FN = better recall)
            self.criterion = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
            print(f"Using Tversky Loss: alpha={tversky_alpha}, beta={tversky_beta}")
            print(f"  alpha < beta: Prioritizes Recall over Precision")
            
        elif loss_type == "focal_tversky":
            # Focal Tversky - Tversky with focal mechanism
            tversky_alpha = config.get("tversky_alpha", 0.3)
            tversky_beta = config.get("tversky_beta", 0.7)
            focal_gamma = config.get("focal_gamma", 1.5)
            self.criterion = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=focal_gamma)
            print(f"Using Focal Tversky Loss: alpha={tversky_alpha}, beta={tversky_beta}, gamma={focal_gamma}")
            
        elif loss_type == "combo_v2":
            # RECOMMENDED: Combines Focal + Tversky + Soft F1 for maximum F1
            focal_alpha = config.get("focal_alpha", 0.75)
            focal_gamma = config.get("focal_gamma", 2.0)
            # Tversky: alpha=FP weight, beta=FN weight
            # BALANCED for F1: alpha=beta=0.5 is equivalent to Dice/F1 loss
            # Previous alpha=0.6, beta=0.4 gave P=0.87, R=0.67 → F1 limited by low recall
            # Now: alpha=0.4, beta=0.6 → Slightly favor recall to balance P/R for higher F1
            tversky_alpha = config.get("tversky_alpha", 0.4)  # Lower FP penalty
            tversky_beta = config.get("tversky_beta", 0.6)    # Higher FN penalty → better recall
            focal_weight = config.get("focal_weight", 0.3)  # 30% focal for imbalance
            tversky_weight = config.get("tversky_weight", 0.4)  # 40% tversky for stronger P/R control
            f1_weight = config.get("f1_weight", 0.3)  # 30% direct F1 optimization
            self.criterion = ComboLossV2(
                focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
                focal_weight=focal_weight, tversky_weight=tversky_weight, f1_weight=f1_weight
            )
            print(f"Using ComboLossV2 (Focal + Tversky + Soft F1):")
            print(f"  Focal: {focal_weight*100:.0f}% weight")
            print(f"  Tversky: {tversky_weight*100:.0f}% weight (alpha={tversky_alpha}, beta={tversky_beta})")
            print(f"  Soft F1: {f1_weight*100:.0f}% weight")
            print(f"  Mode: BALANCED F1 focus (alpha<beta favors recall to maximize F1)")
        
        elif loss_type == "multi_task":
            # =========================================================================
            # MULTI-TASK LEARNING: Classification + Duration Regression
            # This is the KEY for learning gradual transitions!
            # 
            # The duration head teaches the model:
            # - Cuts: Sharp spikes, duration ≈ 1
            # - Gradual: Sustained changes over 5-30+ frames
            # =========================================================================
            from focal_loss import MultiTaskBoundaryLoss
            
            # Classification component params
            focal_alpha = config.get("focal_alpha", 0.75)
            focal_gamma = config.get("focal_gamma", 2.0)
            tversky_alpha = config.get("tversky_alpha", 0.4)
            tversky_beta = config.get("tversky_beta", 0.6)
            
            # Duration regression params
            gradual_weight = config.get("gradual_weight", 3.0)  # 3x weight for gradual
            
            # Task weights
            classification_weight = config.get("classification_weight", 0.6)
            duration_weight = config.get("duration_weight", 0.4)
            
            self.criterion = MultiTaskBoundaryLoss(
                focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
                gradual_weight=gradual_weight,
                classification_weight=classification_weight,
                duration_weight=duration_weight
            )
            self.use_multi_task = True
            print(f"Using MULTI-TASK Loss (Classification + Duration):")
            print(f"  Classification: {classification_weight*100:.0f}% weight")
            print(f"  Duration Regression: {duration_weight*100:.0f}% weight")
            print(f"  Gradual transition boost: {gradual_weight}x")
            print(f"  This teaches the model gradual transition SIGNATURES!")
            
        elif loss_type == "adaptive_focal":
            from focal_loss import AdaptiveFocalLoss
            focal_gamma = config.get("focal_gamma", 2.5)
            self.criterion = AdaptiveFocalLoss(gamma=focal_gamma)
            print(f"Using Adaptive Focal Loss: gamma={focal_gamma}")
            
        else:  # bce
            pos_weight_value = config.get("pos_weight", 20.0)
            pos_weight = torch.tensor([pos_weight_value])
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            print(f"Using BCE Loss: pos_weight={pos_weight_value}")
        
        print(f"  Classification threshold: {self.classification_threshold}")
        print(f"  Goal: F1 > 0.8!")
        
        # Optimizer with STRONG weight decay to prevent overfitting
        # Previous 1e-4 was too weak - model was overfitting severely
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-4  # 5x increase for better regularization
        )
        
        # Learning rate scheduler - ReduceLROnPlateau for smooth training without dips
        # Reduces LR only when validation metric stops improving
        self.scheduler_type = config.get("scheduler_type", "plateau")  # "cosine_warmup", "plateau", "onecycle"
        self.warmup_epochs = config.get("warmup_epochs", 3)
        self.scheduler = None  # Will be initialized in train_epoch
        self._scheduler_initialized = False
        self._warmup_scheduler = None
        
        # Gradient accumulation for more stable updates
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 2)
    
    def _compute_tolerance_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                   tolerance: int = 3) -> Tuple[int, int, int, int]:
        """
        Compute TP, FP, FN, TN with FRAME-LEVEL matching and temporal tolerance.
        
        CRITICAL FIX: Use frame-level metrics instead of region-grouping.
        Region-grouping caused train/val metric mismatch because:
        1. Same transition appears in different batches (overlapping windows)
        2. Region counting in each batch leads to inflated TP counts
        
        New approach: Standard frame-level tolerance matching
        - For each predicted boundary, check if any GT is within ±tolerance
        - For each GT boundary, check if any prediction is within ±tolerance
        - ONE-TO-ONE matching to prevent double counting
        
        Args:
            predictions: Binary predictions (batch, seq_len-1)
            targets: Target labels - can be soft (0-1) or binary (batch, seq_len-1)
            tolerance: Number of frames tolerance (±tolerance)
            
        Returns:
            Tuple of (TP, FP, FN, TN) counts
        """
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        
        batch_size = predictions_np.shape[0] if len(predictions_np.shape) > 1 else 1
        
        for seq_idx in range(batch_size):
            if len(predictions_np.shape) > 1:
                seq_preds = predictions_np[seq_idx]
                seq_targets = targets_np[seq_idx]
            else:
                seq_preds = predictions_np
                seq_targets = targets_np
            
            seq_len = len(seq_preds)
            
            # Get indices of predictions and GT peaks
            pred_indices = list(np.where(seq_preds == 1)[0])
            
            # For GT: Group consecutive positive frames into SINGLE transitions
            # Then take the CENTER of each group as the GT peak
            # This ensures 1 GT peak per transition even with ±1 neighbor labeling
            gt_peaks = []
            gt_positive_indices = np.where(seq_targets >= 0.5)[0]
            
            if len(gt_positive_indices) > 0:
                # Group consecutive indices into transitions
                groups = []
                current_group = [gt_positive_indices[0]]
                
                for idx in gt_positive_indices[1:]:
                    if idx <= current_group[-1] + 1:  # Consecutive or adjacent
                        current_group.append(idx)
                    else:
                        groups.append(current_group)
                        current_group = [idx]
                groups.append(current_group)  # Don't forget last group
                
                # Take center of each group as the GT peak (1 peak per transition)
                for group in groups:
                    center = group[len(group) // 2]
                    gt_peaks.append(center)
            
            # ONE-TO-ONE MATCHING with tolerance
            # Each prediction can match at most one GT, each GT can match at most one prediction
            matched_preds = set()
            matched_gts = set()
            
            # Sort by distance for greedy matching (closest pairs first)
            matches = []
            for p_idx, pred in enumerate(pred_indices):
                for g_idx, gt in enumerate(gt_peaks):
                    dist = abs(pred - gt)
                    if dist <= tolerance:
                        matches.append((dist, p_idx, g_idx))
            
            matches.sort(key=lambda x: x[0])  # Sort by distance
            
            for dist, p_idx, g_idx in matches:
                if p_idx not in matched_preds and g_idx not in matched_gts:
                    matched_preds.add(p_idx)
                    matched_gts.add(g_idx)
            
            # Compute metrics
            tp = len(matched_gts)  # Correctly detected boundaries
            fp = len(pred_indices) - len(matched_preds)  # Spurious predictions
            fn = len(gt_peaks) - len(matched_gts)  # Missed boundaries
            tn = seq_len - len(gt_peaks) - fp  # Non-boundaries correctly not predicted
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += max(0, tn)
        
        return total_tp, total_fp, total_fn, total_tn
    
    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Dice loss for boundary detection - directly optimizes F1-like metric."""
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        return 1 - dice
    
    def _combo_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combination of Focal Loss and Dice Loss."""
        focal = self.focal_criterion(logits, targets)
        dice = self._dice_loss(logits, targets)
        return (1 - self.dice_weight) * focal + self.dice_weight * dice
        
    def train_epoch(self, train_loader, current_epoch: int = 0) -> Tuple[float, float]:
        """
        Train for one epoch with FRAME-LEVEL classification.
        
        Args:
            train_loader: DataLoader for training data
            current_epoch: Current epoch number (for warmup scheduling)
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        
        # Track FRAME-LEVEL metrics
        total_frames = 0
        correct_frames = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        # Initialize scheduler with correct steps if not done
        if not self._scheduler_initialized:
            if self.scheduler_type == "cosine_warmup":
                # Use CosineAnnealingLR with built-in warmup via SequentialLR
                # This ensures smooth transition from warmup to cosine annealing
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,  # Start at 10% of learning rate
                    end_factor=1.0,    # End at full learning rate
                    total_iters=self.warmup_epochs
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.num_epochs - self.warmup_epochs,
                    eta_min=self.learning_rate * 0.01
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_epochs]
                )
                self._uses_epoch_scheduler = True  # Flag to step per epoch
            elif self.scheduler_type == "plateau":
                # ReduceLROnPlateau with manual warmup (handled separately)
                # Warmup is done manually, then plateau takes over
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',  # Maximize F1
                    factor=0.5,
                    patience=2,  # Increased patience for stability after warmup
                    min_lr=self.learning_rate * 0.01
                )
                self._uses_epoch_scheduler = False  # Stepped on validation
            else:  # onecycle - has built-in warmup, no separate warmup needed
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.learning_rate,  # Use base LR as max (not multiplied)
                    epochs=self.num_epochs,
                    steps_per_epoch=len(train_loader) // self.gradient_accumulation_steps,
                    pct_start=0.3,  # 30% warmup phase
                    anneal_strategy='cos',
                    div_factor=25,  # Start at max_lr/25 (smoother start)
                    final_div_factor=1000  # End at max_lr/1000
                )
                self._uses_epoch_scheduler = False  # Stepped per batch internally
            self._scheduler_initialized = True
        
        # Manual warmup ONLY for plateau scheduler (others have built-in warmup)
        is_warmup = current_epoch < self.warmup_epochs and self.scheduler_type == "plateau"
        if is_warmup:
            # Smooth warmup: start from 10% and gradually increase
            warmup_progress = (current_epoch + 1) / self.warmup_epochs
            warmup_factor = 0.1 + 0.9 * warmup_progress  # 10% -> 100%
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * warmup_factor
        
        # Add progress bar for batches
        batch_iterator = tqdm(train_loader, desc="Training batches", leave=False)
        
        # Zero gradients at start for gradient accumulation
        self.optimizer.zero_grad()
        accumulated_loss = 0.0
        
        # Track multi-task losses separately
        accumulated_cls_loss = 0.0
        accumulated_dur_loss = 0.0
        
        for batch_idx, batch in enumerate(batch_iterator):
            # =====================================================================
            # MULTI-TASK TRAINING: Handle both classification and duration labels
            # =====================================================================
            if self.use_multi_task and len(batch) == 3:
                # Multi-task mode: (sequences, class_labels, duration_labels)
                sequences, frame_labels, duration_labels = batch
                sequences = sequences.to(self.device)
                frame_labels = frame_labels.to(self.device)
                duration_labels = duration_labels.to(self.device)
                
                # Forward pass in MULTI-TASK mode
                outputs = self.model(sequences, mode='frame_multi_task')
                
                if isinstance(outputs, dict):
                    class_logits = outputs['class_logits'].squeeze(-1)  # (batch, seq_len-1)
                    duration_pred = outputs['duration_pred'].squeeze(-1)  # (batch, seq_len-1)
                    
                    # Compute multi-task loss
                    loss_dict = self.criterion(class_logits, duration_pred, frame_labels, duration_labels)
                    loss = loss_dict['total'] / self.gradient_accumulation_steps
                    
                    # Track component losses for logging
                    accumulated_cls_loss += loss_dict['classification'].item()
                    accumulated_dur_loss += loss_dict['duration'].item()
                    
                    logits = class_logits  # For metrics computation
                else:
                    # Fallback: model returned tensor (no multi-task head)
                    logits = outputs.squeeze(-1) if outputs.dim() == 3 else outputs
                    loss = self.criterion(logits, frame_labels) / self.gradient_accumulation_steps
            else:
                # Standard single-task mode
                sequences, frame_labels = batch[:2]  # Handle both 2-tuple and 3-tuple
                sequences = sequences.to(self.device)
                frame_labels = frame_labels.to(self.device)
                
                # Forward pass in FRAME mode for frame-level classification
                logits = self.model(sequences, mode='frame')  # (batch, seq_len-1, 1)
                logits = logits.squeeze(-1)  # (batch, seq_len-1)
                
                # Compute loss at FRAME level (scaled for gradient accumulation)
                loss = self.criterion(logits, frame_labels) / self.gradient_accumulation_steps
            
            # Backward pass (accumulate gradients)
            loss.backward()
            accumulated_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Step per-batch scheduler (OneCycleLR) only if not in warmup
                if not is_warmup and self.scheduler_type == "onecycle":
                    try:
                        self.scheduler.step()
                    except:
                        pass  # Ignore scheduler errors at end of training
            
            # FRAME-LEVEL statistics with TEMPORAL TOLERANCE
            total_loss += loss.item() * self.gradient_accumulation_steps  # Unscale for logging
            probs = torch.sigmoid(logits)
            predictions = (probs > self.classification_threshold).float()
            
            # Compute tolerance-based metrics (standard in SBD benchmarks)
            # A prediction within ±tolerance frames of ground truth is TP
            tp, fp, fn, tn = self._compute_tolerance_metrics(
                predictions, frame_labels, self.temporal_tolerance
            )
            
            true_positives += tp
            false_positives += fp
            false_negatives += fn
            true_negatives += tn
            
            # Count frame-level metrics (for accuracy)
            batch_frames = frame_labels.numel()
            total_frames += batch_frames
            correct_frames += (predictions == frame_labels).sum().item()
            
            # Update progress bar with tolerance-based stats
            current_avg_loss = total_loss / (batch_iterator.n + 1)
            frame_precision = true_positives / (true_positives + false_positives + 1e-8)
            frame_recall = true_positives / (true_positives + false_negatives + 1e-8)
            frame_f1 = 2 * frame_precision * frame_recall / (frame_precision + frame_recall + 1e-8)
            
            # Show current LR in progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            batch_iterator.set_postfix({
                'loss': f'{current_avg_loss:.4f}',
                'P': f'{frame_precision:.3f}',
                'R': f'{frame_recall:.3f}',
                'F1': f'{frame_f1:.3f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # Handle remaining gradients if batch count not divisible by accumulation steps
        if len(train_loader) % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Step epoch-level scheduler (SequentialLR for cosine_warmup)
        # This handles both warmup and cosine phases automatically
        if hasattr(self, '_uses_epoch_scheduler') and self._uses_epoch_scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_frames / total_frames
        
        # Print epoch summary
        final_precision = true_positives / (true_positives + false_positives + 1e-8)
        final_recall = true_positives / (true_positives + false_negatives + 1e-8)
        final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-8)
        print(f"\n  Train Epoch Summary (tolerance=±{self.temporal_tolerance} frames):")
        print(f"    TP={true_positives}, FP={false_positives}, FN={false_negatives}, TN={true_negatives}")
        print(f"    Precision={final_precision:.4f}, Recall={final_recall:.4f}, F1={final_f1:.4f}")
        
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float, float, float]:
        """
        Validate the model with FRAME-LEVEL classification.
        
        Returns:
            Tuple of (loss, accuracy, precision, recall)
        """
        self.model.eval()
        total_loss = 0.0
        
        # Track FRAME-LEVEL metrics with TEMPORAL TOLERANCE
        total_frames = 0
        correct_frames = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        # Track duration metrics for multi-task mode
        total_duration_mae = 0.0
        total_duration_samples = 0
        gradual_correct = 0
        gradual_total = 0
        
        # Use SAME threshold as training for fair comparison during training
        # (higher threshold + NMS will be used at final inference time)
        val_threshold = self.classification_threshold
        
        # Add progress bar for validation batches
        batch_iterator = tqdm(val_loader, desc="Validation batches", leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(batch_iterator):
                # =====================================================================
                # MULTI-TASK VALIDATION: Handle both classification and duration
                # =====================================================================
                if self.use_multi_task and len(batch) == 3:
                    sequences, frame_labels, duration_labels = batch
                    sequences = sequences.to(self.device)
                    frame_labels = frame_labels.to(self.device)
                    duration_labels = duration_labels.to(self.device)
                    
                    # Forward pass in MULTI-TASK mode
                    outputs = self.model(sequences, mode='frame_multi_task')
                    
                    if isinstance(outputs, dict):
                        class_logits = outputs['class_logits'].squeeze(-1)
                        duration_pred = outputs['duration_pred'].squeeze(-1)
                        
                        # Compute multi-task loss
                        loss_dict = self.criterion(class_logits, duration_pred, frame_labels, duration_labels)
                        loss = loss_dict['total']
                        
                        # Evaluate duration prediction on boundary frames
                        boundary_mask = frame_labels > 0.5
                        if boundary_mask.sum() > 0:
                            pred_dur = torch.expm1(duration_pred[boundary_mask])  # Convert from log scale
                            true_dur = duration_labels[boundary_mask]
                            duration_mae = torch.abs(pred_dur - true_dur).mean()
                            total_duration_mae += duration_mae.item() * boundary_mask.sum().item()
                            total_duration_samples += boundary_mask.sum().item()
                            
                            # Check gradual detection (duration > 2)
                            # ClipShots format: cuts are [n, n+1] so duration=2
                            # Graduals span more frames, so duration > 2
                            is_gradual_true = true_dur > 2.5
                            is_gradual_pred = pred_dur > 2.5
                            gradual_correct += (is_gradual_true == is_gradual_pred).sum().item()
                            gradual_total += is_gradual_true.numel()
                            
                            # Track actual predictions for debugging - ACCUMULATE all batches
                            if not hasattr(self, '_debug_dur_pred_list'):
                                self._debug_dur_pred_list = []
                                self._debug_dur_true_list = []
                            self._debug_dur_pred_list.append(pred_dur.detach().cpu().numpy())
                            self._debug_dur_true_list.append(true_dur.detach().cpu().numpy())
                        
                        logits = class_logits
                    else:
                        logits = outputs.squeeze(-1) if outputs.dim() == 3 else outputs
                        loss = self.criterion(logits, frame_labels)
                else:
                    # Standard single-task validation
                    sequences, frame_labels = batch[:2]
                    sequences = sequences.to(self.device)
                    frame_labels = frame_labels.to(self.device)
                    
                    # Forward pass in FRAME mode
                    logits = self.model(sequences, mode='frame')  # (batch, seq_len-1, 1)
                    logits = logits.squeeze(-1)  # (batch, seq_len-1)
                    
                    # Compute loss at FRAME level
                    loss = self.criterion(logits, frame_labels)
                
                total_loss += loss.item()
                
                # FRAME-LEVEL predictions - use SAME threshold as training
                predictions = (torch.sigmoid(logits) > val_threshold).float()
                
                # Compute tolerance-based metrics (standard in SBD benchmarks)
                tp, fp, fn, tn = self._compute_tolerance_metrics(
                    predictions, frame_labels, self.temporal_tolerance
                )
                
                true_positives += tp
                false_positives += fp
                false_negatives += fn
                true_negatives += tn
                
                # Count frame-level metrics (for accuracy)
                batch_frames = frame_labels.numel()
                total_frames += batch_frames
                correct_frames += (predictions == frame_labels).sum().item()
                
                # Update progress bar
                current_avg_loss = total_loss / (batch_iterator.n + 1)
                batch_iterator.set_postfix({'loss': f'{current_avg_loss:.4f}'})
        
        # Compute final metrics with tolerance
        accuracy = correct_frames / total_frames if total_frames > 0 else 0
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        avg_loss = total_loss / len(val_loader)
        
        print(f"\n  Validation Summary (tolerance=±{self.temporal_tolerance} frames):")
        print(f"    TP={true_positives}, FP={false_positives}, FN={false_negatives}")
        print(f"    Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # Print duration metrics for multi-task mode
        if self.use_multi_task and total_duration_samples > 0:
            avg_duration_mae = total_duration_mae / total_duration_samples
            gradual_accuracy = gradual_correct / gradual_total if gradual_total > 0 else 0
            print(f"    Duration MAE={avg_duration_mae:.2f} frames")
            print(f"    Gradual/Cut classification accuracy={gradual_accuracy:.4f}")
            
            # Debug: Show actual duration predictions distribution (accumulated across ALL batches)
            if hasattr(self, '_debug_dur_pred_list') and len(self._debug_dur_pred_list) > 0:
                import numpy as np
                pred = np.concatenate(self._debug_dur_pred_list)
                true = np.concatenate(self._debug_dur_true_list)
                # Clear for next validation
                self._debug_dur_pred_list = []
                self._debug_dur_true_list = []
                
                print(f"    Duration preds - Min:{pred.min():.2f}, Max:{pred.max():.2f}, Mean:{pred.mean():.2f}, Std:{pred.std():.2f}")
                print(f"    Duration true  - Min:{true.min():.2f}, Max:{true.max():.2f}, Mean:{true.mean():.2f}")
                # Check discrimination: what % of cuts are predicted as cuts?
                cuts_mask = true <= 2.5  # ClipShots: cuts are [n, n+1] = duration 2
                grads_mask = true > 2.5   # Graduals span more frames
                if cuts_mask.sum() > 0:
                    cut_pred_mean = pred[cuts_mask].mean()
                    cut_pred_std = pred[cuts_mask].std()
                    print(f"    Cuts (n={cuts_mask.sum()}): mean pred = {cut_pred_mean:.2f} ± {cut_pred_std:.2f} (should be ~2.0)")
                else:
                    print(f"    Cuts (n=0): No cuts in validation set")
                if grads_mask.sum() > 0:
                    grad_pred_mean = pred[grads_mask].mean()
                    grad_pred_std = pred[grads_mask].std()
                    print(f"    Graduals (n={grads_mask.sum()}): mean pred = {grad_pred_mean:.2f} ± {grad_pred_std:.2f} (should be >3.0)")
                else:
                    print(f"    Graduals (n=0): No graduals in validation set")
        
        return avg_loss, accuracy, precision, recall
    
    def step_scheduler_on_validation(self, val_f1: float) -> None:
        """
        Step the scheduler based on validation F1 (for ReduceLROnPlateau).
        Call this after validation in training loop.
        
        Note: cosine_warmup scheduler is stepped per epoch in train_epoch().
              onecycle scheduler is stepped per batch automatically.
              Only plateau scheduler needs explicit stepping on validation.
        
        Args:
            val_f1: Validation F1 score
        """
        if self.scheduler_type == "plateau" and self.scheduler is not None:
            self.scheduler.step(val_f1)
        
        # Always print current LR for debugging
        print(f"  Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


# Register the method
DetectionMethodRegistry.register("transformer", TransformerDetection)