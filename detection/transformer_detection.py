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
        self.dropout = config.get("dropout", 0.15)  # Increased from 0.1 for better generalization
        
        # Processing parameters
        self.sequence_length = config.get("sequence_length", 100)
        self.confidence_threshold = config.get("confidence_threshold", 0.3)
        
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
        self.drop_path_rate = config.get("drop_path_rate", 0.1)
        
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
            drop_path_rate=self.drop_path_rate  # Stochastic depth for generalization
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.is_initialized = True
    
    def detect_boundaries(self, features: np.ndarray) -> List[int]:
        """
        Detect boundaries from feature sequences with SOTA enhancements.
        
        Uses:
        - Batch inference for speed
        - Center-frame prediction for reliability (with edge handling)
        - NMS peak detection for clean output
        
        Args:
            features: Feature array of shape (N, feature_dim)
            
        Returns:
            List of boundary frame indices
        """
        if len(features) < 2:
            return []
        
        # Step 1: Collect all windows for batch processing
        windows = []
        window_starts = []
        is_edge_window = []  # Track which windows are edge cases
        
        # Calculate stride based on center-frame strategy
        if self.use_center_frame:
            # Slide by center_ratio of sequence_length (only trust middle frames)
            stride = max(1, int(self.sequence_length * self.center_ratio))
        else:
            stride = self.sequence_length // 2  # 50% overlap
        
        for start_idx in range(0, len(features) - self.sequence_length + 1, stride):
            end_idx = start_idx + self.sequence_length
            windows.append(features[start_idx:end_idx])
            window_starts.append(start_idx)
            is_edge_window.append(False)
        
        # Handle edge case: add final window if needed
        if len(features) >= self.sequence_length:
            last_start = len(features) - self.sequence_length
            if not window_starts or window_starts[-1] != last_start:
                windows.append(features[last_start:])
                window_starts.append(last_start)
                is_edge_window.append(True)  # Mark as edge window
        
        if not windows:
            return []
        
        # Step 2: Batch inference
        all_probs = self._batch_predict(windows)
        
        # Step 3: Aggregate probabilities with center-frame strategy
        # IMPORTANT: Handle edge regions specially to ensure full coverage
        full_probs = np.zeros(len(features) - 1)
        prob_counts = np.zeros(len(features) - 1)
        
        for i, (start_idx, probs) in enumerate(zip(window_starts, all_probs)):
            if self.use_center_frame:
                # Calculate center region
                center_start = int(len(probs) * (0.5 - self.center_ratio / 2))
                center_end = int(len(probs) * (0.5 + self.center_ratio / 2))
                
                # For the FIRST window, also use predictions for early frames (0 to center_start)
                # These frames would otherwise never be covered
                if start_idx == 0:
                    for j in range(0, center_end):
                        global_idx = start_idx + j
                        if 0 <= global_idx < len(full_probs):
                            full_probs[global_idx] += probs[j]
                            prob_counts[global_idx] += 1
                # For the LAST window, also use predictions for late frames (center_end to end)
                elif is_edge_window[i] or start_idx == window_starts[-1]:
                    for j in range(center_start, len(probs)):
                        global_idx = start_idx + j
                        if 0 <= global_idx < len(full_probs):
                            full_probs[global_idx] += probs[j]
                            prob_counts[global_idx] += 1
                else:
                    # Normal center-only for middle windows
                    for j in range(center_start, min(center_end, len(probs))):
                        global_idx = start_idx + j
                        if 0 <= global_idx < len(full_probs):
                            full_probs[global_idx] += probs[j]
                            prob_counts[global_idx] += 1
            else:
                # Use all predictions
                for j, prob in enumerate(probs):
                    global_idx = start_idx + j
                    if 0 <= global_idx < len(full_probs):
                        full_probs[global_idx] += prob
                        prob_counts[global_idx] += 1
        
        # Average overlapping predictions
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
    
    def _batch_predict(self, windows: List[np.ndarray]) -> List[np.ndarray]:
        """
        Batch inference for multiple windows.
        
        Args:
            windows: List of feature windows
            
        Returns:
            List of probability arrays
        """
        all_probs = []
        
        with torch.no_grad():
            # Process in batches
            for batch_start in range(0, len(windows), self.batch_size_inference):
                batch_end = min(batch_start + self.batch_size_inference, len(windows))
                batch_windows = windows[batch_start:batch_end]
                
                # Stack into batch tensor
                batch_tensor = torch.from_numpy(np.stack(batch_windows)).float()
                batch_tensor = batch_tensor.to(self.device)
                
                # Forward pass
                logits = self.model(batch_tensor, mode='frame')
                probs = torch.sigmoid(logits).cpu().numpy()
                
                # Extract each window's predictions
                for i in range(len(batch_windows)):
                    all_probs.append(probs[i, :, 0])
        
        return all_probs
    
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
    - Multi-scale difference features (1st and 2nd order)
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
                 drop_path_rate: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.use_causal_attention = use_causal_attention
        self.use_difference_features = use_difference_features
        self.use_magnitude_skip = use_magnitude_skip
        self.drop_path_rate = drop_path_rate
        
        # Difference layer for computing frame differences
        if use_difference_features:
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
        
        # Temporal Convolutional Network (TCN) for local pattern capture
        # Captures local temporal patterns BEFORE global attention
        # This helps detect sharp transitions that might be diluted by attention
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=1),
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
        
        # Output layer
        self.frame_output = nn.Linear(128, 1)
        self.frame_dropout = nn.Dropout(dropout)
        
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
        
        else:  # mode == 'frame'
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
            
            # Output
            logits = self.frame_output(h2)  # (batch_size, seq_len-1, 1)
            
            return logits


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
    """
    
    def __init__(self, model: BoundaryDetectionTransformer, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        loss_type = config.get("loss_type", "soft_f1")  # "bce", "focal", "combo", "soft_f1", "combo_v2" - DEFAULT: combo_v2
        
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
            # For real-world distribution (1:133 imbalance), we MUST penalize FP aggressively
            # to prevent massive over-prediction on unseen videos.
            # Previous: alpha=0.6, beta=0.4 → Val 83K FP vs 34K TP (2.4x overprediction)
            # Now: alpha=0.7, beta=0.3 → Much stronger FP penalty for better precision
            tversky_alpha = config.get("tversky_alpha", 0.6)  # Balanced FP penalty
            tversky_beta = config.get("tversky_beta", 0.4)    # Balanced FN penalty
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
            print(f"  Mode: HIGH PRECISION focus (alpha >> beta to prevent overprediction)")
            
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
        
        # Optimizer with higher weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4  # Increased for better regularization
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
        Compute TP, FP, FN, TN with temporal tolerance (standard in SBD benchmarks).
        
        Standard SBD evaluation protocol (used in TransNetV2, ClipShots, etc.):
        - Each GT boundary can match at most ONE prediction (closest within tolerance)
        - Each prediction can match at most ONE GT boundary
        - Greedy matching: process GT boundaries and find closest unmatched prediction
        
        This gives the standard Precision/Recall where:
        - Precision = matched_predictions / total_predictions
        - Recall = matched_gt / total_gt
        
        Args:
            predictions: Binary predictions (batch, seq_len-1)
            targets: Binary ground truth (batch, seq_len-1)
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
        
        batch_size = predictions_np.shape[0]
        
        for seq_idx in range(batch_size):
            if len(predictions_np.shape) > 1:
                seq_preds = predictions_np[seq_idx]
                seq_targets = targets_np[seq_idx]
            else:
                seq_preds = predictions_np
                seq_targets = targets_np
            
            seq_len = len(seq_preds)
            
            # Get indices of predicted and ground truth boundaries
            pred_indices = list(np.where(seq_preds == 1)[0])
            gt_indices = list(np.where(seq_targets >= 0.5)[0])
            
            # ONE-TO-ONE MATCHING (standard SBD protocol)
            # Each GT matches at most one prediction, each prediction matches at most one GT
            matched_pred = set()
            matched_gt = set()
            
            # Sort GT by position for consistent matching
            for gt_idx in sorted(gt_indices):
                # Find closest unmatched prediction within tolerance
                best_pred = None
                best_dist = tolerance + 1
                
                for pred_idx in pred_indices:
                    if pred_idx in matched_pred:
                        continue
                    dist = abs(pred_idx - gt_idx)
                    if dist <= tolerance and dist < best_dist:
                        best_dist = dist
                        best_pred = pred_idx
                
                if best_pred is not None:
                    matched_pred.add(best_pred)
                    matched_gt.add(gt_idx)
            
            # Standard SBD metrics
            tp = len(matched_pred)  # = len(matched_gt), matched pairs
            fp = len(pred_indices) - tp  # Unmatched predictions
            fn = len(gt_indices) - tp  # Unmatched GT boundaries
            tn = seq_len - len(pred_indices) - fn  # Non-boundary frames not predicted
            
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
        
        for batch_idx, batch in enumerate(batch_iterator):
            sequences, frame_labels = batch  # frame_labels shape: (batch, seq_len-1)
            sequences = sequences.to(self.device)
            frame_labels = frame_labels.to(self.device)  # Keep frame-level labels!
            
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
        
        # Add progress bar for validation batches
        batch_iterator = tqdm(val_loader, desc="Validation batches", leave=False)
        
        with torch.no_grad():
            for batch in batch_iterator:
                sequences, frame_labels = batch
                sequences = sequences.to(self.device)
                frame_labels = frame_labels.to(self.device)
                
                # Forward pass in FRAME mode
                logits = self.model(sequences, mode='frame')  # (batch, seq_len-1, 1)
                logits = logits.squeeze(-1)  # (batch, seq_len-1)
                
                # Compute loss at FRAME level
                loss = self.criterion(logits, frame_labels)
                total_loss += loss.item()
                
                # FRAME-LEVEL predictions with tolerance
                predictions = (torch.sigmoid(logits) > self.classification_threshold).float()
                
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