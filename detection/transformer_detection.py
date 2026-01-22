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
from focal_loss import FocalLoss

from .base import BaseDetectionMethod, DetectionMethodRegistry

# Try to import scipy for peak detection
try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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
        self.num_layers = config.get("num_layers", 6)
        self.num_heads = config.get("num_heads", 8)
        self.ff_dim = config.get("ff_dim", 2048)
        self.dropout = config.get("dropout", 0.1)
        
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
            use_magnitude_skip=True
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.is_initialized = True
    
    def detect_boundaries(self, features: np.ndarray) -> List[int]:
        """
        Detect boundaries from feature sequences with SOTA enhancements.
        
        Uses:
        - Batch inference for speed
        - Center-frame prediction for reliability
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
        
        # Handle edge case: add final window if needed
        if len(features) >= self.sequence_length:
            last_start = len(features) - self.sequence_length
            if not window_starts or window_starts[-1] != last_start:
                windows.append(features[last_start:])
                window_starts.append(last_start)
        
        if not windows:
            return []
        
        # Step 2: Batch inference
        all_probs = self._batch_predict(windows)
        
        # Step 3: Aggregate probabilities with center-frame strategy
        full_probs = np.zeros(len(features) - 1)
        prob_counts = np.zeros(len(features) - 1)
        
        for i, (start_idx, probs) in enumerate(zip(window_starts, all_probs)):
            if self.use_center_frame:
                # Only trust the center region
                center_start = int(len(probs) * (0.5 - self.center_ratio / 2))
                center_end = int(len(probs) * (0.5 + self.center_ratio / 2))
                
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
                 ff_dim: int = 2048, dropout: float = 0.1,
                 max_sequence_length: int = 1000,
                 use_causal_attention: bool = True,
                 use_difference_features: bool = True,
                 use_magnitude_skip: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.use_causal_attention = use_causal_attention
        self.use_difference_features = use_difference_features
        self.use_magnitude_skip = use_magnitude_skip
        
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
        
        # Input projection (from difference features to hidden dim)
        self.input_projection = nn.Linear(projection_input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout, max_sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
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
        
        # Frame classifier with magnitude skip connection
        # Takes encoded features + raw magnitude for direct boundary signal
        frame_input_dim = hidden_dim + 1 if use_magnitude_skip else hidden_dim
        self.frame_classifier = nn.Sequential(
            nn.Linear(frame_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Binary classification per difference
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
        
        # Step 2: Input projection
        x = self.input_projection(x)  # (batch_size, seq_len(-1), hidden_dim)
        
        # Step 3: Add positional encoding
        x = self.positional_encoding(x)
        
        # Step 4: Apply layer normalization
        x = self.layer_norm(x)
        
        # Step 5: Get causal attention mask (if enabled)
        if self.use_causal_attention:
            causal_mask = self._get_causal_mask(sequence_length, x.device)
        else:
            causal_mask = None
        
        # Step 6: Transformer encoding with causal mask
        encoded = self.transformer_encoder(x, mask=causal_mask)  # (batch_size, seq_len, hidden_dim)
        
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
            # FRAME-LEVEL CLASSIFICATION with magnitude skip
            # Each position in encoded represents a frame transition (difference)
            
            if self.use_magnitude_skip and magnitude is not None:
                # Apply magnitude gating: learn to weight the magnitude contribution
                magnitude_gated = self.magnitude_gate(magnitude) * magnitude * self.magnitude_skip_weight
                
                # Concatenate encoded features with gated magnitude
                encoded_with_mag = torch.cat([encoded, magnitude_gated], dim=-1)  # (batch, seq_len-1, hidden_dim+1)
                logits = self.frame_classifier(encoded_with_mag)  # (batch_size, seq_len-1, 1)
            else:
                logits = self.frame_classifier(encoded)  # (batch_size, seq_len-1, 1)
            
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
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        
        # Classification threshold (can be tuned for precision/recall trade-off)
        self.classification_threshold = config.get("classification_threshold", 0.3)
        
        # CRITICAL: Loss configuration for FRAME-LEVEL classification
        # Frame-level imbalance: ~1.7% boundary frames, ~98.3% non-boundary
        # Balance ratio: 1:56.6 (boundary:non-boundary) - EXTREME IMBALANCE!
        # 
        # Strategy: Use Focal Loss with high alpha to favor minority class (boundaries)
        
        loss_type = config.get("loss_type", "focal")  # "bce", "focal", or "combo"
        
        if loss_type == "focal":
            # Focal Loss for FRAME-LEVEL extreme imbalance (1:56.6)
            # Alpha should favor the MINORITY class (boundary frames)
            # Higher alpha = more weight on positive (boundary) class
            focal_alpha = config.get("focal_alpha", 0.95)   # Alpha > 0.5: favor minority (boundary) class
            focal_gamma = config.get("focal_gamma", 2.5)    # Higher gamma: focus more on hard examples
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"Using Focal Loss for FRAME-LEVEL classification:")
            print(f"  alpha={focal_alpha}, gamma={focal_gamma}")
            print(f"  FRAME-LEVEL: ~1.7% boundary, ~98.3% non-boundary (1:56.6 ratio)")
            print(f"  Alpha {focal_alpha} (>0.5): Heavily favors minority class (boundary frames)")
            print(f"  Gamma {focal_gamma}: Strong focus on hard-to-classify examples")
        elif loss_type == "combo":
            # Combination of Focal Loss and Dice Loss for better boundary detection
            focal_alpha = config.get("focal_alpha", 0.90)
            focal_gamma = config.get("focal_gamma", 2.0)
            self.focal_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.dice_weight = config.get("dice_weight", 0.5)
            self.criterion = self._combo_loss
            print(f"Using Combo Loss (Focal + Dice) for FRAME-LEVEL classification:")
            print(f"  Focal: alpha={focal_alpha}, gamma={focal_gamma}")
            print(f"  Dice weight: {self.dice_weight}")
        elif loss_type == "adaptive_focal":
            # Use Adaptive Focal Loss - automatically adjusts alpha based on class distribution
            from focal_loss import AdaptiveFocalLoss
            focal_gamma = config.get("focal_gamma", 2.5)
            self.criterion = AdaptiveFocalLoss(gamma=focal_gamma)
            print(f"Using Adaptive Focal Loss for FRAME-LEVEL:")
            print(f"  gamma={focal_gamma}")
            print(f"  Automatically adjusts alpha based on observed class distribution")
        else:
            # Use BCE with FRAME-LEVEL weights
            # pos_weight = negative_count / positive_count ≈ 56.6
            pos_weight_value = config.get("pos_weight", 50.0)  # Approximately 56.6
            pos_weight = torch.tensor([pos_weight_value])
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            print(f"Using BCE Loss with FRAME-LEVEL weight:")
            print(f"  pos_weight={pos_weight_value} (compensates for 1:{pos_weight_value:.1f} imbalance)")
        
        print(f"  Classification threshold: {self.classification_threshold}")
        print(f"  Goal: Precise frame-level boundary detection!")
        
        # Optimizer with higher weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4  # Increased for better regularization
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate * 10,
            epochs=self.num_epochs,
            steps_per_epoch=1,  # Will be updated in train_epoch
            pct_start=0.1,
            anneal_strategy='cos'
        )
        self._scheduler_initialized = False
    
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
        
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """
        Train for one epoch with FRAME-LEVEL classification.
        
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
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate * 10,
                epochs=self.num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
                anneal_strategy='cos'
            )
            self._scheduler_initialized = True
        
        # Add progress bar for batches
        batch_iterator = tqdm(train_loader, desc="Training batches", leave=False)
        
        for batch in batch_iterator:
            sequences, frame_labels = batch  # frame_labels shape: (batch, seq_len-1)
            sequences = sequences.to(self.device)
            frame_labels = frame_labels.to(self.device)  # Keep frame-level labels!
            
            # Forward pass in FRAME mode for frame-level classification
            self.optimizer.zero_grad()
            logits = self.model(sequences, mode='frame')  # (batch, seq_len-1, 1)
            logits = logits.squeeze(-1)  # (batch, seq_len-1)
            
            # Compute loss at FRAME level
            loss = self.criterion(logits, frame_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # FRAME-LEVEL statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(logits) > self.classification_threshold).float()
            
            # Count frame-level metrics
            batch_frames = frame_labels.numel()
            total_frames += batch_frames
            correct_frames += (predictions == frame_labels).sum().item()
            
            # Detailed metrics
            true_positives += ((predictions == 1) & (frame_labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (frame_labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (frame_labels == 1)).sum().item()
            true_negatives += ((predictions == 0) & (frame_labels == 0)).sum().item()
            
            # Update progress bar with frame-level stats
            current_avg_loss = total_loss / (batch_iterator.n + 1)
            frame_accuracy = correct_frames / total_frames if total_frames > 0 else 0
            frame_precision = true_positives / (true_positives + false_positives + 1e-8)
            frame_recall = true_positives / (true_positives + false_negatives + 1e-8)
            frame_f1 = 2 * frame_precision * frame_recall / (frame_precision + frame_recall + 1e-8)
            
            batch_iterator.set_postfix({
                'loss': f'{current_avg_loss:.4f}',
                'P': f'{frame_precision:.3f}',
                'R': f'{frame_recall:.3f}',
                'F1': f'{frame_f1:.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_frames / total_frames
        
        # Print epoch summary
        final_precision = true_positives / (true_positives + false_positives + 1e-8)
        final_recall = true_positives / (true_positives + false_negatives + 1e-8)
        final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-8)
        print(f"\n  Train Epoch Summary (Frame-Level):")
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
        
        # Track FRAME-LEVEL metrics
        total_frames = 0
        correct_frames = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
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
                
                # FRAME-LEVEL predictions
                predictions = (torch.sigmoid(logits) > self.classification_threshold).float()
                
                # Count metrics
                batch_frames = frame_labels.numel()
                total_frames += batch_frames
                correct_frames += (predictions == frame_labels).sum().item()
                
                true_positives += ((predictions == 1) & (frame_labels == 1)).sum().item()
                false_positives += ((predictions == 1) & (frame_labels == 0)).sum().item()
                false_negatives += ((predictions == 0) & (frame_labels == 1)).sum().item()
                
                # Update progress bar
                current_avg_loss = total_loss / (batch_iterator.n + 1)
                batch_iterator.set_postfix({'loss': f'{current_avg_loss:.4f}'})
        
        # Compute final FRAME-LEVEL metrics
        accuracy = correct_frames / total_frames if total_frames > 0 else 0
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        avg_loss = total_loss / len(val_loader)
        
        print(f"\n  Validation Summary (Frame-Level):")
        print(f"    TP={true_positives}, FP={false_positives}, FN={false_negatives}")
        print(f"    Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        return avg_loss, accuracy, precision, recall


# Register the method
DetectionMethodRegistry.register("transformer", TransformerDetection)