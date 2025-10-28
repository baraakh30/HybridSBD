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


class TransformerDetection(BaseDetectionMethod):
    """
    Transformer-based boundary detection using sequence modeling.
    
    Features:
    - Transformer encoder architecture
    - Positional encoding for temporal relationships
    - Binary boundary classification
    - Sequence-level training with balanced sampling
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
                - confidence_threshold: Classification threshold (default: 0.5)
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
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
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
            max_sequence_length=self.sequence_length
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.is_initialized = True
    
    def detect_boundaries(self, features: np.ndarray) -> List[int]:
        """
        Detect boundaries from feature sequences.
        
        Args:
            features: Feature array of shape (N, feature_dim)
            
        Returns:
            List of boundary frame indices
        """
        if len(features) < 2:
            return []
        
        boundaries = []
        
        # Process sequences with overlapping windows
        overlap = self.sequence_length // 2
        
        for start_idx in range(0, len(features) - self.sequence_length + 1, overlap):
            end_idx = start_idx + self.sequence_length
            sequence = features[start_idx:end_idx]
            
            # Get boundary probabilities for this sequence
            probs = self._predict_sequence(sequence)
            
            # Find boundaries above threshold
            sequence_boundaries = np.where(probs > self.confidence_threshold)[0]
            
            # Convert to global indices
            global_boundaries = [start_idx + idx for idx in sequence_boundaries]
            boundaries.extend(global_boundaries)
        
        # Remove duplicates and sort
        boundaries = sorted(set(boundaries))
        
        # Post-process boundaries
        boundaries = self._post_process_boundaries(
            boundaries, len(features), self.config.get("min_shot_duration", 30)
        )
        
        return boundaries
    
    def _predict_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict boundary probabilities for a sequence.
        
        Args:
            sequence: Feature sequence of shape (sequence_length, feature_dim)
            
        Returns:
            Boundary probabilities of shape (sequence_length-1,)
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            sequence_tensor = torch.from_numpy(sequence).float()
            sequence_tensor = sequence_tensor.unsqueeze(0).to(self.device)  # (1, seq_len, feature_dim)
            
            # Forward pass in frame mode for localization
            logits = self.model(sequence_tensor, mode='frame')  # (1, seq_len-1, 1)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).squeeze()  # (seq_len-1,)
            
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


class BoundaryDetectionTransformer(nn.Module):
    """
    Transformer model for boundary detection.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, 
                 num_layers: int = 6, num_heads: int = 8,
                 ff_dim: int = 2048, dropout: float = 0.1,
                 max_sequence_length: int = 1000):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
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
        
        # Output head for boundary classification
        # CHANGED: Predict if sequence contains boundary (sequence-level classification)
        # instead of predicting each frame pair (frame-level classification)
        self.sequence_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Binary classification: has_boundary or not
        )
        
        # Keep frame-pair classifier for inference (to localize boundaries within sequence)
        self.frame_pair_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenate adjacent frames
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Binary classification per frame pair
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mode: str = 'sequence') -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            mode: 'sequence' for sequence-level classification (training),
                  'frame' for frame-level classification (inference)
            
        Returns:
            - If mode='sequence': Binary logits of shape (batch_size, 1) indicating if sequence has boundary
            - If mode='frame': Boundary logits of shape (batch_size, sequence_length-1, 1) for each frame pair
        """
        batch_size, sequence_length, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        
        if mode == 'sequence':
            # SEQUENCE-LEVEL CLASSIFICATION (for training)
            # Use global average pooling over the sequence
            sequence_repr = encoded.mean(dim=1)  # (batch_size, hidden_dim)
            logits = self.sequence_classifier(sequence_repr)  # (batch_size, 1)
            return logits
        
        else:  # mode == 'frame'
            # FRAME-LEVEL CLASSIFICATION (for inference/localization)
            # Create frame pairs for boundary classification
            frame_pairs = []
            for i in range(sequence_length - 1):
                current_frame = encoded[:, i, :]      # (batch_size, hidden_dim)
                next_frame = encoded[:, i + 1, :]     # (batch_size, hidden_dim)
                pair = torch.cat([current_frame, next_frame], dim=1)  # (batch_size, hidden_dim * 2)
                frame_pairs.append(pair)
            
            # Stack pairs
            pairs = torch.stack(frame_pairs, dim=1)  # (batch_size, seq_len-1, hidden_dim * 2)
            
            # Classify each pair
            logits = self.frame_pair_classifier(pairs)  # (batch_size, seq_len-1, 1)
            
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
        
        # CRITICAL: Loss configuration for SEQUENCE-LEVEL classification
        # At sequence level: 59.1% have boundaries (71744), 40.9% don't (49652)
        # Balance ratio: 1:0.69 (positive:negative) - MUCH MORE BALANCED than frame-level!
        # 
        # DO NOT use frame-level weights (30-67x) - this causes predicting everything as positive!
        # We need much smaller weights appropriate for sequence-level imbalance.
        
        loss_type = config.get("loss_type", "focal")  # "bce" or "focal"
        
        if loss_type == "focal":
            # Use Focal Loss - excellent for sequence-level classification
            # At sequence level: ~59% have boundaries, ~41% don't (relatively balanced!)
            # Alpha should favor the MINORITY class (non-boundary sequences)
            focal_alpha = config.get("focal_alpha", 0.35)   # Alpha <0.5: favor minority (negative) class
            focal_gamma = config.get("focal_gamma", 2.0)    # Standard focusing parameter
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"Using Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
            print(f"  SEQUENCE-LEVEL: 59% positive (boundary), 41% negative (no boundary)")
            print(f"  Alpha {focal_alpha} (<0.5): Favors minority class (non-boundary sequences)")
            print(f"  This prevents predicting everything as positive!")
        elif loss_type == "adaptive_focal":
            # Use Adaptive Focal Loss - automatically adjusts alpha based on class distribution
            from focal_loss import AdaptiveFocalLoss
            focal_gamma = config.get("focal_gamma", 2.0)
            self.criterion = AdaptiveFocalLoss(gamma=focal_gamma)
            print(f"Using Adaptive Focal Loss: gamma={focal_gamma}")
            print(f"  Automatically adjusts alpha based on observed class distribution")
            print(f"  This adapts to actual imbalance during training")
        else:
            # Use BCE with SEQUENCE-LEVEL weights
            # Balance ratio: 1:0.69 (positive:negative) means minority is negative class
            # Need to DOWN-weight the majority class (positive/boundary sequences)
            # Use pos_weight = negative_count / positive_count = 49652 / 71744 = 0.69
            class_weights = config.get("class_weights", {"negative": 1.3, "positive": 1.0})
            pos_weight = torch.tensor([class_weights["positive"] / class_weights["negative"]])
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            print(f"Using BCE Loss with SEQUENCE-LEVEL weights: negative={class_weights['negative']}, positive={class_weights['positive']}")
            print(f"  Effective pos_weight = {pos_weight.item():.2f} (down-weights majority class)")
        
        print(f"  Goal: Balanced predictions, not everything-as-boundary!")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs
        )
        
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """
        Train for one epoch with SEQUENCE-LEVEL classification.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Track positive and negative class metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_positive_samples = 0
        
        # Add progress bar for batches
        batch_iterator = tqdm(train_loader, desc="Training batches", leave=False)
        
        for batch in batch_iterator:
            sequences, frame_labels = batch  # frame_labels shape: (batch, seq_len-1)
            sequences = sequences.to(self.device)
            
            # Convert frame-level labels to sequence-level labels
            # Label = 1 if any frame in sequence is a boundary, 0 otherwise
            sequence_labels = (frame_labels.sum(dim=1) > 0).float().unsqueeze(1)  # (batch, 1)
            sequence_labels = sequence_labels.to(self.device)
            
            # Forward pass in sequence mode
            self.optimizer.zero_grad()
            logits = self.model(sequences, mode='sequence')  # (batch, 1)
            
            # Compute loss
            loss = self.criterion(logits, sequence_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()  # Use 0.5 for balanced classification
            correct_predictions += (predictions == sequence_labels).sum().item()
            total_predictions += sequence_labels.numel()
            
            # Track boundary detection metrics at sequence level
            positive_mask = (sequence_labels == 1)
            total_positive_samples += positive_mask.sum().item()
            true_positives += ((predictions == 1) & (sequence_labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (sequence_labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (sequence_labels == 1)).sum().item()
            
            # Update progress bar with current stats
            current_avg_loss = total_loss / (batch_iterator.n + 1)
            current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            boundary_recall = true_positives / total_positive_samples if total_positive_samples > 0 else 0
            batch_iterator.set_postfix({
                'loss': f'{current_avg_loss:.4f}',
                'acc': f'{current_accuracy:.4f}',
                'seq_recall': f'{boundary_recall:.4f}'  # Sequence-level recall
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float, float, float]:
        """
        Validate the model with SEQUENCE-LEVEL classification.
        
        Returns:
            Tuple of (loss, accuracy, precision, recall)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Add progress bar for validation batches
        batch_iterator = tqdm(val_loader, desc="Validation batches", leave=False)
        
        with torch.no_grad():
            for batch in batch_iterator:
                sequences, frame_labels = batch
                sequences = sequences.to(self.device)
                
                # Convert frame-level labels to sequence-level labels
                sequence_labels = (frame_labels.sum(dim=1) > 0).float().unsqueeze(1)
                sequence_labels = sequence_labels.to(self.device)
                
                # Forward pass in sequence mode
                logits = self.model(sequences, mode='sequence')
                
                # Compute loss
                loss = self.criterion(logits, sequence_labels)
                total_loss += loss.item()
                
                # Collect predictions and targets
                predictions = (torch.sigmoid(logits) > 0.5).float()  # Use 0.5 for balanced classification
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(sequence_labels.cpu().numpy())
                
                # Update progress bar
                current_avg_loss = total_loss / (batch_iterator.n + 1)
                batch_iterator.set_postfix({'loss': f'{current_avg_loss:.4f}'})
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        accuracy = np.mean(all_predictions == all_targets)
        
        # Precision and recall at sequence level
        tp = np.sum((all_predictions == 1) & (all_targets == 1))
        fp = np.sum((all_predictions == 1) & (all_targets == 0))
        fn = np.sum((all_predictions == 0) & (all_targets == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, precision, recall


# Register the method
DetectionMethodRegistry.register("transformer", TransformerDetection)