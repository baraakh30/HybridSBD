"""
Training script for the transformer detection model.
"""

# CRITICAL: Import OpenCV configuration FIRST to set environment variables
import traceback
import opencv_config  # This MUST be imported before any OpenCV usage

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time
import gc  # For garbage collection to free memory

# Add the parent directory to the Python path to import our modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from core.config import VideoShotConfig
from detection.transformer_detection import TransformerDetection, TransformerTrainer
from core.coordinator import ProcessingCoordinator

warning_filter = "ignore"
import warnings 
warnings.filterwarnings(warning_filter)

# =============================================================================
# FEATURE TYPE CONSTANTS
# =============================================================================
# Individual feature types that can be used for training (not fused)
INDIVIDUAL_FEATURE_TYPES = ['hsv_histogram', 'cnn', 'clip', 'optical_flow']

# Fusion strategies available
FUSION_STRATEGIES = ['concatenation', 'weighted', 'pca', 'cross_transformer', 'hierarchical']

# =============================================================================
# REPRODUCIBILITY: Fixed random seeds for scientific validity
# =============================================================================
DEFAULT_RANDOM_SEED = 42

def set_random_seeds(seed: int = DEFAULT_RANDOM_SEED):
    """
    Set random seeds for reproducibility across all libraries.
    
    This is CRITICAL for scientific reproducibility:
    - Same seed = same results across runs
    - Essential for peer review and publication
    
    Args:
        seed: Random seed value (default: 42)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make PyTorch deterministic (may slightly impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seeds set to {seed} for reproducibility")


# =============================================================================
# SYNTHETIC BOUNDARY GENERATOR
# Creates mathematically realistic transitions in feature space
# =============================================================================
class SyntheticBoundaryGenerator:
    """
    Generates synthetic shot boundaries by mathematically combining features 
    from different video segments. This solves class imbalance and teaches
    the model what transitions "look like" in feature space.
    
    Inspired by TransNetV2's synthetic data augmentation approach.
    
    Transition Types:
    1. Hard Cut: Instant transition (most common in real videos)
    2. Dissolve: Linear blending over N frames (cross-fade)
    3. Fade: A → black → B (fade through black/white)
    4. Wipe: Gradual spatial replacement (simulated in feature space)
    """
    
    # Transition type probabilities (matching real-world distribution)
    TRANSITION_PROBS = {
        'hard_cut': 0.70,    # 70% hard cuts (most common)
        'dissolve': 0.20,    # 20% dissolves
        'fade': 0.05,        # 5% fades
        'wipe': 0.05         # 5% wipes
    }
    
    def __init__(self, feature_files: List[str], sequence_length: int = 100,
                 feature_dim: int = None, quiet: bool = False):
        """
        Initialize the synthetic boundary generator.
        
        Args:
            feature_files: List of paths to feature .npy files (different videos)
            sequence_length: Length of sequences to generate
            feature_dim: Expected feature dimension (auto-detected if None)
            quiet: Suppress print messages
        """
        self.feature_files = feature_files
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.quiet = quiet
        
        # Pre-load feature file metadata for efficient random access
        self._file_metadata = {}
        self._valid_files = []
        self._init_file_metadata()
        
        if not quiet:
            print(f"SyntheticBoundaryGenerator initialized with {len(self._valid_files)} valid video features")
    
    def _init_file_metadata(self):
        """Pre-scan feature files to get their shapes without loading full data."""
        for fpath in self.feature_files:
            try:
                if not os.path.exists(fpath):
                    continue
                    
                # Read only the header to get shape
                with open(fpath, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    shape, _, dtype = np.lib.format._read_array_header(f, version)
                
                if len(shape) == 2 and shape[0] >= self.sequence_length // 2:
                    self._file_metadata[fpath] = {
                        'shape': shape,
                        'dtype': dtype,
                        'num_frames': shape[0],
                        'feature_dim': shape[1]
                    }
                    self._valid_files.append(fpath)
                    
                    # Track MAXIMUM feature dimension across all files
                    # This handles variable PCA dimensions
                    if self.feature_dim is None:
                        self.feature_dim = shape[1]
                    else:
                        self.feature_dim = max(self.feature_dim, shape[1])
                        
            except Exception:
                continue
    
    def _load_random_segment(self, min_frames: int = 30) -> np.ndarray:
        """
        Load a random segment from a random video.
        
        Args:
            min_frames: Minimum frames to load
            
        Returns:
            Feature array of shape (num_frames, feature_dim)
        """
        if not self._valid_files:
            return None
            
        # Pick a random file
        fpath = np.random.choice(self._valid_files)
        meta = self._file_metadata[fpath]
        
        # Pick a random start position
        max_start = max(0, meta['num_frames'] - min_frames)
        start_idx = np.random.randint(0, max_start + 1)
        end_idx = min(start_idx + min_frames, meta['num_frames'])
        
        # Load only the segment we need
        try:
            with open(fpath, 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, _, dtype = np.lib.format._read_array_header(f, version)
                
                bytes_per_row = shape[1] * dtype.itemsize
                f.seek(start_idx * bytes_per_row, 1)
                
                num_rows = end_idx - start_idx
                data = np.frombuffer(f.read(num_rows * bytes_per_row), dtype=dtype)
                return data.reshape(num_rows, shape[1]).copy()
        except Exception:
            return None
    
    def _pad_to_feature_dim(self, segment: np.ndarray) -> np.ndarray:
        """
        Pad segment to match the target feature dimension.
        
        This handles cases where PCA produces different dimensions for different videos.
        
        Args:
            segment: Feature array of shape (num_frames, current_dim)
            
        Returns:
            Padded array of shape (num_frames, feature_dim)
        """
        if segment is None:
            return None
            
        current_dim = segment.shape[1]
        
        if current_dim == self.feature_dim:
            return segment
        elif current_dim < self.feature_dim:
            # Pad with zeros
            padding = np.zeros((segment.shape[0], self.feature_dim - current_dim))
            return np.concatenate([segment, padding], axis=1)
        else:
            # Truncate (shouldn't happen often, but handle it)
            return segment[:, :self.feature_dim]
    
    def generate_synthetic_sequence(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Generate a synthetic sequence with a mathematically created boundary.
        
        Returns:
            Tuple of (features, labels, transition_type):
            - features: (sequence_length, feature_dim) array
            - labels: (sequence_length - 1,) binary labels
            - transition_type: string describing the transition
        """
        # Load two random segments from DIFFERENT videos
        segment_a = self._load_random_segment(self.sequence_length)
        segment_b = self._load_random_segment(self.sequence_length)
        
        if segment_a is None or segment_b is None:
            return None, None, None
        
        # IMPORTANT: Pad segments to match feature_dim (handles variable PCA dimensions)
        segment_a = self._pad_to_feature_dim(segment_a)
        segment_b = self._pad_to_feature_dim(segment_b)
        
        # Choose transition type based on probabilities
        transition_type = np.random.choice(
            list(self.TRANSITION_PROBS.keys()),
            p=list(self.TRANSITION_PROBS.values())
        )
        
        # Choose transition position (middle 40-60% of sequence for stability)
        cut_point = np.random.randint(
            int(self.sequence_length * 0.35),
            int(self.sequence_length * 0.65)
        )
        
        # Generate the synthetic sequence based on transition type
        if transition_type == 'hard_cut':
            features, labels = self._create_hard_cut(segment_a, segment_b, cut_point)
        elif transition_type == 'dissolve':
            features, labels = self._create_dissolve(segment_a, segment_b, cut_point)
        elif transition_type == 'fade':
            features, labels = self._create_fade(segment_a, segment_b, cut_point)
        else:  # wipe
            features, labels = self._create_wipe(segment_a, segment_b, cut_point)
        
        return features, labels, transition_type
    
    def _create_hard_cut(self, seg_a: np.ndarray, seg_b: np.ndarray, 
                         cut_point: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a hard cut transition (instant change).
        
        This is the most common type (70% of real transitions).
        The feature vector changes abruptly from A to B.
        """
        result = np.zeros((self.sequence_length, self.feature_dim))
        labels = np.zeros(self.sequence_length - 1)
        
        # Use beginning of seg_a and end of seg_b (different content)
        a_frames = min(cut_point, len(seg_a))
        b_frames = min(self.sequence_length - cut_point, len(seg_b))
        
        # Part A (before cut)
        result[:a_frames] = seg_a[:a_frames]
        
        # Part B (after cut) - use END of segment B for different content
        b_start = max(0, len(seg_b) - b_frames)
        result[cut_point:cut_point + b_frames] = seg_b[b_start:b_start + b_frames]
        
        # Boundary at cut point
        if 0 <= cut_point - 1 < len(labels):
            labels[cut_point - 1] = 1.0
        
        return result, labels
    
    def _create_dissolve(self, seg_a: np.ndarray, seg_b: np.ndarray,
                         cut_point: int, transition_frames: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a dissolve transition (linear cross-fade).
        
        Features blend linearly: result = α*A + (1-α)*B
        where α goes from 1 to 0 over the transition window.
        """
        if transition_frames is None:
            transition_frames = np.random.randint(5, 20)  # 5-20 frame dissolve
        
        result = np.zeros((self.sequence_length, self.feature_dim))
        labels = np.zeros(self.sequence_length - 1)
        
        half_trans = transition_frames // 2
        trans_start = max(0, cut_point - half_trans)
        trans_end = min(self.sequence_length, cut_point + half_trans + 1)
        
        # Part A (before transition)
        a_frames = min(trans_start, len(seg_a))
        result[:a_frames] = seg_a[:a_frames]
        
        # Transition zone (linear blend)
        for i in range(trans_start, trans_end):
            alpha = (i - trans_start) / max(1, trans_end - trans_start)
            a_idx = min(i, len(seg_a) - 1)
            b_idx = min(i, len(seg_b) - 1)
            result[i] = (1 - alpha) * seg_a[a_idx] + alpha * seg_b[b_idx]
        
        # Part B (after transition)
        for i in range(trans_end, self.sequence_length):
            b_idx = min(i, len(seg_b) - 1)
            result[i] = seg_b[b_idx]
        
        # Boundary at center of transition
        if 0 <= cut_point - 1 < len(labels):
            labels[cut_point - 1] = 1.0
        
        return result, labels
    
    def _create_fade(self, seg_a: np.ndarray, seg_b: np.ndarray,
                     cut_point: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a fade transition (A → black → B).
        
        Features fade to zero (black) then fade in from zero.
        Common in scene transitions.
        """
        fade_frames = np.random.randint(3, 10)  # 3-10 frames per fade
        
        result = np.zeros((self.sequence_length, self.feature_dim))
        labels = np.zeros(self.sequence_length - 1)
        
        fade_out_start = max(0, cut_point - fade_frames)
        fade_in_end = min(self.sequence_length, cut_point + fade_frames)
        
        # Part A (before fade out)
        for i in range(fade_out_start):
            a_idx = min(i, len(seg_a) - 1)
            result[i] = seg_a[a_idx]
        
        # Fade out (A → black)
        for i in range(fade_out_start, cut_point):
            alpha = (i - fade_out_start) / max(1, cut_point - fade_out_start)
            a_idx = min(i, len(seg_a) - 1)
            result[i] = (1 - alpha) * seg_a[a_idx]  # Fade to zero
        
        # Fade in (black → B)
        for i in range(cut_point, fade_in_end):
            alpha = (i - cut_point) / max(1, fade_in_end - cut_point)
            b_idx = min(i, len(seg_b) - 1)
            result[i] = alpha * seg_b[b_idx]  # Fade from zero
        
        # Part B (after fade in)
        for i in range(fade_in_end, self.sequence_length):
            b_idx = min(i, len(seg_b) - 1)
            result[i] = seg_b[b_idx]
        
        # Boundary at center
        if 0 <= cut_point - 1 < len(labels):
            labels[cut_point - 1] = 1.0
        
        return result, labels
    
    def _create_wipe(self, seg_a: np.ndarray, seg_b: np.ndarray,
                     cut_point: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a wipe transition (gradual spatial replacement).
        
        In feature space, we simulate this by blending with a non-linear curve.
        Uses sigmoid blending for smooth wipe effect.
        """
        wipe_frames = np.random.randint(8, 25)  # 8-25 frame wipe
        
        result = np.zeros((self.sequence_length, self.feature_dim))
        labels = np.zeros(self.sequence_length - 1)
        
        half_wipe = wipe_frames // 2
        wipe_start = max(0, cut_point - half_wipe)
        wipe_end = min(self.sequence_length, cut_point + half_wipe + 1)
        
        # Part A (before wipe)
        for i in range(wipe_start):
            a_idx = min(i, len(seg_a) - 1)
            result[i] = seg_a[a_idx]
        
        # Wipe zone (sigmoid blend for smooth transition)
        for i in range(wipe_start, wipe_end):
            # Sigmoid function for smooth S-curve blend
            t = (i - wipe_start) / max(1, wipe_end - wipe_start)
            alpha = 1 / (1 + np.exp(-10 * (t - 0.5)))  # Sigmoid centered at 0.5
            
            a_idx = min(i, len(seg_a) - 1)
            b_idx = min(i, len(seg_b) - 1)
            result[i] = (1 - alpha) * seg_a[a_idx] + alpha * seg_b[b_idx]
        
        # Part B (after wipe)
        for i in range(wipe_end, self.sequence_length):
            b_idx = min(i, len(seg_b) - 1)
            result[i] = seg_b[b_idx]
        
        # Boundary at center
        if 0 <= cut_point - 1 < len(labels):
            labels[cut_point - 1] = 1.0
        
        return result, labels


def print_memory_usage(prefix: str = ""):
    """Print current memory usage for debugging."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024  # Convert to MB
        print(f"{prefix}Memory usage: {mem_mb:.1f} MB ({mem_mb/1024:.2f} GB)")
    except ImportError:
        # psutil not available, skip memory reporting
        pass


def check_features_exist_and_valid(features_dir: str, video_name: str, 
                                  feature_names: List[str] = None,
                                  fusion_strategy: str = "hierarchical", check_only_existence: bool = False) -> Dict[str, bool]:
    """
    Check if all required features exist and are valid for a video.
    
    Args:
        features_dir: Directory containing feature files
        video_name: Name of the video (without extension)
        feature_names: List of feature types to check (if None, uses default set)
        fusion_strategy: Fusion strategy name to check for strategy-specific fused features
        
    Returns:
        Dictionary with validation results:
        - 'all_individual_valid': True if all individual features exist and are valid
        - 'fused_valid': True if fused features exist and are valid
        - 'individual_status': Dict mapping feature names to their validation status
        - 'missing_features': List of missing or corrupted feature names
    """
    if feature_names is None:
        feature_names = ['hsv_histogram', 'cnn', 'clip', 'optical_flow']
    
    individual_status = {}
    missing_features = []
    
    # Expected feature dimensions
    expected_dims = {
        'optical_flow': 40,
        'clip': 512,
        'cnn': 2048,
        'hsv_histogram': 768
    }
    
    # Check individual features
    for feature_name in feature_names:
        # Try multiple naming patterns
        patterns = [
            os.path.join(features_dir, f"{video_name}_{feature_name}.npy"),
            os.path.join(features_dir, f"{video_name}___{feature_name}.npy"),
        ]
        
        feature_file = None
        for pattern in patterns:
            if os.path.exists(pattern):
                feature_file = pattern
                break
        
        try:
            if feature_file is not None:
                # Try to load and validate the numpy array
                if not check_only_existence:
                    feature_array = np.load(feature_file)
                    expected_dim = expected_dims.get(feature_name)
                    
                    if (feature_array.size > 0 and 
                        len(feature_array.shape) >= 2 and 
                        not np.all(feature_array == 0) and  # Check if not all zeros
                        (expected_dim is None or feature_array.shape[1] == expected_dim)):  # Check correct dimensions
                        individual_status[feature_name] = True
                    else:
                        individual_status[feature_name] = False
                        missing_features.append(feature_name)
                else:
                    individual_status[feature_name] = True
            else:
                individual_status[feature_name] = False
                missing_features.append(feature_name)
        except Exception:
            # Corrupted file or loading error
            individual_status[feature_name] = False
            missing_features.append(feature_name)
    
    all_individual_valid = all(individual_status.values())
    
    # Check fused features (both strategy-specific and generic)
    fused_valid = True ########################
    fused_files_to_check = [
        # os.path.join(features_dir, f"{video_name}_{fusion_strategy}.npy"),
        # os.path.join(features_dir, f"{video_name}_fused.npy"),
        # os.path.join(features_dir, f"{video_name}_features.npy")  # Legacy naming
    ]
    
    for fused_file in fused_files_to_check:
        try:
            if os.path.exists(fused_file):
                fused_array = np.load(fused_file)
                if (fused_array.size > 0 and 
                    len(fused_array.shape) >= 2 and 
                    not np.all(fused_array == 0)):  # Check if not all zeros
                    fused_valid = True
                    break
        except Exception:
            continue
    
    return {
        'all_individual_valid': all_individual_valid,
        'fused_valid': fused_valid,
        'individual_status': individual_status,
        'missing_features': missing_features,
        'needs_extraction': not (all_individual_valid and fused_valid)
    }


def load_individual_features(features_dir: str, video_name: str, feature_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Load individual features for a video.
    
    Args:
        features_dir: Directory containing feature files
        video_name: Name of the video (without extension)
        feature_names: List of feature types to load
        
    Returns:
        Dictionary mapping feature names to arrays
    """
    features = {}
    for feature_name in feature_names:
        # Try multiple naming patterns
        patterns = [
            f"{video_name}_{feature_name}.npy",      # Standard: video_clip.npy
            f"{video_name}___{feature_name}.npy",    # Triple underscore: video___clip.npy
        ]
        
        loaded = False
        for pattern in patterns:
            feature_file = os.path.join(features_dir, pattern)
            if os.path.exists(feature_file):
                features[feature_name] = np.load(feature_file)
                loaded = True
                break
        
        # if not loaded:
        #     print(f"Warning: Feature file not found for {video_name}_{feature_name}")
    return features


def load_and_fuse_individual_features(features_dir: str, video_name: str, fusion_strategy: str,
                                     quiet: bool = False, file_name_only: bool = False) -> np.ndarray:
    """
    Load individual features, fuse them using the specified strategy, and save the result.
    
    Args:
        features_dir: Directory containing feature files
        video_name: Name of the video (without extension)
        fusion_strategy: Name of the fusion strategy to use
        quiet: Whether to suppress output messages
        
    Returns:
        Fused feature array
    """
    # Standard feature types that should be available
    feature_types = ['hsv_histogram', 'cnn', 'clip', 'optical_flow']
    
    # Load individual features
    features = load_individual_features(features_dir, video_name, feature_types)
    
    if not features:
        raise FileNotFoundError(f"No individual features found for {video_name}")
    
    # Create minimal config for fusion
    from core.config import VideoShotConfig, FusionConfig
    config = VideoShotConfig()
    config.fusion.strategy = fusion_strategy
    
    # Initialize coordinator to get fusion strategy
    coordinator = ProcessingCoordinator(config)
    coordinator.feature_extractors = {}  # Mock feature extractors for fusion initialization
    
    # Set up feature dimensions based on loaded features
    feature_dims = {}
    for name, feat_array in features.items():
        feature_dims[name] = feat_array.shape[1]
    
    # Initialize fusion strategy manually to avoid full coordinator initialization
    from fusion import FusionStrategyRegistry
    
    if fusion_strategy == "concatenation":
        fusion = FusionStrategyRegistry.get(
            "concatenation",
            feature_dims=feature_dims,
            device="cpu"  # Use CPU for training data processing
        )
    elif fusion_strategy == "weighted":
        fusion = FusionStrategyRegistry.get(
            "weighted",
            feature_dims=feature_dims,
            optimization_method="manual",
            manual_weights=config.fusion.manual_weights,
            device="cpu"
        )
    elif fusion_strategy == "pca":
        fusion = FusionStrategyRegistry.get(
            "pca",
            feature_dims=feature_dims,
            n_components=256,  # Fixed number of components for consistency
            variance_retained=config.fusion.pca_variance_retained,
            device="cpu"
        )
    elif fusion_strategy == "cross_transformer":
        fusion = FusionStrategyRegistry.get(
            "cross_transformer",
            feature_dims=feature_dims,
            hidden_dim=config.fusion.transformer_hidden_dim,
            num_heads=config.fusion.transformer_heads,
            num_layers=config.fusion.transformer_layers,
            dropout=config.fusion.transformer_dropout,
            device="cpu"
        )
    elif fusion_strategy == "hierarchical":
        fusion = FusionStrategyRegistry.get(
            "hierarchical",
            feature_dims=feature_dims,
            conv_kernel=config.fusion.hierarchical_conv_kernel,
            attention_heads=config.fusion.hierarchical_attention_heads,
            device="cpu"
        )
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    # Initialize and fuse features
    fusion.initialize()
    
    try:
        fused_features = fusion.fuse_features(features)
    except ValueError as e:
        # Handle PCA component errors for very short videos
        if "n_components" in str(e) and "must be between" in str(e):
            print(f"Warning: Cannot fuse features for {video_name} - video too short: {e}")
            raise ValueError(f"Video {video_name} has insufficient frames for fusion")
        raise
    
    # Save the fused features for future use
    strategy_file = os.path.join(features_dir, f"{video_name}_{fusion_strategy}.npy")
    try:
        np.save(strategy_file, fused_features)
        if not quiet:
            print(f"Saved {fusion_strategy} fused features for {video_name}: {fused_features.shape}")
    except Exception as e:
        print(f"Error: Failed to save fused features for {video_name}: {e}")
        raise IOError(f"{fused_features.size} requested and 0 written")
    
    if file_name_only:
        return strategy_file
    return fused_features


def load_single_feature(features_dir: str, video_name: str, feature_type: str,
                       quiet: bool = False, file_name_only: bool = False) -> np.ndarray:
    """
    Load a single individual feature type for a video.
    
    Args:
        features_dir: Directory containing feature files
        video_name: Name of the video (without extension)
        feature_type: Feature type to load (hsv_histogram, cnn, clip, optical_flow)
        quiet: Whether to suppress output messages
        file_name_only: If True, return the file path instead of loaded array
        
    Returns:
        Feature array or file path if file_name_only=True
    """
    if feature_type not in INDIVIDUAL_FEATURE_TYPES:
        raise ValueError(f"Invalid feature type: {feature_type}. Must be one of {INDIVIDUAL_FEATURE_TYPES}")
    
    # Try multiple naming patterns
    patterns = [
        f"{video_name}_{feature_type}.npy",      # Standard: video_clip.npy
        f"{video_name}___{feature_type}.npy",    # Triple underscore: video___clip.npy
    ]
    
    for pattern in patterns:
        feature_file = os.path.join(features_dir, pattern)
        if os.path.exists(feature_file):
            if file_name_only:
                return feature_file
            return np.load(feature_file)
    
    raise FileNotFoundError(f"Could not find {feature_type} features for {video_name} in {features_dir}")


def is_individual_feature(feature_type: str) -> bool:
    """Check if a feature type is an individual feature (not a fusion strategy)."""
    return feature_type in INDIVIDUAL_FEATURE_TYPES


def load_features_for_training(features_dir: str, video_name: str, feature_type: str,
                               quiet: bool = False, file_name_only: bool = False) -> np.ndarray:
    """
    Load features for training - either individual feature or fused features.
    
    Args:
        features_dir: Directory containing feature files
        video_name: Name of the video (without extension)
        feature_type: Either an individual feature type (hsv_histogram, cnn, clip, optical_flow)
                     or a fusion strategy (concatenation, weighted, pca, cross_transformer, hierarchical)
        quiet: Whether to suppress output messages
        file_name_only: If True, return the file path instead of loaded array
        
    Returns:
        Feature array or file path if file_name_only=True
    """
    if is_individual_feature(feature_type):
        return load_single_feature(features_dir, video_name, feature_type, quiet, file_name_only)
    else:
        # It's a fusion strategy
        return load_fused_features(features_dir, video_name, feature_type, quiet, file_name_only)


def load_fused_features(features_dir: str, video_name: str, fusion_strategy: str = "hierarchical", 
                      quiet: bool = False,file_name_only: bool = False) -> np.ndarray:
    """
    Load pre-computed fused features for a video with fallback to individual feature fusion.
    
    Args:
        features_dir: Directory containing feature files
        video_name: Name of the video (without extension)
        fusion_strategy: Name of the fusion strategy to use
        quiet: Whether to suppress output messages
        
    Returns:
        Fused feature array
    """
    # Try loading strategy-specific fused features with multiple naming patterns
    strategy_patterns = [
        f"{video_name}_{fusion_strategy}.npy",      # Standard: video_pca.npy
        f"{video_name}___{fusion_strategy}.npy",    # Triple underscore: video___pca.npy
    ]
    
    for pattern in strategy_patterns:
        strategy_file = os.path.join(features_dir, pattern)
        if os.path.exists(strategy_file):
            if file_name_only:
                return strategy_file
            return np.load(strategy_file)
    
    # Fallback to generic fused features (try both patterns)
    fused_patterns = [
        f"{video_name}_fused.npy",
        f"{video_name}___fused.npy",
    ]
    
    for pattern in fused_patterns:
        fused_file = os.path.join(features_dir, pattern)
        if os.path.exists(fused_file):
            if file_name_only:
                return fused_file
            return np.load(fused_file)
    
    # Fallback to old naming convention (try both patterns)
    old_patterns = [
        f"{video_name}_features.npy",
        f"{video_name}___features.npy",
    ]
    
    for pattern in old_patterns:
        old_file = os.path.join(features_dir, pattern)
        if os.path.exists(old_file):
            if file_name_only:
                return old_file
            return np.load(old_file)
    
    # # Last resort: load individual features and fuse them
    # if not quiet:
    #     print(f"No pre-computed fused features found for {video_name}, attempting to load and fuse individual features...")
    try:
        return load_and_fuse_individual_features(features_dir, video_name, fusion_strategy, quiet,file_name_only)
    except Exception as e:
        raise FileNotFoundError(f"Could not load or create fused features for {video_name}: {e}")


def create_custom_fusion(features_dir: str, video_names: List[str], 
                        feature_weights: Dict[str, float] = None) -> Dict[str, np.ndarray]:
    """
    Create custom fusion from individual features.
    
    Args:
        features_dir: Directory containing feature files
        video_names: List of video names to process
        feature_weights: Optional weights for weighted fusion
        
    Returns:
        Dictionary mapping video names to custom fused features
    """
    if feature_weights is None:
        feature_weights = {
            'hsv_histogram': 0.2,
            'cnn': 0.3,
            'clip': 0.3,
            'optical_flow': 0.2
        }
    
    custom_fused = {}
    available_features = list(feature_weights.keys())
    
    for video_name in video_names:
        features = load_individual_features(features_dir, video_name, available_features)
        
        if not features:
            print(f"No features found for {video_name}")
            continue
        
        # Simple weighted concatenation
        weighted_features = []
        for feature_name, weight in feature_weights.items():
            if feature_name in features:
                feat = features[feature_name]
                # Normalize and weight
                feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
                weighted_features.append(feat_norm * weight)
        
        if weighted_features:
            custom_fused[video_name] = np.concatenate(weighted_features, axis=1)
    
    return custom_fused


class BoundaryDataset(torch.utils.data.Dataset):
    """Dataset for training boundary detection models with lazy loading to reduce memory usage.
    
    IMPORTANT FOR PUBLICATION:
    - Supports video-level train/val/test splitting to prevent data leakage
    - Uses fixed random seeds for reproducibility
    - All sequences from a video stay in the same split
    - Synthetic boundary generation for class balance
    - Uses NON-overlapping windows in validation_mode to prevent duplicate counting
    
    MULTI-TASK LEARNING SUPPORT:
    - When use_multi_task=True, returns (features, class_labels, duration_labels)
    - duration_labels: Frame-wise transition duration (0 for non-transitions, N for gradual)
    - This teaches the model to distinguish cuts (duration=2) from gradual (duration>2)
    """
    
    def __init__(self, features_dir: str, ground_truth_file: str, 
                 sequence_length: int = 100, overlap: int = 50, 
                 fusion_strategy: str = "hierarchical", quiet: bool = False,
                 lazy_loading: bool = True, video_names: List[str] = None,
                 include_all_non_boundary: bool = False,
                 synthetic_ratio: float = 0.5,
                 validation_mode: bool = False,
                 feature_type: str = None,
                 use_multi_task: bool = False):
        """
        Initialize dataset.
        
        Args:
            features_dir: Directory containing extracted features
            ground_truth_file: JSON file with ground truth boundaries OR pre-loaded dict
            sequence_length: Length of input sequences
            overlap: Overlap between sequences (IGNORED in validation_mode - uses stride=sequence_length)
            fusion_strategy: Name of the fusion strategy to use for loading features (deprecated, use feature_type)
            quiet: Whether to suppress warning messages
            lazy_loading: If True, only store metadata and load features on-demand (saves memory)
            video_names: List of video names to include (for video-level splitting).
                        If None, uses all videos in ground truth.
                        CRITICAL: Use this to prevent data leakage between train/val/test!
            include_all_non_boundary: If True, include ALL non-boundary sequences (for validation).
                                     If False, subsample non-boundary sequences (for training balance).
            synthetic_ratio: Ratio of synthetic samples to add (0.5 = 50% more samples are synthetic)
            validation_mode: If True, uses NON-overlapping windows (stride = sequence_length)
                            to prevent the same boundary being counted multiple times in metrics.
                            CRITICAL for correct validation metrics!
            feature_type: Feature type to use - can be either:
                         - Individual feature: 'hsv_histogram', 'cnn', 'clip', 'optical_flow'
                         - Fusion strategy: 'concatenation', 'weighted', 'pca', 'cross_transformer', 'hierarchical'
                         If None, uses fusion_strategy parameter for backward compatibility.
            use_multi_task: If True, also return duration labels for multi-task learning.
                           Duration labels help the model learn gradual transition signatures.
        """
        self.features_dir = features_dir
        self.sequence_length = sequence_length
        self.validation_mode = validation_mode
        self.use_multi_task = use_multi_task  # NEW: Multi-task learning flag
        
        # CRITICAL: In validation mode, use NON-overlapping windows
        # This ensures each boundary is counted EXACTLY ONCE in metrics.
        # 
        # IMPORTANT: Use stride = sequence_length for truly non-overlapping
        # Previous stride = sequence_length - 5 still caused some overlap issues
        if validation_mode:
            self.overlap = sequence_length  # stride = 100 for seq_len=100 (no overlap)
            if not quiet:
                print(f"  VALIDATION MODE: Using non-overlapping windows (stride={self.overlap})")
        else:
            self.overlap = overlap  # Training: use overlapping windows for augmentation
        # Handle feature_type vs fusion_strategy for backward compatibility
        # feature_type takes precedence if specified
        if feature_type is not None:
            self.feature_type = feature_type
            self.fusion_strategy = feature_type  # For backward compat in file pattern matching
        else:
            self.feature_type = fusion_strategy  # Use fusion_strategy as feature_type
            self.fusion_strategy = fusion_strategy
        
        self.is_individual_feature = is_individual_feature(self.feature_type)
        self.quiet = quiet
        self.lazy_loading = lazy_loading
        self._feature_dim = None  # Will be set after first valid feature load
        self.training = not validation_mode  # Training mode flag - enables augmentations
        self.video_names = video_names  # For video-level splitting
        self.include_all_non_boundary = include_all_non_boundary
        self.synthetic_ratio = synthetic_ratio
        
        # Load ground truth (can be path or pre-loaded dict)
        if isinstance(ground_truth_file, dict):
            self.ground_truth = ground_truth_file
        else:
            with open(ground_truth_file, 'r') as f:
                self.ground_truth = json.load(f)
        
        # Filter to only specified videos if video_names provided
        if video_names is not None:
            self.ground_truth = {k: v for k, v in self.ground_truth.items() 
                                if k in video_names or k.replace('.mp4', '') in video_names
                                or k + '.mp4' in video_names}
        
        # Create training samples (metadata only if lazy loading)
        self.samples = self._create_samples()
        
        # Collect all valid feature files for synthetic generation
        self._all_feature_files = self._collect_feature_files()
        
        # Initialize synthetic boundary generator (only for training)
        self._synthetic_generator = None
        if self.training and synthetic_ratio > 0 and len(self._all_feature_files) > 10:
            self._init_synthetic_generator()
        
        # Add synthetic samples to the dataset
        self._add_synthetic_samples()
        
        # Cache for loaded features when lazy loading (LRU cache to limit memory)
        if self.lazy_loading:
            from functools import lru_cache
            self._feature_cache = {}
        
        # Print multi-task info
        if use_multi_task and not quiet:
            print(f"  MULTI-TASK MODE: Will return (features, class_labels, duration_labels)")
            print(f"    Duration labels teach the model: cuts (dur=2) vs gradual (dur>2)")
    
    def _collect_feature_files(self) -> List[str]:
        """Collect all valid feature files from the dataset."""
        feature_files = []
        for sample in self.samples:
            if 'feature_file' in sample and sample['feature_file']:
                feature_files.append(sample['feature_file'])
        return list(set(feature_files))  # Remove duplicates
    
    def _init_synthetic_generator(self):
        """Initialize the synthetic boundary generator."""
        if not self.quiet:
            print(f"  Initializing SyntheticBoundaryGenerator with {len(self._all_feature_files)} feature files...")
        
        self._synthetic_generator = SyntheticBoundaryGenerator(
            feature_files=self._all_feature_files,
            sequence_length=self.sequence_length,
            feature_dim=self._feature_dim,
            quiet=self.quiet
        )
    
    def _add_synthetic_samples(self):
        """Add synthetic boundary samples to balance the dataset."""
        if not self.training or self.synthetic_ratio <= 0:
            return
            
        if self._synthetic_generator is None:
            return
        
        # Count current boundary vs non-boundary samples
        boundary_samples = sum(1 for s in self.samples 
                              if any(s['start_frame'] <= b < s['end_frame'] 
                                    for b in s.get('boundaries', [])))
        non_boundary_samples = len(self.samples) - boundary_samples
        
        # Calculate how many synthetic samples to add
        # Goal: Balance boundaries with ~50% synthetic data
        num_synthetic = int(len(self.samples) * self.synthetic_ratio)
        
        if not self.quiet:
            print(f"  Adding {num_synthetic} synthetic boundary samples...")
            print(f"    Original: {boundary_samples} boundary, {non_boundary_samples} non-boundary")
        
        # Add synthetic sample markers (actual generation happens in __getitem__)
        for i in range(num_synthetic):
            self.samples.append({
                'synthetic': True,
                'synthetic_idx': i
            })
        
        if not self.quiet:
            print(f"    After synthetic: {len(self.samples)} total samples")
    
    def _create_samples(self) -> List[Dict]:
        """Create training samples metadata (lazy loading to save memory)."""
        samples = []
        
        total_videos = len(self.ground_truth)
        videos_processed = 0
        videos_with_features = 0
        videos_skipped = 0
        
        if not self.quiet:
            print(f"Creating training samples from {total_videos} videos in ground truth...")
            if self.lazy_loading:
                print(f"  Using lazy loading mode (low memory usage)")
        
        for video_name, video_data in self.ground_truth.items():
            # Strip .mp4 extension from video name to match feature filenames
            video_name_stem = video_name.replace('.mp4', '') if video_name.endswith('.mp4') else video_name
            
            # Extract transitions from ground truth
            # Handle both old format (list of boundaries) and new format (dict with transitions)
            # CRITICAL FIX: Store full transition ranges [start, end] for multi-frame labeling
            transitions = []  # List of (start_frame, end_frame) tuples
            
            if isinstance(video_data, dict) and 'transitions' in video_data:
                # ClipShots format: transitions = [[start, end], ...]
                # Filter out invalid transitions (start >= end, negative values, corrupted)
                for t in video_data['transitions']:
                    if len(t) >= 2:
                        start_frame, end_frame = int(t[0]), int(t[1])
                        # Validate transition: start must be < end, both must be >= 0
                        if start_frame >= 0 and end_frame > start_frame:
                            transitions.append((start_frame, end_frame))
                        elif not self.quiet and start_frame >= end_frame:
                            pass  # Silently skip invalid transitions
            elif isinstance(video_data, list):
                # Old format: direct list of single-frame boundaries
                for boundary in video_data:
                    if isinstance(boundary, (int, float)) and boundary >= 0:
                        transitions.append((int(boundary), int(boundary)))  # Single frame = cut
                    elif isinstance(boundary, (list, tuple)) and len(boundary) >= 2:
                        start_frame, end_frame = int(boundary[0]), int(boundary[1])
                        if start_frame >= 0 and end_frame >= start_frame:
                            transitions.append((start_frame, max(end_frame, start_frame)))
            else:
                if not self.quiet:
                    print(f"Warning: Unknown annotation format for {video_name}")
                continue
            
            if self.lazy_loading:
                # Only load feature file metadata (shape) without loading full data
                # Use unified feature loading that handles both individual features and fused features
                strategy_file = None
                
                try:
                    # Use the unified loading function that handles both individual features and fusion strategies
                    strategy_file = load_features_for_training(
                        self.features_dir, video_name_stem, self.feature_type, 
                        quiet=True, file_name_only=True
                    )
                except FileNotFoundError:
                    # Fallback: try legacy naming patterns only for fusion strategies
                    if not self.is_individual_feature:
                        naming_patterns = [
                            f"{video_name_stem}_{self.fusion_strategy}.npy",
                            f"{video_name_stem}___{self.fusion_strategy}.npy",
                        ]
                        
                        for pattern in naming_patterns:
                            candidate = os.path.join(self.features_dir, pattern)
                            if os.path.exists(candidate):
                                strategy_file = candidate
                                break
                        
                        # Try fallback files with both naming patterns
                        if strategy_file is None:
                            fallback_patterns = [
                                f"{video_name_stem}_fused.npy",
                                f"{video_name_stem}___fused.npy",
                                f"{video_name_stem}_features.npy",
                                f"{video_name_stem}___features.npy",
                            ]
                            
                            for pattern in fallback_patterns:
                                candidate = os.path.join(self.features_dir, pattern)
                                if os.path.exists(candidate):
                                    strategy_file = candidate
                                    break
                
                if strategy_file is None:
                    # if not self.quiet:
                    #     print(f"Warning: Could not find {self.feature_type} features for {video_name_stem}")
                    videos_skipped += 1
                    continue

                # Load only the shape without loading full array (using mmap)
                try:
                    with open(strategy_file, 'rb') as f:
                        # Read numpy header to get shape
                        version = np.lib.format.read_magic(f)
                        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
                    
                    num_frames = shape[0]
                    # Track the MAXIMUM feature dimension across all files
                    # This is important for PCA which can have variable dimensions
                    if len(shape) == 2:
                        current_dim = shape[1]
                        if self._feature_dim is None:
                            self._feature_dim = current_dim
                        else:
                            # Keep track of the maximum dimension
                            self._feature_dim = max(self._feature_dim, current_dim)
                    
                    videos_with_features += 1
                except Exception as e:
                    if not self.quiet:
                        print(f"Warning: Could not read feature file for {video_name_stem}: {e}")
                    videos_skipped += 1
                    continue
                
                # Store validated transitions as metadata
                # Each transition is (start_frame, end_frame) tuple
                # Filter transitions that are within valid frame range
                valid_transitions = []
                for start_frame, end_frame in transitions:
                    if 0 <= start_frame < num_frames and 0 < end_frame <= num_frames:
                        # Clamp to valid range
                        valid_start = max(0, start_frame)
                        valid_end = min(num_frames - 1, end_frame)
                        if valid_start <= valid_end:
                            valid_transitions.append((valid_start, valid_end))
                
                # Create sequence metadata only
                for start_idx in range(0, num_frames - self.sequence_length + 1, self.overlap):
                    end_idx = start_idx + self.sequence_length
                    
                    # Check if sequence overlaps with ANY transition range
                    # A sequence has a boundary if any transition overlaps with [start_idx, end_idx)
                    has_boundary = any(
                        trans_start < end_idx and trans_end >= start_idx
                        for trans_start, trans_end in valid_transitions
                    )
                    
                    # BALANCED SAMPLING with HARD NEGATIVE MINING:
                    # - Include ALL boundary sequences
                    # - Subsample non-boundary sequences that are "hard negatives" (near boundaries)
                    sample_data = {
                        'video_name_stem': video_name_stem,
                        'video_name': video_name,
                        'start_frame': start_idx,
                        'end_frame': end_idx,
                        'transitions': valid_transitions,  # Store transition ranges!
                        'feature_file': strategy_file,
                        'num_frames': num_frames
                    }
                    
                    if has_boundary:
                        samples.append(sample_data)
                    elif self.include_all_non_boundary:
                        # Validation/test: include ALL sequences for fair evaluation
                        samples.append(sample_data)
                    else:
                        # Training: BALANCED SAMPLING (less aggressive than hard negative mining)
                        # Previous hard negative mining (80%/30%) caused severe train/val mismatch
                        # because training saw mostly "hard" cases while validation saw all cases
                        #
                        # New approach: Include 50% of non-boundary sequences uniformly
                        # This maintains reasonable class balance without distribution shift
                        hash_key = f"{video_name_stem}_{start_idx}"
                        hash_val = hash(hash_key) % 100
                        
                        if hash_val < 50:  # 50% of non-boundary sequences
                            samples.append(sample_data)
            else:
                # Original behavior: load all features into memory
                try:
                    features = load_fused_features(self.features_dir, video_name_stem, self.fusion_strategy, self.quiet)
                    videos_with_features += 1
                except FileNotFoundError as e:
                    # if not self.quiet:
                    #     print(f"Warning: Could not load features for {video_name} (using stem: {video_name_stem}): {e}")
                    videos_skipped += 1
                    continue
                
                # Create CENTER-FRAME boundary labels for transitions
                # CRITICAL FIX: Label only CENTER frame(s), not all frames
                labels = np.zeros(len(features) - 1)
                for trans_start, trans_end in transitions:
                    # Label center frame of transition only
                    local_start = max(0, trans_start)
                    local_end = min(len(labels) - 1, trans_end)
                    if local_start <= local_end:
                        center = (local_start + local_end) // 2
                        center = min(max(0, center), len(labels) - 1)
                        labels[center] = 1.0
                        # For wide transitions, also mark neighbors
                        if local_end - local_start > 10:
                            if center > 0:
                                labels[center - 1] = 1.0
                            if center < len(labels) - 1:
                                labels[center + 1] = 1.0
                
                # Create sequences with full data
                for start_idx in range(0, len(features) - self.sequence_length + 1, self.overlap):
                    end_idx = start_idx + self.sequence_length
                    
                    sequence_features = features[start_idx:end_idx]
                    sequence_labels = labels[start_idx:end_idx-1]  # N-1 labels for N frames
                    
                    # Balance the dataset - include sequences with and without boundaries
                    has_boundary = np.any(sequence_labels == 1)
                    
                    # Store transitions for this sequence as well
                    sample_data = {
                        'features': sequence_features,  
                        'labels': sequence_labels,
                        'video_name': video_name,
                        'start_frame': start_idx,
                        'transitions': transitions  # Keep for reference
                    }
                    
                    if has_boundary:
                        samples.append(sample_data)
                    elif self.include_all_non_boundary:
                        # Validation/test: include ALL sequences for fair evaluation
                        samples.append(sample_data)
                    else:
                        # Training: subsample non-boundary using deterministic hash
                        # This ensures reproducibility without relying on random state
                        hash_key = f"{video_name}_{start_idx}"
                        hash_val = hash(hash_key) % 100
                        if hash_val < 50:  # 50% of non-boundary sequences (deterministic)
                            samples.append(sample_data)
            
            videos_processed += 1
        
        if not self.quiet:
            print(f"Dataset creation summary:")
            print(f"  Total videos in ground truth: {total_videos}")
            print(f"  Videos with features loaded: {videos_with_features}")
            print(f"  Videos skipped (no features): {videos_skipped}")
            print(f"  Total training samples created: {len(samples)}")
            if self.lazy_loading:
                print(f"  Memory mode: Lazy loading (features loaded on-demand)")
            else:
                print(f"  Memory mode: Pre-loaded (all features in memory)")
            
            # Calculate class distribution
            if len(samples) > 0:
                boundary_samples = 0
                total_boundary_frames = 0
                total_frames = 0
                
                if self.lazy_loading:
                    # For lazy loading, count boundary samples and transition frames
                    for s in samples:
                        # Count frames that are within any transition range
                        transitions = s.get('transitions', [])
                        seq_start = s['start_frame']
                        seq_end = s['end_frame']
                        
                        # Count how many frames in this sequence are transition frames
                        boundary_frames_in_seq = 0
                        for trans_start, trans_end in transitions:
                            # Overlap of transition range with sequence range
                            overlap_start = max(seq_start, trans_start)
                            overlap_end = min(seq_end - 1, trans_end)  # -1 because labels are N-1
                            if overlap_start <= overlap_end:
                                boundary_frames_in_seq += (overlap_end - overlap_start + 1)
                        
                        if boundary_frames_in_seq > 0:
                            boundary_samples += 1
                        total_boundary_frames += boundary_frames_in_seq
                        total_frames += (seq_end - seq_start - 1)  # -1 for frame pairs
                else:
                    # For pre-loaded, count samples and frames with boundary label = 1
                    for s in samples:
                        if np.any(s['labels'] == 1):
                            boundary_samples += 1
                        total_boundary_frames += np.sum(s['labels'] == 1)
                        total_frames += len(s['labels'])
                
                non_boundary_samples = len(samples) - boundary_samples
                boundary_sample_ratio = boundary_samples / len(samples) * 100
                boundary_frame_ratio = total_boundary_frames / max(1, total_frames) * 100
                
                print(f"\n  CLASS DISTRIBUTION:")
                print(f"    SEQUENCE LEVEL:")
                print(f"      Sequences with boundaries: {boundary_samples} ({boundary_sample_ratio:.1f}%)")
                print(f"      Sequences without boundaries: {non_boundary_samples} ({100-boundary_sample_ratio:.1f}%)")
                print(f"      Balance ratio: 1:{non_boundary_samples/max(1, boundary_samples):.2f}")
                print(f"    FRAME LEVEL (actual labels):")
                print(f"      Boundary frames: {total_boundary_frames} ({boundary_frame_ratio:.2f}%)")
                print(f"      Non-boundary frames: {total_frames - total_boundary_frames} ({100-boundary_frame_ratio:.2f}%)")
                print(f"      Balance ratio: 1:{(total_frames - total_boundary_frames)/max(1, total_boundary_frames):.1f}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # =====================================================================
        # HANDLE SYNTHETIC SAMPLES (mathematically generated boundaries)
        # =====================================================================
        if sample.get('synthetic', False):
            return self._get_synthetic_sample()
        
        if self.lazy_loading:
            # Load features on-demand from disk
            video_stem = sample['video_name_stem']
            start_frame = sample['start_frame']
            end_frame = sample['end_frame']
            feature_file = sample['feature_file']
            transitions = sample.get('transitions', [])  # List of (trans_start, trans_end) tuples
            
            # Load only the required sequence efficiently
            try:
                # Open file and seek to the specific position to load only what we need
                with open(feature_file, 'rb') as f:
                    # Read numpy header to get shape and dtype info
                    version = np.lib.format.read_magic(f)
                    shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
                    
                    # Calculate byte offset for the slice we need
                    if len(shape) == 2:
                        num_frames, feature_dim = shape
                        bytes_per_row = feature_dim * dtype.itemsize
                        offset_bytes = start_frame * bytes_per_row
                        num_rows_to_read = end_frame - start_frame
                        
                        # Seek to the start position
                        f.seek(offset_bytes, 1)  # Seek from current position (after header)
                        
                        # Read only the required bytes
                        bytes_to_read = num_rows_to_read * bytes_per_row
                        data_bytes = f.read(bytes_to_read)
                        
                        # Convert bytes to numpy array
                        sequence_features = np.frombuffer(data_bytes, dtype=dtype).reshape(num_rows_to_read, feature_dim)
                    else:
                        # Fallback: load entire file if unexpected shape
                        f.seek(0)
                        features_full = np.load(f)
                        sequence_features = features_full[start_frame:end_frame].copy()
                    
                # Validate shape
                if len(sequence_features.shape) != 2:
                    raise ValueError(f"Expected 2D features array, got shape {sequence_features.shape}")
                
                if sequence_features.shape[0] != self.sequence_length:
                    # Handle edge cases with padding if needed
                    if sequence_features.shape[0] < self.sequence_length:
                        padding = np.zeros((self.sequence_length - sequence_features.shape[0], sequence_features.shape[1]))
                        sequence_features = np.concatenate([sequence_features, padding], axis=0)
                    else:
                        sequence_features = sequence_features[:self.sequence_length]
                
                # PAD FEATURES TO MAXIMUM DIMENSION IF NEEDED
                current_dim = sequence_features.shape[1]
                if self._feature_dim is not None and current_dim < self._feature_dim:
                    padding = np.zeros((self.sequence_length, self._feature_dim - current_dim))
                    sequence_features = np.concatenate([sequence_features, padding], axis=1)
                elif self._feature_dim is None:
                    self._feature_dim = current_dim
                elif current_dim > self._feature_dim:
                    self._feature_dim = current_dim
                
                # FEATURE AUGMENTATION: Add small noise during training to improve generalization
                if not self.validation_mode and np.random.random() < 0.3:
                    feature_std = np.std(sequence_features) + 1e-8
                    noise = np.random.normal(0, 0.01 * feature_std, sequence_features.shape)
                    sequence_features = sequence_features + noise
                
                # =============================================================
                # CREATE BINARY LABELS WITH SINGLE CENTER PEAK PER TRANSITION
                # =============================================================
                # CRITICAL FIX: Use BINARY labels (0/1) instead of Gaussian.
                # Gaussian labels caused train/validation metric mismatch because:
                # 1. Training loss operates on soft labels (0.0-1.0)
                # 2. Metrics use threshold=0.5, so labels 0.5-1.0 are "positive"
                # 3. Gaussian labels create MANY positive frames per transition
                # 4. This inflates both GT count and prediction matches
                #
                # New approach: Label ONLY the center frame of each transition
                # For hard cuts: single frame gets 1.0
                # For gradual transitions: middle frame(s) get 1.0
                # This ensures 1 GT peak per transition = clean metrics
                
                labels = np.zeros(self.sequence_length - 1)
                duration_labels = np.zeros(self.sequence_length - 1)  # NEW: Duration labels for multi-task
                
                for trans_start, trans_end in transitions:
                    # Convert global frame indices to local sequence indices
                    local_start = max(0, trans_start - start_frame)
                    local_end = min(self.sequence_length - 2, trans_end - start_frame)
                    
                    if local_start <= local_end and local_start < self.sequence_length - 1:
                        # Calculate transition duration (in frames)
                        # Duration = end - start + 1 for original global indices
                        trans_duration = (trans_end - trans_start) + 1
                        
                        # For gradual transitions, label CENTER frame only
                        # This teaches model to output a single peak
                        trans_center = int((local_start + local_end) / 2.0)
                        trans_center = min(max(0, trans_center), self.sequence_length - 2)
                        
                        # Set binary label at center
                        labels[trans_center] = 1.0
                        
                        # =============================================================
                        # MULTI-TASK: Set duration label at center frame
                        # This teaches the model:
                        # - Cuts: duration ≈ 1 (small log value)
                        # - Gradual: duration 5-30+ (larger log value)
                        # =============================================================
                        duration_labels[trans_center] = trans_duration
                        
                        # For wide transitions (>10 frames), also mark ±1 neighbors
                        # to give model some slack for precise localization
                        trans_width = local_end - local_start
                        if trans_width > 10:
                            if trans_center > 0:
                                labels[trans_center - 1] = 1.0
                                duration_labels[trans_center - 1] = trans_duration
                            if trans_center < self.sequence_length - 2:
                                labels[trans_center + 1] = 1.0
                                duration_labels[trans_center + 1] = trans_duration
                
            except Exception as e:
                # Handle any errors during loading
                if not self.quiet:
                    print(f"Error loading features for {video_stem}: {e}")
                # Return zero features as fallback
                sequence_features = np.zeros((self.sequence_length, self._feature_dim or 512))
                labels = np.zeros(self.sequence_length - 1)
                duration_labels = np.zeros(self.sequence_length - 1)
                
        else:
            # ORIGINAL PRE-LOADED BEHAVIOR (kept for backward compatibility)
            # Create sequences with full data
            sequence_features = sample['features']
            labels = sample['labels']
            
            # For pre-loaded data, create duration labels from transitions
            duration_labels = np.zeros(len(labels))
            if 'transitions' in sample:
                for trans_start, trans_end in sample['transitions']:
                    trans_duration = (trans_end - trans_start) + 1
                    # Find where labels are positive and assign duration
                    local_start = max(0, trans_start - sample.get('start_frame', 0))
                    local_end = min(len(labels) - 1, trans_end - sample.get('start_frame', 0))
                    center = (local_start + local_end) // 2
                    if 0 <= center < len(duration_labels):
                        duration_labels[center] = trans_duration
            
            # Pad if needed
            if sequence_features.shape[0] < self.sequence_length:
                padding = np.zeros((self.sequence_length - sequence_features.shape[0], sequence_features.shape[1]))
                sequence_features = np.concatenate([sequence_features, padding], axis=0)
            
            # Apply augmentations for pre-loaded data (synthetic DISABLED)
            # Gaussian smoothing also DISABLED - it inflates training metrics
            # by creating multiple GT points per boundary (labels >= 0.5 count as GT)

        # Convert to tensors
        features_tensor = torch.from_numpy(sequence_features).float()
        labels_tensor = torch.from_numpy(labels).float()
        
        # MULTI-TASK: Return 3 tensors if enabled
        if self.use_multi_task:
            duration_tensor = torch.from_numpy(duration_labels).float()
            return features_tensor, labels_tensor, duration_tensor
        
        return features_tensor, labels_tensor
    
    def _get_synthetic_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a synthetic boundary sample using the SyntheticBoundaryGenerator.
        
        This creates mathematically realistic transitions from features of
        DIFFERENT videos, teaching the model what boundaries "look like".
        
        Returns:
            Tuple of (features_tensor, labels_tensor) or (features, labels, duration) if multi-task
        """
        if self._synthetic_generator is None:
            # Fallback to simple synthetic augmentation if generator not available
            # Create a random non-boundary sequence and augment it
            fallback_idx = np.random.randint(0, len(self.samples) - 1)
            while self.samples[fallback_idx].get('synthetic', False):
                fallback_idx = np.random.randint(0, len(self.samples) - 1)
            
            # Get the real sample and apply synthetic augmentation
            sample = self.samples[fallback_idx]
            if 'feature_file' in sample:
                # Load features
                try:
                    features = np.load(sample['feature_file'])[sample['start_frame']:sample['end_frame']]
                    features, labels, duration_labels = self._synthetic_augmentation(features)
                    labels = self._apply_gaussian_smoothing(labels)
                    
                    if self.use_multi_task:
                        return (torch.from_numpy(features).float(), 
                                torch.from_numpy(labels).float(),
                                torch.from_numpy(duration_labels).float())
                    return torch.from_numpy(features).float(), torch.from_numpy(labels).float()
                except:
                    pass
            
            # Ultimate fallback: zeros
            features = np.zeros((self.sequence_length, self._feature_dim or 256))
            labels = np.zeros(self.sequence_length - 1)
            duration_labels = np.zeros(self.sequence_length - 1)
            
            if self.use_multi_task:
                return (torch.from_numpy(features).float(), 
                        torch.from_numpy(labels).float(),
                        torch.from_numpy(duration_labels).float())
            return torch.from_numpy(features).float(), torch.from_numpy(labels).float()
        
        # Use the sophisticated generator for cross-video synthetic transitions
        features, labels, transition_type = self._synthetic_generator.generate_synthetic_sequence()
        
        if features is None or labels is None:
            # Fallback if generation failed
            features = np.zeros((self.sequence_length, self._feature_dim or 256))
            labels = np.zeros(self.sequence_length - 1)
            labels[self.sequence_length // 2] = 1.0  # Put boundary in middle
        
        # Ensure correct dimensions
        if features.shape[0] != self.sequence_length:
            if features.shape[0] < self.sequence_length:
                padding = np.zeros((self.sequence_length - features.shape[0], features.shape[1]))
                features = np.concatenate([features, padding], axis=0)
            else:
                features = features[:self.sequence_length]
        
        if len(labels) != self.sequence_length - 1:
            new_labels = np.zeros(self.sequence_length - 1)
            new_labels[:min(len(labels), len(new_labels))] = labels[:min(len(labels), len(new_labels))]
            labels = new_labels
        
        # Apply Gaussian smoothing to labels
        labels = self._apply_gaussian_smoothing(labels)
        
        # Create duration labels for synthetic samples
        # Map transition type to approximate duration
        duration_labels = np.zeros(self.sequence_length - 1)
        boundary_idx = np.where(labels > 0.5)[0]
        if len(boundary_idx) > 0:
            # Estimate duration based on transition type
            if transition_type == 'hard_cut':
                synthetic_duration = 1
            elif transition_type == 'dissolve':
                synthetic_duration = np.random.randint(5, 20)
            elif transition_type == 'fade':
                synthetic_duration = np.random.randint(10, 30)
            elif transition_type == 'wipe':
                synthetic_duration = np.random.randint(5, 15)
            else:
                synthetic_duration = np.random.randint(1, 10)
            
            for idx in boundary_idx:
                duration_labels[idx] = synthetic_duration
        
        if self.use_multi_task:
            return (torch.from_numpy(features).float(), 
                    torch.from_numpy(labels).float(),
                    torch.from_numpy(duration_labels).float())
        
        return torch.from_numpy(features).float(), torch.from_numpy(labels).float()

    def _synthetic_augmentation(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a synthetic transition by blending two parts of the sequence.
        This simulates dissolves and wipes mathematically in feature space.
        
        Ref: TransNet V2 / AutoShot augmentation strategy
        
        Returns:
            Tuple of (features, labels, duration_labels)
        """
        seq_len, dim = features.shape
        cut_point = np.random.randint(int(seq_len * 0.3), int(seq_len * 0.7))
        transition_len = np.random.randint(1, 15)  # 1-15 frame transition
        
        # Take two random segments from the sequence (treated as unrelated clips)
        # We scramble the second half to make it look "different"
        clip_a = features.copy()
        clip_b = np.roll(features, shift=int(seq_len/2), axis=0) 
        # Add some noise to make them distinct
        clip_b += np.random.normal(0, 0.1, clip_b.shape)
        
        # Create transition
        result = np.zeros_like(features)
        labels = np.zeros(seq_len - 1)
        duration_labels = np.zeros(seq_len - 1)
        
        # Linear blend (dissolve simulation)
        half_trans = transition_len // 2
        start_trans = max(0, cut_point - half_trans)
        end_trans = min(seq_len, cut_point + half_trans + 1)
        
        # Part A (before transition)
        result[:start_trans] = clip_a[:start_trans]
        
        # Transition zone
        for i in range(start_trans, end_trans):
            alpha = (i - start_trans) / max(1, end_trans - start_trans)
            result[i] = (1 - alpha) * clip_a[i] + alpha * clip_b[i]
            
        # Part B (after transition)
        result[end_trans:] = clip_b[end_trans:]
        
        # Set label at the center of transition
        if 0 <= cut_point < seq_len - 1:
            labels[cut_point] = 1
            # Duration is the transition length
            duration_labels[cut_point] = transition_len
            
        return result, labels, duration_labels

    def _apply_gaussian_smoothing(self, labels: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        DISABLED: No longer apply Gaussian smoothing.
        
        Previously this converted [0, 1, 0] -> [0.1, 0.8, 0.1]
        But this caused train/val metric mismatch because:
        1. Metrics use threshold 0.5 → labels 0.5-1.0 count as positive
        2. This created MULTIPLE positive frames per boundary
        3. Training saw inflated GT counts
        
        Now: Return labels unchanged (keep binary)
        """
        return labels  # Return unchanged - use binary labels



def validate_video_worker(video_file: Path, video_queue: queue.Queue, 
                          load_progress: tqdm = None) -> None:
    """
    Worker function to validate a video file and put metadata in the queue.
    Does NOT load frames into memory to prevent memory bloat with large videos.
    
    Args:
        video_file: Path to the video file
        video_queue: Queue to put video metadata (not frames)
        load_progress: Optional progress bar for validation
    """
    try:
        # Import here to avoid issues with multiprocessing
        import cv2
        
        # Force verify and set OpenCV environment variables for this worker process
        worker_env_vars = {
            'OPENCV_FFMPEG_READ_ATTEMPTS': '16384',
            'OPENCV_LOG_LEVEL': 'ERROR',
            'OPENCV_FFMPEG_DEMUXER_OPTIONS': 'analyzeduration;10000000;probesize;50000000',
            'OPENCV_FFMPEG_CAPTURE_OPTIONS': 'video_stream_id;0'
        }
        
        for key, value in worker_env_vars.items():
            os.environ[key] = value  # Force set (not just setdefault)
        
        # Quick validation first
        if not video_file.exists() or video_file.stat().st_size == 0:
            raise ValueError(f"Invalid video file: {video_file}")
        
        # Get file size for memory estimation
        file_size_mb = video_file.stat().st_size / (1024 * 1024)
        
        # Try multiple backends for better compatibility, prioritizing non-FFmpeg backends first
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG, cv2.CAP_ANY]
        cap = None
        video_info = None
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(str(video_file), backend)
                if not cap.isOpened():
                    if cap:
                        cap.release()
                    continue
                
                # Test read first frame to validate video is readable
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    cap.release()
                    continue
                
                # Get video properties for memory estimation
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Estimate memory usage: frames * height * width * channels * bytes_per_channel
                estimated_memory_mb = (frame_count * height * width * 3 * 1) / (1024 * 1024)
                
                video_info = {
                    'frame_count': frame_count,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'file_size_mb': file_size_mb,
                    'estimated_memory_mb': estimated_memory_mb,
                    'backend': backend
                }
                
                cap.release()
                break  # Successfully validated with this backend
                
            except Exception:
                if cap:
                    cap.release()
                continue
        
        if video_info is None:
            raise RuntimeError("Could not validate video with any backend")
        
        # Put video metadata in queue (NOT the actual frames)
        video_data = {
            'video_file': video_file,
            'video_stem': video_file.stem,
            'video_info': video_info,
            'validated': True
        }
        video_queue.put(video_data)
        
        if load_progress:
            load_progress.update(1)
            mem_info = f"{video_info['estimated_memory_mb']:.0f}MB"
            load_progress.set_postfix({"Validated": video_file.name, "Est.Mem": mem_info})
            
    except Exception as e:
        # Categorize error types for better handling
        error_type = "unknown"
        if "moov atom not found" in str(e).lower():
            error_type = "corrupted_mp4"
        elif "ffmpeg" in str(e).lower():
            error_type = "ffmpeg_error"
        elif "assertion failed" in str(e).lower():
            error_type = "opencv_assertion"
        elif "file" in str(e).lower() and ("not found" in str(e).lower() or "invalid" in str(e).lower()):
            error_type = "file_error"
        
        # Put error information in queue
        error_data = {
            'video_file': video_file,
            'error': str(e),
            'error_type': error_type,
            'video_stem': video_file.stem
        }
        video_queue.put(error_data)
        
        if load_progress:
            load_progress.update(1)
            load_progress.set_postfix({"Error": f"{error_type}: {video_file.name[:15]}"})


# Legacy alias for backward compatibility
load_video_worker = validate_video_worker


def check_video_health(video_dir: str, sample_size: int = 10, quiet: bool = False) -> Dict[str, Any]:
    """
    Check the health of videos in a directory to identify potential issues.
    
    Args:
        video_dir: Directory containing videos
        sample_size: Number of videos to sample for health check (0 = all)
        quiet: Whether to suppress output
        
    Returns:
        Dictionary with health check results
    """
    from utils.video_utils import VideoProcessor
    from utils.video_repair import VideoRepairer
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    if not video_files:
        return {'error': 'No video files found'}
    
    # Sample videos if requested
    if sample_size > 0 and len(video_files) > sample_size:
        import random
        video_files = random.sample(video_files, sample_size)
    
    # Create a simple config object for video processor
    from core.config import ProcessingConfig
    processing_config = ProcessingConfig()
    processing_config.target_size = (224, 224)
    processing_config.frame_rate = 30
    
    processor = VideoProcessor(processing_config)
    repairer = VideoRepairer()
    
    results = {
        'total_checked': len(video_files),
        'healthy': 0,
        'corrupted': 0,
        'unreadable': 0,
        'issues': [],
        'repair_available': repairer.can_repair()
    }
    
    if not quiet:
        print(f"Checking health of {len(video_files)} videos...")
    
    for video_file in video_files:
        try:
            validation = processor.validate_video(str(video_file))
            
            if validation['valid']:
                results['healthy'] += 1
            else:
                error = validation.get('error', 'unknown')
                if 'moov atom' in error.lower():
                    results['corrupted'] += 1
                    results['issues'].append({
                        'file': str(video_file),
                        'type': 'corrupted_mp4',
                        'error': error
                    })
                else:
                    results['unreadable'] += 1
                    results['issues'].append({
                        'file': str(video_file),
                        'type': 'unreadable',
                        'error': error
                    })
        
        except Exception as e:
            results['unreadable'] += 1
            results['issues'].append({
                'file': str(video_file),
                'type': 'exception',
                'error': str(e)
            })
    
    if not quiet:
        print(f"Health check results:")
        print(f"  ✓ Healthy: {results['healthy']}")
        print(f"  ⚠️  Corrupted: {results['corrupted']}")
        print(f"  ✗ Unreadable: {results['unreadable']}")
        
        if results['issues'] and not quiet:
            print(f"\\nIssues found:")
            for issue in results['issues'][:5]:  # Show first 5 issues
                file_name = Path(issue['file']).name
                print(f"  {issue['type']}: {file_name} - {issue['error']}")
            
            if len(results['issues']) > 5:
                print(f"  ... and {len(results['issues']) - 5} more issues")
        
        if results['repair_available'] and results['corrupted'] > 0:
            print(f"\\n💡 Tip: Use --repair_videos to attempt automatic repair of corrupted files")
    
    return results


def extract_training_features_parallel(config: VideoShotConfig, video_dir: str, 
                                     features_dir: str, max_workers: int = 2,
                                     parallel_features: bool = True,
                                     show_progress: bool = True, 
                                     quiet: bool = False, skip_videos: int = 0) -> None:
    """
    Extract features for all videos in the training directory using parallel video loading.
    
    Args:
        config: System configuration
        video_dir: Directory containing training videos
        features_dir: Directory to save extracted features
        max_workers: Maximum number of parallel video loading workers
        parallel_features: Whether to enable parallel feature extraction within each video
        show_progress: Whether to show progress bars
        quiet: Whether to suppress non-essential output
        skip_videos: Number of videos to skip from the beginning (useful for resuming)
    """
    os.makedirs(features_dir, exist_ok=True)
    
    # Configure parallel feature extraction
    if parallel_features:
        config.processing.parallel_feature_extraction = True
        config.processing.feature_extraction_workers = min(4, max(2, max_workers))
    
    # Initialize coordinator with logging configuration (single instance)
    coordinator = ProcessingCoordinator(config)
    coordinator.initialize()
    
    # Get all video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    if not video_files:
        if not quiet:
            print(f"No video files found in {video_dir}")
        return

    # Apply skip_videos if specified
    total_videos_found = len(video_files)
    if skip_videos > 0:
        if skip_videos >= total_videos_found:
            if not quiet:
                print(f"Warning: skip_videos ({skip_videos}) >= total videos found ({total_videos_found}). Nothing to process.")
            return
        video_files = video_files[skip_videos:]
        if not quiet:
            print(f"Skipping first {skip_videos} videos. Processing {len(video_files)} out of {total_videos_found} total videos.")

    total_videos = len(video_files)
    if not quiet:
        print(f"Processing {total_videos} videos with {max_workers} parallel loaders...")    # Create queues for loaded videos
    video_queue = queue.Queue(maxsize=max_workers)  # Buffer for loaded videos ##  * 2
    
    # Setup progress bars
    load_progress = None
    extract_progress = None
    
    if show_progress:
        load_progress = tqdm(total=total_videos, desc="Validating videos", position=0)
        extract_progress = tqdm(total=total_videos, desc="Extracting features", position=1)
    
    videos_processed = 0
    videos_loaded = 0
    
    # Start video validation in background threads (validates without loading frames)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video validation tasks
        loading_futures = []
        for video_file in video_files:
            future = executor.submit(validate_video_worker, video_file, video_queue, load_progress)
            loading_futures.append(future)
        
        # Process loaded videos as they become available
        while videos_processed < total_videos:
            try:
                # Wait for a video to be loaded (with timeout to check completion)
                video_data = video_queue.get(timeout=1.0)
                videos_loaded += 1
                
                # Check if this is an error
                if 'error' in video_data:
                    error_msg = f"Error loading {video_data['video_file']}: {video_data['error']}"
                    error_type = video_data.get('error_type', 'unknown')
                    
                    # Log different error types with appropriate detail
                    if error_type == "corrupted_mp4":
                        if show_progress:
                            tqdm.write(f"⚠️  Corrupted MP4 (moov atom): {video_data['video_file'].name}")
                        elif not quiet:
                            print(f"⚠️  Corrupted MP4 detected: {video_data['video_file'].name}")
                    elif error_type == "ffmpeg_error":
                        if show_progress:
                            tqdm.write(f"⚠️  FFMPEG error: {video_data['video_file'].name}")
                        elif not quiet:
                            print(f"⚠️  FFMPEG read error: {video_data['video_file'].name}")
                    elif error_type == "opencv_assertion":
                        if show_progress:
                            tqdm.write(f"⚠️  OpenCV assertion failed: {video_data['video_file'].name}")
                        elif not quiet:
                            print(f"⚠️  OpenCV processing error: {video_data['video_file'].name}")
                    else:
                        if show_progress:
                            tqdm.write(error_msg)
                        elif not quiet:
                            print(error_msg)
                    
                    videos_processed += 1
                    if extract_progress:
                        extract_progress.update(1)
                        extract_progress.set_postfix({"Status": f"Skipped ({error_type})"})
                    continue
                
                # Extract features from validated video (load frames on-demand)
                video_file = video_data['video_file']
                video_stem = video_data['video_stem']
                video_info = video_data.get('video_info', {})
                
                # Check if all features already exist and are valid
                feature_check = check_features_exist_and_valid(
                    features_dir, video_stem, 
                    feature_names=['hsv_histogram', 'cnn', 'clip', 'optical_flow'],
                    # fusion_strategy=config.fusion.strategy
                )
                
                if not feature_check['needs_extraction']:
                    if extract_progress:
                        extract_progress.set_postfix({"Status": "Skipped (all features exist)"})
                    elif not quiet and not show_progress:
                        print(f"  ✓ All features already exist and are valid for {video_file.name}")
                    
                    videos_processed += 1
                    if extract_progress:
                        extract_progress.update(1)
                    continue
                
                # If some features are missing or corrupted, show what needs to be extracted
                if feature_check['missing_features']:
                    missing_msg = f"Missing: {', '.join(feature_check['missing_features'])}"
                    if extract_progress:
                        extract_progress.set_postfix({"Status": missing_msg})
                    elif not quiet and not show_progress:
                        print(f"  → {missing_msg}")
                
                # Show estimated memory for this video
                est_mem = video_info.get('estimated_memory_mb', 0)
                if extract_progress:
                    extract_progress.set_postfix({"Current": video_file.name, "Est.Mem": f"{est_mem:.0f}MB"})
                elif not quiet and not show_progress:
                    print(f"Extracting features for: {video_file.name} (Est. memory: {est_mem:.0f}MB)")
                
                try:
                    # Load frames on-demand (memory-efficient: only one video in memory at a time)
                    frames = coordinator.video_processor.load_video(str(video_file))
                    
                    if frames is None or len(frames) == 0:
                        raise RuntimeError(f"Could not load frames from {video_file}")
                    
                    # Extract features using the shared coordinator
                    features = coordinator._extract_all_features(frames, parallel_features, max_workers)
                    
                    # Explicitly free frames memory after extraction
                    del frames
                    gc.collect()

                    # Save individual features
                    for feature_name, feature_array in features.items():
                        individual_file = os.path.join(features_dir, f"{video_stem}_{feature_name}.npy")
                        np.save(individual_file, feature_array)
                        if not quiet and not show_progress:
                            print(f"Saved {feature_name} features: {feature_array.shape}")
                    
                    # Explicitly free features memory after saving
                    del features
                    gc.collect()
                    
                    # Fuse features
                    # fused_features = coordinator.fusion_strategy.fuse_features(features)
                    
                    # # Save fused features with strategy name
                    # strategy_name = config.fusion.strategy
                    # strategy_file = os.path.join(features_dir, f"{video_stem}_{strategy_name}.npy")
                    # np.save(strategy_file, fused_features)
                    # if not quiet and not show_progress:
                    #     print(f"Saved {strategy_name} fused features: {fused_features.shape}")
                    
                    # # Also save as generic fused features for backward compatibility
                    # fused_file = os.path.join(features_dir, f"{video_stem}_fused.npy")
                    # np.save(fused_file, fused_features)
                    # if not quiet and not show_progress:
                    #     print(f"Saved generic fused features: {fused_features.shape}")
                        
                    if extract_progress:
                        extract_progress.set_postfix({"Status": "Completed"})
                
                except Exception as e:
                    error_msg = f"Error extracting features from {video_file}: {e}"
                    if show_progress:
                        tqdm.write(error_msg)
                    elif not quiet:
                        print(error_msg)
                    if extract_progress:
                        extract_progress.set_postfix({"Status": "Error"})
                
                videos_processed += 1
                if extract_progress:
                    extract_progress.update(1)
                
            except queue.Empty:
                # Check if all loading is complete
                if all(future.done() for future in loading_futures):
                    # All videos should be loaded, but queue is empty
                    # This shouldn't happen, but break to avoid infinite loop
                    if videos_processed < total_videos:
                        remaining = total_videos - videos_processed
                        if not quiet:
                            print(f"Warning: {remaining} videos not processed (queue empty)")
                    break
                continue
    
    # Close progress bars
    if load_progress:
        load_progress.close()
    if extract_progress:
        extract_progress.close()
    
    if not quiet:
        print(f"Feature extraction completed! Processed {videos_processed}/{total_videos} videos.")


def extract_training_features(config: VideoShotConfig, video_dir: str, 
                            features_dir: str, show_progress: bool = True, 
                            quiet: bool = False, max_workers: int = 1,
                            parallel_features: bool = False, skip_videos: int = 0) -> None:
    """
    Extract features for all videos in the training directory.
    
    Args:
        config: System configuration
        video_dir: Directory containing training videos
        features_dir: Directory to save extracted features
        show_progress: Whether to show progress bars
        quiet: Whether to suppress non-essential output
        max_workers: Number of parallel workers for video loading (1 = sequential)
        parallel_features: Whether to enable parallel feature extraction within each video
        skip_videos: Number of videos to skip from the beginning (useful for resuming)
    """
    # Use parallel version if max_workers > 1
    if max_workers > 1:
        return extract_training_features_parallel(
            config, video_dir, features_dir, max_workers, parallel_features, show_progress, quiet, skip_videos
        )
    
    # Configure parallel feature extraction for sequential video processing
    if parallel_features:
        config.processing.parallel_feature_extraction = True
        config.processing.feature_extraction_workers = min(4, max(2, max_workers))
    
    # Original sequential implementation
    os.makedirs(features_dir, exist_ok=True)
    
    # Initialize coordinator with logging configuration
    coordinator = ProcessingCoordinator(config)
    coordinator.initialize()
    
    # Process all videos
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    if not video_files:
        if not quiet:
            print(f"No video files found in {video_dir}")
        return
    
    # Apply skip_videos if specified
    total_videos_found = len(video_files)
    if skip_videos > 0:
        if skip_videos >= total_videos_found:
            if not quiet:
                print(f"Warning: skip_videos ({skip_videos}) >= total videos found ({total_videos_found}). Nothing to process.")
            return
        video_files = video_files[skip_videos:]
        if not quiet:
            print(f"Skipping first {skip_videos} videos. Processing {len(video_files)} out of {total_videos_found} total videos.")
    
    # Setup progress bar if enabled
    video_iterator = tqdm(video_files, desc="Extracting features", disable=not show_progress) if show_progress else video_files
    
    for video_file in video_iterator:
        if show_progress:
            video_iterator.set_postfix({"Current": video_file.name})
        elif not quiet:
            print(f"Processing: {video_file.name}")
        
        video_stem = video_file.stem
        
        # Check if all features already exist and are valid
        feature_check = check_features_exist_and_valid(
            features_dir, video_stem, 
            feature_names=['hsv_histogram', 'cnn', 'clip', 'optical_flow'],
            # fusion_strategy=config.fusion.strategy
        )
        
        if not feature_check['needs_extraction']:
            if show_progress:
                video_iterator.set_postfix({"Status": "Skipped (all features exist)"})
            elif not quiet:
                print(f"  ✓ All features already exist and are valid for {video_file.name}")
            continue
        
        # If some features are missing or corrupted, show what needs to be extracted
        if feature_check['missing_features']:
            missing_msg = f"Missing/corrupted: {', '.join(feature_check['missing_features'])}"
            if show_progress:
                video_iterator.set_postfix({"Status": missing_msg})
            elif not quiet:
                print(f"  → {missing_msg}")
        
        try:
            # Load video with improved error handling
            frames = coordinator.video_processor.load_video(
                str(video_file), 
                max_retries=3, 
                skip_corrupted=True
            )
            
            # Extract features
            features = coordinator._extract_all_features(frames, parallel_features, max_workers)

            # Save individual features
            for feature_name, feature_array in features.items():
                individual_file = os.path.join(features_dir, f"{video_stem}_{feature_name}.npy")
                np.save(individual_file, feature_array)
                if not quiet and not show_progress:
                    print(f"Saved {feature_name} features: {feature_array.shape}")
            
            # Fuse features
            # fused_features = coordinator.fusion_strategy.fuse_features(features)
            
            # Save fused features with strategy name
            # strategy_name = config.fusion.strategy
            # strategy_file = os.path.join(features_dir, f"{video_stem}_{strategy_name}.npy")
            # np.save(strategy_file, fused_features)
            # if not quiet and not show_progress:
            #     print(f"Saved {strategy_name} fused features: {fused_features.shape}")
            
            # Also save as generic fused features for backward compatibility
            # fused_file = os.path.join(features_dir, f"{video_stem}_fused.npy")
            # np.save(fused_file, fused_features)
            # if not quiet and not show_progress:
            #     print(f"Saved generic fused features: {fused_features.shape}")
            
        except Exception as e:
            error_msg = f"Error processing {video_file}: {e}"
            if show_progress:
                tqdm.write(error_msg)
            elif not quiet:
                print(error_msg)


def train_transformer_model(config: VideoShotConfig, features_dir: str, 
                          ground_truth_file: str, model_save_path: str, 
                          fusion_strategy: str = None, show_progress: bool = True,
                          quiet: bool = False, save_plots: bool = True,
                          random_seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, float]:
    """
    Train the transformer detection model for a specific fusion strategy.
    
    PUBLICATION-READY:
    - Video-level train/val split (no data leakage)
    - Fixed random seeds for reproducibility
    - Deterministic sampling
    
    Args:
        config: System configuration
        features_dir: Directory containing extracted features
        ground_truth_file: JSON file with ground truth boundaries
        model_save_path: Path to save the trained model
        fusion_strategy: Specific fusion strategy to use (overrides config)
        show_progress: Whether to show progress bars
        quiet: Whether to suppress non-essential output
        save_plots: Whether to save training/validation plots
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Dictionary with training results (best F1-score, final metrics)
    """
    # ==========================================================================
    # STEP 0: Set random seeds for reproducibility (CRITICAL FOR PUBLICATION)
    # ==========================================================================
    set_random_seeds(random_seed)
    
    # Track training start time
    training_start_time = time.time()
    
    # Ensure model path has .pth extension
    if not model_save_path.endswith('.pth'):
        model_save_path = model_save_path + '.pth'
        if not quiet:
            print(f"Added .pth extension to model path: {model_save_path}")
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Use specific fusion strategy if provided
    if fusion_strategy:
        config.fusion.strategy = fusion_strategy
    
    if not quiet:
        print(f"Starting transformer model training with {config.fusion.strategy} fusion...")
        print_memory_usage("Before dataset creation: ")
    
    # ==========================================================================
    # STEP 1: VIDEO-LEVEL SPLIT (CRITICAL - Prevents data leakage!)
    # ==========================================================================
    # Load ground truth to get all video names
    with open(ground_truth_file, 'r') as f:
        full_ground_truth = json.load(f)
    
    all_video_names = list(full_ground_truth.keys())
    
    # Shuffle videos with fixed seed for reproducibility
    import random
    random.shuffle(all_video_names)  # Uses the seed we set earlier
    
    # Split at VIDEO level: 80% train, 20% validation
    train_split_idx = int(0.8 * len(all_video_names))
    train_video_names = all_video_names[:train_split_idx]
    val_video_names = all_video_names[train_split_idx:]
    
    if not quiet:
        print(f"\n  VIDEO-LEVEL SPLIT (prevents data leakage):")
        print(f"    Total videos: {len(all_video_names)}")
        print(f"    Training videos: {len(train_video_names)} (80%)")
        print(f"    Validation videos: {len(val_video_names)} (20%)")
        print(f"    Random seed: {random_seed}")
    
    # ==========================================================================
    # STEP 2: Create SEPARATE datasets for train and validation
    # ==========================================================================
    # IMPORTANT: Include all data! The ComboLossV2 handles class imbalance.
    # SYNTHETIC DATA: Add mathematically generated boundaries to help balance classes
    # and teach the model what transitions "look like" in feature space.
    
    # Determine if multi-task learning is enabled based on loss_type
    loss_type = getattr(config.detection, "loss_type", "combo_v2")
    use_multi_task = (loss_type == "multi_task")
    
    # Training dataset: ALL sequences with OVERLAPPING windows for augmentation
    # The overlapping windows help the model see boundaries from different contexts
    train_dataset = BoundaryDataset(
        features_dir=features_dir,
        ground_truth_file=full_ground_truth,  # Pass dict directly
        sequence_length=config.processing.sequence_length,
        feature_type=config.fusion.strategy,  # CRITICAL: Pass feature_type explicitly!
        quiet=quiet,
        lazy_loading=True,
        video_names=train_video_names,  # Only training videos!
        include_all_non_boundary=False,  # Use hard negative mining for training
        synthetic_ratio=0.0,  # DISABLED: Synthetic data caused train/val distribution mismatch
        validation_mode=False,  # Training: use overlapping windows
        use_multi_task=use_multi_task  # Enable duration labels for multi-task learning
    )
    
    # Validation dataset: NON-OVERLAPPING windows for correct metrics
    # CRITICAL: Without validation_mode=True, overlapping windows cause the same boundary
    # to be counted multiple times, inflating GT count and making metrics inconsistent.
    # Example: boundary at frame 75 appears in windows [0:100] AND [50:150]
    # → counted TWICE in metrics → inflated recall denominator → lower apparent recall
    val_dataset = BoundaryDataset(
        features_dir=features_dir,
        ground_truth_file=full_ground_truth,  # Pass dict directly
        sequence_length=config.processing.sequence_length,
        feature_type=config.fusion.strategy,  # CRITICAL: Pass feature_type explicitly!
        quiet=quiet,
        lazy_loading=True,
        video_names=val_video_names,  # Only validation videos!
        include_all_non_boundary=True,  # Include ALL sequences for fair evaluation
        synthetic_ratio=0.0,  # NO synthetic for validation - evaluate on real data only
        validation_mode=True,  # CRITICAL: Non-overlapping windows for correct metrics!
        use_multi_task=use_multi_task  # Same multi-task mode as training
    )
    
    # Force garbage collection after dataset creation
    gc.collect()
    
    if not quiet:
        print(f"\n  DATASET SIZES:")
        print(f"    Training samples: {len(train_dataset)}")
        print(f"    Validation samples: {len(val_dataset)}")
        print(f"  Memory optimization: Lazy loading enabled")
        print_memory_usage("After dataset creation: ")
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        error_msg = f"ERROR: Training dataset is empty! No training samples were created."
        print(error_msg)
        print(f"  - Features directory: {features_dir}")
        print(f"  - Ground truth file: {ground_truth_file}")
        print(f"  - Fusion strategy: {config.fusion.strategy}")
        print(f"  - Sequence length: {config.processing.sequence_length}")
        raise RuntimeError(error_msg)
    
    # Create data loaders with reduced workers to save memory
    # Each worker duplicates the dataset, so fewer workers = less memory
    total_samples = len(train_dataset) + len(val_dataset)
    num_workers = 0 if total_samples > 50000 else 2  # Use 0 workers for large datasets
    
    if not quiet:
        print(f"  DataLoader workers: {num_workers} (memory-optimized)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.detection.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory to save RAM
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.detection.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Determine input dimension based on feature type (individual or fused)
    # Individual feature dimensions:
    individual_feature_dims = {
        "hsv_histogram": 768,     # HSV histogram: 256 bins * 3 channels = 768
        "cnn": 2048,              # ResNet50 features
        "clip": 512,              # CLIP ViT-B/32 features
        "optical_flow": 40,       # Optical flow histogram bins
    }
    
    # Fusion strategy output dimensions:
    # Each fusion strategy outputs different dimensions based on how they combine features
    # Actual feature dimensions: CNN=2048, HSV=768, CLIP=512, Flow=40
    fusion_input_dims = {
        "concatenation": 3368,    # HSV (768) + CNN (2048) + CLIP (512) + Flow (40) = 3368
        "weighted": 3368,         # Same as concatenation before weighted combination
        "pca": 256,              # PCA reduces to 256 components (configured at line 224)
        "cross_transformer": 512, # Transformer fusion outputs 512
        "hierarchical": 512       # Hierarchical fusion progressively reduces to 512 (verified)
    }
    
    # Check if it's an individual feature type first
    feature_type = config.fusion.strategy
    if feature_type in individual_feature_dims:
        input_dim = individual_feature_dims[feature_type]
        if not quiet:
            print(f"Using input_dim={input_dim} for individual feature: {feature_type}")
    else:
        input_dim = fusion_input_dims.get(feature_type, 512)
        if not quiet:
            print(f"Using input_dim={input_dim} for fusion strategy: {feature_type}")
    
    # Initialize model with correct input dimension
    model_config = config.detection.__dict__.copy()
    model_config['input_dim'] = input_dim
    
    detection_method = TransformerDetection(model_config)
    detection_method.initialize()
    
    # Initialize trainer
    trainer = TransformerTrainer(detection_method.model, model_config)
    
    # Training loop with progress bar
    best_f1_score = 0.0
    patience_counter = 0
    final_metrics = {}
    model_saved = False  # Track if we've saved at least one model
    
    # Lists to track training history for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    epochs_list = []
    
    # Setup progress bar for epochs if enabled
    epoch_iterator = tqdm(range(config.detection.num_epochs), desc=f"Training ({config.fusion.strategy})", disable=not show_progress) if show_progress else range(config.detection.num_epochs)
    
    for epoch in epoch_iterator:
        # Train (pass current epoch for warmup scheduling)
        train_loss, train_acc = trainer.train_epoch(train_loader, current_epoch=epoch)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall = trainer.validate(val_loader)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)
        
        # Store metrics for plotting
        epochs_list.append(epoch + 1)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)
        
        # Step scheduler based on validation F1 (for ReduceLROnPlateau)
        trainer.step_scheduler_on_validation(val_f1)
        
        # Update progress bar or print results
        if show_progress:
            epoch_iterator.set_postfix({
                'Train_Loss': f'{train_loss:.4f}',
                'Val_F1': f'{val_f1:.4f}',
                'Best_F1': f'{best_f1_score:.4f}'
            })
        elif not quiet:
            print(f"[{config.fusion.strategy}] Epoch {epoch+1}/{config.detection.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            detection_method.save_model(model_save_path, trainer.optimizer, epoch, val_loss)
            model_saved = True
            patience_counter = 0
            if not quiet and not show_progress:
                print(f"New best model saved with F1-score: {best_f1_score:.4f}")
            
            # Update final metrics
            final_metrics = {
                'best_f1': best_f1_score,
                'final_precision': val_precision,
                'final_recall': val_recall,
                'final_accuracy': val_acc,
                'final_loss': val_loss,
                'epoch': epoch + 1
            }
        else:
            patience_counter += 1
        
        # Save model on first epoch if nothing saved yet (fallback)
        if epoch == 0 and not model_saved:
            best_f1_score = val_f1
            detection_method.save_model(model_save_path, trainer.optimizer, epoch, val_loss)
            model_saved = True
            print(f"Saved initial model checkpoint with F1-score: {val_f1:.4f}")
            
            final_metrics = {
                'best_f1': best_f1_score,
                'final_precision': val_precision,
                'final_recall': val_recall,
                'final_accuracy': val_acc,
                'final_loss': val_loss,
                'epoch': epoch + 1
            }
        
        # Early stopping
        if patience_counter >= config.detection.early_stopping_patience:
            if show_progress:
                tqdm.write(f"Early stopping after {epoch+1} epochs")
            elif not quiet:
                print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Final save if nothing was saved during training (shouldn't happen, but just in case)
    if not model_saved:
        print(f"WARNING: No model was saved during training. Saving final model...")
        detection_method.save_model(model_save_path, trainer.optimizer, config.detection.num_epochs - 1, val_loss)
        final_metrics = {
            'best_f1': 0.0,
            'final_precision': 0.0,
            'final_recall': 0.0,
            'final_accuracy': 0.0,
            'final_loss': val_loss,
            'epoch': config.detection.num_epochs
        }
    
    # Save training plots if requested
    if save_plots and len(epochs_list) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create plots directory
            plots_dir = os.path.join(os.path.dirname(model_save_path), 'training_plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            strategy_name = config.fusion.strategy
            
            # Plot 1: Training and Validation Loss
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_list, train_losses, label='Training Loss', marker='o', linewidth=2)
            plt.plot(epochs_list, val_losses, label='Validation Loss', marker='s', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Training and Validation Loss - {strategy_name}', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            loss_plot_path = os.path.join(plots_dir, f'{strategy_name}_loss.png')
            plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Training and Validation Accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_list, train_accuracies, label='Training Accuracy', marker='o', linewidth=2)
            plt.plot(epochs_list, val_accuracies, label='Validation Accuracy', marker='s', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Training and Validation Accuracy - {strategy_name}', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            acc_plot_path = os.path.join(plots_dir, f'{strategy_name}_accuracy.png')
            plt.savefig(acc_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 3: Validation Metrics (Precision, Recall, F1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_list, val_precisions, label='Precision', marker='o', linewidth=2)
            plt.plot(epochs_list, val_recalls, label='Recall', marker='s', linewidth=2)
            plt.plot(epochs_list, val_f1_scores, label='F1-Score', marker='^', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title(f'Validation Metrics - {strategy_name}', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
            plt.tight_layout()
            metrics_plot_path = os.path.join(plots_dir, f'{strategy_name}_metrics.png')
            plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 4: Combined Overview
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Loss
            axes[0, 0].plot(epochs_list, train_losses, label='Train', marker='o', linewidth=2)
            axes[0, 0].plot(epochs_list, val_losses, label='Val', marker='s', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy
            axes[0, 1].plot(epochs_list, train_accuracies, label='Train', marker='o', linewidth=2)
            axes[0, 1].plot(epochs_list, val_accuracies, label='Val', marker='s', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Validation Metrics
            axes[1, 0].plot(epochs_list, val_precisions, label='Precision', marker='o', linewidth=2)
            axes[1, 0].plot(epochs_list, val_recalls, label='Recall', marker='s', linewidth=2)
            axes[1, 0].plot(epochs_list, val_f1_scores, label='F1', marker='^', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Validation Metrics')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(0, 1.0)
            
            # Best metrics summary (text)
            axes[1, 1].axis('off')
            summary_text = f"""
Best Model Summary
{'='*30}
Strategy: {strategy_name}
Best F1-Score: {best_f1_score:.4f}
Epoch: {final_metrics.get('epoch', 0)}

Final Validation Metrics:
  Precision: {final_metrics.get('final_precision', 0):.4f}
  Recall: {final_metrics.get('final_recall', 0):.4f}
  Accuracy: {final_metrics.get('final_accuracy', 0):.4f}
  Loss: {final_metrics.get('final_loss', 0):.4f}
            """
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                          verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            fig.suptitle(f'Training Overview - {strategy_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            overview_plot_path = os.path.join(plots_dir, f'{strategy_name}_overview.png')
            plt.savefig(overview_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if not quiet:
                print(f"Training plots saved to: {plots_dir}")
                print(f"  - Loss plot: {os.path.basename(loss_plot_path)}")
                print(f"  - Accuracy plot: {os.path.basename(acc_plot_path)}")
                print(f"  - Metrics plot: {os.path.basename(metrics_plot_path)}")
                print(f"  - Overview plot: {os.path.basename(overview_plot_path)}")
                
        except Exception as e:
            if not quiet:
                print(f"Warning: Failed to save training plots: {e}")
    
    # Store training history in final metrics
    final_metrics['training_history'] = {
        'epochs': epochs_list,
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'val_precision': val_precisions,
        'val_recall': val_recalls,
        'val_f1_score': val_f1_scores
    }
    
    # Calculate training duration
    training_end_time = time.time()
    training_duration_seconds = training_end_time - training_start_time
    training_duration_minutes = training_duration_seconds / 60
    training_duration_hours = training_duration_seconds / 3600
    
    # Add timing information to final_metrics
    final_metrics['training_time_seconds'] = training_duration_seconds
    final_metrics['training_time_minutes'] = training_duration_minutes
    final_metrics['training_time_hours'] = training_duration_hours
    final_metrics['training_time_formatted'] = f"{int(training_duration_hours)}h {int(training_duration_minutes % 60)}m {int(training_duration_seconds % 60)}s"
    
    if not quiet:
        print(f"Training completed for {config.fusion.strategy}. Best F1-score: {best_f1_score:.4f}")
        print(f"Training time: {final_metrics['training_time_formatted']}")
        print(f"Model saved to: {model_save_path}")
    return final_metrics


def create_strategy_comparison_plots(results: Dict[str, Dict[str, Any]], 
                                    model_base_path: str, quiet: bool = False) -> None:
    """
    Create comparison plots for all trained strategies.
    
    Args:
        results: Dictionary mapping fusion strategies to their training results
        model_base_path: Base path for saving plots
        quiet: Whether to suppress output messages
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if len(successful_results) < 1:
            return
        
        # Create plots directory
        plots_dir = os.path.join(os.path.dirname(model_base_path), 'training_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract data for comparison
        strategies = list(successful_results.keys())
        best_f1_scores = [successful_results[s]['best_f1'] for s in strategies]
        precisions = [successful_results[s]['final_precision'] for s in strategies]
        recalls = [successful_results[s]['final_recall'] for s in strategies]
        accuracies = [successful_results[s]['final_accuracy'] for s in strategies]
        
        # Plot 1: Bar chart comparing all metrics
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(strategies))
        width = 0.2
        
        ax.bar(x - 1.5*width, best_f1_scores, width, label='F1-Score', alpha=0.8)
        ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        
        ax.set_xlabel('Fusion Strategy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comparison of All Fusion Strategies - Validation Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for i, (f1, prec, rec, acc) in enumerate(zip(best_f1_scores, precisions, recalls, accuracies)):
            ax.text(i - 1.5*width, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i - 0.5*width, prec + 0.02, f'{prec:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + 0.5*width, rec + 0.02, f'{rec:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + 1.5*width, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        comparison_path = os.path.join(plots_dir, 'all_strategies_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: F1-Score ranking
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_indices = np.argsort(best_f1_scores)
        sorted_strategies = [strategies[i] for i in sorted_indices]
        sorted_f1 = [best_f1_scores[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_strategies)))
        bars = ax.barh(sorted_strategies, sorted_f1, color=colors, alpha=0.8)
        
        ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fusion Strategy', fontsize=12, fontweight='bold')
        ax.set_title('Fusion Strategies Ranked by F1-Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1.0)
        
        # Add value labels
        for i, (strategy, f1) in enumerate(zip(sorted_strategies, sorted_f1)):
            ax.text(f1 + 0.01, i, f'{f1:.4f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        ranking_path = os.path.join(plots_dir, 'strategies_f1_ranking.png')
        plt.savefig(ranking_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Training convergence comparison (if training history available)
        has_history = any('training_history' in result for result in successful_results.values())
        
        if has_history:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            for strategy in strategies:
                result = successful_results[strategy]
                if 'training_history' in result:
                    history = result['training_history']
                    epochs = history['epochs']
                    
                    # Plot training loss
                    axes[0, 0].plot(epochs, history['train_loss'], label=strategy, marker='o', linewidth=2, markersize=4)
                    
                    # Plot validation loss
                    axes[0, 1].plot(epochs, history['val_loss'], label=strategy, marker='s', linewidth=2, markersize=4)
                    
                    # Plot validation accuracy
                    axes[1, 0].plot(epochs, history['val_accuracy'], label=strategy, marker='^', linewidth=2, markersize=4)
                    
                    # Plot validation F1
                    axes[1, 1].plot(epochs, history['val_f1_score'], label=strategy, marker='d', linewidth=2, markersize=4)
            
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss Convergence')
            axes[0, 0].legend(fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Validation Loss Convergence')
            axes[0, 1].legend(fontsize=8)
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Validation Accuracy Convergence')
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(0, 1.0)
            
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].set_title('Validation F1-Score Convergence')
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1.0)
            
            fig.suptitle('Training Convergence Comparison - All Strategies', fontsize=16, fontweight='bold')
            plt.tight_layout()
            convergence_path = os.path.join(plots_dir, 'strategies_convergence_comparison.png')
            plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Plot 4: Metrics summary table as image
        fig, ax = plt.subplots(figsize=(12, max(4, len(strategies) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = [['Strategy', 'F1-Score', 'Precision', 'Recall', 'Accuracy', 'Epoch']]
        for strategy in strategies:
            result = successful_results[strategy]
            table_data.append([
                strategy,
                f"{result['best_f1']:.4f}",
                f"{result['final_precision']:.4f}",
                f"{result['final_recall']:.4f}",
                f"{result['final_accuracy']:.4f}",
                str(result.get('epoch', 'N/A'))
            ])
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(table_data[0])):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        
        # Style data rows with alternating colors
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('#ffffff')
        
        # Highlight best F1-score
        best_idx = strategies.index(max(successful_results.items(), key=lambda x: x[1]['best_f1'])[0]) + 1
        for j in range(len(table_data[0])):
            cell = table[(best_idx, j)]
            cell.set_facecolor('#FFD700')
            cell.set_text_props(weight='bold')
        
        plt.title('Validation Metrics Summary - All Strategies', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        table_path = os.path.join(plots_dir, 'strategies_metrics_table.png')
        plt.savefig(table_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if not quiet:
            print(f"\n✓ Strategy comparison plots saved:")
            print(f"  - Metrics comparison: {os.path.basename(comparison_path)}")
            print(f"  - F1-Score ranking: {os.path.basename(ranking_path)}")
            if has_history:
                print(f"  - Convergence comparison: {os.path.basename(convergence_path)}")
            print(f"  - Metrics table: {os.path.basename(table_path)}")
            
    except Exception as e:
        if not quiet:
            print(f"Warning: Failed to create comparison plots: {e}")
        import traceback
        traceback.print_exc()


def train_multiple_fusion_strategies(config: VideoShotConfig, features_dir: str,
                                   ground_truth_file: str, model_base_path: str,
                                   fusion_strategies: List[str] = None,
                                   show_progress: bool = True,
                                   quiet: bool = False,
                                   save_plots: bool = True,
                                   results_file: str = None) -> Dict[str, Dict[str, float]]:
    """
    Train transformer models for multiple fusion strategies.
    
    Args:
        config: System configuration
        features_dir: Directory containing extracted features
        ground_truth_file: JSON file with ground truth boundaries
        model_base_path: Base path for saving models (will append strategy names)
        fusion_strategies: List of fusion strategies to train. If None, uses all available
        show_progress: Whether to show progress bars
        quiet: Whether to suppress non-essential output
        save_plots: Whether to save training/validation plots for each strategy
        results_file: Path to save combined results JSON (will be updated after each strategy)
        
    Returns:
        Dictionary mapping fusion strategy names to their training results
    """
    # Default fusion strategies if none specified
    if fusion_strategies is None:
        fusion_strategies = ["concatenation", "weighted", "pca", "cross_transformer", "hierarchical"]
    
    # Default results file if not specified
    if results_file is None:
        results_file = os.path.join(os.path.dirname(model_base_path), "training_results.json")
    
    # Directory for individual strategy result files
    results_dir = os.path.dirname(results_file)
    
    print(f"Fusion strategies to train: {fusion_strategies}")
    if not quiet:
        print("="*80)
        print(f"TRAINING MULTIPLE TRANSFORMER MODELS")
        print(f"Fusion strategies: {', '.join(fusion_strategies)}")
        print(f"Results will be saved to: {results_file}")
        print(f"Individual results in: {results_dir}")
        print("="*80)
    
    results = {}
    
    # Setup outer progress bar for strategies if enabled
    strategy_iterator = tqdm(fusion_strategies, desc="Training strategies", disable=not show_progress) if show_progress else enumerate(fusion_strategies, 1)
    
    for i, strategy in (enumerate(strategy_iterator, 1) if show_progress else strategy_iterator):
        if show_progress:
            strategy_iterator.set_description(f"Strategy {i}/{len(fusion_strategies)}: {strategy}")
        elif not quiet:
            print(f"\n{'='*60}")
            print(f"TRAINING MODEL {i}/{len(fusion_strategies)}: {strategy.upper()}")
            print(f"{'='*60}")
        
        try:
            # Create strategy-specific model path
            model_path = model_base_path.replace('.pth', f'_{strategy}.pth')
            
            # Ensure model directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Train model for this strategy
            strategy_results = train_transformer_model(
                config=config,
                features_dir=features_dir,
                ground_truth_file=ground_truth_file,
                model_save_path=model_path,
                fusion_strategy=strategy,
                show_progress=show_progress,
                quiet=quiet,
                save_plots=save_plots
            )
            
            # Store results
            results[strategy] = strategy_results
            
            if not quiet:
                print(f"\n✓ Successfully trained {strategy} model")
                print(f"  Model saved to: {model_path}")
                print(f"  Best F1-score: {strategy_results['best_f1']:.4f}")
            
            # Save individual strategy result to its own JSON file
            save_single_strategy_result(strategy, strategy_results, results_dir, quiet)
            
            # Save/update combined results JSON (incremental save)
            save_training_results(results, results_file)
            if not quiet:
                print(f"  Combined results updated: {results_file}")
            
        except Exception as e:
            error_msg = f"\n✗ Failed to train {strategy} model: {e}"
            if show_progress:
                tqdm.write(error_msg)
            elif not quiet:
                print(error_msg)
            results[strategy] = {'error': str(e), 'best_f1': 0.0}
            
            # Save error result for this strategy
            save_single_strategy_result(strategy, results[strategy], results_dir, quiet)
            
            # Save/update combined results even on failure
            save_training_results(results, results_file)
            
            traceback.print_exc()
            exit(1)  # Exit on failure to ensure issues are addressed immediately
    
    # Print summary
    if not quiet:
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print(f"{'='*80}")
        
        successful_strategies = []
        for strategy, result in results.items():
            if 'error' in result:
                print(f"✗ {strategy:15} - FAILED: {result['error']}")
            else:
                f1_score = result['best_f1']
                print(f"✓ {strategy:15} - F1: {f1_score:.4f} | "
                      f"Precision: {result['final_precision']:.4f} | "
                      f"Recall: {result['final_recall']:.4f}")
                successful_strategies.append((strategy, f1_score))
        
        # Find best performing strategy
        if successful_strategies:
            best_strategy, best_f1 = max(successful_strategies, key=lambda x: x[1])
            print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy} (F1: {best_f1:.4f})")
        else:
            print(f"\n⚠️  No models trained successfully!")
        
        print(f"\n📁 Results saved to:")
        print(f"   Combined: {results_file}")
        print(f"   Individual: {results_dir}/training_result_<strategy>.json")
    
    # Create comparison plots for all strategies
    if save_plots and len([r for r in results.values() if 'error' not in r]) > 1:
        try:
            create_strategy_comparison_plots(results, model_base_path, quiet)
        except Exception as e:
            if not quiet:
                print(f"Warning: Failed to create comparison plots: {e}")
    
    return results


def save_training_results(results: Dict[str, Dict[str, float]], 
                        results_file: str) -> None:
    """
    Save training results to a JSON file.
    
    Args:
        results: Dictionary mapping fusion strategies to training results
        results_file: Path to save results JSON file
    """
    import json
    from datetime import datetime
    
    # Prepare results for JSON serialization (remove non-serializable numpy arrays)
    serializable_results = {}
    for strategy, result in results.items():
        serializable_result = {}
        for key, value in result.items():
            if key == 'training_history':
                # Convert numpy arrays to lists if present
                history = {}
                for hist_key, hist_value in value.items():
                    if isinstance(hist_value, (list, tuple)):
                        history[hist_key] = [float(v) if not isinstance(v, (int, str)) else v for v in hist_value]
                    else:
                        history[hist_key] = hist_value
                serializable_result[key] = history
            else:
                serializable_result[key] = value
        serializable_results[strategy] = serializable_result
    
    # Find best strategy
    successful_strategies = {k: v for k, v in serializable_results.items() if 'error' not in v}
    best_strategy = None
    best_f1 = 0.0
    if successful_strategies:
        best_strategy = max(successful_strategies.items(), key=lambda x: x[1]['best_f1'])[0]
        best_f1 = successful_strategies[best_strategy]['best_f1']
    
    # Add timestamp and metadata
    results_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'num_strategies': len(serializable_results),
        'strategies_trained': list(serializable_results.keys()),
        'successful_strategies': list(successful_strategies.keys()),
        'failed_strategies': [k for k, v in serializable_results.items() if 'error' in v],
        'best_strategy': best_strategy,
        'best_f1_score': best_f1,
        'results': serializable_results
    }
    
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training results saved to: {results_file}")
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  Total strategies: {len(serializable_results)}")
    print(f"  Successful: {len(successful_strategies)}")
    print(f"  Failed: {len(results_with_metadata['failed_strategies'])}")
    if best_strategy:
        print(f"  Best strategy: {best_strategy} (F1: {best_f1:.4f})")
    print(f"{'='*80}")


def save_single_strategy_result(strategy: str, result: Dict[str, Any], 
                                results_dir: str, quiet: bool = False) -> str:
    """
    Save training results for a single strategy to its own JSON file.
    
    Args:
        strategy: Name of the fusion strategy or feature type
        result: Training results dictionary for this strategy
        results_dir: Directory to save results JSON file
        quiet: Whether to suppress output messages
        
    Returns:
        Path to the saved JSON file
    """
    import json
    from datetime import datetime
    
    # Prepare result for JSON serialization
    serializable_result = {}
    for key, value in result.items():
        if key == 'training_history':
            # Convert numpy arrays to lists if present
            history = {}
            for hist_key, hist_value in value.items():
                if isinstance(hist_value, (list, tuple)):
                    history[hist_key] = [float(v) if not isinstance(v, (int, str)) else v for v in hist_value]
                else:
                    history[hist_key] = hist_value
            serializable_result[key] = history
        else:
            serializable_result[key] = value
    
    # Add metadata
    result_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'strategy': strategy,
        'is_individual_feature': is_individual_feature(strategy),
        'success': 'error' not in result,
        'best_f1': result.get('best_f1', 0.0),
        'result': serializable_result
    }
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Save to strategy-specific file
    result_file = os.path.join(results_dir, f"training_result_{strategy}.json")
    with open(result_file, 'w') as f:
        json.dump(result_with_metadata, f, indent=2)
    
    if not quiet:
        print(f"  Strategy result saved to: {result_file}")
    
    return result_file


def load_training_results(results_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load training results from a JSON file.
    
    Args:
        results_file: Path to results JSON file
        
    Returns:
        Dictionary mapping fusion strategies to training results
    """
    import json
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    if 'results' in data:
        return data['results']
    else:
        # Legacy format
        return data


def compare_fusion_strategies(results: Dict[str, Dict[str, float]], save_to_file: str = None) -> None:
    """
    Print a comparison table of fusion strategy performance.
    
    Args:
        results: Dictionary mapping fusion strategies to training results
        save_to_file: Optional path to save comparison table as text file
    """
    output_lines = []
    
    header = f"\n{'='*100}"
    output_lines.append(header)
    output_lines.append("FUSION STRATEGY COMPARISON")
    output_lines.append(f"{'='*100}")
    
    table_header = f"{'Strategy':<20} {'F1-Score':<10} {'Precision':<12} {'Recall':<10} {'Accuracy':<12} {'Epoch':<8}"
    output_lines.append(table_header)
    output_lines.append(f"{'-'*100}")
    
    # Sort by F1-score (descending)
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1].get('best_f1', 0), 
                          reverse=True)
    
    for strategy, result in sorted_results:
        if 'error' in result:
            line = f"{strategy:<20} {'FAILED':<10} {'-':<12} {'-':<10} {'-':<12} {'-':<8}"
        else:
            f1 = result.get('best_f1', 0)
            precision = result.get('final_precision', 0)
            recall = result.get('final_recall', 0)
            accuracy = result.get('final_accuracy', 0)
            epoch = result.get('epoch', 0)
            
            line = f"{strategy:<20} {f1:<10.4f} {precision:<12.4f} {recall:<10.4f} {accuracy:<12.4f} {epoch:<8}"
        output_lines.append(line)
    
    footer = f"{'-'*100}"
    output_lines.append(footer)
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Save to file if requested
    if save_to_file:
        try:
            os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
            with open(save_to_file, 'w') as f:
                f.write('\n'.join(output_lines))
            print(f"\nComparison table saved to: {save_to_file}")
        except Exception as e:
            print(f"Warning: Failed to save comparison table: {e}")


def create_model_ensemble(model_dir: str, fusion_strategies: List[str], 
                        ensemble_name: str = "ensemble") -> str:
    """
    Create an ensemble configuration that combines multiple fusion strategy models.
    
    Args:
        model_dir: Directory containing trained models
        fusion_strategies: List of fusion strategies to include in ensemble
        ensemble_name: Name for the ensemble configuration
        
    Returns:
        Path to the ensemble configuration file
    """
    import json
    
    ensemble_config = {
        'name': ensemble_name,
        'type': 'transformer_ensemble',
        'strategies': {},
        'voting_method': 'weighted',  # can be 'majority', 'weighted', or 'average'
        'weights': {}
    }
    
    total_weight = 0
    for strategy in fusion_strategies:
        model_path = os.path.join(model_dir, f"transformer_detection_{strategy}.pth")
        if os.path.exists(model_path):
            ensemble_config['strategies'][strategy] = model_path
            # Default equal weights (can be adjusted based on performance)
            ensemble_config['weights'][strategy] = 1.0
            total_weight += 1.0
    
    # Normalize weights
    for strategy in ensemble_config['weights']:
        ensemble_config['weights'][strategy] /= total_weight
    
    ensemble_file = os.path.join(model_dir, f"{ensemble_name}_config.json")
    with open(ensemble_file, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    print(f"Ensemble configuration saved to: {ensemble_file}")
    print(f"Ensemble includes {len(ensemble_config['strategies'])} models: {list(ensemble_config['strategies'].keys())}")
    
    return ensemble_file


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Extract information from a saved model checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Dictionary with model information
    """
    if not os.path.exists(model_path):
        return {'error': 'Model file not found'}
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'model_path': model_path,
            'config': checkpoint.get('config', {}),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'loss': checkpoint.get('loss', 'unknown'),
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'has_optimizer': 'optimizer_state_dict' in checkpoint
        }
        
        # Extract model architecture info if available
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            num_parameters = sum(p.numel() for p in state_dict.values())
            info['num_parameters'] = num_parameters
        
        return info
        
    except Exception as e:
        return {'error': f'Failed to load model: {e}'}


def list_trained_models(model_dir: str, pattern: str = "transformer_detection_*.pth") -> Dict[str, Dict[str, Any]]:
    """
    List all trained transformer models and their information.
    
    Args:
        model_dir: Directory containing model files
        pattern: Filename pattern to match
        
    Returns:
        Dictionary mapping fusion strategies to model information
    """
    from pathlib import Path
    import glob
    
    models = {}
    pattern_path = os.path.join(model_dir, pattern)
    
    for model_file in glob.glob(pattern_path):
        # Extract strategy name from filename
        filename = os.path.basename(model_file)
        if 'transformer_detection_' in filename:
            strategy = filename.replace('transformer_detection_', '').replace('.pth', '')
            models[strategy] = get_model_info(model_file)
    
    return models


def print_model_summary(model_dir: str) -> None:
    """
    Print a summary of all trained models.
    
    Args:
        model_dir: Directory containing model files
    """
    models = list_trained_models(model_dir)
    
    if not models:
        print("No trained transformer models found.")
        return
    
    print(f"\n{'='*80}")
    print("TRAINED MODELS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<20} {'Size (MB)':<10} {'Parameters':<12} {'Epoch':<8} {'Status'}")
    print(f"{'-'*80}")
    
    for strategy, info in models.items():
        if 'error' in info:
            print(f"{strategy:<20} {'-':<10} {'-':<12} {'-':<8} ERROR: {info['error']}")
        else:
            size_mb = info.get('model_size_mb', 0)
            params = info.get('num_parameters', 0)
            epoch = info.get('epoch', 'unknown')
            
            # Format parameters in human-readable form
            if params > 1e6:
                params_str = f"{params/1e6:.1f}M"
            elif params > 1e3:
                params_str = f"{params/1e3:.1f}K"
            else:
                params_str = str(params)
            
            print(f"{strategy:<20} {size_mb:<10.1f} {params_str:<12} {epoch:<8} OK")
    
    print(f"{'-'*80}")
    print(f"Total models: {len(models)}")


def extract_features_for_all_strategies(config: VideoShotConfig, video_dir: str, 
                                       features_dir: str, 
                                       fusion_strategies: List[str] = None,
                                       show_progress: bool = True,
                                       quiet: bool = False,
                                       max_workers: int = 1,
                                       parallel_features: bool = False,
                                       skip_videos: int = 0) -> None:
    """
    Extract features for all fusion strategies to avoid repeated feature extraction.
    
    Args:
        config: System configuration
        video_dir: Directory containing training videos
        features_dir: Directory to save extracted features
        fusion_strategies: List of fusion strategies to prepare features for
        show_progress: Whether to show progress bars
        quiet: Whether to suppress non-essential output
        max_workers: Number of parallel workers for video loading (1 = sequential)
        parallel_features: Whether to enable parallel feature extraction within each video
        skip_videos: Number of videos to skip from the beginning (useful for resuming)
    """
    if fusion_strategies is None:
        fusion_strategies = ["concatenation", "weighted", "pca", "cross_transformer", "hierarchical"]
    
    if not quiet:
        print("="*80)
        print("EXTRACTING FEATURES FOR ALL FUSION STRATEGIES")
        print("="*80)
    
    os.makedirs(features_dir, exist_ok=True)
    
    # First extract individual features if not already done
    if not quiet:
        print("Extracting individual features...")
    extract_training_features(config, video_dir, features_dir, show_progress, quiet, max_workers, parallel_features, skip_videos)
    
    # Then create fused features for each strategy
    if not quiet:
        print(f"\nCreating strategy-specific fused features...")
    
    # Get list of videos
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    video_names = [video_file.stem for video_file in video_files]
    
    # Setup outer progress bar for strategies if enabled
    strategy_iterator = tqdm(fusion_strategies, desc="Processing strategies", disable=not show_progress) if show_progress else fusion_strategies
    
    for strategy in strategy_iterator:
        if show_progress:
            strategy_iterator.set_postfix({"Strategy": strategy})
        elif not quiet:
            print(f"\nProcessing {strategy} fusion...")
        
        # Setup inner progress bar for videos if enabled
        video_iterator = tqdm(video_names, desc=f"Videos ({strategy})", leave=False, disable=not show_progress) if show_progress else video_names
        
        for video_name in video_iterator:
            if show_progress:
                video_iterator.set_postfix({"Video": video_name[:20] + "..." if len(video_name) > 20 else video_name})
            
            try:
                # Check if strategy-specific features already exist and are valid
                feature_check = check_features_exist_and_valid(
                    features_dir, video_name, 
                    feature_names=['hsv_histogram', 'cnn', 'clip', 'optical_flow'],
                    fusion_strategy=strategy
                )
                
                if not feature_check['needs_extraction']:
                    if not quiet and not show_progress:
                        print(f"  ✓ {video_name} ({strategy}) - all features already exist and are valid")
                    continue
                
                # Check if individual features exist and are valid
                if not feature_check['all_individual_valid']:
                    if not quiet and not show_progress:
                        missing_msg = f"Missing individual features: {', '.join(feature_check['missing_features'])}"
                        print(f"  ⚠ {video_name} ({strategy}) - {missing_msg}")
                        print(f"    Note: Individual features need to be extracted first using main feature extraction")
                    continue
                
                # Individual features exist, but fused features for this strategy don't
                # Load and fuse individual features
                fused_features = load_and_fuse_individual_features(
                    features_dir, video_name, strategy, quiet=quiet or show_progress
                )
                if not quiet and not show_progress:
                    print(f"  ✓ {video_name} ({strategy}): {fused_features.shape}")
                
            except Exception as e:
                error_msg = f"  ✗ {video_name} ({strategy}): {e}"
                if show_progress:
                    tqdm.write(error_msg)
                elif not quiet:
                    print(error_msg)
    
    if not quiet:
        print(f"\n✓ Feature extraction completed for all strategies!")
        print(f"Individual features and strategy-specific fused features saved in: {features_dir}")


def check_features_only(video_dir: str, features_dir: str, feature_types: List[str], fix_gaps: bool = False, config: VideoShotConfig = None) -> None:
    """
    Check for existence of the 4 types of features for videos and print the range of videos that have features extracted.
    Optionally fix gaps by extracting features for missing videos.
    
    Args:
        video_dir: Directory containing videos
        features_dir: Directory containing feature files
        feature_types: List of feature types to check
        fix_gaps: If True, extract features for videos that are missing features
        config: System configuration (required if fix_gaps is True)
    """
    # Get all video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    # Sort video files by name for consistent ordering
    video_files.sort()
    
    # Check features for each video
    videos_with_features = []
    videos_without_features = []
    videos_with_features_indices = []  # Track indices of videos with features
    
    print(f"Checking {len(video_files)} videos for feature existence...")
    print(f"Required feature types: {', '.join(feature_types)}")
    print("-" * 80)
    
    for idx, video_file in enumerate(video_files):
        video_stem = video_file.stem
        
        # Check if all required features exist and are valid
        feature_check = check_features_exist_and_valid(
            features_dir, video_stem, 
            feature_names=feature_types,
            check_only_existence=True
        )
        
        if feature_check['all_individual_valid']:
            videos_with_features.append(video_stem)
            videos_with_features_indices.append(idx)  # 0-based index
        else:
            videos_without_features.append({
                'name': video_stem,
                'missing': feature_check['missing_features'],
                'index': idx  # 0-based index
            })
    
    # Check if indices are continuous
    is_continuous = True
    gaps = []
    if len(videos_with_features_indices) > 1:
        for i in range(1, len(videos_with_features_indices)):
            prev_idx = videos_with_features_indices[i-1]
            curr_idx = videos_with_features_indices[i]
            if curr_idx - prev_idx > 1:
                # There's a gap
                is_continuous = False
                gap_start = prev_idx + 1
                gap_end = curr_idx - 1
                gaps.append((gap_start, gap_end))
    
    # Print summary
    print(f"\nFEATURE EXISTENCE SUMMARY:")
    print(f"=" * 80)
    print(f"Total videos found: {len(video_files)}")
    print(f"Videos with all features: {len(videos_with_features)}")
    print(f"Videos missing features: {len(videos_without_features)}")
    print(f"Continuous range: {'YES' if is_continuous else 'NO'}")
    
    if videos_with_features:
        print(f"\nVIDEOS WITH ALL FEATURES ({len(videos_with_features)}):")
        print("-" * 50)
        # Show range of videos with features including indices
        first_with_features = videos_with_features[0]
        last_with_features = videos_with_features[-1]
        first_idx = videos_with_features_indices[0] + 1  # Convert to 1-based
        last_idx = videos_with_features_indices[-1] + 1  # Convert to 1-based
        print(f"Range: [{first_idx}] {first_with_features} to [{last_idx}] {last_with_features}")
        
        if not is_continuous:
            print(f"\n⚠️  WARNING: Features are NOT continuous!")
            print(f"Found {len(gaps)} gap(s) in the sequence:")
            for gap_start, gap_end in gaps:
                if gap_start == gap_end:
                    missing_video = video_files[gap_start].stem
                    print(f"  Gap at index [{gap_start + 1}]: {missing_video}")
                else:
                    start_video = video_files[gap_start].stem
                    end_video = video_files[gap_end].stem
                    print(f"  Gap at indices [{gap_start + 1}] to [{gap_end + 1}]: {start_video} to {end_video}")
        else:
            print(f"✓ Features form a continuous range from index {first_idx} to {last_idx}")
        
        # If the list is small, show all video names
        if len(videos_with_features) <= 20:
            for i, (video_name, orig_idx) in enumerate(zip(videos_with_features, videos_with_features_indices), 1):
                print(f"  {i:3d}. [{orig_idx + 1:3d}] {video_name}")
        else:
            # Show first and last few
            print(f"First 5:")
            for i, (video_name, orig_idx) in enumerate(zip(videos_with_features[:5], videos_with_features_indices[:5]), 1):
                print(f"  {i:3d}. [{orig_idx + 1:3d}] {video_name}")
            print(f"  ...")
            print(f"Last 5:")
            start_idx = len(videos_with_features) - 4
            for i, (video_name, orig_idx) in enumerate(zip(videos_with_features[-5:], videos_with_features_indices[-5:]), start_idx):
                print(f"  {i:3d}. [{orig_idx + 1:3d}] {video_name}")
    
    if videos_without_features:
        print(f"\nVIDEOS MISSING FEATURES ({len(videos_without_features)}):")
        print("-" * 50)
        
        # Show details about missing features (up to 10 videos)
        for i, video_info in enumerate(videos_without_features[:10], 1):
            missing_features_str = ', '.join(video_info['missing'])
            orig_idx = video_info['index'] + 1  # Convert to 1-based
            print(f"  {i:3d}. [{orig_idx:3d}] {video_info['name']} (missing: {missing_features_str})")
        
        if len(videos_without_features) > 10:
            print(f"  ... and {len(videos_without_features) - 10} more videos")
    
    # Fix gaps if requested
    if fix_gaps and videos_without_features:
        if config is None:
            print(f"\n⚠️  Cannot fix gaps: config parameter is required for feature extraction")
            print(f"Feature check completed.")
            return
        
        # Filter videos_without_features to only include gaps (not files after the last processed)
        if videos_with_features_indices:
            last_processed_idx = max(videos_with_features_indices)
            videos_to_fix = [v for v in videos_without_features if v['index'] <= last_processed_idx]
        else:
            # No videos have features yet, so we can't fix gaps - this would be initial extraction
            print(f"\n⚠️  Cannot fix gaps: No videos have features yet. Use --extract_features for initial extraction.")
            print(f"Feature check completed.")
            return
        
        if not videos_to_fix:
            print(f"\n✓ No gaps to fix within the processed range (up to index {last_processed_idx + 1})")
            print(f"All missing features are beyond the last processed video.")
        else:
            print(f"\n" + "=" * 80)
            print(f"FIXING GAPS - EXTRACTING FEATURES FOR MISSING VIDEOS")
            print(f"=" * 80)
            print(f"Processing range: up to index {last_processed_idx + 1} (last video with features)")
            print(f"Will extract features for {len(videos_to_fix)} videos within this range...")
            print(f"Skipping {len(videos_without_features) - len(videos_to_fix)} videos beyond the processed range")
            
            # Initialize coordinator for feature extraction
            os.makedirs(features_dir, exist_ok=True)
            coordinator = ProcessingCoordinator(config)
            coordinator.initialize()
            
            # Extract features for videos missing features (only gaps)
            success_count = 0
            error_count = 0
            
            from tqdm import tqdm
            progress_bar = tqdm(videos_to_fix, desc="Fixing gaps")
            
            for video_info in progress_bar:
                video_name = video_info['name']
                orig_idx = video_info['index']
                video_file = video_files[orig_idx]
                
                progress_bar.set_postfix({
                    "Current": f"[{orig_idx + 1}] {video_name[:15]}...",
                    "Index": f"{orig_idx + 1}/{last_processed_idx + 1}"
                })
                
                try:
                    # Load video with improved error handling
                    frames = coordinator.video_processor.load_video(
                        str(video_file), 
                        max_retries=3, 
                        skip_corrupted=False
                    )
                    
                    # Extract features
                    features = coordinator._extract_all_features_parallel(frames, max_workers=4)

                    # Save individual features
                    for feature_name, feature_array in features.items():
                        individual_file = os.path.join(features_dir, f"{video_name}_{feature_name}.npy")
                        np.save(individual_file, feature_array)
                    
                    success_count += 1
                    progress_bar.set_postfix({
                        "Status": f"✓ Fixed [{orig_idx + 1}]",
                        "Progress": f"{success_count}/{len(videos_to_fix)}"
                    })
                    
                except Exception as e:
                    error_count += 1
                    tqdm.write(f"✗ Error extracting features for [{orig_idx + 1}] {video_name}: {e}")
                    progress_bar.set_postfix({
                        "Status": f"✗ Error [{orig_idx + 1}]",
                        "Errors": f"{error_count}"
                    })
            
            progress_bar.close()
            
            print(f"\n" + "=" * 80)
            print(f"GAP FIXING COMPLETED")
            print(f"=" * 80)
            print(f"Successfully fixed: {success_count}/{len(videos_to_fix)} gap videos")
            print(f"Errors encountered: {error_count}/{len(videos_to_fix)} gap videos")
            print(f"Range processed: indices 1 to {last_processed_idx + 1}")
            
            if success_count > 0:
                print(f"✓ Gaps have been fixed! Re-run with --check_only to verify continuity.")
            
            if error_count > 0:
                print(f"⚠️  Some gap videos could not be processed. Check the error messages above.")
    
    elif fix_gaps and not videos_without_features:
        print(f"\n✓ No gaps to fix - all videos already have complete features!")
    
    print(f"\n" + "=" * 80)
    print(f"Feature check completed.")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train transformer detection model(s)")
    parser.add_argument("--video_dir", required=True, help="Directory containing training videos")
    parser.add_argument("--ground_truth", required=True, help="JSON file with ground truth boundaries")
    parser.add_argument("--features_dir", default="training_features", help="Directory to save/load features")
    parser.add_argument("--model_path", default="models/transformer_detection.pth", help="Path to save trained model")
    parser.add_argument("--extract_features", action="store_true", help="Extract features before training")
    parser.add_argument("--fusion_strategy", default="hierarchical", 
                       help="Fusion strategy to use (concatenation, weighted, pca, cross_transformer, hierarchical). Ignored if --feature_type is set.")
    
    # Individual feature training (NEW)
    parser.add_argument("--feature_type", type=str, default=None,
                       choices=['hsv_histogram', 'cnn', 'clip', 'optical_flow', 
                               'concatenation', 'weighted', 'pca', 'cross_transformer', 'hierarchical'],
                       help="Feature type for training. Can be individual feature (hsv_histogram, cnn, clip, optical_flow) "
                            "or fusion strategy (concatenation, weighted, pca, cross_transformer, hierarchical). "
                            "If not specified, uses --fusion_strategy.")
    parser.add_argument("--train_all_features", action="store_true",
                       help="Train models for all individual feature types (hsv_histogram, cnn, clip, optical_flow)")
    
    # Multi-strategy training
    parser.add_argument("--train_all_strategies", action="store_true", 
                       help="Train models for all available fusion strategies")
    parser.add_argument("--fusion_strategies", nargs="+",
                       default=[ "pca", "cross_transformer", "hierarchical","concatenation","weighted"], 
                       help="List of fusion strategies to train")
    parser.add_argument("--results_file", default="models/training_results.json",
                       help="File to save training results summary")
    parser.add_argument("--create_ensemble", action="store_true",
                       help="Create ensemble configuration after training")
    parser.add_argument("--list_models", action="store_true",
                       help="List all trained models and exit")
    parser.add_argument("--compare_results", type=str,
                       help="Path to results file for comparison (and exit)")
    
    # Logging and progress control arguments
    parser.add_argument("--disable_logging", action="store_true",
                       help="Disable all logging output")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress non-essential output (quiet mode)")
    parser.add_argument("--no_progress", action="store_true",
                       help="Disable progress bars")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help="Set logging level")
    parser.add_argument("--file_logging_only", action="store_true",
                       help="Only log to file, disable console logging")
    
    # Parallel processing arguments
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Number of parallel workers for video loading (1 = sequential, 2-4 recommended for parallel)")
    parser.add_argument("--parallel", action="store_true",
                       help="Enable parallel video loading (equivalent to --max_workers 2)")
    parser.add_argument("--parallel_features", action="store_true",
                       help="Enable parallel feature extraction within each video (HSV, CNN, CLIP, optical flow in parallel)")
    parser.add_argument("--feature_workers", type=int, default=4,
                       help="Number of parallel workers for feature extraction (used with --parallel_features)")
    
    # Video health and repair arguments
    parser.add_argument("--check_video_health", action="store_true",
                       help="Check video health before processing")
    parser.add_argument("--health_sample_size", type=int, default=10,
                       help="Number of videos to sample for health check (0 = all)")
    parser.add_argument("--repair_videos", action="store_true",
                       help="Attempt to repair corrupted videos before processing")
    parser.add_argument("--skip_corrupted", action="store_true", default=True,
                       help="Skip corrupted videos instead of failing (default: True)")
    
    # Feature extraction control arguments
    parser.add_argument("--skip", type=int, default=0,
                       help="Skip the first N videos during feature extraction (useful for resuming after interruption)")
    parser.add_argument("--check_only", action="store_true",
                       help="Check for existence of the 4 types of features for videos and print the range of videos that have features extracted, then exit")
    parser.add_argument("--fix_gaps", action="store_true",
                       help="Must be used with --check_only. Extract features for videos that are missing features (fill gaps)")
    
    # Legacy arguments
    parser.add_argument("--use_individual_features", action="store_true", help="Use individual features instead of pre-fused")
    parser.add_argument("--feature_weights", type=str, help="JSON string with feature weights")
    parser.add_argument("--save_individual", action="store_true", help="Save individual features separately")
    parser.add_argument("--feature_types", nargs="+", default=["hsv_histogram", "cnn", "clip", "optical_flow"], 
                       help="Types of features to extract/use")
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.fix_gaps and not args.check_only:
        print("Error: --fix_gaps must be used with --check_only")
        parser.print_help()
        return
    
    # Handle utility commands first
    if args.list_models:
        print_model_summary(os.path.dirname(args.model_path))
        return
    
    if args.compare_results:
        results = load_training_results(args.compare_results)
        compare_fusion_strategies(results)
        return
    
    # Handle check_only command
    if args.check_only:
        if args.fix_gaps:
            # Create minimal config for feature extraction
            config = VideoShotConfig()
            config.logging.enabled = not args.disable_logging
            config.logging.quiet_mode = args.quiet
            config.logging.level = args.log_level
            check_features_only(args.video_dir, args.features_dir, args.feature_types, fix_gaps=True, config=config)
        else:
            check_features_only(args.video_dir, args.features_dir, args.feature_types, fix_gaps=False)
        return
    
    # Create configuration
    config = VideoShotConfig()
    config.fusion.strategy = args.fusion_strategy
    config.detection.method = "transformer"
    
    # Configure logging settings based on arguments
    config.logging.enabled = not args.disable_logging
    config.logging.quiet_mode = args.quiet
    config.logging.progress_bars = not args.no_progress
    config.logging.level = args.log_level
    config.logging.console_logging = not args.file_logging_only
    config.logging.log_to_file_only = args.file_logging_only
    
    # Determine progress and quiet settings
    show_progress = config.logging.progress_bars and not args.quiet
    quiet_mode = args.quiet or args.disable_logging
    
    # Handle parallel processing arguments
    max_workers = args.max_workers
    if args.parallel and max_workers == 1:
        max_workers = 2  # Default for parallel mode
    
    parallel_features = args.parallel_features
    if parallel_features:
        config.processing.feature_extraction_workers = args.feature_workers
    
    if max_workers > 1 and not quiet_mode:
        print(f"Using parallel processing with {max_workers} workers for video loading")
    if parallel_features and not quiet_mode:
        print(f"Using parallel feature extraction with {args.feature_workers} workers per video")
    if args.skip > 0 and not quiet_mode:
        print(f"Will skip the first {args.skip} videos during feature extraction")
    
    # Check video health if requested
    if args.check_video_health:
        if not quiet_mode:
            print("\\n" + "="*60)
            print("VIDEO HEALTH CHECK")
            print("="*60)
        
        health_results = check_video_health(
            args.video_dir, 
            args.health_sample_size, 
            quiet_mode
        )
        
        if health_results.get('corrupted', 0) > 0 and args.repair_videos:
            if not quiet_mode:
                print(f"\\nAttempting to repair {health_results['corrupted']} corrupted videos...")
            
            from utils.video_repair import batch_repair_videos
            repair_results = batch_repair_videos(args.video_dir)
            
            if not quiet_mode:
                print(f"Repair results:")
                print(f"  Attempted: {repair_results['attempted_repairs']}")
                print(f"  Successful: {repair_results['successful_repairs']}")
                print(f"  Failed: {repair_results['failed_repairs']}")
    
    # Extract features if requested
    if args.extract_features:
        if args.train_all_strategies:
            if not quiet_mode:
                print(f"Extracting features for all strategies: {args.fusion_strategies}")
            extract_features_for_all_strategies(
                config, args.video_dir, args.features_dir, 
                args.fusion_strategies, show_progress, quiet_mode, max_workers, parallel_features, args.skip
            )
        else:
            if not quiet_mode:
                print(f"Extracting features using {args.fusion_strategy} fusion strategy...")
            extract_training_features(
                config, args.video_dir, args.features_dir, 
                show_progress, quiet_mode, max_workers, parallel_features, args.skip
            )
    
    # Training
    # Ensure model path has correct extension
    if not args.model_path.endswith('.pth'):
        original_path = args.model_path
        args.model_path = args.model_path + '.pth'
        if not quiet_mode:
            print(f"Note: Added .pth extension to model path")
            print(f"  Original: {original_path}")
            print(f"  Updated:  {args.model_path}")
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Determine the effective feature type to use
    # feature_type takes precedence over fusion_strategy
    effective_feature_type = args.feature_type if args.feature_type else args.fusion_strategy
    
    # Results file for individual features
    results_file_features = args.results_file.replace('.json', '_individual_features.json')
    results_dir = os.path.dirname(args.results_file)
    
    if args.train_all_features:
        # Train models for all individual feature types
        if not quiet_mode:
            print(f"Training models for all individual feature types: {INDIVIDUAL_FEATURE_TYPES}")
            print(f"Results will be saved incrementally to: {results_file_features}")
        
        all_results = {}
        for feature_type in INDIVIDUAL_FEATURE_TYPES:
            if not quiet_mode:
                print(f"\\n{'='*60}")
                print(f"Training model for {feature_type} feature...")
                print(f"{'='*60}")
            
            # Create model path with feature type suffix
            model_base = args.model_path.replace('.pth', '')
            model_path = f"{model_base}_{feature_type}.pth"
            
            try:
                result = train_transformer_model(
                    config=config,
                    features_dir=args.features_dir,
                    ground_truth_file=args.ground_truth,
                    model_save_path=model_path,
                    fusion_strategy=feature_type,  # Using feature_type as the "strategy"
                    show_progress=show_progress,
                    quiet=quiet_mode
                )
                all_results[feature_type] = result
                
                # Save individual feature result to its own JSON file
                save_single_strategy_result(feature_type, result, results_dir, quiet_mode)
                
            except Exception as e:
                if not quiet_mode:
                    print(f"Error training {feature_type}: {e}")
                all_results[feature_type] = {'error': str(e), 'best_f1': 0.0}
                
                # Save error result for this feature
                save_single_strategy_result(feature_type, all_results[feature_type], results_dir, quiet_mode)
            
            # Save/update combined results JSON (incremental save after each feature)
            save_training_results(all_results, results_file_features)
            if not quiet_mode:
                print(f"  Combined results updated: {results_file_features}")
        
        if not quiet_mode:
            print(f"\\n{'='*60}")
            print("INDIVIDUAL FEATURE TRAINING COMPLETE")
            print(f"{'='*60}")
            for ft, res in all_results.items():
                if 'error' in res:
                    print(f"  {ft}: ERROR - {res['error']}")
                else:
                    print(f"  {ft}: F1={res.get('best_f1', 'N/A'):.4f}")
            print(f"\\n📁 Results saved to:")
            print(f"   Combined: {results_file_features}")
            print(f"   Individual: {results_dir}/training_result_<feature>.json")
    
    elif args.train_all_strategies:
        # Train multiple fusion strategies
        if not quiet_mode:
            print(f"Training models for strategies: {args.fusion_strategies}")
        results = train_multiple_fusion_strategies(
            config=config,
            features_dir=args.features_dir,
            ground_truth_file=args.ground_truth,
            model_base_path=args.model_path,
            fusion_strategies=args.fusion_strategies,
            show_progress=show_progress,
            quiet=quiet_mode,
            results_file=args.results_file  # Pass results file for incremental saving
        )
        
        # Results are already saved incrementally, but do a final save to be sure
        save_training_results(results, args.results_file)
        
        # Print comparison and save to text file
        if not quiet_mode:
            comparison_txt = args.results_file.replace('.json', '_comparison.txt')
            compare_fusion_strategies(results, save_to_file=comparison_txt)
        
        # Create ensemble if requested
        if args.create_ensemble:
            successful_strategies = [strategy for strategy, result in results.items() 
                                   if 'error' not in result]
            if len(successful_strategies) > 1:
                ensemble_file = create_model_ensemble(
                    model_dir=os.path.dirname(args.model_path),
                    fusion_strategies=successful_strategies,
                    ensemble_name="multi_fusion_ensemble"
                )
                if not quiet_mode:
                    print(f"\n✓ Ensemble configuration created: {ensemble_file}")
            else:
                if not quiet_mode:
                    print(f"\n⚠️  Cannot create ensemble: need at least 2 successful models")

    else:
        # Train single feature type (individual feature or fusion strategy)
        feature_name = effective_feature_type
        is_individual = is_individual_feature(effective_feature_type)
        
        if not quiet_mode:
            if is_individual:
                print(f"Training model for individual feature: {feature_name}")
            else:
                print(f"Training model for fusion strategy: {feature_name}")
        
        results = train_transformer_model(
            config=config,
            features_dir=args.features_dir,
            ground_truth_file=args.ground_truth,
            model_save_path=args.model_path,
            fusion_strategy=effective_feature_type,  # Works for both individual features and fusion strategies
            show_progress=show_progress,
            quiet=quiet_mode
        )
        
        if not quiet_mode:
            print(f"\n✓ Training completed!")
            print(f"Model saved to: {args.model_path}")
            print(f"Feature type: {feature_name} ({'individual' if is_individual else 'fusion'})")
            print(f"Best F1-score: {results['best_f1']:.4f}")


if __name__ == "__main__":
    main()