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
        feature_file = os.path.join(features_dir, f"{video_name}_{feature_name}.npy")
        try:
            if os.path.exists(feature_file):
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
        feature_file = os.path.join(features_dir, f"{video_name}_{feature_name}.npy")
        if os.path.exists(feature_file):
            features[feature_name] = np.load(feature_file)
        else:
            print(f"Warning: Feature file not found: {feature_file}")
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
    # Try loading strategy-specific fused features
    strategy_file = os.path.join(features_dir, f"{video_name}_{fusion_strategy}.npy")
    if os.path.exists(strategy_file):
        if file_name_only:
            return strategy_file
        return np.load(strategy_file)
    
    # Fallback to generic fused features
    fused_file = os.path.join(features_dir, f"{video_name}_fused.npy")
    if os.path.exists(fused_file):
        if file_name_only:
            return fused_file
        return np.load(fused_file)
    
    # Fallback to old naming convention
    old_file = os.path.join(features_dir, f"{video_name}_features.npy")
    if os.path.exists(old_file):
        if file_name_only:
            return old_file
        return np.load(old_file)
    
    # Last resort: load individual features and fuse them
    if not quiet:
        print(f"No pre-computed fused features found for {video_name}, attempting to load and fuse individual features...")
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
    """Dataset for training boundary detection models with lazy loading to reduce memory usage."""
    
    def __init__(self, features_dir: str, ground_truth_file: str, 
                 sequence_length: int = 100, overlap: int = 50, 
                 fusion_strategy: str = "hierarchical", quiet: bool = False,
                 lazy_loading: bool = True):
        """
        Initialize dataset.
        
        Args:
            features_dir: Directory containing extracted features
            ground_truth_file: JSON file with ground truth boundaries
            sequence_length: Length of input sequences
            overlap: Overlap between sequences
            fusion_strategy: Name of the fusion strategy to use for loading features
            quiet: Whether to suppress warning messages
            lazy_loading: If True, only store metadata and load features on-demand (saves memory)
        """
        self.features_dir = features_dir
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.fusion_strategy = fusion_strategy
        self.quiet = quiet
        self.lazy_loading = lazy_loading
        self._feature_dim = None  # Will be set after first valid feature load
        
        # Load ground truth
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)
        
        # Create training samples (metadata only if lazy loading)
        self.samples = self._create_samples()
        
        # Cache for loaded features when lazy loading (LRU cache to limit memory)
        if self.lazy_loading:
            from functools import lru_cache
            self._feature_cache = {}
    
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
            
            # Extract boundaries from transitions
            # Handle both old format (list of boundaries) and new format (dict with transitions)
            if isinstance(video_data, dict) and 'transitions' in video_data:
                # New format: extract first frame of each transition as boundary
                boundaries = [transition[1] for transition in video_data['transitions']]
            elif isinstance(video_data, list):
                # Old format: direct list of boundaries
                boundaries = video_data
            else:
                if not self.quiet:
                    print(f"Warning: Unknown annotation format for {video_name}")
                continue
            
            if self.lazy_loading:
                # Only load feature file metadata (shape) without loading full data
                strategy_file = os.path.join(self.features_dir, f"{video_name_stem}_{self.fusion_strategy}.npy")
                
                # Check if feature file exists
                if not os.path.exists(strategy_file):
                    # Try fallback files
                    fused_file = os.path.join(self.features_dir, f"{video_name_stem}_fused.npy")
                    old_file = os.path.join(self.features_dir, f"{video_name_stem}_features.npy")
                    
                    if os.path.exists(fused_file):
                        strategy_file = fused_file
                    elif os.path.exists(old_file):
                        strategy_file = old_file
                    else:
                        if not self.quiet:
                            print(f"Warning: No feature file found for {video_name_stem}")
                        try:
                            strategy_file = load_fused_features(self.features_dir, video_name_stem, self.fusion_strategy, self.quiet,file_name_only=True)
                        except FileNotFoundError as e:
                            if not self.quiet:
                                print(f"Warning: Could not load features for {video_name} (using stem: {video_name_stem}): {e}")
                            videos_skipped += 1
                            continue
                        if strategy_file is None:
                            if not self.quiet:
                                print(f"Warning: Could not find any feature file for {video_name_stem}")
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
                
                # Create binary boundary labels (just metadata)
                labels_list = []
                for boundary in boundaries:
                    if 0 <= boundary < num_frames - 1:
                        labels_list.append(boundary)
                
                # Create sequence metadata only
                for start_idx in range(0, num_frames - self.sequence_length + 1, self.overlap):
                    end_idx = start_idx + self.sequence_length
                    
                    # Check if sequence has boundary
                    has_boundary = any(start_idx <= b < end_idx - 1 for b in labels_list)
                    
                    # BALANCED SAMPLING: Include ALL boundary sequences and 25% of non-boundary
                    # With 1:56.6 frame imbalance, this gives reasonable sequence balance
                    # Target: ~80% boundary samples, 20% non-boundary samples
                    if has_boundary:
                        samples.append({
                            'video_name_stem': video_name_stem,
                            'video_name': video_name,
                            'start_frame': start_idx,
                            'end_frame': end_idx,
                            'boundaries': labels_list,  # Store boundary positions
                            'feature_file': strategy_file,  # Path to feature file
                            'num_frames': num_frames
                        })
                    elif np.random.random() < 0.50:  # 50% of non-boundary sequences
                        samples.append({
                            'video_name_stem': video_name_stem,
                            'video_name': video_name,
                            'start_frame': start_idx,
                            'end_frame': end_idx,
                            'boundaries': labels_list,  # Store boundary positions
                            'feature_file': strategy_file,  # Path to feature file
                            'num_frames': num_frames
                        })
            else:
                # Original behavior: load all features into memory
                try:
                    features = load_fused_features(self.features_dir, video_name_stem, self.fusion_strategy, self.quiet)
                    videos_with_features += 1
                except FileNotFoundError as e:
                    if not self.quiet:
                        print(f"Warning: Could not load features for {video_name} (using stem: {video_name_stem}): {e}")
                    videos_skipped += 1
                    continue
                
                # Create binary boundary labels
                labels = np.zeros(len(features) - 1)
                for boundary in boundaries:
                    if 0 <= boundary < len(labels):
                        labels[boundary] = 1
                
                # Create sequences with full data
                for start_idx in range(0, len(features) - self.sequence_length + 1, self.overlap):
                    end_idx = start_idx + self.sequence_length
                    
                    sequence_features = features[start_idx:end_idx]
                    sequence_labels = labels[start_idx:end_idx-1]  # N-1 labels for N frames
                    
                    # Balance the dataset - include sequences with and without boundaries
                    has_boundary = np.any(sequence_labels == 1)
                    
                    # BALANCED SAMPLING: Include ALL boundary sequences and 25% of non-boundary
                    # With 1:56.6 frame imbalance, this gives reasonable sequence balance
                    # Target: ~80% boundary samples, 20% non-boundary samples
                    if has_boundary:
                        samples.append({
                            'features': sequence_features,
                            'labels': sequence_labels,
                            'video_name': video_name,
                            'start_frame': start_idx
                        })
                    elif np.random.random() < 0.50:  # 40% of non-boundary sequences
                        samples.append({
                            'features': sequence_features,  
                            'labels': sequence_labels,
                            'video_name': video_name,
                            'start_frame': start_idx
                        })
            
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
                    # For lazy loading, count boundary samples and frames
                    for s in samples:
                        boundaries_in_seq = [b for b in s.get('boundaries', []) 
                                            if s['start_frame'] <= b < s['end_frame']]
                        if len(boundaries_in_seq) > 0:
                            boundary_samples += 1
                        total_boundary_frames += len(boundaries_in_seq)
                        total_frames += (s['end_frame'] - s['start_frame'] - 1)  # -1 for frame pairs
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
        
        if self.lazy_loading:
            # Load features on-demand from disk
            video_stem = sample['video_name_stem']
            start_frame = sample['start_frame']
            end_frame = sample['end_frame']
            feature_file = sample['feature_file']
            boundaries = sample['boundaries']
            
            # Load only the required sequence efficiently
            # IMPORTANT: We avoid numpy's mmap_mode on Windows because it can cause
            # "[WinError 8] Not enough memory resources are available to process this command"
            # This error occurs when Windows runs out of memory mapping handles (not RAM),
            # especially with many DataLoader workers accessing files repeatedly.
            # Solution: Direct file I/O to read only the slice we need without memory mapping.
            try:
                # Open file and seek to the specific position to load only what we need
                # This avoids both mmap issues on Windows and loading entire files
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
                    raise ValueError(f"Expected {self.sequence_length} frames, got {sequence_features.shape[0]}")
                
                # PAD FEATURES TO MAXIMUM DIMENSION IF NEEDED (for variable PCA dimensions)
                # This ensures all samples have the same feature dimension
                current_dim = sequence_features.shape[1]
                if self._feature_dim is not None and current_dim < self._feature_dim:
                    # Pad with zeros to match the maximum dimension
                    padding = np.zeros((self.sequence_length, self._feature_dim - current_dim))
                    sequence_features = np.concatenate([sequence_features, padding], axis=1)
                elif self._feature_dim is None:
                    # First sample, set the dimension
                    self._feature_dim = current_dim
                elif current_dim > self._feature_dim:
                    # Update to larger dimension and we'll need to pad previous samples
                    self._feature_dim = current_dim
                
                # Create labels for this sequence
                labels = np.zeros(self.sequence_length - 1)
                for boundary in boundaries:
                    if start_frame <= boundary < end_frame - 1:
                        labels[boundary - start_frame] = 1
                
                sequence_labels = labels
                
            except Exception as e:
                # Fallback to zeros if loading fails
                print(f"Error loading features for {video_stem} at frame {start_frame}: {e}")
                print(f"  Feature file: {feature_file}")
                print(f"  Frame range: {start_frame}-{end_frame}")
                print(f"  Expected shape: ({self.sequence_length}, feature_dim)")
                print(f"  Video ID: {sample.get('video_name', 'unknown')}")
                # Use the actual feature dimension from the first valid sample
                feature_dim = 256  # Default for PCA
                if hasattr(self, '_feature_dim') and self._feature_dim is not None:
                    feature_dim = self._feature_dim
                sequence_features = np.zeros((self.sequence_length, feature_dim))
                sequence_labels = np.zeros(self.sequence_length - 1)
        else:
            # Features already in memory
            sequence_features = sample['features']
            sequence_labels = sample['labels']
        
        # Final validation before converting to tensors
        assert len(sequence_features.shape) == 2, f"Features must be 2D, got {sequence_features.shape}"
        assert sequence_features.shape[0] == self.sequence_length, f"Expected {self.sequence_length} frames, got {sequence_features.shape[0]}"
        assert len(sequence_labels.shape) == 1, f"Labels must be 1D, got {sequence_labels.shape}"
        assert sequence_labels.shape[0] == self.sequence_length - 1, f"Expected {self.sequence_length-1} labels, got {sequence_labels.shape[0]}"
        
        features = torch.from_numpy(sequence_features).float()
        labels = torch.from_numpy(sequence_labels).float()
        
        return features, labels


def load_video_worker(video_file: Path, video_queue: queue.Queue, 
                     load_progress: tqdm = None) -> None:
    """
    Worker function to load a video file and put it in the queue with robust error handling.
    
    Args:
        video_file: Path to the video file
        video_queue: Queue to put loaded video data
        load_progress: Optional progress bar for loading
    """
    try:
        # Import here to avoid issues with multiprocessing
        import cv2
        import numpy as np
        import logging
        
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
        
        # Try multiple backends for better compatibility, prioritizing non-FFmpeg backends first
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG, cv2.CAP_ANY]  # Try Windows-specific backends first
        cap = None
        frames = []
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(str(video_file), backend)
                if not cap.isOpened():
                    if cap:
                        cap.release()
                    continue
                
                # Test read first frame
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    cap.release()
                    continue
                
                # Reset to beginning and read all frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frames = []
                consecutive_failures = 0
                max_failures = 10
                
                while consecutive_failures < max_failures:
                    ret, frame = cap.read()
                    if not ret:
                        # Check if we've reached end naturally
                        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        if current_pos >= total_frames - 1:
                            break  # Natural end
                        consecutive_failures += 1
                        continue
                    
                    consecutive_failures = 0
                    
                    # Validate and process frame
                    if frame is not None:
                        try:
                            # Check if frame has valid size
                            if hasattr(frame, 'size') and frame.size > 0:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frames.append(frame_rgb)
                        except cv2.error:
                            # Skip corrupted frame
                            continue
                
                cap.release()
                
                if len(frames) > 0:
                    frames = np.array(frames)
                    break  # Successfully loaded with this backend
                
            except Exception:
                if cap:
                    cap.release()
                continue
        
        if len(frames) == 0:
            raise RuntimeError("Could not load any frames with any backend")
        
        # Put the loaded video data in queue
        video_data = {
            'video_file': video_file,
            'frames': frames,
            'video_stem': video_file.stem
        }
        video_queue.put(video_data)
        
        if load_progress:
            load_progress.update(1)
            load_progress.set_postfix({"Loaded": video_file.name})
            
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
        load_progress = tqdm(total=total_videos, desc="Loading videos", position=0)
        extract_progress = tqdm(total=total_videos, desc="Extracting features", position=1)
    
    videos_processed = 0
    videos_loaded = 0
    
    # Start video loading in background threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video loading tasks
        loading_futures = []
        for video_file in video_files:
            future = executor.submit(load_video_worker, video_file, video_queue, load_progress)
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
                
                # Extract features from loaded video
                video_file = video_data['video_file']
                frames = video_data['frames']
                video_stem = video_data['video_stem']
                
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
                
                if extract_progress:
                    extract_progress.set_postfix({"Current": video_file.name})
                elif not quiet and not show_progress:
                    print(f"Extracting features for: {video_file.name}")
                
                try:
                    # Extract features using the shared coordinator
                    features = coordinator._extract_all_features(frames, parallel_features, max_workers)

                    # Save individual features
                    for feature_name, feature_array in features.items():
                        individual_file = os.path.join(features_dir, f"{video_stem}_{feature_name}.npy")
                        np.save(individual_file, feature_array)
                        if not quiet and not show_progress:
                            print(f"Saved {feature_name} features: {feature_array.shape}")
                    
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
                          quiet: bool = False, save_plots: bool = True) -> Dict[str, float]:
    """
    Train the transformer detection model for a specific fusion strategy.
    
    Args:
        config: System configuration
        features_dir: Directory containing extracted features
        ground_truth_file: JSON file with ground truth boundaries
        model_save_path: Path to save the trained model
        fusion_strategy: Specific fusion strategy to use (overrides config)
        show_progress: Whether to show progress bars
        quiet: Whether to suppress non-essential output
        save_plots: Whether to save training/validation plots
        
    Returns:
        Dictionary with training results (best F1-score, final metrics)
    """
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
    
    # Create dataset with fusion strategy support and lazy loading
    dataset = BoundaryDataset(
        features_dir=features_dir,
        ground_truth_file=ground_truth_file,
        sequence_length=config.processing.sequence_length,
        fusion_strategy=config.fusion.strategy,
        quiet=quiet,
        lazy_loading=True  # Enable lazy loading to save memory
    )
    
    # Force garbage collection after dataset creation
    gc.collect()
    
    if not quiet:
        print(f"Created dataset with {len(dataset)} samples using {config.fusion.strategy} fusion")
        print(f"  Memory optimization: Lazy loading enabled")
        print_memory_usage("After dataset creation: ")
    
    # Check if dataset is empty
    if len(dataset) == 0:
        error_msg = f"ERROR: Dataset is empty! No training samples were created."
        print(error_msg)
        print(f"  - Features directory: {features_dir}")
        print(f"  - Ground truth file: {ground_truth_file}")
        print(f"  - Fusion strategy: {config.fusion.strategy}")
        print(f"  - Sequence length: {config.processing.sequence_length}")
        raise RuntimeError(error_msg)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders with reduced workers to save memory
    # Each worker duplicates the dataset, so fewer workers = less memory
    num_workers = 0 if len(dataset) > 50000 else 2  # Use 0 workers for large datasets
    
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
    
    # Determine input dimension based on fusion strategy
    # Each fusion strategy outputs different dimensions based on how they combine features
    # Actual feature dimensions: CNN=2048, HSV=768, CLIP=512, Flow=40
    fusion_input_dims = {
        "concatenation": 3368,    # HSV (768) + CNN (2048) + CLIP (512) + Flow (40) = 3368
        "weighted": 3368,         # Same as concatenation before weighted combination
        "pca": 256,              # PCA reduces to 256 components (configured at line 224)
        "cross_transformer": 512, # Transformer fusion outputs 512
        "hierarchical": 512       # Hierarchical fusion progressively reduces to 512 (verified)
    }
    
    input_dim = fusion_input_dims.get(config.fusion.strategy, 512)
    
    if not quiet:
        print(f"Using input_dim={input_dim} for {config.fusion.strategy} fusion strategy")
    
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
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
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
        
        # Learning rate scheduling
        trainer.scheduler.step()
        
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
    
    if not quiet:
        print(f"Training completed for {config.fusion.strategy}. Best F1-score: {best_f1_score:.4f}")
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
                                   save_plots: bool = True) -> Dict[str, Dict[str, float]]:
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
        
    Returns:
        Dictionary mapping fusion strategy names to their training results
    """
    # Default fusion strategies if none specified
    if fusion_strategies is None:
        fusion_strategies = ["concatenation", "weighted", "pca", "cross_transformer", "hierarchical"]
    print(f"Fusion strategies to train: {fusion_strategies}")
    if not quiet:
        print("="*80)
        print(f"TRAINING MULTIPLE TRANSFORMER MODELS")
        print(f"Fusion strategies: {', '.join(fusion_strategies)}")
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
            
        except Exception as e:
            error_msg = f"\n✗ Failed to train {strategy} model: {e}"
            if show_progress:
                tqdm.write(error_msg)
            elif not quiet:
                print(error_msg)
            results[strategy] = {'error': str(e), 'best_f1': 0.0}
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
    parser.add_argument("--fusion_strategy", default="hierarchical", help="Single fusion strategy to use")
    
    # New arguments for multi-strategy training
    parser.add_argument("--train_all_strategies", action="store_true", 
                       help="Train models for all available fusion strategies")
    parser.add_argument("--fusion_strategies", nargs="+",
                       default=[ "pca", "cross_transformer", "hierarchical"], #"concatenation","weighted",
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
    
    if args.train_all_strategies:
        # Train multiple strategies
        if not quiet_mode:
            print(f"Training models for strategies: {args.fusion_strategies}")
        results = train_multiple_fusion_strategies(
            config=config,
            features_dir=args.features_dir,
            ground_truth_file=args.ground_truth,
            model_base_path=args.model_path,
            fusion_strategies=args.fusion_strategies,
            show_progress=show_progress,
            quiet=quiet_mode
        )
        
        # Save results
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
        # Train single strategy
        if not quiet_mode:
            print(f"Training model for {args.fusion_strategy} strategy...")
        results = train_transformer_model(
            config=config,
            features_dir=args.features_dir,
            ground_truth_file=args.ground_truth,
            model_save_path=args.model_path,
            fusion_strategy=args.fusion_strategy,
            show_progress=show_progress,
            quiet=quiet_mode
        )
        
        if not quiet_mode:
            print(f"\n✓ Training completed!")
            print(f"Model saved to: {args.model_path}")
            print(f"Best F1-score: {results['best_f1']:.4f}")


if __name__ == "__main__":
    main()