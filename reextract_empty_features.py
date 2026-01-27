"""
Script to re-extract features for videos that have empty or corrupted feature files.

This script:
1. Scans the training_features directory for empty/corrupted .npy files
2. Identifies which individual features (clip, cnn, hsv_histogram, optical_flow) need re-extraction
3. Re-extracts the missing individual features from source videos
4. Regenerates fusion features (pca, weighted, concatenation, cross_transformer, hierarchical)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from tqdm import tqdm
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add the parent directory to the Python path to import our modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import OpenCV config first
import opencv_config

from core.config import VideoShotConfig
from core.coordinator import ProcessingCoordinator
from fusion import FusionStrategyRegistry

# Feature types
INDIVIDUAL_FEATURE_TYPES = ['hsv_histogram', 'cnn', 'clip', 'optical_flow']
FUSION_STRATEGIES = ['concatenation', 'weighted', 'pca', 'cross_transformer', 'hierarchical']

# Expected feature dimensions
EXPECTED_DIMS = {
    'hsv_histogram': 768,
    'cnn': 2048,
    'clip': 512,
    'optical_flow': 40
}

# Minimum file size for valid numpy array (128 bytes = empty array header)
EMPTY_ARRAY_SIZE = 128

# Maximum frames to load per video (to avoid memory issues)
MAX_FRAMES_LIMIT = 150000


def find_empty_feature_files(features_dir: str, check_missing: bool = True) -> Dict[str, List[str]]:
    """
    Find all empty, corrupted, or missing feature files in the directory.
    
    Args:
        features_dir: Path to features directory
        check_missing: Also check for missing feature files (not just empty)
        
    Returns:
        Dictionary mapping video names to list of problematic feature types
    """
    empty_features = defaultdict(list)
    all_videos = set()
    existing_features = defaultdict(set)  # video_name -> set of existing feature types
    
    print(f"Scanning directory: {features_dir}")
    
    if not os.path.exists(features_dir):
        print(f"Error: Directory does not exist: {features_dir}")
        return {}
    
    # First pass: find all videos and their existing features
    for filename in os.listdir(features_dir):
        if not filename.endswith('.npy'):
            continue
        
        filepath = os.path.join(features_dir, filename)
        filesize = os.path.getsize(filepath)
        base = filename[:-4]  # Remove .npy
        
        # Check for each feature type
        all_types = INDIVIDUAL_FEATURE_TYPES + FUSION_STRATEGIES
        for feat_type in all_types:
            suffix = f'_{feat_type}'
            if base.endswith(suffix):
                video_name = base[:-len(suffix)]
                all_videos.add(video_name)
                
                # Check if file is empty (128 bytes = numpy empty array header)
                if filesize <= EMPTY_ARRAY_SIZE:
                    empty_features[video_name].append(feat_type)
                else:
                    existing_features[video_name].add(feat_type)
                break
    
    # Second pass: check for missing individual features
    if check_missing:
        for video_name in all_videos:
            for feat_type in INDIVIDUAL_FEATURE_TYPES:
                if feat_type not in existing_features[video_name] and feat_type not in empty_features[video_name]:
                    empty_features[video_name].append(feat_type)
    
    return dict(empty_features)


def validate_feature_file(filepath: str, feature_type: str) -> Tuple[bool, str]:
    """
    Validate a feature file.
    
    Args:
        filepath: Path to the feature file
        feature_type: Type of feature (e.g., 'clip', 'pca')
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(filepath):
        return False, "File does not exist"
    
    try:
        filesize = os.path.getsize(filepath)
        if filesize <= EMPTY_ARRAY_SIZE:
            return False, f"File is empty ({filesize} bytes)"
        
        # Try to load the array
        arr = np.load(filepath)
        
        if arr.size == 0:
            return False, "Array is empty"
        
        if len(arr.shape) < 2:
            return False, f"Invalid shape: {arr.shape}"
        
        # Check expected dimension for individual features
        if feature_type in EXPECTED_DIMS:
            expected_dim = EXPECTED_DIMS[feature_type]
            if arr.shape[1] != expected_dim:
                return False, f"Wrong dimension: expected {expected_dim}, got {arr.shape[1]}"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Load error: {str(e)}"


def find_video_source(video_name: str, video_dirs: List[str]) -> str:
    """
    Find the source video file for a given video name.
    
    Args:
        video_name: Name of the video (without extension)
        video_dirs: List of directories to search
        
    Returns:
        Full path to video file, or None if not found
    """
    # Common video extensions
    extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
    
    for video_dir in video_dirs:
        if not os.path.exists(video_dir):
            continue
        
        for ext in extensions:
            # Try direct match
            video_path = os.path.join(video_dir, f"{video_name}{ext}")
            if os.path.exists(video_path):
                return video_path
            
            # Try with sanitized name
            video_path = os.path.join(video_dir, f"{video_name.replace('___', '/')}{ext}")
            if os.path.exists(video_path):
                return video_path
    
    return None


def extract_single_feature(feat_type: str, frames: np.ndarray, video_name: str, 
                           features_dir: str, config: Any, quiet: bool = False) -> Tuple[str, bool, str]:
    """
    Extract a single feature type from frames.
    
    Args:
        feat_type: Feature type to extract
        frames: Video frames
        video_name: Name of the video
        features_dir: Directory to save features
        config: Configuration object
        quiet: Suppress output
        
    Returns:
        Tuple of (feature_type, success, message)
    """
    from extractors import FeatureExtractorRegistry
    
    try:
        # Initialize extractor based on type
        if feat_type == 'hsv_histogram':
            extractor = FeatureExtractorRegistry.get(
                "hsv_histogram",
                bins=config.features.hsv_bins,
                smoothing=config.features.hsv_smoothing,
                device=config.processing.device
            )
        elif feat_type == 'cnn':
            extractor = FeatureExtractorRegistry.get(
                "cnn",
                model_name=config.features.cnn_model,
                pretrained=config.features.cnn_pretrained,
                device=config.processing.device,
                frame_sampling_rate=config.features.cnn_frame_sampling_rate
            )
        elif feat_type == 'clip':
            extractor = FeatureExtractorRegistry.get(
                "clip",
                model_name=config.features.clip_model,
                device=config.processing.device,
                frame_sampling_rate=config.features.clip_frame_sampling_rate
            )
        elif feat_type == 'optical_flow':
            extractor = FeatureExtractorRegistry.get(
                "optical_flow",
                method=config.features.optical_flow_method,
                device=config.processing.device,
                frame_sampling_rate=config.features.optical_flow_frame_sampling_rate
            )
        else:
            return feat_type, False, f"Unknown feature type: {feat_type}"
        
        # Extract features
        extractor.initialize()
        features = extractor.extract_batch(frames, batch_size=config.processing.batch_size)
        
        # Save features
        output_path = os.path.join(features_dir, f"{video_name}_{feat_type}.npy")
        np.save(output_path, features)
        
        return feat_type, True, f"Extracted {feat_type}: {features.shape}"
        
    except Exception as e:
        return feat_type, False, f"Error extracting {feat_type}: {e}"


def extract_individual_features(video_path: str, video_name: str, features_dir: str,
                                 feature_types: List[str], config: Any,
                                 quiet: bool = False, num_workers: int = 4,
                                 max_frames: int = MAX_FRAMES_LIMIT) -> Dict[str, bool]:
    """
    Extract individual features for a video using parallel workers.
    
    Args:
        video_path: Path to source video
        video_name: Name to use for saving features
        features_dir: Directory to save features
        feature_types: List of feature types to extract
        config: Configuration object
        quiet: Suppress output
        num_workers: Number of parallel workers (one per feature type)
        max_frames: Maximum number of frames to load (default: 150000)
        
    Returns:
        Dictionary mapping feature types to success status
    """
    from utils.video_utils import VideoProcessor
    
    results = {}
    
    try:
        # Load video frames once (with frame limit to avoid memory issues)
        processor = VideoProcessor(config.processing)
        frames = processor.load_video(video_path, max_frames=max_frames)
        
        if frames is None or len(frames) == 0:
            print(f"  Error: Could not load frames from {video_path}")
            return {ft: False for ft in feature_types}
        
        if not quiet:
            print(f"  Loaded {len(frames)} frames (max: {max_frames}), extracting with {min(num_workers, len(feature_types))} workers...")
        
        # Filter to only individual features
        individual_features = [ft for ft in feature_types if ft in INDIVIDUAL_FEATURE_TYPES]
        
        if not individual_features:
            return results
        
        # Use ThreadPoolExecutor to run extractions in parallel
        # (ThreadPool is better here since most work is I/O and GPU bound)
        with ThreadPoolExecutor(max_workers=min(num_workers, len(individual_features))) as executor:
            # Submit all extraction tasks
            future_to_feat = {
                executor.submit(
                    extract_single_feature, 
                    feat_type, frames, video_name, features_dir, config, quiet
                ): feat_type 
                for feat_type in individual_features
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_feat):
                feat_type = future_to_feat[future]
                try:
                    feat_type, success, message = future.result()
                    results[feat_type] = success
                    if not quiet:
                        print(f"    {message}")
                except Exception as e:
                    results[feat_type] = False
                    if not quiet:
                        print(f"    Error in {feat_type} worker: {e}")
        
        return results
        
    except Exception as e:
        print(f"  Error processing video: {e}")
        return {ft: False for ft in feature_types}


def generate_fusion_features(features_dir: str, video_name: str, 
                             fusion_strategies: List[str], quiet: bool = False) -> Dict[str, bool]:
    """
    Generate fusion features for a video.
    
    Args:
        features_dir: Directory containing individual features
        video_name: Name of the video
        fusion_strategies: List of fusion strategies to generate
        quiet: Suppress output
        
    Returns:
        Dictionary mapping strategy names to success status
    """
    results = {}
    
    # Load individual features
    features = {}
    for feat_type in INDIVIDUAL_FEATURE_TYPES:
        feat_path = os.path.join(features_dir, f"{video_name}_{feat_type}.npy")
        if os.path.exists(feat_path):
            try:
                arr = np.load(feat_path)
                if arr.size > 0:
                    features[feat_type] = arr
            except Exception as e:
                if not quiet:
                    print(f"  Error loading {feat_type}: {e}")
    
    if len(features) != len(INDIVIDUAL_FEATURE_TYPES):
        if not quiet:
            print(f"  Missing individual features, cannot generate fusion")
        return {s: False for s in fusion_strategies}
    
    # Get feature dimensions
    feature_dims = {name: arr.shape[1] for name, arr in features.items()}
    
    # Create config
    config = VideoShotConfig()
    
    # Generate each fusion strategy
    for strategy in fusion_strategies:
        if strategy not in FUSION_STRATEGIES:
            continue
        
        try:
            # Initialize fusion strategy
            if strategy == "concatenation":
                fusion = FusionStrategyRegistry.get(
                    "concatenation",
                    feature_dims=feature_dims,
                    device="cpu"
                )
            elif strategy == "weighted":
                fusion = FusionStrategyRegistry.get(
                    "weighted",
                    feature_dims=feature_dims,
                    optimization_method="manual",
                    manual_weights=config.fusion.manual_weights,
                    device="cpu"
                )
            elif strategy == "pca":
                fusion = FusionStrategyRegistry.get(
                    "pca",
                    feature_dims=feature_dims,
                    n_components=256,
                    variance_retained=config.fusion.pca_variance_retained,
                    device="cpu"
                )
            elif strategy == "cross_transformer":
                fusion = FusionStrategyRegistry.get(
                    "cross_transformer",
                    feature_dims=feature_dims,
                    hidden_dim=config.fusion.transformer_hidden_dim,
                    num_heads=config.fusion.transformer_heads,
                    num_layers=config.fusion.transformer_layers,
                    dropout=config.fusion.transformer_dropout,
                    device="cpu"
                )
            elif strategy == "hierarchical":
                fusion = FusionStrategyRegistry.get(
                    "hierarchical",
                    feature_dims=feature_dims,
                    conv_kernel=config.fusion.hierarchical_conv_kernel,
                    attention_heads=config.fusion.hierarchical_attention_heads,
                    device="cpu"
                )
            else:
                results[strategy] = False
                continue
            
            # Initialize and fuse
            fusion.initialize()
            fused = fusion.fuse_features(features)
            
            # Save
            output_path = os.path.join(features_dir, f"{video_name}_{strategy}.npy")
            np.save(output_path, fused)
            
            if not quiet:
                print(f"  Generated {strategy}: {fused.shape}")
            
            results[strategy] = True
            
        except ValueError as e:
            if "n_components" in str(e) or "insufficient" in str(e).lower():
                if not quiet:
                    print(f"  Warning: Video too short for {strategy}")
            else:
                if not quiet:
                    print(f"  Error generating {strategy}: {e}")
            results[strategy] = False
        except Exception as e:
            if not quiet:
                print(f"  Error generating {strategy}: {e}")
            results[strategy] = False
    
    return results


def categorize_empty_features(empty_features: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Categorize empty features into individual and fusion types.
    
    Args:
        empty_features: Dictionary mapping video names to empty feature types
        
    Returns:
        Dictionary with 'individual' and 'fusion' keys, each mapping to video->features
    """
    categorized = {
        'individual': defaultdict(list),
        'fusion': defaultdict(list)
    }
    
    for video_name, feat_types in empty_features.items():
        for feat_type in feat_types:
            if feat_type in INDIVIDUAL_FEATURE_TYPES:
                categorized['individual'][video_name].append(feat_type)
            elif feat_type in FUSION_STRATEGIES:
                categorized['fusion'][video_name].append(feat_type)
    
    return {
        'individual': dict(categorized['individual']),
        'fusion': dict(categorized['fusion'])
    }


def main():
    parser = argparse.ArgumentParser(description='Re-extract empty or corrupted features')
    parser.add_argument('--features-dir', type=str, default='training_features',
                        help='Directory containing feature files')
    parser.add_argument('--video-dirs', type=str, nargs='+', 
                        default=[r'C:\Users\baraa\Documents\ClipShots\videos\train',
                                 r'C:\Users\baraa\Documents\ClipShots\videos',
                                 'data/videos', 'data', 'videos'],
                        help='Directories to search for source videos')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only report what would be done')
    parser.add_argument('--fusion-only', action='store_true',
                        help='Only regenerate fusion features (skip individual feature extraction)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for extraction (cuda/cpu)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--skip-videos', type=str, nargs='*', default=[],
                        help='Video names to skip')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for feature extraction (default: 4, one per feature type)')
    parser.add_argument('--extract-all', action='store_true', default=True,
                        help='Extract ALL individual features when processing a video (default: True)')
    parser.add_argument('--max-frames', type=int, default=150000,
                        help='Maximum frames to load per video (default: 150000)')
    
    args = parser.parse_args()
    
    features_dir = args.features_dir
    if not os.path.isabs(features_dir):
        features_dir = os.path.join(current_dir, features_dir)
    
    # Find empty feature files
    print("=" * 70)
    print("STEP 1: Scanning for empty/corrupted feature files")
    print("=" * 70)
    
    empty_features = find_empty_feature_files(features_dir)
    
    if not empty_features:
        print("\nNo empty feature files found!")
        return
    
    # Categorize
    categorized = categorize_empty_features(empty_features)
    
    print(f"\nFound {len(empty_features)} videos with empty features:")
    print(f"  - Videos with empty individual features: {len(categorized['individual'])}")
    print(f"  - Videos with empty fusion features: {len(categorized['fusion'])}")
    
    # Count total files
    total_individual = sum(len(v) for v in categorized['individual'].values())
    total_fusion = sum(len(v) for v in categorized['fusion'].values())
    print(f"\nTotal empty files:")
    print(f"  - Individual feature files: {total_individual}")
    print(f"  - Fusion feature files: {total_fusion}")
    
    # Show details
    print("\n" + "-" * 70)
    print("Videos with empty INDIVIDUAL features (need video re-extraction):")
    print("-" * 70)
    for video_name, feats in sorted(categorized['individual'].items()):
        print(f"  {video_name}: {', '.join(feats)}")
    
    print("\n" + "-" * 70)
    print("Videos with empty FUSION features (can be regenerated from individual):")
    print("-" * 70)
    for video_name, feats in sorted(categorized['fusion'].items()):
        print(f"  {video_name}: {', '.join(feats)}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would process the above files")
        return
    
    # Setup configuration
    config = VideoShotConfig()
    config.processing.device = args.device
    
    # STEP 2: Regenerate fusion features (for videos that have all individual features)
    print("\n" + "=" * 70)
    print("STEP 2: Regenerating fusion features")
    print("=" * 70)
    
    fusion_success = 0
    fusion_failed = 0
    
    for video_name, missing_fusions in tqdm(categorized['fusion'].items(), desc="Regenerating fusion"):
        if video_name in args.skip_videos:
            continue
        
        # Check if we have all individual features
        has_all_individual = True
        for feat_type in INDIVIDUAL_FEATURE_TYPES:
            feat_path = os.path.join(features_dir, f"{video_name}_{feat_type}.npy")
            is_valid, _ = validate_feature_file(feat_path, feat_type)
            if not is_valid:
                has_all_individual = False
                break
        
        if not has_all_individual:
            if not args.quiet:
                print(f"\n  Skipping {video_name}: missing valid individual features")
            # Add to individual extraction list if not already there
            if video_name not in categorized['individual']:
                categorized['individual'][video_name] = []
            continue
        
        if not args.quiet:
            print(f"\n  Processing {video_name}")
        
        results = generate_fusion_features(features_dir, video_name, missing_fusions, args.quiet)
        
        for strategy, success in results.items():
            if success:
                fusion_success += 1
            else:
                fusion_failed += 1
    
    print(f"\nFusion regeneration complete:")
    print(f"  - Successful: {fusion_success}")
    print(f"  - Failed: {fusion_failed}")
    
    if args.fusion_only:
        print("\n[--fusion-only] Skipping individual feature extraction")
        return
    
    # STEP 3: Re-extract individual features (requires source videos)
    if categorized['individual']:
        print("\n" + "=" * 70)
        print("STEP 3: Re-extracting individual features")
        print("=" * 70)
        
        # Expand video dirs to absolute paths
        video_dirs = []
        for vd in args.video_dirs:
            if not os.path.isabs(vd):
                vd = os.path.join(current_dir, vd)
            video_dirs.append(vd)
        
        print(f"Searching for source videos in: {video_dirs}")
        
        individual_success = 0
        individual_failed = 0
        videos_not_found = []
        
        for video_name, missing_features in tqdm(categorized['individual'].items(), 
                                                  desc="Extracting individual"):
            if video_name in args.skip_videos:
                continue
            
            # Find source video
            video_path = find_video_source(video_name, video_dirs)
            
            if video_path is None:
                videos_not_found.append(video_name)
                if not args.quiet:
                    print(f"\n  Video not found: {video_name}")
                continue
            
            if not args.quiet:
                print(f"\n  Processing {video_name} from {video_path}")
            
            # Extract ALL individual features when processing a video (ensures complete set for fusion)
            features_to_extract = INDIVIDUAL_FEATURE_TYPES if args.extract_all else missing_features
            
            results = extract_individual_features(
                video_path, video_name, features_dir, 
                features_to_extract, config, args.quiet, args.workers, args.max_frames
            )
            
            for feat_type, success in results.items():
                if success:
                    individual_success += 1
                else:
                    individual_failed += 1
            
            # Also regenerate fusion features for this video
            fusion_results = generate_fusion_features(
                features_dir, video_name, FUSION_STRATEGIES, args.quiet
            )
        
        print(f"\nIndividual feature extraction complete:")
        print(f"  - Successful: {individual_success}")
        print(f"  - Failed: {individual_failed}")
        
        if videos_not_found:
            print(f"\nVideos not found ({len(videos_not_found)}):")
            for vn in videos_not_found[:20]:
                print(f"  - {vn}")
            if len(videos_not_found) > 20:
                print(f"  ... and {len(videos_not_found) - 20} more")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
