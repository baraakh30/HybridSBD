"""
Script to generate fusion features for videos that have all 4 individual feature types.

This script scans a features directory, identifies videos with complete individual features
(hsv_histogram, cnn, clip, optical_flow), and generates all fusion types for them.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add the parent directory to the Python path to import our modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Required individual feature types
INDIVIDUAL_FEATURE_TYPES = ['hsv_histogram', 'cnn', 'clip', 'optical_flow']

# All fusion strategies to generate
ALL_FUSION_STRATEGIES = ['concatenation', 'weighted', 'pca', 'cross_transformer', 'hierarchical']


def extract_video_name_from_feature_file(filename: str) -> Tuple[str, str]:
    """
    Extract video name and feature type from a feature filename.
    
    Args:
        filename: Feature filename (e.g., 'video123_clip.npy')
        
    Returns:
        Tuple of (video_name, feature_type) or (None, None) if invalid
    """
    if not filename.endswith('.npy'):
        return None, None
    
    # Remove .npy extension
    base = filename[:-4]
    
    # Check for each feature type suffix
    all_feature_types = INDIVIDUAL_FEATURE_TYPES + ALL_FUSION_STRATEGIES + ['fused']
    
    for feat_type in all_feature_types:
        suffix = f'_{feat_type}'
        if base.endswith(suffix):
            video_name = base[:-len(suffix)]
            return video_name, feat_type
    
    return None, None


def scan_features_directory(features_dir: str) -> Dict[str, Dict[str, bool]]:
    """
    Scan features directory and build a mapping of videos to their available features.
    
    Args:
        features_dir: Path to features directory
        
    Returns:
        Dictionary mapping video names to dict of {feature_type: exists}
    """
    video_features = defaultdict(lambda: {ft: False for ft in INDIVIDUAL_FEATURE_TYPES + ALL_FUSION_STRATEGIES})
    
    # List all files in the directory
    if not os.path.exists(features_dir):
        print(f"Error: Features directory does not exist: {features_dir}")
        return {}
    
    files = os.listdir(features_dir)
    
    for filename in files:
        video_name, feature_type = extract_video_name_from_feature_file(filename)
        if video_name and feature_type:
            if feature_type in INDIVIDUAL_FEATURE_TYPES or feature_type in ALL_FUSION_STRATEGIES:
                video_features[video_name][feature_type] = True
    
    return dict(video_features)


def get_videos_with_all_individual_features(video_features: Dict[str, Dict[str, bool]]) -> List[str]:
    """
    Get list of videos that have all 4 individual feature types.
    
    Args:
        video_features: Mapping of videos to their feature availability
        
    Returns:
        List of video names with complete individual features
    """
    complete_videos = []
    
    for video_name, features in video_features.items():
        has_all = all(features.get(ft, False) for ft in INDIVIDUAL_FEATURE_TYPES)
        if has_all:
            complete_videos.append(video_name)
    
    return sorted(complete_videos)


def get_missing_fusion_strategies(video_features: Dict[str, Dict[str, bool]], video_name: str) -> List[str]:
    """
    Get list of fusion strategies that are missing for a video.
    
    Args:
        video_features: Mapping of videos to their feature availability
        video_name: Name of the video
        
    Returns:
        List of missing fusion strategy names
    """
    features = video_features.get(video_name, {})
    missing = []
    
    for strategy in ALL_FUSION_STRATEGIES:
        if not features.get(strategy, False):
            missing.append(strategy)
    
    return missing


def load_individual_features(features_dir: str, video_name: str) -> Dict[str, np.ndarray]:
    """
    Load individual features for a video.
    
    Args:
        features_dir: Directory containing feature files
        video_name: Name of the video (without extension)
        
    Returns:
        Dictionary mapping feature names to arrays
    """
    features = {}
    for feature_name in INDIVIDUAL_FEATURE_TYPES:
        feature_file = os.path.join(features_dir, f"{video_name}_{feature_name}.npy")
        if os.path.exists(feature_file):
            features[feature_name] = np.load(feature_file)
    return features


def generate_fusion_features(features_dir: str, video_name: str, fusion_strategy: str, 
                            quiet: bool = False) -> bool:
    """
    Generate and save fused features for a video using a specific fusion strategy.
    
    Args:
        features_dir: Directory containing feature files
        video_name: Name of the video
        fusion_strategy: Name of the fusion strategy to use
        quiet: Whether to suppress output messages
        
    Returns:
        True if successful, False otherwise
    """
    from core.config import VideoShotConfig
    from fusion import FusionStrategyRegistry
    
    try:
        # Load individual features
        features = load_individual_features(features_dir, video_name)
        
        if len(features) != len(INDIVIDUAL_FEATURE_TYPES):
            if not quiet:
                print(f"  Skipping {video_name}: missing individual features")
            return False
        
        # Set up feature dimensions
        feature_dims = {name: feat_array.shape[1] for name, feat_array in features.items()}
        
        # Create config for fusion
        config = VideoShotConfig()
        
        # Initialize fusion strategy
        if fusion_strategy == "concatenation":
            fusion = FusionStrategyRegistry.get(
                "concatenation",
                feature_dims=feature_dims,
                device="cpu"
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
                n_components=256,
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
            if not quiet:
                print(f"  Unknown fusion strategy: {fusion_strategy}")
            return False
        
        # Initialize and fuse features
        fusion.initialize()
        fused_features = fusion.fuse_features(features)
        
        # Save fused features
        output_file = os.path.join(features_dir, f"{video_name}_{fusion_strategy}.npy")
        np.save(output_file, fused_features)
        
        return True
        
    except ValueError as e:
        if "n_components" in str(e) or "insufficient" in str(e).lower():
            if not quiet:
                print(f"  Warning: {video_name} - video too short for {fusion_strategy}: {e}")
        else:
            if not quiet:
                print(f"  Error processing {video_name} with {fusion_strategy}: {e}")
        return False
    except Exception as e:
        if not quiet:
            print(f"  Error processing {video_name} with {fusion_strategy}: {e}")
        return False


def _process_video_fusion(args: Tuple[str, str, List[str]]) -> Dict[str, Any]:
    """
    Worker function for parallel processing of a single video's fusion features.
    
    Args:
        args: Tuple of (features_dir, video_name, missing_strategies)
        
    Returns:
        Dictionary with results for this video
    """
    features_dir, video_name, missing_strategies = args
    results = {'video': video_name, 'success': [], 'failed': []}
    
    for strategy in missing_strategies:
        success = generate_fusion_features(features_dir, video_name, strategy, quiet=True)
        if success:
            results['success'].append(strategy)
        else:
            results['failed'].append(strategy)
    
    return results


def generate_all_fusion_features(features_dir: str, fusion_strategies: List[str] = None,
                                 show_progress: bool = True, quiet: bool = False,
                                 dry_run: bool = False, num_workers: int = None) -> Dict[str, Any]:
    """
    Generate all missing fusion features for videos with complete individual features.
    
    Args:
        features_dir: Path to features directory
        fusion_strategies: List of fusion strategies to generate (None = all)
        show_progress: Whether to show progress bars
        quiet: Whether to suppress output messages
        dry_run: If True, only report what would be done without actually generating
        
    Returns:
        Dictionary with summary statistics
    """
    if fusion_strategies is None:
        fusion_strategies = ALL_FUSION_STRATEGIES
    
    # Scan directory
    if not quiet:
        print(f"Scanning features directory: {features_dir}")
    
    video_features = scan_features_directory(features_dir)
    
    if not video_features:
        return {'error': 'No features found in directory'}
    
    # Get videos with all individual features
    complete_videos = get_videos_with_all_individual_features(video_features)
    
    if not quiet:
        print(f"\nFound {len(video_features)} videos total")
        print(f"Videos with all 4 individual features: {len(complete_videos)}")
    
    # Identify videos needing fusion features
    videos_needing_fusion = []
    total_fusions_needed = 0
    
    for video_name in complete_videos:
        missing = get_missing_fusion_strategies(video_features, video_name)
        # Filter to only requested strategies
        missing = [s for s in missing if s in fusion_strategies]
        if missing:
            videos_needing_fusion.append((video_name, missing))
            total_fusions_needed += len(missing)
    
    if not quiet:
        print(f"Videos needing fusion features: {len(videos_needing_fusion)}")
        print(f"Total fusion operations needed: {total_fusions_needed}")
        print(f"Fusion strategies to generate: {', '.join(fusion_strategies)}")
    
    if dry_run:
        if not quiet:
            print("\n[DRY RUN] Would generate fusion features for:")
            for video_name, missing in videos_needing_fusion[:10]:
                print(f"  {video_name}: {', '.join(missing)}")
            if len(videos_needing_fusion) > 10:
                print(f"  ... and {len(videos_needing_fusion) - 10} more videos")
        
        return {
            'total_videos': len(video_features),
            'complete_videos': len(complete_videos),
            'videos_needing_fusion': len(videos_needing_fusion),
            'total_fusions_needed': total_fusions_needed,
            'dry_run': True
        }
    
    if not videos_needing_fusion:
        if not quiet:
            print("\nAll videos already have all fusion features!")
        return {
            'total_videos': len(video_features),
            'complete_videos': len(complete_videos),
            'videos_needing_fusion': 0,
            'total_fusions_needed': 0,
            'generated': 0,
            'failed': 0
        }
    
    # Generate fusion features
    if not quiet:
        print(f"\nGenerating fusion features...")
    
    success_count = 0
    fail_count = 0
    results_by_strategy = {s: {'success': 0, 'fail': 0} for s in fusion_strategies}
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    if not quiet:
        print(f"Using {num_workers} parallel workers")
    
    # Prepare work items
    work_items = [(features_dir, video_name, missing) for video_name, missing in videos_needing_fusion]
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_process_video_fusion, item): item[1] for item in work_items}
        
        # Progress bar for completed tasks
        with tqdm(total=len(futures), desc="Processing videos", disable=not show_progress) as pbar:
            for future in as_completed(futures):
                video_name = futures[future]
                try:
                    result = future.result()
                    
                    # Update counts
                    for strategy in result['success']:
                        success_count += 1
                        results_by_strategy[strategy]['success'] += 1
                    
                    for strategy in result['failed']:
                        fail_count += 1
                        results_by_strategy[strategy]['fail'] += 1
                        
                except Exception as e:
                    if not quiet:
                        print(f"\nError processing {video_name}: {e}")
                    fail_count += 1
                
                pbar.update(1)
                if show_progress:
                    pbar.set_postfix({'success': success_count, 'failed': fail_count})
    
    # Print summary
    if not quiet:
        print(f"\n{'='*60}")
        print("FUSION GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total fusion operations: {success_count + fail_count}")
        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"\nResults by strategy:")
        for strategy, counts in results_by_strategy.items():
            total = counts['success'] + counts['fail']
            if total > 0:
                print(f"  {strategy}: {counts['success']}/{total} successful")
        print(f"{'='*60}")
    
    return {
        'total_videos': len(video_features),
        'complete_videos': len(complete_videos),
        'videos_needing_fusion': len(videos_needing_fusion),
        'total_fusions_needed': total_fusions_needed,
        'generated': success_count,
        'failed': fail_count,
        'by_strategy': results_by_strategy
    }


def print_status_report(features_dir: str) -> None:
    """
    Print a detailed status report of features in the directory.
    
    Args:
        features_dir: Path to features directory
    """
    print(f"\n{'='*70}")
    print(f"FEATURES STATUS REPORT")
    print(f"Directory: {features_dir}")
    print(f"{'='*70}")
    
    video_features = scan_features_directory(features_dir)
    
    if not video_features:
        print("No features found!")
        return
    
    # Count videos by feature completeness
    complete_individual = 0
    partial_individual = 0
    
    # Count by fusion strategy availability
    fusion_counts = {s: 0 for s in ALL_FUSION_STRATEGIES}
    
    for video_name, features in video_features.items():
        # Check individual features
        individual_count = sum(1 for ft in INDIVIDUAL_FEATURE_TYPES if features.get(ft, False))
        if individual_count == len(INDIVIDUAL_FEATURE_TYPES):
            complete_individual += 1
        elif individual_count > 0:
            partial_individual += 1
        
        # Check fusion features
        for strategy in ALL_FUSION_STRATEGIES:
            if features.get(strategy, False):
                fusion_counts[strategy] += 1
    
    print(f"\nTotal videos found: {len(video_features)}")
    print(f"\nIndividual Features Status:")
    print(f"  Complete (all 4 types): {complete_individual}")
    print(f"  Partial (some types):   {partial_individual}")
    print(f"  Missing all:            {len(video_features) - complete_individual - partial_individual}")
    
    print(f"\nFusion Features Status (of {complete_individual} videos with complete individual features):")
    for strategy in ALL_FUSION_STRATEGIES:
        have = fusion_counts[strategy]
        need = complete_individual - have
        pct = (have / complete_individual * 100) if complete_individual > 0 else 0
        print(f"  {strategy:20s}: {have:4d} have, {need:4d} need ({pct:.1f}% complete)")
    
    print(f"\n{'='*70}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate fusion features for videos with complete individual features"
    )
    
    parser.add_argument(
        "--features_dir", 
        default="testing_features",
        help="Directory containing feature files (default: testing_features)"
    )
    
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=ALL_FUSION_STRATEGIES,
        default=None,
        help=f"Fusion strategies to generate (default: all). Choices: {', '.join(ALL_FUSION_STRATEGIES)}"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print status report and exit (don't generate anything)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without actually generating features"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    
    args = parser.parse_args()
    
    # Resolve features directory path
    features_dir = args.features_dir
    if not os.path.isabs(features_dir):
        features_dir = os.path.join(current_dir, features_dir)
    
    if args.status:
        print_status_report(features_dir)
        return
    
    # Generate fusion features
    results = generate_all_fusion_features(
        features_dir=features_dir,
        fusion_strategies=args.strategies,
        show_progress=not args.no_progress,
        quiet=args.quiet,
        dry_run=args.dry_run,
        num_workers=args.workers
    )
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
