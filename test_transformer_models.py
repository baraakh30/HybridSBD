"""
Testing script for trained transformer detection models.

This script tests trained transformer models on the test dataset using
pre-extracted features from testing_features directory.

Usage:
    # Test all available models
    python test_transformer_models.py --test_all
    
    # Test specific fusion strategies
    python test_transformer_models.py --fusion_strategies hierarchical pca concatenation
    
    # Test with custom paths
    python test_transformer_models.py --features_dir testing_features --model_dir models
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from core.config import VideoShotConfig
from detection.transformer_detection import TransformerDetection
from core.coordinator import ProcessingCoordinator

# =============================================================================
# CONSTANTS
# =============================================================================
INDIVIDUAL_FEATURE_TYPES = ['hsv_histogram', 'cnn', 'clip', 'optical_flow']
FUSION_STRATEGIES = ['concatenation', 'weighted', 'pca', 'cross_transformer', 'hierarchical']

# Feature dimensions for each type
FEATURE_DIMS = {
    'hsv_histogram': 768,
    'cnn': 2048,
    'clip': 512,
    'optical_flow': 40,
    'concatenation': 3368,
    'weighted': 3368,
    'pca': 256,
    'cross_transformer': 512,
    'hierarchical': 512
}

# Default paths
DEFAULT_VIDEO_DIR = r"C:\Users\Baraa\Documents\ClipShots\videos\test"
DEFAULT_GROUND_TRUTH = r"C:\Users\Baraa\Documents\ClipShots\annotations\test.json"
DEFAULT_FEATURES_DIR = "testing_features"
DEFAULT_MODEL_DIR = "models"
DEFAULT_RESULTS_DIR = "test_results"


def load_ground_truth(ground_truth_file: str, return_ranges: bool = False) -> Dict[str, List]:
    """
    Load ground truth annotations from JSON file.
    
    Handles ClipShots format: {'video.mp4': {'transitions': [[start, end], ...], 'frame_num': N}}
    
    Args:
        ground_truth_file: Path to ground truth JSON file
        return_ranges: If True, return transition ranges [(start, end), ...]
                      If False (default), return boundary frames [end_frame, ...] for backward compat
        
    Returns:
        Dictionary mapping video names (without .mp4) to list of boundary frames or transition ranges
    """
    with open(ground_truth_file, 'r') as f:
        data = json.load(f)
    
    processed = {}
    
    for video_name, video_data in data.items():
        # Strip .mp4 extension from video name to match feature filenames
        video_name_stem = video_name.replace('.mp4', '') if video_name.endswith('.mp4') else video_name
        
        # Extract boundaries from transitions
        # Handle both old format (list of boundaries) and new format (dict with transitions)
        if isinstance(video_data, dict) and 'transitions' in video_data:
            # ClipShots format: transitions = [[start, end], ...]
            # Filter invalid transitions (start >= end, negative values)
            valid_transitions = []
            for t in video_data['transitions']:
                if len(t) >= 2:
                    start_frame, end_frame = int(t[0]), int(t[1])
                    if start_frame >= 0 and end_frame > start_frame:
                        valid_transitions.append((start_frame, end_frame))
            
            if return_ranges:
                processed[video_name_stem] = valid_transitions
            else:
                # Backward compatibility: return end frames only
                processed[video_name_stem] = [t[1] for t in valid_transitions]
        elif isinstance(video_data, list):
            # Old format: direct list of boundaries
            if return_ranges:
                # Convert single frames to (frame, frame) ranges
                processed[video_name_stem] = [(int(b), int(b)) for b in video_data if int(b) >= 0]
            else:
                processed[video_name_stem] = [int(b) for b in video_data if int(b) >= 0]
        else:
            # Unknown format, skip
            continue
    
    return processed


def get_video_name_from_feature_file(feature_file: str) -> str:
    """
    Extract video name from feature file path.
    
    Args:
        feature_file: Feature file name (e.g., "video_name_clip.npy")
        
    Returns:
        Video name without feature type suffix
    """
    # Remove .npy extension
    name = os.path.splitext(feature_file)[0]
    
    # Remove feature type suffix
    for feature_type in INDIVIDUAL_FEATURE_TYPES + FUSION_STRATEGIES + ['fused', 'features']:
        if name.endswith(f'_{feature_type}'):
            return name[:-len(f'_{feature_type}')]
    
    return name


def get_available_test_videos(features_dir: str) -> List[str]:
    """
    Get list of video names that have all required individual features.
    
    Args:
        features_dir: Directory containing feature files
        
    Returns:
        List of video names with complete features
    """
    feature_files = os.listdir(features_dir)
    video_features = {}
    
    for f in feature_files:
        if not f.endswith('.npy'):
            continue
        
        video_name = get_video_name_from_feature_file(f)
        if video_name not in video_features:
            video_features[video_name] = set()
        
        for feat_type in INDIVIDUAL_FEATURE_TYPES:
            if f.endswith(f'_{feat_type}.npy'):
                video_features[video_name].add(feat_type)
                break
    
    # Return videos with all 4 individual features
    complete_videos = [
        video_name for video_name, features in video_features.items()
        if len(features) == len(INDIVIDUAL_FEATURE_TYPES)
    ]
    
    return sorted(complete_videos)


def load_individual_features(features_dir: str, video_name: str) -> Dict[str, np.ndarray]:
    """
    Load all individual features for a video.
    
    Args:
        features_dir: Directory containing feature files
        video_name: Name of the video
        
    Returns:
        Dictionary mapping feature type to numpy array
    """
    features = {}
    for feat_type in INDIVIDUAL_FEATURE_TYPES:
        feat_file = os.path.join(features_dir, f"{video_name}_{feat_type}.npy")
        if os.path.exists(feat_file):
            features[feat_type] = np.load(feat_file)
    return features


def create_fused_features(individual_features: Dict[str, np.ndarray], 
                          fusion_strategy: str,
                          features_dir: str = None,
                          video_name: str = None) -> np.ndarray:
    """
    Create fused features from individual features using specified strategy.
    
    Args:
        individual_features: Dictionary of individual feature arrays
        fusion_strategy: Name of fusion strategy
        features_dir: Optional directory to check for pre-computed fused features
        video_name: Optional video name for loading pre-computed features
        
    Returns:
        Fused feature array
    """
    # First, check if pre-computed fused features exist
    if features_dir and video_name:
        fused_file = os.path.join(features_dir, f"{video_name}_{fusion_strategy}.npy")
        if os.path.exists(fused_file):
            return np.load(fused_file)
    
    # Create minimal config for fusion
    from core.config import VideoShotConfig
    config = VideoShotConfig()
    config.fusion.strategy = fusion_strategy
    
    # Set up feature dimensions
    feature_dims = {name: arr.shape[1] for name, arr in individual_features.items()}
    
    # Initialize fusion strategy
    from fusion import FusionStrategyRegistry
    
    if fusion_strategy == "concatenation":
        from fusion.concatenation_fusion import ConcatenationFusion
        fusion = ConcatenationFusion(config.fusion, feature_dims)
    elif fusion_strategy == "weighted":
        from fusion.weighted_fusion import WeightedFusion
        fusion = WeightedFusion(config.fusion, feature_dims)
    elif fusion_strategy == "pca":
        from fusion.pca_fusion import PCAFusion
        fusion = PCAFusion(config.fusion, feature_dims)
    elif fusion_strategy == "cross_transformer":
        from fusion.cross_transformer_fusion import CrossTransformerFusion
        fusion = CrossTransformerFusion(config.fusion, feature_dims)
    elif fusion_strategy == "hierarchical":
        from fusion.hierarchical_fusion import HierarchicalFusion
        fusion = HierarchicalFusion(config.fusion, feature_dims)
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    # Initialize and fuse
    fusion.initialize()
    
    try:
        fused = fusion.fuse(individual_features)
    except ValueError as e:
        # Handle frame count mismatch by truncating to minimum
        min_frames = min(arr.shape[0] for arr in individual_features.values())
        truncated = {k: v[:min_frames] for k, v in individual_features.items()}
        fused = fusion.fuse(truncated)
    
    return fused


def load_model(model_path: str, feature_type: str, config: VideoShotConfig) -> TransformerDetection:
    """
    Load a trained transformer model.
    
    Args:
        model_path: Path to the model checkpoint
        feature_type: Feature type or fusion strategy (for input dimension)
        config: System configuration
        
    Returns:
        Initialized TransformerDetection instance
    """
    # Load checkpoint first to extract saved config
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model config from checkpoint if available
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        # saved_config is the model config dict used during training
        if isinstance(saved_config, dict):
            model_config = saved_config.copy()
        else:
            # It might be a config object, extract as dict
            model_config = dict(saved_config) if hasattr(saved_config, '__iter__') else {}
    else:
        model_config = {}
    
    # Ensure input_dim is set correctly for this feature type
    input_dim = FEATURE_DIMS.get(feature_type, 512)
    model_config['input_dim'] = input_dim
    
    # Set defaults for any missing keys (matching training defaults from TransformerDetection)
    model_config.setdefault('hidden_dim', 512)
    model_config.setdefault('num_layers', 4)  # Training uses 4 layers
    model_config.setdefault('num_heads', 8)
    model_config.setdefault('ff_dim', 1024)   # Training uses 1024
    model_config.setdefault('dropout', 0.15)
    model_config.setdefault('sequence_length', 100)
    model_config.setdefault('confidence_threshold', 0.5)
    
    # Initialize detection method with the saved config
    detection = TransformerDetection(model_config)
    detection.initialize()
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        detection.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        detection.model.load_state_dict(checkpoint)
    
    detection.model.eval()
    
    return detection


def get_available_models(model_dir: str) -> Dict[str, str]:
    """
    Get dictionary of available trained models.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Dictionary mapping fusion strategy to model path
    """
    available = {}
    
    for f in os.listdir(model_dir):
        if not f.endswith('.pth'):
            continue
        
        # Parse model name to get strategy
        # Expected format: transformer_detection_<strategy>.pth
        if f.startswith('transformer_detection_'):
            strategy = f.replace('transformer_detection_', '').replace('.pth', '')
            available[strategy] = os.path.join(model_dir, f)
    
    return available


def calculate_metrics(detected: List[int], ground_truth, 
                     tolerance: int = 3) -> Dict[str, float]:
    """
    Calculate detection metrics with tolerance using standard SBD protocol.
    
    CRITICAL FIX: Supports transition RANGES [(start, end), ...] not just end frames.
    A prediction is a TP if it falls within ANY transition range (with tolerance).
    This is the ClipShots evaluation protocol for gradual transitions.
    
    Standard SBD evaluation protocol:
    - Each GT transition can match at most ONE prediction
    - Each prediction can match at most ONE GT transition
    - A prediction matches if it's within [start-tolerance, end+tolerance] of a transition
    
    Args:
        detected: List of detected boundary frames
        ground_truth: List of ground truth - can be either:
                     - List[int]: boundary frames (legacy)
                     - List[Tuple[int, int]]: transition ranges (start, end)
        tolerance: Tolerance window for matching (frames) - default 3
        
    Returns:
        Dictionary with precision, recall, f1_score
    """
    # Convert ground_truth to ranges if not already
    if not ground_truth:
        return {
            'precision': 1.0 if not detected else 0.0,
            'recall': 1.0,
            'f1_score': 1.0 if not detected else 0.0,
            'true_positives': 0,
            'false_positives': len(detected),
            'false_negatives': 0
        }
    
    # Check if ground_truth contains ranges or single frames
    gt_ranges = []
    for gt in ground_truth:
        if isinstance(gt, (list, tuple)) and len(gt) >= 2:
            gt_ranges.append((int(gt[0]), int(gt[1])))
        else:
            gt_ranges.append((int(gt), int(gt)))  # Single frame = instant cut
    
    if not detected:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(gt_ranges)
        }
    
    # ONE-TO-ONE MATCHING with OVERLAP detection
    # Each GT transition matches at most one prediction, each prediction matches at most one GT
    matched_pred = set()
    matched_gt = set()
    
    # Sort GT ranges by start position for consistent matching
    sorted_gt_indices = sorted(range(len(gt_ranges)), key=lambda i: gt_ranges[i][0])
    
    for gt_i in sorted_gt_indices:
        gt_start, gt_end = gt_ranges[gt_i]
        
        # Find best matching prediction
        # A prediction matches if it's within [start-tolerance, end+tolerance]
        best_pred = None
        best_score = float('inf')  # Lower is better (prefer closer to center)
        
        for pred_idx in detected:
            if pred_idx in matched_pred:
                continue
            
            # Check if prediction overlaps with transition range (with tolerance)
            # Overlap = prediction is within extended range [start-tolerance, end+tolerance]
            if gt_start - tolerance <= pred_idx <= gt_end + tolerance:
                # Score by distance to center of transition (prefer predictions near center)
                center = (gt_start + gt_end) / 2
                score = abs(pred_idx - center)
                if score < best_score:
                    best_score = score
                    best_pred = pred_idx
        
        if best_pred is not None:
            matched_pred.add(best_pred)
            matched_gt.add(gt_i)
    
    # Standard SBD metrics
    true_positives = len(matched_pred)
    false_positives = len(detected) - true_positives
    false_negatives = len(gt_ranges) - true_positives
    
    precision = true_positives / len(detected) if detected else 0.0
    recall = true_positives / len(gt_ranges) if gt_ranges else 0.0
    
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def test_single_model(model_path: str, 
                      fusion_strategy: str,
                      features_dir: str,
                      ground_truth: Dict[str, List[int]],
                      video_names: List[str],
                      config: VideoShotConfig,
                      tolerance: int = 2,
                      quiet: bool = False) -> Dict[str, Any]:
    """
    Test a single model on the test dataset.
    
    Args:
        model_path: Path to model checkpoint
        fusion_strategy: Fusion strategy name
        features_dir: Directory containing test features
        ground_truth: Ground truth boundaries
        video_names: List of video names to test
        config: System configuration
        tolerance: Tolerance for boundary matching
        quiet: Suppress output
        
    Returns:
        Dictionary with test results
    """
    # Load model
    if not quiet:
        print(f"\nLoading model: {model_path}")
    
    detection = load_model(model_path, fusion_strategy, config)
    
    # Test on each video
    all_metrics = []
    per_video_results = {}
    total_detected = 0
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Filter to videos that exist in ground truth
    test_videos = [v for v in video_names if v in ground_truth]
    
    if not quiet:
        print(f"Testing on {len(test_videos)} videos with ground truth...")
    
    video_iterator = tqdm(test_videos, desc=f"Testing {fusion_strategy}", disable=quiet)
    
    for video_name in video_iterator:
        try:
            # Load individual features
            individual_features = load_individual_features(features_dir, video_name)
            
            if len(individual_features) < len(INDIVIDUAL_FEATURE_TYPES):
                if not quiet:
                    print(f"  Skipping {video_name}: missing features")
                continue
            
            # Create fused features
            fused_features = create_fused_features(
                individual_features, 
                fusion_strategy,
                features_dir,
                video_name
            )
            
            # Detect boundaries
            detected = detection.detect_boundaries(fused_features)
            
            # Get ground truth
            gt_boundaries = ground_truth.get(video_name, [])
            
            # Calculate metrics
            metrics = calculate_metrics(detected, gt_boundaries, tolerance)
            
            all_metrics.append(metrics)
            per_video_results[video_name] = {
                'detected': detected,
                'ground_truth': gt_boundaries,
                'num_frames': len(fused_features),
                **metrics
            }
            
            total_detected += len(detected)
            total_gt += len(gt_boundaries)
            total_tp += metrics['true_positives']
            total_fp += metrics['false_positives']
            total_fn += metrics['false_negatives']
            
        except Exception as e:
            if not quiet:
                print(f"  Error processing {video_name}: {e}")
            continue
    
    # Calculate aggregate metrics
    if all_metrics:
        # Micro-averaged metrics (based on total counts)
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Macro-averaged metrics (average of per-video metrics)
        macro_precision = np.mean([m['precision'] for m in all_metrics])
        macro_recall = np.mean([m['recall'] for m in all_metrics])
        macro_f1 = np.mean([m['f1_score'] for m in all_metrics])
        
        results = {
            'fusion_strategy': fusion_strategy,
            'model_path': model_path,
            'num_videos_tested': len(test_videos),
            'num_videos_processed': len(all_metrics),
            'tolerance': tolerance,
            'micro_metrics': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1_score': micro_f1,
                'total_detected': total_detected,
                'total_ground_truth': total_gt,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn
            },
            'macro_metrics': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1,
                'precision_std': np.std([m['precision'] for m in all_metrics]),
                'recall_std': np.std([m['recall'] for m in all_metrics]),
                'f1_std': np.std([m['f1_score'] for m in all_metrics])
            },
            'per_video_results': per_video_results
        }
    else:
        results = {
            'fusion_strategy': fusion_strategy,
            'model_path': model_path,
            'num_videos_tested': len(test_videos),
            'num_videos_processed': 0,
            'error': 'No videos could be processed'
        }
    
    return results


def test_all_models(model_dir: str,
                   features_dir: str,
                   ground_truth_file: str,
                   fusion_strategies: List[str] = None,
                   tolerance: int = 2,
                   quiet: bool = False,
                   results_dir: str = None) -> Dict[str, Any]:
    """
    Test all available models.
    
    Args:
        model_dir: Directory containing model files
        features_dir: Directory containing test features
        ground_truth_file: Path to ground truth JSON
        fusion_strategies: List of strategies to test (None = all available)
        tolerance: Tolerance for boundary matching
        quiet: Suppress output
        results_dir: Directory to save results
        
    Returns:
        Dictionary with all test results
    """
    # Load ground truth with TRANSITION RANGES for proper overlap-based matching
    # This is critical for correct evaluation of gradual transitions
    print(f"Loading ground truth from: {ground_truth_file}")
    ground_truth = load_ground_truth(ground_truth_file, return_ranges=True)  # CRITICAL: Use ranges!
    print(f"Ground truth loaded for {len(ground_truth)} videos (using transition ranges)")
    
    # Get available test videos
    print(f"Scanning features directory: {features_dir}")
    available_videos = get_available_test_videos(features_dir)
    print(f"Found {len(available_videos)} videos with complete features")
    
    # Get available models
    available_models = get_available_models(model_dir)
    print(f"Found {len(available_models)} trained models: {list(available_models.keys())}")
    
    # Filter to requested strategies
    if fusion_strategies:
        models_to_test = {s: p for s, p in available_models.items() if s in fusion_strategies}
    else:
        models_to_test = available_models
    
    if not models_to_test:
        print("No models to test!")
        return {}
    
    print(f"\nTesting {len(models_to_test)} models: {list(models_to_test.keys())}")
    
    # Create config
    config = VideoShotConfig()
    
    # Test each model
    all_results = {}
    
    for strategy, model_path in models_to_test.items():
        print(f"\n{'='*80}")
        print(f"Testing: {strategy}")
        print(f"{'='*80}")
        
        results = test_single_model(
            model_path=model_path,
            fusion_strategy=strategy,
            features_dir=features_dir,
            ground_truth=ground_truth,
            video_names=available_videos,
            config=config,
            tolerance=tolerance,
            quiet=quiet
        )
        
        all_results[strategy] = results
        
        # Print summary
        if 'micro_metrics' in results:
            print(f"\n{strategy} Results:")
            print(f"  Micro F1: {results['micro_metrics']['f1_score']:.4f}")
            print(f"  Micro Precision: {results['micro_metrics']['precision']:.4f}")
            print(f"  Micro Recall: {results['micro_metrics']['recall']:.4f}")
            print(f"  Macro F1: {results['macro_metrics']['f1_score']:.4f} (±{results['macro_metrics']['f1_std']:.4f})")
    
    # Save results
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save full results
        results_file = os.path.join(results_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Prepare serializable results (remove numpy arrays from per_video_results)
        serializable_results = {}
        for strategy, result in all_results.items():
            serializable_results[strategy] = {
                k: v for k, v in result.items() 
                if k != 'per_video_results'
            }
            # Include summary of per-video results
            if 'per_video_results' in result:
                serializable_results[strategy]['per_video_summary'] = {
                    video: {k: v for k, v in vresult.items() if k not in ['detected', 'ground_truth']}
                    for video, vresult in result['per_video_results'].items()
                }
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'tolerance': tolerance,
                'num_test_videos': len(available_videos),
                'models_tested': list(models_to_test.keys()),
                'results': serializable_results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    return all_results


def print_comparison_table(results: Dict[str, Any]) -> None:
    """
    Print a comparison table of test results.
    
    Args:
        results: Dictionary with test results for each strategy
    """
    print("\n" + "="*100)
    print("TEST RESULTS COMPARISON")
    print("="*100)
    
    # Header
    print(f"{'Strategy':<20} {'Micro F1':<12} {'Micro P':<12} {'Micro R':<12} {'Macro F1':<12} {'Videos':<10}")
    print("-"*100)
    
    # Sort by micro F1 score
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('micro_metrics', {}).get('f1_score', 0),
        reverse=True
    )
    
    for strategy, result in sorted_results:
        if 'micro_metrics' in result:
            micro = result['micro_metrics']
            macro = result['macro_metrics']
            print(f"{strategy:<20} {micro['f1_score']:<12.4f} {micro['precision']:<12.4f} "
                  f"{micro['recall']:<12.4f} {macro['f1_score']:<12.4f} {result['num_videos_processed']:<10}")
        else:
            print(f"{strategy:<20} {'ERROR':<12} {'-':<12} {'-':<12} {'-':<12} {0:<10}")
    
    print("-"*100)
    
    # Find best strategy
    if sorted_results and 'micro_metrics' in sorted_results[0][1]:
        best_strategy = sorted_results[0][0]
        best_f1 = sorted_results[0][1]['micro_metrics']['f1_score']
        print(f"\nBest Strategy: {best_strategy} (Micro F1: {best_f1:.4f})")
    
    print("="*100)


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test trained transformer models")
    
    # Data paths
    parser.add_argument("--features_dir", default=DEFAULT_FEATURES_DIR,
                       help="Directory containing test features")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                       help="Directory containing trained models")
    parser.add_argument("--ground_truth", default=DEFAULT_GROUND_TRUTH,
                       help="Path to ground truth JSON file")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR,
                       help="Directory to save test results")
    
    # Model selection
    parser.add_argument("--fusion_strategies", nargs="+", default=None,
                       help="List of fusion strategies to test (default: all available)")
    parser.add_argument("--test_all", action="store_true",
                       help="Test all available models")
    
    # Testing parameters
    parser.add_argument("--tolerance", type=int, default=3,
                       help="Tolerance window for boundary matching (frames) - default 3 to match training")
    
    # Output control
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress detailed output")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.features_dir):
        print(f"Error: Features directory not found: {args.features_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)
    
    # Run tests
    print("="*80)
    print("TRANSFORMER MODEL TESTING")
    print("="*80)
    print(f"Features directory: {args.features_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Tolerance: {args.tolerance} frames")
    
    results = test_all_models(
        model_dir=args.model_dir,
        features_dir=args.features_dir,
        ground_truth_file=args.ground_truth,
        fusion_strategies=args.fusion_strategies,
        tolerance=args.tolerance,
        quiet=args.quiet,
        results_dir=None if args.no_save else args.results_dir
    )
    
    # Print comparison table
    if results:
        print_comparison_table(results)
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()
