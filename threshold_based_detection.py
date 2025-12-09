"""
Threshold-based shot boundary detection using the methodology from the paper.
Applies the detection to raw, smoothed, and normalized+smoothed features.

The method:
1. Calculate pairwise similarity/correlation between consecutive frames
2. Apply a threshold to detect boundaries (low similarity = boundary)
3. Handle dissolving frames and frequent transitions
4. Enforce minimum scene length constraints

This script processes features extracted by analyze_feature_distances.py
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Set
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_DIRS = {
    'no_sampling': 'training_features_no_sampling',
    'sampling_8': 'training_features_sampling_8'
}

OUTPUT_DIR = 'threshold_detection_results'

# Feature types to analyze
FEATURE_TYPES = ['cnn', 'hsv_histogram', 'clip', 'optical_flow']

# Thresholds for each feature type (based on paper: CNN ~0.8, HSV correlation ~0.9)
# These are for CORRELATION values (high = similar, low = boundary)
# A boundary occurs when correlation DROPS BELOW threshold
THRESHOLDS = {
    'cnn': 0.90,           # Correlation threshold (high - only strong changes)
    'hsv_histogram': 0.92,  # Correlation threshold (high - very stable scenes)
    'clip': 0.90,          # Correlation threshold
    'optical_flow': 0.85   # Correlation threshold (lower - more sensitive to motion)
}

# Scene constraints (from paper: minimum 2 seconds)
# For sampling_rate=1 at ~25fps: 2 seconds = 50 frames
# For sampling_rate=8: 2 seconds = 6-7 frames
MIN_SCENE_LENGTH = {
    'no_sampling': 50,   # ~2 seconds at 25 FPS
    'sampling_8': 6      # ~2 seconds at 3-4 FPS (25/8)
}

# Dissolving frames window (consecutive frames below threshold)
MAX_DISSOLVE_WINDOW = 5

# Threshold for normalized features (after min-max normalization to [0,1])
# Since normalized features are in [0,1], we need a different threshold
NORMALIZED_THRESHOLD = 0.85  # High value since we want high correlation = same scene


def load_selected_videos() -> List[Tuple[str, int, List, List]]:
    """Load the selected videos from the JSON file."""
    with open('selected_videos.json', 'r') as f:
        data = json.load(f)
    return [(item[0], item[1], item[2], item[3]) for item in data]


def load_analysis_results(sampling_key: str) -> Dict:
    """Load pre-computed analysis results."""
    results_file = os.path.join('feature_analysis_results', f'all_results_{sampling_key}.json')
    with open(results_file, 'r') as f:
        return json.load(f)


def detect_boundaries_simple(correlation: np.ndarray, threshold: float) -> List[int]:
    """
    Simple threshold-based boundary detection.
    Returns frame indices where boundaries occur.
    
    A boundary is detected when correlation drops BELOW the threshold.
    
    Args:
        correlation: Array of correlation scores between consecutive frames (higher = more similar)
        threshold: Threshold below which a boundary is detected
        
    Returns:
        List of boundary frame indices
    """
    boundaries = []
    for i in range(len(correlation)):
        if correlation[i] < threshold:
            boundaries.append(i)
    return boundaries


def detect_boundaries_with_constraints(correlation: np.ndarray, threshold: float, 
                                       min_scene_length: int, 
                                       max_dissolve_window: int = MAX_DISSOLVE_WINDOW) -> List[int]:
    """
    Advanced boundary detection with constraints from the paper:
    1. Minimum scene length enforcement
    2. Dissolving frames handling (consecutive frames below threshold)
    3. Frequent transition filtering
    
    A boundary is detected when correlation drops BELOW the threshold.
    
    Args:
        correlation: Array of correlation scores between consecutive frames (higher = more similar)
        threshold: Threshold below which a boundary is detected
        min_scene_length: Minimum number of frames in a scene
        max_dissolve_window: Maximum consecutive frames below threshold to consider as dissolving
        
    Returns:
        List of boundary frame indices
    """
    # Step 1: Find all points below threshold
    below_threshold = np.where(correlation < threshold)[0]
    
    if len(below_threshold) == 0:
        return []
    
    # Step 2: Group consecutive below-threshold points (potential dissolving frames)
    groups = []
    current_group = [below_threshold[0]]
    
    for i in range(1, len(below_threshold)):
        if below_threshold[i] == below_threshold[i-1] + 1:
            current_group.append(below_threshold[i])
        else:
            groups.append(current_group)
            current_group = [below_threshold[i]]
    groups.append(current_group)
    
    # Step 3: Filter groups
    # - Single points or small groups (≤ max_dissolve_window) are boundaries
    # - Large groups (> max_dissolve_window) are dissolving frames, take only borders
    boundaries = []
    
    for group in groups:
        if len(group) <= max_dissolve_window:
            # Use the middle or first point as boundary
            boundaries.append(group[0])
        else:
            # Dissolving frames: take border boundaries
            boundaries.append(group[0])
            boundaries.append(group[-1])
    
    # Step 4: Enforce minimum scene length
    if len(boundaries) == 0:
        return []
    
    filtered_boundaries = [boundaries[0]]
    
    for boundary in boundaries[1:]:
        # Check distance from last accepted boundary
        if boundary - filtered_boundaries[-1] >= min_scene_length:
            filtered_boundaries.append(boundary)
    
    return filtered_boundaries


def evaluate_detection(detected_boundaries: List[int], 
                       ground_truth_boundaries: List[int],
                       tolerance: int = 5) -> Dict:
    """
    Evaluate detection performance against ground truth.
    Uses frame-level tolerance for matching.
    
    Args:
        detected_boundaries: List of detected boundary frame indices
        ground_truth_boundaries: List of ground truth boundary frame indices
        tolerance: Number of frames within which a detection is considered correct
        
    Returns:
        Dictionary with precision, recall, F1, and counts
    """
    if len(ground_truth_boundaries) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'true_positives': 0,
            'false_positives': len(detected_boundaries),
            'false_negatives': 0,
            'num_detected': len(detected_boundaries),
            'num_ground_truth': 0
        }
    
    detected_set = set(detected_boundaries)
    gt_set = set(ground_truth_boundaries)
    
    # Find true positives (detected boundaries within tolerance of ground truth)
    true_positives = 0
    matched_gt = set()
    matched_detected = set()
    
    for gt_boundary in gt_set:
        for detected_boundary in detected_set:
            if abs(detected_boundary - gt_boundary) <= tolerance:
                true_positives += 1
                matched_gt.add(gt_boundary)
                matched_detected.add(detected_boundary)
                break
    
    false_positives = len(detected_set) - len(matched_detected)
    false_negatives = len(gt_set) - len(matched_gt)
    
    # Calculate metrics
    precision = true_positives / len(detected_set) if len(detected_set) > 0 else 0.0
    recall = true_positives / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'num_detected': len(detected_boundaries),
        'num_ground_truth': len(ground_truth_boundaries)
    }


def plot_detection_results(video_name: str, video_id: int, 
                           correlation: np.ndarray,
                           detected_boundaries: List[int],
                           ground_truth_boundaries: List[int],
                           threshold: float,
                           feature_type: str,
                           version: str,
                           output_dir: str,
                           metrics: Dict):
    """
    Create visualization of detection results.
    
    Args:
        video_name: Name of the video
        video_id: Numeric ID (1-10)
        correlation: Correlation scores array (higher = more similar)
        detected_boundaries: Detected boundary indices
        ground_truth_boundaries: Ground truth boundary indices
        threshold: Detection threshold used
        feature_type: Type of feature
        version: Version of features (raw, smoothed, normalized_smoothed)
        output_dir: Output directory
        metrics: Evaluation metrics
    """
    frames = np.arange(1, len(correlation) + 1)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot correlation
    ax.plot(frames, correlation, linewidth=1.5, alpha=0.7, color='blue', label='Correlation')
    
    # Plot threshold
    ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Threshold ({threshold})')
    
    # Plot ground truth boundaries
    for boundary in ground_truth_boundaries:
        if boundary < len(correlation):
            ax.axvline(x=boundary, color='green', linestyle='-', linewidth=2, alpha=0.6, label='Ground Truth' if boundary == ground_truth_boundaries[0] else '')
    
    # Plot detected boundaries
    for boundary in detected_boundaries:
        if boundary < len(correlation):
            ax.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Detected' if boundary == detected_boundaries[0] else '')
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Correlation Score', fontsize=12)
    ax.set_title(f'Video {video_id} - {feature_type.upper()} ({version}) - Boundary Detection\n'
                f'P={metrics["precision"]:.3f}, R={metrics["recall"]:.3f}, F1={metrics["f1"]:.3f}',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{feature_type}_{version}_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()


def process_video_feature(video_name: str, video_id: int,
                          feature_type: str,
                          video_data: Dict,
                          ground_truth_boundaries: List[int],
                          sampling_key: str,
                          output_dir: str) -> Dict:
    """
    Process a single feature type for a video with all versions.
    
    Args:
        video_name: Name of the video
        video_id: Numeric ID
        feature_type: Type of feature
        video_data: Pre-computed analysis data
        ground_truth_boundaries: Ground truth boundaries
        sampling_key: Sampling rate key
        output_dir: Output directory
        
    Returns:
        Dictionary with results for all versions
    """
    if feature_type not in video_data['results']:
        return {}
    
    results = {}
    min_scene_length = MIN_SCENE_LENGTH[sampling_key]
    
    # Get pre-computed features
    correlation_raw = np.array(video_data['results'][feature_type]['correlation'])
    correlation_smooth = np.array(video_data['results'][feature_type]['correlation_smooth'])
    
    # Process three versions
    versions = {
        'raw': correlation_raw,
        'smoothed': correlation_smooth,
        'normalized_smoothed': correlation_smooth  # Will normalize below
    }
    
    for version, correlation in versions.items():
        # Determine threshold based on version
        if version == 'normalized_smoothed':
            # Normalize to [0, 1] range first
            min_val = np.min(correlation)
            max_val = np.max(correlation)
            if max_val - min_val > 1e-10:
                correlation = (correlation - min_val) / (max_val - min_val)
            else:
                correlation = np.zeros_like(correlation)
            threshold = NORMALIZED_THRESHOLD
        else:
            threshold = THRESHOLDS[feature_type]
        
        # Detect boundaries (simple method)
        detected_boundaries_simple = detect_boundaries_simple(correlation, threshold)
        
        # Detect boundaries (with constraints)
        detected_boundaries_constrained = detect_boundaries_with_constraints(
            correlation, threshold, min_scene_length, MAX_DISSOLVE_WINDOW
        )
        
        # Evaluate both methods
        metrics_simple = evaluate_detection(detected_boundaries_simple, ground_truth_boundaries)
        metrics_constrained = evaluate_detection(detected_boundaries_constrained, ground_truth_boundaries)
        
        # Plot results (constrained method)
        plot_detection_results(
            video_name, video_id, correlation,
            detected_boundaries_constrained, ground_truth_boundaries,
            threshold, feature_type, version, output_dir, metrics_constrained
        )
        
        results[version] = {
            'simple': {
                'detected_boundaries': detected_boundaries_simple,
                'metrics': metrics_simple
            },
            'constrained': {
                'detected_boundaries': detected_boundaries_constrained,
                'metrics': metrics_constrained
            }
        }
    
    return results


def calculate_aggregate_statistics(all_results: Dict) -> Dict:
    """
    Calculate aggregate statistics across all videos.
    
    Args:
        all_results: Dictionary with all results
        
    Returns:
        Dictionary with aggregate statistics
    """
    stats = {}
    
    for feature_type in FEATURE_TYPES:
        stats[feature_type] = {}
        
        for version in ['raw', 'smoothed', 'normalized_smoothed']:
            stats[feature_type][version] = {}
            
            for method in ['simple', 'constrained']:
                precisions = []
                recalls = []
                f1_scores = []
                
                for video_name, video_results in all_results.items():
                    if feature_type in video_results and version in video_results[feature_type]:
                        metrics = video_results[feature_type][version][method]['metrics']
                        precisions.append(metrics['precision'])
                        recalls.append(metrics['recall'])
                        f1_scores.append(metrics['f1'])
                
                stats[feature_type][version][method] = {
                    'mean_precision': float(np.mean(precisions)) if precisions else 0.0,
                    'mean_recall': float(np.mean(recalls)) if recalls else 0.0,
                    'mean_f1': float(np.mean(f1_scores)) if f1_scores else 0.0,
                    'std_precision': float(np.std(precisions)) if precisions else 0.0,
                    'std_recall': float(np.std(recalls)) if recalls else 0.0,
                    'std_f1': float(np.std(f1_scores)) if f1_scores else 0.0,
                    'num_videos': len(precisions)
                }
    
    return stats


def print_statistics(stats: Dict, sampling_key: str):
    """Print aggregate statistics in a readable format."""
    print(f"\n{'='*80}")
    print(f"AGGREGATE STATISTICS - {sampling_key.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    
    for feature_type in FEATURE_TYPES:
        print(f"\n{feature_type.upper()}")
        print("-" * 60)
        
        for version in ['raw', 'smoothed', 'normalized_smoothed']:
            print(f"\n  {version.upper()}:")
            
            for method in ['simple', 'constrained']:
                method_stats = stats[feature_type][version][method]
                print(f"    {method.capitalize():12} - P: {method_stats['mean_precision']:.3f}±{method_stats['std_precision']:.3f}, "
                      f"R: {method_stats['mean_recall']:.3f}±{method_stats['std_recall']:.3f}, "
                      f"F1: {method_stats['mean_f1']:.3f}±{method_stats['std_f1']:.3f}")


def create_summary_comparison_plot(stats: Dict, sampling_key: str, output_dir: str):
    """
    Create bar plots comparing methods across features and versions.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    versions = ['raw', 'smoothed', 'normalized_smoothed']
    version_labels = ['Raw', 'Smoothed', 'Normalized+Smoothed']
    metrics = ['mean_precision', 'mean_recall', 'mean_f1']
    metric_labels = ['Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(FEATURE_TYPES))
    width = 0.12
    
    for ax_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        
        for v_idx, (version, version_label) in enumerate(zip(versions, version_labels)):
            # Simple method
            values_simple = [stats[ft][version]['simple'][metric] for ft in FEATURE_TYPES]
            ax.bar(x + (v_idx * 2 - 2) * width, values_simple, width, 
                   label=f'{version_label} (Simple)', alpha=0.6)
            
            # Constrained method
            values_constrained = [stats[ft][version]['constrained'][metric] for ft in FEATURE_TYPES]
            ax.bar(x + (v_idx * 2 - 1) * width, values_constrained, width,
                   label=f'{version_label} (Constrained)', alpha=0.9)
        
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([ft.upper() for ft in FEATURE_TYPES])
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
    
    axes[-1].set_xlabel('Feature Type', fontsize=11)
    fig.suptitle(f'Detection Performance Comparison - {sampling_key.upper().replace("_", " ")}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_comparison_{sampling_key}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    print("=" * 80)
    print("THRESHOLD-BASED SHOT BOUNDARY DETECTION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Min scene lengths: {MIN_SCENE_LENGTH}")
    print(f"  Max dissolve window: {MAX_DISSOLVE_WINDOW}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load selected videos
    print(f"\nLoading selected videos...")
    selected_videos = load_selected_videos()
    print(f"Found {len(selected_videos)} videos")
    
    # Create video ID mapping
    video_id_map = {video[0]: idx + 1 for idx, video in enumerate(selected_videos)}
    
    # Process each sampling rate
    for sampling_key in FEATURES_DIRS.keys():
        print("\n" + "=" * 80)
        print(f"PROCESSING: {sampling_key.upper().replace('_', ' ')}")
        print("=" * 80)
        
        # Load pre-computed analysis results
        print(f"Loading analysis results for {sampling_key}...")
        analysis_results = load_analysis_results(sampling_key)
        
        # Create output directory for this sampling rate
        sampling_output_dir = os.path.join(OUTPUT_DIR, sampling_key)
        os.makedirs(sampling_output_dir, exist_ok=True)
        
        # Process each video
        all_results = {}
        
        for video_name, frame_count, transitions, boundaries in selected_videos:
            video_id = video_id_map[video_name]
            print(f"\n{'='*60}")
            print(f"Processing: Video {video_id} ({video_name})")
            print(f"  Ground truth boundaries: {len(boundaries)} at {boundaries}")
            
            # Get pre-computed data
            video_data = analysis_results[video_name]
            
            # Create video output directory
            video_output_dir = os.path.join(sampling_output_dir, str(video_id))
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Process each feature type
            video_results = {}
            for feature_type in FEATURE_TYPES:
                print(f"  Processing {feature_type}...")
                
                feature_results = process_video_feature(
                    video_name, video_id, feature_type,
                    video_data, boundaries, sampling_key,
                    video_output_dir
                )
                
                if feature_results:
                    video_results[feature_type] = feature_results
                    
                    # Print metrics summary
                    for version in ['raw', 'smoothed', 'normalized_smoothed']:
                        if version in feature_results:
                            constrained_metrics = feature_results[version]['constrained']['metrics']
                            print(f"    {version:20} - P: {constrained_metrics['precision']:.3f}, "
                                  f"R: {constrained_metrics['recall']:.3f}, "
                                  f"F1: {constrained_metrics['f1']:.3f}")
            
            all_results[video_name] = video_results
        
        # Calculate and display aggregate statistics
        stats = calculate_aggregate_statistics(all_results)
        print_statistics(stats, sampling_key)
        
        # Create summary comparison plot
        create_summary_comparison_plot(stats, sampling_key, sampling_output_dir)
        
        # Save results to JSON
        results_file = os.path.join(sampling_output_dir, 'detection_results.json')
        
        # Convert results to JSON-serializable format
        json_results = {}
        for video_name, video_results in all_results.items():
            json_results[video_name] = {}
            for feature_type, feature_results in video_results.items():
                json_results[video_name][feature_type] = {}
                for version, version_results in feature_results.items():
                    json_results[video_name][feature_type][version] = {}
                    for method, method_results in version_results.items():
                        json_results[video_name][feature_type][version][method] = {
                            'detected_boundaries': [int(b) for b in method_results['detected_boundaries']],
                            'metrics': method_results['metrics']
                        }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Save statistics to JSON
        stats_file = os.path.join(sampling_output_dir, 'detection_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_file}")
    
    print("\n" + "=" * 80)
    print("DETECTION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - detection_results.json: Per-video detection results")
    print(f"  - detection_statistics.json: Aggregate statistics")
    print(f"  - summary_comparison_*.png: Performance comparison plots")
    print(f"  - Video directories (1-10): Individual detection visualizations")


if __name__ == "__main__":
    main()
