"""
Locally Adaptive Robust Shot Boundary Detection.

Implements the algorithm from the paper using:
1. Robust local normalization with median and MAD (Median Absolute Deviation)
2. Adaptive local thresholding based on normalized deviation scores
3. Dissolve detection and handling
4. Minimum scene length constraints

This approach is more robust than fixed global thresholds because it adapts
to local statistics of the similarity signal.
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_DIRS = {
    'no_sampling': 'training_features_no_sampling',
    'sampling_8': 'training_features_sampling_8'
}

OUTPUT_DIR = 'adaptive_threshold_detection_results'

# Feature types to analyze
FEATURE_TYPES = ['cnn', 'hsv_histogram', 'clip', 'optical_flow']

# ============================================================================
# ALGORITHM PARAMETERS
# ============================================================================

# Local window radius (w): Controls temporal neighborhood for local statistics
# Smaller w = more sensitive to rapid cuts, larger w = more stable in slow scenes
# Typical range: [3, 10] frames
LOCAL_WINDOW_RADIUS = 5

# Statistics window multiplier (k): Expands local window to 2*k*w frames for robust estimation
# Separates detection resolution from statistical reliability
STATS_WINDOW_MULTIPLIER = 2

# Sensitivity factor (alpha): Controls threshold in normalized deviation space
# Larger alpha = only strong transitions (hard cuts), smaller = also subtle changes
# Boundary detected when normalized score < -alpha
SENSITIVITY_FACTORS = {
    'cnn': 2.5,           # Conservative - only strong semantic changes
    'hsv_histogram': 2.0,  # Moderate - color histogram changes
    'clip': 2.5,          # Conservative - CLIP semantic similarity
    'optical_flow': 1.5   # More sensitive - motion features
}

# Default sensitivity factor if not specified per feature
DEFAULT_SENSITIVITY = 2.0

# Minimum scene length (L_min): Prevents over-segmentation
# For sampling_rate=1 at ~25fps: 2 seconds = 50 frames
# For sampling_rate=8: 2 seconds = 6-7 frames
MIN_SCENE_LENGTH = {
    'no_sampling': 50,   # ~2 seconds at 25 FPS
    'sampling_8': 6      # ~2 seconds at 3-4 FPS (25/8)
}

# Maximum dissolve window (W_max): Upper bound on gradual transition length
# Sustained low similarity over many frames indicates dissolves/fades
MAX_DISSOLVE_WINDOW = 5

# Stability constant (epsilon): Prevents division by near-zero MAD
# Ensures numerical stability in static or near-static scenes
EPSILON = 1e-6

# Evaluation tolerance (frames within which detection is considered correct)
EVALUATION_TOLERANCE = 5


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


def compute_mad(data: np.ndarray) -> float:
    """
    Compute Median Absolute Deviation (MAD).
    MAD = median(|X - median(X)|)
    
    A robust measure of statistical dispersion, less sensitive to outliers than std.
    
    Args:
        data: 1D array of values
        
    Returns:
        MAD value
    """
    median = np.median(data)
    return np.median(np.abs(data - median))


def consecutive_segments(indices: List[int]) -> List[List[int]]:
    """
    Group consecutive indices into segments.
    
    Args:
        indices: List of indices (sorted)
        
    Returns:
        List of groups where each group contains consecutive indices
    """
    if len(indices) == 0:
        return []
    
    groups = []
    current_group = [indices[0]]
    
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current_group.append(indices[i])
        else:
            groups.append(current_group)
            current_group = [indices[i]]
    groups.append(current_group)
    
    return groups


def locally_adaptive_detection(
    similarity: np.ndarray,
    local_window_radius: int,
    stats_window_multiplier: int,
    sensitivity: float,
    min_scene_length: int,
    max_dissolve_window: int,
    epsilon: float = EPSILON
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """
    Locally Adaptive Robust Shot Boundary Detection Algorithm.
    
    Implements the complete algorithm with:
    1. Local robust normalization using median and MAD
    2. Adaptive thresholding based on normalized deviation scores
    3. Dissolve detection and handling
    4. Minimum scene length constraints
    
    Args:
        similarity: Array of similarity/correlation scores between consecutive frames
                   (higher values = more similar, lower = potential boundary)
        local_window_radius: Radius w for local window (detection resolution)
        stats_window_multiplier: Multiplier k for statistics window (2*k*w total)
        sensitivity: Factor α for threshold in normalized space (threshold = -α)
        min_scene_length: Minimum L_min frames between boundaries
        max_dissolve_window: Maximum W_max consecutive frames for dissolve detection
        epsilon: Stability constant ε to prevent division by zero
        
    Returns:
        Tuple of (boundary_indices, normalized_scores, local_medians, local_mads)
    """
    n = len(similarity)
    kw = stats_window_multiplier * local_window_radius  # Half-width of statistics window
    
    # Initialize arrays for storing local statistics
    normalized_scores = np.zeros(n)
    local_medians = np.zeros(n)
    local_mads = np.zeros(n)
    
    # Mark boundary regions (can't compute full statistics near edges)
    valid_start = kw
    valid_end = n - kw
    
    candidate_indices = []
    
    # Step 1-11: Compute normalized scores and find candidates
    for i in range(n):
        if i < valid_start or i >= valid_end:
            # Edge regions: use available data
            start_idx = max(0, i - kw)
            end_idx = min(n, i + kw + 1)
        else:
            # Full window available
            start_idx = i - kw
            end_idx = i + kw + 1
        
        # Extract local window W_i
        W_i = similarity[start_idx:end_idx]
        
        # Compute robust local statistics
        med_i = np.median(W_i)
        mad_i = max(compute_mad(W_i), epsilon)  # Prevent MAD collapse
        
        local_medians[i] = med_i
        local_mads[i] = mad_i
        
        # Compute normalized deviation score
        # Ĉ_i = (C[i] - med_i) / mad_i
        normalized_scores[i] = (similarity[i] - med_i) / mad_i
        
        # Apply adaptive local threshold θ_i = -α
        # A boundary is detected when Ĉ_i < θ_i (i.e., significant negative deviation)
        if normalized_scores[i] < -sensitivity:
            candidate_indices.append(i)
    
    # Step 12: Group consecutive candidates
    groups = consecutive_segments(candidate_indices)
    
    # Step 13-22: Identify boundaries with dissolve handling
    boundaries = []
    
    for group in groups:
        if len(group) <= max_dissolve_window:
            # Short group: hard cut
            # Select frame with minimum normalized score (maximum semantic contrast)
            min_idx = group[np.argmin([normalized_scores[idx] for idx in group])]
            boundaries.append(min_idx)
        else:
            # Long group: gradual transition (dissolve)
            # Take borders: start and end of transition
            boundaries.append(group[0])
            boundaries.append(group[-1])
    
    # Step 23-29: Enforce minimum scene length
    if len(boundaries) == 0:
        return [], normalized_scores, local_medians, local_mads
    
    filtered_boundaries = [boundaries[0]]
    
    for b in boundaries[1:]:
        if b - filtered_boundaries[-1] >= min_scene_length:
            filtered_boundaries.append(b)
    
    return filtered_boundaries, normalized_scores, local_medians, local_mads


def detect_boundaries_adaptive(
    correlation: np.ndarray,
    feature_type: str,
    sampling_key: str
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply locally adaptive detection to a feature correlation signal.
    
    Args:
        correlation: Correlation scores array (higher = more similar)
        feature_type: Type of feature ('cnn', 'clip', etc.)
        sampling_key: Sampling rate key ('no_sampling' or 'sampling_8')
        
    Returns:
        Tuple of (boundary_indices, normalized_scores, local_medians, local_mads)
    """
    sensitivity = SENSITIVITY_FACTORS.get(feature_type, DEFAULT_SENSITIVITY)
    min_scene_length = MIN_SCENE_LENGTH[sampling_key]
    
    return locally_adaptive_detection(
        similarity=correlation,
        local_window_radius=LOCAL_WINDOW_RADIUS,
        stats_window_multiplier=STATS_WINDOW_MULTIPLIER,
        sensitivity=sensitivity,
        min_scene_length=min_scene_length,
        max_dissolve_window=MAX_DISSOLVE_WINDOW,
        epsilon=EPSILON
    )


def evaluate_detection(
    detected_boundaries: List[int],
    ground_truth_boundaries: List[int],
    tolerance: int = EVALUATION_TOLERANCE
) -> Dict:
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
            'precision': 0.0 if len(detected_boundaries) > 0 else 1.0,
            'recall': 1.0,  # No ground truth to miss
            'f1': 0.0 if len(detected_boundaries) > 0 else 1.0,
            'true_positives': 0,
            'false_positives': len(detected_boundaries),
            'false_negatives': 0,
            'num_detected': len(detected_boundaries),
            'num_ground_truth': 0
        }
    
    if len(detected_boundaries) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(ground_truth_boundaries),
            'num_detected': 0,
            'num_ground_truth': len(ground_truth_boundaries)
        }
    
    detected_set = set(detected_boundaries)
    gt_set = set(ground_truth_boundaries)
    
    # Find true positives (detected boundaries within tolerance of ground truth)
    true_positives = 0
    matched_gt = set()
    matched_detected = set()
    
    for gt_boundary in sorted(gt_set):
        for detected_boundary in sorted(detected_set):
            if detected_boundary in matched_detected:
                continue
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


def plot_adaptive_detection_results(
    video_name: str,
    video_id: int,
    correlation: np.ndarray,
    normalized_scores: np.ndarray,
    local_medians: np.ndarray,
    local_mads: np.ndarray,
    detected_boundaries: List[int],
    ground_truth_boundaries: List[int],
    sensitivity: float,
    feature_type: str,
    version: str,
    output_dir: str,
    metrics: Dict
):
    """
    Create comprehensive visualization of adaptive detection results.
    
    Shows:
    1. Raw correlation signal with local median/MAD bands
    2. Normalized deviation scores with threshold
    3. Detected vs ground truth boundaries
    """
    frames = np.arange(1, len(correlation) + 1)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Panel 1: Raw correlation with local statistics
    ax1 = axes[0]
    ax1.plot(frames, correlation, linewidth=1.5, alpha=0.7, color='blue', label='Correlation')
    ax1.plot(frames, local_medians, linewidth=2, color='orange', label='Local Median', alpha=0.8)
    
    # Plot MAD bands (median ± MAD)
    upper_band = local_medians + local_mads
    lower_band = local_medians - local_mads
    ax1.fill_between(frames, lower_band, upper_band, alpha=0.2, color='orange', label='±MAD Band')
    
    # Mark ground truth boundaries
    for boundary in ground_truth_boundaries:
        if boundary < len(correlation):
            ax1.axvline(x=boundary, color='green', linestyle='-', linewidth=2, alpha=0.6,
                       label='Ground Truth' if boundary == ground_truth_boundaries[0] else '')
    
    ax1.set_ylabel('Correlation Score', fontsize=11)
    ax1.set_title(f'Raw Correlation with Local Statistics', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Normalized deviation scores with threshold
    ax2 = axes[1]
    ax2.plot(frames, normalized_scores, linewidth=1.5, color='purple', alpha=0.8, label='Normalized Score')
    
    # Draw threshold line
    ax2.axhline(y=-sensitivity, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Threshold (−α = {-sensitivity:.1f})')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Highlight regions below threshold
    below_threshold = normalized_scores < -sensitivity
    ax2.fill_between(frames, normalized_scores, -sensitivity,
                    where=below_threshold, alpha=0.3, color='red', label='Below Threshold')
    
    # Mark ground truth boundaries
    for boundary in ground_truth_boundaries:
        if boundary < len(normalized_scores):
            ax2.axvline(x=boundary, color='green', linestyle='-', linewidth=2, alpha=0.6)
    
    ax2.set_ylabel('Normalized Deviation', fontsize=11)
    ax2.set_title(f'Normalized Deviation Scores (Ĉ = (C - median) / MAD)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Detection comparison
    ax3 = axes[2]
    ax3.plot(frames, correlation, linewidth=1.5, alpha=0.6, color='blue', label='Correlation')
    
    # Mark ground truth boundaries (green)
    for i, boundary in enumerate(ground_truth_boundaries):
        if boundary < len(correlation):
            ax3.axvline(x=boundary, color='green', linestyle='-', linewidth=2.5, alpha=0.7,
                       label='Ground Truth' if i == 0 else '')
    
    # Mark detected boundaries (red dashed)
    for i, boundary in enumerate(detected_boundaries):
        if boundary < len(correlation):
            ax3.axvline(x=boundary, color='red', linestyle='--', linewidth=2.5, alpha=0.7,
                       label='Detected' if i == 0 else '')
    
    ax3.set_xlabel('Frame Index', fontsize=12)
    ax3.set_ylabel('Correlation Score', fontsize=11)
    ax3.set_title(f'Detection Results - P={metrics["precision"]:.3f}, R={metrics["recall"]:.3f}, F1={metrics["f1"]:.3f}',
                 fontsize=12, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Video {video_id} - {feature_type.upper()} ({version}) - Adaptive Boundary Detection\n'
                f'Parameters: w={LOCAL_WINDOW_RADIUS}, k={STATS_WINDOW_MULTIPLIER}, α={sensitivity:.1f}',
                fontsize=13, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{feature_type}_{version}_adaptive_detection.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def process_video_feature_adaptive(
    video_name: str,
    video_id: int,
    feature_type: str,
    video_data: Dict,
    ground_truth_boundaries: List[int],
    sampling_key: str,
    output_dir: str
) -> Dict:
    """
    Process a single feature type for a video using adaptive detection.
    
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
    sensitivity = SENSITIVITY_FACTORS.get(feature_type, DEFAULT_SENSITIVITY)
    min_scene_length = MIN_SCENE_LENGTH[sampling_key]
    
    # Get pre-computed correlation signals
    correlation_raw = np.array(video_data['results'][feature_type]['correlation'])
    correlation_smooth = np.array(video_data['results'][feature_type]['correlation_smooth'])
    
    # Process three versions
    versions = {
        'raw': correlation_raw,
        'smoothed': correlation_smooth,
        'normalized_smoothed': None  # Will normalize smoothed below
    }
    
    for version, correlation in versions.items():
        if version == 'normalized_smoothed':
            # Normalize smoothed correlation to [0, 1] range
            min_val = np.min(correlation_smooth)
            max_val = np.max(correlation_smooth)
            if max_val - min_val > 1e-10:
                correlation = (correlation_smooth - min_val) / (max_val - min_val)
            else:
                correlation = np.zeros_like(correlation_smooth)
        
        # Apply locally adaptive detection
        detected_boundaries, normalized_scores, local_medians, local_mads = locally_adaptive_detection(
            similarity=correlation,
            local_window_radius=LOCAL_WINDOW_RADIUS,
            stats_window_multiplier=STATS_WINDOW_MULTIPLIER,
            sensitivity=sensitivity,
            min_scene_length=min_scene_length,
            max_dissolve_window=MAX_DISSOLVE_WINDOW,
            epsilon=EPSILON
        )
        
        # Evaluate detection
        metrics = evaluate_detection(detected_boundaries, ground_truth_boundaries)
        
        # Plot results
        plot_adaptive_detection_results(
            video_name, video_id, correlation,
            normalized_scores, local_medians, local_mads,
            detected_boundaries, ground_truth_boundaries,
            sensitivity, feature_type, version, output_dir, metrics
        )
        
        results[version] = {
            'detected_boundaries': detected_boundaries,
            'metrics': metrics,
            'parameters': {
                'local_window_radius': LOCAL_WINDOW_RADIUS,
                'stats_window_multiplier': STATS_WINDOW_MULTIPLIER,
                'sensitivity': sensitivity,
                'min_scene_length': min_scene_length,
                'max_dissolve_window': MAX_DISSOLVE_WINDOW,
                'epsilon': EPSILON
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
            precisions = []
            recalls = []
            f1_scores = []
            
            for video_name, video_results in all_results.items():
                if feature_type in video_results and version in video_results[feature_type]:
                    metrics = video_results[feature_type][version]['metrics']
                    precisions.append(metrics['precision'])
                    recalls.append(metrics['recall'])
                    f1_scores.append(metrics['f1'])
            
            stats[feature_type][version] = {
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
            version_stats = stats[feature_type][version]
            print(f"  {version:20} - P: {version_stats['mean_precision']:.3f}±{version_stats['std_precision']:.3f}, "
                  f"R: {version_stats['mean_recall']:.3f}±{version_stats['std_recall']:.3f}, "
                  f"F1: {version_stats['mean_f1']:.3f}±{version_stats['std_f1']:.3f}")


def create_summary_comparison_plot(stats: Dict, sampling_key: str, output_dir: str):
    """
    Create bar plots comparing different feature types and versions.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    versions = ['raw', 'smoothed', 'normalized_smoothed']
    version_labels = ['Raw', 'Smoothed', 'Normalized+Smoothed']
    metrics_keys = ['mean_precision', 'mean_recall', 'mean_f1']
    metric_labels = ['Precision', 'Recall', 'F1-Score']
    colors = ['#2ecc71', '#3498db', '#9b59b6']  # Green, Blue, Purple
    
    x = np.arange(len(FEATURE_TYPES))
    width = 0.25
    
    for ax_idx, (metric_key, metric_label) in enumerate(zip(metrics_keys, metric_labels)):
        ax = axes[ax_idx]
        
        for v_idx, (version, version_label, color) in enumerate(zip(versions, version_labels, colors)):
            values = [stats[ft][version][metric_key] for ft in FEATURE_TYPES]
            stds = [stats[ft][version][f'std_{metric_key.replace("mean_", "")}'] for ft in FEATURE_TYPES]
            
            bars = ax.bar(x + (v_idx - 1) * width, values, width,
                         label=version_label, color=color, alpha=0.8,
                         yerr=stds, capsize=3)
        
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([ft.upper() for ft in FEATURE_TYPES])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
    
    axes[-1].set_xlabel('Feature Type', fontsize=11)
    fig.suptitle(f'Adaptive Detection Performance - {sampling_key.upper().replace("_", " ")}\n'
                f'Parameters: w={LOCAL_WINDOW_RADIUS}, k={STATS_WINDOW_MULTIPLIER}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_comparison_{sampling_key}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def create_feature_comparison_heatmap(stats_all: Dict, output_dir: str):
    """
    Create heatmaps comparing F1 scores across features, versions, and sampling rates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax_idx, sampling_key in enumerate(['no_sampling', 'sampling_8']):
        if sampling_key not in stats_all:
            continue
            
        stats = stats_all[sampling_key]
        
        # Build matrix: features x versions
        matrix = np.zeros((len(FEATURE_TYPES), 3))
        versions = ['raw', 'smoothed', 'normalized_smoothed']
        
        for i, feature_type in enumerate(FEATURE_TYPES):
            for j, version in enumerate(versions):
                matrix[i, j] = stats[feature_type][version]['mean_f1']
        
        ax = axes[ax_idx]
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(len(FEATURE_TYPES)):
            for j in range(3):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                              ha='center', va='center', color='black', fontsize=10)
        
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(['Raw', 'Smoothed', 'Norm+Smooth'], fontsize=10)
        ax.set_yticks(np.arange(len(FEATURE_TYPES)))
        ax.set_yticklabels([ft.upper() for ft in FEATURE_TYPES], fontsize=10)
        ax.set_title(f'{sampling_key.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Mean F1-Score', shrink=0.8)
    fig.suptitle('Adaptive Detection - F1 Score Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_heatmap_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    print("=" * 80)
    print("LOCALLY ADAPTIVE ROBUST SHOT BOUNDARY DETECTION")
    print("=" * 80)
    
    print(f"\nAlgorithm Parameters:")
    print(f"  Local window radius (w): {LOCAL_WINDOW_RADIUS}")
    print(f"  Statistics window multiplier (k): {STATS_WINDOW_MULTIPLIER}")
    print(f"  Statistics window size: 2 × k × w = {2 * STATS_WINDOW_MULTIPLIER * LOCAL_WINDOW_RADIUS} frames")
    print(f"  Sensitivity factors (α): {SENSITIVITY_FACTORS}")
    print(f"  Min scene lengths: {MIN_SCENE_LENGTH}")
    print(f"  Max dissolve window: {MAX_DISSOLVE_WINDOW}")
    print(f"  Stability constant (ε): {EPSILON}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load selected videos
    print(f"\nLoading selected videos...")
    selected_videos = load_selected_videos()
    print(f"Found {len(selected_videos)} videos")
    
    # Create video ID mapping
    video_id_map = {video[0]: idx + 1 for idx, video in enumerate(selected_videos)}
    
    # Store all statistics for final comparison
    stats_all = {}
    
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
                print(f"  Processing {feature_type} (α={SENSITIVITY_FACTORS.get(feature_type, DEFAULT_SENSITIVITY):.1f})...")
                
                feature_results = process_video_feature_adaptive(
                    video_name, video_id, feature_type,
                    video_data, boundaries, sampling_key,
                    video_output_dir
                )
                
                if feature_results:
                    video_results[feature_type] = feature_results
                    
                    # Print metrics summary
                    for version in ['raw', 'smoothed', 'normalized_smoothed']:
                        if version in feature_results:
                            metrics = feature_results[version]['metrics']
                            print(f"    {version:20} - P: {metrics['precision']:.3f}, "
                                  f"R: {metrics['recall']:.3f}, "
                                  f"F1: {metrics['f1']:.3f} "
                                  f"(Detected: {metrics['num_detected']}/{metrics['num_ground_truth']})")
            
            all_results[video_name] = video_results
        
        # Calculate and display aggregate statistics
        stats = calculate_aggregate_statistics(all_results)
        stats_all[sampling_key] = stats
        print_statistics(stats, sampling_key)
        
        # Create summary comparison plot
        create_summary_comparison_plot(stats, sampling_key, sampling_output_dir)
        
        # Save results to JSON
        results_file = os.path.join(sampling_output_dir, 'adaptive_detection_results.json')
        
        # Convert results to JSON-serializable format
        json_results = {}
        for video_name, video_results in all_results.items():
            json_results[video_name] = {}
            for feature_type, feature_results in video_results.items():
                json_results[video_name][feature_type] = {}
                for version, version_results in feature_results.items():
                    json_results[video_name][feature_type][version] = {
                        'detected_boundaries': [int(b) for b in version_results['detected_boundaries']],
                        'metrics': version_results['metrics'],
                        'parameters': version_results['parameters']
                    }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Save statistics to JSON
        stats_file = os.path.join(sampling_output_dir, 'adaptive_detection_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_file}")
    
    # Create cross-sampling comparison heatmap
    create_feature_comparison_heatmap(stats_all, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("ADAPTIVE DETECTION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - adaptive_detection_results.json: Per-video detection results")
    print(f"  - adaptive_detection_statistics.json: Aggregate statistics")
    print(f"  - summary_comparison_*.png: Performance comparison plots")
    print(f"  - f1_heatmap_comparison.png: F1 score heatmap across all conditions")
    print(f"  - Video directories (1-10): Individual detection visualizations")
    
    # Print best performing configurations
    print("\n" + "=" * 80)
    print("BEST PERFORMING CONFIGURATIONS:")
    print("=" * 80)
    
    best_f1 = 0
    best_config = None
    
    for sampling_key, stats in stats_all.items():
        for feature_type in FEATURE_TYPES:
            for version in ['raw', 'smoothed', 'normalized_smoothed']:
                f1 = stats[feature_type][version]['mean_f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = (sampling_key, feature_type, version)
    
    if best_config:
        sampling_key, feature_type, version = best_config
        print(f"\nBest overall: {feature_type.upper()} ({version}) on {sampling_key}")
        print(f"  Mean F1: {best_f1:.3f}")
        stats = stats_all[sampling_key][feature_type][version]
        print(f"  Mean Precision: {stats['mean_precision']:.3f} ± {stats['std_precision']:.3f}")
        print(f"  Mean Recall: {stats['mean_recall']:.3f} ± {stats['std_recall']:.3f}")


if __name__ == "__main__":
    main()
