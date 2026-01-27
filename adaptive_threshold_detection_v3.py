"""
Locally Adaptive Robust Shot Boundary Detection - V3 (Strict Algorithm + Grid Search)

This version:
1. Strictly follows the exact algorithm from the paper
2. Performs grid search to find optimal parameters for each feature type
3. No additional filtering beyond what's specified in the algorithm
4. Reports best parameters found per feature/sampling combination
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_DIRS = {
    'no_sampling': 'training_features_no_sampling',
    'sampling_8': 'training_features_sampling_8'
}

OUTPUT_DIR = 'adaptive_threshold_detection_v3_results'

# Feature types to analyze
FEATURE_TYPES = ['cnn', 'hsv_histogram', 'clip', 'optical_flow']

# Minimum scene length constraints (from paper: ~2 seconds)
MIN_SCENE_LENGTH = {
    'no_sampling': 50,   # ~2 seconds at 25 FPS
    'sampling_8': 6      # ~2 seconds at ~3 FPS
}

# Maximum dissolve window
MAX_DISSOLVE_WINDOW = 5

# Stability constant (epsilon)
EPSILON = 1e-6

# Evaluation tolerance
EVALUATION_TOLERANCE = 5

# Grid search parameter ranges
PARAM_GRID = {
    'w': [3, 5, 7, 10],           # Local window radius
    'k': [2, 3, 4],               # Statistics window multiplier  
    'alpha': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # Sensitivity factor
}


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
    """
    median = np.median(data)
    return np.median(np.abs(data - median))


def consecutive_segments(indices: List[int]) -> List[List[int]]:
    """Group consecutive indices into segments."""
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
    C: np.ndarray,
    w: int,
    k: int,
    alpha: float,
    L_min: int,
    W_max: int,
    epsilon: float = EPSILON
) -> Tuple[List[int], np.ndarray]:
    """
    Algorithm: Locally Adaptive Robust Shot Boundary Detection
    
    Exactly as specified in the paper:
    
    Require:
        C: Similarity array
        w: Local window radius
        k: Statistics window multiplier
        alpha: Sensitivity factor
        L_min: Minimum scene length
        W_max: Maximum dissolve window
        epsilon: Stability constant
    
    Ensure:
        B: List of boundary indices
    
    Args:
        C: Similarity array (correlation between consecutive frames)
        w: Local window radius
        k: Statistics window multiplier
        alpha: Sensitivity factor
        L_min: Minimum scene length
        W_max: Maximum dissolve window
        epsilon: Stability constant
        
    Returns:
        Tuple of (boundary_indices, normalized_scores)
    """
    n = len(C)
    kw = k * w  # Statistics window half-width
    
    # Initialize normalized scores array
    C_hat = np.zeros(n)
    
    # Step 1: candidate_indices ← ∅
    candidate_indices = []
    
    # Step 2-11: Main loop for candidate detection
    # for i = k·w to |C| − k·w do
    for i in range(kw, n - kw):
        # Step 3: W_i ← C[i − k·w : i + k·w]
        W_i = C[i - kw : i + kw + 1]  # +1 for inclusive end
        
        # Step 4: med_i ← median(W_i)
        med_i = np.median(W_i)
        
        # Step 5: mad_i ← max(MAD(W_i), ε)  ▷ Prevent MAD collapse
        mad_i = max(compute_mad(W_i), epsilon)
        
        # Step 6: Ĉ_i ← (C[i] − med_i) / mad_i
        C_hat[i] = (C[i] - med_i) / mad_i
        
        # Step 7: θ_i ← −α  ▷ Adaptive local threshold
        theta_i = -alpha
        
        # Step 8-10: if Ĉ_i < θ_i then candidate_indices.append(i)
        if C_hat[i] < theta_i:
            candidate_indices.append(i)
    
    # Handle edge regions (frames outside k·w boundary)
    # Use available window for edge frames
    for i in range(0, kw):
        W_i = C[0 : i + kw + 1]
        if len(W_i) > 1:
            med_i = np.median(W_i)
            mad_i = max(compute_mad(W_i), epsilon)
            C_hat[i] = (C[i] - med_i) / mad_i
            if C_hat[i] < -alpha:
                candidate_indices.append(i)
    
    for i in range(n - kw, n):
        W_i = C[i - kw : n]
        if len(W_i) > 1:
            med_i = np.median(W_i)
            mad_i = max(compute_mad(W_i), epsilon)
            C_hat[i] = (C[i] - med_i) / mad_i
            if C_hat[i] < -alpha:
                candidate_indices.append(i)
    
    # Remove duplicates and sort
    candidate_indices = sorted(set(candidate_indices))
    
    # Step 12: Group consecutive candidate indices
    # groups ← consecutive_segments(candidate_indices)
    groups = consecutive_segments(candidate_indices)
    
    # Step 13-22: Identify boundaries with dissolve handling
    # boundaries ← ∅
    boundaries = []
    
    for group in groups:
        # Step 15-17: if |group| ≤ W_max then b ← argmin Ĉ_i within group
        if len(group) <= W_max:
            # Find frame with minimum normalized score in group
            min_score = float('inf')
            best_boundary = group[0]
            for idx in group:
                if C_hat[idx] < min_score:
                    min_score = C_hat[idx]
                    best_boundary = idx
            boundaries.append(best_boundary)
        else:
            # Step 18-21: Dissolve handling - take start and end
            boundaries.append(group[0])    # Dissolve start
            boundaries.append(group[-1])   # Dissolve end
    
    # Step 23-29: Enforce minimum scene length
    if len(boundaries) == 0:
        return [], C_hat
    
    # B ← [boundaries[0]]
    B = [boundaries[0]]
    
    # for each b in boundaries[1:] do
    for b in boundaries[1:]:
        # if b − B[−1] > L_min then B.append(b)
        if b - B[-1] >= L_min:
            B.append(b)
    
    # Step 30: return B
    return B, C_hat


def evaluate_detection(
    detected_boundaries: List[int],
    ground_truth_boundaries: List[int],
    tolerance: int = EVALUATION_TOLERANCE
) -> Dict:
    """Evaluate detection performance against ground truth."""
    if len(ground_truth_boundaries) == 0:
        return {
            'precision': 0.0 if len(detected_boundaries) > 0 else 1.0,
            'recall': 1.0,
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
    
    # Match detected to ground truth (greedy matching)
    true_positives = 0
    matched_gt = set()
    matched_detected = set()
    
    for gt_boundary in sorted(ground_truth_boundaries):
        best_match = None
        best_dist = float('inf')
        
        for detected_boundary in sorted(detected_boundaries):
            if detected_boundary in matched_detected:
                continue
            dist = abs(detected_boundary - gt_boundary)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_match = detected_boundary
        
        if best_match is not None:
            true_positives += 1
            matched_gt.add(gt_boundary)
            matched_detected.add(best_match)
    
    false_positives = len(detected_boundaries) - len(matched_detected)
    false_negatives = len(ground_truth_boundaries) - len(matched_gt)
    
    precision = true_positives / len(detected_boundaries) if len(detected_boundaries) > 0 else 0.0
    recall = true_positives / len(ground_truth_boundaries) if len(ground_truth_boundaries) > 0 else 0.0
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


def grid_search_parameters(
    all_video_data: List[Tuple[str, np.ndarray, List[int]]],
    sampling_key: str,
    feature_type: str
) -> Dict:
    """
    Perform grid search to find optimal parameters.
    
    Args:
        all_video_data: List of (video_name, correlation_array, ground_truth_boundaries)
        sampling_key: Sampling rate key
        feature_type: Feature type being processed
        
    Returns:
        Dict with best parameters and performance metrics
    """
    L_min = MIN_SCENE_LENGTH[sampling_key]
    
    best_f1 = 0
    best_params = None
    best_metrics_avg = None
    all_results = []
    
    total_combinations = len(PARAM_GRID['w']) * len(PARAM_GRID['k']) * len(PARAM_GRID['alpha'])
    print(f"    Testing {total_combinations} parameter combinations...")
    
    for w in PARAM_GRID['w']:
        for k in PARAM_GRID['k']:
            for alpha in PARAM_GRID['alpha']:
                # Evaluate on all videos
                all_f1 = []
                all_precision = []
                all_recall = []
                
                for video_name, correlation, gt_boundaries in all_video_data:
                    detected, _ = locally_adaptive_detection(
                        C=correlation,
                        w=w,
                        k=k,
                        alpha=alpha,
                        L_min=L_min,
                        W_max=MAX_DISSOLVE_WINDOW,
                        epsilon=EPSILON
                    )
                    
                    metrics = evaluate_detection(detected, gt_boundaries)
                    all_f1.append(metrics['f1'])
                    all_precision.append(metrics['precision'])
                    all_recall.append(metrics['recall'])
                
                mean_f1 = np.mean(all_f1)
                mean_precision = np.mean(all_precision)
                mean_recall = np.mean(all_recall)
                
                all_results.append({
                    'w': w, 'k': k, 'alpha': alpha,
                    'mean_f1': mean_f1,
                    'mean_precision': mean_precision,
                    'mean_recall': mean_recall,
                    'std_f1': np.std(all_f1)
                })
                
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_params = {'w': w, 'k': k, 'alpha': alpha}
                    best_metrics_avg = {
                        'mean_f1': mean_f1,
                        'std_f1': np.std(all_f1),
                        'mean_precision': mean_precision,
                        'std_precision': np.std(all_precision),
                        'mean_recall': mean_recall,
                        'std_recall': np.std(all_recall)
                    }
    
    return {
        'best_params': best_params,
        'best_metrics': best_metrics_avg,
        'all_results': all_results
    }


def plot_detection_results(
    video_name: str,
    video_id: int,
    correlation: np.ndarray,
    normalized_scores: np.ndarray,
    detected_boundaries: List[int],
    ground_truth_boundaries: List[int],
    params: Dict,
    feature_type: str,
    version: str,
    output_dir: str,
    metrics: Dict
):
    """Create visualization of detection results."""
    frames = np.arange(len(correlation))
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Panel 1: Raw correlation
    ax1 = axes[0]
    ax1.plot(frames, correlation, linewidth=1.2, color='blue', alpha=0.8, label='Correlation C')
    for i, boundary in enumerate(ground_truth_boundaries):
        if boundary < len(correlation):
            ax1.axvline(x=boundary, color='green', linestyle='-', linewidth=2, alpha=0.6,
                       label='Ground Truth' if i == 0 else '')
    ax1.set_ylabel('Correlation', fontsize=11)
    ax1.set_title('Similarity Signal C', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Normalized scores with threshold
    ax2 = axes[1]
    ax2.plot(frames, normalized_scores, linewidth=1.2, color='purple', alpha=0.8, label='Ĉ (normalized)')
    ax2.axhline(y=-params['alpha'], color='red', linestyle='--', linewidth=2, alpha=0.8,
                label=f'θ = −α = {-params["alpha"]:.1f}')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Highlight regions below threshold
    below_threshold = normalized_scores < -params['alpha']
    ax2.fill_between(frames, normalized_scores, -params['alpha'],
                    where=below_threshold, alpha=0.3, color='red', label='Candidates')
    
    for boundary in ground_truth_boundaries:
        if boundary < len(normalized_scores):
            ax2.axvline(x=boundary, color='green', linestyle='-', linewidth=2, alpha=0.6)
    
    ax2.set_ylabel('Normalized Score Ĉ', fontsize=11)
    ax2.set_title(f'Locally Normalized Deviation (w={params["w"]}, k={params["k"]})', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Detection results
    ax3 = axes[2]
    ax3.plot(frames, correlation, linewidth=1.2, alpha=0.5, color='blue', label='Correlation')
    
    for i, boundary in enumerate(ground_truth_boundaries):
        if boundary < len(correlation):
            ax3.axvline(x=boundary, color='green', linestyle='-', linewidth=2.5, alpha=0.7,
                       label='Ground Truth' if i == 0 else '')
    
    for i, boundary in enumerate(detected_boundaries):
        if boundary < len(correlation):
            ax3.axvline(x=boundary, color='red', linestyle='--', linewidth=2.5, alpha=0.7,
                       label='Detected' if i == 0 else '')
    
    ax3.set_xlabel('Frame Index', fontsize=12)
    ax3.set_ylabel('Correlation', fontsize=11)
    ax3.set_title(f'Detection Results: P={metrics["precision"]:.3f}, R={metrics["recall"]:.3f}, F1={metrics["f1"]:.3f} | '
                 f'Detected: {metrics["num_detected"]}, GT: {metrics["num_ground_truth"]}',
                 fontsize=12, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(f'Video {video_id} - {feature_type.upper()} ({version})\n'
                f'Parameters: w={params["w"]}, k={params["k"]}, α={params["alpha"]:.1f}, '
                f'L_min={MIN_SCENE_LENGTH.get("no_sampling", 50)}, W_max={MAX_DISSOLVE_WINDOW}',
                fontsize=13, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{feature_type}_{version}_detection.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def create_parameter_heatmap(grid_results: Dict, sampling_key: str, output_dir: str):
    """Create heatmaps showing parameter sensitivity for each feature."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, feature_type in enumerate(FEATURE_TYPES):
        ax = axes[idx]
        
        if feature_type not in grid_results:
            continue
        
        results = grid_results[feature_type]['all_results']
        
        # Create pivot table for alpha vs w (averaging over k)
        alphas = sorted(set(r['alpha'] for r in results))
        ws = sorted(set(r['w'] for r in results))
        
        matrix = np.zeros((len(alphas), len(ws)))
        
        for i, alpha in enumerate(alphas):
            for j, w in enumerate(ws):
                # Average F1 over all k values
                f1_values = [r['mean_f1'] for r in results if r['alpha'] == alpha and r['w'] == w]
                matrix[i, j] = np.mean(f1_values) if f1_values else 0
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(len(alphas)):
            for j in range(len(ws)):
                ax.text(j, i, f'{matrix[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=8)
        
        ax.set_xticks(np.arange(len(ws)))
        ax.set_xticklabels([str(w) for w in ws])
        ax.set_yticks(np.arange(len(alphas)))
        ax.set_yticklabels([f'{a:.1f}' for a in alphas])
        ax.set_xlabel('Window Radius (w)', fontsize=10)
        ax.set_ylabel('Sensitivity (α)', fontsize=10)
        ax.set_title(f'{feature_type.upper()}', fontsize=11, fontweight='bold')
    
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Mean F1-Score', shrink=0.8)
    fig.suptitle(f'Parameter Sensitivity - {sampling_key.upper().replace("_", " ")}\n(averaged over k values)',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parameter_heatmap_{sampling_key}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    print("=" * 80)
    print("LOCALLY ADAPTIVE ROBUST SHOT BOUNDARY DETECTION - V3")
    print("(Strict Algorithm Implementation + Grid Search)")
    print("=" * 80)
    
    print(f"\nAlgorithm: Exact implementation from paper")
    print(f"Grid Search Parameters:")
    print(f"  Window radius (w): {PARAM_GRID['w']}")
    print(f"  Stats multiplier (k): {PARAM_GRID['k']}")
    print(f"  Sensitivity (α): {PARAM_GRID['alpha']}")
    print(f"  Min scene lengths: {MIN_SCENE_LENGTH}")
    print(f"  Max dissolve window: {MAX_DISSOLVE_WINDOW}")
    print(f"  Stability constant (ε): {EPSILON}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nLoading selected videos...")
    selected_videos = load_selected_videos()
    print(f"Found {len(selected_videos)} videos")
    
    video_id_map = {video[0]: idx + 1 for idx, video in enumerate(selected_videos)}
    
    # Store best parameters found for each configuration
    best_params_all = {}
    
    for sampling_key in FEATURES_DIRS.keys():
        print("\n" + "=" * 80)
        print(f"PROCESSING: {sampling_key.upper().replace('_', ' ')}")
        print("=" * 80)
        
        print(f"Loading analysis results...")
        analysis_results = load_analysis_results(sampling_key)
        
        sampling_output_dir = os.path.join(OUTPUT_DIR, sampling_key)
        os.makedirs(sampling_output_dir, exist_ok=True)
        
        # Prepare data for grid search
        video_data_by_feature = {ft: [] for ft in FEATURE_TYPES}
        
        for video_name, frame_count, transitions, boundaries in selected_videos:
            video_data = analysis_results[video_name]
            
            for feature_type in FEATURE_TYPES:
                if feature_type in video_data['results']:
                    # Use smoothed correlation (typically better)
                    correlation = np.array(video_data['results'][feature_type]['correlation_smooth'])
                    video_data_by_feature[feature_type].append((video_name, correlation, boundaries))
        
        # Grid search for each feature type
        grid_results = {}
        best_params_all[sampling_key] = {}
        
        print(f"\n--- GRID SEARCH FOR OPTIMAL PARAMETERS ---")
        for feature_type in FEATURE_TYPES:
            print(f"\n  {feature_type.upper()}:")
            
            grid_result = grid_search_parameters(
                video_data_by_feature[feature_type],
                sampling_key,
                feature_type
            )
            
            grid_results[feature_type] = grid_result
            best_params_all[sampling_key][feature_type] = grid_result['best_params']
            
            bp = grid_result['best_params']
            bm = grid_result['best_metrics']
            print(f"    Best: w={bp['w']}, k={bp['k']}, α={bp['alpha']:.1f}")
            print(f"    Performance: F1={bm['mean_f1']:.3f}±{bm['std_f1']:.3f}, "
                  f"P={bm['mean_precision']:.3f}, R={bm['mean_recall']:.3f}")
        
        # Create parameter heatmap
        create_parameter_heatmap(grid_results, sampling_key, sampling_output_dir)
        
        # Now run detection with best parameters for each video
        print(f"\n--- RUNNING DETECTION WITH OPTIMAL PARAMETERS ---")
        
        all_results = {}
        all_metrics_by_feature = {ft: {'f1': [], 'precision': [], 'recall': []} for ft in FEATURE_TYPES}
        
        for video_name, frame_count, transitions, boundaries in selected_videos:
            video_id = video_id_map[video_name]
            print(f"\n  Video {video_id} ({video_name})")
            print(f"    Ground truth: {len(boundaries)} boundaries")
            
            video_data = analysis_results[video_name]
            video_output_dir = os.path.join(sampling_output_dir, str(video_id))
            os.makedirs(video_output_dir, exist_ok=True)
            
            video_results = {}
            
            for feature_type in FEATURE_TYPES:
                if feature_type not in video_data['results']:
                    continue
                
                # Get optimal parameters
                params = best_params_all[sampling_key][feature_type]
                
                # Process both raw and smoothed versions
                correlation_raw = np.array(video_data['results'][feature_type]['correlation'])
                correlation_smooth = np.array(video_data['results'][feature_type]['correlation_smooth'])
                
                feature_results = {}
                
                for version, correlation in [('raw', correlation_raw), ('smoothed', correlation_smooth)]:
                    detected, normalized_scores = locally_adaptive_detection(
                        C=correlation,
                        w=params['w'],
                        k=params['k'],
                        alpha=params['alpha'],
                        L_min=MIN_SCENE_LENGTH[sampling_key],
                        W_max=MAX_DISSOLVE_WINDOW,
                        epsilon=EPSILON
                    )
                    
                    metrics = evaluate_detection(detected, boundaries)
                    
                    plot_detection_results(
                        video_name, video_id, correlation,
                        normalized_scores, detected, boundaries,
                        params, feature_type, version,
                        video_output_dir, metrics
                    )
                    
                    feature_results[version] = {
                        'detected_boundaries': detected,
                        'metrics': metrics,
                        'parameters': params.copy()
                    }
                    
                    if version == 'smoothed':
                        all_metrics_by_feature[feature_type]['f1'].append(metrics['f1'])
                        all_metrics_by_feature[feature_type]['precision'].append(metrics['precision'])
                        all_metrics_by_feature[feature_type]['recall'].append(metrics['recall'])
                        
                        print(f"    {feature_type:15} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, "
                              f"F1: {metrics['f1']:.3f} ({metrics['num_detected']}/{metrics['num_ground_truth']})")
                
                video_results[feature_type] = feature_results
            
            all_results[video_name] = video_results
        
        # Print aggregate statistics
        print(f"\n{'='*80}")
        print(f"AGGREGATE STATISTICS - {sampling_key.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        for feature_type in FEATURE_TYPES:
            m = all_metrics_by_feature[feature_type]
            params = best_params_all[sampling_key][feature_type]
            print(f"\n{feature_type.upper()} (w={params['w']}, k={params['k']}, α={params['alpha']:.1f}):")
            print(f"  F1:        {np.mean(m['f1']):.3f} ± {np.std(m['f1']):.3f}")
            print(f"  Precision: {np.mean(m['precision']):.3f} ± {np.std(m['precision']):.3f}")
            print(f"  Recall:    {np.mean(m['recall']):.3f} ± {np.std(m['recall']):.3f}")
        
        # Save results
        results_file = os.path.join(sampling_output_dir, 'detection_results.json')
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
        
        # Save best parameters
        params_file = os.path.join(sampling_output_dir, 'best_parameters.json')
        with open(params_file, 'w') as f:
            json.dump({
                'parameters': best_params_all[sampling_key],
                'grid_search_results': {ft: {'best_params': r['best_params'], 'best_metrics': r['best_metrics']} 
                                        for ft, r in grid_results.items()}
            }, f, indent=2)
        
        print(f"\nResults saved to: {sampling_output_dir}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("OPTIMAL PARAMETERS SUMMARY")
    print("=" * 80)
    
    for sampling_key in FEATURES_DIRS.keys():
        print(f"\n{sampling_key.upper().replace('_', ' ')}:")
        for feature_type in FEATURE_TYPES:
            params = best_params_all[sampling_key][feature_type]
            print(f"  {feature_type:15}: w={params['w']:2}, k={params['k']}, α={params['alpha']:.1f}")


if __name__ == "__main__":
    main()
