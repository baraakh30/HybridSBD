"""
Test script for XGBoost Boundary Detector

Evaluates the trained XGBoost model on the test set using:
- Standard SBD metrics (precision, recall, F1)
- Tolerance-based evaluation (±3 frames)
- Per-video and aggregate statistics
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import argparse
import pickle
from datetime import datetime
from scipy.ndimage import uniform_filter1d

# Feature dimensions (must match training)
FEATURE_SPLITS = {
    'clip': (0, 512),
    'cnn': (512, 2560),
    'hsv': (2560, 2816),
    'flow': (2816, 3368),
}


def compute_features(features: np.ndarray, context_window: int = 7) -> Tuple[np.ndarray, List[str]]:
    """
    Compute all features for a video with enhanced feature engineering.
    Must match the training feature computation exactly.
    """
    diffs = features[1:] - features[:-1]
    num_frames = len(diffs)
    
    # Basic magnitudes
    total_mags = np.linalg.norm(diffs, axis=1)
    
    type_mags = {}
    for name, (start, end) in FEATURE_SPLITS.items():
        if end <= features.shape[1]:
            type_mags[name] = np.linalg.norm(diffs[:, start:end], axis=1)
        else:
            # Handle case where features might be shorter
            actual_end = min(end, features.shape[1])
            type_mags[name] = np.linalg.norm(diffs[:, start:actual_end], axis=1)
    
    # Cosine similarity between consecutive diffs
    cosine_sims = np.zeros(num_frames)
    for i in range(1, num_frames):
        n1, n2 = np.linalg.norm(diffs[i-1]), np.linalg.norm(diffs[i])
        if n1 > 1e-8 and n2 > 1e-8:
            cosine_sims[i] = np.dot(diffs[i-1], diffs[i]) / (n1 * n2)
    
    # Magnitude acceleration (first derivative)
    mag_diff = np.zeros(num_frames)
    mag_diff[1:] = total_mags[1:] - total_mags[:-1]
    
    # Second derivative of magnitude
    mag_diff2 = np.zeros(num_frames)
    mag_diff2[1:] = mag_diff[1:] - mag_diff[:-1]
    
    # Local statistics with multiple window sizes
    local_stats = {}
    for window in [context_window, context_window * 2]:
        zscore = np.zeros(num_frames)
        max_ratio = np.zeros(num_frames)
        rank = np.zeros(num_frames)
        percentile_90 = np.zeros(num_frames)
        std_ratio = np.zeros(num_frames)
        
        for i in range(num_frames):
            s = max(0, i - window)
            e = min(num_frames, i + window + 1)
            local = total_mags[s:e]
            
            local_mean = local.mean()
            local_std = local.std() + 1e-8
            zscore[i] = (total_mags[i] - local_mean) / local_std
            max_ratio[i] = total_mags[i] / (local.max() + 1e-8)
            rank[i] = (local < total_mags[i]).sum() / len(local)
            percentile_90[i] = 1.0 if total_mags[i] > np.percentile(local, 90) else 0.0
            std_ratio[i] = total_mags[i] / local_std
        
        suffix = f'_w{window}'
        local_stats[f'zscore{suffix}'] = zscore
        local_stats[f'max_ratio{suffix}'] = max_ratio
        local_stats[f'rank{suffix}'] = rank
        local_stats[f'percentile_90{suffix}'] = percentile_90
        local_stats[f'std_ratio{suffix}'] = std_ratio
    
    # Per-type z-scores and statistics
    type_zscores = {}
    type_ranks = {}
    for name, mags in type_mags.items():
        zscores = np.zeros(num_frames)
        ranks = np.zeros(num_frames)
        for i in range(num_frames):
            s = max(0, i - context_window)
            e = min(num_frames, i + context_window + 1)
            local = mags[s:e]
            zscores[i] = (mags[i] - local.mean()) / (local.std() + 1e-8)
            ranks[i] = (local < mags[i]).sum() / len(local)
        type_zscores[name] = zscores
        type_ranks[name] = ranks
    
    # Feature ratios
    clip_cnn_ratio = (type_mags['clip'] + 1e-8) / (type_mags['cnn'] + 1e-8)
    hsv_flow_ratio = (type_mags['hsv'] + 1e-8) / (type_mags['flow'] + 1e-8)
    hsv_total_ratio = (type_mags['hsv'] + 1e-8) / (total_mags + 1e-8)
    clip_total_ratio = (type_mags['clip'] + 1e-8) / (total_mags + 1e-8)
    
    # Peak detection features
    is_local_peak = np.zeros(num_frames)
    peak_prominence = np.zeros(num_frames)
    for i in range(1, num_frames - 1):
        if total_mags[i] > total_mags[i-1] and total_mags[i] > total_mags[i+1]:
            is_local_peak[i] = 1.0
            peak_prominence[i] = total_mags[i] - max(total_mags[i-1], total_mags[i+1])
    
    # Global statistics
    global_mean = total_mags.mean()
    global_std = total_mags.std() + 1e-8
    global_zscore = (total_mags - global_mean) / global_std
    global_percentile = np.array([np.searchsorted(np.sort(total_mags), m) / num_frames for m in total_mags])
    
    # Gradients (before and after)
    grad_before = np.zeros(num_frames)
    grad_after = np.zeros(num_frames)
    grad_before[1:] = total_mags[1:] - total_mags[:-1]
    grad_after[:-1] = total_mags[1:] - total_mags[:-1]
    
    # Gradient magnitude
    grad_magnitude = np.abs(grad_before) + np.abs(grad_after)
    
    # Rolling statistics
    rolling_mean_3 = uniform_filter1d(total_mags, size=3, mode='nearest')
    rolling_mean_5 = uniform_filter1d(total_mags, size=5, mode='nearest')
    deviation_from_rolling_3 = total_mags - rolling_mean_3
    deviation_from_rolling_5 = total_mags - rolling_mean_5
    
    # Per-type rolling deviations for HSV
    hsv_rolling_mean = uniform_filter1d(type_mags['hsv'], size=5, mode='nearest')
    hsv_deviation = type_mags['hsv'] - hsv_rolling_mean
    
    # Temporal context: features of neighbors
    prev_mag = np.zeros(num_frames)
    next_mag = np.zeros(num_frames)
    prev_mag[1:] = total_mags[:-1]
    next_mag[:-1] = total_mags[1:]
    mag_vs_prev = total_mags - prev_mag
    mag_vs_next = total_mags - next_mag
    
    # HSV-specific neighbor features
    hsv_prev = np.zeros(num_frames)
    hsv_next = np.zeros(num_frames)
    hsv_prev[1:] = type_mags['hsv'][:-1]
    hsv_next[:-1] = type_mags['hsv'][1:]
    hsv_vs_prev = type_mags['hsv'] - hsv_prev
    hsv_vs_next = type_mags['hsv'] - hsv_next
    
    # Combined peak score
    combined_peak_score = (
        local_stats['zscore_w7'] * 0.3 +
        local_stats['rank_w7'] * 0.3 +
        is_local_peak * 0.2 +
        type_zscores['hsv'] * 0.2
    )
    
    # Feature names
    feature_names = [
        'total_mag', 'clip_mag', 'cnn_mag', 'hsv_mag', 'flow_mag',
        'mag_diff', 'mag_diff2', 'cosine_sim',
        'grad_before', 'grad_after', 'grad_magnitude',
        'clip_cnn_ratio', 'hsv_flow_ratio', 'hsv_total_ratio', 'clip_total_ratio',
        'clip_zscore', 'cnn_zscore', 'hsv_zscore', 'flow_zscore',
        'clip_rank', 'cnn_rank', 'hsv_rank', 'flow_rank',
        'is_local_peak', 'peak_prominence', 'global_zscore', 'global_percentile',
        'zscore_w7', 'max_ratio_w7', 'rank_w7', 'percentile_90_w7', 'std_ratio_w7',
        'zscore_w14', 'max_ratio_w14', 'rank_w14', 'percentile_90_w14', 'std_ratio_w14',
        'deviation_from_rolling_3', 'deviation_from_rolling_5',
        'hsv_deviation', 'hsv_vs_prev', 'hsv_vs_next',
        'mag_vs_prev', 'mag_vs_next',
        'combined_peak_score',
    ]
    
    # Combine all features
    all_features = []
    for i in range(num_frames):
        feat = [
            total_mags[i],
            type_mags['clip'][i],
            type_mags['cnn'][i],
            type_mags['hsv'][i],
            type_mags['flow'][i],
            mag_diff[i],
            mag_diff2[i],
            cosine_sims[i],
            grad_before[i],
            grad_after[i],
            grad_magnitude[i],
            clip_cnn_ratio[i],
            hsv_flow_ratio[i],
            hsv_total_ratio[i],
            clip_total_ratio[i],
            type_zscores['clip'][i],
            type_zscores['cnn'][i],
            type_zscores['hsv'][i],
            type_zscores['flow'][i],
            type_ranks['clip'][i],
            type_ranks['cnn'][i],
            type_ranks['hsv'][i],
            type_ranks['flow'][i],
            is_local_peak[i],
            peak_prominence[i],
            global_zscore[i],
            global_percentile[i],
            local_stats['zscore_w7'][i],
            local_stats['max_ratio_w7'][i],
            local_stats['rank_w7'][i],
            local_stats['percentile_90_w7'][i],
            local_stats['std_ratio_w7'][i],
            local_stats['zscore_w14'][i],
            local_stats['max_ratio_w14'][i],
            local_stats['rank_w14'][i],
            local_stats['percentile_90_w14'][i],
            local_stats['std_ratio_w14'][i],
            deviation_from_rolling_3[i],
            deviation_from_rolling_5[i],
            hsv_deviation[i],
            hsv_vs_prev[i],
            hsv_vs_next[i],
            mag_vs_prev[i],
            mag_vs_next[i],
            combined_peak_score[i],
        ]
        all_features.append(feat)
    
    return np.array(all_features, dtype=np.float32), feature_names


def load_ground_truth(gt_path: str, return_ranges: bool = False) -> Dict:
    """
    Load ground truth annotations.
    
    Args:
        gt_path: Path to ground truth JSON file
        return_ranges: If True, return transition ranges; if False, return start frames only
        
    Returns:
        Dictionary mapping video names to transitions
    """
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    result = {}
    for video_name, video_data in gt_data.items():
        if isinstance(video_data, dict) and 'transitions' in video_data:
            transitions = video_data['transitions']
        elif isinstance(video_data, list):
            transitions = video_data
        else:
            continue
        
        # Filter invalid transitions
        valid_transitions = []
        for t in transitions:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                start, end = t[0], t[1]
                if start >= 0 and end > start:
                    if return_ranges:
                        valid_transitions.append((start, end))
                    else:
                        valid_transitions.append(start)  # Just start frame
            elif isinstance(t, (int, float)):
                if return_ranges:
                    valid_transitions.append((int(t), int(t)))
                else:
                    valid_transitions.append(int(t))
        
        result[video_name] = valid_transitions
    
    return result


def calculate_metrics_with_tolerance(
    predictions: List[int],
    ground_truth: List[Tuple[int, int]],
    tolerance: int = 3
) -> Tuple[int, int, int]:
    """
    Calculate TP, FP, FN with temporal tolerance for transition RANGES.
    
    A prediction matches a transition if it falls within [start - tol, end + tol].
    Multiple predictions within the same transition count as ONE TP.
    
    Args:
        predictions: List of predicted boundary frame indices
        ground_truth: List of (start, end) transition ranges
        tolerance: Frame tolerance for matching
        
    Returns:
        Tuple of (TP, FP, FN)
    """
    if not ground_truth:
        # No ground truth - all predictions are FP
        return 0, len(predictions), 0
    
    if not predictions:
        # No predictions - all GT are FN
        return 0, 0, len(ground_truth)
    
    pred_set = set(predictions)
    matched_predictions = set()
    matched_gt = set()
    
    # For each GT transition range, check if any prediction falls within tolerance
    for gt_idx, (gt_start, gt_end) in enumerate(ground_truth):
        match_start = gt_start - tolerance
        match_end = gt_end + tolerance
        
        for pred in pred_set:
            if match_start <= pred <= match_end:
                matched_predictions.add(pred)
                matched_gt.add(gt_idx)
                # Don't break - absorb all predictions in this range
    
    tp = len(matched_gt)  # Number of GT transitions detected
    fp = len(pred_set) - len(matched_predictions)  # Predictions outside any GT range
    fn = len(ground_truth) - tp  # GT transitions not detected
    
    return tp, fp, fn


def apply_nms(predictions: np.ndarray, probabilities: np.ndarray, window: int = 5) -> List[int]:
    """
    Apply non-maximum suppression to predictions.
    Keep only local maxima within the window.
    
    Args:
        predictions: Binary prediction array
        probabilities: Probability array
        window: NMS window size (±window/2)
        
    Returns:
        List of prediction indices after NMS
    """
    pred_indices = np.where(predictions == 1)[0]
    if len(pred_indices) == 0:
        return []
    
    kept = []
    half_window = window // 2
    
    for idx in pred_indices:
        start = max(0, idx - half_window)
        end = min(len(probabilities), idx + half_window + 1)
        
        # Check if this is a local maximum
        if probabilities[idx] >= max(probabilities[start:end]):
            kept.append(idx)
    
    return kept


def test_xgboost(
    features_dir: str = "testing_features",
    ground_truth: str = r"C:\Users\baraa\Documents\ClipShots\annotations\test.json",
    model_path: str = "models/boundary_detector_xgb.pkl",
    fusion_strategy: str = "concatenation",
    tolerance: int = 3,
    apply_nms_filter: bool = True,
    save_results: bool = True,
):
    """
    Test the XGBoost boundary detector on test data.
    """
    print("=" * 70)
    print("XGBOOST BOUNDARY DETECTOR - TEST EVALUATION")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = model_data['model']
    threshold = model_data['threshold']
    feature_names = model_data.get('feature_names', [])
    
    print(f"  Loaded threshold: {threshold:.4f}")
    print(f"  Number of features: {len(feature_names)}")
    
    # Load ground truth
    print(f"\nLoading ground truth from: {ground_truth}")
    gt_data = load_ground_truth(ground_truth, return_ranges=True)
    print(f"  Videos in ground truth: {len(gt_data)}")
    
    # Find available test videos
    features_path = Path(features_dir)
    available_videos = []
    
    for video_name in gt_data.keys():
        video_stem = video_name.replace('.mp4', '')
        # Try different naming patterns
        patterns = [
            f"{video_stem}_{fusion_strategy}.npy",
            f"{video_stem}___{fusion_strategy}.npy",
        ]
        for pattern in patterns:
            if (features_path / pattern).exists():
                available_videos.append(video_name)
                break
    
    print(f"  Videos with features: {len(available_videos)}")
    
    if not available_videos:
        print("Error: No test videos found with features!")
        return
    
    # Evaluate on each video
    print(f"\n{'=' * 70}")
    print("EVALUATION")
    print(f"{'=' * 70}")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    per_video_results = {}
    
    for video_name in tqdm(available_videos, desc="Testing"):
        video_stem = video_name.replace('.mp4', '')
        
        # Find and load features
        feat_file = None
        for suffix in ['_', '___']:
            path = features_path / f"{video_stem}{suffix}{fusion_strategy}.npy"
            if path.exists():
                feat_file = path
                break
        
        if feat_file is None:
            continue
        
        try:
            features = np.load(feat_file)
            if len(features) < 10:
                continue
            
            # Compute test features
            X, _ = compute_features(features)
            
            # Get predictions
            y_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
            
            # Apply NMS if requested
            if apply_nms_filter:
                pred_indices = apply_nms(y_pred, y_proba, window=5)
            else:
                pred_indices = list(np.where(y_pred == 1)[0])
            
            # Get ground truth for this video
            gt_transitions = gt_data.get(video_name, [])
            
            # Calculate metrics with tolerance
            tp, fp, fn = calculate_metrics_with_tolerance(
                pred_indices, gt_transitions, tolerance
            )
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Store per-video results
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            per_video_results[video_name] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_predictions': len(pred_indices),
                'num_gt': len(gt_transitions),
            }
            
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue
    
    # Calculate aggregate metrics
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nTest Configuration:")
    print(f"  Features directory: {features_dir}")
    print(f"  Fusion strategy: {fusion_strategy}")
    print(f"  Tolerance: ±{tolerance} frames")
    print(f"  NMS applied: {apply_nms_filter}")
    print(f"  Classification threshold: {threshold:.4f}")
    
    print(f"\nAggregate Results:")
    print(f"  Videos evaluated: {len(per_video_results)}")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Per-video statistics
    if per_video_results:
        f1_scores = [r['f1'] for r in per_video_results.values()]
        print(f"\nPer-Video F1 Statistics:")
        print(f"  Mean F1: {np.mean(f1_scores):.4f}")
        print(f"  Median F1: {np.median(f1_scores):.4f}")
        print(f"  Std F1: {np.std(f1_scores):.4f}")
        print(f"  Min F1: {np.min(f1_scores):.4f}")
        print(f"  Max F1: {np.max(f1_scores):.4f}")
        
        # Show worst performing videos
        sorted_videos = sorted(per_video_results.items(), key=lambda x: x[1]['f1'])
        print(f"\nWorst 5 Videos:")
        for video_name, results in sorted_videos[:5]:
            print(f"  {video_name}: F1={results['f1']:.3f} "
                  f"(P={results['precision']:.3f}, R={results['recall']:.3f}, "
                  f"Pred={results['num_predictions']}, GT={results['num_gt']})")
        
        # Show best performing videos
        print(f"\nBest 5 Videos:")
        for video_name, results in sorted_videos[-5:][::-1]:
            print(f"  {video_name}: F1={results['f1']:.3f} "
                  f"(P={results['precision']:.3f}, R={results['recall']:.3f}, "
                  f"Pred={results['num_predictions']}, GT={results['num_gt']})")
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results/xgboost_test_results_{timestamp}.json"
        Path("test_results").mkdir(exist_ok=True)
        
        results_data = {
            'config': {
                'features_dir': features_dir,
                'ground_truth': ground_truth,
                'model_path': model_path,
                'fusion_strategy': fusion_strategy,
                'tolerance': tolerance,
                'threshold': threshold,
                'apply_nms': apply_nms_filter,
            },
            'aggregate': {
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_videos': len(per_video_results),
            },
            'per_video': per_video_results,
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_video': per_video_results,
    }


def test_multiple_thresholds(
    features_dir: str = "testing_features",
    ground_truth: str = r"C:\Users\baraa\Documents\ClipShots\annotations\test.json",
    model_path: str = "models/boundary_detector_xgb.pkl",
    fusion_strategy: str = "concatenation",
    tolerance: int = 3,
):
    """Test with multiple thresholds to find optimal."""
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    trained_threshold = model_data['threshold']
    
    # Load ground truth
    gt_data = load_ground_truth(ground_truth, return_ranges=True)
    
    # Find available videos
    features_path = Path(features_dir)
    test_videos = []
    for video_name in gt_data.keys():
        video_stem = video_name.replace('.mp4', '')
        for suffix in ['_', '___']:
            if (features_path / f"{video_stem}{suffix}{fusion_strategy}.npy").exists():
                test_videos.append(video_name)
                break
    
    print(f"Testing with {len(test_videos)} videos")
    
    # Collect all probabilities and labels
    all_proba = []
    all_gt_ranges = []
    video_lengths = []
    
    for video_name in tqdm(test_videos, desc="Loading"):
        video_stem = video_name.replace('.mp4', '')
        
        for suffix in ['_', '___']:
            feat_file = features_path / f"{video_stem}{suffix}{fusion_strategy}.npy"
            if feat_file.exists():
                break
        
        try:
            features = np.load(feat_file)
            if len(features) < 10:
                continue
            
            X, _ = compute_features(features)
            proba = model.predict_proba(X)[:, 1]
            
            all_proba.append(proba)
            all_gt_ranges.append(gt_data.get(video_name, []))
            video_lengths.append(len(proba))
        except:
            continue
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, trained_threshold]
    thresholds = sorted(set(thresholds))
    
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)
    
    best_f1 = 0
    best_thresh = 0
    
    for thresh in thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for proba, gt_ranges in zip(all_proba, all_gt_ranges):
            preds = (proba >= thresh).astype(int)
            pred_indices = apply_nms(preds, proba, window=5)
            tp, fp, fn = calculate_metrics_with_tolerance(pred_indices, gt_ranges, tolerance)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        marker = " <-- TRAINED" if abs(thresh - trained_threshold) < 0.001 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            marker = " <-- BEST" + marker
        
        print(f"{thresh:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}{marker}")
    
    print(f"\nBest threshold: {best_thresh:.4f} with F1={best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test XGBoost Boundary Detector")
    parser.add_argument("--features_dir", default="testing_features",
                        help="Directory containing test features")
    parser.add_argument("--ground_truth", 
                        default=r"C:\Users\baraa\Documents\ClipShots\annotations\test.json",
                        help="Path to test ground truth JSON")
    parser.add_argument("--model_path", default="models/boundary_detector_xgb.pkl",
                        help="Path to trained model")
    parser.add_argument("--fusion_strategy", default="concatenation",
                        help="Feature fusion strategy used")
    parser.add_argument("--tolerance", type=int, default=3,
                        help="Frame tolerance for matching")
    parser.add_argument("--no-nms", action="store_true",
                        help="Disable NMS filtering")
    parser.add_argument("--threshold-analysis", action="store_true",
                        help="Run threshold analysis instead of single evaluation")
    
    args = parser.parse_args()
    
    if args.threshold_analysis:
        test_multiple_thresholds(
            features_dir=args.features_dir,
            ground_truth=args.ground_truth,
            model_path=args.model_path,
            fusion_strategy=args.fusion_strategy,
            tolerance=args.tolerance,
        )
    else:
        test_xgboost(
            features_dir=args.features_dir,
            ground_truth=args.ground_truth,
            model_path=args.model_path,
            fusion_strategy=args.fusion_strategy,
            tolerance=args.tolerance,
            apply_nms_filter=not args.no_nms,
        )
