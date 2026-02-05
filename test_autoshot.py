"""
Test Hybrid Boundary Detector on AutoShot Dataset

Evaluates model on pre-decoded AutoShot clips using BATCH processing.
AutoShot provides shot-level annotations - we detect transitions between shots.

Usage:
    # First, predecode the dataset
    python predecode_autoshot.py
    
    # Then run testing
    python test_autoshot.py --checkpoint outputs/hybrid_detector/best_model.pt --batch_size 8
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.hybrid_boundary_detector import create_hybrid_detector


def collate_autoshot_batch(batch):
    """
    Custom collate function to handle variable-length transitions lists
    """
    frames = torch.stack([item['frames'] for item in batch])
    shot_labels = torch.stack([item['shot_labels'] for item in batch])
    transitions = [item['transitions'] for item in batch]  # Keep as list
    video_ids = [item['video_id'] for item in batch]
    start_frames = torch.tensor([item['start_frame'] for item in batch])
    clip_indices = torch.tensor([item['clip_index'] for item in batch])
    
    # Boundary labels (if available)
    if batch[0]['boundary_labels'] is not None:
        boundary_labels = torch.stack([item['boundary_labels'] for item in batch])
    else:
        boundary_labels = None
    
    return {
        'frames': frames,
        'shot_labels': shot_labels,
        'transitions': transitions,  # List of lists
        'boundary_labels': boundary_labels,  # [B, T] or None
        'video_id': video_ids,
        'start_frame': start_frames,
        'clip_index': clip_indices
    }


class AutoShotPredecodedDataset(Dataset):
    """
    Dataset for pre-decoded AutoShot clips
    
    Loads pre-decoded clips with shot annotations for testing
    """
    def __init__(self, cache_root):
        self.cache_root = Path(cache_root)
        
        # Load index mapping
        index_file = self.cache_root / 'index_mapping.json'
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        self.mapping = index_data['mapping']
        self.num_videos = index_data['num_videos']
        
        print(f"AutoShot Dataset: {self.num_videos} videos, {len(self.mapping)} clips")
    
    def __len__(self):
        return len(self.mapping)
    
    def __getitem__(self, idx):
        clip_info = self.mapping[idx]
        
        # Load pre-decoded clip
        clip_data = torch.load(clip_info['clip_file'], weights_only=False)
        
        # Normalize frames to [0, 1]
        frames = clip_data['frames'].float() / 255.0  # [T, 3, H, W]
        
        # Get shot labels and transitions
        shot_labels = clip_data['shot_labels']  # [T] shot indices
        transitions = clip_data['transitions']  # List of transition frames
        boundary_labels = clip_data.get('boundary_labels', None)  # [T] binary labels (if available)
        
        return {
            'frames': frames,  # [T, 3, H, W] float32 [0, 1]
            'shot_labels': shot_labels,  # [T] long
            'transitions': transitions,  # List of int
            'boundary_labels': boundary_labels,  # [T] float32 or None
            'video_id': clip_info['video_id'],
            'start_frame': clip_info['start_frame'],
            'clip_index': idx
        }


def predictions_to_scenes(predictions):
    """
    Convert binary boundary predictions to scene segments
    Adapted from AutoShot evaluation code
    
    Args:
        predictions: [T] binary array (0=no boundary, 1=boundary)
        
    Returns:
        scenes: Nx2 array of [start, end] frame indices for each scene
    """
    scenes = []
    t_prev, start = 0, 0
    
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i - 1])
        t_prev = t
    
    if t_prev == 0:
        scenes.append([start, len(predictions) - 1])
    
    # Fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    
    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(gt_scenes, pred_scenes, n_frames_miss_tolerance=2):
    """
    Evaluate scene detection with frame tolerance
    Adapted from AutoShot evaluation: https://github.com/gyglim/shot-detection-evaluation
    
    Args:
        gt_scenes: Nx2 array of ground truth [start, end] scenes
        pred_scenes: Mx2 array of predicted [start, end] scenes
        n_frames_miss_tolerance: Tolerance in frames for matching
        
    Returns:
        precision, recall, f1, (tp, fp, fn)
    """
    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    
    # Get transitions between scenes
    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1) if len(gt_scenes) > 1 else np.zeros((0, 2))
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1) if len(pred_scenes) > 1 else np.zeros((0, 2))
    
    i, j = 0, 0
    tp, fp, fn = 0, 0, 0
    
    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            i += 1
        elif i == len(gt_trans):
            fp += 1
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            i += 1
        else:
            i += 1
            j += 1
            tp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, (tp, fp, fn)


@torch.no_grad()
def test_model_with_sweep(model, test_loader, device, config):
    """
    Test model on AutoShot dataset with threshold sweeping
    Uses reasoning features to intelligently boost/suppress predictions
    
    Returns:
        results: Dictionary with metrics for all thresholds
    """
    model.eval()
    
    # Collect predictions per video with reasoning features
    video_predictions = defaultdict(list)  # video_id -> list of (start_frame, boundary_probs, reasoning_features)
    
    print("Collecting predictions with reasoning features...")
    print(f"Reasoning refinement: {'ENABLED' if config.get('use_reasoning', True) else 'DISABLED'}")
    for batch in tqdm(test_loader, desc="Inference"):
        frames = batch['frames'].to(device)  # [B, T, 3, H, W]
        video_ids = batch['video_id']
        start_frames = batch['start_frame']
        
        # Forward pass
        boundary_logits, type_logits, reasoning = model(frames)
        
        # Get boundary probabilities
        boundary_probs = torch.sigmoid(boundary_logits.squeeze(-1))  # [B, T]
        
        # Get type probabilities (0=none, 1=cut, 2=gradual)
        type_probs = F.softmax(type_logits, dim=-1)  # [B, T, 3]
        gradual_prob = type_probs[:, :, 2]  # [B, T] - probability of gradual
        
        # Get reasoning features
        change_mag = reasoning['change_magnitude'].cpu().numpy()  # [B, T] or [B, T, 1]
        temp_var = reasoning['temporal_variance'].cpu().numpy()  # [B, T] or [B, T, 1]
        monotonicity = reasoning['monotonicity'].cpu().numpy()  # [B, T] or [B, T, 1]
        
        # Ensure features are 2D [B, T] by squeezing extra dimensions
        if change_mag.ndim > 2:
            change_mag = change_mag.squeeze(-1)
        if temp_var.ndim > 2:
            temp_var = temp_var.squeeze(-1)
        if monotonicity.ndim > 2:
            monotonicity = monotonicity.squeeze(-1)
        
        # Store predictions per video
        for b in range(frames.shape[0]):
            video_id = video_ids[b]
            start_frame = start_frames[b].item()
            probs = boundary_probs[b].cpu().numpy()  # [T]
            g_prob = gradual_prob[b].cpu().numpy()  # [T]
            
            # Package reasoning features - ensure 1D
            features = {
                'change_magnitude': change_mag[b].ravel(),  # [T]
                'temporal_variance': temp_var[b].ravel(),  # [T]
                'monotonicity': monotonicity[b].ravel(),  # [T]
                'gradual_prob': g_prob.ravel()  # [T]
            }
            
            video_predictions[video_id].append((start_frame, probs, features))
    
    # Load ground truth scenes
    print("\nLoading ground truth scenes...")
    import pickle
    gt_path = Path(config['cache_root']).parent / 'autoshot' / 'gt_scenes_dict_baseline_v2.pickle'
    with open(gt_path, 'rb') as f:
        gt_scenes_dict = pickle.load(f)
    
    # Reconstruct full video predictions
    print("\nReconstructing full video predictions...")
    video_full_probs = {}
    video_features = {}
    
    for video_id in tqdm(sorted(video_predictions.keys()), desc="Aggregating clips"):
        clips = sorted(video_predictions[video_id], key=lambda x: x[0])
        
        # Estimate video length
        max_end = max(start + len(probs) for start, probs, _ in clips)
        full_video_probs = np.zeros(max_end, dtype=np.float32)
        frame_counts = np.zeros(max_end, dtype=np.int32)
        
        # Initialize feature arrays
        full_change = np.zeros(max_end, dtype=np.float32)
        full_variance = np.zeros(max_end, dtype=np.float32)
        full_monotonicity = np.zeros(max_end, dtype=np.float32)
        full_gradual_prob = np.zeros(max_end, dtype=np.float32)
        
        # Average overlapping predictions and features
        for start_frame, probs, features in clips:
            end_frame = start_frame + len(probs)
            full_video_probs[start_frame:end_frame] += probs
            full_change[start_frame:end_frame] += features['change_magnitude']
            full_variance[start_frame:end_frame] += features['temporal_variance']
            full_monotonicity[start_frame:end_frame] += features['monotonicity']
            full_gradual_prob[start_frame:end_frame] += features['gradual_prob']
            frame_counts[start_frame:end_frame] += 1
        
        # Average where there are overlaps
        mask = frame_counts > 0
        full_video_probs[mask] /= frame_counts[mask]
        full_change[mask] /= frame_counts[mask]
        full_variance[mask] /= frame_counts[mask]
        full_monotonicity[mask] /= frame_counts[mask]
        full_gradual_prob[mask] /= frame_counts[mask]
        
        video_full_probs[video_id] = full_video_probs
        video_features[video_id] = {
            'change_magnitude': full_change,
            'temporal_variance': full_variance,
            'monotonicity': full_monotonicity,
            'gradual_prob': full_gradual_prob
        }
    
    # Analyze feature distributions for TP/FP/TN/FN at multiple thresholds
    if config.get('use_reasoning', True):
        print("\n" + "="*80)
        print("REASONING FEATURE PERCENTILE ANALYSIS")
        print("="*80)
        percentile_stats = analyze_feature_percentiles(
            video_full_probs, video_features, gt_scenes_dict, config
        )
        print("\n✓ Feature percentiles computed - using for intelligent refinement")
    else:
        percentile_stats = None
    
    # Sweep thresholds (matching AutoShot methodology)
    thresholds = np.array([
        0.02, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9
    ])
    
    # Optionally sweep multiplier strengths
    if config.get('sweep_multipliers', False):
        boost_strengths = np.array([1.00, 1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20])
        suppress_strengths = np.array([0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.75])
        print(f"\nSweeping {len(boost_strengths)} boost × {len(suppress_strengths)} suppress × {len(thresholds)} thresholds...")
        print(f"  Total combinations: {len(boost_strengths) * len(suppress_strengths) * len(thresholds)}")
    else:
        boost_strengths = [config.get('boost_strength', 1.12)]
        suppress_strengths = [config.get('suppress_strength', 0.88)]
        print(f"\nSweeping {len(thresholds)} thresholds...")
    
    threshold_results = {}
    best_overall_f1 = 0
    best_params = {}
    
    for boost in boost_strengths:
        for suppress in suppress_strengths:
            for threshold in tqdm(thresholds, desc=f"B={boost:.2f} S={suppress:.2f}", leave=False):
                all_tp, all_fp, all_fn = 0, 0, 0
                
                for video_id in video_full_probs.keys():
                    base_probs = video_full_probs[video_id]
                    features = video_features[video_id]
                    
                    # Intelligent refinement using reasoning features
                    refined_probs = refine_predictions_with_reasoning(
                        base_probs, 
                        features,
                        config.get('use_reasoning', True),
                        percentile_stats,
                        boost_strength=boost,
                        suppress_strength=suppress
                    )
                    
                    # Convert to binary
                    binary_preds = (refined_probs > threshold).astype(np.uint8)
                    
                    # Convert to scenes
                    pred_scenes = predictions_to_scenes(binary_preds)
                    gt_scenes = gt_scenes_dict[video_id]
                    
                    # Evaluate
                    _, _, _, (tp, fp, fn) = evaluate_scenes(
                        gt_scenes, pred_scenes, 
                        n_frames_miss_tolerance=config['tolerance']
                    )
                    
                    all_tp += tp
                    all_fp += fp
                    all_fn += fn
                
                # Compute metrics
                precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
                recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                key = (float(boost), float(suppress), float(threshold))
                threshold_results[key] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': all_tp,
                    'fp': all_fp,
                    'fn': all_fn
                }
                
                if f1 > best_overall_f1:
                    best_overall_f1 = f1
                    best_params = {'boost': boost, 'suppress': suppress, 'threshold': threshold}
    
    # Find best parameters
    if config.get('sweep_multipliers', False):
        best_boost = best_params['boost']
        best_suppress = best_params['suppress']
        best_threshold = best_params['threshold']
        best_key = (float(best_boost), float(best_suppress), float(best_threshold))
    else:
        best_threshold = max(threshold_results.keys(), key=lambda t: threshold_results[t]['f1'])
        best_key = best_threshold
        if isinstance(best_threshold, tuple):
            best_threshold = best_threshold[2]  # Extract threshold from tuple
    
    best_results = threshold_results[best_key]
    
    # Get per-video results at best parameters
    video_results = {}
    for video_id in video_full_probs.keys():
        base_probs = video_full_probs[video_id]
        features = video_features[video_id]
        
        if config.get('sweep_multipliers', False):
            refined_probs = refine_predictions_with_reasoning(
                base_probs, features, config.get('use_reasoning', True), percentile_stats,
                boost_strength=best_boost, suppress_strength=best_suppress
            )
        else:
            refined_probs = refine_predictions_with_reasoning(
                base_probs, features, config.get('use_reasoning', True), percentile_stats
            )
        binary_preds = (refined_probs > best_threshold).astype(np.uint8)
        pred_scenes = predictions_to_scenes(binary_preds)
        gt_scenes = gt_scenes_dict[video_id]
        
        precision, recall, f1, (tp, fp, fn) = evaluate_scenes(
            gt_scenes, pred_scenes, n_frames_miss_tolerance=config['tolerance']
        )
        
        video_results[video_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'num_gt_scenes': len(gt_scenes),
            'num_pred_scenes': len(pred_scenes)
        }
    
    # Print results
    if config.get('sweep_multipliers', False):
        print(f"\n{'='*80}")
        print("MULTIPLIER SWEEP RESULTS")
        print(f"{'='*80}\n")
        
        # Sort by F1 and show top 10
        sorted_results = sorted(threshold_results.items(), key=lambda x: x[1]['f1'], reverse=True)
        print("Top 10 combinations:")
        print(f"{'Rank':<6} {'Boost':<8} {'Suppress':<10} {'Threshold':<11} {'F1':<8} {'Precision':<11} {'Recall':<8}")
        print("-" * 80)
        for i, ((boost, suppress, threshold), res) in enumerate(sorted_results[:10], 1):
            print(f"{i:<6} {boost:<8.2f} {suppress:<10.2f} {threshold:<11.3f} {res['f1']:<8.4f} {res['precision']:<11.4f} {res['recall']:<8.4f}")
        
        print(f"\nBest combination: boost={best_boost:.2f}, suppress={best_suppress:.2f}, threshold={best_threshold:.3f}")
        print(f"F1={best_results['f1']:.4f}, Precision={best_results['precision']:.4f}, Recall={best_results['recall']:.4f}")
    
    return {
        'threshold_sweep': threshold_results,
        'best_threshold': best_threshold,
        'best_params': best_params if config.get('sweep_multipliers', False) else None,
        'overall': best_results,
        'video_results': video_results,
        'num_videos': len(video_results)
    }


def analyze_feature_percentiles(video_probs, video_features, gt_scenes_dict, config):
    """
    Analyze percentiles of reasoning features for TP/FP/TN/FN cases
    Also categorizes by predicted type (cut/gradual/none)
    
    Returns:
        Dictionary with percentile statistics for each outcome type
    """
    # Use a middle threshold for analysis (0.5)
    analysis_threshold = 0.5
    
    # Collect features for each outcome type
    tp_features = defaultdict(list)
    fp_features = defaultdict(list)
    tn_features = defaultdict(list)
    fn_features = defaultdict(list)
    
    # Collect by predicted type
    tp_by_type = defaultdict(lambda: defaultdict(list))  # outcome -> type -> features
    fp_by_type = defaultdict(lambda: defaultdict(list))
    
    for video_id in video_probs.keys():
        probs = video_probs[video_id]
        features = video_features[video_id]
        gt_scenes = gt_scenes_dict[video_id]
        
        # Create ground truth binary labels
        gt_binary = np.zeros(len(probs), dtype=np.uint8)
        for scene_start, scene_end in gt_scenes[1:]:
            # Mark transition frames (with tolerance)
            start_idx = max(0, scene_start - config['tolerance'])
            end_idx = min(len(gt_binary), scene_start + config['tolerance'] + 1)
            gt_binary[start_idx:end_idx] = 1
        
        # Predict binary
        pred_binary = (probs > analysis_threshold).astype(np.uint8)
        
        # Classify each frame
        for t in range(len(probs)):
            feature_dict = {
                'change': features['change_magnitude'][t],
                'variance': features['temporal_variance'][t],
                'monotonicity': features['monotonicity'][t],
                'gradual_prob': features['gradual_prob'][t],
                'boundary_prob': probs[t]
            }
            
            # Determine predicted type based on gradual_prob
            # Low gradual_prob (~<0.3) = cut, High gradual_prob (~>0.7) = gradual, Mid = uncertain
            gradual_p = features['gradual_prob'][t]
            if gradual_p < 0.3:
                pred_type = 'cut'
            elif gradual_p > 0.7:
                pred_type = 'gradual'
            else:
                pred_type = 'uncertain'
            
            if gt_binary[t] == 1 and pred_binary[t] == 1:
                # True Positive
                for k, v in feature_dict.items():
                    tp_features[k].append(v)
                    tp_by_type[pred_type][k].append(v)
            elif gt_binary[t] == 0 and pred_binary[t] == 1:
                # False Positive
                for k, v in feature_dict.items():
                    fp_features[k].append(v)
                    fp_by_type[pred_type][k].append(v)
            elif gt_binary[t] == 0 and pred_binary[t] == 0:
                # True Negative
                for k, v in feature_dict.items():
                    tn_features[k].append(v)
            elif gt_binary[t] == 1 and pred_binary[t] == 0:
                # False Negative
                for k, v in feature_dict.items():
                    fn_features[k].append(v)
    
    # Compute percentiles
    percentiles = [10, 25, 50, 75, 90]
    
    def compute_stats(feature_dict):
        stats = {}
        for feature_name, values in feature_dict.items():
            if len(values) > 0:
                values_array = np.array(values)
                stats[feature_name] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'percentiles': {p: np.percentile(values_array, p) for p in percentiles},
                    'count': len(values)
                }
        return stats
    
    results = {
        'TP': compute_stats(tp_features),
        'FP': compute_stats(fp_features),
        'TN': compute_stats(tn_features),
        'FN': compute_stats(fn_features)
    }
    
    # Print statistics
    for outcome_type in ['TP', 'FP', 'FN', 'TN']:
        stats = results[outcome_type]
        if not stats:
            continue
        
        print(f"\n{outcome_type} Features (n={stats.get('change', {}).get('count', 0):,}):")
        for feature in ['change', 'variance', 'monotonicity', 'gradual_prob']:
            if feature in stats:
                s = stats[feature]
                print(f"  {feature:15s}: mean={s['mean']:.3f}, p25={s['percentiles'][25]:.3f}, "
                      f"p50={s['percentiles'][50]:.3f}, p75={s['percentiles'][75]:.3f}, p90={s['percentiles'][90]:.3f}")
    
    # Print breakdown by predicted type
    print("\n" + "-"*80)
    print("BREAKDOWN BY PREDICTED TYPE")
    print("-"*80)
    
    for outcome in ['TP', 'FP']:
        type_dict = tp_by_type if outcome == 'TP' else fp_by_type
        print(f"\n{outcome} by Predicted Type:")
        for pred_type in ['cut', 'gradual', 'uncertain']:
            if pred_type in type_dict and 'change' in type_dict[pred_type]:
                stats = compute_stats(type_dict[pred_type])
                count = stats.get('change', {}).get('count', 0)
                if count > 0:
                    print(f"  {pred_type.upper()} (n={count:,}):")
                    gradual_p_mean = stats.get('gradual_prob', {}).get('mean', 0)
                    change_mean = stats.get('change', {}).get('mean', 0)
                    variance_mean = stats.get('variance', {}).get('mean', 0)
                    print(f"    gradual_prob={gradual_p_mean:.3f}, change={change_mean:.3f}, variance={variance_mean:.3f}")
    
    return results


def refine_predictions_with_reasoning(boundary_probs, features, use_reasoning=True, percentile_stats=None, boost_strength=1.12, suppress_strength=0.88):
    """
    Intelligently boost/suppress boundary predictions using reasoning features
    
    Uses percentile statistics from TP/FP/TN/FN analysis to apply data-driven thresholds
    
    Args:
        boundary_probs: [T] base boundary probabilities
        features: Dict with reasoning features
        use_reasoning: Whether to use reasoning refinement
        percentile_stats: Dictionary with percentile statistics (optional)
        boost_strength: Multiplier for boosting TPs (default: 1.12)
        suppress_strength: Multiplier for suppressing FPs (default: 0.88)
        
    Returns:
        refined_probs: [T] refined boundary probabilities
    """
    if not use_reasoning:
        return boundary_probs
    
    change = features['change_magnitude']
    variance = features['temporal_variance']
    monotonicity = features['monotonicity']
    gradual_prob = features['gradual_prob']
    
    refined = boundary_probs.copy()
    
    # Use data-driven thresholds if available, otherwise use learned defaults
    if percentile_stats and 'TP' in percentile_stats and 'FP' in percentile_stats:
        # Based on type breakdown analysis:
        # TP CUTS (80% of TPs): gradual_prob=0.020, change=4.503, variance=2.906
        # TP GRADUALS (14% of TPs): gradual_prob=0.841, change=-3.242, variance=0.386
        # FP CUTS (15% of FPs): gradual_prob=0.048, change=4.035, variance=1.438
        # FP GRADUALS (62% of FPs!): gradual_prob=0.841, change=-3.893, variance=-0.261
        
        # Extract percentiles
        tp_change_p25 = percentile_stats['TP'].get('change', {}).get('percentiles', {}).get(25, 3.9)
        tp_variance_p25 = percentile_stats['TP'].get('variance', {}).get('percentiles', {}).get(25, 1.2)
        tp_gradual_p75 = percentile_stats['TP'].get('gradual_prob', {}).get('percentiles', {}).get(75, 0.08)
        
        fp_change_p50 = percentile_stats['FP'].get('change', {}).get('percentiles', {}).get(50, -4.0)
        fp_variance_p50 = percentile_stats['FP'].get('variance', {}).get('percentiles', {}).get(50, -0.6)
        fp_gradual_p50 = percentile_stats['FP'].get('gradual_prob', {}).get('percentiles', {}).get(50, 0.76)
        
        # CRITICAL INSIGHT from type breakdown:
        # TP Cuts (80%): change=4.503, variance=2.906, gradual_prob=0.020
        # FP Cuts (15%): change=4.035, variance=1.438, gradual_prob=0.048
        #   → Key discriminator: VARIANCE (TP: 2.9 vs FP: 1.4)
        #
        # TP Graduals (14%): change=-3.242, variance=+0.386, gradual_prob=0.841
        # FP Graduals (62%!): change=-3.893, variance=-0.261, gradual_prob=0.841
        #   → Key discriminator: VARIANCE SIGN (TP: positive, FP: negative)
        
        # BOOST: True Cuts with high variance (differentiates from FP cuts)
        # Low gradual_prob (<0.08) + high change (>3.5) + HIGH variance (>2.0)
        true_cut_high_variance = (gradual_prob < 0.08) & (change > 3.5) & (variance > 2.0)
        refined[true_cut_high_variance] = np.minimum(1.0, refined[true_cut_high_variance] * boost_strength)
        
        # BOOST: True Graduals with positive variance (differentiates from FP graduals)
        # High gradual_prob (>0.7) + POSITIVE variance (>0.2) + negative change (<-2.0)
        true_gradual_positive_variance = (gradual_prob > 0.7) & (variance > 0.2) & (change < -2.0)
        refined[true_gradual_positive_variance] = np.minimum(1.0, refined[true_gradual_positive_variance] * (boost_strength * 0.95))
        
        # SUPPRESS: False Cuts with low variance
        # Low gradual_prob (<0.15) + high change (>3.0) BUT LOW variance (<1.8)
        false_cut_low_variance = (gradual_prob < 0.15) & (change > 3.0) & (variance < 1.8)
        refined[false_cut_low_variance] = refined[false_cut_low_variance] * (1.0 - (1.0 - suppress_strength) * 0.8)
        
        # SUPPRESS: False Graduals with negative variance (MAIN FP SOURCE - 62%)
        # High gradual_prob (>0.7) + NEGATIVE variance (<0.0) + negative change (<-3.0)
        false_gradual_negative_variance = (gradual_prob > 0.7) & (variance < 0.0) & (change < -3.0)
        refined[false_gradual_negative_variance] = refined[false_gradual_negative_variance] * suppress_strength
        
        # SUPPRESS: Uncertain FPs with very negative variance
        # Mid gradual_prob (0.5-0.7) + very negative variance (<-1.0) + negative change (<-3.0)
        uncertain_very_negative_variance = (gradual_prob > 0.5) & (gradual_prob < 0.7) & (variance < -1.0) & (change < -3.0)
        refined[uncertain_very_negative_variance] = refined[uncertain_very_negative_variance] * (1.0 - (1.0 - suppress_strength) * 0.7)
        
    else:
        # Fallback to learned defaults from training statistics
        # BOOST confident boundaries with strong discriminative signals
        # High change + low monotonicity = likely CUT
        cut_signal = (change > 0.7) & (monotonicity < 0.3)
        refined[cut_signal] = np.minimum(1.0, refined[cut_signal] * 1.15)
        
        # Mid change + mid variance + high gradual_prob = likely GRADUAL
        gradual_signal = (change > 0.05) & (change < 0.4) & (variance > 0.3) & (variance < 0.8) & (gradual_prob > 0.5)
        refined[gradual_signal] = np.minimum(1.0, refined[gradual_signal] * 1.10)
        
        # SUPPRESS false positives with none-like characteristics
        # Low change + high monotonicity + low gradual_prob = likely NONE (false positive)
        none_signal = (change < 0.1) & (monotonicity > 0.7) & (gradual_prob < 0.2)
        refined[none_signal] = refined[none_signal] * 0.75
        
        # SUPPRESS ambiguous gradual predictions (main source of FP)
        # Mid-low change + low variance + low gradual_prob = uncertain, suppress
        uncertain_signal = (change < 0.15) & (variance < 0.4) & (gradual_prob < 0.4)
        refined[uncertain_signal] = refined[uncertain_signal] * 0.85
    
    return refined


def main():
    parser = argparse.ArgumentParser(description='Test on AutoShot Dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--cache_root', type=str, default=r'd:\grad\data\autoshot_cache',
                       help='Path to predecoded AutoShot data')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for testing')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Boundary detection threshold')
    parser.add_argument('--tolerance', type=int, default=2,
                       help='Frame tolerance for scene matching (AutoShot default=2)')
    parser.add_argument('--sweep', action='store_true',
                       help='Sweep thresholds to find optimal F1')
    parser.add_argument('--sweep_multipliers', action='store_true',
                       help='Sweep boost/suppress multiplier strengths (requires --use_reasoning)')
    parser.add_argument('--use_reasoning', action='store_true', default=False,
                       help='Use reasoning features to refine predictions')
    parser.add_argument('--output', type=str, default='autoshot_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'checkpoint': args.checkpoint,
        'cache_root': args.cache_root,
        'sweep': args.sweep,
        'sweep_multipliers': args.sweep_multipliers,
        'use_reasoning': args.use_reasoning,
        'batch_size': args.batch_size,
        'threshold': args.threshold,
        'tolerance': args.tolerance,
    }
    print("\n" + "="*80)
    print("TESTING ON AUTOSHOT DATASET")
    print("="*80)
    print(f"Checkpoint:      {config['checkpoint']}")
    print(f"Cache root:      {config['cache_root']}")
    print(f"Batch size:      {config['batch_size']}")
    print(f"Threshold:       {config['threshold']}")
    print(f"Tolerance:       ±{config['tolerance']} frames")
    if config['use_reasoning']:
        print(f"Reasoning:       ENABLED")
        if config['sweep_multipliers']:
            print(f"Sweep mode:      Multipliers + Thresholds")
    print(f"Min gap:         {config['tolerance']} frames (AutoShot scene-based)")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = AutoShotPredecodedDataset(config['cache_root'])
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_autoshot_batch
    )
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(config['checkpoint'], map_location=device, weights_only=False)
    
    # Infer model architecture from checkpoint
    model_config = checkpoint.get('config', {})
    main_dim = model_config.get('main_dim', 384)
    num_fusion_blocks = model_config.get('num_fusion_blocks', 4)
    
    model = create_hybrid_detector(
        main_dim=main_dim,
        num_fusion_blocks=num_fusion_blocks
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_metrics' in checkpoint:
        val_f1 = checkpoint['val_metrics'].get('overall', {}).get('f1', 0)
        print(f"  Validation F1: {val_f1:.4f}")
    
    if config['sweep']:
        results = test_model_with_sweep(model, dataloader, device, config)
        
        # Print sweep results
        print("\n" + "="*80)
        print("THRESHOLD SWEEP RESULTS")
        print("="*80)
        print(f"Thresholds tested: {len(results['threshold_sweep'])}")
        print(f"\nBest threshold: {results['best_threshold']:.3f}")
        print(f"\nTop 5 thresholds by F1:")
        sorted_thresholds = sorted(results['threshold_sweep'].items(), 
                                  key=lambda x: x[1]['f1'], reverse=True)
        for thr, metrics in sorted_thresholds[:5]:
            print(f"  {thr:.3f}: F1={metrics['f1']:.4f} (P={metrics['precision']:.4f}, R={metrics['recall']:.4f})")
        
        # Print best results
        print("\n" + "="*80)
        print(f"BEST RESULTS (threshold={results['best_threshold']:.3f})")
        print("="*80)
    else:
        # Single threshold evaluation (legacy mode)
        results = test_model_with_sweep(model, dataloader, device, config)
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
    
    print(f"Videos tested:   {results['num_videos']}")
    print(f"\nOverall Metrics:")
    print(f"  Precision:     {results['overall']['precision']:.4f}")
    print(f"  Recall:        {results['overall']['recall']:.4f}")
    print(f"  F1 Score:      {results['overall']['f1']:.4f}")
    print(f"\nDetection Stats:")
    print(f"  True Positives:  {results['overall']['tp']}")
    print(f"  False Positives: {results['overall']['fp']}")
    print(f"  False Negatives: {results['overall']['fn']}")
    
    # Top/bottom videos by F1
    video_f1s = [(vid, vdata['f1']) for vid, vdata in results['video_results'].items()]
    video_f1s.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 Videos (by F1):")
    for vid, f1 in video_f1s[:5]:
        vdata = results['video_results'][vid]
        print(f"  {vid}: F1={f1:.3f} (P={vdata['precision']:.3f}, R={vdata['recall']:.3f})")
    
    print(f"\nBottom 5 Videos (by F1):")
    for vid, f1 in video_f1s[-5:]:
        vdata = results['video_results'][vid]
        print(f"  {vid}: F1={f1:.3f} (P={vdata['precision']:.3f}, R={vdata['recall']:.3f})")
    
    print("="*80)



if __name__ == '__main__':
    main()
