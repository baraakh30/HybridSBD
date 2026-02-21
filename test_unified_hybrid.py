"""
Unified Test Script for Hybrid Boundary Detector

Supports multiple evaluation protocols:
1. Frame-level metrics (current approach)
2. Transition-level metrics with tolerances
3. ClipShots-style "graduals as points" evaluation
4. Scene-based evaluation (AutoShot-style)

Handles all datasets: ClipShots, AutoShot, BBC, RAI

Usage:
    # Test on ClipShots (predecoded cache)
    python test_unified_hybrid.py --checkpoint outputs/hybrid_detector/best_model.pt --split test

    # Test with ClipShots point-based evaluation
    python test_unified_hybrid.py --checkpoint outputs/hybrid_detector/best_model.pt --split test --clipshots_eval
    
    # Test with both evaluation methods
    python test_unified_hybrid.py --checkpoint outputs/hybrid_detector/best_model.pt --split test --all_metrics
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.hybrid_boundary_detector import create_hybrid_detector
from train_boundary_detector import PredecodedDataset


# ============================================================================
# ClipShots-style Evaluation Functions
# ============================================================================

def scenes2zero_one_representation(scenes, n_frames):
    """
    Convert scene boundaries to one-hot and many-hot representations
    
    For gradual transitions:
    - one_hot: marks the CENTER of the transition
    - many_hot: marks ALL frames in the transition
    
    Example:
        [0][1] -> one_hot[0]=1 (cut)
        [0][x][2] -> one_hot[1]=1 (gradual center)
        [0][x][x][3] -> one_hot[1]=1 (gradual center)
        [0][x][x][x][4] -> one_hot[2]=1 (gradual center)
    
    Args:
        scenes: List of [start, end] scene boundaries
        n_frames: Total number of frames
        
    Returns:
        one_hot: [n_frames] binary array marking transition centers
        many_hot: [n_frames] binary array marking all transition frames
    """
    prev_end = 0
    one_hot = np.zeros([n_frames], np.uint8)
    many_hot = np.zeros([n_frames], np.uint8)
    
    for start, end in scenes:
        # Mark all transition frames
        for i in range(prev_end, start):
            many_hot[i] = 1
        
        # Mark transition center (one-hot)
        if not (prev_end == 0 and start == 0):
            one_hot_index = prev_end + (start - prev_end) // 2
            one_hot[one_hot_index] = 1
        
        prev_end = end
    
    # Handle final transition
    if prev_end + 1 != n_frames:
        for i in range(prev_end, n_frames):
            many_hot[i] = 1
        
        one_hot_index = prev_end + (n_frames - prev_end) // 2
        one_hot[one_hot_index] = 1
    
    return one_hot, many_hot


def predictions_to_scenes(predictions):
    """
    Convert binary predictions to scene boundaries
    
    Args:
        predictions: [T] binary array (1=boundary, 0=scene)
        
    Returns:
        scenes: Nx2 array of [start, end] for each scene
    """
    scenes = []
    t_prev, start = 0, 0
    
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    
    if t_prev == 0:  # End with a scene
        scenes.append([start, len(predictions)])
    
    # Handle edge case: all predictions are 1 (all boundaries)
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    
    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(gt_scenes, pred_scenes, n_frames_miss_tolerance=2, return_mistakes=False):
    """
    Evaluate scene detection with frame tolerance
    
    Adapted from ClipShots evaluation.
    
    Args:
        gt_scenes: Nx2 array of ground truth [start, end] scenes
        pred_scenes: Mx2 array of predicted [start, end] scenes
        n_frames_miss_tolerance: Tolerance in frames
        return_mistakes: Whether to return FP/FN mistakes
        
    Returns:
        precision, recall, f1, (tp, fp, fn), [fp_mistakes], [fn_mistakes]
    """
    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    
    # Get transitions between scenes
    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1) if len(gt_scenes) > 1 else np.zeros((0, 2))
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1) if len(pred_scenes) > 1 else np.zeros((0, 2))
    
    i, j = 0, 0
    tp, fp, fn = 0, 0, 0
    fp_mistakes, fn_mistakes = [], []
    
    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            if return_mistakes:
                fn_mistakes.append(gt_trans[i])
            i += 1
        elif i == len(gt_trans):
            fp += 1
            if return_mistakes:
                fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            if return_mistakes:
                fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            if return_mistakes:
                fn_mistakes.append(gt_trans[i])
            i += 1
        else:
            i += 1
            j += 1
            tp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    assert tp + fn == len(gt_trans)
    assert tp + fp == len(pred_trans)
    
    if return_mistakes:
        return precision, recall, f1, (tp, fp, fn), fp_mistakes, fn_mistakes
    return precision, recall, f1, (tp, fp, fn)


def mAP_f1_p_fix_r(one_hot_pred, gt_scenes, fixed_r=-1, skip_map_miou=True):
    """
    Compute mAP, F1, P, R with threshold sweeping
    
    EXACTLY matches ClipShots evaluation protocol.
    
    If fixed_r > 0: uses binary search to find threshold that gives target recall
    Otherwise: sweeps thresholds to find best F1
    
    Args:
        one_hot_pred: dict of {video_name: [T] prediction scores}
        gt_scenes: dict of {video_name: Nx2 ground truth scenes}
        fixed_r: target recall (if > 0, e.g., 0.70654 for ClipShots)
        skip_map_miou: whether to skip mAP/mIoU computation
        
    Returns:
        mAP, f1, precision, recall, best_threshold, miou
    """
    # Mode 1: Binary search to hit target recall (ClipShots fixed-recall mode)
    if fixed_r > 0:
        assert skip_map_miou
        eps = 0.001
        l_thr = 0.
        h_thr = 1.
        
        while h_thr - l_thr > eps:
            cur_thr = (l_thr + h_thr) / 2.
            precision = recall = f1 = tp = fp = fn = 0
            
            for file_name, pred in one_hot_pred.items():
                pred_scenes = predictions_to_scenes((pred > np.array([cur_thr])).astype(np.uint8))
                _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes[file_name], pred_scenes)
                tp += tp_
                fp += fp_
                fn += fn_
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if recall > fixed_r + eps:
                l_thr = cur_thr
            elif recall < fixed_r - eps:
                h_thr = cur_thr
            else:
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                return 0, f1, precision, recall, cur_thr, 0
        
        # Final evaluation at l_thr
        precision = recall = f1 = tp = fp = fn = 0
        for file_name, pred in one_hot_pred.items():
            pred_scenes = predictions_to_scenes((pred > np.array([l_thr])).astype(np.uint8))
            _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes[file_name], pred_scenes)
            tp += tp_
            fp += fp_
            fn += fn_
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return 0, f1, precision, recall, l_thr, 0
    
    # Mode 2: Threshold sweeping to find best F1 (default ClipShots mode)
    thresholds = np.array([
        0.02, 0.06, 0.1, 0.15, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25,
        0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.2833, 0.2867, 0.29,
        0.292, 0.294, 0.296, 0.298, 0.3, 0.302, 0.304, 0.306, 0.308,
        0.31, 0.3133, 0.3167, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345,
        0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ])
    
    precision = np.zeros_like(thresholds)
    recall = np.zeros_like(thresholds)
    f1 = np.zeros_like(thresholds)
    tp = np.zeros_like(thresholds)
    fp = np.zeros_like(thresholds)
    fn = np.zeros_like(thresholds)
    
    for i, thr in enumerate(thresholds):
        for file_name, pred in one_hot_pred.items():
            pred_scenes = predictions_to_scenes((pred > thr).astype(np.uint8))
            _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes[file_name], pred_scenes)
            tp[i] += tp_
            fp[i] += fp_
            fn[i] += fn_
        
        precision[i] = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        recall[i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    best_idx = np.argmax(f1)
    
    return 0, f1[best_idx], precision[best_idx], recall[best_idx], thresholds[best_idx], 0


# ============================================================================
# Standard Evaluation Functions (Frame-level & Transition-level)
# ============================================================================

def compute_frame_level_metrics(pred_labels, gt_labels):
    """
    Compute frame-level precision, recall, F1, accuracy
    
    Args:
        pred_labels: [N] predicted labels (0=none, 1=cut, 2=gradual)
        gt_labels: [N] ground truth labels
        
    Returns:
        dict with metrics
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # Overall metrics (any boundary vs none)
    pred_binary = (pred_labels > 0).astype(int)
    gt_binary = (gt_labels > 0).astype(int)
    
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        gt_binary, pred_binary, average='binary', zero_division=0
    )
    acc = accuracy_score(gt_labels, pred_labels)
    
    # Per-class metrics
    p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(
        gt_labels, pred_labels, labels=[0, 1, 2], zero_division=0
    )
    
    return {
        'boundary_precision': float(p_bin),
        'boundary_recall': float(r_bin),
        'boundary_f1': float(f1_bin),
        'accuracy': float(acc),
        'per_class': {
            'none': {'precision': float(p_cls[0]), 'recall': float(r_cls[0]), 'f1': float(f1_cls[0]), 'support': int(support[0])},
            'cut': {'precision': float(p_cls[1]), 'recall': float(r_cls[1]), 'f1': float(f1_cls[1]), 'support': int(support[1])},
            'gradual': {'precision': float(p_cls[2]), 'recall': float(r_cls[2]), 'f1': float(f1_cls[2]), 'support': int(support[2])},
        }
    }


def extract_transitions_from_predictions(pred_labels, min_gap=3):
    """
    Extract cut and gradual transitions from predictions
    
    Args:
        pred_labels: [T] predictions (0=none, 1=cut, 2=gradual)
        min_gap: minimum frames between transitions
        
    Returns:
        dict with 'cuts' and 'graduals' lists of frame indices
    """
    cuts = []
    graduals = []
    
    for i in range(len(pred_labels)):
        if pred_labels[i] == 1:  # Cut
            # Merge if too close to previous
            if len(cuts) > 0 and i - cuts[-1] < min_gap:
                continue
            cuts.append(i)
        elif pred_labels[i] == 2:  # Gradual
            # Find the center of the gradual segment
            if i == 0 or pred_labels[i-1] != 2:  # Start of gradual
                start = i
            if i == len(pred_labels) - 1 or pred_labels[i+1] != 2:  # End of gradual
                end = i + 1
                center = (start + end) // 2
                
                # Merge if too close to previous
                if len(graduals) > 0 and center - graduals[-1] < min_gap:
                    continue
                graduals.append(center)
    
    return {'cuts': cuts, 'graduals': graduals}


def compute_transition_metrics(pred_trans, gt_trans, tolerance=3):
    """
    Compute TP/FP/FN for transitions with frame tolerance
    
    Args:
        pred_trans: list of predicted transition frame indices
        gt_trans: list of ground truth transition frame indices
        tolerance: frame tolerance for matching
        
    Returns:
        dict with tp, fp, fn counts
    """
    pred_trans = sorted(pred_trans)
    gt_trans = sorted(gt_trans)
    
    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    
    for pred in pred_trans:
        matched = False
        for i, gt in enumerate(gt_trans):
            if i in matched_gt:
                continue
            if abs(pred - gt) <= tolerance:
                tp += 1
                matched_gt.add(i)
                matched = True
                break
        if not matched:
            fp += 1
    
    fn = len(gt_trans) - len(matched_gt)
    
    return {'tp': tp, 'fp': fp, 'fn': fn}


# ============================================================================
# Main Testing Function
# ============================================================================

@torch.no_grad()
def test_hybrid_unified(model, dataloader, device, args):
    """
    Unified testing with multiple evaluation protocols
    
    Returns:
        results: dict with all metrics
    """
    model.eval()
    
    # Accumulators for frame-level evaluation
    all_pred_labels = []
    all_gt_labels = []
    all_boundary_probs = []
    all_type_probs = []
    
    # For ClipShots-style evaluation
    clipshots_one_hot_pred = {}  # Probability-based (threshold sweeping)
    clipshots_binary_pred = {}   # Label-based (fixed threshold)
    clipshots_gt_scenes = {}
    
    # For transition-level evaluation
    all_pred_cuts = []
    all_pred_graduals = []
    all_gt_cuts = []
    all_gt_graduals = []
    
    clip_idx = 0
    
    print(f"\nRunning inference...")
    for batch_data in tqdm(dataloader, desc="Testing"):
        frames, labels, transition_lengths = batch_data
        frames = frames.to(device, non_blocking=True)
        labels = labels.cpu().numpy()
        
        # Forward pass
        boundary_logits, type_logits, features = model(frames)
        
        # Move to CPU
        boundary_logits = boundary_logits.cpu().numpy()
        type_logits = type_logits.cpu().numpy()
        
        # Process each clip in batch
        for b in range(frames.shape[0]):
            sample_boundary_logits = boundary_logits[b]
            sample_type_logits = type_logits[b]
            sample_labels = labels[b]
            
            # Get predictions
            boundary_probs = 1 / (1 + np.exp(-sample_boundary_logits.squeeze()))  # Sigmoid
            type_probs = np.exp(sample_type_logits) / np.exp(sample_type_logits).sum(axis=-1, keepdims=True)  # Softmax
            type_preds = np.argmax(type_probs, axis=-1)
            
            # Combine: boundary prob > 0.5 → use type prediction, else none
            pred_labels = np.where(boundary_probs > args.boundary_threshold, type_preds, 0)
            
            # Store for frame-level evaluation
            all_pred_labels.extend(pred_labels.tolist())
            all_gt_labels.extend(sample_labels.tolist())
            all_boundary_probs.extend(boundary_probs.tolist())
            all_type_probs.extend(type_probs.tolist())
            
            # For ClipShots evaluation: type-agnostic boundary detection
            if args.clipshots_eval or args.all_metrics:
                video_name = f"clip_{clip_idx:06d}"
                
                # Convert GT labels to binary (type-agnostic: 0=none, 1=any boundary)
                gt_binary = (sample_labels > 0).astype(np.uint8)
                gt_scenes_clip = predictions_to_scenes(gt_binary)
                clipshots_gt_scenes[video_name] = gt_scenes_clip
                
                # Store boundary probabilities (for probability-based evaluation with threshold sweeping)
                clipshots_one_hot_pred[video_name] = boundary_probs.copy()
                
                # Store binary predictions (type-agnostic: cuts and graduals both count as boundaries)
                pred_binary = (pred_labels > 0).astype(np.float32)
                clipshots_binary_pred[video_name] = pred_binary
            
            # Extract transitions for transition-level evaluation
            pred_trans = extract_transitions_from_predictions(pred_labels)
            gt_trans = extract_transitions_from_predictions(sample_labels)
            
            all_pred_cuts.extend(pred_trans['cuts'])
            all_pred_graduals.extend(pred_trans['graduals'])
            all_gt_cuts.extend(gt_trans['cuts'])
            all_gt_graduals.extend(gt_trans['graduals'])
            
            clip_idx += 1
    
    # ========================================================================
    # Compute Metrics
    # ========================================================================
    
    results = {}
    
    # 1. Frame-level metrics
    print("\n" + "="*80)
    print("FRAME-LEVEL METRICS")
    print("="*80)
    
    all_pred_labels = np.array(all_pred_labels)
    all_gt_labels = np.array(all_gt_labels)
    
    frame_metrics = compute_frame_level_metrics(all_pred_labels, all_gt_labels)
    results['frame_level'] = frame_metrics
    
    print(f"\nBoundary Detection (any type):")
    print(f"  Precision: {frame_metrics['boundary_precision']:.4f}")
    print(f"  Recall:    {frame_metrics['boundary_recall']:.4f}")
    print(f"  F1:        {frame_metrics['boundary_f1']:.4f}")
    print(f"  Accuracy:  {frame_metrics['accuracy']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for cls_name, cls_metrics in frame_metrics['per_class'].items():
        print(f"  {cls_name.upper()}:")
        print(f"    Precision: {cls_metrics['precision']:.4f}")
        print(f"    Recall:    {cls_metrics['recall']:.4f}")
        print(f"    F1:        {cls_metrics['f1']:.4f}")
        print(f"    Support:   {cls_metrics['support']}")
    
    # 2. Transition-level metrics
    print("\n" + "="*80)
    print("TRANSITION-LEVEL METRICS (Tolerance=3 frames)")
    print("="*80)
    
    # Cuts
    cut_metrics = compute_transition_metrics(all_pred_cuts, all_gt_cuts, tolerance=3)
    cut_p = cut_metrics['tp'] / (cut_metrics['tp'] + cut_metrics['fp']) if (cut_metrics['tp'] + cut_metrics['fp']) > 0 else 0
    cut_r = cut_metrics['tp'] / (cut_metrics['tp'] + cut_metrics['fn']) if (cut_metrics['tp'] + cut_metrics['fn']) > 0 else 0
    cut_f1 = 2 * cut_p * cut_r / (cut_p + cut_r) if (cut_p + cut_r) > 0 else 0
    
    print(f"\nCuts:")
    print(f"  Precision: {cut_p:.4f}")
    print(f"  Recall:    {cut_r:.4f}")
    print(f"  F1:        {cut_f1:.4f}")
    print(f"  TP/FP/FN:  {cut_metrics['tp']}/{cut_metrics['fp']}/{cut_metrics['fn']}")
    
    # Graduals
    gradual_metrics = compute_transition_metrics(all_pred_graduals, all_gt_graduals, tolerance=3)
    gradual_p = gradual_metrics['tp'] / (gradual_metrics['tp'] + gradual_metrics['fp']) if (gradual_metrics['tp'] + gradual_metrics['fp']) > 0 else 0
    gradual_r = gradual_metrics['tp'] / (gradual_metrics['tp'] + gradual_metrics['fn']) if (gradual_metrics['tp'] + gradual_metrics['fn']) > 0 else 0
    gradual_f1 = 2 * gradual_p * gradual_r / (gradual_p + gradual_r) if (gradual_p + gradual_r) > 0 else 0
    
    print(f"\nGraduals:")
    print(f"  Precision: {gradual_p:.4f}")
    print(f"  Recall:    {gradual_r:.4f}")
    print(f"  F1:        {gradual_f1:.4f}")
    print(f"  TP/FP/FN:  {gradual_metrics['tp']}/{gradual_metrics['fp']}/{gradual_metrics['fn']}")
    
    results['transition_level'] = {
        'cuts': {'precision': cut_p, 'recall': cut_r, 'f1': cut_f1, **cut_metrics},
        'graduals': {'precision': gradual_p, 'recall': gradual_r, 'f1': gradual_f1, **gradual_metrics}
    }
    
    # 3. ClipShots-style evaluation (type-agnostic boundary detection)
    if args.clipshots_eval or args.all_metrics:
        print("\n" + "="*80)
        print("CLIPSHOTS-STYLE EVALUATION (Type-Agnostic Boundaries)")
        print("="*80)
        print("Note: Treats cuts and graduals as equivalent 'boundary' events")
        print("      Evaluates transitions BETWEEN scenes (more strict than point-based)")
        
        # Method 1: Probability-based (optimal threshold via sweeping)
        print(f"\n1. PROBABILITY-BASED (Threshold Sweeping):")
        mAP, f1_prob, precision_prob, recall_prob, best_thr, miou = mAP_f1_p_fix_r(
            clipshots_one_hot_pred,
            clipshots_gt_scenes,
            fixed_r=-1,
            skip_map_miou=True
        )
        
        print(f"  Best Threshold: {best_thr:.4f}")
        print(f"  Precision:      {precision_prob:.4f}")
        print(f"  Recall:         {recall_prob:.4f}")
        print(f"  F1:             {f1_prob:.4f}")
        
        # Method 2: Label-based (using predictions at args.boundary_threshold)
        print(f"\n2. LABEL-BASED (Fixed Threshold={args.boundary_threshold}):")
        
        # Directly evaluate binary predictions without threshold sweeping
        tp_label = fp_label = fn_label = 0
        for video_name, pred_binary in clipshots_binary_pred.items():
            pred_scenes = predictions_to_scenes(pred_binary.astype(np.uint8))
            _, _, _, (tp_, fp_, fn_) = evaluate_scenes(clipshots_gt_scenes[video_name], pred_scenes)
            tp_label += tp_
            fp_label += fp_
            fn_label += fn_
        
        precision_label = tp_label / (tp_label + fp_label) if (tp_label + fp_label) > 0 else 0
        recall_label = tp_label / (tp_label + fn_label) if (tp_label + fn_label) > 0 else 0
        f1_label = 2 * precision_label * recall_label / (precision_label + recall_label) if (precision_label + recall_label) > 0 else 0
        
        print(f"  Precision:      {precision_label:.4f}")
        print(f"  Recall:         {recall_label:.4f}")
        print(f"  F1:             {f1_label:.4f}")
        print(f"  TP/FP/FN:       {tp_label}/{fp_label}/{fn_label}")
        
        # Calculate actual GT transition count
        total_gt = tp_label + fn_label
        print(f"\n  Ground Truth Transitions: {total_gt}")
        print(f"  Total Predictions:        {tp_label + fp_label}")
        
        results['clipshots_eval'] = {
            'probability_based': {
                'precision': float(precision_prob),
                'recall': float(recall_prob),
                'f1': float(f1_prob),
                'best_threshold': float(best_thr)
            },
            'label_based': {
                'precision': float(precision_label),
                'recall': float(recall_label),
                'f1': float(f1_label),
                'threshold': float(args.boundary_threshold),
                'tp': int(tp_label),
                'fp': int(fp_label),
                'fn': int(fn_label),
                'total_gt': int(total_gt)
            }
        }
    
    print("\n" + "="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Unified Test Script for Hybrid Boundary Detector')
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--main_dim', type=int, default=512,
                       help='Main dimension of model')
    parser.add_argument('--num_fusion_blocks', type=int, default=4,
                       help='Number of fusion blocks')
    
    # Dataset args
    parser.add_argument('--cache_root', type=str,
                       default=r'D:\grad\data\training_cache',
                       help='Path to predecoded cache')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    # Evaluation args
    parser.add_argument('--boundary_threshold', type=float, default=0.5,
                       help='Threshold for boundary detection')
    parser.add_argument('--clipshots_eval', action='store_true',
                       help='Run ClipShots-style evaluation (graduals as points)')
    parser.add_argument('--all_metrics', action='store_true',
                       help='Compute all evaluation metrics')
    
    # Output args
    parser.add_argument('--output_dir', type=str,
                       default='outputs/test_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint to get config
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Update args from checkpoint config
    if 'config' in checkpoint:
        config = checkpoint['config']
        args.main_dim = config.get('main_dim', args.main_dim)
        args.num_fusion_blocks = config.get('num_fusion_blocks', args.num_fusion_blocks)
    
    # Create model
    print("Creating model...")
    model = create_hybrid_detector(
        main_dim=args.main_dim,
        num_fusion_blocks=args.num_fusion_blocks
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded model weights")
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"\nLoading {args.split} dataset from {args.cache_root}...")
    dataset = PredecodedDataset(
        cache_root=args.cache_root,
        split=args.split,
        val_split_ratio=0.15,
        random_seed=42
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Run testing
    results = test_hybrid_unified(model, dataloader, device, args)
    
    # Save results
    output_dir = Path(args.output_dir) / f"{args.split}_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'metrics.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")


if __name__ == '__main__':
    main()
