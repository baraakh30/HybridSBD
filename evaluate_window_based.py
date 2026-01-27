"""
Evaluate test set using the same window-based approach as training validation.
This gives an apples-to-apples comparison with training validation metrics.
"""

import numpy as np
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

from test_transformer_models import load_model, INDIVIDUAL_FEATURE_TYPES, load_individual_features, create_fused_features
from core.config import VideoShotConfig


def compute_tolerance_metrics(predictions: np.ndarray, targets: np.ndarray, 
                              tolerance: int = 3) -> Tuple[int, int, int, int]:
    """
    Compute TP, FP, FN, TN with temporal tolerance (same as training).
    
    Standard SBD evaluation protocol.
    """
    batch_size = predictions.shape[0]
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    for seq_idx in range(batch_size):
        seq_preds = predictions[seq_idx]
        seq_targets = targets[seq_idx]
        seq_len = len(seq_preds)
        
        # Get indices of predicted and ground truth boundaries
        pred_indices = list(np.where(seq_preds == 1)[0])
        gt_indices = list(np.where(seq_targets >= 0.5)[0])
        
        # ONE-TO-ONE MATCHING
        matched_pred = set()
        matched_gt = set()
        
        for gt_idx in sorted(gt_indices):
            best_pred = None
            best_dist = tolerance + 1
            
            for pred_idx in pred_indices:
                if pred_idx in matched_pred:
                    continue
                dist = abs(pred_idx - gt_idx)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_pred = pred_idx
            
            if best_pred is not None:
                matched_pred.add(best_pred)
                matched_gt.add(gt_idx)
        
        tp = len(matched_pred)
        fp = len(pred_indices) - tp
        fn = len(gt_indices) - tp
        tn = seq_len - len(pred_indices) - fn
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += max(0, tn)
    
    return total_tp, total_fp, total_fn, total_tn


def evaluate_window_based(model_path: str, fusion_strategy: str, 
                          features_dir: str, ground_truth_file: str,
                          sequence_length: int = 100,
                          confidence_threshold: float = 0.5,
                          tolerance: int = 3):
    """
    Evaluate test set using window-based approach (same as training validation).
    """
    config = VideoShotConfig()
    detection = load_model(model_path, fusion_strategy, config)
    detection.model.eval()
    
    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)
    
    # Get all test videos
    test_videos = []
    for video_name in gt_data.keys():
        video_stem = video_name.replace('.mp4', '')
        # Check if features exist
        feature_file = os.path.join(features_dir, f"{video_stem}_{fusion_strategy}.npy")
        if os.path.exists(feature_file):
            test_videos.append(video_stem)
    
    print(f"Evaluating {len(test_videos)} videos with {fusion_strategy} features")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    for video_name in tqdm(test_videos, desc="Evaluating"):
        # Load features
        feature_file = os.path.join(features_dir, f"{video_name}_{fusion_strategy}.npy")
        features = np.load(feature_file)
        
        # Get ground truth boundaries
        video_key = video_name if video_name in gt_data else video_name + '.mp4'
        video_gt = gt_data.get(video_key, {})
        if isinstance(video_gt, dict) and 'transitions' in video_gt:
            boundaries = [t[1] for t in video_gt['transitions']]
        else:
            boundaries = video_gt if isinstance(video_gt, list) else []
        
        # Create binary labels for entire video
        labels = np.zeros(len(features) - 1)
        for b in boundaries:
            if 0 <= b < len(labels):
                labels[b] = 1.0
        
        # Slide non-overlapping windows (same as validation_mode in training)
        stride = sequence_length - 5  # Same as training validation
        
        for start_idx in range(0, len(features) - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            
            window_features = features[start_idx:end_idx]
            window_labels = labels[start_idx:end_idx-1]  # seq_len - 1 frame transitions
            
            # Predict
            with torch.no_grad():
                seq_tensor = torch.from_numpy(window_features).float().unsqueeze(0).to(detection.device)
                logits = detection.model(seq_tensor, mode='frame')
                probs = torch.sigmoid(logits).cpu().numpy()[0, :, 0]
            
            # Threshold
            predictions = (probs > confidence_threshold).astype(float)
            
            # Compute metrics for this window
            tp, fp, fn, tn = compute_tolerance_metrics(
                predictions.reshape(1, -1), 
                window_labels.reshape(1, -1), 
                tolerance
            )
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
    
    # Compute final metrics
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    print(f"\nWindow-Based Evaluation Results (tolerance=±{tolerance} frames):")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/transformer_detection_pca.pth')
    parser.add_argument('--fusion', default='pca')
    parser.add_argument('--features_dir', default='testing_features')
    parser.add_argument('--ground_truth', default=r'C:\Users\Baraa\Documents\ClipShots\annotations\test.json')
    parser.add_argument('--tolerance', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    results = evaluate_window_based(
        args.model, args.fusion, args.features_dir, args.ground_truth,
        tolerance=args.tolerance,
        confidence_threshold=args.threshold
    )
