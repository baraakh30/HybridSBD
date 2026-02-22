"""
Comprehensive Testing Script for Hybrid Boundary Detector

Implements:
1. Widely-used SBD evaluation metrics:
   - Frame-level: Precision, Recall, F1, Accuracy
   - Transition-level: Precision, Recall, F1 with multiple tolerances
   - Mean Average Precision (mAP) at different IoU thresholds
   - Confusion matrix analysis
   - Per-class metrics (cuts vs graduals)
   - Miou (Mean Intersection over Union)

2. Advanced post-processing:
   - Temporal smoothing with learnable window
   - Context-aware boundary refinement  
   - Probability calibration
   - Adaptive thresholding based on local statistics
   - Boundary clustering to merge nearby predictions
   - Type-aware non-maximum suppression
   - Multi-scale temporal aggregation
   - Confidence-based filtering with dynamic thresholds

This addresses the challenge of detecting correct boundaries even with low probabilities
by using context and temporal information to boost confidence.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import find_peaks
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.hybrid_boundary_detector import create_hybrid_detector
from datasets import PredecodedDataset
from utils import extract_transitions_from_predictions, compute_transition_metrics
from analyze_hybrid_failures import extract_failure_cases, analyze_per_layer_statistics, compare_failure_modes, print_analysis_summary


def filter_short_gradual_segments(pred_labels, min_length=5):
    """
    Filter out gradual predictions shorter than minimum length
    
    CRITICAL FIX: Diagnostic analysis shows 135/138 FPs are from "none" frames
    being wrongly predicted as graduals. These are typically isolated or short
    segments (1-4 frames). Real graduals span 5-60 frames.
    
    Args:
        pred_labels: [T] prediction array (0=none, 1=cut, 2=gradual)
        min_length: minimum frames for a valid gradual transition
        
    Returns:
        filtered: [T] predictions with short graduals removed
    """
    filtered = pred_labels.copy()
    T = len(pred_labels)
    
    # Find gradual segments
    in_gradual = False
    segment_start = 0
    
    for i in range(T):
        if pred_labels[i] == 2:  # Gradual
            if not in_gradual:
                in_gradual = True
                segment_start = i
        else:
            if in_gradual:
                # End of gradual segment - check length
                segment_length = i - segment_start
                if segment_length < min_length:
                    # Too short, remove it (set to none)
                    filtered[segment_start:i] = 0
                in_gradual = False
    
    # Handle segment extending to end
    if in_gradual:
        segment_length = T - segment_start
        if segment_length < min_length:
            filtered[segment_start:] = 0
    
    return filtered


class AdvancedPostProcessor:
    """
    Advanced post-processing for boundary detection
    
    Handles low probabilities that are correct by:
    1. Temporal smoothing - local consistency
    2. Context-aware refinement - neighboring frames
    3. Adaptive thresholding - local statistics
    4. Boundary clustering - merge nearby predictions
    5. Multi-scale aggregation - different temporal scales
    6. Confidence calibration - probability adjustment
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Smoothing parameters
        self.smooth_window = self.config.get('smooth_window', 3)  # Reduced from 5 (less aggressive)
        self.smooth_method = self.config.get('smooth_method', 'gaussian')  # 'uniform', 'gaussian', 'bilateral'
        
        # Adaptive thresholding - DISABLED by default (was hurting performance)
        self.adaptive = self.config.get('adaptive_threshold', False)  # Changed to False
        self.local_window = self.config.get('local_window', 15)
        self.threshold_offset = self.config.get('threshold_offset', 0.05)  # Positive offset = lower threshold
        
        # Boundary clustering - DISABLED by default (was removing correct detections)
        self.cluster = self.config.get('cluster_boundaries', False)  # Changed to False
        self.cluster_distance = self.config.get('cluster_distance', 3)  # Merge within 3 frames
        
        # Multi-scale aggregation - DISABLED by default (model already has multi-scale)
        self.multi_scale = self.config.get('multi_scale', False)  # Changed to False
        self.scales = self.config.get('scales', [1, 3, 5])  # Reduced scales
        
        # Type-aware NMS - DISABLED by default (was too aggressive)
        self.type_nms = self.config.get('type_nms', False)  # Changed to False
        self.nms_window = self.config.get('nms_window', 5)
        
        # Confidence calibration - ENABLED by default (improves probability estimates)
        self.calibrate = self.config.get('calibrate_probs', True)
        self.calibration_temp = self.config.get('calibration_temp', 1.2)  # Reduced from 1.5 (less smoothing)
        
        # Minimum gradual length filter - CRITICAL for removing FPs
        self.filter_short_graduals = self.config.get('filter_short_graduals', True)  # ENABLED by default
        self.min_gradual_length = self.config.get('min_gradual_length', 5)  # Graduals must be at least 5 frames
        
    def smooth_probabilities(self, probs, method=None):
        """
        Apply temporal smoothing to probabilities
        
        Args:
            probs: [T] probability array
            method: 'uniform', 'gaussian', or 'bilateral'
            
        Returns:
            smoothed: [T] smoothed probabilities
        """
        method = method or self.smooth_method
        
        if method == 'uniform':
            smoothed = uniform_filter1d(probs, size=self.smooth_window, mode='nearest')
        elif method == 'gaussian':
            sigma = self.smooth_window / 4.0
            smoothed = gaussian_filter1d(probs, sigma=sigma, mode='nearest')
        elif method == 'bilateral':
            # Bilateral filtering: smooth but preserve edges
            smoothed = np.copy(probs)
            half_win = self.smooth_window // 2
            for i in range(len(probs)):
                start = max(0, i - half_win)
                end = min(len(probs), i + half_win + 1)
                window = probs[start:end]
                
                # Spatial weights (distance from center)
                spatial_weights = np.exp(-0.5 * ((np.arange(len(window)) - half_win) ** 2) / (half_win / 2) ** 2)
                
                # Range weights (similarity to center value)
                range_weights = np.exp(-0.5 * ((window - probs[i]) ** 2) / 0.1 ** 2)
                
                # Combined weights
                weights = spatial_weights * range_weights
                weights /= weights.sum()
                
                smoothed[i] = (window * weights).sum()
        else:
            smoothed = probs
        
        return smoothed
    
    def adaptive_threshold(self, probs):
        """
        Compute adaptive threshold based on local statistics
        
        Low probabilities in high-activity regions should be kept,
        while high probabilities in low-activity regions might be noise.
        
        Args:
            probs: [T] probability array
            
        Returns:
            thresholds: [T] adaptive thresholds for each frame
        """
        if not self.adaptive:
            return np.full_like(probs, 0.5)
        
        # Local mean and std
        local_mean = uniform_filter1d(probs, size=self.local_window, mode='nearest')
        # Fix numerical stability: E[X^2] - E[X]^2 can be slightly negative due to floating point
        variance = uniform_filter1d(probs ** 2, size=self.local_window, mode='nearest') - local_mean ** 2
        variance = np.maximum(variance, 0)  # Clip negative values to 0
        local_std = np.sqrt(variance)
        
        # Adaptive threshold: local_mean + offset - std_factor * local_std
        # In high-variance regions (transitions), lower the threshold
        # In low-variance regions (static), raise the threshold
        thresholds = local_mean + self.threshold_offset - 0.5 * local_std
        thresholds = np.clip(thresholds, 0.1, 0.9)  # Keep reasonable bounds
        
        return thresholds
    
    def multi_scale_aggregation(self, probs):
        """
        Aggregate probabilities at multiple temporal scales
        
        A true boundary should show high probability at multiple scales.
        
        Args:
            probs: [T] probability array
            
        Returns:
            aggregated: [T] multi-scale aggregated probabilities
        """
        if not self.multi_scale:
            return probs
        
        scale_probs = []
        for scale in self.scales:
            if scale == 1:
                scale_probs.append(probs)
            else:
                smoothed = uniform_filter1d(probs, size=scale, mode='nearest')
                scale_probs.append(smoothed)
        
        # Combine: weighted average (give more weight to smaller scales)
        weights = np.array([1.0 / s for s in self.scales])
        weights /= weights.sum()
        
        aggregated = np.zeros_like(probs)
        for w, sp in zip(weights, scale_probs):
            aggregated += w * sp
        
        return aggregated
    
    def cluster_boundaries(self, boundary_probs, boundary_mask):
        """
        Cluster nearby boundary predictions and keep the strongest
        
        This prevents multiple predictions for the same transition.
        
        Args:
            boundary_probs: [T] boundary probabilities
            boundary_mask: [T] boolean mask of predicted boundaries
            
        Returns:
            clustered_mask: [T] boolean mask with clustered boundaries
            clustered_probs: [T] probabilities with clustered boundaries
        """
        if not self.cluster:
            return boundary_mask, boundary_probs
        
        boundary_indices = np.where(boundary_mask)[0]
        if len(boundary_indices) == 0:
            return boundary_mask, boundary_probs
        
        # Group nearby boundaries
        clusters = []
        current_cluster = [boundary_indices[0]]
        
        for idx in boundary_indices[1:]:
            if idx - current_cluster[-1] <= self.cluster_distance:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        clusters.append(current_cluster)
        
        # For each cluster, keep only the frame with highest probability
        clustered_mask = np.zeros_like(boundary_mask)
        clustered_probs = boundary_probs.copy()
        
        for cluster in clusters:
            cluster_probs = boundary_probs[cluster]
            best_idx = cluster[np.argmax(cluster_probs)]
            clustered_mask[best_idx] = True
        
        return clustered_mask, clustered_probs
    
    def type_aware_nms(self, boundary_indices, type_preds, boundary_probs):
        """
        Type-aware non-maximum suppression
        
        For boundaries of the same type within a window, keep the strongest.
        Different types can coexist (e.g., cut followed by gradual).
        
        Args:
            boundary_indices: array of boundary frame indices
            type_preds: [T] type predictions (1=cut, 2=gradual)
            boundary_probs: [T] boundary probabilities
            
        Returns:
            filtered_indices: filtered boundary indices
        """
        if not self.type_nms or len(boundary_indices) == 0:
            return boundary_indices
        
        # Group by type
        cuts = []
        graduals = []
        
        for idx in boundary_indices:
            if type_preds[idx] == 1:
                cuts.append((idx, boundary_probs[idx]))
            elif type_preds[idx] == 2:
                graduals.append((idx, boundary_probs[idx]))
        
        # Apply NMS within each type
        def nms_single_type(boundaries):
            if len(boundaries) == 0:
                return []
            
            # Sort by probability (descending)
            boundaries = sorted(boundaries, key=lambda x: x[1], reverse=True)
            kept = []
            
            while boundaries:
                # Keep the strongest
                best = boundaries.pop(0)
                kept.append(best[0])
                
                # Remove nearby boundaries
                boundaries = [b for b in boundaries if abs(b[0] - best[0]) > self.nms_window]
            
            return kept
        
        filtered_cuts = nms_single_type(cuts)
        filtered_graduals = nms_single_type(graduals)
        
        # Combine and ensure integer dtype
        if len(filtered_cuts) == 0 and len(filtered_graduals) == 0:
            return np.array([], dtype=np.int64)
        elif len(filtered_cuts) == 0:
            return np.sort(np.array(filtered_graduals, dtype=np.int64))
        elif len(filtered_graduals) == 0:
            return np.sort(np.array(filtered_cuts, dtype=np.int64))
        else:
            return np.sort(np.concatenate([filtered_cuts, filtered_graduals]).astype(np.int64))
    
    def calibrate_probabilities(self, probs, temperature=None):
        """
        Calibrate probabilities using temperature scaling
        
        Temperature > 1: Soften probabilities (less confident)
        Temperature < 1: Sharpen probabilities (more confident)
        
        Args:
            probs: [T] probability array
            temperature: temperature parameter
            
        Returns:
            calibrated: [T] calibrated probabilities
        """
        if not self.calibrate:
            return probs
        
        temperature = temperature or self.calibration_temp
        
        # Convert probs to logits, scale, then back to probs
        logits = np.log(probs + 1e-8) - np.log(1 - probs + 1e-8)
        logits /= temperature
        calibrated = 1 / (1 + np.exp(-logits))
        
        return calibrated
    
    def filter_short_graduals(self, type_preds):
        """
        Filter out gradual predictions shorter than minimum length
        
        Uses the standalone filter_short_gradual_segments function.
        
        Args:
            type_preds: [T] type predictions (0=none, 1=cut, 2=gradual)
            
        Returns:
            filtered: [T] type predictions with short graduals removed
        """
        if not self.filter_short_graduals:
            return type_preds
        
        return filter_short_gradual_segments(type_preds, min_length=self.min_gradual_length)
    
    def __call__(self, boundary_logits, type_logits, base_threshold=0.5):
        """
        Apply full post-processing pipeline
        
        Args:
            boundary_logits: [B, T, 1] or [T,] boundary logits
            type_logits: [B, T, 3] or [T, 3] type logits
            base_threshold: base threshold for boundary detection
            
        Returns:
            boundary_mask: [T] final boundary predictions
            type_preds: [T] final type predictions
            refined_probs: [T] refined boundary probabilities
        """
        # Handle batched or single input
        if len(boundary_logits.shape) == 3:
            boundary_logits = boundary_logits[0]  # Take first sample
            type_logits = type_logits[0]
        
        boundary_logits = boundary_logits.squeeze()
        
        # Convert to numpy
        if isinstance(boundary_logits, torch.Tensor):
            boundary_logits = boundary_logits.cpu().numpy()
        if isinstance(type_logits, torch.Tensor):
            type_logits = type_logits.cpu().numpy()
        
        # Get probabilities
        boundary_probs = 1 / (1 + np.exp(-boundary_logits))  # Sigmoid
        type_probs = np.exp(type_logits) / np.exp(type_logits).sum(axis=-1, keepdims=True)  # Softmax
        type_preds = np.argmax(type_probs, axis=-1)
        
        # 1. Calibrate probabilities (temperature scaling)
        boundary_probs = self.calibrate_probabilities(boundary_probs)
        
        # 2. Multi-scale aggregation
        boundary_probs = self.multi_scale_aggregation(boundary_probs)
        
        # 3. Temporal smoothing
        boundary_probs = self.smooth_probabilities(boundary_probs)
        
        # 4. Adaptive thresholding
        adaptive_thresholds = self.adaptive_threshold(boundary_probs)
        boundary_mask = boundary_probs > adaptive_thresholds
        
        # 5. Boundary clustering
        boundary_mask, boundary_probs = self.cluster_boundaries(boundary_probs, boundary_mask)
        
        # 6. Type-aware NMS
        boundary_indices = np.where(boundary_mask)[0]
        boundary_indices = self.type_aware_nms(boundary_indices, type_preds, boundary_probs)
        
        # Create final mask
        final_mask = np.zeros_like(boundary_mask)
        final_mask[boundary_indices] = True
        
        # 7. Filter short gradual segments (CRITICAL FIX for FPs)
        # Combine boundary mask with type to get final predictions
        final_type_preds = np.where(final_mask, type_preds, 0)
        
        # Filter out short gradual segments (removes isolated FPs from "none" regions)
        final_type_preds = self.filter_short_graduals(final_type_preds)
        
        return final_mask, final_type_preds, boundary_probs


def compute_frame_level_metrics(pred_labels, gt_labels, num_classes=3):
    """
    Compute frame-level classification metrics
    
    Args:
        pred_labels: [N] predicted labels
        gt_labels: [N] ground truth labels
        num_classes: number of classes
        
    Returns:
        metrics: dict with precision, recall, f1, accuracy, confusion matrix
    """
    # Overall accuracy
    accuracy = (pred_labels == gt_labels).mean()
    
    # Per-class metrics
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(num_classes)))
    
    # Precision, recall, F1 per class
    per_class = {}
    class_names = ['none', 'cut', 'gradual']
    
    for i, name in enumerate(class_names):
        if i >= num_classes:
            break
        
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        per_class[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    # Binary boundary detection metrics (boundary vs non-boundary)
    pred_boundary = pred_labels > 0
    gt_boundary = gt_labels > 0
    
    tp_boundary = (pred_boundary & gt_boundary).sum()
    fp_boundary = (pred_boundary & ~gt_boundary).sum()
    fn_boundary = (~pred_boundary & gt_boundary).sum()
    
    boundary_precision = tp_boundary / (tp_boundary + fp_boundary + 1e-8)
    boundary_recall = tp_boundary / (tp_boundary + fn_boundary + 1e-8)
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'boundary': {
            'precision': float(boundary_precision),
            'recall': float(boundary_recall),
            'f1': float(boundary_f1)
        }
    }


def compute_map(pred_boundaries, gt_boundaries, pred_probs, iou_thresholds=[0.3, 0.5, 0.7]):
    """
    Compute Mean Average Precision (mAP) at different IoU thresholds
    
    This is a widely-used metric in object detection, adapted for boundary detection.
    
    Args:
        pred_boundaries: list of predicted boundary frames
        gt_boundaries: list of ground truth boundary frames
        pred_probs: probabilities corresponding to pred_boundaries
        iou_thresholds: list of IoU thresholds for matching
        
    Returns:
        map_scores: dict with mAP at each IoU threshold
    """
    if len(pred_boundaries) == 0 or len(gt_boundaries) == 0:
        return {f'mAP@{t}': 0.0 for t in iou_thresholds}
    
    # Sort predictions by probability (descending)
    sorted_indices = np.argsort(-pred_probs)
    pred_boundaries = pred_boundaries[sorted_indices]
    pred_probs = pred_probs[sorted_indices]
    
    map_scores = {}
    
    for iou_thresh in iou_thresholds:
        # Match predictions to ground truth
        gt_matched = np.zeros(len(gt_boundaries), dtype=bool)
        tp = np.zeros(len(pred_boundaries))
        fp = np.zeros(len(pred_boundaries))
        
        for i, pred in enumerate(pred_boundaries):
            # Compute "IoU" as overlap divided by union (for 1D intervals)
            # Use a window around each boundary
            window = int(1 / iou_thresh) if iou_thresh > 0 else 3
            
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(gt_boundaries):
                if gt_matched[j]:
                    continue
                
                # Distance-based IoU approximation
                dist = abs(pred - gt)
                if dist <= window:
                    iou = 1.0 - (dist / window)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            if best_iou >= iou_thresh:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gt_boundaries)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Compute Average Precision (area under PR curve)
        # Use 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        map_scores[f'mAP@{iou_thresh}'] = ap
    
    return map_scores


def compute_miou(pred_labels, gt_labels, num_classes=3):
    """
    Compute Mean Intersection over Union (mIoU)
    
    This is a standard metric in semantic segmentation, useful for evaluating
    frame-level classification quality.
    
    Args:
        pred_labels: [N] predicted labels
        gt_labels: [N] ground truth labels
        num_classes: number of classes
        
    Returns:
        miou: mean IoU across all classes
        per_class_iou: IoU for each class
    """
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(num_classes)))
    
    ious = []
    class_names = ['none', 'cut', 'gradual']
    per_class_iou = {}
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        iou = tp / (tp + fp + fn + 1e-8)
        ious.append(iou)
        per_class_iou[class_names[i]] = float(iou)
    
    miou = np.mean(ious)
    
    return float(miou), per_class_iou


@torch.no_grad()
def evaluate_model(model, dataloader, device, post_processor=None, tolerances=[1, 3, 5], min_gradual_length=5):
    """
    Comprehensive evaluation with all metrics
    
    Args:
        model: trained model
        dataloader: test data loader
        device: torch device
        post_processor: AdvancedPostProcessor instance
        tolerances: list of frame tolerances for transition-level metrics
        
    Returns:
        results: dict with all evaluation metrics
    """
    model.eval()
    
    # Accumulators
    all_pred_labels = []
    all_gt_labels = []
    all_boundary_probs = []
    
    # Transition-level metrics per tolerance
    transition_metrics = {tol: {'cut': {'tp': 0, 'fp': 0, 'fn': 0},
                                'gradual': {'tp': 0, 'fp': 0, 'fn': 0}}
                         for tol in tolerances}
    
    # mAP computation
    all_pred_boundaries = []
    all_gt_boundaries = []
    all_pred_boundary_probs = []
    
    print("\nEvaluating model...")
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Testing")):
        frames, labels, transition_lengths = batch_data
        
        frames = frames.to(device, non_blocking=True)
        labels = labels.cpu().numpy()
        
        # Forward pass
        boundary_logits, type_logits, reasoning= model(frames)
        
        # Move to CPU/numpy
        boundary_logits = boundary_logits.cpu()
        type_logits = type_logits.cpu()
        
        # Process each sample in batch
        for b in range(frames.shape[0]):
            sample_boundary_logits = boundary_logits[b]
            sample_type_logits = type_logits[b]
            sample_labels = labels[b]
            
            # Apply post-processing if provided
            if post_processor is not None:
                boundary_mask, type_preds, refined_probs = post_processor(
                    sample_boundary_logits,
                    sample_type_logits
                )
                
                # Combine boundary mask with type predictions
                pred_labels = np.where(boundary_mask, type_preds, 0)
                boundary_probs = refined_probs
            else:
                # Standard thresholding
                boundary_probs = torch.sigmoid(sample_boundary_logits).squeeze().numpy()
                type_probs = torch.softmax(sample_type_logits, dim=-1).numpy()
                type_preds = np.argmax(type_probs, axis=-1)
                
                # Apply standard threshold
                boundary_mask = boundary_probs > 0.5
                pred_labels = np.where(boundary_mask, type_preds, 0)
                
                # CRITICAL FIX: Filter short gradual segments (removes FPs)
                pred_labels = filter_short_gradual_segments(pred_labels, min_length=min_gradual_length)
            
            # Accumulate frame-level predictions
            all_pred_labels.extend(pred_labels.tolist())
            all_gt_labels.extend(sample_labels.tolist())
            all_boundary_probs.extend(boundary_probs.tolist())
            
            # Extract transitions for transition-level metrics
            pred_trans = extract_transitions_from_predictions(torch.from_numpy(pred_labels), min_gap=3)
            gt_trans = extract_transitions_from_predictions(torch.from_numpy(sample_labels), min_gap=3)
            
            # Compute transition-level metrics for each tolerance
            for tol in tolerances:
                # Cuts
                cut_metrics = compute_transition_metrics(
                    pred_trans['cuts'], gt_trans['cuts'], tolerance=tol
                )
                transition_metrics[tol]['cut']['tp'] += cut_metrics.get('tp', 0)
                transition_metrics[tol]['cut']['fp'] += cut_metrics.get('fp', 0)
                transition_metrics[tol]['cut']['fn'] += cut_metrics.get('fn', 0)
                
                # Graduals
                gradual_metrics = compute_transition_metrics(
                    pred_trans['graduals'], gt_trans['graduals'], tolerance=tol
                )
                transition_metrics[tol]['gradual']['tp'] += gradual_metrics.get('tp', 0)
                transition_metrics[tol]['gradual']['fp'] += gradual_metrics.get('fp', 0)
                transition_metrics[tol]['gradual']['fn'] += gradual_metrics.get('fn', 0)
            
            # For mAP computation
            pred_boundary_frames = np.where(boundary_mask)[0]
            gt_boundary_frames = np.where(sample_labels > 0)[0]
            
            if len(pred_boundary_frames) > 0:
                all_pred_boundaries.append(pred_boundary_frames)
                all_pred_boundary_probs.append(boundary_probs[pred_boundary_frames])
            
            if len(gt_boundary_frames) > 0:
                all_gt_boundaries.append(gt_boundary_frames)
    
    # Convert to numpy arrays
    all_pred_labels = np.array(all_pred_labels)
    all_gt_labels = np.array(all_gt_labels)
    
    # Compute frame-level metrics
    print("\nComputing frame-level metrics...")
    frame_metrics = compute_frame_level_metrics(all_pred_labels, all_gt_labels)
    
    # Compute mIoU
    print("Computing mIoU...")
    miou, per_class_iou = compute_miou(all_pred_labels, all_gt_labels)
    
    # Compute transition-level metrics
    print("Computing transition-level metrics...")
    transition_results = {}
    for tol in tolerances:
        cut_tp = transition_metrics[tol]['cut']['tp']
        cut_fp = transition_metrics[tol]['cut']['fp']
        cut_fn = transition_metrics[tol]['cut']['fn']
        
        cut_p = cut_tp / (cut_tp + cut_fp + 1e-8)
        cut_r = cut_tp / (cut_tp + cut_fn + 1e-8)
        cut_f1 = 2 * cut_p * cut_r / (cut_p + cut_r + 1e-8)
        
        gradual_tp = transition_metrics[tol]['gradual']['tp']
        gradual_fp = transition_metrics[tol]['gradual']['fp']
        gradual_fn = transition_metrics[tol]['gradual']['fn']
        
        gradual_p = gradual_tp / (gradual_tp + gradual_fp + 1e-8)
        gradual_r = gradual_tp / (gradual_tp + gradual_fn + 1e-8)
        gradual_f1 = 2 * gradual_p * gradual_r / (gradual_p + gradual_r + 1e-8)
        
        overall_tp = cut_tp + gradual_tp
        overall_fp = cut_fp + gradual_fp
        overall_fn = cut_fn + gradual_fn
        
        overall_p = overall_tp / (overall_tp + overall_fp + 1e-8)
        overall_r = overall_tp / (overall_tp + overall_fn + 1e-8)
        overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r + 1e-8)
        
        transition_results[f'tolerance_{tol}'] = {
            'cut': {
                'precision': float(cut_p),
                'recall': float(cut_r),
                'f1': float(cut_f1),
                'tp': int(cut_tp),
                'fp': int(cut_fp),
                'fn': int(cut_fn)
            },
            'gradual': {
                'precision': float(gradual_p),
                'recall': float(gradual_r),
                'f1': float(gradual_f1),
                'tp': int(gradual_tp),
                'fp': int(gradual_fp),
                'fn': int(gradual_fn)
            },
            'overall': {
                'precision': float(overall_p),
                'recall': float(overall_r),
                'f1': float(overall_f1)
            }
        }
    
    # Compute mAP (flatten all predictions)
    print("Computing mAP...")
    if all_pred_boundaries and all_gt_boundaries:
        pred_boundaries_flat = np.concatenate(all_pred_boundaries)
        gt_boundaries_flat = np.concatenate(all_gt_boundaries)
        pred_probs_flat = np.concatenate(all_pred_boundary_probs)
        
        map_scores = compute_map(pred_boundaries_flat, gt_boundaries_flat, pred_probs_flat)
    else:
        map_scores = {'mAP@0.3': 0.0, 'mAP@0.5': 0.0, 'mAP@0.7': 0.0}
    
    # Compile all results
    results = {
        'frame_level': frame_metrics,
        'transition_level': transition_results,
        'miou': miou,
        'per_class_iou': per_class_iou,
        'map': map_scores
    }
    
    return results


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved: {save_path}")


def print_results(results, post_processing=False):
    """
    Print comprehensive results in a readable format
    
    Args:
        results: results dict from evaluate_model
        post_processing: whether post-processing was used
    """
    print("\n" + "="*80)
    print(f"{'COMPREHENSIVE EVALUATION RESULTS' + (' (WITH POST-PROCESSING)' if post_processing else '')}")
    print("="*80)
    
    # Frame-level metrics
    print("\n📊 FRAME-LEVEL METRICS")
    print("-" * 80)
    print(f"Accuracy: {results['frame_level']['accuracy']:.4f}")
    print(f"mIoU: {results['miou']:.4f}")
    
    print("\nPer-class IoU:")
    for cls, iou in results['per_class_iou'].items():
        print(f"  {cls:>8s}: {iou:.4f}")
    
    print("\nBoundary Detection (binary):")
    print(f"  Precision: {results['frame_level']['boundary']['precision']:.4f}")
    print(f"  Recall:    {results['frame_level']['boundary']['recall']:.4f}")
    print(f"  F1:        {results['frame_level']['boundary']['f1']:.4f}")
    
    print("\nPer-class Metrics:")
    for cls, metrics in results['frame_level']['per_class'].items():
        print(f"  {cls.capitalize():>8s} - P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        print(f"           (TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']})")
    
    # Transition-level metrics
    print("\n🎯 TRANSITION-LEVEL METRICS")
    print("-" * 80)
    for tol_key, tol_results in results['transition_level'].items():
        tol = tol_key.split('_')[1]
        print(f"\nTolerance: ±{tol} frames")
        
        print(f"  Cut:")
        print(f"    Precision: {tol_results['cut']['precision']:.4f}")
        print(f"    Recall:    {tol_results['cut']['recall']:.4f}")
        print(f"    F1:        {tol_results['cut']['f1']:.4f}")
        print(f"    (TP: {tol_results['cut']['tp']}, FP: {tol_results['cut']['fp']}, FN: {tol_results['cut']['fn']})")
        
        print(f"  Gradual:")
        print(f"    Precision: {tol_results['gradual']['precision']:.4f}")
        print(f"    Recall:    {tol_results['gradual']['recall']:.4f}")
        print(f"    F1:        {tol_results['gradual']['f1']:.4f}")
        print(f"    (TP: {tol_results['gradual']['tp']}, FP: {tol_results['gradual']['fp']}, FN: {tol_results['gradual']['fn']})")
        
        print(f"  Overall:")
        print(f"    Precision: {tol_results['overall']['precision']:.4f}")
        print(f"    Recall:    {tol_results['overall']['recall']:.4f}")
        print(f"    F1:        {tol_results['overall']['f1']:.4f}")
    
    # mAP scores
    print("\n📈 MEAN AVERAGE PRECISION (mAP)")
    print("-" * 80)
    for map_key, map_val in results['map'].items():
        print(f"{map_key}: {map_val:.4f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Testing for Hybrid Boundary Detector')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--cache_root', type=str, default=r'D:\grad\data\training_cache',
                       help='Path to pre-decoded dataset cache')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results (default: same as checkpoint dir)')
    
    # Post-processing options
    parser.add_argument('--use_post_processing', action='store_true',
                       help='Enable advanced post-processing')
    parser.add_argument('--smooth_window', type=int, default=5,
                       help='Smoothing window size')
    parser.add_argument('--smooth_method', type=str, default='gaussian',
                       choices=['uniform', 'gaussian', 'bilateral'],
                       help='Smoothing method')
    parser.add_argument('--adaptive_threshold', action='store_true',
                       help='Use adaptive thresholding')
    parser.add_argument('--cluster_boundaries', action='store_true',
                       help='Cluster nearby boundaries')
    parser.add_argument('--multi_scale', action='store_true',
                       help='Use multi-scale aggregation')
    parser.add_argument('--type_nms', action='store_true',
                       help='Use type-aware NMS')
    parser.add_argument('--calibrate_probs', action='store_true',
                       help='Calibrate probabilities')
    parser.add_argument('--min_gradual_length', type=int, default=5,
                       help='Minimum length for gradual transitions (filters short FPs)')
    
    # Deep analysis option
    parser.add_argument('--deep_analysis', action='store_true',
                       help='Run deep per-layer failure analysis (FP/FN graduals, FN none)')
    parser.add_argument('--max_analysis_samples', type=int, default=None,
                       help='Maximum clips for deep analysis (None = all)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device,weights_only=False)
    
    config = checkpoint.get('config', {})
    main_dim = config.get('main_dim', 256)
    num_fusion_blocks = config.get('num_fusion_blocks', 3)
    
    # Create model
    print("\nCreating model...")
    model = create_hybrid_detector(main_dim=main_dim, num_fusion_blocks=num_fusion_blocks)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation F1: {checkpoint.get('best_val_f1', 'unknown')}")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = PredecodedDataset(
        args.cache_root,
        split=args.split,
        val_split_ratio=0.15,
        random_seed=42
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup post-processor
    post_processor = None
    if args.use_post_processing:
        print("\n🔧 Setting up advanced post-processing...")
        post_config = {
            'smooth_window': args.smooth_window,
            'smooth_method': args.smooth_method,
            'adaptive_threshold': args.adaptive_threshold,
            'cluster_boundaries': args.cluster_boundaries,
            'multi_scale': args.multi_scale,
            'type_nms': args.type_nms,
            'calibrate_probs': args.calibrate_probs,
            'min_gradual_length': args.min_gradual_length
        }
        post_processor = AdvancedPostProcessor(post_config)
        print("Post-processing configuration:")
        for key, val in post_config.items():
            print(f"  {key}: {val}")
    else:
        print(f"\n✓ Using standard thresholding with minimum gradual length filter ({args.min_gradual_length} frames)")
    
    # Evaluate
    results = evaluate_model(
        model, dataloader, device,
        post_processor=post_processor,
        tolerances=[1, 3, 5],
        min_gradual_length=args.min_gradual_length
    )
    
    # Print results
    print_results(results, post_processing=args.use_post_processing)
    
    # Setup output directory
    if args.output_dir is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.output_dir = checkpoint_dir / f'test_results_{args.split}'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = output_dir / f'comprehensive_results{"_postproc" if args.use_post_processing else ""}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Plot confusion matrix
    cm = np.array(results['frame_level']['confusion_matrix'])
    cm_file = output_dir / f'confusion_matrix{"_postproc" if args.use_post_processing else ""}.png'
    plot_confusion_matrix(cm, ['None', 'Cut', 'Gradual'], cm_file)
    
    # Deep failure analysis (optional)
    if args.deep_analysis:
        print("\n" + "="*80)
        print("RUNNING DEEP PER-LAYER FAILURE ANALYSIS")
        print("="*80)
        
        # Extract failure cases with internal features
        failure_cases = extract_failure_cases(
            model, dataloader, device,
            max_samples=args.max_analysis_samples
        )
        
        # Analyze per-layer statistics
        analysis = analyze_per_layer_statistics(failure_cases)
        
        # Compare failure modes
        comparison = compare_failure_modes(analysis)
        
        # Print summary
        print_analysis_summary(analysis, comparison)
        
        # Prepare raw cases for visualization (sample to keep file size manageable)
        raw_cases_for_viz = {}
        max_viz_samples = 100  # Keep 100 cases per type for visualization
        
        for case_type, cases in failure_cases.items():
            if len(cases) == 0:
                continue
            
            # Sample evenly across the dataset
            num_samples = min(max_viz_samples, len(cases))
            indices = np.linspace(0, len(cases) - 1, num_samples, dtype=int)
            
            # Extract minimal data needed for visualization (no large feature arrays)
            raw_cases_for_viz[case_type] = []
            for idx in indices:
                case = cases[idx]
                raw_cases_for_viz[case_type].append({
                    'frame_idx': case['frame_idx'],
                    'clip_idx': case['clip_idx'],
                    'boundary_prob': case['boundary_prob'],
                    'type_probs': case['type_probs'],
                    'acceleration': case['acceleration'],
                    'jerk': case['jerk'],
                    'pixel_motion': case['pixel_motion'].tolist() if isinstance(case['pixel_motion'], np.ndarray) else case['pixel_motion'],
                    'predicted_label': case.get('predicted_label', None),
                })
        
        # Save detailed analysis
        analysis_file = output_dir / 'deep_failure_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump({
                'analysis': analysis,
                'comparison': comparison,
                'num_cases': {k: len(v) for k, v in failure_cases.items()},
                'raw_cases': raw_cases_for_viz,  # For visualization
            }, f, indent=2)
        
        print(f"\n✅ Deep analysis saved to: {analysis_file}")
    
    print("\n✅ Comprehensive evaluation completed!")
    print("="*80)


if __name__ == '__main__':
    main()
