"""
Training script for Hybrid Boundary Detector - 2026 SOTA

This combines the best insights from:
1. TransNetV2: Multi-scale dilated convolutions
2. Modern methods: Lightweight temporal attention
3. Our research: Explicit frame differences, progressive fusion

Expected improvements:
- Better localization (dilated convs for local patterns)
- Better sequence understanding (lightweight attention)
- More efficient (1.5M params vs 9.2M)
- Faster training (fewer parameters)

USAGE:
  # Train from scratch
  python train_hybrid_detector.py
  
  # Resume from checkpoint
  python train_hybrid_detector.py --resume outputs/hybrid_detector/best_model.pt
  
  # Resume and train for more epochs
  python train_hybrid_detector.py --resume outputs/hybrid_detector/best_model.pt --epochs 100
  
  # Resume with different batch size
  python train_hybrid_detector.py --resume outputs/hybrid_detector/hybrid_detector_epoch50.pt --batch_size 16
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import random as _random
import wandb
from tqdm import tqdm
import json
import random
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.hybrid_boundary_detector import create_hybrid_detector
from core.binary_boundary_loss import create_binary_boundary_loss, create_one_hot_many_hot

# Import existing utilities
from utils import (
    set_seed, 
    worker_init_fn,
    compute_transition_metrics,
    extract_transitions_from_predictions,
    CONFIG as BASE_CONFIG
)
from datasets import PredecodedDataset


# ---------------------------------------------------------------------------
# AutoShot-compatible scene evaluation utilities
# Adapted from: https://github.com/StanislavPetrovV/AutoShot
# ---------------------------------------------------------------------------

def predictions_to_scenes(predictions):
    """Convert many_hot binary array (1=transition, 0=scene) to list of scene [start, end] intervals."""
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(gt_scenes, pred_scenes, return_mistakes=False, n_frames_miss_tolerance=2):
    """
    Standard AutoShot scene detection evaluation.
    Adapted from: https://github.com/gyglim/shot-detection-evaluation

    A predicted transition is a TP if it overlaps a GT transition within
    n_frames_miss_tolerance frames.
    """
    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

    i, j = 0, 0
    tp, fp, fn = 0, 0, 0
    fp_mistakes, fn_mistakes = [], []

    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            fn_mistakes.append(gt_trans[i])
            i += 1
        elif i == len(gt_trans):
            fp += 1
            fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            fn_mistakes.append(gt_trans[i])
            i += 1
        else:
            i += 1
            j += 1
            tp += 1

    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = (2 * p * r) / (p + r) if p + r != 0 else 0

    if return_mistakes:
        return p, r, f1, (tp, fp, fn), fp_mistakes, fn_mistakes
    return p, r, f1, (tp, fp, fn)


def _build_pred_many_hot(y_pred, z_pred, boundary_threshold, gradual_threshold):
    """Combine y (one_hot midpoint) and z (many_hot extent) into a single many_hot array.

    - Gradual transition frames: z_pred > gradual_threshold (already many_hot)
    - Cut transition frames: y_pred > boundary_threshold AND surrounding z is low
      (mark just that single midpoint frame as a 1-frame transition)
    """
    pred_is_boundary = y_pred > boundary_threshold      # sparse midpoint spikes
    pred_is_gradual = z_pred > gradual_threshold         # dense gradual extent
    # Cuts: midpoint spikes that are not already covered by gradual extent
    pred_is_cut = pred_is_boundary & ~pred_is_gradual
    pred_many_hot = (pred_is_gradual | pred_is_cut).astype(np.uint8)
    return pred_many_hot, pred_is_cut


# ---------------------------------------------------------------------------


def augment_frames(frames):
    """
    Inline augmentation on [B, T, C, H, W] frames (values in [0,1]).
    All gating is per-clip (not per-batch), so each of the B clips gets
    an independent decision on whether each augmentation applies.
    Runs on GPU after data is moved — faster than CPU-side __getitem__.
    """
    B, T, C, H, W = frames.shape

    # 1. Horizontal flip — per-clip gate, same across T (preserves temporal structure)
    flip_mask = torch.rand(B, device=frames.device) < 0.5          # [B]
    frames[flip_mask] = frames[flip_mask].flip(-1)

    # 2. Per-frame brightness — each clip independently decides (50% chance)
    apply = torch.rand(B, device=frames.device) < 0.5              # [B]
    if apply.any():
        scale = 1.0 + (torch.rand(B, T, 1, 1, 1, device=frames.device) * 0.4 - 0.2)
        scale[~apply] = 1.0                                         # no-op for non-selected clips
        frames = (frames * scale).clamp(0.0, 1.0)

    # 3. Per-frame contrast — each clip independently decides (50% chance)
    apply = torch.rand(B, device=frames.device) < 0.5              # [B]
    if apply.any():
        factor = 1.0 + (torch.rand(B, T, 1, 1, 1, device=frames.device) * 0.3 - 0.15)
        factor[~apply] = 1.0
        mean = frames.mean(dim=(2, 3, 4), keepdim=True)
        frames = ((frames - mean) * factor + mean).clamp(0.0, 1.0)

    # 4. Gaussian noise — each clip independently decides (50% chance, stronger noise)
    # This directly disrupts the model memorising exact pixel values from training clips.
    apply = torch.rand(B, device=frames.device) < 0.5              # [B] raised from 30% to 50%
    if apply.any():
        noise = torch.randn(B, T, C, H, W, device=frames.device) * 0.02  # raised 0.01→0.02
        noise[~apply] = 0.0
        frames = (frames + noise).clamp(0.0, 1.0)

    # 5. Global brightness shift — per-clip (50% chance)
    # Shifts the whole clip up/down by a constant; different from per-frame scaling.
    apply = torch.rand(B, device=frames.device) < 0.5
    if apply.any():
        shift = (torch.rand(B, 1, 1, 1, 1, device=frames.device) * 0.2 - 0.1)  # [-0.1, +0.1]
        shift[~apply] = 0.0
        frames = (frames + shift).clamp(0.0, 1.0)

    return frames


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config):
    """Train for one epoch with dual-head (one_hot + many_hot) model"""
    model.train()
    
    total_loss = 0
    total_y_loss = 0
    total_z_loss = 0
    
    # Metrics tracking for progress bar
    tp_boundary = 0
    fp_boundary = 0
    fn_boundary = 0
    
    boundary_threshold = config['boundary_threshold']
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch_data in enumerate(pbar):
        frames, labels, transition_lengths = batch_data
        
        frames = frames.to(device, non_blocking=True)  # [B, T, C, H, W]
        frames = augment_frames(frames)                 # random flip + color jitter + noise
        labels = labels.to(device, non_blocking=True)  # [B, T] - 0=none, 1=cut, 2=gradual
        transition_lengths = transition_lengths.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            y_logits, z_logits = model(frames)
            loss, loss_dict = criterion(y_logits, z_logits, labels, transition_lengths)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Tightened 1.0→0.5
        scaler.step(optimizer)
        scaler.update()
        
        # Track losses
        total_loss += loss.item()
        total_y_loss += loss_dict['y_loss']
        total_z_loss += loss_dict['z_loss']
        
        num_batches += 1
        
        # Compute frame-level binary metrics for progress bar.
        # Compare y_pred against one_hot targets (midpoints only) — same signal
        # the loss trains on. Using (labels > 0) is wrong: it penalises y for not
        # firing on every gradual frame, causing recall to look artificially low.
        with torch.no_grad():
            y_probs = torch.sigmoid(y_logits).squeeze(-1)  # [B, T]
            boundary_preds = (y_probs > boundary_threshold).long()
            one_hot_targets, _ = create_one_hot_many_hot(labels)
            boundary_targets = (one_hot_targets > 0).long()
            
            tp_boundary += (boundary_preds & boundary_targets.bool()).sum().item()
            fp_boundary += (boundary_preds & ~boundary_targets.bool()).sum().item()
            fn_boundary += (~boundary_preds.bool() & boundary_targets.bool()).sum().item()
        
        # Compute epoch-averaged metrics for progress bar
        avg_loss = total_loss / num_batches
        precision = tp_boundary / (tp_boundary + fp_boundary + 1e-8)
        recall = tp_boundary / (tp_boundary + fn_boundary + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{avg_loss:.4f}",
            'F1': f"{f1:.3f}",
            'P': f"{precision:.3f}",
            'R': f"{recall:.3f}"
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_y_loss = total_y_loss / len(train_loader)
    avg_z_loss = total_z_loss / len(train_loader)
    
    # Final epoch metrics
    final_precision = tp_boundary / (tp_boundary + fp_boundary + 1e-8)
    final_recall = tp_boundary / (tp_boundary + fn_boundary + 1e-8)
    final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-8)
    
    return {
        'loss': avg_loss,
        'y_loss': avg_y_loss,
        'z_loss': avg_z_loss,
        'boundary_f1': final_f1,
        'boundary_precision': final_precision,
        'boundary_recall': final_recall
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    """
    Validate using the standard AutoShot scene evaluation (evaluate_scenes).

    Primary metric: overall scene-detection F1 at the best threshold found by
    sweeping a small set of candidates over the buffered predictions.
    Secondary metrics: cut-only and gradual-only F1 (approx, for debug).
    """
    model.eval()

    total_loss = 0.0
    total_y_loss = 0.0
    total_z_loss = 0.0

    gradual_threshold = config.get('gradual_threshold', 0.5)

    # Buffer per-clip predictions & GT so we can sweep thresholds at the end.
    # Each entry: (y_pred [T], z_pred [T], gt_labels [T])
    clip_predictions = []

    for batch_data in tqdm(val_loader, desc="Validating"):
        frames, labels, transition_lengths = batch_data

        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        transition_lengths = transition_lengths.to(device, non_blocking=True)

        y_logits, z_logits = model(frames)
        loss, loss_dict = criterion(y_logits, z_logits, labels, transition_lengths)

        total_loss += loss.item()
        total_y_loss += loss_dict['y_loss']
        total_z_loss += loss_dict['z_loss']

        y_probs = torch.sigmoid(y_logits.squeeze(-1)).cpu().numpy()  # [B, T]
        z_probs = torch.sigmoid(z_logits.squeeze(-1)).cpu().numpy()  # [B, T]
        gt_np   = labels.cpu().numpy()                               # [B, T]

        for b in range(frames.shape[0]):
            clip_predictions.append((y_probs[b], z_probs[b], gt_np[b]))

    # -----------------------------------------------------------------------
    # Sweep thresholds on y-head to find the best overall F1
    # (mirrors AutoShot's mAP_f1_p_fix_r threshold search)
    # -----------------------------------------------------------------------
    thresholds = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.8, 0.9])

    best_f1, best_thr = -1.0, config['boundary_threshold']
    thr_results = {}  # threshold -> (p, r, f1, tp, fp, fn)

    for thr in thresholds:
        tp, fp, fn = 0, 0, 0
        for y_pred, z_pred, gt_labels in clip_predictions:
            gt_many_hot = (gt_labels > 0).astype(np.uint8)
            gt_scenes   = predictions_to_scenes(gt_many_hot)

            pred_many_hot, _ = _build_pred_many_hot(y_pred, z_pred, thr, gradual_threshold)
            pred_scenes       = predictions_to_scenes(pred_many_hot)

            _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes, pred_scenes,
                                                        n_frames_miss_tolerance=2)
            tp += tp_
            fp += fp_
            fn += fn_

        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        thr_results[float(thr)] = (p, r, f1, tp, fp, fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    # Model selection uses the BEST SWEPT F1 (same as what is displayed).
    # Using fixed thr=0.5 was causing wrong saves: a model that peaks at thr=0.55
    # would show higher displayed F1 but lower selection_f1, getting discarded.
    # Threshold leakage is not a concern here — the swept thr is only used to pick
    # which checkpoint to save, not to update weights or stop training early.
    selection_f1 = best_f1  # best_f1 from sweep above, used for best-model saving

    # -----------------------------------------------------------------------
    # Sweep y_pred independently for cut-optimal threshold
    # -----------------------------------------------------------------------
    best_cut_f1, best_cut_thr = -1.0, 0.5
    for thr in thresholds:
        ctp, cfp, cfn = 0, 0, 0
        for y_pred, z_pred, gt_labels in clip_predictions:
            gt_cut_many   = (gt_labels == 1).astype(np.uint8)
            gt_cut_scenes = predictions_to_scenes(gt_cut_many)
            pred_cut_scenes = predictions_to_scenes((y_pred > thr).astype(np.uint8))
            if len(gt_cut_scenes) > 1 or gt_cut_many.any():
                _, _, _, (c_, f_, n_) = evaluate_scenes(
                    gt_cut_scenes, pred_cut_scenes, n_frames_miss_tolerance=2)
                ctp += c_; cfp += f_; cfn += n_
        cp = ctp / (ctp + cfp + 1e-8); cr = ctp / (ctp + cfn + 1e-8)
        cf = 2 * cp * cr / (cp + cr + 1e-8)
        if cf > best_cut_f1:
            best_cut_f1, best_cut_thr = cf, float(thr)

    # -----------------------------------------------------------------------
    # Sweep z_pred independently for gradual-optimal threshold
    # -----------------------------------------------------------------------
    best_grad_f1, best_grad_thr = -1.0, 0.5
    for thr in thresholds:
        gtp, gfp, gfn = 0, 0, 0
        for y_pred, z_pred, gt_labels in clip_predictions:
            gt_grad_many   = (gt_labels == 2).astype(np.uint8)
            gt_grad_scenes = predictions_to_scenes(gt_grad_many)
            pred_grad_scenes = predictions_to_scenes((z_pred > thr).astype(np.uint8))
            if len(gt_grad_scenes) > 1 or gt_grad_many.any():
                _, _, _, (g_, f_, n_) = evaluate_scenes(
                    gt_grad_scenes, pred_grad_scenes, n_frames_miss_tolerance=2)
                gtp += g_; gfp += f_; gfn += n_
        gp = gtp / (gtp + gfp + 1e-8); gr = gtp / (gtp + gfn + 1e-8)
        gf = 2 * gp * gr / (gp + gr + 1e-8)
        if gf > best_grad_f1:
            best_grad_f1, best_grad_thr = gf, float(thr)

    # -----------------------------------------------------------------------
    # Compute final metrics at their respective best thresholds
    # -----------------------------------------------------------------------
    ov_tp, ov_fp, ov_fn = 0, 0, 0
    cut_tp, cut_fp, cut_fn = 0, 0, 0
    grad_tp, grad_fp, grad_fn = 0, 0, 0

    for y_pred, z_pred, gt_labels in clip_predictions:
        # --- overall (combined y+z, AutoShot standard) at best_thr ---
        gt_many_hot = (gt_labels > 0).astype(np.uint8)
        gt_scenes   = predictions_to_scenes(gt_many_hot)
        pred_many_hot, _ = _build_pred_many_hot(y_pred, z_pred, best_thr, gradual_threshold)
        pred_scenes = predictions_to_scenes(pred_many_hot)
        _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes, pred_scenes,
                                                    n_frames_miss_tolerance=2)
        ov_tp += tp_;  ov_fp += fp_;  ov_fn += fn_

        # --- cut-only: head_y at its own optimal threshold ---
        gt_cut_many   = (gt_labels == 1).astype(np.uint8)
        gt_cut_scenes = predictions_to_scenes(gt_cut_many)
        pred_cut_scenes = predictions_to_scenes((y_pred > best_cut_thr).astype(np.uint8))
        if len(gt_cut_scenes) > 1 or gt_cut_many.any():
            _, _, _, (c_tp, c_fp, c_fn) = evaluate_scenes(
                gt_cut_scenes, pred_cut_scenes, n_frames_miss_tolerance=2)
            cut_tp += c_tp;  cut_fp += c_fp;  cut_fn += c_fn

        # --- gradual-only: head_z at its own optimal threshold ---
        gt_grad_many   = (gt_labels == 2).astype(np.uint8)
        gt_grad_scenes = predictions_to_scenes(gt_grad_many)
        pred_grad_scenes = predictions_to_scenes((z_pred > best_grad_thr).astype(np.uint8))
        if len(gt_grad_scenes) > 1 or gt_grad_many.any():
            _, _, _, (g_tp, g_fp, g_fn) = evaluate_scenes(
                gt_grad_scenes, pred_grad_scenes, n_frames_miss_tolerance=2)
            grad_tp += g_tp;  grad_fp += g_fp;  grad_fn += g_fn

    def _prf(tp, fp, fn):
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        return p, r, f1

    ov_p,  ov_r,  ov_f1  = _prf(ov_tp, ov_fp, ov_fn)
    cut_p, cut_r, cut_f1 = _prf(cut_tp, cut_fp, cut_fn)
    grd_p, grd_r, grd_f1 = _prf(grad_tp, grad_fp, grad_fn)

    n_batches = len(val_loader)
    return {
        'cut': {
            'tp': cut_tp, 'fp': cut_fp, 'fn': cut_fn,
            'precision': cut_p, 'recall': cut_r, 'f1': cut_f1
        },
        'gradual': {
            'tp': grad_tp, 'fp': grad_fp, 'fn': grad_fn,
            'precision': grd_p, 'recall': grd_r, 'f1': grd_f1
        },
        'overall': {
            'tp': ov_tp, 'fp': ov_fp, 'fn': ov_fn,
            'precision': ov_p, 'recall': ov_r, 'f1': ov_f1
        },
        'best_threshold': best_thr,          # best swept threshold (diagnostic only)
        'selection_f1': selection_f1,          # F1 at fixed thr=0.5 — used for model selection
        'best_cut_threshold': best_cut_thr,    # y_pred optimal for cuts
        'best_grad_threshold': best_grad_thr,  # z_pred optimal for graduals
        'loss':   total_loss   / n_batches,
        'y_loss': total_y_loss / n_batches,
        'z_loss': total_z_loss / n_batches,
    }


def make_serializable(obj):
    """Convert numpy/torch types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    else:
        return obj


def plot_training_history(history, save_dir):
    """
    Generate comprehensive training history plots
    
    Args:
        history: Dictionary with 'train' and 'val' lists of epoch metrics
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not history['train'] or not history['val']:
        print("⚠ No training history to plot")
        return
    
    epochs = [m['epoch'] for m in history['train']]
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Loss curves
    ax1 = plt.subplot(3, 3, 1)
    train_loss = [m['loss'] for m in history['train']]
    val_loss = [m['loss'] for m in history['val']]
    ax1.plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Y-head loss (one_hot - boundary midpoint)
    ax2 = plt.subplot(3, 3, 2)
    train_y = [m.get('y_loss', 0) for m in history['train']]
    val_y = [m.get('y_loss', 0) for m in history['val']]
    ax2.plot(epochs, train_y, 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, val_y, 'r-', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Y Loss')
    ax2.set_title('Y-Head Loss (one_hot: boundary midpoint)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Z-head loss (many_hot - gradual extent)
    ax3 = plt.subplot(3, 3, 3)
    train_z = [m.get('z_loss', 0) for m in history['train']]
    val_z = [m.get('z_loss', 0) for m in history['val']]
    ax3.plot(epochs, train_z, 'b-', label='Train', linewidth=2)
    ax3.plot(epochs, val_z, 'r-', label='Val', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Z Loss')
    ax3.set_title('Z-Head Loss (many_hot: gradual extent)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Overall F1
    ax4 = plt.subplot(3, 3, 4)
    val_overall_f1 = [m.get('overall_f1', 0) for m in history['val']]
    ax4.plot(epochs, val_overall_f1, 'g-', linewidth=2, marker='o', markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Overall F1')
    ax4.set_title('Overall F1 Score (Val)')
    ax4.grid(alpha=0.3)
    best_epoch = epochs[np.argmax(val_overall_f1)]
    best_f1 = max(val_overall_f1)
    ax4.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_f1:.3f} @ epoch {best_epoch}')
    ax4.legend()
    
    # 5. Cut F1
    ax5 = plt.subplot(3, 3, 5)
    val_cut_f1 = [m.get('cut_f1', 0) for m in history['val']]
    ax5.plot(epochs, val_cut_f1, 'b-', linewidth=2, marker='s', markersize=4)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Cut F1')
    ax5.set_title('Cut Detection F1 (Val)')
    ax5.grid(alpha=0.3)
    
    # 6. Gradual F1
    ax6 = plt.subplot(3, 3, 6)
    val_gradual_f1 = [m.get('gradual_f1', 0) for m in history['val']]
    ax6.plot(epochs, val_gradual_f1, 'm-', linewidth=2, marker='^', markersize=4)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Gradual F1')
    ax6.set_title('Gradual Detection F1 (Val)')
    ax6.grid(alpha=0.3)
    
    # 7. Precision & Recall (Overall)
    ax7 = plt.subplot(3, 3, 7)
    val_precision = [m.get('overall_precision', 0) for m in history['val']]
    val_recall = [m.get('overall_recall', 0) for m in history['val']]
    ax7.plot(epochs, val_precision, 'b-', label='Precision', linewidth=2)
    ax7.plot(epochs, val_recall, 'r-', label='Recall', linewidth=2)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Score')
    ax7.set_title('Overall Precision & Recall (Val)')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    # 8. Learning Rate
    ax8 = plt.subplot(3, 3, 8)
    if 'lr' in history['train'][0]:
        lrs = [m['lr'] for m in history['train']]
        ax8.plot(epochs, lrs, 'purple', linewidth=2)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Learning Rate')
        ax8.set_title('Learning Rate Schedule')
        ax8.set_yscale('log')
        ax8.grid(alpha=0.3)
    
    # 9. F1 Comparison
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(epochs, val_overall_f1, 'g-', label='Overall', linewidth=2, marker='o', markersize=3)
    ax9.plot(epochs, val_cut_f1, 'b-', label='Cut', linewidth=2, marker='s', markersize=3)
    ax9.plot(epochs, val_gradual_f1, 'm-', label='Gradual', linewidth=2, marker='^', markersize=3)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('F1 Score')
    ax9.set_title('F1 Comparison (Val)')
    ax9.legend()
    ax9.grid(alpha=0.3)
    
    plt.suptitle('Training History - Hybrid Boundary Detector', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Training history plot saved: {plot_path}")


def merge_histories(history1, history2):
    """
    Merge two training histories, preferring history2 for overlapping epochs
    
    Args:
        history1: First history dict (lower priority)
        history2: Second history dict (higher priority, overwrites conflicts)
        
    Returns:
        Merged history dict
    """
    merged = {'train': [], 'val': []}
    
    # Combine all epochs from both histories
    all_train_epochs = {}
    all_val_epochs = {}
    
    # Add from history1 first
    for record in history1.get('train', []):
        epoch = record.get('epoch')
        if epoch is not None:
            all_train_epochs[epoch] = record
    
    for record in history1.get('val', []):
        epoch = record.get('epoch')
        if epoch is not None:
            all_val_epochs[epoch] = record
    
    # Overwrite with history2 (higher priority)
    for record in history2.get('train', []):
        epoch = record.get('epoch')
        if epoch is not None:
            all_train_epochs[epoch] = record
    
    for record in history2.get('val', []):
        epoch = record.get('epoch')
        if epoch is not None:
            all_val_epochs[epoch] = record
    
    # Sort by epoch and create lists
    merged['train'] = [all_train_epochs[e] for e in sorted(all_train_epochs.keys())]
    merged['val'] = [all_val_epochs[e] for e in sorted(all_val_epochs.keys())]
    
    return merged


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, scaler=None, device='cuda', checkpoint_dir=None):
    """
    Load training checkpoint and resume training state
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        scaler: GradScaler to load state into (optional)
        device: Device to map checkpoint to
        checkpoint_dir: Directory where training outputs are saved (for finding history JSON)
        
    Returns:
        epoch: Last completed epoch number
        best_val_f1: Best validation F1 score
        patience_counter: Current patience counter value
        history: Training history dict
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Model state loaded")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  ✓ Scheduler state loaded")
    
    # Load scaler state
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"  ✓ GradScaler state loaded")
    
    # Get training state
    epoch = checkpoint.get('epoch', 0)
    val_metrics = checkpoint.get('val_metrics', {})
    best_val_f1 = val_metrics.get('selection_f1', val_metrics.get('overall', {}).get('f1', 0.0))
    
    # Try to get patience counter from checkpoint (may not exist in older checkpoints)
    patience_counter = checkpoint.get('patience_counter', 0)
    
    # Get training history from checkpoint (may not exist in older checkpoints)
    history_from_checkpoint = checkpoint.get('history', {'train': [], 'val': []})
    
    # Try to load existing training_history.json file (from previous runs or wandb extraction)
    history_from_json = {'train': [], 'val': []}
    if checkpoint_dir:
        history_json_path = Path(checkpoint_dir) / 'training_history.json'
        if history_json_path.exists():
            try:
                with open(history_json_path, 'r') as f:
                    history_from_json = json.load(f)
                print(f"  ✓ Found existing training_history.json with {len(history_from_json.get('train', []))} epochs")
            except Exception as e:
                print(f"  ⚠ Could not load training_history.json: {e}")
    
    # Merge histories intelligently
    # Priority: checkpoint history > JSON file history (checkpoint is more recent)
    history = merge_histories(history_from_json, history_from_checkpoint)
    
    print(f"  ✓ Resuming from epoch {epoch}")
    print(f"  ✓ Best validation F1: {best_val_f1:.4f}")
    print(f"  ✓ Patience counter: {patience_counter}/{checkpoint.get('config', {}).get('patience', 'N/A')}")
    print(f"  ✓ Training history: {len(history['train'])} epochs loaded (merged from all sources)")
    
    return epoch, best_val_f1, patience_counter, history


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Hybrid Boundary Detector')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from (e.g., outputs/hybrid_detector/best_model.pt)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    args = parser.parse_args()
    
    # Configuration
    config = {
        **BASE_CONFIG,  # Inherit all base config
        
        # Model architecture
        'main_dim': 384,          # Reduced 512→384: 512 has too much capacity for this dataset size
        'num_fusion_blocks': 4,
        
        # Resume training
        'resume': args.resume,  # Path to checkpoint to resume from
        
        # Pre-decoded dataset path (autoshot_cache_train/train has 45,274 clips, 648 videos)
        'predecoded_cache': r"D:\grad\data\autoshot_cache_train",
        'use_predecoded': True,
        
        # Training
        'batch_size': 32,
        'num_epochs': 80,
        'learning_rate': 5e-5,    # 1e-4 caused memorisation by epoch 4
        'warmup_epochs': 5,       # slower ramp prevents early overfitting
        'weight_decay': 0.05,
        'patience': 15,

        # Loss (AutoShot dual-head: one_hot + many_hot)
        # ClipShots facts: 79% cuts, 21% graduals; gradual length min=2 max=243 mean=13 median=10
        # lambda2 raised 0.1→0.3: z-head gets 3× more gradient signal; needed because
        #   ClipShots graduals (mean=13 frames) are shorter than the model's 127-frame window,
        #   so z-head learns slowly without extra weight.
        'lambda1': 5.0,
        'lambda2': 0.3,
        # focal_alpha (y-head): 0.80 — one_hot positives are ~0.8% of frames → high alpha needed.
        #   Run 14 test: lowering to 0.70 hurt AutoShot F1 by 0.008 while gaining only 0.001
        #   on ClipShots. Reverted. The FP problem is a feature gap (fast pans look like cuts
        #   to FrameSimilarity), not a loss-weight problem.
        # focal_alpha_z (z-head): 0.50 balanced; fixed FP inflation on the z-head in run 13.
        'focal_alpha': 0.80,
        'focal_alpha_z': 0.50,
        'focal_gamma': 2.0,
        'boundary_threshold': 0.5,
        'gradual_threshold': 0.5,

        # Data loading
        'val_split_ratio': 0.15,
        'num_workers': 6,
        'prefetch_factor': 4,

        # Paths
        'checkpoint_dir': 'outputs/hybrid_detector19',
        'wandb_project': 'hybrid-boundary-detector13',

        # Mixed training
        # Real dataset sizes:
        #   AutoShot train (85% of 45274): 38,482 clips  → ~1202 batches
        #   ClipShots train total:         73,120 clips
        # With mix_fraction=0.5:  AS=55%, CS=45% — model stays AutoShot-biased
        # With mix_fraction=1.0:  AS=34%, CS=66% — matches the target eval distribution
        'clipshots_cache': r'D:\grad\data\training_cache',
        'clipshots_mix_fraction': 0.7,  # use ALL ClipShots clips; resampled each epoch
        'clipshots_val_clips': 3000,    # ClipShots val clips added to val set (capped for speed)
    }
    
    # Apply command-line overrides
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    # Setup
    set_seed(config.get('random_seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=config['wandb_project'],
        config=config,
        name=f"hybrid_m{config['main_dim']}_b{config['num_fusion_blocks']}"
    )
    
    # Load datasets
    print("\nLoading datasets...")
    
    if config['use_predecoded']:
        # Use pre-decoded dataset (FAST!)
        cache_path = config['predecoded_cache']
        
        # Check if pre-decoded files exist
        train_mapping = Path(cache_path) / 'train' / 'index_mapping.json'
        
        if not train_mapping.exists():
            print("="*80)
            print("⚠️  PRE-DECODED DATASET NOT FOUND!")
            print("="*80)
            print("Please run this command first:")
            print(f"  python predecode_dataset.py --split train --num_workers 8")
            print("="*80)
            return
        
        # Split from the train cache (has all videos)
        train_dataset = PredecodedDataset(
            cache_path, 
            split='train',
            val_split_ratio=config.get('val_split_ratio', 0.15),
            random_seed=config.get('random_seed', 42),
            temporal_jitter=True,   # time-reversal: scrambles fixed similarity matrices
        )
        val_dataset = PredecodedDataset(
            cache_path, 
            split='val',
            val_split_ratio=config.get('val_split_ratio', 0.15),
            random_seed=config.get('random_seed', 42),
            temporal_jitter=False,  # no jitter on val
        )

        # --- Mix in ClipShots training clips ---
        # AutoShot train = 648 videos. The model memorises their specific clip patterns
        # within 4 epochs. Adding ClipShots (different distribution, ~3700 videos) forces
        # the model to learn general features instead of AutoShot-specific ones.
        # The ClipShots SUBSET IS RESAMPLED EVERY EPOCH so all 62K clips are seen
        # over time and no fixed subset can be memorised.
        clipshots_cache = config.get('clipshots_cache', '')
        mix_fraction    = config.get('clipshots_mix_fraction', 0.0)
        clipshots_full_dataset = None    # kept alive for per-epoch resampling
        autoshot_train_dataset = train_dataset   # base AutoShot reference
        if clipshots_cache and mix_fraction > 0 and Path(clipshots_cache).exists():
            clipshots_full_dataset = PredecodedDataset(
                clipshots_cache,
                split='train',
                val_split_ratio=config.get('val_split_ratio', 0.15),
                random_seed=config.get('random_seed', 42),
                temporal_jitter=True,
            )
            # Build epoch-0 subset (will be replaced before epoch 1 in the training loop)
            n_mix_init = int(len(clipshots_full_dataset) * mix_fraction)
            init_indices = list(_random.sample(range(len(clipshots_full_dataset)), n_mix_init))
            clipshots_sub = torch.utils.data.Subset(clipshots_full_dataset, init_indices)
            train_dataset = ConcatDataset([autoshot_train_dataset, clipshots_sub])
            print(f"✓ ClipShots mixing: {n_mix_init} clips ({mix_fraction*100:.0f}% of {len(clipshots_full_dataset)}) — resampled each epoch")
            print(f"✓ Initial combined train size: {len(train_dataset)} clips")
        # ----------------------------------------
        
        print(f"✓ Using pre-decoded dataset (10-20x faster!)")
        print(f"✓ Train: {len(train_dataset)} clips, Val: {len(val_dataset)} clips")
    else:
        # Fallback to original ClipShotsDataset (slower)
        print("Using on-the-fly video decoding (slower)...")
        from datasets import ClipShotsDataset
        
        val_json_path = os.path.join(config['input_root'], 'annotations', 'val.json')
        
        if os.path.exists(val_json_path):
            train_dataset = ClipShotsDataset(
                config['input_root'], 
                split='train',
                seq_len=config['seq_len'],
                cache_capacity=config['cache_capacity'],
                deterministic=False
            )
            val_dataset = ClipShotsDataset(
                config['input_root'],
                split='val',
                seq_len=config['seq_len'],
                cache_capacity=config['cache_capacity'],
                deterministic=True,
                samples_per_video=5
            )
        else:
            temp_ds = ClipShotsDataset(
                config['input_root'],
                split='train',
                seq_len=config['seq_len'],
                cache_capacity=config['cache_capacity']
            )
            
            all_videos = sorted(temp_ds.valid_keys)
            n_val = max(1, int(len(all_videos) * config['val_split_ratio']))
            
            rng = random.Random(config.get('random_seed', 42))
            rng.shuffle(all_videos)
            val_videos = set(all_videos[:n_val])
            
            train_dataset = ClipShotsDataset(
                config['input_root'],
                split='train',
                seq_len=config['seq_len'],
                cache_capacity=config['cache_capacity'],
                deterministic=False,
                val_videos=val_videos
            )
            val_dataset = ClipShotsDataset(
                config['input_root'],
                split='train',
                seq_len=config['seq_len'],
                cache_capacity=config['cache_capacity'],
                deterministic=True,
                val_videos=val_videos,
                samples_per_video=5
            )
            val_dataset.split = 'val'
        
        print(f"✓ Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Ensure per-epoch resampling variables exist even in non-predecoded path
    if 'clipshots_full_dataset' not in dir():
        clipshots_full_dataset = None
    if 'autoshot_train_dataset' not in dir():
        autoshot_train_dataset = train_dataset

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=config['prefetch_factor']  # Use config value
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=config['prefetch_factor']  # Use config value
    )
    
    # Create model
    print("\n" + "="*80)
    model = create_hybrid_detector(
        main_dim=config['main_dim'],
        num_fusion_blocks=config['num_fusion_blocks']
    )
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params = model.count_parameters()
    wandb.config.update({
        'total_params': total_params,
        'trainable_params': trainable_params
    })
    
    # Create loss function (AutoShot dual-head)
    criterion = create_binary_boundary_loss({
        'lambda1': config['lambda1'],
        'lambda2': config['lambda2'],
        'focal_alpha': config['focal_alpha'],
        'focal_alpha_z': config.get('focal_alpha_z', None),
        'focal_gamma': config['focal_gamma'],
        'z_label_smoothing': 0.1,  # Soft many_hot targets: reduces z_loss memorization
    }).to(device)
    
    print(f"\nLoss configuration (AutoShot dual-head):")
    print(f"  lambda1 (y-head, one_hot):  {config['lambda1']:.2f}")
    print(f"  lambda2 (z-head, many_hot): {config['lambda2']:.2f}")
    print(f"  focal_alpha:                {config['focal_alpha']:.3f}")
    print(f"  focal_gamma:                {config['focal_gamma']:.2f}")
    print(f"  boundary_threshold:         {config['boundary_threshold']:.2f}")
    print(f"  gradual_threshold:          {config['gradual_threshold']:.2f}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler: linear warmup then cosine decay
    warmup_epochs = config.get('warmup_epochs', 5)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.25,      # Start at 25% of peak LR (was 10% = too slow)
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] - warmup_epochs,
        eta_min=5e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    if config.get('resume'):
        if os.path.exists(config['resume']):
            start_epoch, best_val_f1, patience_counter, history = load_checkpoint(
                model,
                config['resume'],
                optimizer,
                scheduler,
                scaler,
                device,
                checkpoint_dir=config['checkpoint_dir']
            )
            print(f"\n✓ Successfully resumed from epoch {start_epoch}")
            print(f"  Next epoch will be: {start_epoch + 1}")
        else:
            print(f"\n⚠ Warning: Resume checkpoint not found: {config['resume']}")
            print("  Starting training from scratch...")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    for epoch in range(start_epoch + 1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 80)
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")

        # Resample ClipShots subset each epoch — different 50% every time.
        # All 62K clips are seen over 2 full epochs; the model cannot memorise
        # any fixed subset. Rebuilding DataLoader takes <100ms (index assignment only).
        if clipshots_full_dataset is not None:
            n_mix = int(len(clipshots_full_dataset) * config.get('clipshots_mix_fraction', 0.5))
            epoch_indices = _random.sample(range(len(clipshots_full_dataset)), n_mix)
            epoch_clipshots = torch.utils.data.Subset(clipshots_full_dataset, epoch_indices)
            epoch_train_dataset = ConcatDataset([autoshot_train_dataset, epoch_clipshots])
            train_loader = DataLoader(
                epoch_train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                pin_memory=True,
                persistent_workers=False,   # must be False when recreating each epoch
                worker_init_fn=worker_init_fn,
                prefetch_factor=config['prefetch_factor'],
            )
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track history (make serializable)
        train_record = make_serializable({
            'epoch': epoch,
            'loss': train_metrics['loss'],
            'y_loss': train_metrics['y_loss'],
            'z_loss': train_metrics['z_loss'],
            'boundary_f1': train_metrics['boundary_f1'],
            'boundary_precision': train_metrics['boundary_precision'],
            'boundary_recall': train_metrics['boundary_recall'],
            'lr': current_lr
        })
        
        val_record = make_serializable({
            'epoch': epoch,
            'loss': val_metrics['loss'],
            'y_loss': val_metrics['y_loss'],
            'z_loss': val_metrics['z_loss'],
            'best_threshold': val_metrics['best_threshold'],
            'cut_f1': val_metrics['cut']['f1'],
            'cut_precision': val_metrics['cut']['precision'],
            'cut_recall': val_metrics['cut']['recall'],
            'cut_tp': val_metrics['cut']['tp'],
            'cut_fp': val_metrics['cut']['fp'],
            'cut_fn': val_metrics['cut']['fn'],
            'gradual_f1': val_metrics['gradual']['f1'],
            'gradual_precision': val_metrics['gradual']['precision'],
            'gradual_recall': val_metrics['gradual']['recall'],
            'gradual_tp': val_metrics['gradual']['tp'],
            'gradual_fp': val_metrics['gradual']['fp'],
            'gradual_fn': val_metrics['gradual']['fn'],
            'overall_f1': val_metrics['overall']['f1'],
            'overall_precision': val_metrics['overall']['precision'],
            'overall_recall': val_metrics['overall']['recall'],
            'overall_tp': val_metrics['overall']['tp'],
            'overall_fp': val_metrics['overall']['fp'],
            'overall_fn': val_metrics['overall']['fn'],
        })
        
        history['train'].append(train_record)
        history['val'].append(val_record)
        
        # Log to wandb
        wandb_metrics = {
            'epoch': epoch,
            'lr': current_lr,
            'train/loss': train_metrics['loss'],
            'train/y_loss': train_metrics['y_loss'],
            'train/z_loss': train_metrics['z_loss'],
            'train/boundary_f1': train_metrics['boundary_f1'],
            'train/boundary_precision': train_metrics['boundary_precision'],
            'train/boundary_recall': train_metrics['boundary_recall'],
            'val/loss': val_metrics['loss'],
            'val/y_loss': val_metrics['y_loss'],
            'val/z_loss': val_metrics['z_loss'],
            'val/cut_f1': val_metrics['cut']['f1'],
            'val/cut_precision': val_metrics['cut']['precision'],
            'val/cut_recall': val_metrics['cut']['recall'],
            'val/gradual_f1': val_metrics['gradual']['f1'],
            'val/gradual_precision': val_metrics['gradual']['precision'],
            'val/gradual_recall': val_metrics['gradual']['recall'],
            'val/overall_f1': val_metrics['overall']['f1'],
            'val/overall_precision': val_metrics['overall']['precision'],
            'val/overall_recall': val_metrics['overall']['recall'],
            'val/best_threshold': val_metrics['best_threshold'],
            'val/best_cut_threshold': val_metrics['best_cut_threshold'],
            'val/best_grad_threshold': val_metrics['best_grad_threshold'],
        }
        
        wandb.log(wandb_metrics)
        
        # Print results
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Y: {train_metrics['y_loss']:.4f}, "
              f"Z: {train_metrics['z_loss']:.4f}, "
              f"F1: {train_metrics['boundary_f1']:.3f}, "
              f"P: {train_metrics['boundary_precision']:.3f}, "
              f"R: {train_metrics['boundary_recall']:.3f}")
        
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Y: {val_metrics['y_loss']:.4f}, "
              f"Z: {val_metrics['z_loss']:.4f}")
        
        best_thr = val_metrics['best_threshold']
        best_cut_thr  = val_metrics['best_cut_threshold']
        best_grad_thr = val_metrics['best_grad_threshold']
        print(f"\nScene Detection (tol=2 | overall_thr={best_thr:.2f} | cut_thr={best_cut_thr:.2f} | grad_thr={best_grad_thr:.2f}):")
        print(f"  Overall  - P: {val_metrics['overall']['precision']:.3f}, "
              f"R: {val_metrics['overall']['recall']:.3f}, "
              f"F1: {val_metrics['overall']['f1']:.3f} "
              f"(TP: {val_metrics['overall']['tp']}, "
              f"FP: {val_metrics['overall']['fp']}, "
              f"FN: {val_metrics['overall']['fn']})")
        print(f"  Cut      - P: {val_metrics['cut']['precision']:.3f}, "
              f"R: {val_metrics['cut']['recall']:.3f}, "
              f"F1: {val_metrics['cut']['f1']:.3f} "
              f"(TP: {val_metrics['cut']['tp']}, "
              f"FP: {val_metrics['cut']['fp']}, "
              f"FN: {val_metrics['cut']['fn']})")
        print(f"  Gradual  - P: {val_metrics['gradual']['precision']:.3f}, "
              f"R: {val_metrics['gradual']['recall']:.3f}, "
              f"F1: {val_metrics['gradual']['f1']:.3f} "
              f"(TP: {val_metrics['gradual']['tp']}, "
              f"FP: {val_metrics['gradual']['fp']}, "
              f"FN: {val_metrics['gradual']['fn']})")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            config['checkpoint_dir'],
            f'hybrid_detector_epoch{epoch}.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'config': config,
            'val_metrics': val_metrics,
            'best_val_f1': best_val_f1,
            'patience_counter': patience_counter,
            'history': history
        }, checkpoint_path)
        
        # Save best model — use fixed-threshold F1 to avoid threshold leakage
        val_f1 = val_metrics['selection_f1']
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
                'best_val_f1': best_val_f1,
                'patience_counter': patience_counter,
                'history': history
            }, best_path)
            print(f"\n✓ New best model saved! F1: {val_f1:.3f}")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{config['patience']})")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            break
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation F1: {best_val_f1:.3f}")
    print("="*80)
    
    # Save final training history as JSON
    history_path = os.path.join(config['checkpoint_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved: {history_path}")
    
    # Generate and save training plots
    print("\nGenerating training history plots...")
    plot_training_history(history, config['checkpoint_dir'])
    
    # Save summary statistics
    summary = {
        'total_epochs': len(history['train']),
        'best_epoch': int(np.argmax([m['overall_f1'] for m in history['val']]) + 1) if history['val'] else 0,
        'best_val_f1': float(best_val_f1),
        'final_val_f1': float(history['val'][-1]['overall_f1']) if history['val'] else 0,
        'final_cut_f1': float(history['val'][-1]['cut_f1']) if history['val'] else 0,
        'final_gradual_f1': float(history['val'][-1]['gradual_f1']) if history['val'] else 0,
        'config': make_serializable({
            'lambda1': config['lambda1'],
            'lambda2': config['lambda2'],
            'focal_alpha': config['focal_alpha'],
            'focal_gamma': config['focal_gamma'],
            'boundary_threshold': config['boundary_threshold'],
            'gradual_threshold': config['gradual_threshold']
        })
    }
    
    summary_path = os.path.join(config['checkpoint_dir'], 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Training summary saved: {summary_path}")
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total epochs trained: {summary['total_epochs']}")
    print(f"Best epoch: {summary['best_epoch']} (F1: {summary['best_val_f1']:.4f})")
    print(f"Final validation F1: {summary['final_val_f1']:.4f}")
    print(f"  - Cut F1: {summary['final_cut_f1']:.4f}")
    print(f"  - Gradual F1: {summary['final_gradual_f1']:.4f}")
    print("="*80)
    
    wandb.finish()


if __name__ == '__main__':
    main()
