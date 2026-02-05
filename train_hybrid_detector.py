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
from torch.utils.data import DataLoader, Dataset
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
from core.binary_boundary_loss import create_binary_boundary_loss

# Import existing utilities
from train_turbo_sbd_v2 import (
    set_seed, 
    worker_init_fn,
    compute_transition_metrics,
    extract_transitions_from_predictions,
    CONFIG as BASE_CONFIG
)
from train_boundary_detector import PredecodedDataset


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_focal_loss = 0
    total_type_loss = 0
    total_reasoning_loss = 0
    total_consistency_loss = 0
    
    # Reasoning tracking (motion dynamics features)
    total_acceleration = 0
    total_jerk = 0
    
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
        labels = labels.to(device, non_blocking=True)  # [B, T] - 0=none, 1=cut, 2=gradual
        transition_lengths = transition_lengths.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            boundary_logits, type_logits, reasoning, type_logits_raw = model(frames)
            # WITH reasoning supervision - guide discriminative features!
            loss, loss_dict = criterion(
                boundary_logits, type_logits, labels, transition_lengths,
                reasoning_outputs=reasoning,
                type_logits_raw=type_logits_raw
            )
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Track losses
        total_loss += loss.item()
        total_focal_loss += loss_dict['focal']
        total_type_loss += loss_dict['type']
        total_reasoning_loss += loss_dict.get('reasoning', 0.0)
        total_consistency_loss += loss_dict.get('consistency', 0.0)
        
        # Track reasoning scores (motion dynamics - monitor natural learning)
        with torch.no_grad():
            total_acceleration += reasoning['acceleration'].abs().mean().item()
            total_jerk += reasoning['jerk'].abs().mean().item()
        
        num_batches += 1
        
        # Compute frame-level binary metrics for progress bar
        with torch.no_grad():
            boundary_probs = torch.sigmoid(boundary_logits).squeeze(-1)
            boundary_preds = (boundary_probs > boundary_threshold).long()
            boundary_targets = (labels > 0).long()
            
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
    avg_focal = total_focal_loss / len(train_loader)
    avg_type = total_type_loss / len(train_loader)
    avg_reasoning = total_reasoning_loss / len(train_loader)
    avg_consistency = total_consistency_loss / len(train_loader)
    
    # Final epoch metrics
    final_precision = tp_boundary / (tp_boundary + fp_boundary + 1e-8)
    final_recall = tp_boundary / (tp_boundary + fn_boundary + 1e-8)
    final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-8)
    
    return {
        'loss': avg_loss,
        'focal_loss': avg_focal,
        'type_loss': avg_type,
        'reasoning_loss': avg_reasoning,
        'consistency_loss': avg_consistency,
        'boundary_f1': final_f1,
        'boundary_precision': final_precision,
        'boundary_recall': final_recall,
        # Reasoning statistics (motion dynamics - track learning)
        'reasoning/acceleration': total_acceleration / len(train_loader),
        'reasoning/jerk': total_jerk / len(train_loader)
    }


@torch.no_grad()
@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    """Validate model - REDESIGNED without boosting diagnostics"""
    model.eval()
    
    total_loss = 0
    total_focal_loss = 0
    total_type_loss = 0
    total_reasoning_loss = 0
    total_consistency_loss = 0
    
    # Reasoning tracking (motion dynamics - monitor discrimination)
    reasoning_stats = {
        'cut': {
            'acceleration': [], 'jerk': [],
            'gradual_prob': [], 'count': 0
        },
        'gradual': {
            'acceleration': [], 'jerk': [],
            'gradual_prob': [], 'count': 0
        },
        'none': {
            'acceleration': [], 'jerk': [],
            'gradual_prob': [], 'count': 0
        }
    }
    
    # Confusion tracking
    gradual_confusion = {
        'predicted_as_gradual': 0, 'predicted_as_cut': 0,
        'predicted_as_none': 0, 'total': 0
    }
    
    gradual_fp_sources = {
        'from_cut': 0, 'from_none': 0, 'total': 0
    }
    
    # Aggregate metrics
    cut_tp, cut_fp, cut_fn = 0, 0, 0
    gradual_tp, gradual_fp, gradual_fn = 0, 0, 0
    
    boundary_threshold = config['boundary_threshold']
    
    for batch_data in tqdm(val_loader, desc="Validating"):
        frames, labels, transition_lengths = batch_data
        
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        transition_lengths = transition_lengths.to(device, non_blocking=True)
        
        # Forward pass
        boundary_logits, type_logits, reasoning, type_logits_raw = model(frames)
        
        # Compute loss (WITH reasoning supervision)
        loss, loss_dict = criterion(
            boundary_logits, type_logits, labels, transition_lengths,
            reasoning_outputs=reasoning,
            type_logits_raw=type_logits_raw
        )
        
        total_loss += loss.item()
        total_focal_loss += loss_dict['focal']
        total_type_loss += loss_dict['type']
        
        total_reasoning_loss += loss_dict.get('reasoning', 0.0)
        total_consistency_loss += loss_dict.get('consistency', 0.0)
        # Extract predictions
        boundary_probs = torch.sigmoid(boundary_logits.squeeze(-1))
        type_preds = torch.argmax(type_logits, dim=-1)
        is_boundary = boundary_probs > boundary_threshold
        
        # Collect reasoning statistics (motion dynamics - monitor discrimination)
        with torch.no_grad():
            acceleration = reasoning['acceleration'].abs().squeeze(-1)
            jerk = reasoning['jerk'].abs().squeeze(-1)
            type_probs = torch.softmax(type_logits, dim=-1)
            gradual_prob = type_probs[:, :, 2]
            
            # Flatten
            labels_flat = labels.reshape(-1).cpu().numpy()
            acceleration_flat = acceleration.reshape(-1).cpu().numpy()
            jerk_flat = jerk.reshape(-1).cpu().numpy()
            gradual_prob_flat = gradual_prob.reshape(-1).cpu().numpy()
            
            # Collect by type
            for stats_key, label_val in [('none', 0), ('cut', 1), ('gradual', 2)]:
                mask = (labels_flat == label_val)
                reasoning_stats[stats_key]['acceleration'].extend(acceleration_flat[mask].tolist())
                reasoning_stats[stats_key]['jerk'].extend(jerk_flat[mask].tolist())
                reasoning_stats[stats_key]['gradual_prob'].extend(gradual_prob_flat[mask].tolist())
                reasoning_stats[stats_key]['count'] += mask.sum()
            
            # Confusion analysis
            boundary_pred_flat = is_boundary.reshape(-1).cpu().numpy()
            type_pred_flat = type_preds.reshape(-1).cpu().numpy()
            
            # Gradual GT confusion
            gradual_mask = (labels_flat == 2)
            for idx in np.where(gradual_mask)[0]:
                if boundary_pred_flat[idx]:
                    if type_pred_flat[idx] == 2:
                        gradual_confusion['predicted_as_gradual'] += 1
                    else:
                        gradual_confusion['predicted_as_cut'] += 1
                else:
                    gradual_confusion['predicted_as_none'] += 1
                gradual_confusion['total'] += 1
            
            # Gradual FP sources
            for idx in np.where((boundary_pred_flat == True) & (type_pred_flat == 2))[0]:
                if labels_flat[idx] == 1:
                    gradual_fp_sources['from_cut'] += 1
                    gradual_fp_sources['total'] += 1
                elif labels_flat[idx] == 0:
                    gradual_fp_sources['from_none'] += 1
                    gradual_fp_sources['total'] += 1
        
        # Get boundary predictions with type gating
        preds_3class = torch.where(is_boundary, type_preds, torch.zeros_like(type_preds))
        
        # Process each sample
        for b in range(frames.shape[0]):
            pred_array = preds_3class[b].cpu()
            gt_array = labels[b].cpu()
            
            pred_trans = extract_transitions_from_predictions(pred_array, min_gap=3)
            gt_trans = extract_transitions_from_predictions(gt_array, min_gap=3)
            
            # Cuts
            sample_metrics = compute_transition_metrics(pred_trans['cuts'], gt_trans['cuts'], tolerance=3)
            cut_tp += sample_metrics.get('tp', 0)
            cut_fp += sample_metrics.get('fp', 0)
            cut_fn += sample_metrics.get('fn', 0)
            
            # Graduals
            sample_metrics = compute_transition_metrics(pred_trans['graduals'], gt_trans['graduals'], tolerance=3)
            gradual_tp += sample_metrics.get('tp', 0)
            gradual_fp += sample_metrics.get('fp', 0)
            gradual_fn += sample_metrics.get('fn', 0)
    
    # Calculate metrics
    cut_precision = cut_tp / (cut_tp + cut_fp + 1e-8)
    cut_recall = cut_tp / (cut_tp + cut_fn + 1e-8)
    cut_f1 = 2 * cut_precision * cut_recall / (cut_precision + cut_recall + 1e-8)
    
    gradual_precision = gradual_tp / (gradual_tp + gradual_fp + 1e-8)
    gradual_recall = gradual_tp / (gradual_tp + gradual_fn + 1e-8)
    gradual_f1 = 2 * gradual_precision * gradual_recall / (gradual_precision + gradual_recall + 1e-8)
    
    overall_tp = cut_tp + gradual_tp
    overall_fp = cut_fp + gradual_fp
    overall_fn = cut_fn + gradual_fn
    overall_precision = overall_tp / (overall_tp + overall_fp + 1e-8)
    overall_recall = overall_tp / (overall_tp + overall_fn + 1e-8)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-8)
    
    # Reasoning summary (motion dynamics - monitor discrimination)
    def compute_reasoning_stats(scores_dict):
        return {
            'acceleration_mean': float(np.mean(scores_dict['acceleration'])) if scores_dict['acceleration'] else 0.0,
            'jerk_mean': float(np.mean(scores_dict['jerk'])) if scores_dict['jerk'] else 0.0,
            'gradual_prob_mean': float(np.mean(scores_dict['gradual_prob'])) if scores_dict['gradual_prob'] else 0.0,
            'count': scores_dict['count']
        }
    
    reasoning_summary = {
        'cut': compute_reasoning_stats(reasoning_stats['cut']),
        'gradual': compute_reasoning_stats(reasoning_stats['gradual']),
        'none': compute_reasoning_stats(reasoning_stats['none'])
    }
    
    return {
        'cut': {'tp': cut_tp, 'fp': cut_fp, 'fn': cut_fn,
                'precision': cut_precision, 'recall': cut_recall, 'f1': cut_f1},
        'gradual': {'tp': gradual_tp, 'fp': gradual_fp, 'fn': gradual_fn,
                    'precision': gradual_precision, 'recall': gradual_recall, 'f1': gradual_f1},
        'overall': {'precision': overall_precision, 'recall': overall_recall, 'f1': overall_f1},
        'loss': total_loss / len(val_loader),
        'focal_loss': total_focal_loss / len(val_loader),
        'type_loss': total_type_loss / len(val_loader),
        'reasoning_loss': total_reasoning_loss / len(val_loader),
        'consistency_loss': total_consistency_loss / len(val_loader),
        'reasoning': reasoning_summary,
        'gradual_confusion': gradual_confusion,
        'gradual_fp_sources': gradual_fp_sources
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
    
    # 2. Focal loss
    ax2 = plt.subplot(3, 3, 2)
    train_focal = [m['focal_loss'] for m in history['train']]
    val_focal = [m['focal_loss'] for m in history['val']]
    ax2.plot(epochs, train_focal, 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, val_focal, 'r-', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Focal Loss')
    ax2.set_title('Focal Loss (Boundary Detection)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Type loss
    ax3 = plt.subplot(3, 3, 3)
    train_type = [m['type_loss'] for m in history['train']]
    val_type = [m['type_loss'] for m in history['val']]
    ax3.plot(epochs, train_type, 'b-', label='Train', linewidth=2)
    ax3.plot(epochs, val_type, 'r-', label='Val', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Type Loss')
    ax3.set_title('Type Loss (Cut vs Gradual)')
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
    best_val_f1 = val_metrics.get('overall', {}).get('f1', 0.0)
    
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
        'main_dim': 384,  # Main processing dimension (fusion + detection)
                          # Component dims are FIXED: spatial=256, hist=128, similarity=128
                          # 384 gives ~10-12M params (sweet spot between TransNetV2 7M and AutoShot 15M)
        'num_fusion_blocks': 4,  # Deeper fusion (TransNetV2 uses 3 DDCNN layers + fc)
        
        # Resume training
        'resume': args.resume,  # Path to checkpoint to resume from
        
        # Pre-decoded dataset path
        'predecoded_cache': r"D:\grad\data\training_cache",
        'use_predecoded': True,
        
        # Training
        'batch_size': 32,
        'num_epochs': 80,
        'learning_rate': 1e-4,  # Back to original stable value
        'weight_decay': 0.01,
        'patience': 15,
        
        # Loss (REDESIGNED - back to stable baseline)
        # Reasoning supervision DISABLED temporarily - margin constraints cause instability
        # Model still computes reasoning features and uses them internally
        # 1. Removed coherence/length losses (counterproductive)
        # 2. Rebalanced type weights to prevent none class suppression
        'focal_alpha': 0.85,  # Increased to help recall (0.80 too conservative)
        'focal_gamma': 2.0,   # Standard focal loss focusing
        'type_weight': 0.15,  # Reduced to let boundary head lead
        'type_class_weights': [0.05, 10.0, 5.0],  # FIXED: balance 67.5:1:1.8 distribution
        'reasoning_weight': 0.00,  # DISABLED: margin constraints too strict
        'consistency_weight': 0.02,  # VERY LOW: avoid suppressing boundary head during early training
        'boundary_threshold': 0.55,  # LOWERED to help detect subtle graduals
        
        # Data loading (OPTIMIZED - fix bottleneck!)
        'val_split_ratio': 0.15,
        'num_workers': 6,  # INCREASED 4→12: saturate CPU while GPU computes
        'prefetch_factor': 4,  # INCREASED 2→4: pre-load more batches
        
        # Paths
        'checkpoint_dir': 'outputs/hybrid_detector3',
        'wandb_project': 'hybrid-boundary-detector3'
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
            random_seed=config.get('random_seed', 42)
        )
        val_dataset = PredecodedDataset(
            cache_path, 
            split='val',
            val_split_ratio=config.get('val_split_ratio', 0.15),
            random_seed=config.get('random_seed', 42)
        )
        
        print(f"✓ Using pre-decoded dataset (10-20x faster!)")
        print(f"✓ Train: {len(train_dataset)} clips, Val: {len(val_dataset)} clips")
    else:
        # Fallback to original ClipShotsDataset (slower)
        print("Using on-the-fly video decoding (slower)...")
        from train_turbo_sbd_v2 import ClipShotsDataset
        
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
    
    # Create loss function (REDESIGNED)
    criterion = create_binary_boundary_loss({
        'focal_alpha': config['focal_alpha'],
        'focal_gamma': config['focal_gamma'],
        'type_weight': config['type_weight'],
        'type_class_weights': config['type_class_weights'],
        'reasoning_weight': config['reasoning_weight'],
        'consistency_weight': config['consistency_weight']
    }).to(device)
    
    print(f"\nLoss configuration (REDESIGNED):")
    print(f"  Focal gamma:      {config['focal_gamma']:.2f}")
    print(f"  Focal alpha:      {config['focal_alpha']:.2f}")
    print(f"  Type weight:      {config['type_weight']:.2f}")
    print(f"  Type class weights: {config['type_class_weights']} [none, cut, gradual]")
    print(f"  Reasoning weight: {config['reasoning_weight']:.2f} (discriminative targets)")
    print(f"  Consistency weight: {config['consistency_weight']:.2f} (boundary-type alignment)")
    print(f"  Boundary threshold: {config['boundary_threshold']:.2f}")
    print(f"  REMOVED: coherence_loss, length_loss (counterproductive)")
    print(f"  REMOVED: boosting logic (train/eval mismatch)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
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
            'focal_loss': train_metrics['focal_loss'],
            'type_loss': train_metrics['type_loss'],
            'boundary_f1': train_metrics['boundary_f1'],
            'boundary_precision': train_metrics['boundary_precision'],
            'boundary_recall': train_metrics['boundary_recall'],
            'acceleration': train_metrics['reasoning/acceleration'],
            'jerk': train_metrics['reasoning/jerk'],
            'lr': current_lr
        })
        
        val_record = make_serializable({
            'epoch': epoch,
            'loss': val_metrics['loss'],
            'focal_loss': val_metrics['focal_loss'],
            'type_loss': val_metrics['type_loss'],
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
            'overall_recall': val_metrics['overall']['recall']
        })
        
        history['train'].append(train_record)
        history['val'].append(val_record)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'lr': current_lr,
            'train/loss': train_metrics['loss'],
            'train/focal_loss': train_metrics['focal_loss'],
            'train/type_loss': train_metrics['type_loss'],
            'train/reasoning_loss': train_metrics['reasoning_loss'],
            'train/consistency_loss': train_metrics['consistency_loss'],
            # Reasoning monitoring (motion dynamics - track discrimination)
            'train/reasoning_acceleration': train_metrics['reasoning/acceleration'],
            'train/reasoning_jerk': train_metrics['reasoning/jerk'],
            'val/loss': val_metrics['loss'],
            'val/focal_loss': val_metrics['focal_loss'],
            'val/type_loss': val_metrics['type_loss'],
            'val/reasoning_loss': val_metrics['reasoning_loss'],
            'val/consistency_loss': val_metrics['consistency_loss'],
            'val/cut_f1': val_metrics['cut']['f1'],
            'val/cut_precision': val_metrics['cut']['precision'],
            'val/cut_recall': val_metrics['cut']['recall'],
            'val/gradual_f1': val_metrics['gradual']['f1'],
            'val/gradual_precision': val_metrics['gradual']['precision'],
            'val/gradual_recall': val_metrics['gradual']['recall'],
            'val/overall_f1': val_metrics['overall']['f1'],
            'val/overall_precision': val_metrics['overall']['precision'],
            'val/overall_recall': val_metrics['overall']['recall'],
            # Reasoning diagnostics (motion dynamics by type)
            'reasoning/cut_acceleration': val_metrics['reasoning']['cut']['acceleration_mean'],
            'reasoning/cut_jerk': val_metrics['reasoning']['cut']['jerk_mean'],
            'reasoning/gradual_acceleration': val_metrics['reasoning']['gradual']['acceleration_mean'],
            'reasoning/gradual_jerk': val_metrics['reasoning']['gradual']['jerk_mean'],
            'reasoning/none_acceleration': val_metrics['reasoning']['none']['acceleration_mean'],
            'reasoning/none_jerk': val_metrics['reasoning']['none']['jerk_mean']
        })
        
        # Print results
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Focal: {train_metrics['focal_loss']:.4f}, "
              f"Type: {train_metrics['type_loss']:.4f}, "
              f"Reason: {train_metrics['reasoning_loss']:.4f}, "
              f"Consist: {train_metrics['consistency_loss']:.4f}, "
              f"F1: {train_metrics['boundary_f1']:.3f}, "
              f"P: {train_metrics['boundary_precision']:.3f}, "
              f"R: {train_metrics['boundary_recall']:.3f}")
        
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Focal: {val_metrics['focal_loss']:.4f}, "
              f"Type: {val_metrics['type_loss']:.4f}, "
              f"Reason: {val_metrics['reasoning_loss']:.4f}, "
              f"Consist: {val_metrics['consistency_loss']:.4f}")
        
        print(f"\nTransition Metrics (±3 frame tolerance):")
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
        print(f"  Overall  - P: {val_metrics['overall']['precision']:.3f}, "
              f"R: {val_metrics['overall']['recall']:.3f}, "
              f"F1: {val_metrics['overall']['f1']:.3f}")
        
        # Gradual confusion analysis
        if 'gradual_confusion' in val_metrics:
            print(f"\nGradual GT Confusion (where do gradual GTs go?):")
            total = val_metrics['gradual_confusion']['total']
            if total > 0:
                grad_tp = val_metrics['gradual_confusion']['predicted_as_gradual']
                as_cut = val_metrics['gradual_confusion']['predicted_as_cut']
                as_none = val_metrics['gradual_confusion']['predicted_as_none']
                print(f"  ✓ Correctly predicted as gradual: {grad_tp}/{total} ({100*grad_tp/total:.1f}%)")
                print(f"  ✗ Type error (predicted as cut):  {as_cut}/{total} ({100*as_cut/total:.1f}%)")
                print(f"  ✗ Missed boundary (as none):      {as_none}/{total} ({100*as_none/total:.1f}%)")
        
        # FP source analysis
        if 'gradual_fp_sources' in val_metrics:
            fp_total = val_metrics['gradual_fp_sources']['total']
            if fp_total > 0:
                print(f"\nGradual False Positives (what gets wrongly predicted as gradual?):")
                from_cut = val_metrics['gradual_fp_sources']['from_cut']
                from_none = val_metrics['gradual_fp_sources']['from_none']
                print(f"  From cuts:  {from_cut}/{fp_total} ({100*from_cut/fp_total:.1f}%)")
                print(f"  From none:  {from_none}/{fp_total} ({100*from_none/fp_total:.1f}%)")
        
        print(f"\nReasoning Diagnostics (Motion Dynamics - TOP discriminators):")
        print(f"  Cut frames ({val_metrics['reasoning']['cut']['count']}):")
        print(f"    Acceleration: {val_metrics['reasoning']['cut']['acceleration_mean']:.3f} (expect HIGH spike)")
        print(f"    Jerk: {val_metrics['reasoning']['cut']['jerk_mean']:.3f} (expect HIGH spike)")
        print(f"    Gradual prob: {val_metrics['reasoning']['cut']['gradual_prob_mean']:.3f}")
        
        print(f"  Gradual frames ({val_metrics['reasoning']['gradual']['count']}):")
        print(f"    Acceleration: {val_metrics['reasoning']['gradual']['acceleration_mean']:.3f} (expect MODERATE - smooth change)")
        print(f"    Jerk: {val_metrics['reasoning']['gradual']['jerk_mean']:.3f} (expect LOW - smooth acceleration)")
        print(f"    Gradual prob: {val_metrics['reasoning']['gradual']['gradual_prob_mean']:.3f} (should be HIGH!)")
        
        print(f"  None frames ({val_metrics['reasoning']['none']['count']}):")
        print(f"    Acceleration: {val_metrics['reasoning']['none']['acceleration_mean']:.3f} (expect LOW/ZERO)")
        print(f"    Jerk: {val_metrics['reasoning']['none']['jerk_mean']:.3f} (expect LOW/ZERO)")
        print(f"    Gradual prob: {val_metrics['reasoning']['none']['gradual_prob_mean']:.3f} (should be LOW!)")
        
        # Check if motion dynamics are discriminative
        cut_accel = val_metrics['reasoning']['cut']['acceleration_mean']
        gradual_accel = val_metrics['reasoning']['gradual']['acceleration_mean']
        none_accel = val_metrics['reasoning']['none']['acceleration_mean']
        
        cut_jerk = val_metrics['reasoning']['cut']['jerk_mean']
        gradual_jerk = val_metrics['reasoning']['gradual']['jerk_mean']
        none_jerk = val_metrics['reasoning']['none']['jerk_mean']
        
        print(f"\n🎯 MOTION DISCRIMINATIVE POWER:")
        print(f"  Acceleration separates cut/gradual: {abs(cut_accel - gradual_accel):.3f}")
        print(f"  Acceleration separates gradual/none: {abs(gradual_accel - none_accel):.3f}")
        print(f"  Jerk separates cut/gradual: {abs(cut_jerk - gradual_jerk):.3f}")
        
        # CRITICAL DIAGNOSTIC
        print(f"\n🔍 TYPE HEAD ANALYSIS:")
        gradual_prob_none = val_metrics['reasoning']['none']['gradual_prob_mean']
        gradual_prob_gradual = val_metrics['reasoning']['gradual']['gradual_prob_mean']
        print(f"  None frames gradual prob: {gradual_prob_none:.1%}")
        print(f"  Gradual frames gradual prob: {gradual_prob_gradual:.1%}")
        if gradual_prob_none > 0.3:
            print(f"    ⚠ Type head still confused! Need more training.")
        else:
                print(f"    ✓ Type head learning to separate gradual/none")
        
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
        
        # Save best model
        val_f1 = val_metrics['overall']['f1']
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
            'main_dim': config['main_dim'],
            'num_fusion_blocks': config['num_fusion_blocks'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'num_epochs': config['num_epochs'],
            'focal_alpha': config['focal_alpha'],
            'focal_gamma': config['focal_gamma'],
            'type_weight': config['type_weight'],
            'boundary_threshold': config['boundary_threshold']
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
