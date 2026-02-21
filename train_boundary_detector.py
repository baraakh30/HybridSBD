"""
Training Script for TurboSBD Binary Boundary Detector (2026 SOTA)

BINARY CLASSIFICATION APPROACH:
- Primary: Is this frame a boundary? (yes/no)
- Auxiliary: If boundary, what type? (cut/gradual)

This approach:
✓ Matches how SOTA models work (TransNetV2, AutoShot)
✓ Better class balance (1% boundaries vs 99% non-boundaries)
✓ Reduces false positives (no cut/gradual confusion)
✓ Validates on same metrics as test (transition-level F1)

Key improvements over 3-class frame classification:
1. Binary formulation → simpler, better balanced
2. Auxiliary type task → teaches cut vs gradual distinction
3. Modern 2026 architecture → RoPE, SwiGLU, better attention
4. Proper validation → transition-level metrics, not frame-level
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import gc
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Import existing dataset and utilities
from train_turbo_sbd_v2 import (
    set_seed, 
    worker_init_fn,
    compute_transition_metrics,
    extract_transitions_from_predictions,
    CONFIG as BASE_CONFIG
)

# Import new binary architecture and loss
from core.binary_boundary_loss import create_binary_boundary_loss


class PredecodedDataset(Dataset):
    """
    Fast dataset loader for pre-decoded clips
    
    Loads directly from .pt files created by predecode_dataset.py
    10-20x faster than decoding videos on the fly
    
    Supports:
    - train/val splits from train directory
    - test split from test directory
    - autoshot split from autoshot_cache directory
    """
    def __init__(self, cache_root, split='train', val_split_ratio=0.15, random_seed=42, dataset_type='clipshots', temporal_jitter=False):
        """
        Args:
            cache_root: Path to training_cache directory
            split: 'train', 'val', 'test', or 'autoshot'
            val_split_ratio: Fraction of videos for validation (default 0.15)
            random_seed: Seed for reproducible split (default 42)
            dataset_type: 'clipshots' or 'autoshot' (determines which cache to use)
        """
        self.cache_root = Path(cache_root)
        self.split = split
        self.dataset_type = dataset_type
        self.temporal_jitter = temporal_jitter and (split == 'train')
        
        if split == 'autoshot':
            # Load from autoshot_cache directory (separate dataset)
            mapping_file = self.cache_root.parent / 'autoshot_cache' / 'index_mapping.json'
            if not mapping_file.exists():
                raise FileNotFoundError(
                    f"AutoShot index mapping not found: {mapping_file}\n"
                    f"Please run: python predecode_autoshot.py --num_workers 8"
                )
            
            print(f"Loading pre-decoded AutoShot dataset from {mapping_file}...")
            with open(mapping_file, 'r') as f:
                data = json.load(f)
            
            # Use all AutoShot clips without splitting
            self.mapping = data['mapping']
            self.num_videos = len(set(clip['video_name'] for clip in self.mapping))
            self.num_clips = len(self.mapping)
            
            print(f"✓ AUTOSHOT: {self.num_clips} clips from {self.num_videos} videos")
        
        elif split == 'test':
            # Load from test directory (separate subdirectory)
            mapping_file = self.cache_root / 'test' / 'index_mapping.json'
            if not mapping_file.exists():
                raise FileNotFoundError(
                    f"Test index mapping not found: {mapping_file}\n"
                    f"Please run: python predecode_dataset.py --split test --num_workers 8"
                )
            
            print(f"Loading pre-decoded test dataset from {mapping_file}...")
            with open(mapping_file, 'r') as f:
                data = json.load(f)
            
            # Use all test clips without splitting
            self.mapping = data['mapping']
            self.num_videos = len(set(clip['video_name'] for clip in self.mapping))
            self.num_clips = len(self.mapping)
            
            print(f"✓ TEST split: {self.num_clips} clips from {self.num_videos} videos")
        
        else:
            # Load from train directory and split into train/val
            # OR load from standalone cache (like autoshot_cache_train)
            mapping_file = self.cache_root / 'train' / 'index_mapping.json'
            
            # Check if this is a standalone cache (no train/ subdirectory)
            if not mapping_file.exists():
                standalone_mapping = self.cache_root / 'index_mapping.json'
                if standalone_mapping.exists():
                    # Standalone cache (e.g., autoshot_cache_train)
                    mapping_file = standalone_mapping
                    print(f"Loading standalone pre-decoded dataset from {mapping_file}...")
                else:
                    raise FileNotFoundError(
                        f"Index mapping not found in:\n"
                        f"  {mapping_file}\n"
                        f"  {standalone_mapping}\n"
                        f"Please run: python predecode_dataset.py --split train --num_workers 8\n"
                        f"       or: python predecode_autoshot_train.py --num_workers 8"
                    )
            else:
                print(f"Loading pre-decoded dataset from {mapping_file}...")
            
            with open(mapping_file, 'r') as f:
                data = json.load(f)
            
            all_mapping = data['mapping']
            
            # Get unique videos and create train/val split (MUST match training script logic!)
            all_videos = sorted(set(clip['video_name'] for clip in all_mapping))
            num_val = max(1, int(len(all_videos) * val_split_ratio))
            
            rng = random.Random(random_seed)
            rng.shuffle(all_videos)
            val_videos = set(all_videos[:num_val])
            train_videos = set(all_videos[num_val:])
            
            # Filter clips based on split
            if split == 'val':
                self.mapping = [clip for clip in all_mapping if clip['video_name'] in val_videos]
                self.num_videos = len(val_videos)
            else:  # train
                self.mapping = [clip for clip in all_mapping if clip['video_name'] in train_videos]
                self.num_videos = len(train_videos)
            
            self.num_clips = len(self.mapping)
            
            print(f"✓ {split.upper()} split: {self.num_clips} clips from {self.num_videos} videos")
            print(f"  (Total videos: {len(all_videos)}, Train: {len(train_videos)}, Val: {len(val_videos)})")
    
    def __len__(self):
        return len(self.mapping)
    
    def __getitem__(self, idx):
        """
        Load a pre-decoded clip with optical flow features
        
        Returns:
            frames: [T, 3, H, W] float tensor (normalized to [0, 1])
            targets: [T] long tensor
            transition_lengths: [T] long tensor
            flow_features: [T-1, 4] float tensor (mean_mag, max_mag, std_mag, uniformity)
        """
        clip_info = self.mapping[idx]
        clip_file = clip_info['clip_file']
        
        # Load pre-decoded clip (very fast!)
        data = torch.load(clip_file, weights_only=False)
        
        # Convert frames from uint8 to float32 and normalize to [0, 1]
        frames = data['frames'].float() / 255.0  # [T, 3, H, W]
        targets = data['targets']     # [T] long
        lengths = data['lengths']     # [T] long

        if self.temporal_jitter:
            # Time-reversal: flip the clip in time with 50% probability.
            # A cut at frame t becomes a cut at frame T-1-t.
            # This directly scrambles the position-specific FrameSimilarity and
            # ColorHistSim matrices the model would otherwise memorize.
            if torch.rand(1).item() < 0.5:
                frames  = frames.flip(0)
                targets = targets.flip(0)
                lengths = lengths.flip(0)

        # All-zero frames are VALID (black frames are legitimate video content)
        return (
            frames,   # [T, 3, H, W] float32
            targets,  # [T] long
            lengths,  # [T] long
        )

# Configuration
CONFIG = {
    **BASE_CONFIG,  # Inherit all base config
    
    # Pre-decoded dataset path
    'predecoded_cache': r"D:\grad\data\training_cache",  # Path to pre-decoded clips
    'use_predecoded': True,  # Use pre-decoded dataset for speed
    
    # Training hyperparameters
    'epochs': 60,
    'base_lr': 2e-4,
    'weight_decay': 1e-3,
    'patience': 12,
    
    # Loss weights (binary classification)
    'focal_gamma': 2.0,          # Focal loss gamma (hard example focus)
    'focal_alpha': 0.90,         # Focal loss alpha (need VERY high for 4.2% minority class)
    'type_weight': 0.05,         # Type head weight (reduce - was dominating at 0.1)
    'coherence_weight': 0.0,     # Coherence disabled (let model learn naturally)
    
    # Evaluation
    'eval_interval': 1,  # Validate every epoch
    'use_transition_metrics': True,  # CRITICAL: Use same metrics as test
    
    # Model checkpoint to load from
    'pretrained_checkpoint': 'outputs/stage2_best_final.pth',
}


def validate_transition_level(model, loader, device, tolerance=3, boundary_threshold=0.7):
    """
    Validation using TRANSITION-LEVEL metrics (same as test evaluation)
    
    Binary classification approach:
    - Detect boundaries using boundary_logits
    - Classify type using type_logits (for detected boundaries)
    - Extract transitions and compute transition-level F1
    
    Returns transition-level P/R/F1 for cuts and graduals
    """
    model.eval()
    
    # Frame-level stats for quick metrics
    tp_boundary = 0
    fp_boundary = 0
    fn_boundary = 0
    
    # Transition-level: collect all predictions and GT
    all_pred_cuts = []
    all_pred_graduals = []
    all_gt_cuts = []
    all_gt_graduals = []
    
    total_loss = 0
    num_batches = 0
    
    # Use binary boundary loss
    criterion = create_binary_boundary_loss({
        'focal_gamma': CONFIG['focal_gamma'],
        'focal_alpha': CONFIG['focal_alpha'],
        'type_weight': CONFIG['type_weight'],
        'coherence_weight': CONFIG['coherence_weight']
    }).to(device)
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Validating", leave=False):
            frames, targets, transition_lengths = batch_data
            
            frames = frames.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            transition_lengths = transition_lengths.to(device, non_blocking=True)
            
            # Forward pass - binary classification
            boundary_logits, type_logits = model(frames)
            
            # Loss
            loss, loss_components = criterion(boundary_logits, type_logits, targets, transition_lengths)
            total_loss += loss.item()
            num_batches += 1
            
            # Convert to predictions
            # Binary boundary detection
            boundary_probs = torch.sigmoid(boundary_logits).squeeze(-1)  # [B, T]
            boundary_preds = (boundary_probs > boundary_threshold).long()
            
            # Type classification (for boundaries)
            type_preds = type_logits.argmax(dim=-1)  # [B, T] - 0=cut, 1=gradual
            
            # Combine: create 3-class predictions for compatibility
            # 0=none, 1=cut, 2=gradual
            preds = torch.where(boundary_preds == 1,
                               type_preds + 1,  # 0->1 (cut), 1->2 (gradual)
                               torch.zeros_like(boundary_preds))  # 0 (none)
            
            # Frame-level metrics (for quick feedback)
            boundary_targets = (targets > 0).long()
            tp_boundary += (boundary_preds & boundary_targets.bool()).sum().item()
            fp_boundary += (boundary_preds & ~boundary_targets.bool()).sum().item()
            fn_boundary += (~boundary_preds.bool() & boundary_targets.bool()).sum().item()
            
            # Extract transitions for each sample
            batch_size = preds.shape[0]
            for b in range(batch_size):
                pred_array = preds[b].cpu()
                gt_array = targets[b].cpu()
                
                # Extract transitions
                pred_trans = extract_transitions_from_predictions(pred_array, min_gap=3)
                gt_trans = extract_transitions_from_predictions(gt_array, min_gap=3)
                
                all_pred_cuts.extend(pred_trans['cuts'])
                all_pred_graduals.extend(pred_trans['graduals'])
                all_gt_cuts.extend(gt_trans['cuts'])
                all_gt_graduals.extend(gt_trans['graduals'])
    
    # Compute frame-level F1 (quick indicator for boundary detection)
    boundary_p = tp_boundary / (tp_boundary + fp_boundary + 1e-8)
    boundary_r = tp_boundary / (tp_boundary + fn_boundary + 1e-8)
    boundary_f1 = 2 * boundary_p * boundary_r / (boundary_p + boundary_r + 1e-8)
    
    # Compute transition-level metrics (PRIMARY METRIC - same as test)
    cut_trans_metrics = compute_transition_metrics(all_pred_cuts, all_gt_cuts, tolerance)
    grad_trans_metrics = compute_transition_metrics(all_pred_graduals, all_gt_graduals, tolerance)
    trans_mean_f1 = (cut_trans_metrics['f1'] + grad_trans_metrics['f1']) / 2
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # Print detailed results
    print(f"\n{'='*80}")
    print(f"Validation Results (Loss: {avg_loss:.4f})")
    print(f"{'='*80}")
    print(f"\nFrame-Level Binary Detection (quick indicator):")
    print(f"  Boundary - P: {boundary_p:.4f}, R: {boundary_r:.4f}, F1: {boundary_f1:.4f}")
    print(f"             TP={tp_boundary}, FP={fp_boundary}, FN={fn_boundary}")
    
    print(f"\nTransition-Level Metrics (PRIMARY - same as test):")
    print(f"  Cut      - P: {cut_trans_metrics['precision']:.4f}, R: {cut_trans_metrics['recall']:.4f}, F1: {cut_trans_metrics['f1']:.4f}")
    print(f"             TP={cut_trans_metrics['tp']}, FP={cut_trans_metrics['fp']}, FN={cut_trans_metrics['fn']}")
    print(f"  Gradual  - P: {grad_trans_metrics['precision']:.4f}, R: {grad_trans_metrics['recall']:.4f}, F1: {grad_trans_metrics['f1']:.4f}")
    print(f"             TP={grad_trans_metrics['tp']}, FP={grad_trans_metrics['fp']}, FN={grad_trans_metrics['fn']}")
    print(f"  Mean F1  : {trans_mean_f1:.4f} ⭐ (PRIMARY METRIC)")
    print(f"\nPrediction Stats:")
    print(f"  Predicted cuts: {len(all_pred_cuts)}, GT: {len(all_gt_cuts)}")
    print(f"  Predicted graduals: {len(all_pred_graduals)}, GT: {len(all_gt_graduals)}")
    print(f"{'='*80}")
    
    return {
        'loss': avg_loss,
        'boundary_f1': boundary_f1,
        'boundary_precision': boundary_p,
        'boundary_recall': boundary_r,
        'trans_mean_f1': trans_mean_f1,  # PRIMARY METRIC
        'cut_f1': cut_trans_metrics['f1'],
        'cut_precision': cut_trans_metrics['precision'],
        'cut_recall': cut_trans_metrics['recall'],
        'gradual_f1': grad_trans_metrics['f1'],
        'gradual_precision': grad_trans_metrics['precision'],
        'gradual_recall': grad_trans_metrics['recall'],
    }


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, boundary_threshold=0.7):
    """
    Train one epoch with binary boundary detection
    """
    model.train()
    
    # Metrics tracking
    total_loss = 0
    loss_components_sum = {'focal': 0, 'type': 0, 'coherence': 0, 'total': 0}
    
    # Frame-level stats for quick binary F1 calculation
    tp_boundary = 0
    fp_boundary = 0
    fn_boundary = 0
    
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for frames, targets, transition_lengths in pbar:
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        transition_lengths = transition_lengths.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            boundary_logits, type_logits = model(frames)
            loss, components = criterion(boundary_logits, type_logits, targets, transition_lengths)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        total_loss += loss.item()
        for key in loss_components_sum:
            loss_components_sum[key] += components[key]
        num_batches += 1
        
        # Compute frame-level binary metrics for progress bar
        boundary_probs = torch.sigmoid(boundary_logits).squeeze(-1)
        boundary_preds = (boundary_probs > boundary_threshold).long()
        boundary_targets = (targets > 0).long()
        
        tp_boundary += (boundary_preds & boundary_targets.bool()).sum().item()
        fp_boundary += (boundary_preds & ~boundary_targets.bool()).sum().item()
        fn_boundary += (~boundary_preds.bool() & boundary_targets.bool()).sum().item()
        
        # Update progress bar every 5 batches
        if num_batches % 5 == 0:
            # Compute binary F1 scores
            boundary_p = tp_boundary / (tp_boundary + fp_boundary + 1e-8)
            boundary_r = tp_boundary / (tp_boundary + fn_boundary + 1e-8)
            boundary_f1 = 2 * boundary_p * boundary_r / (boundary_p + boundary_r + 1e-8)
            
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'B_F1': f'{boundary_f1:.3f}',
                'B_P': f'{boundary_p:.3f}',
                'B_R': f'{boundary_r:.3f}',
            })
    
    # Final epoch metrics
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components_sum.items()}
    
    # Final binary F1 scores
    boundary_p = tp_boundary / (tp_boundary + fp_boundary + 1e-8)
    boundary_r = tp_boundary / (tp_boundary + fn_boundary + 1e-8)
    boundary_f1 = 2 * boundary_p * boundary_r / (boundary_p + boundary_r + 1e-8)
    
    return {
        'loss': avg_loss,
        'loss_components': avg_components,
        'boundary_f1': boundary_f1,
        'boundary_precision': boundary_p,
        'boundary_recall': boundary_r,
    }


def main():
    print("="*80)
    print("TRAINING: TurboSBD Binary Boundary Detector (2026 SOTA)")
    print("="*80)
    print("Approach: BINARY CLASSIFICATION + Auxiliary Type Head")
    print("  Like: TransNetV2, AutoShot")
    print("\nKey improvements:")
    print("  ✓ Binary classification (is_boundary? yes/no)")
    print("  ✓ Auxiliary type head (cut vs gradual)")
    print("  ✓ Better balanced (1% boundaries vs 99% non-boundaries)")
    print("  ✓ Reduces false positives (no cut/gradual confusion)")
    print("  ✓ Modern 2026 architecture (RoPE, SwiGLU)")
    print("  ✓ Transition-level validation (same as test)")
    print("="*80)
    
    set_seed(CONFIG['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    os.makedirs('outputs', exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    
    if CONFIG['use_predecoded']:
        # Use pre-decoded dataset (FAST!)
        cache_path = CONFIG['predecoded_cache']
        
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
        train_ds = PredecodedDataset(
            cache_path, 
            split='train',
            val_split_ratio=CONFIG.get('val_split_ratio', 0.15),
            random_seed=CONFIG.get('random_seed', 42)
        )
        val_ds = PredecodedDataset(
            cache_path, 
            split='val',
            val_split_ratio=CONFIG.get('val_split_ratio', 0.15),
            random_seed=CONFIG.get('random_seed', 42)
        )
        
        print(f"✓ Using pre-decoded dataset (10-20x faster!)")
        print(f"✓ Train: {len(train_ds)} clips, Val: {len(val_ds)} clips")
    else:
        # Fallback to original ClipShotsDataset (slower)
        print("Using on-the-fly video decoding (slower)...")
        from train_turbo_sbd_v2 import ClipShotsDataset
        
        val_json_path = os.path.join(CONFIG['input_root'], 'annotations', 'val.json')
        
        if os.path.exists(val_json_path):
            train_ds = ClipShotsDataset(
                CONFIG['input_root'], 
                split='train',
                seq_len=CONFIG['seq_len'],
                cache_capacity=CONFIG['cache_capacity'],
                deterministic=False
            )
            val_ds = ClipShotsDataset(
                CONFIG['input_root'],
                split='val',
                seq_len=CONFIG['seq_len'],
                cache_capacity=CONFIG['cache_capacity'],
                deterministic=True,
                samples_per_video=5
            )
        else:
            temp_ds = ClipShotsDataset(
                CONFIG['input_root'],
                split='train',
                seq_len=CONFIG['seq_len'],
                cache_capacity=CONFIG['cache_capacity']
            )
            
            all_videos = sorted(temp_ds.valid_keys)
            n_val = max(1, int(len(all_videos) * CONFIG['val_split_ratio']))
            
            rng = random.Random(CONFIG['random_seed'])
            rng.shuffle(all_videos)
            val_videos = set(all_videos[:n_val])
            
            train_ds = ClipShotsDataset(
                CONFIG['input_root'],
                split='train',
                seq_len=CONFIG['seq_len'],
                cache_capacity=CONFIG['cache_capacity'],
                deterministic=False,
                val_videos=val_videos
            )
            val_ds = ClipShotsDataset(
                CONFIG['input_root'],
                split='train',
                seq_len=CONFIG['seq_len'],
                cache_capacity=CONFIG['cache_capacity'],
                deterministic=True,
                val_videos=val_videos,
                samples_per_video=5
            )
            val_ds.split = 'val'
        
        print(f"✓ Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=CONFIG['prefetch_factor']
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=CONFIG['prefetch_factor']
    )
    
    # Create model and load pretrained weights
    print("\nInitializing model...")
    model =None
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function (binary boundary detection)
    criterion = create_binary_boundary_loss({
        'focal_gamma': CONFIG['focal_gamma'],
        'focal_alpha': CONFIG['focal_alpha'],
        'type_weight': CONFIG['type_weight'],
        'coherence_weight': CONFIG['coherence_weight']
    }).to(device)
    
    print(f"\nLoss configuration:")
    print(f"  Focal gamma:      {CONFIG['focal_gamma']:.2f} (hard example focus)")
    print(f"  Focal alpha:      {CONFIG['focal_alpha']:.2f} (high weight for rare 4.2% boundaries)")
    print(f"  Type weight:      {CONFIG['type_weight']:.2f} (helps cut vs gradual distinction)")
    print(f"  Coherence weight: {CONFIG['coherence_weight']:.2f} (disabled - let model learn naturally)")
    print(f"  Positive bias:    0.0 (neutral initialization)")
    print(f"  ⚠️  Actual imbalance: 22.7:1 (4.2% boundaries)")
    print(f"  Strategy:         Balanced detection with focal loss")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['base_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['epochs'],
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_boundary_f1': [], 'val_boundary_f1': [],
        'val_cut_f1': [], 'val_gradual_f1': [],
        'val_trans_f1': []  # Transition-level F1 (primary)
    }
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{CONFIG['epochs']}")
        print(f"{'='*80}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        
        print(f"\nTraining Summary:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"    Focal (boundary): {train_metrics['loss_components']['focal']:.4f}")
        print(f"    Type (cut/grad):  {train_metrics['loss_components']['type']:.4f}")
        print(f"    Coherence:        {train_metrics['loss_components']['coherence']:.4f}")
        print(f"  Boundary Detection - F1: {train_metrics['boundary_f1']:.4f}, "
              f"P: {train_metrics['boundary_precision']:.4f}, "
              f"R: {train_metrics['boundary_recall']:.4f}")
        
        # Validate with transition-level metrics
        val_metrics = validate_transition_level(model, val_loader, device, CONFIG['tolerance'])
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_boundary_f1'].append(train_metrics['boundary_f1'])
        history['val_boundary_f1'].append(val_metrics['boundary_f1'])
        history['val_cut_f1'].append(val_metrics['cut_f1'])
        history['val_gradual_f1'].append(val_metrics['gradual_f1'])
        history['val_trans_f1'].append(val_metrics['trans_mean_f1'])
        
        # Step scheduler
        scheduler.step()
        
        # Save best model (based on transition-level F1)
        if val_metrics['trans_mean_f1'] > best_f1:
            best_f1 = val_metrics['trans_mean_f1']
            torch.save(model.state_dict(), 'outputs/binary_boundary_detector_best.pth')
            print(f"\n✓ SAVED BEST MODEL! Transition F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{CONFIG['patience']})")
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            break
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Transition-Level F1: {best_f1:.4f}")
    print(f"Model saved: outputs/binary_boundary_detector_best.pth")
    print("\nThis model uses BINARY CLASSIFICATION:")
    print("  - Primary task: Is this frame a boundary? (yes/no)")
    print("  - Auxiliary task: If boundary, what type? (cut/gradual)")
    print("  - Matches SOTA approaches: TransNetV2, AutoShot")
    print("="*80)
    
    # Save training history
    import json
    with open('outputs/binary_boundary_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("Training history saved: outputs/binary_boundary_training_history.json")


if __name__ == '__main__':
    main()
