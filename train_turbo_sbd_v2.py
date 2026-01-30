import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import glob
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from decord import VideoReader, cpu, gpu
    # Set decord to use CPU by default to avoid VRAM leaks in dataloader workers
    # We copy to GPU tensor later
except ImportError:
    print("Decord not found, please install it: pip install decord")
    exit(1)

from detection.turbo_sbd import TurboSBD
from debug_confusion import ConfusionTracker

class LengthAwareFocalLoss(nn.Module):
    """
    Focal Loss with Length-Aware Class Weights
    
    Combines two strategies:
    1. Focal Loss (γ=2): Focus learning on hard examples
    2. Length-Aware Weights: Boost long graduals that are being missed
    
    For graduals:
      - Short (2-5f):   weight = 3.0  (80% recall - already good)
      - Medium (6-15f): weight = 5.0  (84% recall - decent)
      - Long (16+f):    weight = 12.0 (72% recall - NEED BOOST!)
    """
    def __init__(self, gamma=2.0, base_weights=None):
        super().__init__()
        self.gamma = gamma
        self.base_weights = base_weights  # [None, Cut, Gradual_base]
        
    def forward(self, logits, targets, transition_lengths):
        """
        Args:
            logits: [B*T, 3] - model predictions
            targets: [B*T] - ground truth labels
            transition_lengths: [B*T] - length of transition at each frame (0 for none)
        """
        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get probability of true class
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=3).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # [B*T]
        
        # Focal loss modulation: (1-pt)^γ
        # Hard examples (pt low) get high weight
        # Easy examples (pt high) get low weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Length-aware class weights
        class_weights = torch.ones_like(targets).float()
        
        if self.base_weights is not None:
            # Apply base weights for None, Cut, Gradual
            for c in range(3):
                class_weights[targets == c] = self.base_weights[c]
        
        # BOOST long graduals specifically (label 2 = gradual)
        is_gradual = (targets == 2)
        is_short_gradual = is_gradual & (transition_lengths <= 5)
        is_medium_gradual = is_gradual & (transition_lengths > 5) & (transition_lengths <= 15)
        is_long_gradual = is_gradual & (transition_lengths > 15)
        
        # Override gradual weights based on length
        class_weights[is_short_gradual] = 3.0   # Short: already 80% recall
        class_weights[is_medium_gradual] = 5.0  # Medium: 84% recall
        class_weights[is_long_gradual] = 12.0   # Long: 72% recall - BOOST!
        
        # Compute cross-entropy loss
        ce_loss = torch.nn.functional.cross_entropy(
            logits, targets, reduction='none'
        )
        
        # Combine: focal_weight * class_weight * ce_loss
        loss = focal_weight * class_weights * ce_loss
        
        return loss.mean()

# Configuration
CONFIG = {
    'input_root': r"C:\Users\baraa\Documents\ClipShots",
    'frame_size': (48, 64), # H, W (TransNetV2 standard)
    'seq_len': 100,
    'batch_size': 16,
    'learning_rate': 2e-4, # Reduced from 5e-4 for stability with multi-scale fusion
    'epochs': 100,
    'num_workers': 4, # Standard for Windows
    'cache_capacity': 70, # Number of open videos to keep per worker
    'early_stopping_patience': 15, # Stop if no improvement for 15 epochs
    'weight_decay': 1e-3, # Increased regularization
    'focal_gamma': 2.0, # Focal loss gamma (2.0 = strong focus on hard examples)
    'use_focal_loss': True # Enable focal loss + length-aware weights
}

class LRUVideoCache:
    """
    Keep a limited number of VideoReader objects open to avoid OS file handle limits
    and overhead of reopening files constantly.
    """
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        
        # Determine context - safe to use CPU for loading in workers
        vr = VideoReader(path, ctx=cpu(0), width=CONFIG['frame_size'][1], height=CONFIG['frame_size'][0])
        
        self.cache[path] = vr
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return vr

class ClipShotsDataset(Dataset):
    def __init__(self, root_dir, split='train', seq_len=100, frame_size=(48, 64), cache_capacity=50):
        self.root_dir = root_dir
        self.split = split
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.cache_capacity = cache_capacity
        self.video_cache = None # Initialized in worker
        
        # Load Annotations
        json_path = os.path.join(root_dir, 'annotations', f'{split}.json')
        if not os.path.exists(json_path):
             # Try searching recursively if structure is different
             print(f"Warning: {json_path} not found. Trying recursive search...")
             try:
                json_path = glob.glob(os.path.join(root_dir, '**', f'{split}.json'), recursive=True)[0]
             except IndexError:
                # Fallback for filenames like "clipshots_annotations.json" or "train.json" in weird places
                if split == 'train':
                     json_path = os.path.join(root_dir, 'annotations', 'train.json') 
                else:
                     json_path = os.path.join(root_dir, 'annotations', 'test.json')

        print(f"Loading annotations from {json_path}...")
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
            
        # Verify video files existence and build sample list
        # We assume videos are in videos/split OR just flat
        self.video_paths = {}
        # video_dir might be just "videos" or "videos/train"
        video_dir_candidates = [
            os.path.join(root_dir, 'videos', split),
            os.path.join(root_dir, 'videos')
        ]
        
        video_dir = None
        for d in video_dir_candidates:
            if os.path.exists(d):
                video_dir = d
                break
        
        if not video_dir:
             raise ValueError(f"Could not find video directory in {root_dir}")

        # Optimization: Pre-scan directory once
        print(f"Scanning video directory: {video_dir}")
        # Build map of filename -> fullpath
        found_videos = {}
        for root, dirs, files in os.walk(video_dir):
            for f in files:
                if f.endswith('.mp4'):
                    found_videos[f] = os.path.join(root, f)
        
        # Filter annotations to only include found videos
        self.valid_keys = []
        for vid_name, data in self.annotations.items():
            if vid_name in found_videos:
                self.video_paths[vid_name] = found_videos[vid_name]
                self.valid_keys.append(vid_name)
        
        print(f"Found {len(self.valid_keys)} valid videos for {split}")
        
        # Create samples: (video_name, start_frame)
        # We can implement a sliding window or random sampling.
        # For efficiency in this script: Random sampling per epoch effectively cover dataset
        # For deterministic len, we estimate:
        self.samples_per_video = 10 # Increase sampling density
        
    def __len__(self):
        return len(self.valid_keys) * self.samples_per_video

    def _init_worker_cache(self):
        if self.video_cache is None:
            self.video_cache = LRUVideoCache(self.cache_capacity)

    def __getitem__(self, idx):
        self._init_worker_cache()
        
        # Map linear index to video
        vid_idx = idx // self.samples_per_video
        vid_name = self.valid_keys[vid_idx]
        vid_path = self.video_paths[vid_name]
        
        try:
            vr = self.video_cache.get(vid_path)
            total_frames = len(vr)
            
            # Smart Sampling strategies:
            # 1. Focus on transitions (Positive mining)
            # 2. Focus on normal content (Negative mining)
            transitions = self.annotations[vid_name]['transitions']
            
            # 50% chance to center on a transition if available
            target_frame_start = 0
            has_transition = len(transitions) > 0
            
            if has_transition and random.random() < 0.6: # 60% transition focus
                # Pick a random transition
                t_start, t_end = random.choice(transitions)
                # Center the window around this transition
                center_frame = (t_start + t_end) // 2
                target_frame_start = center_frame - (self.seq_len // 2)
                # Add some jitter
                target_frame_start += random.randint(-15, 15)
            else:
                # Random location
                if total_frames > self.seq_len:
                    target_frame_start = random.randint(0, total_frames - self.seq_len)
            
            # Clamping
            target_frame_start = max(0, min(target_frame_start, total_frames - self.seq_len))
            
            # Load frames
            indices = list(range(target_frame_start, target_frame_start + self.seq_len))
            # Handle short videos
            if total_frames < self.seq_len:
                indices = [i % total_frames for i in indices]
                
            frame_tensor = vr.get_batch(indices).asnumpy() # [T, H, W, C]
            frames = torch.from_numpy(frame_tensor).permute(0, 3, 1, 2).float() / 255.0 # [T, 3, H, W]
            
            # Generate Labels
            targets = torch.zeros(self.seq_len, dtype=torch.long)
            transition_lengths = torch.zeros(self.seq_len, dtype=torch.long)  # NEW: Track lengths
            
            # Fill labels based on transitions
            # Offset transition times by target_frame_start
            for start, end in transitions:
                # Map global frame idx to local window idx
                local_start = start - target_frame_start
                local_end = end - target_frame_start
                
                # Check intersection
                if local_end > 0 and local_start < self.seq_len:
                    # Clip to window
                    s = max(0, local_start)
                    e = min(self.seq_len, local_end) # end is exclusive in annotation?
                    
                    # Determine type: Cut vs Gradual
                    # Heuristic based on transition length:
                    # - Length <= 1: Cut (instant scene change)
                    # - Length > 1: Gradual (dissolve/fade of any duration)
                    length = end - start
                    label = 1 if length <= 1 else 2
                    
                    # For Cut (label 1), we often mark 2 frames around cut
                    if label == 1:
                        # Ensure we mark at least one frame
                        targets[s:max(s+1, e+1)] = 1
                        transition_lengths[s:max(s+1, e+1)] = length  # NEW
                    else:
                        targets[s:e+1] = 2
                        transition_lengths[s:e+1] = length  # NEW: Store length for graduals
                        
            return frames, targets, transition_lengths  # NEW: Return lengths

        except Exception as e:
            # print(f"Error loading {vid_path}: {e}")
            # Return zero sample on error to not crash training
            return torch.zeros(self.seq_len, 3, *self.frame_size), torch.zeros(self.seq_len, dtype=torch.long), torch.zeros(self.seq_len, dtype=torch.long)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Data
    print("Initializing Datasets...")
    try:
        train_ds = ClipShotsDataset(CONFIG['input_root'], split='train', 
                                seq_len=CONFIG['seq_len'], 
                                cache_capacity=CONFIG['cache_capacity'])
                                
        val_ds = ClipShotsDataset(CONFIG['input_root'], split='test', 
                                seq_len=CONFIG['seq_len'],
                                cache_capacity=CONFIG['cache_capacity'])
    except Exception as e:
        print(f"Dataset Init Error: {e}")
        return

    # Num workers determines how many parallel video loaders we have
    # Setting pin_memory=True speeds up transfer to GPU
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
                              
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    
    # 2. Setup Model
    model = TurboSBD(num_classes=3).to(device)
    print(f"TurboSBD Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Loss Function
    if CONFIG.get('use_focal_loss', True):
        # Length-Aware Focal Loss: Combines hard example mining + long gradual boosting
        base_weights = torch.tensor([1.0, 3.0, 4.0]).to(device)  # Base weights (focal loss adds more)
        criterion = LengthAwareFocalLoss(
            gamma=CONFIG.get('focal_gamma', 2.0),
            base_weights=base_weights
        ).to(device)
        print(f"Using Length-Aware Focal Loss (γ={CONFIG.get('focal_gamma', 2.0)})")
        print("  - Short graduals (2-5f): weight 3.0")
        print("  - Medium graduals (6-15f): weight 5.0")
        print("  - Long graduals (16+f): weight 12.0")
    else:
        # Standard CrossEntropyLoss
        weights = torch.tensor([1.0, 3.0, 8.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using standard CrossEntropyLoss")
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], 
                           weight_decay=CONFIG.get('weight_decay', 1e-4))
    # Use CosineAnnealingLR instead of OneCycleLR for smoother decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    # 4. Loop
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    best_f1 = 0.0
    patience_counter = 0
    
    # Track metrics for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cut_f1': [],
        'train_gradual_f1': [],
        'val_cut_f1': [],
        'val_gradual_f1': [],
        'val_mean_f1': [],
        'learning_rate': []
    }
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Running metrics for display - track all confusion matrix elements
        tp_run = {1: 0, 2: 0}
        fp_run = {1: 0, 2: 0}
        tn_run = {1: 0, 2: 0}
        fn_run = {1: 0, 2: 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for frames, targets, transition_lengths in pbar:
            frames, targets, transition_lengths = frames.to(device), targets.to(device), transition_lengths.to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision
            with torch.amp.autocast('cuda', enabled=True):
                 logits = model(frames) # [B, T, 3]
                 
                 # Reshape for loss
                 logits_flat = logits.reshape(-1, 3)
                 targets_flat = targets.reshape(-1)
                 lengths_flat = transition_lengths.reshape(-1)
                 
                 # Compute loss (handles both focal and standard)
                 if CONFIG.get('use_focal_loss', True):
                     loss = criterion(logits_flat, targets_flat, lengths_flat)
                 else:
                     loss = criterion(logits_flat, targets_flat)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Simple metrics update for progress bar (no_grad to save memory)
            with torch.no_grad():
                preds = torch.argmax(logits, dim=2)
                for c in [1, 2]:
                    pm = (preds == c)
                    tm = (targets == c)
                    tp_run[c] += (pm & tm).sum().item()
                    fp_run[c] += (pm & ~tm).sum().item()
                    tn_run[c] += (~pm & ~tm).sum().item()
                    fn_run[c] += (~pm & tm).sum().item()
                
                # Compute current F1s and metrics
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = total_loss / num_batches
                metrics = {'loss': f"{avg_loss:.4f}", 'lr': f"{current_lr:.1e}"}
                
                for c, name in [(1, 'C'), (2, 'G')]:
                    p = tp_run[c] / (tp_run[c] + fp_run[c] + 1e-8)
                    r = tp_run[c] / (tp_run[c] + fn_run[c] + 1e-8)
                    f1 = 2 * p * r / (p + r + 1e-8)
                    metrics[f'{name}_P'] = f"{p:.3f}"
                    metrics[f'{name}_R'] = f"{r:.3f}"
                    metrics[f'{name}_F1'] = f"{f1:.3f}"
            
            pbar.set_postfix(metrics)
            
        # Training Epoch Summary
        avg_train_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1} Training Summary:")
        print(f"  Avg Loss: {avg_train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Compute training F1s
        train_cut_f1 = 0
        train_grad_f1 = 0
        for c, name in [(1, 'Cut'), (2, 'Gradual')]:
            p = tp_run[c] / (tp_run[c] + fp_run[c] + 1e-8)
            r = tp_run[c] / (tp_run[c] + fn_run[c] + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)
            if c == 1:
                train_cut_f1 = f1
            else:
                train_grad_f1 = f1
            print(f"  {name}: TP={tp_run[c]}, FP={fp_run[c]}, TN={tn_run[c]}, FN={fn_run[c]}")
        
        # Store training metrics
        history['train_loss'].append(avg_train_loss)
        history['train_cut_f1'].append(train_cut_f1)
        history['train_gradual_f1'].append(train_grad_f1)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Validation
        val_metrics = validate(model, val_loader, device)
        val_f1 = val_metrics['mean_f1']
        print(f"Epoch {epoch+1} Mean F1: {val_f1:.4f}")
        
        # Store validation metrics
        history['val_loss'].append(val_metrics['loss'])
        history['val_cut_f1'].append(val_metrics['cut_f1'])
        history['val_gradual_f1'].append(val_metrics['gradual_f1'])
        history['val_mean_f1'].append(val_f1)
        
        # Learning rate scheduling (per epoch for CosineAnnealingLR)
        scheduler.step()
        
        # Early Stopping Logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "outputs/turbo_sbd_best.pth")
            print(f"✓ Saved Best Model! (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")
            
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation F1: {best_f1:.4f}")
            break
    
    # Save training history and plots
    print("\n" + "="*80)
    print("Training Complete! Saving results...")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save metrics as JSON
    results = {
        'config': CONFIG,
        'best_f1': best_f1,
        'total_epochs': len(history['train_loss']),
        'history': history
    }
    
    json_path = f"outputs/training_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved metrics to {json_path}")
    
    # 2. Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mean F1
    axes[0, 1].plot(epochs, history['val_mean_f1'], 'g-', label='Mean F1', linewidth=2, marker='o')
    axes[0, 1].axhline(y=best_f1, color='r', linestyle='--', label=f'Best: {best_f1:.4f}')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('F1 Score', fontsize=11)
    axes[0, 1].set_title('Validation Mean F1', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cut F1 (Train vs Val)
    axes[1, 0].plot(epochs, history['train_cut_f1'], 'b--', label='Train Cut F1', linewidth=2, alpha=0.7)
    axes[1, 0].plot(epochs, history['val_cut_f1'], 'b-', label='Val Cut F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('F1 Score', fontsize=11)
    axes[1, 0].set_title('Cut Detection F1', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Gradual F1 (Train vs Val)
    axes[1, 1].plot(epochs, history['train_gradual_f1'], 'r--', label='Train Gradual F1', linewidth=2, alpha=0.7)
    axes[1, 1].plot(epochs, history['val_gradual_f1'], 'r-', label='Val Gradual F1', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('F1 Score', fontsize=11)
    axes[1, 1].set_title('Gradual Detection F1', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"outputs/training_plots_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plots to {plot_path}")
    plt.close()
    
    # 3. Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Total Epochs: {len(history['train_loss'])}")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Final Cut F1: {history['val_cut_f1'][-1]:.4f}")
    print(f"Final Gradual F1: {history['val_gradual_f1'][-1]:.4f}")
    print(f"{'='*80}")

def validate(model, loader, device):
    model.eval()
    
    # Class 1 (Cut), Class 2 (Gradual)
    tp = {1: 0, 2: 0}
    fp = {1: 0, 2: 0}
    tn = {1: 0, 2: 0}
    fn = {1: 0, 2: 0}
    
    total_loss = 0
    num_batches = 0
    
    # Loss criterion (same as training)
    if CONFIG.get('use_focal_loss', True):
        base_weights = torch.tensor([1.0, 3.0, 4.0]).to(device)
        criterion = LengthAwareFocalLoss(
            gamma=CONFIG.get('focal_gamma', 2.0),
            base_weights=base_weights
        ).to(device)
    else:
        weights = torch.tensor([1.0, 3.0, 8.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Confusion tracker for detailed analysis
    confusion = ConfusionTracker()
    
    with torch.no_grad():
        for frames, targets, transition_lengths in loader:
            frames, targets, transition_lengths = frames.to(device), targets.to(device), transition_lengths.to(device)
            logits = model(frames)
            preds = torch.argmax(logits, dim=2)
            
            # Calculate loss
            logits_flat = logits.reshape(-1, 3)
            targets_flat = targets.reshape(-1)
            lengths_flat = transition_lengths.reshape(-1)
            
            if CONFIG.get('use_focal_loss', True):
                loss = criterion(logits_flat, targets_flat, lengths_flat)
            else:
                loss = criterion(logits_flat, targets_flat)
                
            total_loss += loss.item()
            num_batches += 1
            
            # Track confusion patterns with router decisions AND transition lengths
            confusion.update(preds, targets, None, transition_lengths)
            
            for c in [1, 2]:
                pred_mask = (preds == c)
                target_mask = (targets == c)
                
                tp[c] += (pred_mask & target_mask).sum().item()
                fp[c] += (pred_mask & ~target_mask).sum().item()
                tn[c] += (~pred_mask & ~target_mask).sum().item()
                fn[c] += (~pred_mask & target_mask).sum().item()
    
    # Calculate metrics
    avg_val_loss = total_loss / num_batches
    res = {}
    print(f"\nValidation Results (Avg Loss: {avg_val_loss:.4f}):")
    for c, name in [(1, 'Cut'), (2, 'Gradual')]:
        p = tp[c] / (tp[c] + fp[c] + 1e-8)
        r = tp[c] / (tp[c] + fn[c] + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        res[name] = {'P': p, 'R': r, 'F1': f1}
        print(f"  {name:<8} - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f} | TP={tp[c]}, FP={fp[c]}, TN={tn[c]}, FN={fn[c]}")

    mean_f1 = (res['Cut']['F1'] + res['Gradual']['F1']) / 2
    print(f"  Mean F1 : {mean_f1:.4f}")
    
    # Print detailed confusion analysis
    confusion.print_report()
    
    # Return comprehensive metrics
    return {
        'mean_f1': mean_f1,
        'cut_f1': res['Cut']['F1'],
        'gradual_f1': res['Gradual']['F1'],
        'loss': avg_val_loss
    }

if __name__ == '__main__':
    # Ensure optimized decord reading context
    # os.environ['DECORD_NUM_THREADS'] = '4'
    
    # Create output dir
    os.makedirs('outputs', exist_ok=True)
    
    train()
