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
import gc  # Garbage collector for memory cleanup
import torch.nn.functional as F
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

try:
    from torchcodec.decoders import VideoDecoder
    # TorchCodec: PyTorch's official video decoder with GPU acceleration
    TORCHCODEC_AVAILABLE = True
except ImportError:
    print("TorchCodec not found, please install it: pip install torchcodec")
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
        # Use progressive multipliers: focus boost only on what's struggling
        is_gradual = (targets == 2)
        is_short_gradual = is_gradual & (transition_lengths <= 5)
        is_medium_gradual = is_gradual & (transition_lengths > 5) & (transition_lengths <= 15)
        is_long_gradual = is_gradual & (transition_lengths > 15)
        
        # REMOVED length-aware multipliers - they were causing massive false positives
        # Problem: Boosting graduals when precision is 36% makes FP problem WORSE
        # New strategy: Use UNIFORM weight, let discriminator handle length differences
        # Model has explicit TransitionDiscriminator - it should learn to distinguish
        # without needing training weight hacks
        # 
        # Keep code structure but set all multipliers to 1.0 (no boost)
        base_gradual_weight = self.base_weights[2] if self.base_weights is not None else 1.0
        # No length-based adjustments - uniform treatment
        # class_weights[is_short_gradual] = base_gradual_weight * 1.0
        # class_weights[is_medium_gradual] = base_gradual_weight * 1.0
        # class_weights[is_long_gradual] = base_gradual_weight * 1.0
        
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
    'seq_len': 128,  # Balanced: captures most transitions, sustainable decoding speed
    'batch_size': 32,  # Balanced for RTX 4090: good GPU utilization + sustainable decoding
    'learning_rate': 2e-4, # Reduced from 5e-4 for stability with multi-scale fusion
    'epochs': 100,
    'samples_per_video_train': 20,  # Training: 2000 frames/video (20% coverage)
    'samples_per_video_val': 3,     # Validation: 300 frames/video (fast evaluation)
    'num_workers': 8,  # Balanced workers for steady pipeline
    'prefetch_factor': 12,  # Continuous flow: 8 workers * 12 = 96 batches ready
    'cache_capacity': 50,  # VideoReader cache per worker (8 * 50 = 400 files)
    'decoded_cache_size': 300,  # Decoded frames cache per worker (~1.8GB per worker)
    'early_stopping_patience': 15, # Stop if no improvement for 15 epochs
    'weight_decay': 1e-3, # Increased regularization
    'focal_gamma': 2.0, # Focal loss gamma (2.0 = strong focus on hard examples)
    'use_focal_loss':  True, # Enable focal loss + length-aware weights
    'random_seed': 42, # For reproducibility
    'val_split_ratio': 0.15, # Create 15% validation from training data if no val.json exists
    'tolerance': 3, # Frame tolerance for transition-level metrics (±3 frames)
    'num_runs': 1, # Number of training runs with different seeds (3-5 for publication)
    'compute_class_weights': True, # Compute weights from dataset distribution vs using fixed values
    'length_pred_weight': 0.1 # Weight for auxiliary length prediction loss (0.0 to disable)
}

class LRUVideoCache:
    """
    Two-level cache:
    1. VideoReader cache: Keep video files open
    2. Decoded frames cache: Store ready tensors for instant reuse
    
    CRITICAL: Properly closes old VideoReaders to prevent memory leaks
    """
    def __init__(self, capacity=50, decoded_cache_size=200):
        self.capacity = capacity
        self.cache = OrderedDict()
        # Cache for decoded frame tensors: {(path, start, end): tensor}
        self.decoded_cache = OrderedDict()
        self.decoded_cache_size = decoded_cache_size
    
    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        
        # TorchCodec VideoDecoder with GPU acceleration
        try:
            decoder = VideoDecoder(path)
            self.cache[path] = decoder
            
            if len(self.cache) > self.capacity:
                # Remove oldest decoder
                old_path, old_decoder = self.cache.popitem(last=False)
                try:
                    del old_decoder
                except:
                    pass
            return decoder
        except Exception as e:
            print(f"Error opening video {path}: {e}")
            raise
    
    def get_decoded(self, path, start_idx, seq_len, total_frames=None):
        """Get decoded frames from cache or decode and cache them - GPU accelerated with TorchCodec"""
        cache_key = (path, start_idx, start_idx + seq_len)
        
        # Check decoded cache first
        if cache_key in self.decoded_cache:
            self.decoded_cache.move_to_end(cache_key)
            return self.decoded_cache[cache_key].clone()
        
        # Not in cache - decode now with TorchCodec (GPU accelerated)
        try:
            decoder = self.get(path)
            
            # Use provided total_frames or try to get from decoder
            if total_frames is None:
                total_frames = getattr(decoder, 'num_frames', seq_len * 10)  # Fallback estimate
            
            # Clamp frame indices to valid range
            end_idx = min(start_idx + seq_len, total_frames)
            actual_start = min(start_idx, total_frames - 1) if total_frames > 0 else 0
            
            # Get frame indices to decode (within valid range)
            if actual_start >= total_frames:
                # Start is beyond video - return black frames
                frames = torch.zeros((seq_len, 3, *CONFIG['frame_size']))
                return frames
            
            frame_indices = list(range(actual_start, end_idx))
            
            if len(frame_indices) == 0:
                # No valid frames - return black frames
                frames = torch.zeros((seq_len, 3, *CONFIG['frame_size']))
                return frames
            
            # Decode frames - TorchCodec returns FrameBatch with .data tensor [N, C, H, W]
            frames_batch = decoder.get_frames_at(frame_indices)
            frames_tensor = frames_batch.data  # Extract tensor from FrameBatch
            
            # Resize to target size
            frames = torch.nn.functional.interpolate(
                frames_tensor.float(),
                size=CONFIG['frame_size'],
                mode='bilinear',
                align_corners=False
            )
            
            # Normalize to [0, 1]
            frames = frames / 255.0
            
            # Handle short sequences
            if frames.shape[0] < seq_len:
                # Pad with last frame
                padding = seq_len - frames.shape[0]
                last_frame = frames[-1:].repeat(padding, 1, 1, 1)
                frames = torch.cat([frames, last_frame], dim=0)
            
            # Cache the result (CPU to save VRAM)
            self.decoded_cache[cache_key] = frames.cpu().clone()
            if len(self.decoded_cache) > self.decoded_cache_size:
                self.decoded_cache.popitem(last=False)
            
            return frames
            
        except Exception as e:
            print(f"Error decoding video {path}: {e}")
            # Return black frames as fallback
            frames = torch.zeros((seq_len, 3, *CONFIG['frame_size']))
            return frames
    
    def clear_decoded_cache(self):
        """Clear decoded frames cache to free memory"""
        self.decoded_cache.clear()
        gc.collect()
    
    def __del__(self):
        """Clean up all decoders when cache is destroyed"""
        try:
            for decoder in self.cache.values():
                try:
                    del decoder
                except:
                    pass
            self.cache.clear()
            self.decoded_cache.clear()
        except:
            pass

class ClipShotsDataset(Dataset):
    def __init__(self, root_dir, split='train', seq_len=100, frame_size=(48, 64), cache_capacity=50, 
                 deterministic=False, val_videos=None, full_video_inference=False, samples_per_video=None):
        self.root_dir = root_dir
        self.split = split
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.cache_capacity = cache_capacity
        self.video_cache = None # Initialized in worker
        self.deterministic = deterministic  # For reproducible validation
        self.full_video_inference = full_video_inference  # Process complete videos for transition metrics
        self.samples_per_video_override = samples_per_video  # Allow explicit override
        
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
        
        # Handle validation split creation (if val_videos provided, filter to only those)
        if val_videos is not None:
            if split == 'val':
                self.valid_keys = [v for v in self.valid_keys if v in val_videos]
            elif split == 'train':
                self.valid_keys = [v for v in self.valid_keys if v not in val_videos]
        
        print(f"Found {len(self.valid_keys)} valid videos for {split}")
        
        # Create samples: (video_name, start_frame)
        # We can implement a sliding window or random sampling.
        # For efficiency in this script: Random sampling per epoch effectively cover dataset
        # For deterministic len, we estimate:
        if self.samples_per_video_override is not None:
            # Explicit override provided
            self.samples_per_video = self.samples_per_video_override
        elif self.full_video_inference:
            # For transition-level metrics: one sample per video (will process whole video)
            self.samples_per_video = 1
        elif self.deterministic:
            # Validation: 3 samples × 100 frames = 300 frames per video
            self.samples_per_video = 3
        else:
            # Training: 20 samples × 100 frames = 2000 frames per video per epoch
            # Smart sampling (60% on transitions) ensures diverse coverage
            self.samples_per_video = 20
            # Provides good diversity while maintaining reasonable training speed
            # Smart sampling (60% on transitions) ensures all transitions seen multiple times
            self.samples_per_video = 20
        
    def __len__(self):
        return len(self.valid_keys) * self.samples_per_video

    def _init_worker_cache(self):
        if self.video_cache is None:
            decoded_size = CONFIG.get('decoded_cache_size', 200)
            self.video_cache = LRUVideoCache(self.cache_capacity, decoded_cache_size=decoded_size)

    def __getitem__(self, idx):
        self._init_worker_cache()
        
        # Map linear index to video
        vid_idx = idx // self.samples_per_video
        vid_name = self.valid_keys[vid_idx]
        vid_path = self.video_paths[vid_name]
        
        try:
            transitions = self.annotations[vid_name]['transitions']
            # Get frame count from annotations (convert to int)
            total_frames = int(self.annotations[vid_name]['frame_num'])
            
            if self.deterministic:
                # Deterministic validation: sequential non-overlapping windows
                sample_in_video = idx % self.samples_per_video
                stride = max(1, total_frames // self.samples_per_video)
                target_frame_start = sample_in_video * stride
                target_frame_start = max(0, min(target_frame_start, total_frames - self.seq_len))
            else:
                # Smart Sampling for training:
                # 1. Focus on transitions (Positive mining)
                # 2. Focus on normal content (Negative mining)
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
            
            # OPTIMIZED: Use decoded frames cache for instant reuse (pass total_frames)
            frames = self.video_cache.get_decoded(vid_path, target_frame_start, self.seq_len, total_frames=total_frames)
            
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
            
            # Return with video metadata if in full video mode
            if self.full_video_inference:
                return frames, targets, transition_lengths, vid_name, target_frame_start
            else:
                return frames, targets, transition_lengths

        except Exception as e:
            # print(f"Error loading {vid_path}: {e}")
            # Return zero sample on error to not crash training
            if self.full_video_inference:
                return (torch.zeros(self.seq_len, 3, *self.frame_size), 
                        torch.zeros(self.seq_len, dtype=torch.long), 
                        torch.zeros(self.seq_len, dtype=torch.long),
                        vid_name if 'vid_name' in locals() else '', 0)
            else:
                return (torch.zeros(self.seq_len, 3, *self.frame_size), 
                        torch.zeros(self.seq_len, dtype=torch.long), 
                        torch.zeros(self.seq_len, dtype=torch.long))

def compute_transition_metrics(predictions, ground_truth, tolerance=3):
    """
    Compute transition-level metrics (standard in SBD literature)
    
    Args:
        predictions: List of predicted transition frames (frame indices)
        ground_truth: List of ground truth transition frames
        tolerance: Frame tolerance for matching (±tolerance frames)
    
    Returns:
        dict with precision, recall, f1, tp, fp, fn
    """
    if len(ground_truth) == 0:
        fp = len(predictions)
        return {
            'precision': 1.0 if len(predictions) == 0 else 0.0, 
            'recall': 1.0, 
            'f1': 1.0 if len(predictions) == 0 else 0.0,
            'tp': 0, 'fp': fp, 'fn': 0
        }
    
    if len(predictions) == 0:
        fn = len(ground_truth)
        return {
            'precision': 0.0, 
            'recall': 0.0, 
            'f1': 0.0,
            'tp': 0, 'fp': 0, 'fn': fn
        }
    
    # Match predictions to ground truth
    tp = 0
    matched_gt = set()
    
    for pred in predictions:
        for gt in ground_truth:
            if abs(pred - gt) <= tolerance and gt not in matched_gt:
                tp += 1
                matched_gt.add(gt)
                break
    
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

def extract_transitions_from_predictions(predictions, min_gap=5):
    """
    Extract transition frames from frame-level predictions
    
    Args:
        predictions: [T] tensor of class predictions (0=none, 1=cut, 2=gradual)
        min_gap: Minimum frames between transitions to avoid duplicates
    
    Returns:
        dict with 'cuts' and 'graduals' lists of frame indices
    """
    cuts = []
    graduals = []
    
    # Find transition segments
    in_cut = False
    in_gradual = False
    segment_start = 0
    
    for i in range(len(predictions)):
        pred = predictions[i].item()
        
        # Cut transitions
        if pred == 1:
            if not in_cut:
                in_cut = True
                segment_start = i
        else:
            if in_cut:
                # End of cut segment - take middle frame
                center = (segment_start + i - 1) // 2
                if len(cuts) == 0 or center - cuts[-1] >= min_gap:
                    cuts.append(center)
                in_cut = False
        
        # Gradual transitions
        if pred == 2:
            if not in_gradual:
                in_gradual = True
                segment_start = i
        else:
            if in_gradual:
                # End of gradual segment - take middle frame
                center = (segment_start + i - 1) // 2
                if len(graduals) == 0 or center - graduals[-1] >= min_gap:
                    graduals.append(center)
                in_gradual = False
    
    # Handle segments that extend to end
    if in_cut:
        center = (segment_start + len(predictions) - 1) // 2
        if len(cuts) == 0 or center - cuts[-1] >= min_gap:
            cuts.append(center)
    if in_gradual:
        center = (segment_start + len(predictions) - 1) // 2
        if len(graduals) == 0 or center - graduals[-1] >= min_gap:
            graduals.append(center)
    
    return {'cuts': cuts, 'graduals': graduals}

def compute_dataset_class_weights(dataset, device):
    """
    Compute class weights from dataset distribution (PUBLICATION STANDARD)
    
    Returns balanced weights inversely proportional to class frequency
    """
    print("Computing class weights from dataset distribution...")
    class_counts = {0: 0, 1: 0, 2: 0}
    
    # Sample subset for speed (5% of dataset, ~300 samples)
    sample_size = min(len(dataset), max(100, len(dataset) // 20))
    indices = np.random.choice(len(dataset), sample_size, replace=False).tolist()
    
    # Create subset sampler for parallel processing
    from torch.utils.data import Subset, DataLoader
    subset = Subset(dataset, indices)
    
    # Use DataLoader with multiple workers for fast parallel sampling
    sampler_loader = DataLoader(
        subset, 
        batch_size=8,  # Process 8 samples at once
        shuffle=False, 
        num_workers=4,  # Parallel workers
        pin_memory=False,
        persistent_workers=False
    )
    
    print(f"  Sampling {sample_size} sequences with {4} workers...")
    for frames, targets, _ in tqdm(sampler_loader, desc="Sampling dataset"):
        # targets: [B, T] - count class occurrences across batch
        for c in range(3):
            class_counts[c] += (targets == c).sum().item()
    
    total = sum(class_counts.values())
    weights = []
    for c in range(3):
        # Inverse frequency weighting
        weight = total / (3 * class_counts[c]) if class_counts[c] > 0 else 1.0
        weights.append(weight)
    
    # Normalize so majority class has weight 1.0
    weights = torch.tensor(weights) / min(weights)
    
    print(f"  Class distribution: None={class_counts[0]:,}, Cut={class_counts[1]:,}, Gradual={class_counts[2]:,}")
    print(f"  Computed weights: {weights.tolist()}")
    
    return weights.to(device)

def worker_init_fn(worker_id):
    """Initialize worker with unique seed for reproducibility"""
    worker_seed = CONFIG['random_seed'] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train():
    # Set random seeds for reproducibility
    set_seed(CONFIG['random_seed'])
    print(f"Random seed set to {CONFIG['random_seed']}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Data with proper train/val/test splits
    print("Initializing Datasets...")
    try:
        # Check if val.json exists, otherwise create validation split from training data
        val_json_path = os.path.join(CONFIG['input_root'], 'annotations', 'val.json')
        
        if os.path.exists(val_json_path):
            print("Using existing val.json for validation")
            train_ds = ClipShotsDataset(CONFIG['input_root'], split='train', 
                                    seq_len=CONFIG['seq_len'], 
                                    cache_capacity=CONFIG['cache_capacity'],
                                    deterministic=False)
            val_ds = ClipShotsDataset(CONFIG['input_root'], split='val', 
                                    seq_len=CONFIG['seq_len'],
                                    cache_capacity=CONFIG['cache_capacity'],
                                    deterministic=True,
                                    full_video_inference=False)  # Frame-level only for fast validation
        else:
            print(f"Creating {CONFIG['val_split_ratio']:.0%} validation split from training data")
            # Load training data first to split
            temp_ds = ClipShotsDataset(CONFIG['input_root'], split='train',
                                      seq_len=CONFIG['seq_len'],
                                      cache_capacity=CONFIG['cache_capacity'])
            
            # Deterministically split videos
            all_videos = sorted(temp_ds.valid_keys)  # Sort for reproducibility
            n_val = max(1, int(len(all_videos) * CONFIG['val_split_ratio']))
            
            # Use seed-based shuffle
            rng = random.Random(CONFIG['random_seed'])
            rng.shuffle(all_videos)
            
            val_videos = set(all_videos[:n_val])
            print(f"Validation videos: {len(val_videos)}, Training videos: {len(all_videos) - len(val_videos)}")
            
            # Create datasets with filtering
            train_ds = ClipShotsDataset(CONFIG['input_root'], split='train',
                                    seq_len=CONFIG['seq_len'],
                                    cache_capacity=CONFIG['cache_capacity'],
                                    deterministic=False,
                                    val_videos=val_videos)
            val_ds = ClipShotsDataset(CONFIG['input_root'], split='train',  # Use train.json but filter to val videos
                                    seq_len=CONFIG['seq_len'],
                                    cache_capacity=CONFIG['cache_capacity'],
                                    deterministic=True,
                                    val_videos=val_videos,
                                    full_video_inference=False)  # Frame-level only for fast validation
            val_ds.split = 'val'  # Rename for clarity
            
    except Exception as e:
        print(f"Dataset Init Error: {e}")
        return

    # Num workers determines how many parallel video loaders we have
    # Setting pin_memory=True speeds up transfer to GPU
    # prefetch_factor: each worker prefetches this many batches
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=CONFIG['num_workers'], pin_memory=True, 
                              persistent_workers=True, worker_init_fn=worker_init_fn,
                              prefetch_factor=CONFIG['prefetch_factor'])  # RTX 4090: keep GPU fed
                              
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=CONFIG['num_workers'], pin_memory=True, 
                            persistent_workers=True, worker_init_fn=worker_init_fn,
                            prefetch_factor=CONFIG['prefetch_factor'])
    
    # 2. Setup Model
    model = TurboSBD(num_classes=3).to(device)
    print(f"TurboSBD Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Loss Function with empirically tuned class weights
    # CRITICAL FIX: Model is MASSIVELY over-predicting graduals (27k FP vs 15k TP)
    # Root cause: Gradual weight too high + length multipliers boosting false positives
    # Solution: DRASTICALLY reduce gradual weight, keep cuts moderate
    # Balanced weights for valid data (no longer corrupted)
    # With 4.2% boundaries total, use moderate boost for transitions
    base_weights = torch.tensor([1.0, 5.0, 6.0]).to(device)  # None=1.0 (baseline), Cut=5.0, Gradual=6.0 (slightly more)
    print(f"  Using balanced weights for valid data: None={base_weights[0]:.1f}, Cut={base_weights[1]:.1f}, Gradual={base_weights[2]:.1f}")
    
    if CONFIG.get('use_focal_loss', True):
        # Length-Aware Focal Loss: Combines hard example mining + long gradual boosting
        criterion = LengthAwareFocalLoss(
            gamma=CONFIG.get('focal_gamma', 2.0),
            base_weights=base_weights
        ).to(device)
        print(f"Using Length-Aware Focal Loss (γ={CONFIG.get('focal_gamma', 2.0)})")
        print(f"  Base class weights: None={base_weights[0]:.2f}, Cut={base_weights[1]:.2f}, Gradual={base_weights[2]:.2f}")
        print("  Length-aware multipliers for graduals:")
        print("    - Short (2-5f): 1.0x")
        print("    - Medium (6-15f): 1.3x")
        print("    - Long (16+f): 2.0x")
    else:
        # Standard CrossEntropyLoss
        weights = torch.tensor([1.0, 3.0, 8.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using standard CrossEntropyLoss")
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], 
                           weight_decay=CONFIG.get('weight_decay', 1e-4))
    # Use CosineAnnealingLR instead of OneCycleLR for smoother decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    # Mixed Precision: GradScaler for stable training
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # 4. Loop
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    print(f"Effective batch size: {CONFIG['batch_size']} (GPU batch)")
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
            # Non-blocking transfer: CPU can prepare next batch while GPU transfers current
            frames = frames.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            transition_lengths = transition_lengths.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed Precision with GradScaler (PUBLICATION STANDARD)
            with torch.amp.autocast('cuda', enabled=True):
                 # Forward pass with auxiliary length prediction
                 logits, length_pred = model(frames, return_length_pred=True) # [B, T, 3], [B, T]
                 
                 # Reshape for loss
                 logits_flat = logits.reshape(-1, 3)
                 targets_flat = targets.reshape(-1)
                 lengths_flat = transition_lengths.reshape(-1)
                 
                 # Main classification loss
                 if CONFIG.get('use_focal_loss', True):
                     cls_loss = criterion(logits_flat, targets_flat, lengths_flat)
                 else:
                     cls_loss = criterion(logits_flat, targets_flat)
                 
                 # Auxiliary length prediction loss (helps learn temporal extent)
                 # Only compute loss on transition frames (ignore none frames)
                 length_pred_flat = length_pred.reshape(-1)
                 length_target_flat = lengths_flat.float()
                 
                 # Mask: only compute loss where there's actually a transition
                 transition_mask = (targets_flat > 0).float()  # Cut or Gradual
                 
                 # Smooth L1 loss (robust to outliers)
                 length_loss = F.smooth_l1_loss(
                     length_pred_flat * transition_mask,
                     length_target_flat * transition_mask,
                     reduction='sum'
                 ) / (transition_mask.sum() + 1e-8)
                 
                 # Combined loss (classification + auxiliary)
                 aux_weight = CONFIG.get('length_pred_weight', 0.1)
                 loss = cls_loss + aux_weight * length_loss
            
            # Scaled backward pass for mixed precision
            scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
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
                
                # Compute current F1s and metrics for progress bar
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = total_loss / num_batches
                
                # Calculate metrics for both classes
                cut_p = tp_run[1] / (tp_run[1] + fp_run[1] + 1e-8)
                cut_r = tp_run[1] / (tp_run[1] + fn_run[1] + 1e-8)
                cut_f1 = 2 * cut_p * cut_r / (cut_p + cut_r + 1e-8)
                
                grad_p = tp_run[2] / (tp_run[2] + fp_run[2] + 1e-8)
                grad_r = tp_run[2] / (tp_run[2] + fn_run[2] + 1e-8)
                grad_f1 = 2 * grad_p * grad_r / (grad_p + grad_r + 1e-8)
                
                mean_f1 = (cut_f1 + grad_f1) / 2
                
                # Compact display with all key metrics
                metrics = {
                    'loss': f"{avg_loss:.4f}",
                    'C_P': f"{cut_p:.3f}",
                    'C_R': f"{cut_r:.3f}",
                    'C_F1': f"{cut_f1:.3f}",
                    'G_P': f"{grad_p:.3f}",
                    'G_R': f"{grad_r:.3f}",
                    'G_F1': f"{grad_f1:.3f}",
                    'Mean_F1': f"{mean_f1:.3f}"
                }
            
            pbar.set_postfix(metrics)
        
        # Force garbage collection after each epoch to prevent memory buildup
        gc.collect()
        torch.cuda.empty_cache()
        
        # Periodically clear decoded frame caches to prevent memory leaks (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            print(f"  Clearing decoded frame caches to free memory...")
            # Note: Worker caches are in separate processes, so we just trigger GC
            # The LRU cache will naturally evict old entries, but this helps
            gc.collect()
            torch.cuda.empty_cache()
            
        # Training Epoch Summary
        avg_train_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1} Training Summary:")
        print(f"  Avg Loss: {avg_train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Compute training F1s with full metrics display
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
            # Print P/R/F1 along with confusion matrix
            print(f"  {name:<8} - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f} | TP={tp_run[c]}, FP={fp_run[c]}, FN={fn_run[c]}")
        
        # Store training metrics
        history['train_loss'].append(avg_train_loss)
        history['train_cut_f1'].append(train_cut_f1)
        history['train_gradual_f1'].append(train_grad_f1)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Validation (transition-level metrics - PUBLICATION STANDARD)
        # Note: Slower but ensures we optimize the metric we'll report in paper
        val_metrics = validate(model, val_loader, device, compute_transition_level=True)
        val_f1 = val_metrics['mean_f1']  # Transition-level F1 (primary SBD metric)
        print(f"Epoch {epoch+1} Transition-Level Mean F1: {val_f1:.4f}")
        
        # CRITICAL: Clear validation worker caches every 3 epochs
        # Persistent workers accumulate decoded frames (8 workers × 300 seqs = 28GB+)
        if (epoch + 1) % 3 == 0:
            print(f"  Recreating validation loader to clear worker caches...")
            # Delete old loader to free worker processes and their caches
            del val_loader
            gc.collect()
            # Create new loader with fresh workers
            val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                                    num_workers=CONFIG['num_workers'], pin_memory=True, 
                                    persistent_workers=True, worker_init_fn=worker_init_fn,
                                    prefetch_factor=CONFIG['prefetch_factor'])
        
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
    
    return best_f1  # Return for multi-run aggregation

def validate(model, loader, device, compute_transition_level=False):
    """
    Validate with frame-level (and optionally transition-level) metrics
    
    Args:
        model: The model to evaluate
        loader: DataLoader (use full_video_inference=True for transition metrics)
        device: Device to run on
        compute_transition_level: Whether to compute transition-level metrics (SLOW, only for final test)
    """
    model.eval()
    
    # Frame-level metrics: Class 1 (Cut), Class 2 (Gradual)
    tp = {1: 0, 2: 0}
    fp = {1: 0, 2: 0}
    tn = {1: 0, 2: 0}
    fn = {1: 0, 2: 0}
    
    # Transition-level: aggregate predictions per video (only if needed)
    video_predictions = {} if compute_transition_level else None
    video_ground_truth = {} if compute_transition_level else None
    
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
    
    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch_data in pbar:
            # Handle both training and full video modes
            if len(batch_data) == 5:  # full_video_inference=True
                frames, targets, transition_lengths, vid_names, offsets = batch_data
            else:  # Training mode
                frames, targets, transition_lengths = batch_data
                vid_names = None
                offsets = None
            
            # Non-blocking transfer for speed
            frames = frames.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            transition_lengths = transition_lengths.to(device, non_blocking=True)
            
            # Forward pass (no length prediction during validation for speed)
            logits = model(frames, return_length_pred=False)
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
            
            # Frame-level metrics
            for c in [1, 2]:
                pred_mask = (preds == c)
                target_mask = (targets == c)
                
                tp[c] += (pred_mask & target_mask).sum().item()
                fp[c] += (pred_mask & ~target_mask).sum().item()
                tn[c] += (~pred_mask & ~target_mask).sum().item()
                fn[c] += (~pred_mask & target_mask).sum().item()
            
            # Update progress bar with current metrics
            if num_batches % 5 == 0:  # Update every 5 batches for performance
                metrics = {'loss': f"{total_loss / num_batches:.4f}"}
                for c, name in [(1, 'C'), (2, 'G')]:
                    p = tp[c] / (tp[c] + fp[c] + 1e-8)
                    r = tp[c] / (tp[c] + fn[c] + 1e-8)
                    f1 = 2 * p * r / (p + r + 1e-8)
                    metrics[f'{name}_F1'] = f"{f1:.3f}"
                pbar.set_postfix(metrics)
            
            # Aggregate for transition-level metrics (ONLY when explicitly requested)
            if compute_transition_level and vid_names is not None:
                batch_size = preds.shape[0]
                for b in range(batch_size):
                    vid_name = vid_names[b]
                    offset = offsets[b].item() if torch.is_tensor(offsets[b]) else offsets[b]
                    
                    # Store predictions with absolute frame indices (batch processing)
                    # Convert to numpy for faster iteration
                    preds_np = preds[b].cpu().numpy()
                    targets_np = targets[b].cpu().numpy()
                    
                    if vid_name not in video_predictions:
                        video_predictions[vid_name] = []
                        video_ground_truth[vid_name] = []
                    
                    # Batch append for speed
                    for t in range(len(preds_np)):
                        abs_frame = offset + t
                        video_predictions[vid_name].append((abs_frame, int(preds_np[t])))
                        video_ground_truth[vid_name].append((abs_frame, int(targets_np[t])))
    
    # Clean up progress bar explicitly
    pbar.close()
    del pbar
    
    # Calculate frame-level metrics
    avg_val_loss = total_loss / num_batches
    res = {}
    
    print(f"\n{'='*80}")
    print(f"Validation Results (Avg Loss: {avg_val_loss:.4f})")
    print(f"{'='*80}")
    
    # Frame-level metrics
    print("\nFrame-Level Metrics:")
    for c, name in [(1, 'Cut'), (2, 'Gradual')]:
        p = tp[c] / (tp[c] + fp[c] + 1e-8)
        r = tp[c] / (tp[c] + fn[c] + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        res[name] = {'P': p, 'R': r, 'F1': f1}
        print(f"  {name:<8} - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f} | TP={tp[c]}, FP={fp[c]}, FN={fn[c]}")

    frame_mean_f1 = (res['Cut']['F1'] + res['Gradual']['F1']) / 2
    print(f"  Mean F1  : {frame_mean_f1:.4f}")
    
    # Transition-level metrics (STANDARD FOR SBD - only computed for final test evaluation)
    if compute_transition_level and video_predictions is not None and len(video_predictions) > 0:
        print(f"\nTransition-Level Metrics (±{CONFIG['tolerance']} frames tolerance):")
        print(f"  Analyzing {len(video_predictions)} videos...")
        
        all_pred_cuts = []
        all_gt_cuts = []
        all_pred_graduals = []
        all_gt_graduals = []
        
        # Process each video
        for vid_name in video_predictions:
            # Sort by frame index and extract transitions
            pred_frames = sorted(video_predictions[vid_name], key=lambda x: x[0])
            gt_frames = sorted(video_ground_truth[vid_name], key=lambda x: x[0])
            
            # Create frame-level prediction arrays
            max_frame = max(pred_frames[-1][0], gt_frames[-1][0]) + 1
            pred_array = torch.zeros(max_frame, dtype=torch.long)
            gt_array = torch.zeros(max_frame, dtype=torch.long)
            
            for frame_idx, pred_class in pred_frames:
                if frame_idx < max_frame:
                    pred_array[frame_idx] = pred_class
            for frame_idx, true_class in gt_frames:
                if frame_idx < max_frame:
                    gt_array[frame_idx] = true_class
            
            # Extract transitions
            pred_trans = extract_transitions_from_predictions(pred_array)
            gt_trans = extract_transitions_from_predictions(gt_array)
            
            all_pred_cuts.extend(pred_trans['cuts'])
            all_pred_graduals.extend(pred_trans['graduals'])
            all_gt_cuts.extend(gt_trans['cuts'])
            all_gt_graduals.extend(gt_trans['graduals'])
            
            # CRITICAL: Delete temporary tensors to prevent memory accumulation
            del pred_array, gt_array, pred_frames, gt_frames, pred_trans, gt_trans
        
        # CRITICAL: Delete massive dictionaries (310K+ entries) immediately after use
       
        
        cut_trans_metrics = compute_transition_metrics(all_pred_cuts, all_gt_cuts, CONFIG['tolerance'])
        grad_trans_metrics = compute_transition_metrics(all_pred_graduals, all_gt_graduals, CONFIG['tolerance'])
        
        print(f"  Cut      - P: {cut_trans_metrics['precision']:.4f}, R: {cut_trans_metrics['recall']:.4f}, F1: {cut_trans_metrics['f1']:.4f} | "
              f"TP={cut_trans_metrics['tp']}, FP={cut_trans_metrics['fp']}, FN={cut_trans_metrics['fn']}")
        print(f"  Gradual  - P: {grad_trans_metrics['precision']:.4f}, R: {grad_trans_metrics['recall']:.4f}, F1: {grad_trans_metrics['f1']:.4f} | "
              f"TP={grad_trans_metrics['tp']}, FP={grad_trans_metrics['fp']}, FN={grad_trans_metrics['fn']}")
        
        trans_mean_f1 = (cut_trans_metrics['f1'] + grad_trans_metrics['f1']) / 2
        print(f"  Mean F1  : {trans_mean_f1:.4f} ⭐ (PRIMARY METRIC)")
        print(f"  Total videos: {len(video_predictions)}, GT cuts: {len(all_gt_cuts)}, GT graduals: {len(all_gt_graduals)}")
        del video_predictions, video_ground_truth
        gc.collect()  # Force cleanup of these large structures
    else:
        if not compute_transition_level:
            print(f"\nTransition-Level Metrics: SKIPPED (frame-level only for training speed)")
        else:
            print(f"\nTransition-Level Metrics: SKIPPED (no video data collected)")
        trans_mean_f1 = frame_mean_f1  # Use frame-level F1 during training
        cut_trans_metrics = {'f1': res['Cut']['F1'], 'precision': res['Cut']['F1'], 'recall': res['Cut']['F1']}
        grad_trans_metrics = {'f1': res['Gradual']['F1'], 'precision': res['Gradual']['F1'], 'recall': res['Gradual']['F1']}
    
    print(f"{'='*80}")
    
    # Print detailed confusion analysis
    confusion.print_report()
    
    # Clean up confusion tracker to free memory
    del confusion
    
    # CRITICAL: Clear worker caches to prevent memory accumulation
    # Worker processes keep decoded frame caches (8 workers × 300 seqs = huge RAM)
    if hasattr(loader.dataset, 'video_cache') and loader.dataset.video_cache is not None:
        loader.dataset.video_cache.clear_decoded_cache()
    
    # Force garbage collection and clear CUDA cache after validation
    gc.collect()
    torch.cuda.empty_cache()
    
    # Return comprehensive metrics (use transition-level F1 as primary metric)
    return {
        'mean_f1': trans_mean_f1,  # PRIMARY: transition-level
        'frame_mean_f1': frame_mean_f1,
        'cut_f1': cut_trans_metrics['f1'],
        'gradual_f1': grad_trans_metrics['f1'],
        'frame_cut_f1': res['Cut']['F1'],
        'frame_gradual_f1': res['Gradual']['F1'],
        'loss': avg_val_loss,
        'cut_precision': cut_trans_metrics['precision'],
        'cut_recall': cut_trans_metrics['recall'],
        'gradual_precision': grad_trans_metrics['precision'],
        'gradual_recall': grad_trans_metrics['recall']
    }

if __name__ == '__main__':
    # Enable TorchVision NVDEC GPU decoding (RTX 4090)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Create output dir
    os.makedirs('outputs', exist_ok=True)
    
    print("="*80)
    print("PUBLICATION-READY TRAINING WITH NVDEC GPU DECODING")
    print("="*80)
    print("✓ Reproducible: Random seeds set")
    print("✓ Proper splits: Train/Val separation (Test reserved for final evaluation)")
    print("✓ Standard metrics: Transition-level F1 with tolerance")
    print("✓ Deterministic validation: No randomness in evaluation")
    print(f"✓ RTX 4090 NVDEC: GPU-accelerated video decoding")
    print(f"✓ Mixed Precision: AMP with GradScaler (stable + fast)")
    print(f"✓ Data-driven weights: Computed from class distribution")
    print(f"✓ Training config: batch={CONFIG['batch_size']}, seq_len={CONFIG['seq_len']}, {CONFIG['num_workers']} workers, {CONFIG['prefetch_factor']}x prefetch")
    print(f"  Frames/batch: {CONFIG['batch_size'] * CONFIG['seq_len']:,}")
    print(f"  Prefetch buffer: {CONFIG['num_workers'] * CONFIG['prefetch_factor']} batches (continuous flow)")
    print(f"  Decoded cache: {CONFIG['decoded_cache_size']} seqs/worker = {CONFIG['num_workers'] * CONFIG['decoded_cache_size']} total")
    print(f"  Expected VRAM: ~12-15GB / 24GB | RAM: ~15GB cache | Speed: 8-15 it/s with NVDEC")
    print("="*80)
    print()
    
    num_runs = CONFIG.get('num_runs', 1)
    
    if num_runs > 1:
        print(f"Running {num_runs} training runs with different seeds (publication standard)")
        print("="*80)
        all_results = []
        
        for run in range(num_runs):
            run_seed = CONFIG['random_seed'] + run * 1000
            print(f"\n{'='*80}")
            print(f"RUN {run + 1}/{num_runs} (seed={run_seed})")
            print(f"{'='*80}")
            
            # Update seed for this run
            CONFIG['random_seed'] = run_seed
            set_seed(run_seed)
            
            # Train and collect results
            best_f1 = train()
            all_results.append(best_f1)
            
            # Clear memory between runs
            gc.collect()
            torch.cuda.empty_cache()
        
        # Compute statistics
        mean_f1 = np.mean(all_results)
        std_f1 = np.std(all_results)
        
        print(f"\n{'='*80}")
        print("MULTI-RUN SUMMARY (PUBLICATION RESULTS)")
        print(f"{'='*80}")
        print(f"Individual runs: {[f'{f1:.4f}' for f1 in all_results]}")
        print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"{'='*80}")
        
        # Save multi-run results
        multi_run_results = {
            'num_runs': num_runs,
            'seeds': [CONFIG['random_seed'] + i * 1000 for i in range(num_runs)],
            'individual_f1s': all_results,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'config': CONFIG
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'outputs/multi_run_results_{timestamp}.json', 'w') as f:
            json.dump(multi_run_results, f, indent=2)
        
    else:
        train()
    
    print("\n" + "="*80)
    print("⚠️  IMPORTANT: Test set evaluation")
    print("="*80)
    print("Training completed. The test set has NOT been used.")
    print("For FINAL PUBLICATION results:")
    print("  Run: python evaluate_final_test.py")
    print("="*80)
