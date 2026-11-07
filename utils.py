import torch
import os
import numpy as np
import random
import torch

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

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    """Initialize worker with unique seed for reproducibility"""
    worker_seed = CONFIG['random_seed'] + worker_id
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
