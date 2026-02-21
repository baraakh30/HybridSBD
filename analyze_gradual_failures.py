"""
Analyze gradual transition failures:
- False Negatives: graduals being missed (predicted as none)
- False Positives: non-graduals predicted as gradual

Outputs detailed characteristics to understand model weaknesses.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from detection.context_aware_detector import create_context_aware_detector
from train_boundary_detector import PredecodedDataset
from train_turbo_sbd_v2 import (
    extract_transitions_from_predictions,
    compute_transition_metrics,
)


def analyze_frame_characteristics(frames, start_idx, end_idx):
    """
    Analyze characteristics of a transition region.
    
    Args:
        frames: [T, C, H, W] tensor
        start_idx: start frame index
        end_idx: end frame index (exclusive)
    
    Returns:
        dict of characteristics
    """
    region = frames[start_idx:end_idx]
    T = region.shape[0]
    
    if T < 2:
        return {}
    
    # Frame differences
    diffs = torch.diff(region, dim=0)
    diff_norms = torch.norm(diffs.flatten(1), dim=1)
    
    # Color statistics
    mean_color = region.mean(dim=(0, 2, 3))  # [C]
    std_color = region.std(dim=(0, 2, 3))
    
    # Brightness
    brightness = region.mean(dim=(1, 2, 3))  # [T]
    brightness_change = brightness[-1] - brightness[0]
    
    # Motion/change intensity
    avg_diff = diff_norms.mean().item()
    max_diff = diff_norms.max().item()
    diff_std = diff_norms.std().item()
    
    # Temporal smoothness (variance of differences)
    diff_variance = diff_norms.var().item()
    
    return {
        'duration': T,
        'avg_frame_diff': avg_diff,
        'max_frame_diff': max_diff,
        'diff_std': diff_std,
        'diff_variance': diff_variance,
        'brightness_change': brightness_change.item(),
        'mean_r': mean_color[0].item(),
        'mean_g': mean_color[1].item(),
        'mean_b': mean_color[2].item(),
        'std_r': std_color[0].item(),
        'std_g': std_color[1].item(),
        'std_b': std_color[2].item(),
    }


def analyze_boundary_signal(boundary_probs, type_logits, start_idx, end_idx):
    """
    Analyze model's boundary signal in a region.
    
    Args:
        boundary_probs: [T] probabilities
        type_logits: [T, 3] type logits
        start_idx: start frame
        end_idx: end frame
    
    Returns:
        dict of signal characteristics
    """
    region_probs = boundary_probs[start_idx:end_idx]
    region_types = type_logits[start_idx:end_idx]
    
    type_probs = F.softmax(region_types, dim=-1)
    
    return {
        'max_boundary_prob': region_probs.max().item(),
        'mean_boundary_prob': region_probs.mean().item(),
        'median_boundary_prob': region_probs.median().item(),
        'boundary_prob_std': region_probs.std().item(),
        'max_gradual_prob': type_probs[:, 2].max().item(),
        'mean_gradual_prob': type_probs[:, 2].mean().item(),
        'max_cut_prob': type_probs[:, 1].max().item(),
        'mean_cut_prob': type_probs[:, 1].mean().item(),
    }


def extract_segments_with_bounds(predictions, min_gap=5):
    """
    Extract transition segments with start/end bounds.
    
    Returns:
        dict with 'cuts' and 'graduals' lists of dicts {'start', 'end', 'center'}
    """
    cuts = []
    graduals = []
    
    in_cut = False
    in_gradual = False
    segment_start = 0
    
    for i in range(len(predictions)):
        pred = predictions[i].item() if torch.is_tensor(predictions[i]) else predictions[i]
        
        # Cut transitions
        if pred == 1:
            if not in_cut:
                in_cut = True
                segment_start = i
        else:
            if in_cut:
                segment_end = i - 1
                center = (segment_start + segment_end) // 2
                if len(cuts) == 0 or center - cuts[-1]['center'] >= min_gap:
                    cuts.append({'start': segment_start, 'end': segment_end, 'center': center})
                in_cut = False
        
        # Gradual transitions
        if pred == 2:
            if not in_gradual:
                in_gradual = True
                segment_start = i
        else:
            if in_gradual:
                segment_end = i - 1
                center = (segment_start + segment_end) // 2
                if len(graduals) == 0 or center - graduals[-1]['center'] >= min_gap:
                    graduals.append({'start': segment_start, 'end': segment_end, 'center': center})
                in_gradual = False
    
    # Handle segments that extend to end
    if in_cut:
        segment_end = len(predictions) - 1
        center = (segment_start + segment_end) // 2
        if len(cuts) == 0 or center - cuts[-1]['center'] >= min_gap:
            cuts.append({'start': segment_start, 'end': segment_end, 'center': center})
    if in_gradual:
        segment_end = len(predictions) - 1
        center = (segment_start + segment_end) // 2
        if len(graduals) == 0 or center - graduals[-1]['center'] >= min_gap:
            graduals.append({'start': segment_start, 'end': segment_end, 'center': center})
    
    return {'cuts': cuts, 'graduals': graduals}


def find_nearest_prediction(pred_transitions, gt_center, tolerance=10):
    """Find nearest predicted transition to a ground truth center position."""
    if not pred_transitions:
        return None, float('inf')
    
    # pred_transitions is a list of frame indices
    distances = [abs(p - gt_center) for p in pred_transitions]
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    
    if min_dist <= tolerance:
        return pred_transitions[min_idx], min_dist
    return None, min_dist


def analyze_gradual_failures(
    model,
    dataloader,
    device,
    boundary_threshold=0.55,
    tolerance=3,
    max_samples=50,
):
    """
    Analyze gradual transition failures.
    
    Returns:
        dict with FN and FP analyses
    """
    model.eval()
    
    false_negatives = []  # Missed graduals
    false_positives = []  # Wrong gradual predictions
    true_positives = []   # Correct gradual predictions (for comparison)
    
    num_samples = 0
    
    with torch.no_grad():
        for frames, labels, lengths in tqdm(dataloader, desc="Analyzing"):
            if num_samples >= max_samples:
                break
            
            frames = frames.to(device)
            labels = labels.to(device)
            
            b_logits, t_logits, e_scores, seg_emb = model(frames)
            
            boundary_probs = torch.sigmoid(b_logits.squeeze(-1))
            is_boundary = boundary_probs > boundary_threshold
            type_preds = torch.argmax(t_logits, dim=-1)
            preds_3class = torch.where(is_boundary, type_preds, torch.zeros_like(type_preds))
            
            for b in range(frames.shape[0]):
                if num_samples >= max_samples:
                    break
                num_samples += 1
                
                frame_seq = frames[b].cpu()
                label_seq = labels[b].cpu()
                pred_seq = preds_3class[b].cpu()
                boundary_prob_seq = boundary_probs[b].cpu()
                type_logit_seq = t_logits[b].cpu()
                
                # Extract transitions
                # Use extract_segments_with_bounds for GT to get start/end info  
                gt_trans = extract_segments_with_bounds(label_seq, min_gap=3)
                # Use standard function for predictions (returns frame indices)
                pred_trans = extract_transitions_from_predictions(pred_seq, min_gap=3)
                
                # Analyze ground truth graduals
                for gt_grad in gt_trans['graduals']:
                    gt_center = gt_grad['center']
                    
                    # Find if it was detected
                    matched_pred, dist = find_nearest_prediction(
                        pred_trans['graduals'], gt_center, tolerance
                    )
                    
                    # Get characteristics
                    frame_chars = analyze_frame_characteristics(
                        frame_seq, gt_grad['start'], gt_grad['end']
                    )
                    signal_chars = analyze_boundary_signal(
                        boundary_prob_seq, type_logit_seq,
                        gt_grad['start'], gt_grad['end']
                    )
                    
                    analysis = {
                        **frame_chars,
                        **signal_chars,
                        'gt_start': gt_grad['start'],
                        'gt_end': gt_grad['end'],
                        'gt_center': gt_center,
                    }
                    
                    if matched_pred is not None:
                        # True Positive - matched_pred is just a frame index
                        analysis['pred_center'] = matched_pred
                        analysis['detection_error'] = dist
                        true_positives.append(analysis)
                    else:
                        # False Negative
                        # Check if predicted as cut instead
                        matched_cut, cut_dist = find_nearest_prediction(
                            pred_trans['cuts'], gt_center, tolerance
                        )
                        if matched_cut is not None:
                            analysis['misclassified_as'] = 'cut'
                            analysis['misclass_distance'] = cut_dist
                            analysis['pred_center'] = matched_cut
                        else:
                            analysis['misclassified_as'] = 'none'
                        false_negatives.append(analysis)
                
                # Analyze predicted graduals that are false positives
                # Match against all ground truth centers
                gt_centers = [g['center'] for g in gt_trans['graduals']]
                all_gt_centers = [t['center'] for t in gt_trans['cuts'] + gt_trans['graduals']]
                
                for pred_grad in pred_trans['graduals']:
                    # pred_grad is just a frame index
                    pred_center = pred_grad
                    
                    # Check if it matches any GT gradual
                    matched = False
                    for gt_center in gt_centers:
                        if abs(pred_center - gt_center) <= tolerance:
                            matched = True
                            break
                    
                    if not matched:
                        # This is a false positive
                        # Use a window around the predicted center for analysis (±5 frames)
                        window_size = 10
                        window_start = max(0, pred_center - window_size // 2)
                        window_end = min(len(frame_seq), pred_center + window_size // 2)
                        
                        frame_chars = analyze_frame_characteristics(
                            frame_seq, window_start, window_end
                        )
                        signal_chars = analyze_boundary_signal(
                            boundary_prob_seq, type_logit_seq,
                            window_start, window_end
                        )
                        
                        # Find what it actually is in ground truth
                        actual_label = 'none'
                        if all_gt_centers:
                            distances = [abs(pred_center - gc) for gc in all_gt_centers]
                            min_dist = min(distances)
                            if min_dist <= tolerance * 2:
                                min_idx = np.argmin(distances)
                                actual_label = 'cut' if min_idx < len(gt_trans['cuts']) else 'far_gradual'
                        
                        analysis = {
                            **frame_chars,
                            **signal_chars,
                            'pred_center': pred_center,
                            'actual_label': actual_label,
                        }
                        false_positives.append(analysis)
    
    return {
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'true_positives': true_positives,
    }


def compute_statistics(data, name):
    """Compute statistics for a list of characteristic dicts."""
    if not data:
        return {}
    
    # Get all keys
    keys = set()
    for item in data:
        keys.update(item.keys())
    
    stats = {}
    for key in keys:
        values = [item[key] for item in data if key in item and isinstance(item[key], (int, float))]
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
            }
    
    return stats


def plot_comparison(fn_stats, fp_stats, tp_stats, save_dir):
    """Plot comparison of characteristics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Common characteristics to compare
    compare_keys = [
        'duration',
        'avg_frame_diff',
        'max_frame_diff',
        'diff_variance',
        'brightness_change',
        'max_boundary_prob',
        'mean_boundary_prob',
        'max_gradual_prob',
        'mean_gradual_prob',
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, key in enumerate(compare_keys):
        ax = axes[idx]
        
        # Get values
        fn_vals = fn_stats.get(key, {})
        fp_vals = fp_stats.get(key, {})
        tp_vals = tp_stats.get(key, {})
        
        categories = []
        means = []
        stds = []
        
        if fn_vals:
            categories.append('Missed\nGraduals')
            means.append(fn_vals['mean'])
            stds.append(fn_vals['std'])
        
        if fp_vals:
            categories.append('False\nGraduals')
            means.append(fp_vals['mean'])
            stds.append(fp_vals['std'])
        
        if tp_vals:
            categories.append('Detected\nGraduals')
            means.append(tp_vals['mean'])
            stds.append(tp_vals['std'])
        
        if categories:
            x = np.arange(len(categories))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['red', 'orange', 'green'][:len(categories)])
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_title(key.replace('_', ' ').title())
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'gradual_failure_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {save_dir / 'gradual_failure_comparison.png'}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=r'outputs\casbd\best_model.pt', help='Model checkpoint path')
    parser.add_argument('--cache_path', type=str, default=r'D:\grad\data\training_cache')
    parser.add_argument('--output_dir', type=str, default='outputs/gradual_failure_analysis')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=100, help='Max clips to analyze')
    parser.add_argument('--threshold', type=float, default=0.55)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Load full model object from checkpoint (avoids architecture mismatch)
    if 'model' in ckpt:
        model = ckpt['model']
        print(f"Loaded full model from checkpoint (epoch {ckpt.get('epoch', '?')})")
    else:
        # Fallback: create model and load state dict
        model = create_context_aware_detector(
            main_dim=ckpt.get('config', {}).get('main_dim', 384),
            num_blocks=ckpt.get('config', {}).get('num_blocks', 4),
            embed_dim=ckpt.get('config', {}).get('embed_dim', 128),
        )
        incompatible = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"WARNING: Incompatible keys (architecture changed)")
        print(f"Loaded state dict from checkpoint (epoch {ckpt.get('epoch', '?')})")
    
    model.to(device)
    model.eval()
    
    # Load validation data
    val_dataset = PredecodedDataset(
        args.cache_path, split='val',
        val_split_ratio=0.15,
        random_seed=42,
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    
    print(f"Validation set: {len(val_dataset)} clips")
    
    # Analyze
    print("\nAnalyzing gradual failures...")
    results = analyze_gradual_failures(
        model, val_loader, device,
        boundary_threshold=args.threshold,
        tolerance=3,
        max_samples=args.max_samples,
    )
    
    # Compute statistics
    fn_stats = compute_statistics(results['false_negatives'], 'False Negatives')
    fp_stats = compute_statistics(results['false_positives'], 'False Positives')
    tp_stats = compute_statistics(results['true_positives'], 'True Positives')
    
    # Print summary
    print("\n" + "="*80)
    print("GRADUAL FAILURE ANALYSIS")
    print("="*80)
    
    print(f"\nFalse Negatives (Missed Graduals): {len(results['false_negatives'])}")
    if results['false_negatives']:
        misclass_counts = defaultdict(int)
        for fn in results['false_negatives']:
            misclass_counts[fn.get('misclassified_as', 'unknown')] += 1
        print("  Misclassified as:")
        for label, count in misclass_counts.items():
            print(f"    {label}: {count}")
    
    print(f"\nFalse Positives (Wrong Gradual Predictions): {len(results['false_positives'])}")
    if results['false_positives']:
        actual_counts = defaultdict(int)
        for fp in results['false_positives']:
            actual_counts[fp.get('actual_label', 'unknown')] += 1
        print("  Actually:")
        for label, count in actual_counts.items():
            print(f"    {label}: {count}")
    
    print(f"\nTrue Positives (Correct Graduals): {len(results['true_positives'])}")
    
    # Key characteristic comparisons
    print("\n" + "="*80)
    print("KEY CHARACTERISTICS COMPARISON")
    print("="*80)
    
    compare_keys = [
        ('duration', 'Duration (frames)'),
        ('avg_frame_diff', 'Avg Frame Difference'),
        ('max_boundary_prob', 'Max Boundary Probability'),
        ('mean_gradual_prob', 'Mean Gradual Type Probability'),
    ]
    
    for key, label in compare_keys:
        print(f"\n{label}:")
        if key in fn_stats:
            print(f"  Missed Graduals:    {fn_stats[key]['mean']:.3f} ± {fn_stats[key]['std']:.3f}")
        if key in fp_stats:
            print(f"  False Graduals:     {fp_stats[key]['mean']:.3f} ± {fp_stats[key]['std']:.3f}")
        if key in tp_stats:
            print(f"  Detected Graduals:  {tp_stats[key]['mean']:.3f} ± {tp_stats[key]['std']:.3f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        'config': {
            'checkpoint': args.checkpoint,
            'threshold': args.threshold,
            'samples_analyzed': args.max_samples,
        },
        'counts': {
            'false_negatives': len(results['false_negatives']),
            'false_positives': len(results['false_positives']),
            'true_positives': len(results['true_positives']),
        },
        'statistics': {
            'false_negatives': fn_stats,
            'false_positives': fp_stats,
            'true_positives': tp_stats,
        },
        'raw_data': {
            'false_negatives': results['false_negatives'][:50],  # Save first 50
            'false_positives': results['false_positives'][:50],
            'true_positives': results['true_positives'][:20],
        }
    }
    
    output_file = output_dir / 'gradual_failure_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved detailed analysis to: {output_file}")
    
    # Plot comparison
    plot_comparison(fn_stats, fp_stats, tp_stats, output_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
