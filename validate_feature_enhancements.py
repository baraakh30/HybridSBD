"""
Computational Validation of Feature Enhancements

This script validates whether the proposed enhancements actually help:
1. Multi-distance frame differences (1, 4, 8, 16)
2. Histogram evolution tracking
3. Memory and computational cost

We'll analyze REAL data to see if these features are discriminative.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_boundary_detector import PredecodedDataset
from detection.hybrid_boundary_detector import ColorHistogramEncoder


def compute_frame_differences(frames, distances=[1, 4, 8, 16]):
    """
    Compute frame differences at multiple distances
    
    Args:
        frames: [T, C, H, W] tensor
        distances: List of distances to compute
        
    Returns:
        Dict of {distance: [T] mean absolute difference per frame}
    """
    T, C, H, W = frames.shape
    diffs = {}
    
    for dist in distances:
        if dist >= T:
            diffs[dist] = torch.zeros(T)
            continue
            
        # Compute absolute difference
        frame_diff = torch.zeros(T)
        frame_diff[dist:] = torch.abs(frames[dist:] - frames[:-dist]).mean(dim=(1, 2, 3))
        diffs[dist] = frame_diff
    
    return diffs


def compute_histogram_evolution(frames, num_bins=32):
    """
    Compute color histogram evolution over time
    
    Args:
        frames: [T, C, H, W] tensor in [0, 1]
        
    Returns:
        [T, num_bins*3] histogram per frame
        [T] histogram change magnitude from previous frame
    """
    T, C, H, W = frames.shape
    histograms = []
    
    for t in range(T):
        frame = frames[t]  # [C, H, W]
        
        # Compute RGB histograms
        hist = []
        for c in range(3):
            channel = frame[c].flatten()
            h = torch.histc(channel, bins=num_bins, min=0.0, max=1.0)
            h = h / (H * W)  # Normalize
            hist.append(h)
        
        histograms.append(torch.cat(hist))
    
    histograms = torch.stack(histograms)  # [T, num_bins*3]
    
    # Compute histogram changes
    hist_changes = torch.zeros(T)
    hist_changes[1:] = torch.abs(histograms[1:] - histograms[:-1]).mean(dim=1)
    
    return histograms, hist_changes


def analyze_transition_features(frames, labels, transition_type, transition_length):
    """
    Analyze feature discriminative power for a single clip
    
    Args:
        frames: [T, C, H, W]
        labels: [T] - 0=none, 1=cut, 2=gradual
        transition_type: 'cut' or 'gradual'
        transition_length: int
        
    Returns:
        Dict of feature statistics
    """
    T = len(frames)
    
    # 1. Multi-distance differences
    diffs = compute_frame_differences(frames, distances=[1, 4, 8, 16])
    
    # 2. Histogram evolution
    histograms, hist_changes = compute_histogram_evolution(frames)
    
    # Find transition frames
    transition_mask = labels > 0
    transition_indices = torch.where(transition_mask)[0]
    
    if len(transition_indices) == 0:
        return None
    
    # Transition center
    trans_start = transition_indices[0].item()
    trans_end = transition_indices[-1].item()
    trans_center = (trans_start + trans_end) // 2
    
    # Extract features at transition center (±5 frame window)
    window = 5
    start_idx = max(0, trans_center - window)
    end_idx = min(T, trans_center + window + 1)
    
    # Background features (non-transition regions)
    bg_mask = labels == 0
    if bg_mask.sum() == 0:
        return None
    
    features = {
        'type': transition_type,
        'length': transition_length,
        'transition_center': trans_center,
        
        # Multi-distance differences (at transition vs background)
        'diff_1_trans': diffs[1][start_idx:end_idx].mean().item(),
        'diff_4_trans': diffs[4][start_idx:end_idx].mean().item(),
        'diff_8_trans': diffs[8][start_idx:end_idx].mean().item(),
        'diff_16_trans': diffs[16][start_idx:end_idx].mean().item(),
        
        'diff_1_bg': diffs[1][bg_mask].mean().item(),
        'diff_4_bg': diffs[4][bg_mask].mean().item(),
        'diff_8_bg': diffs[8][bg_mask].mean().item(),
        'diff_16_bg': diffs[16][bg_mask].mean().item(),
        
        # Histogram changes
        'hist_change_trans': hist_changes[start_idx:end_idx].mean().item(),
        'hist_change_bg': hist_changes[bg_mask].mean().item(),
    }
    
    # Compute signal-to-noise ratio (transition / background)
    for dist in [1, 4, 8, 16]:
        snr = features[f'diff_{dist}_trans'] / (features[f'diff_{dist}_bg'] + 1e-8)
        features[f'snr_{dist}'] = snr
    
    features['hist_snr'] = features['hist_change_trans'] / (features['hist_change_bg'] + 1e-8)
    
    return features


def analyze_dataset(cache_path, num_samples=2000):
    """
    Analyze feature discriminative power across dataset
    STRATIFIED SAMPLING: Ensure we get long gradual transitions!
    
    Returns:
        Dict with analysis results
    """
    print("\n" + "="*80)
    print("FEATURE DISCRIMINATIVE POWER ANALYSIS (STRATIFIED SAMPLING)")
    print("="*80)
    
    # Load dataset
    dataset = PredecodedDataset(
        cache_path,
        split='train',
        val_split_ratio=0.15,
        random_seed=42
    )
    
    print(f"\nLoaded {len(dataset)} clips")
    print(f"Pre-scanning dataset to find gradual transitions...")
    
    # FIRST PASS: Find all gradual transition indices by length
    gradual_indices = {
        'short': [],   # <10 frames
        'medium': [],  # 10-20 frames
        'long': [],    # 20-40 frames
        'very_long': [] # 40+ frames
    }
    cut_indices = []
    
    # Scan first 10000 clips to find transitions
    for idx in tqdm(range(min(10000, len(dataset))), desc="Pre-scanning"):
        frames, labels, trans_lengths = dataset[idx]
        
        has_cut = (labels == 1).any()
        has_gradual = (labels == 2).any()
        
        if has_cut:
            cut_indices.append(idx)
        elif has_gradual:
            # Get gradual length
            trans_length = trans_lengths[labels == 2][0].item() if (labels == 2).sum() > 0 else 0
            
            if trans_length < 10:
                gradual_indices['short'].append((idx, trans_length))
            elif trans_length < 20:
                gradual_indices['medium'].append((idx, trans_length))
            elif trans_length < 40:
                gradual_indices['long'].append((idx, trans_length))
            else:
                gradual_indices['very_long'].append((idx, trans_length))
    
    print(f"\n📊 Found transitions:")
    print(f"  Cuts: {len(cut_indices)}")
    print(f"  Short graduals (<10f): {len(gradual_indices['short'])}")
    print(f"  Medium graduals (10-20f): {len(gradual_indices['medium'])}")
    print(f"  Long graduals (20-40f): {len(gradual_indices['long'])}")
    print(f"  Very long graduals (40+f): {len(gradual_indices['very_long'])}")
    
    # STRATIFIED SAMPLING: Get balanced samples
    print(f"\nStratified sampling to ensure long gradual coverage...")
    
    # Sample cuts
    num_cuts = min(500, len(cut_indices))
    sampled_cut_indices = np.random.choice(cut_indices, size=num_cuts, replace=False).tolist()
    
    # Sample graduals (prioritize long ones!)
    sampled_gradual_indices = []
    
    # Get ALL very long graduals (they're rare!)
    sampled_gradual_indices.extend(gradual_indices['very_long'])
    
    # Get ALL long graduals
    sampled_gradual_indices.extend(gradual_indices['long'])
    
    # Get medium graduals (up to 100)
    if len(gradual_indices['medium']) > 0:
        n_medium = min(100, len(gradual_indices['medium']))
        medium_sample = np.random.choice(len(gradual_indices['medium']), size=n_medium, replace=False)
        sampled_gradual_indices.extend([gradual_indices['medium'][i] for i in medium_sample])
    
    # Get short graduals (up to 100)
    if len(gradual_indices['short']) > 0:
        n_short = min(100, len(gradual_indices['short']))
        short_sample = np.random.choice(len(gradual_indices['short']), size=n_short, replace=False)
        sampled_gradual_indices.extend([gradual_indices['short'][i] for i in short_sample])
    
    print(f"  Sampled {num_cuts} cuts")
    print(f"  Sampled {len(sampled_gradual_indices)} graduals (prioritizing long ones)")
    
    # SECOND PASS: Analyze sampled clips
    cut_features = []
    gradual_features = []
    
    print(f"\nAnalyzing sampled transitions...")
    
    # Analyze cuts
    for idx in tqdm(sampled_cut_indices, desc="Analyzing cuts"):
        frames, labels, trans_lengths = dataset[idx]
        trans_length = trans_lengths[labels == 1][0].item() if (labels == 1).sum() > 0 else 0
        feat = analyze_transition_features(frames, labels, 'cut', trans_length)
        if feat is not None:
            cut_features.append(feat)
    
    # Analyze graduals
    for idx, trans_length in tqdm(sampled_gradual_indices, desc="Analyzing graduals"):
        frames, labels, trans_lengths = dataset[idx]
        feat = analyze_transition_features(frames, labels, 'gradual', trans_length)
        if feat is not None:
            gradual_features.append(feat)
    
    print(f"\n✓ Analyzed {len(cut_features)} cuts, {len(gradual_features)} gradual transitions")
    
    # Breakdown
    short = len([f for f in gradual_features if f['length'] < 10])
    medium = len([f for f in gradual_features if 10 <= f['length'] < 20])
    long = len([f for f in gradual_features if 20 <= f['length'] < 40])
    very_long = len([f for f in gradual_features if f['length'] >= 40])
    
    print(f"  Gradual breakdown:")
    print(f"    Short (<10f): {short}")
    print(f"    Medium (10-20f): {medium}")
    print(f"    Long (20-40f): {long}")
    print(f"    Very long (40+f): {very_long}")
    
    return {
        'cuts': cut_features,
        'graduals': gradual_features
    }


def plot_feature_analysis(results, output_dir):
    """Generate comprehensive feature analysis plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    cuts = results['cuts']
    graduals = results['graduals']
    
    # Separate graduals by length
    short_graduals = [f for f in graduals if f['length'] < 10]
    medium_graduals = [f for f in graduals if 10 <= f['length'] < 20]
    long_graduals = [f for f in graduals if 20 <= f['length'] < 40]
    very_long_graduals = [f for f in graduals if f['length'] >= 40]
    
    print(f"\nGradual breakdown:")
    print(f"  Short (<10 frames): {len(short_graduals)}")
    print(f"  Medium (10-20 frames): {len(medium_graduals)}")
    print(f"  Long (20-40 frames): {len(long_graduals)}")
    print(f"  Very long (40+ frames): {len(very_long_graduals)}")
    
    # Figure 1: Multi-distance difference SNR comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    distances = [1, 4, 8, 16]
    for i, dist in enumerate(distances):
        ax = axes[i // 3, i % 3]
        
        snr_key = f'snr_{dist}'
        
        cut_snrs = [f[snr_key] for f in cuts if snr_key in f]
        short_snrs = [f[snr_key] for f in short_graduals if snr_key in f]
        medium_snrs = [f[snr_key] for f in medium_graduals if snr_key in f]
        long_snrs = [f[snr_key] for f in long_graduals if snr_key in f]
        very_long_snrs = [f[snr_key] for f in very_long_graduals if snr_key in f]
        
        data = [cut_snrs, short_snrs, medium_snrs, long_snrs, very_long_snrs]
        labels = ['Cuts', 'Short\n(<10f)', 'Medium\n(10-20f)', 'Long\n(20-40f)', 'Very Long\n(40+f)']
        
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        
        # Color boxes
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Signal-to-Noise Ratio', fontsize=11)
        ax.set_title(f'Distance-{dist} Frame Difference SNR', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No signal (SNR=1)')
        
        # Add median values
        medians = [np.median(d) if len(d) > 0 else 0 for d in data]
        for j, median in enumerate(medians):
            ax.text(j + 1, median, f'{median:.2f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Histogram evolution in the 5th subplot
    ax = axes[1, 1]
    
    cut_hist_snrs = [f['hist_snr'] for f in cuts if 'hist_snr' in f]
    short_hist_snrs = [f['hist_snr'] for f in short_graduals if 'hist_snr' in f]
    medium_hist_snrs = [f['hist_snr'] for f in medium_graduals if 'hist_snr' in f]
    long_hist_snrs = [f['hist_snr'] for f in long_graduals if 'hist_snr' in f]
    very_long_hist_snrs = [f['hist_snr'] for f in very_long_graduals if 'hist_snr' in f]
    
    data = [cut_hist_snrs, short_hist_snrs, medium_hist_snrs, long_hist_snrs, very_long_hist_snrs]
    labels = ['Cuts', 'Short\n(<10f)', 'Medium\n(10-20f)', 'Long\n(20-40f)', 'Very Long\n(40+f)']
    
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Signal-to-Noise Ratio', fontsize=11)
    ax.set_title('Histogram Evolution SNR', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    
    # Add median values
    medians = [np.median(d) if len(d) > 0 else 0 for d in data]
    for j, median in enumerate(medians):
        ax.text(j + 1, median, f'{median:.2f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.suptitle('Multi-Distance Frame Difference: Signal Strength by Transition Type', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multidistance_snr_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved multi-distance SNR analysis")
    
    # Figure 2: Best feature per gradual length (keep existing)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group graduals by length bins
    length_bins = list(range(2, 50, 3))  # 2-5, 5-8, 8-11, ...
    best_features_by_length = {dist: [] for dist in distances}
    best_features_by_length['hist'] = []
    bin_centers = []
    
    for bin_start in length_bins:
        bin_end = bin_start + 3
        bin_graduals = [f for f in graduals if bin_start <= f['length'] < bin_end]
        
        if len(bin_graduals) < 5:  # Need enough samples
            continue
        
        bin_centers.append(bin_start + 1.5)
        
        # Get median SNR for each feature
        for dist in distances:
            snrs = [f[f'snr_{dist}'] for f in bin_graduals if f'snr_{dist}' in f]
            best_features_by_length[dist].append(np.median(snrs) if snrs else 0)
        
        hist_snrs = [f['hist_snr'] for f in bin_graduals if 'hist_snr' in f]
        best_features_by_length['hist'].append(np.median(hist_snrs) if hist_snrs else 0)
    
    # Plot lines
    ax.plot(bin_centers, best_features_by_length[1], 'o-', label='Diff-1 (consecutive)', linewidth=2, markersize=6)
    ax.plot(bin_centers, best_features_by_length[4], 's-', label='Diff-4', linewidth=2, markersize=6)
    ax.plot(bin_centers, best_features_by_length[8], '^-', label='Diff-8', linewidth=2, markersize=6)
    ax.plot(bin_centers, best_features_by_length[16], 'd-', label='Diff-16 (long range)', linewidth=2, markersize=6)
    ax.plot(bin_centers, best_features_by_length['hist'], '*-', label='Histogram Evolution', linewidth=2, markersize=8)
    
    ax.set_xlabel('Gradual Transition Length (frames)', fontsize=12)
    ax.set_ylabel('Median Signal-to-Noise Ratio', fontsize=12)
    ax.set_title('Which Features Work Best for Different Gradual Lengths?', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='No signal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_feature_by_length.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved best feature by length analysis")


def memory_analysis():
    """Analyze memory impact of enhancements"""
    print("\n" + "="*80)
    print("MEMORY & COMPUTATIONAL COST ANALYSIS")
    print("="*80)
    
    B, T, C, H, W = 2, 128, 3, 48, 64
    
    # Old approach: Single distance-1 difference
    print("\nOLD APPROACH (single distance-1 difference):")
    print(f"  Input: [{B}, {T}, {C}, {H}, {W}]")
    print(f"  Memory: {B * T * C * H * W * 4 / 1024**2:.2f} MB (fp32)")
    print(f"  Diff computation: 1 distance → {B * T * C * H * W * 4 / 1024**2:.2f} MB")
    print(f"  Total: {2 * B * T * C * H * W * 4 / 1024**2:.2f} MB")
    
    # New approach: Multi-distance differences
    print("\nNEW APPROACH (multi-distance: 1, 4, 8, 16):")
    print(f"  Input: [{B}, {T}, {C}, {H}, {W}]")
    print(f"  Memory: {B * T * C * H * W * 4 / 1024**2:.2f} MB (fp32)")
    print(f"  Diff computation: 4 distances → {4 * B * T * C * H * W * 4 / 1024**2:.2f} MB")
    print(f"  Histogram diff: {B * T * C * H * W * 4 / 1024**2:.2f} MB")
    print(f"  Total: {6 * B * T * C * H * W * 4 / 1024**2:.2f} MB")
    
    # Increase
    old_mem = 2 * B * T * C * H * W * 4 / 1024**2
    new_mem = 6 * B * T * C * H * W * 4 / 1024**2
    increase = ((new_mem - old_mem) / old_mem) * 100
    
    print(f"\n  Memory increase: {increase:.1f}%")
    print(f"  Absolute increase: {new_mem - old_mem:.2f} MB per batch")
    
    # Compute time estimate
    print(f"\n  Estimated compute overhead:")
    print(f"    4x difference computations: ~4x slower preprocessing")
    print(f"    BUT: Differences are cheap (just subtraction)")
    print(f"    Expected total overhead: ~10-15% training time")
    
    # Feature projection change
    print(f"\n  Feature projection:")
    hidden_dim = 192
    print(f"    Old: Linear({hidden_dim * 3} → {hidden_dim})")
    print(f"    New: Linear({hidden_dim * 4} → {hidden_dim})")
    old_params = hidden_dim * 3 * hidden_dim
    new_params = hidden_dim * 4 * hidden_dim
    print(f"    Parameter increase: {new_params - old_params:,} ({((new_params - old_params) / old_params) * 100:.1f}%)")
    
    return {
        'memory_increase_pct': increase,
        'memory_increase_mb': new_mem - old_mem,
        'param_increase': new_params - old_params
    }


def main():
    cache_path = r"D:\grad\data\training_cache"
    output_dir = "outputs/hybrid_detector/feature_validation"
    
    # 1. Memory analysis
    mem_results = memory_analysis()
    
    # 2. Feature discriminative power analysis
    results = analyze_dataset(cache_path, num_samples=1000)
    
    # 3. Generate plots
    print("\n" + "="*80)
    print("GENERATING ANALYSIS PLOTS")
    print("="*80)
    plot_feature_analysis(results, output_dir)
    
    # 4. Summary report
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n📊 MEMORY & COMPUTE:")
    print(f"  Memory increase: {mem_results['memory_increase_pct']:.1f}% (+{mem_results['memory_increase_mb']:.2f} MB/batch)")
    print(f"  Parameter increase: {mem_results['param_increase']:,} params")
    print(f"  Expected training slowdown: ~10-15%")
    
    # Compute average SNRs
    cuts = results['cuts']
    graduals = results['graduals']
    long_graduals = [f for f in graduals if f['length'] >= 20]
    very_long_graduals = [f for f in graduals if f['length'] >= 40]
    
    print(f"\n📈 FEATURE EFFECTIVENESS (median SNR):")
    
    for dist in [1, 4, 8, 16]:
        cut_snr = np.median([f[f'snr_{dist}'] for f in cuts if f'snr_{dist}' in f])
        grad_snr = np.median([f[f'snr_{dist}'] for f in graduals if f'snr_{dist}' in f])
        long_snr = np.median([f[f'snr_{dist}'] for f in long_graduals if f'snr_{dist}' in f]) if long_graduals else 0
        vlong_snr = np.median([f[f'snr_{dist}'] for f in very_long_graduals if f'snr_{dist}' in f]) if very_long_graduals else 0
        
        print(f"\n  Distance-{dist} frame difference:")
        print(f"    Cuts: {cut_snr:.2f}x signal")
        print(f"    All graduals: {grad_snr:.2f}x signal")
        print(f"    Long graduals (20+f): {long_snr:.2f}x signal ({len(long_graduals)} samples)")
        print(f"    Very long graduals (40+f): {vlong_snr:.2f}x signal ({len(very_long_graduals)} samples)")
    
    cut_hist_snr = np.median([f['hist_snr'] for f in cuts if 'hist_snr' in f])
    grad_hist_snr = np.median([f['hist_snr'] for f in graduals if 'hist_snr' in f])
    long_hist_snr = np.median([f['hist_snr'] for f in long_graduals if 'hist_snr' in f]) if long_graduals else 0
    vlong_hist_snr = np.median([f['hist_snr'] for f in very_long_graduals if 'hist_snr' in f]) if very_long_graduals else 0
    
    print(f"\n  Histogram evolution:")
    print(f"    Cuts: {cut_hist_snr:.2f}x signal")
    print(f"    All graduals: {grad_hist_snr:.2f}x signal")
    print(f"    Long graduals (20+f): {long_hist_snr:.2f}x signal ({len(long_graduals)} samples)")
    print(f"    Very long graduals (40+f): {vlong_hist_snr:.2f}x signal ({len(very_long_graduals)} samples)")
    
    print(f"\n✅ RECOMMENDATION:")
    
    # Decision logic - need data from long graduals!
    if len(long_graduals) == 0:
        print(f"  ⚠️  WARNING: No long gradual samples found!")
        print(f"     Cannot validate effectiveness - need more data")
        return
    
    # Check distance-16 for very long graduals
    if len(very_long_graduals) > 0 and vlong_snr > 1.3:
        print(f"  ✅ Distance-16 differences show {vlong_snr:.2f}x signal for very long graduals → VALUABLE")
    elif long_snr > 1.3:
        print(f"  ✅ Distance-16 differences show {long_snr:.2f}x signal for long graduals → VALUABLE")
    else:
        print(f"  ⚠️  Distance-16 differences show only {long_snr:.2f}x signal → WEAK")
    
    if len(very_long_graduals) > 0 and vlong_hist_snr > 1.3:
        print(f"  ✅ Histogram evolution shows {vlong_hist_snr:.2f}x signal for very long graduals → VALUABLE")
    elif long_hist_snr > 1.3:
        print(f"  ✅ Histogram evolution shows {long_hist_snr:.2f}x signal for long graduals → VALUABLE")
    else:
        print(f"  ⚠️  Histogram evolution shows only {long_hist_snr:.2f}x signal → WEAK")
    
    # Final verdict
    best_long_snr = max(long_snr, vlong_snr) if very_long_graduals else long_snr
    best_hist_snr = max(long_hist_snr, vlong_hist_snr) if very_long_graduals else long_hist_snr
    
    if best_long_snr > 1.3 or best_hist_snr > 1.3:
        print(f"\n  🎯 VERDICT: Enhancements PROVIDE MEASURABLE BENEFIT")
        print(f"     Best SNR for long graduals: {max(best_long_snr, best_hist_snr):.2f}x")
        print(f"     Trade-off: +{mem_results['memory_increase_pct']:.1f}% memory, ~10-15% slower")
        print(f"     → RECOMMENDED: Keep enhancements")
    else:
        print(f"\n  ⚠️  VERDICT: Enhancements show WEAK improvement")
        print(f"     Best SNR for long graduals: {max(best_long_snr, best_hist_snr):.2f}x")
        print(f"     → CONSIDER: Alternative approaches or accept the trade-off")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
