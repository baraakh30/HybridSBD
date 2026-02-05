"""
Test potential discriminative features on REAL samples
Compute features using multiprocessing, analyze if they actually help

CANDIDATE FEATURES TO TEST:
1. Spatial coherence: Do motion vectors point in same direction? (pans=yes, graduals=no)
2. Content emergence: Is new content appearing? (graduals=yes, pans=no)
3. Bidirectionality: Motion in both directions? (graduals=yes, pans=one direction)
4. Edge consistency: Do edges stay consistent? (pans=yes, graduals=no)
"""

import torch
import numpy as np
import json
import cv2
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_boundary_detector import PredecodedDataset


def compute_optical_flow_coherence(frames):
    """
    Compute spatial coherence of motion
    Pans: All pixels move in same direction (high coherence)
    Graduals: Pixels move randomly as two scenes blend (low coherence)
    
    Args:
        frames: [T, H, W, 3] numpy array, uint8
    Returns:
        float: coherence score [0,1], higher = more coherent motion
    """
    if len(frames) < 2:
        return 0.0
    
    # Convert to grayscale
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    
    flows = []
    for i in range(len(gray_frames) - 1):
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i+1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append(flow)
    
    if len(flows) == 0:
        return 0.0
    
    # Compute coherence: std of flow directions
    # Low std = coherent (all move same way)
    # High std = incoherent (random motion)
    all_angles = []
    for flow in flows:
        # Flow is [H, W, 2] (dx, dy)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Filter out low-magnitude vectors (noise)
        mask = mag > 0.5
        if mask.sum() > 0:
            all_angles.append(ang[mask])
    
    if len(all_angles) == 0:
        return 0.0
    
    all_angles = np.concatenate(all_angles)
    
    # Compute angular variance (circular)
    # Low variance = coherent, high variance = incoherent
    mean_angle = np.arctan2(np.sin(all_angles).mean(), np.cos(all_angles).mean())
    angular_diff = np.abs(all_angles - mean_angle)
    angular_diff = np.minimum(angular_diff, 2*np.pi - angular_diff)  # Wrap around
    
    coherence = 1.0 - (angular_diff.std() / np.pi)  # Normalize to [0,1]
    
    return float(np.clip(coherence, 0, 1))


def compute_content_emergence(frames):
    """
    Measure if NEW content is appearing (graduals) vs same content moving (pans)
    
    Strategy: Compare feature similarity between first half and second half
    - Graduals: Low similarity (new scene emerging)
    - Pans: High similarity (same scene, just moved)
    
    Args:
        frames: [T, H, W, 3] numpy array, uint8
    Returns:
        float: emergence score [0,1], higher = more new content
    """
    if len(frames) < 4:
        return 0.0
    
    mid = len(frames) // 2
    first_half = frames[:mid]
    second_half = frames[mid:]
    
    # Compute edge maps (edges are content-independent of position)
    def get_edges(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    first_edges = [get_edges(f) for f in first_half]
    second_edges = [get_edges(f) for f in second_half]
    
    # Compare edge patterns
    # If same content (pan), edges should be similar despite motion
    # If new content (gradual), edges should be different
    first_edge_mean = np.mean(first_edges, axis=0)
    second_edge_mean = np.mean(second_edges, axis=0)
    
    # Normalized cross-correlation
    correlation = np.corrcoef(first_edge_mean.flatten(), second_edge_mean.flatten())[0, 1]
    
    # Low correlation = new content emerging
    emergence = 1.0 - max(0, correlation)
    
    return float(np.clip(emergence, 0, 1))


def compute_bidirectionality(frames):
    """
    Check if motion is bidirectional (graduals) vs unidirectional (pans)
    
    Graduals: Some pixels fade out (move one way), some fade in (opposite way)
    Pans: All pixels move in same direction
    
    Args:
        frames: [T, H, W, 3] numpy array, uint8
    Returns:
        float: bidirectionality score [0,1], higher = more bidirectional
    """
    if len(frames) < 3:
        return 0.0
    
    # Compute brightness changes
    gray_frames = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
    
    # Compute temporal gradients
    temporal_grad = np.diff(gray_frames.astype(float), axis=0)  # [T-1, H, W]
    
    # Count pixels increasing vs decreasing
    increasing = (temporal_grad > 5).sum()
    decreasing = (temporal_grad < -5).sum()
    total = increasing + decreasing
    
    if total == 0:
        return 0.0
    
    # Balanced increase/decrease = bidirectional (gradual)
    # Imbalanced = unidirectional (pan or static)
    ratio = min(increasing, decreasing) / (max(increasing, decreasing) + 1e-8)
    
    return float(ratio)


def compute_edge_consistency(frames):
    """
    Measure if edge structure stays consistent (pans) vs changes (graduals)
    
    Pans: Edges move but stay the same edges
    Graduals: Edges disappear and new ones appear
    
    Args:
        frames: [T, H, W, 3] numpy array, uint8
    Returns:
        float: consistency score [0,1], higher = more consistent
    """
    if len(frames) < 2:
        return 1.0
    
    # Compute edge maps
    def get_edges(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges.astype(float) / 255.0
    
    edge_maps = [get_edges(f) for f in frames]
    
    # Compute temporal consistency of edges
    # If edges consistent (pan), adjacent frames have similar edge density
    # If edges changing (gradual), edge density varies
    edge_densities = [em.mean() for em in edge_maps]
    density_std = np.std(edge_densities)
    
    # Low std = consistent, high std = inconsistent
    consistency = np.exp(-density_std * 10)  # Exponential decay
    
    return float(np.clip(consistency, 0, 1))


def process_sample(sample_data, sample_idx):
    """
    Process a single sample and compute all features
    
    Args:
        sample_data: (frames_tensor, label, length)
        sample_idx: index for progress tracking
    Returns:
        dict with computed features
    """
    frames_tensor, label, length = sample_data
    
    # Convert to numpy for OpenCV
    # Tensor is [T, 3, H, W] float32 [0,1]
    frames_np = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, 3]
    frames_np = (frames_np * 255).astype(np.uint8)
    
    # Compute features
    try:
        coherence = compute_optical_flow_coherence(frames_np)
        emergence = compute_content_emergence(frames_np)
        bidirectionality = compute_bidirectionality(frames_np)
        edge_consistency = compute_edge_consistency(frames_np)
    except Exception as e:
        # If computation fails, return zeros
        coherence = 0.0
        emergence = 0.0
        bidirectionality = 0.0
        edge_consistency = 0.0
    
    return {
        'label': int(label),
        'length': int(length),
        'spatial_coherence': coherence,
        'content_emergence': emergence,
        'bidirectionality': bidirectionality,
        'edge_consistency': edge_consistency,
        'sample_idx': sample_idx
    }


def analyze_features(results):
    """
    Analyze computed features to see if they discriminate
    """
    print("\n" + "="*80)
    print("FEATURE DISCRIMINATION ANALYSIS")
    print("="*80)
    
    # Group by label
    none_samples = [r for r in results if r['label'] == 0]
    cut_samples = [r for r in results if r['label'] == 1]
    gradual_samples = [r for r in results if r['label'] == 2]
    
    features = ['spatial_coherence', 'content_emergence', 'bidirectionality', 'edge_consistency']
    
    for feat_name in features:
        print(f"\n{'='*80}")
        print(f"{feat_name.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        if len(none_samples) > 0:
            none_vals = np.array([s[feat_name] for s in none_samples])
            print(f"\nNONE frames ({len(none_samples)} samples):")
            print(f"  Mean: {none_vals.mean():.4f} ± {none_vals.std():.4f}")
            print(f"  Median: {np.median(none_vals):.4f}")
            print(f"  P5-P95: [{np.percentile(none_vals, 5):.4f}, {np.percentile(none_vals, 95):.4f}]")
        
        if len(cut_samples) > 0:
            cut_vals = np.array([s[feat_name] for s in cut_samples])
            print(f"\nCUT frames ({len(cut_samples)} samples):")
            print(f"  Mean: {cut_vals.mean():.4f} ± {cut_vals.std():.4f}")
            print(f"  Median: {np.median(cut_vals):.4f}")
            print(f"  P5-P95: [{np.percentile(cut_vals, 5):.4f}, {np.percentile(cut_vals, 95):.4f}]")
        
        if len(gradual_samples) > 0:
            gradual_vals = np.array([s[feat_name] for s in gradual_samples])
            print(f"\nGRADUAL frames ({len(gradual_samples)} samples):")
            print(f"  Mean: {gradual_vals.mean():.4f} ± {gradual_vals.std():.4f}")
            print(f"  Median: {np.median(gradual_vals):.4f}")
            print(f"  P5-P95: [{np.percentile(gradual_vals, 5):.4f}, {np.percentile(gradual_vals, 95):.4f}]")
        
        # Compute separation
        if len(gradual_samples) > 0 and len(none_samples) > 0:
            gradual_mean = gradual_vals.mean()
            none_mean = none_vals.mean()
            separation = abs(gradual_mean - none_mean)
            
            print(f"\n🎯 SEPARATION: {separation:.4f}")
            if separation > 0.2:
                print(f"   ✅ STRONG discriminator! (>{0.2:.2f})")
            elif separation > 0.1:
                print(f"   ⚠️  Moderate discriminator ({0.1:.2f}-{0.2:.2f})")
            else:
                print(f"   ❌ Weak discriminator (<{0.1:.2f})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='data/training_cache', help='Cache directory')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples to analyze')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--output', default='feature_discrimination_analysis.json', help='Output file')
    args = parser.parse_args()
    
    print("Loading validation dataset...")
    dataset = PredecodedDataset(
        args.cache,
        split='val',
        val_split_ratio=0.15,
        random_seed=42
    )
    
    # Sample frames from different categories
    print("Collecting samples from each category...")
    none_samples = []
    cut_samples = []
    gradual_samples = []
    
    for i in range(len(dataset)):
        frames, labels, lengths = dataset[i]
        
        # Get center frame info
        center_idx = frames.shape[0] // 2
        label = labels[center_idx]
        length = lengths[center_idx]
        
        sample = (frames, label, length)
        
        if label == 0 and len(none_samples) < args.samples // 3:
            none_samples.append((sample, len(none_samples)))
        elif label == 1 and len(cut_samples) < args.samples // 3:
            cut_samples.append((sample, len(cut_samples)))
        elif label == 2 and len(gradual_samples) < args.samples // 3:
            gradual_samples.append((sample, len(gradual_samples)))
        
        if len(none_samples) >= args.samples // 3 and \
           len(cut_samples) >= args.samples // 3 and \
           len(gradual_samples) >= args.samples // 3:
            break
    
    all_samples = none_samples + cut_samples + gradual_samples
    print(f"\nCollected {len(all_samples)} samples:")
    print(f"  None: {len(none_samples)}")
    print(f"  Cut: {len(cut_samples)}")
    print(f"  Gradual: {len(gradual_samples)}")
    
    # Process samples with multiprocessing
    print(f"\nComputing features with {args.workers} workers...")
    print("(This may take a few minutes - optical flow is expensive)")
    
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(
            pool.starmap(process_sample, all_samples),
            total=len(all_samples),
            desc="Processing"
        ))
    
    # Analyze results
    analyze_features(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Check which features are useful
    useful_features = []
    
    gradual_samples_data = [r for r in results if r['label'] == 2]
    none_samples_data = [r for r in results if r['label'] == 0]
    
    features = ['spatial_coherence', 'content_emergence', 'bidirectionality', 'edge_consistency']
    
    for feat_name in features:
        if len(gradual_samples_data) > 0 and len(none_samples_data) > 0:
            gradual_mean = np.mean([s[feat_name] for s in gradual_samples_data])
            none_mean = np.mean([s[feat_name] for s in none_samples_data])
            separation = abs(gradual_mean - none_mean)
            
            if separation > 0.1:  # Threshold for usefulness
                useful_features.append((feat_name, separation))
    
    if len(useful_features) == 0:
        print("\n❌ NONE of the tested features provide strong discrimination!")
        print("   The problem may not be solvable with hand-crafted features.")
        print("   Consider: data augmentation, collecting more diverse graduals,")
        print("   or accepting the current precision as inherent dataset limitation.")
    else:
        print(f"\n✅ Found {len(useful_features)} useful feature(s):")
        for feat_name, sep in sorted(useful_features, key=lambda x: x[1], reverse=True):
            print(f"   - {feat_name}: separation={sep:.4f}")
        print("\n   Recommend implementing these in ReasoningModule!")


if __name__ == '__main__':
    main()
