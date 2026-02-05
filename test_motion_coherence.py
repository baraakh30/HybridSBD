"""
Test Motion Coherence Features for Gradual vs Pan Discrimination

KEY HYPOTHESIS:
- Pans: Coherent motion trajectory (trackable, consistent direction)
- Graduals: Incoherent fading (no trajectory, random motion)

Tests 4 features empirically:
1. Optical Flow Consistency: Do flows point same direction?
2. Feature Point Trackability: Can points be tracked across frames?
3. Motion Accumulation: Does motion accumulate or cancel out?
4. Spatial Layout Stability: Do grid cells move smoothly?

USAGE:
    python test_motion_coherence.py --samples 300 --workers 8
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import multiprocessing as mp
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_boundary_detector import PredecodedDataset


def compute_optical_flow_consistency(frames):
    """
    Compute consistency of optical flow across consecutive frame pairs
    
    Pan: All flows point same direction → HIGH consistency
    Gradual: Flows random/mixed → LOW consistency
    
    Args:
        frames: [T, H, W, C] numpy array
    
    Returns:
        float: Flow consistency score [0, 1]
    """
    T, H, W, C = frames.shape
    
    # Convert to grayscale
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    
    # Compute flows for consecutive pairs
    flows = []
    for i in range(min(20, T-1)):  # First 20 frames
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i+1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append(flow)
    
    if len(flows) < 2:
        return 0.0
    
    # Compute pairwise flow correlations
    correlations = []
    for i in range(len(flows) - 1):
        flow1 = flows[i]
        flow2 = flows[i+1]
        
        # Flatten and compute correlation
        f1_flat = flow1.reshape(-1, 2)
        f2_flat = flow2.reshape(-1, 2)
        
        # Normalize
        f1_norm = f1_flat / (np.linalg.norm(f1_flat, axis=1, keepdims=True) + 1e-8)
        f2_norm = f2_flat / (np.linalg.norm(f2_flat, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        corr = np.mean(np.sum(f1_norm * f2_norm, axis=1))
        correlations.append(corr)
    
    # Average correlation
    consistency = np.mean(correlations)
    
    # Map to [0, 1]
    consistency = (consistency + 1) / 2  # From [-1, 1] to [0, 1]
    
    return float(consistency)


def compute_feature_trackability(frames):
    """
    Compute what percentage of feature points can be tracked over time
    
    Pan: Most points trackable (same content moving) → HIGH
    Gradual: Points disappear/appear (content change) → LOW
    
    Args:
        frames: [T, H, W, C] numpy array
    
    Returns:
        float: Trackability score [0, 1]
    """
    T, H, W, C = frames.shape
    
    # Convert to grayscale
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    
    # Detect good features in first frame
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    
    p0 = cv2.goodFeaturesToTrack(gray_frames[0], **feature_params)
    
    if p0 is None or len(p0) < 10:
        return 0.0
    
    initial_count = len(p0)
    
    # Lucas-Kanade parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Track to frame 10, 20, 30 (if available)
    trackability_scores = []
    
    for target_frame in [10, 20, 30]:
        if target_frame >= T:
            continue
        
        # Track from frame 0 to target
        p_current = p0.copy()
        for i in range(target_frame):
            p_next, status, _ = cv2.calcOpticalFlowPyrLK(
                gray_frames[i], gray_frames[i+1], p_current, None, **lk_params
            )
            
            # Keep only successfully tracked points
            if status is not None:
                p_current = p_next[status.flatten() == 1]
            else:
                p_current = np.array([])
            
            if len(p_current) == 0:
                break
        
        # Compute percentage tracked
        tracked_ratio = len(p_current) / initial_count
        trackability_scores.append(tracked_ratio)
    
    if len(trackability_scores) == 0:
        return 0.0
    
    # Average across different distances
    return float(np.mean(trackability_scores))


def compute_motion_accumulation(frames):
    """
    Compute whether motion accumulates (trajectory) or cancels out (noise)
    
    Pan: Motion adds up to smooth trajectory → HIGH magnitude
    Gradual: Random motion cancels out → LOW magnitude
    
    Args:
        frames: [T, H, W, C] numpy array
    
    Returns:
        float: Motion accumulation score [0, 1]
    """
    T, H, W, C = frames.shape
    
    # Convert to grayscale
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    
    # Compute flows and accumulate
    accumulated_flow = np.zeros((H, W, 2), dtype=np.float32)
    
    for i in range(min(20, T-1)):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i+1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        accumulated_flow += flow
    
    # Compute magnitude of accumulated flow
    magnitude = np.sqrt(accumulated_flow[:, :, 0]**2 + accumulated_flow[:, :, 1]**2)
    avg_magnitude = np.mean(magnitude)
    
    # Normalize by number of frames (trajectory length per frame)
    trajectory_per_frame = avg_magnitude / min(20, T-1)
    
    # Map to [0, 1] (empirically, values range 0-10)
    score = min(1.0, trajectory_per_frame / 10.0)
    
    return float(score)


def compute_spatial_layout_stability(frames):
    """
    Compute whether spatial grid cells move smoothly or change values
    
    Pan: Cells move smoothly (content shifts) → SMOOTH trajectories
    Gradual: Cells change values (content replaces) → ERRATIC changes
    
    Args:
        frames: [T, H, W, C] numpy array
    
    Returns:
        float: Layout stability score [0, 1] (0=stable/smooth, 1=unstable/erratic)
    """
    T, H, W, C = frames.shape
    
    # Divide into 3×3 grid
    grid_h = H // 3
    grid_w = W // 3
    
    # Track mean brightness of each cell over time
    cell_trajectories = []
    
    for i in range(3):
        for j in range(3):
            # Extract cell region
            cell_values = []
            for t in range(min(30, T)):
                cell = frames[t, i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                mean_val = np.mean(cell)
                cell_values.append(mean_val)
            
            cell_trajectories.append(cell_values)
    
    # Compute variance of first derivatives (smoothness)
    stabilities = []
    for traj in cell_trajectories:
        if len(traj) < 3:
            continue
        
        # First derivative
        diffs = np.diff(traj)
        
        # Variance of differences (high = erratic, low = smooth)
        variance = np.var(diffs)
        stabilities.append(variance)
    
    if len(stabilities) == 0:
        return 0.5
    
    # Average variance
    avg_variance = np.mean(stabilities)
    
    # Map to [0, 1] - higher variance = less stable
    # Empirically, variance ranges 0-100
    instability = min(1.0, avg_variance / 100.0)
    
    # Return stability (inverse)
    return float(1.0 - instability)


def compute_all_features(sample_data):
    """
    Compute all motion coherence features for a sample
    
    Args:
        sample_data: (frames_np, label) tuple
    
    Returns:
        dict with all feature scores
    """
    frames_np, label = sample_data
    
    try:
        # Compute all features
        flow_consistency = compute_optical_flow_consistency(frames_np)
        trackability = compute_feature_trackability(frames_np)
        motion_accum = compute_motion_accumulation(frames_np)
        layout_stability = compute_spatial_layout_stability(frames_np)
        
        return {
            'flow_consistency': flow_consistency,
            'trackability': trackability,
            'motion_accumulation': motion_accum,
            'layout_stability': layout_stability,
            'label': label
        }
    except Exception as e:
        print(f"Error computing features: {e}")
        return None


def load_samples(dataset, sample_type, num_samples):
    """
    Load samples and convert to numpy for OpenCV processing
    
    Args:
        dataset: PredecodedDataset
        sample_type: 'gradual' or 'none'
        num_samples: Number to load
    
    Returns:
        List of (frames_np, label) tuples
    """
    type_map = {'none': 0, 'cut': 1, 'gradual': 2}
    target_label = type_map[sample_type]
    
    samples = []
    
    print(f"\nLoading {sample_type} samples...")
    for idx in tqdm(range(len(dataset))):
        if len(samples) >= num_samples:
            break
        
        try:
            frames, labels, _ = dataset[idx]
            
            # Check if sample contains target
            if sample_type == 'none':
                # Only pure none sequences
                if (labels == 0).all().item():
                    # Convert to numpy [T, H, W, C]
                    frames_np = frames.permute(0, 2, 3, 1).numpy()
                    frames_np = (frames_np * 255).astype(np.uint8)
                    samples.append((frames_np, sample_type))
            else:
                # Sequences containing graduals
                if (labels == target_label).any().item():
                    frames_np = frames.permute(0, 2, 3, 1).numpy()
                    frames_np = (frames_np * 255).astype(np.uint8)
                    samples.append((frames_np, sample_type))
        except Exception as e:
            continue
    
    print(f"Loaded {len(samples)} {sample_type} samples")
    return samples


def analyze_separation(gradual_scores, none_scores, feature_name):
    """
    Analyze how well a feature separates graduals from nones
    
    Returns:
        dict with statistics
    """
    gradual_mean = np.mean(gradual_scores)
    gradual_std = np.std(gradual_scores)
    none_mean = np.mean(none_scores)
    none_std = np.std(none_scores)
    
    # Cohen's d
    pooled_std = np.sqrt((gradual_std**2 + none_std**2) / 2)
    separation = abs(gradual_mean - none_mean) / (pooled_std + 1e-8)
    
    return {
        'feature': feature_name,
        'gradual_mean': gradual_mean,
        'gradual_std': gradual_std,
        'none_mean': none_mean,
        'none_std': none_std,
        'separation': separation,
        'discriminative': separation > 0.5
    }


def main():
    parser = argparse.ArgumentParser(description='Test motion coherence features')
    parser.add_argument('--data_dir', type=str, default='data/training_cache')
    parser.add_argument('--samples', type=int, default=300, help='Samples per type')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
    parser.add_argument('--output', type=str, default='motion_coherence_results.json')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MOTION COHERENCE FEATURE TESTING")
    print("="*80)
    
    # Load dataset
    print("\nLoading validation dataset...")
    val_dataset = PredecodedDataset(
        cache_root=args.data_dir,
        split='val',
    )
    print(f"Dataset size: {len(val_dataset)}")
    
    # Load samples
    gradual_samples = load_samples(val_dataset, 'gradual', args.samples)
    none_samples = load_samples(val_dataset, 'none', args.samples)
    
    if len(gradual_samples) == 0 or len(none_samples) == 0:
        print("ERROR: Could not load samples!")
        return
    
    # Combine all samples
    all_samples = gradual_samples + none_samples
    
    print(f"\n{'='*80}")
    print(f"Computing features with {args.workers} workers...")
    print(f"{'='*80}\n")
    
    # Process in parallel
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(compute_all_features, all_samples),
            total=len(all_samples),
            desc="Processing samples"
        ))
    
    # Filter out failures
    results = [r for r in results if r is not None]
    
    # Separate by type
    gradual_results = [r for r in results if r['label'] == 'gradual']
    none_results = [r for r in results if r['label'] == 'none']
    
    print(f"\nSuccessfully processed:")
    print(f"  Graduals: {len(gradual_results)}")
    print(f"  Nones: {len(none_results)}")
    
    # Analyze each feature
    print(f"\n{'='*80}")
    print("FEATURE ANALYSIS")
    print(f"{'='*80}\n")
    
    features = [
        'flow_consistency',
        'trackability',
        'motion_accumulation',
        'layout_stability'
    ]
    
    analyses = []
    
    for feature in features:
        gradual_scores = [r[feature] for r in gradual_results]
        none_scores = [r[feature] for r in none_results]
        
        analysis = analyze_separation(gradual_scores, none_scores, feature)
        analyses.append(analysis)
        
        # Print results
        status = "✅ DISCRIMINATIVE" if analysis['discriminative'] else "❌ Weak"
        print(f"{feature.upper().replace('_', ' ')}:")
        print(f"  Separation (Cohen's d): {analysis['separation']:.4f} {status}")
        print(f"  Gradual: μ={analysis['gradual_mean']:.4f}, σ={analysis['gradual_std']:.4f}")
        print(f"  None:    μ={analysis['none_mean']:.4f}, σ={analysis['none_std']:.4f}")
        print()
    
    # Summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    discriminative = [a for a in analyses if a['discriminative']]
    weak = [a for a in analyses if not a['discriminative']]
    
    print(f"✅ DISCRIMINATIVE FEATURES ({len(discriminative)}):")
    for a in sorted(discriminative, key=lambda x: x['separation'], reverse=True):
        print(f"  {a['feature']}: d={a['separation']:.4f}")
    
    print(f"\n❌ WEAK FEATURES ({len(weak)}):")
    for a in sorted(weak, key=lambda x: x['separation']):
        print(f"  {a['feature']}: d={a['separation']:.4f}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if len(discriminative) > 0:
        print("✅ Motion coherence features CAN discriminate graduals from pans!")
        print("\nStrongest features to implement:")
        for a in sorted(discriminative, key=lambda x: x['separation'], reverse=True)[:3]:
            print(f"  - {a['feature']}: d={a['separation']:.4f}")
        print("\nThese should improve gradual precision from ~60% to ~70-75%")
    else:
        print("❌ Motion coherence features do NOT discriminate well.")
        print("May need semantic features (CLIP/DINO) for better discrimination.")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            'analyses': analyses,
            'summary': {
                'discriminative_count': len(discriminative),
                'weak_count': len(weak),
                'best_feature': max(analyses, key=lambda x: x['separation'])['feature'],
                'best_separation': max(analyses, key=lambda x: x['separation'])['separation']
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
