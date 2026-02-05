"""
Test COMBINED Low-Level Features for Gradual vs Pan Discrimination

HYPOTHESIS: Individual features weak, but combination strong!

Combines:
1. Motion coherence (4 features): flow_consistency, trackability, motion_accum, layout_stability
2. Content emergence: Edge correlation first/second half
3. Spatial patterns: Variance, contrast, edge patterns
4. Temporal patterns: Accumulation, monotonicity

Tests if LINEAR COMBINATION can discriminate better than individuals.

USAGE:
    python test_combined_features.py --samples 300 --workers 8
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_boundary_detector import PredecodedDataset
from test_motion_coherence import (
    compute_optical_flow_consistency,
    compute_feature_trackability,
    compute_motion_accumulation,
    compute_spatial_layout_stability
)


def compute_content_emergence(frames):
    """
    Content emergence: Edge correlation between first/second half
    
    Gradual: Different content → LOW correlation
    Pan: Same content → HIGH correlation
    
    Args:
        frames: [T, H, W, C] numpy array
    
    Returns:
        float: Edge correlation [0, 1]
    """
    T = frames.shape[0]
    
    # Split into first/second half
    mid = T // 2
    first_half = frames[:mid]
    second_half = frames[mid:]
    
    # Convert to grayscale
    gray_first = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in first_half]
    gray_second = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in second_half]
    
    # Compute edge maps (average across half)
    edges_first = []
    edges_second = []
    
    for g in gray_first:
        edges = cv2.Canny(g, 50, 150)
        edges_first.append(edges)
    
    for g in gray_second:
        edges = cv2.Canny(g, 50, 150)
        edges_second.append(edges)
    
    # Average edge maps
    avg_edges_first = np.mean(edges_first, axis=0)
    avg_edges_second = np.mean(edges_second, axis=0)
    
    # Flatten and compute correlation
    e1_flat = avg_edges_first.flatten()
    e2_flat = avg_edges_second.flatten()
    
    # Normalize
    e1_norm = e1_flat / (np.linalg.norm(e1_flat) + 1e-8)
    e2_norm = e2_flat / (np.linalg.norm(e2_flat) + 1e-8)
    
    # Correlation
    correlation = np.dot(e1_norm, e2_norm)
    
    # Map to [0, 1]
    correlation = (correlation + 1) / 2
    
    return float(correlation)


def compute_spatial_variance_pattern(frames):
    """
    Spatial variance over time
    
    Gradual: Variance changes (fading)
    Pan: Variance constant (same content)
    
    Args:
        frames: [T, H, W, C] numpy array
    
    Returns:
        float: Variance change score [0, 1]
    """
    T = frames.shape[0]
    
    # Compute variance per frame
    variances = []
    for t in range(T):
        var = np.var(frames[t])
        variances.append(var)
    
    # Compute variance of variances (how much does variance change?)
    variance_change = np.var(variances)
    
    # Normalize (empirically ranges 0-1000)
    score = min(1.0, variance_change / 1000.0)
    
    return float(score)


def compute_brightness_monotonicity(frames):
    """
    Brightness change monotonicity
    
    Gradual: Brightness changes monotonically (fade in/out)
    Pan: Brightness oscillates
    
    Args:
        frames: [T, H, W, C] numpy array
    
    Returns:
        float: Monotonicity score [0, 1]
    """
    T = frames.shape[0]
    
    # Compute mean brightness per frame
    brightness = []
    for t in range(T):
        b = np.mean(frames[t])
        brightness.append(b)
    
    # Check monotonicity
    diffs = np.diff(brightness)
    
    # Count sign changes (non-monotonic)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    
    # Monotonicity = inverse of sign changes
    max_changes = len(diffs) - 1
    monotonicity = 1.0 - (sign_changes / (max_changes + 1e-8))
    
    return float(monotonicity)


def compute_change_accumulation_ratio(frames):
    """
    Ratio of long-distance to short-distance changes
    
    Gradual: Changes accumulate (ratio > 1)
    Pan: Changes constant (ratio ≈ 1)
    
    Args:
        frames: [T, H, W, C] numpy array
    
    Returns:
        float: Accumulation ratio
    """
    T = frames.shape[0]
    
    # Compute differences at multiple distances
    diff_4 = np.mean(np.abs(frames[4:] - frames[:-4])) if T > 4 else 0
    diff_16 = np.mean(np.abs(frames[16:] - frames[:-16])) if T > 16 else 0
    
    # Ratio (should be ~4x for accumulating)
    ratio = diff_16 / (diff_4 * 4 + 1e-8)
    
    # Gradual: ratio ≈ 1.0 (linear accumulation)
    # Pan: ratio < 1.0 (sublinear)
    
    return float(ratio)


def compute_all_combined_features(sample_data):
    """
    Compute all features for a sample
    
    Returns:
        dict with feature vector and label
    """
    frames_np, label = sample_data
    
    try:
        # Motion coherence (4 features)
        flow_consistency = compute_optical_flow_consistency(frames_np)
        trackability = compute_feature_trackability(frames_np)
        motion_accum = compute_motion_accumulation(frames_np)
        layout_stability = compute_spatial_layout_stability(frames_np)
        
        # Content emergence (1 feature)
        content_emerge = compute_content_emergence(frames_np)
        
        # Spatial patterns (2 features)
        spatial_variance = compute_spatial_variance_pattern(frames_np)
        brightness_mono = compute_brightness_monotonicity(frames_np)
        
        # Temporal accumulation (1 feature)
        change_accum = compute_change_accumulation_ratio(frames_np)
        
        # Total: 9 features
        feature_vector = [
            flow_consistency,
            trackability,
            motion_accum,
            layout_stability,
            content_emerge,
            spatial_variance,
            brightness_mono,
            change_accum,
            # Interaction features (products)
            content_emerge * trackability,  # Content + motion
            spatial_variance * brightness_mono,  # Spatial patterns
            motion_accum * layout_stability,  # Motion patterns
        ]
        
        # Total: 11 features with interactions
        
        return {
            'features': feature_vector,
            'label': 1 if label == 'gradual' else 0  # 1=gradual, 0=none
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


def load_samples(dataset, sample_type, num_samples):
    """Load samples for testing"""
    type_map = {'none': 0, 'cut': 1, 'gradual': 2}
    target_label = type_map[sample_type]
    
    samples = []
    
    print(f"\nLoading {sample_type} samples...")
    for idx in tqdm(range(len(dataset))):
        if len(samples) >= num_samples:
            break
        
        try:
            frames, labels, _ = dataset[idx]
            
            if sample_type == 'none':
                if (labels == 0).all().item():
                    frames_np = frames.permute(0, 2, 3, 1).numpy()
                    frames_np = (frames_np * 255).astype(np.uint8)
                    samples.append((frames_np, sample_type))
            else:
                if (labels == target_label).any().item():
                    frames_np = frames.permute(0, 2, 3, 1).numpy()
                    frames_np = (frames_np * 255).astype(np.uint8)
                    samples.append((frames_np, sample_type))
        except:
            continue
    
    print(f"Loaded {len(samples)} {sample_type} samples")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/training_cache')
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=str, default='combined_features_results.json')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMBINED FEATURES TESTING")
    print("="*80)
    
    # Load dataset
    val_dataset = PredecodedDataset(
        cache_root=args.data_dir,
        split='val',
    )
    
    # Load samples
    gradual_samples = load_samples(val_dataset, 'gradual', args.samples)
    none_samples = load_samples(val_dataset, 'none', args.samples)
    
    if len(gradual_samples) == 0 or len(none_samples) == 0:
        print("ERROR: Could not load samples!")
        return
    
    # Combine
    all_samples = gradual_samples + none_samples
    
    print(f"\n{'='*80}")
    print("Computing combined features...")
    print(f"{'='*80}\n")
    
    # Process in parallel
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(compute_all_combined_features, all_samples),
            total=len(all_samples),
            desc="Processing"
        ))
    
    # Filter failures
    results = [r for r in results if r is not None]
    
    print(f"\nProcessed {len(results)} samples successfully")
    
    # Extract features and labels
    X = np.array([r['features'] for r in results])
    y = np.array([r['label'] for r in results])
    
    print(f"Feature matrix: {X.shape}")
    print(f"Labels: {y.shape} ({np.sum(y)} graduals, {len(y) - np.sum(y)} nones)")
    
    # Split train/test
    n_train = int(0.7 * len(X))
    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train logistic regression
    print(f"\n{'='*80}")
    print("TRAINING LOGISTIC REGRESSION CLASSIFIER")
    print(f"{'='*80}\n")
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"COMBINED FEATURES PERFORMANCE:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    
    # Feature importance
    feature_names = [
        'flow_consistency', 'trackability', 'motion_accum', 'layout_stability',
        'content_emerge', 'spatial_variance', 'brightness_mono', 'change_accum',
        'content×track', 'spatial×bright', 'motion×layout'
    ]
    
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE (Logistic Regression Weights)")
    print(f"{'='*80}\n")
    
    weights = clf.coef_[0]
    importance = list(zip(feature_names, weights))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for name, weight in importance:
        direction = "→ Gradual" if weight > 0 else "→ None"
        print(f"  {name:20s}: {weight:+.4f} {direction}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    if precision >= 0.70:
        print("✅ SUCCESS! Combined features achieve 70%+ precision!")
        print("   This is BETTER than individual features (60%) and current model!")
        print("\n   RECOMMENDATION: Implement these features in the model!")
    elif precision >= 0.65:
        print("⚠️  MODERATE: Combined features achieve 65-70% precision")
        print("   This is a SMALL improvement over current 60%")
        print("\n   RECOMMENDATION: May be worth implementing for incremental gain")
    else:
        print("❌ NO IMPROVEMENT: Combined features still <65% precision")
        print("   Low-level features hit fundamental ceiling")
        print("\n   RECOMMENDATION: Need semantic features (pre-trained backbone)")
    
    # Save
    output = {
        'metrics': {
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'feature_importance': {name: float(w) for name, w in importance},
        'num_features': len(feature_names)
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
