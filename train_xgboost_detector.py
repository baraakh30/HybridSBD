"""
Boundary Detector using XGBoost

Tree-based models often outperform neural networks on tabular data.
XGBoost can:
1. Handle feature interactions naturally
2. Be less prone to overfitting
3. Provide feature importance analysis
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import argparse
import random
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from scipy.ndimage import uniform_filter1d
import pickle

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("imbalanced-learn not installed. Run: pip install imbalanced-learn")


# Feature dimensions
FEATURE_SPLITS = {
    'clip': (0, 512),
    'cnn': (512, 2560),
    'hsv': (2560, 2816),
    'flow': (2816, 3368),
}


def compute_features(features: np.ndarray, context_window: int = 7) -> Tuple[np.ndarray, List[str]]:
    """Compute all features for a video with enhanced feature engineering."""
    diffs = features[1:] - features[:-1]
    num_frames = len(diffs)
    
    # Basic magnitudes
    total_mags = np.linalg.norm(diffs, axis=1)
    
    type_mags = {}
    for name, (start, end) in FEATURE_SPLITS.items():
        type_mags[name] = np.linalg.norm(diffs[:, start:end], axis=1)
    
    # Cosine similarity between consecutive diffs
    cosine_sims = np.zeros(num_frames)
    for i in range(1, num_frames):
        n1, n2 = np.linalg.norm(diffs[i-1]), np.linalg.norm(diffs[i])
        if n1 > 1e-8 and n2 > 1e-8:
            cosine_sims[i] = np.dot(diffs[i-1], diffs[i]) / (n1 * n2)
    
    # Magnitude acceleration (first derivative)
    mag_diff = np.zeros(num_frames)
    mag_diff[1:] = total_mags[1:] - total_mags[:-1]
    
    # Second derivative of magnitude
    mag_diff2 = np.zeros(num_frames)
    mag_diff2[1:] = mag_diff[1:] - mag_diff[:-1]
    
    # Local statistics with multiple window sizes
    local_stats = {}
    for window in [context_window, context_window * 2]:
        zscore = np.zeros(num_frames)
        max_ratio = np.zeros(num_frames)
        rank = np.zeros(num_frames)
        percentile_90 = np.zeros(num_frames)
        std_ratio = np.zeros(num_frames)
        
        for i in range(num_frames):
            s = max(0, i - window)
            e = min(num_frames, i + window + 1)
            local = total_mags[s:e]
            
            local_mean = local.mean()
            local_std = local.std() + 1e-8
            zscore[i] = (total_mags[i] - local_mean) / local_std
            max_ratio[i] = total_mags[i] / (local.max() + 1e-8)
            rank[i] = (local < total_mags[i]).sum() / len(local)
            percentile_90[i] = 1.0 if total_mags[i] > np.percentile(local, 90) else 0.0
            std_ratio[i] = total_mags[i] / local_std
        
        suffix = f'_w{window}'
        local_stats[f'zscore{suffix}'] = zscore
        local_stats[f'max_ratio{suffix}'] = max_ratio
        local_stats[f'rank{suffix}'] = rank
        local_stats[f'percentile_90{suffix}'] = percentile_90
        local_stats[f'std_ratio{suffix}'] = std_ratio
    
    # Per-type z-scores and statistics
    type_zscores = {}
    type_ranks = {}
    for name, mags in type_mags.items():
        zscores = np.zeros(num_frames)
        ranks = np.zeros(num_frames)
        for i in range(num_frames):
            s = max(0, i - context_window)
            e = min(num_frames, i + context_window + 1)
            local = mags[s:e]
            zscores[i] = (mags[i] - local.mean()) / (local.std() + 1e-8)
            ranks[i] = (local < mags[i]).sum() / len(local)
        type_zscores[name] = zscores
        type_ranks[name] = ranks
    
    # Feature ratios
    clip_cnn_ratio = (type_mags['clip'] + 1e-8) / (type_mags['cnn'] + 1e-8)
    hsv_flow_ratio = (type_mags['hsv'] + 1e-8) / (type_mags['flow'] + 1e-8)
    hsv_total_ratio = (type_mags['hsv'] + 1e-8) / (total_mags + 1e-8)
    clip_total_ratio = (type_mags['clip'] + 1e-8) / (total_mags + 1e-8)
    
    # Peak detection features
    is_local_peak = np.zeros(num_frames)
    peak_prominence = np.zeros(num_frames)
    for i in range(1, num_frames - 1):
        if total_mags[i] > total_mags[i-1] and total_mags[i] > total_mags[i+1]:
            is_local_peak[i] = 1.0
            # Prominence: how much higher than neighbors
            peak_prominence[i] = total_mags[i] - max(total_mags[i-1], total_mags[i+1])
    
    # Global statistics
    global_mean = total_mags.mean()
    global_std = total_mags.std() + 1e-8
    global_zscore = (total_mags - global_mean) / global_std
    global_percentile = np.array([np.searchsorted(np.sort(total_mags), m) / num_frames for m in total_mags])
    
    # Gradients (before and after)
    grad_before = np.zeros(num_frames)
    grad_after = np.zeros(num_frames)
    grad_before[1:] = total_mags[1:] - total_mags[:-1]
    grad_after[:-1] = total_mags[1:] - total_mags[:-1]
    
    # Gradient magnitude
    grad_magnitude = np.abs(grad_before) + np.abs(grad_after)
    
    # Rolling statistics
    rolling_mean_3 = uniform_filter1d(total_mags, size=3, mode='nearest')
    rolling_mean_5 = uniform_filter1d(total_mags, size=5, mode='nearest')
    deviation_from_rolling_3 = total_mags - rolling_mean_3
    deviation_from_rolling_5 = total_mags - rolling_mean_5
    
    # Per-type rolling deviations for HSV (most important feature)
    hsv_rolling_mean = uniform_filter1d(type_mags['hsv'], size=5, mode='nearest')
    hsv_deviation = type_mags['hsv'] - hsv_rolling_mean
    
    # Temporal context: features of neighbors
    prev_mag = np.zeros(num_frames)
    next_mag = np.zeros(num_frames)
    prev_mag[1:] = total_mags[:-1]
    next_mag[:-1] = total_mags[1:]
    mag_vs_prev = total_mags - prev_mag
    mag_vs_next = total_mags - next_mag
    
    # HSV-specific neighbor features
    hsv_prev = np.zeros(num_frames)
    hsv_next = np.zeros(num_frames)
    hsv_prev[1:] = type_mags['hsv'][:-1]
    hsv_next[:-1] = type_mags['hsv'][1:]
    hsv_vs_prev = type_mags['hsv'] - hsv_prev
    hsv_vs_next = type_mags['hsv'] - hsv_next
    
    # Combined peak score
    combined_peak_score = (
        local_stats['zscore_w7'] * 0.3 +
        local_stats['rank_w7'] * 0.3 +
        is_local_peak * 0.2 +
        type_zscores['hsv'] * 0.2
    )
    
    # Feature names for interpretability
    feature_names = [
        'total_mag', 'clip_mag', 'cnn_mag', 'hsv_mag', 'flow_mag',
        'mag_diff', 'mag_diff2', 'cosine_sim',
        'grad_before', 'grad_after', 'grad_magnitude',
        'clip_cnn_ratio', 'hsv_flow_ratio', 'hsv_total_ratio', 'clip_total_ratio',
        'clip_zscore', 'cnn_zscore', 'hsv_zscore', 'flow_zscore',
        'clip_rank', 'cnn_rank', 'hsv_rank', 'flow_rank',
        'is_local_peak', 'peak_prominence', 'global_zscore', 'global_percentile',
        'zscore_w7', 'max_ratio_w7', 'rank_w7', 'percentile_90_w7', 'std_ratio_w7',
        'zscore_w14', 'max_ratio_w14', 'rank_w14', 'percentile_90_w14', 'std_ratio_w14',
        'deviation_from_rolling_3', 'deviation_from_rolling_5',
        'hsv_deviation', 'hsv_vs_prev', 'hsv_vs_next',
        'mag_vs_prev', 'mag_vs_next',
        'combined_peak_score',
    ]
    
    # Combine all features
    all_features = []
    for i in range(num_frames):
        feat = [
            total_mags[i],
            type_mags['clip'][i],
            type_mags['cnn'][i],
            type_mags['hsv'][i],
            type_mags['flow'][i],
            mag_diff[i],
            mag_diff2[i],
            cosine_sims[i],
            grad_before[i],
            grad_after[i],
            grad_magnitude[i],
            clip_cnn_ratio[i],
            hsv_flow_ratio[i],
            hsv_total_ratio[i],
            clip_total_ratio[i],
            type_zscores['clip'][i],
            type_zscores['cnn'][i],
            type_zscores['hsv'][i],
            type_zscores['flow'][i],
            type_ranks['clip'][i],
            type_ranks['cnn'][i],
            type_ranks['hsv'][i],
            type_ranks['flow'][i],
            is_local_peak[i],
            peak_prominence[i],
            global_zscore[i],
            global_percentile[i],
            local_stats['zscore_w7'][i],
            local_stats['max_ratio_w7'][i],
            local_stats['rank_w7'][i],
            local_stats['percentile_90_w7'][i],
            local_stats['std_ratio_w7'][i],
            local_stats['zscore_w14'][i],
            local_stats['max_ratio_w14'][i],
            local_stats['rank_w14'][i],
            local_stats['percentile_90_w14'][i],
            local_stats['std_ratio_w14'][i],
            deviation_from_rolling_3[i],
            deviation_from_rolling_5[i],
            hsv_deviation[i],
            hsv_vs_prev[i],
            hsv_vs_next[i],
            mag_vs_prev[i],
            mag_vs_next[i],
            combined_peak_score[i],
        ]
        all_features.append(feat)
    
    return np.array(all_features, dtype=np.float32), feature_names


def load_data(features_dir: str, ground_truth: str, fusion_strategy: str, video_names: List[str]):
    """Load and compute features for videos."""
    features_path = Path(features_dir)
    
    with open(ground_truth, 'r') as f:
        all_gt = json.load(f)
    
    X_list = []
    y_list = []
    feature_names = None
    
    for video_name in tqdm(video_names, desc="Loading"):
        if video_name not in all_gt:
            continue
        
        # Load features
        video_stem = video_name.replace('.mp4', '')
        feat_file = None
        for suffix in ['___', '_']:
            path = features_path / f"{video_stem}{suffix}{fusion_strategy}.npy"
            if path.exists():
                feat_file = path
                break
        
        if feat_file is None:
            continue
        
        features = np.load(feat_file)
        if len(features) < 3:
            continue
        
        # Get boundaries
        video_data = all_gt[video_name]
        if isinstance(video_data, dict) and 'transitions' in video_data:
            boundaries = set(t[0] for t in video_data['transitions'])
        elif isinstance(video_data, list):
            boundaries = set(video_data)
        else:
            continue
        
        # Compute features
        X, feature_names = compute_features(features)
        y = np.array([1 if i in boundaries else 0 for i in range(len(X))])
        
        X_list.append(X)
        y_list.append(y)
    
    return np.vstack(X_list), np.concatenate(y_list), feature_names


def find_best_threshold(y_true, y_proba):
    """Find threshold that maximizes F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5, f1_scores[best_idx]


def resample_data(X, y, strategy='hybrid', random_state=42):
    """Apply resampling to handle class imbalance."""
    if not HAS_IMBLEARN:
        print("Warning: imbalanced-learn not installed, skipping resampling")
        return X, y
    
    print(f"\nApplying {strategy} resampling...")
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    print(f"  Before: Positive={pos_count}, Negative={neg_count}, Ratio=1:{neg_count//pos_count}")
    
    if strategy == 'smote':
        # SMOTE only - may be slow with large datasets
        sampler = SMOTE(random_state=random_state)
    elif strategy == 'undersample':
        # Undersample majority class
        sampler = RandomUnderSampler(
            sampling_strategy=0.1,  # 10:1 ratio
            random_state=random_state
        )
    elif strategy == 'hybrid':
        # Hybrid: undersample then SMOTE
        sampler = ImbPipeline([
            ('under', RandomUnderSampler(sampling_strategy=0.05, random_state=random_state)),
            ('smote', SMOTE(sampling_strategy=0.5, random_state=random_state))
        ])
    else:
        return X, y
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    pos_count = (y_resampled == 1).sum()
    neg_count = (y_resampled == 0).sum()
    print(f"  After: Positive={pos_count}, Negative={neg_count}, Ratio=1:{neg_count//max(1,pos_count)}")
    
    return X_resampled, y_resampled


def train(
    features_dir: str = "training_features",
    ground_truth: str = r"C:\Users\baraa\Documents\ClipShots\annotations\train.json",
    fusion_strategy: str = "concatenation",
    val_split: float = 0.2,
    resample: str = "none",  # 'none', 'smote', 'undersample', 'hybrid'
    n_estimators: int = 1000,
    max_depth: int = 8,
    learning_rate: float = 0.05,
):
    if not HAS_XGB:
        print("Please install xgboost: pip install xgboost")
        return
    
    print("=" * 60)
    print("XGBOOST BOUNDARY DETECTOR (Enhanced)")
    print("=" * 60)
    
    # Get available videos
    with open(ground_truth, 'r') as f:
        all_gt = json.load(f)
    
    features_path = Path(features_dir)
    available_videos = []
    for video_name in all_gt.keys():
        video_stem = video_name.replace('.mp4', '')
        for suffix in ['___', '_']:
            if (features_path / f"{video_stem}{suffix}{fusion_strategy}.npy").exists():
                available_videos.append(video_name)
                break
    
    print(f"Available videos: {len(available_videos)}")
    
    # Split by video
    random.seed(42)
    random.shuffle(available_videos)
    split_idx = int(len(available_videos) * (1 - val_split))
    train_videos = available_videos[:split_idx]
    val_videos = available_videos[split_idx:]
    
    print(f"Train videos: {len(train_videos)}")
    print(f"Val videos: {len(val_videos)}")
    
    # Load data
    print("\nLoading training data...")
    X_train, y_train, feature_names = load_data(features_dir, ground_truth, fusion_strategy, train_videos)
    print(f"  Samples: {len(X_train)}, Positive: {y_train.sum()}")
    print(f"  Features: {len(feature_names)}")
    
    print("\nLoading validation data...")
    X_val, y_val, _ = load_data(features_dir, ground_truth, fusion_strategy, val_videos)
    print(f"  Samples: {len(X_val)}, Positive: {y_val.sum()}")
    
    # Apply resampling if requested
    if resample != 'none':
        X_train, y_train = resample_data(X_train, y_train, strategy=resample)
    
    # Compute scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"\nScale pos weight: {scale_pos_weight:.1f}")
    
    # Train XGBoost with improved parameters
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        min_child_weight=10,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=2.0,
        max_delta_step=1,  # Helps with imbalanced classes
        random_state=42,
        eval_metric='aucpr',  # Area under PR curve - better for imbalanced
        early_stopping_rounds=100,
        n_jobs=-1,
        tree_method='hist',  # Faster training
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Find best threshold
    best_thresh, best_f1 = find_best_threshold(y_val, y_proba)
    y_pred = (y_proba >= best_thresh).astype(int)
    
    p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
    
    print(f"\nBest Threshold: {best_thresh:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Also show results at different threshold levels
    print("\n--- Results at different thresholds ---")
    for thresh in [0.3, 0.5, 0.7, 0.9, best_thresh]:
        y_pred_t = (y_proba >= thresh).astype(int)
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(y_val, y_pred_t, average='binary', zero_division=0)
        marker = " <-- BEST" if thresh == best_thresh else ""
        print(f"  Thresh={thresh:.2f}: P={p_t:.4f}, R={r_t:.4f}, F1={f1_t:.4f}{marker}")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Top 15)")
    print("=" * 60)
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    for i in range(min(15, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    # Save model
    save_path = Path("models") / "boundary_detector_xgb.pkl"
    save_path.parent.mkdir(exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'threshold': best_thresh,
            'f1': f1,
            'precision': p,
            'recall': r,
            'feature_names': feature_names,
        }, f)
    
    print(f"\nModel saved to: {save_path}")
    
    return model, best_thresh, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="training_features")
    parser.add_argument("--ground_truth", default=r"C:\Users\baraa\Documents\ClipShots\annotations\train.json")
    parser.add_argument("--fusion_strategy", default="concatenation")
    parser.add_argument("--resample", default="none", choices=['none', 'smote', 'undersample', 'hybrid'],
                        help="Resampling strategy for class imbalance")
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    
    args = parser.parse_args()
    train(**vars(args))
