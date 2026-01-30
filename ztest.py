"""
ClipShots-style evaluation for shot boundary detection.
Based on: https://github.com/Tangshitao/ClipShots/blob/master/tools/evaluate.py

Usage:
    python clipshots_eval.py --model-path models/transformer_detection_pca.pth --fusion-strategy pca
"""

import json
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def if_overlap(begin1, end1, begin2, end2):
    """Check if two intervals overlap."""
    if begin1 > begin2:
        begin1, end1, begin2, end2 = begin2, end2, begin1, end1
    return end1 >= begin2


def get_union_cnt(set1, set2):
    """Count overlapping intervals between two sets."""
    cnt = 0
    for begin, end in set1:
        for _begin, _end in set2:
            if if_overlap(begin, end, _begin, _end):
                cnt += 1
                break
    return cnt


def recall_pre_f1(correct, gt_count, pred_count):
    """Calculate precision, recall, and F1 score."""
    recall = correct / gt_count if gt_count != 0 else 0
    precision = correct / pred_count if pred_count != 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0
    return precision, recall, f1


def load_model(model_path, feature_dim, device):
    """Load the trained transformer model."""
    from detection.transformer_detection import BoundaryDetectionTransformer
    
    # Load checkpoint first to get config if available
    checkpoint = torch.load(model_path, map_location=device)
    
    # Infer architecture from state_dict (more reliable than config)
    state_dict = checkpoint['model_state_dict']
    
    # Count transformer layers by looking at keys
    num_layers = 0
    for key in state_dict.keys():
        if key.startswith('transformer_layers.'):
            layer_idx = int(key.split('.')[1])
            num_layers = max(num_layers, layer_idx + 1)
    
    # Get hidden_dim from layer_norm weight
    hidden_dim = state_dict['layer_norm.weight'].shape[0]
    
    # Get ff_dim from transformer linear1 weight
    ff_dim = state_dict['transformer_layers.0.linear1.weight'].shape[0]
    
    # Get num_heads from relative_pos_bias
    num_heads = state_dict['relative_pos_bias'].shape[0]
    
    # Get max_sequence_length from positional_encoding
    max_sequence_length = state_dict['positional_encoding.pe'].shape[0]
    
    # Check if model has multi-task duration head
    has_duration_head = any('duration_head' in key for key in state_dict.keys())
    has_magnitude_conv = any('magnitude_temporal_conv' in key for key in state_dict.keys())
    
    print(f"Inferred model config: hidden_dim={hidden_dim}, num_layers={num_layers}, "
          f"ff_dim={ff_dim}, num_heads={num_heads}, max_sequence_length={max_sequence_length}")
    print(f"Multi-task duration head: {'YES' if has_duration_head else 'NO'}")
    print(f"Magnitude temporal conv: {'YES' if has_magnitude_conv else 'NO (old model - will use random init)'}")
    
    # Create the actual neural network model with inferred config
    model = BoundaryDetectionTransformer(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=0.1,
        use_difference_features=True,
        use_magnitude_skip=True,
        use_causal_attention=False,
        max_sequence_length=max_sequence_length,
    )
    
    # Load with strict=False to handle architecture changes
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: Missing keys (randomly initialized): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"WARNING: Unexpected keys (ignored): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    
    model = model.to(device)
    model.eval()
    
    # Store whether this is a multi-task model
    model.has_duration_head = has_duration_head
    
    return model


def compute_feature_distances(features):
    """Compute frame-to-frame feature distances."""
    from numpy.linalg import norm
    distances = []
    for i in range(len(features) - 1):
        dist = norm(features[i+1] - features[i])
        distances.append(dist)
    return np.array(distances)


def detect_transitions(model, features, device, threshold=0.5, window_size=100, min_distance=15, prominence=0.1, return_debug=False):
    """
    Detect shot transitions using CHUNKED inference matching training.
    
    Splits the video into fixed-size chunks (same as training sequence length),
    processes each chunk independently, then combines results.
    This matches the training distribution exactly.
    
    If the model has a multi-task duration head (trained with duration regression),
    uses the predicted durations to classify cuts vs graduals directly.
    Otherwise falls back to heuristic-based gradual detection.
    """
    seq_len = features.shape[0]
    
    if seq_len < 2:
        return ([], []) if return_debug else []
    
    # Check if model has duration prediction capability
    use_multi_task = getattr(model, 'has_duration_head', False) and getattr(model, 'use_multi_task', True)
    
    # Pre-compute feature distances for gradual detection (fallback)
    feature_distances = compute_feature_distances(features) if not use_multi_task else None
    
    # =========================================================================
    # CHUNKED INFERENCE - Match training sequence length exactly
    # Process video in non-overlapping chunks of window_size (default 100 frames)
    # =========================================================================
    chunk_size = window_size  # Use the same sequence length as training
    
    all_probs = np.zeros(seq_len - 1)
    all_durations = np.zeros(seq_len - 1) if use_multi_task else None
    
    # Collect all chunks
    chunks = []
    chunk_starts = []
    
    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)
        chunk_features = features[start_idx:end_idx]
        
        # Pad last chunk if needed to match training sequence length
        if len(chunk_features) < chunk_size:
            padded = np.zeros((chunk_size, features.shape[1]), dtype=features.dtype)
            padded[:len(chunk_features)] = chunk_features
            chunk_features = padded
        
        chunks.append(chunk_features)
        chunk_starts.append(start_idx)
    
    # Batch inference - process multiple chunks at once
    batch_size = 32
    
    with torch.no_grad():
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_starts = chunk_starts[batch_start:batch_end]
            
            # Stack into batch tensor: [B, T, D]
            batch_tensor = torch.FloatTensor(np.stack(batch_chunks)).to(device)
            
            # Use multi-task mode if available to get duration predictions
            if use_multi_task:
                outputs = model(batch_tensor, mode='frame_multi_task')
                if isinstance(outputs, dict):
                    probs = torch.sigmoid(outputs['class_logits']).squeeze(-1).cpu().numpy()  # (B, T-1)
                    # Convert from log(1+duration) back to duration
                    durations = torch.expm1(outputs['duration_pred']).squeeze(-1).cpu().numpy()  # (B, T-1)
                else:
                    probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
                    durations = None
            else:
                # Fallback: classification only
                outputs = model(batch_tensor, mode='frame')
                probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()  # (B, T-1)
                durations = None
            
            # Map each chunk's predictions back to global indices
            for i, start_idx in enumerate(batch_starts):
                chunk_probs = probs[i]  # (T-1,)
                out_len = chunk_probs.shape[0]
                
                # Calculate actual valid length (exclude padding)
                actual_chunk_end = min(start_idx + chunk_size, seq_len)
                actual_len = actual_chunk_end - start_idx - 1  # -1 because outputs are transitions
                
                if actual_len <= 0:
                    continue
                
                # Map to global indices
                global_start = start_idx
                global_end = global_start + actual_len
                
                all_probs[global_start:global_end] = chunk_probs[:actual_len]
                if use_multi_task and durations is not None:
                    all_durations[global_start:global_end] = durations[i, :actual_len]
    
    # Use scipy's find_peaks for proper NMS-like peak detection
    try:
        from scipy.signal import find_peaks
        
        # Find peaks with:
        # - height: above threshold
        # - distance: minimum frames between peaks (NMS)
        # - prominence: peak must stand out from surroundings
        peaks, properties = find_peaks(
            all_probs,
            height=threshold,
            distance=min_distance,
            prominence=prominence  # Use parameter - peak must clearly stand out
        )
        
        transitions = []
        peak_durations = []  # For debugging
        used_multi_task = False  # Track if we actually used multi-task path
        
        # =====================================================================
        # MULTI-TASK PATH: Use model's duration predictions directly
        # The model predicts log(1+duration), so:
        # - Cuts (dur=2): log(3) = 1.10 → expm1 gives 2.0
        # - Graduals (dur=5): log(6) = 1.79 → expm1 gives 5.0
        # 
        # ClipShots format: cuts are [n, n+1] so duration=2, not 1
        # During validation, model shows good accuracy on TRUE boundaries.
        # At test time, we only have detected peaks (which may include FPs).
        # Strategy: Use HIGH-CONFIDENCE peaks for better duration reliability.
        # =====================================================================
        if use_multi_task and all_durations is not None:
            # Always collect duration stats for debugging
            peak_durations_arr = all_durations[peaks] if len(peaks) > 0 else np.array([])
            peak_probs_arr = all_probs[peaks] if len(peaks) > 0 else np.array([])
            peak_durations = peak_durations_arr.tolist()  # For debugging output
            
            # Use multi-task path - trust the model's predictions
            # Higher probability = more likely to be real boundary = more reliable duration
            used_multi_task = True
            
            # Adaptive threshold based on probability confidence
            # High confidence (prob > 0.7): use model's duration prediction
            # Low confidence: be more conservative
            high_conf_threshold = 0.7
            
            for i, peak_idx in enumerate(peaks):
                predicted_duration = all_durations[peak_idx]
                peak_prob = all_probs[peak_idx]
                
                # For high-confidence detections, trust duration prediction
                # For low-confidence, assume cut (safer default)
                # ClipShots: cuts = duration 2, graduals > 2
                if peak_prob >= high_conf_threshold:
                    # High confidence: use model's prediction with threshold 3.0
                    # (log scale: log(1+3)=1.39, so duration ~3 separates cuts from graduals)
                    is_gradual = predicted_duration > 3.0
                else:
                    # Low confidence: only classify as gradual if duration is very high
                    is_gradual = predicted_duration > 5.0
                
                if is_gradual:
                    # Gradual: estimate extent from duration
                    half_duration = int(predicted_duration / 2)
                    start = max(0, peak_idx - half_duration)
                    end = min(len(all_probs) - 1, peak_idx + half_duration)
                    
                    # Ensure minimum extent of 2 frames for gradual
                    if end - start < 2:
                        end = start + 2
                    
                    transitions.append([int(start), int(end)])
                else:
                    # Cut: single frame
                    transitions.append([int(peak_idx), int(peak_idx)])
        
        # =====================================================================
        # FALLBACK PATH: Heuristic-based gradual detection (no duration head
        # or duration head not trained enough)
        # =====================================================================
        if not used_multi_task:
            # Compute median feature distance as baseline for cut detection
            if feature_distances is None:
                feature_distances = compute_feature_distances(features)
            median_dist = np.median(feature_distances)
            std_dist = np.std(feature_distances)
            
            for peak_idx in peaks:
                peak_prob = all_probs[peak_idx]
                
                # Method 1: Check probability spread (width of elevated region)
                prob_threshold = peak_prob * 0.5
                left = peak_idx
                while left > 0 and all_probs[left - 1] > prob_threshold:
                    left -= 1
                right = peak_idx
                while right < len(all_probs) - 1 and all_probs[right + 1] > prob_threshold:
                    right += 1
                prob_width = right - left + 1
                
                # Method 2: Check feature distance pattern
                window_start = max(0, peak_idx - 6)
                window_end = min(len(feature_distances), peak_idx + 6)
                local_distances = feature_distances[window_start:window_end]
                
                # For gradual: multiple frames with moderate-high distance
                # For cut: single frame with very high distance
                elevated_threshold = median_dist + std_dist
                elevated_count = np.sum(local_distances > elevated_threshold)
                
                peak_distance = feature_distances[min(peak_idx, len(feature_distances)-1)]
                is_extreme_peak = peak_distance > median_dist + 2.5 * std_dist
                
                # Classify as gradual if:
                # 1. Probability is spread across multiple frames (width > 3), OR
                # 2. Multiple elevated feature distances AND peak not extremely high
                is_gradual = (prob_width > 3) or (elevated_count >= 4 and not is_extreme_peak)
                
                if is_gradual:
                    # Use the wider of prob-based or feature-based extent
                    feat_start = window_start + np.argmax(local_distances > elevated_threshold) if elevated_count > 0 else peak_idx
                    feat_end = window_start + len(local_distances) - 1 - np.argmax(local_distances[::-1] > elevated_threshold) if elevated_count > 0 else peak_idx
                    
                    start = min(left, feat_start)
                    end = max(right, feat_end)
                    
                    if end - start >= 2:
                        transitions.append([int(start), int(end)])
                    else:
                        transitions.append([int(peak_idx), int(peak_idx)])
                else: 
                    transitions.append([int(peak_idx), int(peak_idx)])
        
        if return_debug:
            peak_probs = all_probs[peaks].tolist() if len(peaks) > 0 else []
            return transitions, peak_durations, peak_probs
        return transitions
        
    except ImportError:
        # Fallback without scipy
        transitions = []
        i = 0
        while i < len(all_probs):
            if all_probs[i] > threshold:
                # Find local maximum
                start = i
                peak_idx = i
                peak_val = all_probs[i]
                
                while i < len(all_probs) and all_probs[i] > threshold * 0.5:
                    if all_probs[i] > peak_val:
                        peak_val = all_probs[i]
                        peak_idx = i
                    i += 1
                
                transitions.append([int(peak_idx), int(peak_idx)])
                i = peak_idx + min_distance
            else:
                i += 1
        
        if return_debug:
            return transitions, [], []
        return transitions


def classify_transitions(transitions):
    """
    Classify transitions as cuts or gradual based on their width.
    Cuts: end - begin <= 1 (single frame or adjacent)
    Gradual: end - begin > 1 (spans multiple frames)
    """
    cuts = []
    graduals = []
    
    for begin, end in transitions:
        # ClipShots definition: gradual if span > 1 frame
        if end - begin <= 1:
            cuts.append([int(begin), int(end)])
        else:
            graduals.append([int(begin), int(end)])
    
    return cuts, graduals


def evaluate(predictions, ground_truth):
    """
    Evaluate predictions against ground truth using ClipShots metrics.
    """
    cut_correct_sum = 0
    gradual_correct_sum = 0
    all_correct_sum = 0
    
    gt_cut_sum = 0
    gt_gradual_sum = 0
    
    predict_cut_sum = 0
    predict_gradual_sum = 0
    
    for videoname, labels in ground_truth.items():
        if videoname in predictions:
            _gts = labels['transitions']
            
            # Separate GT into cuts and graduals
            gt_cuts = [(begin, end) for begin, end in _gts if end - begin <= 1]
            gt_graduals = [(begin, end) for begin, end in _gts if end - begin > 1]
            
            _predicts = predictions[videoname]
            predicts_cut = _predicts['cut']
            predicts_gradual = _predicts['gradual']
            
            # Count overlaps
            cut_correct = get_union_cnt(gt_cuts, predicts_cut)
            gradual_correct = get_union_cnt(gt_graduals, predicts_gradual)
            all_correct = get_union_cnt(_gts, predicts_cut + predicts_gradual)
            
            cut_correct_sum += cut_correct
            gradual_correct_sum += gradual_correct
            all_correct_sum += all_correct
            
            gt_cut_sum += len(gt_cuts)
            gt_gradual_sum += len(gt_graduals)
            
            predict_cut_sum += len(predicts_cut)
            predict_gradual_sum += len(predicts_gradual)
        else:
            print(f"Warning: {videoname} not found in predictions")
    
    # Calculate metrics
    cut_metrics = recall_pre_f1(cut_correct_sum, gt_cut_sum, predict_cut_sum)
    gradual_metrics = recall_pre_f1(gradual_correct_sum, gt_gradual_sum, predict_gradual_sum)
    all_metrics = recall_pre_f1(all_correct_sum, gt_cut_sum + gt_gradual_sum, predict_cut_sum + predict_gradual_sum)
    
    return {
        'cut': {'precision': cut_metrics[0], 'recall': cut_metrics[1], 'f1': cut_metrics[2],
                'gt_count': gt_cut_sum, 'pred_count': predict_cut_sum, 'correct': cut_correct_sum},
        'gradual': {'precision': gradual_metrics[0], 'recall': gradual_metrics[1], 'f1': gradual_metrics[2],
                    'gt_count': gt_gradual_sum, 'pred_count': predict_gradual_sum, 'correct': gradual_correct_sum},
        'all': {'precision': all_metrics[0], 'recall': all_metrics[1], 'f1': all_metrics[2],
                'gt_count': gt_cut_sum + gt_gradual_sum, 'pred_count': predict_cut_sum + predict_gradual_sum,
                'correct': all_correct_sum}
    }


def main():
    parser = argparse.ArgumentParser(description='ClipShots-style evaluation')
    parser.add_argument('--model-path', type=str, default='models/transformer_detection_pca.pth',
                        help='Path to trained model')
    parser.add_argument('--fusion-strategy', type=str, default='pca',
                        help='Feature fusion strategy (pca, pca, etc.)')
    parser.add_argument('--features-dir', type=str, default='testing_features',
                        help='Directory containing test features')
    parser.add_argument('--annotations-path', type=str, 
                        default=r'C:\Users\baraa\Documents\ClipShots\annotations\test.json',
                        help='Path to ClipShots test annotations')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--min-distance', type=int, default=15,
                        help='Minimum frames between detected peaks (NMS)')
    parser.add_argument('--prominence', type=float, default=0.1,
                        help='Peak prominence threshold (how much peak stands out)')
    parser.add_argument('--output', type=str, default='clipshots_results.json',
                        help='Output file for predictions')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load annotations
    print(f"Loading annotations from: {args.annotations_path}")
    with open(args.annotations_path, 'r') as f:
        ground_truth = json.load(f)
    print(f"Loaded {len(ground_truth)} videos from annotations")
    
    # Find feature files
    features_dir = Path(args.features_dir)
    feature_suffix = f"_{args.fusion_strategy}.npy"
    
    # Get list of videos that have features
    available_videos = set()
    for f in features_dir.glob(f"*{feature_suffix}"):
        # Correctly extract video ID by removing the suffix
        video_id = f.stem[:-len(f"_{args.fusion_strategy}")]
        available_videos.add(video_id)
    
    print(f"Found {len(available_videos)} videos with {args.fusion_strategy} features")
    
    # Determine feature dimension from first file
    sample_file = list(features_dir.glob(f"*{feature_suffix}"))[0]
    sample_features = np.load(sample_file)
    feature_dim = sample_features.shape[1]
    print(f"Feature dimension: {feature_dim}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, feature_dim, device)
    print("Model loaded successfully")
    
    # Report if multi-task mode is available
    if getattr(model, 'has_duration_head', False):
        print("✓ Multi-task duration head detected - using MODEL predictions for cut/gradual classification")
    else:
        print("✗ No duration head - using HEURISTIC gradual detection (fallback)")
    
    # For debugging: collect duration and probability statistics
    all_peak_durations = []
    all_peak_probs = []
    
    # Process each video
    predictions = {}
    videos_processed = 0
    videos_skipped = 0
    
    for video_name, video_info in tqdm(ground_truth.items(), desc="Processing videos"):
        # Get video ID (remove .mp4 extension)
        video_id = video_name.replace('.mp4', '')
        
        # Check if features exist
        feature_path = features_dir / f"{video_id}{feature_suffix}"
        if not feature_path.exists():
            videos_skipped += 1
            continue
        
        # Load features
        features = np.load(feature_path)
        
        # Detect transitions (with debug info)
        result = detect_transitions(
            model, features, device, 
            threshold=args.threshold, 
            min_distance=args.min_distance,
            prominence=args.prominence,
            return_debug=True
        )
        transitions, peak_durations, peak_probs = result
        all_peak_durations.extend(peak_durations)
        all_peak_probs.extend(peak_probs)
        
        # Classify into cuts and graduals
        cuts, graduals = classify_transitions(transitions)
        
        predictions[video_name] = {
            'cut': cuts,
            'gradual': graduals
        }
        videos_processed += 1
    
    print(f"\nProcessed {videos_processed} videos, skipped {videos_skipped} (missing features)")
    
    # Debug: show duration statistics
    if all_peak_durations and all_peak_probs:
        durations_array = np.array(all_peak_durations)
        probs_array = np.array(all_peak_probs)
        
        p20 = np.percentile(durations_array, 20)
        p80 = np.percentile(durations_array, 80)
        spread_ratio = p80 / p20 if p20 > 0 else 1.0
        n_clear_gradual = (durations_array > 2.5).sum()
        
        print(f"\n=== Duration Prediction Statistics (Multi-Task Mode ACTIVE) ===")
        print(f"  Total peaks: {len(durations_array)}")
        print(f"  Min: {durations_array.min():.3f}, Max: {durations_array.max():.3f}")
        print(f"  Mean: {durations_array.mean():.3f}, Median: {np.median(durations_array):.3f}")
        print(f"  P20: {p20:.3f}, P80: {p80:.3f}, Spread ratio: {spread_ratio:.2f}")
        
        # Analyze duration by confidence level
        print(f"\n  Duration by confidence level:")
        for prob_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            high_conf_mask = probs_array >= prob_thresh
            if high_conf_mask.sum() > 0:
                hc_durs = durations_array[high_conf_mask]
                hc_std = np.std(hc_durs) if len(hc_durs) > 1 else 0
                print(f"    Prob >= {prob_thresh}: n={high_conf_mask.sum()}, "
                      f"dur mean={hc_durs.mean():.2f}, std={hc_std:.2f}, "
                      f"min={hc_durs.min():.2f}, max={hc_durs.max():.2f}")
        
        print(f"\n  Duration threshold used: 2.0 (high conf) / 3.0 (low conf)")
        for thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
            n_gradual = (durations_array > thresh).sum()
            print(f"    Duration > {thresh}: {n_gradual} ({100*n_gradual/len(durations_array):.1f}%)")
    elif all_peak_durations:
        durations_array = np.array(all_peak_durations)
        print(f"\n=== Duration Stats (no probs) ===")
        print(f"  Total: {len(durations_array)}, Mean: {durations_array.mean():.3f}")
    else:
        print(f"\n=== No Duration Data ===")
        print(f"  (Model may not have duration head or no peaks detected)")
    
    # Debug: count prediction types
    total_cuts = sum(len(p['cut']) for p in predictions.values())
    total_graduals = sum(len(p['gradual']) for p in predictions.values())
    print(f"\nPrediction breakdown: {total_cuts} cuts, {total_graduals} graduals")
    
    # Evaluate
    print("\n" + "="*60)
    print("ClipShots Evaluation Results")
    print("="*60)
    
    results = evaluate(predictions, ground_truth)
    
    print(f"\n{'Type':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'GT':<8} {'Pred':<8} {'Correct':<8}")
    print("-"*72)
    
    for trans_type in ['cut', 'gradual', 'all']:
        r = results[trans_type]
        print(f"{trans_type:<12} {r['precision']:.4f}       {r['recall']:.4f}       {r['f1']:.4f}       "
              f"{r['gt_count']:<8} {r['pred_count']:<8} {r['correct']:<8}")
    
    # Save results (convert numpy types to native Python types)
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    output_data = {
        'predictions': convert_to_native(predictions),
        'metrics': convert_to_native(results),
        'config': {
            'model_path': args.model_path,
            'fusion_strategy': args.fusion_strategy,
            'threshold': args.threshold,
            'videos_processed': videos_processed,
            'videos_skipped': videos_skipped
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
