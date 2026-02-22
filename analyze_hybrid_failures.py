"""
Deep Per-Layer Analysis of Hybrid Model Failure Modes

Analyzes characteristics of:
1. False Positive Graduals (predicted gradual, actually none)
2. False Negative Graduals (predicted none/cut, actually gradual)
3. False Negative None (predicted boundary, actually none)

Extracts and analyzes ALL internal representations:
- RGB features (640-d from spatial encoder)
- Diff features (640-d from spatial encoder)
- Pixel motion features (15-d: magnitude, std, coherence at 5 distances)
- Reasoning features (160-d: acceleration, jerk, spatial, coherence)
- Fused features (main_dim after fusion blocks)
- Boundary logits & type logits

Goal: Understand WHAT the model sees differently in failure cases
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple


def extract_failure_cases(model, dataloader, device, max_samples=None):
    """
    Extract all failure cases with their internal representations
    
    Args:
        model: trained hybrid detector
        dataloader: validation/test dataloader
        device: torch device
        max_samples: maximum clips to process (None = all)
        
    Returns:
        failure_cases: dict with FP/FN cases and their features
    """
    model.eval()
    
    failure_cases = {
        'fp_graduals': [],  # Predicted gradual, GT none
        'fn_graduals': [],  # Predicted none/cut, GT gradual
        'fn_none': [],      # Predicted boundary, GT none
        'tp_cuts': [],      # Predicted cut, GT cut (BASELINE)
        'tp_graduals': [],  # Predicted gradual, GT gradual (BASELINE)
        'tp_none': [],      # Predicted none, GT none (BASELINE)
    }
    
    samples_processed = 0
    
    print("\nExtracting failure cases with internal features...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing")):
            if max_samples and samples_processed >= max_samples:
                break
                
            frames, labels, transition_lengths = batch_data
            frames = frames.to(device, non_blocking=True)
            labels = labels.cpu().numpy()
            
            # Forward pass with feature extraction
            boundary_logits, type_logits, features = model(frames, return_features=True)
            
            # Move to CPU/numpy
            boundary_logits = boundary_logits.cpu().numpy()
            type_logits = type_logits.cpu().numpy()
            
            # Extract features
            rgb_feats = features['rgb'].cpu().numpy()
            diff_feats = features['diff'].cpu().numpy()
            pixel_motion = features['pixel_motion'].cpu().numpy()
            fused_feats = features['fused'].cpu().numpy()
            
            # Reasoning features
            reasoning = features['reasoning']
            acceleration = reasoning['acceleration'].cpu().numpy()
            jerk = reasoning['jerk'].cpu().numpy()
            pixel_motion_raw = reasoning['pixel_motion'].cpu().numpy()
            combined_reasoning = reasoning['combined'].cpu().numpy()
            
            # Process each sample in batch
            for b in range(frames.shape[0]):
                sample_labels = labels[b]
                sample_boundary_logits = boundary_logits[b]
                sample_type_logits = type_logits[b]
                
                # Get predictions
                boundary_probs = 1 / (1 + np.exp(-sample_boundary_logits.squeeze()))
                type_probs = np.exp(sample_type_logits) / np.exp(sample_type_logits).sum(axis=-1, keepdims=True)
                type_preds = np.argmax(type_probs, axis=-1)
                
                # Combine: if boundary prob > 0.5, use type prediction, else none
                pred_labels = np.where(boundary_probs > 0.5, type_preds, 0)
                
                # Find failure frames
                for t in range(len(sample_labels)):
                    gt_label = sample_labels[t]
                    pred_label = pred_labels[t]
                    
                    # Case 1: False Positive Gradual (predicted=2, GT=0)
                    if pred_label == 2 and gt_label == 0:
                        failure_cases['fp_graduals'].append({
                            'frame_idx': t,
                            'clip_idx': samples_processed + b,
                            'boundary_prob': float(boundary_probs[t]),
                            'type_probs': type_probs[t].tolist(),
                            'rgb_features': rgb_feats[b, t],
                            'diff_features': diff_feats[b, t],
                            'pixel_motion': pixel_motion[b, t],
                            'fused_features': fused_feats[b, t],
                            'acceleration': float(acceleration[b, t]),
                            'jerk': float(jerk[b, t]),
                            'combined_reasoning': combined_reasoning[b, t],
                        })
                    
                    # Case 2: False Negative Gradual (predicted=0 or 1, GT=2)
                    elif pred_label != 2 and gt_label == 2:
                        failure_cases['fn_graduals'].append({
                            'frame_idx': t,
                            'clip_idx': samples_processed + b,
                            'predicted_label': int(pred_label),
                            'boundary_prob': float(boundary_probs[t]),
                            'type_probs': type_probs[t].tolist(),
                            'rgb_features': rgb_feats[b, t],
                            'diff_features': diff_feats[b, t],
                            'pixel_motion': pixel_motion[b, t],
                            'fused_features': fused_feats[b, t],
                            'acceleration': float(acceleration[b, t]),
                            'jerk': float(jerk[b, t]),
                            'combined_reasoning': combined_reasoning[b, t],
                        })
                    
                    # Case 3: False Negative None (predicted=1 or 2, GT=0)
                    elif pred_label > 0 and gt_label == 0:
                        failure_cases['fn_none'].append({
                            'frame_idx': t,
                            'clip_idx': samples_processed + b,
                            'predicted_label': int(pred_label),
                            'boundary_prob': float(boundary_probs[t]),
                            'type_probs': type_probs[t].tolist(),
                            'rgb_features': rgb_feats[b, t],
                            'diff_features': diff_feats[b, t],
                            'pixel_motion': pixel_motion[b, t],
                            'fused_features': fused_feats[b, t],
                            'acceleration': float(acceleration[b, t]),
                            'jerk': float(jerk[b, t]),
                            'combined_reasoning': combined_reasoning[b, t],
                        })
                    
                    # TRUE POSITIVES (for baseline comparisons)
                    # TP Cut
                    elif pred_label == 1 and gt_label == 1:
                        failure_cases['tp_cuts'].append({
                            'frame_idx': t,
                            'clip_idx': samples_processed + b,
                            'boundary_prob': float(boundary_probs[t]),
                            'type_probs': type_probs[t].tolist(),
                            'rgb_features': rgb_feats[b, t],
                            'diff_features': diff_feats[b, t],
                            'pixel_motion': pixel_motion[b, t],
                            'fused_features': fused_feats[b, t],
                            'acceleration': float(acceleration[b, t]),
                            'jerk': float(jerk[b, t]),
                            'combined_reasoning': combined_reasoning[b, t],
                        })
                    
                    # TP Gradual
                    elif pred_label == 2 and gt_label == 2:
                        failure_cases['tp_graduals'].append({
                            'frame_idx': t,
                            'clip_idx': samples_processed + b,
                            'boundary_prob': float(boundary_probs[t]),
                            'type_probs': type_probs[t].tolist(),
                            'rgb_features': rgb_feats[b, t],
                            'diff_features': diff_feats[b, t],
                            'pixel_motion': pixel_motion[b, t],
                            'fused_features': fused_feats[b, t],
                            'acceleration': float(acceleration[b, t]),
                            'jerk': float(jerk[b, t]),
                            'combined_reasoning': combined_reasoning[b, t],
                        })
                    
                    # TP None
                    elif pred_label == 0 and gt_label == 0:
                        # Sample 10% to avoid overwhelming memory
                        if np.random.random() < 0.1:
                            failure_cases['tp_none'].append({
                                'frame_idx': t,
                                'clip_idx': samples_processed + b,
                                'boundary_prob': float(boundary_probs[t]),
                                'type_probs': type_probs[t].tolist(),
                                'rgb_features': rgb_feats[b, t],
                                'diff_features': diff_feats[b, t],
                                'pixel_motion': pixel_motion[b, t],
                                'fused_features': fused_feats[b, t],
                                'acceleration': float(acceleration[b, t]),
                                'jerk': float(jerk[b, t]),
                                'combined_reasoning': combined_reasoning[b, t],
                            })
            
            # Update samples processed (whole batch at once)
            samples_processed += frames.shape[0]
            if max_samples and samples_processed >= max_samples:
                break
    
    print(f"\nCases extracted:")
    print(f"  ✅ TP Cuts:      {len(failure_cases['tp_cuts'])}")
    print(f"  ✅ TP Graduals:  {len(failure_cases['tp_graduals'])}")
    print(f"  ✅ TP None:      {len(failure_cases['tp_none'])}  (sampled at 10%)")
    print(f"  ❌ FP Graduals:  {len(failure_cases['fp_graduals'])}")
    print(f"  ❌ FN Graduals:  {len(failure_cases['fn_graduals'])}")
    print(f"  ❌ FN None:      {len(failure_cases['fn_none'])}")
    
    return failure_cases


def analyze_per_layer_statistics(failure_cases):
    """
    Compute per-layer statistics for each failure mode
    
    For each layer, compute:
    - Mean and std of features
    - Distribution characteristics
    - Comparison to successful predictions (if available)
    
    Args:
        failure_cases: dict from extract_failure_cases
        
    Returns:
        analysis: dict with per-layer statistics
    """
    analysis = {}
    
    for failure_type, cases in failure_cases.items():
        if len(cases) == 0:
            continue
            
        print(f"\nAnalyzing {failure_type} ({len(cases)} cases)...")
        
        # Aggregate features
        rgb_features = np.stack([c['rgb_features'] for c in cases])
        diff_features = np.stack([c['diff_features'] for c in cases])
        pixel_motion = np.stack([c['pixel_motion'] for c in cases])
        fused_features = np.stack([c['fused_features'] for c in cases])
        combined_reasoning = np.stack([c['combined_reasoning'] for c in cases])
        
        acceleration = np.array([c['acceleration'] for c in cases])
        jerk = np.array([c['jerk'] for c in cases])
        boundary_probs = np.array([c['boundary_prob'] for c in cases])
        
        analysis[failure_type] = {
            # RGB features (640-d from spatial encoder)
            'rgb_features': {
                'mean': float(rgb_features.mean()),
                'std': float(rgb_features.std()),
                'norm': float(np.linalg.norm(rgb_features, axis=1).mean()),
                'sparsity': float((np.abs(rgb_features) < 0.01).mean()),
                'activation_rate': float((rgb_features > 0).mean()),
            },
            
            # Diff features (640-d from spatial encoder)
            'diff_features': {
                'mean': float(diff_features.mean()),
                'std': float(diff_features.std()),
                'norm': float(np.linalg.norm(diff_features, axis=1).mean()),
                'sparsity': float((np.abs(diff_features) < 0.01).mean()),
                'activation_rate': float((diff_features > 0).mean()),
            },
            
            # Pixel motion features (15-d: 5 distances × 3 stats)
            'pixel_motion': {
                'mean': pixel_motion.mean(axis=0).tolist(),
                'std': pixel_motion.std(axis=0).tolist(),
                'min': pixel_motion.min(axis=0).tolist(),
                'max': pixel_motion.max(axis=0).tolist(),
                'percentile_25': np.percentile(pixel_motion, 25, axis=0).tolist(),
                'percentile_75': np.percentile(pixel_motion, 75, axis=0).tolist(),
            },
            
            # Motion dynamics (scalar values)
            'acceleration': {
                'mean': float(acceleration.mean()),
                'std': float(acceleration.std()),
                'min': float(acceleration.min()),
                'max': float(acceleration.max()),
                'percentile_25': float(np.percentile(acceleration, 25)),
                'percentile_50': float(np.percentile(acceleration, 50)),
                'percentile_75': float(np.percentile(acceleration, 75)),
            },
            
            'jerk': {
                'mean': float(jerk.mean()),
                'std': float(jerk.std()),
                'min': float(jerk.min()),
                'max': float(jerk.max()),
                'percentile_25': float(np.percentile(jerk, 25)),
                'percentile_50': float(np.percentile(jerk, 50)),
                'percentile_75': float(np.percentile(jerk, 75)),
            },
            
            # Reasoning module outputs (combined 160-d feature)
            'combined_reasoning': {
                'mean': float(combined_reasoning.mean()),
                'std': float(combined_reasoning.std()),
                'norm': float(np.linalg.norm(combined_reasoning, axis=1).mean()),
                'sparsity': float((np.abs(combined_reasoning) < 0.01).mean()),
                'activation_rate': float((combined_reasoning > 0).mean()),
            },
            
            # Fused features (after fusion blocks)
            'fused_features': {
                'mean': float(fused_features.mean()),
                'std': float(fused_features.std()),
                'norm': float(np.linalg.norm(fused_features, axis=1).mean()),
                'sparsity': float((np.abs(fused_features) < 0.01).mean()),
                'activation_rate': float((fused_features > 0).mean()),
            },
            
            # Output statistics
            'boundary_probability': {
                'mean': float(boundary_probs.mean()),
                'std': float(boundary_probs.std()),
                'min': float(boundary_probs.min()),
                'max': float(boundary_probs.max()),
            },
            
            # Sample count
            'num_samples': len(cases),
        }
    
    return analysis


def compare_failure_modes(analysis):
    """
    Compare characteristics across different failure modes
    
    Args:
        analysis: dict from analyze_per_layer_statistics
        
    Returns:
        comparison: dict with comparative statistics
    """
    comparison = {}
    
    # Keys to compare
    layers = ['rgb_features', 'diff_features', 'pixel_motion', 'acceleration', 
              'jerk', 'combined_reasoning', 'fused_features', 'boundary_probability']
    
    for layer in layers:
        comparison[layer] = {}
        
        for failure_type in analysis:
            if layer in analysis[failure_type]:
                comparison[layer][failure_type] = analysis[failure_type][layer]
    
    # Compute ratios for key metrics
    if 'fp_graduals' in analysis and 'fn_graduals' in analysis:
        comparison['fp_vs_fn_graduals'] = {
            'acceleration_ratio': analysis['fp_graduals']['acceleration']['mean'] / (analysis['fn_graduals']['acceleration']['mean'] + 1e-8),
            'jerk_ratio': analysis['fp_graduals']['jerk']['mean'] / (analysis['fn_graduals']['jerk']['mean'] + 1e-8),
            'diff_norm_ratio': analysis['fp_graduals']['diff_features']['norm'] / (analysis['fn_graduals']['diff_features']['norm'] + 1e-8),
        }
    
    return comparison


def print_analysis_summary(analysis, comparison):
    """Print human-readable summary of analysis"""
    
    print("\n" + "="*80)
    print("DEEP PER-LAYER FAILURE ANALYSIS WITH TP BASELINES")
    print("="*80)
    
    # Print in groups: TP baseline first, then related failures
    groups = [
        ('GRADUALS', ['tp_graduals', 'fp_graduals', 'fn_graduals']),
        ('CUTS', ['tp_cuts']),
        ('NONE', ['tp_none', 'fn_none']),
    ]
    
    for group_name, types in groups:
        print(f"\n{'#'*80}")
        print(f"# {group_name}")
        print(f"{'#'*80}")
        
        for failure_type in types:
            if failure_type not in analysis:
                continue
                
            stats = analysis[failure_type]
            
            print(f"\n{'='*80}")
            print(f"{failure_type.upper().replace('_', ' ')} ({stats['num_samples']} cases)")
            print("="*80)
        
        print(f"\n📊 PIXEL MOTION FEATURES (15-d: 5 distances × 3 stats)")
        print(f"  Mean: {np.array(stats['pixel_motion']['mean']).mean():.4f}")
        print(f"  Std:  {np.array(stats['pixel_motion']['std']).mean():.4f}")
        print(f"  Min:  {np.array(stats['pixel_motion']['min']).min():.4f}")
        print(f"  Max:  {np.array(stats['pixel_motion']['max']).max():.4f}")
        
        print(f"\n🔥 MOTION DYNAMICS")
        print(f"  Acceleration: μ={stats['acceleration']['mean']:.4f}, σ={stats['acceleration']['std']:.4f}")
        print(f"                range=[{stats['acceleration']['min']:.4f}, {stats['acceleration']['max']:.4f}]")
        print(f"  Jerk:         μ={stats['jerk']['mean']:.4f}, σ={stats['jerk']['std']:.4f}")
        print(f"                range=[{stats['jerk']['min']:.4f}, {stats['jerk']['max']:.4f}]")
        
        print(f"\n🎨 SPATIAL FEATURES (640-d RGB encoder)")
        print(f"  Mean:       {stats['rgb_features']['mean']:.4f}")
        print(f"  Std:        {stats['rgb_features']['std']:.4f}")
        print(f"  Norm:       {stats['rgb_features']['norm']:.4f}")
        print(f"  Sparsity:   {stats['rgb_features']['sparsity']:.2%}")
        print(f"  Active:     {stats['rgb_features']['activation_rate']:.2%}")
        
        print(f"\n📐 DIFFERENCE FEATURES (640-d Diff encoder)")
        print(f"  Mean:       {stats['diff_features']['mean']:.4f}")
        print(f"  Std:        {stats['diff_features']['std']:.4f}")
        print(f"  Norm:       {stats['diff_features']['norm']:.4f}")
        print(f"  Sparsity:   {stats['diff_features']['sparsity']:.2%}")
        print(f"  Active:     {stats['diff_features']['activation_rate']:.2%}")
        
        print(f"\n🧠 REASONING MODULE OUTPUTS (160-d combined)")
        print(f"  Mean:       {stats['combined_reasoning']['mean']:.4f}")
        print(f"  Std:        {stats['combined_reasoning']['std']:.4f}")
        print(f"  Norm:       {stats['combined_reasoning']['norm']:.4f}")
        print(f"  Sparsity:   {stats['combined_reasoning']['sparsity']:.2%}")
        print(f"  Active:     {stats['combined_reasoning']['activation_rate']:.2%}")
        
        print(f"\n🎯 FUSED FEATURES (after fusion blocks)")
        print(f"  Mean:       {stats['fused_features']['mean']:.4f}")
        print(f"  Std:        {stats['fused_features']['std']:.4f}")
        print(f"  Norm:       {stats['fused_features']['norm']:.4f}")
        print(f"  Sparsity:   {stats['fused_features']['sparsity']:.2%}")
        print(f"  Active:     {stats['fused_features']['activation_rate']:.2%}")
        
        print(f"\n📈 OUTPUT")
        print(f"  Boundary prob: μ={stats['boundary_probability']['mean']:.4f}, σ={stats['boundary_probability']['std']:.4f}")
        print(f"                 range=[{stats['boundary_probability']['min']:.4f}, {stats['boundary_probability']['max']:.4f}]")
    
    # Comparative analysis with TP baselines
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS: TP BASELINES vs FAILURES")
    print("="*80)
    
    # Gradual analysis
    if 'tp_graduals' in analysis and ('fp_graduals' in analysis or 'fn_graduals' in analysis):
        print("\n🔬 GRADUAL TRANSITIONS:")
        tp_g = analysis['tp_graduals']
        
        if 'fp_graduals' in analysis:
            fp_g = analysis['fp_graduals']
            acc_ratio = fp_g['acceleration']['mean'] / (tp_g['acceleration']['mean'] + 1e-8)
            jerk_ratio = fp_g['jerk']['mean'] / (tp_g['jerk']['mean'] + 1e-8)
            diff_ratio = fp_g['diff_features']['norm'] / tp_g['diff_features']['norm']
            
            print(f"\n  TP vs FP Graduals:")
            print(f"    TP acceleration:  μ={tp_g['acceleration']['mean']:.4f}, σ={tp_g['acceleration']['std']:.4f}")
            print(f"    FP acceleration:  μ={fp_g['acceleration']['mean']:.4f}, σ={fp_g['acceleration']['std']:.4f} ({acc_ratio:.2f}x)")
            print(f"    TP jerk:          μ={tp_g['jerk']['mean']:.4f}, σ={tp_g['jerk']['std']:.4f}")
            print(f"    FP jerk:          μ={fp_g['jerk']['mean']:.4f}, σ={fp_g['jerk']['std']:.4f} ({jerk_ratio:.2f}x)")
            print(f"    TP diff norm:     {tp_g['diff_features']['norm']:.2f}")
            print(f"    FP diff norm:     {fp_g['diff_features']['norm']:.2f} ({diff_ratio:.2f}x)")
            
            if abs(acc_ratio) > 2.0:
                print(f"    ⚠️  FP graduals have {acc_ratio:.1f}x acceleration → CAMERA MOTION!")
        
        if 'fn_graduals' in analysis:
            fn_g = analysis['fn_graduals']
            diff_ratio = fn_g['diff_features']['norm'] / tp_g['diff_features']['norm']
            prob_ratio = fn_g['boundary_probability']['mean'] / (tp_g['boundary_probability']['mean'] + 1e-8)
            
            print(f"\n  TP vs FN Graduals:")
            print(f"    TP diff norm:     {tp_g['diff_features']['norm']:.2f}")
            print(f"    FN diff norm:     {fn_g['diff_features']['norm']:.2f} ({diff_ratio:.2f}x)")
            print(f"    TP boundary prob: {tp_g['boundary_probability']['mean']:.3f}")
            print(f"    FN boundary prob: {fn_g['boundary_probability']['mean']:.3f} ({prob_ratio:.2f}x)")
            
            if diff_ratio < 0.8:
                print(f"    ⚠️  FN graduals have subtle changes → INCREASE SENSITIVITY")
    
    # None analysis  
    if 'tp_none' in analysis and 'fn_none' in analysis:
        print("\n🔬 NONE (no boundary):")
        tp_n = analysis['tp_none']
        fn_n = analysis['fn_none']
        
        acc_ratio = fn_n['acceleration']['mean'] / (tp_n['acceleration']['mean'] + 1e-8)
        jerk_ratio = fn_n['jerk']['mean'] / (tp_n['jerk']['mean'] + 1e-8)
        motion_ratio = np.array(fn_n['pixel_motion']['mean']).mean() / (np.array(tp_n['pixel_motion']['mean']).mean() + 1e-8)
        
        print(f"\n  TP vs FN None:")
        print(f"    TP acceleration:  μ={tp_n['acceleration']['mean']:.4f}")
        print(f"    FN acceleration:  μ={fn_n['acceleration']['mean']:.4f} ({acc_ratio:.2f}x)")
        print(f"    TP jerk:          μ={tp_n['jerk']['mean']:.4f}")
        print(f"    FN jerk:          μ={fn_n['jerk']['mean']:.4f} ({jerk_ratio:.2f}x)")
        print(f"    TP pixel motion:  {np.array(tp_n['pixel_motion']['mean']).mean():.4f}")
        print(f"    FN pixel motion:  {np.array(fn_n['pixel_motion']['mean']).mean():.4f} ({motion_ratio:.2f}x)")
        
        if abs(acc_ratio) > 3.0 or abs(jerk_ratio) > 3.0:
            print(f"    ⚠️  FN none (false alarms) have extreme motion → ADD MOTION REJECTION")
    
    # Threshold recommendations
    if 'tp_graduals' in analysis and 'fp_graduals' in analysis:
        print(f"\n{'='*80}")
        print("💡 RECOMMENDED POST-PROCESSING THRESHOLDS")
        print("="*80)
        
        tp_g = analysis['tp_graduals']
        fp_g = analysis['fp_graduals']
        
        # Find threshold that separates TP from FP
        tp_acc_mean = tp_g['acceleration']['mean']
        tp_acc_std = tp_g['acceleration']['std']
        fp_acc_mean = fp_g['acceleration']['mean']
        
        # Threshold at TP_mean + 2*TP_std
        acc_threshold = tp_acc_mean + 2 * tp_acc_std
        
        print(f"\n  Reject gradual prediction if:")
        print(f"    |acceleration| > {abs(acc_threshold):.4f}")
        print(f"      (TP: μ={tp_acc_mean:.4f}, σ={tp_acc_std:.4f})")
        print(f"      (FP: μ={fp_acc_mean:.4f})")
        
        if 'fn_none' in analysis:
            fn_n = analysis['fn_none']
            fn_acc_mean = fn_n['acceleration']['mean']
            fn_jerk_std = fn_n['jerk']['std']
            
            print(f"\n  Reject boundary prediction if:")
            print(f"    |acceleration| > {abs(fn_acc_mean) * 0.5:.4f}  OR  |jerk| > {fn_jerk_std * 2:.4f}")
            print(f"      (Extreme motion indicates camera work, not scene change)")


def main():
    """Run deep failure analysis from command line"""
    import argparse
    from detection.hybrid_boundary_detector import create_hybrid_detector
    from datasets import PredecodedDataset
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description='Deep Per-Layer Failure Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--cache_root', type=str, default=r'D:\grad\data\training_cache',
                       help='Path to predecoded dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum clips to process (None = all)')
    parser.add_argument('--output', type=str, default='outputs/failure_analysis.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = create_hybrid_detector(
        main_dim=config.get('main_dim', 512),
        num_fusion_blocks=config.get('num_fusion_blocks', 4)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"\nLoading validation dataset...")
    dataset = PredecodedDataset(
        args.cache_root,
        split='val',
        val_split_ratio=0.15,
        random_seed=42
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Extract failure cases
    failure_cases = extract_failure_cases(model, dataloader, device, args.max_samples)
    
    # Analyze per-layer statistics
    analysis = analyze_per_layer_statistics(failure_cases)
    
    # Compare failure modes
    comparison = compare_failure_modes(analysis)
    
    # Print summary
    print_analysis_summary(analysis, comparison)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'analysis': analysis,
            'comparison': comparison,
        }, f, indent=2)
    
    print(f"\n✅ Analysis saved to: {output_path}")


if __name__ == '__main__':
    main()
