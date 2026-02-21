"""
Analyze why the model confuses high-motion scenes with gradual transitions.

Extracts internal model representations to understand:
1. What embedding distances actually look like for FP vs TP graduals
2. What diff encoder sees in high-motion vs true gradual patterns
3. Where in the architecture the confusion happens
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from detection.context_aware_detector import create_context_aware_detector
from train_boundary_detector import PredecodedDataset
from train_turbo_sbd_v2 import extract_transitions_from_predictions


def extract_internal_features(model, frames):
    """
    Extract internal representations from the model.
    
    Returns:
        dict with spatial_feats, diff_feats, embed_dists, temporal_feats, etc.
    """
    model.eval()
    B, T, C, H, W = frames.shape
    
    with torch.no_grad():
        # 1. Spatial encoding
        x_flat = frames.reshape(B * T, C, H, W)
        spatial_feats = model.spatial_encoder(x_flat)
        spatial_feats = spatial_feats.reshape(B, T, -1)
        
        # 2. Difference encoding
        diff_feats = model.diff_encoder(frames)
        
        # 3. Segment embeddings + distances
        seg_emb, embed_dists = model.segment_embedding(spatial_feats)
        
        # 4. Feature fusion
        fused = torch.cat([spatial_feats, diff_feats, embed_dists], dim=-1)
        fused = model.feature_proj(fused)
        
        # 5. Temporal processing
        temporal_feats = fused
        for block in model.temporal_blocks:
            temporal_feats = block(temporal_feats)
        
        # 6. Final predictions
        b_logits, t_logits, e_scores = model.detection_head(temporal_feats, embed_dists)
        
    return {
        'spatial_feats': spatial_feats.cpu(),
        'diff_feats': diff_feats.cpu(),
        'embed_dists': embed_dists.cpu(),
        'segment_embeddings': seg_emb.cpu(),
        'temporal_feats': temporal_feats.cpu(),
        'boundary_logits': b_logits.cpu(),
        'type_logits': t_logits.cpu(),
        'event_scores': e_scores.cpu(),
    }


def analyze_region(frames, internal_feats, start, end, region_type):
    """
    Analyze a specific region (TP/FP/FN gradual).
    
    Returns dict with statistics about internal representations.
    """
    # Extract region
    T = frames.shape[1]
    start = max(0, start)
    end = min(T, end)
    
    if end <= start:
        return {}
    
    analysis = {
        'region_type': region_type,
        'start': start,
        'end': end,
        'duration': end - start,
    }
    
    # Embedding distance patterns
    embed_dist_region = internal_feats['embed_dists'][0, start:end]  # [T, 64]
    analysis['embed_dist_mean'] = embed_dist_region.mean().item()
    analysis['embed_dist_std'] = embed_dist_region.std().item()
    analysis['embed_dist_max'] = embed_dist_region.max().item()
    analysis['embed_dist_trend'] = (embed_dist_region[-1] - embed_dist_region[0]).mean().item()
    
    # Segment embedding similarity (how much content changes)
    seg_emb_region = internal_feats['segment_embeddings'][0, start:end]  # [T, 128]
    seg_emb_norm = F.normalize(seg_emb_region, p=2, dim=-1)
    
    if len(seg_emb_norm) > 1:
        # Cosine similarity between consecutive frames
        sim = (seg_emb_norm[:-1] * seg_emb_norm[1:]).sum(dim=-1)
        analysis['seg_similarity_mean'] = sim.mean().item()
        analysis['seg_similarity_std'] = sim.std().item()
        
        # Overall drift: similarity between first and last frame
        if len(seg_emb_norm) > 2:
            overall_sim = (seg_emb_norm[0] * seg_emb_norm[-1]).sum().item()
            analysis['seg_drift'] = 1.0 - overall_sim  # Higher = more drift
    
    # Diff features (motion patterns)
    diff_region = internal_feats['diff_feats'][0, start:end]  # [T, 128]
    analysis['diff_feat_mean'] = diff_region.mean().item()
    analysis['diff_feat_std'] = diff_region.std().item()
    analysis['diff_feat_variance'] = diff_region.var(dim=0).mean().item()  # Consistency across time
    
    # Type predictions
    type_logits_region = internal_feats['type_logits'][0, start:end]  # [T, 3]
    type_probs = F.softmax(type_logits_region, dim=-1)
    
    analysis['type_none_prob'] = type_probs[:, 0].mean().item()
    analysis['type_cut_prob'] = type_probs[:, 1].mean().item()
    analysis['type_gradual_prob'] = type_probs[:, 2].mean().item()
    
    # Boundary predictions
    boundary_probs = torch.sigmoid(internal_feats['boundary_logits'][0, start:end].squeeze(-1))
    analysis['boundary_prob_mean'] = boundary_probs.mean().item()
    analysis['boundary_prob_max'] = boundary_probs.max().item()
    
    # Raw frame differences (ground truth motion)
    frame_region = frames[0, start:end]  # [T, 3, H, W]
    if len(frame_region) > 1:
        raw_diffs = torch.diff(frame_region, dim=0)
        raw_diff_norms = raw_diffs.pow(2).mean(dim=(1, 2, 3)).sqrt()
        analysis['raw_diff_mean'] = raw_diff_norms.mean().item()
        analysis['raw_diff_std'] = raw_diff_norms.std().item()
    
    return analysis


def deep_analyze_failures(model, dataloader, device, boundary_threshold=0.55, max_samples=50):
    """
    Deep analysis of model's internal representations for gradual failures.
    """
    model.eval()
    
    tp_analyses = []  # True positive graduals
    fp_analyses = []  # False positive graduals
    fn_analyses = []  # False negative graduals
    
    num_clips = 0
    
    with torch.no_grad():
        for frames, labels, lengths in tqdm(dataloader, desc="Deep Analysis"):
            if num_clips >= max_samples:
                break
            
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Get predictions
            b_logits, t_logits, e_scores, seg_emb = model(frames)
            boundary_probs = torch.sigmoid(b_logits.squeeze(-1))
            is_boundary = boundary_probs > boundary_threshold
            type_preds = torch.argmax(t_logits, dim=-1)
            preds_3class = torch.where(is_boundary, type_preds, torch.zeros_like(type_preds))
            
            # Extract internal features
            internal_feats = extract_internal_features(model, frames)
            
            for b in range(frames.shape[0]):
                if num_clips >= max_samples:
                    break
                num_clips += 1
                
                # Get transitions
                pred_graduals = extract_transitions_from_predictions(preds_3class[b].cpu(), min_gap=3)['graduals']
                gt_graduals_raw = extract_transitions_from_predictions(labels[b].cpu(), min_gap=3)['graduals']
                
                # Convert GT to segments
                gt_graduals = []
                in_gradual = False
                seg_start = 0
                label_arr = labels[b].cpu().numpy()
                
                for i in range(len(label_arr)):
                    if label_arr[i] == 2:
                        if not in_gradual:
                            in_gradual = True
                            seg_start = i
                    else:
                        if in_gradual:
                            gt_graduals.append({'start': seg_start, 'end': i-1, 'center': (seg_start + i - 1) // 2})
                            in_gradual = False
                
                if in_gradual:
                    gt_graduals.append({'start': seg_start, 'end': len(label_arr)-1, 'center': (seg_start + len(label_arr) - 1) // 2})
                
                # Analyze ground truth graduals
                for gt in gt_graduals:
                    # Check if detected
                    matched = False
                    for pred_center in pred_graduals:
                        if abs(pred_center - gt['center']) <= 3:
                            matched = True
                            break
                    
                    region_analysis = analyze_region(
                        frames[b:b+1].cpu(),
                        {k: v[b:b+1] for k, v in internal_feats.items()},
                        gt['start'], gt['end'] + 1,
                        'TP' if matched else 'FN'
                    )
                    
                    if matched:
                        tp_analyses.append(region_analysis)
                    else:
                        fn_analyses.append(region_analysis)
                
                # Analyze predicted graduals (check for FPs)
                for pred_center in pred_graduals:
                    # Check if it matches any GT
                    matched = False
                    for gt in gt_graduals:
                        if abs(pred_center - gt['center']) <= 3:
                            matched = True
                            break
                    
                    if not matched:
                        # False positive - analyze window around prediction
                        window_size = 10
                        start = max(0, pred_center - window_size // 2)
                        end = min(frames.shape[1], pred_center + window_size // 2)
                        
                        region_analysis = analyze_region(
                            frames[b:b+1].cpu(),
                            {k: v[b:b+1] for k, v in internal_feats.items()},
                            start, end,
                            'FP'
                        )
                        fp_analyses.append(region_analysis)
    
    return {
        'true_positives': tp_analyses,
        'false_positives': fp_analyses,
        'false_negatives': fn_analyses,
    }


def compare_statistics(tp_data, fp_data, fn_data):
    """Compare statistics across TP/FP/FN."""
    def get_stats(data, key):
        values = [d[key] for d in data if key in d]
        if not values:
            return None
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    keys = [
        'embed_dist_mean', 'embed_dist_std', 'seg_drift',
        'seg_similarity_mean', 'diff_feat_variance',
        'type_gradual_prob', 'boundary_prob_max',
        'raw_diff_mean', 'raw_diff_std',
    ]
    
    comparison = {}
    for key in keys:
        comparison[key] = {
            'TP': get_stats(tp_data, key),
            'FP': get_stats(fp_data, key),
            'FN': get_stats(fn_data, key),
        }
    
    return comparison


def print_insights(comparison):
    """Print key insights about model confusion."""
    print("\n" + "="*80)
    print("MODEL CONFUSION ANALYSIS - Internal Representations")
    print("="*80)
    
    critical_features = [
        ('seg_drift', 'Semantic Content Drift', 'Higher=more content change'),
        ('embed_dist_mean', 'Embedding Distance', 'Model\'s learned semantic change metric'),
        ('raw_diff_mean', 'Raw Frame Difference', 'Ground truth motion intensity'),
        ('diff_feat_variance', 'Diff Feature Variance', 'Temporal consistency of motion'),
        ('seg_similarity_mean', 'Segment Similarity', 'Lower=more different'),
        ('type_gradual_prob', 'Model Gradual Confidence', 'What model thinks'),
    ]
    
    for key, name, desc in critical_features:
        print(f"\n{name} ({desc}):")
        if key in comparison:
            tp_val = comparison[key].get('TP')
            fp_val = comparison[key].get('FP')
            fn_val = comparison[key].get('FN')
            
            if tp_val:
                print(f"  True Positives:  {tp_val['mean']:.4f} ± {tp_val['std']:.4f}")
            if fp_val:
                print(f"  False Positives: {fp_val['mean']:.4f} ± {fp_val['std']:.4f}")
                if tp_val:
                    ratio = fp_val['mean'] / (tp_val['mean'] + 1e-8)
                    print(f"                   ({ratio:.2f}x vs TP)")
            if fn_val:
                print(f"  False Negatives: {fn_val['mean']:.4f} ± {fn_val['std']:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=r'outputs\casbd\best_model.pt')
    parser.add_argument('--cache_path', type=str, default=r'D:\grad\data\training_cache')
    parser.add_argument('--output_dir', type=str, default='outputs/model_confusion_analysis')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=350)
    parser.add_argument('--threshold', type=float, default=0.55)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    if 'model' in ckpt:
        model = ckpt['model']
        print("Loaded full model from checkpoint")
    else:
        model = create_context_aware_detector(
            main_dim=ckpt.get('config', {}).get('main_dim', 384),
            num_blocks=ckpt.get('config', {}).get('num_blocks', 4),
            embed_dim=ckpt.get('config', {}).get('embed_dim', 128),
        )
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("Loaded state dict from checkpoint")
    
    model.to(device)
    model.eval()
    
    # Load data
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
    
    # Deep analysis
    print("\nAnalyzing model's internal representations...")
    results = deep_analyze_failures(
        model, val_loader, device,
        boundary_threshold=args.threshold,
        max_samples=args.max_samples,
    )
    
    print(f"\nFound:")
    print(f"  True Positives:  {len(results['true_positives'])}")
    print(f"  False Positives: {len(results['false_positives'])}")
    print(f"  False Negatives: {len(results['false_negatives'])}")
    
    # Compare statistics
    comparison = compare_statistics(
        results['true_positives'],
        results['false_positives'],
        results['false_negatives'],
    )
    
    # Print insights
    print_insights(comparison)
    
    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        'config': {
            'checkpoint': args.checkpoint,
            'threshold': args.threshold,
            'samples': args.max_samples,
        },
        'counts': {
            'tp': len(results['true_positives']),
            'fp': len(results['false_positives']),
            'fn': len(results['false_negatives']),
        },
        'comparison': comparison,
        'raw_tp': results['true_positives'][:20],
        'raw_fp': results['false_positives'][:20],
        'raw_fn': results['false_negatives'][:20],
    }
    
    output_file = output_dir / 'model_confusion_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nSaved detailed analysis to: {output_file}")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
