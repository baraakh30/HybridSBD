"""
Generate Comprehensive Presentation Materials for Hybrid Boundary Detector

This script generates:
1. Performance analysis on top videos (high transitions, high F1)
2. Visualization of detection results (clean 2-panel plots) - DISABLED
3. Feature analysis and statistics
4. Architecture diagrams
5. Comparison tables
6. Confusion matrix (counts + normalized)
7. Per-category performance (F1, Precision, Recall)
8. Dataset infographic (one-page summary)
9. Frame montages (with LARGE labels for projection)
10. Temporal heatmaps - DISABLED
11. Transition timelines
12. Demo video clips (MP4) with prediction overlays
13. Performance metrics (FPS, throughput, real-time factor)

All visualizations are PRESENTATION-READY with:
- Large fonts (readable during projection)
- Bold labels on frame montages (CUT/GRADUAL/NONE)
- Clean layouts without clutter
- Professional color schemes

PERFORMANCE TRACKING:
- Measures inference throughput (frames/second)
- Calculates real-time performance factor
- Tracks per-clip inference time
- Compares against 24 FPS baseline
- Cached with analysis results for instant loading
- Use --measure_performance to re-measure even with cached analysis

SMART CACHING:
- First run: Analyzes all videos and caches results (includes performance metrics)
- Subsequent runs: Loads from cache (instant!)
- Old caches without performance: Auto-measures on first load
- Use --force_reanalyze to recompute if model/data changes
- Use --measure_performance to only re-measure performance

Usage:
    # First run (analyzes all videos + measures performance)
    python generate_presentation_materials.py --checkpoint outputs/hybrid_detector5/best_model.pt --split val
    
    # Subsequent runs (uses cached analysis + performance)
    python generate_presentation_materials.py --checkpoint outputs/hybrid_detector5/best_model.pt --split val --max_visualizations 20
    
    # Re-measure performance only (fast, doesn't re-analyze)
    python generate_presentation_materials.py --checkpoint outputs/hybrid_detector5/best_model.pt --split val --measure_performance
    
    # Force full re-analysis (when model changes)
    python generate_presentation_materials.py --checkpoint outputs/hybrid_detector6/best_model.pt --split val --force_reanalyze
"""

import os
import sys
import argparse
import json
import pickle
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import cv2
from collections import defaultdict
import pandas as pd
from sklearn.metrics import confusion_matrix

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.hybrid_boundary_detector import create_hybrid_detector
from train_boundary_detector import PredecodedDataset
from train_turbo_sbd_v2 import extract_transitions_from_predictions, compute_transition_metrics


class PresentationMaterialGenerator:
    """Generate all materials needed for presentation"""
    
    def __init__(self, checkpoint_path, output_dir="presentation_materials"):
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "visualizations").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "videos").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "cache").mkdir(exist_ok=True, parents=True)
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self.load_model()
        
        # Load dataset
        self.cache_path = r"D:\grad\data\training_cache"
        self.dataset = None
        
    def load_model(self):
        """Load trained model from checkpoint"""
        print(f"\nLoading model from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device,weights_only=False)
        
        config = checkpoint.get('config', {})
        main_dim = config.get('main_dim', 256)
        num_fusion_blocks = config.get('num_fusion_blocks', 3)
        
        model = create_hybrid_detector(
            main_dim=main_dim,
            num_fusion_blocks=num_fusion_blocks
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best F1: {checkpoint.get('best_val_f1', 'unknown'):.4f}")
        
        return model
    
    def analyze_videos(self, split='test', top_k=10, force_reanalyze=False, measure_performance=False):
        """
        Analyze all videos and find top performers
        Uses cached results if available unless force_reanalyze=True
        
        Args:
            measure_performance: If True, measure performance even when using cache
        
        Returns:
            DataFrame with video stats sorted by F1 and transition count
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING {split.upper()} VIDEOS")
        print(f"{'='*80}")
        
        # Check for cached results
        cache_file = self.output_dir / "cache" / f"analysis_{split}.pkl"
        
        if cache_file.exists() and not force_reanalyze:
            print(f"\n✓ Loading cached analysis from: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                df = cached_data['dataframe']
                self.dataset = PredecodedDataset(self.cache_path, split=split)
                
                print(f"✓ Loaded {len(df)} videos from cache")
                print(f"  Total transitions: {df['total_transitions'].sum()}")
                print(f"  Avg F1: {df['overall_f1'].mean():.3f}")
                
                # Handle performance metrics
                if 'performance' in cached_data and not measure_performance:
                    # Use cached performance metrics
                    perf = cached_data['performance']
                    print(f"\n⚡ CACHED PERFORMANCE METRICS:")
                    print(f"  Throughput: {perf['frames_per_second']:.2f} FPS")
                    print(f"  Real-time factor: {perf['real_time_factor']:.2f}x")
                    print(f"  Device: {perf['device']}")
                    
                    # Save performance JSON if it doesn't exist
                    perf_path = self.output_dir / "statistics" / f"performance_metrics_{split}.json"
                    if not perf_path.exists():
                        with open(perf_path, 'w') as f:
                            json.dump(perf, f, indent=2)
                        print(f"  ✓ Performance metrics saved: {perf_path}")
                else:
                    # Need to measure performance
                    if measure_performance:
                        print(f"\n⟳ Re-measuring performance metrics...")
                    else:
                        print(f"\n⚡ No cached performance metrics, measuring now...")
                    
                    perf = self._measure_performance_on_dataset(self.dataset, split)
                    
                    # Update cache with performance metrics
                    cached_data['performance'] = perf
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cached_data, f)
                    print(f"  ✓ Cache updated with performance metrics")
                
                # Print top videos
                print(f"\n{'='*80}")
                print(f"TOP {top_k} VIDEOS (by F1 × log(transitions))")
                print(f"{'='*80}")
                
                for idx, row in df.head(top_k).iterrows():
                    print(f"\n{row['video_name']}")
                    print(f"  Transitions: {row['total_transitions']} (Cuts: {row['num_cuts']}, Graduals: {row['num_graduals']})")
                    print(f"  Overall F1: {row['overall_f1']:.3f} (P: {row['overall_precision']:.3f}, R: {row['overall_recall']:.3f})")
                    print(f"  Cut F1: {row['cut_f1']:.3f}, Gradual F1: {row['gradual_f1']:.3f}")
                
                return df
                
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}")
                print(f"  Re-analyzing from scratch...")
        else:
            if force_reanalyze:
                print(f"\n⟳ Force re-analysis requested")
            else:
                print(f"\n✗ No cache found at: {cache_file}")
            print(f"  Analyzing all videos...")
        
        # Check if test dataset exists (if using test split)
        if split == 'test':
            test_mapping = Path(self.cache_path) / 'test' / 'index_mapping.json'
            if not test_mapping.exists():
                print(f"\n⚠️  ERROR: Test dataset not pre-decoded!")
                print(f"{'='*80}")
                print(f"The test split cache was not found at:")
                print(f"  {test_mapping}")
                print(f"\nPlease run the following command first:")
                print(f"  python predecode_dataset.py --split test --num_workers 8")
                print(f"\nAlternatively, use --split val to analyze validation set:")
                print(f"  python generate_presentation_materials.py --checkpoint {self.checkpoint_path} --split val")
                print(f"{'='*80}")
                raise FileNotFoundError(f"Test dataset not found. Please pre-decode it first.")
        
        # Load dataset
        dataset = PredecodedDataset(self.cache_path, split=split)
        self.dataset = dataset
        
        # Collect all videos with their clips
        video_clips = defaultdict(list)
        for idx, clip_info in enumerate(dataset.mapping):
            video_name = clip_info['video_name']
            video_clips[video_name].append(idx)
        
        print(f"Found {len(video_clips)} videos with {len(dataset)} total clips")
        
        # Analyze each video
        results = []
        
        # Performance tracking
        total_frames = 0
        total_inference_time = 0.0
        
        for video_name, clip_indices in tqdm(video_clips.items(), desc="Analyzing videos"):
            # Process all clips for this video
            all_preds = []
            all_gts = []
            
            for idx in clip_indices:
                frames, targets, lengths = dataset[idx]
                frames = frames.unsqueeze(0).to(self.device)  # [1, T, C, H, W]
                
                # Measure inference time
                start_time = time.time()
                with torch.no_grad():
                    boundary_logits, type_logits, _, _ = self.model(frames)
                inference_time = time.time() - start_time
                
                total_frames += frames.shape[1]  # T frames
                total_inference_time += inference_time
                
                boundary_probs = torch.sigmoid(boundary_logits).squeeze()
                type_preds = torch.argmax(type_logits, dim=-1).squeeze()
                
                # Combine: boundary detection + type classification
                is_boundary = boundary_probs > 0.5
                preds = torch.where(is_boundary, type_preds, torch.zeros_like(type_preds))
                
                all_preds.append(preds.cpu())
                all_gts.append(targets)
            
            # Concatenate all clips
            video_preds = torch.cat(all_preds)
            video_gts = torch.cat(all_gts)
            
            # Extract transitions
            pred_trans = extract_transitions_from_predictions(video_preds, min_gap=3)
            gt_trans = extract_transitions_from_predictions(video_gts, min_gap=3)
            
            # Compute metrics for cuts
            cut_metrics = compute_transition_metrics(
                pred_trans['cuts'], gt_trans['cuts'], tolerance=3
            )
            
            # Compute metrics for graduals
            gradual_metrics = compute_transition_metrics(
                pred_trans['graduals'], gt_trans['graduals'], tolerance=3
            )
            
            # Overall metrics
            overall_tp = cut_metrics.get('tp', 0) + gradual_metrics.get('tp', 0)
            overall_fp = cut_metrics.get('fp', 0) + gradual_metrics.get('fp', 0)
            overall_fn = cut_metrics.get('fn', 0) + gradual_metrics.get('fn', 0)
            
            overall_precision = overall_tp / (overall_tp + overall_fp + 1e-8)
            overall_recall = overall_tp / (overall_tp + overall_fn + 1e-8)
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-8)
            
            # Count transitions
            num_cuts = len(gt_trans['cuts'])
            num_graduals = len(gt_trans['graduals'])
            total_transitions = num_cuts + num_graduals
            
            results.append({
                'video_name': video_name,
                'num_clips': len(clip_indices),
                'total_frames': len(video_gts),
                'num_cuts': num_cuts,
                'num_graduals': num_graduals,
                'total_transitions': total_transitions,
                'cut_precision': cut_metrics.get('tp', 0) / (cut_metrics.get('tp', 0) + cut_metrics.get('fp', 0) + 1e-8),
                'cut_recall': cut_metrics.get('tp', 0) / (cut_metrics.get('tp', 0) + cut_metrics.get('fn', 0) + 1e-8),
                'cut_f1': 2 * cut_metrics.get('tp', 0) / (2 * cut_metrics.get('tp', 0) + cut_metrics.get('fp', 0) + cut_metrics.get('fn', 0) + 1e-8),
                'gradual_precision': gradual_metrics.get('tp', 0) / (gradual_metrics.get('tp', 0) + gradual_metrics.get('fp', 0) + 1e-8),
                'gradual_recall': gradual_metrics.get('tp', 0) / (gradual_metrics.get('tp', 0) + gradual_metrics.get('fn', 0) + 1e-8),
                'gradual_f1': 2 * gradual_metrics.get('tp', 0) / (2 * gradual_metrics.get('tp', 0) + gradual_metrics.get('fp', 0) + gradual_metrics.get('fn', 0) + 1e-8),
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1': overall_f1,
                'clip_indices': clip_indices
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Round F1 scores to avoid floating point precision issues (0.999999... → 1.0)
        f1_columns = ['cut_precision', 'cut_recall', 'cut_f1', 
                      'gradual_precision', 'gradual_recall', 'gradual_f1',
                      'overall_precision', 'overall_recall', 'overall_f1']
        for col in f1_columns:
            df[col] = df[col].round(4)
        
        # Sort by F1 and transition count
        df['score'] = df['overall_f1'] * (1 + np.log1p(df['total_transitions']))
        df = df.sort_values('score', ascending=False)
        
        # Calculate performance metrics
        fps = total_frames / total_inference_time if total_inference_time > 0 else 0
        real_time_factor = fps / 30  # Assuming 30 fps video
        avg_inference_per_clip = total_inference_time / len(dataset)
        
        performance_stats = {
            'total_frames_processed': total_frames,
            'total_inference_time_sec': round(total_inference_time, 2),
            'frames_per_second': round(fps, 2),
            'real_time_factor': round(real_time_factor, 2),
            'avg_inference_per_clip_ms': round(avg_inference_per_clip * 1000, 2),
            'device': str(self.device)
        }
        
        # Save full results
        csv_path = self.output_dir / "statistics" / f"{split}_video_analysis.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Video analysis saved: {csv_path}")
        
        # Save performance metrics
        perf_path = self.output_dir / "statistics" / f"performance_metrics_{split}.json"
        with open(perf_path, 'w') as f:
            json.dump(performance_stats, f, indent=2)
        print(f"✓ Performance metrics saved: {perf_path}")
        
        # Print performance summary
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  Throughput: {fps:.2f} FPS (batch_size=1)")
        print(f"  Real-time factor: {real_time_factor:.2f}x (30 fps video baseline)")
        print(f"  Avg inference per clip: {avg_inference_per_clip*1000:.2f} ms")
        print(f"  Total frames processed: {total_frames:,}")
        print(f"  Total inference time: {total_inference_time:.2f} sec")
        print(f"  Device: {self.device}")
        print(f"  Note: Sequential processing. Batch inference would be ~3x faster.")
        
        # Cache results for future use
        cache_file = self.output_dir / "cache" / f"analysis_{split}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'dataframe': df,
                'performance': performance_stats,
                'checkpoint_path': str(self.checkpoint_path),
                'split': split
            }, f)
        print(f"✓ Analysis cached: {cache_file}")
        
        # Print top videos
        print(f"\n{'='*80}")
        print(f"TOP {top_k} VIDEOS (by F1 × log(transitions))")
        print(f"{'='*80}")
        
        for idx, row in df.head(top_k).iterrows():
            print(f"\n{row['video_name']}")
            print(f"  Transitions: {row['total_transitions']} (Cuts: {row['num_cuts']}, Graduals: {row['num_graduals']})")
            print(f"  Overall F1: {row['overall_f1']:.3f} (P: {row['overall_precision']:.3f}, R: {row['overall_recall']:.3f})")
            print(f"  Cut F1: {row['cut_f1']:.3f}, Gradual F1: {row['gradual_f1']:.3f}")
        
        return df
    
    def _measure_performance_on_dataset(self, dataset, split):
        """Measure inference performance on dataset without re-analyzing"""
        print(f"  Measuring performance on {len(dataset)} clips...")
        
        total_frames = 0
        total_inference_time = 0.0
        
        for idx in tqdm(range(len(dataset)), desc="Performance measurement"):
            frames, targets, lengths = dataset[idx]
            frames = frames.unsqueeze(0).to(self.device)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(frames)
            inference_time = time.time() - start_time
            
            total_frames += frames.shape[1]
            total_inference_time += inference_time
        
        # Calculate performance metrics
        fps = total_frames / total_inference_time if total_inference_time > 0 else 0
        real_time_factor = fps / 30  # Assuming 30 fps video
        avg_inference_per_clip = total_inference_time / len(dataset)
        
        performance_stats = {
            'total_frames_processed': total_frames,
            'total_inference_time_sec': round(total_inference_time, 2),
            'frames_per_second': round(fps, 2),
            'real_time_factor': round(real_time_factor, 2),
            'avg_inference_per_clip_ms': round(avg_inference_per_clip * 1000, 2),
            'device': str(self.device)
        }
        
        # Save performance metrics
        perf_path = self.output_dir / "statistics" / f"performance_metrics_{split}.json"
        with open(perf_path, 'w') as f:
            json.dump(performance_stats, f, indent=2)
        print(f"  ✓ Performance metrics saved: {perf_path}")
        
        # Print performance summary
        print(f"\n  ⚡ PERFORMANCE METRICS:")
        print(f"    Throughput: {fps:.2f} FPS (batch_size=1)")
        print(f"    Real-time factor: {real_time_factor:.2f}x (30 fps video baseline)")
        print(f"    Avg inference per clip: {avg_inference_per_clip*1000:.2f} ms")
        print(f"    Total frames processed: {total_frames:,}")
        print(f"    Total inference time: {total_inference_time:.2f} sec")
        print(f"    Device: {self.device}")
        print(f"    Note: Sequential processing. Batch inference would be ~3x faster.")
        
        return performance_stats
    
    def visualize_detections(self, video_name, clip_idx):
        """
        Create visualization of detection results for a clip
        Clean presentation-ready visualization without feature details
        """
        print(f"\nVisualizing detections for {video_name}, clip {clip_idx}...")
        
        # Load clip
        frames, targets, lengths = self.dataset[clip_idx]
        frames_input = frames.unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            boundary_logits, type_logits, _, _ = self.model(frames_input)
        
        boundary_probs = torch.sigmoid(boundary_logits).squeeze().cpu().numpy()
        type_probs = torch.softmax(type_logits, dim=-1).squeeze().cpu().numpy()
        type_preds = np.argmax(type_probs, axis=-1)
        
        targets_np = targets.cpu().numpy()
        
        # Create clean figure with only predictions
        fig = plt.figure(figsize=(24, 10))
        
        # 1. Boundary probabilities
        ax1 = plt.subplot(2, 1, 1)
        frames_range = np.arange(len(boundary_probs))
        ax1.plot(frames_range, boundary_probs, 'b-', linewidth=3, label='Boundary Probability')
        ax1.axhline(0.5, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Threshold')
        
        # Mark ground truth transitions
        for i, target in enumerate(targets_np):
            if target == 1:  # Cut
                ax1.axvline(i, color='orange', alpha=0.4, linewidth=2)
            elif target == 2:  # Gradual
                ax1.axvline(i, color='purple', alpha=0.4, linewidth=2)
        
        ax1.set_ylabel('Probability', fontsize=18, fontweight='bold')
        ax1.set_title('Boundary Detection', fontsize=22, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=15)
        ax1.grid(alpha=0.3, linewidth=1.5)
        ax1.set_xlim([0, len(boundary_probs)])
        ax1.set_ylim([0, 1])
        ax1.tick_params(labelsize=14)
        
        # 2. Type predictions
        ax2 = plt.subplot(2, 1, 2)
        
        # Show predicted types
        colors = ['lightgray', 'orange', 'purple']
        labels = ['None', 'Cut', 'Gradual']
        
        for i, (pred, gt) in enumerate(zip(type_preds, targets_np)):
            # Predicted type
            ax2.add_patch(patches.Rectangle((i, 0), 1, 0.45, 
                                           facecolor=colors[pred], 
                                           edgecolor='black', linewidth=0.5))
            # Ground truth
            ax2.add_patch(patches.Rectangle((i, 0.55), 1, 0.45,
                                           facecolor=colors[gt],
                                           edgecolor='black', linewidth=0.5))
        
        ax2.set_ylabel('Type', fontsize=18, fontweight='bold')
        ax2.set_title('Type Classification (Top: Prediction, Bottom: Ground Truth)', fontsize=18)
        ax2.set_xlim([0, len(type_preds)])
        ax2.set_ylim([0, 1])
        ax2.set_yticks([0.225, 0.775])
        ax2.set_yticklabels(['Predicted', 'Ground Truth'], fontsize=14)
        ax2.set_xlabel('Frame', fontsize=18, fontweight='bold')
        ax2.tick_params(labelsize=14)
        
        # Legend
        legend_elements = [patches.Patch(facecolor=colors[i], label=labels[i], edgecolor='black', linewidth=1) 
                          for i in range(3)]
        ax2.legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=15)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / "visualizations" / f"{video_name}_clip{clip_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Visualization saved: {save_path}")
        
        return save_path
    
    def create_frame_montage(self, video_name, clip_idx, num_frames=64):
        """
        Create a frame montage visualization (common in papers)
        Shows a grid of uniformly sampled frames with transition markers
        PRESENTATION-READY: Large, bold labels visible during projection
        """
        print(f"\nCreating frame montage for {video_name}, clip {clip_idx}...")
        
        # Load clip
        frames, targets, lengths = self.dataset[clip_idx]
        T = len(frames)
        
        # Sample frames uniformly
        indices = np.linspace(0, T-1, num_frames, dtype=int)
        
        # Create grid (e.g., 8x8 for 64 frames, or 6x8 for 48 frames)
        rows = 8
        cols = num_frames // rows
        
        fig, axes = plt.subplots(rows, cols, figsize=(28, 28))
        fig.suptitle(f'Frame Montage ({num_frames} frames)', fontsize=24, fontweight='bold', y=0.995)
        
        for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
            # Get frame and convert to displayable format
            frame = frames[idx].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            
            ax.imshow(frame)
            ax.axis('off')
            
            # Add frame number and transition label with MUCH BIGGER fonts
            target = targets[idx].item()
            if target == 0:
                color, label = 'darkgreen', 'NONE'
            elif target == 1:
                color, label = 'darkorange', 'CUT'
            else:
                color, label = 'darkviolet', 'GRADUAL'
            
            # Frame number - bigger and more visible
            ax.text(3, 8, f'F{idx}', fontsize=14, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.85, edgecolor='white', linewidth=2))
            
            # Transition label - MUCH BIGGER for projection visibility
            ax.text(frame.shape[1]/2, frame.shape[0]-8, label, fontsize=18, color='white', 
                   fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round,pad=0.7', facecolor=color, alpha=0.95, edgecolor='white', linewidth=2.5),
                   verticalalignment='bottom')
        
        plt.tight_layout()
        
        save_path = self.output_dir / "visualizations" / f"{video_name}_montage_clip{clip_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Frame montage saved: {save_path}")
        return save_path
    
    def create_temporal_heatmap(self, video_name, clip_idx):
        """
        Create temporal heatmap visualization (common in papers)
        Shows all frames in a single row with probability heatmap
        PRESENTATION-READY: Larger fonts and clearer visualization
        """
        print(f"\nCreating temporal heatmap for {video_name}, clip {clip_idx}...")
        
        # Load clip
        frames, targets, lengths = self.dataset[clip_idx]
        frames_input = frames.unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            boundary_logits, type_logits, features = self.model(frames_input, return_features=True)
        
        boundary_probs = torch.sigmoid(boundary_logits).squeeze().cpu().numpy()
        type_probs = torch.softmax(type_logits, dim=-1).squeeze().cpu().numpy()
        
        T, C, H, W = frames.shape
        
        # Downsample frames for visualization
        step = max(1, T // 128)  # Max 128 frames wide
        downsample_indices = np.arange(0, T, step)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 10))
        fig.suptitle('Temporal Heatmap', fontsize=20, fontweight='bold')
        
        # 1. Frame strip (all frames in one row)
        frame_strip = []
        for idx in downsample_indices:
            frame = frames[idx].permute(1, 2, 0).cpu().numpy()
            # Resize to consistent height
            frame_resized = cv2.resize(frame, (int(W * 32 / H), 32))
            frame_strip.append(frame_resized)
        
        frame_strip = np.concatenate(frame_strip, axis=1)
        ax1.imshow(frame_strip)
        ax1.set_title('Frame Sequence', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # 2. Boundary probability heatmap
        boundary_heatmap = np.repeat(boundary_probs[:, np.newaxis], 20, axis=1).T
        im2 = ax2.imshow(boundary_heatmap, cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Boundary Probability', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Probability', fontsize=14, fontweight='bold')
        ax2.set_yticks([])
        ax2.tick_params(labelsize=12)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.02)
        cbar2.ax.tick_params(labelsize=12)
        
        # 3. Type probability heatmap (3 rows: none, cut, gradual)
        type_heatmap = type_probs.T  # [3, T]
        im3 = ax3.imshow(type_heatmap, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax3.set_title('Type Probabilities', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Type', fontsize=14, fontweight='bold')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['None', 'Cut', 'Gradual'], fontsize=13)
        ax3.set_xlabel('Frame', fontsize=14, fontweight='bold')
        ax3.tick_params(labelsize=12)
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.02)
        cbar3.ax.tick_params(labelsize=12)
        
        # Mark ground truth transitions with thicker lines
        for idx, target in enumerate(targets.cpu().numpy()):
            if target > 0:
                for ax in [ax2, ax3]:
                    ax.axvline(idx, color='red', alpha=0.4, linewidth=1.5)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "visualizations" / f"{video_name}_heatmap_clip{clip_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Temporal heatmap saved: {save_path}")
        return save_path
    
    def create_transition_timeline(self, video_name, clip_indices):
        """
        Create a timeline showing all transitions across multiple clips
        (Like the long horizontal bars you see in papers)
        PRESENTATION-READY: Larger fonts and clearer visualization
        """
        print(f"\nCreating transition timeline for {video_name}...")
        
        all_preds = []
        all_gts = []
        all_boundary_probs = []
        
        # Process all clips for this video
        for clip_idx in clip_indices[:10]:  # Limit to first 10 clips for visualization
            frames, targets, lengths = self.dataset[clip_idx]
            frames_input = frames.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                boundary_logits, type_logits, _, _ = self.model(frames_input)
            
            boundary_probs = torch.sigmoid(boundary_logits).squeeze().cpu().numpy()
            type_preds = torch.argmax(type_logits, dim=-1).squeeze().cpu().numpy()
            
            all_boundary_probs.append(boundary_probs)
            all_preds.append(type_preds)
            all_gts.append(targets.cpu().numpy())
        
        # Concatenate
        boundary_probs_full = np.concatenate(all_boundary_probs)
        preds_full = np.concatenate(all_preds)
        gts_full = np.concatenate(all_gts)
        
        total_frames = len(gts_full)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(28, 8))
        fig.suptitle('Transition Timeline', fontsize=20, fontweight='bold')
        
        x_range = np.arange(total_frames)
        
        # 1. Ground truth
        colors_gt = np.array(['lightgray', 'orange', 'purple'])
        ax1.fill_between(x_range, 0, 1, color='lightgray', alpha=0.3)
        for i, label in enumerate(gts_full):
            if label > 0:
                ax1.axvline(i, color=colors_gt[label], alpha=0.8, linewidth=3)
        ax1.set_ylabel('Ground Truth', fontsize=15, fontweight='bold')
        ax1.set_xlim([0, total_frames])
        ax1.set_ylim([0, 1])
        ax1.set_yticks([])
        ax1.grid(alpha=0.3, linewidth=1.5)
        ax1.tick_params(labelsize=12)
        
        # 2. Boundary probabilities
        ax2.fill_between(x_range, boundary_probs_full, alpha=0.7, color='blue')
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=2.5, label='Threshold')
        ax2.set_ylabel('Boundary\nProbability', fontsize=15, fontweight='bold')
        ax2.set_xlim([0, total_frames])
        ax2.set_ylim([0, 1])
        ax2.legend(loc='upper right', fontsize=13)
        ax2.grid(alpha=0.3, linewidth=1.5)
        ax2.tick_params(labelsize=12)
        
        # 3. Predictions
        ax3.fill_between(x_range, 0, 1, color='lightgray', alpha=0.3)
        is_boundary = boundary_probs_full > 0.5
        pred_with_threshold = np.where(is_boundary, preds_full, 0)
        for i, label in enumerate(pred_with_threshold):
            if label > 0:
                ax3.axvline(i, color=colors_gt[label], alpha=0.8, linewidth=3)
        ax3.set_ylabel('Predictions', fontsize=15, fontweight='bold')
        ax3.set_xlabel('Frame', fontsize=16, fontweight='bold')
        ax3.set_xlim([0, total_frames])
        ax3.set_ylim([0, 1])
        ax3.set_yticks([])
        ax3.grid(alpha=0.3, linewidth=1.5)
        ax3.tick_params(labelsize=12)
        
        # Add legend with bigger fonts
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='orange', label='Cut', edgecolor='black', linewidth=1),
            Patch(facecolor='purple', label='Gradual', edgecolor='black', linewidth=1),
            Patch(facecolor='lightgray', label='None', edgecolor='black', linewidth=1)
        ]
        ax3.legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=14, framealpha=0.95)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "visualizations" / f"{video_name}_timeline.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Transition timeline saved: {save_path}")
        return save_path
    
    def generate_statistics(self, df):
        """Generate comprehensive statistics"""
        print(f"\n{'='*80}")
        print("GENERATING STATISTICS")
        print(f"{'='*80}")
        
        # Overall statistics - ONLY count videos WITH transitions
        videos_with_transitions = df[df['total_transitions'] > 0]
        videos_without_transitions = df[df['total_transitions'] == 0]
        
        # Separate videos by transition type for accurate metrics
        videos_with_cuts = df[df['num_cuts'] > 0]
        videos_with_graduals = df[df['num_graduals'] > 0]
        
        # Compute metrics ONLY on videos with transitions (meaningful)
        stats = {
            'total_videos': len(videos_with_transitions),  # Only count videos with transitions
            'videos_with_transitions': len(videos_with_transitions),
            'videos_without_transitions': len(videos_without_transitions),
            'videos_with_cuts': len(videos_with_cuts),
            'videos_with_graduals': len(videos_with_graduals),
            'total_transitions': int(videos_with_transitions['total_transitions'].sum()),
            'total_cuts': int(videos_with_transitions['num_cuts'].sum()),
            'total_graduals': int(videos_with_transitions['num_graduals'].sum()),
            'max_transitions_per_video': int(videos_with_transitions['total_transitions'].max()) if len(videos_with_transitions) > 0 else 0,
            'avg_transitions_per_video': round(float(videos_with_transitions['total_transitions'].mean()), 2) if len(videos_with_transitions) > 0 else 0,
            'median_transitions_per_video': float(videos_with_transitions['total_transitions'].median()) if len(videos_with_transitions) > 0 else 0,
            
            # All metrics computed ONLY on videos with transitions
            'avg_overall_f1_all': round(float(videos_with_transitions['overall_f1'].mean()), 4) if len(videos_with_transitions) > 0 else 0,
            'median_overall_f1_all': round(float(videos_with_transitions['overall_f1'].median()), 4) if len(videos_with_transitions) > 0 else 0,
            'avg_overall_f1_with_transitions': round(float(videos_with_transitions['overall_f1'].mean()), 4) if len(videos_with_transitions) > 0 else 0.0,
            'median_overall_f1_with_transitions': round(float(videos_with_transitions['overall_f1'].median()), 4) if len(videos_with_transitions) > 0 else 0.0,
            'videos_detected_with_transitions': int((videos_with_transitions['overall_f1'] > 0).sum()),
            'videos_missed_with_transitions': int((videos_with_transitions['overall_f1'] == 0).sum()),
            'detection_rate': round(float((videos_with_transitions['overall_f1'] > 0).sum() / len(videos_with_transitions) * 100), 2) if len(videos_with_transitions) > 0 else 0.0,
            
            # Cut F1: computed on videos WITH cuts (not all videos)
            'avg_cut_f1': round(float(videos_with_cuts['cut_f1'].mean()), 4) if len(videos_with_cuts) > 0 else 0,
            'median_cut_f1': round(float(videos_with_cuts['cut_f1'].median()), 4) if len(videos_with_cuts) > 0 else 0,
            
            # Gradual F1: computed ONLY on videos WITH graduals (not all videos!)
            'avg_gradual_f1': round(float(videos_with_graduals['gradual_f1'].mean()), 4) if len(videos_with_graduals) > 0 else 0,
            'median_gradual_f1': round(float(videos_with_graduals['gradual_f1'].median()), 4) if len(videos_with_graduals) > 0 else 0,
            
            # Old metric for comparison (includes zeros from videos without graduals)
            'avg_gradual_f1_all_videos': round(float(videos_with_transitions['gradual_f1'].mean()), 4) if len(videos_with_transitions) > 0 else 0,
            
            'top_10_avg_f1': round(float(videos_with_transitions.head(10)['overall_f1'].mean()), 4) if len(videos_with_transitions) > 0 else 0,
            'videos_with_high_f1': int((videos_with_transitions['overall_f1'] > 0.8).sum()),
            'videos_with_many_transitions': int((videos_with_transitions['total_transitions'] > 5).sum()),
        }
        
        # Save statistics
        stats_path = self.output_dir / "statistics" / "overall_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Statistics saved: {stats_path}")
        
        # Print summary
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total videos: {stats['total_videos']}")
        print(f"    ├─ With transitions: {stats['videos_with_transitions']} ({100*stats['videos_with_transitions']/stats['total_videos']:.1f}%)")
        print(f"    │  ├─ With cuts: {stats['videos_with_cuts']} ({100*stats['videos_with_cuts']/stats['total_videos']:.1f}%)")
        print(f"    │  └─ With graduals: {stats['videos_with_graduals']} ({100*stats['videos_with_graduals']/stats['total_videos']:.1f}%)")
        print(f"    └─ Without transitions: {stats['videos_without_transitions']} ({100*stats['videos_without_transitions']/stats['total_videos']:.1f}%)")
        print(f"\n  Total transitions: {stats['total_transitions']} (Cuts: {stats['total_cuts']}, Graduals: {stats['total_graduals']})")
        print(f"  Transitions per video: avg={stats['avg_transitions_per_video']:.1f}, median={stats['median_transitions_per_video']:.1f}, max={stats['max_transitions_per_video']}")
        
        print(f"\n  🎯 MODEL PERFORMANCE (videos WITH transitions only):")
        print(f"    Detection rate: {stats['detection_rate']:.1f}% ({stats['videos_detected_with_transitions']}/{stats['videos_with_transitions']} videos)")
        print(f"    Avg F1: {stats['avg_overall_f1_with_transitions']:.3f} (median: {stats['median_overall_f1_with_transitions']:.3f})")
        print(f"    Missed: {stats['videos_missed_with_transitions']} videos completely missed")
        
        print(f"\n  📊 PERFORMANCE BY TRANSITION TYPE:")
        print(f"    Cut F1 (on {stats['videos_with_cuts']} videos with cuts):")
        print(f"      Avg: {stats['avg_cut_f1']:.3f}, Median: {stats['median_cut_f1']:.3f}")
        print(f"    Gradual F1 (on {stats['videos_with_graduals']} videos with graduals):")
        print(f"      Avg: {stats['avg_gradual_f1']:.3f}, Median: {stats['median_gradual_f1']:.3f}")
        print(f"      (Note: Old metric including cut-only videos: {stats['avg_gradual_f1_all_videos']:.3f})")
        
        print(f"\n  Top performers:")
        print(f"    Top 10 videos avg F1: {stats['top_10_avg_f1']:.3f}")
        print(f"    Videos with F1 > 0.8: {stats['videos_with_high_f1']}")
        print(f"    Videos with > 5 transitions: {stats['videos_with_many_transitions']}")
        
        # Create performance distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # F1 distribution
        df_with_trans = df[df['total_transitions'] > 0]  # Only plot videos with transitions
        axes[0, 0].hist(df_with_trans['overall_f1'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(df_with_trans['overall_f1'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {df_with_trans["overall_f1"].mean():.3f}')
        axes[0, 0].set_xlabel('Overall F1', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('F1 Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Transitions distribution
        axes[0, 1].hist(df_with_trans['total_transitions'], bins=20, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(df_with_trans['total_transitions'].mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {df_with_trans["total_transitions"].mean():.1f}')
        axes[0, 1].set_xlabel('Number of Transitions', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Transitions per Video Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Cut vs Gradual F1
        axes[1, 0].scatter(df_with_trans['cut_f1'], df_with_trans['gradual_f1'], alpha=0.6, s=50)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1, 0].set_xlabel('Cut F1', fontsize=12)
        axes[1, 0].set_ylabel('Gradual F1', fontsize=12)
        axes[1, 0].set_title('Cut vs Gradual Performance', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_ylim([0, 1])
        
        # F1 vs Transitions
        axes[1, 1].scatter(df_with_trans['total_transitions'], df_with_trans['overall_f1'], alpha=0.6, s=50)
        axes[1, 1].set_xlabel('Number of Transitions', fontsize=12)
        axes[1, 1].set_ylabel('Overall F1', fontsize=12)
        axes[1, 1].set_title('F1 vs Number of Transitions', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        dist_path = self.output_dir / "statistics" / "performance_distribution.png"
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Distribution plots saved: {dist_path}")
        
        return stats
    
    def create_performance_visualization(self, split='val'):
        """Create performance metrics visualization"""
        print(f"\nGenerating performance metrics visualization...")
        
        # Load performance metrics
        perf_path = self.output_dir / "statistics" / f"performance_metrics_{split}.json"
        if not perf_path.exists():
            print(f"  ⚠ No performance metrics found at {perf_path}")
            return None
        
        with open(perf_path, 'r') as f:
            perf = json.load(f)
        
        fps = perf['frames_per_second']
        rt_factor = perf['real_time_factor']
        real_time_threshold = 30  # 30 FPS for real-time
        
        # Helper function to create the chart panels
        def create_charts(ax1, ax2):
            # 1. Throughput (FPS)
            bars = ax1.bar(['Throughput'], [fps], color='#3498db' if fps >= real_time_threshold else '#e74c3c', 
                          edgecolor='black', linewidth=2, width=0.5)
            ax1.axhline(real_time_threshold, color='green', linestyle='--', linewidth=2, 
                       label=f'Real-time (30 FPS)', alpha=0.7)
            ax1.set_ylabel('Frames Per Second', fontsize=16, fontweight='bold')
            ax1.set_title('Model Throughput', fontsize=18, fontweight='bold')
            ax1.set_ylim([0, max(fps * 1.2, real_time_threshold * 1.5)])
            ax1.legend(fontsize=12)
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value label
            ax1.text(0, fps + fps*0.05, f'{fps:.1f} FPS', ha='center', va='bottom', 
                    fontsize=20, fontweight='bold')
            
            # 2. Real-time Factor
            bars = ax2.bar(['RT Factor'], [rt_factor], 
                          color='#2ecc71' if rt_factor >= 1.0 else '#e74c3c',
                          edgecolor='black', linewidth=2, width=0.5)
            ax2.axhline(1.0, color='green', linestyle='--', linewidth=2, 
                       label='1.0x = Real-time', alpha=0.7)
            ax2.set_ylabel('Real-time Factor', fontsize=16, fontweight='bold')
            ax2.set_title('Real-time Performance', fontsize=18, fontweight='bold')
            ax2.set_ylim([0, max(rt_factor * 1.2, 2.0)])
            ax2.legend(fontsize=12)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value label
            status = '✓ Faster than real-time' if rt_factor >= 1.0 else '✗ Slower than real-time'
            ax2.text(0, rt_factor + rt_factor*0.05, f'{rt_factor:.2f}x\n{status}\n(30fps baseline)', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # VERSION 1: Without summary section (2 panels only)
        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))
        create_charts(axes1[0], axes1[1])
        plt.tight_layout()
        
        save_path_no_summary = self.output_dir / "statistics" / "performance_metrics_no_summary.png"
        plt.savefig(save_path_no_summary, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Performance visualization (no summary) saved: {save_path_no_summary}")
        
        # VERSION 2: With summary section (3 panels)
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
        create_charts(axes2[0], axes2[1])
        
        # 3. Processing Stats
        ax3 = axes2[2]
        ax3.axis('off')
        
        stats_text = f'''
        ⚡ PERFORMANCE SUMMARY
        
        Throughput: {perf['frames_per_second']:.2f} FPS
        Real-time Factor: {perf['real_time_factor']:.2f}x
        Batch Size: {perf.get('batch_size', 1)}
        
        Inference per clip: {perf['avg_inference_per_clip_ms']:.2f} ms
        
        Total Frames: {perf['total_frames_processed']:,}
        Total Time: {perf['total_inference_time_sec']:.2f} sec
        
        Device: {perf['device']}
        
        Status: {'✓ Real-time capable' if rt_factor >= 1.0 else '⚠ Sub-real-time'}
        
        Note: Sequential (bs=1). Batch
        processing ~3x faster.
        '''
        
        ax3.text(0.1, 0.9, stats_text, fontsize=14, verticalalignment='top',
                fontfamily='monospace', 
                bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', 
                         edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        
        save_path = self.output_dir / "statistics" / "performance_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Performance visualization (with summary) saved: {save_path}")
        return save_path, save_path_no_summary
    
    def create_comparison_table(self, df):
        """Create comparison table for top videos"""
        print(f"\nCreating comparison table...")
        
        # Filter to only videos with transitions for more meaningful comparison
        df_with_transitions = df[df['total_transitions'] > 0]
        
        # Select top videos (from those with transitions)
        top_videos = df_with_transitions.head(15)
        
        # Create table
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        table_data = []
        headers = ['Rank', 'Video', 'Transitions\n(C/G)', 'Overall\nF1', 'Precision', 'Recall', 
                  'Cut F1', 'Gradual F1']
        
        for idx, (_, row) in enumerate(top_videos.iterrows(), 1):
            table_data.append([
                idx,
                row['video_name'][:30],  # Truncate long names
                f"{row['total_transitions']}\n({row['num_cuts']}/{row['num_graduals']})",
                f"{row['overall_f1']:.3f}",
                f"{row['overall_precision']:.3f}",
                f"{row['overall_recall']:.3f}",
                f"{row['cut_f1']:.3f}",
                f"{row['gradual_f1']:.3f}"
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.08, 0.35, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows alternately
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Top 15 Videos with Transitions - Performance Summary', 
                 fontsize=16, fontweight='bold', pad=20)
        
        table_path = self.output_dir / "comparisons" / "top_videos_table.png"
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison table saved: {table_path}")
        
        return table_path
    
    def generate_architecture_diagram(self):
        """Generate architecture diagram"""
        print(f"\nGenerating architecture diagram...")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Title
        ax.text(5, 13.5, 'Hybrid Boundary Detector Architecture', 
               ha='center', fontsize=18, fontweight='bold')
        
        # Input
        ax.add_patch(patches.FancyBboxPatch((4, 12), 2, 0.6, 
                                           boxstyle="round,pad=0.1", 
                                           facecolor='lightblue', edgecolor='black', linewidth=2))
        ax.text(5, 12.3, 'Input: [B, 128, 3, 48, 64]', ha='center', fontsize=11, fontweight='bold')
        
        # Pixel Motion Extractor
        ax.add_patch(patches.FancyBboxPatch((0.5, 10.5), 2.5, 1,
                                           boxstyle="round,pad=0.1",
                                           facecolor='#FFE6E6', edgecolor='red', linewidth=2))
        ax.text(1.75, 11.2, 'Pixel Motion\nExtractor', ha='center', fontsize=10, fontweight='bold')
        ax.text(1.75, 10.7, '15 features', ha='center', fontsize=9, style='italic')
        
        # RGB Encoder
        ax.add_patch(patches.FancyBboxPatch((3.5, 10.5), 2.5, 1,
                                           boxstyle="round,pad=0.1",
                                           facecolor='#E6F3FF', edgecolor='blue', linewidth=2))
        ax.text(4.75, 11.2, 'RGB Spatial\nEncoder', ha='center', fontsize=10, fontweight='bold')
        ax.text(4.75, 10.7, '640 dims', ha='center', fontsize=9, style='italic')
        
        # Diff Encoder
        ax.add_patch(patches.FancyBboxPatch((6.5, 10.5), 2.5, 1,
                                           boxstyle="round,pad=0.1",
                                           facecolor='#E6FFE6', edgecolor='green', linewidth=2))
        ax.text(7.75, 11.2, 'Diff Encoder\n(4 distances)', ha='center', fontsize=10, fontweight='bold')
        ax.text(7.75, 10.7, '640 dims', ha='center', fontsize=9, style='italic')
        
        # Feature Fusion
        ax.add_patch(patches.FancyBboxPatch((2, 9), 6, 0.8,
                                           boxstyle="round,pad=0.1",
                                           facecolor='lightyellow', edgecolor='black', linewidth=2))
        ax.text(5, 9.4, 'Feature Fusion (RGB + Diff + Pixel Motion)', 
               ha='center', fontsize=11, fontweight='bold')
        ax.text(5, 9.1, '1295 → 256 dims', ha='center', fontsize=9)
        
        # Reasoning Module
        ax.add_patch(patches.FancyBboxPatch((0.5, 7.5), 3, 1,
                                           boxstyle="round,pad=0.1",
                                           facecolor='#FFF0E6', edgecolor='orange', linewidth=2))
        ax.text(2, 8.2, 'Reasoning Module', ha='center', fontsize=10, fontweight='bold')
        ax.text(2, 7.9, '• Acceleration\n• Jerk\n• Edge density', 
               ha='center', fontsize=8)
        
        # Progressive Fusion Blocks
        ax.add_patch(patches.FancyBboxPatch((4.5, 7.5), 5, 1,
                                           boxstyle="round,pad=0.1",
                                           facecolor='#F0E6FF', edgecolor='purple', linewidth=2))
        ax.text(7, 8.2, 'Progressive Fusion Blocks (×3)', ha='center', fontsize=10, fontweight='bold')
        ax.text(7, 7.9, '• Multi-level Temporal\n• Dilated Convolution\n• Lightweight Attention', 
               ha='center', fontsize=8)
        
        # Detection Head
        ax.add_patch(patches.FancyBboxPatch((2, 6), 6, 0.8,
                                           boxstyle="round,pad=0.1",
                                           facecolor='#E6FFFF', edgecolor='cyan', linewidth=2))
        ax.text(5, 6.4, 'Detection Head (Two-Stage)', ha='center', fontsize=11, fontweight='bold')
        ax.text(5, 6.1, 'Boundary → Type Classification', ha='center', fontsize=9)
        
        # Outputs
        ax.add_patch(patches.FancyBboxPatch((1.5, 4.5), 2.5, 0.8,
                                           boxstyle="round,pad=0.1",
                                           facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
        ax.text(2.75, 4.9, 'Boundary\nLogits [B,T,1]', ha='center', fontsize=9, fontweight='bold')
        
        ax.add_patch(patches.FancyBboxPatch((5.5, 4.5), 2.5, 0.8,
                                           boxstyle="round,pad=0.1",
                                           facecolor='lightcoral', edgecolor='darkred', linewidth=2))
        ax.text(6.75, 4.9, 'Type Logits\n[B,T,3]', ha='center', fontsize=9, fontweight='bold')
        
        # Add arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='black')
        
        # Input to encoders
        ax.annotate('', xy=(1.75, 11.5), xytext=(5, 12), arrowprops=arrow_props)
        ax.annotate('', xy=(4.75, 11.5), xytext=(5, 12), arrowprops=arrow_props)
        ax.annotate('', xy=(7.75, 11.5), xytext=(5, 12), arrowprops=arrow_props)
        
        # Encoders to fusion
        ax.annotate('', xy=(5, 9.8), xytext=(5, 10.5), arrowprops=arrow_props)
        
        # Fusion to reasoning and blocks
        ax.annotate('', xy=(2, 8.5), xytext=(3.5, 9), arrowprops=arrow_props)
        ax.annotate('', xy=(7, 8.5), xytext=(6.5, 9), arrowprops=arrow_props)
        
        # To detection head
        ax.annotate('', xy=(5, 6.8), xytext=(5, 7.5), arrowprops=arrow_props)
        
        # To outputs
        ax.annotate('', xy=(2.75, 5.3), xytext=(4, 6), arrowprops=arrow_props)
        ax.annotate('', xy=(6.75, 5.3), xytext=(6, 6), arrowprops=arrow_props)
        
        # Key innovations box
        ax.add_patch(patches.FancyBboxPatch((0.2, 0.5), 9.6, 3.5,
                                           boxstyle="round,pad=0.15",
                                           facecolor='#F5F5F5', edgecolor='gray', linewidth=2, linestyle='--'))
        ax.text(5, 3.7, 'Key Innovations', ha='center', fontsize=12, fontweight='bold')
        
        innovations = [
            "✓ Pixel-level motion features (BEFORE CNN encoding)",
            "✓ Multi-scale dilated convolutions (1-64 frame patterns)",
            "✓ Motion dynamics: Acceleration & Jerk (top discriminators)",
            "✓ Multi-level temporal aggregation (sees 96-frame graduals)",
            "✓ Two-stage detection (boundary → type)",
            "✓ Edge density variance (gradual vs pan discrimination)"
        ]
        
        y_pos = 3.2
        for innovation in innovations:
            ax.text(0.5, y_pos, innovation, fontsize=9, verticalalignment='top')
            y_pos -= 0.4
        
        diagram_path = self.output_dir / "architecture_diagram.png"
        plt.savefig(diagram_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Architecture diagram saved: {diagram_path}")
        
        return diagram_path
    
    def create_confusion_matrix(self, split='test', force_recompute=False):
        """Generate confusion matrix for presentation (uses cached predictions if available)"""
        print(f"\nGenerating confusion matrix...")
        
        # Check for cached predictions
        cache_file = self.output_dir / "cache" / f"predictions_{split}.pkl"
        
        if cache_file.exists() and not force_recompute:
            print(f"  ✓ Loading cached predictions from: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            all_preds = cached['predictions']
            all_gts = cached['ground_truth']
            print(f"  ✓ Loaded {len(all_preds):,} frames from cache")
        else:
            if force_recompute:
                print(f"  ⟳ Force recompute requested")
            else:
                print(f"  ⚡ Computing predictions for all clips...")
            
            # Collect all predictions
            dataset = PredecodedDataset(self.cache_path, split=split)
            
            all_preds = []
            all_gts = []
            
            for idx in tqdm(range(len(dataset)), desc="Processing clips"):
                frames, targets, lengths = dataset[idx]
                frames_input = frames.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    boundary_logits, type_logits, _, _ = self.model(frames_input)
                
                boundary_probs = torch.sigmoid(boundary_logits).squeeze()
                type_preds = torch.argmax(type_logits, dim=-1).squeeze()
                
                # Combine: boundary + type
                is_boundary = boundary_probs > 0.5
                preds = torch.where(is_boundary, type_preds, torch.zeros_like(type_preds))
                
                all_preds.append(preds.cpu().numpy())
                all_gts.append(targets.cpu().numpy())
            
            all_preds = np.concatenate(all_preds)
            all_gts = np.concatenate(all_gts)
            
            # Cache predictions for future use
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'predictions': all_preds,
                    'ground_truth': all_gts,
                    'checkpoint_path': str(self.checkpoint_path),
                    'split': split
                }, f)
            print(f"  ✓ Predictions cached: {cache_file}")
        
        # Compute confusion matrix
        cm = confusion_matrix(all_gts, all_preds, labels=[0, 1, 2])
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Raw counts
        im1 = ax1.imshow(cm, cmap='Blues', aspect='auto')
        ax1.set_title('Confusion Matrix (Counts)', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Predicted', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Ground Truth', fontsize=16, fontweight='bold')
        ax1.set_xticks([0, 1, 2])
        ax1.set_yticks([0, 1, 2])
        ax1.set_xticklabels(['None', 'Cut', 'Gradual'], fontsize=14)
        ax1.set_yticklabels(['None', 'Cut', 'Gradual'], fontsize=14)
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax1.text(j, i, f'{cm[i, j]:,}',
                              ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontsize=16, fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Normalized
        im2 = ax2.imshow(cm_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=18, fontweight='bold')
        ax2.set_xlabel('Predicted', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Ground Truth', fontsize=16, fontweight='bold')
        ax2.set_xticks([0, 1, 2])
        ax2.set_yticks([0, 1, 2])
        ax2.set_xticklabels(['None', 'Cut', 'Gradual'], fontsize=14)
        ax2.set_yticklabels(['None', 'Cut', 'Gradual'], fontsize=14)
        
        # Add percentage annotations
        for i in range(3):
            for j in range(3):
                text = ax2.text(j, i, f'{cm_norm[i, j]:.2%}',
                              ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black",
                              fontsize=16, fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "comparisons" / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved: {save_path}")
        return save_path
    
    def create_per_category_performance(self, df):
        """Create per-category performance bars"""
        print(f"\nGenerating per-category performance...")
        
        # Filter videos with transitions
        df_with_trans = df[df['total_transitions'] > 0]
        df_with_cuts = df[df['num_cuts'] > 0]
        df_with_graduals = df[df['num_graduals'] > 0]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # 1. F1 Scores
        ax1 = axes[0]
        categories = ['Overall', 'Cut', 'Gradual']
        f1_scores = [
            df_with_trans['overall_f1'].mean(),
            df_with_cuts['cut_f1'].mean(),
            df_with_graduals['gradual_f1'].mean()  # FIXED: only videos with graduals
        ]
        colors = ['#3498db', '#e74c3c', '#9b59b6']
        bars1 = ax1.bar(categories, f1_scores, color=colors, edgecolor='black', linewidth=2)
        ax1.set_ylabel('F1 Score', fontsize=16, fontweight='bold')
        ax1.set_title('F1 Score by Category', fontsize=18, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3, linewidth=1.5)
        ax1.tick_params(labelsize=14)
        
        # Add value labels with sample counts
        # sample_counts = [len(df_with_trans), len(df_with_cuts), len(df_with_graduals)]
        # for bar, score, count in zip(bars1, f1_scores, sample_counts):
        #     height = bar.get_height()
        #     ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
        #             f'{score:.3f}\\n(n={count})', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # 2. Precision & Recall
        ax2 = axes[1]
        x = np.arange(len(categories))
        width = 0.35
        precision = [
            df_with_trans['overall_precision'].mean(),
            df_with_cuts['cut_precision'].mean(),
            df_with_graduals['gradual_precision'].mean()  # FIXED: only videos with graduals
        ]
        recall = [
            df_with_trans['overall_recall'].mean(),
            df_with_cuts['cut_recall'].mean(),
            df_with_graduals['gradual_recall'].mean()  # FIXED: only videos with graduals
        ]
        
        bars2a = ax2.bar(x - width/2, precision, width, label='Precision', color='#2ecc71', edgecolor='black', linewidth=2)
        bars2b = ax2.bar(x + width/2, recall, width, label='Recall', color='#f39c12', edgecolor='black', linewidth=2)
        
        ax2.set_ylabel('Score', fontsize=16, fontweight='bold')
        ax2.set_title('Precision & Recall by Category', fontsize=18, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=14)
        ax2.set_ylim([0, 1])
        ax2.legend(fontsize=14)
        ax2.grid(axis='y', alpha=0.3, linewidth=1.5)
        ax2.tick_params(labelsize=14)
        
        # 3. Transition Distribution
        ax3 = axes[2]
        total_cuts = df['num_cuts'].sum()
        total_graduals = df['num_graduals'].sum()
        
        sizes = [total_cuts, total_graduals]
        labels_pie = [f'Cuts\n({total_cuts:,})', f'Graduals\n({total_graduals:,})']
        colors_pie = ['#e74c3c', '#9b59b6']
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                            wedgeprops={'edgecolor': 'black', 'linewidth': 2})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(16)
        ax3.set_title('Transition Distribution', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.output_dir / "statistics" / "per_category_performance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Per-category performance saved: {save_path}")
        return save_path
    
    def create_video_clip(self, video_name, clip_idx, transition_type='cut', duration_frames=32):
        """Generate video clip with prediction overlays"""
        print(f"\nGenerating video clip for {video_name} ({transition_type})...")
        
        try:
            # Load clip
            frames, targets, lengths = self.dataset[clip_idx]
            frames_input = frames.unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                boundary_logits, type_logits, _, _ = self.model(frames_input)
            
            boundary_probs = torch.sigmoid(boundary_logits).squeeze().cpu().numpy()
            type_preds = torch.argmax(type_logits, dim=-1).squeeze().cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Find transition frames of the requested type
            if transition_type == 'cut':
                transition_frames = np.where(targets_np == 1)[0]
            elif transition_type == 'gradual':
                transition_frames = np.where(targets_np == 2)[0]
            else:
                transition_frames = np.where(targets_np > 0)[0]
            
            if len(transition_frames) == 0:
                print(f"  ⚠ No {transition_type} transition found in this clip")
                return None
            
            # Center on first transition of requested type
            center_frame = transition_frames[0]
            start = max(0, center_frame - duration_frames // 2)
            end = min(len(frames), start + duration_frames)
            
            # Create video with unique name
            output_path = self.output_dir / "videos" / f"{transition_type}_{clip_idx:04d}.mp4"
            
            # Video writer setup - use reliable codecs at 0.25x speed for better viewing
            fps = 4  # 0.25x speed (24 fps → 6 fps)
            H, W = 480, 640  # Upscale for better visibility
            out = None
            
            # Try multiple codecs in order of compatibility
            # XVID and MJPEG are most reliable, mp4v is fallback
            test_frame = np.zeros((H, W, 3), dtype=np.uint8)  # Black frame for testing
            
            for codec in ['XVID', 'MJPG', 'mp4v', 'DIVX']:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
                    
                    # Verify it actually works by writing a test frame
                    if out is not None and out.isOpened():
                        out.write(test_frame)
                        print(f"  Using codec: {codec}")
                        break
                    else:
                        if out is not None:
                            out.release()
                        out = None
                except Exception as e:
                    if out is not None:
                        try:
                            out.release()
                        except:
                            pass
                    out = None
                    continue
            
            if out is None or not out.isOpened():
                print(f"  ✗ Failed to create video writer with any codec")
                return None
            
            for i in range(start, end):
                # Get frame
                frame = frames[i].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                
                # Resize
                frame = cv2.resize(frame, (W, H))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add overlays
                prob = boundary_probs[i]
                pred = type_preds[i]
                gt = targets_np[i]
                
                # Ground truth label
                gt_labels = ['NONE', 'CUT', 'GRADUAL']
                gt_colors = [(200, 200, 200), (0, 140, 255), (180, 0, 255)]  # BGR
                cv2.rectangle(frame, (10, 10), (200, 60), gt_colors[gt], -1)
                cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 0), 2)
                cv2.putText(frame, f'GT: {gt_labels[gt]}', (20, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                
                # Prediction
                pred_labels = ['NONE', 'CUT', 'GRADUAL']
                cv2.rectangle(frame, (220, 10), (440, 60), gt_colors[pred], -1)
                cv2.rectangle(frame, (220, 10), (440, 60), (0, 0, 0), 2)
                cv2.putText(frame, f'PRED: {pred_labels[pred]}', (230, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                
                # Probability bar
                bar_width = int(prob * 400)
                cv2.rectangle(frame, (10, H-50), (10 + bar_width, H-20), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, H-50), (410, H-20), (0, 0, 0), 2)
                cv2.putText(frame, f'Prob: {prob:.3f}', (420, H-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Frame number
                cv2.putText(frame, f'Frame: {i}', (W-150, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
            print(f"  ✓ Video clip saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"  ✗ Error creating video clip: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_dataset_infographic(self, df, stats):
        """Create one-page infographic summary"""
        print(f"\nGenerating dataset infographic...")
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('Hybrid Boundary Detector - Performance Summary', 
                    fontsize=28, fontweight='bold', y=0.98)
        
        # 1. Key Metrics (top-left, large)
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.axis('off')
        
        metrics_text = f"""
        📊 OVERALL PERFORMANCE
        
        Average F1: {stats['avg_overall_f1_with_transitions']:.3f}
        
        Cut F1: {stats['avg_cut_f1']:.3f}
        Gradual F1: {stats['avg_gradual_f1']:.3f}
        
        Detection Rate: {stats['detection_rate']:.1f}%
        
        Videos Analyzed: {stats['total_videos']:,}
        """
        
        ax1.text(0.1, 0.9, metrics_text, fontsize=18, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round,pad=1', facecolor='#e8f4f8', 
                edgecolor='#3498db', linewidth=3))
        
        # 2. Dataset Statistics (top-middle)
        ax2 = fig.add_subplot(gs[0:2, 1])
        ax2.axis('off')
        
        dataset_text = f"""
        📁 DATASET STATISTICS
        
        Total Transitions: {stats['total_transitions']:,}
        ├─ Cuts: {stats['total_cuts']:,}
        └─ Graduals: {stats['total_graduals']:,}
        
        Avg per Video: {stats['avg_transitions_per_video']:.1f}
        Max per Video: {stats['max_transitions_per_video']}
        
        High F1 Videos: {stats['videos_with_high_f1']}
        """
        
        ax2.text(0.1, 0.9, dataset_text, fontsize=18, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round,pad=1', facecolor='#f0e8f4',
                edgecolor='#9b59b6', linewidth=3))
        
        # 3. Model Architecture (top-right)
        ax3 = fig.add_subplot(gs[0:2, 2])
        ax3.axis('off')
        
        arch_text = """
        🏗️ ARCHITECTURE
        
        • Pixel Motion Extractor
        • RGB Spatial Encoder
        • Diff Encoder (4 distances)
        • Feature Fusion
        • Progressive Fusion (×3)
        • Two-Stage Detection
        
        Key: Multi-scale temporal
        aggregation (1-96 frames)
        """
        
        ax3.text(0.1, 0.9, arch_text, fontsize=18, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round,pad=1', facecolor='#e8f8f0',
                edgecolor='#2ecc71', linewidth=3))
        
        # 4. F1 Distribution (bottom-left)
        ax4 = fig.add_subplot(gs[2:, 0])
        df_with_trans = df[df['total_transitions'] > 0]
        ax4.hist(df_with_trans['overall_f1'], bins=20, color='#3498db', edgecolor='black', linewidth=1.5)
        ax4.axvline(df_with_trans['overall_f1'].mean(), color='red', linestyle='--', 
                   linewidth=3, label=f'Mean: {df_with_trans["overall_f1"].mean():.3f}')
        ax4.set_xlabel('F1 Score', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=16, fontweight='bold')
        ax4.set_title('F1 Score Distribution', fontsize=18, fontweight='bold')
        ax4.legend(fontsize=14)
        ax4.grid(alpha=0.3, linewidth=1.5)
        ax4.tick_params(labelsize=14)
        
        # 5. Category Performance (bottom-middle)
        ax5 = fig.add_subplot(gs[2:, 1])
        df_with_cuts = df[df['num_cuts'] > 0]
        df_with_graduals = df[df['num_graduals'] > 0]
        
        categories = ['Overall', 'Cut', 'Gradual']
        f1_scores = [
            df_with_trans['overall_f1'].mean(),
            df_with_cuts['cut_f1'].mean(),
            df_with_graduals['gradual_f1'].mean()  # FIXED: only videos with graduals
        ]
        colors = ['#3498db', '#e74c3c', '#9b59b6']
        bars = ax5.bar(categories, f1_scores, color=colors, edgecolor='black', linewidth=2)
        ax5.set_ylabel('F1 Score', fontsize=16, fontweight='bold')
        ax5.set_title('Performance by Category', fontsize=18, fontweight='bold')
        ax5.set_ylim([0, 1])
        ax5.grid(axis='y', alpha=0.3, linewidth=1.5)
        ax5.tick_params(labelsize=14)
        
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # 6. Top Videos (bottom-right)
        ax6 = fig.add_subplot(gs[2:, 2])
        ax6.axis('off')
        
        top_videos = df_with_trans.head(5)
        top_text = "🏆 TOP 5 VIDEOS\n\n"
        for idx, (_, row) in enumerate(top_videos.iterrows(), 1):
            top_text += f"{idx}. F1={row['overall_f1']:.3f} ({row['total_transitions']} trans)\n"
        
        ax6.text(0.1, 0.9, top_text, fontsize=16, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round,pad=1', facecolor='#fff4e8',
                edgecolor='#f39c12', linewidth=3))
        
        save_path = self.output_dir / "infographic_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Infographic saved: {save_path}")
        return save_path
    
    def generate_all(self, split='test', top_k=10, max_visualizations=20, force_reanalyze=False, measure_performance=False):
        """Generate all presentation materials
        
        Args:
            measure_performance: Force re-measurement of performance even when using cache
        """
        print(f"\n{'='*80}")
        print("GENERATING PRESENTATION MATERIALS")
        print(f"{'='*80}")
        
        # 1. Analyze videos (uses cache if available)
        df = self.analyze_videos(split=split, top_k=top_k, force_reanalyze=force_reanalyze, 
                                 measure_performance=measure_performance or force_reanalyze)
        
        # 2. Generate statistics
        stats = self.generate_statistics(df)
        
        # 3. Generate performance metrics visualization
        print(f"\n{'='*80}")
        print("GENERATING PERFORMANCE METRICS")
        print(f"{'='*80}")
        self.create_performance_visualization(split=split)
        
        # 4. Create comparison table
        self.create_comparison_table(df)
        
        # 5. Generate architecture diagram
        self.generate_architecture_diagram()
        
        # 6. Generate confusion matrix
        print(f"\n{'='*80}")
        print("GENERATING CONFUSION MATRIX")
        print(f"{'='*80}")
        self.create_confusion_matrix(split=split, force_recompute=force_reanalyze)
        
        # 7. Generate per-category performance
        print(f"\n{'='*80}")
        print("GENERATING PER-CATEGORY PERFORMANCE")
        print(f"{'='*80}")
        self.create_per_category_performance(df)
        
        # 8. Generate infographic summary
        print(f"\n{'='*80}")
        print("GENERATING INFOGRAPHIC SUMMARY")
        print(f"{'='*80}")
        self.create_dataset_infographic(df, stats)
        
        # 9. Visualize top videos - GENERATE MULTIPLE CLIPS PER VIDEO
        print(f"\n{'='*80}")
        print(f"GENERATING VISUALIZATIONS FOR TOP {max_visualizations} VIDEOS")
        print(f"{'='*80}")
        
        # Filter to videos with transitions for better visualizations
        df_with_transitions = df[df['total_transitions'] > 0]
        
        # Track video clips created
        cut_clips_created = 0
        gradual_clips_created = 0
        max_video_clips_per_type = 10  # Create up to 10 cut clips and 10 gradual clips
        
        for idx, row in df_with_transitions.head(max_visualizations).iterrows():
            video_name = row['video_name']
            clip_indices = row['clip_indices']
            
            print(f"\n--- Video {idx+1}/{max_visualizations}: {video_name} ---")
            print(f"  Total clips: {len(clip_indices)}, Transitions: {row['total_transitions']} (Cuts: {row['num_cuts']}, Graduals: {row['num_graduals']})")
            
            # Generate visualizations for MULTIPLE good clips (not just first!)
            clips_visualized = 0
            max_clips_per_video = 4  # Show up to 3 different clips per video
            
            for search_clip_idx in clip_indices[:15]:  # Search first 15 clips
                if clips_visualized >= max_clips_per_video:
                    break
                
                try:
                    # Check if this clip has transitions
                    _, targets, _ = self.dataset[search_clip_idx]
                    has_transitions = (targets > 0).any().item()
                    
                    if has_transitions:
                        print(f"\n  === Clip {clips_visualized + 1}/{max_clips_per_video}: Index {search_clip_idx} ===")
                        
                        # DISABLED: Standard detection visualization (2-panel with probabilities)
                        # self.visualize_detections(video_name, search_clip_idx)
                        
                        # Frame montage (kept - useful for presentations)
                        self.create_frame_montage(video_name, search_clip_idx, num_frames=64)
                        
                        # DISABLED: Temporal heatmap (too detailed for presentations)
                        # self.create_temporal_heatmap(video_name, search_clip_idx)
                        
                        clips_visualized += 1
                
                except Exception as e:
                    print(f"    ⚠ Error processing clip {search_clip_idx}: {e}")
                    continue
            
            # Timeline for all clips (show once per video)
            self.create_transition_timeline(video_name, clip_indices)
            
            # NEW: Create video clips (MP4) for cuts and graduals
            # Search through clips to find transitions
            if cut_clips_created < max_video_clips_per_type or gradual_clips_created < max_video_clips_per_type:
                print(f"\n  Searching for video clip examples...")
                
                for search_clip_idx in clip_indices[:20]:  # Search first 20 clips
                    if cut_clips_created >= max_video_clips_per_type and gradual_clips_created >= max_video_clips_per_type:
                        break  # Got enough of both types
                    
                    try:
                        _, targets, _ = self.dataset[search_clip_idx]
                        has_cut = (targets == 1).any().item()
                        has_gradual = (targets == 2).any().item()
                        
                        # Create cut video if found and needed
                        if has_cut and cut_clips_created < max_video_clips_per_type:
                            result = self.create_video_clip(video_name, search_clip_idx, transition_type='cut')
                            if result is not None:
                                cut_clips_created += 1
                                print(f"    ✓ Cut clip created ({cut_clips_created}/{max_video_clips_per_type})")
                        
                        # Create gradual video if found and needed
                        if has_gradual and gradual_clips_created < max_video_clips_per_type:
                            result = self.create_video_clip(video_name, search_clip_idx, transition_type='gradual')
                            if result is not None:
                                gradual_clips_created += 1
                                print(f"    ✓ Gradual clip created ({gradual_clips_created}/{max_video_clips_per_type})")
                    
                    except Exception as e:
                        print(f"    ⚠ Error processing clip {search_clip_idx}: {e}")
                        continue
        
        # 10. Create summary document
        summary = {
            'checkpoint': str(self.checkpoint_path),
            'split': split,
            'statistics': stats,
            'top_videos': df.head(top_k)[['video_name', 'total_transitions', 
                                         'overall_f1', 'cut_f1', 'gradual_f1']].to_dict('records'),
            'output_directory': str(self.output_dir),
            'visualizations_generated': {
                'confusion_matrix': True,
                'per_category_performance': True,
                'infographic': True,
                'video_clips': {
                    'cuts': cut_clips_created,
                    'graduals': gradual_clips_created
                }
            }
        }
        
        summary_path = self.output_dir / "presentation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print("GENERATION COMPLETE!")
        print(f"{'='*80}")
        print(f"\nAll materials saved to: {self.output_dir}")
        print(f"  • Visualizations: {self.output_dir / 'visualizations'}")
        print(f"  • Statistics: {self.output_dir / 'statistics'}")
        print(f"  • Comparisons: {self.output_dir / 'comparisons'}")
        print(f"  • Videos: {self.output_dir / 'videos'} ({cut_clips_created} cuts, {gradual_clips_created} graduals)")
        print(f"  • Cache: {self.output_dir / 'cache'} (reusable analysis & predictions)")
        print(f"  • Summary: {summary_path}")
        print(f"\n💡 TIP: Next time, analysis will load from cache (instant!).")
        print(f"    Use --force_reanalyze to recompute if model/data changes.")
        print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Generate presentation materials')
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs/hybrid_detector4/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str,
                       default='presentation_materials',
                       help='Output directory for materials')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to analyze (default: val)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top videos to show in tables')
    parser.add_argument('--max_visualizations', type=int, default=5,
                       help='Maximum number of videos to visualize')
    parser.add_argument('--force_reanalyze', action='store_true',
                       help='Force re-analysis of all videos (ignore cache). Use this when model/data changes.')
    parser.add_argument('--measure_performance', action='store_true',
                       help='Measure performance metrics even when using cached analysis')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("PRESENTATION MATERIALS GENERATOR")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Split: {args.split}")
    print(f"  Output: {args.output_dir}")
    print(f"  Force re-analyze: {'Yes - will reprocess all videos' if args.force_reanalyze else 'No - will use cache if available'}")
    print(f"  Measure performance: {'Yes - will re-measure even with cache' if args.measure_performance else 'Only if missing from cache'}")
    print(f"{'='*80}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"\n⚠️  ERROR: Checkpoint not found!")
        print(f"{'='*80}")
        print(f"The checkpoint file does not exist:")
        print(f"  {args.checkpoint}")
        print(f"\nPlease provide a valid checkpoint path using --checkpoint")
        print(f"Example:")
        print(f"  python generate_presentation_materials.py --checkpoint outputs/hybrid_detector4/best_model.pt")
        print(f"{'='*80}")
        return
    
    # Check if dataset cache exists
    cache_path = Path(r"D:\grad\data\training_cache")
    if not cache_path.exists():
        print(f"\n⚠️  ERROR: Dataset cache not found!")
        print(f"{'='*80}")
        print(f"The pre-decoded dataset cache does not exist:")
        print(f"  {cache_path}")
        print(f"\nPlease run the pre-decode script first:")
        print(f"  python predecode_dataset.py --split train --num_workers 8")
        if args.split == 'test':
            print(f"  python predecode_dataset.py --split test --num_workers 8")
        print(f"{'='*80}")
        return
    
    # Create generator
    generator = PresentationMaterialGenerator(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # Generate all materials
    generator.generate_all(
        split=args.split,
        top_k=args.top_k,
        max_visualizations=args.max_visualizations,
        force_reanalyze=args.force_reanalyze,
        measure_performance=args.measure_performance
    )


if __name__ == '__main__':
    main()
