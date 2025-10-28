"""
Visualize detection issues to diagnose poor performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import DEFAULT_RESEARCH_CONFIG
from utils.video_utils import VideoProcessor
from extractors import FeatureExtractorRegistry
from fusion import FusionStrategyRegistry

def load_and_extract_features(video_path):
    """Load video and extract features."""
    config = DEFAULT_RESEARCH_CONFIG
    
    print("Loading video...")
    video_processor = VideoProcessor(config.processing)
    frames = video_processor.load_video(video_path)
    print(f"Loaded {len(frames)} frames")
    
    print("\nExtracting features...")
    features = {}
    
    # HSV
    if config.features.hsv_enabled:
        hsv_extractor = FeatureExtractorRegistry.get("hsv_histogram")
        features['hsv_histogram'] = hsv_extractor.extract_batch(frames)
        print(f"HSV: {features['hsv_histogram'].shape}")
    
    # CNN
    if config.features.cnn_enabled:
        cnn_extractor = FeatureExtractorRegistry.get("cnn", model_name=config.features.cnn_model)
        features['cnn'] = cnn_extractor.extract_batch(frames)
        print(f"CNN: {features['cnn'].shape}")
    
    # CLIP
    if config.features.clip_enabled:
        clip_extractor = FeatureExtractorRegistry.get("clip", model_name=config.features.clip_model)
        features['clip'] = clip_extractor.extract_batch(frames)
        print(f"CLIP: {features['clip'].shape}")
    
    # Optical Flow
    if config.features.optical_flow_enabled:
        flow_extractor = FeatureExtractorRegistry.get("optical_flow", method=config.features.optical_flow_method)
        features['optical_flow'] = flow_extractor.extract_batch(frames)
        print(f"Optical Flow: {features['optical_flow'].shape}")
    
    return frames, features

def compute_distance_signal(features, fusion_strategy='hierarchical'):
    """Compute distance signal using specified fusion strategy."""
    feature_dims = {name: feat.shape[1] for name, feat in features.items()}
    
    print(f"\nUsing {fusion_strategy} fusion strategy...")
    fusion = FusionStrategyRegistry.get(fusion_strategy, feature_dims=feature_dims, device='cpu')
    fusion.initialize()
    
    # Compute distance signal
    distance_signal = fusion(features)
    print(f"Distance signal shape: {distance_signal.shape}")
    
    return distance_signal

def visualize_detection_analysis(video_path, ground_truth_file, video_name):
    """Create comprehensive visualization of detection issues."""
    
    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)
    
    video_key = None
    if video_name in gt_data:
        video_key = video_name
    elif f"{video_name}.mp4" in gt_data:
        video_key = f"{video_name}.mp4"
    else:
        print(f"Ground truth not found for {video_name}")
        print(f"Available keys: {list(gt_data.keys())[:5]}")
        return
    
    # Extract boundaries from transitions (using start frame of each transition)
    if isinstance(gt_data[video_key], dict) and 'transitions' in gt_data[video_key]:
        transitions = gt_data[video_key]['transitions']
        gt_boundaries = [int(t[0]) for t in transitions]  # Use start frame of transition
    elif isinstance(gt_data[video_key], list):
        gt_boundaries = [int(b) for b in gt_data[video_key]]
    else:
        print(f"Unknown ground truth format: {type(gt_data[video_key])}")
        return
    
    print(f"Ground truth: {len(gt_boundaries)} boundaries")
    
    # Extract features
    frames, features = load_and_extract_features(video_path)
    
    # Try different fusion strategies
    fusion_strategies = ['concatenation', 'hierarchical', 'weighted', 'pca']
    
    fig, axes = plt.subplots(len(fusion_strategies), 1, figsize=(16, 4*len(fusion_strategies)))
    if len(fusion_strategies) == 1:
        axes = [axes]
    
    for idx, strategy in enumerate(fusion_strategies):
        try:
            distance_signal = compute_distance_signal(features, strategy)
            
            # Plot distance signal
            ax = axes[idx]
            ax.plot(distance_signal, linewidth=0.5, alpha=0.7, label='Distance Signal')
            
            # Mark ground truth boundaries
            for gt_b in gt_boundaries:
                if gt_b < len(distance_signal):
                    ax.axvline(x=gt_b, color='green', alpha=0.6, linewidth=1.5, label='GT' if gt_b == gt_boundaries[0] else '')
            
            # Compute statistics
            mean = np.mean(distance_signal)
            std = np.std(distance_signal)
            
            # Show different thresholds
            ax.axhline(y=mean, color='blue', linestyle='--', alpha=0.5, label=f'Mean: {mean:.3f}')
            ax.axhline(y=mean + 1.0*std, color='orange', linestyle='--', alpha=0.5, label=f'Mean + 1σ')
            ax.axhline(y=mean + 1.5*std, color='red', linestyle='--', alpha=0.5, label=f'Mean + 1.5σ')
            ax.axhline(y=mean + 2.0*std, color='purple', linestyle='--', alpha=0.5, label=f'Mean + 2σ')
            
            ax.set_title(f'{strategy.capitalize()} Fusion - Mean: {mean:.3f}, Std: {std:.3f}')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Distance')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Print signal statistics at GT boundaries
            gt_signal_values = [distance_signal[min(gt_b, len(distance_signal)-1)] for gt_b in gt_boundaries if gt_b < len(distance_signal)]
            print(f"\n{strategy} - Signal at GT boundaries:")
            print(f"  Mean: {np.mean(gt_signal_values):.3f}")
            print(f"  Median: {np.median(gt_signal_values):.3f}")
            print(f"  Min: {np.min(gt_signal_values):.3f}")
            print(f"  Max: {np.max(gt_signal_values):.3f}")
            print(f"  Overall mean: {mean:.3f}, std: {std:.3f}")
            print(f"  Ratio (GT_mean/Overall_mean): {np.mean(gt_signal_values)/mean:.2f}")
            
        except Exception as e:
            print(f"Error with {strategy}: {e}")
            axes[idx].text(0.5, 0.5, f"Error: {str(e)}", transform=axes[idx].transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig('test_results/distance_signal_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to test_results/distance_signal_analysis.png")
    plt.close()
    
    # Create a zoomed-in view of first 500 frames
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # Use hierarchical fusion for detailed view
    distance_signal = compute_distance_signal(features, 'hierarchical')
    
    # Full view
    axes[0].plot(distance_signal, linewidth=0.5, alpha=0.7)
    for gt_b in gt_boundaries:
        if gt_b < len(distance_signal):
            axes[0].axvline(x=gt_b, color='green', alpha=0.6, linewidth=1.5)
    axes[0].set_title('Full Distance Signal (Hierarchical Fusion)')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Distance')
    axes[0].grid(True, alpha=0.3)
    
    # Zoomed view
    zoom_end = min(500, len(distance_signal))
    axes[1].plot(distance_signal[:zoom_end], linewidth=1, alpha=0.8)
    for gt_b in gt_boundaries:
        if gt_b < zoom_end:
            axes[1].axvline(x=gt_b, color='green', alpha=0.7, linewidth=2, label=f'GT: {gt_b}')
            axes[1].text(gt_b, np.max(distance_signal[:zoom_end])*0.9, str(gt_b), 
                        rotation=90, va='top', fontsize=8)
    
    mean = np.mean(distance_signal[:zoom_end])
    std = np.std(distance_signal[:zoom_end])
    axes[1].axhline(y=mean + 1.5*std, color='red', linestyle='--', alpha=0.5, label=f'Threshold (μ+1.5σ)')
    
    axes[1].set_title('Zoomed View (First 500 frames)')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Distance')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('test_results/distance_signal_detailed.png', dpi=150, bbox_inches='tight')
    print(f"Detailed view saved to test_results/distance_signal_detailed.png")
    plt.close()


if __name__ == "__main__":
    video_path = r"C:\Users\Abdalrahman Khanfar\Documents\ClipShots\videos\train\3505586409.mp4"
    ground_truth = r"C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json"
    video_name = "3505586409"
    
    print("="*60)
    print("VISUALIZING DETECTION ISSUES")
    print("="*60)
    
    visualize_detection_analysis(video_path, ground_truth, video_name)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Check if peaks in distance signal align with GT boundaries")
    print("2. If signal is too noisy, increase smoothing_window")
    print("3. If peaks are weak, problem is with feature extraction or fusion")
    print("4. If peaks are strong but threshold is wrong, adjust adaptive_beta")
