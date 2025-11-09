"""
Analyze correlation and Euclidean distance between consecutive frames for each feature type.
This script helps determine which feature type shows the strongest signal for shot boundary detection.
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Hardcoded configuration from train_transformer.py usage
VIDEO_DIR = r"C:\Users\Abdalrahman Khanfar\Documents\ClipShots\videos\train"
GROUND_TRUTH_FILE = r"C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json"
FEATURES_DIR = "training_features"
OUTPUT_DIR = "feature_analysis_results"

# Feature types to analyze (individual features, not fused)
FEATURE_TYPES = ['cnn', 'hsv_histogram', 'clip', 'optical_flow']

# Expected feature dimensions
FEATURE_DIMS = {
    'optical_flow': 40,
    'clip': 512,
    'cnn': 2048,
    'hsv_histogram': 768
}


def load_ground_truth(ground_truth_file: str) -> Dict:
    """Load ground truth annotations."""
    with open(ground_truth_file, 'r') as f:
        return json.load(f)


def load_selected_videos() -> List[Tuple[str, int, List, List]]:
    """Load the selected videos from the JSON file created by find_short_videos.py."""
    with open('selected_videos.json', 'r') as f:
        data = json.load(f)
    # data format: [(video_name, frame_count, transitions, boundaries), ...]
    return [(item[0], item[1], item[2], item[3]) for item in data]


def load_feature(features_dir: str, video_name: str, feature_type: str) -> np.ndarray:
    """Load a specific feature type for a video."""
    feature_file = os.path.join(features_dir, f"{video_name}_{feature_type}.npy")
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    
    features = np.load(feature_file)
    print(f"  Loaded {feature_type}: shape {features.shape}")
    return features


def calculate_correlation(features: np.ndarray) -> np.ndarray:
    """
    Calculate correlation between consecutive frames.
    
    Args:
        features: Feature array of shape (N, D) where N is frames, D is feature dimension
        
    Returns:
        Array of shape (N-1,) with correlation coefficients between consecutive frames
    """
    correlations = []
    
    for i in range(len(features) - 1):
        frame1 = features[i]
        frame2 = features[i + 1]
        
        # Calculate Pearson correlation coefficient
        # Handle edge cases where features might be constant
        if np.std(frame1) == 0 or np.std(frame2) == 0:
            corr = 1.0  # Perfect correlation if both are constant
        else:
            corr, _ = pearsonr(frame1, frame2)
        
        correlations.append(corr)
    
    return np.array(correlations)


def calculate_euclidean_distance(features: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance between consecutive frames.
    
    Args:
        features: Feature array of shape (N, D) where N is frames, D is feature dimension
        
    Returns:
        Array of shape (N-1,) with Euclidean distances between consecutive frames
    """
    distances = []
    
    for i in range(len(features) - 1):
        frame1 = features[i]
        frame2 = features[i + 1]
        
        # Calculate Euclidean distance
        dist = np.linalg.norm(frame1 - frame2)
        distances.append(dist)
    
    return np.array(distances)


def plot_video_analysis(video_name: str, frame_count: int, boundaries: List[int],
                        results: Dict[str, Dict], output_dir: str):
    """
    Create plots for a single video showing correlation and distance for all features.
    
    Args:
        video_name: Name of the video
        frame_count: Total number of frames
        boundaries: List of boundary frame indices
        results: Dictionary with results for each feature type
        output_dir: Directory to save plots
    """
    # Create video-specific directory
    video_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_dir, exist_ok=True)
    
    num_features = len(FEATURE_TYPES)
    
    # Plot 1: Individual correlation plots for each feature
    for feature_type in FEATURE_TYPES:
        if feature_type not in results:
            continue
        
        correlations = results[feature_type]['correlation']
        
        plt.figure(figsize=(14, 5))
        frames = np.arange(1, len(correlations) + 1)
        
        plt.plot(frames, correlations, linewidth=1.5, color='blue', alpha=0.7)
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Correlation Coefficient', fontsize=12)
        plt.title(f'{video_name} - {feature_type.upper()} Correlation', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(correlations):
                plt.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Boundary' if boundary == boundaries[0] else '')
        
        if boundaries:
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(video_dir, f'{feature_type}_correlation.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Individual distance plots for each feature
    for feature_type in FEATURE_TYPES:
        if feature_type not in results:
            continue
        
        distances = results[feature_type]['distance']
        
        plt.figure(figsize=(14, 5))
        frames = np.arange(1, len(distances) + 1)
        
        plt.plot(frames, distances, linewidth=1.5, color='green', alpha=0.7)
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Euclidean Distance', fontsize=12)
        plt.title(f'{video_name} - {feature_type.upper()} Euclidean Distance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(distances):
                plt.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Boundary' if boundary == boundaries[0] else '')
        
        if boundaries:
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(video_dir, f'{feature_type}_distance.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Combined correlation plot (all features on one plot)
    fig, axes = plt.subplots(num_features, 1, figsize=(14, 3 * num_features), sharex=True)
    if num_features == 1:
        axes = [axes]
    
    for idx, feature_type in enumerate(FEATURE_TYPES):
        if feature_type not in results:
            continue
        
        correlations = results[feature_type]['correlation']
        frames = np.arange(1, len(correlations) + 1)
        
        axes[idx].plot(frames, correlations, linewidth=1.5, alpha=0.7)
        axes[idx].set_ylabel('Correlation', fontsize=10)
        axes[idx].set_title(f'{feature_type.upper()}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(correlations):
                axes[idx].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    axes[-1].set_xlabel('Frame Index', fontsize=12)
    fig.suptitle(f'{video_name} - All Features Correlation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(video_dir, 'all_features_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Combined distance plot (all features on one plot)
    fig, axes = plt.subplots(num_features, 1, figsize=(14, 3 * num_features), sharex=True)
    if num_features == 1:
        axes = [axes]
    
    for idx, feature_type in enumerate(FEATURE_TYPES):
        if feature_type not in results:
            continue
        
        distances = results[feature_type]['distance']
        frames = np.arange(1, len(distances) + 1)
        
        axes[idx].plot(frames, distances, linewidth=1.5, alpha=0.7, color='green')
        axes[idx].set_ylabel('Distance', fontsize=10)
        axes[idx].set_title(f'{feature_type.upper()}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(distances):
                axes[idx].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    axes[-1].set_xlabel('Frame Index', fontsize=12)
    fig.suptitle(f'{video_name} - All Features Euclidean Distance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(video_dir, 'all_features_distance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Side-by-side comparison (correlation and distance) for each feature
    for feature_type in FEATURE_TYPES:
        if feature_type not in results:
            continue
        
        correlations = results[feature_type]['correlation']
        distances = results[feature_type]['distance']
        frames = np.arange(1, len(correlations) + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Correlation subplot
        ax1.plot(frames, correlations, linewidth=1.5, color='blue', alpha=0.7)
        ax1.set_ylabel('Correlation Coefficient', fontsize=11)
        ax1.set_title('Correlation Between Consecutive Frames', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(correlations):
                ax1.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Distance subplot
        ax2.plot(frames, distances, linewidth=1.5, color='green', alpha=0.7)
        ax2.set_ylabel('Euclidean Distance', fontsize=11)
        ax2.set_xlabel('Frame Index', fontsize=11)
        ax2.set_title('Euclidean Distance Between Consecutive Frames', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(distances):
                ax2.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                          label='Boundary' if boundary == boundaries[0] else '')
        
        if boundaries:
            ax2.legend()
        
        fig.suptitle(f'{video_name} - {feature_type.upper()} Side-by-Side Comparison', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(video_dir, f'{feature_type}_side_by_side.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 6: All features side-by-side (2 columns: correlation and distance)
    fig, axes = plt.subplots(num_features, 2, figsize=(18, 3 * num_features), sharex='col')
    if num_features == 1:
        axes = axes.reshape(1, -1)
    
    for idx, feature_type in enumerate(FEATURE_TYPES):
        if feature_type not in results:
            continue
        
        correlations = results[feature_type]['correlation']
        distances = results[feature_type]['distance']
        frames = np.arange(1, len(correlations) + 1)
        
        # Correlation column
        axes[idx, 0].plot(frames, correlations, linewidth=1.5, alpha=0.7, color='blue')
        axes[idx, 0].set_ylabel(f'{feature_type.upper()}\nCorrelation', fontsize=10)
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(correlations):
                axes[idx, 0].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Distance column
        axes[idx, 1].plot(frames, distances, linewidth=1.5, alpha=0.7, color='green')
        axes[idx, 1].set_ylabel(f'{feature_type.upper()}\nDistance', fontsize=10)
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(distances):
                axes[idx, 1].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Set column titles
    axes[0, 0].set_title('Correlation', fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].set_title('Euclidean Distance', fontsize=12, fontweight='bold', pad=10)
    
    # Set x-axis labels
    axes[-1, 0].set_xlabel('Frame Index', fontsize=11)
    axes[-1, 1].set_xlabel('Frame Index', fontsize=11)
    
    fig.suptitle(f'{video_name} - All Features Side-by-Side Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(video_dir, 'all_features_side_by_side.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plots to {video_dir}")


def create_combined_plots(all_results: Dict[str, Dict], output_dir: str):
    """
    Create combined plots showing all videos for each feature type.
    
    Args:
        all_results: Dictionary mapping video_name to results dict
        output_dir: Directory to save plots
    """
    print("\nCreating combined plots for all videos...")
    
    for feature_type in FEATURE_TYPES:
        print(f"  Processing {feature_type}...")
        
        # Correlation plot
        fig, axes = plt.subplots(len(all_results), 1, figsize=(16, 2.5 * len(all_results)), sharex=False)
        if len(all_results) == 1:
            axes = [axes]
        
        for idx, (video_name, video_data) in enumerate(all_results.items()):
            if feature_type not in video_data['results']:
                continue
            
            correlations = video_data['results'][feature_type]['correlation']
            boundaries = video_data['boundaries']
            frames = np.arange(1, len(correlations) + 1)
            
            axes[idx].plot(frames, correlations, linewidth=1.5, alpha=0.7)
            axes[idx].set_ylabel('Correlation', fontsize=9)
            axes[idx].set_title(f'{video_name} ({video_data["frame_count"]} frames, {len(boundaries)} boundaries)', 
                              fontsize=10, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            # Add vertical lines for boundaries
            for boundary in boundaries:
                if boundary < len(correlations):
                    axes[idx].axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        axes[-1].set_xlabel('Frame Index', fontsize=11)
        fig.suptitle(f'All Videos - {feature_type.upper()} Correlation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'all_videos_{feature_type}_correlation.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Distance plot
        fig, axes = plt.subplots(len(all_results), 1, figsize=(16, 2.5 * len(all_results)), sharex=False)
        if len(all_results) == 1:
            axes = [axes]
        
        for idx, (video_name, video_data) in enumerate(all_results.items()):
            if feature_type not in video_data['results']:
                continue
            
            distances = video_data['results'][feature_type]['distance']
            boundaries = video_data['boundaries']
            frames = np.arange(1, len(distances) + 1)
            
            axes[idx].plot(frames, distances, linewidth=1.5, alpha=0.7, color='green')
            axes[idx].set_ylabel('Distance', fontsize=9)
            axes[idx].set_title(f'{video_name} ({video_data["frame_count"]} frames, {len(boundaries)} boundaries)', 
                              fontsize=10, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            # Add vertical lines for boundaries
            for boundary in boundaries:
                if boundary < len(distances):
                    axes[idx].axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        axes[-1].set_xlabel('Frame Index', fontsize=11)
        fig.suptitle(f'All Videos - {feature_type.upper()} Euclidean Distance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'all_videos_{feature_type}_distance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Side-by-side comparison for this feature across all videos
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 2.5 * len(all_results)))
        
        # Create subplots for correlation (left) and distance (right)
        ax_left_subplots = [plt.subplot(len(all_results), 2, i*2 + 1) for i in range(len(all_results))]
        ax_right_subplots = [plt.subplot(len(all_results), 2, i*2 + 2) for i in range(len(all_results))]
        plt.close(fig)  # Close the initial figure
        
        # Create proper figure with subplots
        fig = plt.figure(figsize=(20, 2.5 * len(all_results)))
        gs = fig.add_gridspec(len(all_results), 2, hspace=0.3, wspace=0.3)
        
        for idx, (video_name, video_data) in enumerate(all_results.items()):
            if feature_type not in video_data['results']:
                continue
            
            correlations = video_data['results'][feature_type]['correlation']
            distances = video_data['results'][feature_type]['distance']
            boundaries = video_data['boundaries']
            frames = np.arange(1, len(correlations) + 1)
            
            # Correlation subplot (left column)
            ax_corr = fig.add_subplot(gs[idx, 0])
            ax_corr.plot(frames, correlations, linewidth=1.5, alpha=0.7, color='blue')
            ax_corr.set_ylabel('Correlation', fontsize=9)
            if idx == 0:
                ax_corr.set_title('Correlation', fontsize=11, fontweight='bold')
            ax_corr.text(0.02, 0.95, f'{video_name}\n{video_data["frame_count"]} frames, {len(boundaries)} boundaries',
                        transform=ax_corr.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            ax_corr.grid(True, alpha=0.3)
            
            # Add vertical lines for boundaries
            for boundary in boundaries:
                if boundary < len(correlations):
                    ax_corr.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Distance subplot (right column)
            ax_dist = fig.add_subplot(gs[idx, 1])
            ax_dist.plot(frames, distances, linewidth=1.5, alpha=0.7, color='green')
            ax_dist.set_ylabel('Distance', fontsize=9)
            if idx == 0:
                ax_dist.set_title('Euclidean Distance', fontsize=11, fontweight='bold')
            ax_dist.grid(True, alpha=0.3)
            
            # Add vertical lines for boundaries
            for boundary in boundaries:
                if boundary < len(distances):
                    ax_dist.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Set x-label only for bottom row
            if idx == len(all_results) - 1:
                ax_corr.set_xlabel('Frame Index', fontsize=10)
                ax_dist.set_xlabel('Frame Index', fontsize=10)
        
        fig.suptitle(f'All Videos - {feature_type.upper()} Side-by-Side Comparison', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(output_dir, f'all_videos_{feature_type}_side_by_side.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved combined plots to {output_dir}")


def calculate_statistics(all_results: Dict[str, Dict]) -> Dict:
    """
    Calculate aggregate statistics across all videos for each feature type.
    
    Args:
        all_results: Dictionary mapping video_name to results dict
        
    Returns:
        Dictionary with statistics for each feature type
    """
    stats = {}
    
    for feature_type in FEATURE_TYPES:
        print(f"\n{feature_type.upper()} Statistics:")
        print("=" * 60)
        
        # Collect all correlations and distances at boundaries vs non-boundaries
        boundary_correlations = []
        non_boundary_correlations = []
        boundary_distances = []
        non_boundary_distances = []
        
        for video_name, video_data in all_results.items():
            if feature_type not in video_data['results']:
                continue
            
            correlations = video_data['results'][feature_type]['correlation']
            distances = video_data['results'][feature_type]['distance']
            boundaries = video_data['boundaries']
            
            # Separate boundary vs non-boundary measurements
            for i in range(len(correlations)):
                if i in boundaries:
                    boundary_correlations.append(correlations[i])
                    if i < len(distances):
                        boundary_distances.append(distances[i])
                else:
                    non_boundary_correlations.append(correlations[i])
                    if i < len(distances):
                        non_boundary_distances.append(distances[i])
        
        # Calculate statistics (convert to Python float for JSON serialization)
        stats[feature_type] = {
            'boundary_correlation_mean': float(np.mean(boundary_correlations)) if boundary_correlations else 0.0,
            'boundary_correlation_std': float(np.std(boundary_correlations)) if boundary_correlations else 0.0,
            'non_boundary_correlation_mean': float(np.mean(non_boundary_correlations)) if non_boundary_correlations else 0.0,
            'non_boundary_correlation_std': float(np.std(non_boundary_correlations)) if non_boundary_correlations else 0.0,
            'boundary_distance_mean': float(np.mean(boundary_distances)) if boundary_distances else 0.0,
            'boundary_distance_std': float(np.std(boundary_distances)) if boundary_distances else 0.0,
            'non_boundary_distance_mean': float(np.mean(non_boundary_distances)) if non_boundary_distances else 0.0,
            'non_boundary_distance_std': float(np.std(non_boundary_distances)) if non_boundary_distances else 0.0,
        }
        
        # Calculate separation metrics (convert to Python float)
        if boundary_correlations and non_boundary_correlations:
            corr_separation = abs(stats[feature_type]['boundary_correlation_mean'] - 
                                 stats[feature_type]['non_boundary_correlation_mean'])
            stats[feature_type]['correlation_separation'] = float(corr_separation)
        
        if boundary_distances and non_boundary_distances:
            dist_separation = abs(stats[feature_type]['boundary_distance_mean'] - 
                                 stats[feature_type]['non_boundary_distance_mean'])
            stats[feature_type]['distance_separation'] = float(dist_separation)
        
        # Print statistics
        print(f"Correlation at boundaries:     {stats[feature_type]['boundary_correlation_mean']:.4f} ± {stats[feature_type]['boundary_correlation_std']:.4f}")
        print(f"Correlation at non-boundaries: {stats[feature_type]['non_boundary_correlation_mean']:.4f} ± {stats[feature_type]['non_boundary_correlation_std']:.4f}")
        print(f"Correlation separation:        {stats[feature_type].get('correlation_separation', 0):.4f}")
        print()
        print(f"Distance at boundaries:        {stats[feature_type]['boundary_distance_mean']:.4f} ± {stats[feature_type]['boundary_distance_std']:.4f}")
        print(f"Distance at non-boundaries:    {stats[feature_type]['non_boundary_distance_mean']:.4f} ± {stats[feature_type]['non_boundary_distance_std']:.4f}")
        print(f"Distance separation:           {stats[feature_type].get('distance_separation', 0):.4f}")
        print(f"Number of boundary samples:    {len(boundary_correlations)}")
        print(f"Number of non-boundary samples: {len(non_boundary_correlations)}")
    
    return stats


def main():
    """Main function to analyze features for selected videos."""
    print("=" * 80)
    print("FEATURE DISTANCE AND CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Features directory: {FEATURES_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Feature types: {', '.join(FEATURE_TYPES)}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load selected videos
    print(f"\nLoading selected videos from selected_videos.json...")
    selected_videos = load_selected_videos()
    print(f"Found {len(selected_videos)} videos to analyze")
    
    # Process each video
    all_results = {}
    
    for video_name, frame_count, transitions, boundaries in selected_videos:
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        print(f"  Frames: {frame_count}, Boundaries: {len(boundaries)}")
        print(f"  Boundary positions: {boundaries}")
        
        results = {}
        
        # Process each feature type
        for feature_type in FEATURE_TYPES:
            print(f"\n  Analyzing {feature_type}...")
            
            try:
                # Load features
                features = load_feature(FEATURES_DIR, video_name, feature_type)
                
                # Calculate correlation and distance
                correlations = calculate_correlation(features)
                distances = calculate_euclidean_distance(features)
                
                results[feature_type] = {
                    'correlation': correlations,
                    'distance': distances
                }
                
                print(f"    Correlation shape: {correlations.shape}")
                print(f"    Distance shape: {distances.shape}")
                print(f"    Correlation range: [{correlations.min():.4f}, {correlations.max():.4f}]")
                print(f"    Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
        
        # Store results
        all_results[video_name] = {
            'frame_count': frame_count,
            'boundaries': boundaries,
            'transitions': transitions,
            'results': results
        }
        
        # Create plots for this video
        plot_video_analysis(video_name, frame_count, boundaries, results, OUTPUT_DIR)
    
    # Create combined plots
    create_combined_plots(all_results, OUTPUT_DIR)
    
    # Calculate and display statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS ACROSS ALL VIDEOS")
    print("=" * 80)
    stats = calculate_statistics(all_results)
    
    # Save statistics to JSON
    stats_file = os.path.join(OUTPUT_DIR, 'statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")
    
    # Save results to JSON
    results_file = os.path.join(OUTPUT_DIR, 'all_results.json')
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for video_name, video_data in all_results.items():
        json_results[video_name] = {
            'frame_count': video_data['frame_count'],
            'boundaries': video_data['boundaries'],
            'transitions': video_data['transitions'],
            'results': {}
        }
        for feature_type, feature_data in video_data['results'].items():
            json_results[video_name]['results'][feature_type] = {
                'correlation': feature_data['correlation'].tolist(),
                'distance': feature_data['distance'].tolist()
            }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Detailed results saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - Individual video plots in subdirectories")
    print(f"  - Combined plots in main directory")
    print(f"  - Statistics in statistics.json")
    print(f"  - Detailed results in all_results.json")


if __name__ == "__main__":
    main()
