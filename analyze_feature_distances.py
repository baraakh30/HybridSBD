"""
Analyze correlation and Euclidean distance between consecutive frames for each feature type.
This script helps determine which feature type shows the strongest signal for shot boundary detection.

Runs analysis for two sampling rates:
1. No sampling (rate=1) - all frames processed
2. High sampling (rate=8) - every 8th frame processed

Applies smoothing filters to better visualize peaks and troughs.
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from scipy.signal import savgol_filter
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Hardcoded configuration from train_transformer.py usage
VIDEO_DIR = r"C:\Users\Baraa\Downloads\ClipShots\videos\train"
GROUND_TRUTH_FILE = r"C:\Users\Baraa\Downloads\ClipShots\annotations\train.json"

# Feature directories for different sampling rates
FEATURES_DIRS = {
    'no_sampling': 'training_features_no_sampling',
    'sampling_8': 'training_features_sampling_8'
}

# Main output directory (sampling subdirectories will be created within video folders)
OUTPUT_DIR = 'feature_analysis_results'

# Smoothing parameters
SMOOTHING_WINDOW = 5  # Window size for smoothing filters
SMOOTHING_METHOD = 'gaussian'  # 'gaussian', 'savgol', or 'none'

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
    # Convert to float32 if needed for better numerical stability
    if features.dtype == np.float16:
        features = features.astype(np.float32)
    
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
    
    return np.array(correlations, dtype=np.float32)


def calculate_euclidean_distance(features: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance between consecutive frames.
    
    Args:
        features: Feature array of shape (N, D) where N is frames, D is feature dimension
        
    Returns:
        Array of shape (N-1,) with Euclidean distances between consecutive frames
    """
    # Convert to float32 if needed for better numerical stability
    if features.dtype == np.float16:
        features = features.astype(np.float32)
    
    distances = []
    
    for i in range(len(features) - 1):
        frame1 = features[i]
        frame2 = features[i + 1]
        
        # Calculate Euclidean distance
        dist = np.linalg.norm(frame1 - frame2)
        distances.append(dist)
    
    return np.array(distances, dtype=np.float32)


def apply_smoothing(signal: np.ndarray, window_size: int = 5, method: str = 'gaussian') -> np.ndarray:
    """
    Apply smoothing filter to a 1D signal to better visualize peaks and troughs.
    
    Args:
        signal: 1D array to smooth
        window_size: Size of the smoothing window
        method: Smoothing method ('gaussian', 'savgol', 'none')
        
    Returns:
        Smoothed signal of same shape
    """
    if method == 'none' or len(signal) < window_size:
        return signal
    
    # Convert to float32 if needed (scipy doesn't support float16)
    original_dtype = signal.dtype
    if signal.dtype == np.float16:
        signal = signal.astype(np.float32)
    
    if method == 'gaussian':
        # Gaussian filter - good for general smoothing
        sigma = window_size / 3.0  # Standard deviation
        smoothed = gaussian_filter1d(signal, sigma=sigma, mode='nearest')
    
    elif method == 'savgol':
        # Savitzky-Golay filter - preserves peaks better
        # Window size must be odd
        if window_size % 2 == 0:
            window_size += 1
        # Polynomial order (must be less than window_size)
        polyorder = min(3, window_size - 1)
        smoothed = savgol_filter(signal, window_length=window_size, polyorder=polyorder, mode='nearest')
    
    else:
        return signal
    
    # Convert back to original dtype if it was float16
    if original_dtype == np.float16:
        smoothed = smoothed.astype(np.float16)
    
    return smoothed


def plot_video_analysis(video_name: str, video_id: int, frame_count: int, boundaries: List[int],
                        results: Dict[str, Dict], output_dir: str, sampling_label: str = ""):
    """
    Create plots for a single video showing correlation and distance for all features.
    Now includes both raw and smoothed signals.
    
    Args:
        video_name: Name of the video
        video_id: Numeric ID of the video (1-10)
        frame_count: Total number of frames
        boundaries: List of boundary frame indices
        results: Dictionary with results for each feature type
        output_dir: Directory to save plots (sampling-specific directory)
        sampling_label: Label for sampling rate (e.g., "no_sampling" or "sampling_8")
    """
    # Output directory is already sampling-specific (e.g., video_name/no_sampling/)
    os.makedirs(output_dir, exist_ok=True)
    
    num_features = len(FEATURE_TYPES)
    
    # Plot 5: Side-by-side comparison with smoothing (correlation and distance) for each feature
    for feature_type in FEATURE_TYPES:
        if feature_type not in results:
            continue
        
        correlations = results[feature_type]['correlation']
        distances = results[feature_type]['distance']
        
        # Apply smoothing
        correlations_smooth = apply_smoothing(correlations, SMOOTHING_WINDOW, SMOOTHING_METHOD)
        distances_smooth = apply_smoothing(distances, SMOOTHING_WINDOW, SMOOTHING_METHOD)
        
        frames = np.arange(1, len(correlations) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex='col')
        
        # Top-left: Raw correlation
        axes[0, 0].plot(frames, correlations, linewidth=1.5, color='blue', alpha=0.5, label='Raw')
        axes[0, 0].plot(frames, correlations_smooth, linewidth=2, color='darkblue', alpha=0.9, label='Smoothed')
        axes[0, 0].set_ylabel('Correlation Coefficient', fontsize=11)
        axes[0, 0].set_title('Correlation (Raw vs Smoothed)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(correlations):
                axes[0, 0].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Top-right: Raw distance
        axes[0, 1].plot(frames, distances, linewidth=1.5, color='green', alpha=0.5, label='Raw')
        axes[0, 1].plot(frames, distances_smooth, linewidth=2, color='darkgreen', alpha=0.9, label='Smoothed')
        axes[0, 1].set_ylabel('Euclidean Distance', fontsize=11)
        axes[0, 1].set_title('Distance (Raw vs Smoothed)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(distances):
                axes[0, 1].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Bottom-left: Smoothed correlation only
        axes[1, 0].plot(frames, correlations_smooth, linewidth=2, color='darkblue', alpha=0.9)
        axes[1, 0].set_ylabel('Correlation Coefficient', fontsize=11)
        axes[1, 0].set_xlabel('Frame Index', fontsize=11)
        axes[1, 0].set_title('Smoothed Correlation Only', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(correlations):
                axes[1, 0].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Bottom-right: Smoothed distance only
        axes[1, 1].plot(frames, distances_smooth, linewidth=2, color='darkgreen', alpha=0.9)
        axes[1, 1].set_ylabel('Euclidean Distance', fontsize=11)
        axes[1, 1].set_xlabel('Frame Index', fontsize=11)
        axes[1, 1].set_title('Smoothed Distance Only', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add vertical lines for boundaries
        for boundary in boundaries:
            if boundary < len(distances):
                axes[1, 1].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                          label='Boundary' if boundary == boundaries[0] else '')
        
        if boundaries:
            axes[1, 1].legend()
        
        sampling_suffix = f"_{sampling_label}" if sampling_label else ""
        fig.suptitle(f'Video {video_id} - {feature_type.upper()} with Smoothing ({SMOOTHING_METHOD})', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature_type}_smoothed_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved plots to {output_dir}")


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize a signal to [0, 1] range using min-max normalization.
    
    Args:
        signal: 1D array to normalize
        
    Returns:
        Normalized signal of same shape
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    # Avoid division by zero
    if max_val - min_val < 1e-10:
        return np.zeros_like(signal)
    
    return (signal - min_val) / (max_val - min_val)


def plot_all_features_side_by_side(video_name: str, video_id: int, frame_count: int, boundaries: List[int],
                                     results_no_sampling: Dict[str, Dict], 
                                     results_sampling_8: Dict[str, Dict],
                                     output_dir: str):
    """
    Create 4-column plots showing all features for both sampling rates.
    Columns: Correlation (no_sampling), Correlation (sampling_8), Distance (no_sampling), Distance (sampling_8)
    Creates raw, smoothed, normalized, and normalized+smoothed versions.
    
    Args:
        video_name: Name of the video
        video_id: Numeric ID of the video (1-10)
        frame_count: Total number of frames
        boundaries: List of boundary frame indices
        results_no_sampling: Dictionary with results for no_sampling
        results_sampling_8: Dictionary with results for sampling_8
        output_dir: Directory to save the plot (video-specific directory)
    """
    num_features = len(FEATURE_TYPES)
    
    # Create four versions: raw, smoothed, normalized, and normalized+smoothed
    for version in ['smoothed', 'raw', 'normalized', 'normalized_smoothed']:
        fig, axes = plt.subplots(num_features, 4, figsize=(24, 3 * num_features), sharex='col')
        if num_features == 1:
            axes = axes.reshape(1, -1)
        
        for idx, feature_type in enumerate(FEATURE_TYPES):
            # Process no_sampling data
            if feature_type in results_no_sampling:
                correlations_0 = results_no_sampling[feature_type]['correlation']
                distances_0 = results_no_sampling[feature_type]['distance']
                
                if version == 'smoothed':
                    correlations_0_plot = apply_smoothing(correlations_0, SMOOTHING_WINDOW, SMOOTHING_METHOD)
                    distances_0_plot = apply_smoothing(distances_0, SMOOTHING_WINDOW, SMOOTHING_METHOD)
                elif version == 'normalized':
                    correlations_0_plot = normalize_signal(correlations_0)
                    distances_0_plot = normalize_signal(distances_0)
                elif version == 'normalized_smoothed':
                    correlations_0_plot = apply_smoothing(normalize_signal(correlations_0), SMOOTHING_WINDOW, SMOOTHING_METHOD)
                    distances_0_plot = apply_smoothing(normalize_signal(distances_0), SMOOTHING_WINDOW, SMOOTHING_METHOD)
                else:
                    correlations_0_plot = correlations_0
                    distances_0_plot = distances_0
                
                frames_0 = np.arange(1, len(correlations_0) + 1)
            else:
                correlations_0_plot = distances_0_plot = frames_0 = None
            
            # Process sampling_8 data
            if feature_type in results_sampling_8:
                correlations_8 = results_sampling_8[feature_type]['correlation']
                distances_8 = results_sampling_8[feature_type]['distance']
                
                if version == 'smoothed':
                    correlations_8_plot = apply_smoothing(correlations_8, SMOOTHING_WINDOW, SMOOTHING_METHOD)
                    distances_8_plot = apply_smoothing(distances_8, SMOOTHING_WINDOW, SMOOTHING_METHOD)
                elif version == 'normalized':
                    correlations_8_plot = normalize_signal(correlations_8)
                    distances_8_plot = normalize_signal(distances_8)
                elif version == 'normalized_smoothed':
                    correlations_8_plot = apply_smoothing(normalize_signal(correlations_8), SMOOTHING_WINDOW, SMOOTHING_METHOD)
                    distances_8_plot = apply_smoothing(normalize_signal(distances_8), SMOOTHING_WINDOW, SMOOTHING_METHOD)
                else:
                    correlations_8_plot = correlations_8
                    distances_8_plot = distances_8
                
                frames_8 = np.arange(1, len(correlations_8) + 1)
            else:
                correlations_8_plot = distances_8_plot = frames_8 = None
            
            # Set colors and alpha based on version
            if version in ['smoothed', 'normalized_smoothed']:
                corr_color, dist_color = 'darkblue', 'darkgreen'
                line_alpha = 0.9
                linewidth = 2
            else:
                corr_color, dist_color = 'blue', 'green'
                line_alpha = 0.7
                linewidth = 1.5
            
            # Column 0: Correlation (no_sampling)
            if correlations_0_plot is not None:
                axes[idx, 0].plot(frames_0, correlations_0_plot, linewidth=linewidth, alpha=line_alpha, color=corr_color)
                axes[idx, 0].set_ylabel(f'{feature_type.upper()}\nCorrelation', fontsize=10)
                axes[idx, 0].grid(True, alpha=0.3)
                for boundary in boundaries:
                    if boundary < len(correlations_0_plot):
                        axes[idx, 0].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            # Column 1: Correlation (sampling_8)
            if correlations_8_plot is not None:
                axes[idx, 1].plot(frames_8, correlations_8_plot, linewidth=linewidth, alpha=line_alpha, color=corr_color)
                axes[idx, 1].set_ylabel(f'{feature_type.upper()}\nCorrelation', fontsize=10)
                axes[idx, 1].grid(True, alpha=0.3)
                for boundary in boundaries:
                    if boundary < len(correlations_8_plot):
                        axes[idx, 1].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            # Column 2: Distance (no_sampling)
            if distances_0_plot is not None:
                axes[idx, 2].plot(frames_0, distances_0_plot, linewidth=linewidth, alpha=line_alpha, color=dist_color)
                axes[idx, 2].set_ylabel(f'{feature_type.upper()}\nDistance', fontsize=10)
                axes[idx, 2].grid(True, alpha=0.3)
                for boundary in boundaries:
                    if boundary < len(distances_0_plot):
                        axes[idx, 2].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            # Column 3: Distance (sampling_8)
            if distances_8_plot is not None:
                axes[idx, 3].plot(frames_8, distances_8_plot, linewidth=linewidth, alpha=line_alpha, color=dist_color)
                axes[idx, 3].set_ylabel(f'{feature_type.upper()}\nDistance', fontsize=10)
                axes[idx, 3].grid(True, alpha=0.3)
                for boundary in boundaries:
                    if boundary < len(distances_8_plot):
                        axes[idx, 3].axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Set column titles
        axes[0, 0].set_title('Correlation (Rate 0)', fontsize=12, fontweight='bold', pad=10)
        axes[0, 1].set_title('Correlation (Rate 8)', fontsize=12, fontweight='bold', pad=10)
        axes[0, 2].set_title('Distance (Rate 0)', fontsize=12, fontweight='bold', pad=10)
        axes[0, 3].set_title('Distance (Rate 8)', fontsize=12, fontweight='bold', pad=10)
        
        # Set x-axis labels
        for col in range(4):
            axes[-1, col].set_xlabel('Frame Index', fontsize=11)
        
        # Set title based on version
        if version == 'smoothed':
            version_label = 'Smoothed'
        elif version == 'normalized':
            version_label = 'Normalized'
        elif version == 'normalized_smoothed':
            version_label = 'Normalized + Smoothed'
        else:
            version_label = 'Raw'
        
        fig.suptitle(f'Video {video_id} - All Features Comparison ({version_label})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'all_features_side_by_side_{version}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved 4-column all-features plots (raw, smoothed, normalized, and normalized+smoothed) to {output_dir}")
def create_combined_plots(all_results_by_sampling: Dict[str, Dict[str, Dict]], video_id_map: Dict[str, int], output_dir: str):
    """
    Create combined plots showing all videos for each feature type.
    Now creates 4-column plots (correlation rate 0, correlation rate 8, distance rate 0, distance rate 8).
    Creates raw, smoothed, normalized, and normalized+smoothed versions.
    
    Args:
        all_results_by_sampling: Dictionary mapping sampling_key -> video_name -> results dict
        video_id_map: Dictionary mapping video_name to video_id (1-10)
        output_dir: Directory to save plots (main output directory)
    """
    print("\nCreating combined plots for all videos...")
    
    # Get list of videos (should be same across sampling rates)
    video_names = list(next(iter(all_results_by_sampling.values())).keys())
    
    for feature_type in FEATURE_TYPES:
        print(f"  Processing {feature_type}...")
        
        # Create four versions: raw, smoothed, normalized, and normalized+smoothed
        for version in ['smoothed', 'raw', 'normalized', 'normalized_smoothed']:
            fig = plt.figure(figsize=(28, 2.5 * len(video_names)))
            gs = fig.add_gridspec(len(video_names), 4, hspace=0.3, wspace=0.3)
            
            for idx, video_name in enumerate(video_names):
                # Get data from both sampling rates
                video_data_0 = all_results_by_sampling.get('no_sampling', {}).get(video_name, {})
                video_data_8 = all_results_by_sampling.get('sampling_8', {}).get(video_name, {})
                
                boundaries = video_data_0.get('boundaries', video_data_8.get('boundaries', []))
                
                # Process no_sampling data
                if feature_type in video_data_0.get('results', {}):
                    correlations_0 = video_data_0['results'][feature_type]['correlation']
                    distances_0 = video_data_0['results'][feature_type]['distance']
                    
                    if version == 'smoothed':
                        correlations_0_plot = apply_smoothing(correlations_0, SMOOTHING_WINDOW, SMOOTHING_METHOD)
                        distances_0_plot = apply_smoothing(distances_0, SMOOTHING_WINDOW, SMOOTHING_METHOD)
                    elif version == 'normalized':
                        correlations_0_plot = normalize_signal(correlations_0)
                        distances_0_plot = normalize_signal(distances_0)
                    elif version == 'normalized_smoothed':
                        correlations_0_plot = apply_smoothing(normalize_signal(correlations_0), SMOOTHING_WINDOW, SMOOTHING_METHOD)
                        distances_0_plot = apply_smoothing(normalize_signal(distances_0), SMOOTHING_WINDOW, SMOOTHING_METHOD)
                    else:
                        correlations_0_plot = correlations_0
                        distances_0_plot = distances_0
                    
                    frames_0 = np.arange(1, len(correlations_0) + 1)
                else:
                    correlations_0_plot = distances_0_plot = frames_0 = None
                
                # Process sampling_8 data
                if feature_type in video_data_8.get('results', {}):
                    correlations_8 = video_data_8['results'][feature_type]['correlation']
                    distances_8 = video_data_8['results'][feature_type]['distance']
                    
                    if version == 'smoothed':
                        correlations_8_plot = apply_smoothing(correlations_8, SMOOTHING_WINDOW, SMOOTHING_METHOD)
                        distances_8_plot = apply_smoothing(distances_8, SMOOTHING_WINDOW, SMOOTHING_METHOD)
                    elif version == 'normalized':
                        correlations_8_plot = normalize_signal(correlations_8)
                        distances_8_plot = normalize_signal(distances_8)
                    elif version == 'normalized_smoothed':
                        correlations_8_plot = apply_smoothing(normalize_signal(correlations_8), SMOOTHING_WINDOW, SMOOTHING_METHOD)
                        distances_8_plot = apply_smoothing(normalize_signal(distances_8), SMOOTHING_WINDOW, SMOOTHING_METHOD)
                    else:
                        correlations_8_plot = correlations_8
                        distances_8_plot = distances_8
                    
                    frames_8 = np.arange(1, len(correlations_8) + 1)
                else:
                    correlations_8_plot = distances_8_plot = frames_8 = None
                
                # Set colors and alpha based on version
                if version in ['smoothed', 'normalized_smoothed']:
                    corr_color, dist_color = 'darkblue', 'darkgreen'
                    line_alpha = 0.9
                    linewidth = 2
                else:
                    corr_color, dist_color = 'blue', 'green'
                    line_alpha = 0.7
                    linewidth = 1.5
                
                # Column 0: Correlation (no_sampling)
                ax_corr_0 = fig.add_subplot(gs[idx, 0])
                if correlations_0_plot is not None:
                    ax_corr_0.plot(frames_0, correlations_0_plot, linewidth=linewidth, alpha=line_alpha, color=corr_color)
                    for boundary in boundaries:
                        if boundary < len(correlations_0_plot):
                            ax_corr_0.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax_corr_0.set_ylabel('Correlation', fontsize=9)
                if idx == 0:
                    ax_corr_0.set_title('Correlation (Rate 0)', fontsize=11, fontweight='bold')
                video_id = video_id_map.get(video_name, idx + 1)
                ax_corr_0.text(0.02, 0.95, f'Video {video_id}',
                              transform=ax_corr_0.transAxes, fontsize=8, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                ax_corr_0.grid(True, alpha=0.3)
                
                # Column 1: Correlation (sampling_8)
                ax_corr_8 = fig.add_subplot(gs[idx, 1])
                if correlations_8_plot is not None:
                    ax_corr_8.plot(frames_8, correlations_8_plot, linewidth=linewidth, alpha=line_alpha, color=corr_color)
                    for boundary in boundaries:
                        if boundary < len(correlations_8_plot):
                            ax_corr_8.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax_corr_8.set_ylabel('Correlation', fontsize=9)
                if idx == 0:
                    ax_corr_8.set_title('Correlation (Rate 8)', fontsize=11, fontweight='bold')
                ax_corr_8.grid(True, alpha=0.3)
                
                # Column 2: Distance (no_sampling)
                ax_dist_0 = fig.add_subplot(gs[idx, 2])
                if distances_0_plot is not None:
                    ax_dist_0.plot(frames_0, distances_0_plot, linewidth=linewidth, alpha=line_alpha, color=dist_color)
                    for boundary in boundaries:
                        if boundary < len(distances_0_plot):
                            ax_dist_0.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax_dist_0.set_ylabel('Distance', fontsize=9)
                if idx == 0:
                    ax_dist_0.set_title('Distance (Rate 0)', fontsize=11, fontweight='bold')
                ax_dist_0.grid(True, alpha=0.3)
                
                # Column 3: Distance (sampling_8)
                ax_dist_8 = fig.add_subplot(gs[idx, 3])
                if distances_8_plot is not None:
                    ax_dist_8.plot(frames_8, distances_8_plot, linewidth=linewidth, alpha=line_alpha, color=dist_color)
                    for boundary in boundaries:
                        if boundary < len(distances_8_plot):
                            ax_dist_8.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax_dist_8.set_ylabel('Distance', fontsize=9)
                if idx == 0:
                    ax_dist_8.set_title('Distance (Rate 8)', fontsize=11, fontweight='bold')
                ax_dist_8.grid(True, alpha=0.3)
                
                # Set x-label only for bottom row
                if idx == len(video_names) - 1:
                    ax_corr_0.set_xlabel('Frame Index', fontsize=10)
                    ax_corr_8.set_xlabel('Frame Index', fontsize=10)
                    ax_dist_0.set_xlabel('Frame Index', fontsize=10)
                    ax_dist_8.set_xlabel('Frame Index', fontsize=10)
            
            # Set title based on version
            if version == 'smoothed':
                version_label = 'Smoothed'
            elif version == 'normalized':
                version_label = 'Normalized'
            elif version == 'normalized_smoothed':
                version_label = 'Normalized + Smoothed'
            else:
                version_label = 'Raw'
            
            fig.suptitle(f'All Videos - {feature_type.upper()} ({version_label})', fontsize=14, fontweight='bold')
            plt.savefig(os.path.join(output_dir, f'all_videos_{feature_type}_{version}.png'), dpi=150, bbox_inches='tight')
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
    """Main function to analyze features for selected videos with multiple sampling rates."""
    print("=" * 80)
    print("FEATURE DISTANCE AND CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Smoothing method: {SMOOTHING_METHOD}")
    print(f"  Smoothing window: {SMOOTHING_WINDOW}")
    print(f"  Feature types: {', '.join(FEATURE_TYPES)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load selected videos once
    print(f"\nLoading selected videos from selected_videos.json...")
    selected_videos = load_selected_videos()
    print(f"Found {len(selected_videos)} videos to analyze")
    
    # Dictionary to store results by sampling rate
    all_results_by_sampling = {}
    
    # Create video ID mapping (1-10)
    video_id_map = {video[0]: idx + 1 for idx, video in enumerate(selected_videos)}
    
    # Process each sampling rate
    for sampling_key, features_dir in FEATURES_DIRS.items():
        print("\n" + "=" * 80)
        print(f"PROCESSING: {sampling_key.upper().replace('_', ' ')}")
        print("=" * 80)
        print(f"  Features directory: {features_dir}")
        
        # Process each video
        all_results = {}
        
        for video_name, frame_count, transitions, boundaries in selected_videos:
            video_id = video_id_map[video_name]
            print(f"\n{'='*60}")
            print(f"Processing: Video {video_id} ({video_name})")
            print(f"  Frames: {frame_count}, Boundaries: {len(boundaries)}")
            print(f"  Boundary positions: {boundaries}")
            
            results = {}
            
            # Process each feature type
            for feature_type in FEATURE_TYPES:
                print(f"\n  Analyzing {feature_type}...")
                
                try:
                    # Load features
                    features = load_feature(features_dir, video_name, feature_type)
                    
                    # Calculate correlation and distance
                    correlations = calculate_correlation(features)
                    distances = calculate_euclidean_distance(features)
                    
                    # Apply smoothing
                    correlations_smooth = apply_smoothing(correlations, SMOOTHING_WINDOW, SMOOTHING_METHOD)
                    distances_smooth = apply_smoothing(distances, SMOOTHING_WINDOW, SMOOTHING_METHOD)
                    
                    results[feature_type] = {
                        'correlation': correlations,
                        'distance': distances,
                        'correlation_smooth': correlations_smooth,
                        'distance_smooth': distances_smooth
                    }
                    
                    print(f"    Correlation shape: {correlations.shape}")
                    print(f"    Distance shape: {distances.shape}")
                    print(f"    Correlation range: [{correlations.min():.4f}, {correlations.max():.4f}]")
                    print(f"    Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                    print(f"    Smoothed correlation range: [{correlations_smooth.min():.4f}, {correlations_smooth.max():.4f}]")
                    print(f"    Smoothed distance range: [{distances_smooth.min():.4f}, {distances_smooth.max():.4f}]")
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Store results
            all_results[video_name] = {
                'frame_count': frame_count,
                'boundaries': boundaries,
                'transitions': transitions,
                'results': results
            }
            
            # Create per-feature plots in sampling-specific subdirectory
            video_dir = os.path.join(OUTPUT_DIR, str(video_id))
            sampling_dir = os.path.join(video_dir, sampling_key)
            plot_video_analysis(video_name, video_id, frame_count, boundaries, results, sampling_dir, sampling_key)
        
        # Store results for this sampling rate
        all_results_by_sampling[sampling_key] = all_results
        
        # Calculate and display statistics for this sampling rate
        print("\n" + "=" * 80)
        print(f"AGGREGATE STATISTICS - {sampling_key.upper().replace('_', ' ')}")
        print("=" * 80)
        stats = calculate_statistics(all_results)
        
        # Save statistics to JSON in sampling-specific location
        stats_file = os.path.join(OUTPUT_DIR, f'statistics_{sampling_key}.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_file}")
        
        # Save results to JSON in sampling-specific location
        results_file = os.path.join(OUTPUT_DIR, f'all_results_{sampling_key}.json')
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
                    'distance': feature_data['distance'].tolist(),
                    'correlation_smooth': feature_data['correlation_smooth'].tolist(),
                    'distance_smooth': feature_data['distance_smooth'].tolist()
                }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Detailed results saved to: {results_file}")
        
        print(f"\n✓ Analysis complete for {sampling_key}")
    
    # Create 4-column all-features plot for each video
    print("\n" + "=" * 80)
    print("CREATING 4-COLUMN ALL-FEATURES PLOTS FOR EACH VIDEO")
    print("=" * 80)
    
    for video_name in selected_videos:
        video_name_str = video_name[0]
        video_id = video_id_map[video_name_str]
        frame_count = video_name[1]
        boundaries = video_name[3]
        
        print(f"\nCreating 4-column plot for Video {video_id} ({video_name_str})...")
        
        results_no_sampling = all_results_by_sampling.get('no_sampling', {}).get(video_name_str, {}).get('results', {})
        results_sampling_8 = all_results_by_sampling.get('sampling_8', {}).get(video_name_str, {}).get('results', {})
        
        video_dir = os.path.join(OUTPUT_DIR, str(video_id))
        os.makedirs(video_dir, exist_ok=True)
        
        plot_all_features_side_by_side(video_name_str, video_id, frame_count, boundaries,
                                       results_no_sampling, results_sampling_8, video_dir)
    
    # Create combined 4-column plots in parent directory
    print("\n" + "=" * 80)
    print("CREATING COMBINED 4-COLUMN PLOTS IN PARENT DIRECTORY")
    print("=" * 80)
    create_combined_plots(all_results_by_sampling, video_id_map, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print("=" * 80)
    print(f"\nResults structure:")
    print(f"  Main directory: {OUTPUT_DIR}/")
    print(f"    - Combined 4-column plots for all videos")
    print(f"    - Statistics and results JSON files for each sampling rate")
    print(f"  Video directories: {OUTPUT_DIR}/1/ through {OUTPUT_DIR}/10/")
    print(f"    - 4-column all-features plot")
    print(f"    - Sampling subdirectories: no_sampling/ and sampling_8/")
    print(f"      - Individual per-feature plots")
    print(f"\nVideo ID mapping:")
    for video_name, video_id in sorted(video_id_map.items(), key=lambda x: x[1]):
        print(f"  Video {video_id}: {video_name}")


if __name__ == "__main__":
    main()

