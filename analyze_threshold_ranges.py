"""
Analyze correlation value distributions to understand optimal threshold ranges.
This script helps determine what thresholds to use by examining:
1. Distribution of correlation values at boundaries vs non-boundaries
2. Optimal threshold ranges for each feature type
3. Separation between boundary and non-boundary distributions
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Configuration
ANALYSIS_RESULTS_DIR = 'feature_analysis_results'
OUTPUT_DIR = 'threshold_analysis_results'

# Feature types to analyze
FEATURE_TYPES = ['cnn', 'hsv_histogram', 'clip', 'optical_flow']

# Sampling keys
SAMPLING_KEYS = ['no_sampling', 'sampling_8']


def load_selected_videos() -> List[Tuple[str, int, List, List]]:
    """Load the selected videos from the JSON file."""
    with open('selected_videos.json', 'r') as f:
        data = json.load(f)
    return [(item[0], item[1], item[2], item[3]) for item in data]


def load_analysis_results(sampling_key: str) -> Dict:
    """Load pre-computed analysis results."""
    results_file = os.path.join(ANALYSIS_RESULTS_DIR, f'all_results_{sampling_key}.json')
    with open(results_file, 'r') as f:
        return json.load(f)


def collect_correlation_values(analysis_results: Dict, feature_type: str, version: str = 'raw') -> Dict:
    """
    Collect correlation values at boundaries vs non-boundaries.
    
    Args:
        analysis_results: Pre-computed analysis results
        feature_type: Type of feature to analyze
        version: 'raw', 'smoothed', or 'normalized_smoothed'
        
    Returns:
        Dictionary with boundary and non-boundary correlations
    """
    boundary_correlations = []
    non_boundary_correlations = []
    all_correlations = []
    
    for video_name, video_data in analysis_results.items():
        if feature_type not in video_data['results']:
            continue
        
        # Get correlation values
        if version == 'raw':
            correlations = np.array(video_data['results'][feature_type]['correlation'])
        elif version == 'smoothed':
            correlations = np.array(video_data['results'][feature_type]['correlation_smooth'])
        elif version == 'normalized_smoothed':
            correlations = np.array(video_data['results'][feature_type]['correlation_smooth'])
            # Normalize
            min_val = np.min(correlations)
            max_val = np.max(correlations)
            if max_val - min_val > 1e-10:
                correlations = (correlations - min_val) / (max_val - min_val)
        
        boundaries = video_data['boundaries']
        
        # Separate boundary vs non-boundary
        for i in range(len(correlations)):
            all_correlations.append(correlations[i])
            if i in boundaries:
                boundary_correlations.append(correlations[i])
            else:
                non_boundary_correlations.append(correlations[i])
    
    return {
        'boundary': np.array(boundary_correlations),
        'non_boundary': np.array(non_boundary_correlations),
        'all': np.array(all_correlations)
    }


def calculate_optimal_threshold(boundary_vals: np.ndarray, non_boundary_vals: np.ndarray, 
                                num_thresholds: int = 100) -> Dict:
    """
    Calculate optimal threshold by maximizing F1 score.
    
    Args:
        boundary_vals: Correlation values at boundaries
        non_boundary_vals: Correlation values at non-boundaries
        num_thresholds: Number of thresholds to test
        
    Returns:
        Dictionary with optimal threshold and metrics
    """
    if len(boundary_vals) == 0 or len(non_boundary_vals) == 0:
        return {
            'optimal_threshold': 0.5,
            'best_f1': 0.0,
            'best_precision': 0.0,
            'best_recall': 0.0,
            'thresholds': [],
            'f1_scores': [],
            'precisions': [],
            'recalls': []
        }
    
    # Test range of thresholds
    min_val = min(boundary_vals.min(), non_boundary_vals.min())
    max_val = max(boundary_vals.max(), non_boundary_vals.max())
    thresholds = np.linspace(min_val, max_val, num_thresholds)
    
    f1_scores = []
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        # Predict boundaries: correlation < threshold
        true_positives = np.sum(boundary_vals < threshold)
        false_positives = np.sum(non_boundary_vals < threshold)
        false_negatives = np.sum(boundary_vals >= threshold)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    
    # Find best threshold
    best_idx = np.argmax(f1_scores)
    
    return {
        'optimal_threshold': float(thresholds[best_idx]),
        'best_f1': float(f1_scores[best_idx]),
        'best_precision': float(precisions[best_idx]),
        'best_recall': float(recalls[best_idx]),
        'thresholds': thresholds.tolist(),
        'f1_scores': f1_scores,
        'precisions': precisions,
        'recalls': recalls
    }


def plot_distribution_analysis(data: Dict, feature_type: str, version: str, 
                               optimal_result: Dict, output_dir: str, sampling_key: str):
    """
    Create comprehensive distribution analysis plots.
    """
    boundary_vals = data['boundary']
    non_boundary_vals = data['non_boundary']
    
    if len(boundary_vals) == 0 or len(non_boundary_vals) == 0:
        print(f"    Skipping {feature_type} {version} - no data")
        return
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Histograms
    ax1 = fig.add_subplot(gs[0, :])
    bins = np.linspace(min(boundary_vals.min(), non_boundary_vals.min()),
                       max(boundary_vals.max(), non_boundary_vals.max()), 50)
    ax1.hist(non_boundary_vals, bins=bins, alpha=0.6, label='Non-Boundary', color='blue', density=True)
    ax1.hist(boundary_vals, bins=bins, alpha=0.6, label='Boundary', color='red', density=True)
    ax1.axvline(x=optimal_result['optimal_threshold'], color='green', linestyle='--', 
               linewidth=2, label=f'Optimal Threshold = {optimal_result["optimal_threshold"]:.4f}')
    ax1.set_xlabel('Correlation Value', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution of Correlation Values', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plots
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.boxplot([non_boundary_vals, boundary_vals], labels=['Non-Boundary', 'Boundary'])
    ax2.axhline(y=optimal_result['optimal_threshold'], color='green', linestyle='--', 
               linewidth=2, alpha=0.7)
    ax2.set_ylabel('Correlation Value', fontsize=12)
    ax2.set_title('Box Plot Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Statistics table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    stats_data = [
        ['Metric', 'Boundary', 'Non-Boundary', 'Separation'],
        ['Mean', f'{boundary_vals.mean():.4f}', f'{non_boundary_vals.mean():.4f}', 
         f'{abs(boundary_vals.mean() - non_boundary_vals.mean()):.4f}'],
        ['Median', f'{np.median(boundary_vals):.4f}', f'{np.median(non_boundary_vals):.4f}',
         f'{abs(np.median(boundary_vals) - np.median(non_boundary_vals)):.4f}'],
        ['Std Dev', f'{boundary_vals.std():.4f}', f'{non_boundary_vals.std():.4f}', ''],
        ['Min', f'{boundary_vals.min():.4f}', f'{non_boundary_vals.min():.4f}', ''],
        ['Max', f'{boundary_vals.max():.4f}', f'{non_boundary_vals.max():.4f}', ''],
        ['Count', f'{len(boundary_vals)}', f'{len(non_boundary_vals)}', ''],
        ['', '', '', ''],
        ['Optimal Threshold', f'{optimal_result["optimal_threshold"]:.4f}', '', ''],
        ['Best F1', f'{optimal_result["best_f1"]:.4f}', '', ''],
        ['Best Precision', f'{optimal_result["best_precision"]:.4f}', '', ''],
        ['Best Recall', f'{optimal_result["best_recall"]:.4f}', '', ''],
    ]
    
    table = ax3.table(cellText=stats_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Statistical Summary', fontsize=13, fontweight='bold', pad=20)
    
    # Plot 4: Threshold vs Metrics
    ax4 = fig.add_subplot(gs[2, :])
    thresholds = optimal_result['thresholds']
    ax4.plot(thresholds, optimal_result['precisions'], label='Precision', linewidth=2, color='blue')
    ax4.plot(thresholds, optimal_result['recalls'], label='Recall', linewidth=2, color='red')
    ax4.plot(thresholds, optimal_result['f1_scores'], label='F1 Score', linewidth=2, color='green')
    ax4.axvline(x=optimal_result['optimal_threshold'], color='black', linestyle='--', 
               linewidth=2, alpha=0.7, label='Optimal Threshold')
    ax4.set_xlabel('Threshold Value', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Threshold vs Performance Metrics', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    fig.suptitle(f'{feature_type.upper()} - {version.upper()} ({sampling_key.replace("_", " ").title()})',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.savefig(os.path.join(output_dir, f'{feature_type}_{version}_{sampling_key}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_comparison(all_results: Dict, output_dir: str, sampling_key: str):
    """
    Create summary comparison across all features and versions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    versions = ['raw', 'smoothed', 'normalized_smoothed']
    version_labels = ['Raw', 'Smoothed', 'Normalized+Smoothed']
    
    # Collect data
    optimal_thresholds = {ft: [] for ft in FEATURE_TYPES}
    best_f1s = {ft: [] for ft in FEATURE_TYPES}
    separations = {ft: [] for ft in FEATURE_TYPES}
    
    for feature_type in FEATURE_TYPES:
        for version in versions:
            key = f'{feature_type}_{version}'
            if key in all_results:
                optimal_thresholds[feature_type].append(all_results[key]['optimal_threshold'])
                best_f1s[feature_type].append(all_results[key]['best_f1'])
                
                boundary_mean = all_results[key]['boundary_mean']
                non_boundary_mean = all_results[key]['non_boundary_mean']
                separations[feature_type].append(abs(boundary_mean - non_boundary_mean))
            else:
                optimal_thresholds[feature_type].append(0)
                best_f1s[feature_type].append(0)
                separations[feature_type].append(0)
    
    x = np.arange(len(versions))
    width = 0.2
    
    # Plot 1: Optimal Thresholds
    ax = axes[0, 0]
    for i, feature_type in enumerate(FEATURE_TYPES):
        ax.bar(x + i * width, optimal_thresholds[feature_type], width, 
               label=feature_type.upper(), alpha=0.8)
    ax.set_ylabel('Optimal Threshold', fontsize=11)
    ax.set_xlabel('Version', fontsize=11)
    ax.set_title('Optimal Thresholds by Feature Type', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(version_labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Best F1 Scores
    ax = axes[0, 1]
    for i, feature_type in enumerate(FEATURE_TYPES):
        ax.bar(x + i * width, best_f1s[feature_type], width, 
               label=feature_type.upper(), alpha=0.8)
    ax.set_ylabel('Best F1 Score', fontsize=11)
    ax.set_xlabel('Version', fontsize=11)
    ax.set_title('Best Achievable F1 Score', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(version_labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Plot 3: Distribution Separation
    ax = axes[1, 0]
    for i, feature_type in enumerate(FEATURE_TYPES):
        ax.bar(x + i * width, separations[feature_type], width, 
               label=feature_type.upper(), alpha=0.8)
    ax.set_ylabel('Mean Separation', fontsize=11)
    ax.set_xlabel('Version', fontsize=11)
    ax.set_title('Boundary vs Non-Boundary Separation', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(version_labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary Table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [['Feature', 'Version', 'Optimal Threshold', 'Best F1']]
    for feature_type in FEATURE_TYPES:
        for i, version in enumerate(versions):
            table_data.append([
                feature_type.upper() if i == 0 else '',
                version_labels[i],
                f'{optimal_thresholds[feature_type][i]:.4f}',
                f'{best_f1s[feature_type][i]:.4f}'
            ])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Optimal Thresholds Summary', fontsize=12, fontweight='bold', pad=20)
    
    fig.suptitle(f'Threshold Analysis Summary - {sampling_key.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, f'summary_comparison_{sampling_key}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    print("=" * 80)
    print("THRESHOLD RANGE ANALYSIS")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load selected videos
    print(f"\nLoading selected videos...")
    selected_videos = load_selected_videos()
    print(f"Found {len(selected_videos)} videos")
    
    # Process each sampling rate
    for sampling_key in SAMPLING_KEYS:
        print("\n" + "=" * 80)
        print(f"PROCESSING: {sampling_key.upper().replace('_', ' ')}")
        print("=" * 80)
        
        # Load analysis results
        print(f"Loading analysis results...")
        analysis_results = load_analysis_results(sampling_key)
        
        # Create sampling-specific output directory
        sampling_output_dir = os.path.join(OUTPUT_DIR, sampling_key)
        os.makedirs(sampling_output_dir, exist_ok=True)
        
        # Store results for summary
        all_results = {}
        
        # Process each feature type and version
        for feature_type in FEATURE_TYPES:
            print(f"\n{feature_type.upper()}:")
            
            for version in ['raw', 'smoothed', 'normalized_smoothed']:
                print(f"  {version}:")
                
                # Collect correlation values
                data = collect_correlation_values(analysis_results, feature_type, version)
                
                if len(data['boundary']) == 0 or len(data['non_boundary']) == 0:
                    print(f"    No data available")
                    continue
                
                # Calculate optimal threshold
                optimal_result = calculate_optimal_threshold(data['boundary'], data['non_boundary'])
                
                print(f"    Boundary mean: {data['boundary'].mean():.4f} ± {data['boundary'].std():.4f}")
                print(f"    Non-boundary mean: {data['non_boundary'].mean():.4f} ± {data['non_boundary'].std():.4f}")
                print(f"    Optimal threshold: {optimal_result['optimal_threshold']:.4f}")
                print(f"    Best F1: {optimal_result['best_f1']:.4f} (P={optimal_result['best_precision']:.4f}, R={optimal_result['best_recall']:.4f})")
                
                # Store results
                all_results[f'{feature_type}_{version}'] = {
                    'optimal_threshold': optimal_result['optimal_threshold'],
                    'best_f1': optimal_result['best_f1'],
                    'best_precision': optimal_result['best_precision'],
                    'best_recall': optimal_result['best_recall'],
                    'boundary_mean': float(data['boundary'].mean()),
                    'boundary_std': float(data['boundary'].std()),
                    'non_boundary_mean': float(data['non_boundary'].mean()),
                    'non_boundary_std': float(data['non_boundary'].std()),
                    'separation': float(abs(data['boundary'].mean() - data['non_boundary'].mean()))
                }
                
                # Create detailed plot
                plot_distribution_analysis(data, feature_type, version, optimal_result, 
                                          sampling_output_dir, sampling_key)
        
        # Create summary comparison
        print(f"\nCreating summary comparison...")
        create_summary_comparison(all_results, sampling_output_dir, sampling_key)
        
        # Save results to JSON
        results_file = os.path.join(sampling_output_dir, 'optimal_thresholds.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - optimal_thresholds.json: Optimal thresholds for each feature/version")
    print(f"  - Individual plots: Distribution analysis for each feature")
    print(f"  - summary_comparison_*.png: Overall comparison plots")


if __name__ == "__main__":
    main()
