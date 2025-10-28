"""
Script to check feature files for potential loading issues.
Identifies files that may cause reshape errors during training.
"""

import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def check_feature_file(feature_file: str) -> Dict:
    """
    Check a feature file for potential issues.
    
    Returns:
        Dictionary with file info and any issues found
    """
    result = {
        'file': feature_file,
        'exists': False,
        'readable': False,
        'shape': None,
        'size_bytes': None,
        'issues': []
    }
    
    if not os.path.exists(feature_file):
        result['issues'].append('File does not exist')
        return result
    
    result['exists'] = True
    result['size_bytes'] = os.path.getsize(feature_file)
    
    try:
        # Try to load with memory mapping first (faster for checking)
        with open(feature_file, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
            result['shape'] = shape
            result['dtype'] = str(dtype)
        
        result['readable'] = True
        
        # Check for issues
        if len(shape) != 2:
            result['issues'].append(f'Wrong dimensionality: {len(shape)}D instead of 2D')
        
        if shape[0] == 0:
            result['issues'].append('Zero frames in file')
        
        if len(shape) == 2 and shape[1] == 0:
            result['issues'].append('Zero feature dimensions')
        
        # Check if file size matches expected size
        if len(shape) == 2:
            expected_size = shape[0] * shape[1] * dtype.itemsize + 128  # 128 bytes for header
            if abs(result['size_bytes'] - expected_size) > 1024:  # Allow 1KB tolerance
                result['issues'].append(f'File size mismatch: {result["size_bytes"]} bytes, expected ~{expected_size} bytes')
        
        # Try to actually load a small slice to verify data integrity
        try:
            features = np.load(feature_file)
            if features.size == 0:
                result['issues'].append('File loads but contains no data (size=0)')
            elif np.all(features == 0):
                result['issues'].append('File contains only zeros')
        except Exception as e:
            result['issues'].append(f'Cannot load data: {e}')
            
    except Exception as e:
        result['issues'].append(f'Cannot read header: {e}')
    
    return result


def check_video_features(features_dir: str, video_stem: str, 
                        fusion_strategies: List[str] = None,
                        sequence_length: int = 100,
                        ground_truth: Dict = None) -> Dict:
    """
    Check all feature files for a specific video.
    
    Args:
        features_dir: Directory containing feature files
        video_stem: Video name stem (without extension)
        fusion_strategies: List of fusion strategies to check
        sequence_length: Expected sequence length for training
        ground_truth: Ground truth data to check frame ranges
    
    Returns:
        Dictionary with comprehensive check results
    """
    if fusion_strategies is None:
        fusion_strategies = ["concatenation", "weighted", "pca", "cross_transformer", "hierarchical"]
    
    individual_features = ['hsv_histogram', 'cnn', 'clip', 'optical_flow']
    
    result = {
        'video_stem': video_stem,
        'individual_features': {},
        'fused_features': {},
        'has_issues': False,
        'critical_issues': []
    }
    
    # Check individual features
    for feature_name in individual_features:
        feature_file = os.path.join(features_dir, f"{video_stem}_{feature_name}.npy")
        check = check_feature_file(feature_file)
        result['individual_features'][feature_name] = check
        if check['issues']:
            result['has_issues'] = True
    
    # Check fused features
    for strategy in fusion_strategies:
        feature_file = os.path.join(features_dir, f"{video_stem}_{strategy}.npy")
        check = check_feature_file(feature_file)
        result['fused_features'][strategy] = check
        if check['issues']:
            result['has_issues'] = True
            
            # Check for critical issues that would cause training errors
            if check['shape'] and len(check['shape']) == 2:
                num_frames = check['shape'][0]
                
                # Check if this video would cause issues during training
                if ground_truth and video_stem in ground_truth:
                    video_data = ground_truth[video_stem]
                    if isinstance(video_data, dict) and 'transitions' in video_data:
                        boundaries = [transition[1] for transition in video_data['transitions']]
                    elif isinstance(video_data, list):
                        boundaries = video_data
                    else:
                        boundaries = []
                    
                    # Simulate sequence creation logic
                    for start_idx in range(0, num_frames - sequence_length + 1, 50):  # 50 = overlap
                        end_idx = start_idx + sequence_length
                        
                        # Check if this sequence would fail
                        if end_idx > num_frames:
                            result['critical_issues'].append({
                                'strategy': strategy,
                                'start_frame': start_idx,
                                'end_frame': end_idx,
                                'num_frames': num_frames,
                                'issue': f'Sequence extends beyond file (needs {end_idx}, has {num_frames})'
                            })
    
    return result


def scan_all_features(features_dir: str, ground_truth_file: str = None,
                     fusion_strategies: List[str] = None,
                     sequence_length: int = 100,
                     verbose: bool = False) -> Dict:
    """
    Scan all feature files in a directory for issues.
    
    Args:
        features_dir: Directory containing feature files
        ground_truth_file: Path to ground truth JSON file
        fusion_strategies: List of fusion strategies to check
        sequence_length: Expected sequence length for training
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with scan results
    """
    if fusion_strategies is None:
        fusion_strategies = ["concatenation", "weighted", "pca", "cross_transformer", "hierarchical"]
    
    # Load ground truth if provided
    ground_truth = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r') as f:
            ground_truth_raw = json.load(f)
            # Normalize video names (remove .mp4 extension)
            for video_name, video_data in ground_truth_raw.items():
                video_stem = video_name.replace('.mp4', '') if video_name.endswith('.mp4') else video_name
                ground_truth[video_stem] = video_data
    
    # Find all unique video stems
    video_stems = set()
    for file in os.listdir(features_dir):
        if file.endswith('.npy'):
            # Extract video stem (everything before the last underscore and feature name)
            parts = file.replace('.npy', '').rsplit('_', 1)
            if len(parts) == 2:
                video_stems.add(parts[0])
    
    print(f"Found {len(video_stems)} unique videos in {features_dir}")
    print(f"Checking {len(fusion_strategies)} fusion strategies: {', '.join(fusion_strategies)}")
    print("=" * 80)
    
    # Check each video
    results = {}
    videos_with_issues = []
    videos_with_critical_issues = []
    
    for video_stem in sorted(video_stems):
        result = check_video_features(
            features_dir, video_stem, fusion_strategies, 
            sequence_length, ground_truth
        )
        results[video_stem] = result
        
        if result['has_issues']:
            videos_with_issues.append(video_stem)
        
        if result['critical_issues']:
            videos_with_critical_issues.append(video_stem)
            print(f"\n⚠️  CRITICAL ISSUES FOUND: {video_stem}")
            for issue in result['critical_issues']:
                print(f"    Strategy: {issue['strategy']}")
                print(f"    Frame range: {issue['start_frame']}-{issue['end_frame']}")
                print(f"    Available frames: {issue['num_frames']}")
                print(f"    Issue: {issue['issue']}")
        
        if verbose or result['critical_issues']:
            print(f"\n{'='*80}")
            print(f"Video: {video_stem}")
            print(f"{'='*80}")
            
            # Print individual features
            print("\nIndividual Features:")
            for feature_name, check in result['individual_features'].items():
                status = "✓" if check['readable'] and not check['issues'] else "✗"
                shape_str = str(check['shape']) if check['shape'] else "N/A"
                print(f"  {status} {feature_name:20} Shape: {shape_str}")
                if check['issues']:
                    for issue in check['issues']:
                        print(f"      - {issue}")
            
            # Print fused features
            print("\nFused Features:")
            for strategy, check in result['fused_features'].items():
                status = "✓" if check['readable'] and not check['issues'] else "✗"
                shape_str = str(check['shape']) if check['shape'] else "N/A"
                print(f"  {status} {strategy:20} Shape: {shape_str}")
                if check['issues']:
                    for issue in check['issues']:
                        print(f"      - {issue}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos: {len(video_stems)}")
    print(f"Videos with issues: {len(videos_with_issues)} ({len(videos_with_issues)/len(video_stems)*100:.1f}%)")
    print(f"Videos with CRITICAL issues: {len(videos_with_critical_issues)} ({len(videos_with_critical_issues)/len(video_stems)*100:.1f}%)")
    
    if videos_with_critical_issues:
        print(f"\n⚠️  Videos that will cause training errors:")
        for video_stem in videos_with_critical_issues:
            print(f"  - {video_stem}")
    
    # Count issues by type
    issue_counts = {}
    for video_stem, result in results.items():
        for strategy, check in result['fused_features'].items():
            for issue in check['issues']:
                issue_key = issue.split(':')[0]  # Get issue type
                issue_counts[issue_key] = issue_counts.get(issue_key, 0) + 1
    
    if issue_counts:
        print(f"\nIssues by type:")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {issue_type}: {count}")
    
    return {
        'total_videos': len(video_stems),
        'videos_with_issues': videos_with_issues,
        'videos_with_critical_issues': videos_with_critical_issues,
        'results': results,
        'issue_counts': issue_counts
    }


def main():
    parser = argparse.ArgumentParser(
        description='Check feature files for potential loading issues during training'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        default='training_features',
        help='Directory containing feature files'
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        help='Path to ground truth JSON file (optional, for detailed analysis)'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=100,
        help='Sequence length used in training (default: 100)'
    )
    parser.add_argument(
        '--strategies',
        type=str,
        nargs='+',
        default=None,
        help='Fusion strategies to check (default: all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information for all videos'
    )
    parser.add_argument(
        '--video_id',
        type=str,
        help='Check a specific video ID only'
    )
    
    args = parser.parse_args()
    
    if args.video_id:
        # Check specific video
        print(f"Checking video: {args.video_id}")
        print("=" * 80)
        result = check_video_features(
            args.features_dir, 
            args.video_id, 
            args.strategies,
            args.sequence_length,
            None
        )
        
        print(f"\nVideo: {args.video_id}")
        print(f"Has issues: {result['has_issues']}")
        
        print("\nIndividual Features:")
        for feature_name, check in result['individual_features'].items():
            status = "✓" if check['readable'] and not check['issues'] else "✗"
            shape_str = str(check['shape']) if check['shape'] else "N/A"
            print(f"  {status} {feature_name:20} Shape: {shape_str}")
            if check['issues']:
                for issue in check['issues']:
                    print(f"      - {issue}")
        
        print("\nFused Features:")
        for strategy, check in result['fused_features'].items():
            status = "✓" if check['readable'] and not check['issues'] else "✗"
            shape_str = str(check['shape']) if check['shape'] else "N/A"
            print(f"  {status} {strategy:20} Shape: {shape_str}")
            if check['issues']:
                for issue in check['issues']:
                    print(f"      - {issue}")
        
        if result['critical_issues']:
            print(f"\n⚠️  CRITICAL ISSUES:")
            for issue in result['critical_issues']:
                print(f"  Strategy: {issue['strategy']}")
                print(f"  Frame range: {issue['start_frame']}-{issue['end_frame']}")
                print(f"  Available frames: {issue['num_frames']}")
                print(f"  Issue: {issue['issue']}")
    else:
        # Scan all videos
        scan_all_features(
            args.features_dir,
            args.ground_truth,
            args.strategies,
            args.sequence_length,
            args.verbose
        )


if __name__ == '__main__':
    main()
