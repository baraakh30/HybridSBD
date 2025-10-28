"""
Test improved configurations to validate performance improvements.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from recommended_fixes import (
    IMPROVED_CONFIG_WEIGHTED,
    IMPROVED_CONFIG_CONCAT,
    IMPROVED_CONFIG_HIERARCHICAL,
    AGGRESSIVE_CONFIG,
    CONSERVATIVE_CONFIG
)
from main_test import VideoShotDetectionTester

def test_configs_comparison(video_path: str, ground_truth_file: str, video_name: str):
    """Test all improved configurations."""
    
    print("="*70)
    print("TESTING IMPROVED CONFIGURATIONS")
    print("="*70)
    
    # Initialize tester
    tester = VideoShotDetectionTester("test_results_improved")
    
    # Load ground truth
    gt_boundaries = tester.load_ground_truth(ground_truth_file, video_name)
    if not gt_boundaries:
        gt_boundaries = tester.load_ground_truth(ground_truth_file, f"{video_name}.mp4")
    
    if not gt_boundaries:
        print(f"ERROR: Could not load ground truth for {video_name}")
        return
    
    print(f"\nVideo: {video_name}")
    print(f"Ground Truth: {len(gt_boundaries)} boundaries")
    print(f"Tolerance: 10 frames")
    
    configs_to_test = {
        "1_Weighted_Fusion": IMPROVED_CONFIG_WEIGHTED,
        "2_Concatenation": IMPROVED_CONFIG_CONCAT,
        "3_Hierarchical_Fixed": IMPROVED_CONFIG_HIERARCHICAL,
        "4_Aggressive": AGGRESSIVE_CONFIG,
        "5_Conservative": CONSERVATIVE_CONFIG,
    }
    
    results = {}
    
    # Extract features once
    print("\n" + "="*70)
    print("EXTRACTING FEATURES (cached for all configs)")
    print("="*70)
    cached_frames, cached_features, video_info = tester._extract_features_cached(
        video_path, IMPROVED_CONFIG_WEIGHTED
    )
    
    # Test each configuration
    for config_name, config in configs_to_test.items():
        print(f"\n{'='*70}")
        print(f"TESTING: {config_name}")
        print(f"  Fusion: {config.fusion.strategy}")
        print(f"  Smoothing: {config.detection.smoothing_method} (window={config.detection.smoothing_window})")
        print(f"  Adaptive Beta: {config.detection.adaptive_beta}")
        print(f"  Min Shot Duration: {config.processing.min_shot_duration}")
        print("="*70)
        
        try:
            # Run detection with cached features
            boundaries, metadata = tester._run_detection_with_cached_features(
                video_path, config, cached_frames, cached_features
            )
            
            # Calculate accuracy
            accuracy = tester.calculate_accuracy_metrics(boundaries, gt_boundaries, tolerance=10)
            
            results[config_name] = {
                "boundaries": boundaries,
                "metadata": metadata,
                "accuracy": accuracy,
                "config": {
                    "fusion": config.fusion.strategy,
                    "smoothing": config.detection.smoothing_method,
                    "smoothing_window": config.detection.smoothing_window,
                    "adaptive_beta": config.detection.adaptive_beta,
                    "min_shot_duration": config.processing.min_shot_duration
                }
            }
            
            # Print results
            print(f"\n✓ Results:")
            print(f"  Boundaries detected: {len(boundaries)}")
            print(f"  Processing time: {metadata['processing_time']:.2f}s")
            print(f"  Precision: {accuracy['precision']:.3f}")
            print(f"  Recall: {accuracy['recall']:.3f}")
            print(f"  F1-Score: {accuracy['f1_score']:.3f}")
            print(f"  True Positives: {accuracy['true_positives']}")
            print(f"  False Positives: {accuracy['false_positives']}")
            print(f"  False Negatives: {accuracy['false_negatives']}")
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            results[config_name] = {"error": str(e)}
    
    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Configuration':<25} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Detected':<10} {'FP':<6} {'FN':<6}")
    print("-" * 70)
    
    # Sort by F1 score
    sorted_results = sorted(
        [(name, res) for name, res in results.items() if "accuracy" in res],
        key=lambda x: x[1]["accuracy"]["f1_score"],
        reverse=True
    )
    
    for name, res in sorted_results:
        acc = res["accuracy"]
        name_short = name.replace("_", " ")
        print(f"{name_short:<25} {acc['f1_score']:<8.3f} {acc['precision']:<8.3f} "
              f"{acc['recall']:<8.3f} {len(res['boundaries']):<10} "
              f"{acc['false_positives']:<6} {acc['false_negatives']:<6}")
    
    # Show baseline (original) for comparison
    print("\n" + "="*70)
    print("COMPARISON WITH ORIGINAL (from previous test)")
    print("="*70)
    print("\nOriginal Hierarchical (default):")
    print("  F1: 0.491, Precision: 0.443, Recall: 0.551, Detected: 61, FP: 34, FN: 22")
    print("\nOriginal Aggressive:")
    print("  F1: 0.536, Precision: 0.476, Recall: 0.612, Detected: 63, FP: 33, FN: 19")
    
    if sorted_results:
        best_name, best_res = sorted_results[0]
        best_acc = best_res["accuracy"]
        print(f"\n{'='*70}")
        print("🏆 BEST CONFIGURATION")
        print("="*70)
        print(f"Configuration: {best_name.replace('_', ' ')}")
        print(f"  F1-Score: {best_acc['f1_score']:.3f} (improvement: {best_acc['f1_score']-0.536:.3f})")
        print(f"  Precision: {best_acc['precision']:.3f} (improvement: {best_acc['precision']-0.476:.3f})")
        print(f"  Recall: {best_acc['recall']:.3f} (improvement: {best_acc['recall']-0.612:.3f})")
        print(f"  Detected: {len(best_res['boundaries'])} (GT: {len(gt_boundaries)})")
        
        if best_acc['f1_score'] > 0.536:
            improvement_pct = ((best_acc['f1_score'] - 0.536) / 0.536) * 100
            print(f"\n✅ SUCCESS: {improvement_pct:.1f}% improvement in F1-score!")
        else:
            print(f"\n⚠️  No improvement over original configuration")
    
    # Save detailed results
    output_file = "test_results_improved/comparison_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\n📄 Detailed results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    video_path = r"C:\Users\Abdalrahman Khanfar\Documents\ClipShots\videos\train\3505586409.mp4"
    ground_truth = r"C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json"
    video_name = "3505586409"
    
    results = test_configs_comparison(video_path, ground_truth, video_name)
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review the visualizations:")
    print("   - test_results/distance_signal_analysis.png")
    print("   - test_results/distance_signal_detailed.png")
    print("\n2. Apply the best configuration to DEFAULT_PRODUCTION_CONFIG in core/config.py")
    print("\n3. Test on more videos to ensure consistency:")
    print('   python main_test.py --video_dir "videos/train" --ground_truth "train.json" --method statistical')
    print("\n4. Consider using the transformer method after getting good statistical results")
