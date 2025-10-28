"""
Test balanced configurations to find optimal precision-recall tradeoff.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from balanced_configs import (
    BALANCED_CONFIG_V1,
    BALANCED_CONFIG_V2,
    BALANCED_CONFIG_V3,
    BALANCED_CONFIG_V4,
    BALANCED_CONFIG_V5_NO_HSV,
    TARGET_METRICS
)
from main_test import VideoShotDetectionTester

def test_balanced_configs(video_path: str, ground_truth_file: str, video_name: str):
    """Test all balanced configurations."""
    
    print("="*70)
    print("TESTING BALANCED CONFIGURATIONS")
    print("Goal: Precision ≥ 0.70, Recall ≥ 0.80, F1 ≥ 0.76")
    print("="*70)
    
    # Initialize tester
    tester = VideoShotDetectionTester("test_results_balanced")
    
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
        "V1_Higher_Threshold": BALANCED_CONFIG_V1,
        "V2_Savgol_Filter": BALANCED_CONFIG_V2,
        "V3_Percentile_Top5": BALANCED_CONFIG_V3,
        "V4_Concat_HighBeta": BALANCED_CONFIG_V4,
        "V5_Without_HSV": BALANCED_CONFIG_V5_NO_HSV,
    }
    
    results = {}
    
    # Test each configuration
    for config_name, config in configs_to_test.items():
        print(f"\n{'='*70}")
        print(f"TESTING: {config_name}")
        print(f"  Fusion: {config.fusion.strategy}")
        print(f"  Smoothing: {config.detection.smoothing_method} (window={config.detection.smoothing_window})")
        print(f"  Threshold: {config.detection.threshold_method}", end="")
        if config.detection.threshold_method == "adaptive":
            print(f" (beta={config.detection.adaptive_beta})")
        else:
            print(f" (percentile={config.detection.fixed_threshold})")
        print(f"  Min Shot: {config.processing.min_shot_duration}")
        print(f"  Temporal Tol: {config.processing.temporal_tolerance}")
        print("="*70)
        
        try:
            # Extract features with caching
            cached_frames, cached_features, video_info = tester._extract_features_cached(
                video_path, config
            )
            
            # Run detection
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
                    "threshold_method": config.detection.threshold_method,
                    "adaptive_beta": getattr(config.detection, 'adaptive_beta', None),
                    "fixed_threshold": getattr(config.detection, 'fixed_threshold', None),
                    "min_shot_duration": config.processing.min_shot_duration,
                    "temporal_tolerance": config.processing.temporal_tolerance
                }
            }
            
            # Print results with pass/fail indicators
            print(f"\n✓ Results:")
            print(f"  Boundaries detected: {len(boundaries)}")
            print(f"  Processing time: {metadata['processing_time']:.2f}s")
            
            # Check against targets
            prec_pass = "✅" if accuracy['precision'] >= 0.70 else "❌"
            rec_pass = "✅" if accuracy['recall'] >= 0.80 else "❌"
            f1_pass = "✅" if accuracy['f1_score'] >= 0.76 else "❌"
            fp_pass = "✅" if accuracy['false_positives'] < 20 else "❌"
            fn_pass = "✅" if accuracy['false_negatives'] < 10 else "❌"
            
            print(f"  Precision: {accuracy['precision']:.3f} {prec_pass}")
            print(f"  Recall: {accuracy['recall']:.3f} {rec_pass}")
            print(f"  F1-Score: {accuracy['f1_score']:.3f} {f1_pass}")
            print(f"  True Positives: {accuracy['true_positives']}")
            print(f"  False Positives: {accuracy['false_positives']} {fp_pass}")
            print(f"  False Negatives: {accuracy['false_negatives']} {fn_pass}")
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            results[config_name] = {"error": str(e)}
    
    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Configuration':<22} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Det':<5} {'FP':<5} {'FN':<5} {'Score'}")
    print("-" * 70)
    
    # Sort by F1 score
    sorted_results = sorted(
        [(name, res) for name, res in results.items() if "accuracy" in res],
        key=lambda x: x[1]["accuracy"]["f1_score"],
        reverse=True
    )
    
    for name, res in sorted_results:
        acc = res["accuracy"]
        # Calculate overall score (how many targets met)
        score = 0
        if acc['precision'] >= 0.70: score += 1
        if acc['recall'] >= 0.80: score += 1
        if acc['f1_score'] >= 0.76: score += 1
        if acc['false_positives'] < 20: score += 1
        if acc['false_negatives'] < 10: score += 1
        
        score_str = f"{score}/5 " + ("⭐" * score)
        
        name_short = name.replace("_", " ")
        print(f"{name_short:<22} {acc['f1_score']:<8.3f} {acc['precision']:<8.3f} "
              f"{acc['recall']:<8.3f} {len(res['boundaries']):<5} "
              f"{acc['false_positives']:<5} {acc['false_negatives']:<5} {score_str}")
    
    # Show previous results for comparison
    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS RESULTS")
    print("="*70)
    print("\nOriginal (Hierarchical, beta=1.5):")
    print("  F1: 0.491, Prec: 0.443, Rec: 0.551, Det: 61, FP: 34, FN: 22 | 0/5")
    print("\nImproved (Weighted, beta=1.8):")
    print("  F1: 0.698, Prec: 0.562, Rec: 0.918, Det: 80, FP: 35, FN: 4  | 1/5 ⭐")
    
    # Find best balanced config
    best_balanced = None
    best_score = 0
    
    for name, res in sorted_results:
        if "accuracy" not in res:
            continue
        acc = res["accuracy"]
        score = 0
        if acc['precision'] >= 0.70: score += 1
        if acc['recall'] >= 0.80: score += 1
        if acc['f1_score'] >= 0.76: score += 1
        if acc['false_positives'] < 20: score += 1
        if acc['false_negatives'] < 10: score += 1
        
        if score > best_score or (score == best_score and (best_balanced is None or acc['f1_score'] > best_balanced[1]['accuracy']['f1_score'])):
            best_score = score
            best_balanced = (name, res)
    
    if best_balanced and best_score >= 4:
        name, res = best_balanced
        acc = res["accuracy"]
        print(f"\n{'='*70}")
        print("🏆 BEST BALANCED CONFIGURATION")
        print("="*70)
        print(f"Configuration: {name.replace('_', ' ')}")
        print(f"  F1-Score: {acc['f1_score']:.3f}")
        print(f"  Precision: {acc['precision']:.3f} {'✅ TARGET MET' if acc['precision'] >= 0.70 else '❌'}")
        print(f"  Recall: {acc['recall']:.3f} {'✅ TARGET MET' if acc['recall'] >= 0.80 else '❌'}")
        print(f"  False Positives: {acc['false_positives']} {'✅ TARGET MET' if acc['false_positives'] < 20 else '❌'}")
        print(f"  False Negatives: {acc['false_negatives']} {'✅ TARGET MET' if acc['false_negatives'] < 10 else '❌'}")
        print(f"  Overall Score: {best_score}/5 targets met")
        
        if best_score == 5:
            print("\n🎉 EXCELLENT! All targets achieved!")
        elif best_score == 4:
            print("\n✅ GOOD! Most targets achieved!")
    else:
        print(f"\n{'='*70}")
        print("⚠️ NO CONFIGURATION MEETS ALL TARGETS")
        print("="*70)
        
        if sorted_results:
            name, res = sorted_results[0]
            acc = res["accuracy"]
            print(f"\nBest available: {name.replace('_', ' ')}")
            print(f"  F1: {acc['f1_score']:.3f}, Prec: {acc['precision']:.3f}, Rec: {acc['recall']:.3f}")
            print(f"  FP: {acc['false_positives']}, FN: {acc['false_negatives']}")
            
            print("\n💡 Recommendations:")
            if acc['precision'] < 0.70:
                print(f"  - Precision too low ({acc['precision']:.3f}): Increase adaptive_beta further")
            if acc['recall'] < 0.80:
                print(f"  - Recall too low ({acc['recall']:.3f}): Decrease adaptive_beta")
            if acc['false_positives'] >= 20:
                print(f"  - Too many FP ({acc['false_positives']}): Increase smoothing or temporal_tolerance")
            if acc['false_negatives'] >= 10:
                print(f"  - Too many FN ({acc['false_negatives']}): Lower threshold or min_shot_duration")
    
    # Save results
    output_file = "test_results_balanced/balanced_results.json"
    with open(output_file, 'w') as f:
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
    
    results = test_balanced_configs(video_path, ground_truth, video_name)
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. If a config meets targets, apply it to DEFAULT_PRODUCTION_CONFIG")
    print("\n2. If no config meets targets, try:")
    print("   - Increase adaptive_beta by 0.2-0.3")
    print("   - Increase smoothing_window by 2")
    print("   - Increase temporal_tolerance to 12-15")
    print("\n3. Test on more videos to ensure robustness")
