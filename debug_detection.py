"""Debug script to analyze detection performance issues."""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('test_results/all_results.json', 'r') as f:
    results = json.load(f)

# Get best performing method
best_method = 'aggressive_detection'
data = results['statistical'][best_method]

gt = data['ground_truth']
pred = data['boundaries']

print("="*60)
print("DETECTION PERFORMANCE ANALYSIS")
print("="*60)
print(f"\nGround Truth: {len(gt)} boundaries")
print(f"Predicted: {len(pred)} boundaries")
print(f"Video frames: {data['metadata']['num_frames']}")

# Analyze boundary spacing
gt_diffs = np.diff(gt)
pred_diffs = np.diff(pred)

print(f"\nGround Truth shot lengths:")
print(f"  Mean: {np.mean(gt_diffs):.1f} frames")
print(f"  Median: {np.median(gt_diffs):.1f} frames")
print(f"  Min: {np.min(gt_diffs):.0f} frames")
print(f"  Max: {np.max(gt_diffs):.0f} frames")

print(f"\nPredicted shot lengths:")
print(f"  Mean: {np.mean(pred_diffs):.1f} frames")
print(f"  Median: {np.median(pred_diffs):.1f} frames")
print(f"  Min: {np.min(pred_diffs):.0f} frames")
print(f"  Max: {np.max(pred_diffs):.0f} frames")

# Analyze matching
tolerance = 10
matches = 0
for gt_b in gt:
    for pred_b in pred:
        if abs(gt_b - pred_b) <= tolerance:
            matches += 1
            break

print(f"\nMatching Analysis (tolerance={tolerance}):")
print(f"  Matched GT boundaries: {matches}/{len(gt)} ({100*matches/len(gt):.1f}%)")
print(f"  False positives: {len(pred) - matches} ({100*(len(pred)-matches)/len(pred):.1f}%)")
print(f"  False negatives: {len(gt) - matches} ({100*(len(gt)-matches)/len(gt):.1f}%)")

# Analyze accuracy metrics
metrics = data['accuracy_metrics']
print(f"\nAccuracy Metrics:")
print(f"  Precision: {metrics['precision']:.3f}")
print(f"  Recall: {metrics['recall']:.3f}")
print(f"  F1-Score: {metrics['f1_score']:.3f}")

# Find missed boundaries
missed_gt = []
for gt_b in gt:
    matched = False
    for pred_b in pred:
        if abs(gt_b - pred_b) <= tolerance:
            matched = True
            break
    if not matched:
        missed_gt.append(gt_b)

print(f"\nMissed ground truth boundaries ({len(missed_gt)}):")
if len(missed_gt) > 0:
    print(f"  Samples: {missed_gt[:10]}")
    # Analyze spacing of missed boundaries
    if len(missed_gt) > 1:
        missed_spacing = []
        for i, missed in enumerate(missed_gt):
            # Find nearest GT neighbors
            gt_arr = np.array(gt)
            idx = np.where(gt_arr == missed)[0][0]
            if idx > 0:
                prev_spacing = missed - gt[idx-1]
                missed_spacing.append(prev_spacing)
        
        if missed_spacing:
            print(f"  Spacing before missed shots: mean={np.mean(missed_spacing):.1f}, min={np.min(missed_spacing)}, max={np.max(missed_spacing)}")

# Find false positives
false_pos = []
for pred_b in pred:
    matched = False
    for gt_b in gt:
        if abs(pred_b - gt_b) <= tolerance:
            matched = True
            break
    if not matched:
        false_pos.append(pred_b)

print(f"\nFalse positive boundaries ({len(false_pos)}):")
if len(false_pos) > 0:
    print(f"  Samples: {false_pos[:10]}")

print("\n" + "="*60)
print("POTENTIAL ISSUES")
print("="*60)

issues = []

# Check if many short shots are missed
short_shots_gt = sum(1 for d in gt_diffs if d < 30)
if short_shots_gt > 0:
    issues.append(f"- {short_shots_gt} ground truth shots are < 30 frames (may be filtered by min_shot_duration)")

# Check precision-recall balance
if metrics['precision'] < 0.5 and metrics['recall'] < 0.7:
    issues.append("- Both precision AND recall are low - problem with feature extraction or fusion")
elif metrics['precision'] < 0.5:
    issues.append("- Low precision - too many false positives (threshold too low)")
elif metrics['recall'] < 0.7:
    issues.append("- Low recall - missing real boundaries (threshold too high)")

# Check if prediction count is reasonable
if len(pred) > len(gt) * 1.5:
    issues.append(f"- Predicting {len(pred)/len(gt):.1f}x more boundaries than GT - over-detection")
elif len(pred) < len(gt) * 0.7:
    issues.append(f"- Predicting only {len(pred)/len(gt):.1f}x GT boundaries - under-detection")

if issues:
    for issue in issues:
        print(issue)
else:
    print("No obvious systematic issues detected.")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)
print("1. Visualize the distance signal to check if peaks align with GT")
print("2. Check if feature extraction is working correctly (HSV, CNN, CLIP, Optical Flow)")
print("3. Try different fusion strategies (current: hierarchical)")
print("4. Adjust adaptive_beta parameter (current: 1.2)")
print("5. Check min_shot_duration setting (may filter valid short shots)")
print("6. Consider different smoothing windows")
print("7. Test on more videos to see if this is dataset-specific")
