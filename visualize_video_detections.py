"""
Visualize detection results on actual video frames.
Shows detected boundaries, ground truth, false positives, and false negatives.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from main_test import VideoShotDetectionTester
from balanced_configs import BALANCED_CONFIG_V2


def load_video_frames(video_path: str, max_frames: int = None) -> List[np.ndarray]:
    """Load video frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    print(f"Loaded {len(frames)} frames from video")
    return frames


def categorize_detections(predicted: List[int], ground_truth: List[int], tolerance: int = 10) -> Dict:
    """Categorize detections into TP, FP, FN."""
    pred_set = set(predicted)
    gt_set = set(ground_truth)
    
    # Find true positives and matched boundaries
    true_positives = []
    false_negatives = []
    false_positives = list(predicted)  # Start with all predictions
    
    for gt_b in ground_truth:
        matched = False
        for pred_b in predicted:
            if abs(gt_b - pred_b) <= tolerance:
                if pred_b not in true_positives:  # Avoid double matching
                    true_positives.append(pred_b)
                    false_positives.remove(pred_b) if pred_b in false_positives else None
                matched = True
                break
        
        if not matched:
            false_negatives.append(gt_b)
    
    return {
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
        "matched_gt": sorted([gt for gt in ground_truth if any(abs(gt - tp) <= tolerance for tp in true_positives)])
    }


def create_comparison_frame(frame1: np.ndarray, frame2: np.ndarray, 
                           boundary_type: str, frame_num: int,
                           show_text: bool = True) -> np.ndarray:
    """Create a comparison image showing before and after frames."""
    h, w = frame1.shape[:2]
    
    # Resize frames if too large
    max_width = 800
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        frame1 = cv2.resize(frame1, (new_w, new_h))
        frame2 = cv2.resize(frame2, (new_w, new_h))
        h, w = new_h, new_w
    
    # Create side-by-side comparison
    comparison = np.zeros((h + 80, w * 2 + 10, 3), dtype=np.uint8)
    comparison[:h, :w] = frame1
    comparison[:h, w+10:] = frame2
    
    # Add labels and colored border based on type
    if boundary_type == "TP":
        color = (0, 255, 0)  # Green for True Positive
        label = f"TRUE POSITIVE - Frame {frame_num}"
    elif boundary_type == "FP":
        color = (0, 165, 255)  # Orange for False Positive
        label = f"FALSE POSITIVE - Frame {frame_num}"
    elif boundary_type == "FN":
        color = (0, 0, 255)  # Red for False Negative
        label = f"FALSE NEGATIVE (MISSED) - Frame {frame_num}"
    else:
        color = (255, 255, 255)
        label = f"Frame {frame_num}"
    
    # Draw colored border
    cv2.rectangle(comparison, (0, 0), (w * 2 + 10, h + 80), color, 8)
    
    # Add text labels
    if show_text:
        cv2.putText(comparison, "BEFORE", (20, h + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison, "AFTER", (w + 30, h + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison, label, (20, h + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return comparison


def create_visualization_video(video_path: str, predicted: List[int], ground_truth: List[int],
                               output_path: str, tolerance: int = 10,
                               show_all: bool = False):
    """Create a video visualization of all detections."""
    
    print(f"\nCreating visualization video...")
    print(f"  Predicted: {len(predicted)} boundaries")
    print(f"  Ground Truth: {len(ground_truth)} boundaries")
    print(f"  Output: {output_path}")
    
    # Categorize detections
    categories = categorize_detections(predicted, ground_truth, tolerance)
    
    print(f"\n  True Positives: {len(categories['true_positives'])}")
    print(f"  False Positives: {len(categories['false_positives'])}")
    print(f"  False Negatives: {len(categories['false_negatives'])}")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    # Get all boundary frames to visualize
    boundaries_to_show = []
    
    # Add true positives
    for tp in categories['true_positives']:
        boundaries_to_show.append((tp, "TP"))
    
    # Add false positives
    for fp in categories['false_positives']:
        boundaries_to_show.append((fp, "FP"))
    
    # Add false negatives
    for fn in categories['false_negatives']:
        boundaries_to_show.append((fn, "FN"))
    
    boundaries_to_show.sort(key=lambda x: x[0])
    
    print(f"\n  Total boundaries to visualize: {len(boundaries_to_show)}")
    print(f"  Processing...")
    
    current_boundary_idx = 0
    frames_written = 0
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this is a boundary frame we need to visualize
        if current_boundary_idx < len(boundaries_to_show):
            boundary_frame, boundary_type = boundaries_to_show[current_boundary_idx]
            
            if frame_idx == boundary_frame:
                # Get the previous frame for comparison
                if frame_idx > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                    ret, prev_frame = cap.read()
                    if ret:
                        # Create comparison
                        comparison = create_comparison_frame(prev_frame, frame, boundary_type, frame_idx)
                        
                        # Initialize video writer with first frame size
                        if out is None:
                            h, w = comparison.shape[:2]
                            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                        
                        # Write this frame multiple times to make it visible (3 seconds)
                        for _ in range(int(fps * 2)):  # 2 seconds per boundary
                            out.write(comparison)
                            frames_written += 1
                    
                    # Reset to continue from next frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
                
                current_boundary_idx += 1
                
                if (current_boundary_idx) % 10 == 0:
                    print(f"    Processed {current_boundary_idx}/{len(boundaries_to_show)} boundaries")
    
    cap.release()
    if out:
        out.release()
    
    print(f"  ✓ Video created: {frames_written} frames written")
    print(f"  Duration: {frames_written/fps:.1f} seconds")


def create_summary_image(video_path: str, predicted: List[int], ground_truth: List[int],
                        output_path: str, tolerance: int = 10):
    """Create a summary image showing examples of each category."""
    
    print(f"\nCreating summary image...")
    
    # Categorize detections
    categories = categorize_detections(predicted, ground_truth, tolerance)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    # Collect example frames for each category
    examples = {"TP": [], "FP": [], "FN": []}
    max_examples = 3  # Show 3 examples of each
    
    # True positives
    for i, tp in enumerate(categories['true_positives'][:max_examples]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, tp - 1))
        ret, prev_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, tp)
        ret, curr_frame = cap.read()
        if ret:
            comp = create_comparison_frame(prev_frame, curr_frame, "TP", tp, show_text=True)
            examples["TP"].append(comp)
    
    # False positives
    for i, fp in enumerate(categories['false_positives'][:max_examples]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fp - 1))
        ret, prev_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, fp)
        ret, curr_frame = cap.read()
        if ret:
            comp = create_comparison_frame(prev_frame, curr_frame, "FP", fp, show_text=True)
            examples["FP"].append(comp)
    
    # False negatives
    for i, fn in enumerate(categories['false_negatives'][:max_examples]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fn - 1))
        ret, prev_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, curr_frame = cap.read()
        if ret:
            comp = create_comparison_frame(prev_frame, curr_frame, "FN", fn, show_text=True)
            examples["FN"].append(comp)
    
    cap.release()
    
    # Stack all examples vertically
    all_examples = []
    
    # Add header
    if examples["TP"]:
        header = np.zeros((60, examples["TP"][0].shape[1], 3), dtype=np.uint8)
        cv2.putText(header, "TRUE POSITIVES (Correct Detections)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        all_examples.append(header)
        all_examples.extend(examples["TP"])
    
    if examples["FP"]:
        header = np.zeros((60, examples["FP"][0].shape[1], 3), dtype=np.uint8)
        cv2.putText(header, "FALSE POSITIVES (Wrong Detections)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
        all_examples.append(header)
        all_examples.extend(examples["FP"])
    
    if examples["FN"]:
        header = np.zeros((60, examples["FN"][0].shape[1], 3), dtype=np.uint8)
        cv2.putText(header, "FALSE NEGATIVES (Missed Boundaries)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        all_examples.append(header)
        all_examples.extend(examples["FN"])
    
    if all_examples:
        summary = np.vstack(all_examples)
        
        # Add overall statistics at the top
        stats_height = 150
        stats_panel = np.zeros((stats_height, summary.shape[1], 3), dtype=np.uint8)
        
        y_pos = 35
        cv2.putText(stats_panel, "DETECTION RESULTS SUMMARY", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
        y_pos += 40
        cv2.putText(stats_panel, f"Total Predicted: {len(predicted)}  |  Ground Truth: {len(ground_truth)}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 35
        cv2.putText(stats_panel, f"True Positives: {len(categories['true_positives'])}  |  "
                   f"False Positives: {len(categories['false_positives'])}  |  "
                   f"False Negatives: {len(categories['false_negatives'])}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Calculate metrics
        precision = len(categories['true_positives']) / len(predicted) if predicted else 0
        recall = len(categories['true_positives']) / len(ground_truth) if ground_truth else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        y_pos += 35
        cv2.putText(stats_panel, f"Precision: {precision:.3f}  |  Recall: {recall:.3f}  |  F1-Score: {f1:.3f}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        final_summary = np.vstack([stats_panel, summary])
        
        cv2.imwrite(output_path, final_summary)
        print(f"  ✓ Summary image saved: {output_path}")
        print(f"    Size: {final_summary.shape[1]}x{final_summary.shape[0]} pixels")
    else:
        print("  ✗ No examples to show")


def visualize_detections(video_path: str, ground_truth_file: str, video_name: str,
                         output_dir: str = "visualization_results"):
    """Main visualization function."""
    
    print("="*70)
    print("VIDEO SHOT DETECTION VISUALIZATION")
    print("="*70)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize tester and run detection
    tester = VideoShotDetectionTester(output_dir)
    
    # Load ground truth
    gt_boundaries = tester.load_ground_truth(ground_truth_file, video_name)
    if not gt_boundaries:
        gt_boundaries = tester.load_ground_truth(ground_truth_file, f"{video_name}.mp4")
    
    if not gt_boundaries:
        print(f"ERROR: Could not load ground truth for {video_name}")
        return
    
    print(f"\nVideo: {video_path}")
    print(f"Ground Truth: {len(gt_boundaries)} boundaries")
    
    # Run detection with best balanced config
    print(f"\nRunning detection with balanced configuration...")
    print(f"  Using: Savgol smoothing, weighted fusion, beta=2.0")
    
    cached_frames, cached_features, video_info = tester._extract_features_cached(
        video_path, BALANCED_CONFIG_V2
    )
    
    boundaries, metadata = tester._run_detection_with_cached_features(
        video_path, BALANCED_CONFIG_V2, cached_frames, cached_features
    )
    
    print(f"  Detected: {len(boundaries)} boundaries")
    
    # Calculate accuracy
    accuracy = tester.calculate_accuracy_metrics(boundaries, gt_boundaries, tolerance=10)
    
    print(f"\nAccuracy Metrics:")
    print(f"  Precision: {accuracy['precision']:.3f}")
    print(f"  Recall: {accuracy['recall']:.3f}")
    print(f"  F1-Score: {accuracy['f1_score']:.3f}")
    print(f"  True Positives: {accuracy['true_positives']}")
    print(f"  False Positives: {accuracy['false_positives']}")
    print(f"  False Negatives: {accuracy['false_negatives']}")
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Summary image with examples
    summary_path = f"{output_dir}/detection_summary_{video_name}.jpg"
    create_summary_image(video_path, boundaries, gt_boundaries, summary_path, tolerance=10)
    
    # 2. Full video visualization
    video_output_path = f"{output_dir}/detection_video_{video_name}.mp4"
    create_visualization_video(video_path, boundaries, gt_boundaries, 
                              video_output_path, tolerance=10)
    
    # 3. Save detailed report
    report_path = f"{output_dir}/detection_report_{video_name}.txt"
    with open(report_path, 'w') as f:
        f.write("VIDEO SHOT DETECTION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Ground Truth Boundaries: {len(gt_boundaries)}\n")
        f.write(f"Predicted Boundaries: {len(boundaries)}\n\n")
        
        f.write("ACCURACY METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Precision: {accuracy['precision']:.3f}\n")
        f.write(f"Recall: {accuracy['recall']:.3f}\n")
        f.write(f"F1-Score: {accuracy['f1_score']:.3f}\n")
        f.write(f"True Positives: {accuracy['true_positives']}\n")
        f.write(f"False Positives: {accuracy['false_positives']}\n")
        f.write(f"False Negatives: {accuracy['false_negatives']}\n\n")
        
        # Categorize
        categories = categorize_detections(boundaries, gt_boundaries, tolerance=10)
        
        f.write("TRUE POSITIVE FRAMES (Correct Detections)\n")
        f.write("-"*70 + "\n")
        for tp in categories['true_positives']:
            # Find matching GT
            matching_gt = [gt for gt in gt_boundaries if abs(gt - tp) <= 10]
            if matching_gt:
                f.write(f"Frame {tp} (GT: {matching_gt[0]}, offset: {tp - matching_gt[0]})\n")
        
        f.write("\nFALSE POSITIVE FRAMES (Wrong Detections)\n")
        f.write("-"*70 + "\n")
        for fp in categories['false_positives']:
            f.write(f"Frame {fp}\n")
        
        f.write("\nFALSE NEGATIVE FRAMES (Missed Boundaries)\n")
        f.write("-"*70 + "\n")
        for fn in categories['false_negatives']:
            f.write(f"Frame {fn}\n")
    
    print(f"  ✓ Report saved: {report_path}")
    
    print("\n" + "="*70)
    print("✓ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {summary_path}")
    print(f"  2. {video_output_path}")
    print(f"  3. {report_path}")
    print(f"\nOpen these files to review the detection results!")


if __name__ == "__main__":
    video_path = r"C:\Users\Abdalrahman Khanfar\Documents\ClipShots\videos\train\3505586409.mp4"
    ground_truth = r"C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json"
    video_name = "3505586409"
    
    visualize_detections(video_path, ground_truth, video_name)
