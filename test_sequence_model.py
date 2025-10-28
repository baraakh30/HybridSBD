"""
Test sequence-level transformer model on videos and visualize boundary predictions.

This script:
1. Loads a trained sequence-level model
2. Runs inference on test videos
3. Localizes boundaries within positive sequences
4. Visualizes predicted vs ground truth boundaries
"""

import torch
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Add parent directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from core.config import VideoShotConfig
from detection.transformer_detection import TransformerDetection


class SequenceModelTester:
    """Test and visualize sequence-level boundary detection model."""
    
    def __init__(self, model_path: str, config: VideoShotConfig):
        """
        Initialize tester.
        
        Args:
            model_path: Path to trained model checkpoint
            config: System configuration
        """
        self.model_path = model_path
        self.config = config
        
        # Initialize model
        self.detection_method = TransformerDetection(config.detection.__dict__)
        self.detection_method.initialize()
        self.detection_method.load_model(model_path)
        self.model = self.detection_method.model
        self.device = self.detection_method.device
        
        print(f"Loaded model from: {model_path}")
        print(f"Using device: {self.device}")
    
    def predict_video(self, features: np.ndarray, 
                     sequence_length: int = 100,
                     overlap: int = 50,
                     sequence_threshold: float = 0.5,
                     boundary_selection: str = 'middle') -> Dict[str, any]:
        """
        Predict boundaries for a video using SEQUENCE-LEVEL classification only.
        
        Args:
            features: Feature array of shape (num_frames, feature_dim)
            sequence_length: Length of input sequences
            overlap: Overlap between sequences
            sequence_threshold: Threshold for sequence-level prediction
            boundary_selection: How to select boundary frame from positive sequence:
                - 'middle': Use middle frame of sequence
                - 'max_diff': Frame with maximum feature difference
                - 'first': First frame of sequence
                - 'weighted_middle': Weighted average of overlapping sequence predictions
            
        Returns:
            Dictionary with predictions:
            - 'sequence_predictions': List of (start_frame, end_frame, has_boundary_prob)
            - 'boundary_frames': List of predicted boundary frame indices
            - 'boundary_confidences': Confidence for each boundary
        """
        num_frames = len(features)
        sequence_predictions = []
        positive_sequences = []
        
        self.model.eval()
        with torch.no_grad():
            # Process video in overlapping sequences
            for start_idx in range(0, num_frames - sequence_length + 1, overlap):
                end_idx = start_idx + sequence_length
                sequence = features[start_idx:end_idx]
                
                # Convert to tensor
                sequence_tensor = torch.from_numpy(sequence).float()
                sequence_tensor = sequence_tensor.unsqueeze(0).to(self.device)
                
                # SEQUENCE-LEVEL PREDICTION (does this sequence have a boundary?)
                seq_logits = self.model(sequence_tensor, mode='sequence')
                seq_prob = torch.sigmoid(seq_logits).item()
                
                sequence_predictions.append({
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'has_boundary_prob': seq_prob,
                    'has_boundary': seq_prob > sequence_threshold
                })
                
                # Collect positive sequences
                if seq_prob > sequence_threshold:
                    positive_sequences.append({
                        'start_frame': start_idx,
                        'end_frame': end_idx,
                        'confidence': seq_prob,
                        'features': sequence
                    })
        
        # Select boundary frames from positive sequences
        boundary_frames = []
        boundary_confidences = []
        
        if boundary_selection == 'middle':
            # Use middle frame of each positive sequence
            for seq in positive_sequences:
                boundary_frame = (seq['start_frame'] + seq['end_frame']) // 2
                boundary_frames.append(boundary_frame)
                boundary_confidences.append(seq['confidence'])
        
        elif boundary_selection == 'max_diff':
            # Use frame with maximum feature difference in sequence
            for seq in positive_sequences:
                seq_features = seq['features']
                # Compute feature differences between consecutive frames
                diffs = np.linalg.norm(np.diff(seq_features, axis=0), axis=1)
                max_diff_idx = np.argmax(diffs)
                boundary_frame = seq['start_frame'] + max_diff_idx
                boundary_frames.append(boundary_frame)
                boundary_confidences.append(seq['confidence'])
        
        elif boundary_selection == 'first':
            # Use first frame of each positive sequence
            for seq in positive_sequences:
                boundary_frames.append(seq['start_frame'])
                boundary_confidences.append(seq['confidence'])
        
        elif boundary_selection == 'weighted_middle':
            # Group overlapping sequences and use weighted average
            if positive_sequences:
                # Sort by start frame
                positive_sequences.sort(key=lambda x: x['start_frame'])
                
                # Group overlapping sequences
                groups = []
                current_group = [positive_sequences[0]]
                
                for seq in positive_sequences[1:]:
                    # Check if overlaps with current group
                    if seq['start_frame'] < current_group[-1]['end_frame']:
                        current_group.append(seq)
                    else:
                        groups.append(current_group)
                        current_group = [seq]
                
                groups.append(current_group)
                
                # For each group, compute weighted center
                for group in groups:
                    total_weight = sum(s['confidence'] for s in group)
                    weighted_center = sum(
                        ((s['start_frame'] + s['end_frame']) / 2) * s['confidence']
                        for s in group
                    ) / total_weight
                    
                    boundary_frames.append(int(weighted_center))
                    boundary_confidences.append(max(s['confidence'] for s in group))
        
        # Apply non-maximum suppression to remove nearby duplicates
        if boundary_frames:
            boundary_frames, boundary_confidences = self._non_max_suppression(
                boundary_frames, boundary_confidences, min_distance=sequence_length // 2
            )
        
        return {
            'sequence_predictions': sequence_predictions,
            'boundary_frames': boundary_frames,
            'boundary_confidences': boundary_confidences,
            'positive_sequences': positive_sequences
        }
    
    def _non_max_suppression(self, frames: List[int], confidences: List[float],
                            min_distance: int = 15) -> Tuple[List[int], List[float]]:
        """
        Apply non-maximum suppression to remove nearby duplicate boundaries.
        
        Args:
            frames: List of boundary frame indices
            confidences: Confidence scores for each frame
            min_distance: Minimum distance between boundaries
            
        Returns:
            Filtered frames and confidences
        """
        if not frames:
            return [], []
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_frames = [frames[i] for i in sorted_indices]
        sorted_confidences = [confidences[i] for i in sorted_indices]
        
        kept_frames = []
        kept_confidences = []
        
        for frame, conf in zip(sorted_frames, sorted_confidences):
            # Check if this frame is far enough from all kept frames
            if all(abs(frame - kept_frame) >= min_distance for kept_frame in kept_frames):
                kept_frames.append(frame)
                kept_confidences.append(conf)
        
        # Sort by frame index
        sorted_pairs = sorted(zip(kept_frames, kept_confidences))
        kept_frames = [f for f, _ in sorted_pairs]
        kept_confidences = [c for _, c in sorted_pairs]
        
        return kept_frames, kept_confidences
    
    def evaluate_video(self, video_name: str, features: np.ndarray,
                      ground_truth_boundaries: List[int],
                      tolerance: int = 5) -> Dict[str, any]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            video_name: Name of the video
            features: Feature array
            ground_truth_boundaries: List of ground truth boundary frames
            tolerance: Tolerance window for matching (frames)
            
        Returns:
            Evaluation metrics
        """
        # Get predictions
        predictions = self.predict_video(features)
        predicted_boundaries = predictions['boundary_frames']
        
        # Match predictions to ground truth
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        matched_gt = set()
        matched_pred = set()
        
        # For each prediction, find nearest ground truth
        for pred_idx, pred_frame in enumerate(predicted_boundaries):
            matched = False
            for gt_idx, gt_frame in enumerate(ground_truth_boundaries):
                if abs(pred_frame - gt_frame) <= tolerance and gt_idx not in matched_gt:
                    true_positives += 1
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
                    matched = True
                    break
            
            if not matched:
                false_positives += 1
        
        false_negatives = len(ground_truth_boundaries) - len(matched_gt)
        
        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'video_name': video_name,
            'num_predicted': len(predicted_boundaries),
            'num_ground_truth': len(ground_truth_boundaries),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'predictions': predictions,
            'matched_predictions': matched_pred,
            'matched_ground_truth': matched_gt
        }
    
    def visualize_predictions(self, video_name: str, features: np.ndarray,
                            ground_truth_boundaries: List[int],
                            output_path: Optional[str] = None,
                            show_sequences: bool = True,
                            boundary_selection: str = 'weighted_middle'):
        """
        Visualize predicted vs ground truth boundaries (SEQUENCE-LEVEL only).
        
        Args:
            video_name: Name of the video
            features: Feature array
            ground_truth_boundaries: List of ground truth boundaries
            output_path: Path to save visualization (if None, displays plot)
            show_sequences: Whether to show sequence-level predictions
            boundary_selection: Method for selecting boundary frame from sequence
        """
        # Get predictions (sequence-level only)
        predictions = self.predict_video(features, boundary_selection=boundary_selection)
        predicted_boundaries = predictions['boundary_frames']
        boundary_confidences = predictions['boundary_confidences']
        sequence_preds = predictions['sequence_predictions']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        
        num_frames = len(features)
        
        # Plot sequence-level predictions as background
        for seq in sequence_preds:
            start = seq['start_frame']
            end = seq['end_frame']
            prob = seq['has_boundary_prob']
            has_boundary = seq['has_boundary']
            
            color = 'lightcoral' if has_boundary else 'lightblue'
            alpha = 0.2 + 0.3 * prob  # Opacity based on confidence
            
            ax.axvspan(start, end, alpha=alpha, color=color, zorder=0)
        
        # Plot ground truth boundaries
        for gt_frame in ground_truth_boundaries:
            ax.axvline(x=gt_frame, color='green', linestyle='-', linewidth=3, alpha=0.8, 
                      label='Ground Truth' if gt_frame == ground_truth_boundaries[0] else '', zorder=2)
        
        # Plot predicted boundaries (from sequence selection)
        for i, (pred_frame, conf) in enumerate(zip(predicted_boundaries, boundary_confidences)):
            ax.axvline(x=pred_frame, color='red', linestyle='--', linewidth=3, alpha=0.8,
                      label='Predicted' if i == 0 else '', zorder=3)
            ax.text(pred_frame, 0.95, f'{conf:.2f}', rotation=90,
                   va='bottom', ha='right', fontsize=10, color='red', fontweight='bold')
        
        ax.set_xlim(0, num_frames)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Frame', fontsize=14, fontweight='bold')
        ax.set_ylabel('Sequence Has Boundary', fontsize=14, fontweight='bold')
        ax.set_title(f'Sequence-Level Boundary Detection: {video_name}\n' + 
                    f'Selection Method: {boundary_selection}', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add text annotation
        info_text = (f'Predicted: {len(predicted_boundaries)} | '
                    f'Ground Truth: {len(ground_truth_boundaries)} | '
                    f'Method: {boundary_selection}')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def test_dataset(self, features_dir: str, ground_truth_file: str,
                    output_dir: str = "test_results",
                    num_videos: Optional[int] = None,
                    visualize: bool = True,
                    boundary_selection: str = 'weighted_middle'):
        """
        Test model on multiple videos from dataset (SEQUENCE-LEVEL only).
        
        Args:
            features_dir: Directory containing feature files
            ground_truth_file: JSON file with ground truth
            output_dir: Directory to save results
            num_videos: Number of videos to test (None = all)
            visualize: Whether to create visualizations
            boundary_selection: Method for selecting boundary frame from sequence
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load ground truth
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Select videos to test
        video_names = list(ground_truth.keys())
        if num_videos:
            video_names = video_names[:num_videos]
        
        print(f"Testing on {len(video_names)} videos...")
        
        all_results = []
        aggregate_metrics = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0
        }
        
        for video_name in tqdm(video_names, desc="Testing videos"):
            # Load features
            video_stem = video_name.replace('.mp4', '')
            feature_file = os.path.join(features_dir, f"{video_stem}_hierarchical.npy")
            
            if not os.path.exists(feature_file):
                print(f"Warning: Feature file not found for {video_name}")
                continue
            
            features = np.load(feature_file)
            
            # Get ground truth boundaries
            video_data = ground_truth[video_name]
            if isinstance(video_data, dict) and 'transitions' in video_data:
                gt_boundaries = [t[0] for t in video_data['transitions']]
            else:
                gt_boundaries = video_data
            
            # Evaluate
            result = self.evaluate_video(video_stem, features, gt_boundaries)
            all_results.append(result)
            
            # Aggregate metrics
            aggregate_metrics['total_tp'] += result['true_positives']
            aggregate_metrics['total_fp'] += result['false_positives']
            aggregate_metrics['total_fn'] += result['false_negatives']
            
            # Visualize (if enabled)
            if visualize:
                viz_path = os.path.join(output_dir, f"{video_stem}_predictions.png")
                self.visualize_predictions(video_stem, features, gt_boundaries, viz_path, 
                                          boundary_selection=boundary_selection)
        
        # Compute aggregate metrics
        total_tp = aggregate_metrics['total_tp']
        total_fp = aggregate_metrics['total_fp']
        total_fn = aggregate_metrics['total_fn']
        
        aggregate_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        aggregate_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        aggregate_f1 = 2 * aggregate_precision * aggregate_recall / (aggregate_precision + aggregate_recall) if (aggregate_precision + aggregate_recall) > 0 else 0
        
        # Save results (exclude numpy arrays and non-serializable data)
        serializable_results = []
        for result in all_results:
            serializable_result = {
                'video_name': result['video_name'],
                'num_predicted': result['num_predicted'],
                'num_ground_truth': result['num_ground_truth'],
                'true_positives': result['true_positives'],
                'false_positives': result['false_positives'],
                'false_negatives': result['false_negatives'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'boundary_frames': result['predictions']['boundary_frames'],
                'boundary_confidences': [float(c) for c in result['predictions']['boundary_confidences']]
            }
            serializable_results.append(serializable_result)
        
        results_summary = {
            'model_path': self.model_path,
            'num_videos_tested': len(all_results),
            'aggregate_metrics': {
                'precision': float(aggregate_precision),
                'recall': float(aggregate_recall),
                'f1_score': float(aggregate_f1),
                'total_true_positives': int(total_tp),
                'total_false_positives': int(total_fp),
                'total_false_negatives': int(total_fn)
            },
            'per_video_results': serializable_results
        }
        
        results_file = os.path.join(output_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Videos tested: {len(all_results)}")
        print(f"Precision: {aggregate_precision:.4f}")
        print(f"Recall: {aggregate_recall:.4f}")
        print(f"F1-Score: {aggregate_f1:.4f}")
        print(f"\nResults saved to: {results_file}")
        print(f"{'='*60}")
        
        return results_summary


def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test sequence-level transformer model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--features_dir", required=True, help="Directory containing features")
    parser.add_argument("--ground_truth", required=True, help="Ground truth JSON file")
    parser.add_argument("--output_dir", default="test_results", help="Output directory")
    parser.add_argument("--num_videos", type=int, default=None, help="Number of videos to test")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--sequence_threshold", type=float, default=0.5,
                       help="Threshold for sequence-level prediction")
    parser.add_argument("--boundary_selection", type=str, default='weighted_middle',
                       choices=['middle', 'max_diff', 'first', 'weighted_middle'],
                       help="Method for selecting boundary frame from positive sequence")
    
    args = parser.parse_args()
    
    # Create config
    config = VideoShotConfig()
    config.fusion.strategy = "hierarchical"
    config.detection.method = "transformer"
    
    # Initialize tester
    tester = SequenceModelTester(args.model_path, config)
    
    # Run tests
    tester.test_dataset(
        features_dir=args.features_dir,
        ground_truth_file=args.ground_truth,
        output_dir=args.output_dir,
        num_videos=args.num_videos,
        visualize=args.visualize,
        boundary_selection=args.boundary_selection
    )


if __name__ == "__main__":
    main()
