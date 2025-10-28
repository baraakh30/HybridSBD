"""
Interactive visualization tool for boundary detection results.

Shows:
- Video frames at predicted and ground truth boundaries
- Sequence-level predictions with highlighted regions
- Frame-by-frame navigation
- Comparison between prediction and ground truth
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.patches as patches


class InteractiveBoundaryVisualizer:
    """Interactive visualizer for boundary detection results."""
    
    def __init__(self, video_path: str, results_json: str, ground_truth_json: str):
        """
        Initialize visualizer.
        
        Args:
            video_path: Path to video file
            results_json: Path to test results JSON
            ground_truth_json: Path to ground truth JSON
        """
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        
        # Load results
        with open(results_json, 'r') as f:
            results_data = json.load(f)
        
        # Find results for this video
        self.video_results = None
        for result in results_data['per_video_results']:
            if result['video_name'] == self.video_name:
                self.video_results = result
                break
        
        if self.video_results is None:
            raise ValueError(f"No results found for video: {self.video_name}")
        
        # Load ground truth
        with open(ground_truth_json, 'r') as f:
            gt_data = json.load(f)
        
        # Get ground truth boundaries
        video_key = f"{self.video_name}.mp4"
        if video_key in gt_data:
            gt_entry = gt_data[video_key]
            if isinstance(gt_entry, dict) and 'transitions' in gt_entry:
                self.gt_boundaries = [t[0] for t in gt_entry['transitions']]
            else:
                self.gt_boundaries = gt_entry
        else:
            self.gt_boundaries = []
        
        # Get predictions
        self.pred_boundaries = self.video_results['boundary_frames']
        self.pred_confidences = self.video_results['boundary_confidences']
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Current state
        self.current_frame = 0
        self.playing = False
        
        print(f"Loaded video: {self.video_name}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  FPS: {self.fps}")
        print(f"  Predicted boundaries: {len(self.pred_boundaries)}")
        print(f"  Ground truth boundaries: {len(self.gt_boundaries)}")
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get frame at specific index."""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def create_interactive_view(self):
        """Create interactive matplotlib window."""
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main video frame display
        self.ax_main = fig.add_subplot(gs[0:2, 0:2])
        self.ax_main.set_title(f'Current Frame: {self.current_frame}', fontsize=14, fontweight='bold')
        self.ax_main.axis('off')
        
        # Timeline view
        self.ax_timeline = fig.add_subplot(gs[2, :])
        self.ax_timeline.set_xlabel('Frame', fontsize=12)
        self.ax_timeline.set_ylabel('Boundary', fontsize=12)
        self.ax_timeline.set_title('Video Timeline (Red=Predicted, Green=Ground Truth)', fontsize=12)
        
        # Predicted boundary thumbnails
        self.ax_pred = fig.add_subplot(gs[0, 2])
        self.ax_pred.set_title('Nearest Predicted\nBoundary', fontsize=10, fontweight='bold')
        self.ax_pred.axis('off')
        
        # Ground truth boundary thumbnails
        self.ax_gt = fig.add_subplot(gs[1, 2])
        self.ax_gt.set_title('Nearest Ground Truth\nBoundary', fontsize=10, fontweight='bold')
        self.ax_gt.axis('off')
        
        # Initialize displays
        self.update_display()
        
        # Add slider for frame navigation
        ax_slider = plt.axes([0.15, 0.02, 0.7, 0.02])
        self.slider = Slider(ax_slider, 'Frame', 0, self.total_frames - 1, 
                            valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)
        
        # Add buttons
        ax_prev = plt.axes([0.15, 0.06, 0.08, 0.03])
        ax_next = plt.axes([0.24, 0.06, 0.08, 0.03])
        ax_prev_bound = plt.axes([0.33, 0.06, 0.12, 0.03])
        ax_next_bound = plt.axes([0.46, 0.06, 0.12, 0.03])
        ax_prev_gt = plt.axes([0.59, 0.06, 0.12, 0.03])
        ax_next_gt = plt.axes([0.72, 0.06, 0.12, 0.03])
        
        self.btn_prev = Button(ax_prev, '◄ Prev')
        self.btn_next = Button(ax_next, 'Next ►')
        self.btn_prev_bound = Button(ax_prev_bound, '◄ Prev Pred')
        self.btn_next_bound = Button(ax_next_bound, 'Next Pred ►')
        self.btn_prev_gt = Button(ax_prev_gt, '◄ Prev GT')
        self.btn_next_gt = Button(ax_next_gt, 'Next GT ►')
        
        self.btn_prev.on_clicked(lambda event: self.navigate(-1))
        self.btn_next.on_clicked(lambda event: self.navigate(1))
        self.btn_prev_bound.on_clicked(lambda event: self.jump_to_boundary('pred', -1))
        self.btn_next_bound.on_clicked(lambda event: self.jump_to_boundary('pred', 1))
        self.btn_prev_gt.on_clicked(lambda event: self.jump_to_boundary('gt', -1))
        self.btn_next_gt.on_clicked(lambda event: self.jump_to_boundary('gt', 1))
        
        # Keyboard shortcuts
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.show()
    
    def update_display(self):
        """Update all display elements."""
        # Get current frame
        frame = self.get_frame(self.current_frame)
        
        if frame is not None:
            # Display main frame
            self.ax_main.clear()
            self.ax_main.imshow(frame)
            self.ax_main.axis('off')
            
            # Add frame info
            frame_time = self.current_frame / self.fps if self.fps > 0 else 0
            title = f'Frame: {self.current_frame}/{self.total_frames} | Time: {frame_time:.2f}s'
            
            # Check if near boundary
            near_pred = self.get_nearest_boundary('pred')
            near_gt = self.get_nearest_boundary('gt')
            
            if near_pred is not None and abs(self.current_frame - near_pred) <= 5:
                pred_idx = self.pred_boundaries.index(near_pred)
                conf = self.pred_confidences[pred_idx]
                title += f'\n🔴 PREDICTED BOUNDARY (conf: {conf:.2f})'
            
            if near_gt is not None and abs(self.current_frame - near_gt) <= 5:
                title += '\n🟢 GROUND TRUTH BOUNDARY'
            
            self.ax_main.set_title(title, fontsize=12, fontweight='bold')
        
        # Update timeline
        self.ax_timeline.clear()
        self.ax_timeline.set_xlim(0, self.total_frames)
        self.ax_timeline.set_ylim(-0.5, 1.5)
        
        # Plot predicted boundaries
        for pred_frame, conf in zip(self.pred_boundaries, self.pred_confidences):
            self.ax_timeline.axvline(pred_frame, color='red', alpha=0.6, linewidth=2)
            self.ax_timeline.text(pred_frame, 1.2, f'{conf:.2f}', rotation=90, 
                                 fontsize=8, color='red', ha='right')
        
        # Plot ground truth boundaries
        for gt_frame in self.gt_boundaries:
            self.ax_timeline.axvline(gt_frame, color='green', alpha=0.6, linewidth=2)
        
        # Plot current position
        self.ax_timeline.axvline(self.current_frame, color='blue', linewidth=3, alpha=0.8)
        
        # Add viewing window
        window_size = min(500, self.total_frames // 4)
        view_start = max(0, self.current_frame - window_size // 2)
        view_end = min(self.total_frames, self.current_frame + window_size // 2)
        self.ax_timeline.set_xlim(view_start, view_end)
        
        self.ax_timeline.set_xlabel('Frame', fontsize=12)
        self.ax_timeline.set_yticks([])
        self.ax_timeline.grid(True, alpha=0.3)
        
        # Update predicted boundary thumbnail
        self.ax_pred.clear()
        near_pred = self.get_nearest_boundary('pred')
        if near_pred is not None:
            pred_frame = self.get_frame(near_pred)
            if pred_frame is not None:
                self.ax_pred.imshow(pred_frame)
                pred_idx = self.pred_boundaries.index(near_pred)
                conf = self.pred_confidences[pred_idx]
                distance = abs(self.current_frame - near_pred)
                self.ax_pred.set_title(f'Predicted Boundary\nFrame: {near_pred}\n' + 
                                      f'Confidence: {conf:.2f}\nDistance: {distance} frames',
                                      fontsize=9)
        self.ax_pred.axis('off')
        
        # Update ground truth thumbnail
        self.ax_gt.clear()
        near_gt = self.get_nearest_boundary('gt')
        if near_gt is not None:
            gt_frame = self.get_frame(near_gt)
            if gt_frame is not None:
                self.ax_gt.imshow(gt_frame)
                distance = abs(self.current_frame - near_gt)
                self.ax_gt.set_title(f'Ground Truth Boundary\nFrame: {near_gt}\n' + 
                                    f'Distance: {distance} frames',
                                    fontsize=9)
        self.ax_gt.axis('off')
        
        plt.draw()
    
    def get_nearest_boundary(self, boundary_type: str) -> Optional[int]:
        """Get nearest boundary frame of specified type."""
        boundaries = self.pred_boundaries if boundary_type == 'pred' else self.gt_boundaries
        
        if not boundaries:
            return None
        
        distances = [abs(self.current_frame - b) for b in boundaries]
        min_idx = np.argmin(distances)
        return boundaries[min_idx]
    
    def navigate(self, direction: int):
        """Navigate forward or backward by direction frames."""
        self.current_frame = max(0, min(self.total_frames - 1, 
                                       self.current_frame + direction))
        self.slider.set_val(self.current_frame)
        self.update_display()
    
    def jump_to_boundary(self, boundary_type: str, direction: int):
        """Jump to next/previous boundary."""
        boundaries = self.pred_boundaries if boundary_type == 'pred' else self.gt_boundaries
        
        if not boundaries:
            return
        
        if direction > 0:
            # Find next boundary
            next_bounds = [b for b in boundaries if b > self.current_frame]
            if next_bounds:
                self.current_frame = min(next_bounds)
            else:
                self.current_frame = boundaries[0]  # Wrap to first
        else:
            # Find previous boundary
            prev_bounds = [b for b in boundaries if b < self.current_frame]
            if prev_bounds:
                self.current_frame = max(prev_bounds)
            else:
                self.current_frame = boundaries[-1]  # Wrap to last
        
        self.slider.set_val(self.current_frame)
        self.update_display()
    
    def on_slider_change(self, val):
        """Handle slider change."""
        self.current_frame = int(val)
        self.update_display()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'right':
            self.navigate(1)
        elif event.key == 'left':
            self.navigate(-1)
        elif event.key == 'up':
            self.navigate(10)
        elif event.key == 'down':
            self.navigate(-10)
        elif event.key == 'n':
            self.jump_to_boundary('pred', 1)
        elif event.key == 'p':
            self.jump_to_boundary('pred', -1)
        elif event.key == 'g':
            self.jump_to_boundary('gt', 1)
        elif event.key == 'shift+g':
            self.jump_to_boundary('gt', -1)
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive boundary visualization")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--results", required=True, help="Path to test results JSON")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth JSON")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = InteractiveBoundaryVisualizer(
        video_path=args.video,
        results_json=args.results,
        ground_truth_json=args.ground_truth
    )
    
    # Show help
    print("\n" + "="*60)
    print("INTERACTIVE CONTROLS")
    print("="*60)
    print("Arrow Keys:")
    print("  Left/Right  - Navigate 1 frame")
    print("  Up/Down     - Navigate 10 frames")
    print("\nButtons:")
    print("  Prev/Next        - Navigate 1 frame")
    print("  Prev/Next Pred   - Jump to previous/next predicted boundary")
    print("  Prev/Next GT     - Jump to previous/next ground truth boundary")
    print("\nKeyboard Shortcuts:")
    print("  n - Next predicted boundary")
    print("  p - Previous predicted boundary")
    print("  g - Next ground truth boundary")
    print("="*60)
    print()
    
    # Start interactive view
    visualizer.create_interactive_view()


if __name__ == "__main__":
    main()
