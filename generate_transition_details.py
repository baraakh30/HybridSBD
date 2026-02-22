"""
Generate Detailed Transition Visualizations

This script generates detailed visualizations for cuts and graduals:
- For CUTS: Before/after frames (2 photos) + slowed-down video
- For GRADUALS: 3 key frames (mid-start, mid-end, after-complete) + slowed-down video

Usage:
    python generate_transition_details.py --checkpoint outputs/hybrid_detector5/best_model.pt --dataset_idx 3721 --split val
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.hybrid_boundary_detector import create_hybrid_detector
from datasets import PredecodedDataset


class TransitionDetailGenerator:
    """Generate detailed transition visualizations"""
    
    def __init__(self, checkpoint_path, output_dir="transition_details"):
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "cuts").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "graduals").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "videos").mkdir(exist_ok=True, parents=True)
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self.load_model()
        
        # Load dataset
        self.cache_path = r"D:\grad\data\training_cache"
        self.dataset = None
        
    def load_model(self):
        """Load trained model from checkpoint"""
        print(f"\nLoading model from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        config = checkpoint.get('config', {})
        main_dim = config.get('main_dim', 256)
        num_fusion_blocks = config.get('num_fusion_blocks', 3)
        
        model = create_hybrid_detector(
            main_dim=main_dim,
            num_fusion_blocks=num_fusion_blocks
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best F1: {checkpoint.get('best_val_f1', 'unknown'):.4f}")
        
        return model
    
    def load_dataset(self, split='val'):
        """Load the dataset"""
        if self.dataset is None:
            self.dataset = PredecodedDataset(self.cache_path, split=split)
        return self.dataset
    
    def detect_transitions(self, frames_tensor, targets, threshold=0.5):
        """Detect transitions in a clip and classify them"""
        frames_input = frames_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            boundary_logits, type_logits, reasoning_outputs, _ = self.model(frames_input)
        
        boundary_probs = torch.sigmoid(boundary_logits).squeeze().cpu().numpy()
        type_probs = torch.softmax(type_logits, dim=-1).squeeze().cpu().numpy()
        type_preds = np.argmax(type_probs, axis=-1)
        
        targets_np = targets.cpu().numpy()
        
        # Find transitions (boundary prob > threshold)
        is_boundary = boundary_probs > threshold
        
        # Separate cuts and graduals
        cuts = []
        graduals = []
        
        for i in range(len(boundary_probs)):
            if is_boundary[i]:
                pred_type = type_preds[i]
                gt_type = targets_np[i]
                
                transition_info = {
                    'frame_idx': i,
                    'boundary_prob': boundary_probs[i],
                    'predicted_type': pred_type,  # 0=none, 1=cut, 2=gradual
                    'gt_type': gt_type,
                    'type_probs': type_probs[i]
                }
                
                if pred_type == 1 or gt_type == 1:  # Cut
                    cuts.append(transition_info)
                elif pred_type == 2 or gt_type == 2:  # Gradual
                    graduals.append(transition_info)
        
        return cuts, graduals, boundary_probs, type_preds
    
    def tensor_to_image(self, frame_tensor):
        """Convert tensor [C, H, W] to numpy image [H, W, C]"""
        # frame_tensor is [3, 48, 64] in range [0, 1]
        frame = frame_tensor.cpu().numpy()
        frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
        frame = (frame * 255).astype(np.uint8)
        return frame
    
    def save_cut_visualization(self, frames, cut_info, output_prefix):
        """
        Save cut visualization:
        - 2 photos: before and after frames (individual images)
        - 1 slowed-down video clip
        """
        frame_idx = cut_info['frame_idx']
        
        # Get before and after frames (with boundary check)
        before_idx = max(0, frame_idx - 1)
        after_idx = min(len(frames) - 1, frame_idx)
        
        before_frame = self.tensor_to_image(frames[before_idx])
        after_frame = self.tensor_to_image(frames[after_idx])
        
        # Save individual frames with only frame numbers
        # Before frame
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(before_frame)
        ax.text(5, 10, f'Frame {before_idx}', fontsize=12, color='white', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        ax.axis('off')
        plt.tight_layout()
        before_path = f"{output_prefix}_frame_{before_idx}.png"
        plt.savefig(before_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # After frame
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(after_frame)
        ax.text(5, 10, f'Frame {after_idx}', fontsize=12, color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        ax.axis('off')
        plt.tight_layout()
        after_path = f"{output_prefix}_frame_{after_idx}.png"
        plt.savefig(after_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"  ✓ Cut frames saved: {before_path}, {after_path}")
        
        # Create slowed-down video (show 5 frames before and after, with slow motion)
        video_path = f"{output_prefix}_slow.mp4"
        self.create_slow_video(frames, frame_idx, window=5, slow_factor=3, 
                              output_path=video_path, transition_type='CUT')
        
        return [before_path, after_path], video_path
    
    def save_gradual_visualization(self, frames, gradual_info, output_prefix):
        """
        Save gradual visualization:
        - 3 photos: mid-start, mid-end, after-complete (individual images)
        - 1 slowed-down video clip
        """
        frame_idx = gradual_info['frame_idx']
        
        # For graduals, we estimate the transition window
        # Assume transition starts ~10 frames before detection and ends ~10 frames after
        transition_start = max(0, frame_idx - 10)
        transition_end = min(len(frames) - 1, frame_idx + 10)
        transition_mid = frame_idx
        
        # Calculate key frame indices
        mid_start_idx = (transition_start + transition_mid) // 2  # Mid of first half
        mid_end_idx = (transition_mid + transition_end) // 2      # Mid of second half
        after_idx = min(len(frames) - 1, transition_end + 3)       # 3 frames after complete
        
        # Get frames
        mid_start_frame = self.tensor_to_image(frames[mid_start_idx])
        mid_end_frame = self.tensor_to_image(frames[mid_end_idx])
        after_frame = self.tensor_to_image(frames[after_idx])
        
        # Save individual frames with only frame numbers
        # Frame 1: Mid-start
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(mid_start_frame)
        ax.text(5, 10, f'Frame {mid_start_idx}', fontsize=12, color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        ax.axis('off')
        plt.tight_layout()
        start_path = f"{output_prefix}_frame_{mid_start_idx}.png"
        plt.savefig(start_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Frame 2: Mid-end
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(mid_end_frame)
        ax.text(5, 10, f'Frame {mid_end_idx}', fontsize=12, color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        ax.axis('off')
        plt.tight_layout()
        mid_path = f"{output_prefix}_frame_{mid_end_idx}.png"
        plt.savefig(mid_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Frame 3: After complete
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(after_frame)
        ax.text(5, 10, f'Frame {after_idx}', fontsize=12, color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        ax.axis('off')
        plt.tight_layout()
        after_path = f"{output_prefix}_frame_{after_idx}.png"
        plt.savefig(after_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"  ✓ Gradual frames saved: {start_path}, {mid_path}, {after_path}")
        
        # Create slowed-down video (show full transition window with slow motion)
        video_path = f"{output_prefix}_slow.mp4"
        self.create_slow_video(frames, frame_idx, window=15, slow_factor=2,
                              output_path=video_path, transition_type='GRADUAL')
        
        return [start_path, mid_path, after_path], video_path
    
    def create_slow_video(self, frames, center_frame, window=10, slow_factor=3, 
                         output_path='output.mp4', transition_type='TRANSITION'):
        """
        Create slowed-down video clip around a transition
        
        Args:
            frames: tensor [T, C, H, W]
            center_frame: frame index of transition
            window: number of frames before/after to include
            slow_factor: how much to slow down (e.g., 3 = 3x slower)
            output_path: where to save video
            transition_type: 'CUT' or 'GRADUAL'
        """
        start_idx = max(0, center_frame - window)
        end_idx = min(len(frames) - 1, center_frame + window)
        
        # Extract frames
        clip_frames = []
        for i in range(start_idx, end_idx + 1):
            frame = self.tensor_to_image(frames[i])
            # Upscale for better visibility (from 48x64 to 480x640)
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
            
            # Add only frame number (no transition labels)
            frame_display = frame.copy()
            
            # Add frame number in corner
            text = f"Frame {i}"
            cv2.putText(frame_display, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            clip_frames.append(frame_display)
        
        # Write video with slow motion (repeat each frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 8  # Base FPS (slow)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))
        
        for frame in clip_frames:
            # Repeat each frame for slow motion effect
            for _ in range(slow_factor):
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        
        print(f"  ✓ Slow motion video saved: {output_path}")
        return output_path
    
    def process_clip(self, dataset_idx, split='val'):
        """Process a specific clip and generate all visualizations"""
        print(f"\n{'='*80}")
        print(f"PROCESSING CLIP INDEX: {dataset_idx}")
        print(f"{'='*80}")
        
        # Load dataset
        self.load_dataset(split=split)
        
        if dataset_idx >= len(self.dataset) or dataset_idx < 0:
            print(f"❌ Invalid clip index: {dataset_idx} (dataset has {len(self.dataset)} clips)")
            return
        
        # Get video info
        clip_info = self.dataset.mapping[dataset_idx]
        video_name = clip_info['video_name']
        
        print(f"✓ Found clip: {video_name}")
        print(f"  Dataset index: {dataset_idx}")
        print(f"  Clip file: {clip_info['clip_file']}")
        
        # Load clip
        frames, targets, lengths = self.dataset[dataset_idx]
        print(f"  Frames: {frames.shape}")
        print(f"  Targets: {targets.shape}")
        
        # Detect transitions
        print(f"\nDetecting transitions...")
        cuts, graduals, boundary_probs, type_preds = self.detect_transitions(frames, targets)
        
        print(f"  Found {len(cuts)} cuts")
        print(f"  Found {len(graduals)} graduals")
        
        # Process cuts
        print(f"\n{'='*60}")
        print("GENERATING CUT VISUALIZATIONS")
        print(f"{'='*60}")
        
        cut_frame_paths = []
        cut_video_paths = []
        for idx, cut in enumerate(cuts[:5], 1):  # Limit to first 5
            print(f"\nCut {idx}/{min(len(cuts), 5)} (frame {cut['frame_idx']}):")
            output_prefix = self.output_dir / "cuts" / f"{video_name.replace('.mp4', '')}_idx{dataset_idx}_cut{idx}"
            frame_paths, video_path = self.save_cut_visualization(frames, cut, str(output_prefix))
            cut_frame_paths.extend(frame_paths)
            cut_video_paths.append(video_path)
        
        # Process graduals
        print(f"\n{'='*60}")
        print("GENERATING GRADUAL VISUALIZATIONS")
        print(f"{'='*60}")
        
        gradual_frame_paths = []
        gradual_video_paths = []
        for idx, gradual in enumerate(graduals[:5], 1):  # Limit to first 5
            print(f"\nGradual {idx}/{min(len(graduals), 5)} (frame {gradual['frame_idx']}):")
            output_prefix = self.output_dir / "graduals" / f"{video_name.replace('.mp4', '')}_idx{dataset_idx}_gradual{idx}"
            frame_paths, video_path = self.save_gradual_visualization(frames, gradual, str(output_prefix))
            gradual_frame_paths.extend(frame_paths)
            gradual_video_paths.append(video_path)
        
        # Generate summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Video: {video_name}")
        print(f"Dataset Index: {dataset_idx}")
        print(f"Total cuts: {len(cuts)} (visualized: {min(len(cuts), 5)})")
        print(f"  - {len(cut_frame_paths)} frame images")
        print(f"  - {len(cut_video_paths)} slow-motion videos")
        print(f"Total graduals: {len(graduals)} (visualized: {min(len(graduals), 5)})")
        print(f"  - {len(gradual_frame_paths)} frame images")
        print(f"  - {len(gradual_video_paths)} slow-motion videos")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"  Cuts: {self.output_dir / 'cuts'}")
        print(f"  Graduals: {self.output_dir / 'graduals'}")


def main():
    parser = argparse.ArgumentParser(description='Generate detailed transition visualizations')
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/hybrid_detector5/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset_idx', type=int,
                       default=3721,
                       help='Dataset clip index to process (e.g., 3721 for clip at index 3721)')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split')
    parser.add_argument('--output_dir', type=str,
                       default='transition_details',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("TRANSITION DETAIL GENERATOR")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset Index: {args.dataset_idx}")
    print(f"  Split: {args.split}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*80}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"\n❌ Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            for ckpt in outputs_dir.rglob("*.pt"):
                print(f"  - {ckpt}")
        return
    
    # Create generator
    generator = TransitionDetailGenerator(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # Process clip
    generator.process_clip(
        dataset_idx=args.dataset_idx,
        split=args.split
    )


if __name__ == '__main__':
    main()
