"""
Debug tool to visualize layer activations and intermediate outputs for specific transitions.
Shows what each layer sees and where the model fails.
"""

import torch
import numpy as np
import json
import os
from detection.turbo_sbd import TurboSBD
from decord import VideoReader, cpu

def load_video_segment(video_path, start_frame, num_frames=100):
    """Load a segment of video frames"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Ensure valid range
    start = max(0, start_frame)
    end = min(total_frames, start + num_frames)
    
    # Load frames
    frames = vr.get_batch(range(start, end)).asnumpy()  # [T, H, W, 3]
    
    # Resize to 48x64 and normalize
    import cv2
    resized = []
    for frame in frames:
        frame_resized = cv2.resize(frame, (64, 48))  # W, H
        resized.append(frame_resized)
    
    frames_tensor = torch.from_numpy(np.array(resized)).float() / 255.0  # [T, H, W, 3]
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [T, 3, H, W]
    
    return frames_tensor

def debug_transition(model, video_path, transition_start, transition_end, device='cuda'):
    """
    Debug a specific transition by showing layer activations and predictions.
    
    Args:
        model: TurboSBD model
        video_path: Path to video file
        transition_start: Start frame of transition
        transition_end: End frame of transition
    """
    model.eval()
    
    # Load 100-frame segment centered on transition
    center = (transition_start + transition_end) // 2
    segment_start = max(0, center - 50)
    
    frames = load_video_segment(video_path, segment_start, 100)
    frames = frames.unsqueeze(0).to(device)  # [1, T, 3, H, W]
    
    print(f"\n{'='*80}")
    print(f"DEBUG: Transition at frames {transition_start}-{transition_end} (length: {transition_end - transition_start})")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Loaded segment: frames {segment_start} to {segment_start + frames.shape[1]}")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        # Forward pass with intermediate outputs
        B, T, C, H, W = frames.shape
        
        # 1. Compute differences
        x_diff1 = frames - torch.roll(frames, shifts=1, dims=1)
        x_diff1[:, 0] = 0
        x_diff4 = frames - torch.roll(frames, shifts=4, dims=1)
        x_diff4[:, :4] = 0
        x_diff16 = frames - torch.roll(frames, shifts=16, dims=1)
        x_diff16[:, :16] = 0
        
        # Compute diff magnitudes
        diff1_mag = torch.mean(torch.abs(x_diff1), dim=(2, 3, 4))  # [B, T]
        diff4_mag = torch.mean(torch.abs(x_diff4), dim=(2, 3, 4))
        diff16_mag = torch.mean(torch.abs(x_diff16), dim=(2, 3, 4))
        
        # Compute LUMINANCE signals (Y = 0.299*R + 0.587*G + 0.114*B)
        luma = 0.299 * frames[:, :, 0] + 0.587 * frames[:, :, 1] + 0.114 * frames[:, :, 2]  # [B, T, H, W]
        
        luma_diff1 = luma - torch.roll(luma, shifts=1, dims=1)
        luma_diff1[:, 0] = 0
        luma_diff4 = luma - torch.roll(luma, shifts=4, dims=1)
        luma_diff4[:, :4] = 0
        luma_diff16 = luma - torch.roll(luma, shifts=16, dims=1)
        luma_diff16[:, :16] = 0
        
        luma_diff1_mag = torch.mean(torch.abs(luma_diff1), dim=(2, 3))  # [B, T]
        luma_diff4_mag = torch.mean(torch.abs(luma_diff4), dim=(2, 3))
        luma_diff16_mag = torch.mean(torch.abs(luma_diff16), dim=(2, 3))
        
        eps = 1e-3
        ratio_4_1 = diff4_mag / torch.clamp(diff1_mag, min=eps)
        ratio_16_4 = diff16_mag / torch.clamp(diff4_mag, min=eps)
        ratio_4_1 = torch.clamp(ratio_4_1, 0, 10)
        ratio_16_4 = torch.clamp(ratio_16_4, 0, 10)
        
        # Get model prediction
        logits = model(frames)
        probs = torch.softmax(logits, dim=2)  # [B, T, 3]
        preds = torch.argmax(logits, dim=2)  # [B, T]
        
        # Convert to numpy for analysis
        diff1_mag = diff1_mag.cpu().numpy()[0]  # [T]
        diff4_mag = diff4_mag.cpu().numpy()[0]
        diff16_mag = diff16_mag.cpu().numpy()[0]
        luma_diff1_mag = luma_diff1_mag.cpu().numpy()[0]
        luma_diff4_mag = luma_diff4_mag.cpu().numpy()[0]
        luma_diff16_mag = luma_diff16_mag.cpu().numpy()[0]
        ratio_4_1 = ratio_4_1.cpu().numpy()[0]
        ratio_16_4 = ratio_16_4.cpu().numpy()[0]
        probs = probs.cpu().numpy()[0]  # [T, 3]
        preds = preds.cpu().numpy()[0]  # [T]
        
        # Analyze around transition
        trans_rel_start = transition_start - segment_start
        trans_rel_end = transition_end - segment_start
        
        print("RGB DIFF SIGNAL ANALYSIS:")
        print(f"{'Frame':<8} {'diff1':<10} {'diff4':<10} {'diff16':<10} {'ratio_4/1':<10} {'ratio_16/4':<10} {'Pred':<10} {'P(None)':<10} {'P(Cut)':<10} {'P(Grad)':<10}")
        print("-" * 120)
        
        # Show frames around transition
        for t in range(max(0, trans_rel_start - 5), min(T, trans_rel_end + 6)):
            frame_num = segment_start + t
            pred_name = ['None', 'Cut', 'Grad'][preds[t]]
            in_trans = trans_rel_start <= t <= trans_rel_end
            marker = " <<<" if in_trans else ""
            
            print(f"{frame_num:<8} {diff1_mag[t]:<10.4f} {diff4_mag[t]:<10.4f} {diff16_mag[t]:<10.4f} "
                  f"{ratio_4_1[t]:<10.4f} {ratio_16_4[t]:<10.4f} {pred_name:<10} "
                  f"{probs[t, 0]:<10.4f} {probs[t, 1]:<10.4f} {probs[t, 2]:<10.4f}{marker}")
        
        print("\nLUMINANCE SIGNAL ANALYSIS:")
        print(f"{'Frame':<8} {'luma_d1':<10} {'luma_d4':<10} {'luma_d16':<10} {'Pred':<10} {'P(None)':<10} {'P(Cut)':<10} {'P(Grad)':<10}")
        print("-" * 100)
        
        for t in range(max(0, trans_rel_start - 5), min(T, trans_rel_end + 6)):
            frame_num = segment_start + t
            pred_name = ['None', 'Cut', 'Grad'][preds[t]]
            in_trans = trans_rel_start <= t <= trans_rel_end
            marker = " <<<" if in_trans else ""
            
            print(f"{frame_num:<8} {luma_diff1_mag[t]:<10.4f} {luma_diff4_mag[t]:<10.4f} {luma_diff16_mag[t]:<10.4f} "
                  f"{pred_name:<10} {probs[t, 0]:<10.4f} {probs[t, 1]:<10.4f} {probs[t, 2]:<10.4f}{marker}")
        
        print("\n" + "="*80)
        print("ANALYSIS:")
        
        # Check if transition was detected
        trans_frames = preds[trans_rel_start:trans_rel_end+1]
        detected = (trans_frames == 2).any()
        
        if not detected:
            print("❌ MISSED: No transition detected in ground truth range")
            
            # Analyze RGB signals
            avg_diff1 = np.mean(diff1_mag[trans_rel_start:trans_rel_end+1])
            avg_diff4 = np.mean(diff4_mag[trans_rel_start:trans_rel_end+1])
            avg_ratio = np.mean(ratio_4_1[trans_rel_start:trans_rel_end+1])
            
            # Analyze LUMINANCE signals
            avg_luma1 = np.mean(luma_diff1_mag[trans_rel_start:trans_rel_end+1])
            avg_luma4 = np.mean(luma_diff4_mag[trans_rel_start:trans_rel_end+1])
            avg_luma16 = np.mean(luma_diff16_mag[trans_rel_start:trans_rel_end+1])
            
            print(f"\nRGB signals during transition:")
            print(f"  Avg diff1_mag: {avg_diff1:.4f}")
            print(f"  Avg diff4_mag: {avg_diff4:.4f}")
            print(f"  Avg ratio_4/1: {avg_ratio:.4f}")
            
            print(f"\nLUMINANCE signals during transition:")
            print(f"  Avg luma_diff1: {avg_luma1:.4f}")
            print(f"  Avg luma_diff4: {avg_luma4:.4f}")
            print(f"  Avg luma_diff16: {avg_luma16:.4f}")
            
            # Determine root cause
            luma_is_strong = avg_luma4 > 0.01  # Threshold for "strong" luminance signal
            rgb_is_weak = avg_diff1 < 0.01
            
            if rgb_is_weak and luma_is_strong:
                print(f"\n  🎯 LUMINANCE WOULD HELP! RGB weak ({avg_diff1:.4f}) but luminance strong ({avg_luma4:.4f})")
                print(f"     This is likely a FADE transition (fade-to-black/from-black)")
                print(f"     Luminance signal is {avg_luma4/max(avg_diff4, 1e-6):.1f}x stronger!")
            elif rgb_is_weak and not luma_is_strong:
                print(f"\n  → Very subtle transition - both RGB and luminance signals weak")
            elif avg_ratio < 1.5:
                print(f"\n  → Low smoothness ratio - looks like noise/camera motion")
            else:
                print(f"\n  → Signals present but heads didn't fire - classification issue")
        else:
            print("✓ DETECTED: Gradual correctly identified")
        
        print("="*80)

def find_missed_graduals(model, annotations_path, videos_dir, device='cuda', num_samples=5):
    """
    Find and debug ACTUALLY MISSED gradual transitions by running predictions first
    """
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"\nFinding MISSED graduals in {annotations_path}...")
    print("Running predictions to identify failures...\n")
    
    missed_examples = []
    
    for vid_name, data in annotations.items():
        video_path = os.path.join(videos_dir, vid_name)
        if not os.path.exists(video_path):
            continue
        
        transitions = data.get('transitions', [])
        
        for start, end in transitions:
            length = end - start
            if length <= 1:  # Skip cuts
                continue
            
            try:
                # Load segment and predict
                center = (start + end) // 2
                segment_start = max(0, center - 50)
                frames = load_video_segment(video_path, segment_start, 100)
                frames_batch = frames.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(frames_batch)
                    preds = torch.argmax(logits, dim=2).cpu().numpy()[0]  # [T]
                
                # Check if transition was detected
                trans_rel_start = start - segment_start
                trans_rel_end = end - segment_start
                
                if 0 <= trans_rel_start < len(preds) and 0 <= trans_rel_end < len(preds):
                    trans_preds = preds[trans_rel_start:trans_rel_end+1]
                    
                    # Check if ANY frame in transition was detected as gradual (class 2)
                    detected = (trans_preds == 2).any()
                    
                    if not detected:
                        # MISSED! Save for debugging
                        missed_examples.append({
                            'video': vid_name,
                            'path': video_path,
                            'start': start,
                            'end': end,
                            'length': length,
                            'predictions': trans_preds.tolist()
                        })
                        
                        if len(missed_examples) >= num_samples * 3:  # Collect more than needed
                            break
                            
            except Exception as e:
                continue
        
        if len(missed_examples) >= num_samples * 3:
            break
    
    print(f"Found {len(missed_examples)} missed gradual transitions")
    print(f"Debugging first {num_samples}...\n")
    
    # Sort by length to see which lengths are most problematic
    missed_examples.sort(key=lambda x: x['length'])
    
    for i, example in enumerate(missed_examples[:num_samples]):
        print(f"\n{'='*80}")
        print(f"MISSED GRADUAL #{i+1}/{num_samples}")
        print(f"Length: {example['length']} frames")
        print(f"Predictions in transition: {example['predictions']}")
        print(f"{'='*80}")
        
        debug_transition(model, example['path'], example['start'], example['end'], device)

if __name__ == "__main__":
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TurboSBD(num_classes=3).to(device)
    
    # Load best checkpoint
    checkpoint_path = "outputs/turbo_sbd_best.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("No checkpoint found, using random weights")
    
    model.eval()
    
    # Debug missed graduals
    annotations_path = r"C:\Users\baraa\Documents\ClipShots\annotations\test.json"
    videos_dir = r"C:\Users\baraa\Documents\ClipShots\videos\test"
    
    find_missed_graduals(model, annotations_path, videos_dir, device, num_samples=5)
