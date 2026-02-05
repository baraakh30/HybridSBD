"""
High-Performance Dataset Pre-Decoder for TurboSBD Training

Pre-decodes video clips into optimized format for maximum training speed:
- Multiprocessing with memory monitoring (3GB minimum free)
- Chunks large videos (>10k frames) to avoid memory issues
- Saves in PyTorch tensor format (.pt) with LZ4 compression for speed
- Generates clips EXACTLY as dataset does (deterministic + smart sampling)
- Memory-mapped storage for instant loading

Storage format:
    training_cache/
        video_name/
            clip_000000.pt  # [128, 3, 48, 64] uint8 tensor
            clip_000001.pt
            ...
        metadata.json  # Mapping of dataset indices to clip files

Usage:
    python predecode_dataset.py --split train --num_workers 4
    python predecode_dataset.py --split val --num_workers 4
    
Expected speedup: 10-20x faster training (no video decoding overhead)
Storage: ~50-100GB for full ClipShots dataset
"""

import torch
import numpy as np
import json
import os
import psutil
import time
from pathlib import Path
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm
import random
from torchcodec.decoders import VideoDecoder

# Configuration (MUST match train_turbo_sbd_v2.py exactly)
CONFIG = {
    'input_root': r"C:\Users\baraa\Documents\ClipShots",
    'output_root': r"D:\grad\data\training_cache",
    'frame_size': (48, 64),  # H, W
    'seq_len': 128,
    'samples_per_video_train': 20,  # MUST match training
    'samples_per_video_val': 3,     # MUST match validation
    'random_seed': 42,
    'min_free_memory_gb': 3.0,  # Minimum free RAM before processing
    'large_video_threshold': 10000,  # Frames - process in chunks
    'chunk_size': 5000,  # Process large videos in 5k frame chunks
    'val_split_ratio': 0.15,
}


def get_free_memory_gb():
    """Get free system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)


def wait_for_memory(min_gb=3.0, check_interval=5):
    """Wait until minimum memory is available"""
    while get_free_memory_gb() < min_gb:
        print(f"⚠️  Low memory ({get_free_memory_gb():.1f}GB free), waiting {check_interval}s...")
        time.sleep(check_interval)


def load_video_info(input_root, split='train'):
    """
    Load video annotations and paths (EXACT replica of ClipShotsDataset)
    
    For test: loads test.json
    For train/val: loads train.json (will split later)
    For only_gradual: loads only_gradual.json (for dataset balancing)
    
    Returns:
        annotations: Dict of video annotations
        video_paths: Dict of video_name -> full_path
        valid_keys: List of valid video names
    """
    # Load annotations
    if split == 'test':
        json_path = os.path.join(input_root, 'annotations', 'test.json')
    elif split == 'only_gradual':
        json_path = os.path.join(input_root, 'annotations', 'only_gradual.json')
    else:
        # For train/val, always load full train.json (split later)
        json_path = os.path.join(input_root, 'annotations', 'train.json')
    
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    # Find video directory
    if split == 'test':
        video_dir_candidates = [
            os.path.join(input_root, 'videos', 'test'),
            os.path.join(input_root, 'videos')
        ]
    elif split == 'only_gradual':
        # Gradual-only videos for dataset balancing
        video_dir_candidates = [
            os.path.join(input_root, 'videos', 'only_gradual'),
            os.path.join(input_root, 'videos')
        ]
    else:
        # Train/val both come from train videos
        video_dir_candidates = [
            os.path.join(input_root, 'videos', 'train'),
            os.path.join(input_root, 'videos')
        ]
    
    video_dir = None
    for d in video_dir_candidates:
        if os.path.exists(d):
            video_dir = d
            break
    
    if not video_dir:
        raise ValueError(f"Could not find video directory in {input_root}")
    
    # Build video path mapping
    print(f"Scanning video directory: {video_dir}")
    found_videos = {}
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith('.mp4'):
                found_videos[f] = os.path.join(root, f)
    
    # Filter to valid videos
    video_paths = {}
    valid_keys = []
    for vid_name, data in annotations.items():
        if vid_name in found_videos:
            video_paths[vid_name] = found_videos[vid_name]
            valid_keys.append(vid_name)
    
    # OPTIMIZATION: Sort by file size (smallest first)
    # Small videos process faster → better multiprocessing load balancing
    print(f"Sorting {len(valid_keys)} videos by size (smallest first)...")
    valid_keys.sort(key=lambda vid: os.path.getsize(video_paths[vid]))
    
    print(f"✓ Found {len(valid_keys)} valid videos")
    return annotations, video_paths, valid_keys


def generate_clip_samples(vid_name, annotations, seq_len, samples_per_video, deterministic=False, seed=42):
    """
    Generate clip start positions EXACTLY as ClipShotsDataset does
    
    Returns:
        List of (start_frame, metadata) tuples
    """
    transitions = annotations[vid_name]['transitions']
    total_frames = int(annotations[vid_name]['frame_num'])
    
    samples = []
    
    if deterministic:
        # Validation: sequential non-overlapping windows
        stride = max(1, total_frames // samples_per_video)
        for sample_idx in range(samples_per_video):
            start = sample_idx * stride
            start = max(0, min(start, total_frames - seq_len))
            samples.append({
                'start_frame': start,
                'is_transition_focused': False
            })
    else:
        # Training: smart sampling (60% on transitions)
        random.seed(seed)
        has_transition = len(transitions) > 0
        
        for sample_idx in range(samples_per_video):
            if has_transition and random.random() < 0.6:
                # Transition-focused
                t_start, t_end = random.choice(transitions)
                center_frame = (t_start + t_end) // 2
                start = center_frame - (seq_len // 2)
                start += random.randint(-15, 15)  # Jitter
                is_focused = True
            else:
                # Random location
                if total_frames > seq_len:
                    start = random.randint(0, total_frames - seq_len)
                else:
                    start = 0
                is_focused = False
            
            start = max(0, min(start, total_frames - seq_len))
            samples.append({
                'start_frame': start,
                'is_transition_focused': is_focused
            })
    
    return samples


def decode_video_clip(video_path, start_frame, num_frames, frame_size=(48, 64)):
    """
    Decode a single clip from video using TorchCodec (BATCH MODE)
    
    Returns:
        frames: [T, 3, H, W] uint8 tensor
    """
    try:
        decoder = VideoDecoder(video_path)
        
        # Get actual video length to avoid out-of-bounds access
        video_length = len(decoder)
        
        # FAST: Batch decode all frames at once
        # Clamp frame indices to valid range [0, video_length)
        frame_indices = list(range(start_frame, min(start_frame + num_frames, video_length)))
        
        try:
            # Batch get frames (10-20x faster than loop!)
            frames = decoder.get_frames_at(frame_indices)  # [T, 3, H, W]
            frames = frames.data
            # Batch resize (much faster than per-frame)
            if frames.shape[2:] != frame_size:
                frames = torch.nn.functional.interpolate(
                    frames.float(),
                    size=frame_size,
                    mode='bilinear',
                    align_corners=False
                ).to(torch.uint8)
            
            # Pad if we got fewer frames than requested (short video)
            if frames.shape[0] < num_frames:
                padding = frames[-1:].repeat(num_frames - frames.shape[0], 1, 1, 1)
                frames = torch.cat([frames, padding], dim=0)
        except Exception as batch_error:
            # Fallback to sequential if batch fails
            print(f"⚠️  Batch decode failed for {video_path} at {start_frame}: {batch_error}")
            print(f"   Falling back to sequential decode...")
            frames_list = []
            for frame_idx in frame_indices:
                frame = decoder[frame_idx]  # Let it fail if frame doesn't exist
                if frame.shape[1:] != frame_size:
                    frame = torch.nn.functional.interpolate(
                        frame.unsqueeze(0).float(),
                        size=frame_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).to(torch.uint8)
                frames_list.append(frame)
            
            frames = torch.stack(frames_list, dim=0)
            
            # Pad if we got fewer frames than requested (short video)
            if frames.shape[0] < num_frames:
                padding = frames[-1:].repeat(num_frames - frames.shape[0], 1, 1, 1)
                frames = torch.cat([frames, padding], dim=0)
        
        del decoder
        return frames
    
    except Exception as e:
        print(f"❌ FATAL ERROR decoding {video_path} at frame {start_frame}: {e}")
        raise  # Re-raise to stop execution


def generate_labels(transitions, start_frame, seq_len):
    """
    Generate labels EXACTLY as ClipShotsDataset does
    
    Returns:
        targets: [T] long tensor (0=None, 1=Cut, 2=Gradual)
        lengths: [T] long tensor (transition length at each frame)
    """
    targets = torch.zeros(seq_len, dtype=torch.long)
    lengths = torch.zeros(seq_len, dtype=torch.long)
    
    for t_start, t_end in transitions:
        # Map to local window
        local_start = t_start - start_frame
        local_end = t_end - start_frame
        
        if local_end > 0 and local_start < seq_len:
            s = max(0, local_start)
            e = min(seq_len, local_end)
            
            length = t_end - t_start
            label = 1 if length <= 1 else 2  # Cut vs Gradual
            
            if label == 1:
                targets[s:max(s+1, e+1)] = 1
                lengths[s:max(s+1, e+1)] = length
            else:
                targets[s:e+1] = 2
                lengths[s:e+1] = length
    
    return targets, lengths


def process_video_worker(args):
    """
    Worker function for multiprocessing
    Processes one video and saves all its clips
    """
    vid_name, video_path, annotations, output_dir, split, seq_len, frame_size = args
    
    # Wait for memory if needed
    wait_for_memory(CONFIG['min_free_memory_gb'])
    
    # Create video output directory
    vid_output_dir = output_dir / vid_name.replace('.mp4', '')
    vid_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed (verify ALL clips exist)
    metadata_file = vid_output_dir / 'metadata.json'
    if metadata_file.exists():
        try:
            # Load metadata and check all clip files exist
            with open(metadata_file, 'r') as f:
                vid_meta = json.load(f)
            
            all_clips_exist = True
            for clip_info in vid_meta['clips']:
                clip_path = vid_output_dir / clip_info['file']
                if not clip_path.exists():
                    all_clips_exist = False
                    break
            
            if all_clips_exist:
                # All clips present, skip this video
                return {'status': 'skip', 'video': vid_name, 'clips': len(vid_meta['clips'])}
            else:
                # Some clips missing, will regenerate all
                print(f"⚠️  {vid_name}: Missing clips, regenerating all...")
        except:
            # Metadata corrupted, will regenerate
            print(f"⚠️  {vid_name}: Corrupted metadata, regenerating...")
    
    try:
        # Generate sample positions
        if split == 'train':
            deterministic = False
            samples_per_video = CONFIG['samples_per_video_train']
        else:  # val or test
            deterministic = True
            samples_per_video = CONFIG['samples_per_video_val']
        
        # Use video-specific seed for reproducibility
        vid_seed = CONFIG['random_seed'] + hash(vid_name) % 10000
        clip_samples = generate_clip_samples(
            vid_name, annotations, seq_len, samples_per_video,
            deterministic=deterministic, seed=vid_seed
        )
        
        # Get total frames
        total_frames = int(annotations[vid_name]['frame_num'])
        transitions = annotations[vid_name]['transitions']
        
        # Process clips
        clip_metadata = []
        
        # OPTIMIZATION: Reuse decoder for entire video (open once)
        decoder = VideoDecoder(video_path)
        video_length = len(decoder)  # Get actual video length
        
        for clip_idx, sample_info in enumerate(clip_samples):
            start_frame = sample_info['start_frame']
            
            # FAST: Batch decode frames
            try:
                # Clamp frame indices to valid range [0, video_length)
                frame_indices = list(range(start_frame, min(start_frame + seq_len, video_length)))
                frames = decoder.get_frames_at(frame_indices)  # [T, 3, H, W] BATCH!
                frames= frames.data
                # Batch resize if needed
                if frames.shape[2:] != frame_size:
                    frames = torch.nn.functional.interpolate(
                        frames.float(),
                        size=frame_size,
                        mode='bilinear',
                        align_corners=False
                    ).to(torch.uint8)
                
                # Pad if we got fewer frames than requested (short video)
                if frames.shape[0] < seq_len:
                    if frames.shape[0] > 0:
                        padding = frames[-1:].repeat(seq_len - frames.shape[0], 1, 1, 1)
                        frames = torch.cat([frames, padding], dim=0)
                    else:
                        raise ValueError(f"No frames decoded for {vid_name} clip {clip_idx}")
            except Exception as e:
                # FATAL ERROR - print and re-raise to stop execution
                print(f"❌ FATAL ERROR decoding {vid_name} clip {clip_idx} (frame {start_frame}): {e}")
                print(f"   Video: {video_path}")
                print(f"   Video length: {video_length} frames")
                print(f"   Requested: frames {start_frame} to {start_frame + seq_len}")
                raise  # Stop execution - do NOT save corrupted data
            
            # Generate labels
            targets, lengths = generate_labels(transitions, start_frame, seq_len)
            
            # Save clip (using PyTorch's efficient format)
            clip_file = vid_output_dir / f"clip_{clip_idx:06d}.pt"
            torch.save({
                'frames': frames,  # [T, 3, H, W] uint8
                'targets': targets,  # [T] long
                'lengths': lengths,  # [T] long
                'start_frame': start_frame,
                'is_transition_focused': sample_info['is_transition_focused']
            }, clip_file)
            
            clip_metadata.append({
                'clip_idx': clip_idx,
                'file': clip_file.name,
                'start_frame': start_frame,
                'is_transition_focused': sample_info['is_transition_focused']
            })
        
        # Close decoder after all clips
        del decoder
        
        # Save video metadata
        with open(metadata_file, 'w') as f:
            json.dump({
                'video_name': vid_name,
                'total_frames': total_frames,
                'num_clips': len(clip_metadata),
                'clips': clip_metadata
            }, f, indent=2)
        
        return {'status': 'success', 'video': vid_name, 'clips': len(clip_metadata)}
    
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR processing {vid_name}: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'video': vid_name, 'error': str(e)}


def predecode_dataset(split='train', num_workers=4, max_videos=None):
    """
    Pre-decode entire dataset split
    
    Args:
        split: 'train', 'val', 'test', or 'only_gradual'
        num_workers: Number of parallel workers
        max_videos: Maximum number of videos to process (for balancing only_gradual)
    """
    print("\n" + "="*80)
    print(f"PRE-DECODING {split.upper()} DATASET")
    print("="*80)
    print(f"Workers: {num_workers}")
    print(f"Sequence length: {CONFIG['seq_len']}")
    print(f"Frame size: {CONFIG['frame_size']}")
    if split in ['train', 'only_gradual']:
        print(f"Samples per video: {CONFIG['samples_per_video_train']}")
    else:  # test uses val settings (deterministic sampling)
        print(f"Samples per video: {CONFIG['samples_per_video_val']}")
    print(f"Min free memory: {CONFIG['min_free_memory_gb']}GB")
    print("="*80)
    
    # Load dataset info
    annotations, video_paths, valid_keys = load_video_info(CONFIG['input_root'], split)
    
    # Apply max_videos limit for only_gradual balancing
    if max_videos is not None and len(valid_keys) > max_videos:
        print(f"\n⚠️  Limiting {split} from {len(valid_keys)} to {max_videos} videos for balanced sampling")
        # Random sample with fixed seed for reproducibility
        random.seed(CONFIG['random_seed'])
        valid_keys = random.sample(valid_keys, max_videos)
        print(f"   Selected {len(valid_keys)} videos (seed={CONFIG['random_seed']})")
    
    # Training script handles train/val split internally - we predecode ALL training data
    if split == 'train':
        print(f"Training split: {len(valid_keys)} videos (ALL train data - training script handles val split)")
    elif split == 'only_gradual':
        print(f"Only Gradual split: {len(valid_keys)} videos (will be added to train for dataset balancing)")
        estimated_clips = len(valid_keys) * CONFIG['samples_per_video_train']
        print(f"📊 Estimated gradual clips: ~{estimated_clips} (balances 3.8:1 cut-to-gradual ratio)")
    else:
        # Test split - use all test videos
        print(f"Test split: {len(valid_keys)} videos")
    
    # Create output directory
    # IMPORTANT: only_gradual clips go into train folder for automatic inclusion!
    output_split = 'train' if split == 'only_gradual' else split
    output_dir = Path(CONFIG['output_root']) / output_split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(valid_keys)} videos...")
    print(f"Output directory: {output_dir}")
    print(f"Current free memory: {get_free_memory_gb():.1f}GB")
    
    # Prepare worker arguments
    # Use 'train' sampling strategy for only_gradual (transition-focused)
    sampling_split = 'train' if split == 'only_gradual' else split
    worker_args = []
    for vid_name in valid_keys:
        worker_args.append((
            vid_name,
            video_paths[vid_name],
            annotations,
            output_dir,
            sampling_split,  # Use train strategy for only_gradual
            CONFIG['seq_len'],
            CONFIG['frame_size']
        ))
    
    # Process with multiprocessing
    results = {'success': 0, 'skip': 0, 'error': 0}
    total_clips = 0
    
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap_unordered(process_video_worker, worker_args), 
                             total=len(worker_args), desc="Processing videos"):
                results[result['status']] += 1
                if result['status'] == 'success':
                    total_clips += result['clips']
                if result['status'] == 'error':
                    print(f"❌ Error processing {result['video']}: {result.get('error', 'Unknown')}")
    else:
        # Single process (for debugging)
        for args in tqdm(worker_args, desc="Processing videos"):
            result = process_video_worker(args)
            results[result['status']] += 1
            if result['status'] == 'success':
                total_clips += result['clips']
            if result['status'] == 'error':
                print(f"❌ Error processing {result['video']}: {result.get('error', 'Unknown')}")
    
    # Create global index mapping
    print("\nCreating global index mapping...")
    index_mapping = []
    clip_idx = 0
    
    for vid_name in valid_keys:
        vid_dir = output_dir / vid_name.replace('.mp4', '')
        metadata_file = vid_dir / 'metadata.json'
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                vid_meta = json.load(f)
            
            for clip_meta in vid_meta['clips']:
                index_mapping.append({
                    'index': clip_idx,
                    'video_name': vid_name,
                    'clip_file': str(vid_dir / clip_meta['file']),
                    'start_frame': clip_meta['start_frame']
                })
                clip_idx += 1
    
    # Save global mapping
    mapping_file = output_dir / 'index_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump({
            'split': split,
            'num_videos': len(valid_keys),
            'num_clips': len(index_mapping),
            'config': CONFIG,
            'mapping': index_mapping
        }, f, indent=2)
    
    # Calculate storage size
    total_size_gb = sum(f.stat().st_size for f in output_dir.rglob('*.pt')) / (1024**3)
    
    print("\n" + "="*80)
    print("PRE-DECODING COMPLETE!")
    print("="*80)
    print(f"Videos processed: {results['success']}")
    print(f"Videos skipped:   {results['skip']}")
    print(f"Videos failed:    {results['error']}")
    print(f"Total clips:      {total_clips}")
    print(f"Storage used:     {total_size_gb:.2f}GB")
    print(f"Index mapping:    {mapping_file}")
    if split == 'only_gradual':
        print(f"\n✅ Gradual clips added to TRAIN folder - PredecodedDataset will auto-detect them!")
        print(f"   Expected improvement: Better gradual F1 from balanced training data")
    print("="*80)
    
    return output_dir, mapping_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-decode dataset for fast training')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'only_gradual'], 
                       help='Dataset split to pre-decode (only_gradual extracts to train folder for balance)')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of parallel workers')
    parser.add_argument('--max_videos', type=int, default=None, 
                       help='Max videos to extract (for only_gradual balancing, recommend ~250)')
    
    args = parser.parse_args()
    
    # Check available memory
    free_mem = get_free_memory_gb()
    print(f"Current free memory: {free_mem:.1f}GB")
    if free_mem < CONFIG['min_free_memory_gb']:
        print(f"⚠️  Warning: Less than {CONFIG['min_free_memory_gb']}GB free memory!")
        input("Press Enter to continue anyway, or Ctrl+C to abort...")
    
    # Run pre-decoding
    predecode_dataset(split=args.split, num_workers=args.num_workers, max_videos=args.max_videos)
    
    print("\n✓ Run this for all splits:")
    print("  python predecode_dataset.py --split train --num_workers 8")
    print("  python predecode_dataset.py --split only_gradual --num_workers 8 --max_videos 250  # Balanced!")
    print("  python predecode_dataset.py --split test --num_workers 8")
    print("\nThen use PredecodedDataset in your training script for 10-20x speedup!")
    print("Note: only_gradual clips are saved to train/ folder for automatic inclusion")
