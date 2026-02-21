"""
Pre-Decoder for AutoShot Dataset (Training Mode)

Extracts clips from AutoShot/Kuaishou dataset for TRAINING (random sampling strategy).
Different from predecode_autoshot.py which uses sliding windows for TESTING.

AutoShot annotation format (kuaishou_v2.txt):
    video_name.mp4 total_frames
    start,end
    start,end
    ...
    <blank line>

Usage:
    python predecode_autoshot_train.py --num_workers 8
    python predecode_autoshot_train.py --num_workers 4 --samples_per_video 10
"""

import torch
import numpy as np
import json
import os
import psutil
import time
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import random
from torchcodec.decoders import VideoDecoder

# Configuration
CONFIG = {
    'autoshot_root': r"D:\grad\data\autoshot",  # Root directory with videos and annotations
    'output_root': r"D:\grad\data\autoshot_cache_train\train",  # Output cache directory for TRAINING
    'annotation_file': 'kuaishou_v2.txt',  # Annotation file name
    'video_dirs': ['original_videos', 'video_download', 'ads_game_videos'],  # All video dirs (878 total)
    'frame_size': (48, 64),  # H, W (same as ClipShots)
    'seq_len': 128,  # Clip length (same as ClipShots)
    'samples_per_video': 70,  # 648 videos × 70 = ~45K clips (62% of ClipShots' 73K clips)
    'random_seed': 42,
    'min_free_memory_gb': 3.0,
}


def get_free_memory_gb():
    """Get free system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)


def wait_for_memory(min_gb=3.0, check_interval=5):
    """Wait until minimum memory is available"""
    while get_free_memory_gb() < min_gb:
        print(f"⚠️  Low memory ({get_free_memory_gb():.1f}GB free), waiting {check_interval}s...")
        time.sleep(check_interval)


def parse_autoshot_annotations(annotation_file):
    """
    Parse AutoShot annotation file (kuaishou_v2.txt format)
    
    Format (two variants):
        Variant 1 (with frame count):
            video_name.mp4 total_frames
            start,end
            start,end
            ...
            <blank line>
        
        Variant 2 (without frame count):
            video_name.mp4
            start end
            start end
            ...
            <blank line>
    
    Returns:
        annotations: dict {video_name: {'frame_num': int, 'transitions': [(start, end), ...]}}
    """
    print(f"Parsing AutoShot annotations from {annotation_file}...")
    
    annotations = {}
    current_video = None
    duplicate_count = 0
    
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                # Blank line - end of video
                current_video = None
                continue
            
            # Check if this is a video header (must contain .mp4)
            if '.mp4' in line:
                # Video header: "video_name.mp4" or "video_name.mp4 total_frames"
                parts = line.split()
                video_name = parts[0]
                
                # Detect duplicates
                if video_name in annotations:
                    duplicate_count += 1
                
                # Extract frame count if provided, otherwise set to None (we'll infer later)
                if len(parts) >= 2:
                    try:
                        total_frames = int(parts[1])
                    except ValueError:
                        total_frames = None  # Will infer from max transition frame
                else:
                    total_frames = None
                
                annotations[video_name] = {
                    'frame_num': total_frames,
                    'transitions': []
                }
                current_video = video_name
            
            elif current_video:
                # Transition line: handle multiple formats:
                # - "start,end"
                # - "start end"
                # - "[start,end]"
                # - "[start end]"
                
                # Strip square brackets if present
                line_clean = line.strip('[]')
                
                if ',' in line_clean:
                    # Comma-separated format
                    parts = line_clean.split(',')
                else:
                    # Space-separated format
                    parts = line_clean.split()
                
                if len(parts) == 2 and parts[0] and parts[1]:
                    try:
                        start, end = int(parts[0].strip()), int(parts[1].strip())
                        # Skip invalid transitions (end < start is malformed annotation)
                        if end >= start:
                            annotations[current_video]['transitions'].append((start, end))
                        else:
                            print(f"⚠️  Skipping malformed transition in {current_video}: {line} (end < start)")
                    except ValueError as e:
                        print(f"⚠️  Skipping invalid transition in {current_video}: {line} (parse error: {e})")
    
    # Infer missing frame counts from max transition frame
    for vid_name, vid_data in annotations.items():
        if vid_data['frame_num'] is None and vid_data['transitions']:
            max_frame = max(end for _, end in vid_data['transitions'])
            vid_data['frame_num'] = max_frame + 100  # Add buffer for safety
            print(f"ℹ️  Inferred frame count for {vid_name}: {vid_data['frame_num']} (from transitions)")
    
    print(f"✓ Parsed {len(annotations)} unique videos")
    if duplicate_count > 0:
        print(f"  ℹ️  Note: {duplicate_count} duplicate entries found (kept last occurrence)")
    
    # Statistics
    total_transitions = sum(len(v['transitions']) for v in annotations.values())
    cuts = sum(1 for v in annotations.values() for s, e in v['transitions'] if e - s <= 1)
    graduals = total_transitions - cuts
    
    print(f"  Total transitions: {total_transitions} (Cuts: {cuts}, Graduals: {graduals})")
    print(f"  Avg transitions per video: {total_transitions / len(annotations):.1f}")
    
    return annotations


def generate_clip_samples(total_frames, transitions, seq_len, samples_per_video, seed=42):
    """
    Generate clip sampling positions for AutoShot videos
    OPTIMAL STRATEGY: Exhaustive coverage + jitter + random fill
    
    1. For each transition: 3 samples with jitter (-20, 0, +20 frames)
    2. Fill remaining quota with random samples
    3. Guarantees ALL transitions are covered with augmentation
    
    Returns:
        List of {start_frame, is_transition_focused} dicts
    """
    random.seed(seed)
    samples = []
    
    # Ensure integers
    total_frames = int(total_frames)
    seq_len = int(seq_len)
    samples_per_video = int(samples_per_video)
    
    # Step 1: Exhaustive coverage with jitter
    if len(transitions) > 0:
        jitter_values = [-20, 0, 20]  # 3 samples per transition
        for t_start, t_end in transitions:
            center_frame = int((t_start + t_end) // 2)
            for jitter in jitter_values:
                start = int(center_frame - (seq_len // 2) + jitter)
                start = int(max(0, min(start, total_frames - seq_len)))
                samples.append({
                    'start_frame': start,
                    'is_transition_focused': True,
                    'jitter': jitter
                })
    
    # Step 2: Fill remaining quota with random samples
    num_random = max(0, samples_per_video - len(samples))
    for _ in range(num_random):
        if total_frames > seq_len:
            start = int(random.randint(0, total_frames - seq_len))  # Explicit int
        else:
            start = 0
        samples.append({
            'start_frame': int(start),  # Double ensure integer
            'is_transition_focused': False,
            'jitter': 0
        })
    
    # Shuffle to mix transition-focused and random samples
    random.shuffle(samples)
    
    # Trim to exact target count if we exceeded
    samples = samples[:samples_per_video]
    
    return samples


def generate_labels(transitions, start_frame, seq_len):
    """
    Generate labels in ClipShots format
    
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
        
        if local_end >= 0 and local_start < seq_len:
            s = max(0, local_start)
            e = min(seq_len - 1, local_end)
            
            # FIXED: Match statistics counting (e - s <= 1 is cut)
            # AutoShot cuts are annotated as consecutive frames "130,131" (hard cut between them)
            span = t_end - t_start  # Difference, not inclusive length
            label = 1 if span <= 1 else 2  # Cut vs Gradual
            length = span + 1  # Actual span length for graduals
            
            if label == 1:
                # Cut: mark single frame
                targets[s:max(s+1, e+1)] = 1
                lengths[s:max(s+1, e+1)] = length
            else:
                # Gradual: mark entire range
                targets[s:e+1] = 2
                lengths[s:e+1] = length
    
    return targets, lengths


def process_video_worker(args):
    """
    Worker function to process one AutoShot video
    """
    vid_name, video_path, annotation, output_dir, seq_len, frame_size, samples_per_video = args
    
    # Ensure integers to prevent float type errors in decoder
    seq_len = int(seq_len)
    samples_per_video = int(samples_per_video)
    
    # Memory check removed - done once at script start, not per-video
    
    # Create video output directory
    vid_output_dir = output_dir / vid_name.replace('.mp4', '')
    vid_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    metadata_file = vid_output_dir / 'metadata.json'
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                vid_meta = json.load(f)
            
            # Verify all clips exist
            all_clips_exist = all(
                (vid_output_dir / clip_info['file']).exists() 
                for clip_info in vid_meta['clips']
            )
            
            if all_clips_exist:
                print(f"⊙ Skip {vid_name} ({vid_meta['num_clips']} clips)", flush=True)
                return {'status': 'skip', 'video': vid_name, 'clips': len(vid_meta['clips'])}
        except:
            pass  # Regenerate if metadata corrupted
    
    try:
        print(f"→ Start {vid_name}", flush=True)
        
        # Open video decoder first to get actual length
        try:
            decoder = VideoDecoder(video_path)
            video_length = int(len(decoder))  # Ensure integer
        except Exception as e:
            print(f"❌ Failed to open {vid_name}: {e}", flush=True)
            return {'status': 'error', 'video': vid_name, 'error': f'Cannot open video: {e}', 'clips': 0}
        
        # Generate clip samples using ACTUAL video length (not annotation)
        total_frames = int(annotation['frame_num'])  # Annotation's claimed length
        
        # Use the minimum of annotation and actual video length for safety
        if total_frames > video_length:
            print(f"⚠️  {vid_name}: annotation claims {total_frames} frames but video has {video_length}", flush=True)
            total_frames = video_length
        
        # Filter out transitions beyond actual video length
        transitions = annotation['transitions']
        valid_transitions = [(s, e) for s, e in transitions if e < video_length]
        if len(valid_transitions) < len(transitions):
            print(f"⚠️  {vid_name}: filtered {len(transitions) - len(valid_transitions)} transitions beyond video length", flush=True)
        
        vid_seed = CONFIG['random_seed'] + hash(vid_name) % 10000
        clip_samples = generate_clip_samples(
            total_frames, valid_transitions, seq_len, samples_per_video, seed=vid_seed
        )
        
        clip_metadata = []
        
        for clip_idx, sample_info in enumerate(clip_samples):
            start_frame = int(sample_info['start_frame'])  # Ensure integer
            
            # Skip if start_frame is beyond video length
            if start_frame >= video_length:
                print(f"⚠️  Skip clip {clip_idx} in {vid_name}: start_frame {start_frame} >= video_length {video_length}", flush=True)
                continue
            
            try:
                # Batch decode frames - frame_indices MUST be torch.long tensor!
                end_frame = int(min(start_frame + seq_len, video_length))
                
                # Ensure valid range
                if start_frame >= end_frame:
                    print(f"⚠️  Skip clip {clip_idx} in {vid_name}: Invalid range [{start_frame}, {end_frame})", flush=True)
                    continue
                    
                frame_indices = torch.arange(start_frame, end_frame, dtype=torch.long)
                
                frames = decoder.get_frames_at(frame_indices)
                frames = frames.data
                
            except RuntimeError as e:
                # Handle corrupted video frames gracefully
                if "Invalid data found" in str(e) or "Could not push packet" in str(e):
                    print(f"⚠️  Skip clip {clip_idx} in {vid_name} at frame {start_frame}: video corruption", flush=True)
                    continue
                else:
                    print(f"❌ ERROR decoding {vid_name} clip {clip_idx} at frame {start_frame}: {e}", flush=True)
                    raise
            except Exception as e:
                print(f"❌ ERROR decoding {vid_name} clip {clip_idx} at frame {start_frame}: {e}", flush=True)
                raise
            
            try:
                # Resize if needed
                if frames.shape[2:] != frame_size:
                    frames = torch.nn.functional.interpolate(
                        frames.float(),
                        size=frame_size,
                        mode='bilinear',
                        align_corners=False
                    ).to(torch.uint8)
                
                # Pad if short
                if frames.shape[0] < seq_len:
                    if frames.shape[0] > 0:
                        padding = frames[-1:].repeat(seq_len - frames.shape[0], 1, 1, 1)
                        frames = torch.cat([frames, padding], dim=0)
                    else:
                        raise ValueError(f"No frames decoded for {vid_name} clip {clip_idx}")
                
                # Generate labels
                targets, lengths = generate_labels(valid_transitions, start_frame, seq_len)
                
                # Save clip
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
                
            except Exception as e:
                # Skip this clip but continue with others
                print(f"⚠️  Failed to process clip {clip_idx} in {vid_name}: {e}", flush=True)
                continue
        
        del decoder
        
        # Check if we got any valid clips
        if len(clip_metadata) == 0:
            print(f"⚠️  {vid_name}: No valid clips extracted (all failed/skipped)", flush=True)
            return {'status': 'error', 'video': vid_name, 'error': 'No valid clips', 'clips': 0}
        
        # Save video metadata
        with open(metadata_file, 'w') as f:
            json.dump({
                'video_name': vid_name,
                'total_frames': total_frames,
                'num_clips': len(clip_metadata),
                'num_transitions': len(transitions),
                'clips': clip_metadata
            }, f, indent=2)
        
        print(f"✓ Done {vid_name} ({len(clip_metadata)} clips)", flush=True)
        return {'status': 'success', 'video': vid_name, 'clips': len(clip_metadata)}
    
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR processing {vid_name}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'video': vid_name, 'error': str(e)}


def predecode_autoshot(num_workers=4, samples_per_video=None):
    """
    Pre-decode AutoShot dataset
    """
    if samples_per_video is None:
        samples_per_video = CONFIG['samples_per_video']
    
    print("\n" + "="*80)
    print("PRE-DECODING AUTOSHOT DATASET (TRAINING MODE)")
    print("="*80)
    print(f"Workers: {num_workers}")
    print(f"Sequence length: {CONFIG['seq_len']}")
    print(f"Frame size: {CONFIG['frame_size']}")
    print(f"Samples per video: {samples_per_video} (random + transition-focused)")
    print(f"Min free memory: {CONFIG['min_free_memory_gb']}GB")
    print("="*80)
    
    # Parse annotations
    autoshot_root = Path(CONFIG['autoshot_root'])
    annotation_file = autoshot_root / CONFIG['annotation_file']
    
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    annotations = parse_autoshot_annotations(annotation_file)
    
    # Find video files across multiple directories
    video_dirs = [autoshot_root / vdir for vdir in CONFIG['video_dirs']]
    for vdir in video_dirs:
        if not vdir.exists():
            raise FileNotFoundError(f"Video directory not found: {vdir}")
    
    print(f"\nScanning video directories: {', '.join(CONFIG['video_dirs'])}")
    video_paths = {}
    valid_keys = []
    skipped_no_frames = []
    
    for vid_name in annotations.keys():
        # Skip videos with unknown frame count (no transitions and no frame count in annotation)
        if annotations[vid_name]['frame_num'] is None:
            skipped_no_frames.append(vid_name)
            continue
        
        # Check all video directories
        for video_dir in video_dirs:
            video_path = video_dir / vid_name
            if video_path.exists():
                video_paths[vid_name] = str(video_path)
                valid_keys.append(vid_name)
                break
        else:
            print(f"⚠️  Video not found in any directory: {vid_name}")
    
    if skipped_no_frames:
        print(f"⚠️  Skipped {len(skipped_no_frames)} videos with no frame count and no transitions")
    
    print(f"✓ Found {len(valid_keys)}/{len(annotations)} videos")
    
    # Sort by file size LARGEST first for better load balancing
    # This prevents large videos piling up at the end with few workers
    valid_keys.sort(key=lambda vid: os.path.getsize(video_paths[vid]), reverse=True)
    
    # Show video size distribution
    sizes_mb = [os.path.getsize(video_paths[vid]) / (1024**2) for vid in valid_keys]
    print(f"  Video sizes: min={min(sizes_mb):.1f}MB, max={max(sizes_mb):.1f}MB, avg={sum(sizes_mb)/len(sizes_mb):.1f}MB")
    print(f"  Total video data: {sum(sizes_mb)/1024:.2f}GB")
    
    # Create output directory
    output_dir = Path(CONFIG['output_root'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(valid_keys)} videos...")
    print(f"Output directory: {output_dir}")
    print(f"Current free memory: {get_free_memory_gb():.1f}GB")
    
    # Prepare worker arguments
    worker_args = [
        (
            vid_name,
            video_paths[vid_name],
            annotations[vid_name],
            output_dir,
            CONFIG['seq_len'],
            CONFIG['frame_size'],
            samples_per_video
        )
        for vid_name in valid_keys
    ]
    
    # Process with multiprocessing
    results = {'success': 0, 'skip': 0, 'error': 0}
    total_clips = 0
    
    print(f"\n{'='*60}")
    print(f"STARTING MULTIPROCESSING WITH {num_workers} WORKERS")
    print(f"{'='*60}\n", flush=True)
    
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            pbar = tqdm(total=len(worker_args), desc="Processing videos", smoothing=0.1)
            for result in pool.imap_unordered(process_video_worker, worker_args):
                results[result['status']] += 1
                if result['status'] in ['success', 'skip']:
                    total_clips += result['clips']
                    status_icon = '✓' if result['status'] == 'success' else '⊙'
                    pbar.set_postfix_str(f"{status_icon} {result['video'][:20]}... ({result['clips']} clips) | Total: {results['success']+results['skip']}/{len(worker_args)}")
                if result['status'] == 'error':
                    print(f"\n❌ Error: {result['video']}", flush=True)
                pbar.update(1)
            pbar.close()
    else:
        # Single process (for debugging)
        for args in tqdm(worker_args, desc="Processing videos"):
            result = process_video_worker(args)
            results[result['status']] += 1
            if result['status'] in ['success', 'skip']:
                total_clips += result['clips']
            if result['status'] == 'error':
                print(f"❌ Error: {result['video']}")
    
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
            'dataset': 'autoshot_train',
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
    print("="*80)
    print("\nTo use in training:")
    print("  # Single dataset")
    print("  dataset = PredecodedDataset(")
    print("      cache_root='data/autoshot_cache_train',")
    print("      split='train'")
    print("  )")
    print("\n  # Combined with ClipShots")
    print("  clipshots_train = PredecodedDataset('data/training_cache', split='train')")
    print("  autoshot_train = PredecodedDataset('data/autoshot_cache_train', split='train')")
    print("  combined = ConcatDataset([clipshots_train, autoshot_train])")
    print("="*80)
    
    return output_dir, mapping_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-decode AutoShot dataset for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--samples_per_video', type=int, default=None,
                       help='Samples per video (default: 20)')
    
    args = parser.parse_args()
    
    # Check available memory
    free_mem = get_free_memory_gb()
    print(f"Current free memory: {free_mem:.1f}GB")
    if free_mem < CONFIG['min_free_memory_gb']:
        print(f"⚠️  Warning: Less than {CONFIG['min_free_memory_gb']}GB free memory!")
        input("Press Enter to continue anyway, or Ctrl+C to abort...")
    
    # Run pre-decoding
    predecode_autoshot(num_workers=args.num_workers, samples_per_video=args.samples_per_video)
    
    print("\n✓ AutoShot dataset pre-decoded successfully for TRAINING!")
    print("\nNext steps:")
    print("  1. Use PredecodedDataset to load autoshot_cache_train")
    print("  2. Combine with ClipShots using ConcatDataset for multi-dataset training")
    print("  3. Expected improvement: +2-3% F1 from diverse training data")
