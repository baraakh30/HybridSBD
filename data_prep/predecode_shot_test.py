"""
Pre-decode AutoShot Dataset for Testing

Converts AutoShot videos into predecoded format for fast batch testing.
AutoShot uses shot-level annotations (scene boundaries), not frame-level labels.

Structure:
    data/autoshot_cache/
        video_id/
            clip_000000.pt  # [128, 3, 48, 64] uint8 tensor
            clip_000001.pt
            ...
        metadata.json
        index_mapping.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: lets the script import the repository packages

import json
import os
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from hybridsbd.utils import get_free_memory_gb

# Configuration
CONFIG = {
    "autoshot_root": os.environ.get("AUTOSHOT_ROOT", "data/AutoShot"),
    "output_root": os.environ.get("AUTOSHOT_TEST_CACHE", "data/cache/autoshot_test"),
    "frame_size": (48, 64),  # H, W - match training
    "seq_len": 128,
    "stride": 64,  # Sliding window stride for full video coverage
    "min_free_memory_gb": 3.0,
}


def load_autoshot_annotations():
    """
    Load AutoShot ground truth annotations

    Returns:
        gt_dict: Dictionary mapping video_id -> array of [start, end] shot boundaries
        video_paths: Dictionary mapping video_id -> full video path
    """
    print("Loading AutoShot annotations...")

    # Load ground truth pickle file
    gt_path = Path(CONFIG["autoshot_root"]) / "gt_scenes_dict_baseline_v2.pickle"
    with open(gt_path, "rb") as f:
        gt_scenes_dict = pickle.load(f)

    print(f"  ✓ Loaded annotations for {len(gt_scenes_dict)} videos")

    # Search in multiple video directories
    video_dirs = [
        Path(CONFIG["autoshot_root"]) / "ads_game_videos",
        Path(CONFIG["autoshot_root"]) / "video_download",
        Path(CONFIG["autoshot_root"]) / "original_videos",
    ]

    video_paths = {}

    for video_dir in video_dirs:
        if not video_dir.exists():
            print(f"  ⚠ Directory not found: {video_dir}")
            continue

        print(f"  Searching in: {video_dir.name}...")
        found_count = 0

        for video_file in video_dir.glob("*.mp4"):
            video_id = video_file.stem  # Remove .mp4 extension
            if video_id in gt_scenes_dict and video_id not in video_paths:
                video_paths[video_id] = str(video_file)
                found_count += 1

        print(f"    ✓ Found {found_count} videos")

    print(f"  ✓ Total videos found: {len(video_paths)}")

    # Filter to videos that have both annotations and files
    valid_videos = {vid: gt_scenes_dict[vid] for vid in video_paths.keys()}

    return valid_videos, video_paths


def video_id_to_frame_labels(shot_boundaries, total_frames):
    """
    Convert shot boundaries to frame-level labels

    Args:
        shot_boundaries: Array of [start, end] shot boundaries
        total_frames: Total number of frames in video

    Returns:
        frame_labels: [total_frames] array where each frame = shot_index
    """
    frame_labels = np.zeros(total_frames, dtype=np.int32)

    for shot_idx, (start, end) in enumerate(shot_boundaries):
        # Convert to int to handle numpy float types
        start_idx = int(start)
        end_idx = int(end)
        frame_labels[start_idx : end_idx + 1] = shot_idx

    return frame_labels


def extract_shot_transitions(shot_boundaries):
    """
    Extract transition frame indices from shot boundaries

    Returns list of transition frames (boundaries between shots)
    Note: Transition frame = first frame of the NEW shot (standard convention)
    """
    if len(shot_boundaries) == 0:
        return []

    transitions = []
    for i in range(1, len(shot_boundaries)):
        # Transition is at the START of the new shot
        # Convert to int to handle numpy float types
        transition_frame = int(shot_boundaries[i][0])  # Start of next shot
        transitions.append(transition_frame)

    return transitions


def predecode_video(video_id, video_path, shot_boundaries, output_dir):
    """
    Pre-decode a single AutoShot video into clips

    Args:
        video_id: Video identifier
        video_path: Path to video file
        shot_boundaries: Array of [start, end] shot boundaries
        output_dir: Output directory for this video

    Returns:
        Dict with status and metadata
    """
    try:
        # Create video output directory
        vid_output_dir = output_dir / video_id
        vid_output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already processed
        metadata_file = vid_output_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                vid_meta = json.load(f)
            # Verify all clips exist
            all_exist = all((vid_output_dir / clip["file"]).exists() for clip in vid_meta["clips"])
            if all_exist:
                return {"status": "skip", "video": video_id, "clips": len(vid_meta["clips"])}

        # Open video decoder
        decoder = VideoDecoder(video_path)
        total_frames = len(decoder)

        print(f"\n  Processing {video_id}: {total_frames} frames, {len(shot_boundaries)} shots")

        # Build full-video shot label array once (O(N)) outside the clip loop.
        # Re-computing it inside the loop would be O(N × num_clips) for no gain.
        frame_to_shot = video_id_to_frame_labels(shot_boundaries, total_frames)

        # Generate sliding window clips (stride for full coverage)
        clips_metadata = []
        clip_idx = 0

        for start_frame in range(0, total_frames, CONFIG["stride"]):
            # Don't exceed video length
            if start_frame + CONFIG["seq_len"] > total_frames:
                # Last clip - start from end
                start_frame = max(0, total_frames - CONFIG["seq_len"])

            end_frame = min(start_frame + CONFIG["seq_len"], total_frames)

            # Decode frames (batch mode)
            frame_indices = list(range(start_frame, end_frame))
            from hybridsbd.frame_utils import decode_resized

            frames = decode_resized(decoder, frame_indices, CONFIG["frame_size"])  # [T, 3, H, W], chunked

            # Resize if needed
            if frames.shape[2:] != CONFIG["frame_size"]:
                frames = torch.nn.functional.interpolate(
                    frames.float(), size=CONFIG["frame_size"], mode="bilinear", align_corners=False
                ).to(torch.uint8)

            # Pad if needed
            if frames.shape[0] < CONFIG["seq_len"]:
                padding = frames[-1:].repeat(CONFIG["seq_len"] - frames.shape[0], 1, 1, 1)
                frames = torch.cat([frames, padding], dim=0)

            # Slice pre-computed shot label array for this clip window.
            clip_shot_labels = frame_to_shot[start_frame : start_frame + CONFIG["seq_len"]]

            # Identify transitions in this clip
            transitions = extract_shot_transitions(shot_boundaries)
            clip_transitions = []
            for trans_frame in transitions:
                if start_frame <= trans_frame < start_frame + CONFIG["seq_len"]:
                    clip_transitions.append(trans_frame - start_frame)

            # Generate boundary labels for this clip (1 at transition frames, 0 elsewhere)
            boundary_labels = np.zeros(CONFIG["seq_len"], dtype=np.float32)
            for trans_idx in clip_transitions:
                boundary_labels[trans_idx] = 1.0

            # Save clip
            clip_file = vid_output_dir / f"clip_{clip_idx:06d}.pt"
            torch.save(
                {
                    "frames": frames,  # [128, 3, 48, 64] uint8
                    "shot_labels": torch.from_numpy(clip_shot_labels).long(),  # [128] shot indices
                    "transitions": clip_transitions,  # List of transition frame indices in clip
                    "boundary_labels": torch.from_numpy(boundary_labels),  # [128] binary boundary labels
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                },
                clip_file,
            )

            clips_metadata.append(
                {
                    "clip_idx": clip_idx,
                    "file": clip_file.name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "num_transitions": len(clip_transitions),
                }
            )

            clip_idx += 1

            # Break if we've covered the whole video
            if end_frame >= total_frames:
                break

        # Close decoder
        del decoder

        # Save video metadata
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "video_id": video_id,
                    "total_frames": total_frames,
                    "num_shots": len(shot_boundaries),
                    "num_clips": len(clips_metadata),
                    "clips": clips_metadata,
                },
                f,
                indent=2,
            )

        return {"status": "success", "video": video_id, "clips": len(clips_metadata)}

    except Exception as e:
        print(f"\n❌ Error processing {video_id}: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "video": video_id, "error": str(e)}


def process_video_worker(args):
    """
    Worker function for multiprocessing
    Processes one video and saves all its clips
    """
    video_id, video_path, shot_boundaries, output_dir = args
    return predecode_video(video_id, video_path, shot_boundaries, output_dir)


def main():
    print("\n" + "=" * 80)
    print("PRE-DECODING AUTOSHOT DATASET")
    print("=" * 80)
    print(f"Frame size: {CONFIG['frame_size']}")
    print(f"Sequence length: {CONFIG['seq_len']}")
    print(f"Stride: {CONFIG['stride']} (sliding window)")
    print(f"Min free memory: {CONFIG['min_free_memory_gb']}GB")
    print("=" * 80)

    # Check memory
    free_mem = get_free_memory_gb()
    print(f"\nCurrent free memory: {free_mem:.1f}GB")
    if free_mem < CONFIG["min_free_memory_gb"]:
        print(f"⚠️  Warning: Less than {CONFIG['min_free_memory_gb']}GB free!")

    # Load annotations
    gt_scenes, video_paths = load_autoshot_annotations()

    print(f"\nProcessing {len(video_paths)} videos...")

    # Create output directory
    output_dir = Path(CONFIG["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare worker arguments
    print(f"\nPreparing {len(video_paths)} videos for multiprocessing...")
    worker_args = []
    for video_id in sorted(video_paths.keys()):
        worker_args.append((video_id, video_paths[video_id], gt_scenes[video_id], output_dir))

    # Process videos with multiprocessing
    results = {"success": 0, "skip": 0, "error": 0}
    total_clips = 0

    num_workers = max(1, cpu_count() - 1)  # Leave one core free for the OS
    print(f"Using {num_workers} workers for parallel processing...\n")

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap_unordered(process_video_worker, worker_args), total=len(worker_args), desc="Processing videos"):
                results[result["status"]] += 1
                if result["status"] in ["success", "skip"]:
                    total_clips += result["clips"]
                if result["status"] == "error":
                    print(f"\n❌ Error: {result['video']} - {result.get('error', 'Unknown')}")
    else:
        # Single process (for debugging)
        for args in tqdm(worker_args, desc="Processing videos"):
            result = process_video_worker(args)
            results[result["status"]] += 1
            if result["status"] in ["success", "skip"]:
                total_clips += result["clips"]
            if result["status"] == "error":
                print(f"\n❌ Error: {result['video']} - {result.get('error', 'Unknown')}")

    # Build index mapping
    print("\nBuilding index mapping...")
    index_mapping = []

    for video_id in sorted(video_paths.keys()):
        vid_dir = output_dir / video_id
        metadata_file = vid_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                vid_meta = json.load(f)

            for clip_meta in vid_meta["clips"]:
                index_mapping.append(
                    {
                        "index": len(index_mapping),
                        "video_id": video_id,
                        "clip_file": str(vid_dir / clip_meta["file"]),
                        "start_frame": clip_meta["start_frame"],
                        "num_transitions": clip_meta["num_transitions"],
                    }
                )

    # Save global index mapping
    mapping_file = output_dir / "index_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(
            {
                "dataset": "AutoShot",
                "num_videos": len(video_paths),
                "num_clips": len(index_mapping),
                "config": CONFIG,
                "mapping": index_mapping,
            },
            f,
            indent=2,
        )

    # Calculate storage
    total_size_gb = sum(f.stat().st_size for f in output_dir.rglob("*.pt")) / (1024**3)

    print("\n" + "=" * 80)
    print("PRE-DECODING COMPLETE!")
    print("=" * 80)
    print(f"Videos processed: {results['success']}")
    print(f"Videos skipped:   {results['skip']}")
    print(f"Videos failed:    {results['error']}")
    print(f"Total clips:      {total_clips}")
    print(f"Storage used:     {total_size_gb:.2f}GB")
    print(f"Index mapping:    {mapping_file}")
    print("=" * 80)

    print("\n✓ Now you can use the predecoded data for fast testing!")
    print("  python eval_shot.py --checkpoint outputs/hybrid_detector/best_model.pt")


if __name__ == "__main__":
    main()
