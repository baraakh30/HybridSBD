import gc
import glob
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

from hybridsbd.utils import CONFIG


class PredecodedDataset(Dataset):
    """
    Fast dataset loader for pre-decoded clips

    Loads directly from .pt files created by predecode_clipshots.py
    10-20x faster than decoding videos on the fly

    Supports:
    - train/val splits from train directory
    - test split from test directory
    - autoshot split from autoshot_cache directory
    """

    def __init__(self, cache_root, split="train", val_split_ratio=0.15, random_seed=42, dataset_type="clipshots", temporal_jitter=False):
        """
        Args:
            cache_root: Path to training_cache directory
            split: 'train', 'val', 'test', or 'autoshot'
            val_split_ratio: Fraction of videos for validation (default 0.15)
            random_seed: Seed for reproducible split (default 42)
            dataset_type: 'clipshots' or 'autoshot' (determines which cache to use)
        """
        self.cache_root = Path(cache_root)
        self.split = split
        self.dataset_type = dataset_type
        self.temporal_jitter = temporal_jitter and (split == "train")

        if split == "autoshot":
            # Load from autoshot_cache directory (separate dataset)
            mapping_file = self.cache_root.parent / "autoshot_cache" / "index_mapping.json"
            if not mapping_file.exists():
                raise FileNotFoundError(
                    f"AutoShot index mapping not found: {mapping_file}\nPlease run: python predecode_shot_test.py --num_workers 8"
                )

            print(f"Loading pre-decoded AutoShot dataset from {mapping_file}...")
            with open(mapping_file, "r") as f:
                data = json.load(f)

            # Use all AutoShot clips without splitting
            self.mapping = data["mapping"]
            self.num_videos = len(set(clip["video_name"] for clip in self.mapping))
            self.num_clips = len(self.mapping)

            print(f"✓ AUTOSHOT: {self.num_clips} clips from {self.num_videos} videos")

        elif split == "test":
            # Load from test directory (separate subdirectory)
            mapping_file = self.cache_root / "test" / "index_mapping.json"
            if not mapping_file.exists():
                raise FileNotFoundError(
                    f"Test index mapping not found: {mapping_file}\nPlease run: python predecode_clipshots.py --split test --num_workers 8"
                )

            print(f"Loading pre-decoded test dataset from {mapping_file}...")
            with open(mapping_file, "r") as f:
                data = json.load(f)

            # Use all test clips without splitting
            self.mapping = data["mapping"]
            self.num_videos = len(set(clip["video_name"] for clip in self.mapping))
            self.num_clips = len(self.mapping)

            print(f"✓ TEST split: {self.num_clips} clips from {self.num_videos} videos")

        else:
            # Load from train directory and split into train/val
            # OR load from standalone cache (like autoshot_cache_train)
            mapping_file = self.cache_root / "train" / "index_mapping.json"

            # Check if this is a standalone cache (no train/ subdirectory)
            if not mapping_file.exists():
                standalone_mapping = self.cache_root / "index_mapping.json"
                if standalone_mapping.exists():
                    # Standalone cache (e.g., autoshot_cache_train)
                    mapping_file = standalone_mapping
                    print(f"Loading standalone pre-decoded dataset from {mapping_file}...")
                else:
                    raise FileNotFoundError(
                        f"Index mapping not found in:\n"
                        f"  {mapping_file}\n"
                        f"  {standalone_mapping}\n"
                        f"Please run: python predecode_clipshots.py --split train --num_workers 8\n"
                        f"       or: python predecode_shot_train.py --num_workers 8"
                    )
            else:
                print(f"Loading pre-decoded dataset from {mapping_file}...")

            with open(mapping_file, "r") as f:
                data = json.load(f)

            all_mapping = data["mapping"]

            # Get unique videos and create train/val split (MUST match training script logic!)
            all_videos = sorted(set(clip["video_name"] for clip in all_mapping))
            num_val = max(1, int(len(all_videos) * val_split_ratio))

            rng = random.Random(random_seed)
            rng.shuffle(all_videos)
            val_videos = set(all_videos[:num_val])
            train_videos = set(all_videos[num_val:])

            # Filter clips based on split
            if split == "val":
                self.mapping = [clip for clip in all_mapping if clip["video_name"] in val_videos]
                self.num_videos = len(val_videos)
            else:  # train
                self.mapping = [clip for clip in all_mapping if clip["video_name"] in train_videos]
                self.num_videos = len(train_videos)

            self.num_clips = len(self.mapping)

            print(f"✓ {split.upper()} split: {self.num_clips} clips from {self.num_videos} videos")
            print(f"  (Total videos: {len(all_videos)}, Train: {len(train_videos)}, Val: {len(val_videos)})")

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        """
        Load a pre-decoded clip with optical flow features

        Returns:
            frames: [T, 3, H, W] float tensor (normalized to [0, 1])
            targets: [T] long tensor
            transition_lengths: [T] long tensor
            flow_features: [T-1, 4] float tensor (mean_mag, max_mag, std_mag, uniformity)
        """
        clip_info = self.mapping[idx]
        clip_file = clip_info["clip_file"]

        # Load pre-decoded clip (very fast!)
        data = torch.load(clip_file, weights_only=False)

        # Convert frames from uint8 to float32 and normalize to [0, 1]
        frames = data["frames"].float() / 255.0  # [T, 3, H, W]
        targets = data["targets"]  # [T] long
        lengths = data["lengths"]  # [T] long

        if self.temporal_jitter:
            # Time-reversal: flip the clip in time with 50% probability.
            # A cut at frame t becomes a cut at frame T-1-t.
            # This directly scrambles the position-specific FrameSimilarity and
            # ColorHistSim matrices the model would otherwise memorize.
            if torch.rand(1).item() < 0.5:
                frames = frames.flip(0)
                targets = targets.flip(0)
                lengths = lengths.flip(0)

        # All-zero frames are VALID (black frames are legitimate video content)
        return (
            frames,  # [T, 3, H, W] float32
            targets,  # [T] long
            lengths,  # [T] long
        )


class LRUVideoCache:
    """
    Two-level cache:
    1. VideoReader cache: Keep video files open
    2. Decoded frames cache: Store ready tensors for instant reuse

    CRITICAL: Properly closes old VideoReaders to prevent memory leaks
    """

    def __init__(self, capacity=50, decoded_cache_size=200):
        self.capacity = capacity
        self.cache = OrderedDict()
        # Cache for decoded frame tensors: {(path, start, end): tensor}
        self.decoded_cache = OrderedDict()
        self.decoded_cache_size = decoded_cache_size

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        # TorchCodec VideoDecoder with GPU acceleration
        try:
            decoder = VideoDecoder(path)
            self.cache[path] = decoder

            if len(self.cache) > self.capacity:
                # Remove oldest decoder
                old_path, old_decoder = self.cache.popitem(last=False)
                try:
                    del old_decoder
                except Exception:
                    pass
            return decoder
        except Exception as e:
            print(f"Error opening video {path}: {e}")
            raise

    def get_decoded(self, path, start_idx, seq_len, total_frames=None):
        """Get decoded frames from cache or decode and cache them - GPU accelerated with TorchCodec"""
        cache_key = (path, start_idx, start_idx + seq_len)

        # Check decoded cache first
        if cache_key in self.decoded_cache:
            self.decoded_cache.move_to_end(cache_key)
            return self.decoded_cache[cache_key].clone()

        # Not in cache - decode now with TorchCodec (GPU accelerated)
        try:
            decoder = self.get(path)

            # Use provided total_frames or try to get from decoder
            if total_frames is None:
                total_frames = getattr(decoder, "num_frames", seq_len * 10)  # Fallback estimate

            # Clamp frame indices to valid range
            end_idx = min(start_idx + seq_len, total_frames)
            actual_start = min(start_idx, total_frames - 1) if total_frames > 0 else 0

            # Get frame indices to decode (within valid range)
            if actual_start >= total_frames:
                # Start is beyond video - return black frames
                frames = torch.zeros((seq_len, 3, *CONFIG["frame_size"]))
                return frames

            frame_indices = list(range(actual_start, end_idx))

            if len(frame_indices) == 0:
                # No valid frames - return black frames
                frames = torch.zeros((seq_len, 3, *CONFIG["frame_size"]))
                return frames

            # Decode frames - TorchCodec returns FrameBatch with .data tensor [N, C, H, W]
            frames_batch = decoder.get_frames_at(frame_indices)
            frames_tensor = frames_batch.data  # Extract tensor from FrameBatch

            # Resize to target size
            frames = torch.nn.functional.interpolate(frames_tensor.float(), size=CONFIG["frame_size"], mode="bilinear", align_corners=False)

            # Normalize to [0, 1]
            frames = frames / 255.0

            # Handle short sequences
            if frames.shape[0] < seq_len:
                # Pad with last frame
                padding = seq_len - frames.shape[0]
                last_frame = frames[-1:].repeat(padding, 1, 1, 1)
                frames = torch.cat([frames, last_frame], dim=0)

            # Cache the result (CPU to save VRAM)
            self.decoded_cache[cache_key] = frames.cpu().clone()
            if len(self.decoded_cache) > self.decoded_cache_size:
                self.decoded_cache.popitem(last=False)

            return frames

        except Exception as e:
            print(f"Error decoding video {path}: {e}")
            # Return black frames as fallback
            frames = torch.zeros((seq_len, 3, *CONFIG["frame_size"]))
            return frames

    def clear_decoded_cache(self):
        """Clear decoded frames cache to free memory"""
        self.decoded_cache.clear()
        gc.collect()

    def __del__(self):
        """Clean up all decoders when cache is destroyed"""
        try:
            for decoder in self.cache.values():
                try:
                    del decoder
                except Exception:
                    pass
            self.cache.clear()
            self.decoded_cache.clear()
        except Exception:
            pass


class ClipShotsDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        seq_len=100,
        frame_size=(48, 64),
        cache_capacity=50,
        deterministic=False,
        val_videos=None,
        full_video_inference=False,
        samples_per_video=None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.cache_capacity = cache_capacity
        self.video_cache = None  # Initialized in worker
        self.deterministic = deterministic  # For reproducible validation
        self.full_video_inference = full_video_inference  # Process complete videos for transition metrics
        self.samples_per_video_override = samples_per_video  # Allow explicit override

        # Load Annotations
        json_path = os.path.join(root_dir, "annotations", f"{split}.json")
        if not os.path.exists(json_path):
            # Try searching recursively if structure is different
            print(f"Warning: {json_path} not found. Trying recursive search...")
            try:
                json_path = glob.glob(os.path.join(root_dir, "**", f"{split}.json"), recursive=True)[0]
            except IndexError:
                # Fallback for filenames like "clipshots_annotations.json" or "train.json" in weird places
                if split == "train":
                    json_path = os.path.join(root_dir, "annotations", "train.json")
                else:
                    json_path = os.path.join(root_dir, "annotations", "test.json")

        print(f"Loading annotations from {json_path}...")
        with open(json_path, "r") as f:
            self.annotations = json.load(f)

        # Verify video files existence and build sample list
        # We assume videos are in videos/split OR just flat
        self.video_paths = {}
        # video_dir might be just "videos" or "videos/train"
        video_dir_candidates = [os.path.join(root_dir, "videos", split), os.path.join(root_dir, "videos")]

        video_dir = None
        for d in video_dir_candidates:
            if os.path.exists(d):
                video_dir = d
                break

        if not video_dir:
            raise ValueError(f"Could not find video directory in {root_dir}")

        # Optimization: Pre-scan directory once
        print(f"Scanning video directory: {video_dir}")
        # Build map of filename -> fullpath
        found_videos = {}
        for root, dirs, files in os.walk(video_dir):
            for f in files:
                if f.endswith(".mp4"):
                    found_videos[f] = os.path.join(root, f)

        # Filter annotations to only include found videos
        self.valid_keys = []
        for vid_name, data in self.annotations.items():
            if vid_name in found_videos:
                self.video_paths[vid_name] = found_videos[vid_name]
                self.valid_keys.append(vid_name)

        # Handle validation split creation (if val_videos provided, filter to only those)
        if val_videos is not None:
            if split == "val":
                self.valid_keys = [v for v in self.valid_keys if v in val_videos]
            elif split == "train":
                self.valid_keys = [v for v in self.valid_keys if v not in val_videos]

        print(f"Found {len(self.valid_keys)} valid videos for {split}")

        # Create samples: (video_name, start_frame)
        # We can implement a sliding window or random sampling.
        # For efficiency in this script: Random sampling per epoch effectively cover dataset
        # For deterministic len, we estimate:
        if self.samples_per_video_override is not None:
            # Explicit override provided
            self.samples_per_video = self.samples_per_video_override
        elif self.full_video_inference:
            # For transition-level metrics: one sample per video (will process whole video)
            self.samples_per_video = 1
        elif self.deterministic:
            # Validation: 3 samples × 100 frames = 300 frames per video
            self.samples_per_video = 3
        else:
            # Training: 20 samples × 100 frames = 2000 frames per video per epoch
            # Smart sampling (60% on transitions) ensures diverse coverage
            self.samples_per_video = 20
            # Provides good diversity while maintaining reasonable training speed
            # Smart sampling (60% on transitions) ensures all transitions seen multiple times
            self.samples_per_video = 20

    def __len__(self):
        return len(self.valid_keys) * self.samples_per_video

    def _init_worker_cache(self):
        if self.video_cache is None:
            decoded_size = CONFIG.get("decoded_cache_size", 200)
            self.video_cache = LRUVideoCache(self.cache_capacity, decoded_cache_size=decoded_size)

    def __getitem__(self, idx):
        self._init_worker_cache()

        # Map linear index to video
        vid_idx = idx // self.samples_per_video
        vid_name = self.valid_keys[vid_idx]
        vid_path = self.video_paths[vid_name]

        try:
            transitions = self.annotations[vid_name]["transitions"]
            # Get frame count from annotations (convert to int)
            total_frames = int(self.annotations[vid_name]["frame_num"])

            if self.deterministic:
                # Deterministic validation: sequential non-overlapping windows
                sample_in_video = idx % self.samples_per_video
                stride = max(1, total_frames // self.samples_per_video)
                target_frame_start = sample_in_video * stride
                target_frame_start = max(0, min(target_frame_start, total_frames - self.seq_len))
            else:
                # Smart Sampling for training:
                # 1. Focus on transitions (Positive mining)
                # 2. Focus on normal content (Negative mining)
                target_frame_start = 0
                has_transition = len(transitions) > 0

                if has_transition and random.random() < 0.6:  # 60% transition focus
                    # Pick a random transition
                    t_start, t_end = random.choice(transitions)
                    # Center the window around this transition
                    center_frame = (t_start + t_end) // 2
                    target_frame_start = center_frame - (self.seq_len // 2)
                    # Add some jitter
                    target_frame_start += random.randint(-15, 15)
                else:
                    # Random location
                    if total_frames > self.seq_len:
                        target_frame_start = random.randint(0, total_frames - self.seq_len)

                # Clamping
                target_frame_start = max(0, min(target_frame_start, total_frames - self.seq_len))

            # OPTIMIZED: Use decoded frames cache for instant reuse (pass total_frames)
            frames = self.video_cache.get_decoded(vid_path, target_frame_start, self.seq_len, total_frames=total_frames)

            # Generate Labels
            targets = torch.zeros(self.seq_len, dtype=torch.long)
            transition_lengths = torch.zeros(self.seq_len, dtype=torch.long)  # NEW: Track lengths

            # Fill labels based on transitions
            # Offset transition times by target_frame_start
            for start, end in transitions:
                # Map global frame idx to local window idx
                local_start = start - target_frame_start
                local_end = end - target_frame_start

                # Check intersection
                if local_end > 0 and local_start < self.seq_len:
                    # Clip to window
                    s = max(0, local_start)
                    e = min(self.seq_len, local_end)  # end is exclusive in annotation?

                    # Determine type: Cut vs Gradual
                    # Heuristic based on transition length:
                    # - Length <= 1: Cut (instant scene change)
                    # - Length > 1: Gradual (dissolve/fade of any duration)
                    length = end - start
                    label = 1 if length <= 1 else 2

                    # For Cut (label 1), we often mark 2 frames around cut
                    if label == 1:
                        # Ensure we mark at least one frame
                        targets[s : max(s + 1, e + 1)] = 1
                        transition_lengths[s : max(s + 1, e + 1)] = length  # NEW
                    else:
                        targets[s : e + 1] = 2
                        transition_lengths[s : e + 1] = length  # NEW: Store length for graduals

            # Return with video metadata if in full video mode
            if self.full_video_inference:
                return frames, targets, transition_lengths, vid_name, target_frame_start
            else:
                return frames, targets, transition_lengths

        except Exception as e:
            # print(f"Error loading {vid_path}: {e}")
            # Return zero sample on error to not crash training
            if self.full_video_inference:
                return (
                    torch.zeros(self.seq_len, 3, *self.frame_size),
                    torch.zeros(self.seq_len, dtype=torch.long),
                    torch.zeros(self.seq_len, dtype=torch.long),
                    vid_name if "vid_name" in locals() else "",
                    0,
                )
            else:
                return (
                    torch.zeros(self.seq_len, 3, *self.frame_size),
                    torch.zeros(self.seq_len, dtype=torch.long),
                    torch.zeros(self.seq_len, dtype=torch.long),
                )
