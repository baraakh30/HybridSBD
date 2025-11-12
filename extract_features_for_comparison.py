"""
Extract features with different sampling rates for comparison.
This creates two sets of features:
1. No sampling (rate=1) - all frames processed
2. High sampling (rate=8) - every 8th frame processed

Features are saved in separate directories to avoid overwriting existing features.
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to the Python path to import our modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from core.config import VideoShotConfig, FeatureConfig
from core.coordinator import ProcessingCoordinator

# Configuration
VIDEO_DIR = r"C:\Users\Baraa\Downloads\ClipShots\videos\train"
FEATURES_DIR_NO_SAMPLING = "training_features_no_sampling"
FEATURES_DIR_SAMPLING_8 = "training_features_sampling_8"


def load_selected_videos():
    """Load the selected videos from the JSON file."""
    with open('selected_videos.json', 'r') as f:
        data = json.load(f)
    # data format: [(video_name, frame_count, transitions, boundaries), ...]
    return [(item[0], item[1], item[2], item[3]) for item in data]


def extract_features_with_sampling(video_files, features_dir, sampling_rate, description):
    """
    Extract features with specified sampling rate.
    
    Args:
        video_files: List of video file paths
        features_dir: Directory to save features
        sampling_rate: Sampling rate for all features (1 = no sampling)
        description: Description for progress bar
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING FEATURES - {description}")
    print(f"{'='*80}")
    print(f"Output directory: {features_dir}")
    print(f"Sampling rate: {sampling_rate}")
    print(f"Videos to process: {len(video_files)}")
    
    # Create output directory
    os.makedirs(features_dir, exist_ok=True)
    
    # Create configuration with specified sampling rate
    config = VideoShotConfig()
    config.features = FeatureConfig(
        hsv_enabled=True,
        hsv_bins=256,
        hsv_smoothing=True,
        hsv_frame_sampling_rate=sampling_rate,
        hsv_interpolation_method="linear",
        
        cnn_enabled=True,
        cnn_model="resnet50",
        cnn_pretrained=True,
        cnn_frame_sampling_rate=sampling_rate,
        cnn_interpolation_method="linear",
        
        clip_enabled=True,
        clip_model="ViT-B/32",
        clip_frame_sampling_rate=sampling_rate,
        clip_interpolation_method="linear",
        
        optical_flow_enabled=True,
        optical_flow_method="farneback",
        optical_flow_frame_sampling_rate=sampling_rate,
        optical_flow_interpolation_method="linear"
    )
    
    # Set logging to ERROR to reduce output
    config.logging.level = "ERROR"
    config.logging.quiet_mode = True
    
    # Initialize coordinator
    print("\nInitializing feature extractors...")
    coordinator = ProcessingCoordinator(config)
    coordinator.initialize()
    print(f"Initialized extractors: {', '.join(coordinator.feature_extractors.keys())}")
    
    # Process each video
    for video_file in tqdm(video_files, desc=description):
        video_stem = video_file.stem
        
        # Check if all features already exist
        all_exist = all([
            os.path.exists(os.path.join(features_dir, f"{video_stem}_hsv_histogram.npy")),
            os.path.exists(os.path.join(features_dir, f"{video_stem}_cnn.npy")),
            os.path.exists(os.path.join(features_dir, f"{video_stem}_clip.npy")),
            os.path.exists(os.path.join(features_dir, f"{video_stem}_optical_flow.npy"))
        ])
        
        if all_exist:
            tqdm.write(f"  ✓ Features already exist for {video_file.name}")
            continue
        
        try:
            # Load video
            frames = coordinator.video_processor.load_video(
                str(video_file),
                max_retries=3,
                skip_corrupted=True
            )
            
            if frames is None or len(frames) == 0:
                tqdm.write(f"  ✗ Failed to load video: {video_file.name}")
                continue
            
            tqdm.write(f"  Processing {video_file.name} ({len(frames)} frames)...")
            
            # Extract features
            features = coordinator._extract_all_features(frames, parallel=False, max_workers=1)
            
            # Save individual features
            for feature_name, feature_array in features.items():
                individual_file = os.path.join(features_dir, f"{video_stem}_{feature_name}.npy")
                np.save(individual_file, feature_array)
                tqdm.write(f"    Saved {feature_name}: {feature_array.shape}")
            
        except Exception as e:
            tqdm.write(f"  ✗ Error processing {video_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Feature extraction complete for {description}")


def main():
    """Main function to extract features with different sampling rates."""
    print("="*80)
    print("FEATURE EXTRACTION FOR SAMPLING RATE COMPARISON")
    print("="*80)
    print("\nNOTE: HSV histogram features don't use sampling - they process all frames.")
    print("      Only CNN, CLIP, and Optical Flow will have different sampling rates.")
    
    # Load selected videos
    print("\nLoading selected videos...")
    selected_videos = load_selected_videos()
    print(f"Found {len(selected_videos)} selected videos")
    
    # Get video file paths
    video_dir = Path(VIDEO_DIR)
    video_files = []
    
    for video_name, frame_count, transitions, boundaries in selected_videos:
        # Try different extensions
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_file = video_dir / f"{video_name}{ext}"
            if video_file.exists():
                video_files.append(video_file)
                break
        else:
            print(f"Warning: Video file not found for {video_name}")
    
    print(f"Found {len(video_files)} video files")
    
    if not video_files:
        print("ERROR: No video files found!")
        return
    
    # Extract features with no sampling (rate=1)
    extract_features_with_sampling(
        video_files,
        FEATURES_DIR_NO_SAMPLING,
        sampling_rate=1,
        description="No Sampling (Rate=1)"
    )
    
    # Extract features with sampling rate 8
    extract_features_with_sampling(
        video_files,
        FEATURES_DIR_SAMPLING_8,
        sampling_rate=8,
        description="High Sampling (Rate=8)"
    )
    
    print("\n" + "="*80)
    print("ALL FEATURE EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nFeature directories created:")
    print(f"  1. No sampling:     {FEATURES_DIR_NO_SAMPLING}/")
    print(f"  2. Sampling rate 8: {FEATURES_DIR_SAMPLING_8}/")
    print(f"\nYou can now run the analysis script on both directories to compare results.")


if __name__ == "__main__":
    main()
