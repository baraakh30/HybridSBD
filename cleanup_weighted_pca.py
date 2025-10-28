"""
Script to delete old weighted and PCA fusion feature files that were created with the old format.
These need to be regenerated with the corrected fusion strategies.
"""

import os
from pathlib import Path
import glob

def cleanup_fusion_features(features_dir: str, strategies: list = ["weighted", "pca"]):
    """
    Delete fusion feature files for specified strategies.
    
    Args:
        features_dir: Directory containing feature files
        strategies: List of fusion strategies to clean up
    """
    features_path = Path(features_dir)
    
    if not features_path.exists():
        print(f"Features directory does not exist: {features_dir}")
        return
    
    total_deleted = 0
    
    for strategy in strategies:
        pattern = f"*_{strategy}.npy"
        files = list(features_path.glob(pattern))
        
        print(f"\nFound {len(files)} {strategy} fusion files")
        
        if files:
            response = input(f"Delete all {len(files)} {strategy} files? (yes/no): ")
            if response.lower() == 'yes':
                for file in files:
                    try:
                        file.unlink()
                        total_deleted += 1
                    except Exception as e:
                        print(f"Error deleting {file.name}: {e}")
                print(f"✓ Deleted {len(files)} {strategy} files")
            else:
                print(f"Skipped {strategy} files")
    
    print(f"\n{'='*60}")
    print(f"Total files deleted: {total_deleted}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Re-run training with --extract_features to regenerate the corrected fusion features")
    print("2. Or run: python train_transformer.py --video_dir <dir> --ground_truth <file> --features_dir training_features --extract_features --train_all_strategies")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up old fusion feature files")
    parser.add_argument("--features_dir", default="training_features", help="Features directory")
    parser.add_argument("--strategies", nargs="+", default=["weighted", "pca"], 
                       help="Fusion strategies to clean up")
    
    args = parser.parse_args()
    
    cleanup_fusion_features(args.features_dir, args.strategies)
