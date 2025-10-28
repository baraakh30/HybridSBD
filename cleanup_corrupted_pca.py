"""
Script to delete corrupted PCA feature files.
Deletes PCA files that are 128 bytes or less (header only, no data).
"""

import os
from pathlib import Path

def cleanup_corrupted_pca_files(features_dir: str, dry_run: bool = True):
    """
    Delete corrupted PCA files (those with only header, no data).
    
    Args:
        features_dir: Directory containing feature files
        dry_run: If True, only print what would be deleted without actually deleting
    """
    features_path = Path(features_dir)
    
    if not features_path.exists():
        print(f"Error: Directory does not exist: {features_dir}")
        return
    
    # Find all PCA files
    pca_files = list(features_path.glob("*_pca.npy"))
    
    print(f"\nFound {len(pca_files)} PCA files")
    print(f"Checking for corrupted files (≤ 128 bytes)...\n")
    
    corrupted_files = []
    total_size_freed = 0
    
    for pca_file in pca_files:
        file_size = pca_file.stat().st_size
        
        # Files with 128 bytes or less are corrupted (header only, no data)
        if file_size <= 128:
            corrupted_files.append(pca_file)
            total_size_freed += file_size
            
            if dry_run:
                print(f"[DRY RUN] Would delete: {pca_file.name} ({file_size} bytes)")
            else:
                try:
                    pca_file.unlink()
                    print(f"Deleted: {pca_file.name} ({file_size} bytes)")
                except Exception as e:
                    print(f"Error deleting {pca_file.name}: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"Summary:")
    print(f"  Total PCA files: {len(pca_files)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    print(f"  Space {'that would be' if dry_run else ''} freed: {total_size_freed:,} bytes ({total_size_freed / 1024:.2f} KB)")
    
    if dry_run:
        print(f"\n⚠️  This was a DRY RUN - no files were deleted.")
        print(f"To actually delete the files, run with dry_run=False")
    else:
        print(f"\n✅ Cleanup complete!")
    print(f"{'=' * 70}\n")
    
    return corrupted_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Delete corrupted PCA feature files")
    parser.add_argument("--features_dir", type=str, default="training_features",
                        help="Directory containing feature files")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete files (default is dry-run mode)")
    
    args = parser.parse_args()
    
    # Run cleanup
    dry_run = not args.execute
    
    if dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN MODE - No files will be deleted")
        print("Use --execute flag to actually delete files")
        print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print("⚠️  EXECUTION MODE - Files will be permanently deleted!")
        print("=" * 70 + "\n")
        response = input("Are you sure you want to delete corrupted PCA files? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            exit(0)
    
    cleanup_corrupted_pca_files(args.features_dir, dry_run=dry_run)
