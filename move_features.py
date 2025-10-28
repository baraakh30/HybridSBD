"""
Script to move files containing a specific string from training_features directory to a target path.
"""
import os
import shutil
import argparse
from pathlib import Path


def move_files_with_string(source_dir: str, target_dir: str, search_string: str, dry_run: bool = False):
    """
    Move files containing a specific string from source directory to target directory.
    
    Args:
        source_dir: Source directory to search for files
        target_dir: Target directory to move files to
        search_string: String to search for in filenames
        dry_run: If True, only show what would be moved without actually moving
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Validate source directory exists
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    # Create target directory if it doesn't exist
    if not dry_run:
        target_path.mkdir(parents=True, exist_ok=True)
    
    # Find all files containing the search string
    matching_files = [f for f in source_path.iterdir() 
                     if f.is_file() and search_string in f.name]
    
    if not matching_files:
        print(f"No files found containing '{search_string}' in {source_dir}")
        return
    
    print(f"Found {len(matching_files)} file(s) containing '{search_string}':")
    print("-" * 80)
    
    moved_count = 0
    for file_path in matching_files:
        target_file = target_path / file_path.name
        
        if dry_run:
            print(f"[DRY RUN] Would move: {file_path.name}")
            print(f"          From: {file_path}")
            print(f"          To:   {target_file}")
        else:
            try:
                shutil.move(str(file_path), str(target_file))
                print(f"✓ Moved: {file_path.name}")
                print(f"  From: {file_path}")
                print(f"  To:   {target_file}")
                moved_count += 1
            except Exception as e:
                print(f"✗ Error moving {file_path.name}: {e}")
        print()
    
    if dry_run:
        print(f"\n[DRY RUN] Would move {len(matching_files)} file(s)")
        print("Run without --dry-run to actually move the files.")
    else:
        print(f"\nSuccessfully moved {moved_count} out of {len(matching_files)} file(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Move files containing a specific string from training_features to target directory"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="training_features",
        help="Source directory to search for files (default: training_features)"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target directory to move files to"
    )
    parser.add_argument(
        "--string",
        type=str,
        required=True,
        help="String to search for in filenames"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("File Moving Script")
    print("=" * 80)
    print(f"Source directory: {args.source}")
    print(f"Target directory: {args.target}")
    print(f"Search string:    '{args.string}'")
    print(f"Dry run:          {args.dry_run}")
    print("=" * 80)
    print()
    
    move_files_with_string(args.source, args.target, args.string, args.dry_run)


if __name__ == "__main__":
    main()


# # Dry run first (recommended) - see what would be moved
# python move_features.py --string "_bgW1pAhrfE" --target "backup_features" --dry-run

# # Actually move the files
# python move_features.py --string "_bgW1pAhrfE" --target "backup_features"

# # Move files with specific fusion strategy
# python move_features.py --string "concatenation" --target "concatenation_features"

# # Custom source directory
# python move_features.py --source "training_features" --string "weighted" --target "weighted_backup"