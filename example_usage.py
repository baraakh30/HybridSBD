#!/usr/bin/env python3
"""
Example usage of the train_transformer.py script with new logging and progress control features.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and print its description."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    # Note: This is just showing the commands, not actually running them
    # Uncomment the line below to actually execute the commands
    # subprocess.run(command, check=True)

def main():
    """Show example usage of the training script with different logging options."""
    
    # Base command components
    script = "python train_transformer.py"
    video_dir = "--video_dir /path/to/training/videos"
    ground_truth = "--ground_truth /path/to/ground_truth.json"
    
    print("TRAIN_TRANSFORMER.PY - LOGGING AND PROGRESS CONTROL EXAMPLES")
    print("="*80)
    print("The following examples show how to use the new logging and progress control features.")
    print("Note: Replace the paths with your actual data paths before running.")
    
    # Example 1: Default behavior (normal logging and progress bars)
    run_command([
        "python", "train_transformer.py",
        "--video_dir", "/path/to/training/videos",
        "--ground_truth", "/path/to/ground_truth.json",
        "--extract_features",
        "--fusion_strategy", "hierarchical"
    ], "Default behavior - Normal logging and progress bars")
    
    # Example 2: Quiet mode (suppress non-essential output)
    run_command([
        "python", "train_transformer.py",
        "--video_dir", "/path/to/training/videos", 
        "--ground_truth", "/path/to/ground_truth.json",
        "--extract_features",
        "--quiet",
        "--fusion_strategy", "hierarchical"
    ], "Quiet mode - Suppress non-essential output, keep progress bars")
    
    # Example 3: Disable all logging
    run_command([
        "python", "train_transformer.py",
        "--video_dir", "/path/to/training/videos",
        "--ground_truth", "/path/to/ground_truth.json", 
        "--extract_features",
        "--disable_logging",
        "--fusion_strategy", "hierarchical"
    ], "Disable all logging - No log files or console output")
    
    # Example 4: No progress bars
    run_command([
        "python", "train_transformer.py",
        "--video_dir", "/path/to/training/videos",
        "--ground_truth", "/path/to/ground_truth.json",
        "--extract_features", 
        "--no_progress",
        "--fusion_strategy", "hierarchical"
    ], "No progress bars - Traditional text output")
    
    # Example 5: File logging only
    run_command([
        "python", "train_transformer.py",
        "--video_dir", "/path/to/training/videos",
        "--ground_truth", "/path/to/ground_truth.json",
        "--extract_features",
        "--file_logging_only",
        "--fusion_strategy", "hierarchical"
    ], "File logging only - No console output, logs saved to files")
    
    # Example 6: Train all strategies with progress bars
    run_command([
        "python", "train_transformer.py", 
        "--video_dir", "/path/to/training/videos",
        "--ground_truth", "/path/to/ground_truth.json",
        "--extract_features",
        "--train_all_strategies",
        "--fusion_strategies", "concatenation", "weighted", "hierarchical"
    ], "Train multiple strategies with progress bars")
    
    # Example 7: Train all strategies in quiet mode
    run_command([
        "python", "train_transformer.py",
        "--video_dir", "/path/to/training/videos", 
        "--ground_truth", "/path/to/ground_truth.json",
        "--extract_features",
        "--train_all_strategies",
        "--quiet",
        "--fusion_strategies", "concatenation", "weighted", "hierarchical"
    ], "Train multiple strategies in quiet mode")
    
    # Example 8: Maximum silence for automated scripts
    run_command([
        "python", "train_transformer.py",
        "--video_dir", "/path/to/training/videos",
        "--ground_truth", "/path/to/ground_truth.json", 
        "--extract_features",
        "--train_all_strategies",
        "--disable_logging",
        "--quiet"
    ], "Maximum silence - For automated scripts and batch processing")
    
    print(f"\n{'='*80}")
    print("COMMAND LINE ARGUMENT SUMMARY")
    print(f"{'='*80}")
    print("Logging Control:")
    print("  --disable_logging     : Disable all logging output")
    print("  --quiet, -q          : Suppress non-essential output")
    print("  --file_logging_only  : Only log to files, no console output")
    print("  --log_level LEVEL    : Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    print()
    print("Progress Control:")
    print("  --no_progress        : Disable progress bars")
    print()
    print("Training Options:")
    print("  --train_all_strategies : Train models for multiple fusion strategies")
    print("  --extract_features    : Extract features before training")
    print("  --fusion_strategy STR : Single strategy to use")
    print("  --fusion_strategies LIST : List of strategies for multi-training")

if __name__ == "__main__":
    main()