"""
Script to check weighted and PCA fusion feature files for correct dimensions.
Weighted should be 2D with shape (N, 3368)
PCA should be 2D with shape (N, 512)
"""

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

def check_fusion_features(features_dir: str, strategies: dict = None):
    """
    Check fusion feature files for correct dimensions.
    
    Args:
        features_dir: Directory containing feature files
        strategies: Dict mapping strategy names to expected feature dimensions
                   If None, uses defaults: {"weighted": 3368, "pca": 512}
    """
    if strategies is None:
        strategies = {
            "weighted": 3368,  # Concatenated features
            "pca": 256         # PCA reduced features
        }
    
    features_path = Path(features_dir)
    
    if not features_path.exists():
        print(f"Features directory does not exist: {features_dir}")
        return
    
    results = {}
    
    for strategy, expected_dim in strategies.items():
        print(f"\n{'='*80}")
        print(f"CHECKING {strategy.upper()} FUSION FEATURES")
        print(f"Expected shape: (N_frames, {expected_dim})")
        print(f"{'='*80}")
        
        pattern = f"*_{strategy}.npy"
        files = list(features_path.glob(pattern))
        
        if not files:
            print(f"No {strategy} fusion files found")
            results[strategy] = {
                'total': 0,
                'correct': 0,
                'incorrect': 0,
                'errors': 0,
                'incorrect_files': [],
                'error_files': []
            }
            continue
        
        print(f"Found {len(files)} {strategy} fusion files to check...")
        
        correct_count = 0
        incorrect_count = 0
        error_count = 0
        incorrect_files = []
        error_files = []
        
        # Check each file
        for file in tqdm(files, desc=f"Checking {strategy} files"):
            try:
                # Load only the shape without loading full array
                with open(file, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
                
                # Check if shape is correct
                if len(shape) != 2:
                    incorrect_count += 1
                    incorrect_files.append({
                        'file': file.name,
                        'shape': shape,
                        'reason': f"Wrong dimensions: expected 2D, got {len(shape)}D"
                    })
                elif shape[1] != expected_dim:
                    incorrect_count += 1
                    incorrect_files.append({
                        'file': file.name,
                        'shape': shape,
                        'reason': f"Wrong feature dim: expected {expected_dim}, got {shape[1]}"
                    })
                else:
                    correct_count += 1
                    
            except Exception as e:
                error_count += 1
                error_files.append({
                    'file': file.name,
                    'error': str(e)
                })
        
        # Store results
        results[strategy] = {
            'total': len(files),
            'correct': correct_count,
            'incorrect': incorrect_count,
            'errors': error_count,
            'incorrect_files': incorrect_files,
            'error_files': error_files
        }
        
        # Print summary
        print(f"\n{strategy.upper()} SUMMARY:")
        print(f"  Total files: {len(files)}")
        print(f"  ✓ Correct: {correct_count} ({correct_count/len(files)*100:.1f}%)")
        print(f"  ✗ Incorrect: {incorrect_count} ({incorrect_count/len(files)*100:.1f}%)")
        print(f"  ⚠ Errors: {error_count}")
        
        # Show some examples of incorrect files
        if incorrect_files:
            print(f"\n  First 10 incorrect files:")
            for item in incorrect_files[:10]:
                print(f"    • {item['file']}: shape={item['shape']} - {item['reason']}")
            
            if len(incorrect_files) > 10:
                print(f"    ... and {len(incorrect_files) - 10} more")
        
        # Show error files
        if error_files:
            print(f"\n  Files with errors:")
            for item in error_files[:5]:
                print(f"    • {item['file']}: {item['error']}")
            
            if len(error_files) > 5:
                print(f"    ... and {len(error_files) - 5} more")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_files = sum(r['total'] for r in results.values())
    total_correct = sum(r['correct'] for r in results.values())
    total_incorrect = sum(r['incorrect'] for r in results.values())
    total_errors = sum(r['errors'] for r in results.values())
    
    print(f"Total files checked: {total_files}")
    print(f"✓ Correct: {total_correct} ({total_correct/total_files*100 if total_files > 0 else 0:.1f}%)")
    print(f"✗ Incorrect: {total_incorrect} ({total_incorrect/total_files*100 if total_files > 0 else 0:.1f}%)")
    print(f"⚠ Errors: {total_errors}")
    
    # Recommendations
    if total_incorrect > 0 or total_errors > 0:
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        for strategy, result in results.items():
            if result['incorrect'] > 0 or result['errors'] > 0:
                print(f"\n{strategy.upper()} needs regeneration:")
                print(f"  1. Delete old {strategy} files:")
                print(f"     python cleanup_weighted_pca.py --strategies {strategy}")
                print(f"  2. Regenerate features:")
                print(f"     python train_transformer.py --video_dir <dir> --ground_truth <file> \\")
                print(f"            --features_dir training_features --train_all_strategies \\")
                print(f"            --fusion_strategies {strategy}")
    else:
        print(f"\n✓ All fusion features are correct!")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check fusion feature files for correct dimensions")
    parser.add_argument("--features_dir", default="training_features", help="Features directory")
    parser.add_argument("--weighted_dim", type=int, default=3368, 
                       help="Expected feature dimension for weighted fusion (default: 3368)")
    parser.add_argument("--pca_dim", type=int, default=256, 
                       help="Expected feature dimension for PCA fusion (default: 512)")
    
    args = parser.parse_args()
    
    strategies = {
        "weighted": args.weighted_dim,
        "pca": args.pca_dim
    }
    
    check_fusion_features(args.features_dir, strategies)
