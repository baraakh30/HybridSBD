"""
Check distribution of PCA feature dimensions across all files
"""
import numpy as np
import os
from collections import Counter
from pathlib import Path

features_dir = "training_features"

print("Checking PCA feature dimension distribution...\n")

dimensions = []
errors = []

pca_files = [f for f in os.listdir(features_dir) if f.endswith('_pca.npy')]

print(f"Total PCA files: {len(pca_files)}\n")
print("Sampling files to check dimensions...")

# Check first 100 files to get a sense of distribution
sample_size = min(100, len(pca_files))
for i, filename in enumerate(pca_files[:sample_size]):
    filepath = os.path.join(features_dir, filename)
    try:
        # Read just the header to get shape
        with open(filepath, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
        
        if len(shape) == 2:
            dimensions.append(shape[1])
    except Exception as e:
        errors.append((filename, str(e)))

print(f"Successfully checked: {len(dimensions)} files")
print(f"Errors: {len(errors)} files\n")

if dimensions:
    dim_counts = Counter(dimensions)
    print("Dimension distribution:")
    print("-" * 50)
    for dim, count in sorted(dim_counts.items(), reverse=True):
        percentage = (count / len(dimensions)) * 100
        print(f"  {dim} components: {count} files ({percentage:.1f}%)")
    
    print(f"\nMin dimension: {min(dimensions)}")
    print(f"Max dimension: {max(dimensions)}")
    print(f"Most common: {dim_counts.most_common(1)[0][0]}")
    
    unique_dims = len(dim_counts)
    print(f"\nTotal unique dimensions: {unique_dims}")
    
    if unique_dims == 1:
        print("\n✓ All files have the same dimension - no issues!")
    else:
        print(f"\n⚠ WARNING: {unique_dims} different dimensions found!")
        print("   This will cause shape mismatch errors during training!")
        print(f"   Most common dimension: {dim_counts.most_common(1)[0][0]}")
        print(f"   Consider filtering or padding to consistent dimension.")

if errors:
    print("\n\nFiles with errors (first 10):")
    print("-" * 50)
    for filename, error in errors[:10]:
        print(f"  {filename}: {error}")
