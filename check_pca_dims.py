"""
Quick script to check PCA feature dimensions
"""
import numpy as np
import os
from pathlib import Path

features_dir = "training_features"

# Get all PCA files sorted by size
pca_files = []
for f in os.listdir(features_dir):
    if f.endswith('_pca.npy'):
        filepath = os.path.join(features_dir, f)
        size = os.path.getsize(filepath)
        pca_files.append((f, filepath, size))

# Sort by size descending
pca_files.sort(key=lambda x: x[2], reverse=True)

print("Checking PCA feature dimensions...\n")
print("Top 10 largest PCA files:")
print("-" * 80)

for i, (filename, filepath, size) in enumerate(pca_files[:10]):
    size_mb = size / 1024 / 1024
    try:
        # Use mmap to avoid loading full file
        arr = np.load(filepath, mmap_mode='r')
        print(f"{i+1}. {filename}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Shape: {arr.shape}")
        print(f"   Frames: {arr.shape[0]}, Features: {arr.shape[1]}")
        print()
    except Exception as e:
        print(f"{i+1}. {filename}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   ERROR: {e}")
        print()

print("\nBottom 5 smallest PCA files:")
print("-" * 80)

for i, (filename, filepath, size) in enumerate(pca_files[-5:]):
    size_mb = size / 1024 / 1024
    try:
        arr = np.load(filepath, mmap_mode='r')
        print(f"{i+1}. {filename}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Shape: {arr.shape}")
        print()
    except Exception as e:
        print(f"{i+1}. {filename}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   ERROR: {e}")
        print()

# Check a random sample
print("\n10 random samples:")
print("-" * 80)
import random
random_sample = random.sample(pca_files[50:100], min(10, len(pca_files)-50))
for filename, filepath, size in random_sample:
    size_mb = size / 1024 / 1024
    try:
        arr = np.load(filepath, mmap_mode='r')
        print(f"{filename}: {arr.shape[1]} features, {arr.shape[0]} frames ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"{filename}: ERROR - {e}")
