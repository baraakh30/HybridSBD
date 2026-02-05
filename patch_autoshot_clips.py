"""
Patch existing AutoShot predecoded clips to add boundary_labels

Adds binary boundary labels to already-decoded clips without re-decoding videos.
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Configuration
CACHE_ROOT = Path(r"d:\grad\data\autoshot_cache")
SEQ_LEN = 128

def patch_clip(clip_path):
    """Add boundary_labels to an existing clip file"""
    try:
        # Load existing clip
        clip_data = torch.load(clip_path, weights_only=False)
        
        # Check if already has boundary_labels
        if 'boundary_labels' in clip_data:
            return 'skip'
        
        # Get transitions
        transitions = clip_data['transitions']  # List of transition frame indices
        
        # Generate boundary labels (1 at transition frames, 0 elsewhere)
        boundary_labels = np.zeros(SEQ_LEN, dtype=np.float32)
        for trans_idx in transitions:
            if 0 <= trans_idx < SEQ_LEN:
                boundary_labels[trans_idx] = 1.0
        
        # Add to clip data
        clip_data['boundary_labels'] = torch.from_numpy(boundary_labels)
        
        # Save back
        torch.save(clip_data, clip_path)
        
        return 'success'
        
    except Exception as e:
        print(f"\nError patching {clip_path}: {e}")
        return 'error'


def main():
    print("\n" + "="*80)
    print("PATCHING AUTOSHOT CLIPS - ADDING BOUNDARY LABELS")
    print("="*80)
    
    # Load index mapping to find all clips
    index_file = CACHE_ROOT / 'index_mapping.json'
    if not index_file.exists():
        print(f"Error: Index file not found: {index_file}")
        return
    
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    mapping = index_data['mapping']
    print(f"Found {len(mapping)} clips to patch")
    
    # Patch all clips
    results = {'success': 0, 'skip': 0, 'error': 0}
    
    for clip_info in tqdm(mapping, desc="Patching clips"):
        clip_path = Path(clip_info['clip_file'])
        result = patch_clip(clip_path)
        results[result] += 1
    
    # Summary
    print("\n" + "="*80)
    print("PATCHING COMPLETE!")
    print("="*80)
    print(f"Successfully patched: {results['success']}")
    print(f"Already had labels:   {results['skip']}")
    print(f"Errors:              {results['error']}")
    print("="*80)
    
    if results['success'] > 0:
        print("\n✓ Clips updated! You can now run testing:")
        print("  python test_autoshot.py --checkpoint outputs/hybrid_detector/best_model.pt")


if __name__ == '__main__':
    main()
