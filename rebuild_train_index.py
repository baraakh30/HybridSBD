"""
Rebuild train index_mapping.json to include BOTH original train + only_gradual clips
"""
import json
from pathlib import Path
from tqdm import tqdm

output_dir = Path(r"D:\grad\data\training_cache\train")

print("Scanning all video directories in train folder...")
video_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
print(f"Found {len(video_dirs)} video directories")

index_mapping = []
clip_idx = 0
total_clips = 0

for vid_dir in tqdm(video_dirs, desc="Building index"):
    metadata_file = vid_dir / 'metadata.json'
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            vid_meta = json.load(f)
        
        vid_name = vid_meta['video_name']
        
        for clip_meta in vid_meta['clips']:
            index_mapping.append({
                'index': clip_idx,
                'video_name': vid_name,
                'clip_file': str(vid_dir / clip_meta['file']),
                'start_frame': clip_meta['start_frame']
            })
            clip_idx += 1
        
        total_clips += len(vid_meta['clips'])

# Save combined mapping
mapping_file = output_dir / 'index_mapping.json'
print(f"\nSaving combined index to {mapping_file}...")

with open(mapping_file, 'w') as f:
    json.dump({
        'split': 'train',
        'num_videos': len(video_dirs),
        'num_clips': len(index_mapping),
        'note': 'Combined original train + only_gradual clips for balanced training',
        'mapping': index_mapping
    }, f, indent=2)

print("\n" + "="*80)
print("INDEX REBUILT!")
print("="*80)
print(f"Total videos:  {len(video_dirs)} (original train + only_gradual)")
print(f"Total clips:   {total_clips}")
print(f"Index file:    {mapping_file}")
print("="*80)
