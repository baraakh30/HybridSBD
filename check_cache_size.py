import json
from pathlib import Path
from collections import Counter

# Check ClipShots cache
clipshots_cache = Path(r'D:\grad\data\training_cache\train')
mapping_file = clipshots_cache / 'index_mapping.json'

if mapping_file.exists():
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    
    print('ClipShots Training Cache:')
    print(f'  Videos: {data.get("num_videos", "N/A")}')
    print(f'  Total clips: {data.get("num_clips", len(data.get("mapping", [])))}')
    
    # Check samples per video
    if 'mapping' in data:
        video_counts = Counter(clip['video_name'] for clip in data['mapping'])
        avg_samples = sum(video_counts.values()) / len(video_counts)
        max_samples = max(video_counts.values())
        min_samples = min(video_counts.values())
        
        print(f'  Samples per video:')
        print(f'    Average: {avg_samples:.1f}')
        print(f'    Min: {min_samples}')
        print(f'    Max: {max_samples}')
else:
    print(f'ClipShots cache not found at {mapping_file}')
