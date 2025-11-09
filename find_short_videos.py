import numpy as np
import json
import os

# Load ground truth
gt = json.load(open('C:/Users/Abdalrahman Khanfar/Documents/ClipShots/annotations/train.json'))

# Get videos with frame counts and boundaries
# Extract boundary frame (second element of each transition: [start, end])
videos = []
for k, v in gt.items():
    if 'frame_num' in v and 'transitions' in v:
        video_name = k.replace('.mp4', '')
        frame_count = v['frame_num']
        transitions = v['transitions']
        # Extract the boundary frames (second element of transition pairs)
        boundaries = [transition[1] for transition in transitions]
        
        # Only include videos with at least 4 boundaries
        if len(boundaries) >= 4:
            videos.append((video_name, frame_count, transitions, boundaries))

print(f'Total videos in ground truth with boundaries: {len(videos)}')

# Sort by frame count
videos_sorted = sorted(videos, key=lambda x: x[1])

# Check which have all required features (cnn, hsv, clip, optical_flow)
feat_dir = 'training_features'
existing = []
for v in videos_sorted:
    video_name = v[0]
    frame_count = v[1]
    
    # Check if all 4 feature types exist
    has_all_features = all([
        os.path.exists(os.path.join(feat_dir, f'{video_name}_cnn.npy')),
        os.path.exists(os.path.join(feat_dir, f'{video_name}_hsv_histogram.npy')),
        os.path.exists(os.path.join(feat_dir, f'{video_name}_clip.npy')),
        os.path.exists(os.path.join(feat_dir, f'{video_name}_optical_flow.npy'))
    ])
    
    if has_all_features:
        existing.append(v)

print(f'Videos with all features (cnn, hsv, clip, optical_flow): {len(existing)}')

# Filter for videos under 300 frames (10 seconds at 30fps)
print(f'\nLooking for videos under 300 frames (10 seconds at 30fps):')
existing_under_300 = [v for v in existing if v[1] <= 300]
print(f'Videos under 300 frames with boundaries and features: {len(existing_under_300)}')

# If not enough, expand to 600 frames
if len(existing_under_300) < 10:
    print(f'\nNot enough short videos, expanding to videos under 600 frames (20 seconds):')
    existing_target = [v for v in existing if v[1] <= 600][:10]
else:
    existing_target = existing_under_300[:10]

print(f'\nSelected {len(existing_target)} videos:')
for i, (video_name, frame_count, transitions, boundaries) in enumerate(existing_target, 1):
    num_boundaries = len(boundaries)
    print(f'{i}. {video_name}: {int(frame_count)} frames, {num_boundaries} boundaries')
    print(f'   Boundary frames: {boundaries[:5]}{"..." if len(boundaries) > 5 else ""}')

# Save to a file for the main script
selected = [(v[0], int(v[1]), v[2], v[3]) for v in existing_target]
with open('selected_videos.json', 'w') as f:
    json.dump(selected, f, indent=2)

print(f'\nSaved selected videos to selected_videos.json')
