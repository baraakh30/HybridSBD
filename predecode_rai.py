"""
Pre-decode RAI Dataset for Fast Testing

Videos  : 1.mp4  ...  10.mp4  (in <rai_root>/videos/)
Annots  : scenes_1.txt ... scenes_10.txt  (in <rai_root>/)
Stride  : 64 frames  (50% overlap, identical to AutoShot / BBC cache)
Clip len: 128 frames
Frame sz: 48 x 64  (same as model training)

Annotation format (scenes_X.txt):
    start_frame end_frame
    e.g.:
    1 717
    718 1590

Each scene boundary = last frame of scene i (== scenes[i][1]).

Usage:
    python predecode_rai.py \
        --rai_root "C:/Users/baraa/Documents/RAIDataset" \
        --output_root d:/grad/data/rai_cache \
        --force
"""

import torch
import numpy as np
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from torchcodec.decoders import VideoDecoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    'frame_size': (48, 64),   # H, W — must match model input
    'clip_length': 128,
    'stride': 64,             # 50% overlap, same as AutoShot / BBC cache
}

# Explicit video map: (video_filename, video_id, annotation_filename)
VIDEO_MAP = [
    ('1.mp4',  'rai_01', 'scenes_1.txt'),
    ('2.mp4',  'rai_02', 'scenes_2.txt'),
    ('3.mp4',  'rai_03', 'scenes_3.txt'),
    ('4.mp4',  'rai_04', 'scenes_4.txt'),
    ('5.mp4',  'rai_05', 'scenes_5.txt'),
    ('6.mp4',  'rai_06', 'scenes_6.txt'),
    ('7.mp4',  'rai_07', 'scenes_7.txt'),
    ('8.mp4',  'rai_08', 'scenes_8.txt'),
    ('9.mp4',  'rai_09', 'scenes_9.txt'),
    ('10.mp4', 'rai_10', 'scenes_10.txt'),
]


# ---------------------------------------------------------------------------
# Annotation reader
# ---------------------------------------------------------------------------
def read_scenes(annotation_file):
    """Read 'start_frame end_frame' annotation file. Returns sorted list of (start,end)."""
    scenes = []
    with open(annotation_file, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                scenes.append((int(parts[0]), int(parts[1])))
    scenes.sort()
    return scenes


def scenes_to_boundary_labels(scenes, start_frame, clip_length):
    """Binary boundary labels: 1 at each scene-transition frame within the clip window."""
    labels = torch.zeros(clip_length, dtype=torch.float32)
    for i in range(len(scenes) - 1):
        transition = scenes[i][1]          # last frame of scene i = boundary
        local = transition - start_frame
        if 0 <= local < clip_length:
            labels[local] = 1.0
    return labels


# ---------------------------------------------------------------------------
# Process one video
# ---------------------------------------------------------------------------
def process_video_worker(args):
    """Multiprocessing entry point."""
    return process_video(*args)


def process_video(video_path, annotation_file, video_id, output_dir, force=False):
    vid_dir    = output_dir / video_id
    metadata_f = vid_dir / 'metadata.json'

    # Skip if already done (unless --force)
    if not force and metadata_f.exists():
        meta = json.load(open(metadata_f))
        if all((vid_dir / f['file']).exists() for f in meta['clips']):
            print(f'  SKIP {video_id}  ({meta["num_clips"]} clips already cached)')
            return meta['num_clips']

    vid_dir.mkdir(parents=True, exist_ok=True)

    scenes      = read_scenes(annotation_file)
    transitions = [s[1] for s in scenes[:-1]]   # boundary frames (end of each scene except last)

    decoder    = VideoDecoder(str(video_path))
    total      = len(decoder)
    H, W       = CONFIG['frame_size']
    CL         = CONFIG['clip_length']
    stride     = CONFIG['stride']

    clips_meta = []
    clip_idx   = 0

    pbar = tqdm(range(0, total, stride), desc=f'  {video_id}', leave=False)
    for start in pbar:
        end    = min(start + CL, total)
        actual = end - start
        idxs   = list(range(start, end))

        raw = decoder.get_frames_at(idxs).data   # [T, 3, H_orig, W_orig]

        # Resize to target frame size
        if raw.shape[2] != H or raw.shape[3] != W:
            raw = torch.nn.functional.interpolate(
                raw.float(), size=(H, W), mode='bilinear', align_corners=False
            ).to(torch.uint8)

        # Pad last clip if shorter than CL
        if actual < CL:
            pad = raw[-1:].repeat(CL - actual, 1, 1, 1)
            raw = torch.cat([raw, pad], dim=0)

        boundary_labels  = scenes_to_boundary_labels(scenes, start, CL)
        clip_transitions = [t - start for t in transitions if start <= t < end]

        clip_file = vid_dir / f'clip_{clip_idx:06d}.pt'
        torch.save({
            'frames':           raw,              # [128, 3, 48, 64] uint8
            'boundary_labels':  boundary_labels,  # [128] float32
            'transitions':      clip_transitions,
            'start_frame':      start,
            'end_frame':        end,
            'actual_length':    actual,
        }, clip_file)

        clips_meta.append({
            'clip_idx':        clip_idx,
            'file':            clip_file.name,
            'start_frame':     start,
            'end_frame':       end,
            'num_transitions': len(clip_transitions),
        })
        clip_idx += 1

    del decoder

    meta = {
        'video_id':        video_id,
        'video_path':      str(video_path),
        'total_frames':    total,
        'num_scenes':      len(scenes),
        'num_transitions': len(transitions),
        'num_clips':       len(clips_meta),
        'stride':          stride,
        'clips':           clips_meta,
    }
    json.dump(meta, open(metadata_f, 'w'), indent=2)
    print(f'  DONE  {video_id}  ({len(clips_meta)} clips, {len(transitions)} transitions)')
    return len(clips_meta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rai_root',    default=r'C:\Users\baraa\Documents\RAIDataset',
                        help='Directory containing videos/ and scenes_X.txt files')
    parser.add_argument('--output_root', default=r'D:\grad\data\rai_cache')
    parser.add_argument('--force',       action='store_true',
                        help='Re-process even if cache already exists')
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()

    rai_root   = Path(args.rai_root)
    output_dir = Path(args.output_root)
    videos_dir = rai_root / 'videos'

    print('\n' + '='*80)
    print('PRE-DECODING RAI DATASET')
    print('='*80)
    print(f'Source     : {rai_root}')
    print(f'Output     : {output_dir}')
    print(f'Clip length: {CONFIG["clip_length"]}  Stride: {CONFIG["stride"]}  '
          f'Frame size: {CONFIG["frame_size"]}')
    if args.force:
        print('--force: re-processing all videos')
    print('='*80)

    if args.force and output_dir.exists():
        print(f'\nDeleting old cache at {output_dir}...')
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build work items, skip missing files
    work = []
    for (vid_file, video_id, ann_file) in VIDEO_MAP:
        video_path = videos_dir / vid_file
        ann_path   = rai_root   / ann_file
        if not video_path.exists():
            print(f'  MISSING video: {video_path}')
            continue
        if not ann_path.exists():
            print(f'  MISSING annotation: {ann_path}')
            continue
        work.append((video_path, ann_path, video_id, output_dir, args.force))

    print(f'\nProcessing {len(work)} videos with {args.num_workers} workers...\n')
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(process_video_worker, work)
    total_clips = sum(results)

    # Build global index_mapping.json
    print('\nBuilding index_mapping.json ...')
    all_clips = []
    for (_, video_id, _) in VIDEO_MAP:
        vid_dir   = output_dir / video_id
        meta_file = vid_dir / 'metadata.json'
        if not meta_file.exists():
            continue
        meta = json.load(open(meta_file))
        for c in meta['clips']:
            all_clips.append({
                'video_id':    video_id,
                'clip_file':   str(vid_dir / c['file']),
                'start_frame': c['start_frame'],
                'end_frame':   c['end_frame'],
            })

    mapping = {
        'dataset':    'RAI Dataset',
        'num_videos': sum(1 for (_, vid, _) in VIDEO_MAP
                         if (output_dir / vid / 'metadata.json').exists()),
        'num_clips':  len(all_clips),
        'config':     CONFIG,
        'mapping':    all_clips,
    }
    json.dump(mapping, open(output_dir / 'index_mapping.json', 'w'), indent=2)

    size_gb = sum(f.stat().st_size for f in output_dir.rglob('*.pt')) / 1024**3
    print(f'\nDone.  {len(all_clips)} clips  ({size_gb:.1f} GB)')
    print('='*80)


if __name__ == '__main__':
    main()
