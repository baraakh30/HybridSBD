"""
Pre-decode BBC Planet Earth for Fast Training / Testing

Videos  : bbc_01.mp4  ...  bbc_11.mp4
Stride  : 64 frames  (50% overlap, same as AutoShot)
Clip len: 128 frames
Frame sz: 48 x 64  (same as AutoShot training)

Episode mapping:
  bbc_01 -> 01_From_Pole_to_Pole
  bbc_02 -> 02_Mountains
  bbc_03 -> 03_Ice_Worlds
  bbc_04 -> 04_Great_Plains
  bbc_05 -> 05_Jungles
  bbc_06 -> 06_Seasonal_Forests
  bbc_07 -> 07_Fresh_Water
  bbc_08 -> 08_Ocean_Deep
  bbc_09 -> 09_Shallow_Seas
  bbc_10 -> 10_Caves
  bbc_11 -> 11_Deserts

Usage:
    python predecode_bbc.py \
        --bbc_root "C:/Users/baraa/Documents/bbc" \
        --output_root d:/grad/data/bbc_cache \
        --force
"""

import torch
import numpy as np
import json
import cv2
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
    'stride': 64,             # 50% overlap, same as AutoShot cache
}

# Explicit episode mapping: (video_filename, video_id, annotation_filename)
# annotation_filename must match the file in bbc_root/annotations/shots/
VIDEO_MAP = [
    ('bbc_01.mp4', '01_From_Pole_to_Pole',  '01_From_Pole_to_Pole.txt'),
    ('bbc_02.mp4', '02_Mountains',           '02_Mountains.txt'),
    ('bbc_03.mp4', '03_Ice_Worlds',          '03_Ice Worlds.txt'),
    ('bbc_04.mp4', '04_Great_Plains',        '04_Great Plains.txt'),
    ('bbc_05.mp4', '05_Jungles',             '05_Jungles.txt'),
    ('bbc_06.mp4', '06_Seasonal_Forests',    '06_Seasonal_Forests.txt'),
    ('bbc_07.mp4', '07_Fresh_Water',         '07_Fresh_Water.txt'),
    ('bbc_08.mp4', '08_Ocean_Deep',          '08_Ocean_Deep.txt'),
    ('bbc_09.mp4', '09_Shallow_Seas',        '09_Shallow_Seas.txt'),
    ('bbc_10.mp4', '10_Caves',               '10_Caves.txt'),
    ('bbc_11.mp4', '11_Deserts',             '11_Deserts.txt'),
]


# ---------------------------------------------------------------------------
# Annotation reader
# ---------------------------------------------------------------------------
def read_shots(annotation_file):
    """Read 'start_frame<TAB>end_frame' annotation file. Returns sorted list of (start,end)."""
    shots = []
    with open(annotation_file, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                shots.append((int(parts[0]), int(parts[1])))
    shots.sort()
    return shots


def shots_to_boundary_labels(shots, start_frame, clip_length):
    """Binary boundary labels: 1 at each shot-transition frame that falls in [start, start+clip_length)."""
    labels = torch.zeros(clip_length, dtype=torch.float32)
    for i in range(len(shots) - 1):
        transition = shots[i][1]          # last frame of shot i = boundary
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
    vid_dir      = output_dir / video_id
    metadata_f   = vid_dir / 'metadata.json'

    # Skip if already done (unless --force)
    if not force and metadata_f.exists():
        meta = json.load(open(metadata_f))
        if all((vid_dir / f['file']).exists() for f in meta['clips']):
            print(f'  SKIP {video_id}  ({meta["num_clips"]} clips already cached)')
            return meta['num_clips']

    vid_dir.mkdir(parents=True, exist_ok=True)

    shots       = read_shots(annotation_file)
    transitions = [s[1] for s in shots[:-1]]   # boundary frames

    decoder     = VideoDecoder(str(video_path))
    total       = len(decoder)
    H, W        = CONFIG['frame_size']
    CL          = CONFIG['clip_length']
    stride      = CONFIG['stride']
    frame_size  = CONFIG['frame_size']

    clips_meta = []
    clip_idx   = 0

    pbar = tqdm(range(0, total, stride), desc=f'  {video_id}', leave=False)
    for start in pbar:
        end    = min(start + CL, total)
        actual = end - start
        idxs   = list(range(start, end))

        raw    = decoder.get_frames_at(idxs).data   # [T, 3, H_orig, W_orig]

        # Resize to target frame size
        if raw.shape[2] != H or raw.shape[3] != W:
            raw = torch.nn.functional.interpolate(
                raw.float(), size=(H, W), mode='bilinear', align_corners=False
            ).to(torch.uint8)

        # Pad last clip if shorter than CL
        if actual < CL:
            pad = raw[-1:].repeat(CL - actual, 1, 1, 1)
            raw = torch.cat([raw, pad], dim=0)

        boundary_labels = shots_to_boundary_labels(shots, start, CL)
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
            'clip_idx':       clip_idx,
            'file':           clip_file.name,
            'start_frame':    start,
            'end_frame':      end,
            'num_transitions': len(clip_transitions),
        })
        clip_idx += 1

    del decoder

    meta = {
        'video_id':       video_id,
        'video_path':     str(video_path),
        'total_frames':   total,
        'num_shots':      len(shots),
        'num_transitions': len(transitions),
        'num_clips':      len(clips_meta),
        'stride':         stride,
        'clips':          clips_meta,
    }
    json.dump(meta, open(metadata_f, 'w'), indent=2)
    print(f'  DONE  {video_id}  ({len(clips_meta)} clips, {len(transitions)} transitions)')
    return len(clips_meta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbc_root',    required=True,
                        help='Directory containing bbc_01.mp4 ... bbc_11.mp4')
    parser.add_argument('--output_root', default='D:/grad/data/bbc_cache')
    parser.add_argument('--force',       action='store_true',
                        help='Re-process even if cache already exists')
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()

    bbc_root   = Path(args.bbc_root)
    output_dir = Path(args.output_root)
    ann_dir    = bbc_root / 'annotations' / 'shots'

    print('\n' + '='*80)
    print('PRE-DECODING BBC PLANET EARTH')
    print('='*80)
    print(f'Source     : {bbc_root}')
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

    total_clips  = 0
    all_clips    = []

    # Build work items, skip missing files
    work = []
    for (vid_file, video_id, ann_file) in VIDEO_MAP:
        video_path = bbc_root / vid_file
        ann_path   = ann_dir  / ann_file
        if not video_path.exists():
            print(f'  MISSING video: {video_path}')
            continue
        if not ann_path.exists():
            print(f'  MISSING annotation: {ann_path}')
            continue
        work.append((video_path, ann_path, video_id, output_dir, args.force))

    print(f'\nProcessing {len(work)} videos with {args.num_workers} workers...')
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(process_video_worker, work)
    total_clips = sum(results)

    # Build global index_mapping.json
    print('\nBuilding index_mapping.json ...')
    for (_, video_id, _) in VIDEO_MAP:
        vid_dir    = output_dir / video_id
        meta_file  = vid_dir / 'metadata.json'
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
        'dataset':    'BBC Planet Earth',
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