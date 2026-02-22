"""
Test Hybrid Boundary Detector on ClipShots Test Dataset

Evaluates the model on pre-decoded ClipShots test clips using full-video
sliding-window reconstruction (same methodology as test_autoshot.py).

Ground truth is loaded from ClipShots annotations JSON:
    annotations/test.json  ->  {video_name: {transitions: [[t_start, t_end], ...], frame_num: N}}

Transitions are converted to scenes for evaluate_scenes():
    Scene i = [prev_transition_end, next_transition_start]

Cache layout (from predecode_dataset.py):
    training_cache/test/
        <video_name_no_ext>/
            clip_000000.pt   # {frames, targets, lengths, start_frame, is_transition_focused}
            ...
        index_mapping.json

Usage:
    python test_clipshots.py \\
        --checkpoint outputs/hybrid_detector12/best_model.pt \\
        --cache_root d:/grad/data/training_cache/test \\
        --ann_path "C:/Users/baraa/Documents/ClipShots/annotations/test.json"
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.hybrid_boundary_detector import create_hybrid_detector

# ── shared evaluation engine from test_autoshot ───────────────────────────────
from test_autoshot import (
    predictions_to_scenes,
    evaluate_scenes,
    _build_pred_many_hot,
    run_inference,
    reconstruct_full_video,
    evaluate,
)


# ── ClipShots-specific dataset ────────────────────────────────────────────────

def collate_clipshots_batch(batch):
    frames       = torch.stack([item['frames']      for item in batch])
    video_ids    = [item['video_id']   for item in batch]
    start_frames = torch.tensor([item['start_frame'] for item in batch])
    clip_indices = torch.tensor([item['clip_index']  for item in batch])
    return {
        'frames':      frames,
        'video_id':    video_ids,
        'start_frame': start_frames,
        'clip_index':  clip_indices,
    }


class ClipShotsPredecodedDataset(Dataset):
    """
    Loads pre-decoded ClipShots test clips from training_cache/test/
    The index_mapping has 'video_name' (e.g. 'foo.mp4'); we strip the
    extension to match annotation keys OR keep as-is (both formats checked).
    """
    def __init__(self, cache_root):
        self.cache_root = Path(cache_root)
        index_file = self.cache_root / 'index_mapping.json'
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        self.mapping = index_data['mapping']
        print(f"ClipShots Test Dataset: {index_data['num_videos']} videos, "
              f"{len(self.mapping)} clips")

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        entry     = self.mapping[idx]
        clip_data = torch.load(entry['clip_file'], weights_only=False)
        frames    = clip_data['frames'].float() / 255.0   # [T, 3, H, W]
        # use video_name as the video_id key
        video_id  = entry['video_name']
        return {
            'frames':      frames,
            'video_id':    video_id,
            'start_frame': entry['start_frame'],
            'clip_index':  idx,
        }


# ── GT building ───────────────────────────────────────────────────────────────

def transitions_to_scenes(transitions, total_frames):
    """
    Convert a list of [t_start, t_end] transition pairs to Nx2 scenes array.

    Each transition defines the boundary between two shots:
        Scene i ends at t_start, Scene i+1 starts at t_end.
    So:
        scenes = [[0, t0_start], [t0_end, t1_start], ..., [tN_end, total_frames-1]]

    This gives gt_trans[i] = [scenes[i,1], scenes[i+1,0]] = [ti_start, ti_end],
    matching evaluate_scenes()'s expectation exactly.
    """
    if not transitions:
        return np.array([[0, max(0, int(total_frames) - 1)]], dtype=np.int32)

    sorted_trans = sorted(transitions, key=lambda x: x[0])
    scenes = []
    prev_end = 0
    for t_start, t_end in sorted_trans:
        scenes.append([prev_end, int(t_start)])
        prev_end = int(t_end)
    scenes.append([prev_end, max(prev_end, int(total_frames) - 1)])
    return np.array(scenes, dtype=np.int32)


def load_gt_scenes(ann_path, video_names):
    """
    Load ClipShots GT from annotations JSON and convert to dict of scenes arrays.

    Args:
        ann_path:    path to test.json
        video_names: iterable of video_names as they appear in the cache
                     (e.g. 'foo.mp4' or 'foo')

    Returns:
        gt_scenes_dict: {video_name: Nx2 np.array of scenes}
    """
    with open(ann_path, 'r') as f:
        ann = json.load(f)

    # Build lookup for both 'foo.mp4' and 'foo' keys
    ann_by_key = {}
    for k, v in ann.items():
        ann_by_key[k]                      = v   # original key
        ann_by_key[k.replace('.mp4', '')]  = v   # without extension
        ann_by_key[k + '.mp4']             = v   # with extension (if key had none)

    gt_scenes_dict = {}
    missing = 0
    for vname in video_names:
        entry = ann_by_key.get(vname)
        if entry is None:
            missing += 1
            continue
        gt_scenes_dict[vname] = transitions_to_scenes(
            entry['transitions'], entry['frame_num']
        )

    if missing:
        print(f"  ⚠  {missing} videos not found in annotations (will be skipped)")
    print(f"  ✓ {len(gt_scenes_dict)} videos with GT scenes loaded")
    return gt_scenes_dict


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Test Hybrid Boundary Detector on ClipShots Test Set')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--cache_root', type=str,
                        default=r'd:\grad\data\training_cache\test')
    parser.add_argument('--ann_path', type=str,
                        default=r'C:\Users\baraa\Documents\ClipShots\annotations\test.json')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--tolerance', type=int, default=2,
                        help='Frame tolerance for scene matching')
    parser.add_argument('--output', type=str, default='clipshots_results.json')
    args = parser.parse_args()

    config = vars(args)

    print("\n" + "="*80)
    print("TESTING ON CLIPSHOTS TEST DATASET")
    print("="*80)
    print(f"Checkpoint:  {config['checkpoint']}")
    print(f"Cache root:  {config['cache_root']}")
    print(f"Ann path:    {config['ann_path']}")
    print(f"Batch size:  {config['batch_size']}")
    print(f"Tolerance:   ±{config['tolerance']} frames")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:      {device}")

    # --- dataset ---
    dataset = ClipShotsPredecodedDataset(config['cache_root'])
    loader  = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                         num_workers=4, pin_memory=True,
                         collate_fn=collate_clipshots_batch)

    # --- model ---
    print("\nLoading model...")
    ckpt  = torch.load(config['checkpoint'], map_location=device, weights_only=False)
    mcfg  = ckpt.get('config', {})
    model = create_hybrid_detector(
        main_dim=mcfg.get('main_dim', 512),
        num_fusion_blocks=mcfg.get('num_fusion_blocks', 3),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    print(f"✓ Loaded from epoch {ckpt.get('epoch', '?')}")

    # --- ground truth ---
    print("\nLoading ground truth...")
    video_names = list({e['video_name'] for e in dataset.mapping})
    gt_scenes_dict = load_gt_scenes(config['ann_path'], video_names)

    # --- evaluate ---
    results = evaluate(model, loader, device, gt_scenes_dict, config)

    # --- print ---
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Videos evaluated : {results['num_videos']}")
    print(f"Best boundary thr: {results['best_boundary_threshold']:.2f}")
    print(f"Best gradual thr : {results['best_gradual_threshold']:.2f}")
    ov = results['overall']
    print(f"\nOverall  P={ov['p']:.4f}  R={ov['r']:.4f}  F1={ov['f1']:.4f}")
    print(f"         TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}")

    video_f1s = sorted(results['video_results'].items(),
                       key=lambda x: x[1]['f1'], reverse=True)
    print(f"\nTop 5 videos:")
    for vid, m in video_f1s[:5]:
        print(f"  {vid}: F1={m['f1']:.3f}  P={m['p']:.3f}  R={m['r']:.3f}  "
              f"gt_scenes={m['n_gt']}  pred_scenes={m['n_pred']}")
    print(f"\nBottom 5 videos:")
    for vid, m in video_f1s[-5:]:
        print(f"  {vid}: F1={m['f1']:.3f}  P={m['p']:.3f}  R={m['r']:.3f}  "
              f"gt_scenes={m['n_gt']}  pred_scenes={m['n_pred']}")
    print("="*80)

    with open(config['output'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {config['output']}")


if __name__ == '__main__':
    main()
