"""
Test Hybrid Boundary Detector on BBC Planet Earth Dataset

Evaluates the model on pre-decoded BBC clips using full-video sliding-window
reconstruction, identical methodology to test_autoshot.py.

Model output:
  y_logits  [B, T, 1]  one_hot midpoint predictions
  z_logits  [B, T, 1]  many_hot gradual extent

Usage:
    python test_bbc.py \
        --checkpoint outputs/hybrid_detector13/best_model.pt \
        --cache_root d:/grad/data/bbc_cache \
        --bbc_root "C:/Users/baraa/Documents/bbc" \
        --sweep
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.hybrid_boundary_detector import create_hybrid_detector
from load_ablation_model import load_detector


def collate_bbc_batch(batch):
    frames       = torch.stack([item['frames']       for item in batch])
    video_ids    = [item['video_id']    for item in batch]
    start_frames = torch.tensor([item['start_frame'] for item in batch])
    clip_indices = torch.tensor([item['clip_index']  for item in batch])
    return {
        'frames':      frames,
        'video_id':    video_ids,
        'start_frame': start_frames,
        'clip_index':  clip_indices,
    }


class BBCPredecodedDataset(Dataset):
    """
    Dataset for pre-decoded BBC Planet Earth clips.
    The cache is built with stride=64, so every clip is loaded directly.
    """

    def __init__(self, cache_root):
        self.cache_root = Path(cache_root)

        index_file = self.cache_root / 'index_mapping.json'
        with open(index_file, 'r') as f:
            index_data = json.load(f)

        self.entries    = index_data['mapping']
        self.num_videos = index_data['num_videos']
        self.entries.sort(key=lambda x: (x['video_id'], x['start_frame']))

        print(f"BBC Dataset: {self.num_videos} videos, {len(self.entries)} clips (stride=64)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e      = self.entries[idx]
        data   = torch.load(e['clip_file'], weights_only=False)
        frames = data['frames'].float() / 255.0
        return {
            'frames':      frames,
            'video_id':    e['video_id'],
            'start_frame': e['start_frame'],
            'clip_index':  idx,
        }


def read_bbc_shot_annotations(annotation_file):
    """
    Read BBC shot annotations from a txt file.

    Format: each line is  "start_frame end_frame"

    Returns:
        np.ndarray of shape (N, 2) with [start, end] per shot
    """
    shots = []
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                shots.append([int(parts[0]), int(parts[1])])
    shots.sort(key=lambda x: x[0])
    return np.array(shots, dtype=np.int32)


def predictions_to_scenes(predictions):
    """
    Convert binary many-hot boundary predictions to scene segments.
    Adapted from AutoShot evaluation code.

    Args:
        predictions: [T] binary array (1 = inside a boundary/transition)

    Returns:
        scenes: Nx2 array of [start, end] frame indices for each scene
    """
    scenes = []
    t_prev, start = 0, 0

    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i - 1])
        t_prev = t

    if t_prev == 0:
        scenes.append([start, len(predictions) - 1])

    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(gt_scenes, pred_scenes, n_frames_miss_tolerance=2):
    """
    Evaluate scene detection with frame tolerance.
    Adapted from AutoShot evaluation.

    Returns:
        precision, recall, f1, (tp, fp, fn)
    """
    shift       = n_frames_miss_tolerance / 2
    gt_scenes   = gt_scenes.astype(np.float32)   + np.array([[-0.5 + shift,  0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift,  0.5 - shift]])

    gt_trans   = np.stack([gt_scenes[:-1, 1],   gt_scenes[1:, 0]],   1) if len(gt_scenes)   > 1 else np.zeros((0, 2))
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1) if len(pred_scenes) > 1 else np.zeros((0, 2))

    i = j = tp = fp = fn = 0
    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1; i += 1
        elif i == len(gt_trans):
            fp += 1; j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1; j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1; i += 1
        else:
            tp += 1; i += 1; j += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, (tp, fp, fn)


def _build_pred_many_hot(y_pred, z_pred, boundary_threshold, gradual_threshold):
    """Combine y (one_hot midpoint) and z (many_hot extent) → single many_hot array."""
    pred_is_boundary = y_pred > boundary_threshold
    pred_is_gradual  = z_pred > gradual_threshold
    pred_is_cut      = pred_is_boundary & ~pred_is_gradual
    return (pred_is_gradual | pred_is_cut).astype(np.uint8)


@torch.no_grad()
def run_inference(model, loader, device):
    """Run forward pass over all clips; returns per-clip probabilities keyed by video_id."""
    model.eval()
    video_clips = defaultdict(list)

    for batch in tqdm(loader, desc="Inference"):
        frames   = batch['frames'].to(device, non_blocking=True)   # [B, T, 3, H, W]
        y_logits, z_logits = model(frames)                          # [B, T, 1] each
        y_probs  = torch.sigmoid(y_logits.squeeze(-1)).cpu().numpy()  # [B, T]
        z_probs  = torch.sigmoid(z_logits.squeeze(-1)).cpu().numpy()  # [B, T]

        for b in range(frames.shape[0]):
            video_id    = batch['video_id'][b]
            start_frame = batch['start_frame'][b].item()
            video_clips[video_id].append((start_frame, y_probs[b], z_probs[b]))

    return video_clips


def reconstruct_full_video(video_clips):
    """Average overlapping clip predictions into full-video probability arrays."""
    video_y = {}
    video_z = {}
    for video_id, clips in video_clips.items():
        clips   = sorted(clips, key=lambda x: x[0])
        T_total = max(s + len(y) for s, y, _ in clips)
        full_y  = np.zeros(T_total, dtype=np.float32)
        full_z  = np.zeros(T_total, dtype=np.float32)
        counts  = np.zeros(T_total, dtype=np.int32)
        for start, y, z in clips:
            end = start + len(y)
            full_y[start:end] += y
            full_z[start:end] += z
            counts[start:end] += 1
        mask = counts > 0
        full_y[mask] /= counts[mask]
        full_z[mask] /= counts[mask]
        video_y[video_id] = full_y
        video_z[video_id] = full_z
    return video_y, video_z


@torch.no_grad()
def evaluate(model, loader, device, gt_scenes_dict, config):
    """Full-video evaluation with threshold sweep."""
    video_clips = run_inference(model, loader, device)

    print("\nReconstructing full-video predictions...")
    video_y, video_z = reconstruct_full_video(video_clips)

    boundary_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                           0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    gradual_thresholds  = [0.3, 0.4, 0.5, 0.6]

    print(f"\nSweeping {len(boundary_thresholds)} x {len(gradual_thresholds)} thresholds...")
    best_f1, best_bt, best_gt = 0.0, 0.5, 0.5
    all_results = {}

    for bt in tqdm(boundary_thresholds, desc="Sweep"):
        for gt in gradual_thresholds:
            tp = fp = fn = 0
            for vid in video_y:
                if vid not in gt_scenes_dict:
                    continue
                gt_sc   = gt_scenes_dict[vid]
                # Clip predictions to the annotated frame range to avoid
                # counting predictions in the unannotated video tail as FPs.
                max_annot_frame = int(gt_sc[-1, 1]) + 1
                y_clipped = video_y[vid][:max_annot_frame]
                z_clipped = video_z[vid][:max_annot_frame]
                pred_mh = _build_pred_many_hot(y_clipped, z_clipped, bt, gt)
                pred_sc = predictions_to_scenes(pred_mh)
                _, _, _, (t, f, n) = evaluate_scenes(gt_sc, pred_sc,
                                                      n_frames_miss_tolerance=config['tolerance'])
                tp += t; fp += f; fn += n
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            all_results[(bt, gt)] = dict(p=p, r=r, f1=f1, tp=tp, fp=fp, fn=fn)
            if f1 > best_f1:
                best_f1, best_bt, best_gt = f1, bt, gt

    # Per-video results at best thresholds
    video_results = {}
    for vid in video_y:
        if vid not in gt_scenes_dict:
            continue
        gt_sc   = gt_scenes_dict[vid]
        max_annot_frame = int(gt_sc[-1, 1]) + 1
        y_clipped = video_y[vid][:max_annot_frame]
        z_clipped = video_z[vid][:max_annot_frame]
        pred_mh = _build_pred_many_hot(y_clipped, z_clipped, best_bt, best_gt)
        pred_sc = predictions_to_scenes(pred_mh)
        p, r, f1, (tp, fp, fn) = evaluate_scenes(gt_sc, pred_sc,
                                                   n_frames_miss_tolerance=config['tolerance'])
        video_results[vid] = dict(p=p, r=r, f1=f1, tp=tp, fp=fp, fn=fn,
                                  n_gt=len(gt_sc), n_pred=len(pred_sc))

    best = all_results[(best_bt, best_gt)]
    return {
        'best_boundary_threshold': best_bt,
        'best_gradual_threshold':  best_gt,
        'overall':     best,
        'all_results': {str(k): v for k, v in all_results.items()},
        'video_results': video_results,
        'num_videos':  len(video_results),
    }


def _normalize_episode_name(name: str) -> str:
    """Strip numeric prefix and normalise to lowercase with no separators."""
    # Remove leading digits (and optional separator) e.g. '03_' or '07_'
    import re
    name = re.sub(r'^\d+[_\s]*', '', name)
    # lower-case, drop spaces / underscores / hyphens
    return re.sub(r'[\s_\-]+', '', name).lower()


def load_gt_scenes(bbc_root, cache_root):
    """
    Load ground-truth scenes for all BBC videos.

    Maps video_id (e.g. '01_From_Pole_to_Pole') → np.ndarray shape (N,2)
    of [start_frame, end_frame] shot segments.

    Matching is done by *episode name* (ignoring the numeric prefix) because
    the cache was predecoded in filesystem order which differs from the
    annotation file numbering.
    """
    bbc_root   = Path(bbc_root)
    ann_dir    = bbc_root / 'annotations' / 'shots'
    cache_root = Path(cache_root)

    index_file = cache_root / 'index_mapping.json'
    with open(index_file, 'r') as f:
        index_data = json.load(f)

    # unique video_ids from cache
    seen_vids = []
    for clip in index_data['mapping']:
        vid = clip['video_id']
        if vid not in seen_vids:
            seen_vids.append(vid)

    # Build normalised-name -> annotation-file map
    ann_by_norm = {}
    for ann_file in ann_dir.glob('*.txt'):
        norm = _normalize_episode_name(ann_file.stem)
        ann_by_norm[norm] = ann_file

    gt_scenes_dict = {}
    for vid in seen_vids:
        norm = _normalize_episode_name(vid)
        if norm not in ann_by_norm:
            print(f"  Warning: no annotation file for '{vid}' (normalised='{norm}')")
            print(f"  Available: {sorted(ann_by_norm.keys())}")
            continue
        matched = ann_by_norm[norm]
        scenes  = read_bbc_shot_annotations(matched)
        print(f"  {vid} -> {matched.name}  ({len(scenes)} shots)")
        if len(scenes) > 0:
            gt_scenes_dict[vid] = scenes

    return gt_scenes_dict


def main():
    parser = argparse.ArgumentParser(description='Test Hybrid Boundary Detector on BBC Planet Earth')
    parser.add_argument('--checkpoint',  type=str,   required=True)
    parser.add_argument('--cache_root',  type=str,   default=r'd:\grad\data\bbc_cache')
    parser.add_argument('--bbc_root',    type=str,   required=True,
                        help='Root directory of BBC dataset (contains annotations/shots/)')
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--tolerance',   type=int,   default=2,
                        help='Frame tolerance for scene matching (default=2, same as AutoShot)')
    parser.add_argument('--output',      type=str,   default='bbc_results.json')
    args = parser.parse_args()

    config = vars(args)

    print("\n" + "=" * 80)
    print("TESTING ON BBC PLANET EARTH DATASET")
    print("=" * 80)
    print(f"Checkpoint : {config['checkpoint']}")
    print(f"Cache root : {config['cache_root']}")
    print(f"BBC root   : {config['bbc_root']}")
    print(f"Batch size : {config['batch_size']}")
    print(f"Tolerance  : ±{config['tolerance']} frames")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")

    # --- dataset ---
    dataset = BBCPredecodedDataset(config['cache_root'])
    loader  = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                         num_workers=4, pin_memory=True,
                         collate_fn=collate_bbc_batch)

    # --- model ---
    print("\nLoading model...")
    model, ckpt = load_detector(config['checkpoint'], device)
    print(f"✓ Loaded from epoch {ckpt.get('epoch', '?')}")

    # --- ground truth ---
    print(f"\nLoading ground truth from {config['bbc_root']}...")
    gt_scenes_dict = load_gt_scenes(config['bbc_root'], config['cache_root'])
    print(f"✓ {len(gt_scenes_dict)} videos with GT scenes")

    # --- evaluate ---
    results = evaluate(model, loader, device, gt_scenes_dict, config)

    # --- print ---
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Videos evaluated  : {results['num_videos']}")
    print(f"Best boundary thr : {results['best_boundary_threshold']:.2f}")
    print(f"Best gradual thr  : {results['best_gradual_threshold']:.2f}")
    ov = results['overall']
    print(f"\nOverall  P={ov['p']:.4f}  R={ov['r']:.4f}  F1={ov['f1']:.4f}")
    print(f"         TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}")

    video_f1s = sorted(results['video_results'].items(), key=lambda x: x[1]['f1'], reverse=True)
    print(f"\nTop 5 videos:")
    for vid, m in video_f1s[:5]:
        print(f"  {vid}: F1={m['f1']:.3f}  P={m['p']:.3f}  R={m['r']:.3f}")
    print(f"\nBottom 5 videos:")
    for vid, m in video_f1s[-5:]:
        print(f"  {vid}: F1={m['f1']:.3f}  P={m['p']:.3f}  R={m['r']:.3f}")
    print("=" * 80)

    with open(config['output'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {config['output']}")


if __name__ == '__main__':
    main()
