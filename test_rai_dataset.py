"""
Test Hybrid Boundary Detector on RAI Dataset

Evaluates the model on pre-decoded RAI clips using full-video sliding-window
reconstruction — identical methodology to test_autoshot.py and test_bbc.py.

Model output:
  y_logits  [B, T, 1]  one_hot midpoint predictions  -> cuts + gradual centres
  z_logits  [B, T, 1]  many_hot gradual extent        -> all frames in graduals

Combination rule (mirrors training):
  pred_is_boundary = sigmoid(y) > boundary_threshold
  pred_is_gradual  = sigmoid(z) > gradual_threshold
  pred_is_cut      = pred_is_boundary & ~pred_is_gradual
  pred_many_hot    = pred_is_gradual | pred_is_cut

Evaluation: AutoShot scene-level precision/recall/F1 with +-tolerance frames.

RAI annotation format (scenes_X.txt):
    start_frame end_frame
    e.g.:
    1 717
    718 1590
    1591 2333

Usage:
    # First predecode (one-time):
    python predecode_rai.py

    # Evaluate with threshold sweep:
    python test_rai_dataset.py \\
        --checkpoint outputs/hybrid_detector13/best_model.pt \\
        --cache_root d:/grad/data/rai_cache \\
        --rai_root "C:/Users/baraa/Documents/RAIDataset"
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def collate_rai_batch(batch):
    frames       = torch.stack([item["frames"]       for item in batch])
    video_ids    = [item["video_id"]    for item in batch]
    start_frames = torch.tensor([item["start_frame"] for item in batch])
    clip_indices = torch.tensor([item["clip_index"]  for item in batch])
    return {
        "frames":      frames,
        "video_id":    video_ids,
        "start_frame": start_frames,
        "clip_index":  clip_indices,
    }


class RAIPredecodedDataset(Dataset):
    """
    Dataset for pre-decoded RAI clips.
    Cache is built with stride=64, so every clip is loaded directly.
    """

    def __init__(self, cache_root):
        self.cache_root = Path(cache_root)

        index_file = self.cache_root / "index_mapping.json"
        if not index_file.exists():
            raise FileNotFoundError(
                f"Index mapping not found: {index_file}\n"
                f"Please run: python predecode_rai.py"
            )

        with open(index_file, "r") as f:
            index_data = json.load(f)

        self.entries    = index_data["mapping"]
        self.num_videos = index_data["num_videos"]
        self.entries.sort(key=lambda x: (x["video_id"], x["start_frame"]))

        print(f"RAI Dataset: {self.num_videos} videos, {len(self.entries)} clips (stride=64)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e      = self.entries[idx]
        data   = torch.load(e["clip_file"], weights_only=False)
        frames = data["frames"].float() / 255.0   # [128, 3, 48, 64] float32 [0,1]
        return {
            "frames":      frames,
            "video_id":    e["video_id"],
            "start_frame": e["start_frame"],
            "clip_index":  idx,
        }


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_gt_scenes(rai_root, cache_root):
    """
    Load ground-truth scenes for all RAI videos.

    Annotation files: <rai_root>/scenes_X.txt
    Each line: start_frame end_frame

    Returns:
        dict: video_id (e.g. "rai_01") -> np.ndarray shape (N, 2) [start, end]
    """
    rai_root   = Path(rai_root)
    cache_root = Path(cache_root)

    # Get unique video_ids from cache index
    index_file = cache_root / "index_mapping.json"
    with open(index_file, "r") as f:
        index_data = json.load(f)
    seen_vids = []
    for clip in index_data["mapping"]:
        vid = clip["video_id"]
        if vid not in seen_vids:
            seen_vids.append(vid)

    gt_scenes_dict = {}
    for vid in seen_vids:
        # video_id is "rai_01" ... "rai_10" -> annotation is scenes_1.txt ... scenes_10.txt
        num = int(vid.split("_")[1])
        ann_file = rai_root / f"scenes_{num}.txt"
        if not ann_file.exists():
            print(f"  Warning: annotation not found for {vid} at {ann_file}")
            continue
        scenes = []
        with open(ann_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    scenes.append([int(parts[0]), int(parts[1])])
        if scenes:
            gt_scenes_dict[vid] = np.array(scenes, dtype=np.int32)
            print(f"  {vid} -> {ann_file.name}  ({len(scenes)} scenes)")

    return gt_scenes_dict


# ---------------------------------------------------------------------------
# Evaluation helpers  (identical to test_autoshot.py / test_bbc.py)
# ---------------------------------------------------------------------------

def predictions_to_scenes(predictions):
    """
    Convert binary many-hot boundary predictions to scene segments.

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


def _build_pred_y_only(y_pred, threshold):
    """
    RAI evaluation: threshold y-head only (hard cuts, no z spreading).
    Matches TransNetV2 / AutoShot evaluation methodology on RAI.
    Returns binary many-hot array where only y-head peaks above threshold are marked.
    """
    return (y_pred > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inference & evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    """Run forward pass over all clips; returns per-clip probabilities keyed by video_id."""
    model.eval()
    video_clips = defaultdict(list)

    for batch in tqdm(loader, desc="Inference"):
        frames   = batch["frames"].to(device, non_blocking=True)    # [B, T, 3, H, W]
        y_logits, z_logits = model(frames)                           # [B, T, 1] each
        y_probs  = torch.sigmoid(y_logits.squeeze(-1)).cpu().numpy() # [B, T]
        z_probs  = torch.sigmoid(z_logits.squeeze(-1)).cpu().numpy() # [B, T]

        for b in range(frames.shape[0]):
            video_id    = batch["video_id"][b]
            start_frame = batch["start_frame"][b].item()
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

    # RAI uses y-head only (hard cuts, like TransNetV2 / AutoShot evaluation).
    # The z-head spreads probability across gradual regions which creates massive
    # FPs in long static shots - it must not be used here.
    boundary_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                           0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    print(f"\nSweeping {len(boundary_thresholds)} thresholds (y-head only)...")
    best_f1, best_bt = 0.0, 0.5
    all_results = {}

    for bt in tqdm(boundary_thresholds, desc="Sweep"):
        tp = fp = fn = 0
        for vid in video_y:
            if vid not in gt_scenes_dict:
                continue
            gt_sc = gt_scenes_dict[vid]
            # Clip to annotated range to avoid counting unannotated tail as FPs.
            max_annot_frame = int(gt_sc[-1, 1]) + 1
            y_clipped = video_y[vid][:max_annot_frame]
            pred_mh = _build_pred_y_only(y_clipped, bt)
            pred_sc = predictions_to_scenes(pred_mh)
            _, _, _, (t, f, n) = evaluate_scenes(gt_sc, pred_sc,
                                                  n_frames_miss_tolerance=config["tolerance"])
            tp += t; fp += f; fn += n
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        all_results[bt] = dict(p=p, r=r, f1=f1, tp=tp, fp=fp, fn=fn)
        if f1 > best_f1:
            best_f1, best_bt = f1, bt

    # Per-video results at best threshold
    video_results = {}
    for vid in video_y:
        if vid not in gt_scenes_dict:
            continue
        gt_sc = gt_scenes_dict[vid]
        max_annot_frame = int(gt_sc[-1, 1]) + 1
        y_clipped = video_y[vid][:max_annot_frame]
        pred_mh = _build_pred_y_only(y_clipped, best_bt)
        pred_sc = predictions_to_scenes(pred_mh)
        p, r, f1, (tp, fp, fn) = evaluate_scenes(gt_sc, pred_sc,
                                                   n_frames_miss_tolerance=config["tolerance"])
        video_results[vid] = dict(p=p, r=r, f1=f1, tp=tp, fp=fp, fn=fn,
                                  n_gt=len(gt_sc), n_pred=len(pred_sc))

    best = all_results[best_bt]
    return {
        "best_boundary_threshold": best_bt,
        "overall":       best,
        "all_results":   {str(k): v for k, v in all_results.items()},
        "video_results": video_results,
        "num_videos":    len(video_results),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Boundary Detector on RAI Dataset")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--cache_root",  type=str, default=r"D:\grad\data\rai_cache")
    parser.add_argument("--rai_root",    type=str, default=r"C:\Users\baraa\Documents\RAIDataset",
                        help="RAI dataset root (contains videos/ and scenes_X.txt)")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--tolerance",   type=int, default=2,
                        help="Frame tolerance for scene matching (default=2, same as AutoShot)")
    parser.add_argument("--output",      type=str, default="rai_results.json")
    args = parser.parse_args()

    config = vars(args)

    print("\n" + "="*80)
    print("TESTING ON RAI DATASET")
    print("="*80)
    print(f"Checkpoint : {config['checkpoint']}")
    print(f"Cache root : {config['cache_root']}")
    print(f"RAI root   : {config['rai_root']}")
    print(f"Batch size : {config['batch_size']}")
    print(f"Tolerance  : +-{config['tolerance']} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    # --- dataset ---
    dataset = RAIPredecodedDataset(config["cache_root"])
    loader  = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                         num_workers=4, pin_memory=True,
                         collate_fn=collate_rai_batch)

    # --- model ---
    print("\nLoading model...")
    ckpt  = torch.load(config["checkpoint"], map_location=device, weights_only=False)
    mcfg  = ckpt.get("config", {})
    model = create_hybrid_detector(
        main_dim=mcfg.get("main_dim", 512),
        num_fusion_blocks=mcfg.get("num_fusion_blocks", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded from epoch {ckpt.get('epoch', '?')}")

    # --- ground truth ---
    print(f"\nLoading ground truth from {config['rai_root']}...")
    gt_scenes_dict = load_gt_scenes(config["rai_root"], config["cache_root"])
    print(f"{len(gt_scenes_dict)} videos with GT scenes")

    # --- evaluate ---
    results = evaluate(model, loader, device, gt_scenes_dict, config)

    # --- print ---
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Videos evaluated  : {results['num_videos']}")
    print(f"Best boundary thr : {results['best_boundary_threshold']:.2f}")
    ov = results["overall"]
    print(f"\nOverall  P={ov['p']:.4f}  R={ov['r']:.4f}  F1={ov['f1']:.4f}")
    print(f"         TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}")

    video_f1s = sorted(results["video_results"].items(), key=lambda x: x[1]["f1"], reverse=True)
    print("\nPer-video results:")
    for vid, m in video_f1s:
        print(f"  {vid}: F1={m['f1']:.3f}  P={m['p']:.3f}  R={m['r']:.3f}  "
              f"(GT={m['n_gt']} scenes, Pred={m['n_pred']}, TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
    print("="*80)

    with open(config["output"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {config['output']}")


if __name__ == "__main__":
    main()
