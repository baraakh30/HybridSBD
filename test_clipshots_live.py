"""
Test Hybrid Boundary Detector on ClipShots Test Set — LIVE DENSE INFERENCE

Decodes every test video on-the-fly with a dense stride-64 sliding window.
Speed strategy: one large Dataset covers all 50k clips across all 500 videos;
a multi-worker DataLoader decodes them in parallel while the GPU runs
inference, eliminating the video-by-video sequential bottleneck.

Usage:
    python test_clipshots_live.py \\
        --checkpoint outputs/hybrid_detector12/best_model.pt
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.hybrid_boundary_detector import create_hybrid_detector
from test_autoshot import evaluate_scenes, _build_pred_many_hot,predictions_to_scenes
from test_clipshots import load_gt_scenes

try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    raise SystemExit("torchcodec is required: pip install torchcodec")


# ── dataset: one entry per dense clip across ALL videos ──────────────────────

class DenseClipDataset(Dataset):
    """
    Enumerates every sliding-window clip for every test video upfront.
    Workers open a fresh VideoDecoder per __getitem__ call — cheap because
    the OS file cache keeps the video warm after the first worker touches it.
    """
    def __init__(self, video_dir: Path, ann: dict, seq_len: int,
                 stride: int, frame_size: tuple):
        self.seq_len    = seq_len
        self.frame_size = frame_size
        self.clips: list[tuple] = []   # (video_name, video_path, start, total_frames)

        missing = 0
        for vname, vdata in ann.items():
            vpath = video_dir / vname
            if not vpath.exists():
                vpath = video_dir / (vname + '.mp4')
            if not vpath.exists():
                missing += 1
                continue

            T = int(vdata['frame_num'])
            if T < seq_len:
                starts = [0]
            else:
                starts = list(range(0, T - seq_len + 1, stride))
                if starts[-1] + seq_len < T:
                    starts.append(T - seq_len)

            for s in starts:
                self.clips.append((vname, str(vpath), s, T))

        if missing:
            print(f"  ⚠  {missing} videos not found — skipping")
        print(f"DenseClipDataset: {len(ann) - missing} videos, "
              f"{len(self.clips):,} clips  (stride={stride})")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        vname, vpath, start, T = self.clips[idx]
        H, W = self.frame_size
        seq_len = self.seq_len

        decoder  = VideoDecoder(vpath)
        vid_len  = len(decoder)
        indices  = list(range(start, min(start + seq_len, vid_len)))
        frames   = decoder.get_frames_at(indices).data          # [t, 3, h, w] uint8
        del decoder

        if frames.shape[2:] != (H, W):
            frames = F.interpolate(frames.float(), size=(H, W),
                                   mode='bilinear', align_corners=False
                                   ).to(torch.uint8)

        if frames.shape[0] < seq_len:
            pad    = frames[-1:].repeat(seq_len - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, pad], 0)

        return {
            'frames':       frames.float().div_(255.0),   # [T, 3, H, W]
            'video_name':   vname,
            'start_frame':  start,
            'total_frames': T,
        }


def _collate(batch):
    return {
        'frames':       torch.stack([b['frames']      for b in batch]),
        'video_name':   [b['video_name']   for b in batch],
        'start_frame':  torch.tensor([b['start_frame']  for b in batch]),
        'total_frames': torch.tensor([b['total_frames'] for b in batch]),
    }


# ── inference: single pass over DataLoader, accumulate into per-video arrays ─

@torch.no_grad()
def run_dense_inference(model, loader, device):
    """
    GPU inference over every clip; returns accumulated per-video y/z sum arrays
    and count arrays. Averaging happens after the loop.
    """
    model.eval()

    # We don't know final video lengths upfront, so accumulate dynamically
    # video_name -> {'y': np.array, 'z': np.array, 'cnt': np.array}
    accum: dict = {}

    for batch in tqdm(loader, desc="Inference", dynamic_ncols=True):
        frames = batch['frames'].to(device, non_blocking=True)   # [B, T, 3, H, W]
        y_l, z_l = model(frames)                                  # [B, T, 1]
        y_p = torch.sigmoid(y_l.squeeze(-1)).cpu().numpy()        # [B, T]
        z_p = torch.sigmoid(z_l.squeeze(-1)).cpu().numpy()

        for i in range(len(batch['video_name'])):
            vname   = batch['video_name'][i]
            start   = batch['start_frame'][i].item()
            T_total = batch['total_frames'][i].item()
            seq_len = y_p.shape[1]

            if vname not in accum:
                accum[vname] = {
                    'y':   np.zeros(T_total, np.float32),
                    'z':   np.zeros(T_total, np.float32),
                    'cnt': np.zeros(T_total, np.int32),
                }
            buf = accum[vname]

            end  = start + seq_len
            # Only count unpadded frames (tail clips are padded to seq_len)
            real = min(seq_len, T_total - start)
            buf['y'][start:start + real]   += y_p[i, :real]
            buf['z'][start:start + real]   += z_p[i, :real]
            buf['cnt'][start:start + real] += 1

    # Average
    video_y, video_z = {}, {}
    for vname, buf in accum.items():
        mask = buf['cnt'] > 0
        buf['y'][mask] /= buf['cnt'][mask]
        buf['z'][mask] /= buf['cnt'][mask]
        video_y[vname] = buf['y']
        video_z[vname] = buf['z']

    return video_y, video_z


# ── post-processing helpers ─────────────────────────────────────────────────

def merge_short_scenes(scenes: np.ndarray, min_len: int) -> np.ndarray:
    """
    Iteratively merge scenes shorter than *min_len* frames into their
    longer direct neighbour until no short scenes remain.
    """
    if len(scenes) <= 1 or min_len <= 1:
        return scenes
    scenes = [list(s) for s in scenes]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(scenes):
            width = scenes[i][1] - scenes[i][0] + 1
            if width < min_len and len(scenes) > 1:
                if i == 0:                       # absorb into right
                    scenes[1][0] = scenes[0][0]
                    scenes.pop(0)
                elif i == len(scenes) - 1:       # absorb into left
                    scenes[-2][1] = scenes[-1][1]
                    scenes.pop()
                else:                            # absorb into wider neighbour
                    left_w  = scenes[i - 1][1] - scenes[i - 1][0] + 1
                    right_w = scenes[i + 1][1] - scenes[i + 1][0] + 1
                    if left_w >= right_w:
                        scenes[i - 1][1] = scenes[i][1]
                    else:
                        scenes[i + 1][0] = scenes[i][0]
                    scenes.pop(i)
                changed = True
                break
            i += 1
    return np.array(scenes)


# ── threshold sweep ───────────────────────────────────────────────────────────

def sweep_and_score(video_y, video_z, gt_scenes_dict, tolerance, min_scene_len=1):
    coarse             = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                          0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    gradual_thresholds = [0.3, 0.4, 0.5, 0.6]

    def _score(bt, gt_thr):
        tp = fp = fn = 0
        for vid in video_y:
            if vid not in gt_scenes_dict:
                continue
            pred_mh = _build_pred_many_hot(video_y[vid], video_z[vid], bt, gt_thr)
            pred_sc = predictions_to_scenes(pred_mh)
            if min_scene_len > 1:
                pred_sc = merge_short_scenes(pred_sc, min_scene_len)
            _, _, _, (t, f, n) = evaluate_scenes(gt_scenes_dict[vid], pred_sc, tolerance)
            tp += t; fp += f; fn += n
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return dict(p=p, r=r, f1=f1, tp=tp, fp=fp, fn=fn)

    print(f"\nCoarse sweep {len(coarse)} x {len(gradual_thresholds)} "
          f"(min_scene_len={min_scene_len})...")
    best_f1, best_bt, best_gt_thr = 0.0, 0.5, 0.5
    all_results = {}
    for bt in tqdm(coarse, desc="Coarse", dynamic_ncols=True):
        for gt_thr in gradual_thresholds:
            r = _score(bt, gt_thr)
            all_results[(bt, gt_thr)] = r
            if r['f1'] > best_f1:
                best_f1, best_bt, best_gt_thr = r['f1'], bt, gt_thr

    # Fine sweep +-0.08 around the coarse best at 0.02 steps
    fine_bts = sorted({round(best_bt + d, 3)
                       for d in [-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08]
                       if 0.0 < best_bt + d < 1.0})
    fine_gts = sorted({round(best_gt_thr + d, 2)
                       for d in [-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08]
                       if 0.0 < best_gt_thr + d < 1.0})
    print(f"Fine sweep {len(fine_bts)} x {len(fine_gts)} around "
          f"({best_bt:.2f}, {best_gt_thr:.2f})...")
    for bt in tqdm(fine_bts, desc="Fine  ", dynamic_ncols=True):
        for gt_thr in fine_gts:
            if (bt, gt_thr) in all_results:
                continue
            r = _score(bt, gt_thr)
            all_results[(bt, gt_thr)] = r
            if r['f1'] > best_f1:
                best_f1, best_bt, best_gt_thr = r['f1'], bt, gt_thr

    # Per-video at best thresholds
    video_results = {}
    for vid in video_y:
        if vid not in gt_scenes_dict:
            continue
        pred_mh = _build_pred_many_hot(video_y[vid], video_z[vid], best_bt, best_gt_thr)
        pred_sc = predictions_to_scenes(pred_mh)
        if min_scene_len > 1:
            pred_sc = merge_short_scenes(pred_sc, min_scene_len)
        gt_sc   = gt_scenes_dict[vid]
        p, r, f1, (tp, fp, fn) = evaluate_scenes(gt_sc, pred_sc, tolerance)
        video_results[vid] = dict(p=p, r=r, f1=f1, tp=tp, fp=fp, fn=fn,
                                  n_gt=len(gt_sc), n_pred=len(pred_sc))

    best = all_results[(best_bt, best_gt_thr)]
    return {
        'best_boundary_threshold': best_bt,
        'best_gradual_threshold':  best_gt_thr,
        'min_scene_len':           min_scene_len,
        'overall': best,
        'all_results': {str(k): v for k, v in all_results.items()},
        'video_results': video_results,
        'num_videos': len(video_results),
    }


# ── FP analysis ───────────────────────────────────────────────────────────────

def _record_fp(fp_records, vid, pred_idx, pt, pred_sc, gt_trans, y, z, T):
    """Record one false-positive transition with diagnostic fields."""
    mid_frame   = max(0, min(T - 1, int(round((pt[0] + pt[1]) / 2))))
    left_scene  = pred_sc[pred_idx]
    scene_width = int(left_scene[1]) - int(left_scene[0]) + 1

    if len(gt_trans) > 0:
        gt_mids  = (gt_trans[:, 0] + gt_trans[:, 1]) / 2.0
        dist_to_gt = float(np.abs(gt_mids - mid_frame).min())
    else:
        dist_to_gt = float(T)

    w = 8
    lo, hi   = max(0, mid_frame - w), min(T, mid_frame + w + 1)
    y_peak   = float(y[lo:hi].max())
    z_peak   = float(z[lo:hi].max())
    y_mean   = float(y[lo:hi].mean())
    z_mean   = float(z[lo:hi].mean())

    fp_records.append(dict(
        vid=vid,
        mid_frame=mid_frame,
        T=T,
        pos=round(mid_frame / max(T - 1, 1), 3),
        scene_width=scene_width,
        dist_to_gt=round(dist_to_gt, 1),
        y_peak=round(y_peak, 3),
        z_peak=round(z_peak, 3),
        y_mean=round(y_mean, 3),
        z_mean=round(z_mean, 3),
        z_driven=z_peak > y_peak,
    ))


def analyze_fps(video_y, video_z, gt_scenes_dict, boundary_thr, gradual_thr,
                tolerance=2, min_scene_len=1):
    """
    Reproduce the TP/FP matching from evaluate_scenes for every video and
    collect diagnostic records for every FP transition.
    """
    fp_records = []

    for vid in video_y:
        if vid not in gt_scenes_dict:
            continue
        y   = video_y[vid]
        z   = video_z[vid]
        T   = len(y)

        pred_mh = _build_pred_many_hot(y, z, boundary_thr, gradual_thr)
        pred_sc = predictions_to_scenes(pred_mh)

        # optional min-scene filter
        if min_scene_len > 1:
            filtered = [pred_sc[0]]
            for sc in pred_sc[1:]:
                if sc[1] - sc[0] + 1 < min_scene_len:
                    filtered[-1] = np.array([filtered[-1][0], sc[1]])
                else:
                    filtered.append(sc)
            pred_sc = np.array(filtered)

        gt_sc = gt_scenes_dict[vid]

        # Reproduce evaluate_scenes boundary-to-boundary matching
        shift = tolerance / 2.0
        gt_f  = gt_sc.astype(np.float32)  + np.array([[-0.5 + shift,  0.5 - shift]])
        pr_f  = pred_sc.astype(np.float32) + np.array([[-0.5 + shift,  0.5 - shift]])

        if len(gt_f) < 2 or len(pr_f) < 2:
            continue

        gt_trans   = np.stack([gt_f[:-1,  1], gt_f[1:,  0]], axis=1)   # [N-1, 2]
        pred_trans = np.stack([pr_f[:-1,  1], pr_f[1:,  0]], axis=1)   # [M-1, 2]

        i, j = 0, 0
        while i < len(gt_trans) or j < len(pred_trans):
            if j == len(pred_trans):
                i += 1
            elif i == len(gt_trans):
                _record_fp(fp_records, vid, j, pred_trans[j],
                            pred_sc, gt_trans, y, z, T)
                j += 1
            elif pred_trans[j, 1] < gt_trans[i, 0]:
                _record_fp(fp_records, vid, j, pred_trans[j],
                            pred_sc, gt_trans, y, z, T)
                j += 1
            elif pred_trans[j, 0] > gt_trans[i, 1]:
                i += 1
            else:
                i += 1
                j += 1

    return fp_records


def print_fp_analysis(fp_records):
    """Print a compact diagnostic breakdown of collected FP records."""
    import statistics
    from collections import Counter

    N = len(fp_records)
    if not N:
        print("No FPs to analyse.")
        return

    sep = '=' * 80
    print(f"\n{sep}\nFP ANALYSIS  (n={N})\n{sep}")

    # 1. Driver: z-head vs y-head
    n_z = sum(1 for r in fp_records if r['z_driven'])
    n_y = N - n_z
    print(f"\n[1] DRIVER")
    print(f"    z-driven : {n_z:5d}  ({100 * n_z / N:.1f}%)")
    print(f"    y-driven : {n_y:5d}  ({100 * n_y / N:.1f}%)")
    y_peaks = [r['y_peak'] for r in fp_records]
    z_peaks = [r['z_peak'] for r in fp_records]
    print(f"    y_peak  mean={statistics.mean(y_peaks):.3f}  "
          f"median={statistics.median(y_peaks):.3f}  "
          f"p90={float(np.percentile(y_peaks, 90)):.3f}")
    print(f"    z_peak  mean={statistics.mean(z_peaks):.3f}  "
          f"median={statistics.median(z_peaks):.3f}  "
          f"p90={float(np.percentile(z_peaks, 90)):.3f}")

    # 2. Distance to nearest GT transition
    dists = [r['dist_to_gt'] for r in fp_records]
    print(f"\n[2] DISTANCE TO NEAREST GT TRANSITION")
    print(f"    mean={statistics.mean(dists):.1f}  "
          f"median={statistics.median(dists):.1f}  "
          f"p90={float(np.percentile(dists, 90)):.1f}")
    for lo, hi, label in [
        (0,   5,    "<5 fr  (borderline match)"),
        (5,   25,   "5-25 fr"),
        (25,  100,  "25-100 fr"),
        (100, 1e9,  ">100 fr (fully spurious)"),
    ]:
        n = sum(1 for d in dists if lo <= d < hi)
        print(f"    {label:30s}: {n:5d}  ({100 * n / N:.1f}%)")

    # 3. Position in video
    pos = [r['pos'] for r in fp_records]
    n_start = sum(1 for p in pos if p < 0.05)
    n_mid   = sum(1 for p in pos if 0.05 <= p < 0.95)
    n_end   = sum(1 for p in pos if p >= 0.95)
    print(f"\n[3] POSITION IN VIDEO")
    print(f"    0-5%   (clip edge start): {n_start:5d}  ({100 * n_start / N:.1f}%)")
    print(f"    5-95%  (mid-video)      : {n_mid:5d}  ({100 * n_mid   / N:.1f}%)")
    print(f"    95-100% (clip edge end) : {n_end:5d}  ({100 * n_end   / N:.1f}%)")

    # 4. Scene width of the FP predicted scene
    widths = [r['scene_width'] for r in fp_records]
    print(f"\n[4] PREDICTED SCENE WIDTH AT FP")
    print(f"    mean={statistics.mean(widths):.1f}  "
          f"median={statistics.median(widths):.1f}  "
          f"p90={float(np.percentile(widths, 90)):.1f}")
    for lo, hi, label in [
        (0,  5,   "<5 fr   (spike/noise)"),
        (5,  20,  "5-20 fr"),
        (20, 64,  "20-64 fr"),
        (64, 1e9, ">=64 fr (wide)"),
    ]:
        n = sum(1 for w in widths if lo <= w < hi)
        print(f"    {label:25s}: {n:5d}  ({100 * n / N:.1f}%)")

    # 5. Top 10 worst videos
    print(f"\n[5] TOP 10 WORST FP VIDEOS")
    for vid, cnt in Counter(r['vid'] for r in fp_records).most_common(10):
        T_val   = next(r['T'] for r in fp_records if r['vid'] == vid)
        recs    = [r for r in fp_records if r['vid'] == vid]
        n_z_vid = sum(1 for r in recs if r['z_driven'])
        print(f"    {vid:44s}: {cnt:3d} FPs  T={T_val:5d}  "
              f"z_driven={n_z_vid}/{cnt}")

    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Dense sliding-window evaluation on ClipShots test set')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--video_dir', type=str,
                        default=r'C:\Users\baraa\Documents\ClipShots\videos\test')
    parser.add_argument('--ann_path', type=str,
                        default=r'C:\Users\baraa\Documents\ClipShots\annotations\test.json')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Clips per GPU forward pass')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='DataLoader decode workers')
    parser.add_argument('--stride', type=int, default=64,
                        help='Window stride in frames (64 = 50%% overlap)')
    parser.add_argument('--seq_len',    type=int, default=128)
    parser.add_argument('--frame_size', type=int, nargs=2, default=[48, 64],
                        metavar=('H', 'W'))
    parser.add_argument('--tolerance',    type=int, default=2)
    parser.add_argument('--min_scene_len', type=int, default=0,
                        help='Merge predicted scenes shorter than N frames into neighbours '
                             '(0=disabled). Try 10-30 to suppress FP bursts.')
    parser.add_argument('--output', type=str, default='clipshots_live_results.json')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("CLIPSHOTS TEST SET — DENSE SLIDING-WINDOW INFERENCE")
    print("="*80)
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Video dir    : {args.video_dir}")
    print(f"Stride       : {args.stride} frames  "
          f"(overlap={args.seq_len - args.stride} frames)")
    print(f"Batch / workers : {args.batch_size} / {args.num_workers}")
    print(f"Tolerance    : ±{args.tolerance} frames")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device       : {device}")

    # --- model ---
    print("\nLoading model...")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg  = ckpt.get('config', {})
    model = create_hybrid_detector(
        main_dim=mcfg.get('main_dim', 512),
        num_fusion_blocks=mcfg.get('num_fusion_blocks', 3),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✓ Loaded from epoch {ckpt.get('epoch', '?')}")

    # --- annotations & GT ---
    print("\nLoading annotations...")
    with open(args.ann_path, 'r') as f:
        ann = json.load(f)
    gt_scenes_dict = load_gt_scenes(args.ann_path, list(ann.keys()))

    # --- dataset + loader ---
    dataset = DenseClipDataset(
        Path(args.video_dir), ann,
        seq_len=args.seq_len, stride=args.stride,
        frame_size=tuple(args.frame_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=_collate,
    )

    # --- inference ---
    video_y, video_z = run_dense_inference(model, loader, device)
    print(f"Inference done: {len(video_y)} videos")

    # --- sweep min_scene_len in [1, 2, 3, 4] ---
    msl_candidates = [1, 2, 3, 4]
    all_msl_results = {}
    for msl in msl_candidates:
        print(f"\n{'─'*60}")
        print(f"min_scene_len = {msl}")
        all_msl_results[msl] = sweep_and_score(
            video_y, video_z, gt_scenes_dict, args.tolerance, msl)

    # --- comparison table ---
    print("\n" + "="*80)
    print("MIN SCENE LEN SWEEP — SUMMARY")
    print("="*80)
    print(f"  {'MSL':>4}  {'BT':>5}  {'GT':>5}  {'P':>7}  {'R':>7}  "
          f"{'F1':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}")
    print("  " + "-"*70)
    best_msl, best_results = max(
        all_msl_results.items(), key=lambda kv: kv[1]['overall']['f1'])
    for msl, res in all_msl_results.items():
        ov  = res['overall']
        tag = "  ◄ best" if msl == best_msl else ""
        print(f"  {msl:>4}  {res['best_boundary_threshold']:>5.2f}  "
              f"{res['best_gradual_threshold']:>5.2f}  "
              f"{ov['p']:>7.4f}  {ov['r']:>7.4f}  {ov['f1']:>7.4f}  "
              f"{ov['tp']:>6}  {ov['fp']:>6}  {ov['fn']:>6}{tag}")
    print("="*80)

    # --- detailed print for best MSL ---
    results = best_results
    min_scene_len = best_msl
    ov = results['overall']
    print(f"\nBest overall  min_scene_len={min_scene_len}  "
          f"boundary_thr={results['best_boundary_threshold']:.2f}  "
          f"gradual_thr={results['best_gradual_threshold']:.2f}")
    print(f"P={ov['p']:.4f}  R={ov['r']:.4f}  F1={ov['f1']:.4f}  "
          f"TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}")

    vf1s = sorted(results['video_results'].items(),
                  key=lambda x: x[1]['f1'], reverse=True)
    print(f"\nTop 5 videos:")
    for vid, m in vf1s[:5]:
        print(f"  {vid}: F1={m['f1']:.3f}  P={m['p']:.3f}  R={m['r']:.3f}  "
              f"gt={m['n_gt']} pred={m['n_pred']}")
    print(f"\nBottom 5 videos:")
    for vid, m in vf1s[-5:]:
        print(f"  {vid}: F1={m['f1']:.3f}  P={m['p']:.3f}  R={m['r']:.3f}  "
              f"gt={m['n_gt']} pred={m['n_pred']}")

    # --- FP analysis (best MSL) ---
    fp_records = analyze_fps(
        video_y, video_z, gt_scenes_dict,
        results['best_boundary_threshold'],
        results['best_gradual_threshold'],
        tolerance=args.tolerance,
        min_scene_len=min_scene_len,
    )
    print_fp_analysis(fp_records)

    with open(args.output, 'w') as f:
        json.dump(
            {str(msl): {k: v for k, v in res.items() if k != 'all_results'}
             for msl, res in all_msl_results.items()},
            f, indent=2)
    print(f"\n✓ Results saved to {args.output}")


if __name__ == '__main__':
    main()
