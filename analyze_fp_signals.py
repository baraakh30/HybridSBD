"""
FP Signal Analysis — what does the model actually "see" at false-positive frames?

Loads clipshots_live_results.json (or a similar per-video JSON), identifies the
worst FP videos, re-runs dense inference on ONLY those videos with
return_features=True, then compares per-stream magnitudes at:
  • FP  frames  — predicted transition, no GT transition nearby
  • TP  frames  — predicted transition, matches GT
  • BG  frames  — no prediction, no GT

Prints a compact table showing which streams are elevated at FP sites vs TPs.

Usage:
    python analyze_fp_signals.py \\
        --checkpoint outputs/hybrid_detector13/best_model.pt \\
        --results_json clipshots_live_results.json \\
        --video_dir "C:\\Users\\baraa\\Documents\\ClipShots\\videos\\test" \\
        --ann_path  "C:\\Users\\baraa\\Documents\\ClipShots\\annotations\\test.json" \\
        --top_k 10
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection.hybrid_boundary_detector import create_hybrid_detector
from test_autoshot import evaluate_scenes, _build_pred_many_hot, predictions_to_scenes
from test_clipshots import load_gt_scenes

try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    raise SystemExit("torchcodec required: pip install torchcodec")


# ─── Motion feature names ─────────────────────────────────────────────────────
# PixelMotionExtractor: distances [1,2,4,8,16], 3 stats each = 15 values
MOTION_NAMES = [
    f"mot_d{d}_{s}"
    for d in [1, 2, 4, 8, 16]
    for s in ["mag_mean", "mag_std", "uniformity"]
]

STREAM_NAMES = ["rgb", "diff", "pixel_motion", "frame_sim", "color_hist_sim", "fused"]


# ─── Minimal Dataset: one video at a time ─────────────────────────────────────

class SingleVideoDataset(Dataset):
    def __init__(self, vpath, T, seq_len=128, stride=64, frame_size=(48, 64)):
        self.vpath     = str(vpath)
        self.seq_len   = seq_len
        self.frame_size = frame_size
        self.T         = T
        if T < seq_len:
            self.starts = [0]
        else:
            starts = list(range(0, T - seq_len + 1, stride))
            if starts[-1] + seq_len < T:
                starts.append(T - seq_len)
            self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start   = self.starts[idx]
        H, W    = self.frame_size
        decoder = VideoDecoder(self.vpath)
        vid_len = len(decoder)
        indices = list(range(start, min(start + self.seq_len, vid_len)))
        frames  = decoder.get_frames_at(indices).data
        del decoder
        if frames.shape[2:] != (H, W):
            frames = F.interpolate(frames.float(), size=(H, W),
                                   mode='bilinear', align_corners=False).to(torch.uint8)
        if frames.shape[0] < self.seq_len:
            pad    = frames[-1:].repeat(self.seq_len - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, pad], 0)
        return {'frames': frames.float().div_(255.0), 'start': start}


def collate_fn(batch):
    return {
        'frames': torch.stack([b['frames'] for b in batch]),
        'start':  torch.tensor([b['start']  for b in batch]),
    }


# ─── Per-video inference with feature extraction ──────────────────────────────

@torch.no_grad()
def infer_video_with_features(model, vpath, T, device, seq_len=128, stride=64,
                               frame_size=(48, 64), batch_size=8, num_workers=4):
    """Return per-frame y, z and per-stream L2-norm arrays for one video."""
    ds = SingleVideoDataset(vpath, T, seq_len, stride, frame_size)
    dl_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers,
                     collate_fn=collate_fn, pin_memory=True,
                     persistent_workers=(num_workers > 0))
    if num_workers > 0:
        dl_kwargs['prefetch_factor'] = 2
    dl = DataLoader(ds, **dl_kwargs)
    model.eval()

    # Accumulators
    y_sum   = np.zeros(T, np.float32)
    z_sum   = np.zeros(T, np.float32)
    cnt     = np.zeros(T, np.int32)
    # Per-stream: store sum of L2 norms (scalar per frame)
    stream_sum  = {s: np.zeros(T, np.float32) for s in STREAM_NAMES}
    stream_cnt  = np.zeros(T, np.int32)
    # Raw motion values (15-dim per frame)
    motion_sum  = np.zeros((T, 15), np.float32)

    for batch in dl:
        frames  = batch['frames'].to(device, non_blocking=True)   # [B, T, 3, H, W]
        starts  = batch['start']                                   # [B]
        y_l, z_l, feats = model(frames, return_features=True)

        y_p = torch.sigmoid(y_l.squeeze(-1)).cpu().numpy()         # [B, T]
        z_p = torch.sigmoid(z_l.squeeze(-1)).cpu().numpy()

        # Convert per-stream tensors to per-frame L2 norms
        stream_norms = {}
        for s in STREAM_NAMES:
            t = feats[s]                                           # [B, T, D]
            stream_norms[s] = t.norm(dim=-1).cpu().numpy()        # [B, T]

        mot = feats['pixel_motion'].cpu().numpy()                  # [B, T, 15]

        for i in range(len(starts)):
            start = starts[i].item()
            real  = min(seq_len, T - start)
            sl    = slice(start, start + real)

            y_sum[sl]  += y_p[i, :real]
            z_sum[sl]  += z_p[i, :real]
            cnt[sl]    += 1

            for s in STREAM_NAMES:
                stream_sum[s][sl]  += stream_norms[s][i, :real]
            stream_cnt[sl] += 1

            motion_sum[sl] += mot[i, :real]

    # Average
    m = cnt > 0
    y_sum[m]  /= cnt[m];  z_sum[m]  /= cnt[m]
    sc = stream_cnt > 0
    for s in STREAM_NAMES:
        stream_sum[s][sc] /= stream_cnt[sc]
    motion_sum[sc] /= stream_cnt[sc, None]

    return y_sum, z_sum, stream_sum, motion_sum


# ─── Classify each frame ──────────────────────────────────────────────────────

def classify_frames(y, z, gt_scenes, boundary_thr, gradual_thr,
                    tolerance=2, min_scene_len=4):
    """Return arrays of frame labels: 'FP', 'TP', 'BG'."""
    T = len(y)
    labels = np.full(T, 'BG', dtype=object)

    pred_mh = _build_pred_many_hot(y, z, boundary_thr, gradual_thr)
    pred_sc = predictions_to_scenes(pred_mh)

    if min_scene_len > 1:
        _sc = [list(pred_sc[0])]
        for sc in pred_sc[1:]:
            if sc[1] - sc[0] + 1 < min_scene_len:
                _sc[-1][1] = sc[1]
            else:
                _sc.append(list(sc))
        pred_sc = np.array(_sc)

    shift = tolerance / 2.0
    gt_f  = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift,  0.5 - shift]])
    pr_f  = pred_sc.astype(np.float32)   + np.array([[-0.5 + shift,  0.5 - shift]])

    if len(gt_f) < 2 or len(pr_f) < 2:
        return labels

    gt_trans   = np.stack([gt_f[:-1, 1], gt_f[1:, 0]],  axis=1)
    pred_trans = np.stack([pr_f[:-1, 1], pr_f[1:, 0]], axis=1)

    # Which predicted transitions are TP vs FP
    tp_pred = set(); fp_pred = set()
    i = j = 0
    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            i += 1
        elif i == len(gt_trans):
            fp_pred.add(j); j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp_pred.add(j); j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            i += 1
        else:
            tp_pred.add(j); i += 1; j += 1

    for j in tp_pred:
        mid = int(round((pred_trans[j, 0] + pred_trans[j, 1]) / 2))
        mid = max(0, min(T - 1, mid))
        labels[mid] = 'TP'
    for j in fp_pred:
        mid = int(round((pred_trans[j, 0] + pred_trans[j, 1]) / 2))
        mid = max(0, min(T - 1, mid))
        labels[mid] = 'FP'

    return labels


# ─── Pretty comparison table ──────────────────────────────────────────────────

def print_signal_analysis(all_records):
    """
    all_records: list of dicts with keys
        'label', 'y', 'z', stream names, motion name × 15
    """
    import statistics

    by_label = defaultdict(list)
    for r in all_records:
        by_label[r['label']].append(r)

    sep = '=' * 90
    print(f"\n{sep}")
    print(f"SIGNAL ANALYSIS BY FRAME TYPE  "
          f"(FP={len(by_label['FP'])}  TP={len(by_label['TP'])}  "
          f"BG={len(by_label['BG'])})")
    print(sep)

    cols = ['y_prob', 'z_prob'] + STREAM_NAMES + MOTION_NAMES
    labels = ['FP', 'TP', 'BG']

    # Header
    print(f"\n{'Signal':<28} {'FP (mean)':>12} {'TP (mean)':>12} {'BG (mean)':>12}  {'FP/BG':>7}  {'FP/TP':>7}")
    print('─' * 90)

    for col in cols:
        vals = {}
        for lbl in labels:
            recs = by_label[lbl]
            if recs:
                v = [r[col] for r in recs]
                vals[lbl] = statistics.mean(v)
            else:
                vals[lbl] = float('nan')

        fp_val = vals['FP']
        tp_val = vals['TP']
        bg_val = vals['BG']

        fp_bg = fp_val / (bg_val + 1e-9)
        fp_tp = fp_val / (tp_val + 1e-9)

        # Highlight rows where FP/BG > 1.5 or < 0.67
        flag = ''
        if fp_bg > 1.5:   flag = '  ↑ FP elevated'
        elif fp_bg < 0.67: flag = '  ↓ FP suppressed'

        print(f"  {col:<26} {fp_val:>12.4f} {tp_val:>12.4f} {bg_val:>12.4f}  {fp_bg:>7.2f}  {fp_tp:>7.2f}{flag}")

    # Uniformity at dist=1 is the most diagnostic: cut=low, pan=high
    print(f"\n{sep}")
    print("KEY DISCRIMINATOR: mot_d1_uniformity")
    print("  HIGH uniformity (>0.7) at FP = pan/zoom  ← motion confuses model")
    print("  LOW  uniformity (<0.4) at FP = genuine visual change like a real cut")
    fp_uni = [r['mot_d1_uniformity'] for r in by_label['FP']]
    tp_uni = [r['mot_d1_uniformity'] for r in by_label['TP']]
    if fp_uni:
        for thr, label in [(0.4, '<0.4 (cut-like)'), (0.7, '0.4-0.7'), (1.1, '>0.7 (pan-like)')]:
            lo = 0 if label.startswith('<') else (0.4 if '0.4' in label else 0.7)
            hi = thr
            n  = sum(1 for v in fp_uni if lo <= v < hi)
            pct = 100 * n / len(fp_uni)
            print(f"  FP  {label:25s}: {n:5d} ({pct:.1f}%)")
        print()
        for thr, label in [(0.4, '<0.4 (cut-like)'), (0.7, '0.4-0.7'), (1.1, '>0.7 (pan-like)')]:
            lo = 0 if label.startswith('<') else (0.4 if '0.4' in label else 0.7)
            hi = thr
            n  = sum(1 for v in tp_uni if lo <= v < hi)
            pct = 100 * n / max(len(tp_uni), 1)
            print(f"  TP  {label:25s}: {n:5d} ({pct:.1f}%)")
    print(sep)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FP signal analysis on worst ClipShots videos')
    parser.add_argument('--checkpoint',   type=str, required=True)
    parser.add_argument('--results_json', type=str, default='clipshots_live_results.json')
    parser.add_argument('--video_dir',    type=str,
                        default=r'C:\Users\baraa\Documents\ClipShots\videos\test')
    parser.add_argument('--ann_path',     type=str,
                        default=r'C:\Users\baraa\Documents\ClipShots\annotations\test.json')
    parser.add_argument('--top_k',        type=int, default=10,
                        help='Analyse this many worst FP videos')
    parser.add_argument('--boundary_thr', type=float, default=0.73)
    parser.add_argument('--gradual_thr',  type=float, default=0.58)
    parser.add_argument('--min_scene_len',type=int,   default=4)
    parser.add_argument('--tolerance',    type=int,   default=2)
    parser.add_argument('--seq_len',      type=int,   default=128)
    parser.add_argument('--stride',       type=int,   default=64)
    parser.add_argument('--frame_size',   type=int,   nargs=2, default=[48, 64])
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--num_workers',  type=int,   default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model ──
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg  = ckpt.get('config', {})
    model = create_hybrid_detector(
        main_dim=mcfg.get('main_dim', 512),
        num_fusion_blocks=mcfg.get('num_fusion_blocks', 3),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded from epoch {ckpt.get('epoch', '?')}")

    # ── Load annotations & GT ──
    with open(args.ann_path, 'r') as f:
        ann = json.load(f)
    gt_scenes_dict = load_gt_scenes(args.ann_path, list(ann.keys()))

    # ── Pick worst FP videos from saved results ──
    print(f"\nLoading results: {args.results_json}")
    with open(args.results_json, 'r') as f:
        saved = json.load(f)

    # The JSON saved by the sweep contains per-MSL keys; use the best MSL entry
    # or fall back to a flat structure if saved from an older run.
    if 'video_results' in saved:
        video_results = saved['video_results']
    else:
        # multi-MSL format: pick the key matching min_scene_len
        key = str(args.min_scene_len)
        if key not in saved:
            key = list(saved.keys())[0]
            print(f"  min_scene_len={args.min_scene_len} not in JSON, using key={key}")
        video_results = saved[key].get('video_results', {})

    # Sort by FP count descending
    worst = sorted(video_results.items(), key=lambda kv: kv[1].get('fp', 0), reverse=True)
    worst = worst[:args.top_k]
    print(f"\nTop {len(worst)} FP videos:")
    for vname, m in worst:
        print(f"  {vname}: FP={m['fp']}  TP={m['tp']}  FN={m['fn']}")

    # ── Run inference on each worst video ──
    video_dir = Path(args.video_dir)
    all_records = []

    for vname, _ in tqdm(worst, desc='Analysing videos'):
        vpath = video_dir / vname
        if not vpath.exists():
            vpath = video_dir / (vname + '.mp4')
        if not vpath.exists():
            print(f"  ⚠  {vname} not found, skipping")
            continue

        T = int(ann[vname]['frame_num'])
        if vname not in gt_scenes_dict:
            continue

        y, z, stream_norms, motion = infer_video_with_features(
            model, vpath, T, device,
            seq_len=args.seq_len, stride=args.stride,
            frame_size=tuple(args.frame_size),
            batch_size=args.batch_size, num_workers=args.num_workers,
        )

        labels = classify_frames(
            y, z, gt_scenes_dict[vname],
            args.boundary_thr, args.gradual_thr,
            tolerance=args.tolerance, min_scene_len=args.min_scene_len,
        )

        fp_idx  = np.where(labels == 'FP')[0]
        tp_idx  = np.where(labels == 'TP')[0]
        # Sample BG to avoid it dominating stats
        bg_idx  = np.where(labels == 'BG')[0]
        rng     = np.random.default_rng(42)
        bg_idx  = rng.choice(bg_idx, size=min(len(bg_idx), len(fp_idx) * 5), replace=False) \
                  if len(bg_idx) > 0 else bg_idx

        for idx_set, label in [(fp_idx, 'FP'), (tp_idx, 'TP'), (bg_idx, 'BG')]:
            for t in idx_set:
                rec = {'label': label, 'video': vname, 't': int(t), 'T': T,
                       'y_prob': float(y[t]), 'z_prob': float(z[t])}
                for s in STREAM_NAMES:
                    rec[s] = float(stream_norms[s][t])
                for mi, mname in enumerate(MOTION_NAMES):
                    rec[mname] = float(motion[t, mi])
                all_records.append(rec)

    if not all_records:
        print("No records collected."); return

    print_signal_analysis(all_records)

    # Save raw records for further analysis
    out = 'fp_signal_records.json'
    with open(out, 'w') as f:
        json.dump(all_records[:5000], f, indent=2)   # cap at 5000 for size
    print(f"\n✓ Raw records saved to {out}")


if __name__ == '__main__':
    main()
