"""
Fine-tune Hybrid Boundary Detector on BBC Planet Earth

Uses 8 BBC episodes for training, 3 for evaluation.
Same AutoShot scene-based evaluation (tolerance=2 frames).

Train split : 01-08
Test  split : 09, 10, 11

Usage:
    python finetune_bbc.py \
        --checkpoint outputs/hybrid_detector13/best_model.pt \
        --cache_root d:/grad/data/bbc_cache \
        --bbc_root "C:/Users/baraa/Documents/bbc" \
        --output_dir outputs/bbc_finetune

Training details:
  - LR = 5e-5  (fine-tuning, 10x lower than pre-training)
  - 15 epochs max, early stop on val F1
  - Same dual-head FocalBCE loss (lambda1=5, lambda2=0.1)
  - BBC has only cut transitions -> z-head targets always 0
  - Same augmentations as main training (flip, brightness, contrast, noise)
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
from core.binary_boundary_loss import create_binary_boundary_loss


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TRAIN_VIDEOS = [
    '01_From_Pole_to_Pole',
    '02_Mountains',
    '03_Ice_Worlds',
    '04_Great_Plains',
    '05_Jungles',
    '06_Seasonal_Forests',
    '07_Fresh_Water',
    '08_Ocean_Deep',
]
TEST_VIDEOS = [
    '09_Shallow_Seas',
    '10_Caves',
    '11_Deserts',
]


# ---------------------------------------------------------------------------
# Dataset  (training)
# ---------------------------------------------------------------------------
class BBCFineTuneDataset(Dataset):
    """
    Loads pre-decoded BBC clips for fine-tuning.
    Returns (frames [T,3,H,W], labels [T] 0/1, transition_lengths [T] zeros)
    where labels=1 marks cut-boundary frames.
    """
    def __init__(self, cache_root, video_ids):
        self.cache_root = Path(cache_root)
        index_data = json.load(open(self.cache_root / 'index_mapping.json'))
        self.clips = [
            c for c in index_data['mapping']
            if c['video_id'] in video_ids
        ]
        vids = len(set(c['video_id'] for c in self.clips))
        print(f"  BBC dataset: {vids} videos, {len(self.clips)} clips")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        info = self.clips[idx]
        data = torch.load(info['clip_file'], weights_only=False)
        frames  = data['frames'].float() / 255.0          # [T, 3, H, W]
        labels  = data['boundary_labels'].long()           # 0=none, 1=cut  (no class 2 in BBC)
        tlens   = torch.zeros_like(labels)                 # unused by loss
        return frames, labels, tlens


# ---------------------------------------------------------------------------
# Dataset  (inference / evaluation)
# ---------------------------------------------------------------------------
class BBCInferenceDataset(Dataset):
    """
    Loads pre-decoded BBC clips for inference.
    The cache is already built with stride=64, so no synthetic clips needed.
    """
    def __init__(self, cache_root, video_ids):
        self.cache_root = Path(cache_root)
        index_data = json.load(open(self.cache_root / 'index_mapping.json'))
        video_id_set = set(video_ids)
        self.entries = [
            c for c in index_data['mapping']
            if c['video_id'] in video_id_set
        ]
        self.entries.sort(key=lambda x: (x['video_id'], x['start_frame']))
        vids = len(set(c['video_id'] for c in self.entries))
        print(f"  BBC inference dataset: {vids} videos, {len(self.entries)} clips (stride=64)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        data   = torch.load(e['clip_file'], weights_only=False)
        frames = data['frames'].float() / 255.0
        return {
            'frames':      frames,
            'video_id':    e['video_id'],
            'start_frame': e['start_frame'],
        }


def collate_inference(batch):
    return {
        'frames':      torch.stack([b['frames'] for b in batch]),
        'video_id':    [b['video_id'] for b in batch],
        'start_frame': torch.tensor([b['start_frame'] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Augmentation  (identical to train_hybrid_detector.py)
# ---------------------------------------------------------------------------
def augment_frames(frames):
    B, T, C, H, W = frames.shape
    flip = torch.rand(B, device=frames.device) < 0.5
    frames[flip] = frames[flip].flip(-1)
    if (torch.rand(1) < 0.5).item():
        scale = 1.0 + (torch.rand(B, T, 1, 1, 1, device=frames.device) * 0.4 - 0.2)
        frames = (frames * scale).clamp(0, 1)
    if (torch.rand(1) < 0.5).item():
        f = 1.0 + (torch.rand(B, T, 1, 1, 1, device=frames.device) * 0.3 - 0.15)
        mean = frames.mean(dim=(2, 3, 4), keepdim=True)
        frames = ((frames - mean) * f + mean).clamp(0, 1)
    if (torch.rand(1) < 0.5).item():
        frames = (frames + torch.randn_like(frames) * 0.02).clamp(0, 1)
    if (torch.rand(1) < 0.5).item():
        shift = torch.rand(B, 1, 1, 1, 1, device=frames.device) * 0.2 - 0.1
        frames = (frames + shift).clamp(0, 1)
    return frames


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------
def read_bbc_shot_annotations(annotation_file):
    shots = []
    with open(annotation_file, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                shots.append([int(parts[0]), int(parts[1])])
    shots.sort(key=lambda x: x[0])
    return np.array(shots, dtype=np.int32)


def load_gt_scenes(bbc_root, cache_root, video_ids):
    ann_dir    = Path(bbc_root) / 'annotations' / 'shots'
    index_data = json.load(open(Path(cache_root) / 'index_mapping.json'))
    vid_to_num = {}
    for c in index_data['mapping']:
        v = c['video_id']
        if v not in vid_to_num:
            vid_to_num[v] = v.split('_')[0]
    gt = {}
    for vid in video_ids:
        num     = vid_to_num.get(vid, vid.split('_')[0])
        matched = next((f for f in ann_dir.glob('*.txt')
                        if f.stem.split('_')[0] == num), None)
        if matched is None:
            print(f"  Warning: no annotation for {vid}")
            continue
        scenes = read_bbc_shot_annotations(matched)
        if len(scenes):
            gt[vid] = scenes
    return gt


# ---------------------------------------------------------------------------
# Evaluation helpers  (AutoShot methodology)
# ---------------------------------------------------------------------------
def predictions_to_scenes(predictions):
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
    if not scenes:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(gt_scenes, pred_scenes, n_frames_miss_tolerance=2):
    shift  = n_frames_miss_tolerance / 2
    gt_s   = gt_scenes.astype(np.float32)   + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_s = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    gt_t   = np.stack([gt_s[:-1, 1],   gt_s[1:, 0]],   1) if len(gt_s)   > 1 else np.zeros((0, 2))
    pred_t = np.stack([pred_s[:-1, 1], pred_s[1:, 0]], 1) if len(pred_s) > 1 else np.zeros((0, 2))
    i = j = tp = fp = fn = 0
    while i < len(gt_t) or j < len(pred_t):
        if   j == len(pred_t):            fn += 1; i += 1
        elif i == len(gt_t):              fp += 1; j += 1
        elif pred_t[j, 1] < gt_t[i, 0]:  fp += 1; j += 1
        elif pred_t[j, 0] > gt_t[i, 1]:  fn += 1; i += 1
        else:                             tp += 1; i += 1; j += 1
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, (tp, fp, fn)


def _build_pred_many_hot(y_pred, z_pred, bt, gt):
    is_b = y_pred > bt
    is_g = z_pred > gt
    return (is_g | (is_b & ~is_g)).astype(np.uint8)


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    video_clips = defaultdict(list)
    for batch in loader:
        frames = batch['frames'].to(device, non_blocking=True)
        y_l, z_l = model(frames)
        y_p = torch.sigmoid(y_l.squeeze(-1)).cpu().numpy()
        z_p = torch.sigmoid(z_l.squeeze(-1)).cpu().numpy()
        for b in range(frames.shape[0]):
            video_clips[batch['video_id'][b]].append(
                (batch['start_frame'][b].item(), y_p[b], z_p[b])
            )
    return video_clips


def reconstruct_full_video(video_clips):
    vy, vz = {}, {}
    for vid, clips in video_clips.items():
        clips = sorted(clips, key=lambda x: x[0])
        T     = max(s + len(y) for s, y, _ in clips)
        fy    = np.zeros(T, np.float32)
        fz    = np.zeros(T, np.float32)
        cnt   = np.zeros(T, np.int32)
        for s, y, z in clips:
            e = s + len(y)
            fy[s:e] += y; fz[s:e] += z; cnt[s:e] += 1
        m = cnt > 0
        fy[m] /= cnt[m]; fz[m] /= cnt[m]
        vy[vid] = fy; vz[vid] = fz
    return vy, vz


def evaluate_on_split(model, inf_loader, device, gt_dict, tolerance=2):
    """Sweep thresholds, return best-F1 metrics dict."""
    vc       = run_inference(model, inf_loader, device)
    vy, vz   = reconstruct_full_video(vc)

    bts = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
           0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    gts = [0.3, 0.4, 0.5, 0.6]

    best_f1, best_bt, best_gt_thr = 0.0, 0.5, 0.5
    for bt in bts:
        for gt in gts:
            tp = fp = fn = 0
            for vid in vy:
                if vid not in gt_dict:
                    continue
                pred = _build_pred_many_hot(vy[vid], vz[vid], bt, gt)
                _, _, _, (t, f, n) = evaluate_scenes(
                    gt_dict[vid], predictions_to_scenes(pred), tolerance)
                tp += t; fp += f; fn += n
            p  = tp / (tp + fp + 1e-8)
            r  = tp / (tp + fn + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)
            if f1 > best_f1:
                best_f1, best_bt, best_gt_thr = f1, bt, gt

    tp = fp = fn = 0
    for vid in vy:
        if vid not in gt_dict:
            continue
        pred = _build_pred_many_hot(vy[vid], vz[vid], best_bt, best_gt_thr)
        _, _, _, (t, f, n) = evaluate_scenes(
            gt_dict[vid], predictions_to_scenes(pred), tolerance)
        tp += t; fp += f; fn += n
    p  = tp / (tp + fp + 1e-8)
    r  = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return dict(f1=f1, p=p, r=r, tp=tp, fp=fp, fn=fn,
                best_bt=best_bt, best_gt=best_gt_thr)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = total_y = total_z = n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for frames, labels, tlens in pbar:
        frames = frames.to(device, non_blocking=True)
        frames = augment_frames(frames)
        labels = labels.to(device, non_blocking=True)
        tlens  = tlens.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            y_l, z_l    = model(frames)
            loss, ld    = criterion(y_l, z_l, labels, tlens)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_y    += ld['y_loss']
        total_z    += ld['z_loss']
        n          += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", y=f"{total_y/n:.4f}")

    return dict(loss=total_loss/n, y_loss=total_y/n, z_loss=total_z/n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Fine-tune on BBC Planet Earth')
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--cache_root',  default=r'd:\grad\data\bbc_cache')
    parser.add_argument('--bbc_root',    required=True)
    parser.add_argument('--output_dir',  default='outputs/bbc_finetune2')
    parser.add_argument('--epochs',      type=int,   default=15)
    parser.add_argument('--lr',          type=float, default=5e-5)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--tolerance',   type=int,   default=2)
    parser.add_argument('--num_workers', type=int,   default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("BBC FINE-TUNING")
    print("=" * 80)
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Output dir  : {out_dir}")
    print(f"Train videos: {TRAIN_VIDEOS}")
    print(f"Test videos : {TEST_VIDEOS}")
    print(f"LR={args.lr}  Epochs={args.epochs}  BatchSize={args.batch_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device      : {device}")

    # ---- model ----
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg  = ckpt.get('config', {})
    model = create_hybrid_detector(
        main_dim=mcfg.get('main_dim', 512),
        num_fusion_blocks=mcfg.get('num_fusion_blocks', 3),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    print(f"Loaded from epoch {ckpt.get('epoch', '?')}")

    # ---- datasets ----
    print("\nBuilding datasets...")
    print("  Train:")
    train_ds = BBCFineTuneDataset(args.cache_root, TRAIN_VIDEOS)
    print("  Test (inference):")
    test_inf_ds  = BBCInferenceDataset(args.cache_root, TEST_VIDEOS)
    print("  Train (inference):")
    train_inf_ds = BBCInferenceDataset(args.cache_root, TRAIN_VIDEOS)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_inf_loader = DataLoader(
        test_inf_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_inference)
    train_inf_loader = DataLoader(
        train_inf_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_inference)

    # ---- ground truth ----
    print("\nLoading ground truth...")
    gt_test  = load_gt_scenes(args.bbc_root, args.cache_root, TEST_VIDEOS)
    gt_train = load_gt_scenes(args.bbc_root, args.cache_root, TRAIN_VIDEOS)
    print(f"  Test GT : {len(gt_test)} videos")
    print(f"  Train GT: {len(gt_train)} videos")

    # ---- loss / optimiser / scheduler ----
    criterion = create_binary_boundary_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10)
    scaler = torch.amp.GradScaler('cuda')

    # ---- baseline before fine-tuning ----
    print("\n--- Baseline (epoch 0, no fine-tuning) ---")
    base_test = evaluate_on_split(model, test_inf_loader, device, gt_test, args.tolerance)
    print(f"Test  F1={base_test['f1']:.4f}  P={base_test['p']:.4f}  R={base_test['r']:.4f}  "
          f"TP={base_test['tp']}  FP={base_test['fp']}  FN={base_test['fn']}")
    print(f"      best_bt={base_test['best_bt']:.2f}  best_gt={base_test['best_gt']:.2f}")

    best_f1 = base_test['f1']
    best_epoch = 0
    history = [{'epoch': 0, 'test': base_test}]

    # ---- training loop ----
    print()
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        scheduler.step()

        test_m = evaluate_on_split(model, test_inf_loader, device, gt_test, args.tolerance)

        print(f"Epoch {epoch:02d} | loss={tr['loss']:.4f}  y={tr['y_loss']:.4f} "
              f"z={tr['z_loss']:.4f} | "
              f"TestF1={test_m['f1']:.4f}  P={test_m['p']:.4f}  R={test_m['r']:.4f}  "
              f"TP={test_m['tp']}  FP={test_m['fp']}  FN={test_m['fn']}  "
              f"bt={test_m['best_bt']:.2f}")

        history.append({'epoch': epoch, 'train': tr, 'test': test_m})

        if test_m['f1'] > best_f1:
            best_f1   = test_m['f1']
            best_epoch = epoch
            save_path  = out_dir / 'best_model.pt'
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'config':           mcfg,
                'test_f1':          best_f1,
                'best_bt':          test_m['best_bt'],
                'best_gt':          test_m['best_gt'],
            }, save_path)
            print(f"  *** New best  F1={best_f1:.4f}  saved to {save_path}")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'config': mcfg},
                   out_dir / 'latest_model.pt')

    # ---- final evaluation ----
    print("\n--- Final evaluation (best checkpoint) ---")
    best_ckpt = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    final_test  = evaluate_on_split(model, test_inf_loader,  device, gt_test,  args.tolerance)
    final_train = evaluate_on_split(model, train_inf_loader, device, gt_train, args.tolerance)

    print(f"Best epoch : {best_epoch}")
    print(f"Test   F1={final_test['f1']:.4f}  "
          f"P={final_test['p']:.4f}  R={final_test['r']:.4f}  "
          f"TP={final_test['tp']}  FP={final_test['fp']}  FN={final_test['fn']}")
    print(f"Train  F1={final_train['f1']:.4f}  "
          f"P={final_train['p']:.4f}  R={final_train['r']:.4f}  "
          f"TP={final_train['tp']}  FP={final_train['fp']}  FN={final_train['fn']}")
    print("=" * 80)

    results = {
        'best_epoch':    best_epoch,
        'base_test_f1':  base_test['f1'],
        'best_test_f1':  best_f1,
        'final_test':    final_test,
        'final_train':   final_train,
        'history':       history,
        'config': {
            'checkpoint':   args.checkpoint,
            'lr':           args.lr,
            'epochs':       args.epochs,
            'batch_size':   args.batch_size,
            'tolerance':    args.tolerance,
            'train_videos': TRAIN_VIDEOS,
            'test_videos':  TEST_VIDEOS,
        },
    }
    with open(out_dir / 'finetune_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {out_dir / 'finetune_results.json'}")


if __name__ == '__main__':
    main()