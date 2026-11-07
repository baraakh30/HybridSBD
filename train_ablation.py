"""Train one HybridSBD Protocol-A variant.

This is intentionally a small, cloud-friendly runner.  It uses the same
predecoded tensors and augmentation policy as the original training script,
but makes the ablation configuration explicit and serializes every setting.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from ablation_config import environment_snapshot, get_variant, write_json
from ablation_loss import AblationBoundaryLoss
from ablation_model import create_ablation_detector
from core.binary_boundary_loss import create_one_hot_many_hot
from datasets import PredecodedDataset


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def augment_frames(frames):
    b, t, c, h, w = frames.shape
    flip = torch.rand(b, device=frames.device) < 0.5
    frames[flip] = frames[flip].flip(-1)
    apply = torch.rand(b, device=frames.device) < 0.5
    scale = 1.0 + (torch.rand(b, t, 1, 1, 1, device=frames.device) * 0.4 - 0.2)
    scale[~apply] = 1.0
    frames = (frames * scale).clamp(0, 1)
    apply = torch.rand(b, device=frames.device) < 0.5
    factor = 1.0 + (torch.rand(b, t, 1, 1, 1, device=frames.device) * 0.3 - 0.15)
    factor[~apply] = 1.0
    mean = frames.mean(dim=(2, 3, 4), keepdim=True)
    frames = ((frames - mean) * factor + mean).clamp(0, 1)
    apply = torch.rand(b, device=frames.device) < 0.5
    noise = torch.randn_like(frames) * 0.02
    noise[~apply] = 0
    frames = (frames + noise).clamp(0, 1)
    apply = torch.rand(b, device=frames.device) < 0.5
    shift = torch.rand(b, 1, 1, 1, 1, device=frames.device) * 0.2 - 0.1
    shift[~apply] = 0
    return (frames + shift).clamp(0, 1)


def make_loader(dataset, args, shuffle):
    kwargs = dict(batch_size=args.batch_size, shuffle=shuffle,
                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                  worker_init_fn=worker_init)
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def measured_unified_positive_rate(datasets):
    """Measure u=y OR z over the actual predecoded training labels."""
    positives = total = 0
    for dataset in datasets:
        for entry in dataset.mapping:
            payload = torch.load(entry["clip_file"], map_location="cpu", weights_only=False)
            labels = payload["targets"]
            positives += int((labels > 0).sum())
            total += int(labels.numel())
    return positives / total if total else 0.008


def scene_counts(y, z, labels, cfg, boundary_threshold=0.5, gradual_threshold=0.5):
    from test_autoshot import evaluate_scenes, predictions_to_scenes, _build_pred_many_hot
    y = torch.sigmoid(y.squeeze(-1)).detach().cpu().numpy()
    z = torch.sigmoid(z.squeeze(-1)).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    tp = fp = fn = 0
    for yp, zp, gt in zip(y, z, labels):
        gt_many = (gt > 0).astype(np.uint8)
        pred_many = (yp > boundary_threshold).astype(np.uint8) if cfg.supervision != "dual" else _build_pred_many_hot(yp, zp, boundary_threshold, gradual_threshold)
        _, _, _, (a, b, c) = evaluate_scenes(
            predictions_to_scenes(gt_many), predictions_to_scenes(pred_many),
            n_frames_miss_tolerance=2,
        )
        tp += a; fp += b; fn += c
    return tp, fp, fn


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, cfg, train):
    model.train(train)
    total = y_total = z_total = 0.0
    tp = fp = fn = 0
    for frames, labels, lengths in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        if train:
            frames = augment_frames(frames)
            optimizer.zero_grad(set_to_none=True)
        amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=amp):
            y_logits, z_logits = model(frames)
            loss, parts = loss_fn(y_logits, z_logits, labels, lengths)
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach())
        y_total += float(parts["y_loss"])
        z_total += float(parts["z_loss"])
        a, b, c = scene_counts(y_logits, z_logits, labels, cfg)
        tp += a; fp += b; fn += c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    n = max(1, len(loader))
    return {"loss": total / n, "y_loss": y_total / n, "z_loss": z_total / n,
            "precision": p, "recall": r, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train one Protocol-A ablation")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--autoshot_cache", required=True)
    parser.add_argument("--clipshots_cache", required=True)
    parser.add_argument("--output_root", default="outputs/ablations")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    cfg = get_variant(args.run_id)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_root) / args.run_id
    out.mkdir(parents=True, exist_ok=True)

    auto = PredecodedDataset(args.autoshot_cache, split="train", val_split_ratio=0.15,
                             random_seed=args.seed, temporal_jitter=True)
    auto_val = PredecodedDataset(args.autoshot_cache, split="val", val_split_ratio=0.15,
                                 random_seed=args.seed, temporal_jitter=False)
    clip_full = PredecodedDataset(args.clipshots_cache, split="train", val_split_ratio=0.15,
                                  random_seed=args.seed, temporal_jitter=True)
    rng = random.Random(args.seed)
    n_mix = int(len(clip_full) * 0.7)
    unified_rate = measured_unified_positive_rate([auto, clip_full]) if cfg.supervision == "unified" else None
    model = create_ablation_detector(cfg, unified_positive_rate=unified_rate or 0.008).to(device)
    params = model.count_parameters()
    loss_fn = AblationBoundaryLoss(cfg.supervision).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 5), eta_min=5e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cosine], milestones=[5])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    write_json(out / "config.json", {"variant": cfg.to_dict(), "args": vars(args),
                                     "unified_positive_rate": unified_rate,
                                     "environment": environment_snapshot(), "parameters": params})
    history = []
    best = -1.0
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        indices = rng.sample(range(len(clip_full)), n_mix)
        train_ds = ConcatDataset([auto, Subset(clip_full, indices)])
        train_loader = make_loader(train_ds, args, True)
        val_loader = make_loader(auto_val, args, False)
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, cfg, True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, loss_fn, optimizer, scaler, device, cfg, False)
        scheduler.step()
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                  "train": train_metrics, "val": val_metrics}
        history.append(record)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(), "config": cfg.to_dict(),
                    "metrics": record}, out / f"epoch_{epoch:03d}.pt")
        if val_metrics["f1"] > best:
            best = val_metrics["f1"]
            stale = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "config": cfg.to_dict(), "metrics": record,
                        "best_val_f1": best}, out / "best_model.pt")
        else:
            stale += 1
        write_json(out / "history.json", history)
        print(f"{args.run_id} epoch={epoch} train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f}")
        if stale >= args.patience:
            break
    write_json(out / "summary.json", {"run_id": args.run_id, "best_val_f1": best,
                                      "epochs_completed": len(history),
                                      "train_hours": (time.time() - start) / 3600,
                                      "parameters": params})


if __name__ == "__main__":
    main()
