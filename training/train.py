"""Train one HybridSBD Protocol-A variant.

This is intentionally a small, cloud-friendly runner.  It uses the same
predecoded tensors and augmentation policy as the original training script,
but makes the ablation configuration explicit and serializes every setting.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: lets the script import the repository packages

import argparse
import copy
import datetime
import json
import os
import random
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from hybridsbd.configs.ablation_config import environment_snapshot, get_variant, write_json
from hybridsbd.datasets import PredecodedDataset
from hybridsbd.logging_utils import tee_to_file
from hybridsbd.losses.ablation_loss import AblationBoundaryLoss
from hybridsbd.losses.binary_boundary_loss import create_one_hot_many_hot
from hybridsbd.models.ablation_model import create_ablation_detector


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


def synth_transitions(frames, labels, lengths, prob):
    """Splice synthetic transitions into a fraction of the batch (TransNetV2-style, training videos only).

    Each selected clip i is replaced by a clip made from itself and a different clip j of the same batch:
      * hard cut      (40%): frames[:s] from i, frames[s:] from j; frames s-1 and s labelled 1 (same as real cuts)
      * dissolve      (40%): linear cross-fade over L frames, labelled 2 for all L frames (same as real graduals)
      * fade via black(20%): i fades to black, then j fades in, over L frames, labelled 2
    The splice point is chosen with a 3-frame clear margin, so no real transition of i or j is cut in half or touches the new one. Nothing from the test sets is used.
    Returns the number of synthesized clips.
    """
    b, t = labels.shape
    if prob <= 0 or b < 2:
        return 0
    src_f, src_l, src_n = frames.clone(), labels.clone(), lengths.clone()
    lab = src_l.cpu().numpy()
    made = 0
    for i in np.nonzero(np.random.rand(b) < prob)[0].tolist():
        j = random.randrange(b - 1)
        j += int(j >= i)
        kind = random.random()
        # cs[k] = number of labelled frames before index k; a window is clear when its count does not change
        cs_i = np.concatenate([[0], np.cumsum(lab[i] != 0)])
        cs_j = np.concatenate([[0], np.cumsum(lab[j] != 0)])
        if kind < 0.4:
            starts = np.arange(8, t - 7)
            ok = starts[(cs_i[starts] - cs_i[starts - 3] == 0) & (cs_j[starts + 3] - cs_j[starts] == 0)]
            if len(ok) == 0:
                continue
            s = int(random.choice(ok))
            frames[i] = torch.cat([src_f[i, :s], src_f[j, s:]], 0)
            labels[i] = torch.cat([src_l[i, :s], src_l[j, s:]], 0)
            lengths[i] = torch.cat([src_n[i, :s], src_n[j, s:]], 0)
            labels[i, s - 1 : s + 1] = 1
            lengths[i, s - 1 : s + 1] = 2
        else:
            n = random.randint(3, 30)
            starts = np.arange(8, t - 8 - n + 1)
            ok = starts[(cs_i[starts] - cs_i[starts - 3] == 0) & (cs_j[starts + n + 3] - cs_j[starts + n] == 0)]
            if len(ok) == 0:
                continue
            s = int(random.choice(ok))
            alpha = (torch.arange(n, device=frames.device, dtype=frames.dtype) + 1) / (n + 1)
            alpha = alpha.view(n, 1, 1, 1)
            a, c = src_f[i, s : s + n], src_f[j, s : s + n]
            if kind < 0.8:
                window = (1 - alpha) * a + alpha * c
            else:
                window = torch.where(alpha < 0.5, a * (1 - 2 * alpha), c * (2 * alpha - 1))
            frames[i] = torch.cat([src_f[i, :s], window, src_f[j, s + n :]], 0)
            labels[i] = torch.cat([src_l[i, :s], torch.full((n,), 2, dtype=labels.dtype, device=labels.device), src_l[j, s + n :]], 0)
            lengths[i] = torch.cat([src_n[i, :s], torch.full((n,), n, dtype=lengths.dtype, device=lengths.device), src_n[j, s + n :]], 0)
        made += 1
    return made


def strong_augment(frames, p=0.7):
    """Extra per-clip augmentation (same for all frames of a clip): random zoom/shift, colour gain, rare grayscale."""
    b, t, c, h, w = frames.shape
    dev = frames.device
    on = (torch.rand(b, device=dev) < p).float()
    scale = 1.0 - 0.25 * torch.rand(b, device=dev) * on
    tx = (torch.rand(b, device=dev) * 2 - 1) * (1 - scale)
    ty = (torch.rand(b, device=dev) * 2 - 1) * (1 - scale)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (b, t * c, h, w), align_corners=False)
    x = F.grid_sample(frames.reshape(b, t * c, h, w), grid, mode="bilinear", padding_mode="border", align_corners=False)
    frames = x.reshape(b, t, c, h, w)
    gain = 1.0 + (torch.rand(b, 1, c, 1, 1, device=dev) * 0.2 - 0.1) * (torch.rand(b, 1, 1, 1, 1, device=dev) < p)
    frames = (frames * gain).clamp(0, 1)
    gray = torch.rand(b, 1, 1, 1, 1, device=dev) < 0.1
    return torch.where(gray, frames.mean(2, keepdim=True).expand_as(frames), frames)


class EMA:
    """Exponential moving average of the weights (BatchNorm statistics are copied from the live model)."""

    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        self.decay = decay
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        ema_p = list(self.model.parameters())
        torch._foreach_mul_(ema_p, d)
        torch._foreach_add_(ema_p, [q.detach() for q in model.parameters()], alpha=1 - d)
        for eb, mb in zip(self.model.buffers(), model.buffers()):
            eb.copy_(mb)


def make_loader(dataset, args, shuffle):
    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init,
    )
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def _unified_counts(clip_file):
    labels = torch.load(clip_file, map_location="cpu", weights_only=False)["targets"].reshape(1, -1)
    one_hot, many_hot = create_one_hot_many_hot(labels)
    unified = torch.maximum(one_hot, many_hot)  # u = y OR z (exactly as in the A2 loss)
    return int(unified.sum()), int(unified.numel())


def measured_unified_positive_rate(datasets, workers=32):
    """Measure u=y OR z over the actual predecoded training labels (parallel file reads)."""
    from concurrent.futures import ThreadPoolExecutor

    files = [entry["clip_file"] for dataset in datasets for entry in dataset.mapping]
    with ThreadPoolExecutor(workers) as pool:
        counts = list(pool.map(_unified_counts, files))
    positives = sum(c[0] for c in counts)
    total = sum(c[1] for c in counts)
    return positives / total if total else 0.008


def scene_counts(y, z, labels, cfg, boundary_threshold=0.5, gradual_threshold=0.5):
    # Validation uses each variant's own inference rule at fixed tau=0.5 (the
    # existing validation procedure): dual -> y OR z; A1/A2/A3 -> y only.
    from hybridsbd.scoring import build_pred_many_hot, evaluate_scenes, predictions_to_scenes

    y = torch.sigmoid(y.squeeze(-1)).detach().cpu().numpy()
    z = None if z is None else torch.sigmoid(z.squeeze(-1)).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    tp = fp = fn = 0
    z_iter = z if z is not None else [None] * len(y)
    for yp, zp, gt in zip(y, z_iter, labels):
        gt_many = (gt > 0).astype(np.uint8)
        pred_many = (
            (yp > boundary_threshold).astype(np.uint8)
            if cfg.supervision != "dual"
            else build_pred_many_hot(yp, zp, boundary_threshold, gradual_threshold)
        )
        _, _, _, (a, b, c) = evaluate_scenes(
            predictions_to_scenes(gt_many),
            predictions_to_scenes(pred_many),
            n_frames_miss_tolerance=2,
        )
        tp += a
        fp += b
        fn += c
    return tp, fp, fn


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, cfg, train, ema=None, synth_prob=0.0, strong_aug=False):
    model.train(train)
    synth_clips = 0
    total = y_total = z_total = 0.0
    tp = fp = fn = 0
    for frames, labels, lengths in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        if train:
            if synth_prob > 0:
                synth_clips += synth_transitions(frames, labels, lengths, synth_prob)
            frames = augment_frames(frames)
            if strong_aug:
                frames = strong_augment(frames)
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
            if ema is not None:
                ema.update(model)
        total += float(loss.detach())
        y_total += float(parts["y_loss"])
        z_total += float(parts["z_loss"])
        a, b, c = scene_counts(y_logits, z_logits, labels, cfg)
        tp += a
        fp += b
        fn += c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    n = max(1, len(loader))
    out = {"loss": total / n, "y_loss": y_total / n, "z_loss": z_total / n, "precision": p, "recall": r, "f1": f1}
    if train and synth_prob > 0:
        out["synth_clips"] = synth_clips
    return out


def git_state(out):
    """Save git commit, dirty status and any uncommitted diff next to the run."""
    info = {}
    try:
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        info["dirty"] = bool(status.strip())
        info["status"] = status
        if info["dirty"]:
            (out / "uncommitted.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], text=True), encoding="utf-8")
    except Exception as exc:
        info["error"] = repr(exc)
    return info


def main():
    parser = argparse.ArgumentParser(description="Train one Protocol-A ablation")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--autoshot_cache", required=True)
    parser.add_argument("--clipshots_cache", required=True)
    parser.add_argument("--output_root", default="outputs/ablations")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="seed of the SHOT / ClipShots train-validation video split (kept at 42 for every training seed, "
        "so extra-seed runs use exactly the same split as the seed-42 runs)",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--synth_prob",
        type=float,
        default=0.0,
        help="fraction of training clips replaced by synthetic cut/dissolve/fade splices (0 = off, spec default)",
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.0, help="EMA of the weights, used for validation and checkpoints (0 = off, spec default)"
    )
    parser.add_argument(
        "--strong_aug", action="store_true", help="extra per-clip zoom/shift/colour/grayscale augmentation (off = spec default)"
    )
    parser.add_argument(
        "--shot_repeat", type=int, default=1, help="how many times the SHOT training clips appear per epoch (1 = spec default)"
    )
    parser.add_argument(
        "--no_channels_last", action="store_true", help="disable the channels_last memory layout (default on: ~2x faster steps, same maths)"
    )
    parser.add_argument("--force", action="store_true", help="Archive an existing run directory (never deleted) and retrain")
    args = parser.parse_args()
    cfg = get_variant(args.run_id)
    out = Path(args.output_root) / args.run_id
    if out.exists() and any(out.iterdir()):
        if (out / "summary.json").exists() and not args.force:
            print(f"{args.run_id}: already completed ({out}); use --force to retrain. Skipping.")
            return
        if not args.force:
            raise SystemExit(
                f"{out} already contains an unfinished run; refusing to overwrite. "
                f"Re-run with --force (the old directory is archived, not deleted)."
            )
        archive = out.with_name(f"{out.name}.archived_{datetime.datetime.now():%Y%m%d_%H%M%S}")
        shutil.move(str(out), str(archive))
        print(f"Archived previous run to {archive}")
    out.mkdir(parents=True, exist_ok=True)
    tee_to_file(out / "train.log")
    if os.environ.get("PYTHONHASHSEED") != "0":
        print(
            "WARNING: PYTHONHASHSEED was not 0 at interpreter start; export it in the shell "
            "(the launchers do). Sampling itself uses MD5 seeds and is unaffected."
        )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    git = git_state(out)
    try:
        run_training(args, cfg, out, device, git)
    except BaseException as exc:
        write_json(
            out / "status.json",
            {
                "run_id": args.run_id,
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "time": datetime.datetime.now().isoformat(),
            },
        )
        raise


def split_checksums(auto, auto_val, clip_full):
    """SHA-256 of the sorted video-name lists of the SHOT validation / train and ClipShots train splits."""
    from hybridsbd.utils import names_sha256

    def h(ds):
        names = sorted({c["video_name"] for c in ds.mapping})
        return {"videos": len(names), "sha256": names_sha256(names)}

    return {"shot_val": h(auto_val), "shot_train": h(auto), "clipshots_train": h(clip_full)}


def run_training(args, cfg, out, device, git):

    auto = PredecodedDataset(args.autoshot_cache, split="train", val_split_ratio=0.15, random_seed=args.split_seed, temporal_jitter=True)
    auto_val = PredecodedDataset(args.autoshot_cache, split="val", val_split_ratio=0.15, random_seed=args.split_seed, temporal_jitter=False)
    clip_full = PredecodedDataset(
        args.clipshots_cache, split="train", val_split_ratio=0.15, random_seed=args.split_seed, temporal_jitter=True
    )
    split_info = split_checksums(auto, auto_val, clip_full)
    print("SPLIT_CHECKSUM", json.dumps(split_info), flush=True)
    rng = random.Random(args.seed)
    n_mix = int(len(clip_full) * 0.7)
    unified_rate = measured_unified_positive_rate([auto, clip_full]) if cfg.supervision == "unified" else None
    model = create_ablation_detector(cfg, unified_positive_rate=unified_rate or 0.008).to(device)
    params = model.count_parameters()
    if not args.no_channels_last:
        # Memory-layout change only (NHWC conv weights): cuDNN BatchNorm/conv run in much faster NHWC kernels
        # (measured 0.217 -> 0.107 s/step, A0, B=32, AMP, deterministic cudnn). Values are mathematically identical.
        model = model.to(memory_format=torch.channels_last)
    loss_fn = AblationBoundaryLoss(cfg.supervision).to(device)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 5), eta_min=5e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cosine], milestones=[5])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    write_json(
        out / "config.json",
        {
            "variant": cfg.to_dict(),
            "args": vars(args),
            "split": split_info,
            "unified_positive_rate": unified_rate,
            "environment": environment_snapshot(),
            "git": git,
            "parameters": params,
            "fusion_in": model.fusion_in,
            "emb_dim": model.emb_dim,
            "head_type": type(model.detection_head).__name__,
        },
    )
    write_json(out / "status.json", {"run_id": args.run_id, "status": "running", "started": datetime.datetime.now().isoformat()})
    history = []
    best = -1.0
    best_epoch = 0
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        indices = rng.sample(range(len(clip_full)), n_mix)
        train_ds = ConcatDataset([auto] * args.shot_repeat + [Subset(clip_full, indices)])
        train_loader = make_loader(train_ds, args, True)
        val_loader = make_loader(auto_val, args, False)
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scaler,
            device,
            cfg,
            True,
            ema=ema,
            synth_prob=args.synth_prob,
            strong_aug=args.strong_aug,
        )
        eval_model = ema.model if ema is not None else model  # validation and checkpoints use the EMA weights
        with torch.no_grad():
            val_metrics = run_epoch(eval_model, val_loader, loss_fn, optimizer, scaler, device, cfg, False)
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_hours": (time.time() - start) / 3600,
        }
        history.append(record)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": eval_model.state_dict(),
                "raw_model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": cfg.to_dict(),
                "metrics": record,
            },
            out / f"epoch_{epoch:03d}.pt",
        )
        if val_metrics["f1"] > best:
            best = val_metrics["f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "config": cfg.to_dict(),
                    "metrics": record,
                    "best_val_f1": best,
                },
                out / "best_model.pt",
            )
        else:
            stale += 1
        write_json(out / "history.json", history)
        print(f"{args.run_id} epoch={epoch} train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f}")
        if stale >= args.patience:
            break
    stopped_early = stale >= args.patience
    write_json(
        out / "summary.json",
        {
            "run_id": args.run_id,
            "best_val_f1": best,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "train_hours": (time.time() - start) / 3600,
            "parameters": params,
        },
    )
    write_json(out / "status.json", {"run_id": args.run_id, "status": "completed", "finished": datetime.datetime.now().isoformat()})


if __name__ == "__main__":
    main()
