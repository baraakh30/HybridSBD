"""
Benchmark Hybrid Boundary Detector — Multi-Model Comparison

Measures for each checkpoint:
  - Parameters
  - Inference time / FPS at batch sizes [8, 16, 32]
  - GPU memory (peak)
  - Real-time speedup factor

Usage:
    # Default: benchmark all known hybrid_detector checkpoints
    python benchmark_model.py

    # Specific checkpoints
    python benchmark_model.py --checkpoints outputs/hybrid_detector13/best_model.pt \
                                             outputs/hybrid_detector17/best_model.pt

    # CPU only
    python benchmark_model.py --device cpu
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection.hybrid_boundary_detector import create_hybrid_detector
from load_ablation_model import load_detector


# ---------------------------------------------------------------------------
# Default checkpoints
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINTS = [
    'outputs/hybrid_detector13/best_model.pt',
    'outputs/hybrid_detector15/best_model.pt',
    'outputs/hybrid_detector16/best_model.pt',
    'outputs/hybrid_detector17/best_model.pt',
    'outputs/hybrid_detector18/best_model.pt',
    'outputs/hybrid_detector19/best_model.pt',
]

# Real model input dimensions — must match training / predecoding
FRAME_H, FRAME_W = 48, 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model, device, batch_size, seq_len=128,
                           num_runs=80, warmup=10):
    """Returns timing stats (ms) and derived FPS for one batch_size."""
    model.eval()
    x = torch.randn(batch_size, seq_len, 3, FRAME_H, FRAME_W, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        'mean_ms':        float(times.mean() * 1000),
        'std_ms':         float(times.std()  * 1000),
        'min_ms':         float(times.min()  * 1000),
        'max_ms':         float(times.max()  * 1000),
        'clips_per_sec':  float(batch_size / times.mean()),
        'frames_per_sec': float(batch_size * seq_len / times.mean()),
    }


def measure_peak_memory_mb(model, device, batch_size, seq_len=128):
    """Returns peak GPU memory (MB) during a single forward pass."""
    if device.type != 'cuda':
        return None
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    x = torch.randn(batch_size, seq_len, 3, FRAME_H, FRAME_W, device=device)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


# ---------------------------------------------------------------------------
# Per-checkpoint benchmark
# ---------------------------------------------------------------------------

def benchmark_one(ckpt_path: str, device: torch.device,
                  batch_sizes: list, seq_len: int) -> dict:
    """Load one checkpoint, run all measurements, return results dict."""
    ckpt_path = Path(ckpt_path)
    label = ckpt_path.parent.name   # e.g. 'hybrid_detector13'

    if not ckpt_path.exists():
        print(f"  SKIP {ckpt_path} — file not found")
        return None

    ckpt             = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg             = ckpt.get('config', {})
    main_dim         = mcfg.get('main_dim', 512)
    num_fusion_blocks= mcfg.get('num_fusion_blocks', 3)
    epoch            = ckpt.get('epoch', '?')

    model, ckpt = load_detector(ckpt_path, device)
    model.eval()

    total_params, _ = count_parameters(model)
    size_mb = total_params * 4 / 1024 ** 2

    print(f"\n{'─'*72}")
    print(f"  {label}  |  epoch={epoch}  "
          f"dim={main_dim}  blocks={num_fusion_blocks}  "
          f"params={total_params/1e6:.2f}M")
    print(f"{'─'*72}")

    result = {
        'label':             label,
        'checkpoint':        str(ckpt_path),
        'epoch':             epoch,
        'main_dim':          main_dim,
        'num_fusion_blocks': num_fusion_blocks,
        'total_params':      total_params,
        'size_fp32_mb':      round(size_mb, 1),
        'size_fp16_mb':      round(size_mb / 2, 1),
        'batch_results':     {},
    }

    for bs in batch_sizes:
        print(f"    batch_size={bs:>2} ... ", end='', flush=True)
        timing = measure_inference_time(model, device, bs, seq_len)
        mem_mb = measure_peak_memory_mb(model, device, bs, seq_len)
        mem_str = f"{round(mem_mb):4d} MB" if mem_mb is not None else "CPU"
        print(f"{timing['mean_ms']:7.1f} ms  "
              f"{timing['clips_per_sec']:6.0f} clips/s  "
              f"{timing['frames_per_sec']:8.0f} frames/s  {mem_str}")

        result['batch_results'][bs] = {
            'mean_ms':        timing['mean_ms'],
            'std_ms':         timing['std_ms'],
            'min_ms':         timing['min_ms'],
            'max_ms':         timing['max_ms'],
            'clips_per_sec':  timing['clips_per_sec'],
            'frames_per_sec': timing['frames_per_sec'],
            'peak_memory_mb': mem_mb,
        }

    # Real-time speedup factor (at the median batch size, per-clip latency)
    bs_mid         = batch_sizes[len(batch_sizes) // 2]
    per_clip_ms    = result['batch_results'][bs_mid]['mean_ms'] / bs_mid
    clip_dur_ms    = seq_len / 25 * 1000   # 25 fps content
    speedup        = clip_dur_ms / per_clip_ms
    result['realtime_speedup_x'] = round(speedup, 1)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(all_results: list, batch_sizes: list):
    bs_summary = batch_sizes[len(batch_sizes) // 2]
    W = 106
    print(f"\n{'='*W}")
    print(f"  COMPARISON TABLE  —  batch_size={bs_summary}, seq_len=128, "
          f"frame={FRAME_H}×{FRAME_W}, device={all_results[0].get('device','?')}")
    print('='*W)
    hdr = (f"  {'Model':<22} {'Dim':>4} {'Blk':>3}  {'Params':>7}  "
           f"{'FP32 MB':>7}  {'FP16 MB':>7}  "
           f"{'Latency ms':>10}  {'Clips/s':>7}  "
           f"{'Kframes/s':>9}  {'Mem MB':>7}  {'RT speedup':>10}")
    print(hdr)
    print('─'*W)
    for r in all_results:
        br  = r['batch_results'].get(bs_summary, {})
        mem = br.get('peak_memory_mb')
        print(
            f"  {r['label']:<22} {r['main_dim']:>4} {r['num_fusion_blocks']:>3}  "
            f"{r['total_params']/1e6:>6.2f}M  "
            f"{r['size_fp32_mb']:>7.1f}  {r['size_fp16_mb']:>7.1f}  "
            f"{br.get('mean_ms', 0):>10.1f}  "
            f"{br.get('clips_per_sec', 0):>7.0f}  "
            f"{br.get('frames_per_sec', 0)/1000:>9.1f}  "
            f"{(str(round(mem)) if mem is not None else 'CPU'):>7}  "
            f"{r.get('realtime_speedup_x', 0):>9.1f}x"
        )
    print('='*W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark multiple Hybrid Boundary Detector checkpoints')
    parser.add_argument('--checkpoints', nargs='+',
                        default=DEFAULT_CHECKPOINTS,
                        help='One or more checkpoint paths to benchmark')
    parser.add_argument('--device',      type=str, default='cuda')
    parser.add_argument('--batch_sizes', type=int, nargs='+',
                        default=[8, 16, 32])
    parser.add_argument('--seq_len',     type=int, default=128)
    parser.add_argument('--output',      type=str,
                        default='outputs/benchmark_results.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*72}")
    print(f"  HYBRID BOUNDARY DETECTOR — MULTI-MODEL BENCHMARK")
    print(f"{'='*72}")
    print(f"  Device      : {device}")
    print(f"  Batch sizes : {args.batch_sizes}")
    print(f"  Seq len     : {args.seq_len}")
    print(f"  Frame size  : {FRAME_H}x{FRAME_W}")
    print(f"  Models      : {len(args.checkpoints)}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU         : {props.name}  "
              f"({props.total_memory/1024**3:.1f} GB)")

    all_results = []
    for ckpt_path in args.checkpoints:
        r = benchmark_one(ckpt_path, device, args.batch_sizes, args.seq_len)
        if r is not None:
            r['device'] = str(device)
            all_results.append(r)

    if not all_results:
        print("\nNo checkpoints found — nothing to report.")
        return

    print_comparison_table(all_results, args.batch_sizes)

    # Save JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'timestamp':   datetime.now().isoformat(),
            'device':      str(device),
            'batch_sizes': args.batch_sizes,
            'seq_len':     args.seq_len,
            'frame_size':  [FRAME_H, FRAME_W],
            'models':      all_results,
        }, f, indent=2)
    print(f"\n  Results saved -> {args.output}\n")


if __name__ == '__main__':
    main()
