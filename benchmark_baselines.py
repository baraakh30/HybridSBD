"""
Benchmark Baseline Models — AutoShot (TransNetV2Supernet) & TransNetV2 (PyTorch)

Both models use input resolution 27×48 as per their original designs.
Sequence length is set to 128 frames to match our model benchmark for fair comparison.

Usage:
    python benchmark_baselines.py
    python benchmark_baselines.py --device cpu
    python benchmark_baselines.py --batch_sizes 8 16 32 --seq_len 100
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

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parent
AS_DIR     = ROOT_DIR / 'data' / 'autoshot'
TNV2_DIR   = AS_DIR  / 'transnetv2' / 'inference-pytorch'
AS_CKPT    = AS_DIR  / 'ckpt_0_200_0.pth'

# Baseline models use 27×48 as hard-coded in their forward passes
FRAME_H, FRAME_W = 27, 48


# ------------------------------------------------------------------
# Model loaders
# ------------------------------------------------------------------

def load_autoshot(device: torch.device):
    """Load TransNetV2Supernet (AutoShot) with pretrained weights."""
    sys.path.insert(0, str(AS_DIR))
    from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet

    model = TransNetV2Supernet().to(device)

    if AS_CKPT.exists():
        ckpt = torch.load(AS_CKPT, map_location=device, weights_only=False)
        state = ckpt['net']
        own_state = model.state_dict()
        matched = {k: v for k, v in state.items() if k in own_state}
        own_state.update(matched)
        model.load_state_dict(own_state)
        print(f"    Loaded {len(matched)}/{len(own_state)} keys from {AS_CKPT.name}")
    else:
        print(f"    WARNING: {AS_CKPT} not found — benchmarking with random weights")

    model.eval()
    return model


def load_transnetv2(device: torch.device):
    """Instantiate TransNetV2 (PyTorch) with random weights for throughput benchmarking.
    No .pt weights are provided in the repo (only TF saved_model); architecture is fixed."""
    sys.path.insert(0, str(TNV2_DIR))
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2().to(device)
    model.eval()
    print(f"    TransNetV2 instantiated with random weights (TF→PT conversion not available)")
    return model


# ------------------------------------------------------------------
# Measurement helpers
# ------------------------------------------------------------------

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_dummy_input(model_name: str, batch_size: int, seq_len: int, device: torch.device):
    """Create appropriate dummy input for each model.

    AutoShot (TransNetV2Supernet):
        float32 [B, 3, T, H, W], values 0–255 (reprocess_layer divides /255 internally)
    TransNetV2 (PyTorch):
        uint8   [B, T, H, W, 3], values 0–255 (forward permutes to [B,3,T,H,W] and /255)
    """
    if model_name == 'autoshot':
        return (torch.rand(batch_size, 3, seq_len, FRAME_H, FRAME_W, device=device) * 255)
    else:  # transnetv2
        return torch.randint(0, 256, (batch_size, seq_len, FRAME_H, FRAME_W, 3),
                             dtype=torch.uint8, device=device)


def measure_inference_time(model, model_name, device, batch_size, seq_len,
                           num_runs=80, warmup=10):
    model.eval()
    x = make_dummy_input(model_name, batch_size, seq_len, device)

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


def measure_peak_memory_mb(model, model_name, device, batch_size, seq_len):
    if device.type != 'cuda':
        return None
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    x = make_dummy_input(model_name, batch_size, seq_len, device)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


# ------------------------------------------------------------------
# Per-model benchmark
# ------------------------------------------------------------------

def benchmark_one(label: str, model, model_name: str, device: torch.device,
                  batch_sizes: list, seq_len: int) -> dict:
    total_params, _ = count_parameters(model)
    size_mb = total_params * 4 / 1024 ** 2

    print(f"\n{'─'*72}")
    print(f"  {label}  |  params={total_params/1e6:.2f}M")
    print(f"{'─'*72}")

    result = {
        'label':        label,
        'total_params': total_params,
        'size_fp32_mb': round(size_mb, 1),
        'size_fp16_mb': round(size_mb / 2, 1),
        'batch_results': {},
    }

    for bs in batch_sizes:
        print(f"    batch_size={bs:>2} ... ", end='', flush=True)
        timing = measure_inference_time(model, model_name, device, bs, seq_len)
        mem_mb = measure_peak_memory_mb(model, model_name, device, bs, seq_len)
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

    # Real-time speedup (per-clip at median batch size)
    bs_mid         = batch_sizes[len(batch_sizes) // 2]
    per_clip_ms    = result['batch_results'][bs_mid]['mean_ms'] / bs_mid
    clip_dur_ms    = seq_len / 25 * 1000
    result['realtime_speedup_x'] = round(clip_dur_ms / per_clip_ms, 1)

    return result


# ------------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------------

def print_comparison_table(all_results: list, batch_sizes: list):
    bs_summary = batch_sizes[len(batch_sizes) // 2]
    W = 106
    print(f"\n{'='*W}")
    print(f"  COMPARISON TABLE  —  batch_size={bs_summary}, seq_len={SEQ_LEN_USED}, "
          f"device={all_results[0].get('device','?')}")
    print('='*W)
    hdr = (f"  {'Model':<24} {'Params':>7}  "
           f"{'FP32 MB':>7}  {'FP16 MB':>7}  "
           f"{'Latency ms':>10}  {'Clips/s':>7}  "
           f"{'Kframes/s':>9}  {'Mem MB':>7}  {'RT speedup':>10}")
    print(hdr)
    print('─'*W)
    for r in all_results:
        br  = r['batch_results'].get(bs_summary, {})
        mem = br.get('peak_memory_mb')
        print(
            f"  {r['label']:<24} {r['total_params']/1e6:>6.2f}M  "
            f"{r['size_fp32_mb']:>7.1f}  {r['size_fp16_mb']:>7.1f}  "
            f"{br.get('mean_ms', 0):>10.1f}  "
            f"{br.get('clips_per_sec', 0):>7.0f}  "
            f"{br.get('frames_per_sec', 0)/1000:>9.1f}  "
            f"{(str(round(mem)) if mem is not None else 'CPU'):>7}  "
            f"{r.get('realtime_speedup_x', 0):>9.1f}x"
        )
    print('='*W)


SEQ_LEN_USED = 128  # set at runtime


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    global SEQ_LEN_USED

    parser = argparse.ArgumentParser(
        description='Benchmark AutoShot and TransNetV2 baseline models')
    parser.add_argument('--device',      type=str, default='cuda')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--seq_len',     type=int, default=128)
    parser.add_argument('--output',      type=str,
                        default='outputs/benchmark_baselines.json')
    args = parser.parse_args()

    SEQ_LEN_USED = args.seq_len
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*72}")
    print(f"  BASELINE BENCHMARK — AutoShot & TransNetV2")
    print(f"{'='*72}")
    print(f"  Device      : {device}")
    print(f"  Batch sizes : {args.batch_sizes}")
    print(f"  Seq len     : {args.seq_len}")
    print(f"  Frame size  : {FRAME_H}×{FRAME_W}  (baseline models — same for both)")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU         : {props.name}  ({props.total_memory/1024**3:.1f} GB)")

    models_to_run = [
        ('AutoShot (TransNetV2Supernet)', 'autoshot',   load_autoshot),
        ('TransNetV2 (PyTorch)',          'transnetv2', load_transnetv2),
    ]

    all_results = []
    for label, model_name, loader_fn in models_to_run:
        print(f"\n  Loading: {label}")
        try:
            model = loader_fn(device)
            r = benchmark_one(label, model, model_name, device,
                              args.batch_sizes, args.seq_len)
            r['device'] = str(device)
            all_results.append(r)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR benchmarking {label}: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("\nNo models benchmarked successfully.")
        return

    print_comparison_table(all_results, args.batch_sizes)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'timestamp':   datetime.now().isoformat(),
            'device':      str(device),
            'batch_sizes': args.batch_sizes,
            'seq_len':     args.seq_len,
            'frame_size':  [FRAME_H, FRAME_W],
            'note':        (
                'AutoShot input: float32 [B,3,T,H,W] 0-255 (reprocess_layer does /255). '
                'TransNetV2 input: uint8 [B,T,H,W,3] (forward does permute+/255). '
                'TransNetV2 uses random weights (no .pt file available).'
            ),
            'models': all_results,
        }, f, indent=2)
    print(f"\n  Results saved -> {args.output}\n")


if __name__ == '__main__':
    main()
