"""
Benchmark computational efficiency of Hybrid Boundary Detector

Measures:
1. Inference time (FPS)
2. GPU memory usage
3. Model parameters
4. Throughput (clips/second)
"""

import torch
import time
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.hybrid_boundary_detector import create_hybrid_detector


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model, device, batch_size=32, seq_length=128, num_runs=100, warmup=10):
    """Measure inference time and FPS"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_length, 3, 64, 64, device=device)
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Benchmark
    print(f"\nBenchmarking ({num_runs} iterations)...")
    times = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_runs)):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            
            _ = model(dummy_input)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            
            times.append(end - start)
    
    times = np.array(times)
    
    return {
        'mean_time': float(times.mean()),
        'std_time': float(times.std()),
        'min_time': float(times.min()),
        'max_time': float(times.max()),
        'fps': float(batch_size / times.mean()),
        'frames_per_sec': float(batch_size * seq_length / times.mean())
    }


def measure_memory(model, device, batch_size=32, seq_length=128):
    """Measure GPU memory usage"""
    if device.type != 'cuda':
        return {'memory_mb': 0, 'peak_memory_mb': 0}
    
    model.eval()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Measure baseline
    baseline_memory = torch.cuda.memory_allocated(device) / 1024**2
    
    # Run inference
    dummy_input = torch.randn(batch_size, seq_length, 3, 64, 64, device=device)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Measure peak
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    current_memory = torch.cuda.memory_allocated(device) / 1024**2
    
    return {
        'baseline_memory_mb': baseline_memory,
        'current_memory_mb': current_memory,
        'peak_memory_mb': peak_memory,
        'inference_memory_mb': peak_memory - baseline_memory
    }


def benchmark_model(checkpoint_path, device='cuda', batch_sizes=[8, 16, 32], seq_length=128):
    """Complete benchmark of model"""
    
    # Initialize results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': checkpoint_path,
        'device': str(device),
        'sequence_length': seq_length,
        'batch_sizes_tested': batch_sizes
    }
    
    print("="*80)
    print("HYBRID BOUNDARY DETECTOR - COMPUTATIONAL EFFICIENCY BENCHMARK")
    print("="*80)
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = create_hybrid_detector(main_dim=256,num_fusion_blocks=3).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"⚠ Checkpoint not found, using random initialization")
    
    # 1. Model size
    print("\n" + "="*80)
    print("1. MODEL SIZE")
    print("="*80)
    
    total_params, trainable_params = count_parameters(model)
    model_size_mb = total_params * 4 / 1024**2  # 4 bytes per float32
    
    results['model_size'] = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_fp32_mb': model_size_mb,
        'model_size_fp16_mb': model_size_mb / 2
    }
    
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Model size (FP32):     {model_size_mb:.2f} MB")
    print(f"Model size (FP16):     {model_size_mb/2:.2f} MB")
    
    # 2. Inference speed for different batch sizes
    print("\n" + "="*80)
    print("2. INFERENCE SPEED (different batch sizes)")
    print("="*80)
    
    results['inference_speed'] = {}
    
    for bs in batch_sizes:
        print(f"\nBatch size: {bs}, Sequence length: {seq_length}")
        metrics = measure_inference_time(model, device, batch_size=bs, seq_length=seq_length)
        
        results['inference_speed'][f'batch_{bs}'] = {
            'batch_size': bs,
            'mean_time_ms': float(metrics['mean_time'] * 1000),
            'std_time_ms': float(metrics['std_time'] * 1000),
            'min_time_ms': float(metrics['min_time'] * 1000),
            'max_time_ms': float(metrics['max_time'] * 1000),
            'throughput_clips_per_sec': float(metrics['fps']),
            'throughput_frames_per_sec': float(metrics['frames_per_sec']),
            'latency_per_clip_ms': float(metrics['mean_time'] * 1000),
            'latency_per_frame_ms': float(metrics['mean_time'] / seq_length * 1000)
        }
        
        print(f"  Mean time:        {metrics['mean_time']*1000:.2f} ms")
        print(f"  Std time:         {metrics['std_time']*1000:.2f} ms")
        print(f"  Min time:         {metrics['min_time']*1000:.2f} ms")
        print(f"  Max time:         {metrics['max_time']*1000:.2f} ms")
        print(f"  Throughput:       {metrics['fps']:.2f} clips/sec")
        print(f"  Frame processing: {metrics['frames_per_sec']:.2f} frames/sec")
        print(f"  Latency/clip:     {metrics['mean_time']*1000:.2f} ms")
        print(f"  Latency/frame:    {metrics['mean_time']/seq_length*1000:.2f} ms")
    
    # 3. Memory usage
    if device.type == 'cuda':
        print("\n" + "="*80)
        print("3. GPU MEMORY USAGE")
        print("="*80)
        
        results['memory_usage'] = {}
        
        for bs in batch_sizes:
            print(f"\nBatch size: {bs}")
            mem_metrics = measure_memory(model, device, batch_size=bs, seq_length=seq_length)
            
            results['memory_usage'][f'batch_{bs}'] = {
                'batch_size': bs,
                'baseline_mb': float(mem_metrics['baseline_memory_mb']),
                'peak_mb': float(mem_metrics['peak_memory_mb']),
                'inference_mb': float(mem_metrics['inference_memory_mb'])
            }
            
            print(f"  Baseline (model): {mem_metrics['baseline_memory_mb']:.2f} MB")
            print(f"  Peak:             {mem_metrics['peak_memory_mb']:.2f} MB")
            print(f"  Inference:        {mem_metrics['inference_memory_mb']:.2f} MB")
    
    # 4. Comparison with baselines
    print("\n" + "="*80)
    print("4. COMPARISON WITH BASELINES")
    print("="*80)
    
    print(f"\nModel:                            Params       FPS (bs=32)    Memory")
    print(f"TransNetV2:                       ~7.0M        ~350           ~200 MB")
    print(f"AutoShot:                         ~12.0M       ~180           ~400 MB")
    print(f"Ours (Hybrid Detector):           {total_params/1e6:.1f}M")
    
    # Re-run with bs=32 for comparison
    if 32 in batch_sizes:
        metrics_32 = measure_inference_time(model, device, batch_size=32, seq_length=seq_length, num_runs=50)
        mem_32 = measure_memory(model, device, batch_size=32, seq_length=seq_length) if device.type == 'cuda' else {'peak_memory_mb': 0}
        
        # results['baseline_comparison'] = {
        #     'ours': {
        #         'parameters_millions': float(total_params / 1e6),
        #         'fps_clips_per_sec': float(metrics_32['fps']),
        #         'fps_frames_per_sec': float(metrics_32['frames_per_sec']),
        #         'memory_mb': float(mem_32['peak_memory_mb']),
        #         'latency_ms_per_clip': float(metrics_32['mean_time'] * 1000)
        #     },
        #     'transnetv2': {
        #         'parameters_millions': 7.0,
        #         'fps_clips_per_sec': 350,
        #         'memory_mb': 200
        #     },
        #     'autoshot': {
        #         'parameters_millions': 12.0,
        #         'fps_clips_per_sec': 180,
        #         'memory_mb': 400
        #     }
        # }
        
        print(f"  - FPS:            {metrics_32['fps']:.1f} clips/sec")
        print(f"  - Frame rate:     {metrics_32['frames_per_sec']:.1f} frames/sec")
        print(f"  - Memory:         {mem_32['peak_memory_mb']:.1f} MB")
        print(f"  - Latency:        {metrics_32['mean_time']*1000:.2f} ms/clip")
    
    # 5. Real-time capability
    print("\n" + "="*80)
    print("5. REAL-TIME CAPABILITY")
    print("="*80)
    
    if 32 in batch_sizes:
        metrics_32 = measure_inference_time(model, device, batch_size=32, seq_length=seq_length, num_runs=50)
        
        # Calculate real-time performance
        batch_size = 32
        video_fps = 30  # Assumed video frame rate
        frames_per_clip = seq_length
        
        # Method 1: Using throughput (most accurate)
        frames_per_sec = metrics_32['frames_per_sec']
        speedup_factor = frames_per_sec / video_fps
        
        # Method 2: Using per-clip processing time
        processing_time_per_clip = metrics_32['mean_time'] / batch_size
        video_duration_per_clip = frames_per_clip / video_fps
        speedup_factor_alt = video_duration_per_clip / processing_time_per_clip
        
        print(f"\nVideo specifications:")
        print(f"  Frame rate:           {video_fps} fps")
        print(f"  Clip duration:        {video_duration_per_clip:.2f} sec ({frames_per_clip} frames)")
        print(f"\nProcessing performance:")
        print(f"  Batch processing:     {metrics_32['mean_time']*1000:.2f} ms for {batch_size} clips")
        print(f"  Per-clip processing:  {processing_time_per_clip*1000:.2f} ms per clip")
        print(f"  Throughput:           {frames_per_sec:.1f} frames/sec")
        print(f"\nReal-time capability:")
        print(f"  Speedup factor:       {speedup_factor:.1f}x real-time")
        print(f"  (Verification:        {speedup_factor_alt:.1f}x)")
        
        if speedup_factor > 1.0:
            print(f"\n✓ Can process video {speedup_factor:.1f}x FASTER than real-time!")
            print(f"  → Can process {int(speedup_factor)} videos simultaneously in real-time")
        else:
            print(f"\n✗ Cannot process real-time (need {1/speedup_factor:.1f}x speedup)")
        
        results['realtime_capability'] = {
            'video_fps': video_fps,
            'frames_per_clip': frames_per_clip,
            'video_duration_per_clip_sec': float(video_duration_per_clip),
            'batch_processing_time_ms': float(metrics_32['mean_time'] * 1000),
            'per_clip_processing_time_ms': float(processing_time_per_clip * 1000),
            'throughput_frames_per_sec': float(frames_per_sec),
            'speedup_factor': float(speedup_factor),
            'can_process_realtime': bool(speedup_factor > 1.0),
            'simultaneous_videos': int(speedup_factor) if speedup_factor > 1.0 else 0
        }
    
    # Save results to JSON
    output_dir = Path('presentation_materials')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'benchmark_statistics.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\n✓ Statistics saved to: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Hybrid Boundary Detector')
    parser.add_argument('--checkpoint', type=str, default='outputs/hybrid_detector4/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[8, 16, 32],
                        help='Batch sizes to test')
    
    args = parser.parse_args()
    
    benchmark_model(args.checkpoint, args.device, args.batch_sizes)
