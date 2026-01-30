import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def generate_dissolve(seq_len=100, transition_len=20, height=48, width=64):
    """
    Generates a synthetic dissolve transition between two random noise patterns (or solid colors).
    """
    # Scene A: Random Pattern A
    scene_a = torch.rand(1, 1, 3, height, width) * 0.5 # Darker
    # Scene B: Random Pattern B
    scene_b = torch.rand(1, 1, 3, height, width) * 0.5 + 0.5 # Brighter
    
    frames = torch.zeros(1, seq_len, 3, height, width)
    
    start = (seq_len - transition_len) // 2
    end = start + transition_len
    
    for t in range(seq_len):
        if t < start:
            frames[:, t] = scene_a
        elif t >= end:
            frames[:, t] = scene_b
        else:
            # Linear Blend
            alpha = (t - start) / float(transition_len)
            frames[:, t] = scene_a * (1 - alpha) + scene_b * alpha
            
    return frames, start, end

def analyze_signals():
    print("Generating Synthetic Dissolve (20 frames)...")
    frames, start, end = generate_dissolve()
    
    # 1. Calculate 1-step difference (Current Architecture)
    diff1 = frames - torch.roll(frames, shifts=1, dims=1)
    diff1[:, 0] = 0
    mag1 = diff1.abs().mean(dim=(2, 3, 4)).squeeze() # [T]
    
    # 2. Calculate 4-step difference (Proposed)
    diff4 = frames - torch.roll(frames, shifts=4, dims=1)
    diff4[:, :4] = 0
    mag4 = diff4.abs().mean(dim=(2, 3, 4)).squeeze() # [T]
    
    # 3. Calculate 8-step difference
    diff8 = frames - torch.roll(frames, shifts=8, dims=1)
    diff8[:, :8] = 0
    mag8 = diff8.abs().mean(dim=(2, 3, 4)).squeeze() # [T]
    
    # Print stats in the transition region vs noise
    # Since inputs are static scenes + dissolve, noise is 0 here. 
    # Let's add noise to simulate reality.
    noise = torch.randn_like(frames) * 0.02 # 2% noise
    frames_noisy = frames + noise
    
    diff1_n = frames_noisy - torch.roll(frames_noisy, shifts=1, dims=1)
    diff4_n = frames_noisy - torch.roll(frames_noisy, shifts=4, dims=1)
    
    mag1_n = diff1_n.abs().mean(dim=(2, 3, 4)).squeeze()
    mag4_n = diff4_n.abs().mean(dim=(2, 3, 4)).squeeze()
    
    # Define Regions
    pre = slice(0, start-5)
    trans = slice(start, end)
    
    print("\n--- Signal Analysis (Values) ---")
    print(f"Noise Level: 0.02 (Pixel scale 0-1)")
    print(f"Transition Length: 20 frames")
    
    print(f"\n1. Single-Step Difference (Current Model):")
    print(f"   Avg Magnitude (Static Region): {mag1_n[pre].mean():.4f}")
    print(f"   Avg Magnitude (Transition):    {mag1_n[trans].mean():.4f}")
    print(f"   Signal-to-Noise Ratio:         {mag1_n[trans].mean() / mag1_n[pre].mean():.2f}")
    
    print(f"\n2. 4-Step Difference (Broader Context):")
    print(f"   Avg Magnitude (Static Region): {mag4_n[pre].mean():.4f}")
    print(f"   Avg Magnitude (Transition):    {mag4_n[trans].mean():.4f}")
    print(f"   Signal-to-Noise Ratio:         {mag4_n[trans].mean() / mag4_n[pre].mean():.2f}")

    print("\n--- Conclusion ---")
    if (mag4_n[trans].mean() / mag4_n[pre].mean()) > (mag1_n[trans].mean() / mag1_n[pre].mean()):
        print(">> Multi-step differences capture significantly more context for gradual transitions.")
        print(">> Recommend updating architecture to include diff4 or diff8.")
    else:
        print(">> Single-step difference is sufficient.")

if __name__ == "__main__":
    analyze_signals()
