"""
Analyze ClipShots annotations to understand transition patterns
"""
import json
import glob
import os
import numpy as np
from collections import defaultdict

def analyze_annotations(input_root):
    stats = {
        'total_videos': 0,
        'total_transitions': 0,
        'cut_count': 0,
        'gradual_count': 0,
        'cut_lengths': [],
        'gradual_lengths': [],
        'transitions_per_video': [],
    }
    
    # Check both train and test
    for split in ['train', 'test']:
        json_path = os.path.join(input_root, 'annotations', f'{split}.json')
        
        if not os.path.exists(json_path):
            print(f"❌ {json_path} not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Analyzing {split.upper()} split: {json_path}")
        print(f"{'='*70}")
        
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        for vid_name, data in annotations.items():
            stats['total_videos'] += 1
            
            # Get transitions list
            transitions = data.get('transitions', [])
            
            stats['transitions_per_video'].append(len(transitions))
            stats['total_transitions'] += len(transitions)
            
            for transition in transitions:
                start = transition[0]
                end = transition[1]
                length = end - start
                
                # Heuristic: length <= 1 is Cut, otherwise Gradual
                if length <= 1:
                    stats['cut_count'] += 1
                    stats['cut_lengths'].append(length)
                else:
                    stats['gradual_count'] += 1
                    stats['gradual_lengths'].append(length)
        
        print(f"✓ Processed {stats['total_videos']} videos in {split}")
    
    return stats

def print_stats(stats):
    print(f"\n{'='*70}")
    print("CLIPSHOTS DATASET STATISTICS")
    print(f"{'='*70}")
    
    print(f"\nOverall:")
    print(f"  Total Videos:        {stats['total_videos']:,}")
    print(f"  Total Transitions:   {stats['total_transitions']:,}")
    print(f"  Avg Trans/Video:     {np.mean(stats['transitions_per_video']):.1f}")
    
    print(f"\nTransition Type Distribution:")
    total = stats['cut_count'] + stats['gradual_count']
    print(f"  Cuts:                {stats['cut_count']:,} ({100*stats['cut_count']/total:.1f}%)")
    print(f"  Graduals:            {stats['gradual_count']:,} ({100*stats['gradual_count']/total:.1f}%)")
    print(f"  Cut/Gradual Ratio:   {stats['cut_count']/max(stats['gradual_count'], 1):.2f}:1")
    
    if stats['cut_lengths']:
        print(f"\nCut Transition Lengths (frames):")
        print(f"  Min:    {min(stats['cut_lengths'])}")
        print(f"  Max:    {max(stats['cut_lengths'])}")
        print(f"  Mean:   {np.mean(stats['cut_lengths']):.2f}")
        print(f"  Median: {np.median(stats['cut_lengths']):.0f}")
        
        # Distribution
        unique, counts = np.unique(stats['cut_lengths'], return_counts=True)
        print(f"  Distribution:")
        for length, count in zip(unique[:10], counts[:10]):
            print(f"    Length {length}: {count:,} ({100*count/len(stats['cut_lengths']):.1f}%)")
    
    if stats['gradual_lengths']:
        print(f"\nGradual Transition Lengths (frames):")
        print(f"  Min:    {min(stats['gradual_lengths'])}")
        print(f"  Max:    {max(stats['gradual_lengths'])}")
        print(f"  Mean:   {np.mean(stats['gradual_lengths']):.2f}")
        print(f"  Median: {np.median(stats['gradual_lengths']):.0f}")
        
        # Percentiles
        print(f"  Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(stats['gradual_lengths'], p)
            print(f"    {p}th: {val:.0f} frames")
        
        # Distribution by bins
        bins = [0, 5, 10, 15, 20, 30, 50, 100, 1000]
        hist, _ = np.histogram(stats['gradual_lengths'], bins=bins)
        print(f"  Distribution by length:")
        for i in range(len(bins)-1):
            count = hist[i]
            print(f"    {bins[i]}-{bins[i+1]} frames: {count:,} ({100*count/len(stats['gradual_lengths']):.1f}%)")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    input_root = r"C:\Users\baraa\Documents\ClipShots"
    
    stats = analyze_annotations(input_root)
    print_stats(stats)
