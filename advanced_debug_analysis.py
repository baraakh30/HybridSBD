"""
Advanced Testing & Debug Analysis Script
=========================================
Analyzes model predictions to identify:
1. What transitions are being missed (type, length, location)
2. Why they're missed (signal strength, expert behavior)
3. How to fix them (temporal context, threshold tuning)
"""

import os
import torch
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from decord import VideoReader, cpu

from detection.turbo_sbd import TurboSBD

# Configuration
CONFIG = {
    'input_root': r"C:\Users\baraa\Documents\ClipShots",
    'model_path': 'outputs/turbo_sbd_best.pth',
    'frame_size': (48, 64),
    'seq_len': 100,
    'stride': 90,  # Minimal overlap (10 frames) for faster processing
    'batch_size': 8,  # Process multiple windows in parallel
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class DetailedAnalyzer:
    def __init__(self):
        self.device = CONFIG['device']
        
        # Load model
        print(f"Loading model from {CONFIG['model_path']}...")
        self.model = TurboSBD(num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(CONFIG['model_path'], map_location=self.device))
        self.model.eval()
        
        # Load annotations
        test_json = os.path.join(CONFIG['input_root'], 'annotations', 'test.json')
        with open(test_json, 'r') as f:
            self.annotations = json.load(f)
        
        # Find video directory
        video_dir_candidates = [
            os.path.join(CONFIG['input_root'], 'videos', 'test'),
            os.path.join(CONFIG['input_root'], 'videos')
        ]
        self.video_dir = None
        for d in video_dir_candidates:
            if os.path.exists(d):
                self.video_dir = d
                break
        
        # Statistics tracking
        self.stats = {
            'total_cuts': 0,
            'detected_cuts': 0,
            'missed_cuts': 0,
            'false_cuts': 0,
            
            'total_graduals': 0,
            'detected_graduals': 0,
            'missed_graduals': 0,
            'false_graduals': 0,
            
            # Gradual breakdown by length
            'gradual_short': {'total': 0, 'detected': 0, 'missed': 0},  # 2-5 frames
            'gradual_medium': {'total': 0, 'detected': 0, 'missed': 0},  # 6-15 frames
            'gradual_long': {'total': 0, 'detected': 0, 'missed': 0},  # 16+ frames
        }
        
        # Detailed miss analysis
        self.missed_transitions = []
        self.false_positives = []
        
    def analyze_video(self, video_name):
        """Analyze a single video with batched sliding windows"""
        video_path = os.path.join(self.video_dir, video_name)
        if not os.path.exists(video_path):
            return
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0), 
                           width=CONFIG['frame_size'][1], 
                           height=CONFIG['frame_size'][0])
            total_frames = len(vr)
            
            # Get ground truth transitions
            transitions = self.annotations[video_name]['transitions']
            gt_labels = np.zeros(total_frames, dtype=np.int32)
            transition_lengths = np.zeros(total_frames, dtype=np.int32)
            
            for start, end in transitions:
                length = end - start
                label = 1 if length <= 1 else 2
                gt_labels[start:end+1] = label
                transition_lengths[start:end+1] = length
            
            # OPTIMIZED: Load all frames at once (cached)
            all_frames = vr.get_batch(list(range(total_frames))).asnumpy()
            all_frames = torch.from_numpy(all_frames).permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]
            
            # Run predictions with batched sliding windows
            pred_labels = np.zeros(total_frames, dtype=np.int32)
            pred_probs = np.zeros((total_frames, 3), dtype=np.float32)
            counts = np.zeros(total_frames, dtype=np.int32)  # For averaging overlaps
            
            with torch.no_grad():
                # Create all windows
                windows = []
                window_starts = []
                
                for start_idx in range(0, total_frames, CONFIG['stride']):
                    end_idx = min(start_idx + CONFIG['seq_len'], total_frames)
                    
                    # Extract window
                    window = all_frames[start_idx:end_idx]
                    if len(window) < CONFIG['seq_len']:
                        # Pad with last frame
                        padding = all_frames[end_idx-1:end_idx].repeat(CONFIG['seq_len'] - len(window), 1, 1, 1)
                        window = torch.cat([window, padding], dim=0)
                    
                    windows.append(window)
                    window_starts.append(start_idx)
                    
                    # Process in batches
                    if len(windows) == CONFIG['batch_size'] or start_idx + CONFIG['stride'] >= total_frames:
                        # Stack and predict
                        batch_frames = torch.stack(windows).to(self.device)  # [B, T, 3, H, W]
                        logits = self.model(batch_frames)  # [B, T, 3]
                        probs = torch.softmax(logits, dim=2).cpu().numpy()
                        
                        # Accumulate predictions
                        for b_idx, start in enumerate(window_starts):
                            valid_len = min(total_frames - start, CONFIG['seq_len'])
                            for i in range(valid_len):
                                frame_idx = start + i
                                pred_probs[frame_idx] += probs[b_idx, i]
                                counts[frame_idx] += 1
                        
                        # Clear batch
                        windows = []
                        window_starts = []
                
                # Average probabilities and get final predictions
                for i in range(total_frames):
                    if counts[i] > 0:
                        pred_probs[i] /= counts[i]
                    pred_labels[i] = np.argmax(pred_probs[i])
            
            # Analyze predictions vs ground truth (pass cached frames)
            self._analyze_predictions(video_name, gt_labels, pred_labels, 
                                    transition_lengths, pred_probs, all_frames)
            
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
    
    def _analyze_predictions(self, video_name, gt_labels, pred_labels, 
                           transition_lengths, pred_probs, all_frames):
        """Detailed analysis of predictions vs ground truth"""
        
        # Find all ground truth transitions
        transitions = []
        i = 0
        while i < len(gt_labels):
            if gt_labels[i] > 0:
                start = i
                label = gt_labels[i]
                while i < len(gt_labels) and gt_labels[i] == label:
                    i += 1
                end = i - 1
                transitions.append((start, end, label, transition_lengths[start]))
            else:
                i += 1
        
        # Check each transition
        for start, end, label, length in transitions:
            transition_type = 'Cut' if label == 1 else 'Gradual'
            
            # Check if detected (any frame in range predicted correctly)
            detected = np.any(pred_labels[start:end+1] == label)
            
            # Update stats
            if label == 1:
                self.stats['total_cuts'] += 1
                if detected:
                    self.stats['detected_cuts'] += 1
                else:
                    self.stats['missed_cuts'] += 1
            else:
                self.stats['total_graduals'] += 1
                
                # Categorize by length
                if length <= 5:
                    category = 'gradual_short'
                elif length <= 15:
                    category = 'gradual_medium'
                else:
                    category = 'gradual_long'
                
                self.stats[category]['total'] += 1
                
                if detected:
                    self.stats['detected_graduals'] += 1
                    self.stats[category]['detected'] += 1
                else:
                    self.stats['missed_graduals'] += 1
                    self.stats[category]['missed'] += 1
                    
                    # Detailed miss analysis (use cached frames)
                    self._analyze_miss(video_name, all_frames, start, end, length, 
                                     pred_probs, pred_labels)
    
    def _analyze_miss(self, video_name, all_frames, start, end, length, 
                     pred_probs, pred_labels):
        """Deep dive into why a gradual transition was missed"""
        
        # Extract probabilities around the transition
        context_before = 10
        context_after = 10
        window_start = max(0, start - context_before)
        window_end = min(len(pred_probs), end + context_after + 1)
        
        window_probs = pred_probs[window_start:window_end]
        window_preds = pred_labels[window_start:window_end]
        
        # Analyze what happened
        max_gradual_prob = np.max(window_probs[:, 2])  # Max P(Gradual)
        avg_gradual_prob = np.mean(window_probs[start-window_start:end-window_start+1, 2])
        
        # What was predicted instead?
        predicted_classes = window_preds[start-window_start:end-window_start+1]
        pred_none = np.sum(predicted_classes == 0)
        pred_cut = np.sum(predicted_classes == 1)
        pred_gradual = np.sum(predicted_classes == 2)
        
        # Compute signal strength from cached frames (much faster!)
        try:
            frames = all_frames[window_start:window_end].numpy()  # [T, 3, H, W]
            
            # Compute diff1
            diff1 = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2, 3))
            diff1 = np.concatenate([[0], diff1])  # Pad first frame
            
            transition_diff1 = diff1[start-window_start:end-window_start+1]
            avg_signal = np.mean(transition_diff1)
            max_signal = np.max(transition_diff1)
            
        except:
            avg_signal = 0
            max_signal = 0
        
        # Store detailed miss info
        miss_info = {
            'video': video_name,
            'start': start,
            'end': end,
            'length': length,
            'max_p_gradual': max_gradual_prob,
            'avg_p_gradual': avg_gradual_prob,
            'predicted_none': pred_none,
            'predicted_cut': pred_cut,
            'predicted_gradual': pred_gradual,
            'avg_signal': avg_signal,
            'max_signal': max_signal,
            'failure_mode': self._classify_failure(max_gradual_prob, avg_gradual_prob, 
                                                   pred_cut, pred_none, avg_signal)
        }
        
        self.missed_transitions.append(miss_info)
    
    def _classify_failure(self, max_p_grad, avg_p_grad, pred_cut, pred_none, avg_signal):
        """Classify why the detection failed"""
        
        if max_p_grad < 0.2:
            # Model never even considered it a gradual
            if avg_signal < 0.01:
                return "VERY_WEAK_SIGNAL"  # Genuinely subtle
            else:
                return "WRONG_EXPERT"  # Predicted as cut/none despite signal
        
        elif max_p_grad < 0.4:
            return "LOW_CONFIDENCE"  # Model unsure
        
        elif pred_cut > pred_none:
            return "CONFUSED_WITH_CUT"  # Thought it was a cut
        
        else:
            return "THRESHOLD_ISSUE"  # P(Grad) high but not winning
    
    def run_analysis(self, max_videos=None):
        """Run analysis on test set"""
        print(f"\nAnalyzing test set on {self.device}...")
        
        video_names = list(self.annotations.keys())
        if max_videos:
            video_names = video_names[:max_videos]
        
        for video_name in tqdm(video_names, desc="Analyzing videos"):
            self.analyze_video(video_name)
        
        self._print_report()
    
    def _print_report(self):
        """Print comprehensive analysis report"""
        print("\n" + "="*80)
        print("ADVANCED DEBUG ANALYSIS REPORT")
        print("="*80)
        
        # Overall stats
        print("\n📊 OVERALL STATISTICS")
        print("-" * 80)
        
        total_transitions = self.stats['total_cuts'] + self.stats['total_graduals']
        total_detected = self.stats['detected_cuts'] + self.stats['detected_graduals']
        overall_recall = total_detected / total_transitions if total_transitions > 0 else 0
        
        print(f"Total Transitions: {total_transitions}")
        print(f"Total Detected: {total_detected} ({overall_recall*100:.1f}% recall)")
        
        # Cut analysis
        print(f"\n🔪 CUTS: {self.stats['total_cuts']} total")
        if self.stats['total_cuts'] > 0:
            cut_recall = self.stats['detected_cuts'] / self.stats['total_cuts']
            print(f"  ✓ Detected: {self.stats['detected_cuts']} ({cut_recall*100:.1f}%)")
            print(f"  ✗ Missed: {self.stats['missed_cuts']} ({(1-cut_recall)*100:.1f}%)")
        
        # Gradual analysis by length
        print(f"\n🌊 GRADUALS: {self.stats['total_graduals']} total")
        if self.stats['total_graduals'] > 0:
            grad_recall = self.stats['detected_graduals'] / self.stats['total_graduals']
            print(f"  ✓ Detected: {self.stats['detected_graduals']} ({grad_recall*100:.1f}%)")
            print(f"  ✗ Missed: {self.stats['missed_graduals']} ({(1-grad_recall)*100:.1f}%)")
            
            print("\n  Breakdown by Length:")
            for cat, label in [('gradual_short', 'Short (2-5f)'),
                              ('gradual_medium', 'Medium (6-15f)'),
                              ('gradual_long', 'Long (16+f)')]:
                total = self.stats[cat]['total']
                if total > 0:
                    detected = self.stats[cat]['detected']
                    missed = self.stats[cat]['missed']
                    recall = detected / total
                    print(f"    {label:18s}: {detected:4d}/{total:4d} detected "
                          f"({recall*100:5.1f}% recall, {missed:3d} missed)")
        
        # Failure mode analysis
        if self.missed_transitions:
            print("\n" + "="*80)
            print("🔍 DETAILED MISS ANALYSIS")
            print("="*80)
            
            # Group by failure mode
            failure_modes = defaultdict(list)
            for miss in self.missed_transitions:
                failure_modes[miss['failure_mode']].append(miss)
            
            print(f"\nTotal Missed Graduals: {len(self.missed_transitions)}")
            print("\nFailure Modes:")
            
            for mode, misses in sorted(failure_modes.items(), 
                                      key=lambda x: len(x[1]), 
                                      reverse=True):
                print(f"\n  {mode}: {len(misses)} cases ({len(misses)/len(self.missed_transitions)*100:.1f}%)")
                
                # Compute statistics for this failure mode
                avg_length = np.mean([m['length'] for m in misses])
                avg_max_p = np.mean([m['max_p_gradual'] for m in misses])
                avg_signal = np.mean([m['avg_signal'] for m in misses])
                
                print(f"    Avg Length: {avg_length:.1f} frames")
                print(f"    Avg Max P(Gradual): {avg_max_p:.3f}")
                print(f"    Avg Signal Strength: {avg_signal:.4f}")
                
                # Show top 3 examples
                print(f"    Examples:")
                for i, miss in enumerate(misses[:3]):
                    print(f"      {i+1}. {miss['video']} frames {miss['start']}-{miss['end']} "
                          f"(len={miss['length']}, max_p={miss['max_p_gradual']:.3f})")
            
            # Recommendations
            print("\n" + "="*80)
            print("💡 RECOMMENDATIONS")
            print("="*80)
            
            # Analyze what needs fixing
            very_weak = len(failure_modes.get('VERY_WEAK_SIGNAL', []))
            wrong_expert = len(failure_modes.get('WRONG_EXPERT', []))
            low_conf = len(failure_modes.get('LOW_CONFIDENCE', []))
            confused = len(failure_modes.get('CONFUSED_WITH_CUT', []))
            threshold = len(failure_modes.get('THRESHOLD_ISSUE', []))
            
            total_misses = len(self.missed_transitions)
            
            if very_weak / total_misses > 0.3:
                print("\n⚠️  HIGH: Many very weak signals (>30%)")
                print("   → These may be annotation errors or genuinely imperceptible transitions")
                print("   → Consider: More aggressive pooling, lower detection threshold")
            
            if wrong_expert / total_misses > 0.2:
                print("\n⚠️  HIGH: Wrong expert activation (>20%)")
                print("   → Gradual expert not firing despite visible signal")
                print("   → Consider: Adjust expert routing, increase gradual temporal context")
            
            if confused / total_misses > 0.15:
                print("\n⚠️  MEDIUM: Confused with cuts (>15%)")
                print("   → Model can't distinguish short graduals from cuts")
                print("   → Consider: Better temporal smoothness features, ratio analysis")
            
            if low_conf / total_misses > 0.25:
                print("\n⚠️  MEDIUM: Low confidence (>25%)")
                print("   → Model unsure, P(Gradual) barely rising")
                print("   → Consider: Longer temporal pooling windows, stronger gradual boosting")
            
            if threshold / total_misses > 0.15:
                print("\n⚠️  MEDIUM: Threshold issues (>15%)")
                print("   → P(Gradual) high but not winning")
                print("   → Consider: Adjust class weights, lower classification threshold")
            
            # Length-specific recommendations
            print("\n📏 Length-Specific Issues:")
            
            for cat, label, suggestion in [
                ('gradual_short', 'Short (2-5f)', 
                 'Increase temporal pooling to 7-9 frames, improve onset detection'),
                ('gradual_medium', 'Medium (6-15f)', 
                 'Current 9-frame pooling should work, check expert routing'),
                ('gradual_long', 'Long (16+f)', 
                 'Increase long pooling to 25+ frames, add multi-scale fusion')
            ]:
                if self.stats[cat]['total'] > 0:
                    miss_rate = self.stats[cat]['missed'] / self.stats[cat]['total']
                    if miss_rate > 0.20:
                        print(f"\n   {label}: {miss_rate*100:.1f}% miss rate")
                        print(f"   → {suggestion}")

def main():
    analyzer = DetailedAnalyzer()
    
    # Run on all test videos (or specify max_videos for quick test)
    analyzer.run_analysis(max_videos=None)  # Set to e.g. 50 for quick test

if __name__ == '__main__':
    main()
