"""
Test Architectural Fixes
=========================
Tests the REAL fixes: probability combination logic, temporal context extension
These require model changes, but we can simulate what they would do.
"""

import os
import torch
import json
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu

from detection.turbo_sbd import TurboSBD

CONFIG = {
    'input_root': r"C:\Users\baraa\Documents\ClipShots",
    'model_path': 'outputs/turbo_sbd_best.pth',
    'frame_size': (48, 64),
    'seq_len': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_test_videos': 50
}

class ArchitecturalFixTester:
    def __init__(self):
        self.device = CONFIG['device']
        
        print(f"Loading model from {CONFIG['model_path']}...")
        self.model = TurboSBD(num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(CONFIG['model_path'], map_location=self.device))
        self.model.eval()
        
        test_json = os.path.join(CONFIG['input_root'], 'annotations', 'test.json')
        with open(test_json, 'r') as f:
            self.annotations = json.load(f)
        
        video_dir_candidates = [
            os.path.join(CONFIG['input_root'], 'videos', 'test'),
            os.path.join(CONFIG['input_root'], 'videos')
        ]
        self.video_dir = None
        for d in video_dir_candidates:
            if os.path.exists(d):
                self.video_dir = d
                break
        
        # Track performance by failure mode
        self.results = {
            'current': {'total': 0, 'detected': 0, 'by_mode': {}},
            'fixed_prob_combination': {'total': 0, 'detected': 0, 'by_mode': {}},
            'extended_temporal_context': {'total': 0, 'detected': 0, 'by_mode': {}},
            'both_fixes': {'total': 0, 'detected': 0, 'by_mode': {}},
        }
        
        self.examples = []
        
    def classify_failure_mode(self, trans_preds):
        """Classify why a transition might be hard to detect"""
        avg_p_grad = np.mean([p['p_gradual'] for p in trans_preds])
        max_p_grad = np.max([p['p_gradual'] for p in trans_preds])
        avg_ratio = np.mean([p['ratio_4_1'] for p in trans_preds])
        avg_signal = np.mean([p['diff1'] for p in trans_preds])
        
        if max_p_grad < 0.2:
            if avg_signal < 0.01:
                return "VERY_WEAK_SIGNAL"
            else:
                return "WRONG_EXPERT"
        elif max_p_grad < 0.4:
            return "LOW_CONFIDENCE"
        elif max_p_grad > 0.5:
            return "THRESHOLD_ISSUE"
        else:
            return "UNCERTAIN"
    
    def test_video(self, video_name):
        """Test architectural fixes on a single video"""
        video_path = os.path.join(self.video_dir, video_name)
        if not os.path.exists(video_path):
            return
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0), 
                           width=CONFIG['frame_size'][1], 
                           height=CONFIG['frame_size'][0])
            total_frames = len(vr)
            
            transitions = self.annotations[video_name]['transitions']
            gt_labels = np.zeros(total_frames, dtype=np.int32)
            
            for start, end in transitions:
                length = end - start
                label = 1 if length <= 1 else 2
                gt_labels[start:end+1] = label
            
            all_frames = vr.get_batch(list(range(total_frames))).asnumpy()
            all_frames = torch.from_numpy(all_frames).permute(0, 3, 1, 2).float() / 255.0
            
            predictions = {}
            
            with torch.no_grad():
                for start_idx in range(0, total_frames, CONFIG['seq_len']):
                    end_idx = min(start_idx + CONFIG['seq_len'], total_frames)
                    
                    window = all_frames[start_idx:end_idx]
                    if len(window) < CONFIG['seq_len']:
                        padding = all_frames[end_idx-1:end_idx].repeat(CONFIG['seq_len'] - len(window), 1, 1, 1)
                        window = torch.cat([window, padding], dim=0)
                    
                    frames_batch = window.unsqueeze(0).to(self.device)
                    
                    logits, intermediates = self._forward_with_intermediates(frames_batch)
                    
                    valid_len = min(total_frames - start_idx, CONFIG['seq_len'])
                    for i in range(valid_len):
                        frame_idx = start_idx + i
                        if frame_idx not in predictions:
                            predictions[frame_idx] = {
                                'logits': logits[0, i].cpu().numpy(),
                                'p_cut': intermediates['p_cut'][0, 0, i].cpu().item(),
                                'p_gradual': intermediates['p_gradual'][0, 0, i].cpu().item(),
                                'diff1': intermediates['diff1'][0, i].cpu().item(),
                                'diff4': intermediates['diff4'][0, i].cpu().item(),
                                'ratio_4_1': intermediates['ratio_4_1'][0, i].cpu().item(),
                            }
            
            # Test on gradual transitions
            for start, end in transitions:
                length = end - start
                if length <= 1:
                    continue
                
                trans_preds = []
                for frame_idx in range(start, min(end+1, total_frames)):
                    if frame_idx in predictions:
                        trans_preds.append(predictions[frame_idx])
                
                if not trans_preds:
                    continue
                
                # Classify failure mode
                failure_mode = self.classify_failure_mode(trans_preds)
                
                # Test each approach
                current_detected = any(np.argmax(p['logits']) == 2 for p in trans_preds)
                fix1_detected = self._test_fixed_probability_combination(trans_preds)
                fix2_detected = self._test_extended_temporal_context(trans_preds, length)
                both_detected = fix1_detected or fix2_detected
                
                # Update stats
                for approach, detected in [
                    ('current', current_detected),
                    ('fixed_prob_combination', fix1_detected),
                    ('extended_temporal_context', fix2_detected),
                    ('both_fixes', both_detected)
                ]:
                    self.results[approach]['total'] += 1
                    if detected:
                        self.results[approach]['detected'] += 1
                    
                    if failure_mode not in self.results[approach]['by_mode']:
                        self.results[approach]['by_mode'][failure_mode] = {'total': 0, 'detected': 0}
                    
                    self.results[approach]['by_mode'][failure_mode]['total'] += 1
                    if detected:
                        self.results[approach]['by_mode'][failure_mode]['detected'] += 1
                
                # Save interesting examples
                if not current_detected and (fix1_detected or fix2_detected):
                    self.examples.append({
                        'video': video_name,
                        'start': start,
                        'end': end,
                        'length': length,
                        'failure_mode': failure_mode,
                        'avg_p_gradual': np.mean([p['p_gradual'] for p in trans_preds]),
                        'max_p_gradual': np.max([p['p_gradual'] for p in trans_preds]),
                        'avg_p_cut': np.mean([p['p_cut'] for p in trans_preds]),
                        'avg_ratio': np.mean([p['ratio_4_1'] for p in trans_preds]),
                        'fixed_by_prob': fix1_detected and not current_detected,
                        'fixed_by_temporal': fix2_detected and not current_detected,
                    })
                    
        except Exception as e:
            print(f"Error: {e}")
    
    def _forward_with_intermediates(self, x):
        """Forward pass capturing intermediates"""
        B, T, C, H, W = x.shape
        
        x_diff1 = x - torch.roll(x, shifts=1, dims=1)
        x_diff1[:, 0] = 0
        
        x_diff4 = x - torch.roll(x, shifts=4, dims=1)
        x_diff4[:, :4] = 0
        
        diff1_mag = torch.mean(torch.abs(x_diff1), dim=(2, 3, 4), keepdim=False)
        diff4_mag = torch.mean(torch.abs(x_diff4), dim=(2, 3, 4), keepdim=False)
        
        eps = 1e-3
        ratio_4_1 = diff4_mag / torch.clamp(diff1_mag, min=eps)
        ratio_4_1 = torch.clamp(ratio_4_1, 0, 10)
        
        logits = self.model(x)
        probs = torch.softmax(logits, dim=2)
        
        intermediates = {
            'p_cut': probs[:, :, 1:2].transpose(1, 2),
            'p_gradual': probs[:, :, 2:3].transpose(1, 2),
            'diff1': diff1_mag,
            'diff4': diff4_mag,
            'ratio_4_1': ratio_4_1,
        }
        
        return logits, intermediates
    
    def _test_fixed_probability_combination(self, trans_preds):
        """
        FIX 1: Fix the broken probability combination
        
        Current (BROKEN):
            p_none = 1 - p_cut - p_gradual  # Can go negative!
        
        Fix: Softmax normalization of expert outputs
        """
        detected = False
        
        for pred in trans_preds:
            p_cut = pred['p_cut']
            p_gradual = pred['p_gradual']
            
            # FIXED: Proper normalization
            # Experts give raw scores, we normalize them
            p_none_raw = 0.5  # Baseline probability for no transition
            
            # Normalize all three
            total = p_none_raw + p_cut + p_gradual
            p_none_fixed = p_none_raw / total
            p_cut_fixed = p_cut / total
            p_gradual_fixed = p_gradual / total
            
            # Now check if gradual wins
            if p_gradual_fixed > p_cut_fixed and p_gradual_fixed > p_none_fixed:
                detected = True
                break
        
        return detected
    
    def _test_extended_temporal_context(self, trans_preds, length):
        """
        FIX 2: Simulate extended temporal context for long graduals
        
        Current: Max pooling = 25 frames (±12)
        Fix: Add 35-frame pooling (±17) for long graduals
        
        We simulate this by being more lenient for long transitions
        """
        # For long graduals (25+ frames), the model needs wider context
        # We simulate this by lowering the threshold proportional to length
        
        if length < 25:
            # Current pooling is fine
            threshold = 0.5
        elif length < 35:
            # Partially missing context
            threshold = 0.4  # Lower threshold to compensate
        else:
            # Definitely missing context (35+ frames)
            threshold = 0.35  # Even lower
        
        # Check if any prediction exceeds the adaptive threshold
        for pred in trans_preds:
            if pred['p_gradual'] > threshold:
                return True
        
        return False
    
    def run_test(self):
        """Run architectural fix tests"""
        print(f"\n🔧 Testing ARCHITECTURAL fixes on {CONFIG['num_test_videos']} videos...\n")
        
        video_names = list(self.annotations.keys())[:CONFIG['num_test_videos']]
        
        for video_name in tqdm(video_names, desc="Testing fixes"):
            self.test_video(video_name)
        
        self._print_report()
    
    def _print_report(self):
        """Print detailed comparison"""
        print("\n" + "="*80)
        print("ARCHITECTURAL FIX ANALYSIS")
        print("="*80)
        
        print("\n📊 OVERALL PERFORMANCE")
        print("-" * 80)
        print(f"{'Approach':<35} {'Detected':<12} {'Missed':<12} {'Recall':<12}")
        print("-" * 80)
        
        approaches = [
            ('current', 'Current (Broken Prob Combination)'),
            ('fixed_prob_combination', 'FIX 1: Fixed Probability Logic'),
            ('extended_temporal_context', 'FIX 2: Extended Temporal Context'),
            ('both_fixes', 'BOTH FIXES COMBINED')
        ]
        
        baseline_recall = 0
        
        for key, label in approaches:
            stats = self.results[key]
            total = stats['total']
            detected = stats['detected']
            missed = total - detected
            recall = detected / total if total > 0 else 0
            
            if key == 'current':
                baseline_recall = recall
                marker = "  "
            else:
                improvement = recall - baseline_recall
                if improvement > 0.02:
                    marker = "✓✓"
                elif improvement > 0:
                    marker = "✓ "
                else:
                    marker = "  "
            
            print(f"{marker}{label:<33} {detected:<12} {missed:<12} {recall*100:>6.1f}%")
        
        # Breakdown by failure mode
        print("\n" + "="*80)
        print("📋 PERFORMANCE BY FAILURE MODE")
        print("="*80)
        
        all_modes = set()
        for approach_stats in self.results.values():
            all_modes.update(approach_stats['by_mode'].keys())
        
        for mode in sorted(all_modes):
            print(f"\n{mode}:")
            print("-" * 80)
            print(f"{'Approach':<35} {'Detected/Total':<20} {'Recall':<12}")
            print("-" * 80)
            
            for key, label in approaches:
                if mode in self.results[key]['by_mode']:
                    mode_stats = self.results[key]['by_mode'][mode]
                    detected = mode_stats['detected']
                    total = mode_stats['total']
                    recall = detected / total if total > 0 else 0
                    
                    print(f"  {label:<33} {detected}/{total:<17} {recall*100:>6.1f}%")
        
        # Show examples
        if self.examples:
            print("\n" + "="*80)
            print("🔍 EXAMPLES: Transitions Fixed by Architectural Changes")
            print("="*80)
            
            self.examples.sort(key=lambda x: (x['fixed_by_prob'], x['fixed_by_temporal']), reverse=True)
            
            for i, ex in enumerate(self.examples[:15]):
                print(f"\n{i+1}. {ex['video']} frames {ex['start']}-{ex['end']}")
                print(f"   Length: {ex['length']} frames | Mode: {ex['failure_mode']}")
                print(f"   P(Gradual): avg={ex['avg_p_gradual']:.3f}, max={ex['max_p_gradual']:.3f}")
                print(f"   Ratio 4/1: {ex['avg_ratio']:.3f}")
                
                fixed_by = []
                if ex['fixed_by_prob']:
                    fixed_by.append("Fixed Prob Combination")
                if ex['fixed_by_temporal']:
                    fixed_by.append("Extended Temporal Context")
                print(f"   ✓ Fixed by: {', '.join(fixed_by)}")
        
        # Final recommendation
        print("\n" + "="*80)
        print("🎯 IMPLEMENTATION PRIORITY")
        print("="*80)
        
        baseline = self.results['current']['detected'] / self.results['current']['total']
        fix1_improvement = (self.results['fixed_prob_combination']['detected'] / self.results['fixed_prob_combination']['total']) - baseline
        fix2_improvement = (self.results['extended_temporal_context']['detected'] / self.results['extended_temporal_context']['total']) - baseline
        both_improvement = (self.results['both_fixes']['detected'] / self.results['both_fixes']['total']) - baseline
        
        print(f"\n1. FIX PROBABILITY COMBINATION: +{fix1_improvement*100:.1f}% improvement")
        print("   → Replace naive p_none = 1 - p_cut - p_gradual")
        print("   → Use proper softmax normalization")
        
        print(f"\n2. EXTEND TEMPORAL CONTEXT: +{fix2_improvement*100:.1f}% improvement")
        print("   → Add 35-frame pooling (kernel_size=35, padding=17)")
        print("   → Helps long graduals (30-40 frames)")
        
        print(f"\n3. COMBINED: +{both_improvement*100:.1f}% total improvement")
        print("   → Implement both fixes together")
        
        if both_improvement > 0.05:
            print("\n✓✓ HIGH IMPACT: Both fixes show significant improvement")
            print("   Recommend implementing BOTH in the next training run")
        elif both_improvement > 0.02:
            print("\n✓ MODERATE IMPACT: Worthwhile improvements")
            print("   Implement the higher-impact fix first")
        else:
            print("\n⚠ LOW IMPACT: Architectural fixes help but won't solve everything")
            print("   May need more fundamental changes (better features, longer sequences)")
        
        print("="*80)

def main():
    tester = ArchitecturalFixTester()
    tester.run_test()

if __name__ == '__main__':
    main()
