"""
Fix Strategy Testing Script
============================
Loads the current best model and tests different fixing strategies on real data
to see which ones actually help before implementing them.
"""

import os
import torch
import json
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu

from detection.turbo_sbd import TurboSBD

# Configuration
CONFIG = {
    'input_root': r"C:\Users\baraa\Documents\ClipShots",
    'model_path': 'outputs/turbo_sbd_best.pth',
    'frame_size': (48, 64),
    'seq_len': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_test_videos': 50  # Test on subset for speed
}

class FixStrategyTester:
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
        
        # Statistics for each strategy
        self.strategies = {
            'current': {'correct': 0, 'missed': 0, 'false_pos': 0},
            'gated_routing': {'correct': 0, 'missed': 0, 'false_pos': 0},
            'confidence_threshold': {'correct': 0, 'missed': 0, 'false_pos': 0},
            'boosted_weight': {'correct': 0, 'missed': 0, 'false_pos': 0},
            'combined': {'correct': 0, 'missed': 0, 'false_pos': 0}
        }
        
        # Detailed examples
        self.examples = []
        
    def test_video(self, video_name):
        """Test all strategies on a single video"""
        video_path = os.path.join(self.video_dir, video_name)
        if not os.path.exists(video_path):
            return
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0), 
                           width=CONFIG['frame_size'][1], 
                           height=CONFIG['frame_size'][0])
            total_frames = len(vr)
            
            # Get ground truth
            transitions = self.annotations[video_name]['transitions']
            gt_labels = np.zeros(total_frames, dtype=np.int32)
            
            for start, end in transitions:
                length = end - start
                label = 1 if length <= 1 else 2
                gt_labels[start:end+1] = label
            
            # Process video in windows
            all_frames = vr.get_batch(list(range(total_frames))).asnumpy()
            all_frames = torch.from_numpy(all_frames).permute(0, 3, 1, 2).float() / 255.0
            
            # Run model to get expert outputs (we need to hook intermediate outputs)
            predictions = {}
            
            with torch.no_grad():
                for start_idx in range(0, total_frames, CONFIG['seq_len']):
                    end_idx = min(start_idx + CONFIG['seq_len'], total_frames)
                    
                    window = all_frames[start_idx:end_idx]
                    if len(window) < CONFIG['seq_len']:
                        padding = all_frames[end_idx-1:end_idx].repeat(CONFIG['seq_len'] - len(window), 1, 1, 1)
                        window = torch.cat([window, padding], dim=0)
                    
                    frames_batch = window.unsqueeze(0).to(self.device)  # [1, T, 3, H, W]
                    
                    # Get model predictions and intermediate values
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
            
            # Now test each strategy on the gradual transitions
            for start, end in transitions:
                length = end - start
                if length <= 1:  # Skip cuts, focus on graduals
                    continue
                
                # Check if any frame in this transition was detected
                gt_detected = False
                strategy_results = {name: False for name in self.strategies.keys()}
                
                # Get predictions for this transition
                trans_preds = []
                for frame_idx in range(start, min(end+1, total_frames)):
                    if frame_idx in predictions:
                        trans_preds.append(predictions[frame_idx])
                
                if not trans_preds:
                    continue
                
                # Check each strategy
                for pred in trans_preds:
                    # Current strategy
                    current_pred = np.argmax(pred['logits'])
                    if current_pred == 2:
                        strategy_results['current'] = True
                    
                    # Strategy 1: Gated Routing
                    if self._strategy_gated_routing(pred):
                        strategy_results['gated_routing'] = True
                    
                    # Strategy 2: Confidence Threshold
                    if self._strategy_confidence_threshold(pred):
                        strategy_results['confidence_threshold'] = True
                    
                    # Strategy 3: Boosted Weight (simulate)
                    if self._strategy_boosted_weight(pred):
                        strategy_results['boosted_weight'] = True
                    
                    # Strategy 4: Combined
                    if self._strategy_combined(pred):
                        strategy_results['combined'] = True
                
                # Update statistics
                for strategy_name, detected in strategy_results.items():
                    if detected:
                        self.strategies[strategy_name]['correct'] += 1
                    else:
                        self.strategies[strategy_name]['missed'] += 1
                        
                        # Save example if current missed but others caught it
                        if strategy_name == 'current' and any(strategy_results[s] for s in ['gated_routing', 'confidence_threshold', 'combined']):
                            self.examples.append({
                                'video': video_name,
                                'start': start,
                                'end': end,
                                'length': length,
                                'avg_p_gradual': np.mean([p['p_gradual'] for p in trans_preds]),
                                'avg_p_cut': np.mean([p['p_cut'] for p in trans_preds]),
                                'avg_ratio': np.mean([p['ratio_4_1'] for p in trans_preds]),
                                'fixed_by': [s for s, detected in strategy_results.items() if s != 'current' and detected]
                            })
            
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
    
    def _forward_with_intermediates(self, x):
        """Run forward pass and capture intermediate values"""
        B, T, C, H, W = x.shape
        
        # Compute differences (same as model)
        x_diff1 = x - torch.roll(x, shifts=1, dims=1)
        x_diff1[:, 0] = 0
        
        x_diff4 = x - torch.roll(x, shifts=4, dims=1)
        x_diff4[:, :4] = 0
        
        x_diff16 = x - torch.roll(x, shifts=16, dims=1)
        x_diff16[:, :16] = 0
        
        # Compute diff signals
        diff1_mag = torch.mean(torch.abs(x_diff1), dim=(2, 3, 4), keepdim=False)
        diff4_mag = torch.mean(torch.abs(x_diff4), dim=(2, 3, 4), keepdim=False)
        diff16_mag = torch.mean(torch.abs(x_diff16), dim=(2, 3, 4), keepdim=False)
        
        eps = 1e-3
        ratio_4_1 = diff4_mag / torch.clamp(diff1_mag, min=eps)
        ratio_16_4 = diff16_mag / torch.clamp(diff4_mag, min=eps)
        
        ratio_4_1 = torch.clamp(ratio_4_1, 0, 10)
        ratio_16_4 = torch.clamp(ratio_16_4, 0, 10)
        
        # Run model forward
        logits = self.model(x)
        
        # We need to access intermediate expert outputs
        # For now, we'll recompute the expert probabilities from logits
        probs = torch.softmax(logits, dim=2)
        p_none = probs[:, :, 0:1].transpose(1, 2)
        p_cut = probs[:, :, 1:2].transpose(1, 2)
        p_gradual = probs[:, :, 2:3].transpose(1, 2)
        
        intermediates = {
            'p_cut': p_cut,
            'p_gradual': p_gradual,
            'p_none': p_none,
            'diff1': diff1_mag,
            'diff4': diff4_mag,
            'diff16': diff16_mag,
            'ratio_4_1': ratio_4_1,
            'ratio_16_4': ratio_16_4
        }
        
        return logits, intermediates
    
    def _strategy_gated_routing(self, pred):
        """FIX 1: Use diff ratios to gate which expert to trust"""
        # High ratio + visible change = trust gradual expert
        is_gradual_pattern = (pred['ratio_4_1'] > 0.3) and (pred['diff1'] > 0.01)
        
        if is_gradual_pattern:
            # Trust gradual expert
            return pred['p_gradual'] > 0.4  # Lower threshold when pattern matches
        else:
            # Use default
            return pred['p_gradual'] > pred['p_cut'] and pred['p_gradual'] > 0.5
    
    def _strategy_confidence_threshold(self, pred):
        """FIX 5: Force prediction when confidence is very high"""
        # Strong gradual evidence
        if pred['p_gradual'] > 0.65 and pred['ratio_4_1'] > 0.25:
            return True
        
        # Normal prediction
        return np.argmax(pred['logits']) == 2
    
    def _strategy_boosted_weight(self, pred):
        """FIX 4: Simulate higher gradual class weight"""
        # Simulate softmax with boosted gradual logit
        logits = pred['logits'].copy()
        logits[2] += 0.5  # Boost gradual logit (simulates weight increase)
        
        boosted_probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.argmax(boosted_probs) == 2
    
    def _strategy_combined(self, pred):
        """Combined strategy: gated routing + confidence threshold"""
        # First try confidence threshold
        if pred['p_gradual'] > 0.65 and pred['ratio_4_1'] > 0.25:
            return True
        
        # Then try gated routing
        is_gradual_pattern = (pred['ratio_4_1'] > 0.3) and (pred['diff1'] > 0.01)
        if is_gradual_pattern and pred['p_gradual'] > 0.4:
            return True
        
        # Fallback to boosted weight
        logits = pred['logits'].copy()
        logits[2] += 0.5
        boosted_probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.argmax(boosted_probs) == 2
    
    def run_test(self):
        """Run test on all videos"""
        print(f"\nTesting fix strategies on {CONFIG['num_test_videos']} videos...\n")
        
        video_names = list(self.annotations.keys())[:CONFIG['num_test_videos']]
        
        for video_name in tqdm(video_names, desc="Testing strategies"):
            self.test_video(video_name)
        
        self._print_report()
    
    def _print_report(self):
        """Print comparison report"""
        print("\n" + "="*80)
        print("FIX STRATEGY COMPARISON REPORT")
        print("="*80)
        
        print("\n📊 GRADUAL DETECTION PERFORMANCE")
        print("-" * 80)
        print(f"{'Strategy':<25} {'Detected':<12} {'Missed':<12} {'Recall':<12}")
        print("-" * 80)
        
        for strategy_name in ['current', 'gated_routing', 'confidence_threshold', 'boosted_weight', 'combined']:
            stats = self.strategies[strategy_name]
            total = stats['correct'] + stats['missed']
            recall = stats['correct'] / total if total > 0 else 0
            
            # Color code improvements
            marker = ""
            if strategy_name != 'current':
                current_recall = self.strategies['current']['correct'] / (self.strategies['current']['correct'] + self.strategies['current']['missed'])
                if recall > current_recall:
                    marker = "✓ "
                elif recall < current_recall:
                    marker = "✗ "
            
            print(f"{marker}{strategy_name:<23} {stats['correct']:<12} {stats['missed']:<12} {recall*100:>6.1f}%")
        
        # Show improvement summary
        print("\n" + "="*80)
        print("💡 IMPROVEMENT SUMMARY")
        print("="*80)
        
        current_recall = self.strategies['current']['correct'] / (self.strategies['current']['correct'] + self.strategies['current']['missed'])
        
        for strategy_name in ['gated_routing', 'confidence_threshold', 'boosted_weight', 'combined']:
            stats = self.strategies[strategy_name]
            total = stats['correct'] + stats['missed']
            recall = stats['correct'] / total if total > 0 else 0
            
            improvement = recall - current_recall
            recovered = stats['correct'] - self.strategies['current']['correct']
            
            print(f"\n{strategy_name.upper().replace('_', ' ')}:")
            print(f"  Recall: {recall*100:.1f}% (Δ{improvement*100:+.1f}%)")
            print(f"  Recovered {recovered} missed graduals")
        
        # Show examples
        if self.examples:
            print("\n" + "="*80)
            print("🔍 EXAMPLE FIXES (Current missed, but fixed by strategies)")
            print("="*80)
            
            # Sort by number of strategies that fix it
            self.examples.sort(key=lambda x: len(x['fixed_by']), reverse=True)
            
            for i, ex in enumerate(self.examples[:10]):  # Show top 10
                print(f"\n{i+1}. {ex['video']} frames {ex['start']}-{ex['end']} (len={ex['length']})")
                print(f"   Avg P(Gradual): {ex['avg_p_gradual']:.3f}, Avg Ratio: {ex['avg_ratio']:.3f}")
                print(f"   Fixed by: {', '.join(ex['fixed_by'])}")
        
        # Recommendation
        print("\n" + "="*80)
        print("🎯 RECOMMENDATION")
        print("="*80)
        
        best_strategy = max(
            [(name, self.strategies[name]['correct'] / (self.strategies[name]['correct'] + self.strategies[name]['missed']))
             for name in ['gated_routing', 'confidence_threshold', 'boosted_weight', 'combined']],
            key=lambda x: x[1]
        )
        
        print(f"\nBest strategy: {best_strategy[0].upper().replace('_', ' ')}")
        print(f"Recall: {best_strategy[1]*100:.1f}%")
        print(f"Improvement: +{(best_strategy[1] - current_recall)*100:.1f}%")
        
        if best_strategy[0] == 'combined':
            print("\n✓ Implement: Gated Routing + Confidence Threshold + Boosted Weights")
        elif best_strategy[0] == 'gated_routing':
            print("\n✓ Implement: Gated Routing based on diff ratios")
        elif best_strategy[0] == 'confidence_threshold':
            print("\n✓ Implement: Confidence-based thresholding")
        elif best_strategy[0] == 'boosted_weight':
            print("\n✓ Implement: Increase gradual class weight to 12.0")
        
        print("="*80)

def main():
    tester = FixStrategyTester()
    tester.run_test()

if __name__ == '__main__':
    main()
