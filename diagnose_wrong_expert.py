"""
Deep Diagnostic: Why Gradual Expert Isn't Firing
=================================================
Analyzes the 8 WRONG_EXPERT failures to understand root cause
"""

import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from decord import VideoReader, cpu

from detection.turbo_sbd import TurboSBD

CONFIG = {
    'input_root': r"C:\Users\baraa\Documents\ClipShots",
    'model_path': 'outputs/turbo_sbd_best.pth',
    'frame_size': (48, 64),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

class DeepDiagnostic:
    def __init__(self):
        self.device = CONFIG['device']
        
        print(f"Loading model...")
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
        
        self.wrong_expert_cases = []
        
    def analyze_transition(self, video_name, start, end):
        """Deep analysis of a single transition"""
        video_path = os.path.join(self.video_dir, video_name)
        if not os.path.exists(video_path):
            return None
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0), 
                           width=CONFIG['frame_size'][1], 
                           height=CONFIG['frame_size'][0])
            
            # Extract window around transition (±20 frames)
            window_start = max(0, start - 20)
            window_end = min(len(vr), end + 20)
            
            frames = vr.get_batch(list(range(window_start, window_end))).asnumpy()
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            
            # Pad to seq_len if needed
            if len(frames) < 100:
                padding = frames[-1:].repeat(100 - len(frames), 1, 1, 1)
                frames = torch.cat([frames, padding], dim=0)
            
            frames_batch = frames.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get predictions
                logits = self.model(frames_batch)
                probs = torch.softmax(logits, dim=2).cpu().numpy()
                
                # Compute signals manually
                x = frames_batch
                B, T, C, H, W = x.shape
                
                # Differences
                x_diff1 = x - torch.roll(x, shifts=1, dims=1)
                x_diff1[:, 0] = 0
                x_diff4 = x - torch.roll(x, shifts=4, dims=1)
                x_diff4[:, :4] = 0
                x_diff16 = x - torch.roll(x, shifts=16, dims=1)
                x_diff16[:, :16] = 0
                
                diff1_mag = torch.mean(torch.abs(x_diff1), dim=(2, 3, 4)).cpu().numpy()[0]
                diff4_mag = torch.mean(torch.abs(x_diff4), dim=(2, 3, 4)).cpu().numpy()[0]
                diff16_mag = torch.mean(torch.abs(x_diff16), dim=(2, 3, 4)).cpu().numpy()[0]
                
                ratio_4_1 = diff4_mag / np.clip(diff1_mag, 1e-3, None)
                ratio_16_4 = diff16_mag / np.clip(diff4_mag, 1e-3, None)
            
            # Map to window coordinates
            trans_start_local = start - window_start
            trans_end_local = end - window_start
            
            return {
                'video': video_name,
                'start': start,
                'end': end,
                'length': end - start,
                'window_start': window_start,
                'probs': probs[0],  # [T, 3]
                'diff1': diff1_mag,
                'diff4': diff4_mag,
                'diff16': diff16_mag,
                'ratio_4_1': ratio_4_1,
                'ratio_16_4': ratio_16_4,
                'trans_start_local': trans_start_local,
                'trans_end_local': trans_end_local,
            }
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def find_wrong_expert_cases(self, max_videos=50):
        """Find all WRONG_EXPERT failures"""
        print("\n🔍 Scanning for WRONG_EXPERT failures...")
        
        video_names = list(self.annotations.keys())[:max_videos]
        
        for video_name in tqdm(video_names):
            transitions = self.annotations[video_name]['transitions']
            
            for start, end in transitions:
                length = end - start
                if length <= 1:  # Skip cuts
                    continue
                
                # Quick check: analyze this transition
                result = self.analyze_transition(video_name, start, end)
                if result is None:
                    continue
                
                # Extract probabilities in transition range
                t_start = result['trans_start_local']
                t_end = min(result['trans_end_local'], len(result['probs']))
                
                if t_start >= len(result['probs']) or t_end <= 0:
                    continue
                
                trans_probs = result['probs'][t_start:t_end]
                max_p_gradual = np.max(trans_probs[:, 2])
                
                # WRONG_EXPERT: max P(Gradual) < 0.2 with visible signal
                avg_signal = np.mean(result['diff1'][t_start:t_end])
                
                if max_p_gradual < 0.2 and avg_signal > 0.01:
                    self.wrong_expert_cases.append(result)
        
        print(f"✓ Found {len(self.wrong_expert_cases)} WRONG_EXPERT cases\n")
    
    def visualize_case(self, case_idx):
        """Create detailed visualization of a case"""
        if case_idx >= len(self.wrong_expert_cases):
            print(f"Only {len(self.wrong_expert_cases)} cases available")
            return
        
        case = self.wrong_expert_cases[case_idx]
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        t = np.arange(len(case['probs']))
        trans_start = case['trans_start_local']
        trans_end = case['trans_end_local']
        
        # Plot 1: Probabilities
        axes[0].plot(t, case['probs'][:, 0], 'b-', label='P(None)', linewidth=2)
        axes[0].plot(t, case['probs'][:, 1], 'r-', label='P(Cut)', linewidth=2)
        axes[0].plot(t, case['probs'][:, 2], 'g-', label='P(Gradual)', linewidth=2)
        axes[0].axvspan(trans_start, trans_end, alpha=0.3, color='yellow', label='GT Transition')
        axes[0].set_ylabel('Probability')
        axes[0].set_title(f'{case["video"]} - Frames {case["start"]}-{case["end"]} (Length: {case["length"]})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Diff Magnitudes
        axes[1].plot(t, case['diff1'], 'b-', label='Diff1', linewidth=2)
        axes[1].plot(t, case['diff4'], 'r-', label='Diff4', linewidth=2)
        axes[1].plot(t, case['diff16'], 'g-', label='Diff16', linewidth=2)
        axes[1].axvspan(trans_start, trans_end, alpha=0.3, color='yellow')
        axes[1].set_ylabel('Magnitude')
        axes[1].set_title('Frame Difference Signals')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Ratios
        axes[2].plot(t, case['ratio_4_1'], 'purple', label='Ratio 4/1', linewidth=2)
        axes[2].plot(t, case['ratio_16_4'], 'orange', label='Ratio 16/4', linewidth=2)
        axes[2].axvspan(trans_start, trans_end, alpha=0.3, color='yellow')
        axes[2].axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Gradual threshold (0.3)')
        axes[2].set_ylabel('Ratio')
        axes[2].set_title('Smoothness Ratios (High = Gradual Pattern)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Prediction
        predictions = np.argmax(case['probs'], axis=1)
        axes[3].plot(t, predictions, 'k-', drawstyle='steps-post', linewidth=2)
        axes[3].axvspan(trans_start, trans_end, alpha=0.3, color='yellow')
        axes[3].set_ylabel('Predicted Class')
        axes[3].set_xlabel('Frame')
        axes[3].set_yticks([0, 1, 2])
        axes[3].set_yticklabels(['None', 'Cut', 'Gradual'])
        axes[3].set_title('Model Prediction')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        os.makedirs('outputs/diagnostics', exist_ok=True)
        plt.savefig(f'outputs/diagnostics/wrong_expert_case_{case_idx}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to outputs/diagnostics/wrong_expert_case_{case_idx}.png")
        plt.close()
    
    def print_summary(self):
        """Print summary of all WRONG_EXPERT cases"""
        if not self.wrong_expert_cases:
            print("No WRONG_EXPERT cases found")
            return
        
        print("\n" + "="*80)
        print(f"WRONG_EXPERT ANALYSIS ({len(self.wrong_expert_cases)} cases)")
        print("="*80)
        
        for i, case in enumerate(self.wrong_expert_cases):
            t_start = case['trans_start_local']
            t_end = min(case['trans_end_local'], len(case['probs']))
            
            trans_probs = case['probs'][t_start:t_end]
            trans_diff1 = case['diff1'][t_start:t_end]
            trans_diff4 = case['diff4'][t_start:t_end]
            trans_ratio = case['ratio_4_1'][t_start:t_end]
            
            print(f"\n{i+1}. {case['video']} frames {case['start']}-{case['end']} (len={case['length']})")
            print(f"   Max P(Gradual): {np.max(trans_probs[:, 2]):.3f}")
            print(f"   Max P(Cut):     {np.max(trans_probs[:, 1]):.3f}")
            print(f"   Avg Diff1:      {np.mean(trans_diff1):.4f}")
            print(f"   Avg Diff4:      {np.mean(trans_diff4):.4f}")
            print(f"   Avg Ratio 4/1:  {np.mean(trans_ratio):.3f}")
            
            # Diagnosis
            if np.mean(trans_diff1) < 0.02:
                print(f"   → DIAGNOSIS: Very subtle change (diff1 < 0.02)")
            elif np.mean(trans_ratio) < 0.2:
                print(f"   → DIAGNOSIS: Sharp change pattern (ratio < 0.2) - looks like cut")
            elif np.max(trans_probs[:, 1]) > 0.5:
                print(f"   → DIAGNOSIS: Cut expert dominating")
            else:
                print(f"   → DIAGNOSIS: Both experts failing - feature issue")
        
        # Aggregate stats
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)
        
        lengths = [c['length'] for c in self.wrong_expert_cases]
        max_p_grads = [np.max(c['probs'][c['trans_start_local']:c['trans_end_local'], 2]) 
                       for c in self.wrong_expert_cases]
        avg_signals = [np.mean(c['diff1'][c['trans_start_local']:c['trans_end_local']]) 
                      for c in self.wrong_expert_cases]
        avg_ratios = [np.mean(c['ratio_4_1'][c['trans_start_local']:c['trans_end_local']]) 
                     for c in self.wrong_expert_cases]
        
        print(f"\nLength:          {np.mean(lengths):.1f} frames (range: {np.min(lengths)}-{np.max(lengths)})")
        print(f"Max P(Gradual):  {np.mean(max_p_grads):.3f} (range: {np.min(max_p_grads):.3f}-{np.max(max_p_grads):.3f})")
        print(f"Avg Signal:      {np.mean(avg_signals):.4f} (range: {np.min(avg_signals):.4f}-{np.max(avg_signals):.4f})")
        print(f"Avg Ratio 4/1:   {np.mean(avg_ratios):.3f} (range: {np.min(avg_ratios):.3f}-{np.max(avg_ratios):.3f})")
        
        print("\n" + "="*80)
        print("💡 ROOT CAUSE HYPOTHESIS")
        print("="*80)
        
        low_signal_count = sum(1 for s in avg_signals if s < 0.02)
        low_ratio_count = sum(1 for r in avg_ratios if r < 0.3)
        
        print(f"\n{low_signal_count}/{len(self.wrong_expert_cases)} cases have VERY LOW signal (diff1 < 0.02)")
        print(f"  → These might be annotation errors or imperceptible transitions")
        
        print(f"\n{low_ratio_count}/{len(self.wrong_expert_cases)} cases have LOW ratios (< 0.3)")
        print(f"  → Pattern looks like CUT not GRADUAL (sharp change)")
        print(f"  → Possible annotation errors (labeled gradual but actually cuts)")
        
        both_count = len([i for i, (s, r) in enumerate(zip(avg_signals, avg_ratios)) 
                         if s >= 0.02 and r >= 0.3])
        
        print(f"\n{both_count}/{len(self.wrong_expert_cases)} cases have GOOD signal + GOOD ratio but still fail")
        print(f"  → TRUE MODEL FAILURES - need architectural fixes:")
        print(f"    1. Features not discriminative enough")
        print(f"    2. Expert routing broken")
        print(f"    3. Need direct 3-class classifier instead of binary experts")
        
        print("\n" + "="*80)

def main():
    diag = DeepDiagnostic()
    diag.find_wrong_expert_cases(max_videos=50)
    diag.print_summary()
    
    # Visualize top cases
    if len(diag.wrong_expert_cases) > 0:
        print("\n📊 Generating visualizations for top cases...")
        for i in range(min(5, len(diag.wrong_expert_cases))):
            diag.visualize_case(i)
        print(f"✓ Check outputs/diagnostics/ for visualizations")

if __name__ == '__main__':
    main()
