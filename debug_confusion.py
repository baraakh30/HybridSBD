"""
Debug utility to track where cuts and graduals are being misclassified.
Shows confusion patterns: Cut->None, Cut->Gradual, Gradual->None, Gradual->Cut
"""

import torch
import numpy as np
from collections import defaultdict

class ConfusionTracker:
    """
    Tracks predictions for Cut and Gradual frames specifically.
    Shows where model is confusing transition types.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Track what Cut frames are predicted as: [None, Cut, Gradual]
        self.cut_predictions = defaultdict(int)  # cut_predictions[0/1/2] counts
        
        # Track what Gradual frames are predicted as: [None, Cut, Gradual]
        self.gradual_predictions = defaultdict(int)
        
        # Track graduals by length - need to store (prediction, length) pairs
        self.gradual_lengths_none = []  # Lengths of graduals confused as None
        self.gradual_lengths_cut = []   # Lengths of graduals confused as Cut
        self.gradual_lengths_correct = []  # Lengths of correctly detected graduals
        
        # Track total frames for each class
        self.total_cuts = 0
        self.total_graduals = 0
        self.total_none = 0
        
        # Router decision tracking (optional)
        self.router_decisions = {
            'cut_router1_none': 0,      # Cuts where Router 1 chose None
            'cut_router1_transition': 0,  # Cuts where Router 1 chose Transition
            'cut_router2_cut': 0,       # Cuts where Router 2 chose Cut (among transitions)
            'cut_router2_gradual': 0,   # Cuts where Router 2 chose Gradual (among transitions)
            'gradual_router1_none': 0,
            'gradual_router1_transition': 0,
            'gradual_router2_cut': 0,
            'gradual_router2_gradual': 0,
        }
        self.has_router_info = False
    
    def update(self, predictions, targets, router_info=None, transition_lengths=None):
        """
        predictions: [B, T] - predicted class indices (0=None, 1=Cut, 2=Gradual)
        targets: [B, T] - ground truth class indices
        router_info: dict with 'router_1_weights' [B, T, 2] and 'router_2_weights' [B, T, 2]
        transition_lengths: [B, T] - length of transition at each frame (0 if not transition)
        """
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        if transition_lengths is not None:
            transition_lengths = transition_lengths.cpu().numpy().flatten()
        
        # Process router decisions if provided
        if router_info is not None:
            self.has_router_info = True
            router_1 = router_info['router_1_weights'].cpu().numpy().reshape(-1, 2)  # [B*T, 2]
            router_2 = router_info['router_2_weights'].cpu().numpy().reshape(-1, 2)  # [B*T, 2]
            
            # Router 1: 0=None, 1=Transition
            router_1_decisions = (router_1[:, 1] > 0.5).astype(int)  # 1 if chose Transition
            # Router 2: 0=Cut, 1=Gradual
            router_2_decisions = (router_2[:, 1] > 0.5).astype(int)  # 1 if chose Gradual
        
        # Find where ground truth is Cut (class 1)
        cut_mask = targets == 1
        if cut_mask.any():
            cut_preds = predictions[cut_mask]
            self.total_cuts += len(cut_preds)
            for pred in cut_preds:
                self.cut_predictions[int(pred)] += 1
            
            if router_info is not None:
                cut_r1 = router_1_decisions[cut_mask]
                cut_r2 = router_2_decisions[cut_mask]
                self.router_decisions['cut_router1_none'] += (cut_r1 == 0).sum()
                self.router_decisions['cut_router1_transition'] += (cut_r1 == 1).sum()
                # Among those Router 1 sent to Transition, what did Router 2 choose?
                cut_trans_mask = cut_r1 == 1
                if cut_trans_mask.any():
                    self.router_decisions['cut_router2_cut'] += (cut_r2[cut_trans_mask] == 0).sum()
                    self.router_decisions['cut_router2_gradual'] += (cut_r2[cut_trans_mask] == 1).sum()
        
        # Find where ground truth is Gradual (class 2)
        gradual_mask = targets == 2
        if gradual_mask.any():
            gradual_preds = predictions[gradual_mask]
            self.total_graduals += len(gradual_preds)
            for pred in gradual_preds:
                self.gradual_predictions[int(pred)] += 1
            
            # Track by length if provided
            if transition_lengths is not None:
                gradual_lens = transition_lengths[gradual_mask]
                for pred, length in zip(gradual_preds, gradual_lens):
                    if pred == 0:  # Confused as None
                        self.gradual_lengths_none.append(length)
                    elif pred == 1:  # Confused as Cut
                        self.gradual_lengths_cut.append(length)
                    else:  # Correctly detected
                        self.gradual_lengths_correct.append(length)
            
            if router_info is not None:
                grad_r1 = router_1_decisions[gradual_mask]
                grad_r2 = router_2_decisions[gradual_mask]
                self.router_decisions['gradual_router1_none'] += (grad_r1 == 0).sum()
                self.router_decisions['gradual_router1_transition'] += (grad_r1 == 1).sum()
                # Among those Router 1 sent to Transition, what did Router 2 choose?
                grad_trans_mask = grad_r1 == 1
                if grad_trans_mask.any():
                    self.router_decisions['gradual_router2_cut'] += (grad_r2[grad_trans_mask] == 0).sum()
                    self.router_decisions['gradual_router2_gradual'] += (grad_r2[grad_trans_mask] == 1).sum()
        
        # Track None class count
        none_mask = targets == 0
        self.total_none += none_mask.sum()
    
    def get_report(self):
        """Generate detailed confusion report"""
        report = []
        report.append("\n" + "="*70)
        report.append("CONFUSION ANALYSIS")
        report.append("="*70)
        
        # Cut confusion breakdown
        if self.total_cuts > 0:
            report.append(f"\nCUT frames (total: {self.total_cuts}):")
            cut_as_none = self.cut_predictions.get(0, 0)
            cut_as_cut = self.cut_predictions.get(1, 0)
            cut_as_gradual = self.cut_predictions.get(2, 0)
            
            pct_none = 100 * cut_as_none / self.total_cuts
            pct_cut = 100 * cut_as_cut / self.total_cuts
            pct_gradual = 100 * cut_as_gradual / self.total_cuts
            
            report.append(f"  ✓ Correctly as Cut:      {cut_as_cut:6d} ({pct_cut:5.1f}%)")
            report.append(f"  ✗ Confused as None:      {cut_as_none:6d} ({pct_none:5.1f}%) <- MISSED")
            report.append(f"  ✗ Confused as Gradual:   {cut_as_gradual:6d} ({pct_gradual:5.1f}%) <- WRONG TYPE")
        else:
            report.append("\nNo CUT frames in validation set")
        
        # Gradual confusion breakdown
        if self.total_graduals > 0:
            report.append(f"\nGRADUAL frames (total: {self.total_graduals}):")
            grad_as_none = self.gradual_predictions.get(0, 0)
            grad_as_cut = self.gradual_predictions.get(1, 0)
            grad_as_gradual = self.gradual_predictions.get(2, 0)
            
            pct_none = 100 * grad_as_none / self.total_graduals
            pct_cut = 100 * grad_as_cut / self.total_graduals
            pct_gradual = 100 * grad_as_gradual / self.total_graduals
            
            report.append(f"  ✓ Correctly as Gradual:  {grad_as_gradual:6d} ({pct_gradual:5.1f}%)")
            report.append(f"  ✗ Confused as None:      {grad_as_none:6d} ({pct_none:5.1f}%) <- MISSED")
            report.append(f"  ✗ Confused as Cut:       {grad_as_cut:6d} ({pct_cut:5.1f}%) <- WRONG TYPE")
        else:
            report.append("\nNo GRADUAL frames in validation set")
        
        # Summary
        report.append("\n" + "-"*70)
        report.append("KEY INSIGHTS:")
        if self.total_cuts > 0 and self.total_graduals > 0:
            cut_recall = 100 * self.cut_predictions.get(1, 0) / self.total_cuts
            grad_recall = 100 * self.gradual_predictions.get(2, 0) / self.total_graduals
            
            if cut_recall < 70:
                report.append(f"  ⚠ Cut detection low ({cut_recall:.1f}%) - Router 1 too aggressive")
            if grad_recall < 50:
                report.append(f"  ⚠ Gradual detection low ({grad_recall:.1f}%) - Confused as None/Cut")
            
            cut_to_grad = self.cut_predictions.get(2, 0)
            grad_to_cut = self.gradual_predictions.get(1, 0)
            if cut_to_grad > self.total_cuts * 0.1:
                report.append(f"  ⚠ {cut_to_grad} cuts confused as gradual - Router 2 issue")
            if grad_to_cut > self.total_graduals * 0.1:
                report.append(f"  ⚠ {grad_to_cut} graduals confused as cut - Router 2 issue")
        
        # Gradual length analysis
        if self.gradual_lengths_none or self.gradual_lengths_cut or self.gradual_lengths_correct:
            report.append("\n" + "-"*70)
            report.append("GRADUAL TRANSITION LENGTH ANALYSIS:")
            
            all_lens = self.gradual_lengths_none + self.gradual_lengths_cut + self.gradual_lengths_correct
            if all_lens:
                report.append(f"\nAll graduals - Length distribution:")
                report.append(f"  Short (2-5 frames):   {sum(1 for l in all_lens if 2 <= l <= 5):5d} ({100*sum(1 for l in all_lens if 2 <= l <= 5)/len(all_lens):5.1f}%)")
                report.append(f"  Medium (6-15 frames): {sum(1 for l in all_lens if 6 <= l <= 15):5d} ({100*sum(1 for l in all_lens if 6 <= l <= 15)/len(all_lens):5.1f}%)")
                report.append(f"  Long (16+ frames):    {sum(1 for l in all_lens if l >= 16):5d} ({100*sum(1 for l in all_lens if l >= 16)/len(all_lens):5.1f}%)")
            
            if self.gradual_lengths_none:
                report.append(f"\nGraduals MISSED (confused as None) - By length:")
                report.append(f"  Short (2-5 frames):   {sum(1 for l in self.gradual_lengths_none if 2 <= l <= 5):5d} / {sum(1 for l in all_lens if 2 <= l <= 5):5d} ({100*sum(1 for l in self.gradual_lengths_none if 2 <= l <= 5)/max(sum(1 for l in all_lens if 2 <= l <= 5), 1):5.1f}%)")
                report.append(f"  Medium (6-15 frames): {sum(1 for l in self.gradual_lengths_none if 6 <= l <= 15):5d} / {sum(1 for l in all_lens if 6 <= l <= 15):5d} ({100*sum(1 for l in self.gradual_lengths_none if 6 <= l <= 15)/max(sum(1 for l in all_lens if 6 <= l <= 15), 1):5.1f}%)")
                report.append(f"  Long (16+ frames):    {sum(1 for l in self.gradual_lengths_none if l >= 16):5d} / {sum(1 for l in all_lens if l >= 16):5d} ({100*sum(1 for l in self.gradual_lengths_none if l >= 16)/max(sum(1 for l in all_lens if l >= 16), 1):5.1f}%)")
                
                avg_missed = np.mean(self.gradual_lengths_none) if self.gradual_lengths_none else 0
                avg_correct = np.mean(self.gradual_lengths_correct) if self.gradual_lengths_correct else 0
                report.append(f"\n  Avg length MISSED: {avg_missed:.1f} frames")
                report.append(f"  Avg length DETECTED: {avg_correct:.1f} frames")
                if avg_missed < avg_correct:
                    report.append(f"  → Model struggles more with SHORT graduals")
        
        # Router analysis
        if self.has_router_info:
            report.append("\n" + "-"*70)
            report.append("ROUTER DECISION ANALYSIS:")
            
            if self.total_cuts > 0:
                report.append(f"\nCUTS - Router Decisions:")
                r1_none = self.router_decisions['cut_router1_none']
                r1_trans = self.router_decisions['cut_router1_transition']
                report.append(f"  Router 1 → None:       {r1_none:6d} ({100*r1_none/self.total_cuts:5.1f}%) <- MISSED AT STAGE 1")
                report.append(f"  Router 1 → Transition: {r1_trans:6d} ({100*r1_trans/self.total_cuts:5.1f}%)")
                
                if r1_trans > 0:
                    r2_cut = self.router_decisions['cut_router2_cut']
                    r2_grad = self.router_decisions['cut_router2_gradual']
                    report.append(f"    Router 2 → Cut:        {r2_cut:6d} ({100*r2_cut/r1_trans:5.1f}% of transitions)")
                    report.append(f"    Router 2 → Gradual:    {r2_grad:6d} ({100*r2_grad/r1_trans:5.1f}% of transitions) <- WRONG AT STAGE 2")
            
            if self.total_graduals > 0:
                report.append(f"\nGRADUALS - Router Decisions:")
                r1_none = self.router_decisions['gradual_router1_none']
                r1_trans = self.router_decisions['gradual_router1_transition']
                report.append(f"  Router 1 → None:       {r1_none:6d} ({100*r1_none/self.total_graduals:5.1f}%) <- MISSED AT STAGE 1")
                report.append(f"  Router 1 → Transition: {r1_trans:6d} ({100*r1_trans/self.total_graduals:5.1f}%)")
                
                if r1_trans > 0:
                    r2_cut = self.router_decisions['gradual_router2_cut']
                    r2_grad = self.router_decisions['gradual_router2_gradual']
                    report.append(f"    Router 2 → Cut:        {r2_cut:6d} ({100*r2_cut/r1_trans:5.1f}% of transitions) <- WRONG AT STAGE 2")
                    report.append(f"    Router 2 → Gradual:    {r2_grad:6d} ({100*r2_grad/r1_trans:5.1f}% of transitions)")
        
        report.append("="*70 + "\n")
        
        return "\n".join(report)
    
    def print_report(self):
        print(self.get_report())


if __name__ == "__main__":
    # Test the tracker
    tracker = ConfusionTracker()
    
    # Simulate some predictions
    # Ground truth: 5 cuts, 3 graduals, rest none
    targets = torch.tensor([[0, 0, 1, 0, 1, 0, 2, 1, 0, 2]])  # 3 cuts (indices 2,4,7), 2 graduals (6,9)
    
    # Predictions: some correct, some confused
    predictions = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 1, 0, 1]])  # Cut at 2,7 correct; Cut at 4->None; Grad at 6->None; Grad at 9->Cut
    
    tracker.update(predictions, targets)
    tracker.print_report()
