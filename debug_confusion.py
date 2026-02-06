"""
Confusion Matrix Tracker for Shot Boundary Detection

Tracks detailed confusion patterns including:
- Standard confusion matrix (None/Cut/Gradual)
- Length-stratified analysis (short/medium/long graduals)
- Misclassification patterns
"""

import torch
import numpy as np
from collections import defaultdict


class ConfusionTracker:
    """
    Track and analyze confusion patterns during training/validation
    
    Tracks:
    - Overall confusion matrix (3x3: None/Cut/Gradual)
    - Length-stratified metrics for graduals (short/medium/long)
    - Common misclassification patterns
    """
    
    def __init__(self):
        # 3x3 confusion matrix: [true_class][pred_class]
        # Classes: 0=None, 1=Cut, 2=Gradual
        self.confusion_matrix = torch.zeros(3, 3, dtype=torch.long)
        
        # Length-stratified tracking for graduals (class 2)
        # Categories: short (2-5f), medium (6-15f), long (16+f)
        self.gradual_by_length = {
            'short': {'tp': 0, 'fp': 0, 'fn': 0},    # 2-5 frames
            'medium': {'tp': 0, 'fp': 0, 'fn': 0},   # 6-15 frames
            'long': {'tp': 0, 'fp': 0, 'fn': 0}      # 16+ frames
        }
        
        # Track common error patterns
        self.error_patterns = defaultdict(int)
        
        # Total samples processed
        self.total_samples = 0
    
    def update(self, preds, targets, router_decisions=None, transition_lengths=None):
        """
        Update confusion tracking with new predictions
        
        Args:
            preds: [B, T] predictions (0=None, 1=Cut, 2=Gradual)
            targets: [B, T] ground truth labels
            router_decisions: Optional router decisions (not currently used)
            transition_lengths: [B, T] length of transition at each frame
        """
        # Flatten for easier processing
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        if transition_lengths is not None:
            lengths_flat = transition_lengths.reshape(-1)
        else:
            lengths_flat = None
        
        # Update overall confusion matrix
        for true_class in range(3):
            for pred_class in range(3):
                mask = (targets_flat == true_class) & (preds_flat == pred_class)
                self.confusion_matrix[true_class, pred_class] += mask.sum().item()
        
        # Track length-stratified metrics for graduals
        if lengths_flat is not None:
            # True graduals
            is_gradual = (targets_flat == 2)
            pred_gradual = (preds_flat == 2)
            
            # Short graduals (2-5 frames)
            short_mask = is_gradual & (lengths_flat > 1) & (lengths_flat <= 5)
            self.gradual_by_length['short']['tp'] += (short_mask & pred_gradual).sum().item()
            self.gradual_by_length['short']['fn'] += (short_mask & ~pred_gradual).sum().item()
            
            # Medium graduals (6-15 frames)
            medium_mask = is_gradual & (lengths_flat > 5) & (lengths_flat <= 15)
            self.gradual_by_length['medium']['tp'] += (medium_mask & pred_gradual).sum().item()
            self.gradual_by_length['medium']['fn'] += (medium_mask & ~pred_gradual).sum().item()
            
            # Long graduals (16+ frames)
            long_mask = is_gradual & (lengths_flat > 15)
            self.gradual_by_length['long']['tp'] += (long_mask & pred_gradual).sum().item()
            self.gradual_by_length['long']['fn'] += (long_mask & ~pred_gradual).sum().item()
            
            # False positives (predicted gradual but was not)
            false_graduals = ~is_gradual & pred_gradual
            
            # Categorize false positives by what they should have been
            fp_from_none = false_graduals & (targets_flat == 0)
            fp_from_cut = false_graduals & (targets_flat == 1)
            
            # Distribute FPs across categories (for metric calculation)
            total_fp = false_graduals.sum().item()
            if total_fp > 0:
                # Distribute proportionally
                for category in self.gradual_by_length:
                    self.gradual_by_length[category]['fp'] += total_fp // 3
        
        # Track common error patterns
        errors = (preds_flat != targets_flat)
        for i in range(len(preds_flat)):
            if errors[i]:
                true_c = targets_flat[i].item()
                pred_c = preds_flat[i].item()
                pattern = f"{self._class_name(true_c)}→{self._class_name(pred_c)}"
                self.error_patterns[pattern] += 1
        
        self.total_samples += len(preds_flat)
    
    def _class_name(self, class_idx):
        """Convert class index to name"""
        names = {0: 'None', 1: 'Cut', 2: 'Grad'}
        return names.get(class_idx, f'C{class_idx}')
    
    def print_report(self):
        """Print detailed confusion analysis"""
        print("\n" + "="*80)
        print("CONFUSION ANALYSIS")
        print("="*80)
        
        # 1. Overall Confusion Matrix
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print("           None      Cut    Gradual")
        class_names = ['None    ', 'Cut     ', 'Gradual ']
        
        for i, name in enumerate(class_names):
            row = self.confusion_matrix[i]
            print(f"{name}  {row[0]:8d}  {row[1]:8d}  {row[2]:8d}")
        
        # 2. Per-class metrics
        print("\nPer-Class Metrics:")
        for c in range(3):
            tp = self.confusion_matrix[c, c].item()
            fp = self.confusion_matrix[:, c].sum().item() - tp
            fn = self.confusion_matrix[c, :].sum().item() - tp
            
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)
            
            print(f"  {class_names[c]} P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f} | TP={tp:,}, FP={fp:,}, FN={fn:,}")
        
        # 3. Length-stratified gradual analysis
        print("\nGradual Detection by Length:")
        for category, metrics in self.gradual_by_length.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            if tp + fn > 0:  # Only show if we have samples
                p = tp / (tp + fp + 1e-8)
                r = tp / (tp + fn + 1e-8)
                f1 = 2 * p * r / (p + r + 1e-8)
                
                # Length ranges
                ranges = {'short': '2-5f', 'medium': '6-15f', 'long': '16+f'}
                print(f"  {category.capitalize():<8} ({ranges[category]}): "
                      f"P={p:.3f}, R={r:.3f}, F1={f1:.3f} | TP={tp:,}, FP={fp:,}, FN={fn:,}")
        
        # 4. Top error patterns
        print("\nTop 5 Error Patterns:")
        sorted_errors = sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_errors[:5]:
            pct = 100 * count / self.total_samples
            print(f"  {pattern:<15} {count:8,} ({pct:.2f}%)")
        
        print("="*80)
    
    def reset(self):
        """Reset all tracking"""
        self.confusion_matrix.zero_()
        for category in self.gradual_by_length:
            self.gradual_by_length[category] = {'tp': 0, 'fp': 0, 'fn': 0}
        self.error_patterns.clear()
        self.total_samples = 0
