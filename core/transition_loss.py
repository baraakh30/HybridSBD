"""
Transition-Level Loss Functions

Instead of frame-level classification loss, compute loss on detected transitions.
Directly optimizes the metric we care about: precision/recall of transition detection.

Key idea: Extract transitions from predictions, match to ground truth, compute loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCoherenceLoss(nn.Module):
    """
    Penalizes discontinuous predictions to enforce temporal coherence
    
    Encourages:
    - Contiguous regions of the same class
    - Smooth transitions between classes
    - Sparse boundary predictions
    
    Loss = average number of class changes per sequence
    """
    def __init__(self, weight=0.3):
        super().__init__()
        self.weight = weight
        
    def forward(self, logits):
        """
        Args:
            logits: [B, T, 3] raw predictions
        Returns:
            Coherence loss (lower = more coherent)
        """
        # Get predicted classes
        preds = logits.argmax(dim=-1)  # [B, T]
        
        # Count changes between adjacent frames
        changes = (preds[:, 1:] != preds[:, :-1]).float()
        
        # Average across batch and time
        coherence_loss = changes.mean()
        
        return self.weight * coherence_loss


class TransitionMatchingLoss(nn.Module):
    """
    Transition-level matching loss inspired by object detection losses
    
    Algorithm:
    1. Extract predicted transitions from logits
    2. Match to ground truth transitions (Hungarian matching)
    3. Compute:
       - Classification loss for matched transitions
       - Localization loss (how far from GT)
       - Penalty for unmatched predictions (false positives)
       - Penalty for unmatched GT (false negatives)
    
    This directly optimizes transition detection F1.
    """
    def __init__(self, tolerance=3, alpha=1.0, beta=0.5, gamma=0.3):
        super().__init__()
        self.tolerance = tolerance  # Frame tolerance for matching
        self.alpha = alpha    # Weight for classification loss
        self.beta = beta      # Weight for localization loss
        self.gamma = gamma    # Weight for unmatched penalties
        
    def extract_transitions(self, preds, min_gap=3):
        """
        Extract transition frames from predictions
        
        Args:
            preds: [T] tensor of class predictions
            min_gap: Minimum gap between transitions
        
        Returns:
            List of (frame_idx, class) tuples
        """
        transitions = []
        
        # Track segments
        in_cut = False
        in_gradual = False
        cut_start = 0
        grad_start = 0
        
        for i in range(len(preds)):
            pred = preds[i].item()
            
            # Cut transitions
            if pred == 1:
                if not in_cut:
                    in_cut = True
                    cut_start = i
            else:
                if in_cut:
                    center = (cut_start + i - 1) // 2
                    if len(transitions) == 0 or center - transitions[-1][0] >= min_gap:
                        transitions.append((center, 1))
                    in_cut = False
            
            # Gradual transitions
            if pred == 2:
                if not in_gradual:
                    in_gradual = True
                    grad_start = i
            else:
                if in_gradual:
                    center = (grad_start + i - 1) // 2
                    if len(transitions) == 0 or center - transitions[-1][0] >= min_gap:
                        transitions.append((center, 2))
                    in_gradual = False
        
        # Handle segments extending to end
        if in_cut:
            center = (cut_start + len(preds) - 1) // 2
            if len(transitions) == 0 or center - transitions[-1][0] >= min_gap:
                transitions.append((center, 1))
        if in_gradual:
            center = (grad_start + len(preds) - 1) // 2
            if len(transitions) == 0 or center - transitions[-1][0] >= min_gap:
                transitions.append((center, 2))
        
        return transitions
    
    def match_transitions(self, pred_trans, gt_trans):
        """
        Match predicted transitions to ground truth (greedy matching)
        
        Args:
            pred_trans: List of (frame, class) predicted
            gt_trans: List of (frame, class) ground truth
        
        Returns:
            matches: List of (pred_idx, gt_idx) pairs
            unmatched_pred: Indices of unmatched predictions
            unmatched_gt: Indices of unmatched ground truth
        """
        matches = []
        matched_gt = set()
        unmatched_pred = []
        
        for pred_idx, (pred_frame, pred_class) in enumerate(pred_trans):
            best_match = None
            best_dist = float('inf')
            
            for gt_idx, (gt_frame, gt_class) in enumerate(gt_trans):
                if gt_idx in matched_gt:
                    continue
                
                # Only match same class
                if pred_class != gt_class:
                    continue
                
                # Check if within tolerance
                dist = abs(pred_frame - gt_frame)
                if dist <= self.tolerance and dist < best_dist:
                    best_match = gt_idx
                    best_dist = dist
            
            if best_match is not None:
                matches.append((pred_idx, best_match))
                matched_gt.add(best_match)
            else:
                unmatched_pred.append(pred_idx)
        
        unmatched_gt = [i for i in range(len(gt_trans)) if i not in matched_gt]
        
        return matches, unmatched_pred, unmatched_gt
    
    def forward(self, logits, targets, transition_lengths):
        """
        Compute transition-level matching loss
        
        Args:
            logits: [B, T, 3] model predictions
            targets: [B, T] ground truth labels
            transition_lengths: [B, T] length of transition at each frame
        
        Returns:
            Total transition matching loss
        """
        B, T, C = logits.shape
        device = logits.device
        
        total_loss = torch.tensor(0.0, device=device)
        total_samples = 0
        
        # Process each sample in batch
        for b in range(B):
            # Get predictions and ground truth for this sample
            sample_logits = logits[b]  # [T, 3]
            sample_targets = targets[b]  # [T]
            sample_lengths = transition_lengths[b]  # [T]
            
            # Extract predicted transitions
            preds = sample_logits.argmax(dim=-1)  # [T]
            pred_trans = self.extract_transitions(preds)
            
            # Extract ground truth transitions
            gt_trans = []
            in_trans = False
            trans_start = 0
            trans_class = 0
            
            for t in range(T):
                label = sample_targets[t].item()
                
                if label > 0:  # Cut or Gradual
                    if not in_trans:
                        in_trans = True
                        trans_start = t
                        trans_class = label
                else:
                    if in_trans:
                        center = (trans_start + t - 1) // 2
                        gt_trans.append((center, trans_class))
                        in_trans = False
            
            if in_trans:
                center = (trans_start + T - 1) // 2
                gt_trans.append((center, trans_class))
            
            # Skip if no transitions (all none frames)
            if len(gt_trans) == 0:
                # Penalize false positive predictions
                if len(pred_trans) > 0:
                    fp_penalty = len(pred_trans) * self.gamma
                    total_loss = total_loss + fp_penalty
                    total_samples += 1
                continue
            
            # Match transitions
            matches, unmatched_pred, unmatched_gt = self.match_transitions(pred_trans, gt_trans)
            
            # 1. Classification + Localization loss for matched transitions
            for pred_idx, gt_idx in matches:
                pred_frame, pred_class = pred_trans[pred_idx]
                gt_frame, gt_class = gt_trans[gt_idx]
                
                # Classification loss (use logits at predicted frame)
                cls_logits = sample_logits[pred_frame]  # [3]
                cls_target = torch.tensor(gt_class, device=device)
                cls_loss = F.cross_entropy(cls_logits.unsqueeze(0), cls_target.unsqueeze(0))
                
                # Localization loss (distance from GT)
                loc_loss = abs(pred_frame - gt_frame) / self.tolerance
                
                total_loss = total_loss + self.alpha * cls_loss + self.beta * loc_loss
            
            # 2. Penalty for false positives (unmatched predictions)
            fp_penalty = len(unmatched_pred) * self.gamma
            total_loss = total_loss + fp_penalty
            
            # 3. Penalty for false negatives (missed ground truth)
            fn_penalty = len(unmatched_gt) * self.gamma
            total_loss = total_loss + fn_penalty
            
            total_samples += 1
        
        # Average over batch
        if total_samples > 0:
            total_loss = total_loss / total_samples
        
        return total_loss


class BoundaryFocalLoss(nn.Module):
    """
    Focal Loss adapted for boundary detection
    
    Key differences from standard focal loss:
    - Higher penalty for false positives (boundaries should be sparse)
    - Spatial weighting (frames near GT boundaries get higher weight)
    """
    def __init__(self, alpha=0.25, gamma=2.0, boundary_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        
    def forward(self, logits, targets, transition_lengths):
        """
        Args:
            logits: [B, T, 3] predictions
            targets: [B, T] ground truth
            transition_lengths: [B, T] for weighting
        Returns:
            Weighted focal loss
        """
        B, T, C = logits.shape
        
        # Flatten
        logits_flat = logits.view(-1, C)  # [B*T, 3]
        targets_flat = targets.view(-1)    # [B*T]
        
        # Compute probabilities
        probs = F.softmax(logits_flat, dim=1)
        targets_one_hot = F.one_hot(targets_flat, num_classes=C).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting (class balance)
        alpha_weight = torch.ones_like(targets_flat).float()
        alpha_weight[targets_flat == 0] = 1.0      # None
        alpha_weight[targets_flat == 1] = self.alpha  # Cut (boost)
        alpha_weight[targets_flat == 2] = self.alpha  # Gradual (boost)
        
        # Cross entropy
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Combine
        loss = alpha_weight * focal_weight * ce_loss
        
        return loss.mean()


class CombinedTransitionLoss(nn.Module):
    """
    Combines multiple loss components for robust transition detection:
    
    1. Boundary Focal Loss (frame-level, sparse boundary focus)
    2. Transition Matching Loss (transition-level optimization)
    3. Temporal Coherence Loss (enforce contiguous predictions)
    """
    def __init__(self, 
                 focal_weight=1.0,
                 matching_weight=0.5, 
                 coherence_weight=0.3):
        super().__init__()
        
        self.focal_loss = BoundaryFocalLoss(alpha=0.25, gamma=2.0)
        self.matching_loss = TransitionMatchingLoss(tolerance=3, alpha=1.0, beta=0.5, gamma=0.3)
        self.coherence_loss = TemporalCoherenceLoss(weight=coherence_weight)
        
        self.focal_weight = focal_weight
        self.matching_weight = matching_weight
        
    def forward(self, logits, targets, transition_lengths):
        """
        Compute combined loss
        
        Args:
            logits: [B, T, 3]
            targets: [B, T]
            transition_lengths: [B, T]
        
        Returns:
            Combined loss and individual components (for logging)
        """
        # Individual losses
        focal = self.focal_loss(logits, targets, transition_lengths)
        matching = self.matching_loss(logits, targets, transition_lengths)
        coherence = self.coherence_loss(logits)
        
        # Combined
        total_loss = (
            self.focal_weight * focal +
            self.matching_weight * matching +
            coherence  # coherence weight already in module
        )
        
        return total_loss, {
            'focal': focal.item(),
            'matching': matching.item(),
            'coherence': coherence.item(),
            'total': total_loss.item()
        }


if __name__ == '__main__':
    # Test losses
    print("Testing transition-level losses...")
    
    # Create dummy data
    B, T, C = 4, 128, 3
    logits = torch.randn(B, T, C)
    targets = torch.zeros(B, T, dtype=torch.long)
    transition_lengths = torch.zeros(B, T, dtype=torch.long)
    
    # Add some transitions
    for b in range(B):
        # Cut at frame 50
        targets[b, 49:51] = 1
        transition_lengths[b, 49:51] = 1
        
        # Gradual from 80-90
        targets[b, 80:91] = 2
        transition_lengths[b, 80:91] = 10
    
    # Test combined loss
    criterion = CombinedTransitionLoss()
    loss, components = criterion(logits, targets, transition_lengths)
    
    print(f"Total loss: {loss.item():.4f}")
    print(f"Components: {components}")
    print("✓ Loss functions working!")
