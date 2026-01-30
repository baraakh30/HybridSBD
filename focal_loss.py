"""
Focal Loss for handling extreme class imbalance.
Better than BCELoss with class weights for boundary detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    Focal Loss = -alpha * (1-p)^gamma * log(p)
    
    Where:
    - alpha: Balancing factor for positive/negative classes
    - gamma: Focusing parameter (higher = focus more on hard examples)
    - p: predicted probability
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Balancing factor for positive class (0.25 recommended for 1:3 ratio)
                   For 1:85 ratio, use alpha=0.85-0.95
            gamma: Focusing parameter (2.0 recommended, higher = harder focus on difficult examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Raw model outputs (before sigmoid), shape (N,) or (N, 1)
            targets: Binary targets (0 or 1), shape (N,) or (N, 1)
            
        Returns:
            Loss value
        """
        # Ensure same shape
        logits = logits.view(-1)
        targets = targets.view(-1)
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute focal weight: (1 - p_t)^gamma
        # p_t is the probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha balancing
        # alpha for positive class, (1-alpha) for negative class
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that automatically adjusts alpha based on class distribution.
    """
    
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Adaptive Focal Loss.
        
        Args:
            gamma: Focusing parameter
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('pos_count', torch.tensor(0.0))
        self.register_buffer('neg_count', torch.tensor(0.0))
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive focal loss.
        
        Args:
            logits: Raw model outputs (before sigmoid)
            targets: Binary targets (0 or 1)
            
        Returns:
            Loss value
        """
        # Update counts
        self.pos_count += targets.sum()
        self.neg_count += (1 - targets).sum()
        
        # Compute adaptive alpha
        total = self.pos_count + self.neg_count
        if total > 0:
            alpha = self.neg_count / total  # Higher alpha for minority class
        else:
            alpha = 0.5
        
        # Clip alpha to reasonable range
        alpha = torch.clamp(alpha, 0.1, 0.99)
        
        # Use regular focal loss with adaptive alpha
        focal_loss = FocalLoss(alpha=alpha.item(), gamma=self.gamma, reduction=self.reduction)
        return focal_loss(logits, targets)


def test_focal_loss():
    """Test focal loss implementation."""
    print("Testing Focal Loss...")
    
    # Create dummy data
    logits = torch.randn(100, 1)
    targets = torch.randint(0, 2, (100, 1)).float()
    
    # Test different losses
    bce_loss = nn.BCEWithLogitsLoss()
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    adaptive_focal_loss = AdaptiveFocalLoss(gamma=2.0)
    
    bce_val = bce_loss(logits, targets)
    focal_val = focal_loss(logits, targets)
    adaptive_val = adaptive_focal_loss(logits, targets)
    
    print(f"BCE Loss: {bce_val.item():.4f}")
    print(f"Focal Loss: {focal_val.item():.4f}")
    print(f"Adaptive Focal Loss: {adaptive_val.item():.4f}")
    print("\n✅ All losses computed successfully!")


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss with controllable FP/FN weighting.
    
    Tversky Index = TP / (TP + alpha*FP + beta*FN)
    
    - alpha=beta=0.5 → Dice Loss (F1)
    - alpha < beta → Penalize FN more (improve Recall)
    - alpha > beta → Penalize FP more (improve Precision)
    
    Paper: "Tversky loss function for image segmentation" (Salehi et al., 2017)
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        """
        Initialize Tversky Loss.
        
        Args:
            alpha: Weight for False Positives (lower = tolerate more FP)
            beta: Weight for False Negatives (higher = penalize FN more = better recall)
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky loss.
        
        Args:
            logits: Raw model outputs (before sigmoid)
            targets: Binary targets (0 or 1)
            
        Returns:
            Loss value (1 - Tversky Index)
        """
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives, False Negatives
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky_index


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Combines Tversky with focal mechanism for hard examples.
    
    Loss = (1 - Tversky)^gamma
    
    Higher gamma focuses more on hard examples with low Tversky scores.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.5, smooth: float = 1e-6):
        """
        Initialize Focal Tversky Loss.
        
        Args:
            alpha: Weight for False Positives
            beta: Weight for False Negatives  
            gamma: Focal parameter (higher = more focus on hard samples)
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Focal component: raise to power gamma
        focal_tversky = (1 - tversky_index) ** self.gamma
        
        return focal_tversky


class SoftF1Loss(nn.Module):
    """
    Soft F1 Loss - Directly optimizes the F1 score in a differentiable manner.
    
    This is the most direct way to optimize F1!
    
    Soft F1 = 2*TP / (2*TP + FP + FN)
    Loss = 1 - Soft F1
    """
    
    def __init__(self, smooth: float = 1e-6, label_smoothing: float = 0.0):
        """
        Initialize Soft F1 Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.smooth = smooth
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        
        soft_f1 = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        
        return 1 - soft_f1


class ComboLossV2(nn.Module):
    """
    Enhanced Combo Loss: Focal + Tversky + Soft F1
    
    Combines:
    - Focal Loss: Handles class imbalance + hard examples
    - Tversky Loss: Controllable precision/recall trade-off
    - Soft F1 Loss: Direct F1 optimization
    """
    
    def __init__(self, focal_alpha: float = 0.75, focal_gamma: float = 2.0,
                 tversky_alpha: float = 0.3, tversky_beta: float = 0.7,
                 focal_weight: float = 0.3, tversky_weight: float = 0.3, f1_weight: float = 0.4,
                 label_smoothing: float = 0.0):
        """
        Initialize ComboLossV2.
        
        Args:
            focal_alpha, focal_gamma: Focal loss parameters
            tversky_alpha, tversky_beta: Tversky loss parameters
            focal_weight: Weight for focal loss component
            tversky_weight: Weight for tversky loss component
            f1_weight: Weight for soft F1 loss component (MAIN COMPONENT)
            label_smoothing: Label smoothing factor for regularization
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.soft_f1_loss = SoftF1Loss(label_smoothing=label_smoothing)
        
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.f1_weight = f1_weight
        
        # Normalize weights
        total = focal_weight + tversky_weight + f1_weight
        self.focal_weight /= total
        self.tversky_weight /= total
        self.f1_weight /= total
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(logits, targets)
        tversky = self.tversky_loss(logits, targets)
        soft_f1 = self.soft_f1_loss(logits, targets)
        
        return self.focal_weight * focal + self.tversky_weight * tversky + self.f1_weight * soft_f1


# =============================================================================
# MULTI-TASK LEARNING LOSSES FOR GRADUAL TRANSITION DETECTION
# =============================================================================

class DurationAwareRegressionLoss(nn.Module):
    """
    Regression loss for predicting transition duration with special handling:
    - Log-scale regression (since durations range from 1 to 200+ frames)
    - Higher weight for gradual transitions (duration > 1)
    - Smooth L1 (Huber) loss for robustness to outliers
    - NOW: Also trains on non-boundary frames (target=0) to prevent drift
    - NEW: Contrastive-style margin loss to ensure cuts < 1.0 < graduals
    
    Predicts: log(1 + duration) to handle the wide range of durations
    Target values:
    - Non-boundary: 0
    - Cut (duration=2): log(3) = 1.10 (ClipShots: [n, n+1])
    - Gradual (duration=5): log(6) = 1.79
    - Gradual (duration=10): log(11) = 2.40
    """
    
    def __init__(self, gradual_weight: float = 3.0, smooth_beta: float = 1.0,
                 non_boundary_weight: float = 0.5, margin_weight: float = 0.5):
        """
        Args:
            gradual_weight: Extra weight for gradual transitions (duration > 1)
            smooth_beta: Beta parameter for Smooth L1 loss
            non_boundary_weight: Weight for non-boundary frames (prevents duration drift)
            margin_weight: Weight for contrastive margin loss (cuts < threshold < graduals)
        """
        super().__init__()
        self.gradual_weight = gradual_weight
        self.smooth_beta = smooth_beta
        self.non_boundary_weight = non_boundary_weight
        self.margin_weight = margin_weight
        
        # Threshold for separating cuts from graduals in log space
        # ClipShots: cuts have duration=2 -> log(1+2)=1.1
        # Graduals have duration>2 -> log(1+3)=1.39+
        # Use 1.25 as boundary (corresponds to duration ~2.5)
        self.cut_gradual_threshold = 1.25  # In log scale
        
    def forward(self, pred_duration: torch.Tensor, target_duration: torch.Tensor, 
                is_boundary: torch.Tensor) -> torch.Tensor:
        """
        Compute duration regression loss.
        
        Args:
            pred_duration: Predicted duration (log-scale), shape (N,)
            target_duration: Ground truth duration (raw frames), shape (N,)
            is_boundary: Binary mask for boundary frames, shape (N,)
            
        Returns:
            Weighted regression loss
        """
        pred = pred_duration.view(-1)
        target = target_duration.view(-1)
        boundary_mask = is_boundary.view(-1) > 0.5
        
        # Convert target to log-scale: log(1 + duration)
        # This compresses the range: 0 -> 0, 1 -> 0.69, 10 -> 2.4, 100 -> 4.6
        target_log = torch.log1p(target)
        
        # =====================================================================
        # BOUNDARY FRAMES: Train to predict correct duration
        # =====================================================================
        if boundary_mask.sum() > 0:
            pred_boundary = pred[boundary_mask]
            target_boundary = target_log[boundary_mask]
            
            # Smooth L1 loss (Huber loss) - robust to outliers
            loss_boundary = F.smooth_l1_loss(pred_boundary, target_boundary, 
                                             beta=self.smooth_beta, reduction='none')
            
            # Weight gradual transitions more heavily
            # Gradual = duration > 2 (ClipShots: cuts are [n, n+1] = dur 2)
            is_gradual = (target[boundary_mask] > 2.5).float()
            weights = 1.0 + (self.gradual_weight - 1.0) * is_gradual
            
            boundary_loss = (loss_boundary * weights).mean()
        else:
            boundary_loss = torch.tensor(0.0, device=pred.device)
        
        # =====================================================================
        # NON-BOUNDARY FRAMES: Train to predict LOW duration (near 0)
        # This prevents the model from predicting high duration everywhere!
        # =====================================================================
        non_boundary_mask = ~boundary_mask
        if non_boundary_mask.sum() > 0 and self.non_boundary_weight > 0:
            pred_non_boundary = pred[non_boundary_mask]
            # Target: 0 (no transition) -> log(1+0) = 0
            target_non_boundary = torch.zeros_like(pred_non_boundary)
            
            # Use same Smooth L1 loss
            loss_non_boundary = F.smooth_l1_loss(pred_non_boundary, target_non_boundary,
                                                  beta=self.smooth_beta, reduction='mean')
            
            non_boundary_loss = self.non_boundary_weight * loss_non_boundary
        else:
            non_boundary_loss = torch.tensor(0.0, device=pred.device)
        
        # =====================================================================
        # CONTRASTIVE MARGIN LOSS: Force cuts and graduals to be separable
        # Cuts should predict < threshold, Graduals should predict > threshold
        # This ensures clear discrimination at test time
        # =====================================================================
        margin_loss = torch.tensor(0.0, device=pred.device)
        if boundary_mask.sum() > 0 and self.margin_weight > 0:
            pred_boundary = pred[boundary_mask]
            raw_target = target[boundary_mask]  # Raw duration (not log)
            
            # Identify cuts (duration <= 2) and graduals (duration > 2)
            # ClipShots format: cuts are [n, n+1] so duration=2
            is_cut = raw_target <= 2.5
            is_gradual = raw_target > 2.5
            
            # Margin: cuts should be < threshold - margin, graduals > threshold + margin
            margin = 0.3  # Buffer zone around threshold
            
            if is_cut.sum() > 0:
                # Cuts should predict BELOW threshold - margin
                # Penalize if pred > (threshold - margin)
                cut_preds = pred_boundary[is_cut]
                cut_violations = F.relu(cut_preds - (self.cut_gradual_threshold - margin))
                margin_loss = margin_loss + cut_violations.mean()
            
            if is_gradual.sum() > 0:
                # Graduals should predict ABOVE threshold + margin
                # Penalize if pred < (threshold + margin)
                gradual_preds = pred_boundary[is_gradual]
                gradual_violations = F.relu((self.cut_gradual_threshold + margin) - gradual_preds)
                margin_loss = margin_loss + gradual_violations.mean()
            
            margin_loss = self.margin_weight * margin_loss
        
        return boundary_loss + non_boundary_loss + margin_loss


class MultiTaskBoundaryLoss(nn.Module):
    """
    Multi-Task Loss for boundary detection with duration regression.
    
    Combines:
    1. Binary Classification: Is this frame a transition? (ComboLossV2)
    2. Duration Regression: How long is the transition? (DurationAwareLoss)
    
    The duration head helps the model learn:
    - Cuts: Sharp feature changes, duration ≈ 1
    - Gradual: Smooth feature changes over multiple frames, duration > 1
    
    This explicitly teaches the model the SIGNATURE of gradual transitions!
    """
    
    def __init__(self, 
                 # Classification loss params
                 focal_alpha: float = 0.75, focal_gamma: float = 2.0,
                 tversky_alpha: float = 0.4, tversky_beta: float = 0.6,
                 focal_weight: float = 0.3, tversky_weight: float = 0.4, f1_weight: float = 0.3,
                 # Duration loss params
                 gradual_weight: float = 3.0,
                 margin_weight: float = 0.5,  # NEW: Contrastive margin loss weight
                 # Task balancing
                 classification_weight: float = 0.6,
                 duration_weight: float = 0.4):
        """
        Initialize Multi-Task Loss.
        
        Args:
            focal_alpha, focal_gamma: Focal loss parameters
            tversky_alpha, tversky_beta: Tversky loss parameters  
            focal_weight, tversky_weight, f1_weight: Weights for classification components
            gradual_weight: Extra weight for gradual transitions in duration loss
            margin_weight: Weight for contrastive margin loss (cuts vs graduals separation)
            classification_weight: Weight for classification task (0.6 = 60%)
            duration_weight: Weight for duration regression task (0.4 = 40%)
        """
        super().__init__()
        
        # Classification loss (existing ComboLossV2)
        self.classification_loss = ComboLossV2(
            focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
            focal_weight=focal_weight, tversky_weight=tversky_weight, f1_weight=f1_weight
        )
        
        # Duration regression loss with margin for cut/gradual separation
        self.duration_loss = DurationAwareRegressionLoss(
            gradual_weight=gradual_weight,
            margin_weight=margin_weight
        )
        
        # Task weights
        self.classification_weight = classification_weight
        self.duration_weight = duration_weight
        
    def forward(self, class_logits: torch.Tensor, duration_pred: torch.Tensor,
                class_targets: torch.Tensor, duration_targets: torch.Tensor) -> dict:
        """
        Compute multi-task loss.
        
        Args:
            class_logits: Classification logits (batch, seq_len-1)
            duration_pred: Duration predictions in log-scale (batch, seq_len-1)
            class_targets: Binary boundary targets (batch, seq_len-1)
            duration_targets: Duration targets in frames (batch, seq_len-1)
            
        Returns:
            Dictionary with 'total', 'classification', 'duration' losses
        """
        # Classification loss
        cls_loss = self.classification_loss(class_logits, class_targets)
        
        # Duration regression loss (only for boundary frames)
        dur_loss = self.duration_loss(duration_pred, duration_targets, class_targets)
        
        # Combined loss
        total_loss = self.classification_weight * cls_loss + self.duration_weight * dur_loss
        
        return {
            'total': total_loss,
            'classification': cls_loss,
            'duration': dur_loss
        }


class GradualTransitionFocalLoss(nn.Module):
    """
    Specialized Focal Loss that gives EXTRA weight to gradual transitions.
    
    Problem: Gradual transitions are:
    1. Rarer than cuts (1:3.7 ratio in ClipShots)
    2. Have subtle feature changes (low magnitude differences)
    3. Easy to miss without explicit handling
    
    Solution: Use duration as a weighting factor:
    - Cuts (duration=2): Standard focal loss weight (ClipShots format)
    - Gradual (duration>1): Multiplied by gradual_boost factor
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, 
                 gradual_boost: float = 2.0, min_gradual_duration: int = 2):
        """
        Args:
            alpha: Base focal loss alpha
            gamma: Focal loss gamma (focusing parameter)
            gradual_boost: Multiplier for gradual transition loss (2.0 = 2x weight)
            min_gradual_duration: Minimum duration to be considered gradual
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.gradual_boost = gradual_boost
        self.min_gradual_duration = min_gradual_duration
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                durations: torch.Tensor = None) -> torch.Tensor:
        """
        Compute focal loss with gradual transition boosting.
        
        Args:
            logits: Prediction logits
            targets: Binary targets
            durations: Duration for each frame (optional, for gradual boosting)
        """
        logits = logits.view(-1)
        targets = targets.view(-1)
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha balancing
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Base focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply gradual boost if durations provided
        if durations is not None:
            durations = durations.view(-1)
            is_gradual = (durations >= self.min_gradual_duration).float()
            # Boost gradual transitions
            gradual_weight = 1.0 + (self.gradual_boost - 1.0) * is_gradual * targets
            focal_loss = focal_loss * gradual_weight
        
        return focal_loss.mean()


if __name__ == "__main__":
    test_focal_loss()
