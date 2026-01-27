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
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        
        # Apply label smoothing to prevent overconfidence
        # This helps generalization by making the model less certain
        label_smoothing = 0.05
        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
        
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
                 focal_weight: float = 0.3, tversky_weight: float = 0.3, f1_weight: float = 0.4):
        """
        Initialize ComboLossV2.
        
        Args:
            focal_alpha, focal_gamma: Focal loss parameters
            tversky_alpha, tversky_beta: Tversky loss parameters
            focal_weight: Weight for focal loss component
            tversky_weight: Weight for tversky loss component
            f1_weight: Weight for soft F1 loss component (MAIN COMPONENT)
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.soft_f1_loss = SoftF1Loss()
        
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


if __name__ == "__main__":
    test_focal_loss()
