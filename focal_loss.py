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


if __name__ == "__main__":
    test_focal_loss()
