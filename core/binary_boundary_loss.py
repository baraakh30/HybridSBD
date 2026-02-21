"""
Dual-Head Boundary Loss - AutoShot-Style (2026)

Implements AutoShot's proven dual-head training strategy:
  Head y (one_hot):  Midpoint of each transition (primary, lambda1=5.0)
  Head z (many_hot): All frames within gradual transitions (auxiliary, lambda2=0.1)

Loss: L = lambda1 * FocalBCE(y_hat, y_one_hot) + lambda2 * FocalBCE(z_hat, z_many_hot)

Label Generation:
  one_hot:  For each transition (cut or gradual), mark the single midpoint frame as 1.
            - Cut at frame 30: one_hot[30] = 1
            - Gradual spanning frames 60-70: one_hot[65] = 1
  many_hot: For each GRADUAL transition, mark ALL frames as 1.
            - Cut at frame 30: many_hot[30] = 0 (cuts are instantaneous)
            - Gradual spanning frames 60-70: many_hot[60:71] = 1

WHY THIS WORKS:
  - one_hot forces the model to pinpoint exact boundary locations (sparse prediction)
  - many_hot teaches the model what gradual transitions look like (extent awareness)
  - lambda1 >> lambda2 ensures one_hot head dominates training
  - Focal loss handles extreme class imbalance (~1% positive for one_hot)
  - No complex type classification = no gradient contamination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_one_hot_many_hot(targets):
    """
    Convert 3-class frame labels to AutoShot-style dual targets.
    Fully vectorized — no Python loops over batch.

    Args:
        targets: [B, T] with 0=none, 1=cut, 2=gradual

    Returns:
        one_hot:  [B, T] float - 1 at midpoint of each transition (cuts + graduals)
        many_hot: [B, T] float - 1 at ALL gradual transition frames only
    """
    B, T = targets.shape
    many_hot = (targets == 2).float()

    # Find transition midpoints vectorized:
    # A transition region starts where (target > 0) goes from False → True,
    # ends where it goes from True → False.
    is_trans = (targets > 0).float()           # [B, T]

    # Pad with zeros on both sides to detect edges at boundaries
    padded = F.pad(is_trans, (1, 1), value=0)  # [B, T+2]
    rising  = (padded[:, 1:] - padded[:, :-1]).clamp(min=0)   # 1 at start of region [B, T+1]
    falling = (padded[:, :-1] - padded[:, 1:]).clamp(min=0)   # 1 at end of region   [B, T+1]

    # start_mask[b,t]=1 if a transition starts at frame t
    # end_mask[b,t]=1 if a transition ends AFTER frame t (exclusive end)
    start_mask = rising[:, :-1]    # [B, T] — drop the appended column
    end_mask   = falling[:, 1:]    # [B, T] — drop the prepended column

    # Cumulative sum trick: running start index
    # For each frame, find its region start by propagating the last start position.
    # We do this by computing a "region index" via cumsum on starts.
    region_id = start_mask.long().cumsum(dim=1)  # [B, T] counts how many regions started by t

    # For each region, record its start frame and end frame using scatter
    # max_regions = max number of distinct transitions in any clip
    max_regions = int(start_mask.sum(dim=1).max().item()) + 1

    one_hot = torch.zeros(B, T, device=targets.device, dtype=torch.float)

    if max_regions > 1:
        # For each frame in a transition, its region_id gives which transition it belongs to
        # Collect min (start) and max (end) frame index per region per batch item
        in_trans = is_trans.bool()   # [B, T]

        # Use a loop over batch only when needed — inner work is tensor ops
        for b in range(B):
            starts_b = torch.where(start_mask[b] > 0)[0]   # frame indices where regions start
            ends_b   = torch.where(end_mask[b] > 0)[0]     # frame indices where regions end (inclusive)
            if starts_b.numel() == 0:
                continue
            n = min(starts_b.numel(), ends_b.numel())
            mids = (starts_b[:n] + ends_b[:n]) // 2
            mids = mids.clamp(0, T - 1)
            one_hot[b].scatter_(0, mids, 1.0)

    return one_hot, many_hot


class DualHeadBoundaryLoss(nn.Module):
    """
    AutoShot-style dual-head boundary detection loss.
    
    Loss = lambda1 * FocalBCE(y_hat, one_hot) + lambda2 * FocalBCE(z_hat, many_hot)
    
    Args:
        lambda1: Weight for one_hot head (default 5.0 per AutoShot paper)
        lambda2: Weight for many_hot head (default 0.1 per AutoShot paper)
        focal_gamma: Focal loss gamma for hard example mining (default 2.0)
        focal_alpha: Focal loss alpha for class weighting (default 0.75)
    """
    def __init__(
        self,
        lambda1=5.0,
        lambda2=0.1,
        focal_gamma=2.0,
        focal_alpha=0.75,
        focal_alpha_z=None,       # If None, falls back to focal_alpha for backward compat
        z_label_smoothing=0.1,  # Smooth many_hot targets: reduces z_loss memorization
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        # Separate alpha for z-head (many_hot).  y-head targets are ~0.8% positive
        # (one frame per 128) — high alpha is correct.  z-head targets are 0% positive
        # for cut-only clips (79% of ClipShots) and 10-30% positive for gradual clips;
        # using the same high alpha pushes z-head to fire everywhere on cut clips.
        self.focal_alpha_z = focal_alpha_z if focal_alpha_z is not None else focal_alpha
        self.z_label_smoothing = z_label_smoothing
    
    def focal_bce(self, logits, targets, alpha=None):
        """
        Focal Binary Cross-Entropy loss.
        
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
        Handles extreme class imbalance by:
        - alpha weighting: upweight rare positive class
        - focal weighting: downweight easy examples, focus on hard ones
        
        Args:
            logits: [B, T] raw predictions (before sigmoid)
            targets: [B, T] binary labels (0 or 1), float
            alpha:  Override focal alpha for this call (uses self.focal_alpha if None)
        
        Returns:
            Scalar loss
        """
        a = alpha if alpha is not None else self.focal_alpha
        probs = torch.sigmoid(logits)
        
        # p_t: probability of true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # Alpha weighting
        alpha_weight = a * targets + (1 - a) * (1 - targets)
        
        # BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Combine
        loss = focal_weight * alpha_weight * bce_loss
        
        return loss.mean()
    
    def forward(self, y_logits, z_logits, targets, transition_lengths=None):
        """
        Compute dual-head loss.
        
        Args:
            y_logits: [B, T, 1] one_hot head predictions
            z_logits: [B, T, 1] many_hot head predictions
            targets:  [B, T] original 3-class labels (0=none, 1=cut, 2=gradual)
            transition_lengths: [B, T] unused, kept for API compatibility
        
        Returns:
            total_loss: Scalar
            components: Dict with individual loss values
        """
        # Squeeze channel dimension
        y_logits_flat = y_logits.squeeze(-1)  # [B, T]
        z_logits_flat = z_logits.squeeze(-1)  # [B, T]
        
        # Create one_hot and many_hot targets from 3-class labels
        one_hot, many_hot = create_one_hot_many_hot(targets)
        
        # Head y loss (PRIMARY: midpoint detection)
        # focal_alpha=0.80 is correct: one_hot targets are ~0.8% positive
        loss_y = self.focal_bce(y_logits_flat, one_hot, alpha=self.focal_alpha)
        
        # Head z loss (AUXILIARY: gradual extent) - apply label smoothing
        # Use focal_alpha_z (default 0.50, balanced) instead of focal_alpha (0.80).
        # Rationale: 79% of ClipShots clips are cuts → z-targets are ALL-ZERO for those clips.
        # With alpha=0.80 the cost of z-FP (weight=0.20) is trivial → z-head fires everywhere.
        # With alpha=0.50 FP and FN are penalised equally → z-head learns to be conservative.
        many_hot_smooth = many_hot * (1.0 - self.z_label_smoothing)
        loss_z = self.focal_bce(z_logits_flat, many_hot_smooth, alpha=self.focal_alpha_z)
        
        # Total loss with AutoShot weighting
        total_loss = self.lambda1 * loss_y + self.lambda2 * loss_z
        
        # Compute diagnostics
        with torch.no_grad():
            y_probs = torch.sigmoid(y_logits_flat)
            z_probs = torch.sigmoid(z_logits_flat)
            
            # Count positive targets
            n_one_hot = one_hot.sum().item()
            n_many_hot = many_hot.sum().item()
            total_frames = targets.numel()
            
            # Mean probabilities at positive targets
            y_at_positives = y_probs[one_hot > 0].mean().item() if n_one_hot > 0 else 0.0
            z_at_positives = z_probs[many_hot > 0].mean().item() if n_many_hot > 0 else 0.0
        
        components = {
            'y_loss': loss_y.item(),
            'z_loss': loss_z.item(),
            'total': total_loss.item(),
            'n_one_hot': n_one_hot,
            'n_many_hot': n_many_hot,
            'one_hot_density': n_one_hot / total_frames if total_frames > 0 else 0,
            'many_hot_density': n_many_hot / total_frames if total_frames > 0 else 0,
            'y_prob_at_positives': y_at_positives,
            'z_prob_at_positives': z_at_positives,
        }
        
        return total_loss, components


def create_binary_boundary_loss(config=None):
    """
    Factory function for dual-head boundary loss.
    
    Args:
        config: Optional config dict with loss hyperparameters
    
    Returns:
        DualHeadBoundaryLoss instance
    """
    if config is None:
        config = {}
    
    return DualHeadBoundaryLoss(
        lambda1=config.get('lambda1', 5.0),
        lambda2=config.get('lambda2', 0.1),
        focal_gamma=config.get('focal_gamma', 2.0),
        focal_alpha=config.get('focal_alpha', 0.75),
        focal_alpha_z=config.get('focal_alpha_z', None),  # None → falls back to focal_alpha
        z_label_smoothing=config.get('z_label_smoothing', 0.1),
    )


if __name__ == '__main__':
    print("=" * 80)
    print("TESTING DUAL-HEAD BOUNDARY LOSS (AutoShot-Style)")
    print("=" * 80)
    
    B, T = 4, 128
    
    # Simulate model outputs
    y_logits = torch.randn(B, T, 1)
    z_logits = torch.randn(B, T, 1)
    
    # Ground truth (3-class format)
    targets = torch.zeros(B, T, dtype=torch.long)
    targets[0, 30] = 1      # cut at frame 30
    targets[1, 45] = 1      # cut at frame 45
    targets[2, 60:65] = 2   # gradual spanning 5 frames
    targets[3, 80:90] = 2   # gradual spanning 10 frames
    
    transition_lengths = torch.zeros(B, T, dtype=torch.long)
    
    # Test label generation
    one_hot, many_hot = create_one_hot_many_hot(targets)
    print(f"\nLabel generation test:")
    print(f"  Sample 0 (cut@30):      one_hot peaks at {torch.where(one_hot[0] > 0)[0].tolist()}")
    print(f"  Sample 1 (cut@45):      one_hot peaks at {torch.where(one_hot[1] > 0)[0].tolist()}")
    print(f"  Sample 2 (grad 60-64):  one_hot peaks at {torch.where(one_hot[2] > 0)[0].tolist()}, "
          f"many_hot frames: {torch.where(many_hot[2] > 0)[0].tolist()}")
    print(f"  Sample 3 (grad 80-89):  one_hot peaks at {torch.where(one_hot[3] > 0)[0].tolist()}, "
          f"many_hot frames: {torch.where(many_hot[3] > 0)[0].tolist()}")
    
    # Compute loss
    loss_fn = create_binary_boundary_loss()
    total_loss, components = loss_fn(y_logits, z_logits, targets, transition_lengths)
    
    print(f"\nLoss Components:")
    print(f"  y_loss (one_hot, lambda={loss_fn.lambda1}):  {components['y_loss']:.4f}")
    print(f"  z_loss (many_hot, lambda={loss_fn.lambda2}): {components['z_loss']:.4f}")
    print(f"  Total:             {components['total']:.4f}")
    print(f"  One-hot density:   {components['one_hot_density']:.4f} ({components['n_one_hot']:.0f} frames)")
    print(f"  Many-hot density:  {components['many_hot_density']:.4f} ({components['n_many_hot']:.0f} frames)")
    print(f"  y_prob at positives: {components['y_prob_at_positives']:.4f}")
    print(f"  z_prob at positives: {components['z_prob_at_positives']:.4f}")
    
    # Test gradient flow
    total_loss.backward()
    print(f"\nGradient flow successful!")
    
    print("\n" + "=" * 80)
    print("Loss function test passed!")
    print("=" * 80)
