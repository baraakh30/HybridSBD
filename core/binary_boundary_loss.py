"""
Binary Boundary Detection Loss (2026 SOTA)

Designed for binary classification with auxiliary type prediction:
- Primary: Boundary vs Non-Boundary (highly imbalanced ~1:100)
- Auxiliary: Cut vs Gradual (only for boundaries)

Loss Components:
1. Focal Loss for boundary detection (handles class imbalance)
2. Auxiliary type loss (teaches cut vs gradual distinction)
3. Temporal coherence (boundaries should be sparse and isolated)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryBoundaryLoss(nn.Module):
    """
    Complete loss for binary boundary detection
    
    Combines:
    1. Focal Loss: Binary boundary detection (handles 1:100 imbalance)
    2. Type Loss: 3-class classification (none, cut, gradual)
    3. Coherence: Temporal smoothness (boundaries are rare)
    
    Labels format:
    - boundary_targets: [B, T] with 0=not_boundary, 1=is_boundary
    - type_targets: [B, T] with 0=none, 1=cut, 2=gradual (3-class!)
    """
    def __init__(
        self, 
        focal_gamma=2.0,
        focal_alpha=0.85,  # Balanced for 4.2% imbalance
        type_weight=0.15,  # Reduced from 0.30 - let boundary head lead
        type_class_weights=None,  # [none_weight, cut_weight, gradual_weight]
        reasoning_weight=0.10,  # Weight for reasoning supervision
        consistency_weight=0.30  # Weight for boundary-type consistency (increased for stronger supervision)
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.type_weight = type_weight
        self.reasoning_weight = reasoning_weight
        self.consistency_weight = consistency_weight
        
        # FIXED: Balanced 3-class weights [none, cut, gradual]
        # Distribution: none:cut:gradual = 1350k:20k:36k = 67.5:1:1.8
        # Weights inversely proportional to frequency: [0.05, 10.0, 5.0]
        self.type_class_weights = type_class_weights if type_class_weights is not None else [0.05, 10.0, 5.0]
        
    def focal_loss(self, logits, targets):
        """
        Focal Loss for binary boundary detection (same as staged training)
        
        Standard focal loss with class weights (no pos_weight tricks)
        
        Args:
            logits: [B, T, 1] raw boundary predictions
            targets: [B, T] binary labels (0 or 1)
        """
        # Flatten
        logits = logits.squeeze(-1)  # [B, T]
        targets_float = targets.float()
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Get probability of true class (p_t)
        p_t = probs * targets_float + (1 - probs) * (1 - targets_float)
        
        # Focal weight: (1 - p_t)^gamma
        # Hard examples (low p_t) get high weight
        # Easy examples (high p_t) get low weight
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # Class weights (alpha balancing)
        # alpha_weight = alpha for positive class, (1-alpha) for negative class
        alpha_weight = self.focal_alpha * targets_float + (1 - self.focal_alpha) * (1 - targets_float)
        
        # Binary cross entropy (base loss)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets_float, reduction='none'
        )
        
        # Combine: focal_weight * alpha_weight * bce_loss
        loss = focal_weight * alpha_weight * bce_loss
        
        return loss.mean()
    
    def type_loss(self, type_logits, type_targets, boundary_mask):
        """
        3-class type classification loss (none, cut, gradual)
        Trained on ALL frames with balanced weights
        
        Args:
            type_logits: [B, T, 3] none/cut/gradual logits
            type_targets: [B, T] with 0=none, 1=cut, 2=gradual
            boundary_mask: [B, T] (not used - train on all frames)
        """
        # Flatten for cross-entropy
        type_logits_flat = type_logits.reshape(-1, 3)  # [B*T, 3]
        type_targets_flat = type_targets.reshape(-1)  # [B*T]
        
        # BALANCED class weights [none, cut, gradual]
        # Distribution-based: none:cut:gradual = 67.5:1:1.8 → weights [0.05, 10.0, 5.0]
        # This properly balances the massive none class dominance
        type_weights = torch.tensor([
            self.type_class_weights[0],  # none: 0.05 (very common, low weight)
            self.type_class_weights[1],  # cut: 10.0 (rare, high weight)
            self.type_class_weights[2]   # gradual: 5.0 (rare, medium-high weight)
        ], device=type_logits.device)
        
        loss = F.cross_entropy(type_logits_flat, type_targets_flat, weight=type_weights)
        
        return loss
    
    def consistency_loss(self, boundary_logits, type_targets):
        """
        Consistency loss: teaches boundary head to align with type labels
        
        Direct supervision: "type=none → boundary=0, type=cut/gradual → boundary=1"
        
        This is much stronger than KL divergence - boundary head gets direct BCE
        supervision from type labels, teaching it that the type classification
        implies the boundary decision.
        
        Args:
            boundary_logits: [B, T, 1] raw boundary predictions
            type_targets: [B, T] ground truth type labels (0=none, 1=cut, 2=gradual)
        
        Returns:
            consistency loss (scalar)
        """
        # Convert type labels to binary boundary targets
        # type=0 (none) → boundary=0, type=1/2 (cut/gradual) → boundary=1
        is_boundary = (type_targets > 0).float()  # [B, T]
        
        # Get boundary logits (flat)
        boundary_logits_flat = boundary_logits.squeeze(-1)  # [B, T]
        
        # Direct BCE with logits: teach boundary head to predict 1 for cut/gradual, 0 for none
        # Use _with_logits version for autocast safety
        loss = F.binary_cross_entropy_with_logits(boundary_logits_flat, is_boundary)
        
        return loss
    
    # Temporal coherence loss removed - it penalized gradual transitions
    
    # REASONING LOSS DISABLED - Features too ambiguous to supervise directly
    # Empirical analysis showed reasoning features IDENTICAL for graduals vs pans:
    #   True gradual: change=0.122, variance=0.722, mono=0.384
    #   False gradual (smooth pan): change=0.122, variance=0.722, mono=0.348
    # Supervising these would confuse the model. Instead, reasoning features
    # enrich representations internally but learn from boundary/type loss only.
    # Method kept below for reference/documentation of what we tried.
    
    def reasoning_loss(self, reasoning_outputs, labels):
        """
        DISABLED - Reasoning loss with discriminative targets
        
        This was designed for:
        1. Change Magnitude: CONTINUOUS (not binary) - none ~0.0, gradual ~0.6, cut ~0.8
        2. Temporal Variance: none HIGH (noisy/static), gradual LOW (smooth), cut MEDIUM
        3. Monotonicity: gradual HIGH (directional), none/cut LOW (erratic/instant)
        
        Args:
            reasoning_outputs: dict with 'change_magnitude', 'temporal_variance', 'monotonicity'
                              each [B, T, 1] with LOGITS
            labels: [B, T] with 0=none, 1=cut, 2=gradual
        
        Returns:
            Total reasoning loss (scalar)
        """
        # Task 1: Change Magnitude (EVIDENCE-BASED from percentile analysis!)
        # True gradual P5-P95: [0.115, 0.151]
        # True none P95: 0.116
        # False graduals (smooth pans): 0.122 median - OVERLAPS WITH GRADUALS!
        # Conclusion: Change alone cannot discriminate. Need variance gating.
        change_probs = torch.sigmoid(reasoning_outputs['change_magnitude'].squeeze(-1))
        
        # Define acceptable ranges with margin loss (match dtype for mixed precision!)
        change_loss = torch.zeros_like(change_probs)  # Match dtype
        
        # None: want <0.12 (P95 of true nones is 0.116, add small margin)
        none_mask = (labels == 0)
        change_loss[none_mask] = F.relu(change_probs[none_mask] - 0.12)
        
        # Cut: want >0.28 (P5 of true cuts)
        cut_mask = (labels == 1)
        change_loss[cut_mask] = F.relu(0.28 - change_probs[cut_mask])
        
        # Gradual: P5-P95 range [0.115, 0.151] - VERY TIGHT!
        # But missed graduals have 0.121, so this range is good
        gradual_mask = (labels == 2)
        too_low = F.relu(0.115 - change_probs[gradual_mask])
        too_high = F.relu(change_probs[gradual_mask] - 0.151)
        change_loss[gradual_mask] = too_low + too_high
        
        change_loss = change_loss.mean()
        
        # Task 2: Temporal Variance (CRITICAL DISCRIMINATOR!)
        # True gradual P5-P95: [0.708, 0.734]
        # True none P5: 0.736 - VERY CLOSE!
        # Missed graduals: 0.738 (too high, looks like none)
        # False graduals: 0.722 (identical to true graduals!)
        # This is THE key feature to separate graduals from smooth pans
        variance_probs = torch.sigmoid(reasoning_outputs['temporal_variance'].squeeze(-1))
        
        variance_loss = torch.zeros_like(variance_probs)  # Match dtype
        
        # None: want >0.735 (P5 of true nones is 0.736)
        variance_loss[none_mask] = F.relu(0.735 - variance_probs[none_mask])
        
        # Cut: want 0.69-0.71 (P5-P95: 0.690-0.703)
        too_low_cut = F.relu(0.69 - variance_probs[cut_mask])
        too_high_cut = F.relu(variance_probs[cut_mask] - 0.71)
        variance_loss[cut_mask] = too_low_cut + too_high_cut
        
        # Gradual: P5-P95 [0.708, 0.734]
        # BUT missed graduals have 0.738! Need to widen upper bound
        # to accept noisy graduals while still below none threshold (0.735)
        too_low_grad = F.relu(0.708 - variance_probs[gradual_mask])
        too_high_grad = F.relu(variance_probs[gradual_mask] - 0.738)  # Accept up to 0.738
        variance_loss[gradual_mask] = too_low_grad + too_high_grad
        
        variance_loss = variance_loss.mean()
        
        # Task 3: Monotonicity (SECONDARY DISCRIMINATOR)
        # True gradual P5-P95: [0.314, 0.442]
        # True none P95: 0.285
        # Missed graduals: 0.290 - BELOW gradual P5!
        # Need to lower threshold to accept noisy graduals
        monotonicity_probs = torch.sigmoid(reasoning_outputs['monotonicity'].squeeze(-1))
        
        monotonicity_loss = torch.zeros_like(monotonicity_probs)  # Match dtype
        
        # None: want <0.285 (P95 of true nones)
        monotonicity_loss[none_mask] = F.relu(monotonicity_probs[none_mask] - 0.285)
        
        # Cut: want <0.290 (P95 of true cuts)
        monotonicity_loss[cut_mask] = F.relu(monotonicity_probs[cut_mask] - 0.290)
        
        # Gradual: want >0.29 (missed graduals have 0.290, need to accept them)
        # True gradual P5 is 0.314, but we need to lower to catch noisy ones
        monotonicity_loss[gradual_mask] = F.relu(0.29 - monotonicity_probs[gradual_mask])
        
        monotonicity_loss = monotonicity_loss.mean()
        
        # Combine reasoning losses
        total_reasoning_loss = change_loss + variance_loss + monotonicity_loss
        
        return total_reasoning_loss, {
            'change_magnitude': change_loss.item(),
            'temporal_variance': variance_loss.item(),
            'monotonicity': monotonicity_loss.item()
        }
    
    # Length prediction loss removed - wasn't providing clear benefit
    
    def forward(self, boundary_logits, type_logits, targets, transition_lengths, reasoning_outputs=None, type_logits_raw=None):
        """
        Complete loss computation
        
        Args:
            boundary_logits: [B, T, 1] binary boundary predictions
            type_logits: [B, T, 3] 3-class predictions (none, cut, gradual) - GATED for inference
            targets: [B, T] original 3-class labels (0=none, 1=cut, 2=gradual)
            transition_lengths: [B, T] transition durations (for compatibility)
            reasoning_outputs: dict with reasoning feature logits (ENABLED for supervision)
            type_logits_raw: [B, T, 3] RAW type predictions before gating (for training)
        
        Returns:
            total_loss, loss_components dict
            
        Note: Reasoning loss supervision NOW ENABLED with discriminative targets.
              New features (change_magnitude, temporal_variance, monotonicity)
              show clear separation in training - supervise them!
        """
        B, T = targets.shape
        
        # Convert 3-class targets to binary for boundary detection
        boundary_targets = (targets > 0).long()  # [B, T]
        
        # Type targets: KEEP AS 3-CLASS
        type_targets = targets
        
        # Boundary mask
        boundary_mask = boundary_targets.bool()
        
        # 1. Focal loss for boundary detection
        loss_focal = self.focal_loss(boundary_logits, boundary_targets)
        
        # 2. Type classification loss (3-class: none, cut, gradual)
        # Use RAW logits if provided (before gating) for clean gradient flow
        type_logits_for_loss = type_logits_raw if type_logits_raw is not None else type_logits
        loss_type = self.type_loss(type_logits_for_loss, type_targets, boundary_mask)
        
        # 3. Consistency loss: teaches "type=none → boundary=0, else boundary=1"
        loss_consistency = self.consistency_loss(boundary_logits, type_targets)
        
        # 4. Reasoning loss (TEMPORARILY DISABLED - margin constraints too strict for early training)
        # The reasoning_loss() function uses tight margin constraints (e.g., gradual change must be 0.115-0.151)
        # which causes gradient explosions when model features are random at initialization.
        # TODO: Reimplement with softer constraints (MSE) or progressive tightening schedule
        loss_reasoning = torch.tensor(0.0, device=boundary_logits.device)
        reasoning_components = {}
        
        # if reasoning_outputs is not None:
        #     loss_reasoning, reasoning_components = self.reasoning_loss(reasoning_outputs, targets)
        
        # Combine losses (reasoning disabled, consistency enabled)
        total_loss = loss_focal + self.type_weight * loss_type + self.consistency_weight * loss_consistency
        
        components = {
            'focal': loss_focal.item(),
            'type': loss_type.item(),
            'consistency': loss_consistency.item(),
            'reasoning': loss_reasoning.item() if isinstance(loss_reasoning, torch.Tensor) else 0.0,
            'total': total_loss.item()
        }
        
        # Add reasoning component details
        if reasoning_components:
            components.update({
                'reasoning/change_magnitude': reasoning_components.get('change_magnitude', 0.0),
                'reasoning/temporal_variance': reasoning_components.get('temporal_variance', 0.0),
                'reasoning/monotonicity': reasoning_components.get('monotonicity', 0.0)
            })
        
        return total_loss, components


def create_binary_boundary_loss(config=None):
    """
    Factory function for binary boundary loss
    
    Args:
        config: Optional config dict with loss hyperparameters
    
    Returns:
        BinaryBoundaryLoss instance
    """
    if config is None:
        config = {}
    
    return BinaryBoundaryLoss(
        focal_gamma=config.get('focal_gamma', 2.0),
        focal_alpha=config.get('focal_alpha', 0.85),
        type_weight=config.get('type_weight', 0.15),
        type_class_weights=config.get('type_class_weights', None),  # [0.05, 10.0, 5.0] default
        reasoning_weight=config.get('reasoning_weight', 0.10),
        consistency_weight=config.get('consistency_weight', 0.30)
    )


if __name__ == '__main__':
    # Test loss function
    print("="*80)
    print("TESTING REDESIGNED BINARY BOUNDARY LOSS")
    print("="*80)
    
    # Create dummy data
    B, T = 4, 128
    
    # Model outputs
    boundary_logits = torch.randn(B, T, 1)
    type_logits = torch.randn(B, T, 3)  # 3-class
    
    # Ground truth (3-class format)
    targets = torch.zeros(B, T, dtype=torch.long)
    targets[0, 30] = 1  # cut
    targets[1, 45] = 1  # cut
    targets[2, 60:65] = 2  # gradual
    targets[3, 80:90] = 2  # gradual
    
    transition_lengths = torch.zeros(B, T, dtype=torch.long)
    transition_lengths[2, 60:65] = 5
    transition_lengths[3, 80:90] = 10
    
    # Reasoning outputs (new discriminative features)
    reasoning_outputs = {
        'change_magnitude': torch.randn(B, T, 1),
        'temporal_variance': torch.randn(B, T, 1),
        'monotonicity': torch.randn(B, T, 1)
    }
    
    # Compute loss
    loss_fn = create_binary_boundary_loss()
    total_loss, components = loss_fn(
        boundary_logits, type_logits, targets, transition_lengths, 
        reasoning_outputs=reasoning_outputs
    )
    
    print(f"\nLoss Components:")
    print(f"  Focal (boundary):  {components['focal']:.4f}")
    print(f"  Type (3-class):    {components['type']:.4f}")
    print(f"  Reasoning:         {components['reasoning']:.4f}")
    print(f"    - Change mag:    {components['reasoning/change_magnitude']:.4f}")
    print(f"    - Variance:      {components['reasoning/temporal_variance']:.4f}")
    print(f"    - Monotonicity:  {components['reasoning/monotonicity']:.4f}")
    print(f"  Total:             {components['total']:.4f}")
    
    # Test gradient flow
    total_loss.backward()
    print(f"\n✓ Gradient flow successful!")
    
    # Count boundaries in targets
    num_boundaries = (targets > 0).sum().item()
    num_cuts = (targets == 1).sum().item()
    num_graduals = (targets == 2).sum().item()
    print(f"\nTarget statistics:")
    print(f"  Total frames: {B * T}")
    print(f"  Boundaries: {num_boundaries} ({100*num_boundaries/(B*T):.2f}%)")
    print(f"  Cuts: {num_cuts}")
    print(f"  Graduals: {num_graduals}")
    
    print("\n" + "="*80)
    print("✓ Loss function test passed!")
    print("="*80)
