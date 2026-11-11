"""Protocol-A losses for dual, midpoint-only, and unified supervision."""

from __future__ import annotations

import torch
import torch.nn as nn

from hybridsbd.losses.binary_boundary_loss import create_one_hot_many_hot


class AblationBoundaryLoss(nn.Module):
    def __init__(
        self, supervision="dual", lambda1=5.0, lambda2=0.3, focal_alpha=0.8, focal_alpha_z=0.5, focal_gamma=2.0, z_label_smoothing=0.1
    ):
        super().__init__()
        if supervision not in {"dual", "y_only", "unified"}:
            raise ValueError(supervision)
        self.supervision = supervision
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = focal_alpha
        self.alpha_z = focal_alpha_z
        self.gamma = focal_gamma
        self.z_label_smoothing = z_label_smoothing

    def focal_bce(self, logits, targets, alpha):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        return ((1.0 - p_t).pow(self.gamma) * alpha_t * bce).mean()

    def forward(self, y_logits, z_logits, labels, transition_lengths):
        one_hot, many_hot = create_one_hot_many_hot(labels)
        y = y_logits.squeeze(-1)
        if self.supervision == "unified":
            unified = torch.maximum(one_hot, many_hot)
            loss_u = self.focal_bce(y, unified, self.alpha)
            return loss_u, {"y_loss": loss_u.detach().item(), "z_loss": 0.0, "u_loss": loss_u.detach().item()}
        y_loss = self.focal_bce(y, one_hot, self.alpha)
        if self.supervision == "y_only":
            total = self.lambda1 * y_loss
            z_loss = y_loss.detach() * 0.0
        else:
            z = z_logits.squeeze(-1)
            smooth = many_hot * (1.0 - self.z_label_smoothing)
            z_loss = self.focal_bce(z, smooth, self.alpha_z)
            total = self.lambda1 * y_loss + self.lambda2 * z_loss
        return total, {"y_loss": y_loss.detach().item(), "z_loss": z_loss.detach().item()}
