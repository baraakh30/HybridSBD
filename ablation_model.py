"""Flag-controlled HybridSBD model used by the Protocol-A ablations."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from detection.hybrid_boundary_detector import (
    ColorHistogramSimilarity,
    DualBoundaryHead,
    EfficientSpatialEncoder,
    FrameSimilarity,
    PixelMotionExtractor,
    ProgressiveFusionBlock,
)
from ablation_config import AblationConfig


class UnifiedBoundaryHead(nn.Module):
    """Single A2 output head for u = one_hot OR many_hot."""

    def __init__(self, dim: int, positive_rate: float = 0.008):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(dim, dim // 2), nn.LayerNorm(dim // 2), nn.GELU(), nn.Dropout(0.2),
        )
        self.head = nn.Linear(dim // 2, 1)
        p = min(max(float(positive_rate), 1e-6), 1.0 - 1e-6)
        nn.init.constant_(self.head.bias, math.log(p / (1.0 - p)))

    def forward(self, x):
        return self.head(self.refine(x))


class AblationHybridBoundaryDetector(nn.Module):
    """Hybrid detector with every Protocol-A component behind a real flag."""

    def __init__(self, cfg: AblationConfig, unified_positive_rate: float = 0.008):
        super().__init__()
        self.ablation = cfg
        d = cfg.main_dim
        emb_dim = 640 if cfg.use_edge_head else 512

        self.pixel_motion_extractor = PixelMotionExtractor() if cfg.use_motion else None
        self.rgb_encoder = EfficientSpatialEncoder(input_channels=3, output_dim=512,
                                                   use_edge_head=cfg.use_edge_head)
        self.diff_encoder = (
            EfficientSpatialEncoder(input_channels=15, output_dim=512,
                                    use_edge_head=cfg.use_edge_head)
            if cfg.use_branch_b else None
        )

        self.frame_sim_short = FrameSimilarity(emb_dim, 15, 64) if cfg.use_fsim else None
        self.frame_sim_medium = FrameSimilarity(emb_dim, 51, 64) if cfg.use_fsim else None
        self.frame_sim_long = FrameSimilarity(emb_dim, 127, 64) if cfg.use_fsim else None
        self.color_hist_similarity = ColorHistogramSimilarity(64, 127, 128) if cfg.use_csim else None
        self.feat_dropout = nn.Dropout(0.15)

        stream_dims = []
        stream_dims.append(d // 2)  # RGB
        if cfg.use_branch_b:
            stream_dims.append(d // 2)
        if cfg.use_fsim:
            stream_dims.append(d // 4)
        if cfg.use_csim:
            stream_dims.append(d // 4)
        if cfg.use_delta:
            stream_dims.append(64)
        if cfg.use_motion:
            stream_dims.append(15)
        fuse_in = sum(stream_dims)

        self.rgb_proj = nn.Sequential(nn.Linear(emb_dim, d // 2), nn.LayerNorm(d // 2), nn.GELU())
        self.diff_proj = (
            nn.Sequential(nn.Linear(emb_dim, d // 2), nn.LayerNorm(d // 2), nn.GELU())
            if cfg.use_branch_b else None
        )
        self.feat_delta_proj = (
            nn.Sequential(nn.Linear(emb_dim, 64), nn.LayerNorm(64), nn.GELU())
            if cfg.use_delta else None
        )
        self.frame_sim_proj = (
            nn.Sequential(nn.Linear(192, d // 4), nn.LayerNorm(d // 4), nn.GELU())
            if cfg.use_fsim else None
        )
        self.color_sim_proj = (
            nn.Sequential(nn.Linear(128, d // 4), nn.LayerNorm(d // 4), nn.GELU())
            if cfg.use_csim else None
        )
        self.motion_norm = nn.LayerNorm(15) if cfg.use_motion else None
        self.feature_proj = nn.Sequential(
            nn.Linear(fuse_in, d * 2), nn.LayerNorm(d * 2), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.15),
        )
        self.fusion_blocks = nn.ModuleList([
            ProgressiveFusionBlock(d, use_attention=cfg.use_attention)
            for _ in range(cfg.num_fusion_blocks)
        ])
        self.detection_head = (
            UnifiedBoundaryHead(d, unified_positive_rate)
            if cfg.supervision == "unified" else DualBoundaryHead(d)
        )
        self.main_dim = d
        self.emb_dim = emb_dim
        self.fusion_in = fuse_in

    def forward(self, x, return_features=False):
        b, t, c, h, w = x.shape
        streams = []
        motion = None
        if self.ablation.use_motion:
            motion = self.pixel_motion_extractor(x)
            streams.append(self.motion_norm(motion))

        rgb_features = self.rgb_encoder(x)
        streams.insert(0, self.rgb_proj(rgb_features))

        diff_features = None
        if self.ablation.use_branch_b:
            distances = [1, 4, 8, 16, 32]
            diffs = torch.zeros(b, t, c * len(distances), h, w, device=x.device, dtype=x.dtype)
            for i, dist in enumerate(distances):
                if dist < t:
                    diffs[:, dist:, i*c:(i+1)*c] = x[:, dist:] - x[:, :-dist]
            diff_features = self.diff_encoder(diffs)
            streams.insert(1, self.diff_proj(diff_features))

        relation_features = (
            self.rgb_encoder.forward_per_frame(x)
            if self.ablation.use_branch_c and self.ablation.rel_source == "pf"
            else rgb_features
        )
        relation_features = self.feat_dropout(relation_features)

        if self.ablation.use_delta:
            delta = torch.zeros_like(relation_features)
            delta[:, 1:] = relation_features[:, 1:] - relation_features[:, :-1]
            scale = delta.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            streams.append(self.feat_delta_proj(delta / scale))
        if self.ablation.use_fsim:
            fsim = torch.cat([
                self.frame_sim_short(relation_features),
                self.frame_sim_medium(relation_features),
                self.frame_sim_long(relation_features),
            ], dim=-1)
            streams.append(self.frame_sim_proj(fsim))
        if self.ablation.use_csim:
            streams.append(self.color_sim_proj(self.color_hist_similarity(x)))

        fused = self.feature_proj(torch.cat(streams, dim=-1))
        for block in self.fusion_blocks:
            fused = block(fused)

        if self.ablation.supervision == "unified":
            y_logits = self.detection_head(fused)
            z_logits = torch.zeros_like(y_logits)
        else:
            y_logits, z_logits = self.detection_head(fused)

        if return_features:
            return y_logits, z_logits, {"rgb": rgb_features, "diff": diff_features,
                                       "pixel_motion": motion, "fused": fused}
        return y_logits, z_logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_ablation_detector(cfg: AblationConfig, unified_positive_rate: float = 0.008):
    return AblationHybridBoundaryDetector(cfg, unified_positive_rate)
