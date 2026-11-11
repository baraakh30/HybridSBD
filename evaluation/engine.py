"""Inference and threshold search for one test set (shared by SHOT, ClipShots and BBC)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from hybridsbd.scoring import reconstruct_full_video, sweep


@torch.no_grad()
def run_inference(model, loader, device):
    """Forward every clip once; return ``{video_id: [(start_frame, y_probs[T], z_probs[T]), ...]}``.

    Single-head models (A2) have no ``z`` output; zeros are returned for it (never read by the single-output rule).
    """
    model.eval()
    video_clips = defaultdict(list)

    for batch in tqdm(loader, desc="Inference"):
        frames = batch["frames"].to(device, non_blocking=True)  # [B, T, 3, H, W]
        y_logits, z_logits = model(frames)  # [B, T, 1] each
        y_probs = torch.sigmoid(y_logits.squeeze(-1)).cpu().numpy()  # [B, T]
        if z_logits is None:
            z_probs = np.zeros_like(y_probs)
        else:
            z_probs = torch.sigmoid(z_logits.squeeze(-1)).cpu().numpy()  # [B, T]

        for b in range(frames.shape[0]):
            video_clips[batch["video_id"][b]].append((batch["start_frame"][b].item(), y_probs[b], z_probs[b]))

    return video_clips


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    gt_scenes: Dict[str, np.ndarray],
    *,
    tolerance: int = 2,
    single_output: bool = False,
    min_scene_lengths: Optional[Sequence[int]] = None,
    clip_to_annotated_range: bool = False,
) -> dict:
    """Full-video evaluation: inference, per-video reconstruction and the threshold search of ``hybridsbd.scoring``.

    Returns the result dictionary written to ``*_results.json``; the key ``_predictions`` holds the per-video
    probability tracks ``(video_y, video_z)`` (``video_z`` is ``None`` for single-output models) and is stored
    separately by the caller.
    """
    video_clips = run_inference(model, loader, device)
    print("\nReconstructing full-video predictions...")
    video_y, video_z = reconstruct_full_video(video_clips)

    n_settings = "x".join(str(n) for n in (17, 1 if single_output else 4, len(min_scene_lengths or [1])))
    print(f"\nSweeping {n_settings} threshold settings...")
    result = sweep(
        video_y,
        video_z,
        gt_scenes,
        tolerance=tolerance,
        single_output=single_output,
        min_scene_lengths=min_scene_lengths,
        clip_to_annotated_range=clip_to_annotated_range,
        progress=True,
    )
    result["all_results"] = {str(k): v for k, v in result["all_results"].items()}
    return {"_predictions": (video_y, None if single_output else video_z), **result}
