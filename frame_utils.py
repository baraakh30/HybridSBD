"""Memory-bounded decoding for the predecoders."""

import torch


def decode_resized(decoder, frame_indices, size, chunk=16):
    """Decode ``frame_indices`` and bilinear-resize to ``size`` (H, W) in chunks of ``chunk`` frames.

    Returns uint8 [N, 3, H, W], identical to decoding all frames at once and resizing in one call (every frame is
    resized independently), but peak memory is ~chunk/N of the one-shot version: 128 frames of 1080p video need ~4 GB
    that way, which made decode workers hit the container memory limit.
    """
    idx = frame_indices if isinstance(frame_indices, torch.Tensor) else torch.as_tensor(list(frame_indices), dtype=torch.long)
    parts = []
    for i in range(0, len(idx), chunk):
        part = decoder.get_frames_at(idx[i:i + chunk]).data
        if tuple(part.shape[2:]) != tuple(size):
            part = torch.nn.functional.interpolate(part.float(), size=tuple(size), mode='bilinear',
                                                   align_corners=False).to(torch.uint8)
        parts.append(part)
    return torch.cat(parts, dim=0)
