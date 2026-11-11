"""Scene-based scoring of shot-boundary predictions (numpy only, no PyTorch dependency).

This module is the single implementation of the evaluation protocol used by every test set, the training-time
validation and the offline re-evaluations:

1. Clip probabilities are averaged into one per-video track (:func:`reconstruct_full_video`).
2. For a threshold setting the track becomes a binary many-hot prediction (:func:`build_pred_many_hot`),
   then a list of scenes (:func:`predictions_to_scenes`), optionally merged with a minimum scene length
   (:func:`apply_min_scene_length`).
3. Predicted and ground-truth scenes are matched with a frame tolerance (:func:`evaluate_scenes`, adapted from the
   AutoShot / ``shot-detection-evaluation`` code) and precision, recall and F1 are pooled over all videos
   (:func:`sweep`, which searches the threshold grid and returns the best setting).

Conventions: a scene is a row ``[start, end]`` (inclusive frame indices); the boundary between two consecutive scenes
is the gap ``[end_i, start_{i+1}]``; a ground-truth transition counts as detected when the predicted gap overlaps it
after widening both by the tolerance.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Boundary-threshold grid (tau_c) used by every evaluation.
BOUNDARY_THRESHOLDS: Tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
#: Gradual-threshold grid (tau_g) for dual-head models; single-output models use ``(None,)``.
GRADUAL_THRESHOLDS: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)


def build_pred_many_hot(
    y_pred: np.ndarray,
    z_pred: Optional[np.ndarray],
    boundary_threshold: float,
    gradual_threshold: Optional[float],
) -> np.ndarray:
    """Combine the midpoint track ``y`` and the many-hot track ``z`` into one binary many-hot prediction.

    Dual-head rule: boundary if ``y > tau_c``; gradual if ``z > tau_g``; cut = boundary and not gradual;
    prediction = gradual OR cut. Single-output rule (``gradual_threshold=None``): ``y > tau_c``.
    """
    if gradual_threshold is None:
        return (y_pred > boundary_threshold).astype(np.uint8)
    pred_is_boundary = y_pred > boundary_threshold
    pred_is_gradual = z_pred > gradual_threshold
    pred_is_cut = pred_is_boundary & ~pred_is_gradual
    return (pred_is_gradual | pred_is_cut).astype(np.uint8)


def predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
    """Convert a binary many-hot track (1 = inside a transition) into ``N x 2`` scenes ``[start, end]``.

    Adapted from the AutoShot evaluation code. A track that is all ones yields a single scene covering the video.
    """
    scenes: List[List[int]] = []
    t_prev, start = 0, 0

    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i - 1])
        t_prev = t

    if t_prev == 0:
        scenes.append([start, len(predictions) - 1])

    if len(scenes) == 0:  # all predictions are 1
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def apply_min_scene_length(pred_scenes: np.ndarray, min_scene_len: int) -> np.ndarray:
    """Merge every predicted scene shorter than ``min_scene_len`` frames into the previous scene (``<= 1``: no-op)."""
    if min_scene_len <= 1 or len(pred_scenes) <= 1:
        return pred_scenes
    filtered = [pred_scenes[0].copy()]
    for scene in pred_scenes[1:]:
        if int(scene[1] - scene[0] + 1) < min_scene_len:
            filtered[-1] = np.array([filtered[-1][0], scene[1]], dtype=np.int32)
        else:
            filtered.append(scene.copy())
    return np.asarray(filtered, dtype=np.int32)


def evaluate_scenes(
    gt_scenes: np.ndarray, pred_scenes: np.ndarray, n_frames_miss_tolerance: int = 2
) -> Tuple[float, float, float, Tuple[int, int, int]]:
    """Match predicted to ground-truth transitions with a frame tolerance.

    Adapted from https://github.com/gyglim/shot-detection-evaluation.

    Args:
        gt_scenes: ``N x 2`` ground-truth scenes.
        pred_scenes: ``M x 2`` predicted scenes.
        n_frames_miss_tolerance: tolerance in frames (AutoShot default: 2).

    Returns:
        ``(precision, recall, f1, (tp, fp, fn))`` for this video.
    """
    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1) if len(gt_scenes) > 1 else np.zeros((0, 2))
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1) if len(pred_scenes) > 1 else np.zeros((0, 2))

    i = j = 0
    tp = fp = fn = 0
    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            i += 1
        elif i == len(gt_trans):
            fp += 1
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            i += 1
        else:
            i += 1
            j += 1
            tp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, (tp, fp, fn)


def transitions_to_scenes(transitions: Iterable[Sequence[int]], total_frames: int) -> np.ndarray:
    """Convert ``[t_start, t_end]`` transition pairs into ``N x 2`` scenes.

    Each transition separates two shots: one scene ends at ``t_start`` and the next starts at ``t_end``, so
    ``scenes = [[0, t0_start], [t0_end, t1_start], ..., [tN_end, total_frames - 1]]``.
    """
    transitions = list(transitions)
    if not transitions:
        return np.array([[0, max(0, int(total_frames) - 1)]], dtype=np.int32)

    scenes = []
    prev_end = 0
    for t_start, t_end in sorted(transitions, key=lambda x: x[0]):
        scenes.append([prev_end, int(t_start)])
        prev_end = int(t_end)
    scenes.append([prev_end, max(prev_end, int(total_frames) - 1)])
    return np.array(scenes, dtype=np.int32)


def reconstruct_full_video(
    video_clips: Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Average overlapping clip predictions into per-video ``y`` and ``z`` probability tracks.

    Args:
        video_clips: ``{video_id: [(start_frame, y_probs[T], z_probs[T]), ...]}``.

    Returns:
        ``(video_y, video_z)``: per-video float32 arrays; every frame is the mean over the clips that cover it.
    """
    video_y: Dict[str, np.ndarray] = {}
    video_z: Dict[str, np.ndarray] = {}
    for video_id, clips in video_clips.items():
        clips = sorted(clips, key=lambda x: x[0])
        total = max(s + len(y) for s, y, _ in clips)
        full_y = np.zeros(total, dtype=np.float32)
        full_z = np.zeros(total, dtype=np.float32)
        counts = np.zeros(total, dtype=np.int32)
        for start, y, z in clips:
            end = start + len(y)
            full_y[start:end] += y
            full_z[start:end] += z
            counts[start:end] += 1
        mask = counts > 0
        full_y[mask] /= counts[mask]
        full_z[mask] /= counts[mask]
        video_y[video_id] = full_y
        video_z[video_id] = full_z
    return video_y, video_z


def pooled_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, recall and F1 from pooled counts."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _video_scenes(
    vid: str,
    video_y: Dict[str, np.ndarray],
    video_z: Optional[Dict[str, np.ndarray]],
    gt_sc: np.ndarray,
    bt: float,
    gt: Optional[float],
    lmin: Optional[int],
    clip_to_annotated_range: bool,
) -> np.ndarray:
    y = video_y[vid]
    z = video_z[vid] if video_z is not None else np.zeros_like(y)
    if clip_to_annotated_range:
        # drop predictions after the last annotated frame so that the unannotated tail does not count as false positives
        last = int(gt_sc[-1, 1]) + 1
        y, z = y[:last], z[:last]
    scenes = predictions_to_scenes(build_pred_many_hot(y, z, bt, gt))
    return apply_min_scene_length(scenes, lmin) if lmin is not None else scenes


def sweep(
    video_y: Dict[str, np.ndarray],
    video_z: Optional[Dict[str, np.ndarray]],
    gt_scenes: Dict[str, np.ndarray],
    *,
    tolerance: int = 2,
    single_output: bool = False,
    min_scene_lengths: Optional[Sequence[int]] = None,
    clip_to_annotated_range: bool = False,
    boundary_thresholds: Sequence[float] = BOUNDARY_THRESHOLDS,
    gradual_thresholds: Optional[Sequence[Optional[float]]] = None,
    progress: bool = False,
) -> dict:
    """Search the threshold grid and return the best setting plus every setting's pooled scores.

    The grid is iterated ``boundary_thresholds -> gradual_thresholds -> min_scene_lengths`` and the best setting is the
    first one with a strictly greater F1 (ties keep the earlier setting), which makes results reproducible.

    Args:
        video_y, video_z: per-video probability tracks (``video_z`` may be ``None`` for single-output models).
        gt_scenes: ``{video_id: N x 2 scenes}``; videos without ground truth are skipped.
        tolerance: frame tolerance of the scene matching.
        single_output: single-output model (A1, A2, A3): only the boundary threshold is searched.
        min_scene_lengths: minimum-scene-length grid, or ``None`` for no merging (result keys then have no l_min).
        clip_to_annotated_range: BBC rule, see :func:`_video_scenes`.
        gradual_thresholds: override the tau_g grid (default: ``GRADUAL_THRESHOLDS``, or ``(None,)`` if single-output).
        progress: show a progress bar over the boundary thresholds.

    Returns:
        A dict with ``single_output``, ``best_boundary_threshold``, ``best_gradual_threshold``, ``best_min_scene_len``
        (only if ``min_scene_lengths`` is given), ``overall`` (best setting's ``p, r, f1, tp, fp, fn``),
        ``all_results`` (``{(tau_c, tau_g[, l_min]): scores}``), ``video_results`` (per-video scores at the best
        setting) and ``num_videos``.
    """
    if gradual_thresholds is None:
        gradual_thresholds = (None,) if single_output else GRADUAL_THRESHOLDS
    lmins: Sequence[Optional[int]] = [None] if min_scene_lengths is None else list(min_scene_lengths)

    iterator: Iterable[float] = boundary_thresholds
    if progress:
        from tqdm import tqdm

        iterator = tqdm(boundary_thresholds, desc="Sweep")

    best_f1, best_bt, best_gt, best_lmin = 0.0, 0.5, gradual_thresholds[0], (1 if min_scene_lengths is not None else None)
    all_results: Dict[tuple, dict] = {}
    for bt in iterator:
        for gt in gradual_thresholds:
            for lmin in lmins:
                tp = fp = fn = 0
                for vid in video_y:
                    if vid not in gt_scenes:
                        continue
                    scenes = _video_scenes(vid, video_y, video_z, gt_scenes[vid], bt, gt, lmin, clip_to_annotated_range)
                    _, _, _, (t, f, n) = evaluate_scenes(gt_scenes[vid], scenes, n_frames_miss_tolerance=tolerance)
                    tp += t
                    fp += f
                    fn += n
                p, r, f1 = pooled_prf(tp, fp, fn)
                key = (bt, gt) if lmin is None else (bt, gt, lmin)
                all_results[key] = dict(p=p, r=r, f1=f1, tp=tp, fp=fp, fn=fn)
                if f1 > best_f1:
                    best_f1, best_bt, best_gt, best_lmin = f1, bt, gt, lmin

    video_results = {}
    for vid in video_y:
        if vid not in gt_scenes:
            continue
        gt_sc = gt_scenes[vid]
        scenes = _video_scenes(vid, video_y, video_z, gt_sc, best_bt, best_gt, best_lmin, clip_to_annotated_range)
        p, r, f1, (tp, fp, fn) = evaluate_scenes(gt_sc, scenes, n_frames_miss_tolerance=tolerance)
        video_results[vid] = dict(p=p, r=r, f1=f1, tp=tp, fp=fp, fn=fn, n_gt=len(gt_sc), n_pred=len(scenes))

    best_key = (best_bt, best_gt) if best_lmin is None else (best_bt, best_gt, best_lmin)
    result = {
        "single_output": single_output,
        "best_boundary_threshold": best_bt,
        "best_gradual_threshold": best_gt,
    }
    if min_scene_lengths is not None:
        result["best_min_scene_len"] = best_lmin
    result.update(
        overall=all_results[best_key],
        all_results=all_results,
        video_results=video_results,
        num_videos=len(video_results),
    )
    return result


def best_setting(all_results: Dict[tuple, dict], min_scene_len: Optional[int] = None) -> Optional[Tuple[tuple, dict]]:
    """Best ``(key, scores)`` of a ``sweep`` grid (first strictly greater F1), optionally for one fixed ``l_min``."""
    best = None
    for key, scores in all_results.items():
        if min_scene_len is not None and key[2] != min_scene_len:
            continue
        if best is None or scores["f1"] > best[1]["f1"]:
            best = (key, scores)
    return best
