"""Generate Protocol-A ClipShots Table 8 and Table 9 analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from load_ablation_model import load_detector
from test_clipshots import ClipShotsPredecodedDataset, collate_clipshots_batch, load_gt_scenes
from test_autoshot import apply_min_scene_length, predictions_to_scenes, reconstruct_full_video, run_inference


def transition_intervals(scenes):
    if len(scenes) < 2:
        return []
    return [(int(scenes[i, 1]), int(scenes[i + 1, 0])) for i in range(len(scenes) - 1)]


def match_interval(gt, pred, tolerance=2):
    return pred[1] >= gt[0] - tolerance and pred[0] <= gt[1] + tolerance


def best_match(gt, predictions, used, tolerance=2):
    candidates = [(i, p) for i, p in enumerate(predictions) if i not in used and match_interval(gt, p, tolerance)]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs((item[1][0] + item[1][1]) - (gt[0] + gt[1])))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--ann_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    dataset = ClipShotsPredecodedDataset(args.cache_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_clipshots_batch)
    gt = load_gt_scenes(args.ann_path, {e["video_name"] for e in dataset.mapping})
    rows = {}
    for run_id in ("A0", "A1", "A2", "C6"):
        checkpoint = Path(args.checkpoint_root) / run_id / "best_model.pt"
        if not checkpoint.exists():
            continue
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_detector(checkpoint, device)
        clips = run_inference(model, loader, device)
        video_y, video_z = reconstruct_full_video(clips)
        result_file = Path(args.checkpoint_root) / run_id / "clipshots_results.json"
        result = json.loads(result_file.read_text(encoding="utf-8"))
        bt = result["best_boundary_threshold"]
        gt_thr = result["best_gradual_threshold"]
        lmin = result.get("best_min_scene_len", 1)
        cut_hit = cut_total = grad_hit = grad_total = 0
        gradual_pred_count = gradual_tp = 0
        gradual_ious = []
        duration = {"le_10": [0, 0], "11_30": [0, 0], "gt_30": [0, 0]}
        for vid, y in video_y.items():
            if vid not in gt:
                continue
            z = video_z[vid]
            pred_many = ((z > gt_thr) | ((y > bt) & ~(z > gt_thr))).astype(np.uint8)
            pred_scenes = apply_min_scene_length(predictions_to_scenes(pred_many), lmin)
            preds = transition_intervals(pred_scenes)
            gradual_preds = transition_intervals(predictions_to_scenes((z > gt_thr).astype(np.uint8)))
            gradual_pred_count += len(gradual_preds)
            ann_key = vid
            # load_gt_scenes has already normalized key variants; infer type from scene interval
            gt_intervals = transition_intervals(gt[ann_key])
            used = set()
            for interval in gt_intervals:
                match = best_match(interval, preds, used)
                span = interval[1] - interval[0]
                is_cut = span <= 1
                if is_cut:
                    cut_total += 1
                else:
                    grad_total += 1
                if match is not None:
                    used.add(match)
                    if is_cut:
                        cut_hit += 1
                    else:
                        grad_hit += 1
                        pred = preds[match]
                        inter = max(0, min(interval[1], pred[1]) - max(interval[0], pred[0]) + 1)
                        union = max(interval[1], pred[1]) - min(interval[0], pred[0]) + 1
                        gradual_ious.append(inter / union if union else 0.0)
                        bucket = "le_10" if span <= 10 else "11_30" if span <= 30 else "gt_30"
                        duration[bucket][1] += 1
                        duration[bucket][0] += 1
                elif not is_cut:
                    bucket = "le_10" if span <= 10 else "11_30" if span <= 30 else "gt_30"
                    duration[bucket][1] += 1
            gradual_used = set()
            for interval in gt_intervals:
                if interval[1] - interval[0] <= 1:
                    continue
                match = best_match(interval, gradual_preds, gradual_used)
                if match is not None:
                    gradual_used.add(match)
                    gradual_tp += 1
        gradual_fp = max(0, gradual_pred_count - gradual_tp)
        gradual_precision = gradual_tp / (gradual_tp + gradual_fp) if gradual_tp + gradual_fp else 0.0
        gradual_recall = gradual_tp / grad_total if grad_total else 0.0
        gradual_f1 = (2 * gradual_precision * gradual_recall /
                      (gradual_precision + gradual_recall)
                      if gradual_precision + gradual_recall else 0.0)
        rows[run_id] = {
            "cut_recall": cut_hit / cut_total if cut_total else None,
            "gradual_recall": grad_hit / grad_total if grad_total else None,
            "gradual_iou": float(np.mean(gradual_ious)) if gradual_ious else None,
            "gradual_f1": None if run_id in {"A1", "A2"} else gradual_f1,
            "duration_recall": {key: (value[0] / value[1] if value[1] else None) for key, value in duration.items()},
            "duration_counts": {key: value[1] for key, value in duration.items()},
            "thresholds": {"tau_c": bt, "tau_g": gt_thr, "l_min": lmin},
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
