"""Protocol-A ClipShots analyses: Table 8 (per type) and Table 9 (duration groups).

Spec Section 6. Uses each model's own ClipShots thresholds (tau_c, tau_g, l_min)
from ``clipshots_results.json``. If ``clipshots_results.predictions.npz`` exists
(written by eval_clipshots.py) the analysis is computed offline from it; otherwise
the model is run once.

Definitions (all recorded in the output JSON):
* GT type: the annotation's type label when present, otherwise length <= 1 frame
  (end - start <= 1) = cut, > 1 = gradual. The source used is counted per run.
* Predicted transition = contiguous run of frames the inference rule marks as
  boundary (after the l_min merge): A0/dual: (p_z > tau_g) OR (p_y > tau_c);
  A1/A2: p > tau_c.  Matching is type-agnostic with +-2 frame tolerance, one-to-one.
* Cut / gradual recall: fraction of GT cuts / gradual transitions matched.
* Gradual F1 (dual models only): predicted gradual = predicted runs containing
  >= 1 frame with p_z > tau_g, matched against GT gradual only.
* Gradual IoU: inclusive-frame temporal IoU between the matched GT gradual
  [start, end] and the predicted run of frames above threshold (not scene intervals).
  Reported for A0 and A2 (spec Table 8); A1 is N/A.
* Duration groups: GT gradual length (end - start) <= 10, 11-30, > 30, with n per group.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: lets the script import the repository packages

import argparse
import json
from pathlib import Path

import numpy as np

from hybridsbd.logging_utils import load_predictions, tee_to_file
from hybridsbd.scoring import apply_min_scene_length, build_pred_many_hot, predictions_to_scenes, reconstruct_full_video

TABLE8_RUNS = ("A0", "A1", "A2")
TABLE9_RUNS = ("A0", "C6")
TOLERANCE = 2


def runs_from_scenes(scenes):
    """Frames strictly between consecutive scenes = contiguous predicted-transition runs."""
    return [(int(scenes[i, 1]) + 1, int(scenes[i + 1, 0]) - 1) for i in range(len(scenes) - 1)]


def gt_transitions(entry):
    """Return sorted list of (start, end, is_cut, source) for one annotation entry."""
    transitions = [tuple(int(v) for v in t) for t in entry["transitions"]]
    labels = None
    source = "length_rule"
    for key in ("types", "transition_types", "labels", "type", "transition_labels"):
        value = entry.get(key)
        if isinstance(value, list) and len(value) == len(transitions):
            labels, source = value, f"annotation:{key}"
            break
    items = []
    for idx, (s, e) in enumerate(transitions):
        if labels is not None:
            lab = labels[idx]
            text = str(lab).lower()
            if text in {"cut", "0", "hard", "hard_cut"}:
                is_cut = True
            elif text in {"gradual", "1", "dissolve", "fade", "wipe"} or "grad" in text:
                is_cut = False
            else:
                is_cut, source = (e - s) <= 1, "length_rule"
        else:
            is_cut = (e - s) <= 1
        items.append((s, e, is_cut, source))
    items.sort(key=lambda x: x[0])
    return items


def overlaps(gt, pred, tol=TOLERANCE):
    return pred[1] >= gt[0] - tol and pred[0] <= gt[1] + tol


def best_match(gt, preds, used):
    cand = [(i, p) for i, p in enumerate(preds) if i not in used and overlaps(gt, p)]
    if not cand:
        return None
    return min(cand, key=lambda it: abs((it[1][0] + it[1][1]) - (gt[0] + gt[1])))[0]


def iou(gt, pred):
    inter = max(0, min(gt[1], pred[1]) - max(gt[0], pred[0]) + 1)
    union = max(gt[1], pred[1]) - min(gt[0], pred[0]) + 1
    return inter / union if union else 0.0


def bucket(length):
    return "le_10" if length <= 10 else "11_30" if length <= 30 else "gt_30"


def analyse_run(run_id, video_y, video_z, annotations, result):
    single = bool(result.get("single_output", result.get("metadata", {}).get("single_output_rule", False)))
    bt = result["best_boundary_threshold"]
    gtau = result["best_gradual_threshold"]
    lmin = result.get("best_min_scene_len", 1)

    cut_hit = cut_total = grad_hit = grad_total = 0
    pred_grad_total = pred_grad_tp = 0
    ious = []
    dur_hit = {"le_10": 0, "11_30": 0, "gt_30": 0}
    dur_n = {"le_10": 0, "11_30": 0, "gt_30": 0}
    sources = {}
    n_videos = 0
    for vid, y in video_y.items():
        entry = annotations.get(vid)
        if entry is None:
            continue
        n_videos += 1
        z = video_z[vid] if video_z is not None else None
        pred_many = build_pred_many_hot(y, z if z is not None else np.zeros_like(y), bt, None if single else gtau)
        scenes = apply_min_scene_length(predictions_to_scenes(pred_many), lmin)
        preds = runs_from_scenes(scenes)

        gts = gt_transitions(entry)
        used = set()
        matched_pred_for_gt = {}
        for idx, (s, e, is_cut, src) in enumerate(gts):
            sources[src] = sources.get(src, 0) + 1
            m = best_match((s, e), preds, used)
            if is_cut:
                cut_total += 1
            else:
                grad_total += 1
                dur_n[bucket(e - s)] += 1
            if m is not None:
                used.add(m)
                matched_pred_for_gt[idx] = m
                if is_cut:
                    cut_hit += 1
                else:
                    grad_hit += 1
                    dur_hit[bucket(e - s)] += 1
                    ious.append(iou((s, e), preds[m]))

        if not single and z is not None:
            gradual_preds = [p for p in preds if p[1] >= p[0] and float(z[p[0] : p[1] + 1].max()) > gtau]
            pred_grad_total += len(gradual_preds)
            used_g = set()
            for s, e, is_cut, _ in gts:
                if is_cut:
                    continue
                m = best_match((s, e), gradual_preds, used_g)
                if m is not None:
                    used_g.add(m)
                    pred_grad_tp += 1

    def ratio(a, b):
        return a / b if b else None

    g_prec = ratio(pred_grad_tp, pred_grad_total)
    g_rec = ratio(pred_grad_tp, grad_total)
    g_f1 = (
        2 * g_prec * g_rec / (g_prec + g_rec)
        if g_prec is not None and g_rec is not None and (g_prec + g_rec) > 0
        else (0.0 if g_prec is not None and g_rec is not None else None)
    )
    return {
        "videos": n_videos,
        "thresholds": {"tau_c": bt, "tau_g": gtau, "l_min": lmin, "single_output_rule": single},
        "gt_type_sources": sources,
        "counts": {
            "gt_cuts": cut_total,
            "gt_graduals": grad_total,
            "cut_matched": cut_hit,
            "gradual_matched": grad_hit,
            "pred_gradual_runs": pred_grad_total,
            "pred_gradual_tp": pred_grad_tp,
        },
        "cut_recall": ratio(cut_hit, cut_total),
        "gradual_recall": ratio(grad_hit, grad_total),
        # Table 8 N/A rules: gradual F1 only for the dual-head model; IoU not for A1.
        "gradual_f1": None if single else g_f1,
        "gradual_precision": None if single else g_prec,
        "gradual_iou": (None if run_id == "A1" else (float(np.mean(ious)) if ious else None)),
        "gradual_iou_all_matched": float(np.mean(ious)) if ious else None,
        "duration_recall": {k: ratio(dur_hit[k], dur_n[k]) for k in dur_n},
        "duration_counts": dict(dur_n),
    }


def load_annotations(ann_path, names):
    with open(ann_path, "r", encoding="utf-8") as handle:
        ann = json.load(handle)
    lookup = {}
    for key, value in ann.items():
        lookup[key] = value
        lookup[key.replace(".mp4", "")] = value
        lookup[key + ".mp4"] = value
    return {n: lookup[n] for n in names if n in lookup}


def get_predictions(run_dir, args):
    npz = run_dir / "clipshots_results.predictions.npz"
    if npz.exists():
        print(f"  using saved predictions {npz}")
        return load_predictions(npz)
    print("  saved predictions missing -> running inference")
    import torch
    from torch.utils.data import DataLoader

    from evaluation.data import PredecodedClipDataset, collate_clips
    from evaluation.engine import run_inference
    from hybridsbd.checkpoint import load_detector

    dataset = PredecodedClipDataset(args.cache_root, id_key="video_name", name="ClipShots test set")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_clips)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_detector(run_dir / "best_model.pt", device)
    video_y, video_z = reconstruct_full_video(run_inference(model, loader, device))
    return video_y, video_z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--ann_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    tee_to_file(Path(args.output).with_suffix(".log"))

    rows = {}
    for run_id in sorted(set(TABLE8_RUNS) | set(TABLE9_RUNS)):
        run_dir = Path(args.checkpoint_root) / run_id
        result_file = run_dir / "clipshots_results.json"
        if not result_file.exists():
            print(f"SKIP {run_id}: {result_file} missing")
            rows[run_id] = {"status": "missing clipshots_results.json"}
            continue
        print(f"Analysing {run_id}")
        result = json.loads(result_file.read_text(encoding="utf-8"))
        video_y, video_z = get_predictions(run_dir, args)
        annotations = load_annotations(args.ann_path, video_y.keys())
        rows[run_id] = analyse_run(run_id, video_y, video_z, annotations, result)

    out = {"definitions": __doc__, "tolerance_frames": TOLERANCE, "runs": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
