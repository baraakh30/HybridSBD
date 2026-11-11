"""Evaluate a trained checkpoint on the SHOT, ClipShots or BBC Planet Earth test set.

The model is run over the pre-decoded clips, overlapping windows are averaged into one probability track per video, and
the decision thresholds are searched on the grid of ``hybridsbd.scoring`` (Protocol A: the best setting on this test
set). Writes ``<output>.json`` (results), ``<output>.predictions.npz`` (the probability tracks, which allow offline
re-evaluation with ``analysis/reevaluate.py``) and ``<output>.log``.

Examples::

    python evaluation/evaluate.py shot --checkpoint outputs/ablations/A0/best_model.pt \\
        --cache_root data/cache/autoshot_test --gt_path data/AutoShot/gt_scenes_dict_baseline_v2.pickle
    python evaluation/evaluate.py clipshots --checkpoint ... --cache_root data/cache/clipshots/test \\
        --ann_path data/ClipShots/annotations/test.json --min_scene_lengths 1 2 3 4
    python evaluation/evaluate.py bbc --checkpoint ... --cache_root data/cache/bbc --bbc_root data/BBC
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from evaluation.data import PredecodedClipDataset, collate_clips, load_bbc_gt, load_clipshots_gt, load_shot_gt
from evaluation.engine import evaluate
from hybridsbd.checkpoint import is_single_output, load_detector
from hybridsbd.logging_utils import result_metadata, save_predictions, tee_to_file

DATASETS = {
    # name: (title, dataset id key, sort clips, default min scene lengths, clip to annotated range)
    "shot": ("SHOT (AutoShot test set)", "video_id", False, [1], False),
    "clipshots": ("ClipShots test set", "video_name", False, [1, 2, 3, 4], False),
    "bbc": ("BBC Planet Earth", "video_id", True, None, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="dataset", required=True)
    for name, (title, _, _, default_lmin, _) in DATASETS.items():
        p = sub.add_parser(name, help=f"evaluate on {title}")
        p.add_argument("--checkpoint", required=True, help="best_model.pt of the run")
        p.add_argument("--cache_root", required=True, help="pre-decoded clip cache of the test set")
        p.add_argument("--output", default=f"{name}_results.json")
        p.add_argument("--batch_size", type=int, default=8)
        p.add_argument("--tolerance", type=int, default=2, help="frame tolerance of the scene matching (AutoShot: 2)")
        if name == "shot":
            p.add_argument("--gt_path", default=None, help="gt_scenes_dict_baseline_v2.pickle (default: <cache_root>/../autoshot/...)")
        elif name == "clipshots":
            p.add_argument("--ann_path", required=True, help="ClipShots annotations/test.json")
        else:
            p.add_argument("--bbc_root", required=True, help="BBC root (contains annotations/shots/)")
        if default_lmin is not None:
            p.add_argument("--min_scene_lengths", type=int, nargs="+", default=default_lmin, help="minimum-scene-length grid")
    return parser.parse_args()


def load_ground_truth(args, dataset: PredecodedClipDataset):
    if args.dataset == "shot":
        gt_path = Path(args.gt_path) if args.gt_path else Path(args.cache_root).parent / "autoshot" / "gt_scenes_dict_baseline_v2.pickle"
        print(f"\nLoading ground truth from {gt_path}...")
        return load_shot_gt(gt_path)
    if args.dataset == "clipshots":
        print("\nLoading ground truth...")
        return load_clipshots_gt(args.ann_path, list({e["video_name"] for e in dataset.mapping}))
    print(f"\nLoading ground truth from {args.bbc_root}...")
    return load_bbc_gt(args.bbc_root, args.cache_root)


def print_summary(results: dict) -> None:
    print("\n" + "=" * 80 + "\nRESULTS\n" + "=" * 80)
    print(f"Videos evaluated : {results['num_videos']}")
    print(f"Best boundary thr: {results['best_boundary_threshold']:.2f}")
    gradual = results["best_gradual_threshold"]
    print(f"Best gradual thr : {'n/a (single output)' if gradual is None else format(gradual, '.2f')}")
    overall = results["overall"]
    print(f"\nOverall  P={overall['p']:.4f}  R={overall['r']:.4f}  F1={overall['f1']:.4f}")
    print(f"         TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']}")
    ranked = sorted(results["video_results"].items(), key=lambda x: x[1]["f1"], reverse=True)
    for title, subset in (("Top 5 videos", ranked[:5]), ("Bottom 5 videos", ranked[-5:])):
        print(f"\n{title}:")
        for vid, m in subset:
            print(f"  {vid}: F1={m['f1']:.3f}  P={m['p']:.3f}  R={m['r']:.3f}  gt_scenes={m['n_gt']}  pred_scenes={m['n_pred']}")
    print("=" * 80)


def main() -> None:
    args = parse_args()
    config = vars(args)
    title, id_key, sort_clips, _, clip_to_range = DATASETS[args.dataset]
    tee_to_file(Path(args.output).with_suffix(".log"))

    print("\n" + "=" * 80 + f"\nTESTING ON {title.upper()}\n" + "=" * 80)
    print(
        f"Checkpoint:  {args.checkpoint}\nCache root:  {args.cache_root}\nBatch size:  {args.batch_size}\nTolerance:   ±{args.tolerance} frames"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:      {device}")

    dataset = PredecodedClipDataset(args.cache_root, id_key=id_key, sort_clips=sort_clips, name=title)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(os.environ.get("EVAL_WORKERS", 4)),
        pin_memory=True,
        collate_fn=collate_clips,
    )

    print("\nLoading model...")
    model, ckpt = load_detector(args.checkpoint, device)
    config["single_output"] = is_single_output(ckpt)
    print(f"Loaded from epoch {ckpt.get('epoch', '?')}  single_output={config['single_output']}")

    gt_scenes = load_ground_truth(args, dataset)
    results = evaluate(
        model,
        loader,
        device,
        gt_scenes,
        tolerance=args.tolerance,
        single_output=config["single_output"],
        min_scene_lengths=getattr(args, "min_scene_lengths", None),
        clip_to_annotated_range=clip_to_range,
    )
    print_summary(results)

    predictions = results.pop("_predictions")
    results["metadata"] = result_metadata(args.checkpoint, config["single_output"], config)
    results["predictions_file"] = save_predictions(Path(args.output).with_suffix(".predictions.npz"), *predictions)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
