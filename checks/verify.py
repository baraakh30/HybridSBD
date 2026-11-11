"""Verification checks for the HybridSBD pipeline, grouped as sub-commands.

``preflight``         build all 13 variants and run a forward/backward pass on dummy input (parameter counts, heads).
``a0-architecture``   Check 1 of the A0 reproduction on a TRAINED checkpoint (fusion blocks, no positional encoding,
                      delta projection, width, exact parameter count).
``a0-reproduction``   Check 2 of the A0 reproduction: Protocol-A F1 of a finished A0 run against the paper's ranges.
                      Exit code 0 = within range, 3 = outside (report to the team), 2 = results missing.
``shot-labels``       audit of the SHOT label file (which videos are labelled, irregularities, test-video labels).
``shot-split``        checksum of the SHOT validation videos for training seeds 42/43/44: old rule vs fixed split.
``fresh-pass``        compare saved probability tracks with a fresh forward pass of the same checkpoints.

Examples::

    python checks/verify.py preflight --device cpu --output outputs/preflight.json
    python checks/verify.py a0-reproduction --run_dir outputs/ablations/A0
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

#: Paper values and the acceptable range of the A0 reproduction: ``name: (label, paper, low, high)``.
A0_PAPER_RANGES = {
    "shot": ("SHOT", 0.924, 0.915, 0.930),
    "clipshots": ("ClipShots", 0.793, 0.785, 0.800),
    "bbc": ("BBC", 0.971, 0.965, 0.975),
}


# --------------------------------------------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------------------------------------------


def preflight(args) -> None:
    import torch

    from hybridsbd.configs.ablation_config import VARIANTS, environment_snapshot, write_json
    from hybridsbd.losses.ablation_loss import AblationBoundaryLoss
    from hybridsbd.models.ablation_model import UnifiedBoundaryHead, create_ablation_detector

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    x = torch.randn(1, args.seq_len, 3, 48, 64, device=device)
    results = {"environment": environment_snapshot(), "device": str(device), "variants": {}}

    for run_id, cfg in VARIANTS.items():
        model = create_ablation_detector(cfg).to(device).eval()
        with torch.no_grad():
            y, z = model(x)
        total, trainable = model.count_parameters()
        row = {
            "output_shape_y": list(y.shape),
            "output_shape_z": None if z is None else list(z.shape),
            "fusion_in": model.fusion_in,
            "embedding_dim": model.emb_dim,
            "parameters_total": total,
            "parameters_trainable": trainable,
            "head_type": type(model.detection_head).__name__,
        }
        if run_id == "A0":
            row["expected_fusion_in"] = 655
            row["delta_weight_shape"] = list(model.feat_delta_proj[0].weight.shape)
            row["four_fusion_blocks"] = len(model.fusion_blocks) == 4
            row["a0_parameter_count_rounds_to_millions"] = round(total / 1e6, 1)
            if (
                model.fusion_in != 655
                or len(model.fusion_blocks) != 4
                or tuple(model.feat_delta_proj[0].weight.shape) != (64, 640)
                or round(total / 1e6, 1) != 10.9
            ):
                raise AssertionError(f"A0 architecture mismatch: {row}")
        if run_id == "A2" and not isinstance(model.detection_head, UnifiedBoundaryHead):
            raise AssertionError("A2 does not use a single unified head")
        results["variants"][run_id] = row

        if run_id == "A1":
            labels = torch.zeros((1, args.seq_len), dtype=torch.long, device=device)
            labels[:, args.seq_len // 2] = 1
            model.train()
            yy, zz = model(x)
            loss, _ = AblationBoundaryLoss("y_only")(yy, zz, labels, labels)
            loss.backward()
            z_grads = [p.grad for n, p in model.named_parameters() if "head_z" in n]
            if any(g is not None and torch.any(g != 0) for g in z_grads):
                raise AssertionError("A1 z-head received a gradient")
            row["a1_z_head_no_gradient"] = True

    write_json(args.output, results)
    print(json.dumps(results, indent=2))


# --------------------------------------------------------------------------------------------------------------
# A0 reproduction
# --------------------------------------------------------------------------------------------------------------


def a0_architecture(args) -> None:
    import torch

    from hybridsbd.configs.ablation_config import AblationConfig
    from hybridsbd.models.ablation_model import create_ablation_detector

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    model = create_ablation_detector(AblationConfig(**ckpt["config"]))
    model.load_state_dict(state)  # also proves the checkpoint matches the constructed model
    params = sum(p.numel() for p in model.parameters())
    report = {
        "fusion_blocks": len({k.split(".")[1] for k in state if k.startswith("fusion_blocks.")}),
        "positional_encoding_keys": [k for k in state if "pos" in k.lower()],
        "delta_projection_weight_shape": list(state["feat_delta_proj.0.weight"].shape),
        "main_dim": ckpt["config"]["main_dim"],
        "fusion_in": model.fusion_in,
        "embedding_dim": model.emb_dim,
        "parameters_total_exact": params,
        "parameters_round_to_M": round(params / 1e6, 1),
    }
    report["checks"] = {
        "four_fusion_blocks": report["fusion_blocks"] == 4,
        "no_positional_encoding": not report["positional_encoding_keys"],
        "delta_weight_is_64x640": report["delta_projection_weight_shape"] == [64, 640],
        "D_is_384": report["main_dim"] == 384,
        "params_round_to_10.9M": report["parameters_round_to_M"] == 10.9,
    }
    report["all_pass"] = all(report["checks"].values())
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def a0_reproduction(args) -> None:
    run_dir = Path(args.run_dir)
    report, status = {}, 0
    for key, (label, paper, lo, hi) in A0_PAPER_RANGES.items():
        path = run_dir / f"{key}_results.json"
        if not path.exists():
            report[label] = {"status": "MISSING", "file": str(path)}
            status = max(status, 2)
            continue
        f1 = json.loads(path.read_text(encoding="utf-8"))["overall"]["f1"]
        ok = lo <= f1 <= hi
        report[label] = {"f1": f1, "paper": paper, "acceptable": [lo, hi], "status": "PASS" if ok else "OUT_OF_RANGE"}
        if not ok and status == 0:
            status = 3
    print(json.dumps(report, indent=2))
    verdict = {
        0: "A0 reproduces the paper within the acceptable ranges.",
        2: "Check 2 incomplete: result files missing.",
        3: "Check 2 FAILED: report to the team; ablations may proceed on the same code but the main-paper numbers need discussion (spec Section 1).",
    }[status]
    print(verdict)
    out = Path(args.output) if args.output else run_dir / "a0_reproduction_check.json"
    out.write_text(json.dumps({"checks": report, "verdict": verdict, "exit_code": status}, indent=2), encoding="utf-8")
    raise SystemExit(status)


# --------------------------------------------------------------------------------------------------------------
# SHOT label file and validation split
# --------------------------------------------------------------------------------------------------------------


def _strict_pair(line: str):
    """Strict ``a,b`` / ``a b`` pair only (what a strict parser keeps)."""
    c = line.strip().strip("[]")
    parts = c.split(",") if "," in c else c.split()
    if len(parts) == 2 and parts[0] and parts[1]:
        try:
            x, y = int(parts[0].strip()), int(parts[1].strip())
            return [(x, y)] if y >= x else []
        except ValueError:
            return []
    return []


def shot_labels(args) -> None:
    paths = json.load(open(Path(args.data_root) / "paths.json"))
    root = Path(paths["autoshot_root"])
    gt = pickle.load(open(paths["autoshot_gt_pickle"], "rb"))
    test = {str(k): [(int(x[1]), int(y[0])) for x, y in zip(v[:-1], v[1:])] for k, v in gt.items()}
    lines = open(root / "kuaishou_v2.txt", encoding="utf-8", errors="replace").read().split("\n")
    folders = ("original_videos", "video_download", "ads_game_videos")
    disk_by_dir = {d: {p.stem for p in (root / d).glob("*.mp4") if not p.name.startswith("._")} for d in folders}
    disk = set().union(*disk_by_dir.values())

    entries, cur = [], None
    for line in lines:
        t = line.strip()
        if ".mp4" in t:
            cur = {"vid": re.match(r"\d+", t).group(0), "strict": 0, "robust": [], "odd": 0, "header_shape": re.sub(r"\d", "9", t)[:30]}
            entries.append(cur)
        elif t and cur is not None:
            nums = [int(x) for x in re.findall(r"\d+", t)]
            cur["strict"] += len(_strict_pair(t))
            cur["robust"] += [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
            cur["odd"] += len(nums) % 2
    by = collections.defaultdict(list)
    for e in entries:
        by[e["vid"]].append(e)
    labelled = set(by)
    last = {v: es[-1]["robust"] for v, es in by.items()}
    n_file = n_final = within_2 = 0
    for key, final in test.items():
        final_set = set(final)
        n_file += len(last[key])
        n_final += len(final)
        within_2 += sum(any(abs(t[0] - x) <= 2 and abs(t[1] - y) <= 2 for x, y in final_set) for t in last[key])
    dup_nontest = [(v, es) for v, es in by.items() if len(es) > 1 and v not in test]
    report = {
        "files_in_three_video_folders": sum(len(v) for v in disk_by_dir.values()),
        "per_folder": {k: len(v) for k, v in disk_by_dir.items()},
        "distinct_videos_on_disk": len(disk),
        "labelled_videos": len(labelled),
        "labelled_and_on_disk": len(labelled & disk),
        "labelled_not_on_disk": len(labelled - disk),
        "on_disk_without_labels": len(disk - labelled),
        "test_videos": len(test),
        "test_videos_inside_label_file": len(set(test) & labelled),
        "train_videos_after_excluding_test": len((labelled & disk) - set(test)),
        "header_entries": len(entries),
        "videos_with_duplicate_entries": sum(1 for es in by.values() if len(es) > 1),
        "non_test_duplicate_videos": len(dup_nontest),
        "non_test_duplicates_identical": sum(1 for _, es in dup_nontest if es[0]["robust"] == es[-1]["robust"]),
        "header_shapes": dict(collections.Counter(e["header_shape"] for e in entries)),
        "transitions_recovered_by_tolerant_parse": sum(len(e["robust"]) for e in entries),
        "transitions_kept_by_strict_parse": sum(e["strict"] for e in entries),
        "lines_with_odd_number_of_integers": sum(e["odd"] for e in entries),
        "test_labels_in_file": n_file,
        "test_labels_final_annotation": n_final,
        "test_file_labels_within_2_frames_of_final": within_2,
    }
    report["transitions_dropped_by_strict_parse"] = (
        report["transitions_recovered_by_tolerant_parse"] - report["transitions_kept_by_strict_parse"]
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def shot_split(args) -> None:
    from hybridsbd.datasets import PredecodedDataset
    from hybridsbd.utils import names_sha256

    def checksum(seed: int):
        ds = PredecodedDataset(args.cache, split="val", val_split_ratio=0.15, random_seed=seed)
        names = sorted({c["video_name"] for c in ds.mapping})
        return len(names), names_sha256(names)

    print("OLD RULE (split seed = training seed):")
    for seed in (42, 43, 44):
        print(" ", seed, *checksum(seed))
    print("FIXED (split_seed = 42 for every training seed):")
    for seed in (42, 43, 44):
        print(" ", seed, *checksum(42))


# --------------------------------------------------------------------------------------------------------------
# saved probabilities vs a fresh forward pass
# --------------------------------------------------------------------------------------------------------------


def fresh_pass(args) -> None:
    import numpy as np

    from hybridsbd.logging_utils import load_predictions

    fresh, saved = Path(args.fresh), Path(args.saved)
    keys = ("best_boundary_threshold", "best_gradual_threshold", "best_min_scene_len")
    out = {}
    for run in args.runs:
        out[run] = {}
        for test_set in ("shot", "clipshots", "bbc"):
            ay, az = load_predictions(saved / run / f"{test_set}_results.predictions.npz")
            by, bz = load_predictions(fresh / run / f"{test_set}_results.predictions.npz")
            assert set(ay) == set(by), "video sets differ"
            dy = max(float(np.max(np.abs(ay[k].astype(np.float64) - by[k].astype(np.float64)))) for k in ay)
            dz = None if az is None else max(float(np.max(np.abs(az[k].astype(np.float64) - bz[k].astype(np.float64)))) for k in az)
            a = json.load(open(saved / run / f"{test_set}_results.json"))
            b = json.load(open(fresh / run / f"{test_set}_results.json"))
            same = all(a.get(k) == b.get(k) for k in keys) and all(a["overall"][k] == b["overall"][k] for k in ("tp", "fp", "fn"))
            out[run][test_set] = {
                "videos": len(ay),
                "max_abs_diff_py": dy,
                "max_abs_diff_pz": dz,
                "saved_f1": a["overall"]["f1"],
                "fresh_f1": b["overall"]["f1"],
                "thresholds_and_counts_identical": bool(same),
                "saved": [a.get(k) for k in keys],
                "fresh": [b.get(k) for k in keys],
            }
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="forward/backward pass of all variants on dummy input")
    p.add_argument("--output", default="outputs/ablations/preflight.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq_len", type=int, default=128)
    p.set_defaults(func=preflight)

    p = sub.add_parser("a0-architecture", help="architecture check of a trained A0 checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=a0_architecture)

    p = sub.add_parser("a0-reproduction", help="A0 F1 against the paper's acceptable ranges")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--output")
    p.set_defaults(func=a0_reproduction)

    p = sub.add_parser("shot-labels", help="audit of the SHOT label file")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=shot_labels)

    p = sub.add_parser("shot-split", help="SHOT validation-split checksums for seeds 42/43/44")
    p.add_argument("cache", help="SHOT training cache (autoshot_train)")
    p.set_defaults(func=shot_split)

    p = sub.add_parser("fresh-pass", help="saved probabilities vs a fresh forward pass")
    p.add_argument("runs", nargs="+")
    p.add_argument("--saved", required=True, help="outputs root with the saved run directories")
    p.add_argument("--fresh", required=True, help="directory with the fresh evaluations (<run>/<set>_results.*)")
    p.add_argument("--output", required=True)
    p.set_defaults(func=fresh_pass)
    return parser.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    ns.func(ns)
