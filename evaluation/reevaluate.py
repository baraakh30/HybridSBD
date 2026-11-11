"""Offline re-evaluation of trained runs from their saved probability tracks (no model is run, nothing is trained).

Every test run of ``evaluation/evaluate.py`` stores ``<set>_results.predictions.npz``; this script re-scores those tracks
with the shared protocol of ``hybridsbd.scoring`` under a different threshold rule:

``lmin1``      Protocol A with l_min = 1 on SHOT, ClipShots and BBC: thresholds are searched on each test set and the
               minimum scene length is fixed to 1. For every run it also re-computes the *stored* evaluation (ClipShots with
               the l_min grid 1-4) and records whether it is reproduced exactly. Writes ``reeval_lmin1.json/.csv``.
``protocol-b`` Protocol B: thresholds are selected once on the 67 SHOT validation videos (full videos, l_min = 1; the
               ``shot_val_results.json`` written by ``scripts/protocol_b_validation.sh``) and applied unchanged to the
               three test sets. Writes ``protocolB_results[TAG].json/.csv``.

Examples::

    python evaluation/reevaluate.py lmin1 --data_root /workspace/data --outputs /workspace/outputs --runs A0 A1
    python evaluation/reevaluate.py protocol-b --data_root /workspace/data --outputs /workspace/outputs
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from evaluation import clipshots_type_analysis as analysis
from evaluation.data import load_bbc_gt, load_clipshots_gt, load_shot_gt
from hybridsbd.configs.ablation_config import LADDER_RUNS, MAIN_RUNS, RUN_DIRS
from hybridsbd.logging_utils import load_predictions
from hybridsbd.scoring import best_setting, sweep

DEFAULT_RUNS = MAIN_RUNS + LADDER_RUNS
#: Runs (and their extra seeds) for which the per-type ClipShots analysis (Tables 8 and 9) is computed.
ANALYSED = ("A0", "A1", "A2", "C6")
TEST_SETS = ("shot", "clipshots", "bbc")
SET_NAMES = {"shot": "SHOT", "clipshots": "ClipShots", "bbc": "BBC"}

_CTX: dict = {}


def _init(ctx: dict) -> None:
    _CTX.update(ctx)


def _tau_g(value):
    return "" if value is None else value


def _run_sweep(run_dir: Path, test_set: str, single_output: bool, min_scene_lengths, **grid):
    video_y, video_z = load_predictions(run_dir / f"{test_set}_results.predictions.npz")
    result = sweep(
        video_y,
        video_z,
        _CTX["gt"][test_set],
        single_output=single_output,
        min_scene_lengths=min_scene_lengths,
        clip_to_annotated_range=(test_set == "bbc"),
        **grid,
    )
    return result, video_y, video_z


def _clipshots_analysis(run_dir: Path, run: str, single_output: bool, tau_c, tau_g):
    video_y, video_z = load_predictions(run_dir / "clipshots_results.predictions.npz")
    annotations = analysis.load_annotations(_CTX["cs_ann"], video_y.keys())
    setting = {"single_output": single_output, "best_boundary_threshold": tau_c, "best_gradual_threshold": tau_g, "best_min_scene_len": 1}
    return analysis.analyse_run(run, video_y, video_z, annotations, setting)


# --------------------------------------------------------------------------------------------------------------
# Protocol A with l_min = 1
# --------------------------------------------------------------------------------------------------------------


def evaluate_lmin1(run: str) -> dict:
    t0 = time.time()
    run_dir = Path(_CTX["outputs"]) / RUN_DIRS[run]
    record = {"run": run, "sets": {}}
    for test_set in TEST_SETS:
        stored = json.load(open(run_dir / f"{test_set}_results.json"))
        single = bool(stored.get("single_output", False))
        lmins = [1, 2, 3, 4] if test_set == "clipshots" else [1]
        result, video_y, _ = _run_sweep(run_dir, test_set, single, lmins)
        grid = result["all_results"]
        prev_key, prev = best_setting(grid)  # the stored evaluation (l_min grid on ClipShots)
        new_key, new = best_setting(grid, 1)  # l_min = 1 everywhere
        o = stored["overall"]
        reproduced = (
            abs(prev["f1"] - o["f1"]) < 1e-12
            and prev_key[0] == stored["best_boundary_threshold"]
            and prev_key[1] == stored["best_gradual_threshold"]
            and prev_key[2] == stored.get("best_min_scene_len", 1)
            and (prev["tp"], prev["fp"], prev["fn"]) == (o["tp"], o["fp"], o["fn"])
        )
        stored_l1 = None
        if test_set == "clipshots":  # the stored sweep contained l_min 1..4
            cands = [(ast.literal_eval(k), v) for k, v in stored["all_results"].items() if ast.literal_eval(k)[2] == 1]
            best_k, best_v = None, None
            for k, v in cands:
                if best_v is None or v["f1"] > best_v["f1"]:
                    best_k, best_v = k, v
            stored_l1 = {"key": list(best_k), "f1": best_v["f1"]}
        record["sets"][test_set] = {
            "single_output": single,
            "previous_recomputed": {"tau_c": prev_key[0], "tau_g": prev_key[1], "l_min": prev_key[2], **prev},
            "previous_stored": {
                "tau_c": stored["best_boundary_threshold"],
                "tau_g": stored["best_gradual_threshold"],
                "l_min": stored.get("best_min_scene_len", 1),
                **o,
            },
            "previous_reproduced_exactly": bool(reproduced),
            "lmin1": {"tau_c": new_key[0], "tau_g": new_key[1], "l_min": 1, **new},
            "stored_old_sweep_best_at_lmin1": stored_l1,
            "per_lmin": (
                {
                    str(lm): {"tau_c": best_setting(grid, lm)[0][0], "tau_g": best_setting(grid, lm)[0][1], **best_setting(grid, lm)[1]}
                    for lm in lmins
                }
                if test_set == "clipshots"
                else None
            ),
            "videos": len([v for v in video_y if v in _CTX["gt"][test_set]]),
        }
    if run.split("_s")[0] in ANALYSED:
        l1 = record["sets"]["clipshots"]["lmin1"]
        record["analysis_lmin1"] = _clipshots_analysis(run_dir, run, record["sets"]["clipshots"]["single_output"], l1["tau_c"], l1["tau_g"])
    record["seconds"] = round(time.time() - t0, 1)
    print(f"done {run} in {record['seconds']} s", flush=True)
    return record


# --------------------------------------------------------------------------------------------------------------
# Protocol B
# --------------------------------------------------------------------------------------------------------------


def evaluate_protocol_b(run: str) -> dict:
    t0 = time.time()
    run_dir = Path(_CTX["outputs"]) / RUN_DIRS[run]
    val = json.load(open(Path(_CTX["val_root"]) / run / "shot_val_results.json"))
    single = bool(val.get("single_output", False))
    tau_c, tau_g = val["best_boundary_threshold"], val["best_gradual_threshold"]
    record = {"run": run, "val": {"tau_c": tau_c, "tau_g": tau_g, "l_min": 1, **val["overall"], "videos": val["num_videos"]}, "sets": {}}
    for test_set in TEST_SETS:
        result, _, _ = _run_sweep(run_dir, test_set, single, [1], boundary_thresholds=[tau_c], gradual_thresholds=[tau_g])
        ((_, scores),) = result["all_results"].items()
        record["sets"][test_set] = {"tau_c": tau_c, "tau_g": tau_g, "l_min": 1, **scores}
    if run.split("_s")[0] in ANALYSED:
        record["analysis"] = _clipshots_analysis(run_dir, run, single, tau_c, tau_g)
    record["seconds"] = round(time.time() - t0, 1)
    print("done", run, record["seconds"], flush=True)
    return record


# --------------------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------------------


def write_csv(path: Path, records: list, pick) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "test set", "l_min", "tau_c", "tau_g", "precision", "recall", "F1"])
        for rec in records:
            for test_set in TEST_SETS:
                d = pick(rec, test_set)
                writer.writerow(
                    [rec["run"], SET_NAMES[test_set], 1, d["tau_c"], _tau_g(d["tau_g"]), f"{d['p']:.6f}", f"{d['r']:.6f}", f"{d['f1']:.6f}"]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["lmin1", "protocol-b"])
    parser.add_argument(
        "--data_root", default=os.environ.get("DATA_ROOT", "data"), help="root of the data (AutoShot, ClipShots, BBC, cache)"
    )
    parser.add_argument(
        "--outputs", default=os.environ.get("OUTPUT_ROOT", "outputs"), help="root of the training outputs (run directories)"
    )
    parser.add_argument("--out", default=None, help="output directory (default: <outputs>/reeval_lmin1 or <outputs>/protocolB)")
    parser.add_argument(
        "--val_root", default=None, help="protocol-b: directory with <run>/shot_val_results.json (default: <outputs>/protocolB/val)"
    )
    parser.add_argument("--tag", default="", help="protocol-b: suffix of the output file names")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, choices=sorted(RUN_DIRS))
    parser.add_argument("--procs", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = Path(args.data_root)
    t0 = time.time()
    first = Path(args.outputs) / RUN_DIRS["A0"] / "clipshots_results.predictions.npz"
    clipshots_names = list(load_predictions(first)[0].keys())
    cs_ann = data / "ClipShots" / "ClipShots" / "annotations" / "test.json"
    gt = {
        "shot": load_shot_gt(data / "AutoShot" / "gt_scenes_dict_baseline_v2.pickle"),
        "clipshots": load_clipshots_gt(cs_ann, clipshots_names),
        "bbc": load_bbc_gt(data / "BBC", data / "cache" / "bbc"),
    }
    print({k: len(v) for k, v in gt.items()}, flush=True)
    ctx = {
        "outputs": args.outputs,
        "gt": gt,
        "cs_ann": str(cs_ann),
        "val_root": args.val_root or str(Path(args.outputs) / "protocolB" / "val"),
    }
    worker = evaluate_lmin1 if args.mode == "lmin1" else evaluate_protocol_b
    with Pool(args.procs, initializer=_init, initargs=(ctx,)) as pool:
        records = pool.map(worker, args.runs, chunksize=1)

    out = Path(args.out or Path(args.outputs) / ("reeval_lmin1" if args.mode == "lmin1" else "protocolB"))
    out.mkdir(parents=True, exist_ok=True)
    total = round(time.time() - t0, 1)
    stem, pick = (
        ("reeval_lmin1", lambda r, s: r["sets"][s]["lmin1"])
        if args.mode == "lmin1"
        else (f"protocolB_results{args.tag}", lambda r, s: r["sets"][s])
    )
    (out / f"{stem}.json").write_text(json.dumps({"runtime_seconds": total, "runs": {r["run"]: r for r in records}}, indent=1))
    write_csv(out / f"{stem}.csv", records, pick)
    print("total seconds", total)


if __name__ == "__main__":
    main()
