"""Export the required Section-7 CSV from completed ablation run records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ablation_config import VARIANTS


FIELDS = [
    "run_id", "seed", "git_commit", "python_version", "pytorch_version", "cuda_version", "gpu_model",
    "supervision", "flags_json", "params_total", "params_trainable", "best_epoch", "train_hours",
    "evaluation_protocol", "val_F1", "shot_tau_c", "shot_tau_g", "shot_P", "shot_R", "shot_F1",
    "shot_TP", "shot_FP", "shot_FN", "cs_tau_c", "cs_tau_g", "cs_lmin", "cs_P", "cs_R", "cs_F1",
    "cs_TP", "cs_FP", "cs_FN", "bbc_tau_c", "bbc_tau_g", "bbc_P", "bbc_R", "bbc_F1", "bbc_TP",
    "bbc_FP", "bbc_FN", "latency_B16_ms", "vram_B16_MB",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    missing = {}
    for run_id, cfg in VARIANTS.items():
        run_dir = Path(args.output_root) / run_id
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config.json"
        if not summary_path.exists():
            print(f"SKIP {run_id}: no summary.json")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        env = config.get("environment", {})
        row = {field: "" for field in FIELDS}
        row.update({
            "run_id": run_id, "seed": config.get("args", {}).get("seed", 42),
            "git_commit": env.get("git_commit", ""), "python_version": env.get("python_version", ""),
            "pytorch_version": env.get("pytorch_version", ""), "cuda_version": env.get("cuda_version", ""),
            "gpu_model": env.get("gpu_model", ""), "supervision": cfg.supervision,
            "flags_json": json.dumps(cfg.to_dict(), sort_keys=True),
            "params_total": summary.get("parameters", ["", ""])[0],
            "params_trainable": summary.get("parameters", ["", ""])[1],
            "best_epoch": summary.get("best_epoch", ""), "train_hours": summary.get("train_hours", ""),
            "evaluation_protocol": "A_test_selected",
            "val_F1": summary.get("best_val_f1", ""),
        })
        for prefix, filename in (("shot", "shot_results.json"),
                                ("cs", "clipshots_results.json"),
                                ("bbc", "bbc_results.json")):
            result_path = run_dir / filename
            if not result_path.exists():
                missing.setdefault(run_id, []).append(filename)
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            overall = result.get("overall", {})
            for source, target in (("best_boundary_threshold", f"{prefix}_tau_c"),
                                   ("best_gradual_threshold", f"{prefix}_tau_g"),
                                   ("best_min_scene_len", "cs_lmin")):
                if source in result and target in row:
                    row[target] = result[source]
            for source, target in (("p", f"{prefix}_P"), ("r", f"{prefix}_R"),
                                   ("f1", f"{prefix}_F1"), ("tp", f"{prefix}_TP"),
                                   ("fp", f"{prefix}_FP"), ("fn", f"{prefix}_FN")):
                if source in overall:
                    row[target] = overall[source]
        benchmark_path = Path(args.output_root) / "benchmark_results.json"
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            for item in benchmark.get("models", []):
                if Path(item.get("checkpoint", "")).parent.name == run_id:
                    batch = item.get("batch_results", {}).get("16", item.get("batch_results", {}).get(16, {}))
                    row["latency_B16_ms"] = batch.get("mean_ms", "")
                    row["vram_B16_MB"] = batch.get("peak_memory_mb", "")
                    break
        if row["latency_B16_ms"] == "":
            missing.setdefault(run_id, []).append("benchmark (latency/vram)")
        rows.append(row)
    for run_id in VARIANTS:
        if not (Path(args.output_root) / run_id / "summary.json").exists():
            missing.setdefault(run_id, []).append("summary.json (training not completed)")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    commits = {r["git_commit"] for r in rows if r["git_commit"]}
    if len(commits) > 1:
        missing["_warning"] = [f"runs were trained on different git commits: {sorted(commits)} "
                               "(spec: all runs must use the same commit as A0)"]
    report = Path(args.output).with_suffix(".missing.json")
    report.write_text(json.dumps(missing, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")
    if missing:
        print(f"WARNING: incomplete data, see {report}: {json.dumps(missing)}")


if __name__ == "__main__":
    main()
