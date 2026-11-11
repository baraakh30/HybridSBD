"""Manifest and integrity check of the checkpoints behind the paper's tables.

The paper needs exactly 21 files, one ``best_model.pt`` per run in ``hybridsbd.configs.ablation_config.RUN_DIRS``. They are not
stored in git (``*.pt`` is ignored). ``checkpoints/MANIFEST.csv`` records, for each of them, its size, SHA-256, best epoch,
best SHOT validation F1 and parameter count, so that a copy obtained elsewhere can be verified.

``build``   write ``checkpoints/MANIFEST.csv`` from a training output root (needs PyTorch to read the checkpoint metadata).
``verify``  compare the checkpoints under an output root with ``checkpoints/MANIFEST.csv`` (SHA-256; exit code 1 on mismatch).
            ``--flat`` expects the layout of the Zenodo record: one ``<run>.pt`` per run in a single folder.

Examples::

    python checks/checkpoints.py build --outputs <outputs>
    python checks/checkpoints.py verify --outputs <outputs>
    python checks/checkpoints.py verify --outputs <folder downloaded from Zenodo> --flat
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from hybridsbd.configs.ablation_config import LADDER_RUNS, MAIN_RUNS, RUN_DIRS, SEED_RUNS
from hybridsbd.utils import file_sha256

MANIFEST = Path(__file__).resolve().parent.parent / "checkpoints" / "MANIFEST.csv"
FIELDS = ["run", "seed", "used_in", "file", "size_mb", "sha256", "best_epoch", "best_val_f1", "parameters"]

USED_IN = {
    "A0": "primary model: Tables 3, 4, 6-11",
    **{r: "Table 7 (supervision)" for r in ("A1", "A2", "A3", "A4")},
    **{r: "Table 10 (leave-one-out)" for r in MAIN_RUNS if r.startswith("C")},
    **{r: "Table 11 (build-up)" for r in LADDER_RUNS},
    **{r: "Table 8 (seeds)" for r in SEED_RUNS},
}


def _path(outputs: Path, run: str) -> Path:
    return outputs / RUN_DIRS[run] / "best_model.pt"


def build(outputs: Path) -> int:
    import torch

    from hybridsbd.configs.ablation_config import AblationConfig
    from hybridsbd.models.ablation_model import create_ablation_detector

    rows = []
    for run in RUN_DIRS:
        path = _path(outputs, run)
        if not path.exists():
            print(f"MISSING  {run}: {path}", file=sys.stderr)
            return 1
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        n_params = create_ablation_detector(AblationConfig(**ckpt["config"])).count_parameters()[0]
        rows.append(
            {
                "run": run,
                "seed": int(run.split("_s")[1]) if "_s" in run else 42,
                "used_in": USED_IN[run],
                "file": f"{RUN_DIRS[run]}/best_model.pt",
                "size_mb": f"{path.stat().st_size / 2**20:.1f}",
                "sha256": file_sha256(path),
                "best_epoch": ckpt["epoch"],
                "best_val_f1": f"{ckpt['best_val_f1']:.4f}",
                "parameters": n_params,
            }
        )
        print(f"ok  {run:8s} epoch {ckpt['epoch']:>2}  {rows[-1]['size_mb']:>5} MB")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    total = sum(float(r["size_mb"]) for r in rows)
    print(f"{len(rows)} checkpoints, {total:.0f} MB -> {MANIFEST}")
    return 0


def verify(outputs: Path, flat: bool = False) -> int:
    bad = 0
    with open(MANIFEST, newline="") as handle:
        for row in csv.DictReader(handle):
            path = outputs / (f"{row['run']}.pt" if flat else row["file"])
            if not path.exists():
                status = "MISSING"
            elif file_sha256(path) != row["sha256"]:
                status = "DIFFERENT"
            else:
                status = "ok"
            bad += status != "ok"
            print(f"{status:9s} {row['run']:8s} {row['file']}")
    print("all checkpoints match the manifest" if not bad else f"{bad} checkpoint(s) do not match")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument(
        "--outputs", type=Path, required=True, help="training output root (contains ablations_v2/, ablations_v3/, seed_runs/)"
    )
    parser.add_argument("--flat", action="store_true", help="verify: files are <run>.pt in one folder (Zenodo layout)")
    args = parser.parse_args()
    return build(args.outputs) if args.command == "build" else verify(args.outputs, args.flat)


if __name__ == "__main__":
    sys.exit(main())
