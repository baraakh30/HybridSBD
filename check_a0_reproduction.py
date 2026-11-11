"""A0 reproduction Check 2 (spec Section 1): Protocol-A F1 vs the paper's acceptable ranges.

Check 1 (architecture) is enforced by test_ablation_cpu.py / validate_ablation.py.
Exit code: 0 = all within range, 3 = at least one outside (spec: report to the team;
ablations may proceed on the same code), 2 = results missing.
"""
import argparse
import json
from pathlib import Path

RANGES = {  # name: (paper, low, high)
    "shot": ("SHOT", 0.924, 0.915, 0.930),
    "clipshots": ("ClipShots", 0.793, 0.785, 0.800),
    "bbc": ("BBC", 0.971, 0.965, 0.975),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    report, status = {}, 0
    for key, (label, paper, lo, hi) in RANGES.items():
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
    verdict = {0: "A0 reproduces the paper within the acceptable ranges.",
               2: "Check 2 incomplete: result files missing.",
               3: "Check 2 FAILED: report to the team; ablations may proceed on the same code but the "
                  "main-paper numbers need discussion (spec Section 1)."}[status]
    print(verdict)
    out = Path(args.output) if args.output else run_dir / "a0_reproduction_check.json"
    out.write_text(json.dumps({"checks": report, "verdict": verdict, "exit_code": status}, indent=2), encoding="utf-8")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
