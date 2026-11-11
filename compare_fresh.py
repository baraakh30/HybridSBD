"""Compare saved probabilities with a fresh forward pass of the same checkpoint, and the previous results JSON."""
import json, sys
from pathlib import Path
import numpy as np
from run_logging import load_predictions
FRESH = Path("/workspace/outputs/reeval_lmin1/fresh"); SRC = Path("/workspace/outputs/ablations_v2")
out = {}
for run in sys.argv[1:]:
    out[run] = {}
    for s in ("shot", "clipshots", "bbc"):
        ay, az = load_predictions(SRC / run / f"{s}_results.predictions.npz")
        by, bz = load_predictions(FRESH / run / f"{s}_results.predictions.npz")
        assert set(ay) == set(by), "video sets differ"
        dy = max(float(np.max(np.abs(ay[k].astype(np.float64) - by[k].astype(np.float64)))) for k in ay)
        dz = None if az is None else max(float(np.max(np.abs(az[k].astype(np.float64) - bz[k].astype(np.float64)))) for k in az)
        a = json.load(open(SRC / run / f"{s}_results.json")); b = json.load(open(FRESH / run / f"{s}_results.json"))
        keys = ("best_boundary_threshold", "best_gradual_threshold", "best_min_scene_len")
        same = all(a.get(k) == b.get(k) for k in keys) and all(a["overall"][k] == b["overall"][k] for k in ("tp", "fp", "fn"))
        out[run][s] = {"videos": len(ay), "max_abs_diff_py": dy, "max_abs_diff_pz": dz, "saved_f1": a["overall"]["f1"],
                       "fresh_f1": b["overall"]["f1"], "thresholds_and_counts_identical": bool(same),
                       "saved": [a.get(k) for k in keys], "fresh": [b.get(k) for k in keys]}
Path("/workspace/outputs/reeval_lmin1/fresh_vs_saved.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
