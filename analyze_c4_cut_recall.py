import json, sys
from pathlib import Path
sys.path.insert(0, "/workspace/ShotBoundaryDetection")
import analyze_clipshots_protocol_a as an
ann_path = json.load(open("/workspace/data/paths.json"))["clipshots_test_annotations"]
rows = {}
for rid in ("A0", "C4"):
    d = Path("/workspace/outputs/ablations_v2") / rid
    vy, vz = an.load_predictions(d / "clipshots_results.predictions.npz")
    ann = an.load_annotations(ann_path, vy.keys())
    res = json.load(open(d / "clipshots_results.json"))
    rows[rid] = an.analyse_run(rid, vy, vz, ann, res)
json.dump({"note": "extra analysis for spec Section 9 (C4 hypothesis: lower cut recall than A0); same code/definitions as Table 8", "runs": rows},
          open("/workspace/outputs/ablations_v2/c4_cut_recall.json", "w"), indent=1)
for k, v in rows.items(): print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if "recall" in kk or "count" in kk or "n_" in kk})
