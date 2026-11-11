"""Protocol B: thresholds selected once on the 67 SHOT validation videos (full videos, l_min = 1), then applied unchanged
to the SHOT test, ClipShots test and BBC test sets (l_min = 1). Uses the saved full-video probabilities of every run
(same sweep/scoring code as reeval_lmin1.py, which imports the test scripts' functions)."""
import argparse, csv, json, time
from multiprocessing import Pool
from pathlib import Path

import analyze_clipshots_protocol_a as AN
import reeval_lmin1 as RL
from run_logging import load_predictions

G = {}


def work(run):
    t0 = time.time()
    rdir = Path(G["outputs"]) / RL.RUNS[run]
    val = json.load(open(Path(G["vroot"]) / run / "shot_val_results.json"))
    single = bool(val.get("single_output", False))
    bt, g = val["best_boundary_threshold"], val["best_gradual_threshold"]
    rec = {"run": run, "val": {"tau_c": bt, "tau_g": g, "l_min": 1, **val["overall"], "videos": val["num_videos"]}, "sets": {}}
    for s in RL.SETS:
        vy, vz = load_predictions(rdir / f"{s}_results.predictions.npz")
        saved = RL.BT, RL.GT_GRAD
        RL.BT, RL.GT_GRAD = [bt], [g]
        try:
            res = RL.sweep(vy, vz, G["gt"][s], single, [1], s == "bbc")
        finally:
            RL.BT, RL.GT_GRAD = saved
        (k, v), = res.items()
        rec["sets"][s] = {"tau_c": bt, "tau_g": g, "l_min": 1, **v}
    if run.split("_s")[0] in RL.ANALYSED:
        vy, vz = load_predictions(rdir / "clipshots_results.predictions.npz")
        ann = AN.load_annotations(G["cs_ann"], vy.keys())
        fake = {"single_output": single, "best_boundary_threshold": bt, "best_gradual_threshold": g, "best_min_scene_len": 1}
        rec["analysis"] = AN.analyse_run(run, vy, vz, ann, fake)
    rec["seconds"] = round(time.time() - t0, 1)
    print("done", run, rec["seconds"], flush=True)
    return rec


def init(outputs, vroot, gt, cs_ann):
    G.update(outputs=outputs, vroot=vroot, gt=gt, cs_ann=cs_ann)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="/workspace/outputs")
    ap.add_argument("--val_root", default="/workspace/outputs/protocolB/val")
    ap.add_argument("--shot_gt", default="/workspace/data/AutoShot/gt_scenes_dict_baseline_v2.pickle")
    ap.add_argument("--cs_ann", default="/workspace/data/ClipShots/ClipShots/annotations/test.json")
    ap.add_argument("--bbc_root", default="/workspace/data/BBC")
    ap.add_argument("--bbc_cache", default="/workspace/data/cache/bbc")
    ap.add_argument("--out", default="/workspace/outputs/protocolB")
    ap.add_argument("--tag", default="")
    ap.add_argument("--runs", nargs="*", default=RL.DEFAULT_RUNS)
    ap.add_argument("--procs", type=int, default=17)
    a = ap.parse_args()
    t0 = time.time()
    cs_names = list(load_predictions(Path(a.outputs) / RL.RUNS["A0"] / "clipshots_results.predictions.npz")[0].keys())
    gt = RL.load_gts(a, {"clipshots": cs_names})
    init(a.outputs, a.val_root, gt, a.cs_ann)
    with Pool(a.procs, initializer=init, initargs=(a.outputs, a.val_root, gt, a.cs_ann)) as pool:
        recs = pool.map(work, a.runs, chunksize=1)
    total = round(time.time() - t0, 1)
    out = Path(a.out)
    (out / f"protocolB_results{a.tag}.json").write_text(json.dumps({"runtime_seconds": total, "runs": {r["run"]: r for r in recs}}, indent=1))
    names = {"shot": "SHOT", "clipshots": "ClipShots", "bbc": "BBC"}
    with open(out / f"protocolB_results{a.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "test set", "l_min", "tau_c", "tau_g", "precision", "recall", "F1"])
        for r in recs:
            for s in RL.SETS:
                d = r["sets"][s]
                w.writerow([r["run"], names[s], 1, d["tau_c"], "" if d["tau_g"] is None else d["tau_g"], f"{d['p']:.6f}", f"{d['r']:.6f}", f"{d['f1']:.6f}"])
    print("total seconds", total)


if __name__ == "__main__":
    main()
