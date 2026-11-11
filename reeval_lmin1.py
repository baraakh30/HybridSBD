"""Evaluation-only re-run of the ablation runs with l_min = 1 on SHOT, ClipShots and BBC.

Uses the saved full-video probabilities (<set>_results.predictions.npz) of each run's best (EMA) checkpoint.
Nothing is trained or modified. The sweep code is the same as in test_autoshot.py / test_bbc.py (same functions,
same threshold grids, same iteration order and tie-breaking); only the l_min list changes.

For every run and test set it also reproduces the PREVIOUS evaluation (ClipShots: l_min grid 1-4; SHOT/BBC: no merging)
and compares it with the numbers stored in the run's *_results.json.
"""
import argparse, csv, json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import test_autoshot as TA
import test_bbc as TB
import test_clipshots as TC
from run_logging import load_predictions
import analyze_clipshots_protocol_a as AN

BT = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
GT_GRAD = [0.3, 0.4, 0.5, 0.6]
RUNS = {  # run -> run directory relative to the outputs root
    **{r: f"ablations_v2/{r}" for r in ["A0", "A1", "A2", "A3", "A4", "C1", "C2a", "C2b", "C3", "C4", "C5a", "C5b", "C6"]},
    **{r: f"ablations_v3/{r}" for r in ["M0_min", "M1_edge", "B_noatt", "S1_rel"]},
    **{f"{r}_s{s}": f"seed_runs/{r}_s{s}/{r}" for r in ("A0", "A1") for s in (43, 44)},   # extra training seeds
}
DEFAULT_RUNS = [r for r in RUNS if "_s4" not in r]
ANALYSED = ("A0", "A1", "A2", "C6")
SETS = ("shot", "clipshots", "bbc")


def sweep(vy, vz, gt_dict, single, lmins, bbc):
    """Same loops as evaluate() in the test scripts. Returns {(bt, gt, lmin): {p, r, f1, tp, fp, fn}}."""
    mod = TB if bbc else TA
    gts = [None] if single else GT_GRAD
    out = {}
    for bt in BT:
        for g in gts:
            for lm in lmins:
                tp = fp = fn = 0
                for vid in vy:
                    if vid not in gt_dict:
                        continue
                    gt_sc = gt_dict[vid]
                    y = vy[vid]
                    z = vz[vid] if vz is not None else np.zeros_like(y)
                    if bbc:  # BBC test clips predictions to the annotated frame range (test_bbc.evaluate)
                        m = int(gt_sc[-1, 1]) + 1
                        y, z = y[:m], z[:m]
                    pm = mod._build_pred_many_hot(y, z, bt, g)
                    sc = mod.predictions_to_scenes(pm)
                    if not bbc:
                        sc = TA.apply_min_scene_length(sc, lm)
                    _, _, _, (t, f, n) = mod.evaluate_scenes(gt_sc, sc, n_frames_miss_tolerance=2)
                    tp += t; fp += f; fn += n
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * p * r / (p + r) if p + r else 0.0
                out[(bt, g, lm)] = dict(p=p, r=r, f1=f1, tp=tp, fp=fp, fn=fn)
    return out


def best_of(res, lmin=None):
    """First strictly-greater F1 in iteration order (identical tie-breaking to the test scripts)."""
    best = None
    for k, v in res.items():
        if lmin is not None and k[2] != lmin:
            continue
        if best is None or v["f1"] > best[1]["f1"]:
            best = (k, v)
    return best


def load_gts(args, names_by_set):
    with open(args.shot_gt, "rb") as f:
        shot = pickle.load(f)
    cs = TC.load_gt_scenes(args.cs_ann, names_by_set["clipshots"])
    bbc = TB.load_gt_scenes(args.bbc_root, args.bbc_cache)
    return {"shot": shot, "clipshots": cs, "bbc": bbc}


G = {}


def work(job):
    run, root_name = job
    t0 = time.time()
    rdir = Path(G["out_root"]) / root_name
    rec = {"run": run, "sets": {}}
    for s in SETS:
        npz = rdir / f"{s}_results.predictions.npz"
        old = json.load(open(rdir / f"{s}_results.json"))
        vy, vz = load_predictions(npz)
        single = bool(old.get("single_output", False))
        lmins = [1, 2, 3, 4] if s == "clipshots" else [1]
        res = sweep(vy, vz, G["gt"][s], single, lmins, s == "bbc")
        prev_k, prev = best_of(res)          # previous evaluation (l_min grid for ClipShots)
        new_k, new = best_of(res, 1)         # l_min = 1 everywhere
        # compare with the stored previous results
        o = old["overall"]
        same_prev = (abs(prev["f1"] - o["f1"]) < 1e-12 and prev_k[0] == old["best_boundary_threshold"]
                     and prev_k[1] == old["best_gradual_threshold"]
                     and prev_k[2] == old.get("best_min_scene_len", 1)
                     and (prev["tp"], prev["fp"], prev["fn"]) == (o["tp"], o["fp"], o["fn"]))
        # lmin=1 grid entries stored in the old run (ClipShots only: old sweep contained l_min 1..4)
        old_l1 = None
        if s == "clipshots":
            cands = [(ast_key(k), v) for k, v in old["all_results"].items() if ast_key(k)[2] == 1]
            bk, bv = None, None
            for k, v in cands:
                if bv is None or v["f1"] > bv["f1"]:
                    bk, bv = k, v
            old_l1 = {"key": list(bk), "f1": bv["f1"]}
        rec["sets"][s] = {
            "single_output": single,
            "previous_recomputed": {"tau_c": prev_k[0], "tau_g": prev_k[1], "l_min": prev_k[2], **prev},
            "previous_stored": {"tau_c": old["best_boundary_threshold"], "tau_g": old["best_gradual_threshold"],
                                "l_min": old.get("best_min_scene_len", 1), **o},
            "previous_reproduced_exactly": bool(same_prev),
            "lmin1": {"tau_c": new_k[0], "tau_g": new_k[1], "l_min": 1, **new},
            "stored_old_sweep_best_at_lmin1": old_l1,
            "per_lmin": ({str(lm): {"tau_c": best_of(res, lm)[0][0], "tau_g": best_of(res, lm)[0][1], **best_of(res, lm)[1]}
                          for lm in lmins} if s == "clipshots" else None),
            "videos": len([v for v in vy if v in G["gt"][s]]),
        }
    # Table 8 / 9 analysis at l_min = 1 (ClipShots)
    if run.split("_s")[0] in ANALYSED:
        vy, vz = load_predictions(rdir / "clipshots_results.predictions.npz")
        ann = AN.load_annotations(G["cs_ann"], vy.keys())
        l1 = rec["sets"]["clipshots"]["lmin1"]
        fake = {"single_output": rec["sets"]["clipshots"]["single_output"],
                "best_boundary_threshold": l1["tau_c"], "best_gradual_threshold": l1["tau_g"], "best_min_scene_len": 1}
        rec["analysis_lmin1"] = AN.analyse_run(run, vy, vz, ann, fake)
    rec["seconds"] = round(time.time() - t0, 1)
    print(f"done {run} in {rec['seconds']} s", flush=True)
    return rec


def ast_key(s):
    import ast
    return ast.literal_eval(s)


def init(args, gt):
    G["out_root"] = args.outputs
    G["gt"] = gt
    G["cs_ann"] = args.cs_ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="/workspace/outputs")
    ap.add_argument("--shot_gt", default="/workspace/data/AutoShot/gt_scenes_dict_baseline_v2.pickle")
    ap.add_argument("--cs_ann", default="/workspace/data/ClipShots/ClipShots/annotations/test.json")
    ap.add_argument("--bbc_root", default="/workspace/data/BBC")
    ap.add_argument("--bbc_cache", default="/workspace/data/cache/bbc")
    ap.add_argument("--out", default="/workspace/outputs/reeval_lmin1")
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    first = Path(args.outputs) / RUNS["A0"] / "clipshots_results.predictions.npz"
    cs_names = list(load_predictions(first)[0].keys())
    gt = load_gts(args, {"clipshots": cs_names})
    print({k: len(v) for k, v in gt.items()}, flush=True)
    init(args, gt)
    jobs = [(r, RUNS[r]) for r in args.runs]
    with Pool(args.procs, initializer=init, initargs=(args, gt)) as pool:
        recs = pool.map(work, jobs, chunksize=1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    total = round(time.time() - t0, 1)
    (out / "reeval_lmin1.json").write_text(json.dumps({"runtime_seconds": total, "runs": {r["run"]: r for r in recs}}, indent=1))
    names = {"shot": "SHOT", "clipshots": "ClipShots", "bbc": "BBC"}
    with open(out / "reeval_lmin1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "test set", "l_min", "tau_c", "tau_g", "precision", "recall", "F1"])
        for r in recs:
            for s in SETS:
                d = r["sets"][s]["lmin1"]
                w.writerow([r["run"], names[s], 1, d["tau_c"], "" if d["tau_g"] is None else d["tau_g"],
                            f"{d['p']:.6f}", f"{d['r']:.6f}", f"{d['f1']:.6f}"])
    print("total seconds", total)


if __name__ == "__main__":
    main()
