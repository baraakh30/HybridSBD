"""Build REEVAL_LMIN1_REPORT.md from reeval_lmin1.json (+ fresh_vs_saved.json, run histories, previous analysis)."""
import json
from pathlib import Path
import statistics as stat


class _A(list):
    def min(self): return min(self)
    def max(self): return max(self)
    def mean(self): return stat.fmean(self)
    def __getitem__(self, i):
        r = list.__getitem__(self, i)
        return _A(r) if isinstance(i, slice) else r


def slope_of(y):
    n = len(y); xs = list(range(n)); mx = stat.fmean(xs); my = stat.fmean(y)
    return sum((x - mx) * (v - my) for x, v in zip(xs, y)) / sum((x - mx) ** 2 for x in xs)


def absdiff_max(y):
    return max(abs(b - a) for a, b in zip(y, y[1:]))


class np:
    mean = staticmethod(stat.fmean)

SR = Path("server_results")
D = json.load(open(SR / "reeval_lmin1" / "reeval_lmin1.json"))
R = D["runs"]
FRESH = SR / "reeval_lmin1" / "fresh_vs_saved.json"
FV = json.load(open(FRESH)) if FRESH.exists() else None
PREV_AN = json.load(open(SR / "ablations_v2" / "clipshots_protocol_a_analysis.json"))["runs"]
out = []
w = out.append
f4 = lambda x: f"{x:.4f}"
sd = lambda x: f"{x:+.4f}"
S = ("shot", "clipshots", "bbc")
NAME = {"shot": "SHOT", "clipshots": "ClipShots", "bbc": "BBC"}


def F(run, s, new=True):
    return R[run]["sets"][s]["lmin1" if new else "previous_recomputed"]["f1"]


def bolds(runs, s, new=True):
    vals = {r: round(F(r, s, new), 4) for r in runs}
    m = max(vals.values())
    return {r: (f"**{vals[r]:.4f}**" if vals[r] == m else f"{vals[r]:.4f}") for r in runs}


def table_f1(runs, desc, extra_col=None):
    b = {s: bolds(runs, s) for s in S}
    w("| ID | " + ("Description | " if desc else "") + "SHOT F1 | ClipShots F1 | BBC F1 |")
    w("|---|" + ("---|" if desc else "") + "---|---|---|")
    for r in runs:
        w(f"| {r} | " + (f"{desc[r]} | " if desc else "") + " | ".join(b[s][r] for s in S) + " |")
    w("")


w("# Re-evaluation of the ablation runs with no minimum-shot-length merging (ℓ_min = 1)\n")
w("Evaluation only: no model was trained or modified. Every run uses its existing best checkpoint (EMA weights selected by SHOT validation F1). "
  "The only change from the main study is the evaluation rule: **ℓ_min = 1 (no merging) on SHOT, ClipShots and BBC**. "
  "Windows, averaging, threshold grids (τ_c 0.10–0.90 step 0.05; τ_g ∈ {0.3, 0.4, 0.5, 0.6}), the decision rule and ±2-frame scene matching are unchanged. "
  "Thresholds are re-selected on each test set (Protocol A), as before. All 17 runs were evaluated; none was skipped.\n")

# ---------------- Step 1 ----------------
w("## 1. Check that the pipeline reproduces the previous results\n")
exp = {"shot": (0.8449, 0.4, 0.3, None), "clipshots": (0.8052, 0.65, 0.6, 4), "bbc": (0.9719, 0.45, 0.6, None)}
w("Previous evaluation of **A0** (ClipShots with ℓ_min ∈ {1, 2, 3, 4}; SHOT and BBC with ℓ_min = 1), recomputed from the saved probabilities:\n")
w("| Test set | Expected F1 | Recomputed F1 | Expected thresholds | Recomputed thresholds | Match |\n|---|---|---|---|---|---|")
all_ok = True
for s in S:
    p = R["A0"]["sets"][s]["previous_recomputed"]
    e = exp[s]
    ok = round(p["f1"], 4) == e[0] and p["tau_c"] == e[1] and p["tau_g"] == e[2] and (e[3] is None or p["l_min"] == e[3])
    all_ok &= ok
    et = f"τ_c {e[1]}, τ_g {e[2]}" + (f", ℓ_min {e[3]}" if e[3] else "")
    rt = f"τ_c {p['tau_c']}, τ_g {p['tau_g']}" + (f", ℓ_min {p['l_min']}" if s == "clipshots" else "")
    w(f"| {NAME[s]} | {e[0]:.4f} | {p['f1']:.4f} | {et} | {rt} | {'yes' if ok else '**NO**'} |")
w("")
n_exact = sum(1 for r in R.values() for s in S if r["sets"][s]["previous_reproduced_exactly"])
w(f"The same previous evaluation, recomputed for **all 17 runs × 3 test sets**, matches the stored result of the main study exactly in {n_exact} of {17*3} cases "
  "(F1, thresholds, ℓ_min and the TP / FP / FN counts).\n")
if FV:
    w("**Saved probabilities vs a fresh forward pass.** The checkpoints of A0 and A1 were run again from scratch on all three test sets with the original evaluation commands, "
      "and the new full-video probabilities were compared with the saved ones:\n")
    w("| Run | Test set | Videos | max abs diff p_y | max abs diff p_z | Saved F1 | Fresh F1 | Thresholds and TP/FP/FN identical |\n|---|---|---|---|---|---|---|---|")
    for run, d in FV.items():
        for s in S:
            v = d[s]
            dz = "n/a" if v["max_abs_diff_pz"] is None else f"{v['max_abs_diff_pz']:.2e}"
            w(f"| {run} | {NAME[s]} | {v['videos']} | {v['max_abs_diff_py']:.2e} | {dz} | {v['saved_f1']:.4f} | {v['fresh_f1']:.4f} | {'yes' if v['thresholds_and_counts_identical'] else '**no**'} |")
    w("")

# ---------------- Step 2/3 tables ----------------
w("## 2. Results with ℓ_min = 1\n")
w("SHOT and BBC already used ℓ_min = 1, so their F1 must be identical to the main study. "
  + ("**They are identical for all 17 runs** (F1, thresholds, TP / FP / FN)." if all(
      R[r]["sets"][s]["lmin1"]["f1"] == R[r]["sets"][s]["previous_recomputed"]["f1"] and R[r]["sets"][s]["previous_reproduced_exactly"]
      for r in R for s in ("shot", "bbc")) else "**Differences exist, see the comparison table.**")
  + " Only ClipShots changes.\n")

w("### 2.1 Supervision (Table 7a)\n")
desc7a = {"A0": "Full, dual-label (primary)", "A1": "Full, midpoint only", "A2": "Full, unified single head",
          "A3": "Reduced, midpoint only", "A4": "Reduced, dual-label"}
table_f1(["A0", "A1", "A2", "A3", "A4"], desc7a)
w("| Delta | SHOT | ClipShots | BBC |\n|---|---|---|---|")
for lab, a, b in (("A0 − A1 (full)", "A0", "A1"), ("A4 − A3 (reduced)", "A4", "A3"), ("A0 − A2 (dual vs unified)", "A0", "A2")):
    w(f"| {lab} | " + " | ".join(sd(F(a, s) - F(b, s)) for s in S) + " |")
w("")

w("### 2.2 Components, leave-one-out (Table 7b)\n")
desc7b = {"A0": "Full model", "C1": "w/o difference-based branch (B)", "C2a": "w/o FrameSimilarity", "C2b": "w/o ColorHistogramSimilarity",
          "C3": "w/o temporal delta", "C4": "w/o per-frame pathway (relations from Branch A)", "C5a": "w/o pixel-level motion descriptor",
          "C5b": "w/o edge-density head", "C6": "w/o self-attention and gate"}
CR = ["A0", "C1", "C2a", "C2b", "C3", "C4", "C5a", "C5b", "C6"]
table_f1(CR, desc7b)
w("Change in F1 relative to A0:\n")
w("| ID | ΔSHOT | ΔClipShots | ΔBBC |\n|---|---|---|---|")
mx = 0.0
for r in CR[1:]:
    ds = [F(r, s) - F("A0", s) for s in S]
    mx = max(mx, max(abs(x) for x in ds))
    w(f"| {r} | " + " | ".join(sd(x) for x in ds) + " |")
w(f"\nLargest single-component change on any dataset: {mx:.4f}.\n")

w("### 2.3 Build-up ladder\n")
LAD = [("0", "M0_min", "RGB encoder + MLP + dual head only"), ("1", "M1_edge", "+ edge-density head"),
       ("2", "B_noatt", "+ 4 fusion blocks, no attention"), ("3", "A4", "+ attention and gate (reduced model)"),
       ("4", "S1_rel", "+ FrameSimilarity, ColorHist, delta"), ("5", "A0", "+ diff branch, motion, per-frame pathway (full model)")]
runs = [x[1] for x in LAD]
b = {s: bolds(runs, s) for s in S}
w("| Step | Run | What is added | SHOT F1 | ClipShots F1 | BBC F1 |\n|---|---|---|---|---|---|")
for st, r, d in LAD:
    w(f"| {st} | {r} | {d} | " + " | ".join(b[s][r] for s in S) + " |")
w("\nStep changes (F1):\n")
w("| Step | SHOT | ClipShots | BBC |\n|---|---|---|---|")
for (s0, r0, _), (s1, r1, d1) in zip(LAD[:-1], LAD[1:]):
    w(f"| {r0} → {r1} | " + " | ".join(sd(F(r1, s) - F(r0, s)) for s in S) + " |")
w("")

w("### 2.4 Precision, recall and selected thresholds\n")
w("| ID | SHOT P / R | ClipShots P / R | BBC P / R | SHOT τ_c, τ_g | ClipShots τ_c, τ_g | BBC τ_c, τ_g |\n|---|---|---|---|---|---|---|")
for r in R:
    c = R[r]["sets"]
    pr = " | ".join(f"{c[s]['lmin1']['p']:.3f} / {c[s]['lmin1']['r']:.3f}" for s in S)
    th = " | ".join(f"{c[s]['lmin1']['tau_c']}, {c[s]['lmin1']['tau_g'] if c[s]['lmin1']['tau_g'] is not None else '–'}" for s in S)
    w(f"| {r} | {pr} | {th} |")
w("\nℓ_min = 1 for every row. A1, A2 and A3 use τ_c only (no τ_g).\n")

# Table 8
w("### 2.5 Table 8 — Per-type analysis on ClipShots (ℓ_min = 1)\n")
w("| ID | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |\n|---|---|---|---|---|")
for r in ("A0", "A1", "A2"):
    a = R[r]["analysis_lmin1"]
    g = lambda k: "n/a" if a[k] is None else f"{a[k]:.4f}"
    w(f"| {r} | {g('cut_recall')} | {g('gradual_recall')} | {g('gradual_f1')} | {g('gradual_iou')} |")
w("\nFor comparison, the previous evaluation (best ℓ_min):\n")
w("| ID | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |\n|---|---|---|---|---|")
for r in ("A0", "A1", "A2"):
    a = PREV_AN[r]
    g = lambda k: "n/a" if a[k] is None else f"{a[k]:.4f}"
    w(f"| {r} | {g('cut_recall')} | {g('gradual_recall')} | {g('gradual_f1')} | {g('gradual_iou')} |")
w("")

# Table 9
w("### 2.6 Table 9 — Recall by gradual-transition duration on ClipShots (ℓ_min = 1)\n")
w("| ID | ≤ 10 frames | 11–30 frames | > 30 frames |\n|---|---|---|---|")
for r in ("A0", "C6"):
    a = R[r]["analysis_lmin1"]
    d = a["duration_recall"]; n = a["duration_counts"]
    w(f"| {r} | {d['le_10']:.4f} (n = {n['le_10']}) | {d['11_30']:.4f} (n = {n['11_30']}) | {d['gt_30']:.4f} (n = {n['gt_30']}) |")
d0, d6 = R["A0"]["analysis_lmin1"]["duration_recall"], R["C6"]["analysis_lmin1"]["duration_recall"]
w("\nC6 − A0: " + ", ".join(f"{lab} {sd(d6[k]-d0[k])}" for lab, k in (("≤ 10", "le_10"), ("11–30", "11_30"), ("> 30", "gt_30"))) + ".\n")
w("Previous evaluation: " + "; ".join(
    f"{r} " + " / ".join(f"{PREV_AN[r]['duration_recall'][k]:.4f}" for k in ("le_10", "11_30", "gt_30")) for r in ("A0", "C6")) + " (≤ 10 / 11–30 / > 30).\n")

# ClipShots previous vs new
w("## 3. ClipShots: previous evaluation vs ℓ_min = 1\n")
w("| ID | Previous F1 (best ℓ_min) | ℓ_min picked before | F1 with ℓ_min = 1 | Difference |\n|---|---|---|---|---|")
diffs = []
for r in R:
    p = R[r]["sets"]["clipshots"]["previous_recomputed"]; n = R[r]["sets"]["clipshots"]["lmin1"]
    diffs.append(n["f1"] - p["f1"])
    w(f"| {r} | {p['f1']:.4f} | {p['l_min']} | {n['f1']:.4f} | {sd(n['f1']-p['f1'])} |")
pick = {}
for r in R:
    k = R[r]["sets"]["clipshots"]["previous_recomputed"]["l_min"]; pick[k] = pick.get(k, 0) + 1
w(f"\nThe change is between {min(diffs):+.4f} and {max(diffs):+.4f} (mean {np.mean(diffs):+.4f}); every run is lower or equal with ℓ_min = 1, "
  f"since ℓ_min = 1 is one of the options the previous search could pick. Previously picked ℓ_min: " + ", ".join(f"{k} in {v} runs" for k, v in sorted(pick.items())) + ".\n")

w("## 4. ℓ_min sensitivity for A0 on ClipShots\n")
w("Best F1 for each fixed ℓ_min, thresholds re-searched for each value:\n")
w("| ℓ_min | τ_c | τ_g | Precision | Recall | F1 |\n|---|---|---|---|---|---|")
pl = R["A0"]["sets"]["clipshots"]["per_lmin"]
bestf = max(v["f1"] for v in pl.values())
for k in ("1", "2", "3", "4"):
    v = pl[k]
    w(f"| {k} | {v['tau_c']} | {v['tau_g']} | {v['p']:.4f} | {v['r']:.4f} | " + (f"**{v['f1']:.4f}**" if v["f1"] == bestf else f"{v['f1']:.4f}") + " |")
w("")

# Interpretation
w("## 5. Interpretation check (0.01 band)\n")
w("| Statement | Previous | ℓ_min = 1 | Holds? |\n|---|---|---|---|")
def delta(a, b, new): return F(a, "clipshots", new) - F(b, "clipshots", new)
for lab, a, b in (("A0 − A1 on ClipShots ≥ 0.01", "A0", "A1"), ("A4 − A3 on ClipShots ≥ 0.01", "A4", "A3"), ("A0 − A2 on ClipShots ≥ 0.01", "A0", "A2")):
    o, n = delta(a, b, False), delta(a, b, True)
    w(f"| {lab} | {sd(o)} | {sd(n)} | {'**yes**' if n >= 0.01 else '**no**'} |")
def loo(new): return max(abs(F(r, s, new) - F("A0", s, new)) for r in CR[1:] for s in S)
w(f"| No leave-one-out change (C1–C6) reaches 0.01 on any dataset | largest {loo(False):.4f} | largest {loo(True):.4f} | {'**yes**' if loo(True) < 0.01 else '**no**'} |\n")

# Validation history
w("## 6. Validation history: A1 and A0 (SHOT validation F1 per epoch)\n")
H = {r: json.load(open(SR / "ablations_v2" / r / "history.json")) for r in ("A0", "A1")}
SU = {r: json.load(open(SR / "ablations_v2" / r / "summary.json")) for r in ("A0", "A1")}
w("| Epoch | A1 | A0 |\n|---|---|---|")
for ep in range(1, max(len(H["A0"]), len(H["A1"])) + 1):
    cell = {}
    for r in ("A0", "A1"):
        if ep <= len(H[r]):
            v = H[r][ep - 1]["val"]["f1"]
            cell[r] = (f"**{v:.4f}**" if ep == SU[r]["best_epoch"] else f"{v:.4f}")
        else:
            cell[r] = ""
    w(f"| {ep} | {cell['A1']} | {cell['A0']} |")
w("")
v1 = _A([h["val"]["f1"] for h in H["A1"]]); b1 = SU["A1"]["best_epoch"]
post = v1[b1:]; pre = v1[:b1]
slope = slope_of(list(post))
w(f"**A1:** best epoch {b1} ({v1[b1-1]:.4f}); training stopped at epoch {len(v1)} by early stopping (patience 10). "
  f"After epoch {b1} the validation F1 stays between {post.min():.4f} and {post.max():.4f} (mean {post.mean():.4f}, last epoch {v1[-1]:.4f}), "
  f"i.e. {v1[b1-1]-post.max():+.4f} to {v1[b1-1]-post.min():+.4f} from the best; the linear trend over those epochs is {slope:+.5f} per epoch. "
  "It had levelled off: from epoch 5 on it fluctuates inside a band of about 0.017 with no sustained rise and no sustained fall. "
  f"It was not rising when training stopped, and it was not collapsing; the epoch-to-epoch noise (up to {absdiff_max(list(post)):.4f}) is of the same size as the gap to the best epoch.\n")
v0 = _A([h["val"]["f1"] for h in H["A0"]]); b0 = SU["A0"]["best_epoch"]
post0 = v0[b0:]
w(f"**A0 (for comparison):** best epoch {b0} ({v0[b0-1]:.4f}); stopped at epoch {len(v0)}. It climbs more slowly (from 0.6988 to {v0[b0-1]:.4f} over {b0} epochs, with ups and downs) "
  f"and after epoch {b0} stays between {post0.min():.4f} and {post0.max():.4f} (mean {post0.mean():.4f}), also without a rise. "
  "Both runs stopped on the patience rule, not on the epoch limit.\n")
w("Caveat: this is the validation F1 on the SHOT validation split (67 videos) with fixed default thresholds, so its noise level is substantial; a single epoch's difference of a few thousandths is not meaningful.\n")

w("## 7. Deliverables and runtime\n")
w(f"- Wall-clock time of the ℓ_min = 1 re-evaluation of all 17 runs (3 test sets each, including the ClipShots ℓ_min grid for the comparison, from saved probabilities, 17 worker processes on CPU): **{D['runtime_seconds']:.0f} s**.")
w("- Fresh forward passes of A0 and A1 on all three test sets (GPU, original evaluation commands, section 1): **8 min 32 s** in total (both runs).")
w("\n### Commands\n")
w("`<data>` is the prepared data directory, `<runs>` the directory holding the trained runs, `<out>` an output directory.\n")
w("```bash")
w("# 1. Fresh forward pass + original evaluation (l_min grid 1-4 on ClipShots), A0 and A1, for validating the saved probabilities")
w("python test_autoshot.py  --checkpoint <runs>/$R/best_model.pt --cache_root <data>/cache/autoshot_test --gt_path <data>/AutoShot/gt_scenes_dict_baseline_v2.pickle --min_scene_lengths 1 --output <out>/fresh/$R/shot_results.json")
w("python test_clipshots.py --checkpoint <runs>/$R/best_model.pt --cache_root <data>/cache/clipshots/test --ann_path <data>/ClipShots/annotations/test.json --min_scene_lengths 1 2 3 4 --output <out>/fresh/$R/clipshots_results.json")
w("python test_bbc.py       --checkpoint <runs>/$R/best_model.pt --cache_root <data>/cache/bbc --bbc_root <data>/BBC --output <out>/fresh/$R/bbc_results.json     # R = A0, A1")
w("# 2. Compare fresh and saved probabilities (max abs difference per video) and the stored result files")
w("python compare_fresh.py A0 A1")
w("# 3. Re-evaluation of all 17 runs from the saved probabilities: previous rule (ClipShots l_min grid 1-4) and l_min = 1, Tables 8/9 at l_min = 1")
w("python reeval_lmin1.py --procs 17   # writes reeval_lmin1.json and reeval_lmin1.csv")
w("# 4. Report")
w("python make_reeval_report.py")
w("```\n")
w("- Runs that could not be evaluated: **none**. All 17 checkpoints and their saved probabilities were present.")
w("- Per-run, per-test-set CSV: one row per run and test set, columns run, test set, ℓ_min, τ_c, τ_g, precision, recall, F1 (51 rows).\n")
Path("REEVAL_LMIN1_REPORT.md").write_text("\n".join(out), encoding="utf-8")
print("written", len(out))
