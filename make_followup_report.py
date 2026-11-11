"""Build the follow-up report (extra seeds, Protocol A with l_min = 1, Protocol B) from the saved result files.
   python make_followup_report.py [--shareable]   -> FOLLOWUP_RESULTS.md (or FOLLOWUP_RESULTS_SHAREABLE.md)"""
import csv, json, statistics as stat, sys
from pathlib import Path

SHARE = "--shareable" in sys.argv
PARTB = "--partb" in sys.argv
SR = Path("server_results")
PA = json.load(open(SR / "reeval_lmin1" / "reeval_lmin1.json"))
PB = json.load(open(SR / "protocolB" / "protocolB_results.json"))
PREV_AN = json.load(open(SR / "ablations_v2" / "clipshots_protocol_a_analysis.json"))["runs"]
FV = json.load(open(SR / "reeval_lmin1" / "fresh_vs_saved.json"))
CONC = SR / "seed_runs" / "concurrency_check.json"
CONC = json.load(open(CONC)) if CONC.exists() else None
VV = json.load(open(SR / "protocolB" / "val_videos.json")) if (SR / "protocolB" / "val_videos.json").exists() else None
out = []; w = out.append
S = ("shot", "clipshots", "bbc"); NAME = {"shot": "SHOT", "clipshots": "ClipShots", "bbc": "BBC"}
sd = lambda x: f"{x:+.4f}"
SEEDS = (42, 43, 44)
rid = lambda base, s: base if s == 42 else f"{base}_s{s}"
disp = lambda r: r.replace("_s4", "-s4")
SEED_RUNS = [r for r in PA["runs"] if "_s4" in r]
HAVE_SEEDS = all(rid(b, s) in PA["runs"] and rid(b, s) in PB["runs"] for b in ("A0", "A1") for s in SEEDS)


class Prot:
    def __init__(self, key, label, runs):
        self.key, self.label, self.runs = key, label, runs
    def m(self, run, s):
        if self.key == "A":
            d = self.runs[run]["sets"][s]["lmin1"]
        else:
            d = self.runs[run]["sets"][s]
        return d
    def f1(self, run, s): return self.m(run, s)["f1"]
    def ana(self, run): return self.runs[run]["analysis_lmin1" if self.key == "A" else "analysis"]


PROTS = [Prot("A", "Protocol A with ℓ_min = 1 (thresholds selected on each test set)", PA["runs"]),
         Prot("B", "Protocol B (thresholds selected once on the 67 SHOT validation videos, applied unchanged)", PB["runs"])]


def bolds(P, runs, s):
    v = {r: round(P.f1(r, s), 4) for r in runs}; mx = max(v.values())
    return {r: (f"**{v[r]:.4f}**" if v[r] == mx else f"{v[r]:.4f}") for r in runs}


def table_f1(P, runs, desc=None):
    b = {s: bolds(P, runs, s) for s in S}
    w("| ID | " + ("Description | " if desc else "") + "SHOT F1 | ClipShots F1 | BBC F1 |"); w("|---|" + ("---|" if desc else "") + "---|---|---|")
    for r in runs:
        w(f"| {disp(r)} | " + (f"{desc[r]} | " if desc else "") + " | ".join(b[s][r] for s in S) + " |")
    w("")


CR = ["A0", "C1", "C2a", "C2b", "C3", "C4", "C5a", "C5b", "C6"]
DESC7A = {"A0": "Full, dual-label (primary)", "A1": "Full, midpoint only", "A2": "Full, unified single head", "A3": "Reduced, midpoint only", "A4": "Reduced, dual-label"}
DESC7B = {"A0": "Full model", "C1": "w/o difference-based branch (B)", "C2a": "w/o FrameSimilarity", "C2b": "w/o ColorHistogramSimilarity", "C3": "w/o temporal delta",
          "C4": "w/o per-frame pathway (relations from Branch A)", "C5a": "w/o pixel-level motion descriptor", "C5b": "w/o edge-density head", "C6": "w/o self-attention and gate"}
LAD = [("0", "M0_min", "RGB encoder + MLP + dual head only"), ("1", "M1_edge", "+ edge-density head"), ("2", "B_noatt", "+ 4 fusion blocks, no attention"),
       ("3", "A4", "+ attention and gate (reduced model)"), ("4", "S1_rel", "+ FrameSimilarity, ColorHist, delta"), ("5", "A0", "+ diff branch, motion, per-frame pathway (full model)")]
BASE = [r for r in PA["runs"] if "_s4" not in r]


def seeds_stats(P, base, s):
    xs = [P.f1(rid(base, k), s) for k in SEEDS]
    return xs, stat.fmean(xs), stat.stdev(xs)


def protocol_section(P, num, first=1, head=True):
    if head: w(f"## {num}. {P.label}\n")
    if head and P.key == "B":
        w("Each model's τ_c and τ_g were selected once, by maximising F1 on the 67 SHOT validation videos evaluated as full videos (same windows, averaging and scoring as the test sets, ℓ_min = 1), "
          "and then applied unchanged to SHOT, ClipShots and BBC. Nothing was tuned on a test set. Validation F1 and the selected thresholds are in the thresholds table.\n")
    elif head:
        w("SHOT and BBC already used ℓ_min = 1, so their F1 is identical to the main study (checked for all 17 seed-42 models: F1, thresholds, TP / FP / FN). Only ClipShots changes.\n")
    n = num
    w(f"### {n}.{first+0} Supervision (Table 7a, seed 42)\n"); table_f1(P, ["A0", "A1", "A2", "A3", "A4"], DESC7A)
    w("Differences:\n\n| Delta | SHOT | ClipShots | BBC |\n|---|---|---|---|")
    for lab, a, b in (("A0 − A1 (full)", "A0", "A1"), ("A4 − A3 (reduced)", "A4", "A3"), ("A0 − A2 (dual vs unified)", "A0", "A2")):
        w(f"| {lab} | " + " | ".join(sd(P.f1(a, s) - P.f1(b, s)) for s in S) + " |")
    w("")
    w(f"### {n}.{first+1} Components, leave-one-out (Table 7b)\n"); table_f1(P, CR, DESC7B)
    w("Change in F1 relative to A0:\n\n| ID | ΔSHOT | ΔClipShots | ΔBBC |\n|---|---|---|---|")
    mx = 0
    for r in CR[1:]:
        ds = [P.f1(r, s) - P.f1("A0", s) for s in S]; mx = max(mx, max(abs(x) for x in ds)); w(f"| {r} | " + " | ".join(sd(x) for x in ds) + " |")
    w(f"\nLargest single-component change on any dataset: {mx:.4f}.\n")
    w(f"### {n}.{first+2} Build-up ladder\n")
    runs = [x[1] for x in LAD]; b = {s: bolds(P, runs, s) for s in S}
    w("| Step | Run | What is added | SHOT F1 | ClipShots F1 | BBC F1 |\n|---|---|---|---|---|---|")
    for st, r, d in LAD: w(f"| {st} | {r} | {d} | " + " | ".join(b[s][r] for s in S) + " |")
    w("\nChange at each step:\n\n| Step | SHOT | ClipShots | BBC |\n|---|---|---|---|")
    for (s0, r0, _), (s1, r1, _) in zip(LAD[:-1], LAD[1:]): w(f"| {r0} → {r1} | " + " | ".join(sd(P.f1(r1, s) - P.f1(r0, s)) for s in S) + " |")
    w("")
    w(f"### {n}.{first+3} Precision, recall and selected thresholds\n")
    allr = BASE + SEED_RUNS if HAVE_SEEDS else BASE
    if P.key == "A":
        w("| ID | SHOT P / R | ClipShots P / R | BBC P / R | SHOT τ_c, τ_g | ClipShots τ_c, τ_g | BBC τ_c, τ_g |\n|---|---|---|---|---|---|---|")
        for r in allr:
            c = [P.m(r, s) for s in S]
            w(f"| {disp(r)} | " + " | ".join(f"{x['p']:.3f} / {x['r']:.3f}" for x in c) + " | " + " | ".join(f"{x['tau_c']}, {x['tau_g'] if x['tau_g'] is not None else '–'}" for x in c) + " |")
    else:
        w("| ID | τ_c, τ_g (from validation) | Validation F1 | SHOT P / R | ClipShots P / R | BBC P / R |\n|---|---|---|---|---|---|")
        for r in allr:
            v = PB["runs"][r]["val"]; c = [P.m(r, s) for s in S]
            w(f"| {disp(r)} | {v['tau_c']}, {v['tau_g'] if v['tau_g'] is not None else '–'} | {v['f1']:.4f} | " + " | ".join(f"{x['p']:.3f} / {x['r']:.3f}" for x in c) + " |")
    w("\nA1, A2, A3 and their seeds use τ_c only.\n")
    w(f"### {n}.{first+4} Table 8 — Per-type analysis on ClipShots\n")
    w("| ID | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |\n|---|---|---|---|---|")
    for r in ("A0", "A1", "A2"):
        a = P.ana(r); g = lambda k: "n/a" if a[k] is None else f"{a[k]:.4f}"
        w(f"| {r} | {g('cut_recall')} | {g('gradual_recall')} | {g('gradual_f1')} | {g('gradual_iou')} |")
    w(f"\n### {n}.{first+5} Table 9 — Recall by gradual-transition duration on ClipShots\n")
    w("| ID | ≤ 10 frames | 11–30 frames | > 30 frames |\n|---|---|---|---|")
    for r in ("A0", "C6"):
        a = P.ana(r); d, c = a["duration_recall"], a["duration_counts"]
        w(f"| {r} | {d['le_10']:.4f} (n = {c['le_10']}) | {d['11_30']:.4f} (n = {c['11_30']}) | {d['gt_30']:.4f} (n = {c['gt_30']}) |")
    d0, d6 = P.ana("A0")["duration_recall"], P.ana("C6")["duration_recall"]
    w("\nC6 − A0: " + ", ".join(f"{lab} {sd(d6[k]-d0[k])}" for lab, k in (("≤ 10", "le_10"), ("11–30", "11_30"), ("> 30", "gt_30"))) + ".\n")


def seeds_section(P, num, hdr=None):
    w(hdr or f"## {num}. Seeds — {P.label.split(' (')[0]}\n")
    if not HAVE_SEEDS:
        w("The additional-seed runs are not finished yet; this section is filled in when they are.\n"); return
    w("F1 of A0 and A1 at seeds 42, 43 and 44 (seed 42 = the main-study run). Mean and sample standard deviation (n − 1) over the three seeds.\n")
    w("| Model | Test set | Seed 42 | Seed 43 | Seed 44 | Mean | SD |\n|---|---|---|---|---|---|---|")
    for base in ("A0", "A1"):
        for s in S:
            xs, m, sdv = seeds_stats(P, base, s)
            w(f"| {base} | {NAME[s]} | " + " | ".join(f"{x:.4f}" for x in xs) + f" | {m:.4f} | {sdv:.4f} |")
    w("\n**Paired difference A0 − A1** (same seed for both):\n")
    w("| Test set | Seed 42 | Seed 43 | Seed 44 | Mean | SD |\n|---|---|---|---|---|---|")
    for s in S:
        ds = [P.f1(rid("A0", k), s) - P.f1(rid("A1", k), s) for k in SEEDS]
        w(f"| {NAME[s]} | " + " | ".join(sd(x) for x in ds) + f" | {sd(stat.fmean(ds))} | {stat.stdev(ds):.4f} |")
    w("\n**Cut recall and gradual recall on ClipShots:**\n")
    w("| Model | Metric | Seed 42 | Seed 43 | Seed 44 | Mean | SD |\n|---|---|---|---|---|---|---|")
    for base in ("A0", "A1"):
        for key, lab in (("cut_recall", "Cut recall"), ("gradual_recall", "Gradual recall")):
            xs = [P.ana(rid(base, k))[key] for k in SEEDS]
            w(f"| {base} | {lab} | " + " | ".join(f"{x:.4f}" for x in xs) + f" | {stat.fmean(xs):.4f} | {stat.stdev(xs):.4f} |")
    w("")


def interp(P):
    rows = []
    d = lambda a, b, s: P.f1(a, s) - P.f1(b, s)
    ok = lambda c: "**yes**" if c else "**no**"
    if HAVE_SEEDS:
        ds = [d(rid("A0", k), rid("A1", k), "clipshots") for k in SEEDS]
        rows.append(("A0 − A1 on ClipShots ≥ 0.01, seed 42", sd(ds[0]), ok(ds[0] >= 0.01)))
        rows.append(("A0 − A1 on ClipShots ≥ 0.01, mean over the three seeds", sd(stat.fmean(ds)), ok(stat.fmean(ds) >= 0.01)))
        rows.append(("A0 − A1 on ClipShots positive for every seed", " / ".join(sd(x) for x in ds), ok(all(x > 0 for x in ds))))
    else:
        x = d("A0", "A1", "clipshots"); rows.append(("A0 − A1 on ClipShots ≥ 0.01", sd(x), ok(x >= 0.01)))
        if not PARTB: rows.append(("A0 − A1 on ClipShots ≥ 0.01 (three-seed mean) and positive for every seed", "pending", "pending"))
    x = d("A4", "A3", "clipshots"); rows.append(("A4 − A3 on ClipShots ≥ 0.01", sd(x), ok(x >= 0.01)))
    x, y = d("A0", "A2", "clipshots"), d("A0", "A2", "bbc")
    rows.append(("A0 − A2 on ClipShots ≥ 0.01", sd(x), ok(x >= 0.01))); rows.append(("A0 − A2 on BBC ≥ 0.01", sd(y), ok(y >= 0.01)))
    mx = max(abs(P.f1(r, s) - P.f1("A0", s)) for r in CR[1:] for s in S)
    rows.append(("No leave-one-out change (C1–C6) reaches 0.01 on any test set", f"largest {mx:.4f}", ok(mx < 0.01)))
    w("| Statement | Value | Holds? |\n|---|---|---|")
    for r in rows: w(f"| {r[0]} | {r[1]} | {r[2]} |")
    w("")


def prev_new_section(num, h1=None, h2=None):
    P = PROTS[0]
    w(h1 or f"## {num}. ClipShots: previous evaluation vs ℓ_min = 1 (Protocol A)\n")
    w("| ID | Previous F1 (best ℓ_min) | ℓ_min picked before | F1 with ℓ_min = 1 | Difference |\n|---|---|---|---|---|")
    for r in BASE:
        p = PA["runs"][r]["sets"]["clipshots"]["previous_recomputed"]; n = P.m(r, "clipshots")
        w(f"| {r} | {p['f1']:.4f} | {p['l_min']} | {n['f1']:.4f} | {sd(n['f1']-p['f1'])} |")
    w("")
    w(h2 or f"## {num+1}. ℓ_min sensitivity for A0 on ClipShots (Protocol A)\n")
    w("Best F1 for each fixed ℓ_min, thresholds re-searched for each value:\n")
    w("| ℓ_min | τ_c | τ_g | Precision | Recall | F1 |\n|---|---|---|---|---|---|")
    pl = PA["runs"]["A0"]["sets"]["clipshots"]["per_lmin"]; best = max(v["f1"] for v in pl.values())
    for k in ("1", "2", "3", "4"):
        v = pl[k]; w(f"| {k} | {v['tau_c']} | {v['tau_g']} | {v['p']:.4f} | {v['r']:.4f} | " + (f"**{v['f1']:.4f}**" if v["f1"] == best else f"{v['f1']:.4f}") + " |")
    w("")


def _hist_data():
    cols = [("A1", SR / "ablations_v2" / "A1"), ("A0", SR / "ablations_v2" / "A0")]
    for b in ("A1", "A0"):
        for k in (43, 44):
            cols.append((f"{b}-s{k}", SR / "seed_runs" / f"{b}_s{k}" / b))
    H = {}
    for name, p in cols:
        if (p / "history.json").exists() and (p / "summary.json").exists():
            H[name] = (json.load(open(p / "history.json")), json.load(open(p / "summary.json")))
    return [c[0] for c in cols if c[0] in H], H


def hist_table(hdr):
    names, H = _hist_data()
    w(hdr)
    w("SHOT validation F1 after every epoch; best epoch in bold. Seed 42 runs are A1 and A0.\n")
    w("| Epoch | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for ep in range(1, max(len(H[n][0]) for n in names) + 1):
        cells = []
        for n in names:
            h, su = H[n]
            cells.append("" if ep > len(h) else (f"**{h[ep-1]['val']['f1']:.4f}**" if ep == su["best_epoch"] else f"{h[ep-1]['val']['f1']:.4f}"))
        w(f"| {ep} | " + " | ".join(cells) + " |")
    w("")


def hist_trends(hdr):
    names, H = _hist_data()
    w(hdr)
    w("| Run | Best epoch | Best val F1 | Stopped at epoch | Range after best | Last | Trend per epoch |\n|---|---|---|---|---|---|---|")
    for n in names:
        h, su = H[n]; v = [x["val"]["f1"] for x in h]; b = su["best_epoch"]; post = v[b:]
        n_ = len(post); xs = list(range(n_)); mx_, my_ = stat.fmean(xs), stat.fmean(post)
        slope = sum((a - mx_) * (c - my_) for a, c in zip(xs, post)) / sum((a - mx_) ** 2 for a in xs) if n_ > 1 else 0.0
        w(f"| {n} | {b} | {v[b-1]:.4f} | {len(v)} | {min(post):.4f}–{max(post):.4f} | {v[-1]:.4f} | {slope:+.5f} |")
    w("")



if PARTB:
    w("# Protocol B results: thresholds selected on validation\n")
    w("This report gives the Protocol B evaluation of the trained HybridSBD ablation models (leak-free SHOT training set of 448 videos). "
      "In Protocol B the decision thresholds are **not** tuned on the test sets: for each model, τ_c and τ_g are selected once, by maximising F1 on the 67 SHOT validation videos, "
      "and then applied unchanged to the SHOT test set (200 videos), the ClipShots test set (500 videos) and BBC (11 episodes). The minimum shot length is ℓ_min = 1 (no merging) everywhere. "
      "Everything else is as in the main study: existing best checkpoints (EMA weights selected by SHOT validation F1), sliding windows (T = 128, stride 64) averaged back to a per-video track, "
      "threshold grids τ_c ∈ {0.10, 0.15, …, 0.90} and τ_g ∈ {0.3, 0.4, 0.5, 0.6}, the decision rule (dual-label models: boundary if p_y > τ_c, gradual if p_z > τ_g, track = gradual OR cut; "
      "midpoint-only and unified models: τ_c only) and scoring with ±2-frame scene matching, with precision, recall and F1 pooled over all videos. "
      "Numbers are expected to be lower than with per-test-set tuning (Protocol A), especially on ClipShots and BBC. Single training seed (42) per model.\n")
    w("## 1. Validation data and threshold selection\n")
    w("The SHOT validation split (15 % of the 448 training videos, 67 videos) was previously available only as pre-cut training clips. For Protocol B these 67 videos were decoded completely, "
      "with the same decoder, resizing (48 × 64), window layout and window averaging as the SHOT test set (1,270 windows). "
      + (f"Ground truth comes from the SHOT label file with the same transition-to-scene conversion as ClipShots; {VV['transitions_used']} of {VV['transitions_total']} transitions lie within the decoded video length and are used. " if VV else "")
      + "The label file is the only ground truth available for these videos and is not the authors' final annotation, so validation F1 is noisier than test F1, and the thresholds it selects are noisier too. "
      "The validation videos were also used for early stopping and checkpoint selection, so validation F1 is slightly optimistic.\n")
    w("Validation-selected thresholds and validation F1 are listed in section 2.4.\n")
    protocol_section(PROTS[1], 2)
    w("## 3. Protocol B compared with Protocol A (ℓ_min = 1)\n")
    w("Protocol A selects thresholds on each test set (optimistic); Protocol B does not. F1 under each protocol and the difference:\n")
    w("| ID | SHOT A | SHOT B | Δ | ClipShots A | ClipShots B | Δ | BBC A | BBC B | Δ |\n|---|---|---|---|---|---|---|---|---|---|")
    drops = {s: [] for s in S}
    for r in BASE:
        cells = []
        for sset in S:
            a, b = PROTS[0].f1(r, sset), PROTS[1].f1(r, sset); drops[sset].append(b - a)
            cells += [f"{a:.4f}", f"{b:.4f}", sd(b - a)]
        w(f"| {r} | " + " | ".join(cells) + " |")
    w("\nMean change over the 17 models: " + ", ".join(f"{NAME[x]} {sd(stat.fmean(drops[x]))} (range {sd(min(drops[x]))} to {sd(max(drops[x]))})" for x in S) + ". "
      "No model scores higher under Protocol B than under Protocol A on any test set, as expected, since Protocol A picks the best thresholds on the test set itself.\n")
    w("## 4. Interpretation check (0.01 band)\n")
    interp(PROTS[1])
    w("Differences below 0.01 F1 are treated as inconclusive. Under Protocol B the thresholds themselves add noise: a model's validation-selected τ_c / τ_g can sit far from its best test thresholds, "
      "so single-model differences (for example in the leave-one-out table) are less reliable than under Protocol A.\n")
    w("## 5. SHOT validation F1 per epoch (training histories)\n")
    cols = [("A1", SR / "ablations_v2" / "A1"), ("A0", SR / "ablations_v2" / "A0")]
    H = {n: (json.load(open(pth / "history.json")), json.load(open(pth / "summary.json"))) for n, pth in cols}
    w("| Epoch | A1 | A0 |\n|---|---|---|")
    for ep in range(1, max(len(H[n][0]) for n in H) + 1):
        c = []
        for n in ("A1", "A0"):
            h, su = H[n]; c.append("" if ep > len(h) else (f"**{h[ep-1]['val']['f1']:.4f}**" if ep == su["best_epoch"] else f"{h[ep-1]['val']['f1']:.4f}"))
        w(f"| {ep} | " + " | ".join(c) + " |")
    w("\nBest epoch in bold. This is the validation F1 used for early stopping (pre-cut validation clips, fixed thresholds), not the full-video validation F1 of section 1.\n")
    Path("Protocol_B_Results.md").write_text("\n".join(out), encoding="utf-8")
    print("written Protocol_B_Results.md"); sys.exit(0)


def run_summary(name):
    if "_s" in name:
        base, k = name.split("_s"); pth = SR / "seed_runs" / name / base
    else:
        pth = SR / "ablations_v2" / name
    return json.load(open(pth / "summary.json"))


w("# Follow-up experiments: extra seeds, ℓ_min = 1 and validation-selected thresholds\n")
w("This report covers the three parts of the follow-up to the Protocol-A ablation study of the HybridSBD shot boundary detection models (leak-free SHOT training set of 448 videos): "
  "**Part 1** trains A0 and A1 with two more seeds; **Part 2** re-evaluates every model with ℓ_min = 1 on all three test sets (Protocol A with ℓ_min = 1); **Part 3** evaluates every model with "
  "thresholds selected on the SHOT validation split (Protocol B). A short additional check (validation F1 per epoch) follows. "
  "Evaluation uses the existing best checkpoints (EMA weights selected by SHOT validation F1). Windows (T = 128, stride 64), window averaging, threshold grids (τ_c 0.10–0.90 step 0.05, "
  "τ_g ∈ {0.3, 0.4, 0.5, 0.6}), the decision rule and ±2-frame scene matching are unchanged. Seed 42 is the main-study run.\n")
w("| Part | Content |\n|---|---|\n| Part 1 | Extra seeds for A0 and A1 (training) |\n| Part 2 | Protocol A with ℓ_min = 1 on all test sets (evaluation only) |\n"
  "| Part 3 | Protocol B, validation-selected thresholds (evaluation only) |\n| Part 4 | Additional check: validation F1 for every epoch |\n")

w("---\n\n# Part 1: Extra seeds for A0 and A1 (training)\n")
w("A0 (full model, dual-label) and A1 (full model, midpoint only) were trained with seeds 43 and 44, giving four new runs (A0-s43, A0-s44, A1-s43, A1-s44). Recipe, augmentation, synthetic transitions, "
  "EMA (0.999), early stopping (patience 10 on SHOT validation F1 with the variant's own decision rule) and the 80-epoch maximum are identical to the seed-42 runs. "
  "Only the training seed changes: model initialisation, training-data order, the ClipShots clip drawn each epoch, augmentation and synthetic transitions.\n")
w("## 1.1 Train/validation split is identical for all seeds\n")
w("In the training code the SHOT and ClipShots train/validation split was shuffled with the training seed, so seeds 43 and 44 would have used different validation videos. "
  "I fixed the split to the seed-42 split for every training seed (the option defaults to 42, so the seed-42 runs are unchanged). SHA-256 of the sorted SHOT validation video names:\n")
w("| Training seed | Old rule (split follows the seed) | Fixed split (used) |\n|---|---|---|\n| 42 | f0070e4a… | f0070e4a… |\n| 43 | 88ddfd45… | f0070e4a… |\n| 44 | 792acaf5… | f0070e4a… |\n")
w("All four new runs printed the same checksums before training: SHOT validation 67 videos (f0070e4a…), SHOT train 381 videos (ed9cb6ec…), ClipShots train 3,108 videos (f623f785…). "
  "The pre-extracted training clips use the stable per-video MD5 seed and are unchanged.\n")
w("## 1.2 The four new runs\n")
w("| Run | Seed | Best epoch | Best SHOT val F1 | Epochs run | Stopped early | Training hours |\n|---|---|---|---|---|---|---|")
for nm in ("A0", "A0_s43", "A0_s44", "A1", "A1_s43", "A1_s44"):
    su = run_summary(nm); sdn = 42 if "_s" not in nm else int(nm.split("_s")[1])
    w(f"| {disp(nm)} | {sdn} | {su['best_epoch']} | {su['best_val_f1']:.4f} | {su['epochs_completed']} | {su['stopped_early']} | {su['train_hours']:.2f} |")
w("\nSeed-42 rows are the main-study runs, shown for reference.\n")
hist_table("## 1.3 SHOT validation F1 history of every epoch\n")
w("## 1.4 Does training several runs at once on one GPU change the results?\n")
if CONC: w(CONC["text"] + "\n")

w("---\n\n# Part 2: Protocol A with ℓ_min = 1 (evaluation only)\n")
w("ℓ_min is fixed to 1 (no minimum-shot-length merging) on SHOT, ClipShots and BBC and is not searched. Thresholds are still selected on each test set (Protocol A).\n")
w("## 2.1 Checks\n")
a0 = PA["runs"]["A0"]["sets"]
w("**Previous evaluation reproduced for A0** (ClipShots with the ℓ_min grid; SHOT and BBC with ℓ_min = 1):\n")
w("| Test set | Expected F1 | Reproduced F1 | Expected thresholds | Reproduced thresholds |\n|---|---|---|---|---|")
exp = {"shot": ("0.8449", "τ_c 0.4, τ_g 0.3"), "clipshots": ("0.8052", "τ_c 0.65, τ_g 0.6, ℓ_min 4"), "bbc": ("0.9719", "τ_c 0.45, τ_g 0.6")}
for st_ in S:
    v = a0[st_]["previous_recomputed"]; th = f"τ_c {v['tau_c']}, τ_g {v['tau_g']}" + (f", ℓ_min {v['l_min']}" if st_ == "clipshots" else "")
    w(f"| {NAME[st_]} | {exp[st_][0]} | {v['f1']:.4f} | {exp[st_][1]} | {th} |")
w("\nAll 17 seed-42 models reproduce their stored results exactly (51 of 51 model-test-set results). For SHOT and BBC the ℓ_min = 1 results are identical to the main study for all 17 models (F1, thresholds, TP / FP / FN); only ClipShots changes.\n")
w("**Saved probabilities vs a fresh forward pass** (A0 and A1, all three test sets): the maximum absolute difference of p_y and p_z over all videos is 0.0, and F1, thresholds and TP / FP / FN are identical. "
  "The results in Parts 2 and 3 therefore use the saved probabilities.\n")
w("## Results\n")
protocol_section(PROTS[0], "2", first=2, head=False)
seeds_section(PROTS[0], "2.8", hdr="### 2.8 Seeds table\n")
prev_new_section("2", h1="### 2.9 ClipShots comparison, previous vs new\n", h2="### 2.10 ℓ_min sensitivity for A0 on ClipShots\n")
w("### 2.11 Interpretation check (0.01 band)\n"); interp(PROTS[0])

w("---\n\n# Part 3: Protocol B, validation-selected thresholds (evaluation only)\n")
w("For each model τ_c and τ_g are selected once on the SHOT validation videos and applied unchanged to SHOT, ClipShots and BBC (ℓ_min = 1). Nothing is tuned on a test set. "
  "Midpoint-only (A1, A3) and unified (A2) models select τ_c only.\n")
w("## 3.1 Validation videos as full videos (prerequisite)\n")
if VV:
    w("Before this work the 67 SHOT validation videos existed only as pre-cut training clips, so I **added a data path**: the 67 videos are now decoded completely with the same decoder, resizing (48 × 64), "
      "window layout (T = 128, stride 64) and window averaging as the SHOT test set (1,270 windows), and scored with the same scene matching. Ground truth comes from the SHOT label file with the same "
      f"transition-to-scene conversion as ClipShots; {VV['transitions_used']} of {VV['transitions_total']} transitions lie within the decoded video length and are used. "
      "The label file is the only ground truth available for these videos and is not the authors' final annotation, so validation F1 is noisier than test F1 and the selected thresholds are noisier too.\n")
w("## 3.2 Threshold selection and application\n")
w("τ_c ∈ {0.10, …, 0.90} and τ_g ∈ {0.3, 0.4, 0.5, 0.6} were searched once per model by maximising F1 on the full validation videos with ℓ_min = 1. "
  "The selected values and the validation F1 are in the thresholds table (3.6). They were applied unchanged to the three test sets for every model, including the new seeds. "
  "Numbers are expected to be lower than under Protocol A, especially on ClipShots and BBC.\n")
w("## Results\n")
protocol_section(PROTS[1], "3", first=3, head=False)
seeds_section(PROTS[1], "3.9", hdr="### 3.9 Seeds table\n")
w("### 3.10 Interpretation check (0.01 band)\n"); interp(PROTS[1])
w("Differences below 0.01 F1 are treated as inconclusive. Under Protocol B the thresholds themselves add noise, so single-model differences are less reliable than under Protocol A.\n")

w("---\n\n# Part 4: Additional check — SHOT validation F1 for every epoch\n")
w("The validation F1 of every epoch for A0 and A1 at seed 42 and for the four new runs is in Table 1.3. What happens after each run's best epoch:\n")
hist_trends("## 4.1 Behaviour after the best epoch\n")
w("## 4.2 Did validation F1 level off or fall after the best epoch?\n")
w("In every one of the six runs, including A1 at seed 42, early stopping ended training after 10 epochs without improvement, and validation F1 had **levelled off** rather than still rising or collapsing: "
  "after its best epoch each run moves up and down inside a band about 0.01–0.02 wide just below its best value, with a linear trend close to zero (between −0.0007 and +0.0004 F1 per epoch), "
  "and no run was still climbing when it stopped. A1 at seed 42 (best epoch 7, 0.7863) fluctuated between 0.7695 and 0.7852 afterwards and ended at 0.7846, so it was levelled off and unstable at the 0.01 level, "
  "not falling steadily. The new A1 runs behave the same way (A1-s43 drifts slightly down, −0.0005 per epoch; A1-s44 is flat, −0.0002). "
  "The epoch-to-epoch swings are about as large as the gaps between runs, so the best-epoch choice itself is noisy.\n")

Path("FOLLOWUP_RESULTS_SHAREABLE.md" if SHARE else "FOLLOWUP_RESULTS.md").write_text("\n".join(out), encoding="utf-8")
print("written", len(out))
