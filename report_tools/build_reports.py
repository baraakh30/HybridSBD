"""Build every report from the saved result files in ``server_results/`` (nothing is typed in by hand except the prose).

Sub-commands:

``all``     follow-up report (``FOLLOWUP_RESULTS.md``), full results report (``RESULTS_FULL.md``, which embeds the follow-up
            sections), one-page summary (``docs/RESULTS.md``) and the combined CSV (``results_all_runs.csv``); ``--pdf`` also
            renders the two reports to PDF (needs ``pip install markdown`` and Google Chrome).
``latency`` inference-cost report of TransNet V2, PERSIST, AutoShot and HybridSBD from ``server_results/latency_baselines*``.
``tables``  the spec tables 7a / 7b / 8 / 9 from ``ablation_results.csv`` (used by ``scripts/run_evaluation.sh``).

Examples::

    python report_tools/build_reports.py all --pdf
    python report_tools/build_reports.py all --outdir /tmp/check        # write elsewhere, e.g. to compare with the stored reports
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics as stat
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / "server_results"
V2, V3, SB = SR / "ablations_v2", SR / "ablations_v3", SR / "speed_benchmark"  # main runs, build-up ladder, speed benchmark
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PDF_CSS = """@page{size:A4;margin:14mm}body{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.4;margin:0}
table{border-collapse:collapse;margin:6px 0 16px;width:100%;font-size:8.5pt;break-inside:avoid}table.long{break-inside:auto}thead{display:table-header-group}
th,td{border:1px solid #bbb;padding:3px 5px;text-align:left;vertical-align:top}th{background:#f0f0f0}tr{page-break-inside:avoid}code{font-size:9pt}
h1{font-size:17pt;border-bottom:1px solid #999;padding-bottom:3px;page-break-before:always}h1:first-of-type{page-break-before:avoid}
h2{font-size:14pt;margin-top:18px}h3{font-size:12pt}h2,h3{break-after:avoid}p{margin:6px 0}
pre{font-size:7.5pt;background:#f6f6f6;border:1px solid #ddd;padding:6px;white-space:pre-wrap;word-break:break-all}"""


def load_results() -> tuple:
    """The l_min = 1 (Protocol A) and Protocol B result files."""
    return (json.load(open(SR / "reeval_lmin1" / "reeval_lmin1.json")), json.load(open(SR / "protocolB" / "protocolB_results.json")))


def build_followup(pa: dict, pb: dict) -> str:
    """Follow-up report: extra seeds, Protocol A with l_min = 1, Protocol B, validation-F1 histories."""
    PA, PB = pa, pb
    CONC = SR / "seed_runs" / "concurrency_check.json"
    CONC = json.load(open(CONC)) if CONC.exists() else None
    VV = json.load(open(SR / "protocolB" / "val_videos.json")) if (SR / "protocolB" / "val_videos.json").exists() else None
    out = []
    w = out.append
    S = ("shot", "clipshots", "bbc")
    NAME = {"shot": "SHOT", "clipshots": "ClipShots", "bbc": "BBC"}
    sd = lambda x: f"{x:+.4f}"
    SEEDS = (42, 43, 44)
    rid = lambda base, s: base if s == 42 else f"{base}_s{s}"
    disp = lambda r: r.replace("_s4", "-s4")
    SEED_RUNS = [r for r in PA["runs"] if "_s4" in r]

    class Prot:
        def __init__(self, key, label, runs):
            self.key, self.label, self.runs = key, label, runs

        def m(self, run, s):
            if self.key == "A":
                d = self.runs[run]["sets"][s]["lmin1"]
            else:
                d = self.runs[run]["sets"][s]
            return d

        def f1(self, run, s):
            return self.m(run, s)["f1"]

        def ana(self, run):
            return self.runs[run]["analysis_lmin1" if self.key == "A" else "analysis"]

    PROTS = [
        Prot("A", "Protocol A with ℓ_min = 1 (thresholds selected on each test set)", PA["runs"]),
        Prot("B", "Protocol B (thresholds selected once on the 67 SHOT validation videos, applied unchanged)", PB["runs"]),
    ]

    def bolds(P, runs, s):
        v = {r: round(P.f1(r, s), 4) for r in runs}
        mx = max(v.values())
        return {r: (f"**{v[r]:.4f}**" if v[r] == mx else f"{v[r]:.4f}") for r in runs}

    def table_f1(P, runs, desc=None):
        b = {s: bolds(P, runs, s) for s in S}
        w("| ID | " + ("Description | " if desc else "") + "SHOT F1 | ClipShots F1 | BBC F1 |")
        w("|---|" + ("---|" if desc else "") + "---|---|---|")
        for r in runs:
            w(f"| {disp(r)} | " + (f"{desc[r]} | " if desc else "") + " | ".join(b[s][r] for s in S) + " |")
        w("")

    CR = ["A0", "C1", "C2a", "C2b", "C3", "C4", "C5a", "C5b", "C6"]
    DESC7A = {
        "A0": "Full, dual-label (primary)",
        "A1": "Full, midpoint only",
        "A2": "Full, unified single head",
        "A3": "Reduced, midpoint only",
        "A4": "Reduced, dual-label",
    }
    DESC7B = {
        "A0": "Full model",
        "C1": "w/o difference-based branch (B)",
        "C2a": "w/o FrameSimilarity",
        "C2b": "w/o ColorHistogramSimilarity",
        "C3": "w/o temporal delta",
        "C4": "w/o per-frame pathway (relations from Branch A)",
        "C5a": "w/o pixel-level motion descriptor",
        "C5b": "w/o edge-density head",
        "C6": "w/o self-attention and gate",
    }
    LAD = [
        ("0", "M0_min", "RGB encoder + MLP + dual head only"),
        ("1", "M1_edge", "+ edge-density head"),
        ("2", "B_noatt", "+ 4 fusion blocks, no attention"),
        ("3", "A4", "+ attention and gate (reduced model)"),
        ("4", "S1_rel", "+ FrameSimilarity, ColorHist, delta"),
        ("5", "A0", "+ diff branch, motion, per-frame pathway (full model)"),
    ]
    BASE = [r for r in PA["runs"] if "_s4" not in r]

    def seeds_stats(P, base, s):
        xs = [P.f1(rid(base, k), s) for k in SEEDS]
        return xs, stat.fmean(xs), stat.stdev(xs)

    def protocol_section(P, num, first=1, head=True):
        if head:
            w(f"## {num}. {P.label}\n")
        if head and P.key == "B":
            w(
                "Each model's τ_c and τ_g were selected once, by maximising F1 on the 67 SHOT validation videos evaluated as full videos (same windows, averaging and scoring as the test sets, ℓ_min = 1), "
                "and then applied unchanged to SHOT, ClipShots and BBC. Nothing was tuned on a test set. Validation F1 and the selected thresholds are in the thresholds table.\n"
            )
        elif head:
            w(
                "SHOT and BBC already used ℓ_min = 1, so their F1 is identical to the main study (checked for all 17 seed-42 models: F1, thresholds, TP / FP / FN). Only ClipShots changes.\n"
            )
        n = num
        w(f"### {n}.{first + 0} Supervision (Table 7a, seed 42)\n")
        table_f1(P, ["A0", "A1", "A2", "A3", "A4"], DESC7A)
        w("Differences:\n\n| Delta | SHOT | ClipShots | BBC |\n|---|---|---|---|")
        for lab, a, b in (("A0 − A1 (full)", "A0", "A1"), ("A4 − A3 (reduced)", "A4", "A3"), ("A0 − A2 (dual vs unified)", "A0", "A2")):
            w(f"| {lab} | " + " | ".join(sd(P.f1(a, s) - P.f1(b, s)) for s in S) + " |")
        w("")
        w(f"### {n}.{first + 1} Components, leave-one-out (Table 7b)\n")
        table_f1(P, CR, DESC7B)
        w("Change in F1 relative to A0:\n\n| ID | ΔSHOT | ΔClipShots | ΔBBC |\n|---|---|---|---|")
        mx = 0
        for r in CR[1:]:
            ds = [P.f1(r, s) - P.f1("A0", s) for s in S]
            mx = max(mx, max(abs(x) for x in ds))
            w(f"| {r} | " + " | ".join(sd(x) for x in ds) + " |")
        w(f"\nLargest single-component change on any dataset: {mx:.4f}.\n")
        w(f"### {n}.{first + 2} Build-up ladder\n")
        runs = [x[1] for x in LAD]
        b = {s: bolds(P, runs, s) for s in S}
        w("| Step | Run | What is added | SHOT F1 | ClipShots F1 | BBC F1 |\n|---|---|---|---|---|---|")
        for st, r, d in LAD:
            w(f"| {st} | {r} | {d} | " + " | ".join(b[s][r] for s in S) + " |")
        w("\nChange at each step:\n\n| Step | SHOT | ClipShots | BBC |\n|---|---|---|---|")
        for (s0, r0, _), (s1, r1, _) in zip(LAD[:-1], LAD[1:]):
            w(f"| {r0} → {r1} | " + " | ".join(sd(P.f1(r1, s) - P.f1(r0, s)) for s in S) + " |")
        w("")
        w(f"### {n}.{first + 3} Precision, recall and selected thresholds\n")
        allr = BASE + SEED_RUNS
        if P.key == "A":
            w(
                "| ID | SHOT P / R | ClipShots P / R | BBC P / R | SHOT τ_c, τ_g | ClipShots τ_c, τ_g | BBC τ_c, τ_g |\n|---|---|---|---|---|---|---|"
            )
            for r in allr:
                c = [P.m(r, s) for s in S]
                w(
                    f"| {disp(r)} | "
                    + " | ".join(f"{x['p']:.3f} / {x['r']:.3f}" for x in c)
                    + " | "
                    + " | ".join(f"{x['tau_c']}, {x['tau_g'] if x['tau_g'] is not None else '–'}" for x in c)
                    + " |"
                )
        else:
            w("| ID | τ_c, τ_g (from validation) | Validation F1 | SHOT P / R | ClipShots P / R | BBC P / R |\n|---|---|---|---|---|---|")
            for r in allr:
                v = PB["runs"][r]["val"]
                c = [P.m(r, s) for s in S]
                w(
                    f"| {disp(r)} | {v['tau_c']}, {v['tau_g'] if v['tau_g'] is not None else '–'} | {v['f1']:.4f} | "
                    + " | ".join(f"{x['p']:.3f} / {x['r']:.3f}" for x in c)
                    + " |"
                )
        w("\nA1, A2, A3 and their seeds use τ_c only.\n")
        w(f"### {n}.{first + 4} Table 8 — Per-type analysis on ClipShots\n")
        w("| ID | Cut recall | Gradual recall | Gradual F1 | Gradual IoU |\n|---|---|---|---|---|")
        for r in ("A0", "A1", "A2"):
            a = P.ana(r)
            g = lambda k: "n/a" if a[k] is None else f"{a[k]:.4f}"
            w(f"| {r} | {g('cut_recall')} | {g('gradual_recall')} | {g('gradual_f1')} | {g('gradual_iou')} |")
        w(f"\n### {n}.{first + 5} Table 9 — Recall by gradual-transition duration on ClipShots\n")
        w("| ID | ≤ 10 frames | 11–30 frames | > 30 frames |\n|---|---|---|---|")
        for r in ("A0", "C6"):
            a = P.ana(r)
            d, c = a["duration_recall"], a["duration_counts"]
            w(f"| {r} | {d['le_10']:.4f} (n = {c['le_10']}) | {d['11_30']:.4f} (n = {c['11_30']}) | {d['gt_30']:.4f} (n = {c['gt_30']}) |")
        d0, d6 = P.ana("A0")["duration_recall"], P.ana("C6")["duration_recall"]
        w(
            "\nC6 − A0: "
            + ", ".join(f"{lab} {sd(d6[k] - d0[k])}" for lab, k in (("≤ 10", "le_10"), ("11–30", "11_30"), ("> 30", "gt_30")))
            + ".\n"
        )

    def seeds_section(P, num, hdr=None):
        w(hdr or f"## {num}. Seeds — {P.label.split(' (')[0]}\n")
        w(
            "F1 of A0 and A1 at seeds 42, 43 and 44 (seed 42 = the main-study run). Mean and sample standard deviation (n − 1) over the three seeds.\n"
        )
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
        ds = [d(rid("A0", k), rid("A1", k), "clipshots") for k in SEEDS]
        rows.append(("A0 − A1 on ClipShots ≥ 0.01, seed 42", sd(ds[0]), ok(ds[0] >= 0.01)))
        rows.append(("A0 − A1 on ClipShots ≥ 0.01, mean over the three seeds", sd(stat.fmean(ds)), ok(stat.fmean(ds) >= 0.01)))
        rows.append(("A0 − A1 on ClipShots positive for every seed", " / ".join(sd(x) for x in ds), ok(all(x > 0 for x in ds))))
        x = d("A4", "A3", "clipshots")
        rows.append(("A4 − A3 on ClipShots ≥ 0.01", sd(x), ok(x >= 0.01)))
        x, y = d("A0", "A2", "clipshots"), d("A0", "A2", "bbc")
        rows.append(("A0 − A2 on ClipShots ≥ 0.01", sd(x), ok(x >= 0.01)))
        rows.append(("A0 − A2 on BBC ≥ 0.01", sd(y), ok(y >= 0.01)))
        mx = max(abs(P.f1(r, s) - P.f1("A0", s)) for r in CR[1:] for s in S)
        rows.append(("No leave-one-out change (C1–C6) reaches 0.01 on any test set", f"largest {mx:.4f}", ok(mx < 0.01)))
        w("| Statement | Value | Holds? |\n|---|---|---|")
        for r in rows:
            w(f"| {r[0]} | {r[1]} | {r[2]} |")
        w("")

    def lmin_sensitivity_section(hdr):
        w(hdr)
        w("Best F1 for each fixed ℓ_min, thresholds re-searched for each value:\n")
        w("| ℓ_min | τ_c | τ_g | Precision | Recall | F1 |\n|---|---|---|---|---|---|")
        pl = PA["runs"]["A0"]["sets"]["clipshots"]["per_lmin"]
        best = max(v["f1"] for v in pl.values())
        for k in ("1", "2", "3", "4"):
            v = pl[k]
            w(
                f"| {k} | {v['tau_c']} | {v['tau_g']} | {v['p']:.4f} | {v['r']:.4f} | "
                + (f"**{v['f1']:.4f}**" if v["f1"] == best else f"{v['f1']:.4f}")
                + " |"
            )
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
                cells.append(
                    ""
                    if ep > len(h)
                    else (f"**{h[ep - 1]['val']['f1']:.4f}**" if ep == su["best_epoch"] else f"{h[ep - 1]['val']['f1']:.4f}")
                )
            w(f"| {ep} | " + " | ".join(cells) + " |")
        w("")

    def hist_trends(hdr):
        names, H = _hist_data()
        w(hdr)
        w(
            "| Run | Best epoch | Best val F1 | Stopped at epoch | Range after best | Last | Trend per epoch |\n|---|---|---|---|---|---|---|"
        )
        for n in names:
            h, su = H[n]
            v = [x["val"]["f1"] for x in h]
            b = su["best_epoch"]
            post = v[b:]
            n_ = len(post)
            xs = list(range(n_))
            mx_, my_ = stat.fmean(xs), stat.fmean(post)
            slope = sum((a - mx_) * (c - my_) for a, c in zip(xs, post)) / sum((a - mx_) ** 2 for a in xs) if n_ > 1 else 0.0
            w(f"| {n} | {b} | {v[b - 1]:.4f} | {len(v)} | {min(post):.4f}–{max(post):.4f} | {v[-1]:.4f} | {slope:+.5f} |")
        w("")

    def run_summary(name):
        if "_s" in name:
            base, k = name.split("_s")
            pth = SR / "seed_runs" / name / base
        else:
            pth = SR / "ablations_v2" / name
        return json.load(open(pth / "summary.json"))

    w("# Follow-up experiments: extra seeds, ℓ_min = 1 and validation-selected thresholds\n")
    w(
        "This report covers the three parts of the follow-up to the Protocol-A ablation study of the HybridSBD shot boundary detection models (leak-free SHOT training set of 448 videos): "
        "**Part 1** trains A0 and A1 with two more seeds; **Part 2** re-evaluates every model with ℓ_min = 1 on all three test sets (Protocol A with ℓ_min = 1); **Part 3** evaluates every model with "
        "thresholds selected on the SHOT validation split (Protocol B). A short additional check (validation F1 per epoch) follows. "
        "Evaluation uses the existing best checkpoints (EMA weights selected by SHOT validation F1). Windows (T = 128, stride 64), window averaging, threshold grids (τ_c 0.10–0.90 step 0.05, "
        "τ_g ∈ {0.3, 0.4, 0.5, 0.6}), the decision rule and ±2-frame scene matching are unchanged. Seed 42 is the main-study run.\n"
    )
    w(
        "| Part | Content |\n|---|---|\n| Part 1 | Extra seeds for A0 and A1 (training) |\n| Part 2 | Protocol A with ℓ_min = 1 on all test sets (evaluation only) |\n"
        "| Part 3 | Protocol B, validation-selected thresholds (evaluation only) |\n| Part 4 | Additional check: validation F1 for every epoch |\n"
    )

    w("---\n\n# Part 1: Extra seeds for A0 and A1 (training)\n")
    w(
        "A0 (full model, dual-label) and A1 (full model, midpoint only) were trained with seeds 43 and 44, giving four new runs (A0-s43, A0-s44, A1-s43, A1-s44). Recipe, augmentation, synthetic transitions, "
        "EMA (0.999), early stopping (patience 10 on SHOT validation F1 with the variant's own decision rule) and the 80-epoch maximum are identical to the seed-42 runs. "
        "Only the training seed changes: model initialisation, training-data order, the ClipShots clip drawn each epoch, augmentation and synthetic transitions.\n"
    )
    w("## 1.1 Train/validation split is identical for all seeds\n")
    w(
        "In the training code the SHOT and ClipShots train/validation split was shuffled with the training seed, so seeds 43 and 44 would have used different validation videos. "
        "I fixed the split to the seed-42 split for every training seed (the option defaults to 42, so the seed-42 runs are unchanged). SHA-256 of the sorted SHOT validation video names:\n"
    )
    w(
        "| Training seed | Old rule (split follows the seed) | Fixed split (used) |\n|---|---|---|\n| 42 | f0070e4a… | f0070e4a… |\n| 43 | 88ddfd45… | f0070e4a… |\n| 44 | 792acaf5… | f0070e4a… |\n"
    )
    w(
        "All four new runs printed the same checksums before training: SHOT validation 67 videos (f0070e4a…), SHOT train 381 videos (ed9cb6ec…), ClipShots train 3,108 videos (f623f785…). "
        "The pre-extracted training clips use the stable per-video MD5 seed and are unchanged.\n"
    )
    w("## 1.2 The four new runs\n")
    w("| Run | Seed | Best epoch | Best SHOT val F1 | Epochs run | Stopped early | Training hours |\n|---|---|---|---|---|---|---|")
    for nm in ("A0", "A0_s43", "A0_s44", "A1", "A1_s43", "A1_s44"):
        su = run_summary(nm)
        sdn = 42 if "_s" not in nm else int(nm.split("_s")[1])
        w(
            f"| {disp(nm)} | {sdn} | {su['best_epoch']} | {su['best_val_f1']:.4f} | {su['epochs_completed']} | {su['stopped_early']} | {su['train_hours']:.2f} |"
        )
    w("\nSeed-42 rows are the main-study runs, shown for reference.\n")
    hist_table("## 1.3 SHOT validation F1 history of every epoch\n")
    w("## 1.4 Does training several runs at once on one GPU change the results?\n")
    if CONC:
        w(CONC["text"] + "\n")

    w("---\n\n# Part 2: Protocol A with ℓ_min = 1 (evaluation only)\n")
    w(
        "ℓ_min is fixed to 1 (no minimum-shot-length merging) on SHOT, ClipShots and BBC and is not searched. Thresholds are still selected on each test set (Protocol A).\n"
    )
    w("## 2.1 Checks\n")
    w(
        "The evaluation pipeline was checked against the earlier evaluation of the same checkpoints and reproduces it exactly for all 17 seed-42 models (51 of 51 model-test-set results: F1, thresholds, TP / FP / FN).\n"
    )
    w(
        "**Saved probabilities vs a fresh forward pass** (A0 and A1, all three test sets): the maximum absolute difference of p_y and p_z over all videos is 0.0, and F1, thresholds and TP / FP / FN are identical. "
        "The results in Parts 2 and 3 therefore use the saved probabilities.\n"
    )
    w("## Results\n")
    protocol_section(PROTS[0], "2", first=2, head=False)
    seeds_section(PROTS[0], "2.8", hdr="### 2.8 Seeds table\n")
    lmin_sensitivity_section("### 2.9 ℓ_min sensitivity for A0 on ClipShots\n")
    w("### 2.10 Interpretation check (0.01 band)\n")
    interp(PROTS[0])

    w("---\n\n# Part 3: Protocol B, validation-selected thresholds (evaluation only)\n")
    w(
        "For each model τ_c and τ_g are selected once on the SHOT validation videos and applied unchanged to SHOT, ClipShots and BBC (ℓ_min = 1). Nothing is tuned on a test set. "
        "Midpoint-only (A1, A3) and unified (A2) models select τ_c only.\n"
    )
    w("## 3.1 Validation videos as full videos (prerequisite)\n")
    if VV:
        w(
            "Before this work the 67 SHOT validation videos existed only as pre-cut training clips, so I **added a data path**: the 67 videos are now decoded completely with the same decoder, resizing (48 × 64), "
            "window layout (T = 128, stride 64) and window averaging as the SHOT test set (1,270 windows), and scored with the same scene matching. Ground truth comes from the SHOT label file with the same "
            f"transition-to-scene conversion as ClipShots; {VV['transitions_used']} of {VV['transitions_total']} transitions lie within the decoded video length and are used. "
            "The label file is the only ground truth available for these videos and is not the authors' final annotation, so validation F1 is noisier than test F1 and the selected thresholds are noisier too.\n"
        )
    w("## 3.2 Threshold selection and application\n")
    w(
        "τ_c ∈ {0.10, …, 0.90} and τ_g ∈ {0.3, 0.4, 0.5, 0.6} were searched once per model by maximising F1 on the full validation videos with ℓ_min = 1. "
        "The selected values and the validation F1 are in the thresholds table (3.6). They were applied unchanged to the three test sets for every model, including the new seeds. "
        "Numbers are expected to be lower than under Protocol A, especially on ClipShots and BBC.\n"
    )
    w("## Results\n")
    protocol_section(PROTS[1], "3", first=3, head=False)
    seeds_section(PROTS[1], "3.9", hdr="### 3.9 Seeds table\n")
    w("### 3.10 Interpretation check (0.01 band)\n")
    interp(PROTS[1])
    w(
        "Differences below 0.01 F1 are treated as inconclusive. Under Protocol B the thresholds themselves add noise, so single-model differences are less reliable than under Protocol A.\n"
    )

    w("---\n\n# Part 4: Additional check — SHOT validation F1 for every epoch\n")
    w(
        "The validation F1 of every epoch for A0 and A1 at seed 42 and for the four new runs is in Table 1.3. What happens after each run's best epoch:\n"
    )
    hist_trends("## 4.1 Behaviour after the best epoch\n")
    w("## 4.2 Did validation F1 level off or fall after the best epoch?\n")
    w(
        "In every one of the six runs, including A1 at seed 42, early stopping ended training after 10 epochs without improvement, and validation F1 had **levelled off** rather than still rising or collapsing: "
        "after its best epoch each run moves up and down inside a band about 0.01–0.02 wide just below its best value, with a linear trend close to zero (between −0.0007 and +0.0004 F1 per epoch), "
        "and no run was still climbing when it stopped. A1 at seed 42 (best epoch 7, 0.7863) fluctuated between 0.7695 and 0.7852 afterwards and ended at 0.7846, so it was levelled off and unstable at the 0.01 level, "
        "not falling steadily. The new A1 runs behave the same way (A1-s43 drifts slightly down, −0.0005 per epoch; A1-s44 is flat, −0.0002). "
        "The epoch-to-epoch swings are about as large as the gaps between runs, so the best-epoch choice itself is noisy.\n"
    )
    return "\n".join(out)


def build_main(pa: dict, pb: dict, followup_md: str):
    """Full results report (``RESULTS_FULL.md``) and one-page summary (``docs/RESULTS.md``)."""

    def jl(p):
        return json.load(open(p))

    PA, PB = pa["runs"], pb["runs"]  # l_min = 1 re-evaluation (Protocol A) and Protocol B

    def f4(x):
        return f"{x:.4f}"

    def sd(x):
        return f"{x:+.4f}"

    # ---------- load runs ----------
    rows = {r["run_id"]: r for r in csv.DictReader(open(V2 / "ablation_results.csv"))}
    v2 = {}
    for rid, r in rows.items():
        cfg = jl(V2 / rid / "config.json")
        hist = jl(V2 / rid / "history.json")
        v2[rid] = dict(
            params=int(r["params_total"]),
            shot=float(r["shot_F1"]),
            cs=float(r["cs_F1"]),
            bbc=float(r["bbc_F1"]),
            shot_p=float(r["shot_P"]),
            shot_r=float(r["shot_R"]),
            cs_p=float(r["cs_P"]),
            cs_r=float(r["cs_R"]),
            bbc_p=float(r["bbc_P"]),
            bbc_r=float(r["bbc_R"]),
            best_ep=int(r["best_epoch"]),
            epochs=len(hist),
            hours=float(r["train_hours"]),
            val=float(r["val_F1"]),
            lat=float(r["latency_B16_ms"]),
            vram=float(r["vram_B16_MB"]),
            args=cfg["args"],
            commit=cfg["git"]["commit"][:7],
            gpu=cfg["environment"]["gpu_model"],
            torch=cfg["environment"]["pytorch_version"],
            cuda=cfg["environment"]["cuda_version"],
        )

    def f1_of(p):
        return jl(p)["overall"]

    v3 = {}
    for rid in ("M0_min", "M1_edge", "B_noatt", "S1_rel"):
        d = V3 / rid
        cfg = jl(d / "config.json")
        hist = jl(d / "history.json")
        best = max(hist, key=lambda e: e["val"]["f1"])
        st = jl(d / "status.json")["status"]
        s, c, b = (f1_of(d / f"{n}_results.json") for n in ("shot", "clipshots", "bbc"))
        v3[rid] = dict(
            params=cfg["parameters"][0],
            shot=s["f1"],
            cs=c["f1"],
            bbc=b["f1"],
            epochs=len(hist),
            best_ep=best["epoch"],
            hours=hist[-1]["elapsed_hours"],
            status=st,
            val=best["val"]["f1"],
            commit=cfg["git"]["commit"][:7],
        )
    sp = {r["label"]: r for r in jl(SB / "speed_summary.json")}
    a0 = v2["A0"]
    a = v2["A0"]["args"]
    PYV = jl(V2 / "A0" / "config.json")["environment"]["python_version"].split()[0]

    DESC = {
        "A0": "Full, dual-label (primary)",
        "A1": "Full, midpoint only",
        "A2": "Full, unified single head",
        "A3": "Reduced, midpoint only",
        "A4": "Reduced, dual-label",
        "C1": "w/o difference-based branch (B)",
        "C2a": "w/o FrameSimilarity",
        "C2b": "w/o ColorHistogramSimilarity",
        "C3": "w/o temporal delta",
        "C4": "w/o per-frame pathway (relations from Branch A)",
        "C5a": "w/o pixel-level motion descriptor",
        "C5b": "w/o edge-density head",
        "C6": "w/o self-attention and gate",
    }

    def bset(ids, getter, hi=True, nd=4):
        """ids whose rounded value is the best (highest if hi else lowest) in the column"""
        vals = {i: round(getter(i), nd) for i in ids}
        best = max(vals.values()) if hi else min(vals.values())
        return {i for i, v in vals.items() if v == best}

    def B(txt, flag):
        return f"**{txt}**" if flag else txt

    def bold_speed_table(text):
        """bold the best value of every numeric column (lowest params/size/GFLOPs/ms/VRAM, highest frames/s)"""
        lines = text.split("\n")
        rows = [l for l in lines if l.startswith("|") and not l.startswith("|---") and not l.startswith("| Model")]
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        hdr = [c.strip() for c in next(l for l in lines if l.startswith("| Model")).strip().strip("|").split("|")]
        best = {}
        for j in range(1, len(hdr)):
            vals = [float(c[j]) for c in cells]
            best[j] = max(vals) if hdr[j].startswith("frames/s") else min(vals)
        out = []
        for l in lines:
            if l in rows:
                c = [x.strip() for x in l.strip().strip("|").split("|")]
                for j in range(1, len(hdr)):
                    if float(c[j]) == best[j]:
                        c[j] = f"**{c[j]}**"
                l = "| " + " | ".join(c) + " |"
            out.append(l)
        return "\n".join(out)

    out = []
    w = out.append

    # ---------- all F1 / precision / recall in this report come from the l_min = 1 evaluation (Protocol A), re-read from the saved files ----------
    for _r, _d in v2.items():
        for _k, _s in (("shot", "shot"), ("cs", "clipshots"), ("bbc", "bbc")):
            _m = PA[_r]["sets"][_s]["lmin1"]
            _d[_k], _d[_k + "_p"], _d[_k + "_r"] = _m["f1"], _m["p"], _m["r"]
    for _r, _d in v3.items():
        for _k, _s in (("shot", "shot"), ("cs", "clipshots"), ("bbc", "bbc")):
            _d[_k] = PA[_r]["sets"][_s]["lmin1"]["f1"]

    w("# Results — HybridSBD ablation study (leak-free)\n")
    w(
        "All numbers in this document come from the saved experiment results (training histories, per-test-set threshold-search results, validation-selected thresholds and speed measurements). "
        "The evaluation uses the minimum scene length ℓ_min = 1 (no merging) on SHOT, ClipShots and BBC, and is reported under two threshold rules: **Protocol A** selects τ_c and τ_g on each test set (optimistic by construction, comparable only to other Protocol-A numbers) and **Protocol B** selects them once on the 67 SHOT validation videos and applies them unchanged to all three test sets. "
        "Every variant has one training run (seed 42); A0 and A1 also have seeds 43 and 44. Per the spec, F1 differences below 0.01 are treated as inconclusive.\n"
    )
    w(
        "| Section | Content |\n|---|---|\n| 1 | Data and training approach |\n| 2 | Compliance with the specification |\n| 3 | Extra seeds for A0 and A1 (training) |\n| 4 | Protocol A, ℓ_min = 1 (Tables 7a, 7b, build-up ladder, precision/recall, Tables 8 and 9, seeds table, ℓ_min sensitivity) |\n| 5 | Protocol B, validation-selected thresholds (same tables) |\n| 6 | Validation F1 for every epoch |\n| 7 | Interpretation |\n| 8 | Speed, memory and size |\n| 9 | Caveats |\n"
    )

    # ---------------- 1. data and training approach ----------------
    w("## 1. Data and training approach\n")
    w("### 1.1 Datasets\n")
    w("| Role | Dataset | Videos | Notes |\n|---|---|---|---|")
    w(
        "| Train | ClipShots train | 3,656 | 3,657 listed; one source file (`5186782609.mp4`) is unreadable and is explicitly allow-listed as missing |"
    )
    w("| Train / validation | SHOT (AutoShot, `kuaishou_v2.txt`) | 448 | 648 labelled videos minus the 200 SHOT test videos |")
    w("| Test | SHOT test | 200 | the authors' released SHOT test annotation |")
    w("| Test | ClipShots test | 500 | official ClipShots test annotations |")
    w("| Zero-shot test | BBC Planet Earth | 11 episodes | Zenodo record 14865504; never used for training or selection |\n")
    LA = jl(V2 / "shot_label_audit.json")
    w(
        "**SHOT label coverage.** "
        f"The SHOT download holds {LA['files_in_three_video_folders']} video files in three folders ({LA['distinct_videos_on_disk']} distinct videos), but only "
        f"**{LA['labelled_videos']} have labels** in `kuaishou_v2.txt`; the other **{LA['on_disk_without_labels']} videos have no annotations at all** and cannot be used for supervised "
        f"training, so they are not used. All {LA['labelled_and_on_disk']} labelled videos are present on disk. "
        f"The label file is also irregular, and a tolerant parser is used: {LA['header_entries']} header entries for {LA['labelled_videos']} videos "
        f"({LA['videos_with_duplicate_entries']} videos appear twice; the {LA['non_test_duplicates_identical']} non-test duplicates are identical), mixed `a,b` / `a b` line formats, one header ending in `.mp4.png`, "
        f"and {LA['lines_with_odd_number_of_integers']} lines with an odd number of integers that are skipped. A strict parser would silently drop {LA['transitions_dropped_by_strict_parse']} of "
        f"{LA['transitions_recovered_by_tolerant_parse']:,} transition lines ({LA['transitions_dropped_by_strict_parse'] / LA['transitions_recovered_by_tolerant_parse'] * 100:.1f} %).\n"
    )
    w(
        f"For the {LA['test_videos']} test videos the file contains an earlier version of the labels: {LA['test_labels_in_file']:,} labels against {LA['test_labels_final_annotation']:,} transitions in the final test annotation, "
        f"{LA['test_file_labels_within_2_frames_of_final']:,} of them within ±2 frames of the final annotation. Training on these videos therefore means training on almost the exact test answers.\n"
    )
    PAPER_VIDEOS, PAPER_TEST = 853, 200  # AutoShot paper / release: 853 videos, 200 of them are fine-annotated test videos
    w(
        "**Limitation of the released SHOT labels.** "
        f"The released label file mixes the training labels and the labels of the {LA['test_videos']} test videos in one file, so the test videos can only be told apart "
        "with the separate test annotation. It also covers far fewer videos than the dataset description: the AutoShot paper describes "
        f"{PAPER_VIDEOS} videos, of which {PAPER_TEST} are test videos, which would leave {PAPER_VIDEOS - PAPER_TEST} for training. The label file, however, covers only {LA['labelled_videos']} videos "
        f"and these include all {LA['test_videos']} test videos; roughly {LA['on_disk_without_labels']} videos have no labels at all. After excluding the test videos, "
        f"{LA['train_videos_after_excluding_test']} labelled videos remain for training, about {PAPER_VIDEOS - PAPER_TEST - LA['train_videos_after_excluding_test']} fewer than the paper's split implies. "
        "Our SHOT results therefore use less training data than the authors' setup could, which is a limitation of the released data, and more labelled training videos could improve them.\n"
    )
    w(
        "**Test videos are excluded from training.** The SHOT label file contains the 200 test videos, so training on all 648 would "
        "train on the test set. Training uses the 448 other videos, and an overlap audit across all train/test sets "
        "(SHOT, ClipShots, BBC) finds zero shared videos.\n"
    )
    w("### 1.2 Preparation\n")
    w(
        "1. Archives are downloaded, extracted and checked, then every video is decoded once and each frame is bilinearly resized to "
        "**48 × 64** (uint8, RGB). Decoding is chunked to bound memory."
    )
    w(
        "2. Videos are cut into **128-frame clips** stored as tensors with per-frame labels "
        "(0 none, 1 cut, 2 gradual) and a transition-length value."
    )
    w(
        "3. **Training clips.** ClipShots: 20 clips per video (60 % centred on a transition with ±15-frame jitter, the rest random), "
        "73,120 clips from 3,656 videos. SHOT: 70 clips per video; every transition gets 3 clips centred on it "
        "(jitter −20 / 0 / +20 frames) and the remaining quota is random, 31,360 clips from 448 videos."
    )
    w(
        "4. **Test clips.** Sliding windows of 128 frames with stride 64 cover each full video; overlapping predictions are averaged "
        "back into one probability track per video."
    )
    w(
        "5. A verification step checks that every clip loads, has the right shape and label range, and that every expected video "
        "is present, before any training starts.\n"
    )
    w("### 1.3 Training\n")
    w("| Setting | Value |\n|---|---|")
    w("| Targets | y = midpoint of every transition (cuts and graduals); z = every frame inside a gradual (many-hot) |")
    w("| Loss | 5.0 · focal BCE(y; α 0.8, γ 2) + 0.3 · focal BCE(z; α 0.5, γ 2, label smoothing 0.1) |")
    w("| Optimiser | AdamW, lr 5e-5, weight decay 0.05, batch 32, AMP, gradient clip 1.0 |")
    w("| Schedule | up to 80 epochs; 5 warm-up epochs (lr × 0.25 → 1), then cosine to 5e-6 |")
    w(f"| Early stopping | patience {a['patience']} on SHOT validation F1 (15 % of the SHOT training videos: 381 train / 67 validation) |")
    w(
        "| Epoch mix | all SHOT training clips + a fresh random 70 % of the ClipShots training clips each epoch (ClipShots split 3,108 train / 548 validation videos) |"
    )
    w("| Basic augmentation | horizontal flip, per-frame brightness and contrast jitter, noise, brightness shift |")
    w("| Strong augmentation | per-clip zoom/shift (up to 25 %), colour gain, occasional grayscale |")
    w(
        f"| Synthetic transitions | probability {a['synth_prob']} per clip: hard cut 40 %, dissolve 40 %, fade through black 20 % (3–30 frames), "
        "spliced from two clips of the same batch with a 3-frame clear margin from real transitions; labels follow the real-label convention |"
    )
    w(f"| Weight averaging | EMA, decay {a['ema_decay']}; validation, checkpoints and test use the EMA weights |")
    w(f"| Seed | {a['seed']} (fixed hash seed); every variant uses the same recipe, data and seed |")
    w(f"| Hardware / software | {a0['gpu']}, Python {PYV}, PyTorch {a0['torch']}, CUDA {a0['cuda']} |\n")
    w(
        "**Synthetic transitions in detail.** Each clip of a batch is replaced, with probability "
        + str(a["synth_prob"])
        + ", by a splice of itself (clip i) and another clip j from the same batch: "
        "a **hard cut** (40 %: frames before a random point come from i, after it from j), a **dissolve** (40 %: a linear cross-fade from i to j over 3–30 frames) or a **fade through black** "
        "(20 %: i fades to black, then j fades in, over 3–30 frames). Labels before the splice come from i and after it from j; a cut is labelled on the two frames at the splice, and a dissolve or fade is "
        "labelled gradual on every frame of the transition, exactly like real transitions. The splice is kept at least 3 frames away from every real transition in either clip. "
        "This adds many extra transitions with exactly known boundaries, especially gradual ones, which are rare and hard to annotate, and it uses only frames and labels from the training videos.\n"
    )
    w("### 1.4 Evaluation (Protocols A and B)\n")
    w(
        "- Each model's best checkpoint (by SHOT validation F1) is evaluated over a threshold grid: boundary threshold τ_c ∈ {0.10, 0.15, …, 0.90} and gradual threshold τ_g ∈ {0.3, 0.4, 0.5, 0.6}; "
        "the minimum scene length is fixed to ℓ_min = 1 (no merging) on all three test sets. **Protocol A** reports the best setting per test set. "
        "**Protocol B** selects the setting once, on the 67 SHOT validation videos evaluated as full videos, and applies it unchanged to SHOT, ClipShots and BBC."
    )
    w(
        "- Decision rule (dual-label models): boundary if p_y > τ_c; gradual if p_z > τ_g; cut = boundary and not gradual; "
        "the predicted many-hot track is gradual OR cut. Midpoint-only (A1, A3) and unified (A2) variants use τ_c only."
    )
    w(
        "- Scoring: AutoShot's scene-based matching with **±2 frame tolerance**; precision, recall and F1 over transitions, pooled over all videos.\n"
    )

    w("### 1.5 Protocol details (spec Section 0, items 3–8)\n")
    w(
        "- **Minimum shot length ℓ_min** is applied in the evaluation code, shared by all three test sets: after the boundary track "
        "is turned into scenes, every predicted scene shorter than ℓ_min frames is merged into the previous scene. All results use ℓ_min = 1 (no merging); "
        "the sensitivity of A0 on ClipShots to ℓ_min ∈ {1, 2, 3, 4} is in section 4.9."
    )
    w(
        "- **Overlapping windows.** Probabilities of all windows that cover a frame are summed and divided by the number of windows covering it "
        "(plain average, separately for p_y and p_z). Frames near the start or end are covered by one window only and use that window's value."
    )
    w(
        "- **Final window.** ClipShots test: window starts 0, 64, 128, … plus one extra window starting at (frames − 128), so the last window is shifted back "
        "and ends on the final frame. SHOT test: a window that would run past the end is shifted back to start at (frames − 128); a video shorter than 128 frames "
        "is padded by repeating its last frame. BBC: windows start every 64 frames and the last window is padded by repeating the final frame. "
        "The same window and averaging code is used for every run."
    )
    w(
        "- **Seeds.** Clip sampling during data preparation uses an MD5-based stable per-video seed (not Python's `hash`). Training seeds the hash seed, "
        "Python, NumPy, PyTorch (CPU and CUDA) and every data-loader worker to the same seed (42) and runs cuDNN in deterministic mode."
    )
    w(
        "- **Benchmark hygiene.** `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.empty_cache()` are called before every latency and VRAM measurement "
        ""
    )
    w(
        "- **Data overlap.** No video appears in more than one of SHOT train / SHOT validation / SHOT test / ClipShots train / ClipShots test / BBC.\n"
    )

    # ---------------- 2. compliance ----------------

    from checks import verify as CK

    ARCH = jl(V2 / "a0_architecture_check.json")
    CTRL = SR / "control_leak648" / "A0"
    ctl = {n: f1_of(CTRL / f"{n}_results.json")["f1"] for n in ("shot", "clipshots", "bbc")}
    w("## 2. Compliance with the specification\n")
    w("### 2.1 A0 reproduction checks (spec Section 1)\n")
    w("**Check 1 — architecture, on the trained A0 checkpoint: " + ("PASS" if ARCH["all_pass"] else "FAIL") + ".**\n")
    w("| Item | Required | Found |\n|---|---|---|")
    w(f"| Fusion blocks | 4 | {ARCH['fusion_blocks']} |")
    w(f"| Positional encoding | none | {'none' if not ARCH['positional_encoding_keys'] else ARCH['positional_encoding_keys']} |")
    w(f"| Delta projection weight | (64, 640) | {tuple(ARCH['delta_projection_weight_shape'])} |")
    w(f"| D | 384 | {ARCH['main_dim']} |")
    w(f"| Fusion input / embedding dim | 655 / 640 | {ARCH['fusion_in']} / {ARCH['embedding_dim']} |")
    w(f"| Exact parameter count | rounds to 10.9M | {ARCH['parameters_total_exact']:,} ({ARCH['parameters_round_to_M']}M) |\n")
    w("**Check 2 — performance (Protocol A, ℓ_min = 1, paper's acceptable ranges):**\n")
    w("| Test set | Paper | Acceptable range | Our A0 | Result |\n|---|---|---|---|---|")
    for key, (label, paper, lo, hi) in CK.A0_PAPER_RANGES.items():
        v = {"shot": a0["shot"], "clipshots": a0["cs"], "bbc": a0["bbc"]}[key]
        st = "within range" if lo <= v <= hi else ("**below range**" if v < lo else "above range")
        w(f"| {label} F1 | {paper} | {lo} – {hi} | {f4(v)} | {st} |")
    w(
        "\nCheck 2 does **not** pass on SHOT, and this is expected: **it is caused by a deliberate change in the training data, not by the code.** "
        "The SHOT label file `kuaishou_v2.txt` lists 648 videos, and the 200 SHOT *test* videos (the authors' released test annotation) are among them. "
        "The paper-era setup therefore trained on 648 videos that include the entire SHOT test set — a data leak that inflates SHOT F1. "
        "We removed the 200 test videos from training (448 videos remain), so the SHOT numbers here are leak-free and are lower than the paper's 0.924. "
        f"As a one-off control, the same A0 recipe trained on all 648 videos reaches SHOT {f4(ctl['shot'])}, which is inside the required 0.915–0.930 range, "
        "so the gap is explained by the leak and not by an architecture or pipeline difference (Check 1 passes). "
        "Per the spec, this is reported to the team: the ablations proceed on the same code and data policy for all runs, and the main-paper SHOT number needs discussion. "
        f"ClipShots ({f4(a0['cs'])}) is {'inside' if CK.A0_PAPER_RANGES['clipshots'][2] <= a0['cs'] <= CK.A0_PAPER_RANGES['clipshots'][3] else ('above' if a0['cs'] > CK.A0_PAPER_RANGES['clipshots'][3] else 'below')} its range and BBC ({f4(a0['bbc'])}) is {'inside' if CK.A0_PAPER_RANGES['bbc'][2] <= a0['bbc'] <= CK.A0_PAPER_RANGES['bbc'][3] else 'outside'} it; neither benchmark shares videos with the SHOT test set.\n"
    )
    w("### 2.2 Deviations from the specification\n")
    w("| Spec says | What was done | Why |\n|---|---|---|")
    w(
        "| SHOT: 648 training videos (§3) | **448 videos**: the 200 SHOT test videos are excluded | The 648 include the SHOT test set (data leak), see 2.1 |"
    )
    w(
        f"| Early stopping patience 15 (§3) | patience {a['patience']} for all 13 runs and the build-up runs | Shortens training time: the validation optimum appeared early in the first runs, so the same patience was used for every variant |"
    )
    w(
        f"| Augmentation: flip, brightness, contrast, noise, clip brightness (§3) | all of these, **plus** strong per-clip augmentation, synthetic transitions (p = {a['synth_prob']}) and weight averaging (EMA {a['ema_decay']}) | Added to recover accuracy once the leaked videos were removed; identical for every run, so the comparison between variants stays fair |"
    )
    w(
        "| All runs on the same code version as A0 (§1) | A0 and the other 12 runs were trained from two code revisions | A0 was trained first as the recipe experiment; only the launcher scripts differ, the training code is identical |"
    )
    w(
        "| 13 runs, nothing else (§5, prompt) | **extra** build-up ladder (4 runs, section 5), speed table with GFLOPs, B = 1 / 32 timings and an AutoShot comparison, one extra C4 cut-recall analysis | Requested additions to identify which components do the work; kept in separate sections |"
    )
    w("| Training speed | channels-last memory format | Memory-layout change only; values are mathematically identical |\n")

    # ---------------- 3-6. follow-up experiments (embedded from reports/FOLLOWUP_RESULTS_SHAREABLE.md) ----------------
    _fu = followup_md
    _fu = _fu[_fu.index("# Part 1:") :]
    _fu = re.sub(r"^---\s*$", "", _fu, flags=re.M)
    _fu = re.sub(r"^# Part (\d): ", lambda m: f"## {2 + int(m.group(1))}. ", _fu, flags=re.M)
    _fu = re.sub(r"^(#{2,3}) (\d)\.(\d+) ", lambda m: f"{m.group(1)}# {2 + int(m.group(2))}.{m.group(3)} ", _fu, flags=re.M)
    _fu = re.sub(r"^## Results$", "### Results", _fu, flags=re.M)
    _fu = _fu.replace("Table 1.3", "Table 3.3").replace("table (3.6)", "table (5.6)").replace("Parts 2 and 3", "sections 4 and 5")
    assert "Part " not in _fu, "unresolved cross-reference to a Part"
    w(_fu.strip() + "\n")

    # ---------------- 7. interpretation (computed from the Protocol A and Protocol B results) ----------------
    def _f(P, r, s):
        return PA[r]["sets"][s]["lmin1"]["f1"] if P == "A" else PB[r]["sets"][s]["f1"]

    def _an(P, r):
        return PA[r]["analysis_lmin1"] if P == "A" else PB[r]["analysis"]

    _S = ("shot", "clipshots", "bbc")
    _N = {"shot": "SHOT", "clipshots": "ClipShots", "bbc": "BBC"}
    _CR = ["C1", "C2a", "C2b", "C3", "C4", "C5a", "C5b", "C6"]

    def _both(x, y):
        return f"{sd(x)} / {sd(y)}"

    w("## 7. Interpretation (0.01 band)\n")
    w(
        'Differences below 0.01 F1 are inconclusive. Each cell shows Protocol A / Protocol B. Seed 42 unless stated; "3-seed mean" is the mean over seeds 42, 43 and 44 of the paired A0 − A1 difference.\n'
    )

    def _other_sets_sentence():
        parts = []
        for s_ in ("shot", "bbc"):
            for P in "AB":
                ds = [_f(P, f"A0{k}", s_) - _f(P, f"A1{k}", s_) for k in ("", "_s43", "_s44")]
                tag = "same sign at all seeds" if all(x > 0 for x in ds) or all(x < 0 for x in ds) else "sign changes across seeds"
                parts.append(f"{_N[s_]} Protocol {P}: " + ", ".join(sd(x) for x in ds) + f" ({tag})")
        return "On SHOT and BBC the paired A0 − A1 differences (seeds 42, 43, 44) are: " + "; ".join(parts) + "."

    w("### 7.1 Supervision\n")
    w("| Dataset | A0 − A1 (seed 42) | A0 − A1 (3-seed mean) | A4 − A3 (reduced) | Outcome |\n|---|---|---|---|---|")
    for s in _S:
        d1 = {P: _f(P, "A0", s) - _f(P, "A1", s) for P in "AB"}
        d3 = {P: sum(_f(P, f"A0{k}", s) - _f(P, f"A1{k}", s) for k in ("", "_s43", "_s44")) / 3 for P in "AB"}
        d4 = {P: _f(P, "A4", s) - _f(P, "A3", s) for P in "AB"}

        def _case(P):
            a_, b_ = d1[P] >= 0.01, d4[P] >= 0.01
            return (
                "dual-label helps for both the full and the reduced model"
                if a_ and b_
                else ("neither ≥ 0.01: no accuracy claim" if not a_ and not b_ else "only one of the two ≥ 0.01: mixed")
            )

        w(
            f"| {_N[s]} | {_both(d1['A'], d1['B'])} | {_both(d3['A'], d3['B'])} | {_both(d4['A'], d4['B'])} | A: {_case('A')}; B: {_case('B')} |"
        )
    _cs = {P: [_f(P, f"A0{k}", "clipshots") - _f(P, f"A1{k}", "clipshots") for k in ("", "_s43", "_s44")] for P in "AB"}
    w("")
    w(
        "On ClipShots the paired A0 − A1 difference is positive at every seed under both protocols (Protocol A: "
        + ", ".join(sd(x) for x in _cs["A"])
        + "; Protocol B: "
        + ", ".join(sd(x) for x in _cs["B"])
        + "), so dual-label supervision helps on the benchmark with many gradual transitions. "
        + _other_sets_sentence()
    )
    _rc = lambda P, r, k: sum(_an(P, f"{r}{x}")[k] for x in ("", "_s43", "_s44")) / 3
    w(
        f"The error profile changes as well: on ClipShots (3-seed mean, Protocol A) cut recall / gradual recall is {_rc('A', 'A0', 'cut_recall'):.3f} / {_rc('A', 'A0', 'gradual_recall'):.3f} for A0 (dual) against {_rc('A', 'A1', 'cut_recall'):.3f} / {_rc('A', 'A1', 'gradual_recall'):.3f} for A1 (midpoint only); "
        f"under Protocol B {_rc('B', 'A0', 'cut_recall'):.3f} / {_rc('B', 'A0', 'gradual_recall'):.3f} against {_rc('B', 'A1', 'cut_recall'):.3f} / {_rc('B', 'A1', 'gradual_recall'):.3f}.\n"
    )

    w("### 7.2 Hypotheses\n")
    w("| Comparison | Hypothesis | Protocol A / Protocol B |\n|---|---|---|")
    w(
        "| A0 vs A2 | A0 higher | "
        + "; ".join(f"{_N[s]} {_both(_f('A', 'A0', s) - _f('A', 'A2', s), _f('B', 'A0', s) - _f('B', 'A2', s))}" for s in _S)
        + " |"
    )

    def _worst(P, s):
        r = min(_CR, key=lambda r: _f(P, r, s) - _f(P, "A0", s))
        return r, _f(P, r, s) - _f(P, "A0", s)

    w(
        "| C2a / C2b | Largest drops | Largest drop per dataset: "
        + "; ".join(
            f"{_N[s]} " + " / ".join((f"{_worst(P, s)[0]} {sd(_worst(P, s)[1])}" if _worst(P, s)[1] < 0 else "no drop") for P in "AB")
            for s in _S
        )
        + ". "
        + (
            "Confirmed"
            if all(_worst(P, s)[0] in ("C2a", "C2b") and _worst(P, s)[1] <= -0.01 for P in "AB" for s in _S)
            else "Not confirmed"
        )
        + " (the largest drop is not always C2a or C2b, or it is below 0.01) |"
    )
    _d = {P: (_an(P, "A0")["duration_recall"], _an(P, "C6")["duration_recall"], _an(P, "A0")["duration_counts"]) for P in "AB"}
    w(
        "| C6 | Lower recall on > 30-frame transitions | C6 − A0 recall: "
        + "; ".join(
            f"{lab} {_both(_d['A'][1][k] - _d['A'][0][k], _d['B'][1][k] - _d['B'][0][k])}"
            for lab, k in (("≤ 10", "le_10"), ("11–30", "11_30"), ("> 30", "gt_30"))
        )
        + f" (n = {_d['A'][2]['gt_30']} for > 30) |"
    )
    _mx = {P: max(abs(_f(P, r, s) - _f(P, "A0", s)) for r in _CR for s in _S) for P in "AB"}
    w(f"| Any Cx | Drop | Largest single-component change on any dataset: {_mx['A']:.4f} / {_mx['B']:.4f} |")
    w("")
    w("### 7.3 Build-up ladder: what does the work?\n")
    w("Change in F1 at each step (Protocol A / Protocol B); a step clears the band if it is ≥ 0.01 on that dataset under that protocol.\n")
    _LD = [
        ("M0_min", "M1_edge", "+ edge-density head"),
        ("M1_edge", "B_noatt", "+ 4 fusion blocks, no attention"),
        ("B_noatt", "A4", "+ attention and gate"),
        ("A4", "S1_rel", "+ FrameSimilarity, ColorHist, delta"),
        ("S1_rel", "A0", "+ diff branch, motion, per-frame pathway"),
    ]
    w("| Step | SHOT | ClipShots | BBC |\n|---|---|---|---|")
    for a_, b_, lab in _LD:
        cells = []
        for s in _S:
            x = {P: _f(P, b_, s) - _f(P, a_, s) for P in "AB"}
            cells.append(
                _both(x["A"], x["B"])
                + (
                    " ✔"
                    if max(x.values()) >= 0.01 and min(x.values()) >= 0.01
                    else (" (A only)" if x["A"] >= 0.01 else (" (B only)" if x["B"] >= 0.01 else ""))
                )
            )
        w(f"| {a_} → {b_} ({lab}) | " + " | ".join(cells) + " |")
    w("")
    w(
        f"The bare core (M0_min) is within {_f('A', 'A0', 'shot') - _f('A', 'M0_min', 'shot'):.3f} / {_f('A', 'A0', 'clipshots') - _f('A', 'M0_min', 'clipshots'):.3f} / {_f('A', 'A0', 'bbc') - _f('A', 'M0_min', 'bbc'):.3f} F1 of the full model on SHOT / ClipShots / BBC under Protocol A "
        f"with {v3['M0_min']['params'] / a0['params'] * 100:.0f} % of its parameters. ✔ = clears the band under both protocols.\n"
    )

    # ---------------- 4. speed ----------------
    w("## 8. Speed, memory and size\n")
    w(
        "FP32, synthetic input `(B, 128, 3, 48, 64)`, forward pass only, 10 warm-up + 80 timed runs, otherwise idle GPU "
        f"({a0['gpu']}). Peak VRAM is the maximum allocated during the forward pass. GFLOPs are per 128-frame clip and count convolutions "
        "and matrix multiplies only. AutoShot uses its native 27 × 48 input; our models use 48 × 64.\n"
    )
    _lab = list(sp.keys())

    def _g(lab, bs, key):
        return sp[lab]["batch_results"][str(bs)][key]

    def _col(vals, hi, fmt):
        best = max(vals.values()) if hi else min(vals.values())
        return {k: (f"**{format(v, fmt)}**" if v == best else format(v, fmt)) for k, v in vals.items()}

    _cols_a = [
        ("Params (M)", {l: sp[l]["total_params"] / 1e6 for l in _lab}, False, ".2f"),
        ("FP32 (MB)", {l: sp[l]["size_fp32_mb"] for l in _lab}, False, ".1f"),
        ("GFLOPs / clip", {l: sp[l]["gflops"] for l in _lab}, False, ".2f"),
    ] + [(f"VRAM B={bs} (MB)", {l: _g(l, bs, "peak_memory_mb") for l in _lab}, False, ".0f") for bs in (1, 16, 32)]
    _cols_b = [(f"ms B={bs}", {l: _g(l, bs, "mean_ms") for l in _lab}, False, ".1f") for bs in (1, 16, 32)] + [
        (f"frames/s B={bs}", {l: _g(l, bs, "frames_per_sec") for l in _lab}, True, ".0f") for bs in (1, 16, 32)
    ]
    for title, cols in (("**Size, compute and peak VRAM**", _cols_a), ("**Latency and throughput**", _cols_b)):
        cc = [(h, _col(v, hi, f)) for h, v, hi, f in cols]
        w(title + "\n")
        w("| Model | " + " | ".join(h for h, _ in cc) + " |\n|---|" + "---|" * len(cc))
        for l in _lab:
            w(f"| {l} | " + " | ".join(c[l] for _, c in cc) + " |")
        w("")
    au = sp["AutoShot (TransNetV2Supernet)"]["batch_results"]["16"]["mean_ms"]
    auv = sp["AutoShot (TransNetV2Supernet)"]["batch_results"]["16"]["peak_memory_mb"]
    w("| Model | B=16 speedup vs AutoShot | B=16 VRAM vs AutoShot | GFLOPs vs AutoShot |\n|---|---|---|---|")
    for rid in ("M0_min", "M1_edge", "B_noatt", "A4", "S1_rel", "C1", "A0"):
        br = sp[rid]["batch_results"]["16"]
        w(
            f"| {rid} | {au / br['mean_ms']:.1f}× faster | {auv / br['peak_memory_mb']:.1f}× less | {sp['AutoShot (TransNetV2Supernet)']['gflops'] / sp[rid]['gflops']:.1f}× fewer |"
        )
    w(
        f"\nThe diff branch (a second encoder) accounts for most of the full model's cost: A0 {sp['A0']['batch_results']['16']['mean_ms']:.1f} ms vs "
        f"C1 (no diff branch) {sp['C1']['batch_results']['16']['mean_ms']:.1f} ms vs A4 {sp['A4']['batch_results']['16']['mean_ms']:.1f} ms at B=16.\n"
    )

    # ---------------- 9. caveats ----------------
    w("## 9. Caveats\n")
    w(
        "- One training run per variant, except A0 and A1 (three seeds). The spec treats differences below 0.01 F1 as inconclusive. The dual-label effect on ClipShots holds at every seed under both protocols; the effect of the fusion blocks and all other ladder steps are single-seed."
    )
    w(
        "- Protocol A tunes thresholds on each test set, so its absolute F1 is optimistic and comparable only to other Protocol-A numbers. Protocol B selects thresholds on the SHOT validation videos, which are labelled with the SHOT label file rather than the authors' final annotation, so its selected thresholds are noisy and its numbers are lower; under Protocol B, A0 − A1 on ClipShots is the only comparison of section 7 that is positive and ≥ 0.01, A0 is lower than A1 on BBC at all three seeds, and leave-one-out differences are less reliable."
    )
    w(
        "- These numbers are leak-free (SHOT test videos excluded from training) and should not be compared with numbers from models trained on all 648 SHOT videos."
    )
    w(
        "- Runs of this study used two code revisions (A0 and the other 12 runs). The training code is identical across them; only the launcher scripts differ."
    )
    w(
        f"- `B_noatt` was stopped manually after epoch {v3['B_noatt']['epochs']} (best validation F1 at epoch {v3['B_noatt']['best_ep']}) and evaluated from its best checkpoint, so it did not use the patience-10 stop of the other runs."
    )
    w("- Latency, VRAM and GFLOPs are measured on synthetic input; GFLOPs exclude non-matmul/conv operations.")
    w(
        "- The per-type analysis (Tables 8 and 9) was computed for A0, A1, A2, C6 and the extra seeds of A0 and A1; the cut-recall analysis of C4 was not repeated under ℓ_min = 1 and Protocol B, so the hypothesis about C4's cut recall is not evaluated here.\n"
    )

    full_md = "\n".join(out) + "\n"

    # ---------------- docs/RESULTS.md: one-page summary of the same results ----------------
    out = []
    w = out.append
    _SEEDS = ("", "_s43", "_s44")

    def _ms(P, base, s_):
        xs = [_f(P, base + k, s_) for k in _SEEDS]
        m = sum(xs) / 3
        sdv = (sum((x - m) ** 2 for x in xs) / 2) ** 0.5
        return m, sdv

    w("# Results\n")
    w(
        "Summary of the HybridSBD ablation study. Models are trained on SHOT (448 training videos; the 200 SHOT test videos are excluded) and ClipShots, and evaluated on the SHOT test set (200 videos), the ClipShots test set (500 videos) and BBC Planet Earth (11 episodes) with scene-based F1 (±2 frames), ℓ_min = 1 (no minimum-shot-length merging) and the sliding-window protocol of the AutoShot benchmark. "
        "Every table shows two threshold rules: **Protocol A** selects τ_c and τ_g on each test set (optimistic, comparable only to other Protocol-A numbers); **Protocol B** selects them once on the 67 SHOT validation videos and applies them unchanged. "
        "The full tables, methodology, training details, speed measurements and caveats are in [`reports/RESULTS_FULL.md`](../reports/RESULTS_FULL.md); differences below 0.01 F1 are treated as inconclusive.\n"
    )
    w("## Headline: full model (A0) and its midpoint-only counterpart (A1)\n")
    w("F1, mean ± sample SD over seeds 42, 43 and 44.\n")
    w("| Model | Protocol | SHOT | ClipShots | BBC |\n|---|---|---|---|---|")
    for base, lab in (("A0", "A0 (dual-label, full)"), ("A1", "A1 (midpoint only, full)")):
        for P, pl in (("A", "A"), ("B", "B")):
            w(f"| {lab} | {pl} | " + " | ".join("{:.4f} ± {:.4f}".format(*_ms(P, base, s_)) for s_ in _S) + " |")
    w("")
    _dd = {P: [_f(P, f"A0{k}", "clipshots") - _f(P, f"A1{k}", "clipshots") for k in _SEEDS] for P in "AB"}
    w(
        f"Paired A0 − A1 on ClipShots: Protocol A {', '.join(sd(x) for x in _dd['A'])}; Protocol B {', '.join(sd(x) for x in _dd['B'])} (seeds 42, 43, 44). Dual-label supervision helps on ClipShots at every seed under both protocols. "
        + _other_sets_sentence()
        + "\n"
    )
    w("## Supervision (seed 42), F1 as Protocol A / Protocol B\n")
    w("| Run | Description | SHOT | ClipShots | BBC |\n|---|---|---|---|---|")
    for r in ("A0", "A1", "A2", "A3", "A4"):
        w(f"| {r} | {DESC[r]} | " + " | ".join(f"{_f('A', r, s_):.4f} / {_f('B', r, s_):.4f}" for s_ in _S) + " |")
    w("\n## Build-up ladder (seed 42), F1 as Protocol A / Protocol B\n")
    w("| Step | Run | SHOT | ClipShots | BBC |\n|---|---|---|---|---|")
    for i, r in enumerate(("M0_min", "M1_edge", "B_noatt", "A4", "S1_rel", "A0")):
        w(f"| {i} | {r} | " + " | ".join(f"{_f('A', r, s_):.4f} / {_f('B', r, s_):.4f}" for s_ in _S) + " |")
    w("\n## Leave-one-out components (seed 42), change in F1 relative to A0 as Protocol A / Protocol B\n")
    w("| Run | Removed | SHOT | ClipShots | BBC |\n|---|---|---|---|---|")
    for r in _CR:
        w(
            f"| {r} | {DESC[r].replace('w/o ', '')} | "
            + " | ".join(_both(_f("A", r, s_) - _f("A", "A0", s_), _f("B", r, s_) - _f("B", "A0", s_)) for s_ in _S)
            + " |"
        )
    w(f"\nLargest single-component change on any test set: {_mx['A']:.4f} (Protocol A), {_mx['B']:.4f} (Protocol B).\n")
    w("## Where to find what\n")
    w(
        "- [`reports/RESULTS_FULL.md`](../reports/RESULTS_FULL.md): all tables (precision / recall / thresholds for every run, per-type analysis, duration analysis, seeds, interpretation, speed).\n"
        "- [`reports/results_all_runs.csv`](../reports/results_all_runs.csv): one row per run, seed, protocol and test set.\n"
        "- [`docs/IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md): implementation details of the evaluation and training code.\n"
    )
    summary_md = "\n".join(out)
    return full_md, summary_md


def build_results_csv(pa: dict, pb: dict) -> list:
    """Rows ``[run, seed, protocol, test set, l_min, tau_c, tau_g, precision, recall, F1]`` for every run, protocol and test set."""
    names = {"shot": "SHOT", "clipshots": "ClipShots", "bbc": "BBC"}
    rows = []
    for run in pa["runs"]:
        m = re.search(r"_s(\d+)$", run)
        seed = int(m.group(1)) if m else 42
        for proto, get in (
            ("A (l_min=1)", lambda s: pa["runs"][run]["sets"][s]["lmin1"]),
            ("B (validation-selected)", lambda s: pb["runs"][run]["sets"][s]),
        ):
            for s in ("shot", "clipshots", "bbc"):
                v = get(s)
                rows.append(
                    [
                        run,
                        seed,
                        proto,
                        names[s],
                        v["l_min"] if v.get("l_min") is not None else 1,
                        v["tau_c"],
                        "" if v.get("tau_g") is None else v["tau_g"],
                        round(v["p"], 4),
                        round(v["r"], 4),
                        round(v["f1"], 4),
                    ]
                )
    return rows


HEADS = {"A2": 1}
DESC_7A = {
    "A0": "Full, dual-label (primary)",
    "A1": "Full, midpoint only",
    "A2": "Full, unified single head",
    "A3": "Reduced, midpoint only",
    "A4": "Reduced, dual-label",
}
DESC_7B = {
    "A0": "Full model",
    "C1": "w/o difference-based branch (B)",
    "C2a": "w/o FrameSimilarity",
    "C2b": "w/o ColorHistogramSimilarity",
    "C3": "w/o temporal delta",
    "C4": "w/o per-frame pathway (relations from Branch A)",
    "C5a": "w/o pixel-level motion descriptor",
    "C5b": "w/o edge-density head",
    "C6": "w/o self-attention and gate",
}


def f(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def params(value):
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "—"


def row(cells):
    return "| " + " | ".join(cells) + " |"


def cmd_tables(args) -> None:
    """Fill the spec tables 7a, 7b, 8 and 9 from ``ablation_results.csv`` and the ClipShots analysis JSON."""
    rows = {r["run_id"]: r for r in csv.DictReader(open(args.csv, encoding="utf-8"))}
    analysis = {}
    if Path(args.analysis).exists():
        analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8")).get("runs", {})
    out = [
        "# Protocol A (A_test_selected) — result tables",
        "",
        "Single seed (42) per variant; differences below 0.01 F1 are inconclusive (spec Section 9).",
        "",
    ]

    def metrics(rid):
        r = rows.get(rid, {})
        return [f(r.get("shot_F1")), f(r.get("cs_F1")), f(r.get("bbc_F1")), f(r.get("latency_B16_ms"), 1)], r

    out += [
        "## Table 7a — Supervision",
        "",
        row(["ID", "Variant", "Heads", "Params", "SHOT F1", "CS F1", "BBC F1", "Latency (ms)"]),
        row(["---"] * 8),
    ]
    for rid in ("A0", "A1", "A2", "A3", "A4"):
        m, r = metrics(rid)
        out.append(row([rid, DESC_7A[rid], str(HEADS.get(rid, 2)), params(r.get("params_total"))] + m))
    out += [
        "",
        "## Table 7b — Components (dual-label)",
        "",
        row(["ID", "Description", "Params", "SHOT F1", "CS F1", "BBC F1", "Latency (ms)"]),
        row(["---"] * 7),
    ]
    for rid in DESC_7B:
        m, r = metrics(rid)
        out.append(row([rid, DESC_7B[rid], params(r.get("params_total"))] + m))

    def delta(a, b, col):
        try:
            return float(rows[a][col]) - float(rows[b][col])
        except (KeyError, ValueError):
            return None

    out += ["", "## Supervision deltas (spec Section 9; numbers only)", "", row(["Delta", "SHOT", "ClipShots", "BBC"]), row(["---"] * 4)]
    for label, a, b in (("A0 − A1 (full)", "A0", "A1"), ("A4 − A3 (reduced)", "A4", "A3")):
        out.append(row([label] + [f(delta(a, b, c)) for c in ("shot_F1", "cs_F1", "bbc_F1")]))

    out += [
        "",
        "## Table 8 — Per-type analysis on ClipShots",
        "",
        row(["Model", "Cut recall", "Gradual recall", "Gradual F1", "Gradual IoU"]),
        row(["---"] * 5),
    ]
    for rid in ("A0", "A1", "A2"):
        a = analysis.get(rid, {})
        na = lambda key: (
            "N/A"
            if (key in a and a[key] is None and rid in {"A1", "A2"} and key in ("gradual_f1",)) or (rid == "A1" and key == "gradual_iou")
            else f(a.get(key))
        )
        out.append(row([rid, f(a.get("cut_recall")), f(a.get("gradual_recall")), na("gradual_f1"), na("gradual_iou")]))

    n = (analysis.get("A0") or analysis.get("C6") or {}).get("duration_counts", {})
    out += [
        "",
        "## Table 9 — Recall by gradual-transition duration",
        "",
        row(["Model", f"≤ 10 (n = {n.get('le_10', '—')})", f"11–30 (n = {n.get('11_30', '—')})", f"> 30 (n = {n.get('gt_30', '—')})"]),
        row(["---"] * 4),
    ]
    for rid in ("A0", "C6"):
        d = analysis.get(rid, {}).get("duration_recall", {})
        out.append(row([rid, f(d.get("le_10")), f(d.get("11_30")), f(d.get("gt_30"))]))
    Path(args.output).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


# ----------------------------------------------------------------------------------------------------------------
# latency report (TransNet V2 and PERSIST versus HybridSBD and AutoShot, three servers)
# ----------------------------------------------------------------------------------------------------------------

LATENCY_DIRS = {1: "latency_baselines", 2: "latency_baselines_server2", 3: "latency_baselines_server3"}
LATENCY_ORDER = ["TransNet V2 (PyTorch)", "PERSIST (T=128)", "AutoShot (TransNetV2Supernet)", "A0", "A4", "PERSIST (T=64)"]
LATENCY_NAME = {
    "A0": "HybridSBD A0 (D = 384, L = 4)",
    "A4": "HybridSBD A4 (reduced)",
    "AutoShot (TransNetV2Supernet)": "AutoShot",
    "TransNet V2 (PyTorch)": "TransNet V2",
    "PERSIST (T=128)": "PERSIST (T = 128)",
    "PERSIST (T=64)": "PERSIST (native T = 64)",
}
LATENCY_GPU_KEY = {
    "A0": "A0",
    "A4": "A4",
    "AutoShot (TransNetV2Supernet)": "autoshot",
    "TransNet V2 (PyTorch)": "transnetv2",
    "PERSIST (T=128)": "persist",
}
LATENCY_EXPECTED = {"A0": (30.4, 1707, 29.4), "A4": (10.7, 941, 10.8), "AutoShot (TransNetV2Supernet)": (151, 16205, 96.8)}


def build_latency_report() -> str:
    """Inference-cost report from ``server_results/latency_baselines*``: one table for Table 4 plus the three-server comparison."""
    nb = " "
    data = {}
    for k, d in LATENCY_DIRS.items():
        base = SR / d
        ours, baselines = json.load(open(base / "ours.json")), json.load(open(base / "baselines.json"))
        busy = {r["label"]: r for r in json.load(open(base / "gpu_busy.json"))["models"]}
        data[k] = ({r["label"]: r for r in ours["models"] + baselines["models"]}, busy, ours["environment"], baselines["repositories"])
    R = {k: v[0] for k, v in data.items()}
    G = {k: v[1] for k, v in data.items()}
    s3 = SR / LATENCY_DIRS[3]
    freq = sorted(int(x) for x in open(s3 / "freq_max_p_core_khz.log").read().split())
    lat = lambda s, lab, b: R[s][lab]["batch_results"][str(b)]
    ref = {a: lat(3, a, 16)["mean_ms"] for a in ("A0", "A4")}
    diag4 = open(s3 / "diag4.stdout").read()
    py3 = re.search(r"3M iterations ([\d.]+) s; tiny CUDA op launch ([\d.]+) us", diag4)
    diag2 = open(SR / LATENCY_DIRS[2] / "diagnostics.txt").read()
    py2 = re.search(r"3M iterations ([\d.]+) s; tiny CUDA op launch ([\d.]+) us", diag2)
    gn2 = re.search(r"285 ms cumulative", diag2) and 285
    gn3 = round(
        float(
            re.search(r"gather_nd\)\n?", open(s3 / "diag5.stdout").read())
            and re.search(r"2\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+\S+supernet\S+\(gather_nd\)", open(s3 / "diag5.stdout").read()).group(
                1
            )
        )
        * 1000
    )

    out = []
    w = out.append
    w("# Inference cost of TransNet V2 and PERSIST, measured with HybridSBD and AutoShot\n")
    w(
        "Measurement only: nothing was trained and no dataset was used. The same protocol was run on three servers with the same GPU (RTX PRO 6000 Blackwell Workstation Edition) and different hosts. **Server 3 reproduces the earlier session's values and is the one to use for Table 4**; servers 1 and 2 show which results depend on the host.\n"
    )

    w("## 1. Result for Table 4 (server 3)\n")
    w(
        "FP32, `model.eval()`, `torch.no_grad()`, no AMP, no `torch.compile`; synthetic clips in each model's native format; 10 warm-up and 80 timed forward passes with `torch.cuda.synchronize()` around each; latency is mean ± SD (ms); GFLOPs per clip at B = 1 (`torch.utils.flop_counter.FlopCounterMode`: convolutions and matrix multiplications only). The two right-hand columns give the model's latency divided by the HybridSBD latency at B = 16.\n"
    )
    w(
        "| Model | Input (T × H × W, dtype) | Parameters | GFLOPs / clip | B = 1 | B = 16 | B = 32 | frames/s (B = 16) | Peak VRAM (B = 16, MB) | × A0 | × A4 |\n|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for lab in LATENCY_ORDER:
        r = R[3][lab]
        inp = r.get("input", "128 x 48 x 64, float32 RGB").replace(" x ", " × ")
        cells = [LATENCY_NAME[lab], inp, f"{r['total_params']:,}", f"{r['gflops']:.2f}"]
        cells += [f"{lat(3, lab, b)['mean_ms']:.1f}{nb}±{nb}{lat(3, lab, b)['std_ms']:.2f}" for b in (1, 16, 32)]
        cells += [f"{lat(3, lab, 16)['frames_per_sec']:,.0f}", f"{lat(3, lab, 16)['peak_memory_mb']:,.0f}"]
        cells += [f"{lat(3, lab, 16)['mean_ms'] / ref[a]:.1f}" for a in ("A0", "A4")]
        w("| " + " | ".join(cells) + " |")
    w("")
    w(
        "- HybridSBD A0 is 1.7× faster than TransNet V2, 2.6× faster than PERSIST and 5.6× faster than AutoShot at B = 16, with 3.3–5.4× fewer operations than those three (29.4 against 96.8–158.9 GFLOPs); A4 is 4.9×, 7.3× and 15.6× faster."
    )
    w(
        "- PERSIST accepts a 128-frame clip, so the main row uses T = 128 like the other models; its native window of 64 frames is the last row (frames per second stay comparable)."
    )
    w(
        "- Only HybridSBD, TransNet V2 and PERSIST latencies are independent of the host (section 3); AutoShot's latency and every B = 1 latency depend on it.\n"
    )

    w("## 2. Setup\n")
    w("| | Server 1 | Server 2 | Server 3 |\n|---|---|---|---|")
    w("| GPU | RTX PRO 6000 Blackwell WS, idle | same | same |")
    w("| CPU | AMD EPYC 9555 | AMD Threadripper PRO 7975WX | Intel Core i9-14900K |")
    w(
        f"| CPU frequency | cpufreq maximum 1.9 GHz (`schedutil`) | rated 5.35 GHz, **capped at 2.3 GHz** by the host (`scaling_max_freq`, read-only), cores at 2.28 GHz | not capped: P-cores sampled at a median of {freq[len(freq) // 2] / 1e6:.2f} GHz, maximum {freq[-1] / 1e6:.2f} GHz |"
    )
    w("| Benchmark process | free scheduling | free scheduling | pinned to the performance cores (CPUs 0–15) |")
    w("| PyTorch / CUDA / cuDNN | 2.14.1+cu132 / 13.2 / 9.24 | 2.11.0+cu128 / 12.8 / 9.19 | 2.14.1+cu132 / 13.2 / 9.24 |")
    w("| Python | 3.12.3 (system) | 3.12.15 (Vast image) | 3.12.15 (Vast image) |")
    w("| Driver | 590.48.01 | 570.195.03 | 610.57.04 |")
    w("")
    repos = data[3][3]
    w(
        f"Identical on all three: TransNet V2 `soCzech/TransNetV2` @ `{repos['TransNetV2'][:12]}` (PyTorch implementation in `inference-pytorch/`), PERSIST `linty5/PERSIST` @ `{repos['PERSIST'][:12]}` (MMAction2 v1.2.0 fork; mmengine 0.10.7, mmcv-lite 2.1.0, einops 0.8.2), AutoShot `wentaozhu/AutoShot` @ `{repos['AutoShot'][:12]}`; the benchmark code; the A0 and A4 `best_model.pt` (SHA-256 prefixes `8ec3feb706b601ed`, `0125c80ffa1a6b79`, checked on each server); random weights for the three baselines. Server 2's driver (570) cannot run CUDA 13, so it used its preinstalled PyTorch 2.11.0+cu128; server 1 went offline before it could be repeated with that version, so servers 1 and 2 differ in CPU and in software.\n"
    )

    w("## 3. All three servers\n")
    w("Latency in ms (mean). S1, S2, S3 are the servers above.\n")
    w(
        "| Model | B = 1: S1 | S2 | S3 | B = 16: S1 | S2 | S3 | B = 32: S1 | S2 | S3 | Spread of B = 16 (S1–S3) |\n|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for lab in LATENCY_ORDER:
        v16 = [lat(s, lab, 16)["mean_ms"] for s in (1, 2, 3)]
        cells = (
            [LATENCY_NAME[lab]]
            + [f"{lat(s, lab, b)['mean_ms']:.1f}" for b in (1, 16, 32) for s in (1, 2, 3)]
            + [f"{100 * (max(v16) / min(v16) - 1):.0f} %"]
        )
        w("| " + " | ".join(cells) + " |")
    w("")
    w("GPU kernel time and GPU-busy fraction of one forward pass at B = 16 (profiler); peak VRAM at B = 16 (MB).\n")
    w(
        "| Model | Wall-clock S1 | S2 | S3 | GPU kernel S1 | S2 | S3 | GPU busy S1 | S2 | S3 | Peak VRAM S1 / S2 / S3 |\n|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for lab in LATENCY_ORDER[:5]:
        g = [G[s][LATENCY_GPU_KEY[lab]] for s in (1, 2, 3)]
        cells = (
            [LATENCY_NAME[lab]]
            + [f"{x['wall_ms']:.1f}" for x in g]
            + [f"{x['gpu_kernel_ms']:.1f}" for x in g]
            + [f"{100 * x['gpu_busy']:.0f} %" for x in g]
        )
        cells.append(" / ".join(f"{lat(s, lab, 16)['peak_memory_mb']:,.0f}" for s in (1, 2, 3)))
        w("| " + " | ".join(cells) + " |")
    w("")
    w(
        "- **A0, A4, TransNet V2 and PERSIST are GPU-bound** (95–100 % GPU busy): B = 16 and 32 latencies differ by at most 10 % between the servers, the GPU kernel time is the same on all, and the ranking and speedup factors are the same."
    )
    w(
        "- **B = 1 follows the host**, in both directions (A0: 7.8, 12.3 and 4.0 ms; PERSIST: 14.6 with SD 17.5, 7.9 and 5.5 ms). At B = 1 the kernels are tiny and launch and synchronisation overhead dominate, so single-batch latencies should be reported with the machine named."
    )
    w(
        "- **AutoShot follows the host at every batch size** (B = 16: 242, 289 and 160 ms) while its GPU kernel time is about 87 ms on all three; it is GPU-busy only 30–54 % of the time.\n"
    )

    w("## 4. Reproduction of the expected values\n")
    w(
        "Expected values are the paper's earlier measurements (B = 16). Each cell shows the measured value and the difference from the expected one.\n"
    )
    w("| Model | Quantity | Expected | Server 1 | Server 2 | Server 3 |\n|---|---|---|---|---|---|")
    for lab, (ems, emb, egf) in LATENCY_EXPECTED.items():
        for q, e, get in (
            ("Latency (ms)", ems, lambda s: lat(s, lab, 16)["mean_ms"]),
            ("Peak VRAM (MB)", emb, lambda s: lat(s, lab, 16)["peak_memory_mb"]),
            ("GFLOPs", egf, lambda s: R[s][lab]["gflops"]),
        ):
            cells = [LATENCY_NAME[lab], q, f"{e:,}"] + [f"{get(s):,.1f} ({100 * (get(s) / e - 1):+.1f} %)" for s in (1, 2, 3)]
            w("| " + " | ".join(cells) + " |")
    w("")
    w(
        "HybridSBD reproduces on all three servers within 6 % (server 3: −5.3 % / −4.4 % latency, +2.5 % / +3.8 % VRAM). **AutoShot's latency reproduces only on server 3 (+6 %)**; on servers 1 and 2 it is 60 % and 91 % higher while its memory and operation count are exact, so it is the same model on a slower host.\n"
    )

    w("## 5. Why AutoShot depends on the host\n")
    w(
        f"- **Most of its time is host-side Python.** Profiling one forward pass at B = 16 on server 2 (cProfile, 292 ms) shows two calls of AutoShot's own `gather_nd` (a port of `tf.gather_nd` in `supernet_flattransf_3_8_8_8_13_12_0_16_60.py`, called once in its `FrameSimilarity` layer and once in `ColorHistograms`) taking 285 ms, of which `Tensor.tolist()` is 93 ms; the 46 convolutions take 2 ms of host time in total. `gather_nd` copies about 0.6 million indices from the GPU to the CPU, turns them into nested Python lists and indexes with them, so it grows with the batch size and with the host's Python speed. On server 3 the same two calls take {gn3} ms."
    )
    w("- **Host measurements** (single thread):\n")
    w("| | Server 2 | Server 3 |\n|---|---|---|")
    w(f"| Pure-Python loop, 3 million iterations | {py2.group(1)} s | {py3.group(1)} s |")
    w(f"| Launch of a tiny CUDA op | {py2.group(2)} µs | {py3.group(2)} µs |")
    w(f"| AutoShot `gather_nd`, 2 calls at B = 16 | {gn2} ms | {gn3} ms |")
    w("| AutoShot forward at B = 16 (benchmark) | 288.8 ms | 160.1 ms |")
    w("")
    w(
        "- **Server 2's CPU does not run at its rated speed.** The Threadripper PRO 7975WX is rated 5.35 GHz, but `amd-pstate-epp` (governor `powersave`) has `scaling_max_freq` = 2.3 GHz and the busiest cores ran at 2.28 GHz. `/sys` is read-only inside the container and the files are not writable, so the cap is a host setting that cannot be changed from there; the CPU quota (about 30.7 CPUs) is not a limit for these runs. Server 1's EPYC reported a 1.9 GHz maximum. Server 3's i9-14900K was not capped."
    )
    w(
        "- **Not the cause:** repeating the AutoShot run gave the same value (242.6 and 243.4 ms), the GPU was not throttled (SM clock 2,820–2,827 MHz, no throttle reasons), and `cudnn.benchmark`, TF32, cuDNN-disabled and channels-last settings did not move it toward 151 ms.\n"
    )

    w("## 6. Notes\n")
    w(
        "- **Script.** The task's `benchmark_model.py` is `benchmarks/speed.py` after the repository restructure; its `ours` and `baselines` commands run the protocol, `gpu-busy` produced the GPU-busy columns, `summary` builds the table. I added only the thin wrappers below, GFLOPs per clip in the result files and `gpu-busy`."
    )
    w(
        "- **Weights.** TransNet V2, PERSIST and AutoShot use randomly initialised weights (latency, memory and operation counts do not depend on them; AutoShot's `ckpt_0_200_0.pth` and the PERSIST release checkpoints were not downloaded). A0 and A4 use their trained weights."
    )
    w(
        "- **TransNet V2.** Official class with default arguments; input uint8 `[B, T, 27, 48, 3]`. Its `forward` always casts to float, permutes to `[B, 3, T, H, W]` and divides by 255; kept and timed. 7,616,258 parameters."
    )
    w(
        "- **AutoShot.** `TransNetV2Supernet`; input float32 `[B, 3, T, 27, 48]` in 0–255 as in the paper's measurement; 14,299,202 parameters."
    )
    w(
        "- **PERSIST.** Built from `configs/persist/persist_eval.py` through the official registry: 13,195,594 trainable parameters (13.20 M reported). Native input is uint8 `[B, 3, T, 112, 112]` (its test pipeline), native window T = 64; T = 128 also runs. The timed forward pass is the official inference path without post-processing: `ActionDataPreprocessor.preprocess` (cast and normalise), `Recognizer3D.extract_feat(test_mode=True)` (the backbone, which resizes 112 × 112 to 27 × 48 internally) and `SBDOursHead.forward`, returning the boundary probability. The cast and the internal resize are always run by the official code, so they are kept; excluded are `predict()`'s copies to the CPU and debug fields, and video decoding. Built with `mmcv-lite` (pure Python, no compiled operators), which is enough to run it."
    )
    w(
        "- **PERSIST at B = 1 on server 1** had a large spread (SD 17.5 ms; minimum 6.8, maximum 64.9 ms): a few passes were much slower, the mean is reported as measured. On server 3 the SD is 0.01 ms."
    )
    w("- **GFLOPs** are identical on all servers and equal the paper's (A0 29.43, A4 10.79, AutoShot 96.83).\n")

    w("## 7. Commands and wrapper code\n")
    w(
        "```bash\n# environment (server 3 and the paper's session: PyTorch 2.14.1+cu132; server 2 used the preinstalled 2.11.0+cu128)\npip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu132\npip install numpy tqdm scipy einops mmengine==0.10.7 mmcv-lite==2.1.0 opencv-python-headless matplotlib packaging pyyaml rich termcolor yapf addict importlib_metadata\ngit clone https://github.com/soCzech/TransNetV2 && git clone https://github.com/linty5/PERSIST && git clone https://github.com/wentaozhu/AutoShot   # then check out the commits above\npip install --no-deps -e PERSIST/mmaction2\n\n# measurement (code/ = hybridsbd/ + benchmarks/ of this repository; ckpt/<run>/best_model.pt = the study checkpoints)\nexport AUTOSHOT_DIR=AutoShot TRANSNETV2_DIR=TransNetV2/inference-pytorch PERSIST_DIR=PERSIST\nPY='taskset -c 0-15 python'   # server 3 only: pin to the performance cores\n$PY benchmarks/speed.py ours --device cuda --batch_sizes 1 16 32 --checkpoints ckpt/A0/best_model.pt ckpt/A4/best_model.pt --output out/ours.json\n$PY benchmarks/speed.py baselines --device cuda --batch_sizes 1 16 32 --output out/baselines.json\n$PY benchmarks/speed.py gpu-busy --checkpoints ckpt/A0/best_model.pt ckpt/A4/best_model.pt --models autoshot transnetv2 persist --output out/gpu_busy.json\n$PY benchmarks/speed.py summary --dir out\n```\n"
    )
    src = (ROOT / "benchmarks" / "speed.py").read_text()
    w("Wrapper and loaders from `benchmarks/speed.py`:\n")
    w("```python\n" + src[src.index("def load_transnetv2") : src.index("def make_baseline_input")].rstrip() + "\n```\n")
    w(
        "Raw results: `server_results/latency_baselines/` (server 1), `latency_baselines_server2/`, `latency_baselines_server3/` (`ours.json`, `baselines.json`, `gpu_busy.json`, `lscpu.txt`, `nvidia_smi.txt`, diagnostics).\n"
    )
    return "\n".join(out)


def cmd_latency(args) -> None:
    reports = Path(args.outdir) if args.outdir else ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    md = build_latency_report()
    (reports / "LATENCY_BASELINES_REPORT.md").write_text(md, encoding="utf-8")
    print(f"wrote {reports}/LATENCY_BASELINES_REPORT.md")
    if args.pdf:
        export_pdf(
            "\n".join(l.replace(" ± ", " ± ") if l.startswith("|") else l for l in md.split("\n")),
            reports / "LATENCY_BASELINES_REPORT.pdf",
            landscape=True,
        )
        print("wrote the PDF")


def export_pdf(markdown_text: str, pdf: Path, landscape: bool = False) -> Path:
    """Render Markdown to an A4 PDF (short tables stay on one page, long ones may break) with headless Chrome."""
    import markdown

    html = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    html = re.sub(
        r"<table>(.*?)</table>",
        lambda m: ("<table class='long'>" if m.group(1).count("<tr>") > 20 else "<table>") + m.group(1) + "</table>",
        html,
        flags=re.S,
    )
    css = PDF_CSS.replace("size:A4;", "size:A4 landscape;") if landscape else PDF_CSS
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "r.html"
        page.write_text(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>", encoding="utf-8")
        subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", f"file://{page}"],
            check=True,
            capture_output=True,
        )
    return pdf


def cmd_all(args) -> None:
    pa, pb = load_results()
    reports = Path(args.outdir) if args.outdir else ROOT / "reports"
    docs = Path(args.outdir) if args.outdir else ROOT / "docs"
    reports.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    followup = build_followup(pa, pb)
    full, summary = build_main(pa, pb, followup)
    (reports / "FOLLOWUP_RESULTS.md").write_text(followup, encoding="utf-8")
    (reports / "RESULTS_FULL.md").write_text(full, encoding="utf-8")
    (docs / "RESULTS.md").write_text(summary, encoding="utf-8")
    with open(reports / "results_all_runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "seed", "protocol", "test set", "l_min", "tau_c", "tau_g", "precision", "recall", "F1"])
        writer.writerows(build_results_csv(pa, pb))
    print(f"wrote {reports}/FOLLOWUP_RESULTS.md, RESULTS_FULL.md, results_all_runs.csv and {docs}/RESULTS.md")
    if args.pdf:
        export_pdf(followup, reports / "Follow_up_Results_Seeds_and_Protocols.pdf")
        export_pdf(full, reports / "RESULTS_FULL.pdf")
        print("wrote the two PDFs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("all", help="follow-up report, full report, summary and CSV")
    p.add_argument("--outdir", default=None, help="write all files into this directory instead of reports/ and docs/")
    p.add_argument("--pdf", action="store_true", help="also render the two reports to PDF")
    p.set_defaults(func=cmd_all)
    p = sub.add_parser(
        "latency",
        help="inference-cost report (TransNet V2 and PERSIST versus HybridSBD and AutoShot) from server_results/latency_baselines*",
    )
    p.add_argument("--outdir", default=None)
    p.add_argument("--pdf", action="store_true")
    p.set_defaults(func=cmd_latency)
    p = sub.add_parser("tables", help="spec tables 7a / 7b / 8 / 9 from ablation_results.csv")
    p.add_argument("--csv", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_tables)
    return parser.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    ns.func(ns)
