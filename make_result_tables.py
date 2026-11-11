"""Fill spec Section 8 tables (7a, 7b, 8, 9) from the CSV and the ClipShots analysis JSON.

Pure formatting: every number is read from files produced by the pipeline; missing values are shown as '—'.
"""
import argparse
import csv
import json
from pathlib import Path

HEADS = {"A2": 1}
DESC_7A = {"A0": "Full, dual-label (primary)", "A1": "Full, midpoint only", "A2": "Full, unified single head",
           "A3": "Reduced, midpoint only", "A4": "Reduced, dual-label"}
DESC_7B = {"A0": "Full model", "C1": "w/o difference-based branch (B)", "C2a": "w/o FrameSimilarity",
           "C2b": "w/o ColorHistogramSimilarity", "C3": "w/o temporal delta",
           "C4": "w/o per-frame pathway (relations from Branch A)", "C5a": "w/o pixel-level motion descriptor",
           "C5b": "w/o edge-density head", "C6": "w/o self-attention and gate"}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = {r["run_id"]: r for r in csv.DictReader(open(args.csv, encoding="utf-8"))}
    analysis = {}
    if Path(args.analysis).exists():
        analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8")).get("runs", {})
    out = ["# Protocol A (A_test_selected) — result tables", "",
           "Single seed (42) per variant; differences below 0.01 F1 are inconclusive (spec Section 9).", ""]

    def metrics(rid):
        r = rows.get(rid, {})
        return [f(r.get("shot_F1")), f(r.get("cs_F1")), f(r.get("bbc_F1")), f(r.get("latency_B16_ms"), 1)], r

    out += ["## Table 7a — Supervision", "",
            row(["ID", "Variant", "Heads", "Params", "SHOT F1", "CS F1", "BBC F1", "Latency (ms)"]),
            row(["---"] * 8)]
    for rid in ("A0", "A1", "A2", "A3", "A4"):
        m, r = metrics(rid)
        out.append(row([rid, DESC_7A[rid], str(HEADS.get(rid, 2)), params(r.get("params_total"))] + m))
    out += ["", "## Table 7b — Components (dual-label)", "",
            row(["ID", "Description", "Params", "SHOT F1", "CS F1", "BBC F1", "Latency (ms)"]), row(["---"] * 7)]
    for rid in DESC_7B:
        m, r = metrics(rid)
        out.append(row([rid, DESC_7B[rid], params(r.get("params_total"))] + m))

    def delta(a, b, col):
        try:
            return float(rows[a][col]) - float(rows[b][col])
        except (KeyError, ValueError):
            return None
    out += ["", "## Supervision deltas (spec Section 9; numbers only)", "",
            row(["Delta", "SHOT", "ClipShots", "BBC"]), row(["---"] * 4)]
    for label, a, b in (("A0 − A1 (full)", "A0", "A1"), ("A4 − A3 (reduced)", "A4", "A3")):
        out.append(row([label] + [f(delta(a, b, c)) for c in ("shot_F1", "cs_F1", "bbc_F1")]))

    out += ["", "## Table 8 — Per-type analysis on ClipShots", "",
            row(["Model", "Cut recall", "Gradual recall", "Gradual F1", "Gradual IoU"]), row(["---"] * 5)]
    for rid in ("A0", "A1", "A2"):
        a = analysis.get(rid, {})
        na = lambda key: "N/A" if (key in a and a[key] is None and rid in {"A1", "A2"} and key in ("gradual_f1",)) or \
            (rid == "A1" and key == "gradual_iou") else f(a.get(key))
        out.append(row([rid, f(a.get("cut_recall")), f(a.get("gradual_recall")), na("gradual_f1"), na("gradual_iou")]))

    n = (analysis.get("A0") or analysis.get("C6") or {}).get("duration_counts", {})
    out += ["", "## Table 9 — Recall by gradual-transition duration", "",
            row(["Model", f"≤ 10 (n = {n.get('le_10', '—')})", f"11–30 (n = {n.get('11_30', '—')})",
                 f"> 30 (n = {n.get('gt_30', '—')})"]), row(["---"] * 4)]
    for rid in ("A0", "C6"):
        d = analysis.get(rid, {}).get("duration_recall", {})
        out.append(row([rid, f(d.get("le_10")), f(d.get("11_30")), f(d.get("gt_30"))]))
    Path(args.output).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
