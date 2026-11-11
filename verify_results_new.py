"""Independent check of RESULTS_NEW.md: parse the numbers in the markdown tables and compare each one to the RAW result files
(shot/clipshots/bbc_results.json, history.json, config.json, benchmark / speed JSONs, RESULTS.md). Does not import make_results_new.py.
   python verify_results_new.py"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SR = ROOT / "server_results"
md = (ROOT / "RESULTS_NEW.md").read_text().splitlines()
fails, n_ok = [], 0


def check(name, got, want, tol=0.00006):
    global n_ok
    try:
        ok = abs(float(got) - float(want)) <= tol
    except Exception:
        ok = str(got).strip() == str(want).strip()
    if ok:
        n_ok += 1
    else:
        fails.append(f"{name}: md={got} file={want}")


def section(title_prefix):
    start = next(i for i, l in enumerate(md) if l.startswith(title_prefix))
    end = next((i for i in range(start + 1, len(md)) if md[i].startswith("## ") or (title_prefix.startswith("###") and md[i].startswith("### "))), len(md))
    return md[start:end]


def rows(lines):
    out = []
    for l in lines:
        if l.startswith("|") and not re.match(r"^\|[-| ]+\|$", l):
            out.append([c.strip() for c in l.strip().strip("|").split("|")])
    return out[1:]  # drop header


def num(x):
    return float(re.match(r"[-+]?[\d.,]+", x.replace("×", "").replace("*", "").strip()).group(0).replace(",", ""))


def raw(run_dir):
    r = {}
    for k, f in (("shot", "shot"), ("cs", "clipshots"), ("bbc", "bbc")):
        d = json.load(open(run_dir / f"{f}_results.json"))
        r[k] = d["overall"]
        r[k + "_cfg"] = d
    r["hist"] = json.load(open(run_dir / "history.json"))
    r["cfg"] = json.load(open(run_dir / "config.json"))
    return r


V2, V3 = SR / "ablations_v2", SR / "ablations_v3"
bench = {m["label"]: m for m in json.load(open(V2 / "benchmark_results.json"))["models"]}
runs = {r: raw(V2 / r) for r in ("A0", "A1", "A2", "A3", "A4", "C1", "C2a", "C2b", "C3", "C4", "C5a", "C5b", "C6")}
runs.update({r: raw(V3 / r) for r in ("M0_min", "M1_edge", "B_noatt", "S1_rel")})

# ---- 3.1 / 3.2 main ablation tables
for tab in ("### 3.1", "### 3.2"):
    for r in rows(section(tab)):
        rid = r[0]
        if len(r) != (8 if tab == "### 3.1" else 7) or " " in rid:
            continue
        if tab == "### 3.1":
            params, shot, cs, bbc, lat = r[3], r[4], r[5], r[6], r[7]
        else:
            params, shot, cs, bbc, lat = r[2], r[3], r[4], r[5], r[6]
        d = runs[rid]
        check(f"{tab} {rid} params", num(params), d["cfg"]["parameters"][0], 0.5)
        check(f"{tab} {rid} SHOT", num(shot), d["shot"]["f1"])
        check(f"{tab} {rid} CS", num(cs), d["cs"]["f1"])
        check(f"{tab} {rid} BBC", num(bbc), d["bbc"]["f1"])
        check(f"{tab} {rid} latency", num(lat), bench[rid]["batch_results"]["16"]["mean_ms"], 0.06)
for r in rows(section("### 3.2")):
    if len(r) == 4 and r[0] in runs and r[0] != "A0":
        d = runs[r[0]]
        check(f"3.2 delta {r[0]} SHOT", num(r[1]), d["shot"]["f1"] - runs["A0"]["shot"]["f1"])
        check(f"3.2 delta {r[0]} CS", num(r[2]), d["cs"]["f1"] - runs["A0"]["cs"]["f1"])
        check(f"3.2 delta {r[0]} BBC", num(r[3]), d["bbc"]["f1"] - runs["A0"]["bbc"]["f1"])

# ---- 3.3 P/R and thresholds
for r in rows(section("### 3.3")):
    if r[0] not in runs:
        continue
    d = runs[r[0]]
    if len(r) == 4:
        for col, k in ((1, "shot"), (2, "cs"), (3, "bbc")):
            p_, rc = [float(x) for x in r[col].split("/")]
            check(f"3.3 {r[0]} {k} P", p_, d[k]["p"], 0.0006); check(f"3.3 {r[0]} {k} R", rc, d[k]["r"], 0.0006)
    elif len(r) == 6:
        check(f"3.3 {r[0]} shot tau_c", r[1].split(",")[0], d["shot_cfg"]["best_boundary_threshold"])
        check(f"3.3 {r[0]} cs tau_c", r[2].split(",")[0], d["cs_cfg"]["best_boundary_threshold"])
        check(f"3.3 {r[0]} cs lmin", r[2].split(",")[-1], d["cs_cfg"]["best_min_scene_len"])
        check(f"3.3 {r[0]} bbc tau_c", r[3].split(",")[0], d["bbc_cfg"]["best_boundary_threshold"])
        bst = max(d["hist"], key=lambda e: e["val"]["f1"])["epoch"]
        check(f"3.3 {r[0]} best/epochs", r[4], f"{bst} / {len(d['hist'])}")
        check(f"3.3 {r[0]} hours", r[5], d["hist"][-1]["elapsed_hours"], 0.006)

# ---- Table 8 / 9 vs raw analysis
an = json.load(open(V2 / "clipshots_protocol_a_analysis.json"))["runs"]
t89 = "\n".join(section("### 3.4"))
for rid in ("A0", "A1", "A2"):
    m = re.search(rf"\| {rid} \| ([\d.]+) \| ([\d.]+) \|", t89)
    check(f"T8 {rid} cut recall", m.group(1), an[rid]["cut_recall"])
    check(f"T8 {rid} grad recall", m.group(2), an[rid]["gradual_recall"])
for rid in ("A0", "C6"):
    m = re.findall(rf"\| {rid} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|\n?", t89)[-1]
    for g, v in zip(("le_10", "11_30", "gt_30"), m):
        check(f"T9 {rid} {g}", v, an[rid]["duration_recall"][g])

# ---- ladder
lad = {r[1]: r for r in rows(section("## 5."))[:7] if len(r) > 5 and r[0].isdigit()}
spd = {m["label"]: m for m in json.load(open(SR / "speed_benchmark" / "ours.json"))["models"]}
prev = None
for rid in ("M0_min", "M1_edge", "B_noatt", "A4", "S1_rel", "A0"):
    r, d = lad[rid], runs[rid]
    check(f"ladder {rid} params", num(r[3]), d["cfg"]["parameters"][0], 0.5)
    check(f"ladder {rid} SHOT", num(r[4]), d["shot"]["f1"]); check(f"ladder {rid} CS", num(r[5]), d["cs"]["f1"]); check(f"ladder {rid} BBC", num(r[6]), d["bbc"]["f1"])
    check(f"ladder {rid} latency", num(r[7]), spd[rid]["batch_results"]["16"]["mean_ms"], 0.06)
    prev = d
tall = "\n".join(md)
m = re.search(r"`B_noatt` was stopped manually after epoch (\d+) \(best validation F1 at epoch (\d+)\)", tall)
check("B_noatt epochs", m.group(1), len(runs["B_noatt"]["hist"]))
check("B_noatt best epoch", m.group(2), max(runs["B_noatt"]["hist"], key=lambda e: e["val"]["f1"])["epoch"])

# ---- speed table
auto = {m["label"]: m for m in json.load(open(SR / "speed_benchmark" / "autoshot.json"))["models"]}
spd.update({k: v for k, v in auto.items() if k.startswith("AutoShot")})
gf = {r["label"]: r["gflops"] for r in json.load(open(SR / "speed_benchmark" / "speed_summary.json"))}
sec6 = section("## 6.")
def table_after(title):
    i = next(k for k, l in enumerate(sec6) if l.startswith(title))
    j = i + 1
    while not sec6[j].startswith("|"):
        j += 1
    k = j
    while k < len(sec6) and sec6[k].startswith("|"):
        k += 1
    return rows(sec6[j:k])
for r in table_after("**Size, compute and peak VRAM**"):
    d = spd[r[0]]; b = d["batch_results"]
    check(f"speedA {r[0]} params", num(r[1]), d["total_params"] / 1e6, 0.006); check(f"speedA {r[0]} MB", num(r[2]), d["size_fp32_mb"], 0.06)
    check(f"speedA {r[0]} GFLOPs", num(r[3]), gf[r[0]], 0.006)
    for i, bs in ((4, "1"), (5, "16"), (6, "32")):
        check(f"speedA {r[0]} vram B={bs}", num(r[i]), b[bs]["peak_memory_mb"], 0.6)
for r in table_after("**Latency and throughput**"):
    b = spd[r[0]]["batch_results"]
    for i, bs in ((1, "1"), (2, "16"), (3, "32")):
        check(f"speedB {r[0]} ms B={bs}", num(r[i]), b[bs]["mean_ms"], 0.06)
    for i, bs in ((4, "1"), (5, "16"), (6, "32")):
        check(f"speedB {r[0]} fps B={bs}", num(r[i]), b[bs]["frames_per_sec"], 0.6)
a16 = auto["AutoShot (TransNetV2Supernet)"]["batch_results"]["16"]
for r in rows(section("## 6.")):
    if len(r) == 4 and r[0] in spd and "faster" in r[1]:
        d = spd[r[0]]["batch_results"]["16"]
        check(f"speedup {r[0]}", num(r[1]), a16["mean_ms"] / d["mean_ms"], 0.06)
        check(f"vram ratio {r[0]}", num(r[2]), a16["peak_memory_mb"] / d["peak_memory_mb"], 0.06)
        check(f"gflops ratio {r[0]}", num(r[3]), gf["AutoShot (TransNetV2Supernet)"] / gf[r[0]], 0.06)

# ---- 6.1 repro vs raw + vs RESULTS.md
rep = {m["label"]: m for m in json.load(open(SR / "speed_repro" / "ours_old_arch.json"))["models"]}
rep.update({m["label"]: m for m in json.load(open(SR / "speed_repro" / "baselines.json"))["models"]})
old = {}
for ln in (ROOT / "RESULTS.md").read_text().splitlines():
    c = [x.strip() for x in ln.strip().strip("|").split("|")]
    if len(c) == 9 and c[0].startswith("hybrid_detector") and c[4].isdigit():
        old[(c[0], int(c[4]))] = (float(c[3].rstrip("M")), float(c[5]), float(c[8].replace(",", "")))
    if len(c) == 7 and c[0].startswith("AutoShot") and c[2].isdigit():
        old[("AutoShot (TransNetV2Supernet)", int(c[2]))] = (float(c[1].rstrip("M")), float(c[3].replace(",", "")), float(c[6].replace(",", "")))
VAR = {"D = 512, 3 fusion blocks": "hybrid_detector13", "D = 256, 3 fusion blocks": "hybrid_detector15", "D = 384, 4 fusion blocks (A0 architecture)": "hybrid_detector16",
       "D = 256, 4 fusion blocks": "hybrid_detector17", "D = 512, 4 fusion blocks": "hybrid_detector18", "D = 384, 3 fusion blocks": "hybrid_detector19",
       "AutoShot (TransNetV2Supernet)": "AutoShot (TransNetV2Supernet)"}
for r in rows(section("### 6.1")):
    if len(r) == 6 and r[0] in VAR:
        n = VAR[r[0]]; b = rep[n]["batch_results"]
        check(f"6.1 {n} params old", float(r[1].split("→")[0]), old[(n, 16)][0], 0.006)
        check(f"6.1 {n} params new", float(r[1].split("→")[1]), rep[n]["total_params"] / 1e6, 0.006)
        for i, bs in ((2, 8), (3, 16), (4, 32)):
            o, nw = [float(x) for x in r[i].split("→")]
            check(f"6.1 {n} old ms B={bs}", o, old[(n, bs)][1], 0.06); check(f"6.1 {n} new ms B={bs}", nw, b[str(bs)]["mean_ms"], 0.06)
        o, nw = [float(x) for x in r[5].split("→")]
        check(f"6.1 {n} old vram", o, old[(n, 16)][2], 0.6); check(f"6.1 {n} new vram", nw, b["16"]["peak_memory_mb"], 0.6)

# ---- 2.1 compliance
arch = json.load(open(V2 / "a0_architecture_check.json"))
t2 = "\n".join(section("### 2.1"))
check("A0 params exact", re.search(r"Exact parameter count \| rounds to 10.9M \| ([\d,]+)", t2).group(1).replace(",", ""), arch["parameters_total_exact"])
for lab, k in (("SHOT", "shot"), ("ClipShots", "cs"), ("BBC", "bbc")):
    m = re.search(rf"\| {lab} F1 \| [\d.]+ \| [\d.]+ – [\d.]+ \| ([\d.]+) \|", t2)
    check(f"check2 {lab}", m.group(1), runs["A0"][k]["f1"])
ctl = json.load(open(SR / "control_leak648" / "A0" / "shot_results.json"))["overall"]["f1"]
check("control SHOT", re.search(r"reaches SHOT ([\d.]+)", t2).group(1), ctl)

# ---- 4 interpretation numbers (recompute)
t4 = "\n".join(section("## 4."))
def d(a, b, k): return runs[a][k]["f1"] - runs[b][k]["f1"]
m = re.search(r"\| ClipShots \| ([+-][\d.]+) \| ([+-][\d.]+) \|", t4)
check("4.1 CS A0-A1", m.group(1), d("A0", "A1", "cs")); check("4.1 CS A4-A3", m.group(2), d("A4", "A3", "cs"))
c4 = json.load(open(V2 / "c4_cut_recall.json"))["runs"]
m = re.search(r"cut recall ([\d.]+) vs ([\d.]+) for A0", t4)
check("C4 cut recall", m.group(1), c4["C4"]["cut_recall"]); check("A0 cut recall (extra analysis)", m.group(2), c4["A0"]["cut_recall"])
check("A0 cut recall == Table 8", c4["A0"]["cut_recall"], an["A0"]["cut_recall"])

la = json.load(open(V2 / "shot_label_audit.json"))
t11 = "\n".join(md)
m = re.search(r"holds (\d+) video files in three folders \((\d+) distinct videos\), but only \*\*(\d+) have labels\*\*.*?other \*\*(\d+) videos have no annotations", t11)
for got, key in zip(m.groups(), ("files_in_three_video_folders", "distinct_videos_on_disk", "labelled_videos", "on_disk_without_labels")):
    check(f"label audit {key}", got, la[key])
m = re.search(r"([\d,]+) labels against ([\d,]+) transitions in the final test annotation, ([\d,]+) of them within", t11)
check("test labels in file", m.group(1).replace(",", ""), la["test_labels_in_file"]); check("final test transitions", m.group(2).replace(",", ""), la["test_labels_final_annotation"]); check("within 2", m.group(3).replace(",", ""), la["test_file_labels_within_2_frames_of_final"])
m = re.search(r"silently drop (\d+) of ([\d,]+) transition lines", t11)
check("strict parse dropped", m.group(1), la["transitions_dropped_by_strict_parse"]); check("tolerant total", m.group(2).replace(",", ""), la["transitions_recovered_by_tolerant_parse"])
print(f"checked OK: {n_ok}   FAILED: {len(fails)}")
for f in fails[:40]:
    print("  FAIL", f)
raise SystemExit(1 if fails else 0)
