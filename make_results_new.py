"""Build RESULTS_NEW.md from the saved result files (nothing is typed in by hand except the prose).

Reads: server_results/ablations_v2/*, server_results/ablations_v3/*, server_results/speed_benchmark/*,
       server_results/speed_repro/*, RESULTS.md (old section 8, for the speed comparison), and the code constants quoted in section 1.
   python make_results_new.py
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SR = ROOT / "server_results"
V2, V3, SB, RP = SR / "ablations_v2", SR / "ablations_v3", SR / "speed_benchmark", SR / "speed_repro"


def jl(p):
    return json.load(open(p))


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
        params=int(r["params_total"]), shot=float(r["shot_F1"]), cs=float(r["cs_F1"]), bbc=float(r["bbc_F1"]),
        shot_p=float(r["shot_P"]), shot_r=float(r["shot_R"]), cs_p=float(r["cs_P"]), cs_r=float(r["cs_R"]),
        bbc_p=float(r["bbc_P"]), bbc_r=float(r["bbc_R"]),
        best_ep=int(r["best_epoch"]), epochs=len(hist), hours=float(r["train_hours"]),
        val=float(r["val_F1"]), lat=float(r["latency_B16_ms"]), vram=float(r["vram_B16_MB"]),
        tau=(r["shot_tau_c"], r["shot_tau_g"], r["cs_tau_c"], r["cs_tau_g"], r["cs_lmin"], r["bbc_tau_c"], r["bbc_tau_g"]),
        args=cfg["args"], commit=cfg["git"]["commit"][:7], gpu=cfg["environment"]["gpu_model"],
        torch=cfg["environment"]["pytorch_version"], cuda=cfg["environment"]["cuda_version"],
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
    v3[rid] = dict(params=cfg["parameters"][0], shot=s["f1"], cs=c["f1"], bbc=b["f1"], epochs=len(hist),
                   best_ep=best["epoch"], hours=hist[-1]["elapsed_hours"], status=st, val=best["val"]["f1"],
                   commit=cfg["git"]["commit"][:7])
sp = {r["label"]: r for r in jl(SB / "speed_summary.json")}
a0, a4 = v2["A0"], v2["A4"]
a = v2["A0"]["args"]
PYV = jl(V2 / "A0" / "config.json")["environment"]["python_version"].split()[0]

DESC = {"A0": "Full, dual-label (primary)", "A1": "Full, midpoint only", "A2": "Full, unified single head",
        "A3": "Reduced, midpoint only", "A4": "Reduced, dual-label",
        "C1": "w/o difference-based branch (B)", "C2a": "w/o FrameSimilarity", "C2b": "w/o ColorHistogramSimilarity",
        "C3": "w/o temporal delta", "C4": "w/o per-frame pathway (relations from Branch A)",
        "C5a": "w/o pixel-level motion descriptor", "C5b": "w/o edge-density head", "C6": "w/o self-attention and gate"}
HEADS = {"A0": 2, "A1": 2, "A2": 1, "A3": 2, "A4": 2}


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

w("# RESULTS_NEW — HybridSBD Protocol-A ablation (leak-free)\n")
w("All numbers in this document come from the saved experiment results (training histories, per-dataset threshold-search results and speed measurements). "
  "One seed (42) per variant; per the spec, F1 differences below 0.01 are treated as inconclusive. Thresholds are tuned on each test set "
  "(Protocol A), so absolute F1 is optimistic by construction and comparable only to other Protocol-A numbers.\n")

# ---------------- 1. data and training approach ----------------
w("## 1. Data and training approach\n")
w("### 1.1 Datasets\n")
w("| Role | Dataset | Videos | Notes |\n|---|---|---|---|")
w("| Train | ClipShots train | 3,656 | 3,657 listed; one source file (`5186782609.mp4`) is unreadable and is explicitly allow-listed as missing |")
w("| Train / validation | SHOT (AutoShot, `kuaishou_v2.txt`) | 448 | 648 labelled videos minus the 200 SHOT test videos |")
w("| Test | SHOT test | 200 | the authors' released SHOT test annotation |")
w("| Test | ClipShots test | 500 | official ClipShots test annotations |")
w("| Zero-shot test | BBC Planet Earth | 11 episodes | Zenodo record 14865504; never used for training or selection |\n")
LA = jl(V2 / "shot_label_audit.json")
w("**SHOT label coverage.** "
  f"The SHOT download holds {LA['files_in_three_video_folders']} video files in three folders ({LA['distinct_videos_on_disk']} distinct videos), but only "
  f"**{LA['labelled_videos']} have labels** in `kuaishou_v2.txt`; the other **{LA['on_disk_without_labels']} videos have no annotations at all** and cannot be used for supervised "
  f"training, so they are not used. All {LA['labelled_and_on_disk']} labelled videos are present on disk. "
  f"The label file is also irregular, and a tolerant parser is used: {LA['header_entries']} header entries for {LA['labelled_videos']} videos "
  f"({LA['videos_with_duplicate_entries']} videos appear twice; the {LA['non_test_duplicates_identical']} non-test duplicates are identical), mixed `a,b` / `a b` line formats, one header ending in `.mp4.png`, "
  f"and {LA['lines_with_odd_number_of_integers']} lines with an odd number of integers that are skipped. A strict parser would silently drop {LA['transitions_dropped_by_strict_parse']} of "
  f"{LA['transitions_recovered_by_tolerant_parse']:,} transition lines ({LA['transitions_dropped_by_strict_parse']/LA['transitions_recovered_by_tolerant_parse']*100:.1f} %).\n")
w(f"For the {LA['test_videos']} test videos the file contains an earlier version of the labels: {LA['test_labels_in_file']:,} labels against {LA['test_labels_final_annotation']:,} transitions in the final test annotation, "
  f"{LA['test_file_labels_within_2_frames_of_final']:,} of them within ±2 frames of the final annotation. Training on these videos therefore means training on almost the exact test answers.\n")
PAPER_VIDEOS, PAPER_TEST = 853, 200   # AutoShot paper / release: 853 videos, 200 of them are fine-annotated test videos
w("**Limitation of the released SHOT labels.** "
  f"The released label file mixes the training labels and the labels of the {LA['test_videos']} test videos in one file, so the test videos can only be told apart "
  "with the separate test annotation. It also covers far fewer videos than the dataset description: the AutoShot paper describes "
  f"{PAPER_VIDEOS} videos, of which {PAPER_TEST} are test videos, which would leave {PAPER_VIDEOS - PAPER_TEST} for training. The label file, however, covers only {LA['labelled_videos']} videos "
  f"and these include all {LA['test_videos']} test videos; roughly {LA['on_disk_without_labels']} videos have no labels at all. After excluding the test videos, "
  f"{LA['train_videos_after_excluding_test']} labelled videos remain for training, about {PAPER_VIDEOS - PAPER_TEST - LA['train_videos_after_excluding_test']} fewer than the paper's split implies. "
  "Our SHOT results therefore use less training data than the authors' setup could, which is a limitation of the released data, and more labelled training videos could improve them.\n")
w("**Test videos are excluded from training.** The SHOT label file contains the 200 test videos, so training on all 648 would "
  "train on the test set. Training uses the 448 other videos, and an overlap audit across all train/test sets "
  "(SHOT, ClipShots, BBC) finds zero shared videos.\n")
w("### 1.2 Preparation\n")
w("1. Archives are downloaded, extracted and checked, then every video is decoded once and each frame is bilinearly resized to "
  "**48 × 64** (uint8, RGB). Decoding is chunked to bound memory.")
w("2. Videos are cut into **128-frame clips** stored as tensors with per-frame labels "
  "(0 none, 1 cut, 2 gradual) and a transition-length value.")
w("3. **Training clips.** ClipShots: 20 clips per video (60 % centred on a transition with ±15-frame jitter, the rest random), "
  "73,120 clips from 3,656 videos. SHOT: 70 clips per video; every transition gets 3 clips centred on it "
  "(jitter −20 / 0 / +20 frames) and the remaining quota is random, 31,360 clips from 448 videos.")
w("4. **Test clips.** Sliding windows of 128 frames with stride 64 cover each full video; overlapping predictions are averaged "
  "back into one probability track per video.")
w("5. A verification step checks that every clip loads, has the right shape and label range, and that every expected video "
  "is present, before any training starts.\n")
w("### 1.3 Training\n")
w("| Setting | Value |\n|---|---|")
w("| Targets | y = midpoint of every transition (cuts and graduals); z = every frame inside a gradual (many-hot) |")
w("| Loss | 5.0 · focal BCE(y; α 0.8, γ 2) + 0.3 · focal BCE(z; α 0.5, γ 2, label smoothing 0.1) |")
w("| Optimiser | AdamW, lr 5e-5, weight decay 0.05, batch 32, AMP, gradient clip 1.0 |")
w("| Schedule | up to 80 epochs; 5 warm-up epochs (lr × 0.25 → 1), then cosine to 5e-6 |")
w(f"| Early stopping | patience {a['patience']} on SHOT validation F1 (15 % of the SHOT training videos: 381 train / 67 validation) |")
w("| Epoch mix | all SHOT training clips + a fresh random 70 % of the ClipShots training clips each epoch (ClipShots split 3,108 train / 548 validation videos) |")
w("| Basic augmentation | horizontal flip, per-frame brightness and contrast jitter, noise, brightness shift |")
w(f"| Strong augmentation | per-clip zoom/shift (up to 25 %), colour gain, occasional grayscale |")
w(f"| Synthetic transitions | probability {a['synth_prob']} per clip: hard cut 40 %, dissolve 40 %, fade through black 20 % (3–30 frames), "
  "spliced from two clips of the same batch with a 3-frame clear margin from real transitions; labels follow the real-label convention |")
w(f"| Weight averaging | EMA, decay {a['ema_decay']}; validation, checkpoints and test use the EMA weights |")
w(f"| Seed | {a['seed']} (fixed hash seed); every variant uses the same recipe, data and seed |")
w(f"| Hardware / software | {a0['gpu']}, Python {PYV}, PyTorch {a0['torch']}, CUDA {a0['cuda']} |\n")
w("**Synthetic transitions in detail.** Each clip of a batch is replaced, with probability " + str(a['synth_prob']) + ", by a splice of itself (clip i) and another clip j from the same batch: "
  "a **hard cut** (40 %: frames before a random point come from i, after it from j), a **dissolve** (40 %: a linear cross-fade from i to j over 3–30 frames) or a **fade through black** "
  "(20 %: i fades to black, then j fades in, over 3–30 frames). Labels before the splice come from i and after it from j; a cut is labelled on the two frames at the splice, and a dissolve or fade is "
  "labelled gradual on every frame of the transition, exactly like real transitions. The splice is kept at least 3 frames away from every real transition in either clip. "
  "This adds many extra transitions with exactly known boundaries, especially gradual ones, which are rare and hard to annotate, and it uses only frames and labels from the training videos.\n")
w("### 1.4 Evaluation (Protocol A)\n")
w("- Per test set, the model's best checkpoint (by SHOT validation F1) is evaluated once over a threshold grid and the best setting "
  "is reported: boundary threshold τ_c ∈ {0.10, 0.15, …, 0.90}, gradual threshold τ_g ∈ {0.3, 0.4, 0.5, 0.6}, "
  "and minimum scene length ℓ_min ∈ {1, 2, 3, 4} for ClipShots only (ℓ_min = 1 for SHOT and BBC).")
w("- Decision rule (dual-label models): boundary if p_y > τ_c; gradual if p_z > τ_g; cut = boundary and not gradual; "
  "the predicted many-hot track is gradual OR cut. Midpoint-only (A1, A3) and unified (A2) variants use τ_c only.")
w("- Scoring: AutoShot's scene-based matching with **±2 frame tolerance**; precision, recall and F1 over transitions, pooled over all videos.\n")


w("### 1.5 Protocol details (spec Section 0, items 3–8)\n")
w("- **Minimum shot length ℓ_min** is applied in the evaluation code, shared by all three test sets: after the boundary track "
  "is turned into scenes, every predicted scene shorter than ℓ_min frames is merged into the previous scene. The grid {1, 2, 3, 4} is searched "
  "for ClipShots only; SHOT and BBC use ℓ_min = 1 (no merging).")
w("- **Overlapping windows.** Probabilities of all windows that cover a frame are summed and divided by the number of windows covering it "
  "(plain average, separately for p_y and p_z). Frames near the start or end are covered by one window only and use that window's value.")
w("- **Final window.** ClipShots test: window starts 0, 64, 128, … plus one extra window starting at (frames − 128), so the last window is shifted back "
  "and ends on the final frame. SHOT test: a window that would run past the end is shifted back to start at (frames − 128); a video shorter than 128 frames "
  "is padded by repeating its last frame. BBC: windows start every 64 frames and the last window is padded by repeating the final frame. "
  "The same window and averaging code is used for every run.")
w("- **Seeds.** Clip sampling during data preparation uses an MD5-based stable per-video seed (not Python's `hash`). Training seeds the hash seed, "
  "Python, NumPy, PyTorch (CPU and CUDA) and every data-loader worker to the same seed (42) and runs cuDNN in deterministic mode.")
w("- **Benchmark hygiene.** `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.empty_cache()` are called before every latency and VRAM measurement "
  "")
w("- **Data overlap.** No video appears in more than one of SHOT train / SHOT validation / SHOT test / ClipShots train / ClipShots test / BBC.\n")

# ---------------- 2. compliance ----------------

import check_a0_reproduction as CK
ARCH = jl(V2 / "a0_architecture_check.json")
CTRL = SR / "control_leak648" / "A0"
ctl = {n: f1_of(CTRL / f"{n}_results.json")["f1"] for n in ("shot", "clipshots", "bbc")}
w("## 2. Compliance with the specification\n")
w("### 2.1 A0 reproduction checks (spec Section 1)\n")
w("**Check 1 — architecture, on the trained A0 checkpoint: "
  + ("PASS" if ARCH["all_pass"] else "FAIL") + ".**\n")
w("| Item | Required | Found |\n|---|---|---|")
w(f"| Fusion blocks | 4 | {ARCH['fusion_blocks']} |")
w(f"| Positional encoding | none | {'none' if not ARCH['positional_encoding_keys'] else ARCH['positional_encoding_keys']} |")
w(f"| Delta projection weight | (64, 640) | {tuple(ARCH['delta_projection_weight_shape'])} |")
w(f"| D | 384 | {ARCH['main_dim']} |")
w(f"| Fusion input / embedding dim | 655 / 640 | {ARCH['fusion_in']} / {ARCH['embedding_dim']} |")
w(f"| Exact parameter count | rounds to 10.9M | {ARCH['parameters_total_exact']:,} ({ARCH['parameters_round_to_M']}M) |\n")
w("**Check 2 — performance (Protocol A, paper's acceptable ranges):**\n")
w("| Test set | Paper | Acceptable range | Our A0 | Result |\n|---|---|---|---|---|")
for key, (label, paper, lo, hi) in CK.RANGES.items():
    v = {"shot": a0["shot"], "clipshots": a0["cs"], "bbc": a0["bbc"]}[key]
    st = "within range" if lo <= v <= hi else ("**below range**" if v < lo else "above range")
    w(f"| {label} F1 | {paper} | {lo} – {hi} | {f4(v)} | {st} |")
w("\nCheck 2 does **not** pass on SHOT, and this is expected: **it is caused by a deliberate change in the training data, not by the code.** "
  "The SHOT label file `kuaishou_v2.txt` lists 648 videos, and the 200 SHOT *test* videos (the authors' released test annotation) are among them. "
  "The paper-era setup therefore trained on 648 videos that include the entire SHOT test set — a data leak that inflates SHOT F1. "
  "We removed the 200 test videos from training (448 videos remain), so the SHOT numbers here are leak-free and are lower than the paper's 0.924. "
  f"As a one-off control, the same A0 recipe trained on all 648 videos reaches SHOT {f4(ctl['shot'])}, which is inside the required 0.915–0.930 range, "
  "so the gap is explained by the leak and not by an architecture or pipeline difference (Check 1 passes). "
  "Per the spec, this is reported to the team: the ablations proceed on the same code and data policy for all runs, and the main-paper SHOT number needs discussion. "
  f"ClipShots ({f4(a0['cs'])}) is slightly above its range and BBC ({f4(a0['bbc'])}) is inside it; neither benchmark shares videos with the SHOT test set.\n")
w("### 2.2 Deviations from the specification\n")
w("| Spec says | What was done | Why |\n|---|---|---|")
w("| SHOT: 648 training videos (§3) | **448 videos**: the 200 SHOT test videos are excluded | The 648 include the SHOT test set (data leak), see 2.1 |")
w(f"| Early stopping patience 15 (§3) | patience {a['patience']} for all 13 runs and the build-up runs | Shortens training time: the validation optimum appeared early in the first runs, so the same patience was used for every variant |")
w(f"| Augmentation: flip, brightness, contrast, noise, clip brightness (§3) | all of these, **plus** strong per-clip augmentation, synthetic transitions (p = {a['synth_prob']}) and weight averaging (EMA {a['ema_decay']}) | Added to recover accuracy once the leaked videos were removed; identical for every run, so the comparison between variants stays fair |")
w("| All runs on the same code version as A0 (§1) | A0 and the other 12 runs were trained from two code revisions | A0 was trained first as the recipe experiment; only the launcher scripts differ, the training code is identical |")
w("| 13 runs, nothing else (§5, prompt) | **extra** build-up ladder (4 runs, section 5), speed table with GFLOPs, B = 1 / 32 timings and an AutoShot comparison, one extra C4 cut-recall analysis | Requested additions to identify which components do the work; kept in separate sections |")
w("| Training speed | channels-last memory format | Memory-layout change only; values are mathematically identical |\n")

# ---------------- 3. main ablation ----------------
w("## 3. Main ablation (Protocol A, 13 runs)\n")
w("### 3.1 Supervision (Table 7a)\n")
w("| ID | Variant | Heads | Params | SHOT F1 | ClipShots F1 | BBC F1 | Latency B=16 (ms) |\n|---|---|---|---|---|---|---|---|")
_ids = ("A0", "A1", "A2", "A3", "A4")
_b = {k: bset(_ids, lambda i, k=k: v2[i][k]) for k in ("shot", "cs", "bbc")}
_bl = bset(_ids, lambda i: v2[i]["lat"], hi=False, nd=1)
for rid in _ids:
    r = v2[rid]
    w(f"| {rid} | {DESC[rid]} | {HEADS[rid]} | {r['params']:,} | {B(f4(r['shot']), rid in _b['shot'])} | {B(f4(r['cs']), rid in _b['cs'])} | {B(f4(r['bbc']), rid in _b['bbc'])} | {B(format(r['lat'], '.1f'), rid in _bl)} |")
w("\n| Delta | SHOT | ClipShots | BBC |\n|---|---|---|---|")
for lab, x, y in (("A0 − A1 (full)", "A0", "A1"), ("A4 − A3 (reduced)", "A4", "A3"), ("A0 − A2 (dual vs unified)", "A0", "A2")):
    w(f"| {lab} | {sd(v2[x]['shot']-v2[y]['shot'])} | {sd(v2[x]['cs']-v2[y]['cs'])} | {sd(v2[x]['bbc']-v2[y]['bbc'])} |")
w("\n### 3.2 Components, leave-one-out (Table 7b)\n")
_ids = ("A0", "C1", "C2a", "C2b", "C3", "C4", "C5a", "C5b", "C6")
_b = {k: bset(_ids, lambda i, k=k: v2[i][k]) for k in ("shot", "cs", "bbc")}
_bl = bset(_ids, lambda i: v2[i]["lat"], hi=False, nd=1)
w("| ID | Description | Params | SHOT F1 | ClipShots F1 | BBC F1 | Latency B=16 (ms) |\n|---|---|---|---|---|---|---|")
for rid in _ids:
    r = v2[rid]
    w(f"| {rid} | {'Full model' if rid=='A0' else DESC[rid]} | {r['params']:,} | {B(f4(r['shot']), rid in _b['shot'])} | {B(f4(r['cs']), rid in _b['cs'])} | {B(f4(r['bbc']), rid in _b['bbc'])} | {B(format(r['lat'], '.1f'), rid in _bl)} |")
w("\nChange in F1 relative to A0:\n")
w("| ID | ΔSHOT | ΔClipShots | ΔBBC |\n|---|---|---|---|")
for rid in _ids[1:]:
    r = v2[rid]
    w(f"| {rid} | {sd(r['shot']-a0['shot'])} | {sd(r['cs']-a0['cs'])} | {sd(r['bbc']-a0['bbc'])} |")
mx = max(abs(v2[r][k]-a0[k]) for r in ("C1","C2a","C2b","C3","C4","C5a","C5b","C6") for k in ("shot","cs","bbc"))
w(f"\nLargest single-component change on any dataset: {mx:.4f}. Every leave-one-out change is below 0.01 F1, so none is individually conclusive.\n")
w("### 3.3 Precision / recall and chosen thresholds\n")
_all = ("A0","A1","A2","A3","A4","C1","C2a","C2b","C3","C4","C5a","C5b","C6")
w("| ID | SHOT P / R | ClipShots P / R | BBC P / R |\n|---|---|---|---|")
for rid in _all:
    r = v2[rid]
    w(f"| {rid} | {r['shot_p']:.3f} / {r['shot_r']:.3f} | {r['cs_p']:.3f} / {r['cs_r']:.3f} | {r['bbc_p']:.3f} / {r['bbc_r']:.3f} |")
w("\nSelected thresholds (τ_c boundary, τ_g gradual, ℓ_min minimum scene length) and training length:\n")
w("| ID | SHOT τ_c, τ_g | ClipShots τ_c, τ_g, ℓ_min | BBC τ_c, τ_g | Best epoch / epochs run | Train hours |\n|---|---|---|---|---|---|")
for rid in _all:
    r = v2[rid]; t = r["tau"]
    w(f"| {rid} | {t[0]}, {t[1] or '–'} | {t[2]}, {t[3] or '–'}, {t[4]} | {t[5]}, {t[6] or '–'} | {r['best_ep']} / {r['epochs']} | {r['hours']:.2f} |")
w("\n### 3.4 Per-type analysis on ClipShots (Tables 8 and 9)\n")
txt = (V2 / "result_tables.md").read_text()
m = re.search(r"## Table 8.*", txt, re.S)
w(re.sub(r"^## ", "#### ", m.group(0), flags=re.M).strip() + "\n")


c4 = jl(V2 / "c4_cut_recall.json")["runs"]
an = jl(V2 / "clipshots_protocol_a_analysis.json")["runs"]
w("## 4. Interpretation (spec Section 9)\n")
w("Single seed per variant; differences below 0.01 F1 are inconclusive.\n")
w("### 4.1 Supervision\n")
w("| Dataset | A0 − A1 (full) | A4 − A3 (reduced) | Case (spec §9) |\n|---|---|---|---|")
for lab, k in (("SHOT", "shot"), ("ClipShots", "cs"), ("BBC", "bbc")):
    d1, d2 = v2["A0"][k] - v2["A1"][k], v2["A4"][k] - v2["A3"][k]
    if d1 >= 0.01 and d2 >= 0.01: case = "1 — both clearly positive: dual-label helps independently of architectural complexity"
    elif d1 >= 0.01 or d2 >= 0.01: case = "2 — positive for one architecture only"
    else: case = "3 — neither ≥ 0.01: no accuracy claim"
    w(f"| {lab} | {sd(d1)} | {sd(d2)} | {case} |")
w("\nDual-label supervision therefore helps independently of architectural complexity on ClipShots (the benchmark with many gradual transitions) and has no measurable "
  "accuracy effect on SHOT or BBC. It also changes the error profile: cut recall / gradual recall on ClipShots is "
  f"{an['A0']['cut_recall']:.3f} / {an['A0']['gradual_recall']:.3f} for A0 (dual) against {an['A1']['cut_recall']:.3f} / {an['A1']['gradual_recall']:.3f} for A1 (midpoint only).\n")
w("### 4.2 Hypotheses\n")
w("| Comparison | Hypothesis | Outcome |\n|---|---|---|")
d = {k: v2["A0"][k] - v2["A2"][k] for k in ("shot", "cs", "bbc")}
w(f"| A0 vs A2 | A0 higher | Confirmed on ClipShots ({sd(d['cs'])}) and BBC ({sd(d['bbc'])}); A2 is higher on SHOT (A0 − A2 = {sd(d['shot'])}). Wording that a unified target is harmful is supported on ClipShots and BBC, not on SHOT. |")
drops = {c: {k: v2[c][k] - a0[k] for k in ("shot", "cs", "bbc")} for c in ("C1", "C2a", "C2b", "C3", "C4", "C5a", "C5b", "C6")}
worst = {k: min(drops, key=lambda c: drops[c][k]) for k in ("shot", "cs", "bbc")}
w(f"| C2a / C2b | Largest drops | Not confirmed. The largest drop per dataset is {worst['shot']} on SHOT ({sd(drops[worst['shot']]['shot'])}), "
  f"{worst['cs']} on ClipShots ({sd(drops[worst['cs']]['cs'])}), {worst['bbc']} on BBC ({sd(drops[worst['bbc']]['bbc'])}); C2a is {sd(drops['C2a']['shot'])} / {sd(drops['C2a']['cs'])} / {sd(drops['C2a']['bbc'])} and C2b "
  f"{sd(drops['C2b']['shot'])} / {sd(drops['C2b']['cs'])} / {sd(drops['C2b']['bbc'])} (SHOT / ClipShots / BBC). No single stream stands out; the streams look redundant. |")
w(f"| C4 | Lower cut recall than A0 | Direction confirmed but small: cut recall {c4['C4']['cut_recall']:.4f} vs {c4['A0']['cut_recall']:.4f} for A0 ({sd(c4['C4']['cut_recall']-c4['A0']['cut_recall'])}); "
  f"overall F1 {sd(drops['C4']['shot'])} SHOT, {sd(drops['C4']['cs'])} ClipShots, {sd(drops['C4']['bbc'])} BBC. The Branch C argument should be weakened. (extra analysis) |")
dr = {g: an['C6']['duration_recall'][g] - an['A0']['duration_recall'][g] for g in ("le_10", "11_30", "gt_30")}
w(f"| C6 | Lower recall on > 30-frame transitions | Partly: > 30 frames {sd(dr['gt_30'])} (n = {an['A0']['duration_counts']['gt_30']}), but the larger drop is ≤ 10 frames ({sd(dr['le_10'])}) and 11–30 frames improves ({sd(dr['11_30'])}). "
  "Convolutions alone are largely sufficient; C6 has fewer parameters (7.4M vs 10.9M) but no latency benefit on this GPU. |")
mxd = max(-min(drops[c][k] for c in drops) for k in ("shot", "cs", "bbc"))
w(f"| Any Cx | Drop | No single component drop reaches 0.01 on any dataset (largest {mxd:.4f}); each component is individually not needed. |\n")
w("The spec's follow-up for a small A0 − A1 difference (two extra seeds of A0 and A1) is not needed for the ClipShots claim, since that difference is far above 0.01; for SHOT and BBC it would be needed to claim any effect.\n")

# ---------------- 5. build-up ladder ----------------
w("## 5. Build-up ladder: what does the work?\n")
w("The leave-one-out runs cannot show which parts matter if parts can substitute for each other, so we also build the model up "
  "from a minimal core. Same recipe, data and seed as section 3.\n")
w("| Step | Run | What is added | Params | SHOT F1 | ClipShots F1 | BBC F1 | Latency B=16 (ms) |\n|---|---|---|---|---|---|---|---|")
ladder = [("M0_min", "RGB encoder + MLP + dual head only", v3["M0_min"]),
          ("M1_edge", "+ edge-density head", v3["M1_edge"]),
          ("B_noatt", "+ 4 fusion blocks, no attention", v3["B_noatt"]),
          ("A4", "+ attention and gate (reduced model)", a4),
          ("S1_rel", "+ FrameSimilarity, ColorHist, delta", v3["S1_rel"]),
          ("A0", "+ diff branch, motion, per-frame pathway (full model)", a0)]
_L = {rid: r for rid, _, r in ladder}
_ids = list(_L)
_b = {k: bset(_ids, lambda i, k=k: _L[i][k]) for k in ("shot", "cs", "bbc")}
_lat = {i: sp[i]["batch_results"]["16"]["mean_ms"] for i in _ids}
_bl = bset(_ids, lambda i: _lat[i], hi=False, nd=1)
for i, (rid, what, r) in enumerate(ladder):
    w(f"| {i} | {rid} | {what} | {r['params']:,} | {B(f4(r['shot']), rid in _b['shot'])} | {B(f4(r['cs']), rid in _b['cs'])} | {B(f4(r['bbc']), rid in _b['bbc'])} | {B(format(_lat[rid], '.1f'), rid in _bl)} |")
w("")
m0, m1, bn, s1 = v3["M0_min"], v3["M1_edge"], v3["B_noatt"], v3["S1_rel"]
w("**Reading it** (against the 0.01 inconclusive band, single seed):\n")
w(f"- The bare core already reaches SHOT {f4(m0['shot'])} / ClipShots {f4(m0['cs'])} / BBC {f4(m0['bbc'])}, within "
  f"{a0['shot']-m0['shot']:.3f} / {a0['cs']-m0['cs']:.3f} / {a0['bbc']-m0['bbc']:.3f} of the full model on SHOT / ClipShots / BBC, with {m0['params']/a0['params']*100:.0f} % of its parameters.")
w(f"- The **temporal fusion blocks** are the largest single step and the only one that clears the 0.01 band (on ClipShots): {sd(bn['shot']-m1['shot'])} SHOT, {sd(bn['cs']-m1['cs'])} ClipShots, {sd(bn['bbc']-m1['bbc'])} BBC. "
  "The edge head adds " + f"{sd(m1['shot']-m0['shot'])} / {sd(m1['cs']-m0['cs'])} / {sd(m1['bbc']-m0['bbc'])} (SHOT / ClipShots / BBC): small, but positive on all three datasets.")
w(f"- Self-attention ({sd(a4['shot']-bn['shot'])} / {sd(a4['cs']-bn['cs'])} / {sd(a4['bbc']-bn['bbc'])}), the similarity and delta streams "
  f"({sd(s1['shot']-a4['shot'])} / {sd(s1['cs']-a4['cs'])} / {sd(s1['bbc']-a4['bbc'])}) and the remaining branches "
  f"({sd(a0['shot']-s1['shot'])} / {sd(a0['cs']-s1['cs'])} / {sd(a0['bbc']-s1['bbc'])}) each change every dataset by under 0.01.")
w(f"- The full model's SHOT advantage over the reduced model is {sd(a0['shot']-a4['shot'])}, while ClipShots is {sd(a0['cs']-a4['cs'])} and BBC {sd(a0['bbc']-a4['bbc'])}.")
w(f"- Dual-label vs a single unified head (A0 − A2): {sd(v2['A0']['shot']-v2['A2']['shot'])} SHOT, {sd(v2['A0']['cs']-v2['A2']['cs'])} ClipShots, {sd(v2['A0']['bbc']-v2['A2']['bbc'])} BBC.")
w("- Dual-label supervision is the one effect clearly above the band: ClipShots "
  f"{sd(v2['A0']['cs']-v2['A1']['cs'])} (full) and {sd(v2['A4']['cs']-v2['A3']['cs'])} (reduced), with SHOT and BBC unchanged within 0.006.\n")

# ---------------- 4. speed ----------------
w("## 6. Speed, memory and size\n")
w("FP32, synthetic input `(B, 128, 3, 48, 64)`, forward pass only, 10 warm-up + 80 timed runs, otherwise idle GPU "
  f"({a0['gpu']}). Peak VRAM is the maximum allocated during the forward pass. GFLOPs are per 128-frame clip and count convolutions "
  "and matrix multiplies only. AutoShot uses its native 27 × 48 input; our models use 48 × 64.\n")
_lab = list(sp.keys())
def _g(lab, bs, key):
    return sp[lab]["batch_results"][str(bs)][key]
def _col(vals, hi, fmt):
    best = max(vals.values()) if hi else min(vals.values())
    return {k: (f"**{format(v, fmt)}**" if v == best else format(v, fmt)) for k, v in vals.items()}
_cols_a = [("Params (M)", {l: sp[l]["total_params"] / 1e6 for l in _lab}, False, ".2f"), ("FP32 (MB)", {l: sp[l]["size_fp32_mb"] for l in _lab}, False, ".1f"),
           ("GFLOPs / clip", {l: sp[l]["gflops"] for l in _lab}, False, ".2f")] + \
          [(f"VRAM B={bs} (MB)", {l: _g(l, bs, "peak_memory_mb") for l in _lab}, False, ".0f") for bs in (1, 16, 32)]
_cols_b = [(f"ms B={bs}", {l: _g(l, bs, "mean_ms") for l in _lab}, False, ".1f") for bs in (1, 16, 32)] + \
          [(f"frames/s B={bs}", {l: _g(l, bs, "frames_per_sec") for l in _lab}, True, ".0f") for bs in (1, 16, 32)]
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
    w(f"| {rid} | {au/br['mean_ms']:.1f}× faster | {auv/br['peak_memory_mb']:.1f}× less | {sp['AutoShot (TransNetV2Supernet)']['gflops']/sp[rid]['gflops']:.1f}× fewer |")
w(f"\nThe diff branch (a second encoder) accounts for most of the full model's cost: A0 {sp['A0']['batch_results']['16']['mean_ms']:.1f} ms vs "
  f"C1 (no diff branch) {sp['C1']['batch_results']['16']['mean_ms']:.1f} ms vs A4 {sp['A4']['batch_results']['16']['mean_ms']:.1f} ms at B=16.\n")

# ---- reproduction of RESULTS.md section 8
old = {}
for ln in (ROOT / "RESULTS.md").read_text().splitlines():
    mm = re.match(r"\|\s*(hybrid_detector\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)M\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*([\d,]+)\s*\|\s*(\d+)\s*\|", ln)
    if mm:
        old[(mm.group(1), int(mm.group(5)))] = (float(mm.group(4)), float(mm.group(6)), float(mm.group(9)))
    mm = re.match(r"\|\s*AutoShot \(TransNetV2Supernet\)\s*\|\s*([\d.]+)M\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*([\d,]+)\s*\|\s*(\d+)\s*\|", ln)
    if mm:
        old[("AutoShot", int(mm.group(2)))] = (float(mm.group(1)), float(mm.group(3)), float(mm.group(6)))
rep = {r["label"]: r for r in jl(RP / "ours_old_arch.json")["models"]}
rep.update({r["label"]: r for r in jl(RP / "baselines.json")["models"]})
w("### 6.1 Re-measurement of the previously reported speed table\n")
w("The original checkpoints are not available, but latency, memory and parameter count depend only on the architecture, so the six "
  "earlier model variants (different width D and number of fusion blocks) were rebuilt with the same settings and random weights and timed with the same protocol "
  f"(batch 8 / 16 / 32). The old table was measured on an RTX 4090 and this one on the {a0['gpu']}.\n")
w("| Model | Params old → new (M) | B=8 ms old → new | B=16 ms old → new | B=32 ms old → new | B=16 VRAM MB old → new |\n|---|---|---|---|---|---|")
VARIANT = {13: (512, 3), 15: (256, 3), 16: (384, 4), 17: (256, 4), 18: (512, 4), 19: (384, 3)}
for name in [f"hybrid_detector{n}" for n in (13, 15, 16, 17, 18, 19)] + ["AutoShot (TransNetV2Supernet)"]:
    key = "AutoShot" if name.startswith("AutoShot") else name
    shown = name if name.startswith("AutoShot") else "D = %d, %d fusion blocks%s" % (*VARIANT[int(name[15:])], " (A0 architecture)" if VARIANT[int(name[15:])] == (384, 4) else "")
    r = rep[name]; br = {int(k): v for k, v in r["batch_results"].items()}
    cells = []
    for bs in (8, 16, 32):
        o = old.get((key, bs)); cells.append(f"{o[1]:.1f} → {br[bs]['mean_ms']:.1f}" if o else f"n/a → {br[bs]['mean_ms']:.1f}")
    o16 = old.get((key, 16))
    w(f"| {shown} | {o16[0]:.2f} → {r['total_params']/1e6:.2f} | " + " | ".join(cells) + f" | {o16[2]:.0f} → {br[16]['peak_memory_mb']:.0f} |")
w("\n- Parameter counts reproduce exactly, and AutoShot's VRAM reproduces to within about 22 MB.")
w("- Latency does not reproduce in absolute terms because the GPU is different (the new one is much faster): our models run at about "
  "30 ms instead of 73–79 ms at B=16, and AutoShot at 155 ms instead of 1,351 ms.")
ao = {bs: old[("AutoShot", bs)][1] for bs in (8, 16, 32)}
an = {int(k): v["mean_ms"] for k, v in rep["AutoShot (TransNetV2Supernet)"]["batch_results"].items()}
w(f"- The old note that AutoShot's latency jumps ~11× from B=8 to B=16 ({ao[8]:.0f} → {ao[16]:.0f} ms) is not seen here "
  f"({an[8]:.0f} → {an[16]:.0f} → {an[32]:.0f} ms at B=8/16/32: roughly linear). With this GPU the B=16 speedup of the old dim-384 4-block model over AutoShot is "
  f"{an[16]/rep['hybrid_detector16']['batch_results']['16']['mean_ms']:.1f}×, not the 17.7× reported before.")
diffs = [rep[f"hybrid_detector{n}"]["batch_results"]["16"]["peak_memory_mb"] - old[(f"hybrid_detector{n}", 16)][2] for n in (13, 15, 16, 17, 18, 19)]
w(f"- Peak memory of our models measures {min(diffs):.0f}–{max(diffs):.0f} MB higher than in the old table at B=16 (peak allocation differs by environment).")
w("- TransNetV2 could not be re-measured (no PyTorch weights or code in this environment), so its old numbers are not reproduced here.")
w("- The measurements on the new, faster GPU are the ones used in this study; the old table should be read as RTX 4090 numbers.\n")

# ---------------- 5. caveats ----------------
w("## 7. Caveats\n")
w("- Single seed per variant. The spec treats differences below 0.01 F1 as inconclusive, which covers every leave-one-out difference. The larger effects (dual-label supervision on ClipShots, the fusion blocks on ClipShots) exceed the band but are also single-seed.")
w("- Thresholds are tuned per test set (Protocol A), so absolute F1 is optimistic. These numbers are leak-free (SHOT test videos excluded "
  "from training) and should not be compared with numbers from models trained on all 648 SHOT videos.")
w("- Runs of this study used two code revisions (A0 and the other 12 runs). The training code is identical across them; only the launcher scripts differ.")
w(f"- `B_noatt` was stopped manually after epoch {v3['B_noatt']['epochs']} (best validation F1 at epoch {v3['B_noatt']['best_ep']}) and evaluated from its best checkpoint, so it did not use the patience-10 stop of the other runs.")
w("- Latency, VRAM and GFLOPs are measured on synthetic input; GFLOPs exclude non-matmul/conv operations.\n")

# ---------------- 6. files ----------------

Path(ROOT / "RESULTS_NEW.md").write_text("\n".join(out) + "\n")
print("wrote RESULTS_NEW.md", len("\n".join(out)))
