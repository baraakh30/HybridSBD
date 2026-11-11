# HybridSBD: dual-head shot boundary detection

HybridSBD detects shot boundaries in video, both **hard cuts** and **gradual transitions** (dissolves, fades). A model predicts, for every frame of a 128-frame window, whether a transition is centred on it (head y, midpoint) and whether the frame lies inside a gradual transition (head z, many-hot). This repository contains the model, the training and evaluation code, and an ablation study of its components and supervision.

## Results

Models are trained on SHOT (448 videos; the 200 SHOT test videos are excluded) and ClipShots, and evaluated on the SHOT test set, the ClipShots test set and BBC Planet Earth with scene-based F1 (±2 frames) and no minimum-shot-length merging. **Protocol A** selects the decision thresholds on each test set; **Protocol B** selects them once on the SHOT validation videos and applies them unchanged. F1, mean ± sample SD over three training seeds:

| Model | Protocol | SHOT | ClipShots | BBC |
|---|---|---|---|---|
| A0 (dual-label, full model) | A | 0.8466 ± 0.0018 | 0.7990 ± 0.0047 | 0.9693 ± 0.0034 |
| A0 (dual-label, full model) | B | 0.8268 ± 0.0202 | 0.7633 ± 0.0039 | 0.9446 ± 0.0066 |
| A1 (midpoint only, full model) | A | 0.8458 ± 0.0021 | 0.7569 ± 0.0043 | 0.9686 ± 0.0015 |
| A1 (midpoint only, full model) | B | 0.8281 ± 0.0046 | 0.7374 ± 0.0032 | 0.9579 ± 0.0029 |

The full model has 10.9 M parameters. Dual-label supervision (A0) beats midpoint-only supervision (A1) on ClipShots at every seed under both protocols. See [docs/RESULTS.md](docs/RESULTS.md) for the summary tables and [reports/RESULTS_FULL.md](reports/RESULTS_FULL.md) for every table, the methodology and the caveats. All per-run numbers are in [reports/results_all_runs.csv](reports/results_all_runs.csv).

## Model

- **Heads.** y (midpoint of each transition, loss weight 5.0) and z (every frame inside a gradual transition, loss weight 0.3), trained with focal losses. Prediction: boundary if sigmoid(y) > τ_c; gradual if sigmoid(z) > τ_g; the predicted track is gradual OR cut.
- **Backbone (A0).** RGB spatial encoder, difference branch, per-frame pathway, FrameSimilarity, ColorHistogramSimilarity, temporal delta, pixel-motion descriptor and an edge-density head, fused by 4 temporal fusion blocks (dilated convolutions, self-attention and a gated residual), model width 384.
- **Variants.** A0 full model; A1 midpoint-only supervision; A2 single unified head; A3 and A4 reduced models (A3 midpoint-only, A4 dual-label); C1–C6 each remove one component (difference branch, FrameSimilarity, ColorHistogramSimilarity, temporal delta, per-frame pathway, pixel motion, edge head, attention and gate); M0_min, M1_edge, B_noatt and S1_rel form the build-up ladder. The definitions are in `hybridsbd/configs/ablation_config.py`.

## Checkpoints

The paper's tables need **21 checkpoints** (749 MB in total): one `best_model.pt` per run, holding the EMA weights of the epoch with the best SHOT validation F1, the run configuration and the epoch.

| Runs | Used for |
|---|---|
| A0 | the primary model (Tables 3, 4, 6-11) |
| A1, A2, A3, A4 | supervision ablation (Table 7) |
| C1, C2a, C2b, C3, C4, C5a, C5b, C6 | leave-one-out components (Table 10) |
| M0_min, M1_edge, B_noatt, S1_rel | build-up ladder (Table 11, with A4 and A0) |
| A0_s43, A0_s44, A1_s43, A1_s44 | extra seeds (Table 8) |

- **Not stored in git.** `*.pt` is ignored. The files are in a Zenodo record, one `<run>.pt` per run (DOI `10.5281/zenodo.24144245`, reserved; the record becomes available when it is published).
- **Integrity.** [checkpoints/MANIFEST.csv](checkpoints/MANIFEST.csv) lists, for every file, the run, seed, table, size, SHA-256, best epoch, best validation F1 and parameter count. Check a download with `python checks/checkpoints.py verify --outputs <folder> --flat`.
- **Use.** `from hybridsbd.checkpoint import load_detector; model, ckpt = load_detector("A0.pt", "cuda")`. To re-run the evaluations, place the files in the training layout (`ablations_v2/<run>/best_model.pt`, `ablations_v3/<run>/best_model.pt`, `seed_runs/<run>/<A0|A1>/best_model.pt`) and pass that folder as `--outputs`.
- **Not needed for the paper and not published:** the per-epoch snapshots (`epoch_*.pt`, 12.8 GB) and the development runs that are not in any table.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.x and `torchcodec` for video decoding.

## Usage

Run everything from the repository root.

```bash
# Everything in the study: data preparation, training of all variants, evaluation, tables
bash scripts/run_full_study.sh

# One variant with the final training recipe, then SHOT / ClipShots / BBC evaluation
bash scripts/run_experiment.sh <name> --synth_prob 0.25 --ema_decay 0.999 --strong_aug

# Queues of runs: all variants, the build-up ladder, or the extra seeds of A0 and A1
bash scripts/run_queue.sh {variants|buildup|seeds}

# Evaluate a checkpoint on one test set (shot, clipshots or bbc)
python evaluation/evaluate.py shot --checkpoint <run>/best_model.pt --cache_root <cache> --gt_path <gt.pickle> --output shot_results.json

# Re-evaluate from the saved probability tracks: l_min = 1, and Protocol B (validation-selected thresholds)
python evaluation/reevaluate.py lmin1 --data_root <data> --outputs <outputs>
python evaluation/reevaluate.py protocol-b --data_root <data> --outputs <outputs>

# Speed benchmark, result checks
python benchmarks/speed.py ours --checkpoints <run>/best_model.pt
python checks/verify.py preflight --device cpu
python checks/checkpoints.py verify --outputs <outputs>   # SHA-256 of the 21 paper checkpoints against checkpoints/MANIFEST.csv

# Build the reports from server_results/ (add --pdf for the PDFs)
python report_tools/build_reports.py all

# Tests: scoring unit tests (pytest) and the CPU validation suite (dummy inputs, no data needed)
pytest tests && python tests/test_pipeline_cpu.py
```

The exact commands used for the extra seeds, ℓ_min = 1 and Protocol B are in [docs/follow_up_commands.md](docs/follow_up_commands.md).

## Project structure

```
hybridsbd/         Shared library (import as `hybridsbd.<package>.<module>`)
  models/          hybrid_boundary_detector.py (HybridBoundaryDetector, the main model and its building blocks),
                   ablation_model.py (flag-controlled variants built from those blocks; A0 == the main model)
  losses/          binary_boundary_loss.py (focal BCE dual-head loss), ablation_loss.py (dual / midpoint-only / unified)
  configs/         ablation_config.py (definitions of A0-A4, C1-C6 and the build-up ladder, the list of paper runs, environment snapshot)
  scoring.py       Scene-based scoring protocol and threshold search (numpy only; the single implementation used everywhere)
  datasets.py, frame_utils.py        Training datasets and video decoding
  checkpoint.py, logging_utils.py, opencv_setup.py, utils.py
training/          train.py: train one model variant; train_hybrid_detector.py: stand-alone trainer for the base HybridBoundaryDetector
evaluation/        evaluate.py (shot | clipshots | bbc), engine.py, data.py, reevaluate.py (lmin1 | protocol-b),
                   clipshots_type_analysis.py (cut / gradual analysis), export_results_csv.py
benchmarks/        speed.py (ours | baselines | summary)
data_prep/         Download, pre-decoding and audits (prepare_data.py, predecode_*.py, verify_data_cache.py, audit_dataset_overlap.py)
checks/            verify.py (preflight | a0-architecture | a0-reproduction | shot-labels | shot-split | fresh-pass),
                   checkpoints.py (build | verify the manifest of the 21 paper checkpoints)
checkpoints/       MANIFEST.csv (the 21 checkpoints: table, size, SHA-256, epoch, parameters); the .pt files are not in git
scripts/           Shell launchers: run_full_study.sh, run_experiment.sh, run_queue.sh, run_evaluation.sh, rerun_evaluations.sh, sync.sh,
                   upload_zenodo.sh (uploads a folder of checkpoints to a private Zenodo draft), ...
tests/             test_scoring.py (pytest), test_pipeline_cpu.py (CPU validation suite)
report_tools/      build_reports.py: all reports, the summary, the CSV and the PDFs from server_results/
reports/           Generated reports (Markdown, PDF) and the combined results CSV
docs/              Results summary, implementation notes, commands used for the follow-up experiments
server_results/    Local copy of the training server output (logs, histories, result JSON, checkpoints); not in git
```

The scripts add the repository root to the import path themselves.

## Citation

The manuscript is under review.

```bibtex
@unpublished{khanfar2026hybridsbd,
  title  = {HybridSBD: Efficient Dual-Label Shot Boundary Detection via Multi-Scale Dilated Convolutions and Temporal Attention},
  author = {Khanfar, Baraa and Qaroush, Aziz and Jubran, Mohammad and Qady, Mohammad and Kahla, Abdalmoumen},
  year   = {2026},
  note   = {Manuscript under review}
}
```
