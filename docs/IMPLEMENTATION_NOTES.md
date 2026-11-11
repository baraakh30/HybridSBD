# HybridSBD Protocol-A ablation — implementation notes

Authoritative specs: `ABLATION_SPEC_REQUIRED_protocolA.md`, `ABLATION_IMPLEMENTATION_PROMPT.md`.

## Run everything (one command)

```bash
pip install -r requirements.txt
# BBC (Zenodo 14865504) is downloaded + md5-verified automatically into data/BBC by data_prep/prepare_bbc_data.py;
# set BBC_ROOT=/path only to reuse an existing copy (bbc_01.mp4 ... + annotations/shots/).
git add -A && git commit -m "Protocol-A ablation code"   # all 13 runs must share one commit
bash scripts/run_full_study.sh
```

Order executed: manifest -> CPU validation (Check 1) -> data download/extract/predecode ->
SHOT-test + BBC predecode + data-overlap audit -> **A0 train, evaluate, Check 2** ->
remaining 12 runs (spec Section 5 order) -> Table 8/9 analysis -> B=16 benchmark -> Section-7 CSV.
Re-running resumes (finished runs / result files are skipped; nothing is deleted;
`FORCE=1` archives an unfinished run to `<run>.archived_<time>` before retraining).
Single steps: `RUNS="A0" bash scripts/run_variants.sh`, `STAGE=setup|runs|post bash scripts/run_evaluation.sh`.

## What is saved (nothing is only printed)

| Where | Content |
|---|---|
| `outputs/ablations/logs/pipeline.log` | full console log of the pipeline (+ per-step logs) |
| `outputs/ablations/run_manifest.json` | git commit, dirty flag, Python/PyTorch/CUDA/GPU |
| `<run>/train.log` | every line printed during training, incl. tracebacks |
| `<run>/config.json` | variant flags, args, environment, git state, exact parameter counts, fusion size, A2 measured positive rate |
| `<run>/history.json`, `epoch_NNN.pt`, `best_model.pt` | per-epoch metrics, every checkpoint, best checkpoint |
| `<run>/summary.json`, `status.json` | best_val_f1, **best_epoch**, hours, early stop flag; running/completed/failed (+ traceback) |
| `<run>/uncommitted.diff` | code differences from the commit, if the tree was dirty |
| `<run>/{shot,clipshots,bbc}_results.json` | full threshold-search grid (`all_results`), per-video results, best thresholds, metadata (checkpoint SHA-256, inference rule, environment) |
| `<run>/*_results.predictions.npz` | per-video probabilities: thresholds / tables can be recomputed without a GPU |
| `<run>/*_results.log` | evaluation console log |
| `outputs/ablations/clipshots_protocol_a_analysis.json` | Tables 8 and 9 (+ definitions, GT-type source counts) |
| `outputs/ablations/benchmark_results.json` | B=16 latency/VRAM + environment |
| `outputs/ablations/result_tables.md` | spec Section 8 Tables 7a, 7b, 8, 9 filled from the CSV and analysis JSON (`report_tools/build_reports.py`) |
| `outputs/ablations/ablation_results.csv` (+`.missing.json`) | spec Section 7 CSV; `.missing.json` lists anything incomplete or commit mismatches |
| `outputs/ablations/data_overlap.json`, `preflight.json`, `cpu_validation.json` | overlap audit, per-variant preflight, CPU test results |

## Implementation decisions

* **`model.py`** is a flag-driven copy of `HybridBoundaryDetector` (original untouched).
  Stream order is identical to the original `[rgb, diff, delta, fsim, csim, motion]` and the feature
  dropout (0.15) is applied **only** to the FrameSimilarity input (default 3 of the prompt: whichever
  feature feeds FrameSimilarity); delta reads the un-dropped feature. A0 is verified identical to the original
  (same keys, shapes, parameter count and outputs with the same weights).
* **A2** has one head; its forward returns `z = None` so no consumer can mistake a placeholder for a prediction.
  Head bias prior = measured positive rate of `u = max(one_hot, many_hot)` over the training clips (default 2).
* **Inference rules** (default 1): validation during training and all test scripts use each variant's own rule:
  dual `b = (p_z > tau_g) OR (p_y > tau_c)`; A1/A2/A3 `b = p_y > tau_c`, tau_c grid only
  (`gradual_threshold=None` -> `best_gradual_threshold: null`).
* **Gradual IoU / gradual F1 / cut-gradual typing** follow spec Section 6: see the docstring of
  `evaluation/clipshots_type_analysis.py`. GT type uses annotation type labels when present
  (keys `types`, `transition_types`, `labels`, `type`, `transition_labels`), otherwise `end - start <= 1` = cut (default 4);
  the number of transitions typed by each source is recorded in the output.
* **A0 reproduction**: Check 1 (architecture) is enforced by `tests/test_pipeline_cpu.py` / `checks/verify.py`;
  Check 2 (F1 ranges 0.915–0.930 / 0.785–0.800 / 0.965–0.975) by `checks/verify.py`
  (a failure is reported and does not stop the ablations, as the spec allows).

## Spec checklist documentation (Section 0)

* **l_min (item 4):** applied in `eval_shot.apply_min_scene_length` inside the threshold sweep
  (`evaluate`). Predicted scenes shorter than l_min are merged into the preceding scene. The grid {1,2,3,4}
  is used for ClipShots only; SHOT is evaluated with l_min = 1 and BBC does not apply it.
* **Sliding window (item 5):** T = 128, stride 64. Overlapping predictions are averaged
  (`reconstruct_full_video`: sum / count per frame). Frames 0–63 are covered by one window only (no special edge
  handling). SHOT/BBC predecoders start windows at `range(0, total, 64)` and pad an incomplete final window by repeating the
  last frame; BBC clips predictions to the annotated range. ClipShots test windows (rewritten predecoder) are
  complete 128-frame windows at stride 64 plus one final aligned window (no padding). Identical for all runs.
  Videos shorter than 128 frames in the ClipShots test split have not been checked (needs the data).
* **Seeds (item 3):** MD5-based sampling seed in all predecoders; `PYTHONHASHSEED=0` exported by the launchers
  (a value set inside Python after start-up has no effect); Python/NumPy/PyTorch/CUDA and DataLoader workers seeded.
  The ablation trainer does not use full deterministic-algorithm mode beyond `cudnn.deterministic`.
* **Overlap check (item 6):** `data_prep/audit_dataset_overlap.py`, run before any training; aborts on overlap.
* **Benchmark hygiene (item 7):** `reset_peak_memory_stats` + `empty_cache` before both latency and VRAM measurement.
* **Environment (item 8):** recorded per run (`config.json`) and for the pipeline (`run_manifest.json`).

## Validation actually executed (CPU, dummy inputs, PyTorch 2.15.0, Python 3.15.2)

`python tests/test_pipeline_cpu.py` -> **17/17 passed** (details in `outputs/ablations/cpu_validation.json`):
default config == legacy model (402 keys, 10,938,880 parameters, strict load, identical outputs);
A0 architecture (4 blocks, delta weight (64, 640), fusion 655, no positional encoding, 10.9M);
construction/forward/fusion size/parameter counts for all 13 variants; component bypassing (incl. C4 never calling
the per-frame path, C5b 512-d embeddings); heads (A2 = one head, 223,105 fewer parameters than A0); loss + backward
for all 13 (A1/A3 z-head gradient is zero); AdamW + clipping step; checkpoint round-trip; stable seeds;
evaluation pipeline for dual and single-output models; l_min merge and window averaging; Table 8/9 logic on a hand-built
case; CSV header equals spec Section 7.
In addition, on synthetic caches (random weights, meaningless F1): 2-epoch training of A0 and A2 produced all
records; SHOT/ClipShots evaluation, Table 8/9 analysis, reproduction check and CSV export ran end to end.

Exact parameter counts: A0 10,938,880; A1 10,938,880; A2 10,715,775; A3 = A4 8,983,570; C1 9,293,744;
C2a 10,833,504; C2b 10,835,936; C3 10,848,576; C4 10,938,880; C5a 10,927,330; C5b 10,716,864; C6 7,393,792.

## NOT executed (needs your machine / data)

Dataset download and predecoding, real training of the 13 runs, real SHOT/ClipShots/BBC evaluation, the A0
reproduction Check 2, the GPU B=16 latency/VRAM benchmark, and the real-data overlap audit. No experimental number is
claimed. The Google Drive folders could not be enumerated from the authoring environment; if `gdown` is rate-limited,
place the archives in `data/downloads/{clipshots,autoshot}` and re-run with the same command.
