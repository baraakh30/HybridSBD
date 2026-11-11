# HybridSBD Ablation Implementation Summary

## Purpose

This document records the requested work, the clarified decisions, corrections
to earlier assumptions, and the implementation completed in the repository.

## Original request

The requested task was to prepare the project for a complete HybridSBD
ablation study. The project needed to:

- Download the ClipShots and AutoShot datasets from the supplied Google Drive
  folders.
- Correctly handle the archive files, including tar archives and possible split
  archive parts.
- Extract and predecode the datasets.
- Retrain the required models because the original trained checkpoints were
  not present in the repository or Git history.
- Implement every experiment required by the supplied ablation documents.
- Put all optional architectural components and training choices behind flags.
- Keep the intended current model as the default A0 model.
- Provide the required training launcher, evaluation scripts, tests, analyses,
  records, checkpoints, and CSV output.
- Handle cloud-server execution and failures clearly.
- Avoid unnecessary extra experiments and avoid claiming results that were not
  actually executed.

The authoritative documents were:

- [ABLATION_SPEC_REQUIRED_protocolA.md](./ABLATION_SPEC_REQUIRED_protocolA.md)
- [ABLATION_IMPLEMENTATION_PROMPT.md](./ABLATION_IMPLEMENTATION_PROMPT.md)

The user also clarified that evaluation should cover everything required by
those documents, including BBC, and that retraining is required when the old
checkpoint cannot be found.

## Clarifications and corrected assumptions

### Baseline model

An earlier inspection found that the current training script used
`main_dim=384` and three fusion blocks. The README results table clarified that
the intended reported baseline is:

```text
main_dim = 384
fusion blocks = 4
dual-head model
```

This is the model reported as 10.9M parameters with the target results:

```text
SHOT F1       = 0.9238
ClipShots F1  = 0.7934
BBC F1        = 0.9706
```

Therefore, A0 is defined as the full 384-dimensional, four-block,
dual-head model. The old three-block setting is treated as configuration drift,
not as the intended A0 baseline.

### BBC evaluation

BBC testing is required by the ablation specification. The required result
tables contain SHOT, ClipShots, and BBC F1 for every relevant variant. The
evaluation launcher therefore requires `BBC_ROOT` and `BBC_CACHE` unless the
user explicitly sets `SKIP_BBC=1` for an intentionally incomplete run.

### Evaluation versus retraining

The evaluation launcher does not retrain models. The workflow is separated:

1. Prepare data and train all 13 variants.
2. Evaluate the saved checkpoints.
3. Run the required analyses and export the final CSV.

## Implementation completed

### Variant configuration

[ablation_config.py](./ablation_config.py) defines exactly these 13 variants:

```text
A0 A1 A2 A3 A4 C1 C2a C2b C3 C4 C5a C5b C6
```

The default A0 configuration is full dual supervision with D=384 and four
fusion blocks. The flags cover:

- dual, midpoint-only, and unified supervision
- Branch B
- Branch C
- relational feature source
- FrameSimilarity
- ColorHistogramSimilarity
- temporal delta
- pixel motion
- edge-density features
- self-attention and its gate

### Flag-controlled model

[ablation_model.py](./ablation_model.py) implements the configurable model.

The model supports:

- dynamically computed fusion input dimensions
- Branch C fallback to Branch A for C4
- edge-head removal with consistent 512-dimensional embeddings
- attention and gate bypass for C6
- single unified output head for A2
- measured unified-target positive-rate bias initialization for A2

The supporting architecture changes are in
[hybrid_boundary_detector.py](./detection/hybrid_boundary_detector.py).

### Losses and training

[ablation_loss.py](./ablation_loss.py) implements:

- dual-label loss for A0/A4/C1/C2a/C2b/C3/C4/C5a/C5b/C6
- midpoint-only loss for A1/A3
- unified-target loss for A2
- z-head label smoothing for dual supervision
- no z-head gradient for midpoint-only variants

[train_ablation.py](./train_ablation.py) trains one variant at a time and
records:

- complete variant configuration
- measured A2 unified positive rate
- parameter counts
- environment and Git metadata
- checkpoints every epoch
- best checkpoint
- validation history
- training duration and summary

### Dataset preparation

[prepare_ablation_data.py](./prepare_ablation_data.py) provides a cloud-oriented
preparation path. It supports:

- the supplied ClipShots Google Drive folder
- the supplied AutoShot Google Drive folder
- `gdown` folder downloads
- tar, tar.gz, tgz, and zip archives
- clearly identifiable split archive parts
- safe extraction with path-traversal checks
- archive SHA-256 manifests
- extraction markers for resumability
- dataset-root discovery and validation
- invocation of the existing predecoders

The existing predecoders were updated to accept configurable roots and to use
stable MD5-based video-name seeds instead of Python's randomized `hash()`.

Relevant files:

- [predecode_dataset.py](./predecode_dataset.py)
- [predecode_autoshot_train.py](./predecode_autoshot_train.py)
- [predecode_autoshot.py](./predecode_autoshot.py)

### Training launcher

[run_ablation.sh](./run_ablation.sh) defines one training command for each of
the 13 variants.

### Evaluation and required analyses

[run_protocol_a_evaluation.sh](./run_protocol_a_evaluation.sh) now performs:

- all-variant preflight validation
- AutoShot test preparation
- dataset-overlap audit
- SHOT evaluation
- ClipShots evaluation
- ClipShots `l_min` sweep over `{1,2,3,4}`
- BBC evaluation
- B=16 latency and VRAM benchmarking
- ClipShots Table 8 and Table 9 analyses
- Section-7 CSV export

Supporting files:

- [validate_ablation.py](./validate_ablation.py)
- [audit_dataset_overlap.py](./audit_dataset_overlap.py)
- [analyze_clipshots_protocol_a.py](./analyze_clipshots_protocol_a.py)
- [load_ablation_model.py](./load_ablation_model.py)
- [export_ablation_csv.py](./export_ablation_csv.py)

The existing test scripts were updated to load ablation checkpoints:

- [test_autoshot.py](./test_autoshot.py)
- [test_clipshots.py](./test_clipshots.py)
- [test_bbc.py](./test_bbc.py)

The ClipShots test predecoder now produces complete 128-frame windows with
stride 64 and includes the final aligned window.

### Reproducibility

[utils.py](./utils.py) and the predecoders were updated to improve reproducibility:

- PyTorch is imported correctly for seed operations.
- `PYTHONHASHSEED` is set.
- Python, NumPy, and PyTorch seeds are initialized.
- DataLoader workers receive deterministic seeds.
- Video-name sampling uses stable MD5-based seeds.
- Environment and Git metadata are saved per run.

### Dependencies

[requirements.txt](./requirements.txt) now includes the packages required by
the current pipeline, including `torchcodec`, `psutil`, and `gdown`.

## Cloud execution

From a Linux cloud server:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare the data and train every variant:

```bash
bash run_ablation.sh
```

Run every required evaluation and analysis:

```bash
export BBC_ROOT=/path/to/bbc
export BBC_CACHE=/path/to/bbc_cache
bash run_protocol_a_evaluation.sh
```

The final outputs are written under `outputs/ablations/`, including:

- one directory per variant
- checkpoints
- per-run configuration and environment metadata
- validation history and summaries
- per-dataset result JSON files
- preflight validation JSON
- overlap audit JSON
- ClipShots analysis JSON
- benchmark JSON
- the Section-7 CSV

## Validation status

Superseded by the executed CPU validation recorded in [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md)
(`python test_ablation_cpu.py`, 17/17 passed on dummy inputs). Dataset download, predecoding, real training,
real-data evaluation, the A0 reproduction check, and GPU benchmarking have still NOT been executed; no
experimental numbers are claimed. Run everything with `bash run_protocol_a_all.sh` (see the notes).

## Files changed or added

Added:

- [ablation_config.py](./ablation_config.py)
- [ablation_loss.py](./ablation_loss.py)
- [ablation_model.py](./ablation_model.py)
- [analyze_clipshots_protocol_a.py](./analyze_clipshots_protocol_a.py)
- [audit_dataset_overlap.py](./audit_dataset_overlap.py)
- [export_ablation_csv.py](./export_ablation_csv.py)
- [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md)
- [load_ablation_model.py](./load_ablation_model.py)
- [prepare_ablation_data.py](./prepare_ablation_data.py)
- [run_ablation.sh](./run_ablation.sh)
- [run_protocol_a_evaluation.sh](./run_protocol_a_evaluation.sh)
- [train_ablation.py](./train_ablation.py)
- [validate_ablation.py](./validate_ablation.py)

Modified:

- [detection/hybrid_boundary_detector.py](./detection/hybrid_boundary_detector.py)
- [predecode_autoshot.py](./predecode_autoshot.py)
- [predecode_autoshot_train.py](./predecode_autoshot_train.py)
- [predecode_dataset.py](./predecode_dataset.py)
- [requirements.txt](./requirements.txt)
- [test_autoshot.py](./test_autoshot.py)
- [test_bbc.py](./test_bbc.py)
- [test_clipshots.py](./test_clipshots.py)
- [train_hybrid_detector.py](./train_hybrid_detector.py)
- [utils.py](./utils.py)

No trained model results are included because the training and evaluation jobs
still need to run on the cloud server with the external datasets and GPU.
