# HybridSBD Protocol-A implementation notes

## Baseline

The intended A0 baseline is the reported `main_dim=384`, four-fusion-block,
dual-head model from `README.md`. The ablation defaults are therefore full
components, dual supervision, D=384, and L=4.

## Implemented pieces

- `ablation_config.py` defines exactly A0, A1, A2, A3, A4, C1, C2a, C2b, C3,
  C4, C5a, C5b, and C6.
- `ablation_model.py` gates all required feature streams, edge features,
  attention/gating, and prediction-head choices.
- `ablation_loss.py` implements dual, midpoint-only, and unified supervision.
- `train_ablation.py` records environment metadata, parameters, checkpoints,
  validation history, and A2's measured unified-target positive rate.
- `prepare_ablation_data.py` downloads from the supplied Drive folders when
  `gdown` is available, extracts tar/zip archives safely, joins clear split
  archive members, validates dataset roots, and invokes both predecoders.
- `run_ablation.sh` contains all 13 training commands.
- `export_ablation_csv.py` emits the Section-7 CSV schema.
- Existing test and benchmark loaders now accept ablation checkpoints.
- ClipShots test predecoding now creates complete 128-frame windows at stride
  64 and includes the final aligned window.
- Protocol-A threshold search now supports ClipShots minimum scene lengths
  `{1, 2, 3, 4}`. Short predicted scenes are merged with the preceding scene,
  matching the repository's existing evaluation interpretation.
- Seed handling now imports PyTorch correctly, sets `PYTHONHASHSEED`, seeds
  workers, and replaces video-name `hash()` sampling with stable MD5 sampling.

## Validation status

VS Code diagnostics reported no errors for the modified and added Python files.
Full forward/backward tests and data preparation were not run because this
environment has no Python executable, CUDA runtime, or downloaded datasets.
The first cloud run should execute the CPU dummy-variant validation before
starting any training.

## Dataset limitation

The Drive folders were not enumerable from this environment. The preparation
script therefore discovers archives after download and fails with a clear
message if the folder requires authentication, is rate-limited, or does not
contain recognizable ClipShots/AutoShot roots. It does not silently merge
unrelated archives.

## Remaining execution work

1. Install dependencies, including `torchcodec`, `psutil`, and `gdown`.
2. Run `prepare_ablation_data.py` on the cloud server.
3. Run the 13 launcher commands in `run_ablation.sh`.
4. Run the required SHOT, ClipShots, BBC, and B=16 benchmark evaluations on
   the resulting checkpoints and merge their JSON results into the CSV.

No experimental result is claimed until those commands execute successfully.
