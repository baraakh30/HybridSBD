#!/usr/bin/env bash
# Train one variant with extra training options, then run the normal SHOT / ClipShots / BBC evaluation.
# Leak-free data (SHOT test videos excluded), patience 10, spec thresholds. NOT part of the original 13-run pipeline.
#   bash run_experiment.sh synth_ema --synth_prob 0.25 --ema_decay 0.999 --strong_aug          # A0 -> outputs/exp_synth_ema/A0
#   RUN_ID=C6 OUT=/workspace/outputs/ablations_v2 bash run_experiment.sh v2_C6 <same flags>      # another variant, shared root
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/venv/main/bin:$PATH PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
EXP="$1"; shift
RUN_ID="${RUN_ID:-A0}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}" OUTPUT_ROOT="${OUT:-/workspace/outputs/exp_$EXP}"
source ./protocol_a_env.sh
mkdir -p "$OUTPUT_ROOT/logs"
exec > >(tee -a "$OUTPUT_ROOT/logs/exp_$EXP.log") 2>&1
echo "== experiment $EXP run $RUN_ID: extra args: $* ($(date -Iseconds))"
"$PY" train_ablation.py --run_id "$RUN_ID" --autoshot_cache "$AUTOSHOT_TRAIN_CACHE" --clipshots_cache "$CLIPSHOTS_CACHE" \
  --output_root "$OUTPUT_ROOT" --epochs 80 --num_workers "$TRAIN_WORKERS" "$@"
STAGE=runs RUNS="$RUN_ID" bash run_protocol_a_evaluation.sh
echo "== experiment $EXP run $RUN_ID done ($(date -Iseconds))"
