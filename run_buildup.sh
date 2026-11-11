#!/usr/bin/env bash
# Build-up ablation queue (minimal ladder M0->M1->B_noatt->A4, then add-one-in from the backbone A4 + cumulative stacks + backbone-internals + seed repeats).
# Same leak-free data, patience 10 and winning recipe as ablations_v2.   bash run_buildup.sh
set -euo pipefail
cd "$(dirname "$0")"
export OUT="${OUT:-/workspace/outputs/ablations_v3}"
RECIPE_ARGS="--synth_prob 0.25 --ema_decay 0.999 --strong_aug"
# first wave = 4 runs; the rest of the planned queue (not run yet): U_fsim U_csim U_delta U_diff U_motion S2_rel_pf S3_rel_pf_motion B_noedge
RUNS="${RUNS:-M0_min M1_edge B_noatt S1_rel}"
PARALLEL="${PARALLEL:-4}"
mkdir -p "$OUT/logs"
echo "buildup: runs [$RUNS] parallel $PARALLEL ($(date -Iseconds))" | tee -a "$OUT/logs/v3_queue.log"
export RECIPE_ARGS
echo $RUNS | tr ' ' '\n' | xargs -P "$PARALLEL" -I{} bash -c 'RUN_ID={} bash run_experiment.sh v3_{} $RECIPE_ARGS || echo "FAILED {}" >> "$OUT/logs/v3_queue.log"'
# seed repeat of the backbone (noise floor for the base): separate root because run folder = run id
if [ "${SEED_RUN:-0}" = 1 ]; then
  OUT=/workspace/outputs/exp_A4_s43 RUN_ID=A4 bash run_experiment.sh A4_s43 $RECIPE_ARGS --seed 43 || echo "FAILED A4_s43" >> "$OUT/logs/v3_queue.log"
fi
echo "buildup finished ($(date -Iseconds))" | tee -a "$OUT/logs/v3_queue.log"
