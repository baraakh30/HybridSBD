#!/usr/bin/env bash
# Re-run the ablation variants (everything except A0) with the winning training recipe, into one shared output root.
# A0 of this recipe is the experiment run itself and is linked in as ablations_v2/A0 afterwards.
#   RECIPE_ARGS="--synth_prob 0.25 --ema_decay 0.999 --strong_aug" bash run_ablation_v2.sh
set -euo pipefail
cd "$(dirname "$0")"
export OUT="${OUT:-/workspace/outputs/ablations_v2}"
RUNS="${RUNS:-A1 A3 A4 A2 C2a C2b C4 C6 C1 C3 C5a C5b}"
PARALLEL="${PARALLEL:-3}"
: "${RECIPE_ARGS:?set RECIPE_ARGS to the winning recipe flags}"
mkdir -p "$OUT/logs"
echo "ablation v2: runs [$RUNS] parallel $PARALLEL recipe [$RECIPE_ARGS] ($(date -Iseconds))" | tee -a "$OUT/logs/v2_queue.log"
export RECIPE_ARGS
echo $RUNS | tr ' ' '\n' | xargs -P "$PARALLEL" -I{} bash -c 'RUN_ID={} bash run_experiment.sh v2_{} $RECIPE_ARGS || echo "FAILED {}" >> "$OUT/logs/v2_queue.log"'
echo "ablation v2 finished ($(date -Iseconds))" | tee -a "$OUT/logs/v2_queue.log"
