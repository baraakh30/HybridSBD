#!/usr/bin/env bash
# Extra seeds for A0 and A1 (seeds 43, 44). Same recipe as the seed-42 runs; the train/val split stays at split_seed 42.
# Each run gets its own output root (run folder = run id), like the earlier seed repeat:  /workspace/outputs/seed_runs/<RUN>_s<SEED>/<RUN>
set -uo pipefail
cd "$(dirname "$0")"
RECIPE_ARGS="--synth_prob 0.25 --ema_decay 0.999 --strong_aug"
mkdir -p /workspace/outputs/seed_runs/logs
for JOB in ${JOBS:-A0:43 A0:44 A1:43 A1:44}; do
  R=${JOB%%:*}; S=${JOB##*:}
  ( OUT=/workspace/outputs/seed_runs/${R}_s${S} RUN_ID=$R bash run_experiment.sh ${R}_s${S} $RECIPE_ARGS --seed $S --split_seed 42 \
      || echo "FAILED ${R}_s${S}" >> /workspace/outputs/seed_runs/logs/queue.log ) &
done
wait
echo "seed runs finished $(date -Iseconds)" >> /workspace/outputs/seed_runs/logs/queue.log
