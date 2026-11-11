#!/usr/bin/env bash
# Queues of training runs, each run trained and evaluated by scripts/run_experiment.sh with the final recipe.
#
#   RECIPE_ARGS="--synth_prob 0.25 --ema_decay 0.999 --strong_aug" bash scripts/run_queue.sh variants
#   bash scripts/run_queue.sh buildup      # build-up ladder M0_min, M1_edge, B_noatt, S1_rel (SEED_RUN=1 adds A4 seed 43)
#   bash scripts/run_queue.sh seeds        # extra seeds 43 and 44 of A0 and A1 (JOBS="A0:43 A1:44" selects some)
#
# variants: every ablation variant except A0 (A0 of the recipe is the experiment run itself), PARALLEL jobs at a time,
#           into one shared output root OUT (default /workspace/outputs/ablations_v2).
# buildup:  the build-up ladder, PARALLEL jobs at a time, OUT default /workspace/outputs/ablations_v3.
# seeds:    all jobs at once on one GPU; each gets its own output root <SEEDS_OUT>/<RUN>_s<SEED>/<RUN>, and the train /
#           validation split stays at --split_seed 42 so that only the training seed changes.
# Failed runs are appended to <queue log> as "FAILED <run>".
set -euo pipefail
cd "$(dirname "$0")/.."
RECIPE_FINAL="--synth_prob 0.25 --ema_decay 0.999 --strong_aug"

parallel_queue() {  # parallel_queue <name> <runs> <parallel> <out> <prefix>
  local name="$1" runs="$2" parallel="$3" out="$4" prefix="$5" log="$4/logs/${5}_queue.log"
  mkdir -p "$out/logs"
  echo "$name: runs [$runs] parallel $parallel recipe [$RECIPE_ARGS] ($(date -Iseconds))" | tee -a "$log"
  export OUT="$out" RECIPE_ARGS
  echo "$runs" | tr ' ' '\n' | xargs -P "$parallel" -I{} bash -c 'RUN_ID={} bash scripts/run_experiment.sh '"$prefix"'_{} $RECIPE_ARGS || echo "FAILED {}" >> "'"$log"'"'
  echo "$name finished ($(date -Iseconds))" | tee -a "$log"
}

case "${1:-}" in
  variants)
    : "${RECIPE_ARGS:?set RECIPE_ARGS to the final recipe flags}"
    parallel_queue "ablation variants" "${RUNS:-A1 A3 A4 A2 C2a C2b C4 C6 C1 C3 C5a C5b}" "${PARALLEL:-3}" "${OUT:-/workspace/outputs/ablations_v2}" v2
    ;;
  buildup)
    RECIPE_ARGS="$RECIPE_FINAL"
    OUT="${OUT:-/workspace/outputs/ablations_v3}"
    parallel_queue "build-up ladder" "${RUNS:-M0_min M1_edge B_noatt S1_rel}" "${PARALLEL:-4}" "$OUT" v3
    if [ "${SEED_RUN:-0}" = 1 ]; then  # seed repeat of the backbone (noise floor); own root because run folder = run id
      OUT=/workspace/outputs/exp_A4_s43 RUN_ID=A4 bash scripts/run_experiment.sh A4_s43 $RECIPE_ARGS --seed 43 || echo "FAILED A4_s43" >> "$OUT/logs/v3_queue.log"
    fi
    ;;
  seeds)
    SEEDS_OUT="${SEEDS_OUT:-/workspace/outputs/seed_runs}"
    mkdir -p "$SEEDS_OUT/logs"
    for JOB in ${JOBS:-A0:43 A0:44 A1:43 A1:44}; do
      R=${JOB%%:*}; S=${JOB##*:}
      ( OUT="$SEEDS_OUT/${R}_s${S}" RUN_ID=$R bash scripts/run_experiment.sh "${R}_s${S}" $RECIPE_FINAL --seed "$S" --split_seed 42 \
          || echo "FAILED ${R}_s${S}" >> "$SEEDS_OUT/logs/queue.log" ) &
    done
    wait
    echo "seed runs finished $(date -Iseconds)" >> "$SEEDS_OUT/logs/queue.log"
    ;;
  *) echo "usage: bash scripts/run_queue.sh {variants|buildup|seeds}" >&2; exit 2 ;;
esac
