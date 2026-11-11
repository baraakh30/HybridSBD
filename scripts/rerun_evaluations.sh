#!/usr/bin/env bash
# Re-run evaluations of finished checkpoints for the offline re-evaluations (evaluation/reevaluate.py).
#
#   RUNS="A0 A1" bash scripts/rerun_evaluations.sh fresh        # fresh forward pass on the three test sets (checks the saved probabilities)
#   RUNS="A0 A1" bash scripts/rerun_evaluations.sh validation   # the 67 SHOT validation videos as full videos (Protocol B thresholds)
#
# fresh:      the stored evaluation (ClipShots with the l_min grid 1-4, SHOT and BBC without merging) into
#             <outputs>/reeval_lmin1/fresh/<run>/; compare with `python checks/verify.py fresh-pass`.
# validation: SHOT validation videos, l_min = 1, into <outputs>/protocolB/val/<run>/shot_val_results.json; skips finished runs.
#             Needs the cache written by data_prep/predecode_shot_val.py.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/venv/main/bin:$PATH PYTHONUNBUFFERED=1
OUTPUTS="${OUTPUTS:-/workspace/outputs}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"

# run -> directory of its checkpoint (the layout used on the training server)
checkpoint() {
  local r="$1"
  case "$r" in
    A0_s4[34]|A1_s4[34]) echo "$OUTPUTS/seed_runs/$r/${r%_s*}/best_model.pt" ;;
    M0_min|M1_edge|B_noatt|S1_rel) echo "$OUTPUTS/ablations_v3/$r/best_model.pt" ;;
    *) echo "$OUTPUTS/ablations_v2/$r/best_model.pt" ;;
  esac
}

case "${1:-}" in
  fresh)
    export OUTPUT_ROOT="$OUTPUTS/reeval_lmin1"
    source scripts/env.sh
    for R in ${RUNS:-A0 A1}; do
      D="$OUTPUT_ROOT/fresh/$R"; CK=$(checkpoint "$R"); mkdir -p "$D"
      "$PY" evaluation/evaluate.py shot --checkpoint "$CK" --cache_root "$AUTOSHOT_TEST_CACHE" --gt_path "$AUTOSHOT_GT" --min_scene_lengths 1 --output "$D/shot_results.json"
      "$PY" evaluation/evaluate.py clipshots --checkpoint "$CK" --cache_root "$CLIPSHOTS_TEST_CACHE" --ann_path "$CLIPSHOTS_ANN" --min_scene_lengths 1 2 3 4 --output "$D/clipshots_results.json"
      "$PY" evaluation/evaluate.py bbc --checkpoint "$CK" --cache_root "$BBC_CACHE" --bbc_root "$BBC_ROOT" --output "$D/bbc_results.json"
      echo "FRESH_DONE $R $(date -Iseconds)"
    done
    ;;
  validation)
    export EVAL_WORKERS=8
    VAL="$DATA_ROOT/cache/autoshot_val_full"; OUT="$OUTPUTS/protocolB/val"
    mkdir -p "$OUT"
    for R in ${RUNS:-A0 A1 A2 A3 A4 C1 C2a C2b C3 C4 C5a C5b C6 M0_min M1_edge B_noatt S1_rel}; do
      mkdir -p "$OUT/$R"
      [ -f "$OUT/$R/shot_val_results.json" ] || python evaluation/evaluate.py shot --checkpoint "$(checkpoint "$R")" --cache_root "$VAL" \
          --gt_path "$VAL/val_gt_scenes.pickle" --min_scene_lengths 1 --output "$OUT/$R/shot_val_results.json" > "$OUT/$R/val.log" 2>&1
      echo "VAL_DONE $R $(date -Iseconds)"
    done
    ;;
  *) echo "usage: bash scripts/rerun_evaluations.sh {fresh|validation}" >&2; exit 2 ;;
esac
