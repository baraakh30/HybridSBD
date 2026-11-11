#!/usr/bin/env bash
# Train Protocol-A variants. One command per run, all 13 predefined.
#   bash scripts/run_variants.sh                 # all 13 in spec order
#   RUNS="A0" bash scripts/run_variants.sh       # only some runs
# Completed runs are skipped; an unfinished run aborts (use FORCE=1 to archive+retrain).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
RUNS="${RUNS:-$ALL_RUNS}"
EPOCHS="${EPOCHS:-80}"
FORCE_FLAG=""; [[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  "$PY" data_prep/prepare_data.py --data_root "$DATA_ROOT" --num_workers "$NUM_WORKERS" $EXCLUDE_FLAG 2>&1 | tee -a "$LOG_DIR/prepare_data.log"
fi

for RUN_ID in $RUNS; do
  echo "=== training $RUN_ID ==="
  "$PY" training/train.py --run_id "$RUN_ID" \
    --autoshot_cache "$AUTOSHOT_TRAIN_CACHE" --clipshots_cache "$CLIPSHOTS_CACHE" \
    --output_root "$OUTPUT_ROOT" --epochs "$EPOCHS" --num_workers "$TRAIN_WORKERS" $FORCE_FLAG
done
