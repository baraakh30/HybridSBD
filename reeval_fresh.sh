#!/usr/bin/env bash
# Fresh forward pass + ORIGINAL evaluation (l_min grid 1-4 on ClipShots) for a few runs, into a scratch dir.
# Used to confirm that saved probabilities equal a fresh forward pass and that the previous numbers reproduce.
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/venv/main/bin:$PATH PYTHONUNBUFFERED=1 DATA_ROOT=/workspace/data OUTPUT_ROOT=/workspace/outputs/reeval_lmin1
source ./protocol_a_env.sh
SRC=/workspace/outputs; ROOT=$OUTPUT_ROOT/fresh
for R in ${RUNS:-A0 A1}; do
  D=$ROOT/$R; mkdir -p "$D"; CK=$SRC/ablations_v2/$R/best_model.pt
  "$PY" test_autoshot.py --checkpoint "$CK" --cache_root "$AUTOSHOT_TEST_CACHE" --gt_path "$AUTOSHOT_GT" --min_scene_lengths 1 --output "$D/shot_results.json"
  "$PY" test_clipshots.py --checkpoint "$CK" --cache_root "$CLIPSHOTS_TEST_CACHE" --ann_path "$CLIPSHOTS_ANN" --min_scene_lengths 1 2 3 4 --output "$D/clipshots_results.json"
  "$PY" test_bbc.py --checkpoint "$CK" --cache_root "$BBC_CACHE" --bbc_root "$BBC_ROOT" --output "$D/bbc_results.json"
  echo "FRESH_DONE $R $(date -Iseconds)"
done
