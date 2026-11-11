#!/usr/bin/env bash
# Protocol-A evaluation (test-set-selected thresholds), in separable stages:
#   STAGE=setup  SHOT test predecode + data-overlap audit   (run BEFORE training)
#   STAGE=runs   SHOT/ClipShots/BBC evaluation of $RUNS     (default: all 13)
#   STAGE=post   Table 8/9 analysis + B=16 benchmark + Section-7 CSV
#   STAGE=all    setup + runs + post (default)
# Finished result files are kept (resumable); delete a file to recompute it.
set -euo pipefail
cd "$(dirname "$0")"
source ./protocol_a_env.sh
STAGE="${STAGE:-all}"
RUNS="${RUNS:-$ALL_RUNS}"
SKIP_BBC="${SKIP_BBC:-0}"

setup() {
  if [[ "$SKIP_BBC" != "1" ]]; then
    if [[ ! -f "$BBC_ROOT/bbc_manifest.json" ]]; then   # written only after all 23 files are md5-verified
      "$PY" prepare_bbc_data.py --output "$BBC_ROOT" 2>&1 | tee -a "$LOG_DIR/prepare_bbc.log"
    fi
    if [[ ! -f "$BBC_CACHE/index_mapping.json" ]]; then
      "$PY" predecode_bbc.py --bbc_root "$BBC_ROOT" --output_root "$BBC_CACHE" 2>&1 | tee -a "$LOG_DIR/predecode_bbc.log"
    fi
  fi
  if [[ ! -f "$AUTOSHOT_TEST_CACHE/index_mapping.json" ]]; then
    AUTOSHOT_ROOT="$AUTOSHOT_ROOT" AUTOSHOT_TEST_CACHE="$AUTOSHOT_TEST_CACHE" \
      "$PY" predecode_autoshot.py 2>&1 | tee -a "$LOG_DIR/predecode_autoshot_test.log"
  fi
  [[ -f "$CLIPSHOTS_TEST_CACHE/index_mapping.json" ]] || { echo "ClipShots test cache missing: $CLIPSHOTS_TEST_CACHE" >&2; exit 1; }
  AUDIT=(--dataset clipshots_train "$CLIPSHOTS_CACHE/train" --dataset clipshots_test "$CLIPSHOTS_TEST_CACHE"
         --dataset autoshot_train "$AUTOSHOT_TRAIN_CACHE" --dataset autoshot_test "$AUTOSHOT_TEST_CACHE")
  [[ "$SKIP_BBC" == "1" ]] || AUDIT+=(--dataset bbc "$BBC_CACHE")
  "$PY" audit_dataset_overlap.py "${AUDIT[@]}" $ALLOW_PAIR_FLAG --output "$OUTPUT_ROOT/data_overlap.json" 2>&1 | tee -a "$LOG_DIR/audit_overlap.log"
}

evaluate_run() {
  local RUN_ID="$1" CKPT="$OUTPUT_ROOT/$1/best_model.pt" D="$OUTPUT_ROOT/$1"
  [[ -f "$CKPT" ]] || { echo "Missing checkpoint $CKPT" >&2; exit 1; }
  [[ -f "$D/shot_results.json" ]] || "$PY" test_autoshot.py --checkpoint "$CKPT" --cache_root "$AUTOSHOT_TEST_CACHE" \
      --gt_path "$AUTOSHOT_GT" --min_scene_lengths 1 --output "$D/shot_results.json"
  [[ -f "$D/clipshots_results.json" ]] || "$PY" test_clipshots.py --checkpoint "$CKPT" --cache_root "$CLIPSHOTS_TEST_CACHE" \
      --ann_path "$CLIPSHOTS_ANN" --min_scene_lengths 1 2 3 4 --output "$D/clipshots_results.json"
  if [[ "$SKIP_BBC" != "1" && ! -f "$D/bbc_results.json" ]]; then
    "$PY" test_bbc.py --checkpoint "$CKPT" --cache_root "$BBC_CACHE" --bbc_root "$BBC_ROOT" --output "$D/bbc_results.json"
  fi
}

post() {
  "$PY" analyze_clipshots_protocol_a.py --checkpoint_root "$OUTPUT_ROOT" --cache_root "$CLIPSHOTS_TEST_CACHE" \
      --ann_path "$CLIPSHOTS_ANN" --output "$OUTPUT_ROOT/clipshots_protocol_a_analysis.json"
  if ! "$PY" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "WARNING: no CUDA GPU -> benchmark latency/VRAM would be CPU numbers, NOT valid for Table 7. Skipping benchmark." >&2
  else
    "$PY" benchmark_model.py --device cuda --batch_sizes 16 \
        --checkpoints $(for r in $ALL_RUNS; do echo "$OUTPUT_ROOT/$r/best_model.pt"; done) \
        --output "$OUTPUT_ROOT/benchmark_results.json"
  fi
  "$PY" export_ablation_csv.py --output_root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/ablation_results.csv"
  "$PY" make_result_tables.py --csv "$OUTPUT_ROOT/ablation_results.csv" \
      --analysis "$OUTPUT_ROOT/clipshots_protocol_a_analysis.json" --output "$OUTPUT_ROOT/result_tables.md"
}

case "$STAGE" in
  setup) setup ;;
  runs)  for r in $RUNS; do echo "=== evaluating $r ==="; evaluate_run "$r"; done ;;
  post)  post ;;
  all)   setup; for r in $RUNS; do echo "=== evaluating $r ==="; evaluate_run "$r"; done; post ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 1 ;;
esac
