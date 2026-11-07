#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablations}"
AUTOSHOT_CACHE="${AUTOSHOT_TEST_CACHE:-$DATA_ROOT/cache/autoshot_test}"
AUTOSHOT_TRAIN_CACHE="${AUTOSHOT_TRAIN_CACHE:-$DATA_ROOT/cache/autoshot_train}"
CLIPSHOTS_CACHE="${CLIPSHOTS_TEST_CACHE:-$DATA_ROOT/cache/clipshots/test}"
CLIPSHOTS_ANN="${CLIPSHOTS_ANN:-$DATA_ROOT/ClipShots/annotations/test.json}"
BBC_CACHE="${BBC_CACHE:-$DATA_ROOT/cache/bbc}"
BBC_ROOT="${BBC_ROOT:-}"

python validate_ablation.py --device cpu --output "$OUTPUT_ROOT/preflight.json"

export AUTOSHOT_ROOT="${AUTOSHOT_ROOT:-$DATA_ROOT/AutoShot}"
export AUTOSHOT_TEST_CACHE="$AUTOSHOT_CACHE"
python predecode_autoshot.py

if [[ ! -f "$CLIPSHOTS_CACHE/index_mapping.json" && ! -f "$CLIPSHOTS_CACHE/test/index_mapping.json" ]]; then
  echo "ClipShots test cache is missing: $CLIPSHOTS_CACHE" >&2
  exit 1
fi
if [[ -z "$BBC_ROOT" && "${SKIP_BBC:-0}" != "1" ]]; then
  echo "BBC_ROOT is required by Protocol A. Set SKIP_BBC=1 only for an explicitly incomplete run." >&2
  exit 1
fi
if [[ ! -d "$BBC_CACHE" && "${SKIP_BBC:-0}" != "1" ]]; then
  echo "BBC_CACHE is required by Protocol A: $BBC_CACHE" >&2
  exit 1
fi

if [[ -d "$BBC_CACHE" && "${SKIP_BBC:-0}" != "1" ]]; then
  python audit_dataset_overlap.py \
    --dataset clipshots_train "$DATA_ROOT/cache/clipshots/train" \
    --dataset clipshots_test "$CLIPSHOTS_CACHE" \
    --dataset autoshot_train "$AUTOSHOT_TRAIN_CACHE" \
    --dataset autoshot_test "$AUTOSHOT_CACHE" \
    --dataset bbc "$BBC_CACHE" \
    --output "$OUTPUT_ROOT/data_overlap.json"
fi

for RUN_ID in A0 A1 A2 A3 A4 C1 C2a C2b C3 C4 C5a C5b C6; do
  CKPT="$OUTPUT_ROOT/$RUN_ID/best_model.pt"
  test -f "$CKPT"
  python test_autoshot.py --checkpoint "$CKPT" --cache_root "$AUTOSHOT_CACHE" \
    --min_scene_lengths 1 --output "$OUTPUT_ROOT/$RUN_ID/shot_results.json"
  python test_clipshots.py --checkpoint "$CKPT" --cache_root "$CLIPSHOTS_CACHE" \
    --ann_path "$CLIPSHOTS_ANN" --min_scene_lengths 1 2 3 4 \
    --output "$OUTPUT_ROOT/$RUN_ID/clipshots_results.json"
  if [[ -n "$BBC_ROOT" ]]; then
    python test_bbc.py --checkpoint "$CKPT" --cache_root "$BBC_CACHE" \
      --bbc_root "$BBC_ROOT" --output "$OUTPUT_ROOT/$RUN_ID/bbc_results.json"
  fi
done

python analyze_clipshots_protocol_a.py \
  --checkpoint_root "$OUTPUT_ROOT" \
  --cache_root "$CLIPSHOTS_CACHE" \
  --ann_path "$CLIPSHOTS_ANN" \
  --output "$OUTPUT_ROOT/clipshots_protocol_a_analysis.json"

python benchmark_model.py --device cuda --batch_sizes 16 \
  --checkpoints "$OUTPUT_ROOT"/*/best_model.pt \
  --output "$OUTPUT_ROOT/benchmark_results.json"
python export_ablation_csv.py --output_root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/ablation_results.csv"
