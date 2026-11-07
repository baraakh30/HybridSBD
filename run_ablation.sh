#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
AUTOSHOT_CACHE="${AUTOSHOT_CACHE:-$DATA_ROOT/cache/autoshot_train}"
CLIPSHOTS_CACHE="${CLIPSHOTS_CACHE:-$DATA_ROOT/cache/clipshots}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablations}"
EPOCHS="${EPOCHS:-80}"

python prepare_ablation_data.py --data_root "$DATA_ROOT" --num_workers "${NUM_WORKERS:-8}"

for RUN_ID in A0 A1 A2 A3 A4 C1 C2a C2b C3 C4 C5a C5b C6; do
  python train_ablation.py --run_id "$RUN_ID" \
    --autoshot_cache "$AUTOSHOT_CACHE" \
    --clipshots_cache "$CLIPSHOTS_CACHE" \
    --output_root "$OUTPUT_ROOT" --epochs "$EPOCHS"
done

python export_ablation_csv.py --output_root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/ablation_results.csv"
