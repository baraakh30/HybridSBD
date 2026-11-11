#!/usr/bin/env bash
# Train + evaluate ONE variant (used by scripts/run_full_study.sh, also runnable by hand: bash scripts/run_one_variant.sh C6)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
RUN_ID="$1"
exec > >(tee -a "$LOG_DIR/run_${RUN_ID}.log") 2>&1
echo "##### $RUN_ID start $(date -Iseconds) #####"
# Several jobs share the machine: cap CPU threads per process (default 48-thread pools thrash badly).
export OMP_NUM_THREADS="${THREADS_PER_JOB:-4}" MKL_NUM_THREADS="${THREADS_PER_JOB:-4}"   # overrides the 1-thread default of scripts/env.sh for training
SKIP_PREP=1 RUNS="$RUN_ID" bash scripts/run_variants.sh
STAGE=runs RUNS="$RUN_ID" bash scripts/run_evaluation.sh
if [[ "$RUN_ID" == "A0" ]]; then
  set +e
  "$PY" checks/verify.py a0-reproduction --run_dir "$OUTPUT_ROOT/A0"
  echo "A0 reproduction check exit code: $? (0 pass, 3 out of range -> report, ablations continue on the same code)"
  set -e
fi
echo "##### $RUN_ID done $(date -Iseconds) #####"
