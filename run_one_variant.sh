#!/usr/bin/env bash
# Train + evaluate ONE variant (used by run_protocol_a_all.sh, also runnable by hand: bash run_one_variant.sh C6)
set -euo pipefail
cd "$(dirname "$0")"
source ./protocol_a_env.sh
RUN_ID="$1"
exec > >(tee -a "$LOG_DIR/run_${RUN_ID}.log") 2>&1
echo "##### $RUN_ID start $(date -Iseconds) #####"
# Several jobs share the machine: cap CPU threads per process (default 48-thread pools thrash badly).
export OMP_NUM_THREADS="${THREADS_PER_JOB:-4}" MKL_NUM_THREADS="${THREADS_PER_JOB:-4}"   # overrides the 1-thread default of protocol_a_env.sh for training
SKIP_PREP=1 RUNS="$RUN_ID" bash run_ablation.sh
STAGE=runs RUNS="$RUN_ID" bash run_protocol_a_evaluation.sh
if [[ "$RUN_ID" == "A0" ]]; then
  set +e
  "$PY" check_a0_reproduction.py --run_dir "$OUTPUT_ROOT/A0"
  echo "A0 reproduction check exit code: $? (0 pass, 3 out of range -> report, ablations continue on the same code)"
  set -e
fi
echo "##### $RUN_ID done $(date -Iseconds) #####"
