#!/usr/bin/env bash
# ONE COMMAND for everything the two ABLATION md files require, in spec order.
#
#   (BBC is downloaded automatically from Zenodo record 14865504 into $DATA_ROOT/BBC;
#    set BBC_ROOT=/path to use an existing copy laid out as bbc_01.mp4 ... + annotations/shots/)
#   bash scripts/run_full_study.sh
#
# Optional: DATA_ROOT, OUTPUT_ROOT, EPOCHS, NUM_WORKERS, SKIP_PREP=1 (data already prepared),
#           SKIP_BBC=1 (explicitly incomplete run), FORCE=1 (archive + retrain unfinished runs).
# Re-running resumes: finished trainings and result files are skipped, nothing is deleted.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1
echo "##### Protocol-A pipeline start $(date -Iseconds) #####"

step() { echo; echo "########## $* ##########"; }

step "0. manifest (git commit, dirty state, environment)"
"$PY" - <<PYEOF
from hybridsbd.configs.ablation_config import environment_snapshot, write_json
import subprocess
env = environment_snapshot()
env["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
env["note"] = "All 13 runs must use this commit (spec Section 1)."
write_json("$OUTPUT_ROOT/run_manifest.json", env)
print(env)
if env["git_dirty"]:
    print("WARNING: working tree has uncommitted changes. Commit first so all 13 runs share one reproducible commit "
          "(each run also stores uncommitted.diff).")
PYEOF
"$PY" -c "import torch, torchcodec, psutil, tqdm, numpy" || { echo "Missing dependencies: pip install -r requirements.txt" >&2; exit 1; }

step "1. checklist: CPU validation of all variants (Check 1: architecture, parameter counts, A0 == legacy model)"
"$PY" tests/test_pipeline_cpu.py
"$PY" checks/verify.py preflight --device cpu --output "$OUTPUT_ROOT/preflight.json" > "$LOG_DIR/preflight.log" 2>&1

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  step "2. download, extract, predecode ClipShots + AutoShot (train)"
  "$PY" data_prep/prepare_data.py --data_root "$DATA_ROOT" --num_workers "$NUM_WORKERS" $EXCLUDE_FLAG 2>&1 | tee -a "$LOG_DIR/prepare_data.log"
fi
source scripts/env.sh   # re-read data/paths.json written by the preparation step

step "3. SHOT test + BBC predecode, data-overlap audit (before any training)"
STAGE=setup bash scripts/run_evaluation.sh

step "3b. verify every cache (all clips readable and well-formed, no video missing) - training starts only if this passes"
"$PY" data_prep/verify_data_cache.py --data_root "$DATA_ROOT" --output "$OUTPUT_ROOT/data_verification.json" $VERIFY_FLAGS --workers "$NUM_WORKERS"

if [[ "${STOP_AFTER_SETUP:-0}" == "1" ]]; then
  echo "STOP_AFTER_SETUP=1: data prepared and overlap audit passed. Re-run without it to start training."
  exit 0
fi

step "4. train + evaluate all 13 runs, ${PARALLEL} at a time (A0 first; its reproduction Check 2 runs when it finishes)"
# Runs are independent processes sharing the GPU: each keeps its own seed, so results do not depend on
# the degree of parallelism (only wall-clock does). Benchmarks (step 6) run alone afterwards.
RUN_ORDER="A0 A1 A3 A4 A2 C2a C2b C4 C6 C1 C3 C5a C5b"   # spec Section 5 order
set +e
printf '%s\n' $RUN_ORDER | xargs -P "${PARALLEL}" -I{} bash scripts/run_one_variant.sh {}
set -e
FAILED=""
for r in $ALL_RUNS; do
  D="$OUTPUT_ROOT/$r"
  [[ -f "$D/summary.json" && -f "$D/shot_results.json" && -f "$D/clipshots_results.json" ]] || FAILED="$FAILED $r"
  [[ "${SKIP_BBC:-0}" == "1" || -f "$D/bbc_results.json" ]] || FAILED="$FAILED $r"
done
if [[ -n "$FAILED" ]]; then
  echo "INCOMPLETE runs:$FAILED  (see $LOG_DIR/run_<ID>.log). Re-run the same command to resume; finished work is kept." >&2
  exit 1
fi

step "5. Table 8/9 analysis, B=16 benchmark, Section-7 CSV"
STAGE=post bash scripts/run_evaluation.sh

step "DONE"
echo "CSV:            $OUTPUT_ROOT/ablation_results.csv   (+ .missing.json = anything incomplete)"
echo "Tables 7a/7b/8/9: $OUTPUT_ROOT/result_tables.md"
echo "Table 8/9 data:  $OUTPUT_ROOT/clipshots_protocol_a_analysis.json"
echo "A0 check 2:     $OUTPUT_ROOT/A0/a0_reproduction_check.json"
echo "Logs:           $LOG_DIR and <run>/train.log, <run>/*_results.log"
