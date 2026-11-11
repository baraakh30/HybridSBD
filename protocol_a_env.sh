# Shared settings for the Protocol-A launchers (sourced, not executed).
export PYTHONHASHSEED=0                      # spec checklist item 3
PY="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablations}"
ALL_RUNS="A0 A1 A2 A3 A4 C1 C2a C2b C3 C4 C5a C5b C6"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

json_get() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$1" "$2" 2>/dev/null || true; }
if [[ -f "$DATA_ROOT/paths.json" ]]; then
  : "${AUTOSHOT_ROOT:=$(json_get "$DATA_ROOT/paths.json" autoshot_root)}"
  : "${AUTOSHOT_GT:=$(json_get "$DATA_ROOT/paths.json" autoshot_gt_pickle)}"
  : "${CLIPSHOTS_ANN:=$(json_get "$DATA_ROOT/paths.json" clipshots_test_annotations)}"
fi
AUTOSHOT_ROOT="${AUTOSHOT_ROOT:-$DATA_ROOT/AutoShot}"
AUTOSHOT_GT="${AUTOSHOT_GT:-$AUTOSHOT_ROOT/gt_scenes_dict_baseline_v2.pickle}"
CLIPSHOTS_ANN="${CLIPSHOTS_ANN:-$DATA_ROOT/ClipShots/annotations/test.json}"
AUTOSHOT_TRAIN_CACHE="${AUTOSHOT_TRAIN_CACHE:-$DATA_ROOT/cache/autoshot_train}"
AUTOSHOT_TEST_CACHE="${AUTOSHOT_TEST_CACHE:-$DATA_ROOT/cache/autoshot_test}"
CLIPSHOTS_CACHE="${CLIPSHOTS_CACHE:-$DATA_ROOT/cache/clipshots}"
CLIPSHOTS_TEST_CACHE="${CLIPSHOTS_TEST_CACHE:-$CLIPSHOTS_CACHE/test}"
BBC_ROOT="${BBC_ROOT:-$DATA_ROOT/BBC}"   # filled by prepare_bbc_data.py (Zenodo 14865504) if absent
BBC_CACHE="${BBC_CACHE:-$DATA_ROOT/cache/bbc}"

# SHOT training policy (decided with the user): kuaishou_v2.txt lists 648 videos, which includes the 200 SHOT test videos
# (the keys of gt_scenes_dict_baseline_v2.pickle = the authors' test annotation; verified). Default 1 = the test set is
# defined by that pickle and those 200 videos are EXCLUDED from training (448 videos), so SHOT test results are clean.
# 0 = train on all 648 as the original spec says (the known overlap is then reported by the audit, not fatal).
AUTOSHOT_EXCLUDE_TEST="${AUTOSHOT_EXCLUDE_TEST:-1}"
EXCLUDE_FLAG=""; [[ "$AUTOSHOT_EXCLUDE_TEST" == "1" ]] && EXCLUDE_FLAG="--autoshot_exclude_test"
ALLOW_PAIR_FLAG=""; [[ "$AUTOSHOT_EXCLUDE_TEST" != "1" ]] && ALLOW_PAIR_FLAG="--allow_pair autoshot_train autoshot_test"
# Single-thread math per process: containers are CPU-quota limited and decoding/loading is process-parallel.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# ---- machine-aware defaults (container CPU quota and RAM, not the host's nproc) ----
CORES=$(nproc --all)      # NOT plain nproc: it honours OMP_NUM_THREADS (set to 1 above)
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r _q _p < /sys/fs/cgroup/cpu.max
  if [[ "$_q" != "max" ]]; then _c=$(( _q / _p )); (( _c < CORES )) && CORES=$_c; fi
fi
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
_w=$(( CORES - 16 )); _r=$(( RAM_GB - 24 ))            # leave 16 cores / 24 GB free; ~1 GB RAM per decode worker
(( _r < _w )) && _w=$_r; (( _w > 96 )) && _w=96; (( _w < 4 )) && _w=4
NUM_WORKERS="${NUM_WORKERS:-$_w}"                       # data preparation (video decoding) workers
TRAIN_WORKERS="${TRAIN_WORKERS:-12}"                    # DataLoader workers per training job
export EVAL_WORKERS="${EVAL_WORKERS:-16}"               # DataLoader workers per evaluation job
# One training run already keeps the GPU 100% busy (measured: step time scales linearly with concurrent runs), so
# concurrency does not shorten total training time; 2 just overlaps data stalls/evaluation and yields A0 early.
PARALLEL="${PARALLEL:-2}"
echo "machine: ${CORES} usable cores, ${RAM_GB} GB RAM -> NUM_WORKERS=$NUM_WORKERS TRAIN_WORKERS=$TRAIN_WORKERS EVAL_WORKERS=$EVAL_WORKERS PARALLEL=$PARALLEL" >&2

# Videos that may be absent from a cache because the source file itself is unreadable (approved by the user).
VERIFY_ALLOW_MISSING="${VERIFY_ALLOW_MISSING:-clipshots_train:5186782609.mp4}"
VERIFY_FLAGS=""; [[ "$AUTOSHOT_EXCLUDE_TEST" == "1" ]] && VERIFY_FLAGS="--exclude_test"
for _m in $VERIFY_ALLOW_MISSING; do VERIFY_FLAGS="$VERIFY_FLAGS --allow_missing $_m"; done
