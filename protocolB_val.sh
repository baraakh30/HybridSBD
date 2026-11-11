#!/usr/bin/env bash
# Protocol B, step 1: evaluate each model on the 67 SHOT validation videos (full videos, same pipeline as the test sets)
# with the full threshold grid, l_min = 1. The best (tau_c, tau_g) per model are then applied unchanged to the test sets.
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/venv/main/bin:$PATH PYTHONUNBUFFERED=1 EVAL_WORKERS=8
VAL=/workspace/data/cache/autoshot_val_full; OUT=/workspace/outputs/protocolB/val
mkdir -p $OUT
declare -A ROOT; for r in A0 A1 A2 A3 A4 C1 C2a C2b C3 C4 C5a C5b C6; do ROOT[$r]=/workspace/outputs/ablations_v2; done
for r in M0_min M1_edge B_noatt S1_rel; do ROOT[$r]=/workspace/outputs/ablations_v3; done
for r in A0_s43 A0_s44 A1_s43 A1_s44; do ROOT[$r]=/workspace/outputs/seed_runs/${r}; done
MOD() { case $1 in *_s4[34]) echo ${1%_s*};; *) echo $1;; esac; }
for r in ${RUNS:-A0 A1 A2 A3 A4 C1 C2a C2b C3 C4 C5a C5b C6 M0_min M1_edge B_noatt S1_rel}; do
  mkdir -p $OUT/$r
  [ -f $OUT/$r/shot_val_results.json ] || python test_autoshot.py --checkpoint ${ROOT[$r]}/$(MOD $r)/best_model.pt --cache_root $VAL \
      --gt_path $VAL/val_gt_scenes.pickle --min_scene_lengths 1 --output $OUT/$r/shot_val_results.json > $OUT/$r/val.log 2>&1
  echo "VAL_DONE $r $(date -Iseconds)"
done
