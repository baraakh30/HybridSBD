#!/usr/bin/env bash
# Speed benchmark of every ablation checkpoint (A0..C6 from ablations_v2 + the build-up runs from ablations_v3) and AutoShot,
# on an otherwise IDLE GPU (run only after training has finished). FP32, 10 warm-up + 80 timed, synthetic input, batch 1/16/32.
#   bash run_speed_benchmark.sh      -> /workspace/outputs/speed_benchmark/{ours.json,autoshot.json,*.log}
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/venv/main/bin:$PATH
OUT=/workspace/outputs/speed_benchmark; mkdir -p "$OUT"
if [ "$(ps aux | grep -c '[t]rain_ablation.py')" != "0" ]; then echo "training still running; refusing to benchmark" >&2; exit 1; fi
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv | tee "$OUT/gpu_state.txt"
CK=""
for r in A0 A1 A2 A3 A4 C1 C2a C2b C3 C4 C5a C5b C6; do CK="$CK /workspace/outputs/ablations_v2/$r/best_model.pt"; done
for r in M0_min M1_edge B_noatt S1_rel; do CK="$CK /workspace/outputs/ablations_v3/$r/best_model.pt"; done
python benchmark_model.py --device cuda --batch_sizes 1 16 32 --checkpoints $CK --output "$OUT/ours.json"
python benchmark_baselines.py --device cuda --batch_sizes 1 16 32 --output "$OUT/autoshot.json"
CUDA_VISIBLE_DEVICES= python speed_summary.py --dir "$OUT"
