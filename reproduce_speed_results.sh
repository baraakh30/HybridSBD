#!/usr/bin/env bash
# Re-measure the speed numbers in RESULTS.md section 8 (hybrid_detector13/15/16/17/18/19 + AutoShot, batch 8/16/32, FP32, 128x48x64).
# The original checkpoints are not available, but latency/VRAM/params depend only on the architecture, so each model is built with the
# same (main_dim, num_fusion_blocks) and random weights. Same protocol as before: 10 warm-up + 80 timed runs, synthetic input.
#   bash reproduce_speed_results.sh   -> /workspace/outputs/speed_repro/
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/venv/main/bin:$PATH
OUT=/workspace/outputs/speed_repro; mkdir -p "$OUT"
if [ "$(ps aux | grep -c '[t]rain_ablation.py')" != "0" ]; then echo "training running; refusing" >&2; exit 1; fi
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv | tee "$OUT/gpu_state.txt"
python - <<'PY'
import torch
from detection.hybrid_boundary_detector import create_hybrid_detector
CFG = {13: (512, 3), 15: (256, 3), 16: (384, 4), 17: (256, 4), 18: (512, 4), 19: (384, 3)}
for n, (d, b) in CFG.items():
    torch.manual_seed(0)
    m = create_hybrid_detector(main_dim=d, num_fusion_blocks=b)
    p = f"/workspace/outputs/speed_repro/hybrid_detector{n}"
    import os; os.makedirs(p, exist_ok=True)
    torch.save({"model_state_dict": m.state_dict(), "config": {"main_dim": d, "num_fusion_blocks": b}, "epoch": "random-init"}, f"{p}/best_model.pt")
PY
python benchmark_model.py --device cuda --batch_sizes 8 16 32 --output "$OUT/ours_old_arch.json" \
  --checkpoints $(for n in 13 15 16 17 18 19; do echo $OUT/hybrid_detector$n/best_model.pt; done)
python benchmark_baselines.py --device cuda --batch_sizes 8 16 32 --output "$OUT/baselines.json" || true
echo "repro done $(date -Iseconds)"
