#!/bin/bash
# Pull small result files (logs, json, npz, best_model.pt) from the GPU server. Skips per-epoch checkpoints.
D="$(cd "$(dirname "$0")" && pwd)/server_results"
rsync -aL --partial -e "ssh -i $HOME/.ssh/id_ed25519_sbd -o IdentitiesOnly=yes -o BatchMode=yes -p 61020" \
  --exclude 'epoch_*.pt' --exclude '*.archived_*' root@136.41.67.173:/workspace/outputs/ablations/ "$D/" 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
mkdir -p "$D/control_leak648"
rsync -aL --partial -e "ssh -i $HOME/.ssh/id_ed25519_sbd -o IdentitiesOnly=yes -o BatchMode=yes -p 61020" \
  --exclude 'epoch_*.pt' --exclude '*.archived_*' root@136.41.67.173:/workspace/outputs/control_leak648/ "$D/control_leak648/" 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
for E in exp_synth exp_synth_ema exp_synth50_ema exp_synth_ema_shot2x exp_synth_ema_s43 ablations_v2 ablations_v3 exp_A4_s43 seed_runs protocolB reeval_lmin1; do
  mkdir -p "$D/$E"
  rsync -aL --partial -e "ssh -i $HOME/.ssh/id_ed25519_sbd -o IdentitiesOnly=yes -o BatchMode=yes -p 61020" \
    --exclude 'epoch_*.pt' --exclude '*.archived_*' root@136.41.67.173:/workspace/outputs/$E/ "$D/$E/" 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
done
echo "$(date -u +%FT%TZ) synced $(du -sh "$D" | cut -f1)" >> "$D/sync.log"
mkdir -p "$D/speed_benchmark"
rsync -aL --partial -e "ssh -i $HOME/.ssh/id_ed25519_sbd -o IdentitiesOnly=yes -o BatchMode=yes -p 61020" root@136.41.67.173:/workspace/outputs/speed_benchmark/ "$D/speed_benchmark/" 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
