#!/bin/bash
# Pull the newest N (default 5) per-epoch checkpoints of every run in ablations_v2 / ablations_v3 (+ exp_A4_s43) to server_results/.
N="${N:-5}"
D="$(cd "$(dirname "$0")" && pwd)/server_results"
SSH="ssh -i $HOME/.ssh/id_ed25519_sbd -o IdentitiesOnly=yes -o BatchMode=yes -p 61020"
LIST=$($SSH root@136.41.67.173 "cd /workspace/outputs && for d in ablations_v2/*/ ablations_v3/*/ exp_A4_s43/*/; do [ -d \$d ] && ls -t \${d}epoch_*.pt 2>/dev/null | head -$N; done" 2>/dev/null | grep '\.pt$')
[ -z "$LIST" ] && { echo "no epoch files listed"; exit 1; }
echo "$LIST" | rsync -a --partial --files-from=- -e "$SSH" root@136.41.67.173:/workspace/outputs/ "$D/" 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
echo "$(date -u +%FT%TZ) epochs synced: $(find "$D" -name 'epoch_*.pt' | wc -l) files" >> "$D/sync.log"
