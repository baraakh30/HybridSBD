#!/usr/bin/env bash
# Pull results from the GPU server into server_results/ (small result files only unless stated otherwise).
#
#   REMOTE=root@<host> REMOTE_PORT=<port> SSH_KEY=~/.ssh/<key> bash scripts/sync.sh results
#   REMOTE=... bash scripts/sync.sh epochs        # newest N (default 5) per-epoch checkpoints of every run
#
# Environment: REMOTE (required, user@host), REMOTE_PORT (default 22), SSH_KEY (optional),
#              REMOTE_OUTPUTS (default /workspace/outputs), N (epochs to pull, default 5).
set -euo pipefail
: "${REMOTE:?set REMOTE=user@host}"
REMOTE_OUTPUTS="${REMOTE_OUTPUTS:-/workspace/outputs}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/server_results"
SSH="ssh -o IdentitiesOnly=yes -o BatchMode=yes -p ${REMOTE_PORT:-22}${SSH_KEY:+ -i $SSH_KEY}"
quiet() { grep -v "^Welcome\|^Have fun\|^AI agents" || true; }

pull() {  # pull <remote dir> <local dir>: result files without per-epoch checkpoints and archived runs
  mkdir -p "$2"
  rsync -aL --partial -e "$SSH" --exclude 'epoch_*.pt' --exclude '*.archived_*' "$REMOTE:$REMOTE_OUTPUTS/$1/" "$2/" 2>&1 | quiet
}

sync_results() {
  pull ablations "$DEST"
  pull control_leak648 "$DEST/control_leak648"
  for E in exp_synth exp_synth_ema exp_synth50_ema exp_synth_ema_shot2x exp_synth_ema_s43 ablations_v2 ablations_v3 exp_A4_s43 seed_runs protocolB reeval_lmin1 speed_benchmark; do
    pull "$E" "$DEST/$E"
  done
  echo "$(date -u +%FT%TZ) synced $(du -sh "$DEST" | cut -f1)" >> "$DEST/sync.log"
}

sync_epochs() {
  local n="${N:-5}" list
  list=$($SSH "$REMOTE" "cd $REMOTE_OUTPUTS && for d in ablations_v2/*/ ablations_v3/*/ exp_A4_s43/*/; do [ -d \$d ] && ls -t \${d}epoch_*.pt 2>/dev/null | head -$n; done" 2>/dev/null | grep '\.pt$' || true)
  [ -z "$list" ] && { echo "no epoch files listed"; exit 1; }
  echo "$list" | rsync -a --partial --files-from=- -e "$SSH" "$REMOTE:$REMOTE_OUTPUTS/" "$DEST/" 2>&1 | quiet
  echo "$(date -u +%FT%TZ) epochs synced: $(find "$DEST" -name 'epoch_*.pt' | wc -l) files" >> "$DEST/sync.log"
}

case "${1:-}" in
  results) sync_results ;;
  epochs) sync_epochs ;;
  *) echo "usage: bash scripts/sync.sh {results|epochs}" >&2; exit 2 ;;
esac
