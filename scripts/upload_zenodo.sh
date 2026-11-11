#!/usr/bin/env bash
# Upload a folder of checkpoints to a NEW, PRIVATE Zenodo draft. Nothing is published: review the draft on zenodo.org, add
# the authors, description and licence there, then press "Publish" yourself (a published record cannot be removed).
#
#   ZENODO_TOKEN=<personal access token, scopes deposit:write deposit:actions> \
#   bash scripts/upload_zenodo.sh <folder with the .pt files> ["record title"]
#
# ZENODO_HOST=sandbox.zenodo.org uses Zenodo's test site (no real DOI). The token is read from the environment only.
set -euo pipefail

DIR="${1:?usage: upload_zenodo.sh <folder> [title]}"
TITLE="${2:-HybridSBD: trained checkpoints for the paper tables}"
HOST="${ZENODO_HOST:-zenodo.org}"
: "${ZENODO_TOKEN:?set ZENODO_TOKEN}"
AUTH=(-H "Authorization: Bearer ${ZENODO_TOKEN}")

echo "Creating draft on ${HOST} ..."
RESP=$(curl -sS --fail-with-body "${AUTH[@]}" -H "Content-Type: application/json" -X POST -d '{}' "https://${HOST}/api/deposit/depositions")
ID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"$RESP")
BUCKET=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["links"]["bucket"])' <<<"$RESP")
echo "Draft id: ${ID}"

for f in "$DIR"/*.pt; do
  name=$(basename "$f")
  size=$(du -h "$f" | cut -f1)
  echo "Uploading ${name} [${size}] ..."
  curl -sS --fail-with-body "${AUTH[@]}" --upload-file "$f" "${BUCKET}/${name}" >/dev/null
done

# Title only (best effort); authors, description and licence are added on the website before publishing.
curl -sS "${AUTH[@]}" -H "Content-Type: application/json" -X PUT \
  -d "{\"metadata\": {\"title\": \"${TITLE}\", \"upload_type\": \"other\"}}" \
  "https://${HOST}/api/deposit/depositions/${ID}" >/dev/null || true

echo "Done. Private draft (not published): https://${HOST}/deposit/${ID}"
