#!/usr/bin/env bash
set -euo pipefail

# Submit a simple demo job to the MCP server.
# This is intended for non-technical users to validate end-to-end wiring.
#
# Usage:
#   ./scripts/submit_demo_job.sh
#   ./scripts/submit_demo_job.sh "MKT..."
#
# Env:
#   MCP_SERVER_URL (default: http://localhost:8011)

MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:${MCP_SERVER_HOST_PORT:-8011}}"

SEQUENCE="${1:-}"
if [[ -z "$SEQUENCE" ]]; then
  # Short, valid-ish AA sequence for a quick smoke test.
  SEQUENCE="MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
fi

JOB_NAME="${JOB_NAME:-Demo Job}"
NUM_DESIGNS="${NUM_DESIGNS:-2}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl not found; please install curl" >&2
  exit 1
fi

echo "Submitting demo job to: $MCP_SERVER_URL"
RESP="$(curl -fsS "$MCP_SERVER_URL/api/jobs" \
  -H 'Content-Type: application/json' \
  -d "{\"sequence\":\"$SEQUENCE\",\"job_name\":\"$JOB_NAME\",\"num_designs\":$NUM_DESIGNS}")"

echo
if command -v jq >/dev/null 2>&1; then
  echo "$RESP" | jq
  JOB_ID="$(echo "$RESP" | jq -r '.job_id')"
else
  echo "$RESP"
  JOB_ID="$(python3 - <<PY
import json,sys
print(json.loads(sys.stdin.read()).get('job_id',''))
PY
<<<"$RESP" 2>/dev/null || true)"
fi

echo
if [[ -n "$JOB_ID" ]]; then
  echo "Created job_id: $JOB_ID"
  echo "Dashboard: http://localhost:${MCP_DASHBOARD_HOST_PORT:-3000}"
  echo "API status: $MCP_SERVER_URL/api/jobs/$JOB_ID"
else
  echo "Job created, but could not parse job_id (install jq for nicer output)."
fi
