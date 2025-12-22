#!/usr/bin/env bash
set -euo pipefail

# Simple job monitor for the MCP server.
# Prints status/progress/updated_at repeatedly and warns if updated_at stops moving.
#
# Usage:
#   ./scripts/monitor_job.sh <job_id> [--interval 10] [--stale-seconds 120] [--url http://localhost:8011] [--metrics] [--residency]

JOB_ID="${1:-}"
if [[ -z "$JOB_ID" || "$JOB_ID" == "-h" || "$JOB_ID" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/monitor_job.sh <job_id> [--interval 10] [--stale-seconds 120] [--url http://localhost:8011] [--metrics] [--residency]

Notes:
- A job is considered "possibly hanging" if updated_at hasn't changed for --stale-seconds.
- This is a polling monitor; it does not require jq.
- Use --metrics to print MemAvailable/Cached from the AlphaFold host metrics (best-effort).
- Use --residency to also request DB page-cache residency sampling (slower).
EOF
  exit 2
fi
shift || true

INTERVAL_S=10
STALE_S=120
BASE_URL="http://localhost:${MCP_SERVER_HOST_PORT:-8011}"
INCLUDE_METRICS=0
INCLUDE_RESIDENCY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      INTERVAL_S="${2:?missing value}"
      shift 2
      ;;
    --stale-seconds)
      STALE_S="${2:?missing value}"
      shift 2
      ;;
    --url)
      BASE_URL="${2:?missing value}"
      shift 2
      ;;
    --metrics)
      INCLUDE_METRICS=1
      shift
      ;;
    --residency)
      INCLUDE_METRICS=1
      INCLUDE_RESIDENCY=1
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

last_updated_at=""
last_updated_epoch=0

python_parse='import json,sys
j=json.load(sys.stdin)
progress=j.get("progress") or {}

progress_pct=j.get("progress_pct", None)
progress_msg=j.get("progress_message", None)

def _g(obj, *keys, default=None):
  cur=obj
  for k in keys:
    if not isinstance(cur, dict) or k not in cur:
      return default
    cur=cur[k]
  return cur

metrics=j.get("metrics") or {}
alph=_g(metrics, "alphafold_host", "latest", default=None)

mem_total=_g(alph, "meminfo", "gib", "MemTotal", default=None)
mem_avail=_g(alph, "meminfo", "gib", "MemAvailable", default=None)
cached=_g(alph, "meminfo", "gib", "Cached", default=None)
db_tier=_g(alph, "alphafold", "db_tier", default=None)

res_list=_g(alph, "db_cache_residency", default=None)
avg_res=None
min_res=None
n_res=0
if isinstance(res_list, list):
  vals=[]
  for x in res_list:
    if isinstance(x, dict) and x.get("ok") is True:
      v=x.get("resident_pct")
      if isinstance(v, (int, float)):
        vals.append(float(v))
  n_res=len(vals)
  if vals:
    avg_res=round(sum(vals)/len(vals), 1)
    min_res=round(min(vals), 1)

cache_bits=[]
if mem_total is not None:
  cache_bits.append(f"MemTotalGiB={mem_total}")
if mem_avail is not None:
  cache_bits.append(f"MemAvailGiB={mem_avail}")
  try:
    if mem_total is not None:
      used = round(float(mem_total) - float(mem_avail), 3)
      cache_bits.append(f"UsedGiB={used}")
  except Exception:
    pass
if cached is not None:
  cache_bits.append(f"CachedGiB={cached}")
if db_tier is not None:
  cache_bits.append(f"DB={db_tier}")
if avg_res is not None:
  cache_bits.append(f"DBresAvgPct={avg_res}")
  cache_bits.append(f"DBresMinPct={min_res}")
  cache_bits.append(f"DBresN={n_res}")

cache_summary=" ".join(cache_bits)

print("\t".join([
  str(j.get("status","?")),
  str(j.get("updated_at","")),
  "pct="+str(progress_pct),
  "msg="+str(progress_msg),
  "alphafold="+str(progress.get("alphafold")),
  "rfdiffusion="+str(progress.get("rfdiffusion")),
  "proteinmpnn="+str(progress.get("proteinmpnn")),
  "multimer="+str(progress.get("alphafold_multimer")),
  cache_summary,
]))
'

while true; do
  now_ts="$(date -Is)"
  url="$BASE_URL/api/jobs/$JOB_ID?include_metrics=${INCLUDE_METRICS}&include_residency=${INCLUDE_RESIDENCY}"

  if ! body="$(curl -sS --max-time 3 "$url")"; then
    echo "$now_ts ERROR: failed to fetch $url" >&2
    sleep "$INTERVAL_S"
    continue
  fi

  # Parse in a robust way without shell-escaping JSON into argv.
  if ! line="$(printf '%s' "$body" | python -c "$python_parse" 2>/dev/null)"; then
    echo "$now_ts ERROR: invalid JSON from server (first 200 chars):" >&2
    echo "${body:0:200}" >&2
    sleep "$INTERVAL_S"
    continue
  fi

  status="${line%%$'\t'*}"
  rest="${line#*$'\t'}"
  updated_at="${rest%%$'\t'*}"

  if [[ -n "$updated_at" && "$updated_at" != "$last_updated_at" ]]; then
    last_updated_at="$updated_at"
    last_updated_epoch="$(date +%s)"
  fi

  # Show a concise one-line status.
  echo "$now_ts status=$status updated_at=$updated_at ${line#*$'\t'$updated_at$'\t'}"

  # Hang detection: updated_at hasn't changed for STALE_S.
  if [[ -n "$last_updated_at" ]]; then
    now_epoch="$(date +%s)"
    age=$(( now_epoch - last_updated_epoch ))
    if (( age >= STALE_S )); then
      echo "$now_ts WARN: updated_at has not changed for ${age}s (threshold ${STALE_S}s). Job may be hung." >&2
    fi
  fi

  # Stop automatically when job terminal state is reached.
  if [[ "$status" == "completed" || "$status" == "failed" ]]; then
    exit 0
  fi

  sleep "$INTERVAL_S"
done
