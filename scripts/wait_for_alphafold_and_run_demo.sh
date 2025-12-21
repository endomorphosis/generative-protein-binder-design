#!/usr/bin/env bash
set -euo pipefail

# Wait for AlphaFold reduced DB downloads to complete, then run an end-to-end demo job.
# Logs:
#   outputs/alphafold-wait-and-demo.log
#
# This script is safe to run multiple times.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_LOG="$ROOT_DIR/outputs/alphafold-wait-and-demo.log"

DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
PID_FILE="/tmp/alphafold_download_reduced.pid"
DL_LOG="/tmp/alphafold_download_reduced.log"

# Robustness knobs (override via env)
STALL_MINUTES="${ALPHAFOLD_DOWNLOAD_STALL_MINUTES:-20}"
MIN_FREE_GB="${ALPHAFOLD_MIN_FREE_GB:-150}"

mkdir -p "$ROOT_DIR/outputs"

echo "[$(date -Is)] Starting watcher" | tee -a "$OUT_LOG"
echo "[$(date -Is)] DATA_DIR=$DATA_DIR" | tee -a "$OUT_LOG"
echo "[$(date -Is)] PID_FILE=$PID_FILE" | tee -a "$OUT_LOG"
echo "[$(date -Is)] DL_LOG=$DL_LOG" | tee -a "$OUT_LOG"

required_paths=(
  "$DATA_DIR/params/params_model_1.npz"
  "$DATA_DIR/uniref90/uniref90.fasta"
  "$DATA_DIR/mgnify/mgy_clusters_2022_05.fa"
  "$DATA_DIR/uniref30/UniRef30_2021_03_hhm.ffdata"
  "$DATA_DIR/uniref30/UniRef30_2021_03_hhm.ffindex"
  "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta"
  "$DATA_DIR/pdb_mmcif/mmcif_files"
  "$DATA_DIR/pdb_mmcif/obsolete.dat"
  "$DATA_DIR/pdb70/pdb70_hhm.ffdata"
  "$DATA_DIR/pdb70/pdb70_hhm.ffindex"
  "$DATA_DIR/uniprot/uniprot.fasta"
  "$DATA_DIR/pdb_seqres/pdb_seqres.txt"
)

all_present() {
  local p
  for p in "${required_paths[@]}"; do
    if [[ ! -e "$p" ]]; then
      return 1
    fi
  done
  return 0
}

print_missing() {
  local p
  for p in "${required_paths[@]}"; do
    if [[ -e "$p" ]]; then
      echo "  ✓ $p"
    else
      echo "  ✗ $p"
    fi
  done
}

start_downloader_if_needed() {
  # Start (or restart) the downloader in the background; it writes/updates PID_FILE.
  echo "[$(date -Is)] Starting/restarting reduced DB downloader..." | tee -a "$OUT_LOG"
  nohup bash "$ROOT_DIR/scripts/start_alphafold_reduced_download.sh" >/dev/null 2>&1 &
  echo "[$(date -Is)] Downloader spawn pid=$!" | tee -a "$OUT_LOG"
}

free_gb() {
  # Prints free space (GiB) for the filesystem containing DATA_DIR.
  # Uses df -BG, which yields integer GiB (e.g. 586G).
  local free
  free="$(df -BG "$DATA_DIR" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')"
  if [[ "$free" =~ ^[0-9]+$ ]]; then
    echo "$free"
  else
    echo "0"
  fi
}

download_log_stale() {
  # Returns 0 if the download log exists and hasn't been updated recently.
  [[ -f "$DL_LOG" ]] || return 1
  local now_ts last_ts delta
  now_ts="$(date +%s)"
  last_ts="$(stat -c %Y "$DL_LOG" 2>/dev/null || echo 0)"
  delta=$(( now_ts - last_ts ))
  (( delta > STALL_MINUTES * 60 ))
}

gzip_extract() {
  # gzip_extract INPUT.gz OUTPUT
  local in_gz="$1"
  local out_file="$2"
  local tmp
  tmp="${out_file}.tmp"

  if [[ ! -f "$in_gz" ]]; then
    return 1
  fi

  # If aria2 is still downloading, it leaves a .aria2 control file.
  if [[ -f "${in_gz}.aria2" ]]; then
    return 1
  fi

  # Validate archive before extracting.
  if ! gzip -t "$in_gz" >/dev/null 2>&1; then
    echo "[$(date -Is)] WARNING: gzip integrity check failed: $in_gz" | tee -a "$OUT_LOG"
    return 1
  fi

  echo "[$(date -Is)] Extracting: $in_gz -> $out_file" | tee -a "$OUT_LOG"
  rm -f "$tmp" 2>/dev/null || true
  if command -v pigz >/dev/null 2>&1; then
    pigz -dc "$in_gz" >"$tmp"
  else
    gzip -dc "$in_gz" >"$tmp"
  fi
  mv -f "$tmp" "$out_file"
  return 0
}

attempt_post_download_repairs() {
  # If downloads are complete but the official scripts failed before producing
  # the final expected files, try safe repairs.

  # UniRef90: expected output is uniref90/uniref90.fasta.
  if [[ ! -f "$DATA_DIR/uniref90/uniref90.fasta" && -f "$DATA_DIR/uniref90/uniref90.fasta.gz" ]]; then
    gzip_extract "$DATA_DIR/uniref90/uniref90.fasta.gz" "$DATA_DIR/uniref90/uniref90.fasta" || true
  fi

  # UniProt: expected output is uniprot/uniprot.fasta (concatenated sprot+trembl).
  if [[ ! -f "$DATA_DIR/uniprot/uniprot.fasta" ]]; then
    local trembl_gz="$DATA_DIR/uniprot/uniprot_trembl.fasta.gz"
    local sprot_gz="$DATA_DIR/uniprot/uniprot_sprot.fasta.gz"
    local trembl_fa="$DATA_DIR/uniprot/uniprot_trembl.fasta"
    local sprot_fa="$DATA_DIR/uniprot/uniprot_sprot.fasta"
    local out="$DATA_DIR/uniprot/uniprot.fasta"
    local tmp_out="${out}.tmp"

    # Only attempt if both downloads look complete.
    if [[ -f "$trembl_gz" && -f "$sprot_gz" && ! -f "${trembl_gz}.aria2" && ! -f "${sprot_gz}.aria2" ]]; then
      # Extract if needed.
      if [[ ! -f "$trembl_fa" ]]; then
        gzip_extract "$trembl_gz" "$trembl_fa" || true
      fi
      if [[ ! -f "$sprot_fa" ]]; then
        gzip_extract "$sprot_gz" "$sprot_fa" || true
      fi

      if [[ -f "$trembl_fa" && -f "$sprot_fa" ]]; then
        echo "[$(date -Is)] Building: $out (SwissProt + TrEMBL)" | tee -a "$OUT_LOG"
        rm -f "$tmp_out" 2>/dev/null || true
        cat "$sprot_fa" "$trembl_fa" >"$tmp_out"
        mv -f "$tmp_out" "$out"
        # Clean up intermediate unzipped files to save space.
        rm -f "$sprot_fa" "$trembl_fa" 2>/dev/null || true
      fi
    fi
  fi

  # PDB SeqRes: expected output is pdb_seqres/pdb_seqres.txt filtered to proteins.
  local seqres_txt="$DATA_DIR/pdb_seqres/pdb_seqres.txt"
  local seqres_tmp="$DATA_DIR/pdb_seqres/pdb_seqres_filtered.tmp"
  if [[ -f "$seqres_txt" && ! -f "${seqres_txt}.aria2" ]]; then
    # Keep only protein sequences. Safe to run multiple times.
    if grep --after-context=1 --no-group-separator '>.* mol:protein' "$seqres_txt" >"$seqres_tmp" 2>/dev/null; then
      mv -f "$seqres_tmp" "$seqres_txt"
      echo "[$(date -Is)] Re-filtered PDB SeqRes protein-only sequences: $seqres_txt" | tee -a "$OUT_LOG"
    else
      rm -f "$seqres_tmp" 2>/dev/null || true
    fi
  fi
}

# Wait loop
while ! all_present; do
  echo "[$(date -Is)] Waiting for AlphaFold DBs..." | tee -a "$OUT_LOG"
  print_missing | tee -a "$OUT_LOG"

  # If the downloads have completed but the final files are missing, attempt repairs.
  attempt_post_download_repairs || true

  # Disk space guard: downloading/extracting UniRef/UniProt needs lots of headroom.
  free="$(free_gb)"
  if (( free < MIN_FREE_GB )); then
    echo "[$(date -Is)] WARNING: low disk space: ${free}GiB free under DATA_DIR filesystem (min recommended: ${MIN_FREE_GB}GiB)." | tee -a "$OUT_LOG"
    echo "[$(date -Is)] WARNING: Pausing downloads to avoid partial/corrupt datasets. Free space and rerun this watcher." | tee -a "$OUT_LOG"
    sleep 300
    continue
  fi

  pid=""
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  fi
  if [[ -z "$pid" ]] || ! ps -p "$pid" >/dev/null 2>&1; then
    echo "[$(date -Is)] Downloader not running (pid='${pid:-}')." | tee -a "$OUT_LOG"
    echo "[$(date -Is)] Last download log lines:" | tee -a "$OUT_LOG"
    tail -n 60 "$DL_LOG" 2>/dev/null | tee -a "$OUT_LOG" || true
    start_downloader_if_needed
  else
    # If downloader is supposedly running but the download log isn't changing, it likely stalled.
    if download_log_stale; then
      echo "[$(date -Is)] WARNING: download log has not updated for >${STALL_MINUTES} minutes; restarting downloader pid=$pid" | tee -a "$OUT_LOG"
      kill "$pid" 2>/dev/null || true
      sleep 2
      start_downloader_if_needed
    fi
  fi

  if [[ -f "$DL_LOG" ]]; then
    echo "[$(date -Is)] Last download log lines:" | tee -a "$OUT_LOG"
    tail -n 30 "$DL_LOG" 2>/dev/null | tee -a "$OUT_LOG" || true
  fi

  sleep 60
done

echo "[$(date -Is)] Required AlphaFold DB files appear present." | tee -a "$OUT_LOG"

# Keep the AlphaFold2 submodule clean (python imports often create __pycache__).
"$ROOT_DIR/scripts/clean_alphafold2_submodule_artifacts.sh" >>"$OUT_LOG" 2>&1 || true

# Ensure OpenMM is installed in the alphafold2 environment.
echo "[$(date -Is)] Ensuring OpenMM is installed in conda env alphafold2..." | tee -a "$OUT_LOG"
set +e
(
  eval "$(conda shell.bash hook)" && conda activate alphafold2 && \
  python -c 'import openmm, openmm.app; print("openmm ok")'
) >>"$OUT_LOG" 2>&1
rc=$?
set -e
if [[ "$rc" != "0" ]]; then
  (
    eval "$(conda shell.bash hook)" && conda activate alphafold2 && \
    mamba install -y -q -c conda-forge openmm && \
    python -c 'import openmm, openmm.app; print("openmm ok")'
  ) >>"$OUT_LOG" 2>&1
fi

# Restart host-native services so they pick up current env/tools.
echo "[$(date -Is)] Restarting host-native model services..." | tee -a "$OUT_LOG"
set +e
# Kill anything on the ports.
if command -v ss >/dev/null 2>&1; then
  for port in 18081 18082 18084; do
    pids=$(ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p {print $0}' | sed -n 's/.*pid=\([0-9][0-9]*\),.*/\1/p' | sort -u)
    for pid in $pids; do
      kill "$pid" 2>/dev/null || true
    done
  done
fi
set -e

nohup bash "$ROOT_DIR/scripts/run_arm64_native_model_services.sh" >>"$OUT_LOG" 2>&1 &
echo "[$(date -Is)] Started services pid=$!" | tee -a "$OUT_LOG"

# Wait for readiness.
for i in {1..60}; do
  if curl -fsS http://localhost:18081/v1/health/ready >/dev/null 2>&1 \
    && curl -fsS http://localhost:18082/v1/health/ready >/dev/null 2>&1 \
    && curl -fsS http://localhost:18084/v1/health/ready >/dev/null 2>&1; then
    echo "[$(date -Is)] Host-native services are ready." | tee -a "$OUT_LOG"
    break
  fi
  sleep 2
done

# Submit demo job.
echo "[$(date -Is)] Submitting demo job..." | tee -a "$OUT_LOG"
resp="$($ROOT_DIR/scripts/submit_demo_job.sh 2>&1)"
echo "$resp" | tee -a "$OUT_LOG"

job_id=""
if command -v jq >/dev/null 2>&1; then
  job_id=$(echo "$resp" | sed -n '/^{/,$p' | jq -r '.job_id' 2>/dev/null || true)
else
  job_id=$(python3 - <<'PY'
import json,sys
text=sys.stdin.read()
# find first JSON object in output
start=text.find('{')
if start==-1:
  print('')
  raise SystemExit(0)
obj=json.loads(text[start:])
print(obj.get('job_id',''))
PY
<<<"$resp" 2>/dev/null || true)
fi

if [[ -z "$job_id" || "$job_id" == "null" ]]; then
  echo "[$(date -Is)] Could not parse job_id from demo response" | tee -a "$OUT_LOG"
  exit 3
fi

echo "[$(date -Is)] job_id=$job_id" | tee -a "$OUT_LOG"

# Poll job status.
server_url="http://localhost:8011"
if [[ -n "${MCP_SERVER_URL:-}" ]]; then
  server_url="$MCP_SERVER_URL"
fi

echo "[$(date -Is)] Polling job status at $server_url/api/jobs/$job_id" | tee -a "$OUT_LOG"

timeout_seconds=$((2 * 60 * 60))
start_ts=$(date +%s)
while true; do
  now_ts=$(date +%s)
  if (( now_ts - start_ts > timeout_seconds )); then
    echo "[$(date -Is)] Timeout waiting for job completion" | tee -a "$OUT_LOG"
    exit 4
  fi

  status_json=$(curl -sS "$server_url/api/jobs/$job_id" || true)
  echo "[$(date -Is)] $status_json" | tee -a "$OUT_LOG"

  # Try to detect completion.
  if echo "$status_json" | grep -q '"status"\s*:\s*"completed"'; then
    echo "[$(date -Is)] Job completed successfully" | tee -a "$OUT_LOG"
    exit 0
  fi
  if echo "$status_json" | grep -q '"status"\s*:\s*"failed"'; then
    echo "[$(date -Is)] Job failed" | tee -a "$OUT_LOG"
    exit 5
  fi

  sleep 10
done
