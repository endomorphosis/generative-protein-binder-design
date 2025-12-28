#!/usr/bin/env bash
set -euo pipefail

# Run AlphaFold with profiling: GPU (nvidia-smi), CPU/mem (pidstat), disk (iostat)
# Usage:
#   scripts/run_profiled_inference.sh <fasta_path> <msa_mode> [output_dir]
#   msa_mode: jackhmmer | mmseqs2

FASTA_PATH=${1:-}
MSA_MODE=${2:-}
OUT_DIR=${3:-"/tmp/af_profile_$(date +%Y%m%d_%H%M%S)"}

if [[ -z "$FASTA_PATH" || -z "$MSA_MODE" ]]; then
  echo "Usage: $0 <fasta_path> <msa_mode: jackhmmer|mmseqs2> [output_dir]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load AlphaFold env variables if present
if [[ -f "$ROOT_DIR/tools/generated/alphafold2/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/tools/generated/alphafold2/.env"
fi

ALPHAFOLD_DIR="${ALPHAFOLD_DIR:-$ROOT_DIR/tools/alphafold2}"
DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
MMSEQS2_DB_PATH="${ALPHAFOLD_MMSEQS2_DATABASE_PATH:-}"

# Speed knobs (override via env vars)
CPU_THREADS_DEFAULT=$(nproc)
if [[ "$CPU_THREADS_DEFAULT" -gt 32 ]]; then CPU_THREADS_DEFAULT=32; fi
AF_CPU_THREADS="${AF_CPU_THREADS:-$CPU_THREADS_DEFAULT}"
AF_MMSEQS2_MAX_SEQS="${AF_MMSEQS2_MAX_SEQS:-512}"
AF_DISABLE_TEMPLATES="${AF_DISABLE_TEMPLATES:-0}"
AF_NUM_RECYCLES="${AF_NUM_RECYCLES:--1}"
AF_NUM_ENSEMBLE="${AF_NUM_ENSEMBLE:--1}"

# Persistent MSA cache (copy MSAs into the run output dir and enable --use_precomputed_msas)
MSA_CACHE_DIR="${ALPHAFOLD_MSA_CACHE_DIR:-$DATA_DIR/msa_cache}"

mkdir -p "$OUT_DIR"

log() { echo "[PROFILE] $*"; }

# System snapshot
log "System snapshot → $OUT_DIR/system_info.txt"
{
  echo "Date: $(date)"
  echo "Host: $(hostname)"
  echo "CPU cores: $(nproc)"
  echo "RAM: $(free -h | awk '/^Mem:/{print $2}')"
  echo "GPU(s):"; nvidia-smi -L || true
  echo "GPU driver:"; nvidia-smi || true
  echo "Conda env: ${ALPHAFOLD_CONDA_ENV:-unknown}"
} > "$OUT_DIR/system_info.txt"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "${ALPHAFOLD_CONDA_ENV:-alphafold2}"

# Avoid CPU thread oversubscription during feature generation / HHsearch / mmseqs2 wrappers.
export OMP_NUM_THREADS="$AF_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$AF_CPU_THREADS"
export MKL_NUM_THREADS="$AF_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$AF_CPU_THREADS"
export TF_NUM_INTRAOP_THREADS="$AF_CPU_THREADS"
export TF_NUM_INTEROP_THREADS=1

# Prepare monitoring
GPU_CSV="$OUT_DIR/gpu_util.csv"
GPU_DMON="$OUT_DIR/gpu_dmon.csv"
PIDSTAT_LOG="$OUT_DIR/pidstat.csv"
IOSTAT_LOG="$OUT_DIR/iostat.csv"
AF_LOG="$OUT_DIR/run.log"

log "Starting GPU and system monitors"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,memory.used,memory.total --format=csv,noheader,nounits -l 1 > "$GPU_CSV" 2>/dev/null &
GPU_MON_PID=$!

nvidia-smi dmon -s pucmet -d 1 > "$GPU_DMON" 2>/dev/null &
GPU_DMON_PID=$!

# pidstat across system; filter later by PID if needed
pidstat -h -r -u -d 1 > "$PIDSTAT_LOG" 2>/dev/null &
PIDSTAT_PID=$!

iostat -y -x 1 > "$IOSTAT_LOG" 2>/dev/null &
IOSTAT_PID=$!

# Trap cleanup on exit/interrupt
cleanup_monitors() {
  log "Cleaning up monitors..."
  kill $GPU_MON_PID $GPU_DMON_PID $PIDSTAT_PID $IOSTAT_PID 2>/dev/null || true
  wait $GPU_MON_PID $GPU_DMON_PID $PIDSTAT_PID $IOSTAT_PID 2>/dev/null || true
}
trap cleanup_monitors EXIT INT TERM

# Build AlphaFold command
THREADS="$AF_CPU_THREADS"
AF_CMD=(
  python "$ALPHAFOLD_DIR/run_alphafold.py"
  --fasta_paths="$FASTA_PATH"
  --max_template_date=2022-12-31
  --model_preset=monomer
  --db_preset=reduced_dbs
  --data_dir="$DATA_DIR"
  --output_dir="$OUT_DIR"
  --use_gpu_relax=false
  --models_to_relax=none
  --logtostderr
  --jackhmmer_n_cpu="$THREADS"
  --hhsearch_n_cpu="$THREADS"
  --hmmsearch_n_cpu="$THREADS"
  --uniref90_database_path="$DATA_DIR/uniref90/uniref90.fasta"
  --mgnify_database_path="$DATA_DIR/mgnify/mgy_clusters_2022_05.fa"
  --small_bfd_database_path="$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta"
  --pdb70_database_path="$DATA_DIR/pdb70/pdb70"
  --template_mmcif_dir="$DATA_DIR/pdb_mmcif/mmcif_files"
  --obsolete_pdbs_path="$DATA_DIR/pdb_mmcif/obsolete.dat"
)

# Optional speed-related model knobs
if [[ "$AF_DISABLE_TEMPLATES" == "1" ]]; then
  AF_CMD+=(--disable_templates)
fi
if [[ "$AF_NUM_RECYCLES" != "-1" ]]; then
  AF_CMD+=(--num_recycles="$AF_NUM_RECYCLES")
fi
if [[ "$AF_NUM_ENSEMBLE" != "-1" ]]; then
  AF_CMD+=(--num_ensemble="$AF_NUM_ENSEMBLE")
fi

# Optional MSA caching
FASTA_STEM="$(basename "$FASTA_PATH")"
FASTA_STEM="${FASTA_STEM%.*}"
FASTA_SHA="$(sha256sum "$FASTA_PATH" | awk '{print $1}')"
RUN_MSA_DIR="$OUT_DIR/$FASTA_STEM/msas"
mkdir -p "$RUN_MSA_DIR"

CACHE_TAG="${MSA_MODE}_mm${AF_MMSEQS2_MAX_SEQS}_tmpl${AF_DISABLE_TEMPLATES}"
CACHE_MSA_DIR="$MSA_CACHE_DIR/$CACHE_TAG/$FASTA_SHA"
USE_PRECOMPUTED_MSAS=0
if [[ -d "$CACHE_MSA_DIR" ]]; then
  log "Found cached MSAs: $CACHE_MSA_DIR"
  cp -a "$CACHE_MSA_DIR"/. "$RUN_MSA_DIR"/ 2>/dev/null || true
  USE_PRECOMPUTED_MSAS=1
fi

case "$MSA_MODE" in
  jackhmmer)
    AF_CMD+=(--msa_mode=jackhmmer)
    ;;
  mmseqs2)
    AF_CMD+=(--msa_mode=mmseqs2)
    # Ensure MMseqs2 DB path exists; auto-discover if missing
    if [[ -z "$MMSEQS2_DB_PATH" || ! -e "$MMSEQS2_DB_PATH" ]]; then
      CAND=$(find "$DATA_DIR/mmseqs2" -maxdepth 1 -type f -name "*_db" | grep -v "_h$" | head -1 || true)
      if [[ -n "$CAND" ]]; then
        MMSEQS2_DB_PATH="$CAND"
        log "Auto-selected MMseqs2 DB: $MMSEQS2_DB_PATH"
      else
        log "Warning: MMseqs2 DB not found under $DATA_DIR/mmseqs2; continuing without explicit path."
      fi
    fi
    if [[ -n "$MMSEQS2_DB_PATH" ]]; then
      AF_CMD+=(--mmseqs2_database_path="$MMSEQS2_DB_PATH" --mmseqs2_max_seqs="$AF_MMSEQS2_MAX_SEQS")
    fi
    ;;
  *)
    echo "Invalid msa_mode: $MSA_MODE" >&2
    exit 2
    ;;
esac

if [[ "$USE_PRECOMPUTED_MSAS" == "1" ]]; then
  AF_CMD+=(--use_precomputed_msas=true)
fi

log "Running AlphaFold (msa_mode=$MSA_MODE)"
START=$(date +%s)
set +e
"${AF_CMD[@]}" &> "$AF_LOG"
APP_RC=$?
set -e
END=$(date +%s)

TOTAL_SEC=$((END - START))
echo "$TOTAL_SEC" > "$OUT_DIR/total_seconds.txt"

# Monitors will be cleaned up by trap

# Quick summary
SUMMARY="$OUT_DIR/summary.txt"
{
  echo "=== AlphaFold Profiling Summary ==="
  echo "Date: $(date)"
  echo "FASTA: $FASTA_PATH"
  echo "MSA mode: $MSA_MODE"
  echo "Total time (s): $TOTAL_SEC"
  echo "Logs:"
  echo "  AlphaFold: $AF_LOG"
  echo "  GPU util (csv): $GPU_CSV"
  echo "  GPU dmon: $GPU_DMON"
  echo "  pidstat: $PIDSTAT_LOG"
  echo "  iostat: $IOSTAT_LOG"
  if [[ $APP_RC -ne 0 ]]; then
    echo "Result: FAILURE (rc=$APP_RC)"
  else
    echo "Result: SUCCESS"
  fi
} > "$SUMMARY"

log "Done. Summary at $SUMMARY"

# Update cache after success
if [[ $APP_RC -eq 0 ]]; then
  mkdir -p "$CACHE_MSA_DIR"
  cp -a "$RUN_MSA_DIR"/. "$CACHE_MSA_DIR"/ 2>/dev/null || true
  log "Updated MSA cache: $CACHE_MSA_DIR"
fi
