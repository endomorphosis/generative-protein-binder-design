#!/usr/bin/env bash
# Benchmark a single MMseqs2 lookup against the prepared DB.
# Measures wall-clock time for createdb -> search -> result2msa.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Config
CONDA_ENV="${ALPHAFOLD_CONDA_ENV:-alphafold2}"
DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
MMSEQS_DB_PREFIX="${MMSEQS_DB_PREFIX:-$DATA_DIR/mmseqs2/uniref90_db}"
THREADS="${MMSEQS_BENCH_THREADS:-16}"
MAX_SEQS="${MMSEQS_BENCH_MAX_SEQS:-512}"
SEQ="${MMSEQS_BENCH_SEQ:-MNIFEMLRIDKVEELLS}"
OUT_DIR="${MMSEQS_BENCH_OUT_DIR:-/tmp/mmseqs2_bench_$$}"
TMP_DIR="${MMSEQS_BENCH_TMP_DIR:-$DATA_DIR/mmseqs2/tmp_bench}"

mkdir -p "$OUT_DIR" "$TMP_DIR" 2>/dev/null || true

log_info "MMseqs2 Lookup Benchmark"
log_info "  Env:       $CONDA_ENV"
log_info "  Data dir:  $DATA_DIR"
log_info "  DB prefix: $MMSEQS_DB_PREFIX"
log_info "  Threads:   $THREADS"
log_info "  Max seqs:  $MAX_SEQS"
log_info "  Tmp dir:   $TMP_DIR"

# Check prerequisites
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

if ! command -v mmseqs >/dev/null 2>&1; then
  log_error "mmseqs not found on PATH"
  exit 1
fi

if [[ ! -f "${MMSEQS_DB_PREFIX}.dbtype" ]]; then
  log_error "MMseqs DB not found: ${MMSEQS_DB_PREFIX}.dbtype"
  exit 1
fi

# Prepare query FASTA
FASTA="$OUT_DIR/query.fasta"
cat > "$FASTA" <<EOF
>query
$SEQ
EOF

# Run benchmark
START=$(date +%s)

mmseqs createdb "$FASTA" "$OUT_DIR/querydb"
mmseqs search "$OUT_DIR/querydb" "$MMSEQS_DB_PREFIX" "$OUT_DIR/resultdb" "$TMP_DIR" --threads "$THREADS" --max-seqs "$MAX_SEQS"
# Try direct A3M; fall back to convert if needed
if mmseqs result2msa "$OUT_DIR/querydb" "$MMSEQS_DB_PREFIX" "$OUT_DIR/resultdb" "$OUT_DIR/out.a3m" 2>/dev/null; then
  :
else
  mmseqs convertalis "$OUT_DIR/querydb" "$MMSEQS_DB_PREFIX" "$OUT_DIR/resultdb" "$OUT_DIR/out.a3m" --format-output query,target,pairwise,taxid --threads "$THREADS"
fi

END=$(date +%s)
DUR=$((END-START))

# Analyze results
A3M="$OUT_DIR/out.a3m"
SEQS=0
if [[ -f "$A3M" ]]; then
  SEQS=$(grep -c '^>' "$A3M" || echo 0)
fi

log_success "Lookup completed in ${DUR}s; MSA sequences: ${SEQS}"

# Keep outputs for inspection
log_info "Outputs in: $OUT_DIR"
log_info "  - Query DB: $OUT_DIR/querydb"
log_info "  - Result DB: $OUT_DIR/resultdb"
log_info "  - A3M: $A3M"
