#!/usr/bin/env bash
set -euo pipefail

# End-to-end AlphaFold2 run using MMseqs2 MSA mode.
#
# This is intentionally a *manual/opt-in* e2e check because it requires:
# - AlphaFold2 conda env + model params
# - Reduced/full AlphaFold DBs
# - An MMseqs2 searchable DB prefix (built from reduced FASTA)
#
# Usage:
#   ./scripts/e2e_test_alphafold_mmseqs2.sh
#
# Optional env:
#   ALPHAFOLD_MMSEQS2_DATABASE_PATH  (defaults to $ALPHAFOLD_DATA_DIR/mmseqs2/uniref90_db)
#   ALPHAFOLD_MMSEQS2_BINARY_PATH    (defaults to `command -v mmseqs`)
#   ALPHAFOLD_MMSEQS2_MAX_SEQS       (default 512)
#   ALPHAFOLD_E2E_OUTDIR             (default: /tmp/alphafold_mmseqs2_e2e_<timestamp>)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

maybe_source_env_file() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ -z "$line" ]] && continue
      [[ "$line" == \#* ]] && continue
      if [[ "$line" == export\ * ]]; then
        line="${line#export }"
      fi
      [[ "$line" != *=* ]] && continue
      local key="${line%%=*}"
      local value="${line#*=}"
      key="${key%"${key##*[![:space:]]}"}"
      key="${key#"${key%%[![:space:]]*}"}"
      value="${value#"${value%%[![:space:]]*}"}"
      if [[ ${#value} -ge 2 ]]; then
        if [[ "$value" == \"*\" && "$value" == *\" ]]; then
          value="${value:1:${#value}-2}"
        elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
          value="${value:1:${#value}-2}"
        fi
      fi
      export "$key=$value"
    done < "$env_file"
  fi
}

# Load the AlphaFold native command template and data dir from the installer output.
if [[ -z "${ALPHAFOLD_NATIVE_CMD:-}" ]]; then
  maybe_source_env_file "$ROOT_DIR/tools/generated/alphafold2/.env"
  maybe_source_env_file "$ROOT_DIR/tools/alphafold2/.env"
fi

if [[ -z "${ALPHAFOLD_NATIVE_CMD:-}" ]]; then
  log_error "ALPHAFOLD_NATIVE_CMD is not set. Run ./scripts/install_alphafold2_complete.sh first."
  exit 2
fi

DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
MMSEQS_BIN="${ALPHAFOLD_MMSEQS2_BINARY_PATH:-}"
if [[ -z "$MMSEQS_BIN" ]]; then
  MMSEQS_BIN="$(command -v mmseqs 2>/dev/null || true)"
fi
if [[ -z "$MMSEQS_BIN" ]]; then
  log_error "mmseqs not found. Install with: ./scripts/install_mmseqs2.sh --conda-env alphafold2 --install-only"
  exit 2
fi

MMSEQS_DB="${ALPHAFOLD_MMSEQS2_DATABASE_PATH:-$DATA_DIR/mmseqs2/uniref90_db}"
if [[ ! -f "${MMSEQS_DB}.dbtype" && ! -f "${MMSEQS_DB}.0" ]]; then
  log_error "MMseqs2 DB prefix not found at: $MMSEQS_DB"
  log_error "Build it with: ./scripts/install_mmseqs2.sh --conda-env alphafold2 --data-dir '$DATA_DIR' --db-tier reduced --build-db"
  exit 2
fi

MAX_SEQS="${ALPHAFOLD_MMSEQS2_MAX_SEQS:-512}"

OUT_DIR="${ALPHAFOLD_E2E_OUTDIR:-/tmp/alphafold_mmseqs2_e2e_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

FASTA_PATH="$OUT_DIR/query.fasta"
cat > "$FASTA_PATH" <<'EOF'
>query
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV
EOF

log_info "Running AlphaFold2 e2e with MMseqs2 MSA"
log_info "  OUT_DIR:  $OUT_DIR"
log_info "  FASTA:    $FASTA_PATH"
log_info "  MMSEQS:   $MMSEQS_BIN"
log_info "  MMSEQSDB: $MMSEQS_DB"

# Inject mmseqs flags at the end of the command template.
cmd="$ALPHAFOLD_NATIVE_CMD --msa_mode=mmseqs2 --mmseqs2_binary_path=$MMSEQS_BIN --mmseqs2_database_path=$MMSEQS_DB --mmseqs2_max_seqs=$MAX_SEQS"

# Fill placeholders.
cmd="${cmd//\{fasta\}/$FASTA_PATH}"
cmd="${cmd//\{out_dir\}/$OUT_DIR}"

LOG_PATH="$OUT_DIR/run.log"
set +e
bash -lc "$cmd" >"$LOG_PATH" 2>&1
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  log_error "AlphaFold run failed (exit=$rc). Log: $LOG_PATH"
  tail -n 80 "$LOG_PATH" || true
  exit $rc
fi

# Basic output checks.
expected_pdb="${ALPHAFOLD_NATIVE_OUTPUT_PDB:-ranked_0.pdb}"
if [[ ! -f "$OUT_DIR/$expected_pdb" ]]; then
  log_error "Expected output PDB not found: $OUT_DIR/$expected_pdb"
  log_error "See log: $LOG_PATH"
  exit 1
fi

if ! find "$OUT_DIR" -maxdepth 3 -name "mmseqs_hits.a3m" | grep -q .; then
  log_error "mmseqs_hits.a3m not found under output dir (MMseqs2 MSA may not have run)"
  log_error "See log: $LOG_PATH"
  exit 1
fi

log_success "E2E OK: produced $expected_pdb and mmseqs_hits.a3m"
log_info "Artifacts: $OUT_DIR"
