#!/usr/bin/env bash
# End-to-end test: AlphaFold2 with MMseqs2 MSA mode
#
# This test verifies that AlphaFold2 can successfully use MMseqs2 for MSA
# generation instead of JackHMMER, which should improve GPU utilization by
# reducing MSA CPU bottleneck time.

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
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Test configuration
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-/tmp/alphafold_mmseqs2_e2e_test_$$}"
ALPHAFOLD_DIR="${ALPHAFOLD_DIR:-$ROOT_DIR/tools/alphafold2}"
ALPHAFOLD_DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
CONDA_ENV="${ALPHAFOLD_CONDA_ENV:-alphafold2}"

# Small test sequence (10 residues from T4 lysozyme)
TEST_SEQUENCE="MNIFEMLRID"

log_info "AlphaFold2 + MMseqs2 End-to-End Test"
log_info "======================================"
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v conda >/dev/null 2>&1; then
  log_error "conda not found"
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  log_error "Conda env '$CONDA_ENV' not found. Run: ./scripts/install_alphafold2_complete.sh"
  exit 1
fi

conda activate "$CONDA_ENV"

if ! command -v mmseqs >/dev/null 2>&1; then
  log_error "mmseqs not found in env '$CONDA_ENV'. Run: ./scripts/install_mmseqs2.sh --conda-env $CONDA_ENV --install-only"
  exit 1
fi

if [[ ! -d "$ALPHAFOLD_DIR" ]]; then
  log_error "AlphaFold directory not found: $ALPHAFOLD_DIR"
  exit 1
fi

if [[ ! -f "$ALPHAFOLD_DIR/run_alphafold.py" ]]; then
  log_error "AlphaFold run_alphafold.py not found"
  exit 1
fi

MMSEQS_DB="${ALPHAFOLD_DATA_DIR}/mmseqs2/uniref90_db"
if [[ ! -f "${MMSEQS_DB}.dbtype" ]]; then
  log_error "MMseqs2 DB not found: $MMSEQS_DB"
  log_error "Run: ./scripts/install_mmseqs2.sh --conda-env $CONDA_ENV --data-dir $ALPHAFOLD_DATA_DIR --db-tier reduced --build-db"
  exit 1
fi

log_success "Prerequisites OK"
log_info "  mmseqs: $(which mmseqs) ($(mmseqs version 2>&1 || echo 'unknown'))"
log_info "  MMseqs DB: $MMSEQS_DB"
log_info "  AlphaFold: $ALPHAFOLD_DIR"
log_info "  Data dir: $ALPHAFOLD_DATA_DIR"
echo ""

# Prepare test input
log_info "Preparing test input..."
mkdir -p "$TEST_OUTPUT_DIR"
TEST_FASTA="$TEST_OUTPUT_DIR/test.fasta"
cat > "$TEST_FASTA" <<EOF
>test_sequence
$TEST_SEQUENCE
EOF

log_info "Test FASTA: $TEST_FASTA"
log_info "Output dir: $TEST_OUTPUT_DIR"
echo ""

# Run AlphaFold with MMseqs2 mode
log_info "Running AlphaFold2 with --msa_mode=mmseqs2..."
START_TIME=$(date +%s)

export PYTHONPATH="$ALPHAFOLD_DIR:${PYTHONPATH:-}"

python "$ALPHAFOLD_DIR/run_alphafold.py" \
  --fasta_paths="$TEST_FASTA" \
  --output_dir="$TEST_OUTPUT_DIR" \
  --data_dir="$ALPHAFOLD_DATA_DIR" \
  --db_preset=reduced_dbs \
  --model_preset=monomer \
  --models_to_relax=none \
  --use_gpu_relax=false \
  --max_template_date=2022-12-31 \
  --msa_mode=mmseqs2 \
  --mmseqs2_binary_path="$(which mmseqs)" \
  --mmseqs2_database_path="$MMSEQS_DB" \
  --mmseqs2_max_seqs=256 \
  --uniref90_database_path="$ALPHAFOLD_DATA_DIR/uniref90/uniref90.fasta" \
  --mgnify_database_path="$ALPHAFOLD_DATA_DIR/mgnify/mgy_clusters_2022_05.fa" \
  --pdb70_database_path="$ALPHAFOLD_DATA_DIR/pdb70/pdb70" \
  --small_bfd_database_path="$ALPHAFOLD_DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" \
  --template_mmcif_dir="$ALPHAFOLD_DATA_DIR/pdb_mmcif/mmcif_files" \
  --obsolete_pdbs_path="$ALPHAFOLD_DATA_DIR/pdb_mmcif/obsolete.dat" \
  2>&1 | tee "$TEST_OUTPUT_DIR/run.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
log_info "AlphaFold run completed in ${DURATION}s"
echo ""

# Verify output
log_info "Verifying output..."

RESULT_DIR="$TEST_OUTPUT_DIR/test_sequence"
if [[ ! -d "$RESULT_DIR" ]]; then
  log_error "Result directory not created: $RESULT_DIR"
  exit 1
fi

# Check for MMseqs2 MSA output
MMSEQS_MSA="$RESULT_DIR/mmseqs_hits.a3m"
if [[ -f "$MMSEQS_MSA" ]]; then
  MSA_SIZE=$(grep -c "^>" "$MMSEQS_MSA" || echo 0)
  log_success "MMseqs2 MSA generated: $MMSEQS_MSA ($MSA_SIZE sequences)"
else
  log_error "MMseqs2 MSA file not found: $MMSEQS_MSA"
  log_error "Check logs in: $TEST_OUTPUT_DIR/run.log"
  exit 1
fi

# Check for ranked PDB output
RANKED_PDB="$RESULT_DIR/ranked_0.pdb"
if [[ -f "$RANKED_PDB" ]]; then
  PDB_SIZE=$(wc -l < "$RANKED_PDB")
  log_success "Structure prediction succeeded: $RANKED_PDB ($PDB_SIZE lines)"
else
  log_error "Ranked PDB not found: $RANKED_PDB"
  exit 1
fi

# Check timings.json for MSA time
TIMINGS="$RESULT_DIR/timings.json"
if [[ -f "$TIMINGS" ]]; then
  log_info "Timings saved: $TIMINGS"
  if command -v jq >/dev/null 2>&1; then
    echo ""
    log_info "MSA timing breakdown:"
    jq -r '.features' "$TIMINGS" 2>/dev/null || echo "  (timings not in expected format)"
  fi
else
  log_warning "Timings file not found: $TIMINGS"
fi

echo ""
log_success "===================================="
log_success "End-to-End Test PASSED"
log_success "===================================="
echo ""
log_info "Test output preserved in: $TEST_OUTPUT_DIR"
log_info ""
log_info "Key results:"
log_info "  - MMseqs2 MSA: $MMSEQS_MSA"
log_info "  - Structure: $RANKED_PDB"
log_info "  - Runtime: ${DURATION}s"
log_info ""
log_info "To compare with JackHMMER mode, run the same test without --msa_mode=mmseqs2"
echo ""
