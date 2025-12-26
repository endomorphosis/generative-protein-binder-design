#!/usr/bin/env bash
# Compare JackHMMER vs MMseqs2 MSA generation performance
#
# Measures:
# - MSA generation time
# - AlphaFold total inference time
# - GPU utilization during inference
# - MSA quality metrics (depth, coverage)

set -euo pipefail

BENCH_DIR="${1:-/tmp/msa_bench_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$BENCH_DIR"

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

# Test sequence (lysozyme fragment, 70 residues)
TEST_SEQUENCE="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"

log_info "╔═══════════════════════════════════════════════════════════╗"
log_info "║  MSA Performance Comparison: JackHMMER vs MMseqs2         ║"
log_info "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate alphafold2

# System info
log_info "System Information:"
echo "  CPU cores: $(nproc)"
echo "  RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
ALPHAFOLD_DIR="$ROOT_DIR/tools/alphafold2"

# Ensure AlphaFold environment is set
export ALPHAFOLD_DATA_DIR="$DATA_DIR"

# Set required database paths for reduced_dbs preset
UNIREF90_DB="$DATA_DIR/uniref90/uniref90.fasta"
MGNIFY_DB="$DATA_DIR/mgnify/mgy_clusters_2022_05.fa"
SMALL_BFD_DB="$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta"
PDB70_DB="$DATA_DIR/pdb70/pdb70"
TEMPLATE_MMCIF="$DATA_DIR/pdb_mmcif/mmcif_files"
OBSOLETE_PDBS="$DATA_DIR/pdb_mmcif/obsolete.dat"

# Test FASTA
TEST_FASTA="$BENCH_DIR/test_sequence.fasta"
cat > "$TEST_FASTA" <<EOF
>test_sequence
$TEST_SEQUENCE
EOF

log_info "Test sequence: $(echo $TEST_SEQUENCE | cut -c1-40)... (${#TEST_SEQUENCE} residues)"
echo ""

#############################################
# Benchmark 1: JackHMMER MSA
#############################################
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Benchmark 1: JackHMMER MSA Generation"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

JACKHMMER_OUT="$BENCH_DIR/jackhmmer"
mkdir -p "$JACKHMMER_OUT"

log_info "Starting GPU monitoring..."
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits -l 1 > "$JACKHMMER_OUT/gpu_util.csv" &
GPU_MON_PID=$!

log_info "Running AlphaFold with JackHMMER MSA..."
START=$(date +%s)

THREADS=$(nproc)
log_info "Using $THREADS CPU threads for JackHMMER..."
echo ""

cd "$ALPHAFOLD_DIR"
python run_alphafold.py \
    --fasta_paths="$TEST_FASTA" \
    --max_template_date=2022-01-01 \
    --model_preset=monomer \
    --db_preset=reduced_dbs \
    --data_dir="$DATA_DIR" \
    --output_dir="$JACKHMMER_OUT" \
    --use_gpu_relax=false \
    --models_to_relax=none \
    --logtostderr \
    --msa_mode=jackhmmer \
    --jackhmmer_n_cpu="$THREADS" \
    --hhsearch_n_cpu="$THREADS" \
    --hmmsearch_n_cpu="$THREADS" \
    --uniref90_database_path="$UNIREF90_DB" \
    --mgnify_database_path="$MGNIFY_DB" \
    --small_bfd_database_path="$SMALL_BFD_DB" \
    --pdb70_database_path="$PDB70_DB" \
    --template_mmcif_dir="$TEMPLATE_MMCIF" \
    --obsolete_pdbs_path="$OBSOLETE_PDBS" \
    &> "$JACKHMMER_OUT/run.log" || {
        log_error "AlphaFold (JackHMMER) failed"
        kill $GPU_MON_PID 2>/dev/null || true
        cat "$JACKHMMER_OUT/run.log" | tail -50
        exit 1
    }

END=$(date +%s)
JACKHMMER_TIME=$((END - START))

kill $GPU_MON_PID 2>/dev/null || true
wait $GPU_MON_PID 2>/dev/null || true

log_success "JackHMMER run completed in ${JACKHMMER_TIME}s"

# Extract MSA stats
MSA_DIR="$JACKHMMER_OUT/test_sequence/msas"
if [[ -d "$MSA_DIR" ]]; then
    NUM_SEQS=$(grep -c "^>" "$MSA_DIR"/*.a3m 2>/dev/null | awk -F: '{s+=$2} END {print s}' || echo 0)
    log_info "  MSA sequences: $NUM_SEQS"
fi

# Calculate avg GPU utilization
if [[ -f "$JACKHMMER_OUT/gpu_util.csv" ]]; then
    AVG_GPU=$(awk -F, '{sum+=$1; count++} END {printf "%.1f", sum/count}' "$JACKHMMER_OUT/gpu_util.csv")
    AVG_MEM=$(awk -F, '{sum+=$2; count++} END {printf "%.1f", sum/count}' "$JACKHMMER_OUT/gpu_util.csv")
    log_info "  Avg GPU utilization: ${AVG_GPU}%"
    log_info "  Avg GPU memory: ${AVG_MEM}%"
else
    AVG_GPU="N/A"
    AVG_MEM="N/A"
fi

echo ""

#############################################
# Benchmark 2: MMseqs2 MSA
#############################################
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Benchmark 2: MMseqs2 MSA Generation"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

MMSEQS2_OUT="$BENCH_DIR/mmseqs2"
mkdir -p "$MMSEQS2_OUT"

# Find MMseqs2 DB (prefer env value)
if [[ -n "${ALPHAFOLD_MMSEQS2_DATABASE_PATH:-}" ]]; then
    MMSEQS2_DB="$ALPHAFOLD_MMSEQS2_DATABASE_PATH"
else
    MMSEQS2_DB=$(find "$DATA_DIR/mmseqs2" -name "*_db" -type f | grep -v "_h$" | grep -v ".idx" | head -1)
fi
if [[ -z "$MMSEQS2_DB" ]]; then
    log_error "MMseqs2 DB not found (env or $DATA_DIR/mmseqs2)"
    exit 1
fi
log_info "Using MMseqs2 DB: $MMSEQS2_DB"
log_info "Using $THREADS CPU threads for MMseqs2..."
echo ""

log_info "Starting GPU monitoring..."
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,memory.used,memory.total --format=csv,noheader,nounits -l 1 > "$MMSEQS2_OUT/gpu_util.csv" &
GPU_MON_PID=$!

log_info "Running AlphaFold with MMseqs2 MSA..."
START=$(date +%s)

cd "$ALPHAFOLD_DIR"
python run_alphafold.py \
    --fasta_paths="$TEST_FASTA" \
    --max_template_date=2022-01-01 \
    --model_preset=monomer \
    --db_preset=reduced_dbs \
    --data_dir="$DATA_DIR" \
    --output_dir="$MMSEQS2_OUT" \
    --use_gpu_relax=false \
    --models_to_relax=none \
    --logtostderr \
    --msa_mode=mmseqs2 \
    --jackhmmer_n_cpu="$THREADS" \
    --hhsearch_n_cpu="$THREADS" \
    --hmmsearch_n_cpu="$THREADS" \
    --mmseqs2_database_path="$MMSEQS2_DB" \
    --mmseqs2_max_seqs=512 \
    --uniref90_database_path="$UNIREF90_DB" \
    --mgnify_database_path="$MGNIFY_DB" \
    --small_bfd_database_path="$SMALL_BFD_DB" \
    --pdb70_database_path="$PDB70_DB" \
    --template_mmcif_dir="$TEMPLATE_MMCIF" \
    --obsolete_pdbs_path="$OBSOLETE_PDBS" \
    &> "$MMSEQS2_OUT/run.log" || {
        log_error "AlphaFold (MMseqs2) failed"
        kill $GPU_MON_PID 2>/dev/null || true
        cat "$MMSEQS2_OUT/run.log" | tail -50
        exit 1
    }

END=$(date +%s)
MMSEQS2_TIME=$((END - START))

kill $GPU_MON_PID 2>/dev/null || true
wait $GPU_MON_PID 2>/dev/null || true

log_success "MMseqs2 run completed in ${MMSEQS2_TIME}s"

# Extract MSA stats
MSA_DIR="$MMSEQS2_OUT/test_sequence/msas"
if [[ -d "$MSA_DIR" ]]; then
    NUM_SEQS=$(grep -c "^>" "$MSA_DIR"/*.a3m 2>/dev/null | awk -F: '{s+=$2} END {print s}' || echo 0)
    log_info "  MSA sequences: $NUM_SEQS"
fi

# Calculate avg GPU utilization
if [[ -f "$MMSEQS2_OUT/gpu_util.csv" ]]; then
    AVG_GPU_MM=$(awk -F, '{sum+=$1; count++} END {printf "%.1f", sum/count}' "$MMSEQS2_OUT/gpu_util.csv")
    AVG_MEM_MM=$(awk -F, '{sum+=$2; count++} END {printf "%.1f", sum/count}' "$MMSEQS2_OUT/gpu_util.csv")
    log_info "  Avg GPU utilization: ${AVG_GPU_MM}%"
    log_info "  Avg GPU memory: ${AVG_MEM_MM}%"
else
    AVG_GPU_MM="N/A"
    AVG_MEM_MM="N/A"
fi

echo ""

#############################################
# Summary Comparison
#############################################
log_success "╔═══════════════════════════════════════════════════════════╗"
log_success "║  Performance Comparison Summary                           ║"
log_success "╚═══════════════════════════════════════════════════════════╝"
echo ""

SPEEDUP=$(echo "scale=2; $JACKHMMER_TIME / $MMSEQS2_TIME" | bc)

{
    echo "=== MSA Performance Comparison ==="
    echo "Date: $(date)"
    echo "Test sequence: $TEST_SEQUENCE"
    echo ""
    echo "JackHMMER:"
    echo "  Total time: ${JACKHMMER_TIME}s"
    echo "  Avg GPU utilization: ${AVG_GPU}%"
    echo "  Avg GPU memory: ${AVG_MEM}%"
    echo ""
    echo "MMseqs2:"
    echo "  Total time: ${MMSEQS2_TIME}s"
    echo "  Avg GPU utilization: ${AVG_GPU_MM}%"
    echo "  Avg GPU memory: ${AVG_MEM_MM}%"
    echo ""
    echo "Speedup: ${SPEEDUP}x"
    echo "Time saved: $((JACKHMMER_TIME - MMSEQS2_TIME))s"
} > "$BENCH_DIR/comparison_summary.txt"

cat "$BENCH_DIR/comparison_summary.txt"

echo ""
log_info "Results saved to: $BENCH_DIR"
log_info "  JackHMMER output: $JACKHMMER_OUT"
log_info "  MMseqs2 output: $MMSEQS2_OUT"
log_info "  Summary: $BENCH_DIR/comparison_summary.txt"
