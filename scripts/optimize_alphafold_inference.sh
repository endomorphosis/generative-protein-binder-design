#!/usr/bin/env bash
# Comprehensive AlphaFold2 + MMseqs2 Optimization Plan
# 
# Problem: MMseqs2 search against 570GB indexed DB is getting killed (OOM)
# Goal: Speed up AlphaFold2 inference by migrating from JackHMMER → MMseqs2
#
# This script implements a multi-phase approach to benchmark and optimize.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
MMSEQS2_DIR="$DATA_DIR/mmseqs2"

cat << 'BANNER'
╔═══════════════════════════════════════════════════════════════════╗
║  AlphaFold2 Inference Optimization Plan                          ║
║  MMseqs2 Migration for GPU Utilization Improvement               ║
╚═══════════════════════════════════════════════════════════════════╝
BANNER

echo ""
log_info "Analysis of current bottlenecks:"
echo "  1. JackHMMER MSA generation is CPU-bound and slow"
echo "  2. MMseqs2 can reduce MSA time 10-100x"
echo "  3. Current 570GB MMseqs2 DB causes OOM during search"
echo "  4. Need optimized DB size vs accuracy tradeoff"
echo ""

log_info "Multi-phase optimization strategy:"
echo ""

cat << 'PLAN'
PHASE 1: BASELINE BENCHMARKS (5-10 min)
────────────────────────────────────────
□ Benchmark JackHMMER MSA time on test sequence
□ Measure current GPU utilization during full pipeline
□ Establish baseline: MSA time, GPU%, total inference time

PHASE 2: DB SIZE OPTIMIZATION (30-60 min)
───────────────────────────────────────────
□ Test reduced MMseqs2 DB sizes:
  - Tiny: 50K seqs (~5GB, fast functional test)
  - Small: 500K seqs (~50GB, balanced)
  - Medium: 5M seqs (~300GB, production candidate)
□ Benchmark search time vs MSA quality for each
□ Choose optimal size based on memory footprint + search speed

PHASE 3: MMSEQS2 TUNING (15-30 min)
────────────────────────────────────
□ Optimize MMseqs2 search parameters:
  - --max-seqs: test 128, 256, 512, 1024
  - --sensitivity: test 7.5 (default), 6.0 (faster)
  - --threads: match available CPUs
□ Test memory-mapped vs in-memory DB access
□ Benchmark each config on test sequence

PHASE 4: PIPELINE INTEGRATION (10-20 min)
──────────────────────────────────────────
□ Run full AlphaFold2 pipeline with optimized MMseqs2
□ Measure end-to-end time breakdown:
  - MSA generation
  - Template search  
  - Model inference (GPU)
  - Relaxation
□ Compare GPU utilization: JackHMMER vs MMseqs2

PHASE 5: COMPREHENSIVE COMPARISON (5 min)
─────────────────────────────────────────
□ Side-by-side metrics:
  - Total inference time
  - MSA generation time
  - GPU utilization %
  - Peak memory usage
  - MSA depth/quality
□ Generate optimization report

ESTIMATED TOTAL TIME: 1-2 hours
PLAN

echo ""
read -p "Start Phase 1: Baseline Benchmarks? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Exiting. Run individual phases with:"
    echo "  Phase 1: bash $ROOT_DIR/scripts/bench_jackhmmer_baseline.sh"
    echo "  Phase 2: bash $ROOT_DIR/scripts/optimize_mmseqs2_db_size.sh"
    echo "  Phase 3: bash $ROOT_DIR/scripts/tune_mmseqs2_params.sh"
    echo "  Phase 4: bash $ROOT_DIR/scripts/bench_full_pipeline.sh"
    echo "  Phase 5: bash $ROOT_DIR/scripts/generate_optimization_report.sh"
    exit 0
fi

log_step "PHASE 1: Baseline Benchmarks"
echo ""

# Create benchmark output directory
BENCH_DIR="/tmp/alphafold_optimization_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCH_DIR"
log_info "Benchmark results: $BENCH_DIR"

# Test sequence (70 residues from T4 lysozyme)
TEST_SEQ="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"
echo ">test_query" > "$BENCH_DIR/test.fasta"
echo "$TEST_SEQ" >> "$BENCH_DIR/test.fasta"

log_info "Test sequence: ${#TEST_SEQ} residues"
echo ""

# Benchmark 1: JackHMMER MSA time
log_info "[1/3] Benchmarking JackHMMER MSA generation..."
if command -v jackhmmer >/dev/null 2>&1 && [[ -f "$DATA_DIR/uniref90/uniref90.fasta" ]]; then
    START=$(date +%s)
    timeout 300 jackhmmer \
        --cpu 8 \
        -N 1 \
        -o /dev/null \
        -A "$BENCH_DIR/jackhmmer_uniref90.sto" \
        "$BENCH_DIR/test.fasta" \
        "$DATA_DIR/uniref90/uniref90.fasta" \
        > "$BENCH_DIR/jackhmmer.log" 2>&1 || log_warning "JackHMMER timed out or failed"
    END=$(date +%s)
    JACKHMMER_TIME=$((END - START))
    log_success "JackHMMER MSA time: ${JACKHMMER_TIME}s"
    echo "$JACKHMMER_TIME" > "$BENCH_DIR/jackhmmer_time.txt"
else
    log_warning "JackHMMER or UniRef90 FASTA not available, skipping"
    echo "N/A" > "$BENCH_DIR/jackhmmer_time.txt"
fi
echo ""

# Benchmark 2: Check current MMseqs2 DB and rebuild if needed
log_info "[2/3] Analyzing current MMseqs2 DB..."
CURRENT_DB="$MMSEQS2_DIR/uniref90_db"
if [[ -f "${CURRENT_DB}.dbtype" ]]; then
    DB_SIZE=$(du -sh "$MMSEQS2_DIR" | awk '{print $1}')
    log_info "Current MMseqs2 DB size: $DB_SIZE"
    
    # Check available memory
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    AVAIL_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
    log_info "System memory: ${AVAIL_MEM_GB}GB available / ${TOTAL_MEM_GB}GB total"
    
    if [[ "$DB_SIZE" == *"G" ]]; then
        SIZE_NUM=$(echo "$DB_SIZE" | sed 's/G//')
        if (( $(echo "$SIZE_NUM > $AVAIL_MEM_GB" | bc -l) )); then
            log_warning "DB size ($DB_SIZE) exceeds available memory (${AVAIL_MEM_GB}GB)"
            log_warning "This will cause OOM during search. Need to rebuild smaller DB."
            REBUILD_DB=true
        else
            REBUILD_DB=false
        fi
    fi
else
    log_warning "MMseqs2 DB not found at: $CURRENT_DB"
    REBUILD_DB=true
fi
echo ""

# Benchmark 3: System info
log_info "[3/3] Collecting system info..."
{
    echo "=== System Info ==="
    echo "Date: $(date)"
    echo "CPU cores: $(nproc)"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
    echo "Disk free: $(df -h "$DATA_DIR" | tail -1 | awk '{print $4}')"
    echo ""
    echo "=== AlphaFold Config ==="
    echo "Data dir: $DATA_DIR"
    echo "MMseqs2 DB: $CURRENT_DB"
    echo "Test sequence: ${#TEST_SEQ} residues"
} > "$BENCH_DIR/system_info.txt"
log_success "System info saved"
echo ""

log_success "Phase 1 Complete!"
echo ""
cat "$BENCH_DIR/system_info.txt"
echo ""

if [[ "$REBUILD_DB" == true ]]; then
    log_step "PHASE 2: DB Size Optimization"
    echo ""
    log_info "Current MMseqs2 DB is too large for available memory."
    log_info "Need to build a smaller, optimized DB."
    echo ""
    
    read -p "Build optimized MMseqs2 DB? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash "$ROOT_DIR/scripts/rebuild_optimized_mmseqs2_db.sh" "$BENCH_DIR"
    fi
fi

log_info "Benchmark results saved to: $BENCH_DIR"
log_info "Review results and proceed with remaining phases."
