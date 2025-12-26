#!/usr/bin/env bash
set -euo pipefail

# Robust AlphaFold benchmark harness for testing optimizations
# Handles cleanup, timeouts, and clear result reporting

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[BENCH]${NC} $*"; }
success() { echo -e "${GREEN}[✓]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; }
warning() { echo -e "${YELLOW}[!]${NC} $*"; }

# Configuration
BENCH_DIR="${1:-/tmp/alphafold_bench_$(date +%Y%m%d_%H%M%S)}"
TEST_FASTA="${2:-/tmp/test_seq.fasta}"
MSA_MODE="${3:-mmseqs2}"  # mmseqs2 or jackhmmer
TIMEOUT_MINUTES=15

mkdir -p "$BENCH_DIR"

# Create test sequence if not exists
if [[ ! -f "$TEST_FASTA" ]]; then
  log "Creating test FASTA at $TEST_FASTA"
  cat > "$TEST_FASTA" << 'EOF'
>test_70aa
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV
EOF
fi

log "╔════════════════════════════════════════════════════════════════╗"
log "║  AlphaFold Optimization Benchmark Suite                       ║"
log "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Setup GPU optimization environment
log "Setting up GPU optimizations..."
if [[ -f "$ROOT_DIR/scripts/setup_gpu_optimization.sh" ]]; then
  source "$ROOT_DIR/scripts/setup_gpu_optimization.sh" 2>&1 | grep -v "^$" | sed 's/^/  /' || true
  log "GPU optimization environment configured"
else
  warning "GPU optimization script not found at $ROOT_DIR/scripts/setup_gpu_optimization.sh"
fi

# Validate GPU setup
log "Validating GPU/CUDA setup..."
if [[ -f "$ROOT_DIR/scripts/validate_gpu_cuda.sh" ]]; then
  bash "$ROOT_DIR/scripts/validate_gpu_cuda.sh" 2>&1 | tail -20 | sed 's/^/  /' || true
else
  warning "GPU validation script not found"
fi

echo ""
log "Benchmark directory: $BENCH_DIR"
log "Test FASTA: $TEST_FASTA ($(grep -v '^>' "$TEST_FASTA" | tr -d '\\n' | wc -c) residues)"
log "MSA mode: $MSA_MODE"
log "Timeout: ${TIMEOUT_MINUTES} minutes per run"
echo ""

# Cleanup function
cleanup() {
  log "Cleaning up background processes..."
  pkill -P $$ 2>/dev/null || true
  sleep 1
}
trap cleanup EXIT INT TERM

# Run single benchmark
run_benchmark() {
  local name=$1
  local out_dir=$2
  shift 2
  local env_vars=("$@")
  
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "Running: $name"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  
  # Build environment
  local env_str=""
  for var in "${env_vars[@]}"; do
    export "$var"
    env_str="$env_str $var"
  done
  
  log "Settings:$env_str"
  
  # Run with timeout
  local start_time=$(date +%s)
  local status="SUCCESS"
  local timeout_sec=$((TIMEOUT_MINUTES * 60))
  
  if timeout ${timeout_sec}s bash scripts/run_profiled_inference.sh "$TEST_FASTA" "$MSA_MODE" "$out_dir" &> "$out_dir/benchmark.log"; then
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo "$elapsed" > "$out_dir/elapsed_seconds.txt"
    success "$name completed in ${elapsed}s"
  else
    local exit_code=$?
    if [[ $exit_code -eq 124 ]]; then
      error "$name TIMED OUT after ${TIMEOUT_MINUTES} minutes"
      status="TIMEOUT"
    else
      error "$name FAILED (exit code: $exit_code)"
      status="FAILED"
      tail -50 "$out_dir/benchmark.log" | head -30
    fi
    echo "FAILED" > "$out_dir/elapsed_seconds.txt"
  fi
  
  echo "$status" > "$out_dir/status.txt"
  echo ""
  
  return 0
}

# Extract metrics
extract_metrics() {
  local out_dir=$1
  local name=$2
  
  if [[ ! -f "$out_dir/status.txt" || "$(cat "$out_dir/status.txt")" != "SUCCESS" ]]; then
    echo "  Status: $(cat "$out_dir/status.txt" 2>/dev/null || echo "UNKNOWN")"
    return
  fi
  
  local elapsed=$(cat "$out_dir/elapsed_seconds.txt" 2>/dev/null || echo "N/A")
  local gpu_avg=$(awk -F, 'NR>1 {sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "$out_dir/gpu_util.csv" 2>/dev/null || echo "N/A")
  
  echo "  Time: ${elapsed}s"
  echo "  GPU Util: ${gpu_avg}%"
  
  # Extract stage timings if available
  if [[ -f "$out_dir/run.log" ]]; then
    local features_time=$(grep -oP "features.*?time.*?[\d.]+s" "$out_dir/run.log" 2>/dev/null | grep -oP "[\d.]+" | head -1 || echo "")
    if [[ -n "$features_time" ]]; then
      echo "  Features: ${features_time}s"
    fi
  fi
}

# Benchmark configurations
log "Starting benchmark suite..."
echo ""

# 1. Baseline
run_benchmark \
  "Baseline (templates=ON, recycles=default)" \
  "$BENCH_DIR/baseline" \
  "AF_DISABLE_TEMPLATES=0" \
  "AF_NUM_RECYCLES=-1" \
  "AF_MMSEQS2_MAX_SEQS=512" \
  "AF_CPU_THREADS=16"

# 2. Templates OFF
run_benchmark \
  "Optimized (templates=OFF, recycles=default)" \
  "$BENCH_DIR/no_templates" \
  "AF_DISABLE_TEMPLATES=1" \
  "AF_NUM_RECYCLES=-1" \
  "AF_MMSEQS2_MAX_SEQS=512" \
  "AF_CPU_THREADS=16"

# 3. Reduced recycles
run_benchmark \
  "Optimized (templates=OFF, recycles=3)" \
  "$BENCH_DIR/reduced_recycles" \
  "AF_DISABLE_TEMPLATES=1" \
  "AF_NUM_RECYCLES=3" \
  "AF_MMSEQS2_MAX_SEQS=256" \
  "AF_CPU_THREADS=16"

# 4. Cached run (rerun with same settings)
run_benchmark \
  "Cached rerun (templates=OFF, recycles=3)" \
  "$BENCH_DIR/cached" \
  "AF_DISABLE_TEMPLATES=1" \
  "AF_NUM_RECYCLES=3" \
  "AF_MMSEQS2_MAX_SEQS=256" \
  "AF_CPU_THREADS=16"

# Results summary
log "╔════════════════════════════════════════════════════════════════╗"
log "║  Benchmark Results Summary                                     ║"
log "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "1. Baseline (templates=ON, recycles=default):"
extract_metrics "$BENCH_DIR/baseline" "Baseline"
echo ""

echo "2. Optimized (templates=OFF, recycles=default):"
extract_metrics "$BENCH_DIR/no_templates" "No templates"
echo ""

echo "3. Optimized (templates=OFF, recycles=3):"
extract_metrics "$BENCH_DIR/reduced_recycles" "Reduced recycles"
echo ""

echo "4. Cached rerun (templates=OFF, recycles=3):"
extract_metrics "$BENCH_DIR/cached" "Cached"
echo ""

# Calculate speedups if all succeeded
if [[ -f "$BENCH_DIR/baseline/elapsed_seconds.txt" && \
      -f "$BENCH_DIR/reduced_recycles/elapsed_seconds.txt" && \
      "$(cat "$BENCH_DIR/baseline/status.txt")" == "SUCCESS" && \
      "$(cat "$BENCH_DIR/reduced_recycles/status.txt")" == "SUCCESS" ]]; then
  
  baseline_time=$(cat "$BENCH_DIR/baseline/elapsed_seconds.txt")
  optimized_time=$(cat "$BENCH_DIR/reduced_recycles/elapsed_seconds.txt")
  
  if [[ "$baseline_time" != "FAILED" && "$optimized_time" != "FAILED" ]]; then
    speedup=$(echo "scale=2; $baseline_time / $optimized_time" | bc)
    saved=$(echo "$baseline_time - $optimized_time" | bc)
    
    success "═══════════════════════════════════════════════════"
    success "Speedup: ${speedup}x (saved ${saved}s)"
    success "Baseline: ${baseline_time}s → Optimized: ${optimized_time}s"
    success "═══════════════════════════════════════════════════"
  fi
fi

log "Full logs available in: $BENCH_DIR"
log "Done!"
