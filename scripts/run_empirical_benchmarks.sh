#!/usr/bin/env bash
set -euo pipefail

# Empirical benchmarks for MMseqs2 AlphaFold runs.
# Produces run artifacts + a compact summary under a workspace-local directory.
#
# Usage:
#   scripts/run_empirical_benchmarks.sh [bench_dir]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${1:-$ROOT_DIR/benchmarks/af_empirical_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$BENCH_DIR"

log() { echo "[BENCH] $*"; }

FASTA="$BENCH_DIR/test70.fasta"
cat >"$FASTA" <<"EOF"
>test70
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV
EOF

# Default knobs (override via env)
AF_CPU_THREADS="${AF_CPU_THREADS:-16}"

run_case() {
  local name="$1"; shift
  local out="$BENCH_DIR/$name"
  mkdir -p "$out"

  log "Running $name → $out"
  ("$ROOT_DIR/scripts/run_profiled_inference.sh" "$FASTA" mmseqs2 "$out")
}

extract_metrics() {
  local name="$1"
  local dir="$BENCH_DIR/$name"

  local total=""
  local result=""
  if [[ -f "$dir/summary.txt" ]]; then
    total=$(grep -E "^Total time \(s\):" "$dir/summary.txt" | awk '{print $4}' || true)
    result=$(grep -E "^Result:" "$dir/summary.txt" | awk '{print $2}' || true)
  fi

  local avg_gpu=""
  local avg_pwr=""
  local samples=""
  if [[ -f "$dir/gpu_util.csv" ]]; then
    avg_gpu=$(awk -F, '{sum+=$1; n++} END{if(n) printf "%.1f", sum/n; else printf ""}' "$dir/gpu_util.csv")
    avg_pwr=$(awk -F, '{sum+=$5; n++} END{if(n) printf "%.2f", sum/n; else printf ""}' "$dir/gpu_util.csv" 2>/dev/null || true)
    samples=$(wc -l "$dir/gpu_util.csv" | awk '{print $1}')
  fi

  echo -e "${name}\t${result}\t${total}\t${avg_gpu}\t${avg_pwr}\t${samples}"
}

# Case 1: baseline
export AF_MMSEQS2_MAX_SEQS="${AF_MMSEQS2_MAX_SEQS_BASELINE:-512}"
export AF_DISABLE_TEMPLATES=0
export AF_NUM_RECYCLES=-1
export AF_NUM_ENSEMBLE=-1
export AF_CPU_THREADS
run_case mmseqs2_baseline

# Case 2: optimized (templates off + fewer recycles + smaller max_seqs)
export AF_MMSEQS2_MAX_SEQS="${AF_MMSEQS2_MAX_SEQS_OPTIMIZED:-256}"
export AF_DISABLE_TEMPLATES=1
export AF_NUM_RECYCLES="${AF_NUM_RECYCLES_OPTIMIZED:-3}"
export AF_NUM_ENSEMBLE="${AF_NUM_ENSEMBLE_OPTIMIZED:-1}"
run_case mmseqs2_optimized

# Case 3: optimized cache hit
run_case mmseqs2_optimized_cached

# Summaries
TSV="$BENCH_DIR/results.tsv"
{
  echo -e "case\tresult\ttotal_s\tavg_gpu_util_pct\tavg_gpu_power_w\tsamples"
  extract_metrics mmseqs2_baseline
  extract_metrics mmseqs2_optimized
  extract_metrics mmseqs2_optimized_cached
} > "$TSV"

MD="$BENCH_DIR/results.md"
{
  echo "# Empirical MMseqs2 Benchmarks"
  echo
  echo "Bench dir: $BENCH_DIR"
  echo
  echo "| case | result | total (s) | avg GPU util (%) | avg GPU power (W) | samples |"
  echo "|---|---:|---:|---:|---:|---:|"
  tail -n +2 "$TSV" | awk -F"\t" '{printf "| %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6}'
  echo
  echo "FASTA: $FASTA"
  echo
  echo "Notes:"
  echo "- baseline: templates ON, recycles default, max_seqs=$AF_MMSEQS2_MAX_SEQS_BASELINE"
  echo "- optimized: templates OFF, recycles=$AF_NUM_RECYCLES_OPTIMIZED, max_seqs=$AF_MMSEQS2_MAX_SEQS_OPTIMIZED"
  echo "- optimized_cached: same as optimized, should reuse cached MSAs when available"
} > "$MD"

log "Done. Results: $MD"
