#!/usr/bin/env bash
set -euo pipefail

# Pre-warm AlphaFold databases into Linux page cache.
#
# Important:
# - This does NOT lock memory (no mlock). The kernel may evict these pages under pressure.
# - This is meant to reduce cold-start I/O latency for MSA/search phases.
# - For safety, this script stops early if MemAvailable is below a threshold.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/warm_alphafold_page_cache.sh [--data-dir PATH] [--min-mem-gb N] [--tier reduced|full|auto]

Options:
  --data-dir PATH   AlphaFold data dir (default: $ALPHAFOLD_DATA_DIR or ~/.cache/alphafold)
  --min-mem-gb N    Stop prewarming if MemAvailable < N GiB (default: 6)
  --tier VALUE      reduced|full|auto (default: auto)

Notes:
  - Uses vmtouch -t if installed (best). Falls back to sequential reads.
  - Only warms the big MSA/search DB files by default (not the entire mmCIF directory).
EOF
}

DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
MIN_MEM_GB=6
TIER=auto

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"; shift 2 ;;
    --min-mem-gb)
      MIN_MEM_GB="$2"; shift 2 ;;
    --tier)
      TIER="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mem_available_kb() {
  awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0
}

check_mem_or_stop() {
  local avail_kb
  avail_kb="$(mem_available_kb)"
  local min_kb=$((MIN_MEM_GB * 1024 * 1024))
  if [[ "$avail_kb" -lt "$min_kb" ]]; then
    echo "[warm-cache] MemAvailable is low ($(printf "%.1f" "$(awk -v kb="$avail_kb" 'BEGIN{print kb/1024/1024}')") GiB). Stopping early to avoid memory pressure." >&2
    return 1
  fi
  return 0
}

# Decide tier based on what exists on disk.
if [[ "$TIER" == "auto" ]]; then
  if [[ -f "$DATA_DIR/uniref90/uniref90.fasta" && -f "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" ]]; then
    TIER=reduced
  else
    TIER=full
  fi
fi

# Build list of files to prewarm.
# Keep this conservative: focus on the largest, most frequently scanned FASTA/indices.
FILES=(
  "$DATA_DIR/uniref90/uniref90.fasta"
  "$DATA_DIR/mgnify/mgy_clusters_2022_05.fa"
  "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta"
)

if [[ "$TIER" == "full" ]]; then
  # Full DB tier often uses BFD/uniref30; include if present.
  FILES+=(
    "$DATA_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
    "$DATA_DIR/uniref30/UniRef30_2021_03"
  )
fi

# Template search uses PDB70; warming the index can help a bit.
FILES+=(
  "$DATA_DIR/pdb70/pdb70"
)

# Multimer extras (only if present).
FILES+=(
  "$DATA_DIR/pdb_seqres/pdb_seqres.txt"
  "$DATA_DIR/uniprot/uniprot.fasta"
)

warm_one() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    return 0
  fi

  check_mem_or_stop || return 1

  if command -v vmtouch >/dev/null 2>&1; then
    echo "[warm-cache] vmtouch -t $path"
    vmtouch -t "$path" >/dev/null
  else
    # Fallback: sequentially read file into page cache.
    # Avoid O_DIRECT; we want the kernel cache to fill naturally and be evictable.
    echo "[warm-cache] read $path"
    dd if="$path" of=/dev/null bs=16M status=none || cat "$path" >/dev/null
  fi
}

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[warm-cache] Data dir not found: $DATA_DIR" >&2
  exit 1
fi

echo "[warm-cache] AlphaFold data dir: $DATA_DIR"
echo "[warm-cache] Tier: $TIER"
echo "[warm-cache] Min MemAvailable: ${MIN_MEM_GB} GiB"
echo "[warm-cache] Mode: evictable page cache (not locked)"

touched=0
for f in "${FILES[@]}"; do
  if [[ -e "$f" ]]; then
    warm_one "$f" || break
    touched=$((touched+1))
  fi
done

echo "[warm-cache] Done. Warmed $touched file(s)."
