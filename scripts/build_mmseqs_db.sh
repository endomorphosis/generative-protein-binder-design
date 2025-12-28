#!/usr/bin/env bash
set -euo pipefail

# Build an MMseqs2 searchable database from a FASTA file.
#
# Usage:
#   ./scripts/build_mmseqs_db.sh <input_fasta> <output_db_prefix> [OPTIONS]
#
# Example:
#   ./scripts/build_mmseqs_db.sh /data/uniref90.fasta /data/mmseqs/uniref90_db --threads 32 --tmp-dir /data/mmseqs/tmp --split-memory-limit 80G

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/build_mmseqs_db.sh <input_fasta> <output_db_prefix> [OPTIONS]

Options:
  --threads N             Number of threads for createindex (default: 16)
  --tmp-dir PATH          Temp directory for createindex (default: <output_dir>/tmp_mmseqs)
  --split-memory-limit X  Limit memory during indexing (e.g., 64G). Enables split index to fit RAM.

Notes:
- Requires `mmseqs` on PATH.
- This creates a DB and an index (recommended for faster searches).
- The output is a DB *prefix* (MMseqs2 writes multiple files with that prefix).
- The temp dir can consume significant disk space during indexing; ensure it has enough room.
EOF
  exit 2
fi

INPUT_FASTA="$1"
OUT_DB="$2"
shift 2

THREADS=16
TMP_DIR=""
SPLIT_MEM_LIMIT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      THREADS="${2:?missing value}"
      shift 2
      ;;
    --tmp-dir)
      TMP_DIR="${2:?missing value}"
      shift 2
      ;;
    --split-memory-limit)
      SPLIT_MEM_LIMIT="${2:?missing value}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v mmseqs >/dev/null 2>&1; then
  echo "mmseqs not found on PATH" >&2
  exit 1
fi

if [[ ! -f "$INPUT_FASTA" ]]; then
  echo "Input FASTA not found: $INPUT_FASTA" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_DB")"

# Default temp dir to output dir/tmp_mmseqs (avoids filling /tmp on small partitions)
if [[ -z "$TMP_DIR" ]]; then
  TMP_DIR="$(dirname "$OUT_DB")/tmp_mmseqs"
fi
mkdir -p "$TMP_DIR"

# Trap to ensure cleanup even on errors
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Building MMseqs2 DB: $OUT_DB"
mmseqs createdb "$INPUT_FASTA" "$OUT_DB"

echo "Creating MMseqs2 index (threads=$THREADS, tmp=$TMP_DIR${SPLIT_MEM_LIMIT:+, split-memory-limit=$SPLIT_MEM_LIMIT})"
if [[ -n "$SPLIT_MEM_LIMIT" ]]; then
  mmseqs createindex "$OUT_DB" "$TMP_DIR" --threads "$THREADS" --split-memory-limit "$SPLIT_MEM_LIMIT"
else
  mmseqs createindex "$OUT_DB" "$TMP_DIR" --threads "$THREADS"
fi

echo "Cleaning up temp directory: $TMP_DIR"
rm -rf "$TMP_DIR"

echo "Done. Use --mmseqs2_database_path=$OUT_DB"
