#!/usr/bin/env bash
set -euo pipefail

# Build an MMseqs2 searchable database from a FASTA file.
#
# Usage:
#   ./scripts/build_mmseqs_db.sh <input_fasta> <output_db_prefix> [--threads 16]
#
# Example:
#   ./scripts/build_mmseqs_db.sh /data/uniref90.fasta /data/mmseqs/uniref90_db --threads 32

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/build_mmseqs_db.sh <input_fasta> <output_db_prefix> [--threads 16]

Notes:
- Requires `mmseqs` on PATH.
- This creates a DB and an index (recommended for faster searches).
- The output is a DB *prefix* (MMseqs2 writes multiple files with that prefix).
EOF
  exit 2
fi

INPUT_FASTA="$1"
OUT_DB="$2"
shift 2

THREADS=16
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      THREADS="${2:?missing value}"
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

TMP_DIR="${TMPDIR:-/tmp}/mmseqs_db_build_$$"
mkdir -p "$TMP_DIR"

echo "Building MMseqs2 DB: $OUT_DB"
mmseqs createdb "$INPUT_FASTA" "$OUT_DB"

echo "Creating MMseqs2 index (threads=$THREADS)"
mmseqs createindex "$OUT_DB" "$TMP_DIR" --threads "$THREADS"

echo "Done. Use --mmseqs2_database_path=$OUT_DB"
