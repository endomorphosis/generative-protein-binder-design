#!/usr/bin/env bash
# Rebuild MMseqs2 DB with optimized size for available memory
#
# Strategy:
# 1. Remove existing oversized DB
# 2. Create subset of UniRef90 FASTA (target: use 30-40% of available RAM)
# 3. Build new MMseqs2 DB with optimized parameters
# 4. Benchmark single search to verify performance

set -euo pipefail

BENCH_DIR="${1:-/tmp/mmseqs2_rebuild_$(date +%Y%m%d_%H%M%S)}"
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

DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
MMSEQS2_DIR="$DATA_DIR/mmseqs2"
SOURCE_FASTA="$DATA_DIR/uniref90/uniref90.fasta"

if [[ ! -f "$SOURCE_FASTA" ]]; then
    log_error "UniRef90 FASTA not found: $SOURCE_FASTA"
    exit 1
fi

# Calculate optimal DB size
AVAIL_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
TARGET_DB_GB=$((AVAIL_MEM_GB * 30 / 100))  # Use 30% of available memory

log_info "Available memory: ${AVAIL_MEM_GB}GB"
log_info "Target DB size: ~${TARGET_DB_GB}GB"

# Estimate sequences needed (rough: 100-120 bytes per sequence in DB)
TARGET_SEQS=$((TARGET_DB_GB * 1024 * 1024 * 1024 / 120))
log_info "Target sequences: ~$TARGET_SEQS"
echo ""

# Round to nice numbers for common use cases
if (( TARGET_SEQS < 100000 )); then
    NUM_SEQS=50000
    LABEL="tiny"
elif (( TARGET_SEQS < 1000000 )); then
    NUM_SEQS=500000
    LABEL="small"
elif (( TARGET_SEQS < 10000000 )); then
    NUM_SEQS=2000000
    LABEL="medium"
else
    NUM_SEQS=5000000
    LABEL="large"
fi

log_info "Building '$LABEL' DB with $NUM_SEQS sequences"
echo ""

# Backup existing DB
if [[ -d "$MMSEQS2_DIR" ]]; then
    BACKUP_DIR="${MMSEQS2_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    log_warning "Backing up existing DB to: $BACKUP_DIR"
    mv "$MMSEQS2_DIR" "$BACKUP_DIR"
fi

mkdir -p "$MMSEQS2_DIR"

# Create subset FASTA
SUBSET_FASTA="$MMSEQS2_DIR/uniref90_${LABEL}_${NUM_SEQS}.fasta"
log_info "Creating FASTA subset: $SUBSET_FASTA"
log_info "Extracting first $NUM_SEQS sequences from $SOURCE_FASTA..."

START=$(date +%s)
export SOURCE_FASTA
export SUBSET_FASTA
export NUM_SEQS
python3 - <<'PY'
import sys
import os

source = os.environ['SOURCE_FASTA']
dest = os.environ['SUBSET_FASTA']
target_seqs = int(os.environ['NUM_SEQS'])

written = 0
with open(source, 'r', encoding='utf-8', errors='ignore') as fin:
    with open(dest, 'w', encoding='utf-8') as fout:
        current_seq = []
        for line in fin:
            if line.startswith('>'):
                if current_seq:
                    # Write previous sequence
                    fout.write(''.join(current_seq))
                    current_seq = []
                    written += 1
                    if written >= target_seqs:
                        break
                    if written % 10000 == 0:
                        print(f"Extracted {written} sequences...", file=sys.stderr, flush=True)
                current_seq = [line]
            else:
                current_seq.append(line)
        
        # Write last sequence
        if current_seq and written < target_seqs:
            fout.write(''.join(current_seq))
            written += 1

print(f"Wrote {written} sequences to {dest}", file=sys.stderr)
PY

END=$(date +%s)
log_success "FASTA subset created in $((END - START))s"

SUBSET_SIZE=$(du -sh "$SUBSET_FASTA" | awk '{print $1}')
log_info "Subset FASTA size: $SUBSET_SIZE"
echo ""

# Activate conda environment
log_info "Activating alphafold2 conda environment..."
eval "$(conda shell.bash hook)"
conda activate alphafold2

# Verify mmseqs is available
if ! command -v mmseqs &> /dev/null; then
    log_error "mmseqs not found in alphafold2 environment"
    exit 1
fi
log_success "mmseqs $(mmseqs version 2>&1 | head -1) found"
echo ""

# Build MMseqs2 DB
DB_PREFIX="$MMSEQS2_DIR/uniref90_${LABEL}_db"
log_info "Building MMseqs2 DB: $DB_PREFIX"

# Use data dir for temp to avoid filling /tmp
TMP_DIR="$MMSEQS2_DIR/tmp_build"
mkdir -p "$TMP_DIR"

THREADS=$(nproc)
if (( THREADS > 32 )); then
    THREADS=32
fi

START=$(date +%s)
bash "$ROOT_DIR/scripts/build_mmseqs_db.sh" \
    "$SUBSET_FASTA" \
    "$DB_PREFIX" \
    --threads "$THREADS" \
    --tmp-dir "$TMP_DIR"
END=$(date +%s)

rm -rf "$TMP_DIR"

log_success "MMseqs2 DB built in $((END - START))s"
echo ""

# Verify DB
DB_SIZE=$(du -sh "$MMSEQS2_DIR" | awk '{print $1}')
log_success "Final DB size: $DB_SIZE"

# Count sequences in DB
NUM_DB_SEQS=$(mmseqs linsearch "$DB_PREFIX" --threads 1 2>&1 | grep -oP '(?<=Database: )\d+' | head -1 || echo "$NUM_SEQS")
log_info "Sequences in DB: $NUM_DB_SEQS"
echo ""

# Benchmark search
log_info "Benchmarking single MMseqs2 search..."
TEST_FASTA="$BENCH_DIR/test_query.fasta"
cat > "$TEST_FASTA" <<'EOF'
>test_query
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV
EOF

START=$(date +%s)
TMP_SEARCH="$BENCH_DIR/mmseqs_search_test"
mkdir -p "$TMP_SEARCH"

mmseqs createdb "$TEST_FASTA" "$TMP_SEARCH/querydb"
mmseqs search \
    "$TMP_SEARCH/querydb" \
    "$DB_PREFIX" \
    "$TMP_SEARCH/resultdb" \
    "$TMP_SEARCH/tmp" \
    --threads "$THREADS" \
    --max-seqs 512 \
    2>&1 | tee "$BENCH_DIR/search.log"

END=$(date +%s)
SEARCH_TIME=$((END - START))

rm -rf "$TMP_SEARCH"

log_success "Search completed in ${SEARCH_TIME}s"
echo ""

# Save results
{
    echo "=== MMseqs2 DB Rebuild Results ==="
    echo "Date: $(date)"
    echo "Label: $LABEL"
    echo "Sequences: $NUM_DB_SEQS"
    echo "DB size: $DB_SIZE"
    echo "DB path: $DB_PREFIX"
    echo "Search time: ${SEARCH_TIME}s"
    echo "Build time: $((END - START))s"
} > "$BENCH_DIR/rebuild_results.txt"

cat "$BENCH_DIR/rebuild_results.txt"
echo ""

# Update environment
log_info "Updating AlphaFold .env files..."
for env_file in "$ROOT_DIR/tools/generated/alphafold2/.env" "$ROOT_DIR/tools/alphafold2/.env"; do
    if [[ -f "$env_file" ]]; then
        # Update or add ALPHAFOLD_MMSEQS2_DATABASE_PATH
        if grep -q "ALPHAFOLD_MMSEQS2_DATABASE_PATH=" "$env_file"; then
            sed -i "s|ALPHAFOLD_MMSEQS2_DATABASE_PATH=.*|ALPHAFOLD_MMSEQS2_DATABASE_PATH=$DB_PREFIX|" "$env_file"
        else
            echo "ALPHAFOLD_MMSEQS2_DATABASE_PATH=$DB_PREFIX" >> "$env_file"
        fi
        log_info "Updated: $env_file"
    fi
done

echo ""
log_success "╔═══════════════════════════════════════════════════════════╗"
log_success "║  MMseqs2 DB Rebuild Complete!                            ║"
log_success "╚═══════════════════════════════════════════════════════════╝"
echo ""
log_info "New DB: $DB_PREFIX"
log_info "Size: $DB_SIZE ($NUM_DB_SEQS sequences)"
log_info "Search benchmark: ${SEARCH_TIME}s"
echo ""
log_info "To use this DB, set:"
echo "  export ALPHAFOLD_MMSEQS2_DATABASE_PATH=\"$DB_PREFIX\""
echo ""
log_info "Results saved to: $BENCH_DIR"
