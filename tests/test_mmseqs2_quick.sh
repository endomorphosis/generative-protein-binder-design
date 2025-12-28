#!/usr/bin/env bash
# Quick MMseqs2 integration test
# Tests just the MSA generation step with MMseqs2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[TEST] MMseqs2 MSA Generation Test"
echo "===================================="
echo ""

# Setup
eval "$(conda shell.bash hook)"
conda activate alphafold2

export ALPHAFOLD_DIR="${ALPHAFOLD_DIR:-$ROOT_DIR/tools/alphafold2}"
export ALPHAFOLD_DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
export PYTHONPATH="$ALPHAFOLD_DIR:${PYTHONPATH:-}"

MMSEQS_DB="$ALPHAFOLD_DATA_DIR/mmseqs2/uniref90_db"
TEST_SEQ="MNIFEMLRID"
TEST_OUT="/tmp/mmseqs2_quick_test_$$"

mkdir -p "$TEST_OUT"

# Test that we can call MMseqs2 directly
echo "[1/3] Testing mmseqs2 binary..."
if ! mmseqs version; then
  echo "ERROR: mmseqs not found"
  exit 1
fi
echo "OK"
echo ""

# Test that we can import the module
echo "[2/3] Testing MMseqs2 Python wrapper..."
python - <<PY
import sys
from alphafold.data.tools import mmseqs2

print("Creating MMseqs2 runner...")
runner = mmseqs2.MMseqs2(
    binary_path='$(which mmseqs)',
    database_path='$MMSEQS_DB',
    max_seqs=256
)

print("Testing query...")
fasta_path = '$TEST_OUT/test.fasta'
with open(fasta_path, 'w') as f:
    f.write('>test\\n$TEST_SEQ\\n')

try:
    result = runner.query(fasta_path)
    if result and len(result) > 0:
        a3m_content = result[0].get('a3m', '')
        print(f"SUCCESS: Got A3M with {len(a3m_content)} bytes")
        if a3m_content:
            print(f"First 200 chars: {a3m_content[:200]}")
    else:
        print("WARNING: Got empty result")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PY

if [[ $? -eq 0 ]]; then
  echo "OK"
else
  echo "FAILED"
  exit 1
fi
echo ""

# Test the full pipeline integration
echo "[3/3] Testing pipeline integration..."
python - <<PY
from alphafold.data import pipeline, parsers
from alphafold.data.tools import mmseqs2
import os

mmseqs_db = '$MMSEQS_DB'
binary_path = '$(which mmseqs)'

print(f"MMseqs2 DB: {mmseqs_db}")
print(f"Binary: {binary_path}")

# Create a minimal pipeline with MMseqs2
print("Creating monomer data pipeline with msa_mode=mmseqs2...")
try:
    data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path='/usr/bin/jackhmmer',  # fallback, not used in mmseqs2 mode
        hhblits_binary_path='/usr/bin/hhblits',      # fallback
        hhsearch_binary_path='/usr/bin/hhsearch',    # fallback  
        uniref90_database_path='$ALPHAFOLD_DATA_DIR/uniref90/uniref90.fasta',
        mgnify_database_path='$ALPHAFOLD_DATA_DIR/mgnify/mgy_clusters_2022_05.fa',
        bfd_database_path=None,
        small_bfd_database_path='$ALPHAFOLD_DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta',
        uniref30_database_path=None,
        pdb70_database_path='$ALPHAFOLD_DATA_DIR/pdb70/pdb70',
        template_featurizer=None,  # Skip templates for this test
        use_small_bfd=True,
        msa_mode='mmseqs2',
        mmseqs2_binary_path=binary_path,
        mmseqs2_database_path=mmseqs_db,
        mmseqs2_max_seqs=256
    )
    print("Pipeline created successfully with MMseqs2 mode!")
    print(f"MSA mode: {data_pipeline.msa_mode if hasattr(data_pipeline, 'msa_mode') else 'unknown'}")
except Exception as e:
    print(f"ERROR creating pipeline: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PY

if [[ $? -eq 0 ]]; then
  echo "OK"
else
  echo "FAILED"
  exit 1
fi
echo ""

echo "===================================="
echo "[SUCCESS] All MMseqs2 tests passed!"
echo "===================================="
echo ""
echo "MMseqs2 is ready for end-to-end AlphaFold testing."
echo "The built DB at: $MMSEQS_DB"
echo ""

rm -rf "$TEST_OUT"
