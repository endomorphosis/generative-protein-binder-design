#!/usr/bin/env bash
# Wait for MMseqs2 DB build to complete, then run benchmarks
set -euo pipefail

LOG_FILE="/tmp/mmseqs2_rebuild.log"
REBUILD_PID=$(pgrep -f "rebuild_optimized_mmseqs2_db.sh" || echo "")

if [[ -z "$REBUILD_PID" ]]; then
    echo "No rebuild process found. Checking if DB already exists..."
    if [[ -f ~/.cache/alphafold/mmseqs2/uniref90_large_db ]]; then
        echo "DB exists! Proceeding to benchmarks."
    else
        echo "ERROR: No rebuild running and no DB found. Please run rebuild script first."
        exit 1
    fi
else
    echo "Waiting for MMseqs2 DB rebuild (PID: $REBUILD_PID) to complete..."
    echo "Monitoring log: $LOG_FILE"
    echo ""
    
    # Wait for process to complete
    while kill -0 $REBUILD_PID 2>/dev/null; do
        # Show last line of progress
        if [[ -f "$LOG_FILE" ]]; then
            tail -1 "$LOG_FILE" | grep -E "(\[.*\]|eta|SUCCESS|ERROR)" || true
        fi
        sleep 5
    done
    
    echo ""
    echo "Rebuild process completed!"
    echo ""
fi

# Check if rebuild was successful
if tail -20 "$LOG_FILE" 2>/dev/null | grep -q "MMseqs2 DB Rebuild Complete"; then
    echo "✓ DB rebuild successful!"
    echo ""
    
    # Show results
    tail -30 "$LOG_FILE" | grep -A 20 "MMseqs2 DB Rebuild Complete"
    echo ""
    
    # Run benchmarks
    echo "Starting performance benchmarks..."
    echo ""
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    ./scripts/bench_msa_comparison.sh
    
else
    echo "✗ DB rebuild may have failed. Check log:"
    echo "  tail -50 $LOG_FILE"
    exit 1
fi
