#!/bin/bash
# Test database conversions for all tiers
# Minimal, Reduced, and Full tier testing

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${MAGENTA}[STEP]${NC} $1"; }
log_header() { echo -e "${MAGENTA}════════════════════════════════════════${NC}"; echo -e "${MAGENTA}$1${NC}"; echo -e "${MAGENTA}════════════════════════════════════════${NC}"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

MINIMAL_ONLY=false
REDUCED_ONLY=false
SKIP_MINIMAL=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --minimal-only) MINIMAL_ONLY=true; shift ;;
        --reduced-only) REDUCED_ONLY=true; shift ;;
        --skip-minimal) SKIP_MINIMAL=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) shift ;;
    esac
done

is_tier_complete() {
    local tier="$1"
    local output_dir="$HOME/.cache/alphafold/mmseqs2_test_${tier}"
    
    # Check if directory exists
    if [[ ! -d "$output_dir" ]]; then
        return 1
    fi
    
    # Check if any .dbtype files exist (indicates successful conversion)
    local db_count=0
    for db_file in "$output_dir"/*_db.dbtype; do
        if [[ -f "$db_file" ]]; then
            db_count=$((db_count + 1))
        fi
    done
    
    # Return success if at least one database was created
    [[ $db_count -gt 0 ]]
}

test_tier() {
    local tier="$1"
    local start_time=$(date +%s)
    
    log_header "Testing: $tier tier"
    
    # Calculate expected sizes
    local expected_size=""
    local expected_time=""
    case "$tier" in
        minimal)
            expected_size="~30GB"
            expected_time="15-20 minutes"
            ;;
        reduced)
            expected_size="~50GB"
            expected_time="40-60 minutes"
            ;;
        full)
            expected_size="~200GB+"
            expected_time="3-5 hours"
            ;;
    esac
    
    echo "Expected Size: $expected_size"
    echo "Expected Time: $expected_time"
    echo ""
    
    # Check output directory
    OUTPUT_DIR="$HOME/.cache/alphafold/mmseqs2_test_${tier}"
    mkdir -p "$OUTPUT_DIR"
    
    log_step "Running database conversion..."
    
    if $DRY_RUN; then
        log_info "DRY RUN: Would execute:"
        echo "  $SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh \\"
        echo "    --tier $tier \\"
        echo "    --output-dir $OUTPUT_DIR \\"
        echo "    --gpu"
        return 0
    fi
    
    # Run conversion
    if bash "$SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh" \
        --tier "$tier" \
        --output-dir "$OUTPUT_DIR" \
        --gpu; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local mins=$((duration / 60))
        local secs=$((duration % 60))
        
        log_success "$tier conversion completed in ${mins}m ${secs}s"
        
        # Verify databases were created
        log_step "Verifying databases..."
        
        local db_count=0
        for db_file in "$OUTPUT_DIR"/*_db.dbtype; do
            if [[ -f "$db_file" ]]; then
                local db_name=$(basename "$db_file" .dbtype)
                local db_size=$(du -sh "$OUTPUT_DIR/${db_name}"* 2>/dev/null | awk '{print $1}' | head -1)
                echo "  ✓ ${db_name}: $db_size"
                db_count=$((db_count + 1))
            fi
        done
        
        if [[ $db_count -gt 0 ]]; then
            log_success "$db_count database(s) created successfully"
            return 0
        else
            log_error "No databases created for $tier tier"
            return 1
        fi
    else
        log_error "$tier conversion failed"
        return 1
    fi
}

main() {
    log_header "MMseqs2 Database Conversion Testing"
    echo "Project: $PROJECT_ROOT"
    echo "Script: $SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh"
    echo ""
    
    # Check if converter exists
    if [[ ! -f "$SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh" ]]; then
        log_error "Converter script not found: $SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh"
        exit 1
    fi
    
    # Check if mmseqs is available
    if ! command -v mmseqs >/dev/null 2>&1; then
        log_error "mmseqs not found. Install MMseqs2 first."
        exit 1
    fi
    
    log_success "MMseqs2 found: $(mmseqs --version 2>/dev/null | head -1)"
    echo ""
    
    # Run tests
    RESULTS=()
    
    if $MINIMAL_ONLY; then
        if test_tier "minimal"; then
            RESULTS+=("minimal: ✓ PASSED")
        else
            RESULTS+=("minimal: ✗ FAILED")
        fi
    elif $REDUCED_ONLY; then
        if test_tier "reduced"; then
            RESULTS+=("reduced: ✓ PASSED")
        else
            RESULTS+=("reduced: ✗ FAILED")
        fi
    else
        # Test all tiers
        echo "Running tests for all database tiers..."
        echo ""
        
        # Check if minimal tier is already complete
        if is_tier_complete "minimal"; then
            log_info "Minimal tier already complete, skipping..."
            RESULTS+=("minimal: ⏭️ SKIPPED (already complete)")
        else
            if test_tier "minimal"; then
                RESULTS+=("minimal: ✓ PASSED")
            else
                RESULTS+=("minimal: ✗ FAILED")
            fi
        fi
        echo ""
        
        if test_tier "reduced"; then
            RESULTS+=("reduced: ✓ PASSED")
        else
            RESULTS+=("reduced: ✗ FAILED")
        fi
        echo ""
        
        log_header "Full Tier (Long Running)"
        echo "Note: Full tier conversion takes 3-5 hours"
        read -p "Run full tier conversion? (y/N) " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if test_tier "full"; then
                RESULTS+=("full: ✓ PASSED")
            else
                RESULTS+=("full: ✗ FAILED")
            fi
        else
            log_info "Skipping full tier"
            RESULTS+=("full: ⏭️ SKIPPED")
        fi
    fi
    
    # Print summary
    echo ""
    log_header "Test Results Summary"
    for result in "${RESULTS[@]}"; do
        echo "  $result"
    done
    echo ""
    
    # Check for failures
    local failed=0
    for result in "${RESULTS[@]}"; do
        if [[ $result == *"FAILED"* ]]; then
            failed=$((failed + 1))
        fi
    done
    
    if [[ $failed -eq 0 ]]; then
        log_success "All tests passed!"
        echo ""
        echo "Databases created at:"
        echo "  ~/.cache/alphafold/mmseqs2_test_minimal/"
        echo "  ~/.cache/alphafold/mmseqs2_test_reduced/"
        if [[ " ${RESULTS[@]} " =~ "full" ]]; then
            echo "  ~/.cache/alphafold/mmseqs2_test_full/"
        fi
    else
        log_error "$failed test(s) failed"
        exit 1
    fi
}

main
