#!/bin/bash
# Test MMseqs2 integration in zero-touch installer

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${MAGENTA}[STEP]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log_step "Verifying MMseqs2 installer integration..."
echo ""

# Check that conversion script exists
if [[ ! -f "$SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh" ]]; then
    log_error "Conversion script not found: $SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh"
    exit 1
fi
log_success "✓ Conversion script exists"

# Check that install_mmseqs2.sh exists
if [[ ! -f "$SCRIPT_DIR/install_mmseqs2.sh" ]]; then
    log_error "Install script not found: $SCRIPT_DIR/install_mmseqs2.sh"
    exit 1
fi
log_success "✓ Install script exists"

# Check that main installer exists
if [[ ! -f "$SCRIPT_DIR/install_all_native.sh" ]]; then
    log_error "Main installer not found: $SCRIPT_DIR/install_all_native.sh"
    exit 1
fi
log_success "✓ Main installer exists"

# Verify the conversion script is integrated into main installer
if grep -q "convert_alphafold_db_to_mmseqs2_multistage.sh" "$SCRIPT_DIR/install_all_native.sh"; then
    log_success "✓ Conversion script integrated into main installer"
else
    log_error "Conversion script NOT integrated into main installer"
    exit 1
fi

# Verify output path is correct in installer
if grep -q "\$HOME/.cache/alphafold/mmseqs2" "$SCRIPT_DIR/install_all_native.sh"; then
    log_success "✓ Output path set to ~/.cache/alphafold/mmseqs2"
else
    log_error "Output path not correctly set in installer"
    exit 1
fi

# Verify conversion script has correct defaults
if grep -q "OUTPUT_DIR=.*DATA_DIR/mmseqs2" "$SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh"; then
    log_success "✓ Conversion script defaults to ~/.cache/alphafold/mmseqs2"
else
    log_error "Conversion script default output not correct"
    exit 1
fi

echo ""
log_step "Testing conversion script with --help..."
if bash "$SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh" --help > /dev/null 2>&1; then
    log_success "✓ Conversion script is functional"
else
    log_error "Conversion script has errors"
    exit 1
fi

echo ""
log_step "Checking for MMseqs2 in environment..."
if command -v mmseqs >/dev/null 2>&1; then
    MMSEQS_VERSION=$(mmseqs --version 2>&1 | head -1 || echo "unknown")
    log_success "✓ MMseqs2 available: $MMSEQS_VERSION"
else
    log_error "MMseqs2 not found in PATH"
    exit 1
fi

echo ""
log_step "Verifying AlphaFold source databases..."
ALPHAFOLD_DATA="$HOME/.cache/alphafold"

# Check minimal requirements
if [[ -f "$ALPHAFOLD_DATA/uniref90/uniref90.fasta" ]]; then
    SIZE=$(ls -lh "$ALPHAFOLD_DATA/uniref90/uniref90.fasta" | awk '{print $5}')
    log_success "✓ UniRef90: $SIZE"
else
    log_error "UniRef90 database not found"
    exit 1
fi

if [[ -d "$ALPHAFOLD_DATA/small_bfd" ]]; then
    SIZE=$(du -sh "$ALPHAFOLD_DATA/small_bfd" | awk '{print $1}')
    log_success "✓ Small BFD: $SIZE"
else
    log_error "Small BFD database not found"
    exit 1
fi

echo ""
log_step "Checking current MMseqs2 output directory..."
MMSEQS2_OUTPUT="$ALPHAFOLD_DATA/mmseqs2"
if [[ -d "$MMSEQS2_OUTPUT" ]]; then
    DB_COUNT=$(find "$MMSEQS2_OUTPUT" -name "*_db.dbtype" 2>/dev/null | wc -l)
    if [[ $DB_COUNT -gt 0 ]]; then
        log_success "✓ MMseqs2 output directory exists with $DB_COUNT database(s)"
        du -sh "$MMSEQS2_OUTPUT" 2>/dev/null | awk '{print "  Size: " $1}'
    else
        log_info "  MMseqs2 output directory exists (empty, will be populated during install)"
    fi
else
    log_info "  MMseqs2 output directory will be created during install"
fi

echo ""
echo "=========================================="
log_success "All integration checks passed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  • Conversion script: $SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh"
echo "  • Install script: $SCRIPT_DIR/install_mmseqs2.sh"
echo "  • Main installer: $SCRIPT_DIR/install_all_native.sh"
echo "  • Output path: $MMSEQS2_OUTPUT"
echo ""
echo "The MMseqs2 database will be built automatically during:"
echo "  bash scripts/install_all_native.sh --minimal"
echo "  bash scripts/install_all_native.sh --recommended"
echo "  bash scripts/install_all_native.sh --full"
echo ""
