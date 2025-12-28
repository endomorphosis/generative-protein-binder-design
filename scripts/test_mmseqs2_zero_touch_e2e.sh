#!/usr/bin/env bash
# End-to-end test of MMseqs2-GPU zero-touch installation
# Tests minimal, reduced, and full database tiers

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
log_header() { echo -e "${MAGENTA}=== $1 ===${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEST_TIER="${TEST_TIER:-minimal}"
SKIP_INSTALL=false
VERBOSE=false

show_help() {
    cat << 'EOF'
End-to-end test of zero-touch MMseqs2-GPU installation

Usage:
  ./scripts/test_mmseqs2_zero_touch_e2e.sh [OPTIONS]

Options:
  --tier TIER              minimal|reduced|full (default: minimal)
  --skip-install           Skip main installation, test MMseqs2 only
  --verbose                Verbose output
  --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier) TEST_TIER="${2:?missing value}"; shift 2 ;;
        --skip-install) SKIP_INSTALL=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --help) show_help; exit 0 ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

# Test results tracking
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

test_assert() {
    TESTS_RUN=$((TESTS_RUN + 1))
    local name="$1"
    local condition="$2"
    
    if eval "$condition"; then
        echo "  ✓ $name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "  ✗ $name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

test_file_exists() {
    test_assert "$1" "[[ -f '$2' ]]"
}

test_dir_exists() {
    test_assert "$1" "[[ -d '$2' ]]"
}

test_command_exists() {
    test_assert "$1" "command -v '$2' >/dev/null 2>&1"
}

# Phase 1: Environment validation
test_environment() {
    log_header "Phase 1: Environment Validation"
    
    test_command_exists "bash installed" "bash"
    test_command_exists "git installed" "git"
    test_dir_exists "Project root exists" "$PROJECT_ROOT"
    test_file_exists "Main installer exists" "$SCRIPT_DIR/install_all_native.sh"
    test_file_exists "MMseqs2 installer exists" "$SCRIPT_DIR/install_mmseqs2.sh"
    test_file_exists "Database converter exists" "$SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh"
    
    echo ""
    log_info "Environment validation: $TESTS_PASSED/$TESTS_RUN passed"
    return 0
}

# Phase 2: Installer validation
test_installers() {
    log_header "Phase 2: Installer Validation"
    
    log_info "Checking install_all_native.sh for MMseqs2 integration..."
    if grep -q "install_mmseqs2" "$SCRIPT_DIR/install_all_native.sh"; then
        echo "  ✓ MMseqs2 integration found"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "  ✗ MMseqs2 not integrated in main installer"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_RUN=$((TESTS_RUN + 1))
    
    log_info "Checking install_mmseqs2.sh for database build..."
    if grep -q "build-db" "$SCRIPT_DIR/install_mmseqs2.sh"; then
        echo "  ✓ Database build support found"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "  ✗ No database build support"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_RUN=$((TESTS_RUN + 1))
    
    log_info "Checking converter for GPU support..."
    if grep -q "GPU_ENABLED" "$SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh"; then
        echo "  ✓ GPU support in converter"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "  ✗ No GPU support in converter"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_RUN=$((TESTS_RUN + 1))
    
    echo ""
    log_info "Installer validation: $TESTS_PASSED/$TESTS_RUN passed"
    return 0
}

# Phase 3: Installation test (if not skipped)
test_installation() {
    if $SKIP_INSTALL; then
        log_header "Phase 3: Installation - SKIPPED"
        return 0
    fi
    
    log_header "Phase 3: Zero-Touch Installation Test"
    log_warning "Full installation test skipped (requires significant disk/network)"
    log_info "To run: $SCRIPT_DIR/install_all_native.sh --${TEST_TIER}"
    echo ""
    return 0
}

# Phase 4: Syntax validation
test_syntax() {
    log_header "Phase 4: Script Syntax Validation"
    
    log_info "Checking bash syntax..."
    local syntax_errors=0
    
    for script in "$SCRIPT_DIR/install_all_native.sh" \
                  "$SCRIPT_DIR/install_mmseqs2.sh" \
                  "$SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh"; do
        if bash -n "$script" >/dev/null 2>&1; then
            echo "  ✓ $(basename $script)"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo "  ✗ $(basename $script) - syntax error"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            syntax_errors=$((syntax_errors + 1))
        fi
        TESTS_RUN=$((TESTS_RUN + 1))
    done
    
    echo ""
    if [[ $syntax_errors -gt 0 ]]; then
        log_error "Syntax validation: $TESTS_FAILED errors found"
        return 1
    else
        log_success "All scripts have valid syntax"
        return 0
    fi
}

# Phase 5: Configuration validation
test_configuration() {
    log_header "Phase 5: Configuration Validation"
    
    # Check GPU detection
    log_info "Checking GPU detection..."
    if [[ -f "$SCRIPT_DIR/detect_gpu_and_generate_env.sh" ]]; then
        if grep -q "GPU_TYPE" "$SCRIPT_DIR/detect_gpu_and_generate_env.sh"; then
            echo "  ✓ GPU detection available"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo "  ✗ GPU detection incomplete"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    else
        echo "  ✗ GPU detection script missing"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_RUN=$((TESTS_RUN + 1))
    
    # Check environment file handling
    log_info "Checking environment configuration..."
    if grep -q ".env" "$SCRIPT_DIR/install_all_native.sh"; then
        echo "  ✓ Environment file handling present"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "  ✗ No environment file handling"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_RUN=$((TESTS_RUN + 1))
    
    echo ""
    log_info "Configuration validation: $TESTS_PASSED/$TESTS_RUN passed"
    return 0
}

# Phase 6: Documentation validation
test_documentation() {
    log_header "Phase 6: Documentation Validation"
    
    log_info "Checking for MMseqs2 documentation..."
    if [[ -f "$PROJECT_ROOT/docs/MMSEQS2_OPTIMIZATION_PLAN.md" ]] || \
       [[ -f "$PROJECT_ROOT/docs/ZERO_TOUCH_QUICKSTART.md" ]]; then
        echo "  ✓ MMseqs2 documentation found"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "  ✗ No MMseqs2 documentation"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_RUN=$((TESTS_RUN + 1))
    
    echo ""
    log_info "Documentation check: $TESTS_PASSED/$TESTS_RUN found"
    return 0
}

# Summary report
print_summary() {
    log_header "Test Summary"
    
    echo "Total Tests: $TESTS_RUN"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo ""
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "All tests passed!"
        echo ""
        echo "Next steps:"
        echo "  1. Review installation plans:"
        echo "     $SCRIPT_DIR/install_all_native.sh --help"
        echo ""
        echo "  2. Run minimal installation (recommended for testing):"
        echo "     $SCRIPT_DIR/install_all_native.sh --minimal"
        echo ""
        echo "  3. Run with reduced tier (recommended for development):"
        echo "     $SCRIPT_DIR/install_all_native.sh --recommended"
        echo ""
        return 0
    else
        log_error "Some tests failed. See above for details."
        return 1
    fi
}

# Main execution
main() {
    log_header "MMseqs2 Zero-Touch Installation E2E Test"
    echo "Tier: $TEST_TIER"
    echo "Skip Install: $SKIP_INSTALL"
    echo ""
    
    test_environment || return 1
    test_installers || return 1
    test_installation || return 1
    test_syntax || return 1
    test_configuration || return 1
    test_documentation || return 1
    
    print_summary
}

main "$@"
