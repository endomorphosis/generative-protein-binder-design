#!/bin/bash
# End-to-End Validation Script
# Tests complete installation and workflow

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_test() { echo -e "${CYAN}[TEST]${NC} $1"; }

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools"

# Test results
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    log_test "$test_name"
    
    if eval "$test_command" > /dev/null 2>&1; then
        log_success "  PASSED"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        log_error "  FAILED"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Skip a test
skip_test() {
    local test_name="$1"
    local reason="$2"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    log_test "$test_name"
    log_warning "  SKIPPED: $reason"
}

# Print banner
clear
echo "=================================================================="
echo "  Native Installation Validation"
echo "=================================================================="
echo ""
log_info "Project: $PROJECT_ROOT"
log_info "Platform: $(uname -s) $(uname -m)"
echo ""

# Test 1: Check Conda
echo "=================================================================="
echo "  System Tests"
echo "=================================================================="
echo ""

run_test "Conda installed" "command -v conda"
run_test "Mamba installed" "command -v mamba"
run_test "Git installed" "command -v git"
run_test "Wget or curl installed" "command -v wget || command -v curl"

echo ""

# Test 2: Check AlphaFold2
echo "=================================================================="
echo "  AlphaFold2 Tests"
echo "=================================================================="
echo ""

AF_CODE_DIR="$TOOLS_DIR/alphafold2"
AF_WRAPPER_DIR="$TOOLS_DIR/generated/alphafold2"
AF_WRAPPER_DIR_LEGACY="$TOOLS_DIR/alphafold2"
AF_ENV_FILE="$AF_WRAPPER_DIR/.env"
AF_ENV_FILE_LEGACY="$AF_WRAPPER_DIR_LEGACY/.env"

pick_af_env_file() {
    if [ -f "$AF_ENV_FILE" ]; then
        echo "$AF_ENV_FILE"
    elif [ -f "$AF_ENV_FILE_LEGACY" ]; then
        echo "$AF_ENV_FILE_LEGACY"
    else
        echo ""
    fi
}

AF_ENV_SELECTED="$(pick_af_env_file)"
AF_WRAPPER_SELECTED="$AF_WRAPPER_DIR"
if [ "$AF_ENV_SELECTED" = "$AF_ENV_FILE_LEGACY" ]; then
    AF_WRAPPER_SELECTED="$AF_WRAPPER_DIR_LEGACY"
fi

if [ -d "$AF_CODE_DIR" ] || [ -d "$AF_WRAPPER_DIR" ] || [ -d "$AF_WRAPPER_DIR_LEGACY" ]; then
    run_test "AlphaFold2 code directory exists" "[ -d '$AF_CODE_DIR' ]"
    run_test "AlphaFold2 activation script exists" "[ -f '$AF_WRAPPER_SELECTED/activate.sh' ]"
    run_test "AlphaFold2 validation script exists" "[ -f '$AF_WRAPPER_SELECTED/validate.py' ]"
    run_test "AlphaFold2 env file exists" "[ -n '$AF_ENV_SELECTED' ] && [ -f '$AF_ENV_SELECTED' ]"
    
    # Check conda environment
    if eval "$(conda shell.bash hook)" 2>/dev/null; then
        if conda env list | grep -q "alphafold2"; then
            log_test "AlphaFold2 conda environment"
            if conda activate alphafold2 2>/dev/null && python "$AF_WRAPPER_SELECTED/validate.py"; then
                log_success "  PASSED"
                TESTS_PASSED=$((TESTS_PASSED + 1))
            else
                log_error "  FAILED"
                TESTS_FAILED=$((TESTS_FAILED + 1))
            fi
            TESTS_RUN=$((TESTS_RUN + 1))
            conda deactivate 2>/dev/null || true
        else
            skip_test "AlphaFold2 conda environment" "Environment not found"
        fi
    else
        skip_test "AlphaFold2 conda environment" "Conda not initialized"
    fi
    
    # Check data directory
    if [ -n "$AF_ENV_SELECTED" ] && [ -f "$AF_ENV_SELECTED" ]; then
        DATA_DIR=$(grep ALPHAFOLD_DATA_DIR "$AF_ENV_SELECTED" | cut -d'=' -f2)
        run_test "AlphaFold2 data directory exists" "[ -d '$DATA_DIR' ]"
        run_test "AlphaFold2 parameters exist" "[ -d '$DATA_DIR/params' ] && [ $(ls -1 '$DATA_DIR/params'/*.npz 2>/dev/null | wc -l) -gt 0 ]"
        # Biologist-proof DB check (UniRef30/PDB70 are HH-suite prefix DBs, not single files).
        run_test "AlphaFold2 databases (reduced) present" "'$PROJECT_ROOT/scripts/check_alphafold_data_dir.sh' '$DATA_DIR'"
    else
        skip_test "AlphaFold2 data directory" "No .env file"
        skip_test "AlphaFold2 parameters" "No .env file"
        skip_test "AlphaFold2 databases (reduced) present" "No .env file"
    fi
else
    skip_test "AlphaFold2 installation" "Not installed"
    skip_test "AlphaFold2 activation script" "Not installed"
    skip_test "AlphaFold2 validation script" "Not installed"
    skip_test "AlphaFold2 env file" "Not installed"
    skip_test "AlphaFold2 conda environment" "Not installed"
    skip_test "AlphaFold2 data directory" "Not installed"
    skip_test "AlphaFold2 parameters" "Not installed"
fi

echo ""

# Test 3: Check RFDiffusion
echo "=================================================================="
echo "  RFDiffusion Tests"
echo "=================================================================="
echo ""

RF_CODE_DIR="$TOOLS_DIR/rfdiffusion/RFdiffusion"
RF_WRAPPER_DIR="$TOOLS_DIR/generated/rfdiffusion"
RF_WRAPPER_DIR_LEGACY="$TOOLS_DIR/rfdiffusion"
RF_ENV_FILE="$RF_WRAPPER_DIR/.env"
RF_ENV_FILE_LEGACY="$RF_WRAPPER_DIR_LEGACY/.env"
RF_ENV_FILE_LEGACY2="$TOOLS_DIR/rfdiffusion/RFdiffusion/.env"

pick_rf_env_file() {
    if [ -f "$RF_ENV_FILE" ]; then
        echo "$RF_ENV_FILE"
    elif [ -f "$RF_ENV_FILE_LEGACY" ]; then
        echo "$RF_ENV_FILE_LEGACY"
    elif [ -f "$RF_ENV_FILE_LEGACY2" ]; then
        echo "$RF_ENV_FILE_LEGACY2"
    else
        echo ""
    fi
}

RF_ENV_SELECTED="$(pick_rf_env_file)"
RF_WRAPPER_SELECTED="$RF_WRAPPER_DIR"
if [ "$RF_ENV_SELECTED" = "$RF_ENV_FILE_LEGACY" ] || [ "$RF_ENV_SELECTED" = "$RF_ENV_FILE_LEGACY2" ]; then
    RF_WRAPPER_SELECTED="$RF_WRAPPER_DIR_LEGACY"
fi

if [ -d "$TOOLS_DIR/rfdiffusion" ]; then
    run_test "RFDiffusion directory exists" "[ -d '$TOOLS_DIR/rfdiffusion/RFdiffusion' ]"
    run_test "RFDiffusion activation script exists" "[ -f '$RF_WRAPPER_SELECTED/activate.sh' ]"
    run_test "RFDiffusion validation script exists" "[ -f '$RF_WRAPPER_SELECTED/validate.py' ]"
    run_test "RFDiffusion env file exists" "[ -n '$RF_ENV_SELECTED' ] && [ -f '$RF_ENV_SELECTED' ]"
    
    # Check conda environment
    if eval "$(conda shell.bash hook)" 2>/dev/null; then
        if conda env list | grep -q "rfdiffusion"; then
            log_test "RFDiffusion conda environment"
            if conda activate rfdiffusion 2>/dev/null && python "$RF_WRAPPER_SELECTED/validate.py"; then
                log_success "  PASSED"
                TESTS_PASSED=$((TESTS_PASSED + 1))
            else
                log_error "  FAILED"
                TESTS_FAILED=$((TESTS_FAILED + 1))
            fi
            TESTS_RUN=$((TESTS_RUN + 1))
            conda deactivate 2>/dev/null || true
        else
            skip_test "RFDiffusion conda environment" "Environment not found"
        fi
    else
        skip_test "RFDiffusion conda environment" "Conda not initialized"
    fi
    
    # Check models directory
    if [ -n "$RF_ENV_SELECTED" ] && [ -f "$RF_ENV_SELECTED" ]; then
        MODELS_DIR=$(grep RFDIFFUSION_MODELS "$RF_ENV_SELECTED" | cut -d'=' -f2)
        run_test "RFDiffusion models directory exists" "[ -d '$MODELS_DIR' ]"
        run_test "RFDiffusion models exist" "[ $(ls -1 '$MODELS_DIR'/*.pt 2>/dev/null | wc -l) -gt 0 ]"
    else
        skip_test "RFDiffusion models directory" "No .env file"
        skip_test "RFDiffusion models" "No .env file"
    fi
else
    skip_test "RFDiffusion installation" "Not installed"
    skip_test "RFDiffusion activation script" "Not installed"
    skip_test "RFDiffusion validation script" "Not installed"
    skip_test "RFDiffusion env file" "Not installed"
    skip_test "RFDiffusion conda environment" "Not installed"
    skip_test "RFDiffusion models directory" "Not installed"
    skip_test "RFDiffusion models" "Not installed"
fi

echo ""

# Test 4: Check ProteinMPNN
echo "=================================================================="
echo "  ProteinMPNN Tests"
echo "=================================================================="
echo ""

if [ -d "$TOOLS_DIR/proteinmpnn" ]; then
    run_test "ProteinMPNN directory exists" "[ -d '$TOOLS_DIR/proteinmpnn/ProteinMPNN' ]"
    run_test "ProteinMPNN run script exists" "[ -f '$TOOLS_DIR/proteinmpnn/run_proteinmpnn_arm64.sh' ]"
    
    # Check conda environment
    if eval "$(conda shell.bash hook)" 2>/dev/null; then
        if conda env list | grep -q "proteinmpnn_arm64"; then
            log_test "ProteinMPNN conda environment"
            if conda activate proteinmpnn_arm64 2>/dev/null && python "$TOOLS_DIR/proteinmpnn/test_proteinmpnn.py"; then
                log_success "  PASSED"
                TESTS_PASSED=$((TESTS_PASSED + 1))
            else
                log_error "  FAILED"
                TESTS_FAILED=$((TESTS_FAILED + 1))
            fi
            TESTS_RUN=$((TESTS_RUN + 1))
            conda deactivate 2>/dev/null || true
        else
            skip_test "ProteinMPNN conda environment" "Environment not found"
        fi
    else
        skip_test "ProteinMPNN conda environment" "Conda not initialized"
    fi
else
    skip_test "ProteinMPNN installation" "Not installed"
    skip_test "ProteinMPNN run script" "Not installed"
    skip_test "ProteinMPNN conda environment" "Not installed"
fi

echo ""

# Test 5: Check MCP Server Configuration
echo "=================================================================="
echo "  MCP Server Tests"
echo "=================================================================="
echo ""

if [ -f "$PROJECT_ROOT/mcp-server/.env.native" ]; then
    run_test "MCP native configuration exists" "[ -f '$PROJECT_ROOT/mcp-server/.env.native' ]"
    run_test "MCP native configuration has MODEL_BACKEND" "grep -q 'MODEL_BACKEND=native' '$PROJECT_ROOT/mcp-server/.env.native'"
    log_info "  MCP Server can be started with: cd mcp-server && MODEL_BACKEND=native python server.py"
else
    skip_test "MCP native configuration" "Not configured"
    skip_test "MCP MODEL_BACKEND setting" "Not configured"
fi

if [ -f "$PROJECT_ROOT/activate_native.sh" ]; then
    run_test "Native activation script exists" "[ -f '$PROJECT_ROOT/activate_native.sh' ]"
else
    skip_test "Native activation script" "Not created"
fi

echo ""

# Test 6: Check Native Services
echo "=================================================================="
echo "  Native Services Tests"
echo "=================================================================="
echo ""

run_test "Native services directory exists" "[ -d '$PROJECT_ROOT/native_services' ]"
run_test "AlphaFold native service exists" "[ -f '$PROJECT_ROOT/native_services/alphafold_service.py' ]"
run_test "RFDiffusion native service exists" "[ -f '$PROJECT_ROOT/native_services/rfdiffusion_service.py' ]"
run_test "Native service runner exists" "[ -f '$PROJECT_ROOT/scripts/run_arm64_native_model_services.sh' ]"

echo ""

# Final summary
echo "=================================================================="
echo "  Validation Summary"
echo "=================================================================="
echo ""
echo "Tests Run:     $TESTS_RUN"
echo "Tests Passed:  $TESTS_PASSED"
echo "Tests Failed:  $TESTS_FAILED"
echo "Tests Skipped: $TESTS_SKIPPED"
echo ""

SUCCESS_RATE=0
if [ $TESTS_RUN -gt 0 ]; then
    SUCCESS_RATE=$((TESTS_PASSED * 100 / TESTS_RUN))
fi

echo "Success Rate:  ${SUCCESS_RATE}%"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    if [ $TESTS_SKIPPED -gt 0 ]; then
        log_warning "Validation passed with skipped tests"
        log_info "Some components may not be installed"
        echo ""
        echo "To install all components, run:"
        echo "  $PROJECT_ROOT/scripts/install_all_native.sh --recommended"
    else
        log_success "All tests passed! Installation is complete and ready to use."
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Activate environment:"
    echo "     source $PROJECT_ROOT/activate_native.sh"
    echo ""
    echo "  2. Start native services:"
    echo "     bash $PROJECT_ROOT/scripts/run_arm64_native_model_services.sh"
    echo ""
    echo "  3. Start dashboard:"
    echo "     $PROJECT_ROOT/scripts/run_dashboard_stack.sh --arm64-host-native up"
    echo ""
    echo "  4. Open: http://localhost:3000"
    echo ""
    
    exit 0
else
    log_error "Some tests failed. Please review the errors above."
    echo ""
    echo "Common issues:"
    echo "  - Conda environments not created: Run installers again"
    echo "  - Missing dependencies: Check internet connection"
    echo "  - Permission errors: Check file permissions"
    echo ""
    echo "For help, see:"
    echo "  $PROJECT_ROOT/docs/NATIVE_TROUBLESHOOTING.md"
    echo ""
    
    exit 1
fi
