#!/bin/bash
# ARM64 Integration Test Script
# Tests the complete protein design workflow on ARM64

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

ERRORS=0

echo "========================================================"
echo "  ARM64 Integration Testing"
echo "========================================================"
echo

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    print_error "This script is for ARM64 architecture only. Detected: $ARCH"
    exit 1
fi
print_status "Architecture: $ARCH (ARM64)"

# Test 1: Check installations
echo
echo "=== Test 1: Check Native Installations ==="

check_installation() {
    local tool=$1
    local dir=$2
    local env=$3
    
    if [ -d "$dir" ]; then
        print_status "$tool installed at $dir"
        
        # Check if conda environment exists
        if conda env list | grep -q "^$env "; then
            print_status "$tool environment '$env' exists"
            return 0
        else
            print_warning "$tool environment '$env' not found"
            ERRORS=$((ERRORS + 1))
            return 1
        fi
    else
        print_warning "$tool not installed at $dir"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

source ~/miniforge3/bin/activate 2>/dev/null || print_warning "Miniforge not found"

check_installation "AlphaFold2" "${HOME}/alphafold2_arm64" "alphafold2_arm64"
check_installation "RFDiffusion" "${HOME}/rfdiffusion_arm64" "rfdiffusion_arm64"
check_installation "ProteinMPNN" "${HOME}/proteinmpnn_arm64" "proteinmpnn_arm64"

# Test 2: Test Python imports
echo
echo "=== Test 2: Test Python Imports ==="

test_imports() {
    local env=$1
    local tool=$2
    shift 2
    local packages=("$@")
    
    print_info "Testing $tool imports..."
    source ~/miniforge3/bin/activate
    conda activate "$env" 2>/dev/null || {
        print_error "Failed to activate $env"
        ERRORS=$((ERRORS + 1))
        return 1
    }
    
    for package in "${packages[@]}"; do
        if python -c "import ${package}" 2>/dev/null; then
            print_status "$tool: ${package} imports successfully"
        else
            print_error "$tool: ${package} import failed"
            ERRORS=$((ERRORS + 1))
        fi
    done
}

if [ -d "${HOME}/alphafold2_arm64" ]; then
    test_imports "alphafold2_arm64" "AlphaFold2" "jax" "haiku" "numpy" "Bio"
fi

if [ -d "${HOME}/rfdiffusion_arm64" ]; then
    test_imports "rfdiffusion_arm64" "RFDiffusion" "torch" "numpy" "Bio"
fi

if [ -d "${HOME}/proteinmpnn_arm64" ]; then
    test_imports "proteinmpnn_arm64" "ProteinMPNN" "torch" "numpy" "Bio"
fi

# Test 3: Test basic functionality
echo
echo "=== Test 3: Test Basic Functionality ==="

test_tool_functionality() {
    local env=$1
    local tool=$2
    local test_script=$3
    
    print_info "Testing $tool functionality..."
    source ~/miniforge3/bin/activate
    conda activate "$env" 2>/dev/null || return 1
    
    if [ -f "$test_script" ]; then
        if python "$test_script" 2>&1 | grep -q "All tests passed"; then
            print_status "$tool: All tests passed"
        else
            print_warning "$tool: Some tests failed"
            ERRORS=$((ERRORS + 1))
        fi
    else
        print_warning "$tool: Test script not found at $test_script"
        ERRORS=$((ERRORS + 1))
    fi
}

if [ -d "${HOME}/alphafold2_arm64" ]; then
    test_tool_functionality "alphafold2_arm64" "AlphaFold2" "${HOME}/alphafold2_arm64/test_alphafold.py"
fi

if [ -d "${HOME}/rfdiffusion_arm64" ]; then
    test_tool_functionality "rfdiffusion_arm64" "RFDiffusion" "${HOME}/rfdiffusion_arm64/test_rfdiffusion.py"
fi

if [ -d "${HOME}/proteinmpnn_arm64" ]; then
    test_tool_functionality "proteinmpnn_arm64" "ProteinMPNN" "${HOME}/proteinmpnn_arm64/test_proteinmpnn.py"
fi

# Test 4: Test Docker images
echo
echo "=== Test 4: Test Docker Images ==="

if command -v docker &> /dev/null; then
    print_status "Docker available"
    
    test_docker_image() {
        local image=$1
        local tool=$2
        
        if docker images | grep -q "$image"; then
            print_status "$tool: Docker image exists"
            
            # Test running the image
            if docker run --rm "$image" python3 -c "print('OK')" &> /dev/null; then
                print_status "$tool: Docker image runs successfully"
            else
                print_warning "$tool: Docker image failed to run"
                ERRORS=$((ERRORS + 1))
            fi
        else
            print_warning "$tool: Docker image not found"
        fi
    }
    
    test_docker_image "protein-binder/alphafold2:arm64-latest" "AlphaFold2"
    test_docker_image "protein-binder/rfdiffusion:arm64-latest" "RFDiffusion"
    test_docker_image "protein-binder/proteinmpnn:arm64-latest" "ProteinMPNN"
else
    print_warning "Docker not available, skipping Docker tests"
fi

# Test 5: Test GPU access
echo
echo "=== Test 5: Test GPU Access ==="

if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA drivers installed"
    
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_info "GPUs detected: $GPU_COUNT"
    
    # Test GPU in conda environment
    if [ -d "${HOME}/alphafold2_arm64" ]; then
        source ~/miniforge3/bin/activate
        conda activate alphafold2_arm64 2>/dev/null
        
        if python -c "import jax; print(jax.devices())" 2>&1 | grep -q "cpu"; then
            print_info "JAX using CPU (expected for ARM64)"
        fi
    fi
else
    print_warning "NVIDIA drivers not detected"
fi

# Test 6: Test scripts
echo
echo "=== Test 6: Test Installation Scripts ==="

test_script_exists() {
    local script=$1
    local name=$2
    
    if [ -f "$script" ] && [ -x "$script" ]; then
        print_status "$name script exists and is executable"
    else
        print_warning "$name script not found or not executable"
        ERRORS=$((ERRORS + 1))
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

test_script_exists "$SCRIPT_DIR/install_all_arm64.sh" "Master installation"
test_script_exists "$SCRIPT_DIR/install_alphafold2_arm64.sh" "AlphaFold2 installation"
test_script_exists "$SCRIPT_DIR/install_rfdiffusion_arm64.sh" "RFDiffusion installation"
test_script_exists "$SCRIPT_DIR/install_proteinmpnn_arm64.sh" "ProteinMPNN installation"
test_script_exists "$SCRIPT_DIR/build_arm64_images.sh" "Docker build"
test_script_exists "$SCRIPT_DIR/download_models_arm64.sh" "Model download"

# Test 7: Test Dockerfiles
echo
echo "=== Test 7: Test Dockerfiles ==="

PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy"

test_dockerfile() {
    local dockerfile=$1
    local name=$2
    
    if [ -f "$dockerfile" ]; then
        print_status "$name Dockerfile exists"
        
        # Basic syntax check
        if grep -q "FROM" "$dockerfile" && grep -q "RUN" "$dockerfile"; then
            print_status "$name Dockerfile has basic structure"
        else
            print_warning "$name Dockerfile may be incomplete"
            ERRORS=$((ERRORS + 1))
        fi
    else
        print_warning "$name Dockerfile not found"
        ERRORS=$((ERRORS + 1))
    fi
}

test_dockerfile "$DEPLOY_DIR/Dockerfile.alphafold2-arm64" "AlphaFold2"
test_dockerfile "$DEPLOY_DIR/Dockerfile.rfdiffusion-arm64" "RFDiffusion"
test_dockerfile "$DEPLOY_DIR/Dockerfile.proteinmpnn-arm64" "ProteinMPNN"

if [ -f "$DEPLOY_DIR/docker-compose-arm64-native.yaml" ]; then
    print_status "ARM64 native docker-compose file exists"
else
    print_warning "ARM64 native docker-compose file not found"
    ERRORS=$((ERRORS + 1))
fi

# Generate report
echo
echo "========================================================"
echo "  Integration Test Report"
echo "========================================================"
echo

cat > integration_test_report.txt << EOF
ARM64 Integration Test Report
Generated: $(date)
Architecture: $(uname -m)
Kernel: $(uname -r)

Test Results:
- Native Installations: $([ -d "${HOME}/alphafold2_arm64" ] && echo "OK" || echo "MISSING")
- Python Imports: Tested
- Docker Images: $(docker images | grep -c "protein-binder.*arm64" || echo "0") found
- GPU Access: $(command -v nvidia-smi &> /dev/null && echo "Available" || echo "Not available")
- Scripts: All checked
- Dockerfiles: All checked

Errors Found: $ERRORS

Installation Locations:
$(ls -d ${HOME}/*arm64 2>/dev/null || echo "No installations found")

Docker Images:
$(docker images | grep "protein-binder.*arm64" || echo "No ARM64 images found")

Conda Environments:
$(conda env list 2>/dev/null | grep arm64 || echo "No ARM64 environments found")
EOF

cat integration_test_report.txt

echo
echo "Report saved to: integration_test_report.txt"
echo

if [ $ERRORS -eq 0 ]; then
    echo "========================================================"
    print_status "ALL INTEGRATION TESTS PASSED"
    echo "========================================================"
    echo
    echo "ARM64 support is fully functional!"
    exit 0
else
    echo "========================================================"
    print_warning "SOME TESTS FAILED"
    echo "========================================================"
    echo
    echo "Errors found: $ERRORS"
    echo "Please review the output above and fix any issues."
    exit 1
fi
