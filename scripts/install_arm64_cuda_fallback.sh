#!/bin/bash
# Install script for ARM64 CUDA Fallback Module
# This script sets up the fallback module and configures the environment

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

echo "=================================================================="
echo "  ARM64 CUDA Fallback Module - Installation"
echo "=================================================================="
echo

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FALLBACK_DIR="$PROJECT_ROOT/src/arm64_cuda_fallback"

# Check architecture
ARCH=$(uname -m)
print_info "Detected architecture: $ARCH"

if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    print_warning "This module is designed for ARM64 systems"
    print_warning "Your system ($ARCH) may not need this fallback"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not found"
    exit 1
fi
print_status "Python 3 found: $(python3 --version)"

# Check if fallback module exists
if [ ! -d "$FALLBACK_DIR" ]; then
    print_error "Fallback module not found at: $FALLBACK_DIR"
    exit 1
fi
print_status "Fallback module found"

# Install dependencies
print_info "Installing Python dependencies..."

# Add fallback module to Python path
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Try to install PyTorch (optional)
print_info "Checking for PyTorch..."
if python3 -c "import torch" 2>/dev/null; then
    print_status "PyTorch already installed"
else
    print_warning "PyTorch not found"
    read -p "Install PyTorch? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing PyTorch (CPU version)..."
        pip3 install torch --index-url https://download.pytorch.org/whl/cpu || print_warning "PyTorch installation failed"
    fi
fi

# Try to install JAX (optional)
print_info "Checking for JAX..."
if python3 -c "import jax" 2>/dev/null; then
    print_status "JAX already installed"
else
    print_warning "JAX not found"
    read -p "Install JAX? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing JAX (CPU version)..."
        pip3 install jax jaxlib || print_warning "JAX installation failed"
    fi
fi

# Test the module
print_info "Testing fallback module..."
cd "$PROJECT_ROOT"
if python3 -m arm64_cuda_fallback info 2>/dev/null; then
    print_status "Module test passed"
else
    print_warning "Module test had issues (this may be normal if frameworks aren't installed)"
fi

# Create activation script
ACTIVATE_SCRIPT="$PROJECT_ROOT/activate_arm64_fallback.sh"
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Activate ARM64 CUDA Fallback environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Configure for CPU performance
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)

# JAX CPU configuration
export JAX_PLATFORM_NAME=cpu

echo "ARM64 CUDA Fallback environment activated"
echo "Python path: $PYTHONPATH"
echo "Using $(nproc) CPU threads"
EOF

chmod +x "$ACTIVATE_SCRIPT"
print_status "Created activation script: $ACTIVATE_SCRIPT"

# Create wrapper script
WRAPPER_SCRIPT="$PROJECT_ROOT/check_arm64_cuda.sh"
cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
# Quick check for ARM64 CUDA status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

python3 -m arm64_cuda_fallback detect
EOF

chmod +x "$WRAPPER_SCRIPT"
print_status "Created wrapper script: $WRAPPER_SCRIPT"

echo
echo "=================================================================="
echo "  Installation Complete!"
echo "=================================================================="
echo
echo "Next steps:"
echo
echo "1. Activate the environment:"
echo "   source $ACTIVATE_SCRIPT"
echo
echo "2. Check CUDA status:"
echo "   ./check_arm64_cuda.sh"
echo
echo "3. Use in your code:"
echo "   from arm64_cuda_fallback import get_optimal_device"
echo "   device = get_optimal_device()"
echo
echo "4. Read the documentation:"
echo "   $FALLBACK_DIR/README.md"
echo
print_warning "This is a TEMPORARY module and will be deprecated when upstream"
print_warning "ARM64 CUDA support is available. Plan for migration."
echo
