#!/bin/bash
# ProteinMPNN Native ARM64 Installation Script
# This script installs ProteinMPNN natively on ARM64 systems

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

echo "================================================"
echo "  ProteinMPNN ARM64 Native Installation"
echo "================================================"
echo

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    print_error "This script is for ARM64 architecture only. Detected: $ARCH"
    exit 1
fi
print_status "Architecture: $ARCH (ARM64)"

# Check for conda/mamba
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Miniforge first"
    exit 1
fi
print_status "Conda found"

# Check for mamba
if ! command -v mamba &> /dev/null; then
    print_info "Installing mamba..."
    conda install -y mamba -n base -c conda-forge
fi
print_status "Mamba available"

# Create ProteinMPNN environment
ENV_NAME="proteinmpnn_arm64"
print_info "Creating conda environment: $ENV_NAME"

if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment $ENV_NAME already exists. Removing..."
    conda env remove -n $ENV_NAME -y
fi

mamba create -n $ENV_NAME python=3.9 -y
print_status "Environment created"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
print_status "Environment activated"

# Install PyTorch for ARM64
print_info "Installing PyTorch for ARM64..."
pip install --upgrade pip
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install ProteinMPNN dependencies
print_info "Installing ProteinMPNN dependencies..."
pip install \
    numpy \
    biopython \
    pandas \
    matplotlib \
    scipy

# Get project root and create tools directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools"
INSTALL_DIR="$TOOLS_DIR/proteinmpnn"
mkdir -p "$INSTALL_DIR"

print_info "Installing ProteinMPNN to: $INSTALL_DIR"

if [ -d "$INSTALL_DIR/ProteinMPNN" ]; then
    print_warning "Directory exists. Updating..."
    cd "$INSTALL_DIR/ProteinMPNN"
    git pull
else
    cd "$INSTALL_DIR"
    git clone https://github.com/dauparas/ProteinMPNN.git
    cd ProteinMPNN
fi

# Create necessary directories
mkdir -p outputs inputs models
print_status "Created working directories"

# Create run script
RUN_SCRIPT="${INSTALL_DIR}/run_proteinmpnn_arm64.sh"
cat > "$RUN_SCRIPT" << EOF
#!/bin/bash
# ProteinMPNN ARM64 Runner Script

# Activate conda environment
eval "\$(conda shell.bash hook)"
conda activate proteinmpnn_arm64

# Set environment variables
export PROTEINMPNN_HOME="$INSTALL_DIR"

# Run ProteinMPNN
python "\${PROTEINMPNN_HOME}/ProteinMPNN/protein_mpnn_run.py" "\$@"
EOF

chmod +x "$RUN_SCRIPT"
print_status "Created run script: $RUN_SCRIPT"

# Create test script
TEST_SCRIPT="${INSTALL_DIR}/test_proteinmpnn.py"
cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""Test ProteinMPNN ARM64 installation"""

import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import Bio
        print(f"✓ BioPython")
    except ImportError as e:
        print(f"✗ BioPython import failed: {e}")
        return False
    
    return True

def test_functionality():
    """Test basic ProteinMPNN functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        import numpy as np
        
        # Simple tensor operation
        x = torch.randn(10, 20)
        y = torch.mean(x)
        print(f"✓ Tensor operations working: mean = {y.item():.4f}")
        
        # Test protein sequence encoding
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        test_seq = "ACDEFG"
        encoded = [aa_to_idx[aa] for aa in test_seq]
        print(f"✓ Sequence encoding: {test_seq} -> {encoded}")
        
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ProteinMPNN ARM64 Installation Test")
    print("=" * 60)
    print()
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_functionality():
        success = False
    
    print()
    if success:
        print("=" * 60)
        print("✓ All tests passed! ProteinMPNN ARM64 is ready.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("✗ Some tests failed. Please check the errors above.")
        print("=" * 60)
        sys.exit(1)
EOF

chmod +x "$TEST_SCRIPT"
print_status "Created test script: $TEST_SCRIPT"

# Run tests
print_info "Running installation tests..."
python "$TEST_SCRIPT"

echo
echo "================================================"
echo "  ✓ ProteinMPNN ARM64 Installation Complete!"
echo "================================================"
echo
echo "Project root: $PROJECT_ROOT"
echo "Installation directory: $INSTALL_DIR"
echo
echo "To use ProteinMPNN:"
echo "  1. Activate environment: conda activate $ENV_NAME"
echo "  2. Run ProteinMPNN: $RUN_SCRIPT"
echo
echo "Next steps:"
echo "  1. Download model weights"
echo "  2. Prepare input PDB files"
echo "  3. Test with sample structures"
echo
print_warning "Note: Model downloads require ~100MB of disk space"
echo
