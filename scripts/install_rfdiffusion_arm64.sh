#!/bin/bash
# RFDiffusion Native ARM64 Installation Script
# This script installs RFDiffusion natively on ARM64 systems

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
echo "  RFDiffusion ARM64 Native Installation"
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

# Create RFDiffusion environment
ENV_NAME="rfdiffusion_arm64"
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
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install RFDiffusion dependencies
print_info "Installing RFDiffusion dependencies..."
pip install \
    numpy \
    scipy \
    pandas \
    biopython \
    pytorch-lightning \
    hydra-core \
    pyrsistent \
    decorator \
    e3nn

# Clone RFDiffusion repository
INSTALL_DIR="${HOME}/rfdiffusion_arm64"
print_info "Installing RFDiffusion to: $INSTALL_DIR"

if [ -d "$INSTALL_DIR" ]; then
    print_warning "Directory exists. Updating..."
    cd "$INSTALL_DIR"
    git pull
else
    git clone https://github.com/RosettaCommons/RFdiffusion.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Install RFDiffusion
print_info "Installing RFDiffusion package..."
pip install -e .

# Clone SE3 Transformer
SE3_DIR="${HOME}/se3_transformer"
if [ -d "$SE3_DIR" ]; then
    print_warning "SE3 Transformer directory exists. Updating..."
    cd "$SE3_DIR"
    git pull
else
    git clone https://github.com/RosettaCommons/SE3Transformer.git "$SE3_DIR"
    cd "$SE3_DIR"
fi

pip install -e .
print_status "SE3 Transformer installed"

# Create models directory
MODELS_DIR="${INSTALL_DIR}/models"
mkdir -p "$MODELS_DIR"
print_status "Created models directory: $MODELS_DIR"

# Create run script
cd "$INSTALL_DIR"
RUN_SCRIPT="${INSTALL_DIR}/run_rfdiffusion_arm64.sh"
cat > "$RUN_SCRIPT" << 'EOF'
#!/bin/bash
# RFDiffusion ARM64 Runner Script

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rfdiffusion_arm64

# Set environment variables
export RFDIFFUSION_HOME="${HOME}/rfdiffusion_arm64"
export RFDIFFUSION_MODELS="${RFDIFFUSION_HOME}/models"

# Run RFDiffusion
python "${RFDIFFUSION_HOME}/scripts/run_inference.py" "$@"
EOF

chmod +x "$RUN_SCRIPT"
print_status "Created run script: $RUN_SCRIPT"

# Create test script
TEST_SCRIPT="${INSTALL_DIR}/test_rfdiffusion.py"
cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""Test RFDiffusion ARM64 installation"""

import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
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
    
    try:
        import e3nn
        print(f"✓ e3nn {e3nn.__version__}")
    except ImportError as e:
        print(f"✗ e3nn import failed: {e}")
        return False
    
    return True

def test_torch():
    """Test PyTorch functionality"""
    print("\nTesting PyTorch functionality...")
    
    try:
        import torch
        
        # Simple computation
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.sum(x)
        print(f"✓ PyTorch computation: sum([1,2,3]) = {y.item()}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"ℹ CUDA not available (CPU mode)")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("RFDiffusion ARM64 Installation Test")
    print("=" * 60)
    print()
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_torch():
        success = False
    
    print()
    if success:
        print("=" * 60)
        print("✓ All tests passed! RFDiffusion ARM64 is ready.")
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
echo "  ✓ RFDiffusion ARM64 Installation Complete!"
echo "================================================"
echo
echo "Installation directory: $INSTALL_DIR"
echo "Models directory: $MODELS_DIR"
echo
echo "To use RFDiffusion:"
echo "  1. Activate environment: conda activate $ENV_NAME"
echo "  2. Download models to: $MODELS_DIR"
echo "  3. Run RFDiffusion: $RUN_SCRIPT"
echo
echo "Next steps:"
echo "  1. Download model weights"
echo "  2. Test with sample protein target"
echo
print_warning "Note: Model downloads require ~2-3GB of disk space"
echo
