#!/bin/bash
# AlphaFold2 Native ARM64 Installation Script
# This script installs AlphaFold2 natively on ARM64 systems

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
echo "  AlphaFold2 ARM64 Native Installation"
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
    print_error "Conda not found. Please install Miniforge first:"
    echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    echo "  bash Miniforge3-Linux-aarch64.sh -b -p \$HOME/miniforge3"
    echo "  source \$HOME/miniforge3/bin/activate"
    exit 1
fi
print_status "Conda found"

# Check for mamba
if ! command -v mamba &> /dev/null; then
    print_info "Installing mamba for faster package resolution..."
    conda install -y mamba -n base -c conda-forge
fi
print_status "Mamba available"

# Create AlphaFold2 environment
ENV_NAME="alphafold2_arm64"
print_info "Creating conda environment: $ENV_NAME"

if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment $ENV_NAME already exists. Removing..."
    conda env remove -n $ENV_NAME -y
fi

mamba create -n $ENV_NAME python=3.10 -y
print_status "Environment created"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
print_status "Environment activated"

# Install system-level dependencies via conda
print_info "Installing system dependencies..."
mamba install -y \
    openmm=7.7.0 \
    pdbfixer \
    cudatoolkit=11.8 \
    -c conda-forge

# Install JAX for ARM64
print_info "Installing JAX for ARM64..."
pip install --upgrade pip
pip install "jax[cpu]>=0.4.13" jaxlib

# Install AlphaFold dependencies
print_info "Installing AlphaFold dependencies..."
pip install \
    absl-py \
    biopython \
    chex \
    dm-haiku \
    dm-tree \
    docker \
    immutabledict \
    ml-collections \
    numpy \
    pandas \
    scipy \
    tensorflow-cpu

# Clone AlphaFold repository
INSTALL_DIR="${HOME}/alphafold2_arm64"
print_info "Installing AlphaFold2 to: $INSTALL_DIR"

if [ -d "$INSTALL_DIR" ]; then
    print_warning "Directory exists. Updating..."
    cd "$INSTALL_DIR"
    git pull
else
    git clone https://github.com/deepmind/alphafold.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Install AlphaFold
print_info "Installing AlphaFold2 Python package..."
pip install -r requirements.txt

# Create data directory
DATA_DIR="${INSTALL_DIR}/data"
mkdir -p "$DATA_DIR"
print_status "Created data directory: $DATA_DIR"

# Create run script
RUN_SCRIPT="${INSTALL_DIR}/run_alphafold_arm64.sh"
cat > "$RUN_SCRIPT" << 'EOF'
#!/bin/bash
# AlphaFold2 ARM64 Runner Script

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate alphafold2_arm64

# Set environment variables
export ALPHAFOLD_DATA_DIR="${HOME}/alphafold2_arm64/data"

# Run AlphaFold
python "${HOME}/alphafold2_arm64/run_alphafold.py" "$@"
EOF

chmod +x "$RUN_SCRIPT"
print_status "Created run script: $RUN_SCRIPT"

# Create wrapper for Jupyter notebook
NOTEBOOK_SCRIPT="${INSTALL_DIR}/jupyter_alphafold.sh"
cat > "$NOTEBOOK_SCRIPT" << 'EOF'
#!/bin/bash
# Start Jupyter with AlphaFold2 environment

eval "$(conda shell.bash hook)"
conda activate alphafold2_arm64

# Install Jupyter if not already installed
pip install jupyter ipykernel

# Register kernel
python -m ipykernel install --user --name alphafold2_arm64 --display-name "AlphaFold2 ARM64"

# Start Jupyter
cd "${HOME}/alphafold2_arm64"
jupyter notebook
EOF

chmod +x "$NOTEBOOK_SCRIPT"
print_status "Created Jupyter script: $NOTEBOOK_SCRIPT"

# Create test script
TEST_SCRIPT="${INSTALL_DIR}/test_alphafold.py"
cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""Test AlphaFold2 ARM64 installation"""

import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import jax
        print(f"✓ JAX {jax.__version__}")
    except ImportError as e:
        print(f"✗ JAX import failed: {e}")
        return False
    
    try:
        import haiku
        print(f"✓ Haiku (dm-haiku)")
    except ImportError as e:
        print(f"✗ Haiku import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import biopython
        print(f"✓ BioPython")
    except ImportError:
        try:
            import Bio
            print(f"✓ Bio (BioPython)")
        except ImportError as e:
            print(f"✗ BioPython import failed: {e}")
            return False
    
    return True

def test_jax():
    """Test JAX functionality"""
    print("\nTesting JAX functionality...")
    
    try:
        import jax.numpy as jnp
        
        # Simple computation
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        print(f"✓ JAX computation: sum([1,2,3]) = {y}")
        
        # Check devices
        devices = jax.devices()
        print(f"✓ JAX devices: {devices}")
        
        return True
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("AlphaFold2 ARM64 Installation Test")
    print("=" * 60)
    print()
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_jax():
        success = False
    
    print()
    if success:
        print("=" * 60)
        print("✓ All tests passed! AlphaFold2 ARM64 is ready.")
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
echo "  ✓ AlphaFold2 ARM64 Installation Complete!"
echo "================================================"
echo
echo "Installation directory: $INSTALL_DIR"
echo "Data directory: $DATA_DIR"
echo
echo "To use AlphaFold2:"
echo "  1. Activate environment: conda activate $ENV_NAME"
echo "  2. Run AlphaFold: $RUN_SCRIPT"
echo "  3. Or use in Jupyter: $NOTEBOOK_SCRIPT"
echo
echo "Next steps:"
echo "  1. Download model parameters to: $DATA_DIR"
echo "  2. Configure input data paths"
echo "  3. Test with sample protein sequence"
echo
print_warning "Note: Model downloads require ~3-4GB of disk space"
echo
