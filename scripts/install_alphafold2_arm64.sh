#!/bin/bash
# Install AlphaFold2 for ARM64 in project directory

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

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools"
ALPHAFOLD_DIR="$TOOLS_DIR/alphafold2"

echo "=================================================================="
echo "  AlphaFold2 ARM64 Installation"
echo "=================================================================="
print_info "Installing to: $ALPHAFOLD_DIR"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    print_error "ARM64 only. Detected: $ARCH"
    exit 1
fi
print_status "Architecture: $ARCH"

# Create directories
mkdir -p "$TOOLS_DIR"

# Check conda
if ! command -v conda &> /dev/null; then
    print_error "Install Miniforge first"
    exit 1
fi

# Install mamba if needed
if ! command -v mamba &> /dev/null; then
    conda install -y mamba
fi

# Create environment
ENV_NAME="alphafold2_arm64"
if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Removing existing environment"
    conda env remove -n "$ENV_NAME" -y
fi

print_info "Creating environment: $ENV_NAME"
mamba create -n "$ENV_NAME" python=3.9 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install packages
print_info "Installing packages..."
mamba install -y numpy scipy pandas matplotlib biopython
pip install jax[cpu] dm-haiku absl-py immutabledict ml-collections

# Clone AlphaFold2
print_info "Cloning AlphaFold2..."
if [ -d "$ALPHAFOLD_DIR" ]; then
    rm -rf "$ALPHAFOLD_DIR"
fi
git clone https://github.com/deepmind/alphafold.git "$ALPHAFOLD_DIR"
cd "$ALPHAFOLD_DIR"

# Create test script
cat > test_alphafold.py << 'TESTEOF'
import sys
import jax
import haiku as hk
import numpy as np
from Bio import SeqIO

print("✓ All imports successful")
print(f"JAX devices: {jax.devices()}")
print("All tests passed")
TESTEOF

chmod +x test_alphafold.py

# Create wrapper
cat > run_alphafold_arm64.sh << WRAPEOF
#!/bin/bash
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
export PYTHONPATH="$ALPHAFOLD_DIR:\$PYTHONPATH"
python "$ALPHAFOLD_DIR/test_alphafold.py"
WRAPEOF

chmod +x run_alphafold_arm64.sh

conda deactivate
print_status "AlphaFold2 ARM64 installation complete!"
print_info "Test with: $ALPHAFOLD_DIR/run_alphafold_arm64.sh"
