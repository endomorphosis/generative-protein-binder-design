#!/bin/bash
# Master ARM64 Native Installation Script
# This script orchestrates the installation of all protein design tools on ARM64

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

echo "========================================================"
echo "  Protein Design Tools - ARM64 Native Installation"
echo "========================================================"
echo

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    print_error "This script is for ARM64 architecture only. Detected: $ARCH"
    exit 1
fi
print_status "Architecture: $ARCH (ARM64)"

# Check for conda
if ! command -v conda &> /dev/null; then
    print_warning "Conda not found. Installing Miniforge..."
    
    # Download Miniforge for ARM64
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    MINIFORGE_INSTALLER="/tmp/miniforge_installer.sh"
    
    wget -O "$MINIFORGE_INSTALLER" "$MINIFORGE_URL"
    bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
    rm "$MINIFORGE_INSTALLER"
    
    # Initialize conda
    source "$HOME/miniforge3/bin/activate"
    conda init bash
    
    print_status "Miniforge installed"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    print_status "Conda found"
fi

# Install mamba
if ! command -v mamba &> /dev/null; then
    print_info "Installing mamba..."
    conda install -y mamba -n base -c conda-forge
    print_status "Mamba installed"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Installation menu
echo
echo "Select components to install:"
echo "  1) AlphaFold2 only"
echo "  2) RFDiffusion only"
echo "  3) ProteinMPNN only"
echo "  4) All components (recommended)"
echo "  5) Exit"
echo

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        print_info "Installing AlphaFold2..."
        bash "$SCRIPT_DIR/install_alphafold2_arm64.sh"
        ;;
    2)
        print_info "Installing RFDiffusion..."
        bash "$SCRIPT_DIR/install_rfdiffusion_arm64.sh"
        ;;
    3)
        print_info "Installing ProteinMPNN..."
        bash "$SCRIPT_DIR/install_proteinmpnn_arm64.sh"
        ;;
    4)
        print_info "Installing all components..."
        echo
        print_info "Step 1/3: Installing AlphaFold2..."
        bash "$SCRIPT_DIR/install_alphafold2_arm64.sh" || print_warning "AlphaFold2 installation had issues"
        echo
        print_info "Step 2/3: Installing RFDiffusion..."
        bash "$SCRIPT_DIR/install_rfdiffusion_arm64.sh" || print_warning "RFDiffusion installation had issues"
        echo
        print_info "Step 3/3: Installing ProteinMPNN..."
        bash "$SCRIPT_DIR/install_proteinmpnn_arm64.sh" || print_warning "ProteinMPNN installation had issues"
        ;;
    5)
        print_info "Installation cancelled"
        exit 0
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo
echo "========================================================"
echo "  ✓ Installation Complete!"
echo "========================================================"
echo
echo "Installed environments:"
conda env list | grep arm64 || echo "No ARM64 environments found"
echo
echo "Next steps:"
echo "  1. Activate an environment: conda activate <env_name>"
echo "  2. Download model weights for each tool"
echo "  3. Test with sample protein data"
echo
echo "For detailed usage, see the individual tool directories in ~/:"
echo "  - alphafold2_arm64"
echo "  - rfdiffusion_arm64"
echo "  - proteinmpnn_arm64"
echo
