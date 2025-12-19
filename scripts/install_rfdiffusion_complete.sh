#!/bin/bash
# Complete RFDiffusion Zero-Touch Installation
# Supports: x86_64, ARM64, Linux, macOS

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
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# Default options
GPU_MODE="auto"
FORCE_INSTALL=false
CONDA_ENV="rfdiffusion"
SKIP_VALIDATION=false

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools"
RFDIFFUSION_DIR="$TOOLS_DIR/rfdiffusion"
MODELS_DIR="${RFDIFFUSION_MODELS_DIR:-$HOME/.cache/rfdiffusion/models}"

# Parse arguments
show_help() {
    cat << EOF
RFDiffusion Complete Installation Script

Usage: $0 [OPTIONS]

Options:
  --gpu MODE            GPU mode: auto, cuda, metal, cpu (default: auto)
  --models-dir DIR      Models directory (default: ~/.cache/rfdiffusion/models)
  --conda-env NAME      Conda environment name (default: rfdiffusion)
  --force               Force reinstallation
  --skip-validation     Skip validation tests
  --help                Show this help message

Examples:
  # Quick CPU-only installation
  $0 --gpu cpu

  # Full GPU installation (auto-detect)
  $0

  # Custom models directory
  $0 --models-dir /data/rfdiffusion/models

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_MODE="$2"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --force)
            FORCE_INSTALL=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Print configuration
echo "=================================================================="
echo "  RFDiffusion Zero-Touch Installation"
echo "=================================================================="
log_info "Configuration:"
echo "  GPU Mode: $GPU_MODE"
echo "  Models Directory: $MODELS_DIR"
echo "  Conda Environment: $CONDA_ENV"
echo "  Project Root: $PROJECT_ROOT"
echo "  Installation Directory: $RFDIFFUSION_DIR"
echo ""

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)
log_info "Platform: $OS $ARCH"

# Check disk space (need ~5GB)
check_disk_space() {
    local required=5
    local dir=$(dirname "$MODELS_DIR")
    mkdir -p "$dir"
    local available=$(df -BG "$dir" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "100")
    
    if [ "$available" -lt "$required" ]; then
        log_error "Insufficient disk space: ${available}GB available, ${required}GB required"
        return 1
    fi
    log_success "Disk space: ${available}GB available"
}

check_disk_space || exit 1

# Step 1: Check/Install system dependencies
log_step "Step 1/7: Checking system dependencies"

install_system_deps() {
    if [[ "$OS" == "Linux" ]]; then
        log_info "Installing system packages (requires sudo)..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq \
                build-essential \
                cmake \
                git \
                wget \
                curl \
                aria2 2>/dev/null || log_warning "Some packages may have failed"
        elif command -v yum &>/dev/null; then
            sudo yum install -y gcc gcc-c++ cmake git wget curl aria2
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        log_info "Installing system packages via Homebrew..."
        if ! command -v brew &>/dev/null; then
            log_error "Homebrew not found. Install from https://brew.sh"
            exit 1
        fi
        brew install cmake wget git aria2 2>/dev/null || log_warning "Some packages may have failed"
    fi
}

# Check for essential tools
for tool in git wget curl; do
    if ! command -v $tool &>/dev/null; then
        log_warning "$tool not found, installing system dependencies..."
        install_system_deps
        break
    fi
done

log_success "System dependencies ready"

# Step 2: Setup Conda environment
log_step "Step 2/7: Setting up Conda environment"

if ! command -v conda &>/dev/null; then
    log_info "Conda not found. Installing Miniforge..."
    
    if [[ "$ARCH" == "x86_64" ]]; then
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        if [[ "$OS" == "Darwin" ]]; then
            MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
        fi
    else
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
        if [[ "$OS" == "Darwin" ]]; then
            MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
        fi
    fi
    
    wget -q "$MINIFORGE_URL" -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm /tmp/miniforge.sh
    
    export PATH="$HOME/miniforge3/bin:$PATH"
    eval "$(conda shell.bash hook)"
    conda init bash
    log_success "Miniforge installed"
else
    log_success "Conda found"
fi

# Initialize conda for this script
eval "$(conda shell.bash hook)"

# Install mamba if not present
if ! command -v mamba &>/dev/null; then
    log_info "Installing mamba..."
    conda install -y -q mamba -n base -c conda-forge
fi

# Create or update environment
if conda env list | grep -q "^$CONDA_ENV "; then
    if [ "$FORCE_INSTALL" = true ]; then
        log_info "Removing existing environment..."
        conda env remove -n "$CONDA_ENV" -y
    else
        log_warning "Environment $CONDA_ENV already exists. Use --force to reinstall."
        conda activate "$CONDA_ENV"
        log_info "Using existing environment"
    fi
fi

if ! conda env list | grep -q "^$CONDA_ENV "; then
    log_info "Creating conda environment: $CONDA_ENV"
    mamba create -n "$CONDA_ENV" python=3.10 -y -q
fi

conda activate "$CONDA_ENV"
log_success "Conda environment ready: $CONDA_ENV"

# Step 3: Detect and configure GPU
log_step "Step 3/7: Configuring GPU support"

detect_gpu() {
    if [[ "$GPU_MODE" == "cpu" ]]; then
        echo "cpu"
        return
    fi
    
    if [[ "$GPU_MODE" != "auto" ]]; then
        echo "$GPU_MODE"
        return
    fi
    
    # Auto-detect
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "cuda"
    elif [[ "$OS" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
        echo "metal"
    else
        echo "cpu"
    fi
}

GPU_TYPE=$(detect_gpu)
log_info "GPU mode: $GPU_TYPE"

# Step 4: Install PyTorch
log_step "Step 4/7: Installing PyTorch"

log_info "Installing PyTorch for $GPU_TYPE..."

install_torch_with_fallbacks() {
    local index_url="$1"
    shift
    local versions=("$@")

    for v in "${versions[@]}"; do
        if [[ -z "$v" ]]; then
            continue
        fi
        log_info "Trying torch==$v ($index_url)"
        if pip install -q "torch==${v}" --index-url "$index_url"; then
            # Keep torchvision/torchaudio aligned without letting pip upgrade torch.
            case "$v" in
                2.0.1)
                    pip install -q "torchvision==0.15.2" "torchaudio==2.0.2" --no-deps --index-url "$index_url" || true
                    ;;
                2.0.0)
                    pip install -q "torchvision==0.15.1" "torchaudio==2.0.1" --no-deps --index-url "$index_url" || true
                    ;;
                *)
                    # Best effort for other versions.
                    pip install -q torchvision torchaudio --no-deps --index-url "$index_url" || true
                    ;;
            esac
            return 0
        fi
    done
    return 1
}

ARCH=$(uname -m)
case $GPU_TYPE in
    cuda)
        # CUDA wheels availability varies by architecture. Prefer cu121, but fall back.
        if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
            # aarch64 often lags behind x86_64 in PyTorch wheel availability.
            install_torch_with_fallbacks "https://download.pytorch.org/whl/cu121" \
                "${TORCH_VERSION:-2.1.0}" 2.0.1 2.0.0 || {
                log_warning "CUDA torch wheels not available; falling back to CPU wheels"
                install_torch_with_fallbacks "https://download.pytorch.org/whl/cpu" 2.1.0 2.0.1 2.0.0 || true
            }
        else
            install_torch_with_fallbacks "https://download.pytorch.org/whl/cu121" "${TORCH_VERSION:-2.1.0}" 2.1.0 2.0.1 2.0.0
        fi
        ;;
    metal)
        install_torch_with_fallbacks "https://pypi.org/simple" "${TORCH_VERSION:-2.1.0}" 2.1.0 2.0.1 2.0.0
        ;;
    cpu)
        install_torch_with_fallbacks "https://download.pytorch.org/whl/cpu" "${TORCH_VERSION:-2.1.0}" 2.1.0 2.0.1 2.0.0
        ;;
esac

python -c 'import torch; print(torch.__version__)' >/dev/null 2>&1 || {
    log_error "PyTorch installation failed (no importable torch)"
    exit 1
}

# Create a constraints file so later pip installs do not upgrade torch.
TORCH_VERSION_FULL="$(python - <<'PY'
import re, torch
m = re.match(r'^(\d+\.\d+\.\d+)', torch.__version__)
print(m.group(1) if m else torch.__version__)
PY
)"
CONSTRAINTS_FILE="/tmp/rfdiffusion_constraints_${CONDA_ENV}.txt"
cat > "$CONSTRAINTS_FILE" <<EOF
torch==${TORCH_VERSION_FULL}
EOF
log_info "Pinned torch version in constraints: $TORCH_VERSION_FULL"

log_success "PyTorch installed"

# Step 5: Install RFDiffusion and dependencies
log_step "Step 5/7: Installing RFDiffusion"

mkdir -p "$TOOLS_DIR"

# Clone RFDiffusion
if [ -d "$RFDIFFUSION_DIR/RFdiffusion" ]; then
    if [ "$FORCE_INSTALL" = true ]; then
        log_info "Removing existing installation..."
        rm -rf "$RFDIFFUSION_DIR"
    else
        log_info "RFDiffusion directory exists, updating..."
        cd "$RFDIFFUSION_DIR/RFdiffusion"
        git pull -q || log_warning "Git pull failed, continuing..."
    fi
fi

if [ ! -d "$RFDIFFUSION_DIR/RFdiffusion" ]; then
    mkdir -p "$RFDIFFUSION_DIR"
    cd "$RFDIFFUSION_DIR"
    log_info "Cloning RFDiffusion repository..."
    git clone -q https://github.com/RosettaCommons/RFdiffusion.git
fi

cd "$RFDIFFUSION_DIR/RFdiffusion"

# Install dependencies
log_info "Installing RFDiffusion dependencies..."
pip install -q -c "$CONSTRAINTS_FILE" \
    numpy \
    scipy \
    pandas \
    biopython \
    pillow \
    requests \
    pytorch-lightning \
    hydra-core \
    pyrsistent \
    decorator \
    e3nn \
    omegaconf

# Install RFDiffusion in development mode.
# RFdiffusion declares a dependency on "se3-transformer" which is not published on PyPI.
# We install SE3Transformer from git below, so avoid pip trying (and failing) to resolve it.
pip install -q -e . --no-deps

log_success "RFDiffusion installed"

# Install SE3 Transformer
log_info "Installing SE3 Transformer..."
SE3_LOCAL_DIR="$RFDIFFUSION_DIR/RFdiffusion/env/SE3Transformer"
if [ ! -d "$SE3_LOCAL_DIR" ]; then
    log_error "Expected vendored SE3Transformer at: $SE3_LOCAL_DIR"
    log_error "Your RFdiffusion checkout may be incomplete. Re-run with --force."
    exit 1
fi

pip install -q -e "$SE3_LOCAL_DIR"
log_success "SE3 Transformer installed"

# Step 6: Download model weights
log_step "Step 6/7: Downloading model weights"

mkdir -p "$MODELS_DIR"

download_models() {
    log_info "Downloading RFDiffusion model weights..."

    local download_script="$RFDIFFUSION_DIR/RFdiffusion/scripts/download_models.sh"
    if [ -x "$download_script" ]; then
        bash "$download_script" "$MODELS_DIR" || log_warning "Model download script failed"
    else
        log_warning "RFdiffusion download_models.sh not found/executable at: $download_script"
    fi

    # Optional checkpoint used by some workflows.
    local optional_url="http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt"
    local optional_dest="$MODELS_DIR/Complex_beta_ckpt.pt"
    if [ ! -s "$optional_dest" ]; then
        log_info "  Downloading optional Complex_beta_ckpt.pt..."
        wget -q --show-progress -c "$optional_url" -O "$optional_dest" || {
            log_warning "  Failed to download Complex_beta_ckpt.pt"
            log_info "    You may need to download manually from: $optional_url"
            rm -f "$optional_dest"
        }
    fi
}

download_models
log_success "Model weights processed"

# Step 7: Create wrapper scripts and validate
log_step "Step 7/7: Creating wrapper scripts"

# Create activation script
cat > "$RFDIFFUSION_DIR/activate.sh" << EOF
#!/bin/bash
# RFDiffusion Environment Activation Script

eval "\$(conda shell.bash hook)"
conda activate $CONDA_ENV

export RFDIFFUSION_DIR="$RFDIFFUSION_DIR/RFdiffusion"
export RFDIFFUSION_MODELS="$MODELS_DIR"
export PYTHONPATH="$RFDIFFUSION_DIR/RFdiffusion:\$PYTHONPATH"

echo "RFDiffusion environment activated"
echo "  Installation: \$RFDIFFUSION_DIR"
echo "  Models: \$RFDIFFUSION_MODELS"
EOF

chmod +x "$RFDIFFUSION_DIR/activate.sh"

# Create run script
cat > "$RFDIFFUSION_DIR/run_rfdiffusion.sh" << EOF
#!/bin/bash
# RFDiffusion Run Script

source "$RFDIFFUSION_DIR/activate.sh"

python "\$RFDIFFUSION_DIR/scripts/run_inference.py" \\
    inference.model_directory_path="\$RFDIFFUSION_MODELS" \\
    "\$@"
EOF

chmod +x "$RFDIFFUSION_DIR/run_rfdiffusion.sh"

# Create validation script
cat > "$RFDIFFUSION_DIR/validate.py" << 'VALEOF'
#!/usr/bin/env python3
"""Validate RFDiffusion installation"""

import sys
import os

def validate_imports():
    """Test imports"""
    print("Validating imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        from Bio import SeqIO
        print(f"  ✓ BioPython")
    except ImportError as e:
        print(f"  ✗ BioPython import failed: {e}")
        return False
    
    try:
        import e3nn
        print(f"  ✓ e3nn {e3nn.__version__}")
    except ImportError as e:
        print(f"  ✗ e3nn import failed: {e}")
        return False
    
    return True

def validate_models():
    """Validate model directory"""
    print("\nValidating models...")
    
    models_dir = os.environ.get('RFDIFFUSION_MODELS')
    if not models_dir:
        print("  ✗ RFDIFFUSION_MODELS not set")
        return False
    
    print(f"  Models directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"  ✗ Models directory not found: {models_dir}")
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    if len(model_files) == 0:
        print(f"  ⚠ No model files found (download may be needed)")
    else:
        print(f"  ✓ Found {len(model_files)} model files")
        for model in model_files:
            size_mb = os.path.getsize(os.path.join(models_dir, model)) / (1024*1024)
            print(f"    - {model} ({size_mb:.1f} MB)")
    
    return True

def validate_installation():
    """Validate RFDiffusion installation"""
    print("\nValidating installation...")
    
    rfdiffusion_dir = os.environ.get('RFDIFFUSION_DIR')
    if not rfdiffusion_dir:
        print("  ✗ RFDIFFUSION_DIR not set")
        return False
    
    print(f"  Installation directory: {rfdiffusion_dir}")
    
    inference_script = os.path.join(rfdiffusion_dir, 'scripts', 'run_inference.py')
    if not os.path.exists(inference_script):
        print(f"  ✗ Inference script not found: {inference_script}")
        return False
    
    print(f"  ✓ Inference script found")
    
    return True

if __name__ == '__main__':
    print("="*60)
    print("RFDiffusion Installation Validation")
    print("="*60)
    print()
    
    success = (
        validate_imports() and
        validate_models() and
        validate_installation()
    )
    
    print()
    if success:
        print("="*60)
        print("✓ RFDiffusion installation is valid and ready to use!")
        print("="*60)
        sys.exit(0)
    else:
        print("="*60)
        print("✗ Validation failed. Please check errors above.")
        print("="*60)
        sys.exit(1)
VALEOF

chmod +x "$RFDIFFUSION_DIR/validate.py"

log_success "Wrapper scripts created"

# Run validation
if [ "$SKIP_VALIDATION" = false ]; then
    log_info "Running validation tests..."
    RFDIFFUSION_INSTALL_ROOT="$RFDIFFUSION_DIR"
    source "$RFDIFFUSION_INSTALL_ROOT/activate.sh"
    python "$RFDIFFUSION_INSTALL_ROOT/validate.py" || log_warning "Validation had warnings"
fi

# Create environment file for MCP server integration
RFDIFFUSION_INSTALL_ROOT="${RFDIFFUSION_INSTALL_ROOT:-$RFDIFFUSION_DIR}"
cat > "$RFDIFFUSION_INSTALL_ROOT/.env" << EOF
RFDIFFUSION_CONDA_ENV=$CONDA_ENV
RFDIFFUSION_DIR=$RFDIFFUSION_INSTALL_ROOT/RFdiffusion
RFDIFFUSION_MODELS=$MODELS_DIR
RFDIFFUSION_GPU_TYPE=$GPU_TYPE
RFDIFFUSION_NATIVE_CMD=$RFDIFFUSION_INSTALL_ROOT/run_rfdiffusion.sh inference.input_pdb={target_pdb} inference.output_prefix={out_dir}/design_{design_id} inference.num_designs=1
EOF

# Final success message
echo ""
echo "=================================================================="
echo "  ✓ RFDiffusion Installation Complete!"
echo "=================================================================="
echo ""
echo "Installation Summary:"
echo "  Environment: $CONDA_ENV"
echo "  Location: $RFDIFFUSION_DIR"
echo "  Models: $MODELS_DIR"
echo "  GPU Support: $GPU_TYPE"
echo ""
echo "To use RFDiffusion:"
echo "  1. Activate: source $RFDIFFUSION_INSTALL_ROOT/activate.sh"
echo "  2. Run: $RFDIFFUSION_INSTALL_ROOT/run_rfdiffusion.sh 'contigmap.contigs=[50-50]'"
echo "  3. Validate: python $RFDIFFUSION_INSTALL_ROOT/validate.py"
echo ""
echo "MCP Server Integration:"
echo "  Configuration saved to: $RFDIFFUSION_INSTALL_ROOT/.env"
echo "  Import with: export \$(cat $RFDIFFUSION_INSTALL_ROOT/.env | xargs)"
echo ""
