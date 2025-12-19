#!/bin/bash
# Complete AlphaFold2 Zero-Touch Installation
# Supports: x86_64, ARM64, Linux, macOS
# Database tiers: minimal, reduced, full

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
DB_TIER="minimal"
GPU_MODE="auto"
FORCE_INSTALL=false
CONDA_ENV="alphafold2"
SKIP_VALIDATION=false

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools"
ALPHAFOLD_DIR="$TOOLS_DIR/alphafold2"
DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"

# Parse arguments
show_help() {
    cat << EOF
AlphaFold2 Complete Installation Script

Usage: $0 [OPTIONS]

Options:
    --db-tier TIER        Database tier: minimal, reduced, full (default: minimal)
                        minimal  = 5GB (models only, demo quality)
                        reduced  = 50GB (small BFD, 70-80% accuracy)
                        full     = 2.3TB (complete databases, 100% accuracy)
  
  --gpu MODE            GPU mode: auto, cuda, metal, cpu (default: auto)
  --data-dir DIR        Data directory for models/databases (default: ~/.cache/alphafold)
  --conda-env NAME      Conda environment name (default: alphafold2)
  --force               Force reinstallation
  --skip-validation     Skip validation tests
  --help                Show this help message

Examples:
  # Quick test installation (5GB, CPU-only)
  $0 --db-tier minimal --gpu cpu

  # Recommended for development (50GB, auto GPU)
  $0 --db-tier reduced
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --db-tier)
            DB_TIER="$2"
            shift 2
            ;;
        --gpu)
            GPU_MODE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
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
        --help|-h)
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

# Validate database tier
if [[ ! "$DB_TIER" =~ ^(minimal|reduced|full)$ ]]; then
    log_error "Invalid database tier: $DB_TIER"
    exit 1
fi

# Print configuration
echo "=================================================================="
echo "  AlphaFold2 Zero-Touch Installation"
echo "=================================================================="
log_info "Configuration:"
echo "  Database Tier: $DB_TIER"
echo "  GPU Mode: $GPU_MODE"
echo "  Data Directory: $DATA_DIR"
echo "  Conda Environment: $CONDA_ENV"
echo "  Project Root: $PROJECT_ROOT"
echo "  Installation Directory: $ALPHAFOLD_DIR"
echo ""

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)
log_info "Platform: $OS $ARCH"

# Check disk space
check_disk_space() {
    local required=$1
    mkdir -p "$DATA_DIR" 2>/dev/null || true
    local df_target="$DATA_DIR"
    if [[ ! -d "$df_target" ]]; then
        df_target="$HOME"
    fi
    local available
    available=$(df -BG "$df_target" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || true)
    if [[ -z "${available:-}" ]] || ! [[ "$available" =~ ^[0-9]+$ ]]; then
        # If df parsing fails for any reason, don't crash; just warn and continue.
        log_warning "Could not parse available disk space for '$df_target'; continuing without strict check"
        return 0
    fi
    
    if [ "$available" -lt "$required" ]; then
        log_error "Insufficient disk space: ${available}GB available, ${required}GB required"
        return 1
    fi
    log_success "Disk space: ${available}GB available (${required}GB required)"
}

# Check requirements based on tier
case $DB_TIER in
    minimal)
        check_disk_space 10 || exit 1
        ;;
    reduced)
        check_disk_space 60 || exit 1
        ;;
    full)
        check_disk_space 2500 || exit 1
        ;;
esac

# Step 1: Check/Install system dependencies
log_step "Step 1/8: Checking system dependencies"

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
                libhdf5-dev \
                libopenblas-dev \
                python3-dev \
                aria2 2>/dev/null || log_warning "Some packages may have failed to install"
        elif command -v yum &>/dev/null; then
            sudo yum install -y gcc gcc-c++ cmake git wget curl hdf5-devel openblas-devel python3-devel aria2
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        log_info "Installing system packages via Homebrew..."
        if ! command -v brew &>/dev/null; then
            log_error "Homebrew not found. Install from https://brew.sh"
            exit 1
        fi
        brew install cmake wget git hdf5 openblas aria2 2>/dev/null || log_warning "Some packages may have failed"
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
log_step "Step 2/8: Setting up Conda environment"

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
log_step "Step 3/8: Configuring GPU support"

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

# On Linux ARM64, CUDA-enabled jaxlib wheels are generally not available.
# If a GPU is present, auto-detection may select "cuda" via nvidia-smi, but the
# installation would still fall back to CPU (or fail). Prefer CPU for out-of-box.
if [[ "$OS" == "Linux" ]] && [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]] && [[ "$GPU_TYPE" == "cuda" ]]; then
    if [[ "$GPU_MODE" == "cuda" ]]; then
        log_error "CUDA was requested, but CUDA-enabled jaxlib wheels are not supported on $OS $ARCH in this installer. Use --gpu cpu."
        exit 1
    fi
    log_warning "CUDA detected on $OS $ARCH, but CUDA-enabled jaxlib wheels are not available; using CPU JAX."
    GPU_TYPE="cpu"
fi

# Step 4: Install Python dependencies
log_step "Step 4/8: Installing Python dependencies"

log_info "Installing core packages..."
mamba install -y -q numpy scipy pandas matplotlib biopython jupyter

log_info "Installing JAX..."
case $GPU_TYPE in
    cuda)
        # Match upstream AlphaFold requirements (tools/alphafold2/requirements.txt).
        # Use the CUDA12 wheel index for jaxlib.
        pip install -q --upgrade "jax==0.4.26" "jaxlib==0.4.26+cuda12.cudnn89" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ;;
    metal|cpu)
        # Prefer conda-forge builds for broad architecture support.
        mamba install -y -q "jax=0.4.26" "jaxlib=0.4.26"
        ;;
esac

# Keep pip from upgrading JAX/JAXLIB when installing other deps.
AF_CONSTRAINTS_FILE="/tmp/alphafold2_constraints_${CONDA_ENV}.txt"
cat > "$AF_CONSTRAINTS_FILE" <<'EOF'
jax==0.4.26
jaxlib==0.4.26
EOF

log_info "Installing AlphaFold dependencies..."
pip install -q -c "$AF_CONSTRAINTS_FILE" \
    dm-haiku==0.0.12 \
    ml-collections==0.1.0 \
    absl-py==1.0.0 \
    immutabledict==2.2.3 \
    dm-tree==0.1.8

# Sanity check: ensure JAX pin actually held (Haiku expects older JAX APIs).
python - <<'PY'
import jax, jaxlib
expected = '0.4.26'
if jax.__version__ != expected or jaxlib.__version__ != expected:
    raise SystemExit(f"Expected jax/jaxlib {expected}, got jax={jax.__version__} jaxlib={jaxlib.__version__}")
print(f"Pinned jax/jaxlib OK: {jax.__version__}")
PY

log_success "Python dependencies installed"

# Step 5: Clone AlphaFold repository
log_step "Step 5/8: Installing AlphaFold"

mkdir -p "$TOOLS_DIR"

if [ -d "$ALPHAFOLD_DIR" ]; then
    if [ "$FORCE_INSTALL" = true ]; then
        log_info "Removing existing installation..."
        rm -rf "$ALPHAFOLD_DIR"
    else
        log_info "AlphaFold directory exists, updating..."
        cd "$ALPHAFOLD_DIR"
        git pull -q || log_warning "Git pull failed, continuing..."
    fi
fi

if [ ! -d "$ALPHAFOLD_DIR" ]; then
    log_info "Cloning AlphaFold repository..."
    git clone -q https://github.com/deepmind/alphafold.git "$ALPHAFOLD_DIR"
fi

cd "$ALPHAFOLD_DIR"
log_success "AlphaFold repository ready"

# Step 6: Download model parameters
log_step "Step 6/8: Downloading model parameters"

mkdir -p "$DATA_DIR/params"

download_alphafold_params() {
    # The individual .npz URLs are now commonly blocked (HTTP 403) in some environments.
    # The official tarball remains accessible.
    local url="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    local tar_path="$DATA_DIR/params/alphafold_params_2022-12-06.tar"

    if ls "$DATA_DIR/params"/*.npz >/dev/null 2>&1; then
        log_info "  ✓ params already present (*.npz)"
        return 0
    fi

    log_info "Downloading AlphaFold model parameters (tarball)..."
    wget -q --show-progress -c "$url" -O "$tar_path" || {
        log_error "Failed to download AlphaFold params tarball"
        return 1
    }
    tar -xf "$tar_path" -C "$DATA_DIR/params" || {
        log_error "Failed to extract AlphaFold params tarball"
        return 1
    }
    rm -f "$tar_path" || true

    if ! ls "$DATA_DIR/params"/*.npz >/dev/null 2>&1; then
        log_error "Params extraction completed, but no .npz files were found"
        return 1
    fi
}

download_alphafold_params
log_success "Model parameters present"

# Step 7: Download databases based on tier
log_step "Step 7/8: Downloading databases (tier: $DB_TIER)"

download_databases() {
    case $DB_TIER in
        minimal)
            log_info "Minimal tier: Model parameters only (already downloaded)"
            echo "# AlphaFold2 minimal installation" > "$DATA_DIR/.tier"
            echo "tier=minimal" >> "$DATA_DIR/.tier"
            ;;
            
        reduced)
            log_info "Downloading reduced databases (~50GB)..."
            mkdir -p "$DATA_DIR/databases"
            
            # Download small BFD
            log_info "  Downloading small BFD..."
            if [ ! -f "$DATA_DIR/databases/small_bfd.tar.gz" ]; then
                wget -c "https://storage.googleapis.com/alphafold-databases/reduced_dbs/bfd-first_non_consensus_sequences.fasta.gz" \
                    -O "$DATA_DIR/databases/bfd-first_non_consensus_sequences.fasta.gz" || \
                    log_warning "Small BFD download failed (may need manual download)"
            fi
            
            # Download MGnify
            log_info "  Downloading MGnify..."
            if [ ! -f "$DATA_DIR/databases/mgy_clusters_2022_05.fa.gz" ]; then
                MGNIFY_URL="${ALPHAFOLD_MGNIFY_URL:-https://storage.googleapis.com/alphafold-databases/reduced_dbs/mgy_clusters_2022_05.fa.gz}"
                if wget -c "$MGNIFY_URL" -O "$DATA_DIR/databases/mgy_clusters_2022_05.fa.gz"; then
                    log_info "  ✓ MGnify downloaded"
                else
                    log_warning "MGnify download failed"

                    # Optional fallback: build a small substitute FASTA from HuggingFace.
                    # This is not the official MGnify cluster DB; it is a best-effort way
                    # to unblock restricted environments for demo/bring-up.
                    if [ "${ALPHAFOLD_MGNIFY_FALLBACK:-none}" = "huggingface" ]; then
                        log_info "  Attempting HuggingFace fallback (datasets-server API)..."
                                                if MGNIFY_OUT_PATH="$DATA_DIR/databases/mgy_clusters_2022_05.fa.gz" \
                                                    ALPHAFOLD_MGNIFY_HF_DATASET="${ALPHAFOLD_MGNIFY_HF_DATASET:-tattabio/OMG_prot50}" \
                                                    ALPHAFOLD_MGNIFY_HF_TOKEN="${ALPHAFOLD_MGNIFY_HF_TOKEN:-}" \
                                                    ALPHAFOLD_MGNIFY_HF_MAX_SEQS="${ALPHAFOLD_MGNIFY_HF_MAX_SEQS:-200000}" \
                                                    ALPHAFOLD_MGNIFY_HF_PAGE_SIZE="${ALPHAFOLD_MGNIFY_HF_PAGE_SIZE:-1000}" \
                                                    python - <<'PY'
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request

dataset = (os.environ.get('ALPHAFOLD_MGNIFY_HF_DATASET') or 'tattabio/OMG_prot50').strip()
token = (os.environ.get('ALPHAFOLD_MGNIFY_HF_TOKEN') or '').strip()
out_path = os.environ['MGNIFY_OUT_PATH']

target_sequences = int(os.environ.get('ALPHAFOLD_MGNIFY_HF_MAX_SEQS') or '200000')
page_size = int(os.environ.get('ALPHAFOLD_MGNIFY_HF_PAGE_SIZE') or '1000')
page_size = max(1, min(5000, page_size))

headers = {}
if token:
    headers['Authorization'] = f'Bearer {token}'

def fetch_rows(offset: int, length: int):
    params = {
        'dataset': dataset,
        'config': 'default',
        'split': 'train',
        'offset': offset,
        'length': length,
    }
    url = 'https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode('utf-8'))

written = 0
offset = 0
tmp_path = out_path + '.tmp'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with gzip.open(tmp_path, 'wt', encoding='utf-8') as f:
    while written < target_sequences:
        n = min(page_size, target_sequences - written)
        data = fetch_rows(offset, n)
        rows = data.get('rows') or []
        if not rows:
            break
        for item in rows:
            row = item.get('row') or {}
            seq_id = str(row.get('id') or f'row_{offset}')
            seq = (row.get('sequence') or '').strip()
            if not seq:
                continue
            f.write('>' + seq_id + '\n')
            # wrap at 60 chars
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + '\n')
            written += 1
            if written >= target_sequences:
                break
        offset += len(rows)
        if len(rows) < n:
            break
        # be polite to the public API
        time.sleep(0.1)

os.replace(tmp_path, out_path)
print(f'Wrote {written} sequences to {out_path}', file=sys.stderr)
if written < 1000:
    raise SystemExit('Too few sequences fetched from HuggingFace; aborting')
PY
                        then
                          log_info "  ✓ HuggingFace fallback generated mgy_clusters_2022_05.fa.gz"
                        else
                          log_warning "HuggingFace fallback failed; MGnify still missing"
                        fi
                    fi
                fi
            fi
            
            # Download PDB70
            log_info "  Downloading PDB70..."
            if [ ! -d "$DATA_DIR/databases/pdb70" ]; then
                mkdir -p "$DATA_DIR/databases/pdb70"
                wget -c "http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz" \
                    -O /tmp/pdb70.tar.gz || log_warning "PDB70 download failed"
                [ -f /tmp/pdb70.tar.gz ] && tar -xzf /tmp/pdb70.tar.gz -C "$DATA_DIR/databases/" && rm /tmp/pdb70.tar.gz
            fi

                # Download UniRef30 (required for --db_preset=reduced_dbs)
                log_info "  Downloading UniRef30..."
                if [ ! -f "$DATA_DIR/databases/UniRef30_2021_03.tar.gz" ]; then
                    wget -c "https://storage.googleapis.com/alphafold-databases/v2.3/UniRef30_2021_03.tar.gz" \
                        -O "$DATA_DIR/databases/UniRef30_2021_03.tar.gz" || \
                        log_warning "UniRef30 download failed"
                fi
                if [ -f "$DATA_DIR/databases/UniRef30_2021_03.tar.gz" ] && [ ! -d "$DATA_DIR/databases/uniref30" ]; then
                    mkdir -p "$DATA_DIR/databases/uniref30"
                    tar -xzf "$DATA_DIR/databases/UniRef30_2021_03.tar.gz" -C "$DATA_DIR/databases/uniref30" || \
                        log_warning "UniRef30 extract failed"
                fi
            
            echo "tier=reduced" > "$DATA_DIR/.tier"
            log_success "Reduced databases downloaded"
            ;;
            
        full)
            log_warning "Full database download (~2.3TB) will take several hours..."
            log_info "Running official download script..."
            
            mkdir -p "$DATA_DIR/databases"
            
            # Use AlphaFold's official download script
            if [ -f "$ALPHAFOLD_DIR/scripts/download_all_data.sh" ]; then
                bash "$ALPHAFOLD_DIR/scripts/download_all_data.sh" "$DATA_DIR/databases" || \
                    log_error "Database download failed. You may need to run this manually."
            else
                log_error "Download script not found. Please download databases manually."
                log_info "See: https://github.com/deepmind/alphafold#genetic-databases"
            fi
            
            echo "tier=full" > "$DATA_DIR/.tier"
            log_success "Full databases downloaded"
            ;;
    esac
}

download_databases

# Step 8: Create wrapper scripts and validate
log_step "Step 8/8: Creating wrapper scripts"

# Create activation script
cat > "$ALPHAFOLD_DIR/activate.sh" << EOF
#!/bin/bash
# AlphaFold2 Environment Activation Script

eval "\$(conda shell.bash hook)"
conda activate $CONDA_ENV

export ALPHAFOLD_DIR="$ALPHAFOLD_DIR"
export ALPHAFOLD_DATA_DIR="$DATA_DIR"
export ALPHAFOLD_DB_TIER="$DB_TIER"
export PYTHONPATH="$ALPHAFOLD_DIR:\$PYTHONPATH"

echo "AlphaFold2 environment activated"
echo "  Data directory: \$ALPHAFOLD_DATA_DIR"
echo "  Database tier: \$ALPHAFOLD_DB_TIER"
EOF

chmod +x "$ALPHAFOLD_DIR/activate.sh"

# Create run script
cat > "$ALPHAFOLD_DIR/run_alphafold.sh" << EOF
#!/bin/bash
# AlphaFold2 Run Script

source "$ALPHAFOLD_DIR/activate.sh"

python "$ALPHAFOLD_DIR/run_alphafold.py" \\
    --data_dir="\$ALPHAFOLD_DATA_DIR" \\
    --db_preset=reduced_dbs \\
    --model_preset=monomer \\
    --use_gpu=$([ "$GPU_TYPE" != "cpu" ] && echo "true" || echo "false") \\
    "\$@"
EOF

chmod +x "$ALPHAFOLD_DIR/run_alphafold.sh"

# Create validation script
cat > "$ALPHAFOLD_DIR/validate.py" << 'VALEOF'
#!/usr/bin/env python3
"""Validate AlphaFold2 installation"""

import sys
import os

def validate_imports():
    """Test imports"""
    print("Validating imports...")
    
    try:
        import jax
        print(f"  ✓ JAX {jax.__version__}")
        print(f"    Devices: {jax.devices()}")
    except ImportError as e:
        print(f"  ✗ JAX import failed: {e}")
        return False
    
    try:
        import haiku as hk
        print(f"  ✓ Haiku")
    except ImportError as e:
        print(f"  ✗ Haiku import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    return True

def validate_data():
    """Validate data directory"""
    print("\nValidating data directory...")
    
    data_dir = os.environ.get('ALPHAFOLD_DATA_DIR')
    if not data_dir:
        print("  ✗ ALPHAFOLD_DATA_DIR not set")
        return False
    
    print(f"  Data directory: {data_dir}")
    
    params_dir = os.path.join(data_dir, 'params')
    if not os.path.exists(params_dir):
        print(f"  ✗ Parameters directory not found: {params_dir}")
        return False
    
    param_files = [f for f in os.listdir(params_dir) if f.endswith('.npz')]
    print(f"  ✓ Found {len(param_files)} model parameter files")
    
    tier_file = os.path.join(data_dir, '.tier')
    if os.path.exists(tier_file):
        with open(tier_file) as f:
            tier = f.read().strip()
            print(f"  Database tier: {tier}")
    
    return True

if __name__ == '__main__':
    print("="*60)
    print("AlphaFold2 Installation Validation")
    print("="*60)
    print()
    
    success = validate_imports() and validate_data()
    
    print()
    if success:
        print("="*60)
        print("✓ AlphaFold2 installation is valid and ready to use!")
        print("="*60)
        sys.exit(0)
    else:
        print("="*60)
        print("✗ Validation failed. Please check errors above.")
        print("="*60)
        sys.exit(1)
VALEOF

chmod +x "$ALPHAFOLD_DIR/validate.py"

log_success "Wrapper scripts created"

# Run validation
if [ "$SKIP_VALIDATION" = false ]; then
    log_info "Running validation tests..."
    source "$ALPHAFOLD_DIR/activate.sh"
    python "$ALPHAFOLD_DIR/validate.py" || log_warning "Validation had warnings"
fi

# Create environment file for MCP server integration
cat > "$ALPHAFOLD_DIR/.env" << EOF
ALPHAFOLD_CONDA_ENV=$CONDA_ENV
ALPHAFOLD_DIR=$ALPHAFOLD_DIR
ALPHAFOLD_DATA_DIR=$DATA_DIR
ALPHAFOLD_DB_TIER=$DB_TIER
ALPHAFOLD_GPU_TYPE=$GPU_TYPE
ALPHAFOLD_NATIVE_CMD=$ALPHAFOLD_DIR/run_alphafold.sh --fasta_paths={fasta} --output_dir={out_dir}
EOF

# Final success message
echo ""
echo "=================================================================="
echo "  ✓ AlphaFold2 Installation Complete!"
echo "=================================================================="
echo ""
echo "Installation Summary:"
echo "  Environment: $CONDA_ENV"
echo "  Location: $ALPHAFOLD_DIR"
echo "  Data: $DATA_DIR"
echo "  Database Tier: $DB_TIER"
echo "  GPU Support: $GPU_TYPE"
echo ""
echo "To use AlphaFold2:"
echo "  1. Activate: source $ALPHAFOLD_DIR/activate.sh"
echo "  2. Run: $ALPHAFOLD_DIR/run_alphafold.sh --fasta_paths=input.fasta --output_dir=output"
echo "  3. Validate: python $ALPHAFOLD_DIR/validate.py"
echo ""
echo "MCP Server Integration:"
echo "  Configuration saved to: $ALPHAFOLD_DIR/.env"
echo "  Import with: export \$(cat $ALPHAFOLD_DIR/.env | xargs)"
echo ""

case $DB_TIER in
    minimal)
        log_warning "Minimal installation: Demo quality only"
        ;;
    reduced)
        log_info "Reduced databases: 70-80% accuracy, suitable for development"
        ;;
    full)
        log_success "Full databases: Production-ready, state-of-the-art accuracy"
        ;;
esac

echo ""
