#!/bin/bash
# Enable GPU support for MMseqs2 searches
# This script updates scripts and configurations to use --gpu 1 flag

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

echo "================================================================"
echo "  Enabling MMseqs2 GPU Support"
echo "================================================================"
echo ""

# Check if GPU is available
if ! nvidia-smi >/dev/null 2>&1; then
    log_error "No GPU detected. Cannot enable GPU mode."
    exit 1
fi

log_info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Check if MMseqs2 has GPU support
if ! mmseqs search --help 2>&1 | grep -q "\-\-gpu"; then
    log_error "MMseqs2 binary does not have GPU support"
    exit 1
fi

log_success "MMseqs2 binary has GPU support"

# Create GPU configuration file
GPU_CONFIG="/tmp/mmseqs2_gpu.conf"
cat > "$GPU_CONFIG" << 'EOF'
# MMseqs2 GPU Configuration
# Auto-generated GPU settings for MMseqs2

# Enable GPU acceleration
export MMSEQS2_GPU=1
export MMSEQS2_GPU_FLAGS="--gpu 1"

# GPU server mode (optional, for multiple queries)
export MMSEQS2_GPU_SERVER=0

# Search parameters with GPU
export MMSEQS2_SEARCH_PARAMS="--gpu 1 -v 2"
EOF

log_success "GPU configuration created: $GPU_CONFIG"

# Update environment
if [[ -f "$HOME/.cache/alphafold/.env.mmseqs2" ]]; then
    log_info "Updating ~/.cache/alphafold/.env.mmseqs2"
    echo "" >> "$HOME/.cache/alphafold/.env.mmseqs2"
    echo "# GPU Configuration (added $(date))" >> "$HOME/.cache/alphafold/.env.mmseqs2"
    cat "$GPU_CONFIG" >> "$HOME/.cache/alphafold/.env.mmseqs2"
    log_success "Environment file updated"
fi

# Create wrapper script for GPU-enabled MMseqs2 search
WRAPPER_SCRIPT="$HOME/.local/bin/mmseqs2-gpu-search"
mkdir -p "$(dirname "$WRAPPER_SCRIPT")"

cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
# Wrapper for GPU-accelerated MMseqs2 search

# Default to GPU mode
GPU_FLAG="--gpu 1"

# Pass through all arguments, adding GPU flag if not present
if [[ "$*" == *"--gpu"* ]]; then
    # GPU flag already present
    exec mmseqs "$@"
else
    # Add GPU flag
    exec mmseqs search "$GPU_FLAG" "$@"
fi
EOF

chmod +x "$WRAPPER_SCRIPT"
log_success "GPU wrapper created: $WRAPPER_SCRIPT"

echo ""
echo "================================================================"
echo "  MMseqs2 GPU Support Enabled"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  - GPU config: $GPU_CONFIG"
echo "  - Wrapper: $WRAPPER_SCRIPT"
echo ""
echo "To use GPU mode in searches, add: --gpu 1"
echo ""
echo "Example:"
echo "  mmseqs search query.db target.db result.db tmp/ --gpu 1"
echo ""
echo "Or use the wrapper:"
echo "  $WRAPPER_SCRIPT query.db target.db result.db tmp/"
echo ""
echo "To verify GPU usage:"
echo "  nvidia-smi dmon -s u  # Run in another terminal during search"
echo ""
