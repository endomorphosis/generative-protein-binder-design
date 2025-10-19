#!/bin/bash
# ARM64 Model Download Script
# Downloads and configures model weights for all protein design tools

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
echo "  ARM64 Model Downloads"
echo "========================================================"
echo

# Check for wget or curl
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -c -O"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -C - -o"
else
    print_error "Neither wget nor curl found. Please install one of them."
    exit 1
fi
print_status "Download tool available"

# Check disk space
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
REQUIRED_SPACE=10  # 10GB minimum
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    print_error "Insufficient disk space: ${AVAILABLE_SPACE}GB available, ${REQUIRED_SPACE}GB required"
    exit 1
fi
print_status "Disk space: ${AVAILABLE_SPACE}GB available"

# Download menu
echo
echo "Select models to download:"
echo "  1) AlphaFold2 models (~3-4GB)"
echo "  2) RFDiffusion models (~2-3GB)"
echo "  3) ProteinMPNN models (~100MB)"
echo "  4) All models (~5-7GB total)"
echo "  5) Exit"
echo

read -p "Enter your choice (1-5): " choice

download_alphafold2_models() {
    print_info "Downloading AlphaFold2 models..."
    
    ALPHAFOLD_DIR="${HOME}/alphafold2_arm64/data"
    if [ ! -d "$ALPHAFOLD_DIR" ]; then
        mkdir -p "$ALPHAFOLD_DIR"
        print_info "Created directory: $ALPHAFOLD_DIR"
    fi
    
    cd "$ALPHAFOLD_DIR"
    
    print_info "AlphaFold2 models require manual download from:"
    echo "  https://github.com/deepmind/alphafold#genetic-databases"
    echo
    echo "Download the following to: $ALPHAFOLD_DIR"
    echo "  - params_model_1.npz"
    echo "  - params_model_2.npz"
    echo "  - params_model_3.npz"
    echo "  - params_model_4.npz"
    echo "  - params_model_5.npz"
    echo
    
    print_warning "Note: Full databases require ~2.2TB. For testing, models alone (~3-4GB) are sufficient."
    
    # Create a script to help with download
    cat > download_alphafold_models.sh << 'EOF'
#!/bin/bash
# AlphaFold2 Model Download Helper

echo "Downloading AlphaFold2 model parameters..."

BASE_URL="https://storage.googleapis.com/alphafold"

for i in {1..5}; do
    echo "Downloading model $i..."
    wget -c "${BASE_URL}/alphafold_params_2022-12-06/params_model_${i}.npz" || \
        echo "Failed to download model $i"
done

echo "Download complete!"
ls -lh params_model_*.npz
EOF
    
    chmod +x download_alphafold_models.sh
    print_status "Created download helper: $ALPHAFOLD_DIR/download_alphafold_models.sh"
    
    read -p "Run download now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash download_alphafold_models.sh
    fi
}

download_rfdiffusion_models() {
    print_info "Downloading RFDiffusion models..."
    
    RFDIFFUSION_DIR="${HOME}/rfdiffusion_arm64/models"
    if [ ! -d "$RFDIFFUSION_DIR" ]; then
        mkdir -p "$RFDIFFUSION_DIR"
        print_info "Created directory: $RFDIFFUSION_DIR"
    fi
    
    cd "$RFDIFFUSION_DIR"
    
    print_info "RFDiffusion models available from:"
    echo "  https://files.ipd.uw.edu/pub/RFdiffusion/"
    echo
    
    # Download main model weights
    MODELS=(
        "Complex_base_ckpt.pt"
        "Base_ckpt.pt"
    )
    
    for model in "${MODELS[@]}"; do
        if [ -f "$model" ]; then
            print_status "$model already exists"
        else
            print_info "Downloading $model..."
            wget -c "https://files.ipd.uw.edu/pub/RFdiffusion/${model}" || \
                print_warning "Failed to download $model"
        fi
    done
    
    print_status "RFDiffusion models downloaded to: $RFDIFFUSION_DIR"
    ls -lh "$RFDIFFUSION_DIR"
}

download_proteinmpnn_models() {
    print_info "Downloading ProteinMPNN models..."
    
    PROTEINMPNN_DIR="${HOME}/proteinmpnn_arm64/models"
    if [ ! -d "$PROTEINMPNN_DIR" ]; then
        mkdir -p "$PROTEINMPNN_DIR"
        print_info "Created directory: $PROTEINMPNN_DIR"
    fi
    
    cd "$PROTEINMPNN_DIR"
    
    # ProteinMPNN models are typically included in the repo
    # But we can download specific checkpoints if needed
    
    print_info "ProteinMPNN typically includes models in the repository."
    print_info "Checking for models in: ${HOME}/proteinmpnn_arm64/ProteinMPNN/"
    
    PROTEINMPNN_REPO="${HOME}/proteinmpnn_arm64/ProteinMPNN"
    if [ -d "$PROTEINMPNN_REPO" ]; then
        if [ -d "$PROTEINMPNN_REPO/vanilla_model_weights" ]; then
            print_status "ProteinMPNN models found in repository"
            ls -lh "$PROTEINMPNN_REPO/vanilla_model_weights"
        else
            print_warning "Models not found. They should be included in the git repo."
            print_info "If missing, clone again: git clone https://github.com/dauparas/ProteinMPNN.git"
        fi
    else
        print_warning "ProteinMPNN not installed. Run install_proteinmpnn_arm64.sh first."
    fi
}

case $choice in
    1)
        download_alphafold2_models
        ;;
    2)
        download_rfdiffusion_models
        ;;
    3)
        download_proteinmpnn_models
        ;;
    4)
        print_info "Downloading all models..."
        echo
        download_alphafold2_models
        echo
        download_rfdiffusion_models
        echo
        download_proteinmpnn_models
        ;;
    5)
        print_info "Download cancelled"
        exit 0
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo
echo "========================================================"
echo "  ✓ Model Download Complete!"
echo "========================================================"
echo
echo "Model locations:"
echo "  AlphaFold2: ${HOME}/alphafold2_arm64/data"
echo "  RFDiffusion: ${HOME}/rfdiffusion_arm64/models"
echo "  ProteinMPNN: ${HOME}/proteinmpnn_arm64/models"
echo
echo "Verify downloads:"
echo "  du -sh ${HOME}/*/models ${HOME}/alphafold2_arm64/data"
echo
print_warning "Remember: Full AlphaFold2 databases require ~2.2TB for production use"
echo
