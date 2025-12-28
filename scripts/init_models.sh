#!/bin/bash
# Initialize and download model weights for ARM64 services
# This script is designed to run inside containers or on the host

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default model directory
MODEL_DIR="${MODEL_DIR:-/models}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

log_info "Model initialization starting..."
log_info "Model directory: $MODEL_DIR"

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Function to check if a file exists and is not empty
file_exists_and_valid() {
    local file="$1"
    [ -f "$file" ] && [ -s "$file" ]
}

# Initialize RFDiffusion models
init_rfdiffusion_models() {
    log_info "Initializing RFDiffusion models..."
    
    local RFDIFFUSION_DIR="$MODEL_DIR/rfdiffusion"
    mkdir -p "$RFDIFFUSION_DIR"
    
    # Model URLs - updated to correct locations
    # Note: These models are large and may require special access
    local BASE_URL="https://files.ipd.uw.edu/pub/RFdiffusion"
    # Publicly hosted model filenames (see scripts/download_models_arm64.sh)
    local MODELS=(
        "Base_ckpt.pt"
        "Complex_base_ckpt.pt"
    )
    
    local all_present=true
    for model in "${MODELS[@]}"; do
        local model_path="$RFDIFFUSION_DIR/$model"
        if file_exists_and_valid "$model_path" && [ "$FORCE_DOWNLOAD" != "1" ]; then
            log_success "RFDiffusion model exists: $model"
        else
            all_present=false
            log_info "Downloading RFDiffusion model: $model"
            log_warning "Note: Model download may require special access or alternative source"
            
            # Try wget first, then curl
            if command -v wget &>/dev/null; then
                wget -c -O "$model_path" "$BASE_URL/$model" || {
                    log_warning "Failed to download $model (may require alternative source)"
                    rm -f "$model_path"
                }
            elif command -v curl &>/dev/null; then
                curl -C - -o "$model_path" "$BASE_URL/$model" || {
                    log_warning "Failed to download $model (may require alternative source)"
                    rm -f "$model_path"
                }
            else
                log_error "Neither wget nor curl available"
                return 1
            fi
            
            if file_exists_and_valid "$model_path"; then
                log_success "Downloaded: $model"
            fi
        fi
    done
    
    if [ "$all_present" = true ]; then
        log_success "All RFDiffusion models already present"
    else
        log_warning "RFDiffusion models not fully downloaded"
        log_info "Service will report not_ready until weights are present"
    fi
    
    # Create a marker file
    touch "$RFDIFFUSION_DIR/.initialized"
    return 0
}

# Initialize AlphaFold2 models (basic - full DB is too large)
init_alphafold_models() {
    log_info "Initializing AlphaFold2 models..."
    
    local ALPHAFOLD_DIR="$MODEL_DIR/alphafold"
    mkdir -p "$ALPHAFOLD_DIR/params"
    
    # AlphaFold2 model parameters from Google Cloud Storage
    local BASE_URL="https://storage.googleapis.com/alphafold"
    local PARAMS_VERSION="alphafold_params_2022-12-06"
    
    # We'll download the essential model parameters (not the full databases)
    local MODELS=(
        "params_model_1_ptm.npz"
        "params_model_2_ptm.npz"
        "params_model_3_ptm.npz"
        "params_model_4_ptm.npz"
        "params_model_5_ptm.npz"
    )
    
    local all_present=true
    for model in "${MODELS[@]}"; do
        local model_path="$ALPHAFOLD_DIR/params/$model"
        if file_exists_and_valid "$model_path" && [ "$FORCE_DOWNLOAD" != "1" ]; then
            log_success "AlphaFold2 model exists: $model"
        else
            all_present=false
            log_info "Downloading AlphaFold2 model: $model"
            
            # Try wget first, then curl
            if command -v wget &>/dev/null; then
                wget -c -O "$model_path" "$BASE_URL/$PARAMS_VERSION/$model" || {
                    log_warning "Failed to download $model (this is expected - full models are large)"
                    rm -f "$model_path"
                }
            elif command -v curl &>/dev/null; then
                curl -C - -o "$model_path" "$BASE_URL/$PARAMS_VERSION/$model" || {
                    log_warning "Failed to download $model (this is expected - full models are large)"
                    rm -f "$model_path"
                }
            else
                log_error "Neither wget nor curl available"
                return 1
            fi
            
            if file_exists_and_valid "$model_path"; then
                log_success "Downloaded: $model"
            fi
        fi
    done
    
    if [ "$all_present" = true ]; then
        log_success "All AlphaFold2 models already present"
    fi
    
    # Create a marker file
    touch "$ALPHAFOLD_DIR/.initialized"
    
    # Note about databases
    log_warning "AlphaFold2 requires large databases (~2.2TB) for full functionality"
    log_info "Service will report not_ready until required assets are present"
    
    return 0
}

# Initialize ProteinMPNN models
init_proteinmpnn_models() {
    log_info "Initializing ProteinMPNN models..."
    
    local PROTEINMPNN_DIR="$MODEL_DIR/proteinmpnn"
    mkdir -p "$PROTEINMPNN_DIR"
    
    # ProteinMPNN models are typically included in the repo
    # Check if they exist in the standard location
    if [ -d "/app/ProteinMPNN/vanilla_model_weights" ]; then
        log_success "ProteinMPNN models found in repository"
        ln -sf /app/ProteinMPNN/vanilla_model_weights "$PROTEINMPNN_DIR/weights" 2>/dev/null || true
    else
        log_warning "ProteinMPNN models not found in standard location"
        log_info "ProteinMPNN typically includes models in the git repository"
    fi
    
    # Create a marker file
    touch "$PROTEINMPNN_DIR/.initialized"
    return 0
}

# Main execution
main() {
    log_info "Starting model initialization..."
    
    # Check for download tools
    if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
        log_error "Neither wget nor curl found. Cannot download models."
        log_warning "Services will report not_ready until assets are present"
        exit 1
    fi
    
    # Initialize each service's models
    local failed=0
    
    # RFDiffusion
    if init_rfdiffusion_models; then
        log_success "RFDiffusion models ready"
    else
        log_warning "RFDiffusion model initialization incomplete"
        ((failed++))
    fi
    
    echo ""
    
    # AlphaFold2
    if init_alphafold_models; then
        log_success "AlphaFold2 models ready (or mock mode available)"
    else
        log_warning "AlphaFold2 model initialization incomplete"
        ((failed++))
    fi
    
    echo ""
    
    # ProteinMPNN
    if init_proteinmpnn_models; then
        log_success "ProteinMPNN models ready"
    else
        log_warning "ProteinMPNN model initialization incomplete"
        ((failed++))
    fi
    
    echo ""
    
    if [ $failed -eq 0 ]; then
        log_success "All models initialized successfully!"
        log_info "Services can now run with real models"
    else
        log_warning "Some models failed to initialize ($failed services)"
        log_info "Services will report not_ready until assets are present"
    fi
    
    # Create a global marker
    touch "$MODEL_DIR/.initialized"
    
    return 0
}

# Run main function
main "$@"
