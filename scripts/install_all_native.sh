#!/bin/bash
# Unified Zero-Touch Installation Script
# Installs AlphaFold2, RFDiffusion, ProteinMPNN, and MCP Server

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }
log_header() { echo -e "${MAGENTA}$1${NC}"; }

# Default options
DB_TIER="minimal"
GPU_MODE="auto"
INSTALL_ALPHAFOLD=true
INSTALL_RFDIFFUSION=true
INSTALL_PROTEINMPNN=true
INSTALL_MCP=true
FORCE_INSTALL=false
SKIP_VALIDATION=false

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
show_help() {
    cat << EOF
Unified Zero-Touch Installation for Protein Binder Design

Usage: $0 [OPTIONS]

Installation Profiles:
  --minimal             Minimal installation (5GB, CPU-only, ~15 min)
                        AlphaFold2 models only, RFDiffusion, ProteinMPNN, MMseqs2
  
  --recommended         Recommended installation (50GB, GPU, ~1 hour)
                        AlphaFold2 reduced databases, all tools with GPU support,
                        MMseqs2 optimized database build
  
  --full                Full production installation (2.3TB, GPU, ~6 hours)
                        Complete AlphaFold2 databases, all tools,
                        MMseqs2 complete database

Component Selection:
  --alphafold-only      Install only AlphaFold2
  --rfdiffusion-only    Install only RFDiffusion
  --proteinmpnn-only    Install only ProteinMPNN
  --no-alphafold        Skip AlphaFold2 installation
  --no-rfdiffusion      Skip RFDiffusion installation
  --no-proteinmpnn      Skip ProteinMPNN installation
  --no-mcp              Skip MCP server configuration

Options:
  --db-tier TIER        AlphaFold database tier: minimal, reduced, full
  --gpu MODE            GPU mode: auto, cuda, metal, cpu
  --force               Force reinstallation of all components
  --skip-validation     Skip validation tests
  --help                Show this help message

Examples:
  # Quick test installation (5GB, ~15 minutes)
  $0 --minimal

  # Recommended for development (50GB, ~1 hour)
  $0 --recommended

  # Production installation (2.3TB, ~6 hours)
  $0 --full

  # Install only RFDiffusion with CPU
  $0 --rfdiffusion-only --gpu cpu

  # Custom installation
  $0 --db-tier reduced --gpu cuda --no-proteinmpnn

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            DB_TIER="minimal"
            GPU_MODE="cpu"
            shift
            ;;
        --recommended)
            DB_TIER="reduced"
            GPU_MODE="auto"
            shift
            ;;
        --full)
            DB_TIER="full"
            GPU_MODE="auto"
            shift
            ;;
        --alphafold-only)
            INSTALL_RFDIFFUSION=false
            INSTALL_PROTEINMPNN=false
            INSTALL_MCP=false
            shift
            ;;
        --rfdiffusion-only)
            INSTALL_ALPHAFOLD=false
            INSTALL_PROTEINMPNN=false
            INSTALL_MCP=false
            shift
            ;;
        --proteinmpnn-only)
            INSTALL_ALPHAFOLD=false
            INSTALL_RFDIFFUSION=false
            INSTALL_MCP=false
            shift
            ;;
        --no-alphafold)
            INSTALL_ALPHAFOLD=false
            shift
            ;;
        --no-rfdiffusion)
            INSTALL_RFDIFFUSION=false
            shift
            ;;
        --no-proteinmpnn)
            INSTALL_PROTEINMPNN=false
            shift
            ;;
        --no-mcp)
            INSTALL_MCP=false
            shift
            ;;
        --db-tier)
            DB_TIER="$2"
            shift 2
            ;;
        --gpu)
            GPU_MODE="$2"
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

# Display banner
clear
log_header "=================================================================="
log_header "  Protein Binder Design - Zero-Touch Installation"
log_header "=================================================================="
echo ""
log_info "Platform: $(uname -s) $(uname -m)"
log_info "Project: $PROJECT_ROOT"
echo ""

# Display installation plan
log_header "Installation Plan:"
echo "  Components:"
[ "$INSTALL_ALPHAFOLD" = true ] && echo "    ✓ AlphaFold2 (database tier: $DB_TIER) + MMseqs2"
[ "$INSTALL_RFDIFFUSION" = true ] && echo "    ✓ RFDiffusion"
[ "$INSTALL_PROTEINMPNN" = true ] && echo "    ✓ ProteinMPNN"
[ "$INSTALL_MCP" = true ] && echo "    ✓ MCP Server configuration"
echo "  GPU Mode: $GPU_MODE"
echo "  Force Install: $FORCE_INSTALL"
echo ""

# Estimate disk space and time
estimate_resources() {
    local disk=0
    local time_min=0
    
    if [ "$INSTALL_ALPHAFOLD" = true ]; then
        case $DB_TIER in
            minimal) disk=$((disk + 5)); time_min=$((time_min + 10)) ;;
            reduced) disk=$((disk + 50)); time_min=$((time_min + 45)) ;;
            full) disk=$((disk + 2300)); time_min=$((time_min + 300)) ;;
        esac
    fi
    
    [ "$INSTALL_RFDIFFUSION" = true ] && disk=$((disk + 5)) && time_min=$((time_min + 5))
    [ "$INSTALL_PROTEINMPNN" = true ] && disk=$((disk + 1)) && time_min=$((time_min + 3))
    
    echo "  Estimated disk space: ${disk}GB"
    echo "  Estimated time: $((time_min / 60))h $((time_min % 60))min"
}

estimate_resources
echo ""

# Confirmation prompt
read -p "Continue with installation? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Installation cancelled"
    exit 0
fi

echo ""

# Track installation status
INSTALLATION_LOG="$PROJECT_ROOT/.installation.log"
START_TIME=$(date +%s)

log_installation() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$INSTALLATION_LOG"
}

log_installation "=== Installation Started ==="
log_installation "DB Tier: $DB_TIER, GPU: $GPU_MODE"

# Install AlphaFold2
if [ "$INSTALL_ALPHAFOLD" = true ]; then
    log_step "Installing AlphaFold2..."
    log_installation "AlphaFold2: Starting"
    
    ALPHAFOLD_ARGS="--db-tier $DB_TIER --gpu $GPU_MODE"
    [ "$FORCE_INSTALL" = true ] && ALPHAFOLD_ARGS="$ALPHAFOLD_ARGS --force"
    [ "$SKIP_VALIDATION" = true ] && ALPHAFOLD_ARGS="$ALPHAFOLD_ARGS --skip-validation"
    
    if bash "$SCRIPT_DIR/install_alphafold2_complete.sh" $ALPHAFOLD_ARGS; then
        log_success "AlphaFold2 installation complete"
        log_installation "AlphaFold2: SUCCESS"
    else
        log_error "AlphaFold2 installation failed"
        log_installation "AlphaFold2: FAILED"
        exit 1
    fi
    echo ""
    
    # Install MMseqs2 for optimized MSA generation
    log_step "Installing MMseqs2 for optimized MSA..."
    log_installation "MMseqs2: Starting"
    
    MMSEQS2_ARGS="--conda-env alphafold2 --data-dir $HOME/.cache/alphafold --db-tier $DB_TIER --build-db"
    if bash "$SCRIPT_DIR/install_mmseqs2.sh" $MMSEQS2_ARGS; then
        log_success "MMseqs2 installation and database build complete"
        log_installation "MMseqs2: SUCCESS"
        
        # Capture MMseqs2 environment variables for MCP config
        MMSEQS2_EXPORTS=$(bash "$SCRIPT_DIR/install_mmseqs2.sh" --conda-env alphafold2 --print-env 2>/dev/null || echo "")
        if [ -n "$MMSEQS2_EXPORTS" ]; then
            echo "$MMSEQS2_EXPORTS" >> "$INSTALLATION_LOG"
        fi
    else
        log_warning "MMseqs2 installation failed (non-critical, continuing...)"
        log_installation "MMseqs2: WARNING"
    fi
    echo ""
fi

# Install RFDiffusion
if [ "$INSTALL_RFDIFFUSION" = true ]; then
    log_step "Installing RFDiffusion..."
    log_installation "RFDiffusion: Starting"
    
    RFDIFFUSION_ARGS="--gpu $GPU_MODE"
    [ "$FORCE_INSTALL" = true ] && RFDIFFUSION_ARGS="$RFDIFFUSION_ARGS --force"
    [ "$SKIP_VALIDATION" = true ] && RFDIFFUSION_ARGS="$RFDIFFUSION_ARGS --skip-validation"
    
    if bash "$SCRIPT_DIR/install_rfdiffusion_complete.sh" $RFDIFFUSION_ARGS; then
        log_success "RFDiffusion installation complete"
        log_installation "RFDiffusion: SUCCESS"
    else
        log_error "RFDiffusion installation failed"
        log_installation "RFDiffusion: FAILED"
        exit 1
    fi
    echo ""
fi

# Install ProteinMPNN
if [ "$INSTALL_PROTEINMPNN" = true ]; then
    log_step "Installing ProteinMPNN..."
    log_installation "ProteinMPNN: Starting"
    
    PROTEINMPNN_ARGS=""
    [ "$FORCE_INSTALL" = true ] && PROTEINMPNN_ARGS="$PROTEINMPNN_ARGS --force"
    
    if bash "$SCRIPT_DIR/install_proteinmpnn_arm64.sh"; then
        log_success "ProteinMPNN installation complete"
        log_installation "ProteinMPNN: SUCCESS"
    else
        log_warning "ProteinMPNN installation had warnings (continuing...)"
        log_installation "ProteinMPNN: WARNING"
    fi
    echo ""
fi

# Configure MCP Server
if [ "$INSTALL_MCP" = true ]; then
    log_step "Configuring MCP Server integration..."
    log_installation "MCP: Configuring"
    
    # Create combined environment file
    MCP_ENV_FILE="$PROJECT_ROOT/mcp-server/.env.native"
    
    cat > "$MCP_ENV_FILE" << 'EOF'
# Native Backend Configuration
# Auto-generated by unified installer

MODEL_BACKEND=native

# Component paths
EOF

    AF_ENV_FILE="$PROJECT_ROOT/tools/generated/alphafold2/.env"
    AF_ENV_FILE_LEGACY="$PROJECT_ROOT/tools/alphafold2/.env"
    AF_ACTIVATE="$PROJECT_ROOT/tools/generated/alphafold2/activate.sh"
    AF_ACTIVATE_LEGACY="$PROJECT_ROOT/tools/alphafold2/activate.sh"
    
    # Add AlphaFold2 config if installed
    if [ "$INSTALL_ALPHAFOLD" = true ] && { [ -f "$AF_ENV_FILE" ] || [ -f "$AF_ENV_FILE_LEGACY" ]; }; then
        log_info "  Adding AlphaFold2 configuration..."
        if [ -f "$AF_ENV_FILE" ]; then
            cat "$AF_ENV_FILE" >> "$MCP_ENV_FILE"
        else
            cat "$AF_ENV_FILE_LEGACY" >> "$MCP_ENV_FILE"
        fi
        echo "" >> "$MCP_ENV_FILE"
    fi
    
    # Add RFDiffusion config if installed
    RF_ENV_FILE="$PROJECT_ROOT/tools/generated/rfdiffusion/.env"
    RF_ENV_FILE_LEGACY="$PROJECT_ROOT/tools/rfdiffusion/.env"
    RF_ENV_FILE_LEGACY2="$PROJECT_ROOT/tools/rfdiffusion/RFdiffusion/.env"

    if [ "$INSTALL_RFDIFFUSION" = true ] && { [ -f "$RF_ENV_FILE" ] || [ -f "$RF_ENV_FILE_LEGACY" ] || [ -f "$RF_ENV_FILE_LEGACY2" ]; }; then
        log_info "  Adding RFDiffusion configuration..."
        if [ -f "$RF_ENV_FILE" ]; then
            cat "$RF_ENV_FILE" >> "$MCP_ENV_FILE"
        elif [ -f "$RF_ENV_FILE_LEGACY" ]; then
            cat "$RF_ENV_FILE_LEGACY" >> "$MCP_ENV_FILE"
        else
            cat "$RF_ENV_FILE_LEGACY2" >> "$MCP_ENV_FILE"
        fi
        echo "" >> "$MCP_ENV_FILE"
    fi
    
    # Add ProteinMPNN config if installed
    if [ "$INSTALL_PROTEINMPNN" = true ] && [ -d "$PROJECT_ROOT/tools/proteinmpnn" ]; then
        log_info "  Adding ProteinMPNN configuration..."
        cat >> "$MCP_ENV_FILE" << EOF
PROTEINMPNN_CONDA_ENV=proteinmpnn_arm64
PROTEINMPNN_DIR=$PROJECT_ROOT/tools/proteinmpnn/ProteinMPNN
EOF
    fi
    
    log_success "MCP Server configuration complete"
    log_installation "MCP: SUCCESS"
    echo ""
fi

# Setup GPU optimization (auto-enable for recommended/full, optional for minimal)
log_step "Setting up GPU optimizations..."
log_installation "GPU Optimization: Starting"

if bash "$SCRIPT_DIR/detect_gpu_and_generate_env.sh"; then
    log_success "GPU configuration generated"
    log_installation "GPU Optimization: SUCCESS"
    
    # Add GPU config to MCP environment
    if [ -f "$PROJECT_ROOT/.env.gpu" ]; then
        cat >> "$MCP_ENV_FILE" << EOF

# GPU Optimization Configuration (auto-generated)
EOF
        grep "^[A-Z_]*=" "$PROJECT_ROOT/.env.gpu" | grep -v "^#" >> "$MCP_ENV_FILE" || true
    fi
else
    log_warning "GPU optimization setup failed (non-critical, continuing...)"
    log_installation "GPU Optimization: WARNING"
fi
echo ""

# Create activation script
log_step "Creating activation script..."

ACTIVATE_SCRIPT="$PROJECT_ROOT/activate_native.sh"
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Activate Native Backend Environment with GPU Optimizations

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Activating Native Backend Environment..."
echo ""

# Load GPU optimizations if available
if [ -f "$PROJECT_ROOT/.env.gpu" ]; then
    echo "[GPU] Loading GPU optimizations..."
    set -a
    source "$PROJECT_ROOT/.env.gpu"
    set +a
    echo "✓ GPU config loaded: $GPU_TYPE (count: $GPU_COUNT)"
    echo ""
fi

# Load MCP environment
if [ -f "$PROJECT_ROOT/mcp-server/.env.native" ]; then
    export $(cat "$PROJECT_ROOT/mcp-server/.env.native" | grep -v '^#' | xargs)
    echo "✓ MCP environment loaded"
fi

# Show available tools
echo ""
echo "Available tools:"

if [ -f "$PROJECT_ROOT/tools/generated/alphafold2/activate.sh" ] || [ -f "$PROJECT_ROOT/tools/alphafold2/activate.sh" ]; then
    echo "  • AlphaFold2"
    if [ -f "$PROJECT_ROOT/tools/generated/alphafold2/activate.sh" ]; then
        echo "    Activate: source $PROJECT_ROOT/tools/generated/alphafold2/activate.sh"
    else
        echo "    Activate: source $PROJECT_ROOT/tools/alphafold2/activate.sh"
    fi
fi

if [ -f "$PROJECT_ROOT/tools/generated/rfdiffusion/activate.sh" ] || [ -f "$PROJECT_ROOT/tools/rfdiffusion/activate.sh" ]; then
    echo "  • RFDiffusion"
    if [ -f "$PROJECT_ROOT/tools/generated/rfdiffusion/activate.sh" ]; then
        echo "    Activate: source $PROJECT_ROOT/tools/generated/rfdiffusion/activate.sh"
    else
        echo "    Activate: source $PROJECT_ROOT/tools/rfdiffusion/activate.sh"
    fi
fi

if [ -d "$PROJECT_ROOT/tools/proteinmpnn" ]; then
    echo "  • ProteinMPNN"
    echo "    Activate: conda activate proteinmpnn_arm64"
fi

echo ""
echo "To start MCP server with native backend:"
echo "  cd $PROJECT_ROOT/mcp-server"
echo "  MODEL_BACKEND=native python server.py"
echo ""
EOF

chmod +x "$ACTIVATE_SCRIPT"
log_success "Activation script created: $ACTIVATE_SCRIPT"

# Calculate installation time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

log_installation "=== Installation Completed in ${MINUTES}m ${SECONDS}s ==="

# Final summary
echo ""
log_header "=================================================================="
log_header "  ✓ Installation Complete!"
log_header "=================================================================="
echo ""
echo "Installation Summary:"
echo "  Duration: ${MINUTES}m ${SECONDS}s"
echo "  Log: $INSTALLATION_LOG"
echo ""

if [ "$INSTALL_ALPHAFOLD" = true ]; then
    echo "AlphaFold2:"
    echo "  Location: $PROJECT_ROOT/tools/alphafold2"
    echo "  Database: $DB_TIER"
    if [ -f "$PROJECT_ROOT/tools/generated/alphafold2/activate.sh" ]; then
        echo "  Activate: source $PROJECT_ROOT/tools/generated/alphafold2/activate.sh"
    else
        echo "  Activate: source $PROJECT_ROOT/tools/alphafold2/activate.sh"
    fi
    
    # Show MMseqs2 database info if AlphaFold2 installed
    MMSEQS2_DB="${HOME}/.cache/alphafold/mmseqs2/uniref90_db"
    if [ -f "${MMSEQS2_DB}.dbtype" ] || [ -d "$MMSEQS2_DB" ]; then
        echo "  MMseqs2 Database: $MMSEQS2_DB"
        echo "  MSA Mode: mmseqs2 (optimized) or jackhmmer"
    fi
    echo ""
fi

if [ "$INSTALL_RFDIFFUSION" = true ]; then
    echo "RFDiffusion:"
    echo "  Location: $PROJECT_ROOT/tools/rfdiffusion"
    if [ -f "$PROJECT_ROOT/tools/generated/rfdiffusion/activate.sh" ]; then
        echo "  Activate: source $PROJECT_ROOT/tools/generated/rfdiffusion/activate.sh"
    else
        echo "  Activate: source $PROJECT_ROOT/tools/rfdiffusion/activate.sh"
    fi
    echo ""
fi

if [ "$INSTALL_PROTEINMPNN" = true ]; then
    echo "ProteinMPNN:"
    echo "  Location: $PROJECT_ROOT/tools/proteinmpnn"
    echo "  Activate: conda activate proteinmpnn_arm64"
    echo ""
fi

if [ "$INSTALL_MCP" = true ]; then
    echo "MCP Server:"
    echo "  Configuration: $PROJECT_ROOT/mcp-server/.env.native"
    echo "  Start: cd mcp-server && MODEL_BACKEND=native python server.py"
    echo ""
fi

echo "Quick Start:"
echo "  1. Activate environment:"
echo "     source $ACTIVATE_SCRIPT"
echo ""
echo "  2. Start native model services:"
echo "     bash $PROJECT_ROOT/scripts/run_arm64_native_model_services.sh"
echo ""
echo "  3. Start dashboard:"
echo "     $PROJECT_ROOT/scripts/run_dashboard_stack.sh --arm64-host-native up"
echo ""
echo "  4. Open dashboard: http://localhost:3000"
echo ""

# Show next steps based on database tier
case $DB_TIER in
    minimal)
        log_warning "Note: Minimal installation provides demo-quality results only"
        echo "  For better accuracy, reinstall with: $0 --recommended"
        ;;
    reduced)
        log_info "Reduced databases provide 70-80% accuracy - suitable for development"
        echo "  For production, reinstall with: $0 --full"
        ;;
    full)
        log_success "Full databases installed - production ready!"
        ;;
esac

echo ""
log_success "Ready to use! See documentation at $PROJECT_ROOT/docs/"
echo ""
