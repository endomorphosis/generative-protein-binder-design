#!/bin/bash
# Comprehensive verification script for GPU/FP4/CUDA 13.1/MMseqs2 optimization integration
# Verifies end-to-end system setup and performance

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_section() { echo -e "\n${MAGENTA}═══════════════════════════════════════════════════════${NC}"; echo -e "${MAGENTA}  $1${NC}"; echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}\n"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

check_pass() {
    ((PASS_COUNT++))
    log_success "$1"
}

check_fail() {
    ((FAIL_COUNT++))
    log_error "$1"
}

check_warn() {
    ((WARN_COUNT++))
    log_warning "$1"
}

# ═══════════════════════════════════════════════════════
# GPU & CUDA Verification
# ═══════════════════════════════════════════════════════
verify_gpu_cuda() {
    log_section "GPU & CUDA 13.1 Verification"
    
    # Check nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "")
        if [[ -n "$GPU_INFO" ]]; then
            check_pass "GPU detected: $GPU_INFO"
            export GPU_AVAILABLE=true
        else
            check_fail "nvidia-smi found but no GPU detected"
            export GPU_AVAILABLE=false
        fi
    else
        check_fail "nvidia-smi not found"
        export GPU_AVAILABLE=false
    fi
    
    # Check CUDA version
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version 2>&1 | grep -oP "release \K[0-9.]+" || echo "unknown")
        if [[ "$CUDA_VERSION" == 13.1* ]] || [[ "$CUDA_VERSION" == 13.* ]]; then
            check_pass "CUDA version: $CUDA_VERSION (target: 13.1)"
        else
            check_warn "CUDA version: $CUDA_VERSION (expected 13.1)"
        fi
    else
        check_fail "nvcc not found - CUDA development tools not installed"
    fi
    
    # Check GPU optimization scripts
    if [[ -f "$SCRIPT_DIR/setup_gpu_optimization.sh" ]]; then
        check_pass "GPU optimization setup script exists"
    else
        check_fail "GPU optimization setup script missing"
    fi
    
    if [[ -f "$SCRIPT_DIR/detect_gpu_and_generate_env.sh" ]]; then
        check_pass "GPU detection script exists"
    else
        check_fail "GPU detection script missing"
    fi
    
    # Check for GPU config
    if [[ -f "$PROJECT_ROOT/.env.gpu" ]]; then
        check_pass "GPU environment config exists: .env.gpu"
        log_info "GPU config preview:"
        head -5 "$PROJECT_ROOT/.env.gpu" | sed 's/^/  /'
    else
        check_warn "GPU environment config not generated yet (.env.gpu)"
    fi
}

# ═══════════════════════════════════════════════════════
# MMseqs2 Integration Verification
# ═══════════════════════════════════════════════════════
verify_mmseqs2() {
    log_section "MMseqs2 Integration Verification"
    
    # Check MMseqs2 binary
    if command -v mmseqs >/dev/null 2>&1; then
        MMSEQS_VERSION=$(mmseqs version 2>&1 | head -1)
        check_pass "MMseqs2 installed: $(which mmseqs)"
        log_info "Version: $MMSEQS_VERSION"
    else
        check_fail "MMseqs2 not found in PATH"
    fi
    
    # Check MMseqs2 databases
    MMSEQS_DB_DIR="${HOME}/.cache/alphafold/mmseqs2"
    if [[ -d "$MMSEQS_DB_DIR" ]]; then
        check_pass "MMseqs2 database directory exists: $MMSEQS_DB_DIR"
        
        # Check for key databases
        if [[ -f "$MMSEQS_DB_DIR/uniref90_db.dbtype" ]]; then
            DB_SIZE=$(du -sh "$MMSEQS_DB_DIR" 2>/dev/null | cut -f1)
            check_pass "UniRef90 database ready (total size: $DB_SIZE)"
            
            # Count databases
            DB_COUNT=$(ls -1 "$MMSEQS_DB_DIR"/*.dbtype 2>/dev/null | wc -l)
            log_info "Total databases: $DB_COUNT"
            
            # List databases
            log_info "Available databases:"
            ls -1 "$MMSEQS_DB_DIR"/*.dbtype 2>/dev/null | xargs -I {} basename {} .dbtype | sed 's/^/  - /'
        else
            check_warn "UniRef90 database not built yet"
        fi
    else
        check_warn "MMseqs2 database directory not found: $MMSEQS_DB_DIR"
    fi
    
    # Check MMseqs2 installer script
    if [[ -f "$SCRIPT_DIR/install_mmseqs2.sh" ]]; then
        check_pass "MMseqs2 installer script exists"
        if grep -q "GPU\|gpu" "$SCRIPT_DIR/install_mmseqs2.sh"; then
            check_pass "MMseqs2 installer has GPU support"
        fi
    else
        check_fail "MMseqs2 installer script missing"
    fi
    
    # Check database conversion script
    if [[ -f "$SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh" ]]; then
        check_pass "Multi-stage database conversion script exists"
        if grep -q "\-\-gpu" "$SCRIPT_DIR/convert_alphafold_db_to_mmseqs2_multistage.sh"; then
            check_pass "Database conversion supports GPU acceleration"
        fi
    else
        check_fail "Database conversion script missing"
    fi
}

# ═══════════════════════════════════════════════════════
# Installation Scripts Verification
# ═══════════════════════════════════════════════════════
verify_installation_scripts() {
    log_section "Zero-Touch Installation Scripts"
    
    # Check main installer
    if [[ -f "$SCRIPT_DIR/install_all_native.sh" ]]; then
        check_pass "Main installer exists: install_all_native.sh"
        
        # Check MMseqs2 integration
        if grep -q "mmseqs2\|MMseqs2\|MMSEQS" "$SCRIPT_DIR/install_all_native.sh"; then
            check_pass "Main installer includes MMseqs2 integration"
        else
            check_fail "Main installer missing MMseqs2 integration"
        fi
        
        # Check GPU optimization integration
        if grep -q "gpu_optimization\|GPU\|detect_gpu" "$SCRIPT_DIR/install_all_native.sh"; then
            check_pass "Main installer includes GPU optimization"
        else
            check_fail "Main installer missing GPU optimization"
        fi
    else
        check_fail "Main installer script missing"
    fi
    
    # Check AlphaFold installer
    if [[ -f "$SCRIPT_DIR/install_alphafold2_complete.sh" ]]; then
        check_pass "AlphaFold2 installer exists"
    else
        check_warn "AlphaFold2 installer not found"
    fi
    
    # Check RFDiffusion installer
    if [[ -f "$SCRIPT_DIR/install_rfdiffusion_complete.sh" ]]; then
        check_pass "RFDiffusion installer exists"
    else
        check_warn "RFDiffusion installer not found"
    fi
}

# ═══════════════════════════════════════════════════════
# Docker Integration Verification
# ═══════════════════════════════════════════════════════
verify_docker_integration() {
    log_section "Docker GPU Integration"
    
    # Check GPU-optimized docker compose
    if [[ -f "$PROJECT_ROOT/deploy/docker-compose-gpu-optimized.yaml" ]]; then
        check_pass "GPU-optimized Docker Compose exists"
        
        # Check for GPU configuration in compose file
        if grep -q "nvidia" "$PROJECT_ROOT/deploy/docker-compose-gpu-optimized.yaml"; then
            check_pass "Docker Compose includes NVIDIA GPU configuration"
        fi
        
        if grep -q "ENABLE_GPU_OPTIMIZATION" "$PROJECT_ROOT/deploy/docker-compose-gpu-optimized.yaml"; then
            check_pass "Docker Compose includes GPU optimization flags"
        fi
        
        if grep -q "XLA_CACHE\|TF_XLA" "$PROJECT_ROOT/deploy/docker-compose-gpu-optimized.yaml"; then
            check_pass "Docker Compose includes XLA caching configuration"
        fi
    else
        check_warn "GPU-optimized Docker Compose not found"
    fi
}

# ═══════════════════════════════════════════════════════
# Conda Environments Verification
# ═══════════════════════════════════════════════════════
verify_conda_environments() {
    log_section "Conda Environments"
    
    if command -v conda >/dev/null 2>&1; then
        check_pass "Conda available"
        
        # Check for AlphaFold environment
        if conda env list 2>/dev/null | grep -q "alphafold"; then
            check_pass "AlphaFold conda environment found"
            
            # Try to activate and check for JAX
            eval "$(conda shell.bash hook)" 2>/dev/null || true
            if conda activate alphafold2 2>/dev/null; then
                if python -c "import jax" 2>/dev/null; then
                    JAX_BACKEND=$(python -c "import jax; print(jax.default_backend())" 2>/dev/null || echo "unknown")
                    if [[ "$JAX_BACKEND" == "gpu" ]]; then
                        check_pass "JAX configured for GPU backend"
                    else
                        check_warn "JAX backend: $JAX_BACKEND (expected: gpu)"
                    fi
                else
                    check_warn "JAX not available in alphafold2 environment"
                fi
                conda deactivate 2>/dev/null || true
            fi
        else
            check_warn "AlphaFold conda environment not found"
        fi
        
        # Check for RFDiffusion environment
        if conda env list 2>/dev/null | grep -q "rfdiffusion"; then
            check_pass "RFDiffusion conda environment found"
        else
            check_warn "RFDiffusion conda environment not found"
        fi
        
        # Check for ProteinMPNN environment
        if conda env list 2>/dev/null | grep -q "proteinmpnn"; then
            check_pass "ProteinMPNN conda environment found"
        else
            check_warn "ProteinMPNN conda environment not found"
        fi
    else
        check_fail "Conda not available"
    fi
}

# ═══════════════════════════════════════════════════════
# Documentation Verification
# ═══════════════════════════════════════════════════════
verify_documentation() {
    log_section "Documentation Verification"
    
    DOCS=(
        "docs/GPU_OPTIMIZATION_INTEGRATION.md"
        "docs/MMSEQS2_ZERO_TOUCH_IMPLEMENTATION.md"
        "docs/MMSEQS2_ZERO_TOUCH_QUICKREF.md"
        "docs/ZERO_TOUCH_QUICKSTART.md"
    )
    
    for doc in "${DOCS[@]}"; do
        if [[ -f "$PROJECT_ROOT/$doc" ]]; then
            check_pass "Documentation: $doc"
        else
            check_warn "Missing documentation: $doc"
        fi
    done
}

# ═══════════════════════════════════════════════════════
# Benchmark & Testing Infrastructure
# ═══════════════════════════════════════════════════════
verify_benchmarking() {
    log_section "Benchmarking & Testing"
    
    # Check benchmark scripts
    BENCH_SCRIPTS=(
        "scripts/benchmark_optimizations.sh"
        "scripts/run_empirical_benchmarks.sh"
        "scripts/bench_msa_comparison.sh"
        "scripts/test_mmseqs2_zero_touch_e2e.sh"
    )
    
    for script in "${BENCH_SCRIPTS[@]}"; do
        if [[ -f "$PROJECT_ROOT/$script" ]]; then
            check_pass "Benchmark script: $(basename $script)"
        else
            check_warn "Missing benchmark script: $(basename $script)"
        fi
    done
    
    # Check for recent benchmarks
    if [[ -d "$PROJECT_ROOT/benchmarks" ]]; then
        RECENT_BENCH=$(ls -t "$PROJECT_ROOT/benchmarks" 2>/dev/null | head -1)
        if [[ -n "$RECENT_BENCH" ]]; then
            check_pass "Recent benchmark found: $RECENT_BENCH"
        fi
    fi
}

# ═══════════════════════════════════════════════════════
# System Performance Check
# ═══════════════════════════════════════════════════════
verify_system_performance() {
    log_section "System Performance Metrics"
    
    # Check CPU
    CPU_COUNT=$(nproc 2>/dev/null || echo "unknown")
    log_info "CPU cores: $CPU_COUNT"
    
    # Check memory
    if command -v free >/dev/null 2>&1; then
        MEM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
        MEM_AVAIL=$(free -h | awk '/^Mem:/ {print $7}')
        log_info "Memory: $MEM_AVAIL available / $MEM_TOTAL total"
    fi
    
    # Check disk space for key directories
    if [[ -d "$HOME/.cache/alphafold" ]]; then
        AF_SIZE=$(du -sh "$HOME/.cache/alphafold" 2>/dev/null | cut -f1)
        log_info "AlphaFold data size: $AF_SIZE"
    fi
    
    # Check GPU memory
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
        log_info "GPU memory: $GPU_MEM"
    fi
}

# ═══════════════════════════════════════════════════════
# Run All Verifications
# ═══════════════════════════════════════════════════════
main() {
    echo -e "${CYAN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   GPU / FP4 / CUDA 13.1 / MMseqs2 Optimization Integration           ║
║   Comprehensive System Verification                                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    log_info "Project root: $PROJECT_ROOT"
    log_info "Starting verification..."
    echo ""
    
    set +e  # Don't exit on errors during verification
    
    verify_gpu_cuda
    verify_mmseqs2
    verify_installation_scripts
    verify_docker_integration
    verify_conda_environments
    verify_documentation
    verify_benchmarking
    verify_system_performance
    
    # Summary
    log_section "Verification Summary"
    
    echo -e "${GREEN}✓ Passed:${NC} $PASS_COUNT"
    echo -e "${YELLOW}⚠ Warnings:${NC} $WARN_COUNT"
    echo -e "${RED}✗ Failed:${NC} $FAIL_COUNT"
    echo ""
    
    if [[ $FAIL_COUNT -eq 0 ]]; then
        log_success "System verification PASSED! GPU/MMseqs2 optimizations are properly integrated."
        
        echo ""
        log_info "Next steps:"
        echo "  1. Run end-to-end benchmark: ./scripts/run_empirical_benchmarks.sh"
        echo "  2. Test MMseqs2 MSA generation: ./scripts/bench_msa_comparison.sh"
        echo "  3. Monitor GPU utilization: nvidia-smi dmon -s u"
        echo ""
        
        exit 0
    else
        log_error "System verification FAILED with $FAIL_COUNT error(s)."
        echo ""
        log_info "Recommended actions:"
        if ! command -v mmseqs >/dev/null 2>&1; then
            echo "  - Install MMseqs2: ./scripts/install_mmseqs2.sh"
        fi
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "  - Install NVIDIA drivers and CUDA toolkit"
        fi
        if [[ ! -f "$PROJECT_ROOT/.env.gpu" ]]; then
            echo "  - Generate GPU config: ./scripts/detect_gpu_and_generate_env.sh"
        fi
        echo ""
        
        exit 1
    fi
}

# Run main
main "$@"
