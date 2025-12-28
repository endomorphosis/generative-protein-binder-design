#!/bin/bash
# Quick smoke test for GPU + MMseqs2 integration
# Tests that all components are accessible and working

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}GPU + MMseqs2 Integration Smoke Test${NC}"
echo "======================================"
echo ""

# Test 1: GPU availability
echo -n "Test 1: GPU detection... "
if nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    echo -e "${GREEN}PASS${NC} ($GPU_NAME)"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 2: CUDA version
echo -n "Test 2: CUDA 13.1... "
if nvcc --version 2>&1 | grep -q "release 13"; then
    CUDA_VER=$(nvcc --version 2>&1 | grep -oP "release \K[0-9.]+")
    echo -e "${GREEN}PASS${NC} (version $CUDA_VER)"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 3: MMseqs2 binary
echo -n "Test 3: MMseqs2 installation... "
if command -v mmseqs >/dev/null 2>&1; then
    echo -e "${GREEN}PASS${NC} ($(which mmseqs))"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 4: MMseqs2 databases
echo -n "Test 4: MMseqs2 databases... "
if [[ -f "$HOME/.cache/alphafold/mmseqs2/uniref90_db.dbtype" ]]; then
    DB_COUNT=$(ls -1 "$HOME/.cache/alphafold/mmseqs2"/*.dbtype 2>/dev/null | wc -l)
    echo -e "${GREEN}PASS${NC} ($DB_COUNT databases)"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 5: GPU environment config
echo -n "Test 5: GPU configuration... "
if [[ -f ".env.gpu" ]] && grep -q "GPU_TYPE=cuda" .env.gpu; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 6: Conda alphafold2 environment
echo -n "Test 6: AlphaFold2 conda environment... "
if conda env list 2>/dev/null | grep -q "alphafold2"; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 7: JAX GPU backend (requires conda activation)
echo -n "Test 7: JAX GPU backend... "
eval "$(conda shell.bash hook)" 2>/dev/null || true
if conda activate alphafold2 2>/dev/null; then
    if python -c "import jax; assert jax.default_backend() == 'gpu', 'Not GPU backend'" 2>/dev/null; then
        echo -e "${GREEN}PASS${NC}"
        conda deactivate 2>/dev/null || true
    else
        echo -e "${RED}FAIL${NC}"
        conda deactivate 2>/dev/null || true
        exit 1
    fi
else
    echo -e "${RED}FAIL${NC} (could not activate environment)"
    exit 1
fi

# Test 8: Docker GPU compose
echo -n "Test 8: Docker GPU configuration... "
if [[ -f "deploy/docker-compose-gpu-optimized.yaml" ]] && \
   grep -q "nvidia" deploy/docker-compose-gpu-optimized.yaml; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 9: Installation scripts
echo -n "Test 9: Zero-touch installer... "
if [[ -f "scripts/install_all_native.sh" ]] && \
   grep -q "mmseqs2" scripts/install_all_native.sh; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 10: Documentation
echo -n "Test 10: Documentation... "
if [[ -f "docs/GPU_OPTIMIZATION_INTEGRATION.md" ]] && \
   [[ -f "docs/MMSEQS2_ZERO_TOUCH_IMPLEMENTATION.md" ]]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All smoke tests passed!${NC}"
echo ""
echo "System is ready for:"
echo "  • GPU-accelerated AlphaFold inference"
echo "  • MMseqs2-based fast MSA generation"
echo "  • CUDA 13.1 optimized operations"
echo "  • Zero-touch deployments"
echo ""
