#!/bin/bash
# GPU and CUDA Compatibility Validation Script for AlphaFold
# Validates GPU availability, CUDA versions, JAX/CUDA compatibility, and XLA configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}GPU/CUDA Compatibility Check${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check for nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found. NVIDIA GPU tools not installed.${NC}"
    echo "  Install NVIDIA CUDA Toolkit to enable GPU support."
    GPU_AVAILABLE=0
else
    echo -e "${GREEN}✓ nvidia-smi found${NC}"
    GPU_AVAILABLE=1
fi

echo ""
echo -e "${BLUE}GPU Status:${NC}"
if [ $GPU_AVAILABLE -eq 1 ]; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
        echo "  $line"
    done
else
    echo -e "${YELLOW}  No GPU detected - will use CPU${NC}"
fi

echo ""
echo -e "${BLUE}CUDA Toolkit Information:${NC}"

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    echo -e "  ${GREEN}✓ CUDA ${CUDA_VERSION}${NC}"
else
    echo -e "  ${YELLOW}✗ CUDA compiler (nvcc) not found${NC}"
    CUDA_VERSION="Unknown"
fi

# Check cuDNN
if [ -n "$CUDNN_PATH" ]; then
    CUDNN_VERSION=$(cat $CUDNN_PATH/include/cudnn.h 2>/dev/null | grep "CUDNN_MAJOR" -A 2 | head -1 | awk '{print $3}' || echo "Unknown")
    echo -e "  ${GREEN}✓ cuDNN found at $CUDNN_PATH${NC}"
else
    echo -e "  ${YELLOW}✗ cuDNN path not set. Set CUDNN_PATH environment variable.${NC}"
fi

# Check NVIDIA driver
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
echo -e "  NVIDIA Driver: ${GREEN}${DRIVER_VERSION}${NC}"

echo ""
echo -e "${BLUE}Python JAX Configuration:${NC}"

# Check if Python can import JAX and diagnose backend
python3 << 'EOF'
import sys
import os

try:
    import jax
    import jax.numpy as jnp
    print(f"  JAX version: {jax.__version__}")
    
    # Check available devices
    devices = jax.devices()
    print(f"  Available devices: {devices}")
    
    # Check for GPU devices
    gpu_devices = [d for d in devices if 'gpu' in str(d).lower()]
    if gpu_devices:
        print(f"  ✓ GPU devices detected: {len(gpu_devices)}")
        for gpu in gpu_devices:
            print(f"    - {gpu}")
    else:
        print(f"  ⚠ No GPU devices detected - using CPU")
    
    # Check JAX configuration
    print(f"  JAX 64-bit mode: {jax.config.jax_enable_x64}")
    print(f"  JAX default device: {jax.config.jax_default_device}")
    
    # Try a simple GPU computation
    try:
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        print(f"  ✓ Basic JAX GPU computation successful")
    except Exception as e:
        print(f"  ✗ JAX computation failed: {e}")
    
except ImportError as e:
    print(f"  ✗ JAX import failed: {e}")
    print(f"     Install JAX with: pip install jax[cuda]")
    sys.exit(1)
EOF

echo ""
echo -e "${BLUE}XLA/CUDA Environment:${NC}"

# Check XLA configuration
if [ -z "$XLA_FLAGS" ]; then
    echo -e "  ${YELLOW}⚠ XLA_FLAGS not set (will use defaults)${NC}"
else
    echo -e "  ${GREEN}✓ XLA_FLAGS set: $XLA_FLAGS${NC}"
fi

# Check GPU memory fraction
if [ -z "$XLA_PYTHON_CLIENT_MEM_FRACTION" ]; then
    echo -e "  ${YELLOW}⚠ XLA_PYTHON_CLIENT_MEM_FRACTION not set (will use dynamic allocation)${NC}"
else
    echo -e "  ${GREEN}✓ GPU memory fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION${NC}"
fi

# Check thread configuration
echo -e "  OMP_NUM_THREADS: ${OMP_NUM_THREADS:-not set (using default)}"
echo -e "  TF_NUM_INTRAOP_THREADS: ${TF_NUM_INTRAOP_THREADS:-not set (using default)}"
echo -e "  TF_NUM_INTEROP_THREADS: ${TF_NUM_INTEROP_THREADS:-not set (using default)}"

echo ""
echo -e "${BLUE}Recommendations:${NC}"

if [ $GPU_AVAILABLE -eq 1 ]; then
    if [ "$CUDA_VERSION" != "Unknown" ]; then
        echo -e "  ${GREEN}✓ GPU and CUDA available - optimal for AlphaFold inference${NC}"
    else
        echo -e "  ${YELLOW}⚠ GPU detected but CUDA compiler not found${NC}"
        echo "    Install CUDA Toolkit development tools: sudo apt install nvidia-cuda-toolkit"
    fi
    
    echo -e "  For optimal performance:"
    echo "    1. Enable XLA caching: export TF_XLA_CACHE_DIR=~/.cache/jax/xla_cache"
    echo "    2. Set GPU memory fraction: export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9"
    echo "    3. Configure threads: export OMP_NUM_THREADS=\$(nproc)"
    echo "    4. Run with JIT profiling: python run_alphafold.py --benchmark"
else
    echo -e "  ${YELLOW}⚠ No GPU detected - CPU mode will be slow${NC}"
    echo "    Install NVIDIA drivers and CUDA toolkit for GPU acceleration"
    echo "    Visit: https://developer.nvidia.com/cuda-downloads"
fi

echo ""
echo -e "${BLUE}Setup Script:${NC}"
echo "To apply recommended settings, run:"
echo "  source scripts/setup_gpu_optimization.sh"

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "Check complete!"
echo -e "${BLUE}================================${NC}"
