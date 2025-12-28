#!/bin/bash
# GPU Optimization Setup Script for AlphaFold
# Configures JAX, CUDA, XLA, and thread pool settings for optimal performance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "GPU Optimization Setup"
echo "================================"
echo ""

# Detect CPU count
CPU_COUNT=$(nproc 2>/dev/null || echo "4")
echo "Detected CPU cores: $CPU_COUNT"

# Setup XLA caching directory
XLA_CACHE_DIR="$HOME/.cache/jax/xla_cache"
mkdir -p "$XLA_CACHE_DIR"
echo "Created XLA cache directory: $XLA_CACHE_DIR"

# Function to export environment variable
export_var() {
    local name=$1
    local value=$2
    echo "export $name=$value"
    export "$name=$value"
}

echo ""
echo "Setting environment variables..."
echo ""

# XLA/CUDA Optimization
export_var "TF_XLA_CACHE_DIR" "$XLA_CACHE_DIR"
export_var "JAX_PLATFORMS" "gpu"
export_var "XLA_FLAGS" "--xla_gpu_fuse_operations=true --xla_gpu_kernel_lazy_compilation_threshold=10000 --xla_gpu_enable_cudnn_frontend=true"
export_var "XLA_PYTHON_CLIENT_MEM_FRACTION" "0.9"

# Thread Pool Configuration
export_var "OMP_NUM_THREADS" "$CPU_COUNT"
export_var "TF_NUM_INTRAOP_THREADS" "$CPU_COUNT"
export_var "TF_NUM_INTEROP_THREADS" "$((CPU_COUNT / 2))"

echo ""
echo "GPU Optimization environment variables configured."
echo ""
echo "To persist these settings, add to your ~/.bashrc:"
echo ""
echo "# AlphaFold GPU Optimization"
echo "export TF_XLA_CACHE_DIR=$XLA_CACHE_DIR"
echo "export JAX_PLATFORMS=gpu"
echo 'export XLA_FLAGS="--xla_gpu_fuse_operations=true --xla_gpu_kernel_lazy_compilation_threshold=10000 --xla_gpu_enable_cudnn_frontend=true"'
echo "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9"
echo "export OMP_NUM_THREADS=$CPU_COUNT"
echo "export TF_NUM_INTRAOP_THREADS=$CPU_COUNT"
echo "export TF_NUM_INTEROP_THREADS=$((CPU_COUNT / 2))"
echo ""

# Validate setup
echo "Validating setup..."
python3 << 'EOF'
import os
import sys

try:
    import jax
    import jax.numpy as jnp
    
    print("✓ JAX import successful")
    
    # Check devices
    devices = jax.devices()
    gpu_devices = [d for d in devices if 'gpu' in str(d).lower()]
    
    if gpu_devices:
        print(f"✓ GPU devices available: {len(gpu_devices)}")
        
        # Quick performance test
        @jax.jit
        def dummy_compute(x):
            return jnp.sum(x ** 2)
        
        x = jnp.ones((10000, 10000))
        result = dummy_compute(x)
        result.block_until_ready()
        print("✓ GPU computation test passed")
    else:
        print("⚠ No GPU devices detected - using CPU")
    
    print("\n✓ GPU optimization setup complete!")
    
except Exception as e:
    print(f"✗ Setup validation failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
