#!/bin/bash

set -e

echo "==========================================="
echo "PyTorch ARM64 Build for NVIDIA GB10 GPU"
echo "CUDA Capability 12.1 (Blackwell)"
echo "==========================================="

# Build configuration
BUILD_DIR="$HOME/pytorch_build_gb10_fixed"
BUILD_THREADS=19

# Install dependencies
echo "Installing Python dependencies (ARM64 optimized)..."
pip install numpy pyyaml setuptools cmake cffi typing_extensions

echo "Installing system dependencies for OpenBLAS..."
sudo apt-get update
sudo apt-get install -y libopenblas-dev liblapack-dev gfortran

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone PyTorch
if [ ! -d "pytorch" ]; then
    echo "Cloning PyTorch repository..."
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
else
    echo "PyTorch repository already exists, updating..."
    cd pytorch
    git pull
    git submodule update --init --recursive
fi

# Set build environment variables for GB10 GPU (CUDA Capability 12.1)
export CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)')
export USE_CUDA=1
export USE_CUDNN=1
export USE_MKL=0
export USE_MKLDNN=0
export BUILD_TEST=0
export USE_FBGEMM=0
export USE_KINETO=1
export USE_DISTRIBUTED=1
export MAX_JOBS=$BUILD_THREADS

# GB10 (Blackwell) specific settings - CUDA Capability 12.1
# Include compute_90 for Ada Lovelace compatibility and add compute_121 for Blackwell
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 12.1"
export CUDA_HOME=/usr/local/cuda

# Use OpenBLAS instead of MKL for ARM64
export USE_BLAS=1
export BLAS=OpenBLAS
export OpenBLAS_HOME=/usr
export OpenBLAS_INCLUDE_DIR=/usr/include/aarch64-linux-gnu
export OpenBLAS_LIB=/usr/lib/aarch64-linux-gnu/libopenblas.so

# Disable Flash Attention to avoid compute_70 issues
export USE_FLASH_ATTENTION=0

# NCCL configuration for GB10 GPU (including Blackwell support)
export NCCL_CUDA_ARCH_LIST="8.0 8.6 9.0 12.1"
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_121,code=sm_121"

echo "Build environment for GB10 (Blackwell CUDA 12.1):"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  USE_CUDA: $USE_CUDA"
echo "  USE_MKL: $USE_MKL"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  NCCL_CUDA_ARCH_LIST: $NCCL_CUDA_ARCH_LIST"
echo "  NVCC_GENCODE: $NVCC_GENCODE"
echo "  USE_FLASH_ATTENTION: $USE_FLASH_ATTENTION"
echo "  MAX_JOBS: $MAX_JOBS"

# Apply patches for NCCL and Flash Attention if needed
echo "Applying GB10 GPU compatibility patches..."

# Create a custom NCCL patch script
cat > patch_nccl_gb10.py << 'EOF'
#!/usr/bin/env python3
import os
import re

def patch_nccl_makefiles():
    """Patch NCCL makefiles to use only supported CUDA architectures including Blackwell."""
    nccl_dir = "third_party/nccl/nccl"
    if not os.path.exists(nccl_dir):
        print("NCCL directory not found, skipping patch")
        return
    
    # Find all Makefiles in NCCL
    for root, dirs, files in os.walk(nccl_dir):
        for file in files:
            if file in ['Makefile', 'rules.mk', 'Makefile.*']:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Replace unsupported compute capabilities with supported ones
                    updated_content = re.sub(
                        r'compute_70[^,\s]*', 'compute_80', content
                    )
                    updated_content = re.sub(
                        r'compute_75[^,\s]*', 'compute_86', updated_content
                    )
                    updated_content = re.sub(
                        r'sm_70[^,\s]*', 'sm_80', updated_content
                    )
                    updated_content = re.sub(
                        r'sm_75[^,\s]*', 'sm_86', updated_content
                    )
                    
                    # Add Blackwell support
                    if 'compute_90' in updated_content and 'compute_121' not in updated_content:
                        updated_content = updated_content.replace(
                            'compute_90', 'compute_90 -gencode=arch=compute_121,code=sm_121'
                        )
                    
                    if content != updated_content:
                        with open(filepath, 'w') as f:
                            f.write(updated_content)
                        print(f"Patched: {filepath}")
                except Exception as e:
                    print(f"Failed to patch {filepath}: {e}")

if __name__ == "__main__":
    patch_nccl_makefiles()
EOF

python3 patch_nccl_gb10.py

echo "Building PyTorch for GB10 GPU with Blackwell support..."
echo "This may take 1-3 hours depending on your system..."

# Clean build
python3 setup.py clean
rm -rf build/

# Start build
python3 setup.py bdist_wheel

echo ""
echo "==========================================="
echo "Build completed!"
echo "==========================================="

# Check if build was successful
if [ -d "dist" ] && [ "$(ls -A dist/*.whl 2>/dev/null)" ]; then
    echo "✅ PyTorch wheel built successfully:"
    ls -la dist/*.whl
    echo ""
    echo "To install: pip install dist/*.whl"
else
    echo "❌ Build failed - no wheel file found"
    exit 1
fi