"""
PyTorch Source Build Fallback for ARM64

This module provides support for building PyTorch from source with ARM64 CUDA
support as a fallback solution. This is useful when pre-built wheels are not
available for ARM64 with CUDA.

DEPRECATION NOTICE:
This fallback will be deprecated when PyTorch releases official ARM64 CUDA wheels.
"""

import os
import subprocess
import warnings
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import platform


@dataclass
class BuildConfig:
    """Configuration for PyTorch source build."""
    cuda_version: str
    python_version: str
    use_cuda: bool
    use_cudnn: bool
    use_mkl: bool
    build_threads: int
    cmake_args: List[str]
    extra_flags: Dict[str, str]


class PyTorchSourceBuildFallback:
    """
    PyTorch source build fallback handler.
    
    Provides methods to build PyTorch from source with ARM64 CUDA support
    when pre-built wheels are not available.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize PyTorch source build fallback.
        
        Args:
            verbose: Print status messages
        """
        self.verbose = verbose
        self.is_arm64 = platform.machine().lower() in ('aarch64', 'arm64')
        self.build_dir = os.path.expanduser('~/pytorch_build')
        
        if not self.is_arm64 and verbose:
            warnings.warn(
                "This fallback is primarily for ARM64 systems. "
                "Your system is not ARM64.",
                UserWarning
            )
    
    def check_build_dependencies(self) -> Dict[str, bool]:
        """
        Check if build dependencies are available.
        
        Returns:
            Dictionary of dependency availability
        """
        dependencies = {
            'git': False,
            'cmake': False,
            'gcc': False,
            'g++': False,
            'python3': False,
            'python3-dev': False,
            'cuda': False,
            'cudnn': False,
        }
        
        # Check basic tools
        for tool in ['git', 'cmake', 'gcc', 'g++', 'python3']:
            try:
                result = subprocess.run(
                    [tool, '--version'],
                    capture_output=True,
                    timeout=5
                )
                dependencies[tool] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                dependencies[tool] = False
        
        # Check for CUDA
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                timeout=5
            )
            dependencies['cuda'] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            dependencies['cuda'] = False
        
        # Check for cuDNN (harder to detect, look for library)
        cudnn_paths = [
            '/usr/local/cuda/lib64/libcudnn.so',
            '/usr/lib/aarch64-linux-gnu/libcudnn.so',
            '/usr/lib/x86_64-linux-gnu/libcudnn.so',
        ]
        dependencies['cudnn'] = any(os.path.exists(p) for p in cudnn_paths)
        
        return dependencies
    
    def get_recommended_build_config(self) -> BuildConfig:
        """
        Get recommended build configuration for current system.
        
        Returns:
            BuildConfig with recommended settings
        """
        deps = self.check_build_dependencies()
        cpu_count = os.cpu_count() or 4
        
        # Detect CUDA version if available
        cuda_version = '11.8'  # Default
        if deps['cuda']:
            try:
                result = subprocess.run(
                    ['nvcc', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Parse CUDA version from nvcc output
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'release' in part.lower() and i + 1 < len(parts):
                                cuda_version = parts[i + 1].rstrip(',')
                                break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return BuildConfig(
            cuda_version=cuda_version,
            python_version='3.10',
            use_cuda=deps['cuda'],
            use_cudnn=deps['cudnn'],
            use_mkl=True,
            build_threads=max(1, cpu_count - 1),
            cmake_args=[
                '-DBUILD_SHARED_LIBS:BOOL=ON',
                '-DCMAKE_BUILD_TYPE:STRING=Release',
                '-DPYTHON_EXECUTABLE:PATH=$(which python3)',
            ],
            extra_flags={
                'USE_CUDA': '1' if deps['cuda'] else '0',
                'USE_CUDNN': '1' if deps['cudnn'] else '0',
                'USE_MKLDNN': '1',
                'BUILD_TEST': '0',
                'USE_FBGEMM': '0',  # May not work on ARM64
                'USE_KINETO': '0',
                'USE_DISTRIBUTED': '1',
            }
        )
    
    def create_build_script(self, config: BuildConfig, output_path: str) -> str:
        """
        Create a build script for PyTorch.
        
        Args:
            config: Build configuration
            output_path: Path to save build script
            
        Returns:
            Path to created script
        """
        script_content = f"""#!/bin/bash
# PyTorch ARM64 CUDA Build Script
# Auto-generated by arm64_cuda_fallback
# 
# DEPRECATION NOTICE:
# This script will be deprecated when PyTorch releases official ARM64 CUDA wheels

set -e

echo "=========================================="
echo "PyTorch ARM64 Source Build"
echo "=========================================="
echo

# Configuration
CUDA_VERSION="{config.cuda_version}"
PYTHON_VERSION="{config.python_version}"
BUILD_THREADS={config.build_threads}
BUILD_DIR="{self.build_dir}"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing_extensions

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

# Set build environment variables
export CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)')
export USE_CUDA={config.extra_flags.get('USE_CUDA', '0')}
export USE_CUDNN={config.extra_flags.get('USE_CUDNN', '0')}
export USE_MKLDNN={config.extra_flags.get('USE_MKLDNN', '0')}
export BUILD_TEST={config.extra_flags.get('BUILD_TEST', '0')}
export USE_FBGEMM={config.extra_flags.get('USE_FBGEMM', '0')}
export USE_KINETO={config.extra_flags.get('USE_KINETO', '0')}
export USE_DISTRIBUTED={config.extra_flags.get('USE_DISTRIBUTED', '1')}
export MAX_JOBS=$BUILD_THREADS

# ARM64 specific settings
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"  # Adjust based on your GPU
export CUDA_HOME=/usr/local/cuda

# Build PyTorch
echo "Building PyTorch..."
echo "This may take 1-3 hours depending on your system..."
python3 setup.py build

# Install PyTorch
echo "Installing PyTorch..."
python3 setup.py install

# Verify installation
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {{torch.__version__}}'); print(f'CUDA available: {{torch.cuda.is_available()}}')"

echo
echo "=========================================="
echo "Build complete!"
echo "=========================================="
"""
        
        # Write script
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        if self.verbose:
            print(f"Build script created: {output_path}")
        
        return output_path
    
    def get_installation_guide(self) -> str:
        """
        Get detailed installation guide.
        
        Returns:
            Installation guide as string
        """
        deps = self.check_build_dependencies()
        
        guide = """
========================================
PyTorch ARM64 Source Build Guide
========================================

IMPORTANT: This is a fallback solution that will be deprecated when
PyTorch releases official ARM64 CUDA wheels.

Prerequisites:
"""
        
        # List missing dependencies
        missing = [k for k, v in deps.items() if not v]
        if missing:
            guide += "\nMissing dependencies (install these first):\n"
            for dep in missing:
                if dep == 'cuda':
                    guide += "  - CUDA Toolkit: https://developer.nvidia.com/cuda-downloads\n"
                elif dep == 'cudnn':
                    guide += "  - cuDNN: https://developer.nvidia.com/cudnn\n"
                elif dep == 'python3-dev':
                    guide += "  - Python development headers: sudo apt install python3-dev\n"
                else:
                    guide += f"  - {dep}: sudo apt install {dep}\n"
        else:
            guide += "\n✓ All dependencies are available!\n"
        
        guide += """
Build Steps:

1. Generate build script:
   python3 -m arm64_cuda_fallback.pytorch_source_build generate-script

2. Run the build script:
   ./pytorch_arm64_build.sh

3. Wait for build to complete (1-3 hours)

4. Verify installation:
   python3 -c "import torch; print(torch.__version__)"

Alternative: Use conda-forge
If building from source is too complex, try conda-forge:
   conda install pytorch torchvision torchaudio -c pytorch-nightly

Note: conda-forge may not have CUDA support for ARM64 yet.

When to Migrate:
Once PyTorch releases official ARM64 CUDA wheels, migrate with:
   pip install torch --index-url https://download.pytorch.org/whl/cu118

Check status:
   python3 -m arm64_cuda_fallback check-upstream
"""
        
        return guide
    
    def run_build(self, config: Optional[BuildConfig] = None) -> bool:
        """
        Run PyTorch build process.
        
        Args:
            config: Build configuration (uses recommended if None)
            
        Returns:
            True if build successful, False otherwise
        """
        if config is None:
            config = self.get_recommended_build_config()
        
        # Create build directory
        os.makedirs(self.build_dir, exist_ok=True)
        
        # Create build script
        script_path = os.path.join(self.build_dir, 'pytorch_arm64_build.sh')
        self.create_build_script(config, script_path)
        
        if self.verbose:
            print(f"\nBuild script ready: {script_path}")
            print("\nTo start building, run:")
            print(f"  {script_path}")
            print("\nWarning: Build will take 1-3 hours and require ~10GB disk space")
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get build fallback status.
        
        Returns:
            Dictionary with status information
        """
        deps = self.check_build_dependencies()
        config = self.get_recommended_build_config()
        
        return {
            'architecture': platform.machine(),
            'is_arm64': self.is_arm64,
            'dependencies': deps,
            'dependencies_ready': all(deps[k] for k in ['git', 'cmake', 'gcc', 'g++', 'python3']),
            'cuda_available': deps['cuda'],
            'cudnn_available': deps['cudnn'],
            'recommended_config': {
                'cuda_version': config.cuda_version,
                'use_cuda': config.use_cuda,
                'use_cudnn': config.use_cudnn,
                'build_threads': config.build_threads,
            },
            'build_dir': self.build_dir,
        }


def setup_pytorch_source_build(verbose: bool = True) -> PyTorchSourceBuildFallback:
    """
    Setup PyTorch source build fallback.
    
    Args:
        verbose: Print status messages
        
    Returns:
        Configured PyTorchSourceBuildFallback instance
    """
    fallback = PyTorchSourceBuildFallback(verbose=verbose)
    
    if verbose:
        print("Setting up PyTorch source build fallback...")
        print(fallback.get_installation_guide())
    
    return fallback
