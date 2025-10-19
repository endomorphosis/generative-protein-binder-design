"""
CUDA Detection Module for ARM64

Detects CUDA availability and provides fallback recommendations.
"""

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional, List
import warnings


@dataclass
class DeviceInfo:
    """Information about the current device and CUDA availability."""
    architecture: str
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_count: int
    gpu_names: List[str]
    device_type: str  # 'cuda', 'cpu', 'mps'
    pytorch_available: bool
    jax_available: bool
    arm64_cuda_supported: bool
    
    def __str__(self) -> str:
        lines = [
            f"Architecture: {self.architecture}",
            f"CUDA Available: {self.cuda_available}",
            f"CUDA Version: {self.cuda_version or 'N/A'}",
            f"GPU Count: {self.gpu_count}",
            f"Device Type: {self.device_type}",
            f"PyTorch Available: {self.pytorch_available}",
            f"JAX Available: {self.jax_available}",
            f"ARM64 CUDA Supported: {self.arm64_cuda_supported}",
        ]
        if self.gpu_names:
            lines.append(f"GPUs: {', '.join(self.gpu_names)}")
        return "\n".join(lines)


class CUDADetector:
    """Detect CUDA availability and provide fallback recommendations."""
    
    def __init__(self):
        self.architecture = platform.machine().lower()
        self.is_arm64 = self.architecture in ('aarch64', 'arm64')
        
    def detect(self) -> DeviceInfo:
        """
        Detect CUDA and device information.
        
        Returns:
            DeviceInfo object with detection results
        """
        # Detect architecture
        arch = self.architecture
        
        # Try to detect NVIDIA GPUs
        gpu_count = 0
        gpu_names = []
        cuda_version = None
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_names = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]
                gpu_count = len(gpu_names)
                
            # Get CUDA version
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                cuda_version = result.stdout.strip().split('\n')[0].strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        # Check PyTorch
        pytorch_available = False
        pytorch_cuda = False
        try:
            import torch
            pytorch_available = True
            pytorch_cuda = torch.cuda.is_available()
        except ImportError:
            pass
        
        # Check JAX
        jax_available = False
        jax_cuda = False
        try:
            import jax
            jax_available = True
            # JAX on ARM64 typically doesn't support CUDA yet
            devices = jax.devices()
            jax_cuda = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
        except ImportError:
            pass
        
        # Determine CUDA availability
        cuda_available = bool(gpu_count > 0 and (pytorch_cuda or jax_cuda))
        
        # Determine if ARM64 CUDA is supported
        arm64_cuda_supported = self.is_arm64 and cuda_available
        
        # Determine device type
        if cuda_available:
            device_type = 'cuda'
        elif platform.system() == 'Darwin' and self.is_arm64:
            # Apple Silicon - could use MPS
            device_type = 'mps'
        else:
            device_type = 'cpu'
        
        return DeviceInfo(
            architecture=arch,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            device_type=device_type,
            pytorch_available=pytorch_available,
            jax_available=jax_available,
            arm64_cuda_supported=arm64_cuda_supported,
        )
    
    def get_recommendation(self) -> str:
        """
        Get recommendation for device usage.
        
        Returns:
            String with recommendation
        """
        device_info = self.detect()
        
        if not self.is_arm64:
            return "Native AMD64 system - use standard CUDA support"
        
        if device_info.arm64_cuda_supported:
            return "ARM64 CUDA detected - using GPU acceleration"
        
        if device_info.gpu_count > 0:
            return (
                "GPUs detected but CUDA not available on ARM64. "
                "Fallback options:\n"
                "  1. Use CPU-only mode (recommended for compatibility)\n"
                "  2. Wait for upstream ARM64 CUDA support\n"
                "  3. Use Docker with AMD64 emulation (may be slower)"
            )
        
        return "No GPU detected - using CPU-only mode"
    
    def check_compatibility(self, verbose: bool = True) -> bool:
        """
        Check if the system has ARM64 CUDA support.
        
        Args:
            verbose: Print detailed information
            
        Returns:
            True if ARM64 CUDA is supported, False otherwise
        """
        device_info = self.detect()
        
        if verbose:
            print(device_info)
            print()
            print("Recommendation:")
            print(self.get_recommendation())
            print()
            
            if not device_info.arm64_cuda_supported and self.is_arm64:
                print("NOTE: This system uses the ARM64 CUDA fallback module.")
                print("Performance may be limited to CPU execution.")
                print()
        
        return device_info.arm64_cuda_supported
