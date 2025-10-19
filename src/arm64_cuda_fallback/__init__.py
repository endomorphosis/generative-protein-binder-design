"""
ARM64 CUDA Fallback Module

This module provides fallback options for ARM64 CUDA support until upstream
support is completed. It will be deprecated once native ARM64 CUDA support
is available in PyTorch and JAX.

DEPRECATION NOTICE:
This module is temporary and will be removed when:
1. PyTorch adds native ARM64 CUDA support
2. JAX adds native ARM64 CUDA support
3. All upstream dependencies support ARM64 CUDA

Version: 1.1.0
Expected Deprecation: Q2 2026 (subject to upstream progress)
"""

import warnings
from typing import Optional, Dict, Any

from .detector import CUDADetector, DeviceInfo
from .pytorch_fallback import PyTorchFallback
from .jax_fallback import JAXFallback
from .ngc_fallback import NGCFallback, setup_ngc_fallback
from .pytorch_source_build import PyTorchSourceBuildFallback, setup_pytorch_source_build
from .utils import get_optimal_device, format_device_info

__version__ = "1.1.0"
__all__ = [
    "CUDADetector",
    "DeviceInfo",
    "PyTorchFallback",
    "JAXFallback",
    "NGCFallback",
    "setup_ngc_fallback",
    "PyTorchSourceBuildFallback",
    "setup_pytorch_source_build",
    "get_optimal_device",
    "format_device_info",
]

# Deprecation warning
warnings.warn(
    "arm64_cuda_fallback is a temporary module and will be deprecated once "
    "upstream ARM64 CUDA support is complete. Plan to migrate to native "
    "implementations.",
    FutureWarning,
    stacklevel=2
)


def get_fallback_config() -> Dict[str, Any]:
    """
    Get the current fallback configuration.
    
    Returns:
        Dictionary containing fallback status and configuration
    """
    detector = CUDADetector()
    device_info = detector.detect()
    
    return {
        "version": __version__,
        "architecture": device_info.architecture,
        "cuda_available": device_info.cuda_available,
        "cuda_version": device_info.cuda_version,
        "device_type": device_info.device_type,
        "fallback_active": not device_info.cuda_available,
        "recommended_backend": "cpu" if not device_info.cuda_available else "cuda",
    }


def print_deprecation_notice():
    """Print detailed deprecation notice and migration guide."""
    print("=" * 70)
    print("ARM64 CUDA FALLBACK MODULE - DEPRECATION NOTICE")
    print("=" * 70)
    print()
    print("This module provides temporary fallback support for ARM64 systems")
    print("without native CUDA support. It will be deprecated once upstream")
    print("support is completed.")
    print()
    print("Current Fallback Options:")
    print("  1. CPU-only mode (always available)")
    print("  2. Auto-detection with graceful degradation")
    print()
    print("Migration Path:")
    print("  - Monitor upstream PyTorch ARM64 CUDA support")
    print("  - Monitor upstream JAX ARM64 CUDA support")
    print("  - Test with native implementations when available")
    print("  - Remove this module when native support is stable")
    print()
    print("=" * 70)
