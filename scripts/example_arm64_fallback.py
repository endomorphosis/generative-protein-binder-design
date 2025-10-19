#!/usr/bin/env python3
"""
Example: Using ARM64 CUDA Fallback with Protein Design Tools

This example demonstrates how to integrate the ARM64 CUDA fallback module
with AlphaFold2, RFDiffusion, and ProteinMPNN installations.
"""

import sys
import os

# Add fallback module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arm64_cuda_fallback import (
    get_fallback_config,
    CUDADetector,
    PyTorchFallback,
    JAXFallback,
)
from arm64_cuda_fallback.utils import configure_environment_for_cpu


def setup_alphafold2():
    """Setup AlphaFold2 with fallback support."""
    print("=" * 70)
    print("AlphaFold2 with ARM64 CUDA Fallback")
    print("=" * 70)
    
    # Initialize JAX fallback (AlphaFold2 uses JAX)
    jax_fallback = JAXFallback(verbose=True)
    
    if not jax_fallback.available:
        print("ERROR: JAX is not installed")
        print("Install with: pip install jax jaxlib")
        return None
    
    # Get devices
    devices = jax_fallback.get_devices()
    device_info = jax_fallback.get_device_info()
    
    print("\nAlphaFold2 will use:")
    if device_info['gpu_available']:
        print("  ✓ GPU acceleration (ARM64 CUDA detected)")
    else:
        print("  ℹ CPU-only mode (recommended for ARM64)")
        print("  → For better performance, consider cloud GPU instances")
    
    # Configure for CPU if needed
    if not device_info['gpu_available']:
        jax_fallback.configure_for_cpu()
        configure_environment_for_cpu()
    
    return jax_fallback


def setup_rfdiffusion():
    """Setup RFDiffusion with fallback support."""
    print("\n" + "=" * 70)
    print("RFDiffusion with ARM64 CUDA Fallback")
    print("=" * 70)
    
    # Initialize PyTorch fallback (RFDiffusion uses PyTorch)
    pytorch_fallback = PyTorchFallback(verbose=True)
    
    if not pytorch_fallback.available:
        print("ERROR: PyTorch is not installed")
        print("Install with: pip install torch")
        return None
    
    # Get device
    device = pytorch_fallback.get_device()
    device_info = pytorch_fallback.get_device_info()
    
    print("\nRFDiffusion will use:")
    if device_info['cuda_available']:
        print(f"  ✓ GPU: {pytorch_fallback.get_device_name()}")
    else:
        print("  ℹ CPU-only mode")
        print("  → This may be slow for large proteins")
        print("  → Consider reducing design count or using smaller models")
    
    # Configure for CPU if needed
    if not device_info['cuda_available']:
        pytorch_fallback.configure_for_cpu()
        configure_environment_for_cpu()
    
    return pytorch_fallback


def setup_proteinmpnn():
    """Setup ProteinMPNN with fallback support."""
    print("\n" + "=" * 70)
    print("ProteinMPNN with ARM64 CUDA Fallback")
    print("=" * 70)
    
    # Initialize PyTorch fallback (ProteinMPNN uses PyTorch)
    pytorch_fallback = PyTorchFallback(verbose=True)
    
    if not pytorch_fallback.available:
        print("ERROR: PyTorch is not installed")
        print("Install with: pip install torch")
        return None
    
    # Get device
    device = pytorch_fallback.get_device()
    device_info = pytorch_fallback.get_device_info()
    
    print("\nProteinMPNN will use:")
    if device_info['cuda_available']:
        print(f"  ✓ GPU: {pytorch_fallback.get_device_name()}")
    else:
        print("  ℹ CPU-only mode (acceptable for ProteinMPNN)")
        print("  → ProteinMPNN is relatively fast on CPU")
    
    # Configure for CPU if needed
    if not device_info['cuda_available']:
        pytorch_fallback.configure_for_cpu()
    
    return pytorch_fallback


def demonstrate_usage():
    """Demonstrate the complete workflow."""
    print("\n" + "=" * 70)
    print("ARM64 CUDA Fallback - Complete Demonstration")
    print("=" * 70)
    print()
    
    # Check system
    detector = CUDADetector()
    device_info = detector.detect()
    
    print("System Information:")
    print(device_info)
    print()
    print("Recommendation:")
    print(detector.get_recommendation())
    print()
    
    # Check fallback configuration
    config = get_fallback_config()
    print("Fallback Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Setup each tool
    alphafold_fb = setup_alphafold2()
    rfdiffusion_fb = setup_rfdiffusion()
    proteinmpnn_fb = setup_proteinmpnn()
    
    # Summary
    print("\n" + "=" * 70)
    print("Setup Summary")
    print("=" * 70)
    print()
    
    tools_ready = []
    if alphafold_fb:
        tools_ready.append("AlphaFold2 (JAX)")
    if rfdiffusion_fb:
        tools_ready.append("RFDiffusion (PyTorch)")
    if proteinmpnn_fb:
        tools_ready.append("ProteinMPNN (PyTorch)")
    
    if tools_ready:
        print("✓ Ready to use:")
        for tool in tools_ready:
            print(f"  • {tool}")
    else:
        print("✗ No tools are ready")
        print("  Install PyTorch and/or JAX to continue")
    
    print()
    if config['fallback_active']:
        print("⚠ Running in CPU fallback mode")
        print("  Performance may be limited")
        print("  For production workloads, consider:")
        print("  • Cloud GPU instances (AWS, GCP, Azure)")
        print("  • AMD64 systems with NVIDIA GPUs")
        print("  • Waiting for upstream ARM64 CUDA support")
    else:
        print("✓ GPU acceleration available")
    
    print()
    print("=" * 70)


def example_pytorch_code():
    """Example of using PyTorch with fallback."""
    print("\n" + "=" * 70)
    print("Example: PyTorch Code with Fallback")
    print("=" * 70)
    print()
    
    from arm64_cuda_fallback import PyTorchFallback
    
    fallback = PyTorchFallback(verbose=True)
    
    if not fallback.available:
        print("PyTorch not available - skipping example")
        return
    
    # Get device
    device = fallback.get_device()
    
    # Create a simple model
    import torch
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create and move model to device
    model = SimpleModel()
    model = fallback.create_model_wrapper(model)
    
    # Create and move tensor to device
    x = torch.randn(1, 10)
    x = fallback.move_to_device(x)
    
    # Forward pass
    output = model(x)
    
    print(f"\nModel device: {next(model.parameters()).device}")
    print(f"Input device: {x.device}")
    print(f"Output shape: {output.shape}")
    print("\n✓ PyTorch example completed successfully")


def example_jax_code():
    """Example of using JAX with fallback."""
    print("\n" + "=" * 70)
    print("Example: JAX Code with Fallback")
    print("=" * 70)
    print()
    
    from arm64_cuda_fallback import JAXFallback
    
    fallback = JAXFallback(verbose=True)
    
    if not fallback.available:
        print("JAX not available - skipping example")
        return
    
    # Get devices
    devices = fallback.get_devices()
    default_device = fallback.get_default_device()
    
    # Create array on device
    import jax.numpy as jnp
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    array = fallback.create_array_on_device(data)
    
    # Simple computation
    result = jnp.sum(array * 2)
    
    print(f"\nArray: {array}")
    print(f"Result: {result}")
    print(f"Device: {fallback.check_array_device(array)}")
    print("\n✓ JAX example completed successfully")


if __name__ == '__main__':
    # Run main demonstration
    demonstrate_usage()
    
    # Run code examples
    example_pytorch_code()
    example_jax_code()
    
    print("\n" + "=" * 70)
    print("For more information:")
    print("  • Module documentation: src/arm64_cuda_fallback/README.md")
    print("  • CLI tool: python -m arm64_cuda_fallback info")
    print("  • Check migration: python -m arm64_cuda_fallback check-upstream")
    print("=" * 70)
