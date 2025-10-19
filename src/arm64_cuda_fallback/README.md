# ARM64 CUDA Fallback Module

**Version:** 1.0.0  
**Status:** Active (Temporary - Will be deprecated when upstream support is complete)

## Overview

This module provides fallback support for ARM64 systems without native CUDA support in PyTorch and JAX. It automatically detects CUDA availability and gracefully falls back to CPU execution when GPU acceleration is not available.

### Purpose

ARM64 architecture (including Apple Silicon and ARM-based servers) currently has limited CUDA support in popular deep learning frameworks. This module:

1. **Detects CUDA availability** on ARM64 systems
2. **Provides automatic fallback** to CPU when CUDA is unavailable
3. **Offers a consistent API** across PyTorch and JAX
4. **Will be deprecated** once upstream frameworks add native ARM64 CUDA support

## Deprecation Notice

⚠️ **This module is temporary and will be deprecated** when:
- PyTorch adds native ARM64 CUDA support
- JAX adds native ARM64 GPU support
- All dependencies support ARM64 CUDA natively

**Expected deprecation:** Q2 2026 (subject to upstream progress)

When deprecated, this module will be removed and users should migrate to native framework implementations.

## Installation

The module is included in the repository. No additional installation is required beyond the base requirements:

```bash
# Install with PyTorch support
pip install torch

# Install with JAX support  
pip install jax jaxlib

# Both frameworks (recommended)
pip install torch jax jaxlib
```

## Quick Start

### Command-Line Interface

```bash
# Check CUDA availability
python -m arm64_cuda_fallback detect

# Check PyTorch device
python -m arm64_cuda_fallback pytorch

# Check JAX devices
python -m arm64_cuda_fallback jax

# Show comprehensive information
python -m arm64_cuda_fallback info

# Check if module can be deprecated
python -m arm64_cuda_fallback check-upstream
```

### Python API

#### Automatic Device Detection

```python
from arm64_cuda_fallback import get_optimal_device

# Auto-detect framework and get optimal device
device = get_optimal_device()

# Force specific framework
device = get_optimal_device(framework='pytorch')
device = get_optimal_device(framework='jax')

# Force CPU (testing/debugging)
device = get_optimal_device(force_cpu=True)
```

#### PyTorch Example

```python
from arm64_cuda_fallback import PyTorchFallback

# Create fallback handler
fallback = PyTorchFallback(verbose=True)

# Get device (automatically falls back to CPU if needed)
device = fallback.get_device()

# Use with your model
import torch
model = torch.nn.Linear(10, 5)
model = fallback.create_model_wrapper(model)

# Move tensors to device
tensor = torch.randn(10, 10)
tensor = fallback.move_to_device(tensor)

# Get device information
info = fallback.get_device_info()
print(info)
```

#### JAX Example

```python
from arm64_cuda_fallback import JAXFallback

# Create fallback handler
fallback = JAXFallback(verbose=True)

# Get available devices
devices = fallback.get_devices()

# Get default device
device = fallback.get_default_device()

# Create array on device
import jax.numpy as jnp
data = [1, 2, 3, 4, 5]
array = fallback.create_array_on_device(data)

# Get device information
info = fallback.get_device_info()
print(info)
```

#### CUDA Detection

```python
from arm64_cuda_fallback import CUDADetector

# Detect CUDA availability
detector = CUDADetector()
device_info = detector.detect()

print(device_info)
# Output:
# Architecture: aarch64
# CUDA Available: False
# CUDA Version: N/A
# GPU Count: 0
# Device Type: cpu
# PyTorch Available: True
# JAX Available: True
# ARM64 CUDA Supported: False

# Get recommendation
recommendation = detector.get_recommendation()
print(recommendation)
```

## Features

### 1. Automatic Fallback

The module automatically detects when CUDA is not available and falls back to CPU:

```python
from arm64_cuda_fallback import PyTorchFallback

fallback = PyTorchFallback()
device = fallback.get_device()

# On ARM64 without CUDA:
# "ARM64 CUDA not available - falling back to CPU"
# device = cpu

# On ARM64 with CUDA:
# "ARM64 CUDA detected - using GPU 0"
# device = cuda:0
```

### 2. Cross-Framework Support

Works with both PyTorch and JAX:

```python
# PyTorch
from arm64_cuda_fallback import PyTorchFallback
pytorch_device = PyTorchFallback().get_device()

# JAX
from arm64_cuda_fallback import JAXFallback
jax_devices = JAXFallback().get_devices()

# Auto-detect
from arm64_cuda_fallback import get_optimal_device
device = get_optimal_device()  # Uses available framework
```

### 3. Performance Optimization

Helpers for optimizing CPU performance:

```python
from arm64_cuda_fallback import PyTorchFallback
from arm64_cuda_fallback.utils import configure_environment_for_cpu, print_performance_tips

# Configure PyTorch for CPU
fallback = PyTorchFallback()
fallback.configure_for_cpu()

# Configure environment variables
configure_environment_for_cpu()

# Get optimization tips
print_performance_tips()
```

### 4. Migration Support

Check when the module can be deprecated:

```python
from arm64_cuda_fallback.utils import (
    check_upstream_support,
    should_deprecate,
    get_migration_guide
)

# Check support status
support = check_upstream_support()
print(f"PyTorch ARM64 CUDA: {support['pytorch_arm64_cuda']}")
print(f"JAX ARM64 GPU: {support['jax_arm64_cuda']}")
print(f"Fallback needed: {support['fallback_needed']}")

# Check if can deprecate
if should_deprecate():
    print("Module can be deprecated!")
    print(get_migration_guide())
```

## API Reference

### Classes

- **`CUDADetector`**: Detect CUDA availability and system information
- **`DeviceInfo`**: Dataclass containing device information
- **`PyTorchFallback`**: PyTorch device management with fallback
- **`JAXFallback`**: JAX device management with fallback

### Functions

- **`get_optimal_device()`**: Get optimal device with auto-framework detection
- **`get_fallback_config()`**: Get current fallback configuration
- **`check_upstream_support()`**: Check upstream ARM64 CUDA support
- **`should_deprecate()`**: Check if module should be deprecated
- **`get_migration_guide()`**: Get migration instructions
- **`format_device_info()`**: Format device info as string
- **`configure_environment_for_cpu()`**: Optimize environment for CPU
- **`print_performance_tips()`**: Print CPU performance tips

## CLI Commands

```bash
# Detect CUDA
python -m arm64_cuda_fallback detect

# PyTorch device info
python -m arm64_cuda_fallback pytorch
python -m arm64_cuda_fallback pytorch --force-cpu
python -m arm64_cuda_fallback pytorch --configure-cpu

# JAX device info
python -m arm64_cuda_fallback jax
python -m arm64_cuda_fallback jax --force-cpu

# Check upstream support
python -m arm64_cuda_fallback check-upstream -v

# Configure for CPU
python -m arm64_cuda_fallback configure-cpu --tips

# Migration guide
python -m arm64_cuda_fallback migration

# Deprecation notice
python -m arm64_cuda_fallback deprecation

# Complete information
python -m arm64_cuda_fallback info
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m arm64_cuda_fallback.test_fallback

# Run with unittest
python -m unittest arm64_cuda_fallback.test_fallback

# Run specific test
python -m unittest arm64_cuda_fallback.test_fallback.TestCUDADetector
```

## Examples

### Example 1: Protein Folding with AlphaFold2

```python
from arm64_cuda_fallback import JAXFallback

# Initialize fallback
fallback = JAXFallback(verbose=True)

# Your AlphaFold2 code here
import jax
devices = fallback.get_devices()
# Uses CPU on ARM64, GPU if available
```

### Example 2: Protein Design with RFDiffusion

```python
from arm64_cuda_fallback import PyTorchFallback

# Initialize fallback
fallback = PyTorchFallback(verbose=True)
device = fallback.get_device()

# Your RFDiffusion code here
import torch
model = load_rfdiffusion_model()
model = model.to(device)
```

### Example 3: Integration Script

```python
#!/usr/bin/env python3
"""Integration example with fallback support."""

from arm64_cuda_fallback import get_fallback_config, get_optimal_device

# Check configuration
config = get_fallback_config()
print(f"Running on: {config['architecture']}")
print(f"CUDA available: {config['cuda_available']}")
print(f"Fallback active: {config['fallback_active']}")

# Get device
device = get_optimal_device(framework='pytorch', verbose=True)

# Your code here
import torch
model = YourModel()
model = model.to(device)
```

## Migration Plan

When upstream support becomes available:

### Step 1: Check Support Status

```python
from arm64_cuda_fallback.utils import check_upstream_support

support = check_upstream_support()
if not support['fallback_needed']:
    print("Native support available! Time to migrate.")
```

### Step 2: Update Code

Replace fallback imports with native implementations:

```python
# Before (with fallback)
from arm64_cuda_fallback import PyTorchFallback
fallback = PyTorchFallback()
device = fallback.get_device()

# After (native)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Step 3: Remove Module

```bash
# Remove the arm64_cuda_fallback directory
rm -rf src/arm64_cuda_fallback/

# Update any imports in your code
# Remove from requirements if listed
```

## Troubleshooting

### Issue: "PyTorch is not installed"

```bash
pip install torch
```

### Issue: "JAX is not installed"

```bash
pip install jax jaxlib
```

### Issue: Poor CPU performance

```python
from arm64_cuda_fallback.utils import configure_environment_for_cpu, print_performance_tips

configure_environment_for_cpu()
print_performance_tips()
```

### Issue: Want to force CPU for testing

```python
from arm64_cuda_fallback import PyTorchFallback

fallback = PyTorchFallback(force_cpu=True)
device = fallback.get_device()  # Always returns CPU
```

## Contributing

This is a temporary module that will be deprecated. Contributions should focus on:
- Bug fixes
- Documentation improvements
- Testing improvements
- Migration guides

Do not add new features - instead, encourage upstream framework development.

## License

Same as parent project.

## Support

For issues related to:
- **This module**: Open an issue in the repository
- **PyTorch ARM64 CUDA**: https://github.com/pytorch/pytorch
- **JAX ARM64 GPU**: https://github.com/google/jax

## Changelog

### Version 1.0.0 (2025-10-19)
- Initial release
- PyTorch fallback support
- JAX fallback support
- CUDA detection
- CLI tool
- Comprehensive documentation
- Test suite
