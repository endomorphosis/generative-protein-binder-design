# ARM64 CUDA Fallback - Quick Reference

## Overview

Two fallback solutions for ARM64 systems without native CUDA support:

1. **CPU Fallback Module** - Automatic CPU fallback (recommended for development)
2. **Cloud GPU Alternative** - Use cloud instances (recommended for production)

## Quick Start

### Check CUDA Status

```bash
# Quick check
./check_arm64_cuda.sh

# Detailed info
python -m arm64_cuda_fallback info
```

### Installation

```bash
# Install fallback module
./scripts/install_arm64_cuda_fallback.sh

# Activate environment
source activate_arm64_fallback.sh
```

### Usage in Python

```python
# Automatic device detection
from arm64_cuda_fallback import get_optimal_device
device = get_optimal_device()

# PyTorch
from arm64_cuda_fallback import PyTorchFallback
fallback = PyTorchFallback()
device = fallback.get_device()

# JAX
from arm64_cuda_fallback import JAXFallback
fallback = JAXFallback()
devices = fallback.get_devices()
```

## Common Commands

```bash
# Detect CUDA
python -m arm64_cuda_fallback detect

# Check PyTorch
python -m arm64_cuda_fallback pytorch

# Check JAX
python -m arm64_cuda_fallback jax

# Check upstream support
python -m arm64_cuda_fallback check-upstream

# Configure for CPU
python -m arm64_cuda_fallback configure-cpu --tips

# Run example
python scripts/example_arm64_fallback.py

# Run tests
python -m arm64_cuda_fallback.test_fallback
```

## Integration Examples

### AlphaFold2 (JAX)

```python
from arm64_cuda_fallback import JAXFallback

fallback = JAXFallback(verbose=True)
if not fallback.is_gpu_available():
    fallback.configure_for_cpu()

# Continue with AlphaFold2 code
import jax
devices = jax.devices()
```

### RFDiffusion (PyTorch)

```python
from arm64_cuda_fallback import PyTorchFallback

fallback = PyTorchFallback(verbose=True)
device = fallback.get_device()

if str(device) == 'cpu':
    fallback.configure_for_cpu()

# Continue with RFDiffusion code
import torch
model = model.to(device)
```

### ProteinMPNN (PyTorch)

```python
from arm64_cuda_fallback import PyTorchFallback

fallback = PyTorchFallback(verbose=True)
device = fallback.get_device()

# Continue with ProteinMPNN code
import torch
model = model.to(device)
```

## When to Use Each Solution

| Use Case | Solution |
|----------|----------|
| Development | CPU Fallback |
| Testing | CPU Fallback |
| Small proteins | CPU Fallback |
| Production | Cloud GPU |
| Large proteins | Cloud GPU |
| Time-critical | Cloud GPU |

## Deprecation

This module is **temporary** and will be deprecated when upstream ARM64 CUDA support is complete.

**Check deprecation status:**
```bash
python -m arm64_cuda_fallback check-upstream
```

**Migration when ready:**
```python
# Old (with fallback)
from arm64_cuda_fallback import PyTorchFallback
device = PyTorchFallback().get_device()

# New (native)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Troubleshooting

### Slow Performance
```python
from arm64_cuda_fallback.utils import configure_environment_for_cpu
configure_environment_for_cpu()
```

### Force CPU for Testing
```python
fallback = PyTorchFallback(force_cpu=True)
```

### Check Installation
```bash
python -m arm64_cuda_fallback info
```

## Documentation

- **Full Guide**: `ARM64_CUDA_FALLBACK_GUIDE.md`
- **Module Docs**: `src/arm64_cuda_fallback/README.md`
- **Examples**: `scripts/example_arm64_fallback.py`
- **Tests**: `src/arm64_cuda_fallback/test_fallback.py`

## Key Points

✅ Automatic CUDA detection and fallback  
✅ Works with PyTorch and JAX  
✅ Easy migration path  
✅ Comprehensive documentation  
⚠️ Temporary - will be deprecated  
⚠️ CPU performance limited  

## Support

- **Module issues**: Repository issues
- **PyTorch ARM64**: https://github.com/pytorch/pytorch
- **JAX ARM64**: https://github.com/google/jax
