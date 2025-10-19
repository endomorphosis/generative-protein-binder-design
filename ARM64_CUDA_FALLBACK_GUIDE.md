# ARM64 CUDA Fallback Solutions

This document explains the two fallback options available for ARM64 systems without native CUDA support.

## Problem Statement

ARM64 systems (including Apple Silicon and ARM-based servers) currently have limited CUDA support in PyTorch and JAX. This creates challenges for running GPU-accelerated protein design workflows.

## Solution Overview

We've implemented **two fallback solutions** that can be used until upstream CUDA support is complete:

### Solution 1: Automatic CPU Fallback (Recommended)
- **What**: Automatically detects CUDA availability and falls back to CPU
- **When to use**: Development, testing, light workloads
- **Pros**: Simple, reliable, works everywhere
- **Cons**: Slower than GPU

### Solution 2: Cloud GPU Alternative
- **What**: Use cloud instances with native AMD64 CUDA support
- **When to use**: Production, heavy workloads, time-sensitive tasks
- **Pros**: Full GPU performance, no compatibility issues
- **Cons**: Requires cloud account, ongoing costs

## Implementation: ARM64 CUDA Fallback Module

### Overview

The `arm64_cuda_fallback` module provides:
- CUDA availability detection
- Automatic fallback to CPU when CUDA unavailable
- Support for both PyTorch and JAX
- Easy migration path when upstream support arrives
- Deprecation warnings and migration guides

### Quick Start

#### Installation

```bash
# Install the fallback module
./scripts/install_arm64_cuda_fallback.sh

# Activate the environment
source activate_arm64_fallback.sh
```

#### Basic Usage

```python
from arm64_cuda_fallback import get_optimal_device

# Automatically get the best available device
device = get_optimal_device()

# Use with your model
import torch
model = MyModel()
model = model.to(device)
```

#### Check Status

```bash
# Quick check
./check_arm64_cuda.sh

# Detailed information
python -m arm64_cuda_fallback info

# Check if can be deprecated
python -m arm64_cuda_fallback check-upstream
```

### Solution 1: CPU Fallback Mode

#### For PyTorch (RFDiffusion, ProteinMPNN)

```python
from arm64_cuda_fallback import PyTorchFallback

# Initialize fallback handler
fallback = PyTorchFallback(verbose=True)

# Get device (automatically falls back to CPU)
device = fallback.get_device()

# Configure for optimal CPU performance
fallback.configure_for_cpu()

# Use with your model
import torch
model = torch.nn.Linear(100, 50)
model = fallback.create_model_wrapper(model)

# Move tensors to device
data = torch.randn(32, 100)
data = fallback.move_to_device(data)

# Run inference
output = model(data)
```

#### For JAX (AlphaFold2)

```python
from arm64_cuda_fallback import JAXFallback

# Initialize fallback handler
fallback = JAXFallback(verbose=True)

# Get devices (automatically uses CPU if GPU unavailable)
devices = fallback.get_devices()
default_device = fallback.get_default_device()

# Configure for optimal CPU performance
fallback.configure_for_cpu()

# Use with JAX
import jax.numpy as jnp

# Create array on device
data = [1, 2, 3, 4, 5]
array = fallback.create_array_on_device(data)

# Run computation
result = jnp.sum(array * 2)
```

#### Performance Optimization Tips

When running on CPU:

```python
from arm64_cuda_fallback.utils import (
    configure_environment_for_cpu,
    print_performance_tips
)

# Configure environment variables for CPU
configure_environment_for_cpu()

# Get optimization tips
print_performance_tips()
```

Key optimizations:
1. **Reduce batch sizes** - Smaller batches run faster on CPU
2. **Use CPU-optimized builds** - Install PyTorch/JAX with MKL or OpenBLAS
3. **Enable multi-threading** - Use all available CPU cores
4. **Quantize models** - Use int8 or float16 when possible
5. **Use smaller models** - Consider distilled versions

### Solution 2: Cloud GPU Alternative

For production workloads, consider cloud GPU instances:

#### AWS EC2
```bash
# Launch instance with GPU
aws ec2 run-instances \
  --image-id ami-xxxxx \
  --instance-type g4dn.xlarge \
  --key-name your-key
```

#### Google Cloud Platform
```bash
# Launch instance with GPU
gcloud compute instances create protein-design \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1
```

#### Azure
```bash
# Launch instance with GPU
az vm create \
  --name protein-design \
  --size Standard_NC6 \
  --image UbuntuLTS
```

### Integration with Existing Tools

#### AlphaFold2

```python
# In your AlphaFold2 setup
from arm64_cuda_fallback import JAXFallback

fallback = JAXFallback(verbose=True)
if not fallback.is_gpu_available():
    print("Using CPU mode - AlphaFold2 will be slower")
    fallback.configure_for_cpu()

# Your AlphaFold2 code continues normally
import jax
devices = jax.devices()  # Will use CPU automatically
```

#### RFDiffusion

```python
# In your RFDiffusion setup
from arm64_cuda_fallback import PyTorchFallback

fallback = PyTorchFallback(verbose=True)
device = fallback.get_device()

if str(device) == 'cpu':
    print("Using CPU mode - consider reducing num_designs")
    fallback.configure_for_cpu()

# Your RFDiffusion code
import torch
model = load_rfdiffusion_model()
model = model.to(device)
```

#### ProteinMPNN

```python
# In your ProteinMPNN setup
from arm64_cuda_fallback import PyTorchFallback

fallback = PyTorchFallback(verbose=True)
device = fallback.get_device()

# ProteinMPNN runs well on CPU
if str(device) == 'cpu':
    print("ProteinMPNN CPU mode - acceptable performance")

# Your ProteinMPNN code
import torch
model = load_proteinmpnn_model()
model = model.to(device)
```

### Deprecation and Migration

The fallback module is **temporary** and will be deprecated when upstream support is available.

#### Check Deprecation Status

```bash
# Check if module should be deprecated
python -m arm64_cuda_fallback check-upstream

# Show migration guide
python -m arm64_cuda_fallback migration
```

#### Migration Steps

When upstream support is available:

1. **Check status**:
```python
from arm64_cuda_fallback.utils import check_upstream_support

support = check_upstream_support()
if not support['fallback_needed']:
    print("Time to migrate!")
```

2. **Update code** (remove fallback imports):
```python
# Before (with fallback)
from arm64_cuda_fallback import PyTorchFallback
fallback = PyTorchFallback()
device = fallback.get_device()

# After (native)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

3. **Remove module**:
```bash
rm -rf src/arm64_cuda_fallback/
```

### Testing

```bash
# Run all tests
python -m arm64_cuda_fallback.test_fallback

# Run specific test
python -m unittest arm64_cuda_fallback.test_fallback.TestCUDADetector

# Test detection
python -m arm64_cuda_fallback detect

# Test PyTorch
python -m arm64_cuda_fallback pytorch

# Test JAX
python -m arm64_cuda_fallback jax
```

### Complete Example

See the complete working example:

```bash
# Run the example
python scripts/example_arm64_fallback.py
```

This demonstrates:
- System detection
- Framework setup
- AlphaFold2 integration
- RFDiffusion integration
- ProteinMPNN integration
- PyTorch usage examples
- JAX usage examples

## Comparison Matrix

| Feature | CPU Fallback | Cloud GPU |
|---------|-------------|-----------|
| **Setup Time** | Minutes | 10-30 minutes |
| **Cost** | Free (uses local CPU) | $0.50-5/hour |
| **Performance** | Slow (CPU only) | Fast (native GPU) |
| **Compatibility** | 100% compatible | 100% compatible |
| **Best For** | Development, testing | Production, large jobs |
| **Maintenance** | None | Cloud account needed |

## Recommendations

### For Development & Testing
- ✅ Use **CPU Fallback Mode**
- Simple setup, no costs
- Good for testing workflows
- Acceptable for small proteins

### For Production & Heavy Workloads
- ✅ Use **Cloud GPU Instances**
- Full GPU performance
- Cost-effective for large jobs
- No ARM64 limitations

### For Mixed Workloads
- Use **CPU Fallback** for development
- Switch to **Cloud GPU** for production runs
- Keep both options available

## Troubleshooting

### Issue: Slow performance on CPU

**Solution**: Optimize for CPU
```python
from arm64_cuda_fallback.utils import configure_environment_for_cpu
configure_environment_for_cpu()
```

### Issue: Out of memory on CPU

**Solution**: Reduce batch size or model size
- Use smaller proteins
- Reduce num_designs parameter
- Add swap space

### Issue: Want to force CPU for testing

**Solution**: Force CPU mode
```python
from arm64_cuda_fallback import PyTorchFallback
fallback = PyTorchFallback(force_cpu=True)
```

### Issue: Check if CUDA is working

**Solution**: Run detection
```bash
python -m arm64_cuda_fallback detect
```

## Additional Resources

- **Module Documentation**: `src/arm64_cuda_fallback/README.md`
- **Example Code**: `scripts/example_arm64_fallback.py`
- **Installation Script**: `scripts/install_arm64_cuda_fallback.sh`
- **Test Suite**: `src/arm64_cuda_fallback/test_fallback.py`

## Support Timeline

| Date | Status |
|------|--------|
| 2025-10-19 | ✅ Module created |
| 2025-Q2 | Check upstream progress |
| 2026-Q2 | Expected deprecation |

**Monitor upstream support**:
- PyTorch ARM64 CUDA: https://github.com/pytorch/pytorch
- JAX ARM64 GPU: https://github.com/google/jax

## Summary

Both fallback solutions are now available:

1. **CPU Fallback Module** (`arm64_cuda_fallback`)
   - Automatic detection and fallback
   - Easy integration with existing code
   - Temporary until upstream support
   - Full documentation and examples

2. **Cloud GPU Alternative**
   - Use AMD64 instances with native CUDA
   - Best for production workloads
   - Standard setup guides available

Choose based on your needs:
- **Development/Testing**: CPU Fallback
- **Production/Performance**: Cloud GPU
- **Mixed**: Use both as needed

The fallback module will be deprecated when upstream ARM64 CUDA support is complete.
