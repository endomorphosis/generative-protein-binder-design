# ARM64 CUDA Fallback Solutions

This document explains the **four fallback options** available for ARM64 systems without native CUDA support.

## Problem Statement

ARM64 systems (including Apple Silicon and ARM-based servers) currently have limited CUDA support in PyTorch and JAX. This creates challenges for running GPU-accelerated protein design workflows.

## Solution Overview

We've implemented **four fallback solutions** that can be used until upstream CUDA support is complete:

### Solution 1: Automatic CPU Fallback (Recommended for Development)
- **What**: Automatically detects CUDA availability and falls back to CPU
- **When to use**: Development, testing, light workloads
- **Pros**: Simple, reliable, works everywhere
- **Cons**: Slower than GPU

### Solution 2: NVIDIA NGC Containers (Recommended for Production)
- **What**: Use NVIDIA NGC containers via Docker emulation or cloud instances
- **When to use**: Production workloads, need GPU acceleration
- **Pros**: Official NVIDIA containers, full software stack included
- **Cons**: Requires NGC API key, may need cloud instances for best performance

### Solution 3: PyTorch Source Build (Advanced Users)
- **What**: Build PyTorch from source with ARM64 CUDA support
- **When to use**: Need native ARM64 CUDA, have build expertise
- **Pros**: Native performance, full control
- **Cons**: Complex, time-consuming (1-3 hours), requires expertise

### Solution 4: Cloud GPU Alternative
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

### Solution 2: NVIDIA NGC Containers

NVIDIA NGC (NVIDIA GPU Cloud) provides pre-built containers with optimized software stacks for protein design workflows.

#### Prerequisites

```bash
# Set NGC API key (get from https://catalog.ngc.nvidia.com/)
export NGC_CLI_API_KEY=your_api_key_here

# Login to NGC registry
docker login nvcr.io --username='$oauthtoken' --password="${NGC_CLI_API_KEY}"
```

#### Check NGC Fallback Status

```bash
# Check NGC fallback status
python -m arm64_cuda_fallback ngc

# Setup NGC registry
python -m arm64_cuda_fallback ngc --setup
```

#### Using NGC Containers

```python
from arm64_cuda_fallback import NGCFallback

# Initialize NGC fallback
ngc = NGCFallback(use_emulation=True, verbose=True)

# Check status
status = ngc.get_status()
print(f"Docker available: {status['docker_available']}")
print(f"NGC logged in: {status['ngc_logged_in']}")

# Setup NGC registry
ngc.setup_ngc_registry()

# Get configuration for AlphaFold2
af2_config = ngc.get_alphafold_config(
    data_dir='/path/to/alphafold_data',
    output_dir='/path/to/output'
)

# Run container (example)
# result = ngc.run_container(af2_config, command='...')
```

#### NGC Container Options

**On ARM64 Systems:**
- Use Docker emulation (slower but functional)
- OR use cloud AMD64 instances (faster, recommended for production)

**On AMD64 Systems:**
- Native execution with full GPU support

**Available NGC Containers:**
- AlphaFold2: `nvcr.io/nvidia/clara/alphafold2:latest`
- PyTorch base: `nvcr.io/nvidia/pytorch:24.01-py3`
- Custom containers can be built on these bases

### Solution 3: Build PyTorch from Source

For ARM64 systems that need native CUDA support, you can build PyTorch from source.

⚠️ **Warning**: This is complex and time-consuming (1-3 hours). Only recommended for advanced users.

#### Prerequisites Check

```bash
# Check build dependencies
python -m arm64_cuda_fallback pytorch-build

# Generate build script
python -m arm64_cuda_fallback pytorch-build --generate-script
```

#### Manual Build Process

```python
from arm64_cuda_fallback import PyTorchSourceBuildFallback

# Initialize build fallback
builder = PyTorchSourceBuildFallback(verbose=True)

# Check dependencies
status = builder.get_status()
print(f"Build dependencies ready: {status['dependencies_ready']}")
print(f"CUDA available: {status['cuda_available']}")

# Get installation guide
print(builder.get_installation_guide())

# Generate build script
config = builder.get_recommended_build_config()
script_path = builder.build_dir + '/pytorch_arm64_build.sh'
builder.create_build_script(config, script_path)

print(f"\nRun: {script_path}")
```

#### Build Steps

**Option A: Automated Build with GitHub Actions (Recommended)**

1. **Install GitHub CLI**:
```bash
sudo apt install gh  # Ubuntu/Debian
brew install gh      # macOS
```

2. **Trigger build workflow**:
```bash
./scripts/trigger_pytorch_build.sh
```

Or manually:
```bash
gh workflow run pytorch-arm64-build.yml \
  -f cuda_version=11.8 \
  -f use_cuda=true \
  -f upload_artifact=true
```

3. **Monitor progress**:
```bash
gh run watch
```

4. **Download artifacts** when complete (includes wheel and build logs)

**Option B: Local Manual Build**

1. **Install dependencies**:
```bash
sudo apt update
sudo apt install build-essential cmake git python3-dev
# Install CUDA Toolkit from NVIDIA (optional)
# Install cuDNN from NVIDIA (optional, not critical for build)
```

2. **Generate and run build script**:
```bash
python -m arm64_cuda_fallback pytorch-build --generate-script
bash ~/pytorch_build/pytorch_arm64_build.sh
```

3. **Wait for build** (1-3 hours depending on system)

4. **Verify installation**:
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

#### When to Use PyTorch Source Build

✅ **Use when:**
- You need native ARM64 CUDA performance
- You have ARM64 hardware with NVIDIA GPUs
- You have advanced Linux and build system knowledge
- You can dedicate 1-3 hours for building
- Pre-built wheels don't support your configuration

❌ **Don't use when:**
- You just want to test the workflow (use CPU fallback)
- You're on a deadline (use cloud instances)
- You lack build system experience (use NGC containers)

### Solution 4: Cloud GPU Alternative

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

## Comparison Matrix

| Feature | CPU Fallback | NGC Containers | PyTorch Build | Cloud GPU |
|---------|-------------|----------------|---------------|-----------|
| **Setup Time** | Minutes | Minutes | Hours (1-3) | 10-30 min |
| **Cost** | Free | Free (local) | Free | $0.50-5/hour |
| **Performance** | Slow (CPU only) | Moderate-Fast | Fast (native) | Fast (native) |
| **Complexity** | Low | Low-Medium | High | Medium |
| **GPU Support** | No | Via emulation | Yes (if built) | Yes (native) |
| **Best For** | Development, testing | Production (with emulation) | Advanced users | Production |
| **Maintenance** | None | Low | High | Low |
| **Deprecation** | Easy | Easy | Easy | N/A |

## Recommendations

### For Development & Testing
- ✅ Use **CPU Fallback Mode**
- Simple setup, no costs
- Good for testing workflows
- Acceptable for small proteins

### For Production with ARM64 Hardware
- ✅ Use **NGC Containers** (recommended)
  - Official NVIDIA containers
  - Easy deployment
  - Works via emulation or cloud
- Consider **PyTorch Source Build** (advanced)
  - Only if you need native performance
  - Requires expertise

### For Production & Heavy Workloads
- ✅ Use **Cloud GPU Instances**
- Full GPU performance
- Cost-effective for large jobs
- No ARM64 limitations

### For Mixed Workloads
- Use **CPU Fallback** for development
- Use **NGC Containers** for production on ARM64
- Switch to **Cloud GPU** for heavy production runs
- Keep all options available for flexibility

## Command Reference

```bash
# Check all fallback options
python -m arm64_cuda_fallback info

# Solution 1: CPU Fallback
python -m arm64_cuda_fallback detect
python -m arm64_cuda_fallback pytorch
python -m arm64_cuda_fallback jax
python -m arm64_cuda_fallback configure-cpu --tips

# Solution 2: NGC Containers
python -m arm64_cuda_fallback ngc
python -m arm64_cuda_fallback ngc --setup

# Solution 3: PyTorch Source Build
python -m arm64_cuda_fallback pytorch-build
python -m arm64_cuda_fallback pytorch-build --generate-script

# Check deprecation status
python -m arm64_cuda_fallback check-upstream
python -m arm64_cuda_fallback migration
```

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
