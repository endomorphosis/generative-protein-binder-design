# PyTorch ARM64 Build Scripts

## Overview

This directory contains scripts for building PyTorch from source with CUDA support on ARM64 systems, specifically optimized for NVIDIA GB10 GPUs (Blackwell architecture).

## Scripts

### `build_pytorch_arm64_blackwell.sh`

**Purpose**: Builds PyTorch from source with full CUDA support for ARM64 systems with NVIDIA GB10 GPUs.

**Key Features**:
- Supports CUDA compute capability 12.1 (Blackwell architecture)
- Uses OpenBLAS instead of MKL for ARM64 compatibility
- Includes NCCL patches for proper CUDA architecture support
- Disables Flash Attention to avoid compute_70 conflicts
- Optimized for 19-thread parallel compilation

**Usage**:
```bash
./scripts/build_pytorch_arm64_blackwell.sh
```

**Requirements**:
- ARM64 Ubuntu system
- CUDA 13.0+ installed
- NVIDIA GB10 GPU
- 8GB+ free disk space
- 2-3 hours build time

**Build Environment Variables**:
- `TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 12.1"`
- `USE_FLASH_ATTENTION=0`
- `USE_MKL=0` (uses OpenBLAS instead)
- `USE_CUDA=1`

### `monitor_pytorch_build.sh`

**Purpose**: Real-time monitoring dashboard for PyTorch build progress.

**Features**:
- Live process status monitoring
- Disk usage tracking
- Build phase detection
- Progress log tailing
- Interactive dashboard with 10-second refresh

**Usage**:
```bash
./scripts/monitor_pytorch_build.sh
```

**Dashboard Information**:
- Process status and PID
- Build directory disk usage
- Current build phase
- Last 3 log entries
- Helpful commands

## Build Process

1. **Start Build**:
   ```bash
   nohup ./scripts/build_pytorch_arm64_blackwell.sh > pytorch_build.log 2>&1 &
   ```

2. **Monitor Progress**:
   ```bash
   ./scripts/monitor_pytorch_build.sh
   ```

3. **Install Built Wheel**:
   ```bash
   pip install ~/pytorch_build_gb10_fixed/pytorch/dist/*.whl --force-reinstall
   ```

## Troubleshooting

### Common Issues

1. **CUDA Architecture Errors**:
   - Error: `nvcc fatal: Unsupported gpu architecture 'compute_70'`
   - Solution: The script automatically patches NCCL to use supported architectures

2. **MKL Dependency Issues**:
   - Error: MKL not available on ARM64
   - Solution: Script uses OpenBLAS (`USE_MKL=0`)

3. **Flash Attention Conflicts**:
   - Error: Flash Attention compute_70 requirements
   - Solution: Script disables Flash Attention (`USE_FLASH_ATTENTION=0`)

### Build Requirements

- **System**: ARM64 Ubuntu 24.04+
- **GPU**: NVIDIA GB10 (Blackwell, compute capability 12.1)
- **CUDA**: 13.0+ with development tools
- **Memory**: 16GB+ RAM recommended
- **Storage**: 15GB+ free space
- **Time**: 1-3 hours depending on system

### Verification

After successful build and installation:

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
"
```

Expected output:
```
PyTorch: 2.10.0a0+gite939651
CUDA Available: True
GPU: NVIDIA GB10
Compute Capability: (12, 1)
```

## Related Files

- `ARM64_CUDA_FALLBACK_IMPLEMENTATION.md` - Comprehensive ARM64 CUDA support guide
- `src/arm64_cuda_fallback/` - ARM64 CUDA fallback module
- `scripts/install_*_arm64.sh` - Individual tool installation scripts

## Success Metrics

- ✅ PyTorch wheel builds without errors
- ✅ CUDA support enabled and functional
- ✅ GPU operations work correctly
- ✅ Neural network training succeeds
- ✅ Full 119.7GB GPU memory accessible