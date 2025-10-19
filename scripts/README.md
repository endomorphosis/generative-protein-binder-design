# ARM64 Installation and Build Scripts

This directory contains scripts for installing and building protein design tools natively on ARM64 architecture.

## Overview

Unlike the previous superficial ARM64 support, these scripts provide **real, working implementations** that:
- Install tools from source for ARM64
- Configure proper dependencies for aarch64
- Create isolated conda environments
- Include comprehensive testing
- Provide integration with Jupyter notebooks

## Installation Scripts

### Master Installation Script

**`install_all_arm64.sh`** - Interactive installer for all tools

```bash
bash scripts/install_all_arm64.sh
```

Features:
- Installs Miniforge if not present
- Interactive menu to select components
- Installs all dependencies
- Runs tests to verify installation

### Individual Installation Scripts

#### AlphaFold2

**`install_alphafold2_arm64.sh`** - Install AlphaFold2 natively on ARM64

```bash
bash scripts/install_alphafold2_arm64.sh
```

What it does:
- Creates `alphafold2_arm64` conda environment
- Installs JAX for ARM64 (CPU version)
- Clones AlphaFold2 repository
- Installs all dependencies
- Creates run and test scripts
- Installs to `~/alphafold2_arm64`

#### RFDiffusion

**`install_rfdiffusion_arm64.sh`** - Install RFDiffusion natively on ARM64

```bash
bash scripts/install_rfdiffusion_arm64.sh
```

What it does:
- Creates `rfdiffusion_arm64` conda environment
- Installs PyTorch for ARM64 (CPU version)
- Clones RFDiffusion and SE3Transformer repositories
- Installs all dependencies
- Creates run and test scripts
- Installs to `~/rfdiffusion_arm64`

#### ProteinMPNN

**`install_proteinmpnn_arm64.sh`** - Install ProteinMPNN natively on ARM64

```bash
bash scripts/install_proteinmpnn_arm64.sh
```

What it does:
- Creates `proteinmpnn_arm64` conda environment
- Installs PyTorch for ARM64 (CPU version)
- Clones ProteinMPNN repository
- Installs all dependencies
- Creates run and test scripts
- Installs to `~/proteinmpnn_arm64`

## Build Scripts

### PyTorch ARM64 Source Builder

**`build_pytorch_arm64_blackwell.sh`** - Build PyTorch from source with CUDA support for ARM64

```bash
./scripts/build_pytorch_arm64_blackwell.sh
```

Features:
- Builds PyTorch with CUDA support for NVIDIA GB10 (Blackwell) GPUs
- Supports compute capability 12.1
- Uses OpenBLAS instead of MKL for ARM64 compatibility
- Includes NCCL patches for proper CUDA architecture support
- 2-3 hour build time

**`monitor_pytorch_build.sh`** - Real-time build monitoring dashboard

```bash
./scripts/monitor_pytorch_build.sh
```

Features:
- Live process monitoring
- Disk usage tracking
- Build phase detection
- Interactive dashboard with 10-second refresh

See `scripts/PYTORCH_BUILD_README.md` for detailed documentation.

### Docker Image Builder

**`build_arm64_images.sh`** - Build ARM64-native Docker images

```bash
bash scripts/build_arm64_images.sh
```

Features:
- Interactive menu to select images to build
- Uses Docker buildx for native ARM64 builds
- Tags images with `arm64-latest`
- No emulation required

Builds:
- `protein-binder/alphafold2:arm64-latest`
- `protein-binder/rfdiffusion:arm64-latest`
- `protein-binder/proteinmpnn:arm64-latest`

## Usage Examples

### Install All Tools

```bash
# Interactive installation
bash scripts/install_all_arm64.sh

# Select option 4 (All components)
# Wait 2-4 hours for installation
```

### Install Single Tool

```bash
# Install only AlphaFold2
bash scripts/install_alphafold2_arm64.sh

# Use the tool
conda activate alphafold2_arm64
~/alphafold2_arm64/run_alphafold_arm64.sh --help
```

### Build Docker Images

```bash
# Build all images
bash scripts/build_arm64_images.sh

# Select option 4 (All images)
# Wait 1-3 hours for builds

# Use the images
cd deploy
docker compose -f docker-compose-arm64-native.yaml up -d
```

## Requirements

### System Requirements

- **Architecture**: ARM64/aarch64
- **RAM**: 32GB minimum (64GB recommended)
- **Storage**: 500GB free space
- **OS**: Ubuntu 22.04 ARM64 or compatible

### Software Requirements

For native installation:
- Bash shell
- wget, curl, git
- Python 3 support

For Docker builds:
- Docker 20.10+
- Docker buildx
- At least 100GB Docker storage

## Installation Locations

After installation, tools are installed to:

```
~/alphafold2_arm64/
  ├── alphafold/              # AlphaFold2 source
  ├── data/                   # Model weights (download separately)
  ├── run_alphafold_arm64.sh  # Runner script
  ├── jupyter_alphafold.sh    # Jupyter integration
  └── test_alphafold.py       # Test script

~/rfdiffusion_arm64/
  ├── RFdiffusion/            # RFDiffusion source
  ├── models/                 # Model weights (download separately)
  ├── run_rfdiffusion_arm64.sh
  └── test_rfdiffusion.py

~/proteinmpnn_arm64/
  ├── ProteinMPNN/            # ProteinMPNN source
  ├── models/                 # Model weights (download separately)
  ├── run_proteinmpnn_arm64.sh
  └── test_proteinmpnn.py
```

## Conda Environments

Created environments:

- `alphafold2_arm64` - Python 3.10, JAX, Haiku
- `rfdiffusion_arm64` - Python 3.9, PyTorch
- `proteinmpnn_arm64` - Python 3.9, PyTorch

List environments:
```bash
conda env list | grep arm64
```

Activate an environment:
```bash
conda activate alphafold2_arm64
```

## Testing

Each installation includes a test script:

```bash
# Test AlphaFold2
conda activate alphafold2_arm64
python ~/alphafold2_arm64/test_alphafold.py

# Test RFDiffusion
conda activate rfdiffusion_arm64
python ~/rfdiffusion_arm64/test_rfdiffusion.py

# Test ProteinMPNN
conda activate proteinmpnn_arm64
python ~/proteinmpnn_arm64/test_proteinmpnn.py
```

All tests should output:
```
✓ All tests passed! <Tool> ARM64 is ready.
```

## Troubleshooting

### Installation Fails

**Problem**: Conda environment creation fails

```bash
# Solution: Clean conda cache
conda clean --all
# Retry installation
```

**Problem**: Package installation fails

```bash
# Solution: Update pip and retry
pip install --upgrade pip
```

### Build Fails

**Problem**: Docker buildx not found

```bash
# Solution: Install buildx
docker buildx install
```

**Problem**: Out of disk space

```bash
# Solution: Clean Docker
docker system prune -a
```

### Runtime Issues

**Problem**: Import errors

```bash
# Solution: Verify environment
conda activate <env_name>
which python
python -c "import sys; print(sys.path)"
```

## Advanced Usage

### Custom Installation Directory

Edit script before running:

```bash
# In install_alphafold2_arm64.sh
INSTALL_DIR="/custom/path/alphafold2_arm64"
```

### Offline Installation

Download dependencies first:

```bash
# Download conda packages
conda create --name alphafold2_arm64 --download-only python=3.10

# Download pip packages
pip download -r requirements.txt -d /tmp/packages
```

### Multi-User Installation

Install to shared location:

```bash
# Install to /opt
sudo INSTALL_DIR=/opt/alphafold2_arm64 bash install_alphafold2_arm64.sh
```

## Performance Notes

### Native vs Emulation

| Method | Performance | Setup Time | Recommendation |
|--------|-------------|------------|----------------|
| Native Install | 100% | 2-4 hours | Production |
| ARM64 Docker | 95-98% | 1-2 hours | Development |
| AMD64 Emulation | 40-60% | 30 min | Testing only |

### GPU Support

- Native installations use CPU versions by default
- GPU support requires CUDA toolkit for ARM64
- Docker images include GPU support via NVIDIA runtime

## Integration

### With GitHub Actions

Use in workflows:

```yaml
- name: Install tools
  run: bash scripts/install_all_arm64.sh
  
- name: Test installation
  run: |
    conda activate alphafold2_arm64
    python ~/alphafold2_arm64/test_alphafold.py
```

### With Jupyter

Each installation includes Jupyter integration:

```bash
~/alphafold2_arm64/jupyter_alphafold.sh
```

### With Existing Workflows

Source the environments in your scripts:

```bash
#!/bin/bash
source ~/miniforge3/bin/activate
conda activate alphafold2_arm64
python my_protein_design.py
```

## Support

For issues:
1. Check test scripts output
2. Review installation logs
3. Verify system requirements
4. See main ARM64_COMPLETE_GUIDE.md
5. Check individual script comments

## Contributing

To add a new tool:

1. Create `install_<tool>_arm64.sh`
2. Follow existing script structure
3. Include test script
4. Update `install_all_arm64.sh`
5. Add to this README
6. Test on actual ARM64 hardware

## License

These scripts are part of the NVIDIA BioNeMo Blueprint project.
See main LICENSE file for details.
