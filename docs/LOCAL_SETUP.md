# Local Development Setup Guide

This guide will help you set up the NVIDIA BioNeMo Blueprint for Protein Binder Design on your local machine.

## Installation Options

Choose the setup method that best fits your system:

1. **Docker-based Setup (Recommended)** - This guide
   - Works on most systems with Docker
   - Easiest and fastest setup
   - Uses pre-built containers

2. **ARM64 Systems with Docker** - See [ARM64_COMPATIBILITY.md](ARM64_COMPATIBILITY.md)
   - For Apple Silicon (M1/M2/M3) or ARM servers
   - Uses Docker with platform emulation

3. **ARM64 Native Installation (Advanced)** - See [ARM64_NATIVE_INSTALLATION.md](ARM64_NATIVE_INSTALLATION.md)
   - For experienced users on ARM64 systems
   - Requires building tools from source
   - Takes several days to set up

## Quick Start

For a streamlined setup experience, run the automated setup script:

```bash
../scripts/setup_local.sh
```

This script will check your system requirements, install dependencies, and configure your environment.

## Manual Setup Instructions

If you prefer to set up the environment manually, follow these detailed steps:

### 1. System Requirements

**Minimum Hardware Requirements:**
- **Storage**: At least 1.3TB of fast NVMe SSD space
- **CPU**: Modern CPU with at least 24 cores (20 detected - may work but performance may be impacted)
- **RAM**: At least 64GB (120GB detected ✓)
- **GPU**: Two or more NVIDIA L40s, A100, or H100 GPUs (currently 1 GPU detected - may limit functionality)

**Current System Status:**
- ✓ Docker installed (v28.3.3)
- ✓ Docker Compose installed (v2.39.1)
- ✓ NVIDIA drivers installed (580.95.05)
- ✓ NVIDIA Container Runtime installed (1.17.9)
- ✓ Sufficient disk space (3.3TB available)
- ✓ Sufficient RAM (120GB available)
- ⚠ Only 1 GPU detected (minimum 2 recommended for optimal performance)
- ⚠ Only 20 CPU cores (24 recommended)

### 2. Prerequisites Installation

#### Install Required System Packages

```bash
# Update package list
sudo apt update

# Install Docker (if not already installed)
sudo apt-get install -y docker.io docker-compose

# Install NVIDIA Container Toolkit (if not already installed)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Add User to Docker Group

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 3. Environment Configuration

#### Set up NGC API Key

1. Visit [NGC Website](https://ngc.nvidia.com/)
2. Sign in or create an account
3. Go to Setup > Generate API Key
4. Generate a key with appropriate permissions
5. Set the environment variable:

```bash
export NGC_CLI_API_KEY="your_api_key_here"
echo 'export NGC_CLI_API_KEY="your_api_key_here"' >> ~/.bashrc
```

#### Configure NIM Cache

```bash
# Create NIM cache directory
mkdir -p ~/.cache/nim
chmod -R 777 ~/.cache/nim

# Set environment variable
export HOST_NIM_CACHE=~/.cache/nim
echo 'export HOST_NIM_CACHE=~/.cache/nim' >> ~/.bashrc
```

#### Login to NGC Container Registry

```bash
docker login nvcr.io --username='$oauthtoken' --password="${NGC_CLI_API_KEY}"
```

### 4. Python Environment Setup

#### Install Python Dependencies

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 5. Starting the Services

#### Deploy with Docker Compose

```bash
cd deploy/
docker compose up
```

**Important Notes:**
- Initial startup will take 3-7 hours due to model downloads
- Models will be cached to `~/.cache/nim` for faster subsequent startups
- Monitor disk space during download

#### Verify Services

Check that all services are ready:

```bash
curl localhost:8081/v1/health/ready  # AlphaFold2
curl localhost:8082/v1/health/ready  # RFDiffusion
curl localhost:8083/v1/health/ready  # ProteinMPNN
curl localhost:8084/v1/health/ready  # AlphaFold2-Multimer
```

Each should return: `{"status":"ready"}`

### 6. Running the Notebook

```bash
cd src/
source ../.venv/bin/activate
jupyter notebook
```

Open `protein-binder-design.ipynb` and follow the examples.

## Troubleshooting

### Common Issues

**Docker Permission Denied:**
```bash
sudo usermod -aG docker $USER
newgrp docker
# Or logout and login again
```

**Out of Disk Space:**
- Models require ~1.3TB total
- Check available space: `df -h`
- Clean up if needed: `docker system prune -a`

**GPU Not Available:**
- Verify: `nvidia-smi`
- Check Docker GPU access: `docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi`

**Services Not Ready:**
- Check logs: `docker compose logs [service_name]`
- Verify environment variables: `echo $NGC_CLI_API_KEY`
- Ensure sufficient memory and disk space

**Network Issues:**
- Models download from NGC, ensure internet connectivity
- Check firewall settings if needed

### Performance Optimization

**With Limited GPUs:**
- If you have fewer than 4 GPUs, you may need to modify the docker-compose.yaml to assign multiple services to the same GPU
- Edit `device_ids` in the compose file to reuse GPUs

**Memory Management:**
- Monitor RAM usage with `htop`
- Consider reducing the number of concurrent services if memory is limited

### Service Configuration

**Modifying Resource Allocation:**
Edit `deploy/docker-compose.yaml` to adjust:
- GPU assignments (`device_ids`)
- Memory limits
- Port mappings

**Environment Variables:**
- `NIM_DISABLE_MODEL_DOWNLOAD=True` to skip model downloads (if models already cached)
- `CUDA_VISIBLE_DEVICES` to control GPU visibility per service

## Additional Resources

- [Official Documentation](../README.md)
- [Docker Compose Documentation](deploy/README.md)
- [Helm Chart Documentation](protein-design-chart/README.md)
- [NVIDIA NGC Container Registry](https://catalog.ngc.nvidia.com/)

## Support

For issues:
1. Check this troubleshooting guide
2. Review service logs: `docker compose logs`
3. Verify system requirements
4. Check NVIDIA NGC documentation for specific NIM containers