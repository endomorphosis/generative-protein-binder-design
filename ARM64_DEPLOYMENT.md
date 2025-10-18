# ARM64 Deployment Guide

This guide provides comprehensive instructions for deploying the Protein Binder Design project on ARM64 architecture systems.

## Quick Start

### 1. Check Your Platform

Run the platform detection script to understand your system:

```bash
./detect_platform.sh
```

This will automatically detect your architecture and provide tailored recommendations.

## Deployment Options

### Option A: Docker with Emulation (Quick Start)

**Best for:** Quick testing, evaluation, or when native installation is not feasible

**Pros:**
- Fastest to set up (minutes)
- Uses official NVIDIA NIM containers
- Consistent environment
- Easy to update

**Cons:**
- Performance impact from AMD64 emulation on ARM64
- Higher memory usage
- May encounter compatibility issues

**Steps:**

1. **Ensure QEMU emulation is available:**
   ```bash
   docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
   ```

2. **Set up NGC API key:**
   ```bash
   export NGC_CLI_API_KEY=your_api_key_here
   ```

3. **Start services:**
   ```bash
   cd deploy
   docker compose up -d
   ```

4. **Monitor for platform warnings:**
   ```bash
   docker compose logs -f
   ```

5. **Use single GPU mode if needed:**
   ```bash
   ./deploy/run_single_gpu.sh
   ```

### Option B: Native Installation (Recommended for ARM64)

**Best for:** Production use, best performance on ARM64

**Pros:**
- Native ARM64 performance
- No emulation overhead
- Full control over environment
- Better GPU utilization

**Cons:**
- Complex installation (days, not hours)
- Requires advanced technical skills
- Manual dependency management
- Ongoing maintenance

**Steps:**

1. **Run the automated native installation workflow:**
   ```bash
   gh workflow run native-install-test.yml -f component=all
   ```

2. **Or follow the detailed manual guide:**
   See [ARM64_NATIVE_INSTALLATION.md](ARM64_NATIVE_INSTALLATION.md)

3. **Use the native pipeline:**
   ```bash
   gh workflow run protein-design-pipeline.yml -f use_native=true
   ```

### Option C: Hybrid Approach

**Best for:** Balancing performance and convenience

**Description:** Use native installations for critical components and Docker for others.

**Example Configuration:**

1. **Install PyTorch and JAX natively for ARM64:**
   ```bash
   # Install miniforge for ARM64
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
   bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
   source $HOME/miniforge3/bin/activate
   
   # Install PyTorch
   conda create -n pytorch python=3.10 -y
   conda activate pytorch
   pip install torch torchvision torchaudio
   ```

2. **Use Docker for services not available natively:**
   ```bash
   # Start only specific services
   docker compose up -d alphafold
   ```

3. **Create wrapper scripts to integrate native and Docker components**

## GitHub Actions Workflows

The project includes several workflows optimized for ARM64:

### Validation Workflows

1. **ARM64 Platform Validation:**
   ```bash
   gh workflow run arm64-validation.yml -f test_mode=full
   ```
   Comprehensive platform compatibility check

2. **System Health Check:**
   ```bash
   gh workflow run system-health.yml -f deep_check=true
   ```
   Monitor system resources and health

3. **Runner Connection Test:**
   ```bash
   gh workflow run runner-test.yml
   ```
   Verify GitHub Actions runner connectivity

### Testing Workflows

1. **Docker Compatibility Test:**
   ```bash
   # Test ARM64 native containers
   gh workflow run docker-test.yml -f platform=linux/arm64
   
   # Test AMD64 emulation
   gh workflow run docker-test.yml -f platform=linux/amd64
   
   # Test both
   gh workflow run docker-test.yml -f platform=both
   ```

2. **Native Installation Test:**
   ```bash
   # Test specific component
   gh workflow run native-install-test.yml -f component=alphafold2
   
   # Test all components
   gh workflow run native-install-test.yml -f component=all
   ```

3. **Jupyter Notebook Test:**
   ```bash
   gh workflow run jupyter-test.yml
   ```

### Pipeline Workflows

1. **Full Protein Design Pipeline:**
   ```bash
   # Using native installation
   gh workflow run protein-design-pipeline.yml \
     -f target_protein=7BZ5 \
     -f num_designs=10 \
     -f use_native=true
   
   # Using Docker
   gh workflow run protein-design-pipeline.yml \
     -f target_protein=7BZ5 \
     -f num_designs=10 \
     -f use_native=false
   ```

## Platform-Specific Optimizations

### ARM64 Docker Optimizations

1. **Enable BuildKit:**
   ```bash
   export DOCKER_BUILDKIT=1
   ```

2. **Use ARM64-native base images when possible:**
   ```dockerfile
   FROM ubuntu:22.04  # Supports ARM64 natively
   ```

3. **Specify platform explicitly:**
   ```yaml
   platform: linux/amd64  # For NVIDIA NIM containers
   ```

4. **Monitor emulation performance:**
   ```bash
   # Check if emulation is active
   docker info | grep -i emulation
   
   # Monitor resource usage
   docker stats
   ```

### GPU Configuration for ARM64

1. **Verify NVIDIA drivers work with ARM64:**
   ```bash
   nvidia-smi
   ```

2. **Check CUDA compatibility:**
   ```bash
   # Check CUDA version
   nvidia-smi | grep "CUDA Version"
   
   # Test GPU in container
   docker run --rm --gpus all --platform=linux/arm64 \
     ubuntu:22.04 nvidia-smi
   ```

3. **Configure GPU memory:**
   ```bash
   # Set GPU memory fraction if needed
   export CUDA_MEM_FRACTION=0.8
   ```

## Troubleshooting

### Common ARM64 Issues

#### 1. Platform Mismatch Warnings

**Symptom:**
```
WARNING: The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64)
```

**Solution:**
- This is expected for NVIDIA NIM containers
- Docker will use emulation automatically
- Add `--platform=linux/amd64` explicitly to suppress warning

#### 2. Slow Container Startup

**Symptom:**
- Containers take much longer to start on ARM64

**Solution:**
- This is due to emulation overhead
- Consider native installation for critical components
- Use cached images when possible
- Increase startup timeout values

#### 3. GPU Not Accessible in Container

**Symptom:**
```
nvidia-smi: command not found
```

**Solution:**
```bash
# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 4. Out of Memory Errors

**Symptom:**
- Containers crash with OOM errors

**Solution:**
```bash
# Increase Docker memory limit
# Edit /etc/docker/daemon.json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-shm-size": "64G"
}

sudo systemctl restart docker
```

#### 5. AMD64 Container Build Failures

**Symptom:**
- Building AMD64 images fails on ARM64

**Solution:**
```bash
# Set up QEMU for cross-platform builds
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Use buildx for multi-platform builds
docker buildx create --name multiplatform --use
docker buildx inspect --bootstrap
```

## Performance Benchmarking

### Test Container Performance

```bash
# Test native ARM64 performance
time docker run --rm --platform=linux/arm64 ubuntu:22.04 \
  bash -c "for i in {1..1000000}; do echo test > /dev/null; done"

# Test AMD64 emulation performance
time docker run --rm --platform=linux/amd64 ubuntu:22.04 \
  bash -c "for i in {1..1000000}; do echo test > /dev/null; done"
```

### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Log GPU usage
nvidia-smi dmon -s pucvmet -c 100 > gpu_usage.log
```

## Best Practices

### For ARM64 Development

1. **Use native ARM64 Python packages when available:**
   ```bash
   pip install --platform=linux_aarch64 package_name
   ```

2. **Test on both platforms:**
   - Develop on ARM64
   - Validate on AMD64
   - Use GitHub Actions for automated testing

3. **Document platform-specific issues:**
   - Tag issues with architecture labels
   - Include performance comparisons
   - Share workarounds

### For CI/CD

1. **Use self-hosted ARM64 runners:**
   ```yaml
   runs-on: [self-hosted, ARM64, gpu]
   ```

2. **Cache dependencies:**
   ```yaml
   - uses: actions/cache@v4
     with:
       path: ~/.cache/pip
       key: ${{ runner.os }}-${{ runner.arch }}-pip-${{ hashFiles('**/requirements.txt') }}
   ```

3. **Parallelize tests:**
   ```yaml
   strategy:
     matrix:
       component: [alphafold2, rfdiffusion, proteinmpnn]
   ```

## Migration Path

### From AMD64 to ARM64

1. **Assess compatibility:**
   ```bash
   ./detect_platform.sh
   gh workflow run arm64-validation.yml
   ```

2. **Start with Docker emulation:**
   - Low risk, quick start
   - Validate functionality
   - Benchmark performance

3. **Migrate critical components to native:**
   - Start with ML frameworks (PyTorch, JAX)
   - Move to protein design tools
   - Keep non-critical services in Docker

4. **Optimize and tune:**
   - Profile performance
   - Adjust resource limits
   - Fine-tune batch sizes

## Additional Resources

- [Docker Multi-platform builds](https://docs.docker.com/build/building/multi-platform/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [ARM64 Linux Kernel](https://www.kernel.org/doc/html/latest/arm64/index.html)
- [Miniforge for ARM64](https://github.com/conda-forge/miniforge)

## Support

For ARM64-specific issues:

1. Check existing documentation
2. Run diagnostic workflows
3. Search GitHub issues
4. Open a new issue with:
   - Platform information (`uname -a`)
   - Docker version
   - GPU driver version
   - Error logs
   - Output from `./detect_platform.sh`

## Contributing

When contributing ARM64-specific changes:

1. Test on ARM64 hardware
2. Verify AMD64 compatibility not broken
3. Update documentation
4. Add workflow tests
5. Include performance notes
