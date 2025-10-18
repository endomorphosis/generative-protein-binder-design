# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the Protein Binder Design project, specifically optimized for ARM64 self-hosted runners with NVIDIA GPU support.

## Available Workflows

### 1. System Health Check (`system-health.yml`)
- **Trigger**: Push to main/develop, PRs, daily schedule, manual dispatch
- **Purpose**: Monitor system health and resource availability
- **Features**:
  - System information reporting
  - GPU status monitoring
  - Docker environment validation
  - Storage analysis
  - Network connectivity checks

### 2. ARM64 Native Installation Test (`native-install-test.yml`)
- **Trigger**: Manual dispatch with component selection
- **Purpose**: Test native ARM64 installation of protein design tools
- **Components**:
  - AlphaFold2 (JAX-based structure prediction)
  - RFDiffusion (PyTorch-based protein diffusion)
  - ProteinMPNN (sequence design)
- **Features**: Environment isolation, comprehensive testing, cleanup

### 3. Docker Compatibility Test (`docker-test.yml`)
- **Trigger**: Manual dispatch with platform selection
- **Purpose**: Test Docker container compatibility on ARM64
- **Platforms**: 
  - `linux/arm64` (native)
  - `linux/amd64` (emulated)
- **Features**: Platform emulation, NVIDIA runtime testing, image building

### 4. Protein Design Pipeline (`protein-design-pipeline.yml`)
- **Trigger**: Manual dispatch with configuration options
- **Purpose**: Full protein binder design pipeline
- **Pipeline Steps**:
  1. Target protein download/preparation
  2. AlphaFold2 structure prediction
  3. RFDiffusion binder generation
  4. ProteinMPNN sequence design
  5. AlphaFold2 Multimer validation
- **Features**: Native ARM64 execution, result packaging, progress tracking

### 5. Jupyter Notebook Test (`jupyter-test.yml`)
- **Trigger**: Changes to notebooks, PRs, manual dispatch
- **Purpose**: Test Jupyter notebook functionality
- **Features**:
  - Environment setup and kernel installation
  - Notebook execution testing
  - HTML conversion
  - Scientific package validation
  - GPU access verification

## Runner Requirements

These workflows are designed for self-hosted runners with the following labels:
- `self-hosted`
- `ARM64` 
- `gpu`

### System Requirements
- **Architecture**: ARM64 (aarch64)
- **OS**: Ubuntu 22.04 or 24.04
- **GPU**: NVIDIA GPU with drivers ≥ 525.60.13
- **Memory**: ≥ 120GB RAM recommended
- **Storage**: ≥ 1TB available space
- **Docker**: ≥ 28.0 with NVIDIA Container Runtime

## Usage Examples

### Run System Health Check
```bash
# Automatic - runs daily at 8:00 AM UTC
# Manual trigger with deep analysis:
gh workflow run system-health.yml --ref main -f deep_check=true
```

### Test Native Installation
```bash
# Test AlphaFold2 only
gh workflow run native-install-test.yml --ref main -f component=alphafold2

# Test all components
gh workflow run native-install-test.yml --ref main -f component=all

# Skip conda installation (if already installed)
gh workflow run native-install-test.yml --ref main -f component=alphafold2 -f skip_conda=true
```

### Test Docker Compatibility
```bash
# Test ARM64 native containers
gh workflow run docker-test.yml --ref main -f platform=linux/arm64

# Test AMD64 emulation
gh workflow run docker-test.yml --ref main -f platform=linux/amd64

# Test both platforms with forced rebuild
gh workflow run docker-test.yml --ref main -f platform=both -f force_build=true
```

### Run Protein Design Pipeline
```bash
# Design binders for PDB 7BZ5 (10 designs, native installation)
gh workflow run protein-design-pipeline.yml --ref main \
  -f target_protein=7BZ5 \
  -f num_designs=10 \
  -f use_native=true

# Design binders using Docker (fallback)
gh workflow run protein-design-pipeline.yml --ref main \
  -f target_protein=1ABC \
  -f num_designs=5 \
  -f use_native=false
```

### Test Jupyter Notebooks
```bash
# Test main notebook
gh workflow run jupyter-test.yml --ref main

# Test specific notebook
gh workflow run jupyter-test.yml --ref main \
  -f notebook_path=src/protein-binder-design.ipynb
```

## Workflow Artifacts

Each workflow generates artifacts that are stored for 30 days:

- **System Health**: `system-health-report-{run_number}`
- **Native Install**: `arm64-installation-report-{component}-{run_number}`
- **Docker Test**: `docker-compatibility-report-{platform}-{run_number}`
- **Pipeline**: `protein-design-results-{target}-{run_number}`
- **Jupyter**: `jupyter-test-results-{run_number}`

## Troubleshooting

### Common Issues

1. **GPU Access Denied**
   - Ensure runner has proper NVIDIA drivers
   - Check `nvidia-container-runtime` installation
   - Verify Docker daemon configuration

2. **ARM64 Container Issues**
   - Use native ARM64 base images when possible
   - Enable platform emulation for AMD64 containers
   - Consider using native installation instead

3. **Memory/Storage Issues**
   - Monitor system resources with health check
   - Clean up conda environments between runs
   - Use Docker image pruning

4. **Network Connectivity**
   - Ensure access to GitHub, NGC, PyPI
   - Check corporate firewall settings
   - Verify DNS resolution

### Runner Maintenance

```bash
# Check runner status
sudo systemctl status github-runner

# View runner logs  
sudo journalctl -u github-runner -f

# Restart runner service
sudo systemctl restart github-runner

# Update runner (when new versions available)
cd /opt/actions-runner
sudo ./config.sh remove
# Re-run setup_github_runner.sh
```

## Security Considerations

- Workflows run with repository secrets access
- Self-hosted runners have access to local system
- Consider using Docker isolation for untrusted code
- Regularly update runner and system packages
- Monitor workflow logs for security issues

## Contributing

When adding new workflows:

1. Follow existing naming conventions
2. Include comprehensive error handling
3. Add artifact upload for important results
4. Test on ARM64 runner before committing
5. Update this README with new workflow details

## Support

For issues with workflows:
1. Check workflow run logs in GitHub Actions tab
2. Review runner system logs
3. Test components individually before full pipeline
4. Consult ARM64 compatibility documentation