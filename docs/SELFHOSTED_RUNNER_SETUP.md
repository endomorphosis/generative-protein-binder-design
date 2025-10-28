# Self-Hosted Runner CI/CD Setup

This document describes the self-hosted GitHub Actions runner setup for the generative protein binder design project.

## Overview

The project now includes a self-hosted runner on the `spark-b271` ARM64 system that provides:

- **Automated testing** on ARM64 architecture
- **Docker container building and testing**
- **Native model environment validation**
- **CI/CD pipeline for protein design workflows**

## Runner Details

- **Runner Name**: `arm64-gpu-runner-spark-b271`
- **Labels**: `self-hosted`, `ARM64`, `docker`, `protein-design`
- **Location**: `/home/barberb/actions-runner`
- **Service**: `actions.runner.hallucinate-llc-generative-protein-binder-design.arm64-gpu-runner-spark-b271.service`

## Current Workflows

### 1. Self-Hosted CI/CD (`selfhosted-ci.yml`)

**Triggers:**
- Push to `main` or `dgx-spark` branches
- Manual workflow dispatch

**Jobs:**
- **System Info**: Display system specifications
- **Docker Build & Test**: Build and test MCP Server container
- **Cleanup**: Remove test containers and images

## Setup Scripts

### `scripts/setup-github-runner.sh`

Comprehensive script to install, configure, and manage the GitHub Actions runner:

```bash
# Install runner
./scripts/setup-github-runner.sh install

# Check status  
./scripts/setup-github-runner.sh status

# Remove runner
./scripts/setup-github-runner.sh remove
```

**Features:**
- Automatic runner download and setup
- Service installation and management
- Environment validation
- GitHub CLI integration

## Manual Runner Management

### Service Control

```bash
# Check status
sudo systemctl status actions.runner.hallucinate-llc-generative-protein-binder-design.arm64-gpu-runner-spark-b271.service

# Start service
sudo systemctl start actions.runner.hallucinate-llc-generative-protein-binder-design.arm64-gpu-runner-spark-b271.service

# Stop service
sudo systemctl stop actions.runner.hallucinate-llc-generative-protein-binder-design.arm64-gpu-runner-spark-b271.service

# View logs
sudo systemctl logs actions.runner.hallucinate-llc-generative-protein-binder-design.arm64-gpu-runner-spark-b271.service
```

### Alternative Service Control

```bash
cd /home/barberb/actions-runner

# Service operations
sudo ./svc.sh status
sudo ./svc.sh start
sudo ./svc.sh stop
sudo ./svc.sh install
sudo ./svc.sh uninstall
```

## Docker Configuration

The MCP Server includes a Dockerfile optimized for the CI/CD pipeline:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "server.py"]
```

## Environment Requirements

The runner system includes:

- **Docker**: Container runtime for building and testing
- **Python 3.11+**: For MCP server and testing
- **GitHub CLI**: For runner authentication and management
- **Conda**: For native model environments (optional)
- **Git**: For source control integration

## Troubleshooting

### Runner Not Starting

1. Check service status:
   ```bash
   ./scripts/setup-github-runner.sh status
   ```

2. Check system logs:
   ```bash
   sudo journalctl -u actions.runner.* -f
   ```

### Docker Issues

1. Verify Docker is accessible:
   ```bash
   docker ps
   ```

2. Check user permissions:
   ```bash
   groups $USER | grep docker
   ```

### Workflow Failures

1. Check workflow logs in GitHub Actions UI
2. Verify runner labels match workflow requirements
3. Ensure Docker containers can bind to ports

## Security Considerations

- Runner runs as user `barberb` (non-root)
- Docker access is limited to the runner user
- Sensitive secrets are managed through GitHub Secrets
- Containers are cleaned up after each test

## Monitoring

- **GitHub Actions UI**: View workflow runs and logs
- **System logs**: Monitor runner service health
- **Resource usage**: Monitor CPU, memory, and disk usage during builds

## Future Enhancements

Planned improvements:
- Multi-job parallel testing
- Integration with native protein design models
- Automated deployment to staging environments
- Performance benchmarking in CI pipeline
- Security scanning integration

## Contact

For issues with the runner setup, check:
1. GitHub Actions workflow logs
2. System service logs  
3. Runner configuration in `/home/barberb/actions-runner`