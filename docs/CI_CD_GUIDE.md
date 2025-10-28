# CI/CD Pipeline Guide

This document describes the CI/CD pipeline setup for the Protein Design project.

## Overview

The project uses GitHub Actions for continuous integration and deployment with multiple specialized workflows:

### Workflows

1. **Main CI/CD Pipeline** (`ci-cd-main.yml`)
   - Code quality checks (linting, formatting, security)
   - Unit and integration tests
   - Multi-platform Docker builds
   - Security scanning
   - Staging deployment

2. **ARM64 Native Pipeline** (`arm64-native-cicd.yml`)
   - ARM64-specific testing on self-hosted runners
   - Native model environment validation
   - Performance testing
   - GPU acceleration testing

3. **Docker Multi-Platform Pipeline** (`docker-multiplatform-cicd.yml`)
   - Multi-architecture Docker builds (AMD64/ARM64)
   - Container security scanning
   - Integration testing

4. **Protein Design Pipeline** (`protein-design-pipeline.yml`)
   - End-to-end protein design workflow
   - Native tool execution
   - Results validation

## Setup Requirements

### For Standard CI/CD (Ubuntu runners):
- Python 3.8+
- Node.js 18+
- Docker with Buildx
- Standard Ubuntu tools

### For ARM64 Native Testing (Self-hosted runners):
- ARM64 Linux system
- NVIDIA GPU support (optional but recommended)
- Conda/Miniforge
- Python 3.8+
- Node.js 18+
- Docker (optional)

## Triggering Workflows

### Automatic Triggers:
- Push to `main`, `develop`, or `dgx-spark` branches
- Pull requests to main branches
- Tag pushes (for releases)

### Manual Triggers:
All workflows support `workflow_dispatch` for manual execution with parameters.

## Configuration

### Environment Variables:
- `MODEL_BACKEND`: `nim` | `native` | `hybrid`
- `NEXT_PUBLIC_MCP_SERVER_URL`: Dashboard backend URL

### Secrets Required:
- `GITHUB_TOKEN`: Automatically provided
- Additional secrets for deployment environments

## Development Workflow

1. **Local Development:**
   ```bash
   # Setup pre-commit hooks
   pre-commit install
   
   # Run local tests
   ./scripts/ci-cd/run-local-tests.sh
   ```

2. **Pull Request Process:**
   - Create feature branch
   - Implement changes
   - Push to GitHub (triggers CI)
   - Address any CI failures
   - Request review

3. **Release Process:**
   - Merge to `develop` for staging
   - Test in staging environment
   - Merge to `main` for production
   - Tag release for version management

## Monitoring and Troubleshooting

### Logs and Artifacts:
- All workflows generate detailed reports
- Artifacts are retained for 30-90 days
- Security scan results in GitHub Security tab

### Common Issues:
1. **ARM64 Runner Availability:** Check self-hosted runner status
2. **Docker Build Failures:** Often related to platform compatibility
3. **Test Failures:** Check environment setup and dependencies

## Best Practices

1. **Code Quality:**
   - Use pre-commit hooks
   - Follow linting standards
   - Write comprehensive tests

2. **Docker Images:**
   - Multi-stage builds for smaller images
   - Non-root users for security
   - Proper health checks

3. **Security:**
   - Regular dependency updates
   - Vulnerability scanning
   - Secrets management

## Extending the Pipeline

To add new workflows or modify existing ones:

1. Create/modify workflow files in `.github/workflows/`
2. Test with `act` tool locally (optional)
3. Validate YAML syntax
4. Test on feature branch first
5. Document any new requirements

## Support

For CI/CD issues:
1. Check workflow logs in GitHub Actions tab
2. Review this documentation
3. Check system requirements
4. Contact maintainers if needed
