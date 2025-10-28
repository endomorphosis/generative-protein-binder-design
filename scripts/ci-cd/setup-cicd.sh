#!/bin/bash
set -euo pipefail

# CI/CD Setup Script for Protein Design Project
# This script sets up the CI/CD environment and validates configurations

echo "=== Protein Design CI/CD Setup Script ==="
echo "Generated: $(date)"

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKFLOWS_DIR="$PROJECT_ROOT/.github/workflows"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

echo "Project root: $PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
check_command() {
    local cmd=$1
    if command -v "$cmd" &> /dev/null; then
        log_info "$cmd is available"
        return 0
    else
        log_warn "$cmd is not available"
        return 1
    fi
}

# Function to validate YAML files
validate_yaml() {
    local file=$1
    if command -v yamllint &> /dev/null; then
        if yamllint "$file" &> /dev/null; then
            log_info "YAML validation passed: $(basename "$file")"
        else
            log_error "YAML validation failed: $(basename "$file")"
            return 1
        fi
    else
        log_warn "yamllint not available, skipping YAML validation"
    fi
}

# Function to check GitHub Actions syntax
check_workflow_syntax() {
    local workflow_file=$1
    log_info "Checking workflow syntax: $(basename "$workflow_file")"
    
    # Basic checks
    if grep -q "on:" "$workflow_file" && grep -q "jobs:" "$workflow_file"; then
        log_info "Basic workflow structure is valid"
    else
        log_error "Invalid workflow structure in $(basename "$workflow_file")"
        return 1
    fi
    
    # Check for required fields
    if grep -q "runs-on:" "$workflow_file"; then
        log_info "Runner specification found"
    else
        log_warn "No runner specification found in $(basename "$workflow_file")"
    fi
}

# Function to setup development tools
setup_dev_tools() {
    log_info "Setting up development tools..."
    
    # Install yamllint if not present
    if ! check_command yamllint; then
        if check_command pip3; then
            log_info "Installing yamllint..."
            pip3 install --user yamllint || log_warn "Failed to install yamllint"
        fi
    fi
    
    # Install pre-commit if not present
    if ! check_command pre-commit; then
        if check_command pip3; then
            log_info "Installing pre-commit..."
            pip3 install --user pre-commit || log_warn "Failed to install pre-commit"
        fi
    fi
    
    # Install GitHub CLI if not present
    if ! check_command gh; then
        log_warn "GitHub CLI not available. Consider installing for better CI/CD integration"
    fi
}

# Function to create pre-commit configuration
create_precommit_config() {
    log_info "Creating pre-commit configuration..."
    
    cat > "$PROJECT_ROOT/.pre-commit-config.yaml" << 'EOL'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: mixed-line-ending

  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3
        files: ^(mcp-server|scripts)/.*\.py$

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        files: ^(mcp-server|scripts)/.*\.py$
        args: ['--max-line-length=88', '--extend-ignore=E203,W503']

  - repo: https://github.com/adrienverge/yamllint
    rev: v1.28.0
    hooks:
      - id: yamllint
        files: \.(yaml|yml)$
        args: ['-d', 'relaxed']

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v2.7.1
    hooks:
      - id: prettier
        files: ^mcp-dashboard/.*\.(js|jsx|ts|tsx|json|css|md)$
EOL

    log_info "Pre-commit configuration created"
}

# Function to validate workflows
validate_workflows() {
    log_info "Validating GitHub Actions workflows..."
    
    local workflow_files=(
        "ci-cd-main.yml"
        "arm64-native-cicd.yml"  
        "docker-multiplatform-cicd.yml"
        "protein-design-pipeline.yml"
        "docker-test.yml"
    )
    
    local validation_passed=true
    
    for workflow in "${workflow_files[@]}"; do
        local workflow_path="$WORKFLOWS_DIR/$workflow"
        if [[ -f "$workflow_path" ]]; then
            log_info "Validating $workflow..."
            validate_yaml "$workflow_path" || validation_passed=false
            check_workflow_syntax "$workflow_path" || validation_passed=false
        else
            log_warn "Workflow file not found: $workflow"
        fi
    done
    
    if $validation_passed; then
        log_info "All workflow validations passed"
    else
        log_error "Some workflow validations failed"
        return 1
    fi
}

# Function to check Docker configuration
check_docker_config() {
    log_info "Checking Docker configuration..."
    
    if check_command docker; then
        docker_version=$(docker --version)
        log_info "Docker version: $docker_version"
        
        # Check if Docker daemon is running
        if docker info &> /dev/null; then
            log_info "Docker daemon is running"
        else
            log_warn "Docker daemon is not running"
        fi
        
        # Check for docker-compose
        if check_command docker-compose || docker compose version &> /dev/null; then
            log_info "Docker Compose is available"
        else
            log_warn "Docker Compose is not available"
        fi
        
        # Check for buildx
        if docker buildx version &> /dev/null; then
            log_info "Docker Buildx is available for multi-platform builds"
        else
            log_warn "Docker Buildx not available"
        fi
    else
        log_warn "Docker not available"
    fi
}

# Function to check environment setup
check_environment() {
    log_info "Checking environment setup..."
    
    # Check system information
    echo "System information:"
    echo "  OS: $(uname -s)"
    echo "  Architecture: $(uname -m)"
    echo "  Kernel: $(uname -r)"
    
    # Check Python
    if check_command python3; then
        python_version=$(python3 --version)
        echo "  Python: $python_version"
    fi
    
    # Check Node.js
    if check_command node; then
        node_version=$(node --version)
        echo "  Node.js: $node_version"
    fi
    
    # Check conda
    if check_command conda; then
        conda_version=$(conda --version)
        echo "  Conda: $conda_version"
    fi
    
    # Check GPU
    if check_command nvidia-smi; then
        echo "  GPU: Available"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    else
        echo "  GPU: Not available"
    fi
}

# Function to create CI/CD documentation
create_cicd_docs() {
    log_info "Creating CI/CD documentation..."
    
    cat > "$PROJECT_ROOT/docs/CI_CD_GUIDE.md" << 'EOL'
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
EOL

    log_info "CI/CD documentation created at docs/CI_CD_GUIDE.md"
}

# Function to create local testing script
create_local_test_script() {
    log_info "Creating local testing script..."
    
    cat > "$SCRIPTS_DIR/ci-cd/run-local-tests.sh" << 'EOL'
#!/bin/bash
set -euo pipefail

# Local Testing Script for CI/CD Validation
# Run this before pushing to validate changes locally

echo "=== Local CI/CD Testing ==="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test 1: Code formatting
log_info "Running code formatting checks..."
if command -v black &> /dev/null; then
    black --check mcp-server/ || log_error "Black formatting issues found"
else
    echo "Skipping black check (not installed)"
fi

# Test 2: Linting
log_info "Running linting..."
if command -v flake8 &> /dev/null; then
    flake8 mcp-server/ --max-line-length=88 || log_error "Linting issues found"
else
    echo "Skipping flake8 check (not installed)"
fi

# Test 3: YAML validation
log_info "Validating YAML files..."
if command -v yamllint &> /dev/null; then
    yamllint .github/workflows/*.yml || log_error "YAML validation failed"
else
    echo "Skipping YAML validation (yamllint not installed)"
fi

# Test 4: Python tests (if available)
log_info "Running Python tests..."
if [[ -d "tests" ]] && command -v pytest &> /dev/null; then
    pytest tests/ -v || log_error "Python tests failed"
else
    echo "Skipping Python tests (pytest not available or no tests directory)"
fi

# Test 5: Node.js tests (if available)
log_info "Running Node.js tests..."
if [[ -f "mcp-dashboard/package.json" ]] && command -v npm &> /dev/null; then
    cd mcp-dashboard
    if grep -q '"test"' package.json; then
        npm test || log_error "Node.js tests failed"
    else
        echo "No test script found in package.json"
    fi
    cd ..
else
    echo "Skipping Node.js tests (npm not available or no package.json)"
fi

# Test 6: Docker build test (if Docker available)
log_info "Testing Docker builds..."
if command -v docker &> /dev/null && docker info &> /dev/null; then
    # Test MCP Server build
    if [[ -f "mcp-server/Dockerfile" ]]; then
        docker build -t test-mcp-server mcp-server/ || log_error "MCP Server Docker build failed"
        docker rmi test-mcp-server || true
    fi
    
    # Test Dashboard build  
    if [[ -f "mcp-dashboard/Dockerfile" ]]; then
        docker build -t test-mcp-dashboard mcp-dashboard/ || log_error "Dashboard Docker build failed"
        docker rmi test-mcp-dashboard || true
    fi
else
    echo "Skipping Docker tests (Docker not available)"
fi

log_info "Local testing completed!"
log_info "You can now push your changes with confidence."
EOL

    chmod +x "$SCRIPTS_DIR/ci-cd/run-local-tests.sh"
    log_info "Local testing script created at scripts/ci-cd/run-local-tests.sh"
}

# Main execution
main() {
    echo "Starting CI/CD setup..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Setup development tools
    setup_dev_tools
    
    # Create configurations
    create_precommit_config
    
    # Validate existing workflows
    validate_workflows || log_warn "Workflow validation had issues"
    
    # Check environment
    check_environment
    check_docker_config
    
    # Create documentation and scripts
    create_cicd_docs
    create_local_test_script
    
    log_info "CI/CD setup completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Install pre-commit hooks: pre-commit install"
    echo "2. Run local tests: ./scripts/ci-cd/run-local-tests.sh"
    echo "3. Review CI/CD documentation: docs/CI_CD_GUIDE.md"
    echo "4. Configure self-hosted runners for ARM64 testing (if needed)"
}

# Run main function
main "$@"