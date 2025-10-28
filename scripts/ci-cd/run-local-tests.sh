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
