#!/bin/bash
# Platform Detection and Guidance Script
# This script detects the system architecture and provides guidance for running the project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================"
echo "  Protein Binder Design - Platform Detection"
echo "================================================"
echo

# Detect architecture
ARCH=$(uname -m)
echo -e "${BLUE}System Architecture:${NC} $ARCH"

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo -e "${BLUE}Operating System:${NC} $NAME $VERSION_ID"
else
    echo -e "${YELLOW}Operating System: Unknown${NC}"
fi

# Detect kernel
echo -e "${BLUE}Kernel:${NC} $(uname -r)"
echo

# Check for GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    echo "  GPU Count: $GPU_COUNT"
    echo "  GPU Model: $GPU_NAME"
    echo "  Driver Version: $GPU_DRIVER"
else
    echo -e "${RED}✗ No NVIDIA GPU detected${NC}"
    echo "  Note: GPU is required for protein design workflows"
fi
echo

# Check Docker
echo "Checking for Docker..."
if command -v docker >/dev/null 2>&1; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
    echo -e "${GREEN}✓ Docker detected${NC}"
    echo "  Version: $DOCKER_VERSION"
    
    # Check if Docker daemon is running
    if docker info >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Docker daemon is running${NC}"
        
        # Check for NVIDIA Container Runtime
        if docker info 2>/dev/null | grep -q nvidia; then
            echo -e "${GREEN}✓ NVIDIA Container Runtime available${NC}"
        else
            echo -e "${YELLOW}⚠ NVIDIA Container Runtime not detected${NC}"
        fi
    else
        echo -e "${RED}✗ Docker daemon not accessible${NC}"
    fi
else
    echo -e "${RED}✗ Docker not found${NC}"
fi
echo

# Platform-specific recommendations
echo "================================================"
echo "  Platform Analysis & Recommendations"
echo "================================================"
echo

case "$ARCH" in
    aarch64|arm64)
        echo -e "${GREEN}Platform: ARM64 (aarch64)${NC}"
        echo
        echo "Your system is running on ARM64 architecture."
        echo
        echo -e "${YELLOW}IMPORTANT:${NC} NVIDIA NIM containers are built for AMD64/x86_64 architecture."
        echo "When running on ARM64, Docker will use emulation, which may cause:"
        echo "  • Performance degradation"
        echo "  • Increased memory usage"
        echo "  • Potential compatibility issues"
        echo
        echo -e "${BLUE}Recommended Approaches:${NC}"
        echo
        echo "1. ${GREEN}Native Installation (Recommended)${NC}"
        echo "   Install protein design tools natively on ARM64 for best performance:"
        echo "   → See: ARM64_NATIVE_INSTALLATION.md"
        echo "   → Use GitHub Actions workflow: native-install-test.yml"
        echo "   → Command: gh workflow run native-install-test.yml -f component=all"
        echo
        echo "2. ${YELLOW}Docker with Emulation (Fallback)${NC}"
        echo "   Run AMD64 containers with emulation:"
        echo "   → Ensure QEMU emulation is set up"
        echo "   → Use: ./scripts/run_dashboard_stack.sh --emulated up -d"
        echo "   → Note: Expect slower performance"
        echo
        echo "3. ${BLUE}Hybrid Approach${NC}"
        echo "   Mix native tools with Docker containers:"
        echo "   → Install PyTorch/JAX natively for ARM64"
        echo "   → Use Docker only for unavailable components"
        echo "   → See native installation guide for details"
        echo
        echo -e "${BLUE}Quick Start Commands:${NC}"
        echo "  # Run platform validation"
        echo "  gh workflow run arm64-validation.yml"
        echo
        echo "  # Test native installation"
        echo "  gh workflow run native-install-test.yml -f component=alphafold2"
        echo
        echo "  # Run protein design pipeline (native)"
        echo "  gh workflow run protein-design-pipeline.yml -f use_native=true"
        ;;
        
    x86_64|amd64)
        echo -e "${GREEN}Platform: AMD64 (x86_64)${NC}"
        echo
        echo "Your system is running on AMD64/x86_64 architecture."
        echo "This is the native architecture for NVIDIA NIM containers."
        echo
        echo -e "${GREEN}✓ Optimal configuration for this project${NC}"
        echo
        echo -e "${BLUE}Recommended Approach:${NC}"
        echo
        echo "1. ${GREEN}Docker Compose (Recommended)${NC}"
        echo "   Run all services using Docker:"
        echo "   → Use: ./scripts/run_dashboard_stack.sh up -d"
        echo "   → All containers will run natively without emulation"
        echo "   → Best performance and compatibility"
        echo
        echo -e "${BLUE}Quick Start Commands:${NC}"
        echo "  # Set up NGC API key"
        echo "  export NGC_CLI_API_KEY=your_key_here"
        echo
        echo "  # Start all services"
        echo "  cd deploy && docker compose up -d"
        echo
        echo "  # Or use single GPU mode"
        echo "  ./deploy/run_single_gpu.sh"
        ;;
        
    *)
        echo -e "${RED}Platform: Unknown ($ARCH)${NC}"
        echo
        echo "This architecture is not officially supported."
        echo "The project is designed for AMD64 and ARM64 architectures."
        echo
        echo "You may encounter compatibility issues."
        ;;
esac

echo
echo "================================================"
echo "  Additional Resources"
echo "================================================"
echo
echo "Documentation:"
echo "  • README.md - Project overview"
echo "  • LOCAL_SETUP.md - Detailed setup instructions"
echo "  • ARM64_COMPATIBILITY.md - ARM64 compatibility guide"
echo "  • ARM64_NATIVE_INSTALLATION.md - Native installation guide"
echo "  • .github/workflows/README.md - GitHub Actions workflows"
echo
echo "GitHub Actions Workflows:"
echo "  • runner-test.yml - Test runner connectivity"
echo "  • system-health.yml - System health monitoring"
echo "  • docker-test.yml - Docker compatibility testing"
echo "  • native-install-test.yml - Native installation testing"
echo "  • protein-design-pipeline.yml - Full pipeline execution"
echo "  • arm64-validation.yml - ARM64 platform validation"
echo
echo "For help, visit: https://github.com/hallucinate-llc/generative-protein-binder-design"
echo
