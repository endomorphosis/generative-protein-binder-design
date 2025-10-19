#!/bin/bash
# Trigger PyTorch ARM64 Source Build Workflow
# This script triggers the GitHub Actions workflow to build PyTorch from source

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

echo "========================================================"
echo "  PyTorch ARM64 Source Build - Workflow Trigger"
echo "========================================================"
echo

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    print_error "GitHub CLI (gh) is not installed"
    echo
    echo "Install it with:"
    echo "  Ubuntu/Debian: sudo apt install gh"
    echo "  macOS: brew install gh"
    echo "  Or visit: https://cli.github.com/"
    exit 1
fi

print_status "GitHub CLI found"

# Check if authenticated
if ! gh auth status &> /dev/null; then
    print_warning "Not authenticated with GitHub"
    echo
    print_info "Authenticating..."
    gh auth login
fi

print_status "Authenticated with GitHub"

# Get options
CUDA_VERSION="${1:-11.8}"
USE_CUDA="${2:-true}"
UPLOAD_ARTIFACT="${3:-true}"

echo
print_info "Build Configuration:"
echo "  CUDA Version: $CUDA_VERSION"
echo "  Use CUDA: $USE_CUDA"
echo "  Upload Artifact: $UPLOAD_ARTIFACT"
echo

# Confirm
read -p "Trigger PyTorch ARM64 build workflow? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Cancelled"
    exit 0
fi

# Trigger workflow
print_info "Triggering workflow..."
gh workflow run pytorch-arm64-build.yml \
    -f cuda_version="$CUDA_VERSION" \
    -f use_cuda="$USE_CUDA" \
    -f upload_artifact="$UPLOAD_ARTIFACT"

if [ $? -eq 0 ]; then
    print_status "Workflow triggered successfully!"
    echo
    print_info "Monitor progress:"
    echo "  gh run list --workflow=pytorch-arm64-build.yml"
    echo "  gh run watch"
    echo
    print_info "Or visit:"
    echo "  https://github.com/hallucinate-llc/generative-protein-binder-design/actions"
    echo
    print_warning "Build will take 1-3 hours to complete"
else
    print_error "Failed to trigger workflow"
    exit 1
fi
