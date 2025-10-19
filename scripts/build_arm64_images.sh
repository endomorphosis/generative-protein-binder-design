#!/bin/bash
# Build ARM64 Native Docker Images
# This script builds all ARM64-native Docker images for protein design tools

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

echo "========================================================"
echo "  ARM64 Native Docker Images Build"
echo "========================================================"
echo

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    print_warning "Warning: Building on $ARCH, not ARM64"
    print_info "Images will be cross-compiled"
fi
print_status "Host architecture: $ARCH"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker first."
    exit 1
fi
print_status "Docker found"

# Check buildx
if ! docker buildx version &> /dev/null; then
    print_warning "Docker buildx not found. Installing..."
    docker buildx install
fi
print_status "Docker buildx available"

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy"

cd "$DEPLOY_DIR"

# Create buildx builder for multi-platform if needed
BUILDER_NAME="arm64-builder"
if ! docker buildx ls | grep -q "$BUILDER_NAME"; then
    print_info "Creating buildx builder: $BUILDER_NAME"
    docker buildx create --name "$BUILDER_NAME" --use
    docker buildx inspect --bootstrap
fi

print_info "Using builder: $BUILDER_NAME"
docker buildx use "$BUILDER_NAME"

# Build menu
echo
echo "Select images to build:"
echo "  1) AlphaFold2 ARM64"
echo "  2) RFDiffusion ARM64"
echo "  3) ProteinMPNN ARM64"
echo "  4) All images"
echo "  5) Exit"
echo

read -p "Enter your choice (1-5): " choice

build_alphafold() {
    print_info "Building AlphaFold2 ARM64 image..."
    docker buildx build \
        --platform linux/arm64 \
        -f Dockerfile.alphafold2-arm64 \
        -t protein-binder/alphafold2:arm64-latest \
        --load \
        . || print_error "AlphaFold2 build failed"
    
    print_status "AlphaFold2 ARM64 image built"
}

build_rfdiffusion() {
    print_info "Building RFDiffusion ARM64 image..."
    docker buildx build \
        --platform linux/arm64 \
        -f Dockerfile.rfdiffusion-arm64 \
        -t protein-binder/rfdiffusion:arm64-latest \
        --load \
        . || print_error "RFDiffusion build failed"
    
    print_status "RFDiffusion ARM64 image built"
}

build_proteinmpnn() {
    print_info "Building ProteinMPNN ARM64 image..."
    docker buildx build \
        --platform linux/arm64 \
        -f Dockerfile.proteinmpnn-arm64 \
        -t protein-binder/proteinmpnn:arm64-latest \
        --load \
        . || print_error "ProteinMPNN build failed"
    
    print_status "ProteinMPNN ARM64 image built"
}

case $choice in
    1)
        build_alphafold
        ;;
    2)
        build_rfdiffusion
        ;;
    3)
        build_proteinmpnn
        ;;
    4)
        print_info "Building all ARM64 images..."
        echo
        build_alphafold
        echo
        build_rfdiffusion
        echo
        build_proteinmpnn
        ;;
    5)
        print_info "Build cancelled"
        exit 0
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo
echo "========================================================"
echo "  ✓ Build Complete!"
echo "========================================================"
echo
echo "Built images:"
docker images | grep "protein-binder.*arm64" || echo "No ARM64 images found"
echo
echo "To test an image:"
echo "  docker run --rm protein-binder/alphafold2:arm64-latest"
echo "  docker run --rm protein-binder/rfdiffusion:arm64-latest"
echo "  docker run --rm protein-binder/proteinmpnn:arm64-latest"
echo
echo "To start all services:"
echo "  cd deploy && docker compose -f docker-compose-arm64-native.yaml up -d"
echo
