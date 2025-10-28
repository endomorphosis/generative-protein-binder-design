#!/bin/bash
set -euo pipefail

# Docker Build and Test Script for CI/CD
# This script builds and tests Docker images locally before pushing

echo "=== Docker Build and Test Script ==="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
REGISTRY="ghcr.io"
IMAGE_NAME="hallucinate-llc/generative-protein-binder-design"
TAG="${1:-local-test}"
PLATFORMS="${2:-linux/amd64}"
PUSH="${3:-false}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check for buildx
    if ! docker buildx version &> /dev/null; then
        log_warn "Docker Buildx not available, using legacy build"
        USE_BUILDX=false
    else
        USE_BUILDX=true
        log_info "Docker Buildx available for multi-platform builds"
    fi
    
    log_info "Prerequisites check passed"
}

# Function to create Dockerfiles if they don't exist
create_dockerfiles() {
    log_info "Ensuring Dockerfiles exist..."
    
    # MCP Server Dockerfile
    if [[ ! -f "mcp-server/Dockerfile" ]]; then
        log_info "Creating MCP Server Dockerfile..."
        cat > mcp-server/Dockerfile << 'EOL'
FROM python:3.9-slim

# Multi-platform build args
ARG TARGETPLATFORM
ARG BUILDPLATFORM
RUN echo "Building MCP Server on $BUILDPLATFORM for $TARGETPLATFORM"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app
USER mcpuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Environment variables
ENV MODEL_BACKEND=native
ENV PYTHONPATH=/app

# Start command
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
EOL
    fi
    
    # MCP Dashboard Dockerfile
    if [[ ! -f "mcp-dashboard/Dockerfile" ]]; then
        log_info "Creating MCP Dashboard Dockerfile..."
        cat > mcp-dashboard/Dockerfile << 'EOL'
# Build stage
FROM node:18-slim AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM
RUN echo "Building Dashboard on $BUILDPLATFORM for $TARGETPLATFORM"

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage  
FROM node:18-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built application
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules

# Create non-root user
RUN useradd -m -u 1000 dashuser && \
    chown -R dashuser:dashuser /app
USER dashuser

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:3000 || exit 1

# Environment variables
ENV NODE_ENV=production
ENV NEXT_PUBLIC_MCP_SERVER_URL=http://localhost:8001

# Start command
CMD ["npm", "start"]
EOL
    fi
}

# Function to build Docker image
build_image() {
    local component=$1
    local dockerfile_path="$component/Dockerfile"
    local context_path="$component"
    local image_tag="$REGISTRY/$IMAGE_NAME-$component:$TAG"
    
    log_info "Building $component image..."
    
    if [[ ! -f "$dockerfile_path" ]]; then
        log_error "Dockerfile not found: $dockerfile_path"
        return 1
    fi
    
    local build_args=(
        "--tag" "$image_tag"
        "--file" "$dockerfile_path"
    )
    
    # Add platform if using buildx
    if $USE_BUILDX && [[ "$PLATFORMS" != "linux/amd64" ]]; then
        build_args+=("--platform" "$PLATFORMS")
    fi
    
    # Add push flag if requested
    if [[ "$PUSH" == "true" ]] && $USE_BUILDX; then
        build_args+=("--push")
    fi
    
    # Add build context
    build_args+=("$context_path")
    
    # Execute build
    if $USE_BUILDX; then
        docker buildx build "${build_args[@]}" || return 1
    else
        docker build "${build_args[@]}" || return 1
    fi
    
    log_info "$component image built successfully"
    
    # Store image info for testing (only for local builds)
    if [[ "$PUSH" != "true" ]]; then
        echo "$image_tag" >> /tmp/built_images.txt
    fi
}

# Function to test Docker image
test_image() {
    local component=$1
    local image_tag="$REGISTRY/$IMAGE_NAME-$component:$TAG"
    local port
    
    # Skip testing if pushed to registry
    if [[ "$PUSH" == "true" ]]; then
        log_info "Skipping local test for pushed image: $component"
        return 0
    fi
    
    log_info "Testing $component image..."
    
    # Determine port
    case "$component" in
        "mcp-server")
            port=8001
            ;;
        "mcp-dashboard")
            port=3000
            ;;
        *)
            log_warn "Unknown component for testing: $component"
            return 0
            ;;
    esac
    
    # Start container
    local container_name="test-$component-$$"
    
    log_info "Starting test container: $container_name"
    
    if ! docker run -d \
        --name "$container_name" \
        -p "$port:$port" \
        "$image_tag"; then
        log_error "Failed to start container: $container_name"
        return 1
    fi
    
    # Wait for container to start
    log_info "Waiting for container to start..."
    sleep 10
    
    # Test health endpoint
    local test_passed=true
    case "$component" in
        "mcp-server")
            if curl -f "http://localhost:$port/health" &> /dev/null; then
                log_info "MCP Server health check passed"
            else
                log_error "MCP Server health check failed"
                test_passed=false
            fi
            ;;
        "mcp-dashboard")
            if curl -f "http://localhost:$port" &> /dev/null; then
                log_info "Dashboard accessibility check passed"
            else
                log_error "Dashboard accessibility check failed" 
                test_passed=false
            fi
            ;;
    esac
    
    # Get container logs
    log_info "Container logs for $component:"
    docker logs "$container_name" 2>&1 | tail -20
    
    # Cleanup
    log_info "Cleaning up test container..."
    docker stop "$container_name" &> /dev/null || true
    docker rm "$container_name" &> /dev/null || true
    
    if $test_passed; then
        log_info "$component image test passed"
    else
        log_error "$component image test failed"
        return 1
    fi
}

# Function to build and test all components
build_all() {
    local components=("mcp-server" "mcp-dashboard")
    local failed_builds=()
    local failed_tests=()
    
    # Clean previous results
    rm -f /tmp/built_images.txt
    touch /tmp/built_images.txt
    
    log_info "Building all components for platforms: $PLATFORMS"
    
    for component in "${components[@]}"; do
        log_info "Processing component: $component"
        
        if build_image "$component"; then
            log_info "Build succeeded for $component"
            
            # Test if not pushing (local build only)
            if [[ "$PUSH" != "true" ]]; then
                if test_image "$component"; then
                    log_info "Test passed for $component"
                else
                    failed_tests+=("$component")
                fi
            fi
        else
            log_error "Build failed for $component"
            failed_builds+=("$component")
        fi
        
        echo "---"
    done
    
    # Report results
    echo
    log_info "Build and test summary:"
    
    if [[ ${#failed_builds[@]} -eq 0 ]]; then
        log_info "✅ All builds succeeded"
    else
        log_error "❌ Failed builds: ${failed_builds[*]}"
    fi
    
    if [[ ${#failed_tests[@]} -eq 0 ]] && [[ "$PUSH" != "true" ]]; then
        log_info "✅ All tests passed"
    elif [[ ${#failed_tests[@]} -gt 0 ]]; then
        log_error "❌ Failed tests: ${failed_tests[*]}"
    fi
    
    if [[ ${#failed_builds[@]} -gt 0 ]] || [[ ${#failed_tests[@]} -gt 0 ]]; then
        return 1
    fi
}

# Function to cleanup test images
cleanup() {
    log_info "Cleaning up test images..."
    
    if [[ -f /tmp/built_images.txt ]]; then
        while IFS= read -r image; do
            if [[ -n "$image" ]]; then
                log_info "Removing image: $image"
                docker rmi "$image" &> /dev/null || true
            fi
        done < /tmp/built_images.txt
        rm -f /tmp/built_images.txt
    fi
    
    # Clean up test containers
    docker ps -a --filter "name=test-*" -q | xargs -r docker rm -f &> /dev/null || true
    
    log_info "Cleanup completed"
}

# Function to show usage
show_usage() {
    cat << 'EOL'
Usage: ./docker-build-test.sh [TAG] [PLATFORMS] [PUSH]

Arguments:
  TAG        Image tag (default: local-test)
  PLATFORMS  Target platforms (default: linux/amd64)
  PUSH       Push to registry (default: false)

Examples:
  # Build and test locally for AMD64
  ./docker-build-test.sh

  # Build for specific tag
  ./docker-build-test.sh v1.0.0

  # Build for multiple platforms (requires buildx)
  ./docker-build-test.sh latest "linux/amd64,linux/arm64"

  # Build and push to registry
  ./docker-build-test.sh latest "linux/amd64,linux/arm64" true

Environment Variables:
  DOCKER_REGISTRY  Override default registry
  SKIP_TESTS       Skip image testing (true/false)
EOL
}

# Main execution
main() {
    # Handle help flag
    if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    log_info "Docker Build and Test Script"
    log_info "Tag: $TAG"
    log_info "Platforms: $PLATFORMS" 
    log_info "Push: $PUSH"
    echo
    
    # Check prerequisites
    check_prerequisites
    
    # Ensure Dockerfiles exist
    create_dockerfiles
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Build and test all components
    if build_all; then
        log_info "🎉 All builds and tests completed successfully!"
        
        if [[ "$PUSH" == "true" ]]; then
            log_info "Images have been pushed to registry"
        else
            log_info "Images are available locally for testing"
        fi
    else
        log_error "❌ Some builds or tests failed"
        exit 1
    fi
}

# Run main function
main "$@"