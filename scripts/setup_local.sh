#!/bin/bash

# NVIDIA BioNeMo Blueprint: Protein Binder Design Setup Script
# This script helps prepare your local environment for running the protein binder design workflow

set -e

echo "======================================"
echo "🧬 NVIDIA BioNeMo Protein Binder Design Setup"
echo "======================================"
echo

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system architecture
check_architecture() {
    echo "=== System Architecture Check ==="
    echo
    
    ARCH=$(uname -m)
    print_info "Detected architecture: $ARCH"
    
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        print_warning "ARM64 architecture detected"
        echo
        echo "The NVIDIA NIM containers are built for x86_64/AMD64 architecture."
        echo "You have several options:"
        echo
        echo "1. Continue with Docker platform emulation (this script will set it up)"
        echo "   - Quick to set up"
        echo "   - May have performance impact"
        echo "   - Some compatibility issues possible"
        echo
        echo "2. Native installation (advanced users only)"
        echo "   - See ARM64_NATIVE_INSTALLATION.md for detailed instructions"
        echo "   - Requires building tools from source"
        echo "   - Takes several days to complete"
        echo "   - Requires advanced technical skills"
        echo
        echo "3. Use cloud instances with x86_64 architecture"
        echo "   - Best performance and compatibility"
        echo "   - AWS, GCP, Azure instances with NVIDIA GPUs"
        echo
        read -p "Continue with Docker platform emulation setup? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Setup cancelled. See ARM64_NATIVE_INSTALLATION.md for native installation guide."
            exit 0
        fi
        print_info "Continuing with Docker platform emulation setup..."
    else
        print_status "x86_64/AMD64 architecture detected (optimal)"
    fi
    
    echo
}

# Function to check system requirements
check_system_requirements() {
    echo "=== System Requirements Check ==="
    echo
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    REQUIRED_SPACE=1300
    
    if [ "$AVAILABLE_SPACE" -ge "$REQUIRED_SPACE" ]; then
        print_status "Disk space: ${AVAILABLE_SPACE}GB available (${REQUIRED_SPACE}GB required)"
    else
        print_error "Insufficient disk space: ${AVAILABLE_SPACE}GB available, ${REQUIRED_SPACE}GB required"
        echo "Please free up disk space before continuing."
        exit 1
    fi
    
    # Check RAM
    AVAILABLE_RAM=$(free -g | awk 'NR==2{print $2}')
    REQUIRED_RAM=64
    
    if [ "$AVAILABLE_RAM" -ge "$REQUIRED_RAM" ]; then
        print_status "RAM: ${AVAILABLE_RAM}GB available (${REQUIRED_RAM}GB required)"
    else
        print_warning "RAM: ${AVAILABLE_RAM}GB available, ${REQUIRED_RAM}GB recommended"
        print_info "You may experience performance issues with less than 64GB RAM"
    fi
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    REQUIRED_CORES=24
    
    if [ "$CPU_CORES" -ge "$REQUIRED_CORES" ]; then
        print_status "CPU cores: ${CPU_CORES} available (${REQUIRED_CORES} required)"
    else
        print_warning "CPU cores: ${CPU_CORES} available, ${REQUIRED_CORES} recommended"
        print_info "You may experience performance issues with fewer than 24 CPU cores"
    fi
    
    echo
}

# Function to check Docker installation
check_docker() {
    echo "=== Docker Setup Check ==="
    echo
    
    if command_exists docker; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        print_status "Docker installed: $DOCKER_VERSION"
    else
        print_error "Docker is not installed"
        echo "Please install Docker from https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_VERSION=$(docker compose version | cut -d' ' -f4)
        print_status "Docker Compose installed: $COMPOSE_VERSION"
    elif command_exists docker-compose; then
        COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        print_status "Docker Compose installed: $COMPOSE_VERSION (legacy)"
        print_warning "Consider upgrading to Docker Compose V2 (docker compose)"
    else
        print_error "Docker Compose is not installed"
        echo "Installing Docker Compose..."
        
        # Install Docker Compose plugin
        sudo apt-get update
        sudo apt-get install -y docker-compose-plugin
        
        if docker compose version >/dev/null 2>&1; then
            COMPOSE_VERSION=$(docker compose version | cut -d' ' -f4)
            print_status "Docker Compose installed: $COMPOSE_VERSION"
        else
            print_error "Failed to install Docker Compose"
            exit 1
        fi
    fi
    
    # Check if user can run docker without sudo
    if docker ps >/dev/null 2>&1; then
        print_status "Docker permissions configured correctly"
    else
        print_error "Cannot run Docker commands without sudo"
        echo "Please add your user to the docker group:"
        echo "  sudo usermod -aG docker \$USER"
        echo "  newgrp docker"
        exit 1
    fi
    
    echo
}

# Function to check NVIDIA setup
check_nvidia() {
    echo "=== NVIDIA GPU Setup Check ==="
    echo
    
    if command_exists nvidia-smi; then
        print_status "NVIDIA drivers installed"
        
        # Count GPUs
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ "$GPU_COUNT" -ge 2 ]; then
            print_status "GPUs detected: $GPU_COUNT (minimum 2 required)"
        else
            print_warning "GPUs detected: $GPU_COUNT (minimum 2 recommended)"
            print_info "Some workflows may not work optimally with fewer than 2 GPUs"
        fi
        
        # Show GPU info
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | nl -v0 -s": "
    else
        print_error "NVIDIA drivers not found"
        echo "Please install NVIDIA drivers and nvidia-smi"
        exit 1
    fi
    
    if command_exists nvidia-container-runtime; then
        RUNTIME_VERSION=$(nvidia-container-runtime --version | head -1 | cut -d' ' -f4)
        print_status "NVIDIA Container Runtime installed: $RUNTIME_VERSION"
    else
        print_error "NVIDIA Container Runtime not found"
        echo "Please install nvidia-container-toolkit"
        exit 1
    fi
    
    echo
}

# Function to setup environment variables
setup_environment() {
    echo "=== Environment Variables Setup ==="
    echo
    
    # Check for NGC API Key
    if [ -z "$NGC_CLI_API_KEY" ]; then
        print_warning "NGC_CLI_API_KEY not set"
        echo
        echo "You need an NGC Personal API Key to download the NIM containers."
        echo "1. Go to https://ngc.nvidia.com/"
        echo "2. Sign in or create an account"
        echo "3. Go to Setup > Generate API Key"
        echo "4. Generate a key with appropriate permissions"
        echo
        read -p "Enter your NGC API Key: " NGC_KEY
        if [ -n "$NGC_KEY" ]; then
            echo "export NGC_CLI_API_KEY=\"$NGC_KEY\"" >> ~/.bashrc
            export NGC_CLI_API_KEY="$NGC_KEY"
            print_status "NGC_CLI_API_KEY configured"
            print_info "Added to ~/.bashrc for persistence"
        else
            print_error "NGC API Key is required"
            exit 1
        fi
    else
        print_status "NGC_CLI_API_KEY already configured"
    fi
    
    # Setup NIM cache directory
    NIM_CACHE_DIR="$HOME/.cache/nim"
    if [ -z "$HOST_NIM_CACHE" ]; then
        echo "export HOST_NIM_CACHE=\"$NIM_CACHE_DIR\"" >> ~/.bashrc
        export HOST_NIM_CACHE="$NIM_CACHE_DIR"
        print_status "HOST_NIM_CACHE configured: $NIM_CACHE_DIR"
    else
        print_status "HOST_NIM_CACHE already configured: $HOST_NIM_CACHE"
        NIM_CACHE_DIR="$HOST_NIM_CACHE"
    fi
    
    # Create and configure NIM cache directory
    mkdir -p "$NIM_CACHE_DIR"
    chmod -R 777 "$NIM_CACHE_DIR"
    print_status "NIM cache directory created and configured: $NIM_CACHE_DIR"
    
    echo
}

# Function to login to NGC
setup_docker_login() {
    echo "=== Docker Registry Login ==="
    echo
    
    if [ -n "$NGC_CLI_API_KEY" ]; then
        print_info "Logging into nvcr.io Docker registry..."
        echo "$NGC_CLI_API_KEY" | docker login nvcr.io --username='$oauthtoken' --password-stdin
        if [ $? -eq 0 ]; then
            print_status "Successfully logged into nvcr.io"
        else
            print_error "Failed to login to nvcr.io"
            echo "Please check your NGC API Key"
            exit 1
        fi
    else
        print_error "NGC_CLI_API_KEY not available for Docker login"
        exit 1
    fi
    
    echo
}

# Function to setup Python environment
setup_python() {
    echo "=== Python Environment Setup ==="
    echo
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python installed: $PYTHON_VERSION"
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ -d ".venv" ]; then
        print_status "Virtual environment already exists"
    else
        print_info "Creating Python virtual environment..."
        python3 -m venv .venv
        print_status "Virtual environment created"
    fi
    
    # Activate virtual environment and install requirements
    source .venv/bin/activate
    print_info "Installing Python packages..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    print_status "Python packages installed"
    
    echo
}

# Function to show next steps
show_next_steps() {
    echo "======================================"
    echo "🎉 Setup Complete!"
    echo "======================================"
    echo
    echo "Next steps:"
    echo "1. Restart your shell or run: source ~/.bashrc"
    echo "2. Navigate to the deploy directory: cd deploy/"
    echo "3. Start the NIM services: docker compose up"
    echo "4. Wait for model downloads (this can take 3-7 hours)"
    echo "5. Once ready, navigate to src/ and start Jupyter: cd ../src && jupyter notebook"
    echo
    echo "Important notes:"
    echo "• Model downloads require ~1.3TB of disk space"
    echo "• First startup will take several hours due to model downloads"
    echo "• Check service readiness with: curl localhost:808[1-4]/v1/health/ready"
    echo
    print_info "For troubleshooting, see the README files in deploy/ and src/ directories"
}

# Main execution
main() {
    check_architecture
    check_system_requirements
    check_docker
    check_nvidia
    setup_environment
    setup_docker_login
    setup_python
    show_next_steps
}

# Run main function
main