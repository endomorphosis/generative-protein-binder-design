#!/bin/bash
set -euo pipefail

# GitHub Actions Self-Hosted Runner Setup Script
# This script sets up a self-hosted GitHub Actions runner for ARM64 systems

echo "=== GitHub Actions Self-Hosted Runner Setup ==="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
REPO_OWNER="${GITHUB_REPOSITORY_OWNER:-hallucinate-llc}"
REPO_NAME="${GITHUB_REPOSITORY_NAME:-generative-protein-binder-design}"
RUNNER_NAME="${RUNNER_NAME:-spark-b271-arm64}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,ARM64,docker,protein-design}"
RUNNER_GROUP="${RUNNER_GROUP:-default}"
RUNNER_VERSION="2.320.0"  # Latest version as of 2024

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

log_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        log_info "GitHub Actions runners should run as a regular user"
        exit 1
    fi
    
    # Check required commands
    local required_commands=("curl" "tar" "git" "docker" "gh")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        else
            log_info "✓ $cmd is available"
        fi
    done
    
    # Check Docker permissions
    if ! docker ps &> /dev/null; then
        log_warn "Docker is not accessible without sudo"
        log_info "Adding user to docker group..."
        sudo usermod -aG docker "$USER"
        log_warn "Please log out and log back in for docker group changes to take effect"
        log_warn "Then run this script again"
        exit 1
    else
        log_info "✓ Docker is accessible"
    fi
    
    # Check GitHub CLI authentication
    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI is not authenticated"
        log_info "Please run: gh auth login"
        exit 1
    else
        log_info "✓ GitHub CLI is authenticated"
    fi
    
    # Check repository access
    if ! gh repo view "$REPO_OWNER/$REPO_NAME" &> /dev/null; then
        log_error "Cannot access repository: $REPO_OWNER/$REPO_NAME"
        exit 1
    else
        log_info "✓ Repository access confirmed"
    fi
}

# Function to create runner directory
setup_runner_directory() {
    log_header "Setting up Runner Directory"
    
    local runner_dir="$HOME/actions-runner"
    
    if [[ -d "$runner_dir" ]]; then
        log_warn "Runner directory already exists: $runner_dir"
        read -p "Remove existing runner? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing runner..."
            cd "$runner_dir"
            if [[ -f "./svc.sh" ]]; then
                sudo ./svc.sh stop || true
                sudo ./svc.sh uninstall || true
            fi
            ./config.sh remove --token "$(get_removal_token)" || true
            cd "$HOME"
            rm -rf "$runner_dir"
        else
            log_info "Keeping existing runner directory"
            return 0
        fi
    fi
    
    log_info "Creating runner directory: $runner_dir"
    mkdir -p "$runner_dir"
    cd "$runner_dir"
}

# Function to get registration token
get_registration_token() {
    log_info "Getting registration token from GitHub..."
    gh api \
        --method POST \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "/repos/$REPO_OWNER/$REPO_NAME/actions/runners/registration-token" \
        --jq '.token'
}

# Function to get removal token
get_removal_token() {
    log_info "Getting removal token from GitHub..."
    gh api \
        --method POST \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "/repos/$REPO_OWNER/$REPO_NAME/actions/runners/remove-token" \
        --jq '.token' 2>/dev/null || echo ""
}

# Function to download and extract runner
download_runner() {
    log_header "Downloading GitHub Actions Runner"
    
    local runner_dir="$HOME/actions-runner"
    cd "$runner_dir"
    
    # Detect architecture
    local arch
    case "$(uname -m)" in
        x86_64) arch="x64" ;;
        aarch64) arch="arm64" ;;
        *) 
            log_error "Unsupported architecture: $(uname -m)"
            exit 1
            ;;
    esac
    
    local download_url="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${arch}-${RUNNER_VERSION}.tar.gz"
    
    log_info "Downloading runner for architecture: $arch"
    log_info "Download URL: $download_url"
    
    if [[ -f "actions-runner-linux-${arch}-${RUNNER_VERSION}.tar.gz" ]]; then
        log_info "Runner package already downloaded"
    else
        curl -o "actions-runner-linux-${arch}-${RUNNER_VERSION}.tar.gz" -L "$download_url"
    fi
    
    log_info "Extracting runner package..."
    tar xzf "actions-runner-linux-${arch}-${RUNNER_VERSION}.tar.gz"
    
    log_info "Setting up dependencies..."
    sudo ./bin/installdependencies.sh
}

# Function to configure runner
configure_runner() {
    log_header "Configuring GitHub Actions Runner"
    
    local runner_dir="$HOME/actions-runner"
    cd "$runner_dir"
    
    # Get registration token
    local token
    token=$(get_registration_token)
    
    if [[ -z "$token" ]]; then
        log_error "Failed to get registration token"
        exit 1
    fi
    
    log_info "Configuring runner with:"
    log_info "  Repository: $REPO_OWNER/$REPO_NAME"
    log_info "  Runner Name: $RUNNER_NAME"
    log_info "  Labels: $RUNNER_LABELS"
    log_info "  Group: $RUNNER_GROUP"
    
    # Configure the runner
    ./config.sh \
        --url "https://github.com/$REPO_OWNER/$REPO_NAME" \
        --token "$token" \
        --name "$RUNNER_NAME" \
        --labels "$RUNNER_LABELS" \
        --runnergroup "$RUNNER_GROUP" \
        --work "_work" \
        --replace
}

# Function to install runner as service
install_service() {
    log_header "Installing Runner as System Service"
    
    local runner_dir="$HOME/actions-runner"
    cd "$runner_dir"
    
    log_info "Installing runner service..."
    sudo ./svc.sh install
    
    log_info "Starting runner service..."
    sudo ./svc.sh start
    
    log_info "Checking service status..."
    sudo ./svc.sh status
}

# Function to create runner test workflow
create_test_workflow() {
    log_header "Creating Test Workflow"
    
    local workflow_dir="$PROJECT_ROOT/.github/workflows"
    mkdir -p "$workflow_dir"
    
    cat > "$workflow_dir/test-self-hosted-runner.yml" << 'EOF'
name: Test Self-Hosted Runner

on:
  workflow_dispatch:
  push:
    branches: [ dgx-spark ]
    paths:
      - '.github/workflows/test-self-hosted-runner.yml'

jobs:
  test-runner:
    runs-on: [self-hosted, ARM64]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: System Information
      run: |
        echo "=== System Information ==="
        uname -a
        echo "Architecture: $(uname -m)"
        echo "CPU Info:"
        lscpu | head -20
        echo "Memory Info:"
        free -h
        echo "Disk Usage:"
        df -h
        
    - name: Docker Information
      run: |
        echo "=== Docker Information ==="
        docker --version
        docker info
        echo "Available Docker Images:"
        docker images
        
    - name: Test Docker Container
      run: |
        echo "=== Testing Docker Container ==="
        docker run --rm hello-world
        
    - name: Python Environment Test
      run: |
        echo "=== Python Environment Test ==="
        python3 --version
        pip3 --version
        which python3
        
    - name: Project Structure
      run: |
        echo "=== Project Structure ==="
        ls -la
        echo "Python files:"
        find . -name "*.py" | head -10
        
    - name: Test MCP Server Dependencies
      run: |
        echo "=== Testing MCP Server Dependencies ==="
        cd mcp-server
        python3 -c "import sys; print(f'Python: {sys.version}')"
        if [ -f requirements.txt ]; then
          echo "Requirements found:"
          cat requirements.txt
        fi
        
    - name: Conda Environment Check
      run: |
        echo "=== Conda Environment Check ==="
        if command -v conda &> /dev/null; then
          conda --version
          conda env list
        else
          echo "Conda not found"
        fi
        
    - name: Success Message
      run: |
        echo "🎉 Self-hosted runner test completed successfully!"
        echo "Runner: ${{ runner.name }}"
        echo "OS: ${{ runner.os }}"
        echo "Architecture: ${{ runner.arch }}"
EOF

    log_info "Test workflow created: .github/workflows/test-self-hosted-runner.yml"
}

# Function to show runner status
show_runner_status() {
    log_header "Runner Status"
    
    local runner_dir="$HOME/actions-runner"
    
    if [[ -d "$runner_dir" ]]; then
        cd "$runner_dir"
        
        log_info "Service Status:"
        sudo ./svc.sh status || true
        
        echo
        log_info "Runner Configuration:"
        if [[ -f ".runner" ]]; then
            cat .runner | jq '.' 2>/dev/null || cat .runner
        fi
        
        echo
        log_info "GitHub Runners List:"
        gh api "/repos/$REPO_OWNER/$REPO_NAME/actions/runners" \
            --jq '.runners[] | "\(.name): \(.status) (\(.os)/\(.architecture))"' \
            2>/dev/null || log_warn "Failed to get runners list"
    else
        log_warn "Runner directory not found: $runner_dir"
    fi
}

# Function to remove runner
remove_runner() {
    log_header "Removing GitHub Actions Runner"
    
    local runner_dir="$HOME/actions-runner"
    
    if [[ ! -d "$runner_dir" ]]; then
        log_warn "Runner directory not found: $runner_dir"
        return 0
    fi
    
    cd "$runner_dir"
    
    # Stop and uninstall service
    if [[ -f "./svc.sh" ]]; then
        log_info "Stopping runner service..."
        sudo ./svc.sh stop || true
        
        log_info "Uninstalling runner service..."
        sudo ./svc.sh uninstall || true
    fi
    
    # Remove runner from GitHub
    local token
    token=$(get_removal_token)
    
    if [[ -n "$token" ]]; then
        log_info "Removing runner from GitHub..."
        ./config.sh remove --token "$token" || true
    fi
    
    # Remove directory
    log_info "Removing runner directory..."
    cd "$HOME"
    rm -rf "$runner_dir"
    
    log_info "Runner removed successfully"
}

# Function to show usage
show_usage() {
    cat << 'EOL'
Usage: ./setup-github-runner.sh [COMMAND]

Commands:
  install     Install and configure the GitHub Actions runner (default)
  status      Show runner status
  remove      Remove the runner
  test        Create test workflow
  help        Show this help message

Environment Variables:
  RUNNER_NAME     Runner name (default: spark-b271-arm64)
  RUNNER_LABELS   Runner labels (default: self-hosted,ARM64,docker,protein-design)
  RUNNER_GROUP    Runner group (default: default)
  GITHUB_REPOSITORY_OWNER   Repository owner (default: hallucinate-llc)
  GITHUB_REPOSITORY_NAME    Repository name (default: generative-protein-binder-design)

Examples:
  ./setup-github-runner.sh install    # Install runner
  ./setup-github-runner.sh status     # Check status
  ./setup-github-runner.sh remove     # Remove runner
  ./setup-github-runner.sh test       # Create test workflow

Notes:
  - This script must be run as a regular user (not root)
  - Docker must be installed and accessible without sudo
  - GitHub CLI must be authenticated
  - Repository access is required
EOL
}

# Main execution
main() {
    local command="${1:-install}"
    
    case $command in
        "install")
            check_prerequisites
            setup_runner_directory
            download_runner
            configure_runner
            install_service
            create_test_workflow
            show_runner_status
            log_info "✅ GitHub Actions runner installed successfully!"
            log_info "You can now trigger workflows that use 'runs-on: [self-hosted, ARM64]'"
            ;;
        "status")
            show_runner_status
            ;;
        "remove")
            remove_runner
            ;;
        "test")
            create_test_workflow
            log_info "Test workflow created. Push to trigger or run manually."
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n\nSetup interrupted."; exit 1' INT

# Run main function
main "$@"