#!/bin/bash

# GitHub Actions Self-Hosted Runner Setup Script
# For ARM64 systems with NVIDIA GPU support
# This script sets up a GitHub Actions runner for the protein binder design repository

set -e

echo "=============================================="
echo "🚀 GitHub Actions Self-Hosted Runner Setup"
echo "=============================================="
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

# Configuration
RUNNER_VERSION="2.321.0"  # Latest version as of Oct 2025
RUNNER_DIR="$HOME/actions-runner"
REPO_URL=""
RUNNER_TOKEN=""
RUNNER_NAME="arm64-gpu-runner-$(hostname)"
RUNNER_LABELS="self-hosted,ARM64,Linux,gpu,nvidia,protein-design"

# Function to check system requirements
check_system_requirements() {
    echo "=== System Requirements Check ==="
    echo
    
    # Check architecture
    ARCH=$(uname -m)
    if [ "$ARCH" != "aarch64" ]; then
        print_error "This script is designed for ARM64 (aarch64) systems. Detected: $ARCH"
        exit 1
    fi
    print_status "ARM64 architecture confirmed: $ARCH"
    
    # Check available disk space (runners need ~1GB + workspace)
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 5 ]; then
        print_error "Insufficient disk space: ${AVAILABLE_SPACE}GB available, at least 5GB required"
        exit 1
    fi
    print_status "Disk space: ${AVAILABLE_SPACE}GB available"
    
    # Check if Docker is available (for containerized workflows)
    if command -v docker >/dev/null 2>&1; then
        print_status "Docker available for containerized workflows"
    else
        print_warning "Docker not found - some workflows may not work"
    fi
    
    # Check if NVIDIA GPU is available
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_status "NVIDIA GPU available: $GPU_COUNT GPU(s) detected"
    else
        print_warning "NVIDIA GPU not detected - GPU workflows will not work"
    fi
    
    echo
}

# Function to install dependencies
install_dependencies() {
    echo "=== Installing Dependencies ==="
    echo
    
    # Update package list
    print_info "Updating package list..."
    sudo apt-get update -q
    
    # Install required packages
    print_info "Installing required packages..."
    sudo apt-get install -y \
        curl \
        wget \
        tar \
        jq \
        git \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3 \
        python3-pip \
        python3-venv \
        nodejs \
        npm
    
    print_status "Dependencies installed"
    echo
}

# Function to download and setup GitHub Actions runner
setup_github_runner() {
    echo "=== Setting Up GitHub Actions Runner ==="
    echo
    
    # Create runner directory
    if [ -d "$RUNNER_DIR" ]; then
        print_warning "Runner directory already exists: $RUNNER_DIR"
        read -p "Remove existing runner directory? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$RUNNER_DIR"
        else
            print_info "Using existing directory"
        fi
    fi
    
    if [ ! -d "$RUNNER_DIR" ]; then
        mkdir -p "$RUNNER_DIR"
        print_status "Created runner directory: $RUNNER_DIR"
    fi
    
    cd "$RUNNER_DIR"
    
    # Download the runner package for ARM64
    print_info "Downloading GitHub Actions runner v${RUNNER_VERSION} for ARM64..."
    RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"
    
    if [ ! -f "actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz" ]; then
        curl -o "actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz" -L "$RUNNER_URL"
        print_status "Runner package downloaded"
    else
        print_info "Runner package already downloaded"
    fi
    
    # Extract the runner package
    if [ ! -f "config.sh" ]; then
        print_info "Extracting runner package..."
        tar xzf "actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"
        print_status "Runner package extracted"
    else
        print_info "Runner package already extracted"
    fi
    
    # Install additional dependencies
    print_info "Installing runner dependencies..."
    sudo ./bin/installdependencies.sh
    print_status "Runner dependencies installed"
    
    echo
}

# Function to get repository information
get_repository_info() {
    echo "=== Repository Configuration ==="
    echo
    
    # Try to detect repository URL from git remote
    if git remote -v >/dev/null 2>&1; then
        DETECTED_URL=$(git remote get-url origin 2>/dev/null || echo "")
        if [ -n "$DETECTED_URL" ]; then
            # Convert SSH URL to HTTPS if needed
            if [[ $DETECTED_URL == git@github.com:* ]]; then
                DETECTED_URL=$(echo "$DETECTED_URL" | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/\.git$//')
            fi
            print_info "Detected repository URL: $DETECTED_URL"
            
            read -p "Use detected repository URL? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                REPO_URL="$DETECTED_URL"
            fi
        fi
    fi
    
    # If no URL detected or user declined, ask for it
    if [ -z "$REPO_URL" ]; then
        echo "Please enter your GitHub repository URL:"
        echo "Example: https://github.com/username/repository-name"
        read -p "Repository URL: " REPO_URL
    fi
    
    if [ -z "$REPO_URL" ]; then
        print_error "Repository URL is required"
        exit 1
    fi
    
    print_status "Repository URL: $REPO_URL"
    echo
}

# Function to get runner token
get_runner_token() {
    echo "=== Runner Token Configuration ==="
    echo
    
    print_info "To get a runner token:"
    echo "1. Go to: $REPO_URL/settings/actions/runners"
    echo "2. Click 'New self-hosted runner'"
    echo "3. Select 'Linux' and 'ARM64'"
    echo "4. Copy the token from the configuration command"
    echo
    echo "The token will look like: AABCD1234567890ABCDEF..."
    echo
    
    read -p "Enter the runner token: " RUNNER_TOKEN
    
    if [ -z "$RUNNER_TOKEN" ]; then
        print_error "Runner token is required"
        exit 1
    fi
    
    print_status "Runner token configured"
    echo
}

# Function to configure the runner
configure_runner() {
    echo "=== Configuring GitHub Actions Runner ==="
    echo
    
    cd "$RUNNER_DIR"
    
    # Check if runner is already configured
    if [ -f ".runner" ]; then
        print_warning "Runner is already configured"
        read -p "Reconfigure runner? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ./config.sh remove --token "$RUNNER_TOKEN" || true
        else
            print_info "Skipping configuration"
            return
        fi
    fi
    
    # Configure the runner
    print_info "Configuring runner with the following settings:"
    echo "  - Name: $RUNNER_NAME"
    echo "  - Labels: $RUNNER_LABELS"
    echo "  - Work folder: _work"
    echo
    
    ./config.sh \
        --url "$REPO_URL" \
        --token "$RUNNER_TOKEN" \
        --name "$RUNNER_NAME" \
        --labels "$RUNNER_LABELS" \
        --work "_work" \
        --replace
    
    print_status "Runner configured successfully"
    echo
}

# Function to create systemd service
create_systemd_service() {
    echo "=== Creating Systemd Service ==="
    echo
    
    SERVICE_NAME="github-actions-runner"
    SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
    
    print_info "Creating systemd service: $SERVICE_NAME"
    
    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=GitHub Actions Runner
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$RUNNER_DIR
ExecStart=$RUNNER_DIR/run.sh
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Environment variables for GPU access
Environment=NVIDIA_VISIBLE_DEVICES=all
Environment=NVIDIA_DRIVER_CAPABILITIES=all

# Increase limits for protein design workflows
LimitNOFILE=65536
LimitNPROC=32768

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable the service
    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME"
    
    print_status "Systemd service created and enabled"
    echo
}

# Function to create runner management scripts
create_management_scripts() {
    echo "=== Creating Runner Management Scripts ==="
    echo
    
    # Create start script
    cat > "$HOME/start-github-runner.sh" <<'EOF'
#!/bin/bash
SERVICE_NAME="github-actions-runner"

echo "Starting GitHub Actions Runner..."
sudo systemctl start "$SERVICE_NAME"
sudo systemctl status "$SERVICE_NAME" --no-pager -l
EOF
    
    # Create stop script
    cat > "$HOME/stop-github-runner.sh" <<'EOF'
#!/bin/bash
SERVICE_NAME="github-actions-runner"

echo "Stopping GitHub Actions Runner..."
sudo systemctl stop "$SERVICE_NAME"
sudo systemctl status "$SERVICE_NAME" --no-pager -l
EOF
    
    # Create status script
    cat > "$HOME/status-github-runner.sh" <<'EOF'
#!/bin/bash
SERVICE_NAME="github-actions-runner"

echo "GitHub Actions Runner Status:"
sudo systemctl status "$SERVICE_NAME" --no-pager -l
echo
echo "Recent logs:"
sudo journalctl -u "$SERVICE_NAME" --no-pager -l -n 20
EOF
    
    # Create logs script
    cat > "$HOME/logs-github-runner.sh" <<'EOF'
#!/bin/bash
SERVICE_NAME="github-actions-runner"

echo "GitHub Actions Runner Logs (use Ctrl+C to exit):"
sudo journalctl -u "$SERVICE_NAME" -f
EOF
    
    # Make scripts executable
    chmod +x "$HOME/start-github-runner.sh"
    chmod +x "$HOME/stop-github-runner.sh"
    chmod +x "$HOME/status-github-runner.sh"
    chmod +x "$HOME/logs-github-runner.sh"
    
    print_status "Management scripts created in $HOME"
    echo
}

# Function to test runner
test_runner() {
    echo "=== Testing Runner Setup ==="
    echo
    
    cd "$RUNNER_DIR"
    
    print_info "Testing runner configuration..."
    
    # Start the runner service
    sudo systemctl start github-actions-runner
    sleep 5
    
    # Check status
    if sudo systemctl is-active --quiet github-actions-runner; then
        print_status "Runner service is running"
    else
        print_error "Runner service failed to start"
        sudo systemctl status github-actions-runner --no-pager -l
        return 1
    fi
    
    print_info "Check the GitHub repository settings to confirm the runner is connected:"
    echo "  URL: $REPO_URL/settings/actions/runners"
    echo
}

# Function to show completion message
show_completion_message() {
    echo "================================================"
    echo "🎉 GitHub Actions Runner Setup Complete!"
    echo "================================================"
    echo
    echo "Runner Details:"
    echo "  - Name: $RUNNER_NAME"
    echo "  - Labels: $RUNNER_LABELS"
    echo "  - Directory: $RUNNER_DIR"
    echo "  - Service: github-actions-runner"
    echo
    echo "Management Commands:"
    echo "  - Start runner:   ~/start-github-runner.sh"
    echo "  - Stop runner:    ~/stop-github-runner.sh"
    echo "  - Check status:   ~/status-github-runner.sh"
    echo "  - View logs:      ~/logs-github-runner.sh"
    echo
    echo "Service Commands:"
    echo "  - sudo systemctl start github-actions-runner"
    echo "  - sudo systemctl stop github-actions-runner"
    echo "  - sudo systemctl restart github-actions-runner"
    echo "  - sudo systemctl status github-actions-runner"
    echo
    echo "Next Steps:"
    echo "1. Verify the runner appears in: $REPO_URL/settings/actions/runners"
    echo "2. Create workflows in .github/workflows/ directory"
    echo "3. Use labels 'self-hosted', 'ARM64', 'gpu' to target this runner"
    echo
    print_info "Runner is now ready to process GitHub Actions workflows!"
}

# Main execution function
main() {
    print_info "Starting GitHub Actions self-hosted runner setup..."
    echo
    
    check_system_requirements
    install_dependencies
    setup_github_runner
    
    # Get repository and token information
    get_repository_info
    get_runner_token
    
    # Configure and start runner
    configure_runner
    create_systemd_service
    create_management_scripts
    test_runner
    
    show_completion_message
}

# Handle script interruption
trap 'echo -e "\n${RED}Setup interrupted${NC}"; exit 1' INT

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

# Run main function
main "$@"