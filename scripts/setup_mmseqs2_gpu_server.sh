#!/bin/bash
# Setup MMseqs2 GPU Server Mode
# This enables GPU acceleration for MMseqs2 searches without rebuilding the database

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# Configuration
DB_PATH="${1:-$HOME/.cache/alphafold/mmseqs2/uniref90_db}"
GPU_SERVER_PORT="${2:-8080}"
LOG_DIR="/var/log/mmseqs2-gpu-server"
SYSTEMD_SERVICE="/etc/systemd/system/mmseqs2-gpu-server.service"

echo "================================================================"
echo "  MMseqs2 GPU Server Setup"
echo "================================================================"
echo ""

# Verify GPU
if ! nvidia-smi >/dev/null 2>&1; then
    log_error "No GPU detected"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
log_success "GPU detected: $GPU_NAME"

# Verify MMseqs2
if ! command -v mmseqs >/dev/null 2>&1; then
    log_error "MMseqs2 not found"
    exit 1
fi

MMSEQS_BIN=$(which mmseqs)
log_success "MMseqs2: $MMSEQS_BIN"

# Verify database
if [[ ! -f "${DB_PATH}.dbtype" ]]; then
    log_error "Database not found: $DB_PATH"
    exit 1
fi

DB_SIZE=$(du -sh "$(dirname "$DB_PATH")" | cut -f1)
log_success "Database: $DB_PATH ($DB_SIZE)"

# Check if running as root (for systemd service)
if [[ $EUID -eq 0 ]]; then
    INSTALL_SYSTEMD=true
    log_info "Running as root - will install systemd service"
else
    INSTALL_SYSTEMD=false
    log_warning "Not running as root - systemd service requires sudo"
fi

# Create log directory
if $INSTALL_SYSTEMD; then
    mkdir -p "$LOG_DIR"
    chown $SUDO_USER:$SUDO_USER "$LOG_DIR" 2>/dev/null || true
fi

# Create systemd service
if $INSTALL_SYSTEMD; then
    log_info "Creating systemd service..."
    
    cat > "$SYSTEMD_SERVICE" << EOF
[Unit]
Description=MMseqs2 GPU Server
Documentation=https://github.com/soedinglab/MMseqs2
After=network.target

[Service]
Type=simple
User=${SUDO_USER:-$(whoami)}
Environment="PATH=$PATH"
ExecStart=$MMSEQS_BIN gpuserver $DB_PATH --threads 4
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/mmseqs2-gpu-server.log
StandardError=append:$LOG_DIR/mmseqs2-gpu-server-error.log

# Resource limits
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

    log_success "Systemd service created: $SYSTEMD_SERVICE"
    
    # Reload systemd
    systemctl daemon-reload
    log_success "Systemd reloaded"
    
    # Enable and start service
    systemctl enable mmseqs2-gpu-server
    log_success "Service enabled"
    
    systemctl start mmseqs2-gpu-server
    log_success "Service started"
    
    sleep 2
    
    # Check status
    if systemctl is-active --quiet mmseqs2-gpu-server; then
        log_success "GPU server is running"
        systemctl status mmseqs2-gpu-server --no-pager | head -15
    else
        log_error "GPU server failed to start"
        journalctl -u mmseqs2-gpu-server -n 20 --no-pager
        exit 1
    fi
else
    # Non-systemd mode - create run script
    RUN_SCRIPT="$HOME/.local/bin/run-mmseqs2-gpu-server.sh"
    mkdir -p "$(dirname "$RUN_SCRIPT")"
    
    cat > "$RUN_SCRIPT" << EOF
#!/bin/bash
# Start MMseqs2 GPU Server
# Run in background: nohup $RUN_SCRIPT &

set -euo pipefail

LOG_FILE="\$HOME/.cache/mmseqs2-gpu-server.log"

echo "Starting MMseqs2 GPU Server..."
echo "Database: $DB_PATH"
echo "Log: \$LOG_FILE"

$MMSEQS_BIN gpuserver $DB_PATH --threads 4 >> "\$LOG_FILE" 2>&1
EOF
    
    chmod +x "$RUN_SCRIPT"
    log_success "Run script created: $RUN_SCRIPT"
    
    log_info "To start the server:"
    echo "  nohup $RUN_SCRIPT &"
fi

# Create client configuration
CLIENT_CONFIG="$HOME/.mmseqs2_gpu_client"
cat > "$CLIENT_CONFIG" << EOF
# MMseqs2 GPU Client Configuration
# Source this file before running MMseqs2 searches

export MMSEQS2_USE_GPU_SERVER=1
export MMSEQS2_GPU_SERVER_FLAGS="--gpu-server 1"

# Helper function to run searches with GPU server
mmseqs2_gpu_search() {
    mmseqs search "\$@" --gpu-server 1
}

echo "MMseqs2 GPU Server client configured"
echo "Use: mmseqs search <args> --gpu-server 1"
EOF

log_success "Client config: $CLIENT_CONFIG"

# Update environment file
if [[ -f "$HOME/.cache/alphafold/.env.mmseqs2" ]]; then
    if ! grep -q "GPU_SERVER" "$HOME/.cache/alphafold/.env.mmseqs2"; then
        echo "" >> "$HOME/.cache/alphafold/.env.mmseqs2"
        echo "# GPU Server Configuration (added $(date))" >> "$HOME/.cache/alphafold/.env.mmseqs2"
        echo "export MMSEQS2_USE_GPU_SERVER=1" >> "$HOME/.cache/alphafold/.env.mmseqs2"
        echo "export MMSEQS2_GPU_SERVER_FLAGS=\"--gpu-server 1\"" >> "$HOME/.cache/alphafold/.env.mmseqs2"
        log_success "AlphaFold env updated"
    fi
fi

echo ""
echo "================================================================"
echo "  MMseqs2 GPU Server Setup Complete"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  Database: $DB_PATH"
echo "  GPU: $GPU_NAME"
echo "  Client config: $CLIENT_CONFIG"
echo ""
if $INSTALL_SYSTEMD; then
    echo "Service Management:"
    echo "  Status:  sudo systemctl status mmseqs2-gpu-server"
    echo "  Stop:    sudo systemctl stop mmseqs2-gpu-server"
    echo "  Restart: sudo systemctl restart mmseqs2-gpu-server"
    echo "  Logs:    sudo journalctl -u mmseqs2-gpu-server -f"
    echo ""
else
    echo "To start server manually:"
    echo "  nohup $HOME/.local/bin/run-mmseqs2-gpu-server.sh &"
    echo ""
fi
echo "To use in searches:"
echo "  source $CLIENT_CONFIG"
echo "  mmseqs search query.db target.db result.db tmp/ --gpu-server 1"
echo ""
echo "Expected performance: 5-10x speedup (580s → 60-120s)"
echo ""
