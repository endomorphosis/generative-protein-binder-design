# Sudo Keepalive Implementation

## Problem
The zero-touch installer required sudo password multiple times during installation:
1. At the beginning for system dependencies (apt packages)
2. During AlphaFold external binaries installation (hmmer, hhsuite, kalign)
3. Potentially for other system-level operations

This interrupted the "zero-touch" experience and required user monitoring.

## Solution
Implemented a sudo keepalive mechanism that:

1. **Requests sudo once at startup** - User enters password only once after confirming installation
2. **Maintains sudo credentials** - Background process refreshes sudo every 4 minutes
3. **Automatic cleanup** - Stops keepalive on exit, interrupt, or termination
4. **Cross-platform compatible** - Works on both DGX Spark (ARM64) and HP ZBook (x86_64)

## Implementation Details

### Changes to `scripts/install_all_native.sh`

Added after confirmation prompt (line ~215):

```bash
# Request sudo access upfront and start keepalive
log_info "Requesting sudo access for system dependencies..."
if ! sudo -v; then
    log_error "Sudo access required for installation"
    exit 1
fi

# Start sudo keepalive in background
# Refreshes sudo timestamp every 4 minutes (default timeout is 5 minutes)
SUDO_KEEPALIVE_PID=""
start_sudo_keepalive() {
    while true; do
        sleep 240  # 4 minutes
        sudo -v
    done &
    SUDO_KEEPALIVE_PID=$!
}

# Stop sudo keepalive
stop_sudo_keepalive() {
    if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
        kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
        wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    fi
}

# Trap to cleanup keepalive on exit
trap stop_sudo_keepalive EXIT INT TERM

# Start the keepalive
start_sudo_keepalive
log_success "Sudo keepalive started (will refresh automatically)"
```

### Changes to `run_zero_touch_install.sh`

Enhanced wrapper script with clear user messaging:

```bash
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Zero-Touch Installation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}This installer will:${NC}"
echo "  ✓ Request sudo once at start"
echo "  ✓ Keep sudo alive automatically"
echo "  ✓ Install all components uninterrupted"
echo ""
```

## How It Works

1. **Initial Request**: After user confirms installation, script requests sudo with `sudo -v`
2. **Background Process**: Spawns a background process that runs `sudo -v` every 4 minutes
3. **PID Tracking**: Stores the keepalive process PID for cleanup
4. **Trap Handler**: Registers cleanup function to kill keepalive on script exit
5. **Automatic Refresh**: Linux sudo timeout is typically 5 minutes; refreshing every 4 minutes ensures no expiration

## Benefits

✅ **True Zero-Touch**: User enters password once, walks away  
✅ **Uninterrupted Installation**: No mid-installation password prompts  
✅ **Automatic Cleanup**: No lingering background processes  
✅ **Error Handling**: Proper cleanup on errors or interrupts  
✅ **Cross-Platform**: Works on all tested systems

## Installation Flow

```
User runs: ./run_zero_touch_install.sh --recommended
         ↓
1. Display welcome message
         ↓
2. Show installation plan
         ↓
3. Request confirmation (y/N)
         ↓
4. Request sudo password (ONCE)
         ↓
5. Start sudo keepalive in background
         ↓
6. Install all components (45-60 minutes)
         |
         |-- System dependencies (no password needed)
         |-- AlphaFold external binaries (no password needed)
         |-- MMseqs2 databases (no password needed)
         |-- RFDiffusion (no password needed)
         |-- ProteinMPNN (no password needed)
         ↓
7. Stop sudo keepalive
         ↓
8. Installation complete!
```

## Testing

Tested on:
- ✅ HP ZBook (Ubuntu 24.04 x86_64, NVIDIA RTX 5000 Ada)
- ✅ DGX Spark (ARM64) - original development platform

## Usage Examples

### Recommended Installation
```bash
./run_zero_touch_install.sh --recommended
```

### Minimal Installation  
```bash
./run_zero_touch_install.sh --minimal
```

### Full Installation
```bash
./run_zero_touch_install.sh --full
```

## Security Notes

- Sudo credentials are only maintained for the duration of the installation
- Background keepalive process is properly cleaned up on exit
- No credentials are stored or logged
- Standard sudo security policies apply
- User can monitor `ps aux | grep "sudo -v"` to verify keepalive is running

## Implementation Date

January 11, 2026

## Files Modified

1. `scripts/install_all_native.sh` - Added sudo keepalive mechanism
2. `run_zero_touch_install.sh` - Enhanced user messaging

## Related Documentation

- [ZERO_TOUCH_GPU_COMPLETE.md](ZERO_TOUCH_GPU_COMPLETE.md) - GPU auto-configuration
- [START_HERE.md](START_HERE.md) - Quick start guide
- [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) - Overall integration status
