# Zero-Touch Installer Improvements

## Date
January 12, 2026

## Overview
Enhanced the zero-touch installer to handle all edge cases automatically, ensuring true unattended installation across platforms.

## Problems Addressed

### 1. **Multiple Sudo Password Prompts**
**Problem**: Installation required sudo password multiple times:
- System dependencies installation
- AlphaFold external binaries (hmmer, hhsuite, kalign)
- Potentially other system operations

**Solution**: Sudo keepalive mechanism
- Requests sudo once at startup (after confirmation)
- Background process refreshes every 4 minutes
- Automatic cleanup on exit/interrupt
- Zero additional prompts during 45-60 minute installation

### 2. **Conda Detection Failures**
**Problem**: Script failed to detect existing Miniforge installations if not in PATH

**Solution**: Enhanced detection logic
- Checks for `$HOME/miniforge3` directory before attempting install
- Sources conda profile automatically
- Validates conda executable exists
- Only installs if truly missing

### 3. **Wrong AlphaFold Repository**
**Problem**: `tools/alphafold2` contained wrong repository (ProteinCAD fork instead of DeepMind official)
- Missing download scripts
- Database downloads failed
- Incomplete installation

**Solution**: Repository validation and auto-correction
- Validates git remote URL matches `github.com/deepmind/alphafold`
- Checks for required download scripts
- Automatically removes incorrect repo and clones correct one
- Updates existing repos when appropriate

### 4. **Silent Database Download Failures**
**Problem**: Database downloads failed with only warnings, unclear which databases succeeded

**Solution**: Enhanced download tracking
- Counts successes and failures
- Clear progress messages for each database
- Summary at end: "X succeeded, Y failed"
- Saves results to `.tier` file for troubleshooting
- Guidance on how to retry

## Implementation Details

### Sudo Keepalive (`scripts/install_all_native.sh`)

```bash
# Request sudo upfront
if ! sudo -v; then
    log_error "Sudo access required for installation"
    exit 1
fi

# Start background keepalive (refreshes every 4 minutes)
start_sudo_keepalive() {
    while true; do
        sleep 240
        sudo -v
    done &
    SUDO_KEEPALIVE_PID=$!
}

# Cleanup on exit
trap stop_sudo_keepalive EXIT INT TERM
```

### Conda Detection (`scripts/install_alphafold2_complete.sh`)

```bash
# Check if Miniforge is already installed
if [[ -d "$HOME/miniforge3" ]] && [[ -f "$HOME/miniforge3/bin/conda" ]]; then
    log_info "Found existing Miniforge installation"
    export PATH="$HOME/miniforge3/bin:$PATH"
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    log_success "Conda found"
elif ! command -v conda &>/dev/null; then
    # Install if not found
    ...
fi
```

### AlphaFold Repository Validation (`scripts/install_alphafold2_complete.sh`)

```bash
NEEDS_CLONE=false
if [ -d "$ALPHAFOLD_DIR" ]; then
    # Check if it's the correct repository
    if git -C "$ALPHAFOLD_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        REMOTE_URL=$(git -C "$ALPHAFOLD_DIR" remote get-url origin 2>/dev/null || echo "")
        
        # Validate it's deepmind/alphafold
        if [[ "$REMOTE_URL" =~ github.com[:/]deepmind/alphafold ]]; then
            # Verify download scripts exist
            if [ -f "$ALPHAFOLD_DIR/scripts/download_pdb70.sh" ]; then
                log_success "Valid AlphaFold repository"
            else
                # Missing scripts, re-clone
                rm -rf "$ALPHAFOLD_DIR"
                NEEDS_CLONE=true
            fi
        else
            # Wrong repo, remove and re-clone
            log_warning "Wrong repository, re-cloning..."
            rm -rf "$ALPHAFOLD_DIR"
            NEEDS_CLONE=true
        fi
    fi
fi
```

### Download Tracking (`scripts/install_alphafold2_complete.sh`)

```bash
DOWNLOAD_FAILURES=0
DOWNLOAD_SUCCESSES=0

log_info "Downloading Small BFD..."
if run_official_download_script download_small_bfd.sh; then
    DOWNLOAD_SUCCESSES=$((DOWNLOAD_SUCCESSES + 1))
else
    log_warning "Small BFD download failed"
    DOWNLOAD_FAILURES=$((DOWNLOAD_FAILURES + 1))
fi

# ... repeat for each database ...

log_info "Summary: $DOWNLOAD_SUCCESSES succeeded, $DOWNLOAD_FAILURES failed"
```

## Files Modified

1. **`scripts/install_all_native.sh`**
   - Added sudo keepalive mechanism (lines 225-256)
   - Enhanced user messaging

2. **`scripts/install_alphafold2_complete.sh`**
   - Fixed `$ROOT_DIR` → `$PROJECT_ROOT` references
   - Enhanced conda detection (lines 212-220)
   - Repository validation logic (lines 426-480)
   - Download tracking and reporting (lines 688-770)

3. **`run_zero_touch_install.sh`** (NEW)
   - Wrapper script with conda auto-sourcing
   - Clear user messaging about single-password requirement

4. **`SUDO_KEEPALIVE_IMPLEMENTATION.md`** (NEW)
   - Comprehensive documentation of sudo keepalive

5. **`ZERO_TOUCH_INSTALLER_IMPROVEMENTS.md`** (NEW - this file)
   - Complete improvement documentation

## Testing Results

### Tested On
- ✅ HP ZBook (Ubuntu 24.04 x86_64, NVIDIA RTX 5000 Ada 16GB)
- ✅ DGX Spark (ARM64) - original development platform

### Test Scenarios
1. ✅ Fresh installation (no prior conda/alphafold)
2. ✅ Existing conda, no alphafold
3. ✅ Wrong alphafold repository (auto-corrects)
4. ✅ Partial database downloads (tracks and reports)
5. ✅ Sudo timeout handling (keepalive works)
6. ✅ Installation interruption (cleanup works)

## User Experience

### Before
```
$ ./scripts/install_all_native.sh --recommended

[... installation starts ...]

[sudo] password for user: ****     # Prompt 1

[... 20 minutes later ...]

[sudo] password for user: ****     # Prompt 2 😞

[... installation continues with various errors ...]
```

### After
```
$ ./run_zero_touch_install.sh --recommended

========================================
  Zero-Touch Installation
========================================

This installer will:
  ✓ Request sudo once at start
  ✓ Keep sudo alive automatically
  ✓ Install all components uninterrupted

[... confirmation ...]

[sudo] password for user: ****     # ONLY prompt

[SUCCESS] Sudo keepalive started

[... 45-60 minutes of UNINTERRUPTED installation ...]

[INFO] Database download summary: 7 succeeded, 1 failed
[SUCCESS] Installation complete!
```

## Benefits

✅ **True Zero-Touch**: Single password prompt, walk away for 45-60 minutes  
✅ **Cross-Platform**: Works on ARM64 and x86_64  
✅ **Self-Healing**: Auto-corrects wrong repositories  
✅ **Informative**: Clear progress and error reporting  
✅ **Resumable**: Failed downloads can be retried  
✅ **Safe**: Automatic cleanup, no lingering processes  

## Usage

### Recommended Installation (50GB, ~1 hour)
```bash
./run_zero_touch_install.sh --recommended
```

### Minimal Installation (5GB, ~15 min)
```bash
./run_zero_touch_install.sh --minimal
```

### Full Installation (2.3TB, ~6 hours)
```bash
./run_zero_touch_install.sh --full
```

### Retry Failed Downloads
```bash
# Re-run installer - it will skip completed downloads
./run_zero_touch_install.sh --recommended
```

## Future Improvements

Potential enhancements for future versions:

1. **Parallel Downloads**: Download multiple databases simultaneously
2. **Resume Downloads**: aria2c already supports this, could expose more options
3. **Mirror Selection**: Auto-select fastest mirror for downloads
4. **Pre-flight Checks**: Verify disk space, network before starting
5. **Progress Bar**: Overall installation progress indicator
6. **Notification**: Desktop notification when complete
7. **Rollback**: Ability to rollback to previous state on failure

## Related Documentation

- [ZERO_TOUCH_GPU_COMPLETE.md](ZERO_TOUCH_GPU_COMPLETE.md) - GPU auto-configuration
- [SUDO_KEEPALIVE_IMPLEMENTATION.md](SUDO_KEEPALIVE_IMPLEMENTATION.md) - Sudo keepalive details
- [START_HERE.md](START_HERE.md) - Quick start guide
- [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) - Overall integration status

## Conclusion

The zero-touch installer now handles all common edge cases automatically:
- ✅ Single sudo password prompt
- ✅ Correct repository validation
- ✅ Comprehensive error reporting
- ✅ Cross-platform compatibility
- ✅ Truly unattended installation

Scripts developed on DGX Spark now run flawlessly on HP ZBook and vice versa! 🎉
