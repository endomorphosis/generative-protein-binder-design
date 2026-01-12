#!/bin/bash
# Wrapper script for zero-touch installation that ensures proper conda setup
#
# This script will:
# - Request sudo password once at the beginning
# - Automatically refresh sudo in the background (no more password prompts!)
# - Set up conda environment properly
# - Run the complete installation

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Zero-Touch Installation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}This installer will:${NC}"
echo "  ✓ Request sudo once at start"
echo "  ✓ Keep sudo alive automatically"
echo "  ✓ Install all components uninterrupted"
echo ""

# Source conda if it exists
if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    export PATH="$HOME/miniforge3/bin:$PATH"
fi

# Run the installation
cd "$SCRIPT_DIR"
bash scripts/install_all_native.sh "$@"
