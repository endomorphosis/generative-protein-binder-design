#!/usr/bin/env bash
set -euo pipefail

# Provision host-native AlphaFold2 + RFDiffusion on ARM64 and emit environment
# suitable for scripts/run_arm64_native_model_services.sh.
#
# This script intentionally leans on the existing "complete" installers which:
# - download weights/DBs (tiered for AlphaFold2)
# - create wrapper scripts under tools/
# - write integration env files:
#     tools/generated/alphafold2/.env
#     tools/generated/rfdiffusion/.env
#
# Usage:
#   ./scripts/provision_arm64_host_native_models.sh
#   ./scripts/provision_arm64_host_native_models.sh --start-services
#   ./scripts/provision_arm64_host_native_models.sh --db-tier minimal
#

DB_TIER="minimal"
START_SERVICES=0
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/provision_arm64_host_native_models.sh [--db-tier minimal|reduced|full] [--start-services] [--force]

Notes:
- AlphaFold2 "full" DB tier is multi-terabyte and may take a long time.
- The installers may require sudo to install system packages.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --db-tier)
      DB_TIER="${2:-}"
      shift 2
      ;;
    --start-services)
      START_SERVICES=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
 done

case "$DB_TIER" in
  minimal|reduced|full) ;;
  *)
    echo "Invalid --db-tier: $DB_TIER" >&2
    usage
    exit 2
    ;;
 esac

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "ERR: This helper is intended for ARM64 hosts. Detected: $ARCH" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Provisioning host-native models on $ARCH"
echo "- AlphaFold2 DB tier: $DB_TIER"
echo

echo "Ensuring AlphaFold external binaries (hmmer/hhsuite/kalign) are installed..."
bash "$ROOT_DIR/scripts/ensure_alphafold_external_binaries.sh" || {
  echo "ERR: Failed to ensure AlphaFold external binaries." >&2
  exit 1
}

AF_FORCE_ARGS=()
RF_FORCE_ARGS=()
if [[ "$FORCE" == "1" ]]; then
  AF_FORCE_ARGS+=(--force)
  RF_FORCE_ARGS+=(--force)
fi

# AlphaFold2
AF_ENV_FILE="$ROOT_DIR/tools/generated/alphafold2/.env"
AF_ENV_FILE_LEGACY="$ROOT_DIR/tools/alphafold2/.env"
if [[ "$FORCE" != "1" ]] && [[ -f "$AF_ENV_FILE" || -f "$AF_ENV_FILE_LEGACY" ]]; then
  echo "Found AlphaFold2 env file; skipping AlphaFold2 install."
else
  echo "Installing AlphaFold2 (this downloads weights/DBs)..."
  bash "$ROOT_DIR/scripts/install_alphafold2_complete.sh" --db-tier "$DB_TIER" "${AF_FORCE_ARGS[@]}" || {
    echo "ERR: AlphaFold2 installer failed." >&2
    exit 1
  }
fi

echo

# RFDiffusion
RF_ENV_FILE="$ROOT_DIR/tools/generated/rfdiffusion/.env"
RF_ENV_FILE_LEGACY="$ROOT_DIR/tools/rfdiffusion/.env"
RF_ENV_FILE_LEGACY2="$ROOT_DIR/tools/rfdiffusion/RFdiffusion/.env"
if [[ "$FORCE" != "1" ]] && [[ -f "$RF_ENV_FILE" || -f "$RF_ENV_FILE_LEGACY" || -f "$RF_ENV_FILE_LEGACY2" ]]; then
  echo "Found RFDiffusion env file; skipping RFDiffusion install."
else
  echo "Installing RFDiffusion (this downloads weights)..."
  bash "$ROOT_DIR/scripts/install_rfdiffusion_complete.sh" "${RF_FORCE_ARGS[@]}" || {
    echo "ERR: RFDiffusion installer failed." >&2
    exit 1
  }
fi

echo

cat <<EOF
Done.

Next steps:
- Start the host-native services (these will auto-load tools/*/.env):
  bash ./scripts/run_arm64_native_model_services.sh

- Start the dashboard stack (routes to host ports 18081/18082):
    ./scripts/run_dashboard_stack.sh --arm64-host-native up -d --build

If you need AlphaFold-Multimer on ARM64:
- Use the AMD64 NIM stack under emulation:
    ./scripts/run_dashboard_stack.sh --emulated up -d
  (or point the dashboard at a remote NIM deployment)
EOF

if [[ "$START_SERVICES" == "1" ]]; then
  echo
  echo "Starting host-native services..."
  exec bash "$ROOT_DIR/scripts/run_arm64_native_model_services.sh"
fi
