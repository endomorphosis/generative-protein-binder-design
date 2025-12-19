#!/usr/bin/env bash
set -euo pipefail

# Run the MCP Dashboard + MCP Server stack on either AMD64 or ARM64.
#
# Auto mode:
# - AMD64/x86_64  -> deploy/docker-compose-dashboard.yaml (local NIM services; linux/amd64)
# - ARM64/aarch64 -> deploy/docker-compose-dashboard-arm64-native.yaml (local ARM64-native services)
#
# Explicit mode:
# - --control-plane -> deploy/docker-compose-dashboard-default.yaml (dashboard+server only; configure cloud via UI)
#
# Examples:
#   ./scripts/run_dashboard_stack.sh up -d
#   ./scripts/run_dashboard_stack.sh up -d --build
#   ./scripts/run_dashboard_stack.sh --amd64 up -d
#   ./scripts/run_dashboard_stack.sh --arm64 up -d --build
#   ./scripts/run_dashboard_stack.sh --emulated up -d   # run AMD64 stack on ARM64 (requires qemu/binfmt)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_dashboard_stack.sh [--amd64|--arm64|--arm64-host-native|--emulated|--control-plane] <docker compose args...>

Modes:
  --amd64     Force the AMD64 NIM stack (deploy/docker-compose-dashboard.yaml)
  --arm64     Force the ARM64-native stack (deploy/docker-compose-dashboard-arm64-native.yaml)
  --arm64-host-native  ARM64 host-native AlphaFold2/RFdiffusion (deploy/docker-compose-dashboard-arm64-host-native.yaml)
  --emulated  Same as --amd64, but prints extra guidance for ARM64 hosts
  --control-plane  MCP server + dashboard only (deploy/docker-compose-dashboard-default.yaml)

Examples:
  ./scripts/run_dashboard_stack.sh up -d
  ./scripts/run_dashboard_stack.sh up -d --build
  ./scripts/run_dashboard_stack.sh --arm64 up -d --build
EOF
}

MODE="auto"
if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --amd64)
      MODE="amd64"
      shift
      ;;
    --arm64)
      MODE="arm64"
      shift
      ;;
    --arm64-host-native)
      MODE="arm64-host-native"
      shift
      ;;
    --emulated)
      MODE="emulated"
      shift
      ;;
    --control-plane)
      MODE="control-plane"
      shift
      ;;
    *)
      break
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found; install Docker first." >&2
  exit 1
fi

ARCH="$(uname -m)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COMPOSE_FILE=""
case "$MODE" in
  auto)
    case "$ARCH" in
      x86_64|amd64)
        COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard.yaml"
        ;;
      aarch64|arm64)
        COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard-arm64-native.yaml"
        ;;
      *)
        echo "Unsupported architecture: $ARCH" >&2
        echo "Use --amd64, --arm64, or --control-plane to select a compose file." >&2
        exit 1
        ;;
    esac
    ;;
  amd64|emulated)
    COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard.yaml"
    ;;
  arm64)
    COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard-arm64-native.yaml"
    ;;
  arm64-host-native)
    COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard-arm64-host-native.yaml"
    ;;
  control-plane)
    COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard-default.yaml"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac

if [[ "$MODE" == "emulated" ]] || ([[ "$MODE" == "auto" ]] && [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]] && [[ "$COMPOSE_FILE" == *docker-compose-dashboard.yaml ]]); then
  cat <<'EOF'
NOTE: You are running the AMD64 NIM stack on an ARM64 host.
- This requires qemu/binfmt emulation to be installed and enabled.
- Performance and stability may be significantly worse than native.
If this fails, use:
  ./scripts/run_dashboard_stack.sh --arm64 up -d --build
EOF
fi

exec docker compose -f "$COMPOSE_FILE" "$@"
