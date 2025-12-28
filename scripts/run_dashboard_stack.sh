#!/usr/bin/env bash
set -euo pipefail

# Run the MCP Dashboard + MCP Server stack on either AMD64 or ARM64.
#
# Auto mode:
# - AMD64/x86_64  -> deploy/docker-compose-dashboard.yaml (local NIM services; linux/amd64)
# - ARM64/aarch64 -> deploy/docker-compose-dashboard-arm64-host-native.yaml (routes to host-native services)
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
  scripts/run_dashboard_stack.sh [--amd64|--arm64|--arm64-host-native|--host-native|--emulated|--control-plane] <docker compose args...>

Modes:
  --amd64     Force the AMD64 NIM stack (deploy/docker-compose-dashboard.yaml)
  --arm64     Force the ARM64-native stack (deploy/docker-compose-dashboard-arm64-native.yaml)
  --arm64-host-native  ARM64 host-native AlphaFold2/RFdiffusion (deploy/docker-compose-dashboard-arm64-host-native.yaml)
  --host-native  Host-native wrappers (no NIM) (deploy/docker-compose-dashboard-host-native.yaml)
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
    --host-native)
      MODE="host-native"
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

is_host_native_wrappers_healthy() {
  # Best-effort probe: if the user has started the host-native wrappers on the host,
  # we should prefer them over NIM.
  local af_port="${ALPHAFOLD_NATIVE_PORT:-18081}"
  local rf_port="${RFDIFFUSION_NATIVE_PORT:-18082}"
  local afm_port="${ALPHAFOLD_MULTIMER_NATIVE_PORT:-18084}"

  curl -fsS "http://localhost:${af_port}/v1/health/ready" >/dev/null 2>&1 \
    && curl -fsS "http://localhost:${rf_port}/v1/health/ready" >/dev/null 2>&1 \
    && curl -fsS "http://localhost:${afm_port}/v1/health/ready" >/dev/null 2>&1
}

has_ngc_key() {
  [[ -n "${NGC_CLI_API_KEY:-}" ]]
}

case "$MODE" in
  auto)
    case "$ARCH" in
      x86_64|amd64)
        # Prefer local inference when available; only fall back to NIM when the user
        # provided an NGC key. Otherwise run the control-plane stack.
        if is_host_native_wrappers_healthy; then
          COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard-host-native.yaml"
        elif has_ngc_key; then
          COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard.yaml"
        else
          COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard-default.yaml"
        fi
        ;;
      aarch64|arm64)
        # Prefer the host-native stack on ARM64. The ARM64-native compose file is
        # primarily for CI shims / containerized builds and will often conflict
        # with host-native services bound on 18081/18082.
        COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard-arm64-host-native.yaml"
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
  host-native)
    COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose-dashboard-host-native.yaml"
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

ensure_nim_prereqs() {
  # Applies only to the AMD64 NIM stack.
  if [[ "$COMPOSE_FILE" != *"docker-compose-dashboard.yaml" ]]; then
    return 0
  fi

  if [[ -z "${NGC_CLI_API_KEY:-}" ]]; then
    cat >&2 <<'EOF'
NGC_CLI_API_KEY is required to pull and run the NIM model containers.

Set it in your shell, then re-run:
  export NGC_CLI_API_KEY="<YOUR_NGC_PERSONAL_RUN_KEY>"

If you haven't logged in to nvcr.io yet:
  echo "$NGC_CLI_API_KEY" | docker login nvcr.io --username='$oauthtoken' --password-stdin

You can also run the guided setup:
  ./scripts/setup_local.sh
EOF
    exit 2
  fi

  # Docker Compose doesn't reliably expand '~' in volume paths. Prefer an explicit path.
  if [[ -z "${HOST_NIM_CACHE:-}" ]]; then
    export HOST_NIM_CACHE="$HOME/.cache/nim"
  fi

  mkdir -p "$HOST_NIM_CACHE"
  # NIM containers may run as a non-root user; they need RW access to the host cache.
  chmod -R 777 "$HOST_NIM_CACHE" 2>/dev/null || true
}

if [[ "$MODE" == "emulated" ]] || ([[ "$MODE" == "auto" ]] && [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]] && [[ "$COMPOSE_FILE" == *docker-compose-dashboard.yaml ]]); then
  cat <<'EOF'
NOTE: You are running the AMD64 NIM stack on an ARM64 host.
- This requires qemu/binfmt emulation to be installed and enabled.
- Performance and stability may be significantly worse than native.
If this fails, use:
  ./scripts/run_dashboard_stack.sh --arm64 up -d --build
EOF
fi

ensure_nim_prereqs

if [[ "$COMPOSE_FILE" == *"docker-compose-dashboard.yaml" ]] && [[ "${1:-}" == "up" ]]; then
  cat <<'EOF'
INFO: First start will download large model assets into HOST_NIM_CACHE.
- AlphaFold2 / Multimer downloads can take hours (and require significant disk).
- Subsequent restarts are much faster once cached.
EOF
fi

exec docker compose -f "$COMPOSE_FILE" "$@"
