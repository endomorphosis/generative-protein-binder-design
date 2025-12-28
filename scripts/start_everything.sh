#!/usr/bin/env bash
set -euo pipefail

# Start MCP dashboard stack AND any required backend services together.
#
# Default behavior:
# - Starts the appropriate docker-compose stack (auto-detect arch) with `up -d --build`.
# - If using ARM64 host-native mode, also starts the host-native AlphaFold2+RFDiffusion
#   wrapper services in the background (unless they're already healthy).
#
# Usage:
#   ./scripts/start_everything.sh
#   ./scripts/start_everything.sh --arm64-host-native
#   ./scripts/start_everything.sh --host-native
#   ./scripts/start_everything.sh --amd64
#   ./scripts/start_everything.sh --arm64
#   ./scripts/start_everything.sh --control-plane
#
# Options:
#   --no-build         Do not pass --build to docker compose
#   --provision        For --arm64-host-native: run provisioning if tools/*/.env missing
#   --db-tier <tier>   For --provision: minimal|reduced|full (default: minimal)
#   --                Everything after -- is passed through to docker compose

usage() {
  cat <<'EOF'
Usage:
  scripts/start_everything.sh [--amd64|--arm64|--arm64-host-native|--host-native|--emulated|--control-plane]
                             [--no-build] [--provision] [--db-tier minimal|reduced|full] [-- <compose-args...>]

Examples:
  ./scripts/start_everything.sh
  ./scripts/start_everything.sh --arm64-host-native --provision --db-tier minimal
  ./scripts/start_everything.sh --host-native
  ./scripts/start_everything.sh --amd64 -- --pull always
EOF
}

MODE="auto"
BUILD=1
PROVISION=0
DB_TIER="minimal"
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --amd64|--arm64|--arm64-host-native|--host-native|--emulated|--control-plane)
      MODE="${1#--}"
      shift
      ;;
    --no-build)
      BUILD=0
      shift
      ;;
    --provision)
      PROVISION=1
      shift
      ;;
    --db-tier)
      DB_TIER="${2:-}"
      shift 2
      ;;
    --)
      shift
      PASSTHRU+=("$@")
      break
      ;;
    *)
      # Unrecognized args are treated as compose args.
      PASSTHRU+=("$1")
      shift
      ;;
  esac
done

case "$DB_TIER" in
  minimal|reduced|full) ;;
  "") DB_TIER="minimal" ;;
  *)
    echo "ERR: invalid --db-tier '$DB_TIER'" >&2
    usage
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Auto-select the most useful mode.
# On ARM64, the containerized "native" stack is CI-only and does not perform real inference.
# For real out-of-box results, default to host-native mode and provision minimal assets.
if [[ "$MODE" == "auto" ]]; then
  ARCH="$(uname -m)"
  if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    MODE="arm64-host-native"
    if [[ "$PROVISION" == "0" ]]; then
      PROVISION=1
    fi
  else
    # On x86_64, prefer host-native wrappers when they have been provisioned.
    if [[ -f "$ROOT_DIR/tools/generated/alphafold2/.env" || -f "$ROOT_DIR/tools/alphafold2/.env" ]]; then
      if [[ -f "$ROOT_DIR/tools/generated/rfdiffusion/.env" || -f "$ROOT_DIR/tools/rfdiffusion/.env" || -f "$ROOT_DIR/tools/rfdiffusion/RFdiffusion/.env" ]]; then
        MODE="host-native"
      fi
    fi
  fi
fi

start_host_native_services_if_needed() {
  local alphafold_port="${ALPHAFOLD_NATIVE_PORT:-18081}"
  local alphafold_multimer_port="${ALPHAFOLD_MULTIMER_NATIVE_PORT:-18084}"
  local rfdiffusion_port="${RFDIFFUSION_NATIVE_PORT:-18082}"
  local af_env_file="$ROOT_DIR/tools/generated/alphafold2/.env"
  local af_env_file_legacy="$ROOT_DIR/tools/alphafold2/.env"
  local rf_env_file="$ROOT_DIR/tools/generated/rfdiffusion/.env"
  local rf_env_file_legacy="$ROOT_DIR/tools/rfdiffusion/.env"
  local rf_env_file_legacy2="$ROOT_DIR/tools/rfdiffusion/RFdiffusion/.env"

  # Provision on demand (only if requested).
  if [[ "$PROVISION" == "1" ]]; then
    # Ensure required external binaries exist (AlphaFold shells out to these).
    "$ROOT_DIR/scripts/ensure_alphafold_external_binaries.sh" || true
    if [[ (! -f "$af_env_file" && ! -f "$af_env_file_legacy") || (! -f "$rf_env_file" && ! -f "$rf_env_file_legacy" && ! -f "$rf_env_file_legacy2") ]]; then
      echo "Provisioning host-native tools/assets (db-tier=$DB_TIER)..."
      "$ROOT_DIR/scripts/provision_arm64_host_native_models.sh" --db-tier "$DB_TIER"
    fi
  fi

  if [[ (! -f "$af_env_file" && ! -f "$af_env_file_legacy") || (! -f "$rf_env_file" && ! -f "$rf_env_file_legacy" && ! -f "$rf_env_file_legacy2") ]]; then
    cat >&2 <<EOF
ERR: Host-native services are not configured yet.

Run provisioning first (recommended):
  ./scripts/provision_arm64_host_native_models.sh --db-tier minimal

Then retry:
  ./scripts/start_everything.sh --arm64-host-native
EOF
    exit 2
  fi

  # If already healthy, don't start another copy.
  if curl -fsS "http://localhost:${alphafold_port}/v1/health/ready" >/dev/null 2>&1 \
    && curl -fsS "http://localhost:${alphafold_port}/v1/metrics" >/dev/null 2>&1 \
    && curl -fsS "http://localhost:${alphafold_multimer_port}/v1/health/ready" >/dev/null 2>&1 \
    && curl -fsS "http://localhost:${rfdiffusion_port}/v1/health/ready" >/dev/null 2>&1; then
    echo "Host-native AlphaFold2/RFDiffusion services already healthy (:${alphafold_port}, :${alphafold_multimer_port}, :${rfdiffusion_port})."
    return 0
  fi

  mkdir -p "$ROOT_DIR/outputs"
  local log_file="$ROOT_DIR/outputs/host-native-services.log"
  local pid_file="$ROOT_DIR/outputs/host-native-services.pid"

  # Start a lightweight memory watchdog that evicts our DB page-cache under pressure.
  # Safe for non-technical users: it does not lock memory and does not drop global caches.
  local mem_pid_file="$ROOT_DIR/outputs/memory-watchdog.pid"
  local mem_log_file="$ROOT_DIR/outputs/memory-watchdog.log"
  if [[ -f "$mem_pid_file" ]]; then
    local oldpid
    oldpid="$(cat "$mem_pid_file" 2>/dev/null || true)"
    if [[ -n "$oldpid" ]] && kill -0 "$oldpid" >/dev/null 2>&1; then
      echo "Memory watchdog already running (pid=$oldpid)."
    else
      rm -f "$mem_pid_file" || true
    fi
  fi
  if [[ ! -f "$mem_pid_file" ]]; then
    echo "Starting memory watchdog in background..."
    echo "- Logs: $mem_log_file"
    nohup python3 "$ROOT_DIR/scripts/memory_watchdog.py" >"$mem_log_file" 2>&1 &
    echo $! > "$mem_pid_file"
  fi

  echo "Starting host-native AlphaFold2/RFDiffusion services in background..."
  echo "- Logs: $log_file"
  nohup bash "$ROOT_DIR/scripts/run_arm64_native_model_services.sh" >"$log_file" 2>&1 &
  echo $! > "$pid_file"

  echo "Waiting for host-native services to become healthy..."
  for _ in $(seq 1 90); do
    if curl -fsS "http://localhost:${alphafold_port}/v1/health/ready" >/dev/null 2>&1 \
      && curl -fsS "http://localhost:${alphafold_port}/v1/metrics" >/dev/null 2>&1 \
      && curl -fsS "http://localhost:${alphafold_multimer_port}/v1/health/ready" >/dev/null 2>&1 \
      && curl -fsS "http://localhost:${rfdiffusion_port}/v1/health/ready" >/dev/null 2>&1; then
      echo "Host-native services are healthy."
      return 0
    fi
    sleep 2
  done

  echo "WARN: Host-native services did not become healthy in time." >&2
  echo "Check logs: $log_file" >&2
  exit 1
}

# If the selected mode is ARM64 host-native, ensure host-native services are running.
if [[ "$MODE" == "arm64-host-native" || "$MODE" == "host-native" ]]; then
  start_host_native_services_if_needed
fi

# Bring up the docker compose stack (MCP server + dashboard + any containerized backends).
COMPOSE_ARGS=(up -d)
if [[ "$BUILD" == "1" ]]; then
  COMPOSE_ARGS+=(--build)
fi
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  COMPOSE_ARGS+=("${PASSTHRU[@]}")
fi

if [[ "$MODE" == "auto" ]]; then
  exec "$ROOT_DIR/scripts/run_dashboard_stack.sh" "${COMPOSE_ARGS[@]}"
else
  exec "$ROOT_DIR/scripts/run_dashboard_stack.sh" "--$MODE" "${COMPOSE_ARGS[@]}"
fi
