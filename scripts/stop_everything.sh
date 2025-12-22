#!/usr/bin/env bash
set -euo pipefail

# Stop MCP dashboard stack AND any auxiliary backend services started by
# scripts/start_everything.sh.
#
# Usage:
#   ./scripts/stop_everything.sh
#   ./scripts/stop_everything.sh --arm64-host-native
#   ./scripts/stop_everything.sh --amd64 -- --volumes
#
# Notes:
# - For ARM64 host-native mode, this attempts to stop the background process
#   launched by start_everything.sh (PID stored in outputs/host-native-services.pid).

usage() {
  cat <<'EOF'
Usage:
  scripts/stop_everything.sh [--amd64|--arm64|--arm64-host-native|--emulated|--control-plane] [-- <compose-args...>]

Examples:
  ./scripts/stop_everything.sh
  ./scripts/stop_everything.sh --arm64-host-native
  ./scripts/stop_everything.sh --amd64 -- down --volumes
EOF
}

MODE="auto"
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --amd64|--arm64|--arm64-host-native|--emulated|--control-plane)
      MODE="${1#--}"
      shift
      ;;
    --)
      shift
      PASSTHRU+=("$@")
      break
      ;;
    *)
      PASSTHRU+=("$1")
      shift
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

stop_host_native_services_if_present() {
  local pid_file="$ROOT_DIR/outputs/host-native-services.pid"
  local log_file="$ROOT_DIR/outputs/host-native-services.log"
  local mem_pid_file="$ROOT_DIR/outputs/memory-watchdog.pid"
  local mem_log_file="$ROOT_DIR/outputs/memory-watchdog.log"

  # Stop memory watchdog first so it doesn't fight teardown.
  if [[ -f "$mem_pid_file" ]]; then
    local mpid
    mpid="$(cat "$mem_pid_file" 2>/dev/null || true)"
    if [[ -n "$mpid" ]] && kill -0 "$mpid" >/dev/null 2>&1; then
      echo "Stopping memory watchdog (pid=$mpid)..."
      kill "$mpid" >/dev/null 2>&1 || true
      for _ in $(seq 1 20); do
        if kill -0 "$mpid" >/dev/null 2>&1; then
          sleep 0.2
        else
          break
        fi
      done
      if kill -0 "$mpid" >/dev/null 2>&1; then
        kill -9 "$mpid" >/dev/null 2>&1 || true
      fi
      echo "Memory watchdog stopped. Logs were at: $mem_log_file"
    fi
    rm -f "$mem_pid_file" || true
  fi

  if [[ ! -f "$pid_file" ]]; then
    # Nothing to stop (or was started manually).
    return 0
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file" || true
    return 0
  fi

  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "Stopping host-native services (pid=$pid)..."
    kill "$pid" >/dev/null 2>&1 || true

    # Wait briefly for clean shutdown.
    for _ in $(seq 1 20); do
      if kill -0 "$pid" >/dev/null 2>&1; then
        sleep 0.5
      else
        break
      fi
    done

    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "Host-native services did not exit; sending SIGKILL..." >&2
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi

    echo "Host-native services stopped. Logs were at: $log_file"
  else
    echo "Host-native services pid file found, but process is not running (pid=$pid)."
  fi

  rm -f "$pid_file" || true
}

# If ARM64 host-native, stop the background host-native services as well.
if [[ "$MODE" == "arm64-host-native" ]]; then
  stop_host_native_services_if_present
fi

# Default compose action is a safe 'down --remove-orphans'.
COMPOSE_ARGS=(down --remove-orphans)
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  COMPOSE_ARGS=("${PASSTHRU[@]}")
fi

exec "$ROOT_DIR/scripts/run_dashboard_stack.sh" "--$MODE" "${COMPOSE_ARGS[@]}"
