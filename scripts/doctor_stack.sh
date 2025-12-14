#!/usr/bin/env bash
set -euo pipefail

# Quick, human-friendly diagnostics for the MCP dashboard stack.
# Works on both AMD64 and ARM64, and follows the same compose selection logic
# as scripts/run_dashboard_stack.sh.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ok() { echo -e "${GREEN}OK${NC} $*"; }
warn() { echo -e "${YELLOW}WARN${NC} $*"; }
err() { echo -e "${RED}ERR${NC} $*"; }
info() { echo -e "${BLUE}INFO${NC} $*"; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="$(uname -m)"

MCP_DASHBOARD_HOST_PORT="${MCP_DASHBOARD_HOST_PORT:-3000}"
MCP_SERVER_HOST_PORT="${MCP_SERVER_HOST_PORT:-8011}"

MODE="auto"
if [[ ${1:-} == "--amd64" ]]; then MODE="amd64"; shift; fi
if [[ ${1:-} == "--arm64" ]]; then MODE="arm64"; shift; fi
if [[ ${1:-} == "--emulated" ]]; then MODE="emulated"; shift; fi

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
        err "Unsupported architecture: $ARCH"
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
  *)
    err "Unknown mode: $MODE"
    exit 2
    ;;
esac

info "Architecture: $ARCH"
info "Compose file: $COMPOSE_FILE"
info "Dashboard URL: http://localhost:${MCP_DASHBOARD_HOST_PORT}"
info "MCP Server URL: http://localhost:${MCP_SERVER_HOST_PORT}"

STACK_RUNNING=0
if docker compose -f "$COMPOSE_FILE" ps -q >/dev/null 2>&1; then
  if [[ -n "$(docker compose -f "$COMPOSE_FILE" ps -q 2>/dev/null | head -n 1)" ]]; then
    STACK_RUNNING=1
  fi
fi

echo
info "Checking Docker..."
if ! command -v docker >/dev/null 2>&1; then
  err "docker not found"
  exit 1
fi
if docker info >/dev/null 2>&1; then
  ok "Docker daemon reachable"
else
  err "Docker daemon not reachable (is it running? do you need sudo?)"
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  ok "Docker Compose v2 available"
else
  warn "docker compose not available (install Docker Compose v2)"
fi

echo
info "Checking expected ports..."
if command -v ss >/dev/null 2>&1; then
  if ss -ltnH "( sport = :${MCP_DASHBOARD_HOST_PORT} )" | grep -q .; then
    if [[ "$STACK_RUNNING" == "1" ]]; then
      ok "Port ${MCP_DASHBOARD_HOST_PORT} is in use (dashboard running)"
    else
      warn "Port ${MCP_DASHBOARD_HOST_PORT} is already in use (dashboard)"
    fi
  else
    ok "Port ${MCP_DASHBOARD_HOST_PORT} is free"
  fi
  if ss -ltnH "( sport = :${MCP_SERVER_HOST_PORT} )" | grep -q .; then
    if [[ "$STACK_RUNNING" == "1" ]]; then
      ok "Port ${MCP_SERVER_HOST_PORT} is in use (mcp-server running)"
    else
      warn "Port ${MCP_SERVER_HOST_PORT} is already in use (mcp-server)"
    fi
  else
    ok "Port ${MCP_SERVER_HOST_PORT} is free"
  fi
else
  warn "ss not found; skipping port checks"
fi

echo
info "Checking stack status (docker compose ps)..."
if docker compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  docker compose -f "$COMPOSE_FILE" ps
  ok "Compose project reachable"
else
  warn "Compose project not running (or compose file has issues)"
  echo "Try: ./scripts/run_dashboard_stack.sh up -d --build"
fi

echo
info "Checking MCP server /health..."
if curl -fsS "http://localhost:${MCP_SERVER_HOST_PORT}/health" >/dev/null 2>&1; then
  ok "MCP server /health is reachable"
else
  err "MCP server /health failed"
  echo "Try: ./scripts/run_dashboard_stack.sh up -d --build"
fi

echo
info "Checking backend services status (/api/services/status)..."
if curl -fsS "http://localhost:${MCP_SERVER_HOST_PORT}/api/services/status" >/dev/null 2>&1; then
  if command -v jq >/dev/null 2>&1; then
    curl -fsS "http://localhost:${MCP_SERVER_HOST_PORT}/api/services/status" | jq
  else
    curl -fsS "http://localhost:${MCP_SERVER_HOST_PORT}/api/services/status"
    echo
    warn "Install jq for prettier JSON output"
  fi
  ok "Service status endpoint reachable"
else
  warn "Service status endpoint not reachable"
fi

echo
info "Checking dashboard HTTP..."
if curl -fsS "http://localhost:${MCP_DASHBOARD_HOST_PORT}" >/dev/null 2>&1; then
  ok "Dashboard is reachable"
else
  warn "Dashboard not reachable on http://localhost:${MCP_DASHBOARD_HOST_PORT}"
  echo "If port is busy, try: MCP_DASHBOARD_HOST_PORT=3005 ./scripts/run_dashboard_stack.sh up -d --build"
fi

echo
ok "Doctor finished"
