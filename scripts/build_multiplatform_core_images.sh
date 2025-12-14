#!/usr/bin/env bash
set -euo pipefail

# Build multi-arch images for the "core" services we own:
# - MCP server (mcp-server/Dockerfile)
# - MCP dashboard (mcp-dashboard/Dockerfile)
#
# By default, this builds and PUSHES a multi-arch manifest using buildx.
# For local-only development, it is usually simpler to just rely on
# `docker compose ... up --build` on each host (it will build for that host arch).
#
# Usage examples:
#   REGISTRY=ghcr.io/hallucinate-llc TAG=dev PUSH=1 ./scripts/build_multiplatform_core_images.sh
#   REGISTRY=protein-binder TAG=local PUSH=0 ./scripts/build_multiplatform_core_images.sh  # builds single-arch locally

REGISTRY="${REGISTRY:-protein-binder}"
TAG="${TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-1}"
BUILDER_NAME="${BUILDER_NAME:-multiarch-builder}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found; install Docker first." >&2
  exit 1
fi

if [[ "$PUSH" == "1" ]]; then
  if ! docker buildx version >/dev/null 2>&1; then
    echo "docker buildx not available. Try: docker buildx install" >&2
    exit 1
  fi

  if ! docker buildx ls | awk '{print $1}' | grep -q "^${BUILDER_NAME}$"; then
    docker buildx create --name "$BUILDER_NAME" --use >/dev/null
    docker buildx inspect --bootstrap >/dev/null
  else
    docker buildx use "$BUILDER_NAME" >/dev/null
  fi

  echo "Building multi-arch (push) -> $REGISTRY/mcp-server:$TAG"
  docker buildx build \
    --platform "$PLATFORMS" \
    -t "$REGISTRY/mcp-server:$TAG" \
    -f "$ROOT_DIR/mcp-server/Dockerfile" \
    "$ROOT_DIR/mcp-server" \
    --push

  echo "Building multi-arch (push) -> $REGISTRY/mcp-dashboard:$TAG"
  docker buildx build \
    --platform "$PLATFORMS" \
    -t "$REGISTRY/mcp-dashboard:$TAG" \
    -f "$ROOT_DIR/mcp-dashboard/Dockerfile" \
    "$ROOT_DIR/mcp-dashboard" \
    --push

  echo "Done. Published:"
  echo "  $REGISTRY/mcp-server:$TAG"
  echo "  $REGISTRY/mcp-dashboard:$TAG"
else
  # Local-only build for the current host architecture.
  echo "PUSH=0 -> building local single-arch images with docker build"
  docker build -t "$REGISTRY/mcp-server:$TAG" -f "$ROOT_DIR/mcp-server/Dockerfile" "$ROOT_DIR/mcp-server"
  docker build -t "$REGISTRY/mcp-dashboard:$TAG" -f "$ROOT_DIR/mcp-dashboard/Dockerfile" "$ROOT_DIR/mcp-dashboard"
  echo "Done. Built locally:"
  echo "  $REGISTRY/mcp-server:$TAG"
  echo "  $REGISTRY/mcp-dashboard:$TAG"
fi
