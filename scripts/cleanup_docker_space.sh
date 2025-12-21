#!/usr/bin/env bash
set -euo pipefail

# Safe(ish) Docker disk cleanup.
# - Removes stopped containers
# - Removes unused images
# - Removes build cache
# By default does NOT remove volumes.
#
# Usage:
#   ./scripts/cleanup_docker_space.sh
#   ./scripts/cleanup_docker_space.sh --volumes

PRUNE_VOLUMES=0
if [[ "${1:-}" == "--volumes" ]]; then
  PRUNE_VOLUMES=1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERR: docker not found"
  exit 1
fi

echo "=== Docker disk usage (before) ==="
docker system df || true

echo
echo "=== Pruning stopped containers ==="
docker container prune -f || true

echo
echo "=== Pruning unused images ==="
# -a removes all unused images, not just dangling.
docker image prune -a -f || true

echo
echo "=== Pruning build cache ==="
docker builder prune -a -f || true

if (( PRUNE_VOLUMES == 1 )); then
  echo
  echo "=== Pruning unused volumes (WARNING) ==="
  docker volume prune -f || true
else
  echo
  echo "Skipping volume prune (run with --volumes to enable)."
fi

echo
echo "=== Docker disk usage (after) ==="
docker system df || true
