#!/usr/bin/env bash
set -euo pipefail

# Removes Python bytecode/artifacts from the AlphaFold2 submodule so it doesn't
# show up as a dirty submodule (common after running python imports).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE_DIR="$ROOT_DIR/tools/alphafold2"

if [[ ! -d "$SUBMODULE_DIR" ]]; then
  echo "AlphaFold2 submodule not found at: $SUBMODULE_DIR" >&2
  exit 0
fi

# Safety: only delete known generated artifacts.
find "$SUBMODULE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$SUBMODULE_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
find "$SUBMODULE_DIR" -type d -name ".pytest_cache" -prune -exec rm -rf {} +

# Some runs create caches under the repo root too.
find "$SUBMODULE_DIR" -type f -name ".coverage" -delete 2>/dev/null || true

echo "Cleaned AlphaFold2 submodule artifacts."