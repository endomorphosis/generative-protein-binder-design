#!/usr/bin/env bash
set -euo pipefail

# Ensure required external binaries for AlphaFold are installed on the host.
# AlphaFold shells out to:
#   - jackhmmer (hmmer)
#   - hhblits/hhsearch (hhsuite)
#   - kalign
#
# This script is intended to be run by provisioning/start scripts (foreground),
# because package installation may require sudo.

need_cmds=(jackhmmer hhblits hhsearch kalign)

missing=()
for c in "${need_cmds[@]}"; do
  if ! command -v "$c" >/dev/null 2>&1; then
    missing+=("$c")
  fi
done

if [[ ${#missing[@]} -eq 0 ]]; then
  echo "OK: AlphaFold external binaries already present."
  exit 0
fi

echo "Missing AlphaFold external binaries: ${missing[*]}" >&2

if command -v apt-get >/dev/null 2>&1; then
  # Prefer apt-get on Ubuntu/Debian (DGX Spark typical).
  if [[ $EUID -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      echo "Installing via apt-get (requires sudo)..." >&2
      sudo apt-get update -y
      sudo apt-get install -y hmmer hhsuite kalign
    else
      echo "ERR: Need root/sudo to install packages, but sudo not found." >&2
      exit 2
    fi
  else
    apt-get update -y
    apt-get install -y hmmer hhsuite kalign
  fi
else
  echo "ERR: Unsupported system (apt-get not found). Install these tools manually:" >&2
  echo "  - jackhmmer (hmmer)" >&2
  echo "  - hhblits/hhsearch (hhsuite)" >&2
  echo "  - kalign" >&2
  exit 2
fi

# Verify
missing_after=()
for c in "${need_cmds[@]}"; do
  if ! command -v "$c" >/dev/null 2>&1; then
    missing_after+=("$c")
  fi
done

if [[ ${#missing_after[@]} -ne 0 ]]; then
  echo "ERR: Still missing after install: ${missing_after[*]}" >&2
  exit 1
fi

echo "OK: Installed AlphaFold external binaries."
