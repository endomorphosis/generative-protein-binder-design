#!/usr/bin/env bash
set -euo pipefail

# Starts host-native NIM-compatible services for AlphaFold2 + RFdiffusion.
# Intended for ARM64 hosts (e.g. DGX Spark) where you installed the real tools
# in conda envs.
#
# Required env:
#   ALPHAFOLD_NATIVE_CMD        Command template that produces {out_dir}/result.pdb
#   RFDIFFUSION_NATIVE_CMD      Command template that produces {out_dir}/design_{design_id}.pdb
#
# Optional env:
#   ALPHAFOLD_NATIVE_PORT (default 18081)
#   RFDIFFUSION_NATIVE_PORT (default 18082)
#   ALPHAFOLD_NATIVE_TIMEOUT_SECONDS (default 7200)
#   RFDIFFUSION_NATIVE_TIMEOUT_SECONDS (default 7200)
#
# Example:
#   export ALPHAFOLD_NATIVE_CMD='conda run -n alphafold python /path/to/run_alphafold_wrapper.py --fasta {fasta} --out {out_dir} --db /data/alphafold'
#   export RFDIFFUSION_NATIVE_CMD='conda run -n rfdiffusion python /path/to/run_rfdiffusion_wrapper.py --target {target_pdb} --out {out_dir} --design {design_id}'
#   ./scripts/run_arm64_native_model_services.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALPHAFOLD_PORT="${ALPHAFOLD_NATIVE_PORT:-18081}"
RFDIFFUSION_PORT="${RFDIFFUSION_NATIVE_PORT:-18082}"
ALPHAFOLD_MULTIMER_PORT="${ALPHAFOLD_MULTIMER_NATIVE_PORT:-18084}"

# Avoid creating __pycache__ under the AlphaFold2 submodule (keeps the submodule clean).
export PYTHONDONTWRITEBYTECODE=1

maybe_source_env_file() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    # Load KEY=VALUE pairs without executing the file as shell.
    # This is important because command templates often contain spaces.
    while IFS= read -r line || [[ -n "$line" ]]; do
      # Trim leading/trailing whitespace
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"

      [[ -z "$line" ]] && continue
      [[ "$line" == \#* ]] && continue

      if [[ "$line" == export\ * ]]; then
        line="${line#export }"
      fi

      [[ "$line" != *"="* ]] && continue

      local key="${line%%=*}"
      local value="${line#*=}"

      key="${key%"${key##*[![:space:]]}"}"
      key="${key#"${key%%[![:space:]]*}"}"
      value="${value#"${value%%[![:space:]]*}"}"

      # Strip a single layer of surrounding quotes, if present.
      if [[ ${#value} -ge 2 ]]; then
        if [[ "$value" == \"*\" && "$value" == *\" ]]; then
          value="${value:1:${#value}-2}"
        elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
          value="${value:1:${#value}-2}"
        fi
      fi

      export "$key=$value"
    done < "$env_file"
  fi
}

# If the user didn't provide command templates, try to auto-load the ones
# produced by our "zero-touch" installers.
if [[ -z "${ALPHAFOLD_NATIVE_CMD:-}" ]]; then
  maybe_source_env_file "$ROOT_DIR/tools/generated/alphafold2/.env"
  # Backward compatibility: older installers wrote the .env under alphafold2/.
  maybe_source_env_file "$ROOT_DIR/tools/alphafold2/.env"
fi
if [[ -z "${RFDIFFUSION_NATIVE_CMD:-}" ]]; then
  maybe_source_env_file "$ROOT_DIR/tools/generated/rfdiffusion/.env"
  # Backward compatibility: older installers wrote the .env under tools/rfdiffusion/.
  maybe_source_env_file "$ROOT_DIR/tools/rfdiffusion/.env"
  # Backward compatibility: older installers wrote the .env under RFdiffusion/.
  maybe_source_env_file "$ROOT_DIR/tools/rfdiffusion/RFdiffusion/.env"
fi

if [[ -z "${ALPHAFOLD_NATIVE_CMD:-}" ]]; then
  echo "ERR: ALPHAFOLD_NATIVE_CMD is not set" >&2
  echo "Hint: run ./scripts/install_alphafold2_complete.sh (it writes tools/generated/alphafold2/.env)" >&2
  exit 2
fi

# Best-effort: if the user didn't configure a separate multimer command, derive
# one from the monomer command (the installer writes --model_preset=monomer).
if [[ -z "${ALPHAFOLD_MULTIMER_NATIVE_CMD:-}" ]]; then
  if [[ "${ALPHAFOLD_NATIVE_CMD:-}" == *"--model_preset=monomer"* ]]; then
    ALPHAFOLD_MULTIMER_NATIVE_CMD="${ALPHAFOLD_NATIVE_CMD/--model_preset=monomer/--model_preset=multimer}"
    export ALPHAFOLD_MULTIMER_NATIVE_CMD
  fi
fi

if [[ -z "${ALPHAFOLD_MULTIMER_NATIVE_OUTPUT_PDB:-}" && -n "${ALPHAFOLD_NATIVE_OUTPUT_PDB:-}" ]]; then
  export ALPHAFOLD_MULTIMER_NATIVE_OUTPUT_PDB="$ALPHAFOLD_NATIVE_OUTPUT_PDB"
fi
if [[ -z "${RFDIFFUSION_NATIVE_CMD:-}" ]]; then
  echo "ERR: RFDIFFUSION_NATIVE_CMD is not set" >&2
  echo "Hint: run ./scripts/install_rfdiffusion_complete.sh (it writes tools/generated/rfdiffusion/.env)" >&2
  exit 2
fi

ensure_conda_on_path() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  local candidates=(
    "$HOME/miniconda3/bin/conda"
    "$HOME/miniforge3/bin/conda"
    "$HOME/mambaforge/bin/conda"
    "$HOME/anaconda3/bin/conda"
    "/opt/conda/bin/conda"
    "/usr/local/bin/conda"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      export PATH="$(dirname "$candidate"):$PATH"
      return 0
    fi
  done
}

ensure_conda_on_path

python3 -m venv "$ROOT_DIR/.venv-native-services" >/dev/null 2>&1 || true
source "$ROOT_DIR/.venv-native-services/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q -r "$ROOT_DIR/native_services/requirements.txt"

# Best-effort cleanup of already-created artifacts inside the submodule.
"$ROOT_DIR/scripts/clean_alphafold2_submodule_artifacts.sh" >/dev/null 2>&1 || true

echo "Starting AlphaFold2 native service on :$ALPHAFOLD_PORT"
uvicorn native_services.alphafold_service:app --host 0.0.0.0 --port "$ALPHAFOLD_PORT" &
AF_PID=$!

if [[ -n "${ALPHAFOLD_MULTIMER_NATIVE_CMD:-}" ]]; then
  echo "Starting AlphaFold2-multimer native service on :$ALPHAFOLD_MULTIMER_PORT"
  uvicorn native_services.alphafold_multimer_service:app --host 0.0.0.0 --port "$ALPHAFOLD_MULTIMER_PORT" &
  AFM_PID=$!
else
  AFM_PID=""
  echo "WARN: ALPHAFOLD_MULTIMER_NATIVE_CMD is not set; multimer service will not start" >&2
fi

echo "Starting RFdiffusion native service on :$RFDIFFUSION_PORT"
uvicorn native_services.rfdiffusion_service:app --host 0.0.0.0 --port "$RFDIFFUSION_PORT" &
RF_PID=$!

cleanup() {
  kill "$AF_PID" >/dev/null 2>&1 || true
  if [[ -n "${AFM_PID:-}" ]]; then
    kill "$AFM_PID" >/dev/null 2>&1 || true
  fi
  kill "$RF_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "PIDs: alphafold=$AF_PID alphafold_multimer=${AFM_PID:-disabled} rfdiffusion=$RF_PID"

echo "Health checks:"
for i in 1 2 3 4 5; do
  curl -fsS "http://localhost:$ALPHAFOLD_PORT/v1/health/ready" >/dev/null 2>&1 && break || true
  sleep 1
done
curl -fsS "http://localhost:$ALPHAFOLD_PORT/v1/health/ready" || true
if [[ -n "${AFM_PID:-}" ]]; then
  curl -fsS "http://localhost:$ALPHAFOLD_MULTIMER_PORT/v1/health/ready" || true
fi
curl -fsS "http://localhost:$RFDIFFUSION_PORT/v1/health/ready" || true

echo
echo "Services running. Press Ctrl+C to stop."
wait
