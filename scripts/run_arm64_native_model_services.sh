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

if [[ -z "${ALPHAFOLD_NATIVE_CMD:-}" ]]; then
  echo "ERR: ALPHAFOLD_NATIVE_CMD is not set" >&2
  exit 2
fi
if [[ -z "${RFDIFFUSION_NATIVE_CMD:-}" ]]; then
  echo "ERR: RFDIFFUSION_NATIVE_CMD is not set" >&2
  exit 2
fi

python3 -m venv "$ROOT_DIR/.venv-native-services" >/dev/null 2>&1 || true
source "$ROOT_DIR/.venv-native-services/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q -r "$ROOT_DIR/native_services/requirements.txt"

echo "Starting AlphaFold2 native service on :$ALPHAFOLD_PORT"
uvicorn native_services.alphafold_service:app --host 0.0.0.0 --port "$ALPHAFOLD_PORT" &
AF_PID=$!

echo "Starting RFdiffusion native service on :$RFDIFFUSION_PORT"
uvicorn native_services.rfdiffusion_service:app --host 0.0.0.0 --port "$RFDIFFUSION_PORT" &
RF_PID=$!

echo "PIDs: alphafold=$AF_PID rfdiffusion=$RF_PID"

echo "Health checks:"
for i in 1 2 3 4 5; do
  curl -fsS "http://localhost:$ALPHAFOLD_PORT/v1/health/ready" >/dev/null 2>&1 && break || true
  sleep 1
done
curl -fsS "http://localhost:$ALPHAFOLD_PORT/v1/health/ready" || true
curl -fsS "http://localhost:$RFDIFFUSION_PORT/v1/health/ready" || true

echo
echo "Services running. Press Ctrl+C to stop."
wait
