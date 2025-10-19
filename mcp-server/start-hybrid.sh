#!/bin/bash
# Start MCP Server with Hybrid Backend
# Tries Native backend first, falls back to NIM if unavailable

echo "Starting MCP Server with Hybrid Backend..."
echo "Backend Mode: Hybrid (Native + NIM Fallback)"
echo ""

# Set backend mode
export MODEL_BACKEND=hybrid

# Set model paths for native backend
export ALPHAFOLD_PATH=${ALPHAFOLD_PATH:-/opt/alphafold}
export RFDIFFUSION_PATH=${RFDIFFUSION_PATH:-/opt/rfdiffusion}
export PROTEINMPNN_PATH=${PROTEINMPNN_PATH:-/opt/proteinmpnn}

# Set NIM URLs for fallback
export ALPHAFOLD_URL=${ALPHAFOLD_URL:-http://localhost:8081}
export RFDIFFUSION_URL=${RFDIFFUSION_URL:-http://localhost:8082}
export PROTEINMPNN_URL=${PROTEINMPNN_URL:-http://localhost:8083}
export ALPHAFOLD_MULTIMER_URL=${ALPHAFOLD_MULTIMER_URL:-http://localhost:8084}

echo "Configuration:"
echo "  Native paths:"
echo "    ALPHAFOLD_PATH: $ALPHAFOLD_PATH"
echo "    RFDIFFUSION_PATH: $RFDIFFUSION_PATH"
echo "    PROTEINMPNN_PATH: $PROTEINMPNN_PATH"
echo ""
echo "  NIM fallback URLs:"
echo "    ALPHAFOLD_URL: $ALPHAFOLD_URL"
echo "    RFDIFFUSION_URL: $RFDIFFUSION_URL"
echo "    PROTEINMPNN_URL: $PROTEINMPNN_URL"
echo "    ALPHAFOLD_MULTIMER_URL: $ALPHAFOLD_MULTIMER_URL"
echo ""

# Start server
cd "$(dirname "$0")"
python3 server.py
