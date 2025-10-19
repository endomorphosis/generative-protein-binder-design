#!/bin/bash
# Start MCP Server with Native Backend for DGX Spark
# This runs models directly on hardware without NIM containers

echo "Starting MCP Server with Native Backend..."
echo "Backend Mode: Native (Direct Model Execution)"
echo ""

# Set backend mode
export MODEL_BACKEND=native

# Set model paths (adjust these for your installation)
export ALPHAFOLD_PATH=${ALPHAFOLD_PATH:-/opt/alphafold}
export ALPHAFOLD_DATA_DIR=${ALPHAFOLD_DATA_DIR:-/data/alphafold_params}
export RFDIFFUSION_PATH=${RFDIFFUSION_PATH:-/opt/rfdiffusion}
export RFDIFFUSION_MODELS=${RFDIFFUSION_MODELS:-/data/rfdiffusion_models}
export PROTEINMPNN_PATH=${PROTEINMPNN_PATH:-/opt/proteinmpnn}
export PROTEINMPNN_MODELS=${PROTEINMPNN_MODELS:-/data/proteinmpnn_models}

# Set GPU configuration (optional - specify which GPUs to use)
# export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Configuration:"
echo "  ALPHAFOLD_PATH: $ALPHAFOLD_PATH"
echo "  RFDIFFUSION_PATH: $RFDIFFUSION_PATH"
echo "  PROTEINMPNN_PATH: $PROTEINMPNN_PATH"
echo ""

# Start server
cd "$(dirname "$0")"
python3 server.py
