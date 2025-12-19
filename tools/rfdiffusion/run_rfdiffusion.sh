#!/bin/bash
# RFDiffusion Run Script

source "/home/barberb/generative-protein-binder-design/tools/rfdiffusion/activate.sh"

python "$RFDIFFUSION_DIR/scripts/run_inference.py" \
    inference.model_directory_path="$RFDIFFUSION_MODELS" \
    "$@"
