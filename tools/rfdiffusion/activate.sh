#!/bin/bash
# RFDiffusion Environment Activation Script

eval "$(conda shell.bash hook)"
conda activate rfdiffusion

export RFDIFFUSION_DIR="/home/barberb/generative-protein-binder-design/tools/rfdiffusion/RFdiffusion"
export RFDIFFUSION_MODELS="/home/barberb/.cache/rfdiffusion/models"
export PYTHONPATH="/home/barberb/generative-protein-binder-design/tools/rfdiffusion/RFdiffusion:$PYTHONPATH"

echo "RFDiffusion environment activated"
echo "  Installation: $RFDIFFUSION_DIR"
echo "  Models: $RFDIFFUSION_MODELS"
