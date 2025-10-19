#!/bin/bash
# ProteinMPNN ARM64 Runner Script

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate proteinmpnn_arm64

# Set environment variables
export PROTEINMPNN_HOME="/home/barberb/generative-protein-binder-design/tools/proteinmpnn"

# Run ProteinMPNN
python "${PROTEINMPNN_HOME}/ProteinMPNN/protein_mpnn_run.py" "$@"
