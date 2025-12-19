#!/usr/bin/env python3
"""Validate RFDiffusion installation"""

import sys
import os

def validate_imports():
    """Test imports"""
    print("Validating imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        from Bio import SeqIO
        print(f"  ✓ BioPython")
    except ImportError as e:
        print(f"  ✗ BioPython import failed: {e}")
        return False
    
    try:
        import e3nn
        print(f"  ✓ e3nn {e3nn.__version__}")
    except ImportError as e:
        print(f"  ✗ e3nn import failed: {e}")
        return False
    
    return True

def validate_models():
    """Validate model directory"""
    print("\nValidating models...")
    
    models_dir = os.environ.get('RFDIFFUSION_MODELS')
    if not models_dir:
        print("  ✗ RFDIFFUSION_MODELS not set")
        return False
    
    print(f"  Models directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"  ✗ Models directory not found: {models_dir}")
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    if len(model_files) == 0:
        print(f"  ⚠ No model files found (download may be needed)")
    else:
        print(f"  ✓ Found {len(model_files)} model files")
        for model in model_files:
            size_mb = os.path.getsize(os.path.join(models_dir, model)) / (1024*1024)
            print(f"    - {model} ({size_mb:.1f} MB)")
    
    return True

def validate_installation():
    """Validate RFDiffusion installation"""
    print("\nValidating installation...")
    
    rfdiffusion_dir = os.environ.get('RFDIFFUSION_DIR')
    if not rfdiffusion_dir:
        print("  ✗ RFDIFFUSION_DIR not set")
        return False
    
    print(f"  Installation directory: {rfdiffusion_dir}")
    
    inference_script = os.path.join(rfdiffusion_dir, 'scripts', 'run_inference.py')
    if not os.path.exists(inference_script):
        print(f"  ✗ Inference script not found: {inference_script}")
        return False
    
    print(f"  ✓ Inference script found")
    
    return True

if __name__ == '__main__':
    print("="*60)
    print("RFDiffusion Installation Validation")
    print("="*60)
    print()
    
    success = (
        validate_imports() and
        validate_models() and
        validate_installation()
    )
    
    print()
    if success:
        print("="*60)
        print("✓ RFDiffusion installation is valid and ready to use!")
        print("="*60)
        sys.exit(0)
    else:
        print("="*60)
        print("✗ Validation failed. Please check errors above.")
        print("="*60)
        sys.exit(1)
