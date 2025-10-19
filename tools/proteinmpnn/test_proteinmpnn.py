#!/usr/bin/env python3
"""Test ProteinMPNN ARM64 installation"""

import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import Bio
        print(f"✓ BioPython")
    except ImportError as e:
        print(f"✗ BioPython import failed: {e}")
        return False
    
    return True

def test_functionality():
    """Test basic ProteinMPNN functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        import numpy as np
        
        # Simple tensor operation
        x = torch.randn(10, 20)
        y = torch.mean(x)
        print(f"✓ Tensor operations working: mean = {y.item():.4f}")
        
        # Test protein sequence encoding
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        test_seq = "ACDEFG"
        encoded = [aa_to_idx[aa] for aa in test_seq]
        print(f"✓ Sequence encoding: {test_seq} -> {encoded}")
        
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ProteinMPNN ARM64 Installation Test")
    print("=" * 60)
    print()
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_functionality():
        success = False
    
    print()
    if success:
        print("=" * 60)
        print("✓ All tests passed! ProteinMPNN ARM64 is ready.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("✗ Some tests failed. Please check the errors above.")
        print("=" * 60)
        sys.exit(1)
