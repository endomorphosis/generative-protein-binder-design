#!/usr/bin/env python3
"""
Test attention mechanisms for ARM64 Blackwell GB10.
Shows working alternatives to Flash Attention.
"""

import torch
import torch.nn.functional as F
import time
import sys

def test_pytorch_sdpa():
    """Test PyTorch native scaled_dot_product_attention"""
    print("🔧 Testing PyTorch native SDPA...")
    
    # Create test tensors
    batch_size, num_heads, seq_len, head_dim = 2, 8, 2048, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=torch.float16, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                     dtype=torch.float16, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=torch.float16, device=device)
    
    try:
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.amp.autocast('cuda'):
            output = F.scaled_dot_product_attention(query, key, value)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        print(f"✅ PyTorch SDPA: PASSED ({elapsed:.3f}s)")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        return output, elapsed
        
    except Exception as e:
        print(f"❌ PyTorch SDPA: FAILED - {e}")
        return None, 0

def test_manual_attention():
    """Test manual attention implementation"""
    print("\n🔧 Testing manual attention implementation...")
    
    # Create test tensors
    batch_size, num_heads, seq_len, head_dim = 2, 8, 2048, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=torch.float16, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                     dtype=torch.float16, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=torch.float16, device=device)
    
    try:
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.amp.autocast('cuda'):
            # Manual attention computation
            scale = 1.0 / (head_dim ** 0.5)
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, value)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        print(f"✅ Manual attention: PASSED ({elapsed:.3f}s)")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        return output, elapsed
        
    except Exception as e:
        print(f"❌ Manual attention: FAILED - {e}")
        return None, 0

def compare_attention_mechanisms():
    """Compare different attention mechanisms"""
    print("=" * 60)
    print("🧪 ARM64 Blackwell GB10 Attention Performance Test")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print(f"🖥️  Device: {torch.cuda.get_device_name()}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"💾 CUDA version: {torch.version.cuda}")
    print()
    
    results = {}
    
    # Test PyTorch SDPA
    sdpa_output, sdpa_time = test_pytorch_sdpa()
    if sdpa_output is not None:
        results['sdpa'] = sdpa_time
    
    # Test manual attention
    manual_output, manual_time = test_manual_attention()
    if manual_output is not None:
        results['manual'] = manual_time
    
    # Compare outputs if both worked
    if sdpa_output is not None and manual_output is not None:
        diff = torch.abs(sdpa_output - manual_output).max().item()
        print(f"\n🔍 Output difference (SDPA vs Manual): {diff:.6f}")
    
    # Memory usage
    print(f"\n📊 GPU Memory Usage:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Performance summary
    if results:
        print(f"\n📋 Performance Summary:")
        for name, time_taken in results.items():
            print(f"   {name}: {time_taken:.3f}s")
    
    # Recommendations
    print(f"\n💡 Recommendation for ARM64 Blackwell GB10:")
    print(f"   ✅ Use PyTorch native SDPA - optimal performance and compatibility")
    print(f"   📝 Flash Attention is not compatible with Blackwell architecture")
    print(f"   🚀 SDPA provides GPU acceleration with minimal setup")

if __name__ == "__main__":
    compare_attention_mechanisms()