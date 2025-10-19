#!/usr/bin/env python3
"""
Test Flash Attention functionality on ARM64 Blackwell GB10 GPU
Using PyTorch native SDPA as fallback for Blackwell architecture compatibility
"""

import torch
import torch.nn.functional as F
import time
import gc

def test_attention_backends():
    """Test different attention implementations"""
    print(f"🚀 Testing Flash Attention on ARM64 Blackwell GB10")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()
    
    # Test parameters
    batch_size = 2
    seq_len = 2048
    num_heads = 8
    head_dim = 64
    device = torch.device('cuda')
    
    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    
    print(f"📊 Test configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Number of heads: {num_heads}")
    print(f"   Head dimension: {head_dim}")
    print(f"   Data type: {q.dtype}")
    print()
    
    results = {}
    
    # Test 1: PyTorch native SDPA
    print("🔧 Testing PyTorch native SDPA...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.cuda.amp.autocast():
            output_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        torch.cuda.synchronize()
        sdpa_time = time.time() - start_time
        
        print(f"✅ Native SDPA: PASSED ({sdpa_time:.3f}s)")
        print(f"   Output shape: {output_sdpa.shape}")
        print(f"   Output range: [{output_sdpa.min():.3f}, {output_sdpa.max():.3f}]")
        results['sdpa'] = {'time': sdpa_time, 'success': True, 'output': output_sdpa}
    except Exception as e:
        print(f"❌ Native SDPA: FAILED - {e}")
        results['sdpa'] = {'time': 0, 'success': False, 'error': str(e)}
    print()
    
    # Test 2: Flash Attention (if available)
    print("🔧 Testing Flash Attention...")
    try:
        import flash_attn
        print(f"   Flash Attention version: {flash_attn.__version__}")
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Flash attention expects different tensor layout
        q_fa = q.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        k_fa = k.transpose(1, 2)
        v_fa = v.transpose(1, 2)
        
        from flash_attn import flash_attn_func
        output_fa = flash_attn_func(q_fa, k_fa, v_fa)
        output_fa = output_fa.transpose(1, 2)  # Back to [batch, num_heads, seq_len, head_dim]
        
        torch.cuda.synchronize()
        fa_time = time.time() - start_time
        
        print(f"✅ Flash Attention: PASSED ({fa_time:.3f}s)")
        print(f"   Output shape: {output_fa.shape}")
        print(f"   Output range: [{output_fa.min():.3f}, {output_fa.max():.3f}]")
        results['flash_attn'] = {'time': fa_time, 'success': True, 'output': output_fa}
        
        # Compare outputs
        if 'sdpa' in results and results['sdpa']['success']:
            diff = torch.abs(output_sdpa - output_fa).max()
            print(f"   Difference from SDPA: {diff:.6f}")
        
    except ImportError:
        print("⚠️  Flash Attention: Not installed")
        results['flash_attn'] = {'time': 0, 'success': False, 'error': 'Not installed'}
    except Exception as e:
        print(f"❌ Flash Attention: FAILED - {e}")
        results['flash_attn'] = {'time': 0, 'success': False, 'error': str(e)}
    print()
    
    # Test 3: Manual attention implementation
    print("🔧 Testing manual attention implementation...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.cuda.amp.autocast():
            scale = head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(scores, dim=-1)
            output_manual = torch.matmul(attn_weights, v)
        
        torch.cuda.synchronize()
        manual_time = time.time() - start_time
        
        print(f"✅ Manual attention: PASSED ({manual_time:.3f}s)")
        print(f"   Output shape: {output_manual.shape}")
        print(f"   Output range: [{output_manual.min():.3f}, {output_manual.max():.3f}]")
        results['manual'] = {'time': manual_time, 'success': True, 'output': output_manual}
        
        # Compare with SDPA
        if 'sdpa' in results and results['sdpa']['success']:
            diff = torch.abs(output_sdpa - output_manual).max()
            print(f"   Difference from SDPA: {diff:.6f}")
        
    except Exception as e:
        print(f"❌ Manual attention: FAILED - {e}")
        results['manual'] = {'time': 0, 'success': False, 'error': str(e)}
    print()
    
    # Memory usage
    print("📊 GPU Memory Usage:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print()
    
    # Summary
    print("📋 Performance Summary:")
    for method, result in results.items():
        if result['success']:
            print(f"   {method}: {result['time']:.3f}s")
        else:
            print(f"   {method}: FAILED ({result.get('error', 'Unknown error')})")
    
    # Recommendation
    print()
    print("💡 Recommendation for ARM64 Blackwell GB10:")
    if results.get('flash_attn', {}).get('success'):
        print("   ✅ Flash Attention is working - use it for optimal performance")
    elif results.get('sdpa', {}).get('success'):
        print("   ✅ Use PyTorch native SDPA - good performance and compatibility")
    elif results.get('manual', {}).get('success'):
        print("   ⚠️  Use manual attention as fallback - slower but functional")
    else:
        print("   ❌ No attention mechanism working properly")
    
    return results

if __name__ == "__main__":
    test_attention_backends()