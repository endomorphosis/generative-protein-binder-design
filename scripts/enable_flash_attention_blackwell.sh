#!/bin/bash

# Enable Flash Attention for ARM64 Blackwell GB10 GPU
# Multiple approaches to get Flash Attention working on sm121 architecture

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Flash Attention Enablement for ARM64 Blackwell GB10"
echo "=================================================="

# Check GPU architecture
echo "📋 GPU Information:"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits || true
echo ""

# Function to build Flash Attention from source with Blackwell support
build_flash_attention_source() {
    echo "🔨 Building Flash Attention from source with Blackwell support..."
    
    cd /tmp
    
    # Clean previous builds
    rm -rf flash-attention || true
    
    # Clone Flash Attention repository
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    
    # Get latest stable version
    git checkout v2.5.9
    
    echo "🔧 Configuring build for Blackwell architecture..."
    
    # Set environment variables for Blackwell support
    export TORCH_CUDA_ARCH_LIST="8.6;9.0;12.1"
    export CUDA_HOME=/usr/local/cuda
    export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/gcc'
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
    
    # Install build dependencies
    pip install ninja packaging wheel setuptools
    
    # Build and install
    echo "⚙️  Building Flash Attention (this will take 10-15 minutes)..."
    python setup.py bdist_wheel
    
    # Install the wheel
    pip install dist/flash_attn-*.whl --force-reinstall
    
    echo "✅ Flash Attention built and installed from source"
    
    cd "$PROJECT_ROOT"
}

# Function to use PyTorch's native SDPA with proper backend selection
enable_native_sdpa() {
    echo "🔧 Configuring PyTorch native Scaled Dot Product Attention..."
    
    # Create configuration script
    cat > /tmp/configure_sdpa.py << 'EOF'
import torch
import torch.nn.functional as F

print("🔍 Checking PyTorch SDPA backends...")

# Check available backends
backends = []
if hasattr(torch.backends.cuda, 'sdp_kernel'):
    if torch.backends.cuda.sdp_kernel.can_use_flash():
        backends.append("Flash Attention")
    if torch.backends.cuda.sdp_kernel.can_use_efficient():
        backends.append("Memory Efficient")
    if torch.backends.cuda.sdp_kernel.can_use_math():
        backends.append("Math (fallback)")

print(f"Available SDPA backends: {', '.join(backends)}")

# Test with different backends
if torch.cuda.is_available():
    device = torch.device('cuda')
    batch_size, seq_len, embed_dim, num_heads = 2, 1024, 512, 8
    head_dim = embed_dim // num_heads
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    print("\n🧪 Testing SDPA backends:")
    
    # Test Math backend (always available)
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            out_math = F.scaled_dot_product_attention(q, k, v)
        print("✅ Math backend: WORKING")
    except Exception as e:
        print(f"❌ Math backend: {e}")
    
    # Test Memory Efficient backend
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False):
            out_efficient = F.scaled_dot_product_attention(q, k, v)
        print("✅ Memory Efficient backend: WORKING")
    except Exception as e:
        print(f"❌ Memory Efficient backend: {e}")
    
    # Test Flash Attention backend
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
            out_flash = F.scaled_dot_product_attention(q, k, v)
        print("✅ Flash Attention backend: WORKING")
    except Exception as e:
        print(f"❌ Flash Attention backend: {e}")
        
    print(f"\n📊 Output tensor shape: {out_math.shape}")
    print(f"📊 Output tensor device: {out_math.device}")
    print(f"📊 Output tensor dtype: {out_math.dtype}")

else:
    print("❌ CUDA not available")
EOF
    
    python3 /tmp/configure_sdpa.py
}

# Function to install pre-built Flash Attention wheels
install_prebuilt_wheels() {
    echo "📦 Installing pre-built Flash Attention wheels..."
    
    # Try official wheels first
    pip install flash-attn --no-build-isolation --force-reinstall || {
        echo "⚠️  Official wheels failed, trying alternative sources..."
        
        # Try installing with specific CUDA version
        pip install flash-attn --index-url https://download.pytorch.org/whl/cu121 --force-reinstall || {
            echo "⚠️  Pre-built wheels not available for this configuration"
            return 1
        }
    }
}

# Function to create Flash Attention wrapper that handles architecture fallbacks
create_flash_attention_wrapper() {
    echo "🔧 Creating Flash Attention wrapper with architecture fallbacks..."
    
    cat > "$PROJECT_ROOT/src/flash_attention_wrapper.py" << 'EOF'
"""
Flash Attention wrapper with Blackwell GB10 support and graceful fallbacks
"""
import torch
import torch.nn.functional as F
import warnings
from typing import Optional, Tuple

class FlashAttentionWrapper:
    """Wrapper for Flash Attention with multiple backend support"""
    
    def __init__(self):
        self.flash_available = False
        self.backend_priority = ['flash', 'efficient', 'math']
        self.detected_backend = None
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize and detect available backends"""
        if not torch.cuda.is_available():
            print("❌ CUDA not available, using CPU fallback")
            return
            
        # Test Flash Attention availability
        try:
            # Create small test tensors
            q = torch.randn(1, 1, 64, 64, device='cuda', dtype=torch.float16)
            k = torch.randn(1, 1, 64, 64, device='cuda', dtype=torch.float16)
            v = torch.randn(1, 1, 64, 64, device='cuda', dtype=torch.float16)
            
            # Test backends in order of preference
            for backend in self.backend_priority:
                try:
                    if backend == 'flash':
                        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                            _ = F.scaled_dot_product_attention(q, k, v)
                        self.detected_backend = 'flash'
                        self.flash_available = True
                        print("✅ Flash Attention backend available")
                        break
                    elif backend == 'efficient':
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False):
                            _ = F.scaled_dot_product_attention(q, k, v)
                        self.detected_backend = 'efficient'
                        print("✅ Memory Efficient backend available")
                        break
                    elif backend == 'math':
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                            _ = F.scaled_dot_product_attention(q, k, v)
                        self.detected_backend = 'math'
                        print("✅ Math backend available (fallback)")
                        break
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"⚠️  Backend detection failed: {e}")
            self.detected_backend = 'math'
    
    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute scaled dot product attention with automatic backend selection
        """
        if not torch.cuda.is_available():
            # CPU fallback
            return self._manual_attention(query, key, value, attn_mask, dropout_p, is_causal, scale)
        
        # Try backends in order of preference
        for backend in self.backend_priority:
            if backend == self.detected_backend:
                try:
                    if backend == 'flash':
                        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                            return F.scaled_dot_product_attention(
                                query, key, value, attn_mask, dropout_p, is_causal, scale
                            )
                    elif backend == 'efficient':
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False):
                            return F.scaled_dot_product_attention(
                                query, key, value, attn_mask, dropout_p, is_causal, scale
                            )
                    elif backend == 'math':
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                            return F.scaled_dot_product_attention(
                                query, key, value, attn_mask, dropout_p, is_causal, scale
                            )
                except Exception as e:
                    warnings.warn(f"Backend {backend} failed: {e}")
                    continue
        
        # Ultimate fallback to manual implementation
        return self._manual_attention(query, key, value, attn_mask, dropout_p, is_causal, scale)
    
    def _manual_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """Manual attention implementation as ultimate fallback"""
        if scale is None:
            scale = 1.0 / (query.size(-1) ** 0.5)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply causal mask if needed
        if is_causal:
            seq_len = query.size(-2)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores += attn_mask
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training if hasattr(self, 'training') else False)
        
        # Apply attention to values
        return torch.matmul(attn_weights, value)

# Global instance
flash_attention = FlashAttentionWrapper()

# Convenience function
def scaled_dot_product_attention(*args, **kwargs):
    """Convenience function that uses the global Flash Attention wrapper"""
    return flash_attention.scaled_dot_product_attention(*args, **kwargs)
EOF

    echo "✅ Flash Attention wrapper created at $PROJECT_ROOT/src/flash_attention_wrapper.py"
}

# Function to test Flash Attention functionality
test_flash_attention() {
    echo "🧪 Testing Flash Attention functionality..."
    
    cat > /tmp/test_flash_attention.py << 'EOF'
import sys
import torch
import torch.nn.functional as F
import time

print("🧪 Flash Attention Comprehensive Test")
print("====================================")

# Test system information
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n🔍 Testing different attention implementations:")

if torch.cuda.is_available():
    device = torch.device('cuda')
    batch_size, num_heads, seq_len, head_dim = 2, 8, 2048, 64
    
    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    print(f"Test tensor shape: {q.shape}")
    print(f"Test tensor dtype: {q.dtype}")
    print(f"Test tensor device: {q.device}")
    
    # Test different backends
    backends = [
        ('Math Backend', {'enable_flash': False, 'enable_mem_efficient': False, 'enable_math': True}),
        ('Memory Efficient', {'enable_flash': False, 'enable_mem_efficient': True, 'enable_math': False}),
        ('Flash Attention', {'enable_flash': True, 'enable_mem_efficient': False, 'enable_math': False}),
        ('Auto (PyTorch Default)', {})
    ]
    
    for name, kwargs in backends:
        try:
            start_time = time.time()
            
            if kwargs:
                with torch.backends.cuda.sdp_kernel(**kwargs):
                    out = F.scaled_dot_product_attention(q, k, v)
            else:
                out = F.scaled_dot_product_attention(q, k, v)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            print(f"✅ {name}: PASSED ({elapsed:.3f}s)")
            print(f"   Output shape: {out.shape}")
            print(f"   Output range: [{out.min():.3f}, {out.max():.3f}]")
            
        except Exception as e:
            print(f"❌ {name}: FAILED - {str(e)[:100]}...")
    
    # Test with wrapper if available
    try:
        sys.path.insert(0, '/home/barberb/generative-protein-binder-design/src')
        from flash_attention_wrapper import scaled_dot_product_attention as wrapper_sdpa
        
        start_time = time.time()
        out_wrapper = wrapper_sdpa(q, k, v)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        print(f"✅ Custom Wrapper: PASSED ({elapsed:.3f}s)")
        print(f"   Output shape: {out_wrapper.shape}")
        
    except ImportError:
        print("⚠️  Custom wrapper not available")
    except Exception as e:
        print(f"❌ Custom Wrapper: FAILED - {e}")

    # Memory usage test
    print(f"\n📊 GPU Memory Usage:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

else:
    print("❌ CUDA not available - cannot test GPU acceleration")

print("\n✅ Flash Attention test completed")
EOF

    python3 /tmp/test_flash_attention.py
}

# Main execution
main() {
    echo "🎯 Choose Flash Attention enablement strategy:"
    echo "1. Build Flash Attention from source (recommended for Blackwell)"
    echo "2. Use PyTorch native SDPA with backend selection"
    echo "3. Install pre-built wheels (may not support sm121)"
    echo "4. Create wrapper with fallbacks"
    echo "5. Test current Flash Attention setup"
    echo "6. All approaches (comprehensive setup)"
    
    read -p "Enter choice (1-6): " choice
    
    case $choice in
        1)
            build_flash_attention_source
            test_flash_attention
            ;;
        2)
            enable_native_sdpa
            ;;
        3)
            install_prebuilt_wheels
            test_flash_attention
            ;;
        4)
            create_flash_attention_wrapper
            test_flash_attention
            ;;
        5)
            test_flash_attention
            ;;
        6)
            echo "🚀 Comprehensive Flash Attention setup..."
            create_flash_attention_wrapper
            enable_native_sdpa
            echo ""
            echo "🔨 Attempting source build..."
            build_flash_attention_source || {
                echo "⚠️  Source build failed, trying pre-built wheels..."
                install_prebuilt_wheels || echo "⚠️  Pre-built wheels also failed"
            }
            test_flash_attention
            ;;
        *)
            echo "❌ Invalid choice"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"