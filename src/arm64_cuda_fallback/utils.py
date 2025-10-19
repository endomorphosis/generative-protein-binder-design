"""
Utility functions for ARM64 CUDA fallback module.
"""

from typing import Optional, Dict, Any
import platform


def get_optimal_device(framework: str = 'auto', force_cpu: bool = False, verbose: bool = True) -> Any:
    """
    Get the optimal device for computation based on available framework.
    
    Args:
        framework: 'pytorch', 'jax', or 'auto' to detect automatically
        force_cpu: Force CPU usage even if GPU is available
        verbose: Print information about device selection
        
    Returns:
        Device object appropriate for the framework
    """
    is_arm64 = platform.machine().lower() in ('aarch64', 'arm64')
    
    if framework == 'auto':
        # Try PyTorch first, then JAX
        try:
            import torch
            framework = 'pytorch'
        except ImportError:
            try:
                import jax
                framework = 'jax'
            except ImportError:
                raise RuntimeError(
                    "Neither PyTorch nor JAX is available. "
                    "Please install at least one of them."
                )
    
    if framework == 'pytorch':
        from .pytorch_fallback import get_pytorch_device
        return get_pytorch_device(force_cpu=force_cpu, verbose=verbose)
    
    elif framework == 'jax':
        from .jax_fallback import JAXFallback
        fallback = JAXFallback(force_cpu=force_cpu, verbose=verbose)
        return fallback.get_default_device()
    
    else:
        raise ValueError(f"Unknown framework: {framework}. Use 'pytorch', 'jax', or 'auto'")


def format_device_info(info: Dict[str, Any]) -> str:
    """
    Format device information dictionary as a readable string.
    
    Args:
        info: Device information dictionary
        
    Returns:
        Formatted string
    """
    lines = ["Device Information:"]
    lines.append("=" * 60)
    
    for key, value in info.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}: {', '.join(str(v) for v in value)}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def check_upstream_support() -> Dict[str, bool]:
    """
    Check if upstream ARM64 CUDA support is available.
    
    This function checks if PyTorch and JAX have native ARM64 CUDA support,
    which would make this fallback module obsolete.
    
    Returns:
        Dictionary with support status for each framework
    """
    is_arm64 = platform.machine().lower() in ('aarch64', 'arm64')
    
    result = {
        "is_arm64": is_arm64,
        "pytorch_arm64_cuda": False,
        "jax_arm64_cuda": False,
        "fallback_needed": True,
    }
    
    if not is_arm64:
        result["fallback_needed"] = False
        return result
    
    # Check PyTorch ARM64 CUDA support
    try:
        import torch
        if torch.cuda.is_available():
            result["pytorch_arm64_cuda"] = True
    except ImportError:
        pass
    
    # Check JAX ARM64 GPU support
    try:
        import jax
        devices = jax.devices()
        if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
            result["jax_arm64_cuda"] = True
    except ImportError:
        pass
    
    # Determine if fallback is still needed
    result["fallback_needed"] = not (
        result["pytorch_arm64_cuda"] or result["jax_arm64_cuda"]
    )
    
    return result


def should_deprecate() -> bool:
    """
    Check if this module should be deprecated.
    
    Returns:
        True if both PyTorch and JAX have ARM64 CUDA support, False otherwise
    """
    support = check_upstream_support()
    
    # Module should be deprecated if we're not on ARM64 or if support is available
    return (
        not support["is_arm64"] or
        (support["pytorch_arm64_cuda"] and support["jax_arm64_cuda"])
    )


def get_migration_guide() -> str:
    """
    Get migration guide for when upstream support becomes available.
    
    Returns:
        String with migration instructions
    """
    support = check_upstream_support()
    
    if should_deprecate():
        return """
=============================================================================
MIGRATION GUIDE: ARM64 CUDA Fallback Module
=============================================================================

Good news! Native ARM64 CUDA support is now available.
This fallback module is no longer needed.

Migration Steps:
1. Remove imports from arm64_cuda_fallback module
2. Use standard PyTorch/JAX device handling:
   
   For PyTorch:
   ```python
   import torch
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   ```
   
   For JAX:
   ```python
   import jax
   devices = jax.devices()  # Will include GPU if available
   ```

3. Remove arm64_cuda_fallback from your dependencies
4. Test your code with native CUDA support
5. Delete the arm64_cuda_fallback module

=============================================================================
"""
    else:
        status_lines = []
        if not support["pytorch_arm64_cuda"]:
            status_lines.append("- PyTorch ARM64 CUDA: Not yet available")
        else:
            status_lines.append("- PyTorch ARM64 CUDA: ✓ Available")
            
        if not support["jax_arm64_cuda"]:
            status_lines.append("- JAX ARM64 GPU: Not yet available")
        else:
            status_lines.append("- JAX ARM64 GPU: ✓ Available")
        
        return f"""
=============================================================================
ARM64 CUDA Fallback Module Status
=============================================================================

This module is still needed. Upstream support status:

{chr(10).join(status_lines)}

Continue using the fallback module until both frameworks have full support.
Check periodically for updates to PyTorch and JAX.

To check status programmatically:
```python
from arm64_cuda_fallback.utils import check_upstream_support
status = check_upstream_support()
print(status)
```

=============================================================================
"""


def configure_environment_for_cpu():
    """
    Configure environment variables for optimal CPU performance.
    
    Sets various environment variables that can improve CPU performance
    when GPU is not available.
    """
    import os
    
    # Get CPU count
    num_cpus = os.cpu_count() or 4
    
    # Configure threading for various libraries
    env_vars = {
        'OMP_NUM_THREADS': str(num_cpus),
        'MKL_NUM_THREADS': str(num_cpus),
        'OPENBLAS_NUM_THREADS': str(num_cpus),
        'VECLIB_MAXIMUM_THREADS': str(num_cpus),
        'NUMEXPR_NUM_THREADS': str(num_cpus),
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    print(f"Configured environment for CPU with {num_cpus} threads")


def print_performance_tips(framework: str = 'auto'):
    """
    Print performance optimization tips for CPU execution.
    
    Args:
        framework: 'pytorch', 'jax', or 'auto'
    """
    print("=" * 70)
    print("ARM64 CPU Performance Tips")
    print("=" * 70)
    print()
    print("Since GPU acceleration is not available, here are some tips to")
    print("improve CPU performance:")
    print()
    print("1. Use optimized CPU builds:")
    print("   - For PyTorch: Install with MKL or OpenBLAS support")
    print("   - For JAX: Use jaxlib built with optimized BLAS")
    print()
    print("2. Reduce batch sizes:")
    print("   - Smaller batches = less memory = faster on CPU")
    print()
    print("3. Use model quantization:")
    print("   - Convert models to int8 or float16 when possible")
    print()
    print("4. Enable CPU threading:")
    print("   - Run configure_environment_for_cpu() to set optimal threads")
    print()
    print("5. Consider using smaller models:")
    print("   - Distilled or compressed versions run faster on CPU")
    print()
    print("6. Use cloud GPU instances for production:")
    print("   - For heavy workloads, cloud GPUs may be more cost-effective")
    print()
    print("=" * 70)
