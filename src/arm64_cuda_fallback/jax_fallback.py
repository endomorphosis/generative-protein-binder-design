"""
JAX Fallback for ARM64

Provides automatic fallback to CPU when CUDA is not available on ARM64 with JAX.
"""

import warnings
from typing import Optional, List, Any
import platform


class JAXFallback:
    """
    JAX fallback handler for ARM64 systems.
    
    Automatically falls back to CPU when GPU is not available.
    """
    
    def __init__(self, force_cpu: bool = False, verbose: bool = True):
        """
        Initialize JAX fallback.
        
        Args:
            force_cpu: Force CPU usage even if GPU is available
            verbose: Print information about device selection
        """
        self.force_cpu = force_cpu
        self.verbose = verbose
        self.is_arm64 = platform.machine().lower() in ('aarch64', 'arm64')
        
        # Try to import JAX
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
            self.available = True
            
            # Configure JAX for CPU if needed
            if force_cpu or (self.is_arm64 and not self._check_gpu()):
                import os
                os.environ['JAX_PLATFORM_NAME'] = 'cpu'
                if verbose:
                    if self.is_arm64:
                        print("JAX configured for CPU (ARM64 GPU support not yet available)")
                    else:
                        print("JAX configured for CPU")
                    
        except ImportError:
            self.jax = None
            self.jnp = None
            self.available = False
            if verbose:
                warnings.warn(
                    "JAX is not installed. Please install with: "
                    "pip install jax jaxlib",
                    UserWarning
                )
    
    def _check_gpu(self) -> bool:
        """
        Internal method to check GPU availability.
        
        Returns:
            True if GPU is available, False otherwise
        """
        if not self.available:
            return False
        
        try:
            devices = self.jax.devices()
            return any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
        except Exception:
            return False
    
    def get_devices(self) -> List[Any]:
        """
        Get available JAX devices.
        
        Returns:
            List of JAX device objects
        """
        if not self.available:
            raise RuntimeError("JAX is not available")
        
        devices = self.jax.devices()
        
        if self.verbose:
            print(f"Available JAX devices: {devices}")
            if self.is_arm64 and all('cpu' in str(d).lower() for d in devices):
                print("NOTE: Running on CPU. ARM64 GPU support in JAX is not yet available.")
        
        return devices
    
    def get_default_device(self) -> Any:
        """
        Get the default device for computation.
        
        Returns:
            JAX device object
        """
        if not self.available:
            raise RuntimeError("JAX is not available")
        
        devices = self.get_devices()
        return devices[0]  # Return first device (CPU or GPU)
    
    def is_gpu_available(self) -> bool:
        """
        Check if GPU is available in JAX.
        
        Returns:
            True if GPU is available, False otherwise
        """
        if not self.available:
            return False
        return self._check_gpu()
    
    def get_device_info(self) -> dict:
        """
        Get comprehensive device information.
        
        Returns:
            Dictionary with device information
        """
        if not self.available:
            return {
                "available": False,
                "error": "JAX not installed"
            }
        
        devices = self.jax.devices()
        
        info = {
            "available": True,
            "version": self.jax.__version__,
            "gpu_available": self.is_gpu_available(),
            "architecture": platform.machine(),
            "is_arm64": self.is_arm64,
            "devices": [str(d) for d in devices],
            "default_device": str(devices[0]),
            "device_count": len(devices),
        }
        
        if not info["gpu_available"] and self.is_arm64:
            info["fallback_reason"] = (
                "ARM64 GPU support not yet available in JAX. "
                "Using CPU backend."
            )
        
        return info
    
    def configure_for_cpu(self):
        """
        Configure JAX for optimal CPU performance.
        
        Sets recommended CPU settings.
        """
        if not self.available:
            raise RuntimeError("JAX is not available")
        
        import os
        
        # Force CPU platform
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        
        # Set threading
        num_threads = os.cpu_count() or 4
        os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_threads}'
        
        if self.verbose:
            print(f"Configured JAX for CPU with {num_threads} virtual devices")
    
    def create_array_on_device(self, data: Any, device: Optional[Any] = None) -> Any:
        """
        Create a JAX array on a specific device.
        
        Args:
            data: Input data (numpy array, list, etc.)
            device: Target device (uses default if None)
            
        Returns:
            JAX array on the target device
        """
        if not self.available:
            raise RuntimeError("JAX is not available")
        
        if device is None:
            device = self.get_default_device()
        
        arr = self.jnp.array(data)
        return self.jax.device_put(arr, device)
    
    def check_array_device(self, array: Any) -> str:
        """
        Check which device an array is on.
        
        Args:
            array: JAX array
            
        Returns:
            Device name as string
        """
        if not self.available:
            raise RuntimeError("JAX is not available")
        
        try:
            devices = array.devices()
            return str(devices)
        except AttributeError:
            return "unknown"


def get_jax_devices(force_cpu: bool = False, verbose: bool = True) -> List[Any]:
    """
    Convenience function to get JAX devices with fallback.
    
    Args:
        force_cpu: Force CPU usage
        verbose: Print device information
        
    Returns:
        List of JAX device objects
    """
    fallback = JAXFallback(force_cpu=force_cpu, verbose=verbose)
    return fallback.get_devices()


def check_jax_gpu() -> bool:
    """
    Check if JAX GPU is available.
    
    Returns:
        True if GPU is available, False otherwise
    """
    try:
        import jax
        devices = jax.devices()
        return any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
    except ImportError:
        return False
