"""
PyTorch Fallback for ARM64

Provides automatic fallback to CPU when CUDA is not available on ARM64.
"""

import warnings
from typing import Optional, Union, Any
import platform


class PyTorchFallback:
    """
    PyTorch fallback handler for ARM64 systems.
    
    Automatically falls back to CPU when CUDA is not available.
    """
    
    def __init__(self, force_cpu: bool = False, verbose: bool = True):
        """
        Initialize PyTorch fallback.
        
        Args:
            force_cpu: Force CPU usage even if CUDA is available
            verbose: Print information about device selection
        """
        self.force_cpu = force_cpu
        self.verbose = verbose
        self.is_arm64 = platform.machine().lower() in ('aarch64', 'arm64')
        
        # Try to import PyTorch
        try:
            import torch
            self.torch = torch
            self.available = True
        except ImportError:
            self.torch = None
            self.available = False
            if verbose:
                warnings.warn(
                    "PyTorch is not installed. Please install with: "
                    "pip install torch",
                    UserWarning
                )
    
    def get_device(self, device_id: int = 0) -> Any:
        """
        Get the appropriate device for computation.
        
        Args:
            device_id: GPU device ID (ignored if falling back to CPU)
            
        Returns:
            torch.device object
        """
        if not self.available:
            raise RuntimeError("PyTorch is not available")
        
        if self.force_cpu:
            if self.verbose:
                print("Forcing CPU usage as requested")
            return self.torch.device('cpu')
        
        # Check CUDA availability
        if self.torch.cuda.is_available():
            if self.is_arm64 and self.verbose:
                print(f"ARM64 CUDA detected - using GPU {device_id}")
            elif self.verbose:
                print(f"Using GPU {device_id}")
            return self.torch.device(f'cuda:{device_id}')
        else:
            if self.verbose:
                if self.is_arm64:
                    print("ARM64 CUDA not available - falling back to CPU")
                    print("NOTE: This is expected on most ARM64 systems.")
                    print("      For GPU acceleration, wait for upstream PyTorch ARM64 CUDA support.")
                else:
                    print("CUDA not available - using CPU")
            return self.torch.device('cpu')
    
    def move_to_device(self, tensor: Any, device: Optional[Any] = None) -> Any:
        """
        Move a tensor to the appropriate device.
        
        Args:
            tensor: PyTorch tensor to move
            device: Target device (uses auto-detection if None)
            
        Returns:
            Tensor on the target device
        """
        if not self.available:
            raise RuntimeError("PyTorch is not available")
        
        if device is None:
            device = self.get_device()
        
        return tensor.to(device)
    
    def is_cuda_available(self) -> bool:
        """
        Check if CUDA is available.
        
        Returns:
            True if CUDA is available, False otherwise
        """
        if not self.available:
            return False
        return self.torch.cuda.is_available()
    
    def get_device_name(self, device_id: int = 0) -> str:
        """
        Get the name of the device.
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Device name string
        """
        if not self.available:
            return "PyTorch not available"
        
        if self.is_cuda_available():
            try:
                return self.torch.cuda.get_device_name(device_id)
            except Exception:
                return "CPU (fallback)"
        return "CPU"
    
    def get_device_info(self) -> dict:
        """
        Get comprehensive device information.
        
        Returns:
            Dictionary with device information
        """
        if not self.available:
            return {
                "available": False,
                "error": "PyTorch not installed"
            }
        
        info = {
            "available": True,
            "version": self.torch.__version__,
            "cuda_available": self.is_cuda_available(),
            "architecture": platform.machine(),
            "is_arm64": self.is_arm64,
        }
        
        if info["cuda_available"]:
            info["cuda_version"] = self.torch.version.cuda
            info["device_count"] = self.torch.cuda.device_count()
            info["devices"] = [
                self.torch.cuda.get_device_name(i)
                for i in range(self.torch.cuda.device_count())
            ]
            info["current_device"] = self.torch.cuda.current_device()
        else:
            info["fallback_reason"] = (
                "ARM64 CUDA support not yet available in PyTorch"
                if self.is_arm64
                else "CUDA not available"
            )
        
        return info
    
    def configure_for_cpu(self):
        """
        Configure PyTorch for optimal CPU performance.
        
        Sets recommended CPU threading settings.
        """
        if not self.available:
            raise RuntimeError("PyTorch is not available")
        
        import os
        
        # Set number of threads for CPU operations
        num_threads = os.cpu_count() or 4
        self.torch.set_num_threads(num_threads)
        
        if self.verbose:
            print(f"Configured PyTorch for CPU with {num_threads} threads")
    
    def create_model_wrapper(self, model: Any, auto_device: bool = True):
        """
        Wrap a model to automatically handle device placement.
        
        Args:
            model: PyTorch model
            auto_device: Automatically move model to optimal device
            
        Returns:
            Model on the appropriate device
        """
        if not self.available:
            raise RuntimeError("PyTorch is not available")
        
        if auto_device:
            device = self.get_device()
            model = model.to(device)
            
            if self.verbose:
                print(f"Model moved to {device}")
        
        return model


def get_pytorch_device(force_cpu: bool = False, verbose: bool = True) -> Any:
    """
    Convenience function to get PyTorch device with fallback.
    
    Args:
        force_cpu: Force CPU usage
        verbose: Print device information
        
    Returns:
        torch.device object
    """
    fallback = PyTorchFallback(force_cpu=force_cpu, verbose=verbose)
    return fallback.get_device()


def check_pytorch_cuda() -> bool:
    """
    Check if PyTorch CUDA is available.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
