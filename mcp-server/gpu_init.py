#!/usr/bin/env python3
"""
GPU Initialization Helper for MCP Server
Provides easy integration of GPU optimizations into server startup and request handlers
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GPUInitializationError(Exception):
    """Raised when GPU initialization fails"""
    pass


class ServerGPUOptimizer:
    """Wrapper for GPU optimization in server context"""
    
    def __init__(self, enabled: bool = True, profile_inference: bool = False):
        """Initialize GPU optimizer for server.
        
        Args:
            enabled: Whether to enable GPU optimizations
            profile_inference: Whether to profile inference times
        """
        self.enabled = enabled
        self.profile_inference = profile_inference
        self.gpu_available = False
        self.diagnostics: Dict[str, Any] = {}
        
        if enabled:
            try:
                self._initialize_gpu()
            except Exception as e:
                logger.warning(f"GPU optimization initialization failed: {e}")
                self.enabled = False

    def _initialize_gpu(self):
        """Initialize GPU and perform diagnostics."""
        try:
            # Import GPU optimizer - try multiple paths for zero-touch compatibility
            gpu_optimizer = self._import_gpu_optimizer()
            if gpu_optimizer is None:
                logger.warning("GPU optimizer module not available")
                return
            
            optimizer = gpu_optimizer.get_gpu_optimizer()
            self.gpu_available, self.diagnostics = optimizer.validate_gpu_availability()
            
            if self.gpu_available:
                # Setup optimal configuration
                optimizer.setup_optimal_gpu_config()
                logger.info("GPU optimizations initialized successfully")
                logger.info(f"GPU count: {self.diagnostics.get('gpu_count', 0)}")
                logger.info(f"JAX backend: {self.diagnostics.get('jax_backend', 'unknown')}")
            else:
                logger.warning("GPU not available, using CPU")
                
        except Exception as e:
            logger.error(f"GPU initialization error: {e}")
            raise GPUInitializationError(str(e))

    def _import_gpu_optimizer(self):
        """Try to import gpu_optimizer from multiple paths for zero-touch compatibility."""
        import importlib.util
        
        # Path 1: From alphafold module (if installed in conda env)
        try:
            from alphafold.model import gpu_optimizer
            return gpu_optimizer
        except ImportError:
            pass
        
        # Path 2: From project tools directory (relative)
        mcp_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(mcp_dir)
        gpu_opt_path = os.path.join(project_root, 'tools', 'alphafold2', 'gpu_optimizer.py')
        
        if os.path.exists(gpu_opt_path):
            spec = importlib.util.spec_from_file_location("gpu_optimizer", gpu_opt_path)
            if spec and spec.loader:
                gpu_optimizer = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gpu_optimizer)
                return gpu_optimizer
        
        # Path 3: Added to sys.path during conda environment activation
        try:
            import gpu_optimizer
            return gpu_optimizer
        except ImportError:
            pass
        
        return None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get GPU diagnostics."""
        return {
            'gpu_available': self.gpu_available,
            'gpu_optimizations_enabled': self.enabled,
            **self.diagnostics
        }

    @contextmanager
    def profile_inference(self, operation_name: str = "inference"):
        """Context manager for profiling inference operations.
        
        Usage:
            with optimizer.profile_inference("predict_structure"):
                # Run inference
                result = model.predict(...)
        """
        if not self.profile_inference:
            yield
            return
        
        import time
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            logger.info(f"{operation_name} completed in {elapsed:.2f}s")


def setup_gpu_for_server(app=None, profile_inference: bool = False) -> ServerGPUOptimizer:
    """Setup GPU optimizations for FastAPI/server application.
    
    Can be used as:
    1. Direct: optimizer = setup_gpu_for_server()
    2. With app: optimizer = setup_gpu_for_server(app)
    
    Args:
        app: FastAPI application (optional)
        profile_inference: Whether to profile inference operations
    
    Returns:
        ServerGPUOptimizer instance for use in handlers
    """
    # Check if GPU optimization is enabled via environment
    enabled = os.getenv('ENABLE_GPU_OPTIMIZATION', 'true').lower() in {'true', '1', 'yes'}
    
    optimizer = ServerGPUOptimizer(
        enabled=enabled,
        profile_inference=profile_inference
    )
    
    # Optional: add GPU diagnostics endpoint if FastAPI app provided
    if app and hasattr(app, 'get'):
        @app.get('/api/gpu/status')
        async def gpu_status():
            """Get GPU optimization status"""
            return optimizer.get_diagnostics()
        
        logger.info("Added /api/gpu/status endpoint")
    
    return optimizer


# Global instance for server
_gpu_optimizer: Optional[ServerGPUOptimizer] = None


def get_server_gpu_optimizer() -> ServerGPUOptimizer:
    """Get or create global GPU optimizer instance."""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = setup_gpu_for_server()
    return _gpu_optimizer
