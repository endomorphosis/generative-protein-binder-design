"""
Tests for ARM64 CUDA Fallback Module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import platform


class TestCUDADetector(unittest.TestCase):
    """Test CUDA detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from arm64_cuda_fallback.detector import CUDADetector
        self.detector = CUDADetector()
    
    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.architecture)
    
    def test_detect_returns_device_info(self):
        """Test detect() returns DeviceInfo object."""
        device_info = self.detector.detect()
        self.assertIsNotNone(device_info)
        self.assertIsNotNone(device_info.architecture)
        self.assertIsInstance(device_info.cuda_available, bool)
        self.assertIsInstance(device_info.gpu_count, int)
    
    def test_get_recommendation_returns_string(self):
        """Test get_recommendation() returns a string."""
        recommendation = self.detector.get_recommendation()
        self.assertIsInstance(recommendation, str)
        self.assertTrue(len(recommendation) > 0)


class TestPyTorchFallback(unittest.TestCase):
    """Test PyTorch fallback functionality."""
    
    def test_fallback_initialization(self):
        """Test PyTorch fallback initializes correctly."""
        from arm64_cuda_fallback.pytorch_fallback import PyTorchFallback
        fallback = PyTorchFallback(verbose=False)
        self.assertIsNotNone(fallback)
    
    def test_get_device_info_returns_dict(self):
        """Test get_device_info() returns a dictionary."""
        from arm64_cuda_fallback.pytorch_fallback import PyTorchFallback
        fallback = PyTorchFallback(verbose=False)
        
        info = fallback.get_device_info()
        self.assertIsInstance(info, dict)
        self.assertIn('available', info)
    
    def test_force_cpu_mode(self):
        """Test forcing CPU mode."""
        from arm64_cuda_fallback.pytorch_fallback import PyTorchFallback
        fallback = PyTorchFallback(force_cpu=True, verbose=False)
        
        if fallback.available:
            device = fallback.get_device()
            self.assertEqual(str(device), 'cpu')
    
    def test_check_pytorch_cuda(self):
        """Test check_pytorch_cuda utility function."""
        from arm64_cuda_fallback.pytorch_fallback import check_pytorch_cuda
        result = check_pytorch_cuda()
        self.assertIsInstance(result, bool)


class TestJAXFallback(unittest.TestCase):
    """Test JAX fallback functionality."""
    
    def test_fallback_initialization(self):
        """Test JAX fallback initializes correctly."""
        from arm64_cuda_fallback.jax_fallback import JAXFallback
        fallback = JAXFallback(verbose=False)
        self.assertIsNotNone(fallback)
    
    def test_get_device_info_returns_dict(self):
        """Test get_device_info() returns a dictionary."""
        from arm64_cuda_fallback.jax_fallback import JAXFallback
        fallback = JAXFallback(verbose=False)
        
        info = fallback.get_device_info()
        self.assertIsInstance(info, dict)
        self.assertIn('available', info)
    
    def test_check_jax_gpu(self):
        """Test check_jax_gpu utility function."""
        from arm64_cuda_fallback.jax_fallback import check_jax_gpu
        result = check_jax_gpu()
        self.assertIsInstance(result, bool)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_check_upstream_support(self):
        """Test check_upstream_support() returns dict."""
        from arm64_cuda_fallback.utils import check_upstream_support
        support = check_upstream_support()
        
        self.assertIsInstance(support, dict)
        self.assertIn('is_arm64', support)
        self.assertIn('pytorch_arm64_cuda', support)
        self.assertIn('jax_arm64_cuda', support)
        self.assertIn('fallback_needed', support)
    
    def test_should_deprecate(self):
        """Test should_deprecate() returns bool."""
        from arm64_cuda_fallback.utils import should_deprecate
        result = should_deprecate()
        self.assertIsInstance(result, bool)
    
    def test_get_migration_guide(self):
        """Test get_migration_guide() returns string."""
        from arm64_cuda_fallback.utils import get_migration_guide
        guide = get_migration_guide()
        self.assertIsInstance(guide, str)
        self.assertTrue(len(guide) > 0)
    
    def test_format_device_info(self):
        """Test format_device_info() formats dict correctly."""
        from arm64_cuda_fallback.utils import format_device_info
        
        test_info = {
            'architecture': 'aarch64',
            'cuda_available': False,
            'device_type': 'cpu'
        }
        
        result = format_device_info(test_info)
        self.assertIsInstance(result, str)
        self.assertIn('architecture', result)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_get_fallback_config(self):
        """Test get_fallback_config() returns complete config."""
        from arm64_cuda_fallback import get_fallback_config
        config = get_fallback_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('version', config)
        self.assertIn('architecture', config)
        self.assertIn('cuda_available', config)
        self.assertIn('fallback_active', config)
    
    def test_module_imports(self):
        """Test all module imports work."""
        try:
            from arm64_cuda_fallback import (
                CUDADetector,
                DeviceInfo,
                PyTorchFallback,
                JAXFallback,
                get_optimal_device,
                format_device_info,
            )
            # If we get here, imports worked
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Module import failed: {e}")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
