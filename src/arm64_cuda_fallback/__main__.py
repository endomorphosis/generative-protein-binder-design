#!/usr/bin/env python3
"""
ARM64 CUDA Fallback CLI Tool

Command-line interface for detecting CUDA availability and managing fallbacks.
"""

import sys
import argparse
from typing import Optional

from . import get_fallback_config, print_deprecation_notice
from .detector import CUDADetector
from .pytorch_fallback import PyTorchFallback, check_pytorch_cuda
from .jax_fallback import JAXFallback, check_jax_gpu
from .utils import (
    check_upstream_support,
    should_deprecate,
    get_migration_guide,
    configure_environment_for_cpu,
    print_performance_tips,
    format_device_info,
)


def cmd_detect(args):
    """Detect CUDA availability and print information."""
    detector = CUDADetector()
    device_info = detector.detect()
    
    print(device_info)
    print()
    print("Recommendation:")
    print(detector.get_recommendation())
    
    if args.verbose:
        print()
        print("Fallback Configuration:")
        config = get_fallback_config()
        for key, value in config.items():
            print(f"  {key}: {value}")


def cmd_pytorch(args):
    """Check PyTorch CUDA status and get device."""
    fallback = PyTorchFallback(force_cpu=args.force_cpu, verbose=True)
    
    if not fallback.available:
        print("ERROR: PyTorch is not installed")
        return 1
    
    print("\nPyTorch Device Information:")
    info = fallback.get_device_info()
    print(format_device_info(info))
    
    if args.configure_cpu:
        print()
        fallback.configure_for_cpu()
    
    return 0


def cmd_jax(args):
    """Check JAX GPU status and get devices."""
    fallback = JAXFallback(force_cpu=args.force_cpu, verbose=True)
    
    if not fallback.available:
        print("ERROR: JAX is not installed")
        return 1
    
    print("\nJAX Device Information:")
    info = fallback.get_device_info()
    print(format_device_info(info))
    
    if args.configure_cpu:
        print()
        fallback.configure_for_cpu()
    
    return 0


def cmd_check_upstream(args):
    """Check if upstream ARM64 CUDA support is available."""
    support = check_upstream_support()
    
    print("Upstream ARM64 CUDA Support Status:")
    print("=" * 60)
    print(f"Architecture: {'ARM64' if support['is_arm64'] else 'Not ARM64'}")
    print(f"PyTorch ARM64 CUDA: {'✓ Available' if support['pytorch_arm64_cuda'] else '✗ Not available'}")
    print(f"JAX ARM64 GPU: {'✓ Available' if support['jax_arm64_cuda'] else '✗ Not available'}")
    print(f"Fallback Needed: {'Yes' if support['fallback_needed'] else 'No'}")
    print("=" * 60)
    
    if should_deprecate():
        print("\n⚠️  This module can be deprecated!")
        if args.verbose:
            print(get_migration_guide())
    else:
        print("\nℹ️  Fallback module is still needed")
        if args.verbose:
            print(get_migration_guide())


def cmd_configure_cpu(args):
    """Configure environment for optimal CPU performance."""
    configure_environment_for_cpu()
    
    if args.tips:
        print()
        print_performance_tips()


def cmd_migration(args):
    """Show migration guide."""
    print(get_migration_guide())


def cmd_deprecation(args):
    """Show deprecation notice."""
    print_deprecation_notice()


def cmd_info(args):
    """Show comprehensive information about the fallback module."""
    print()
    print_deprecation_notice()
    print()
    
    # Detect system
    detector = CUDADetector()
    device_info = detector.detect()
    print("System Information:")
    print(device_info)
    print()
    
    # Check frameworks
    print("Framework Status:")
    print("=" * 60)
    
    # PyTorch
    try:
        pytorch_fallback = PyTorchFallback(verbose=False)
        if pytorch_fallback.available:
            print(f"✓ PyTorch {pytorch_fallback.torch.__version__} installed")
            print(f"  CUDA Available: {pytorch_fallback.is_cuda_available()}")
        else:
            print("✗ PyTorch not installed")
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
    
    # JAX
    try:
        jax_fallback = JAXFallback(verbose=False)
        if jax_fallback.available:
            print(f"✓ JAX {jax_fallback.jax.__version__} installed")
            print(f"  GPU Available: {jax_fallback.is_gpu_available()}")
        else:
            print("✗ JAX not installed")
    except Exception as e:
        print(f"✗ JAX check failed: {e}")
    
    print("=" * 60)
    print()
    
    # Check upstream
    support = check_upstream_support()
    print("Upstream Support:")
    print("=" * 60)
    print(f"PyTorch ARM64 CUDA: {'✓' if support['pytorch_arm64_cuda'] else '✗'}")
    print(f"JAX ARM64 GPU: {'✓' if support['jax_arm64_cuda'] else '✗'}")
    print(f"Fallback Needed: {'Yes' if support['fallback_needed'] else 'No'}")
    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ARM64 CUDA Fallback Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect CUDA availability
  python -m arm64_cuda_fallback detect
  
  # Check PyTorch device
  python -m arm64_cuda_fallback pytorch
  
  # Check JAX devices
  python -m arm64_cuda_fallback jax
  
  # Check upstream support status
  python -m arm64_cuda_fallback check-upstream
  
  # Show complete information
  python -m arm64_cuda_fallback info
  
  # Configure environment for CPU
  python -m arm64_cuda_fallback configure-cpu --tips
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # detect command
    detect_parser = subparsers.add_parser(
        'detect',
        help='Detect CUDA availability'
    )
    detect_parser.set_defaults(func=cmd_detect)
    
    # pytorch command
    pytorch_parser = subparsers.add_parser(
        'pytorch',
        help='Check PyTorch CUDA status'
    )
    pytorch_parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage'
    )
    pytorch_parser.add_argument(
        '--configure-cpu',
        action='store_true',
        help='Configure for optimal CPU performance'
    )
    pytorch_parser.set_defaults(func=cmd_pytorch)
    
    # jax command
    jax_parser = subparsers.add_parser(
        'jax',
        help='Check JAX GPU status'
    )
    jax_parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage'
    )
    jax_parser.add_argument(
        '--configure-cpu',
        action='store_true',
        help='Configure for optimal CPU performance'
    )
    jax_parser.set_defaults(func=cmd_jax)
    
    # check-upstream command
    upstream_parser = subparsers.add_parser(
        'check-upstream',
        help='Check upstream ARM64 CUDA support'
    )
    upstream_parser.set_defaults(func=cmd_check_upstream)
    
    # configure-cpu command
    cpu_parser = subparsers.add_parser(
        'configure-cpu',
        help='Configure environment for CPU'
    )
    cpu_parser.add_argument(
        '--tips',
        action='store_true',
        help='Show performance tips'
    )
    cpu_parser.set_defaults(func=cmd_configure_cpu)
    
    # migration command
    migration_parser = subparsers.add_parser(
        'migration',
        help='Show migration guide'
    )
    migration_parser.set_defaults(func=cmd_migration)
    
    # deprecation command
    deprecation_parser = subparsers.add_parser(
        'deprecation',
        help='Show deprecation notice'
    )
    deprecation_parser.set_defaults(func=cmd_deprecation)
    
    # info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show comprehensive information'
    )
    info_parser.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        return args.func(args) or 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
