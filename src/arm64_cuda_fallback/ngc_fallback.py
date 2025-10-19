"""
NVIDIA NGC Containers Fallback for ARM64

This module provides support for using NVIDIA NGC containers as a fallback
solution when native ARM64 CUDA is not available. NGC containers can run
via emulation or on AMD64 cloud instances.

DEPRECATION NOTICE:
This fallback will be deprecated when native ARM64 CUDA support is available.
"""

import os
import subprocess
import warnings
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class NGCContainerConfig:
    """Configuration for NGC container deployment."""
    name: str
    image: str
    gpu_required: bool
    platform: str  # 'linux/amd64' or 'linux/arm64'
    volumes: List[str]
    environment: Dict[str, str]
    command: Optional[str] = None


class NGCFallback:
    """
    NVIDIA NGC Containers fallback handler.
    
    Provides methods to use NGC containers as a fallback when native
    ARM64 CUDA is not available. Can run via:
    1. Docker emulation (slower)
    2. Cloud AMD64 instances (faster)
    """
    
    def __init__(self, use_emulation: bool = True, verbose: bool = True):
        """
        Initialize NGC fallback handler.
        
        Args:
            use_emulation: Use Docker emulation on ARM64 (vs recommending cloud)
            verbose: Print status messages
        """
        self.use_emulation = use_emulation
        self.verbose = verbose
        self.ngc_api_key = os.environ.get('NGC_CLI_API_KEY')
        
        if not self.ngc_api_key and verbose:
            warnings.warn(
                "NGC_CLI_API_KEY not set. You'll need this to pull NGC containers. "
                "Get your API key from: https://catalog.ngc.nvidia.com/",
                UserWarning
            )
    
    def check_docker_available(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def check_ngc_login(self) -> bool:
        """Check if logged into NGC registry."""
        if not self.ngc_api_key:
            return False
        
        try:
            result = subprocess.run(
                ['docker', 'login', 'nvcr.io', '--username=$oauthtoken', 
                 f'--password={self.ngc_api_key}', '--password-stdin'],
                capture_output=True,
                timeout=10,
                input=self.ngc_api_key.encode()
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_alphafold_config(self, data_dir: str, output_dir: str) -> NGCContainerConfig:
        """
        Get NGC container configuration for AlphaFold2.
        
        Args:
            data_dir: Path to AlphaFold2 databases
            output_dir: Output directory for results
            
        Returns:
            NGCContainerConfig for AlphaFold2
        """
        return NGCContainerConfig(
            name='alphafold2-nim',
            image='nvcr.io/nvidia/clara/alphafold2:latest',
            gpu_required=True,
            platform='linux/amd64',
            volumes=[
                f'{data_dir}:/data',
                f'{output_dir}:/output',
            ],
            environment={
                'NVIDIA_VISIBLE_DEVICES': 'all',
            }
        )
    
    def get_rfdiffusion_config(self, models_dir: str, output_dir: str) -> NGCContainerConfig:
        """
        Get NGC container configuration for RFDiffusion.
        
        Args:
            models_dir: Path to RFDiffusion models
            output_dir: Output directory for results
            
        Returns:
            NGCContainerConfig for RFDiffusion
        """
        # Note: RFDiffusion may not have official NGC container
        # This is a placeholder for custom NGC container
        return NGCContainerConfig(
            name='rfdiffusion-custom',
            image='nvcr.io/nvidia/pytorch:24.01-py3',
            gpu_required=True,
            platform='linux/amd64',
            volumes=[
                f'{models_dir}:/models',
                f'{output_dir}:/output',
            ],
            environment={
                'NVIDIA_VISIBLE_DEVICES': 'all',
            }
        )
    
    def run_container(self, config: NGCContainerConfig, command: Optional[str] = None) -> subprocess.CompletedProcess:
        """
        Run an NGC container.
        
        Args:
            config: Container configuration
            command: Command to run in container
            
        Returns:
            Completed process result
        """
        if not self.check_docker_available():
            raise RuntimeError("Docker is not available")
        
        # Build docker run command
        cmd = ['docker', 'run', '--rm']
        
        # Add GPU support if required
        if config.gpu_required:
            cmd.extend(['--gpus', 'all'])
        
        # Add platform
        cmd.extend(['--platform', config.platform])
        
        # Add volumes
        for volume in config.volumes:
            cmd.extend(['-v', volume])
        
        # Add environment variables
        for key, value in config.environment.items():
            cmd.extend(['-e', f'{key}={value}'])
        
        # Add image
        cmd.append(config.image)
        
        # Add command if provided
        if command or config.command:
            cmd.append(command or config.command)
        
        if self.verbose:
            print(f"Running NGC container: {config.name}")
            print(f"Command: {' '.join(cmd)}")
        
        # Run container
        return subprocess.run(cmd, capture_output=True, text=True)
    
    def get_recommendation(self) -> str:
        """
        Get recommendation for NGC container usage.
        
        Returns:
            Recommendation string
        """
        import platform
        is_arm64 = platform.machine().lower() in ('aarch64', 'arm64')
        
        if not is_arm64:
            return "AMD64 system - NGC containers will run natively with full GPU support"
        
        if self.use_emulation:
            return (
                "ARM64 system - NGC containers will run via emulation\n"
                "Recommendations:\n"
                "  1. Use Docker emulation (current setting)\n"
                "     - Works but may have performance impact\n"
                "     - Good for testing and development\n"
                "  2. Use cloud AMD64 instances for production\n"
                "     - Full GPU performance\n"
                "     - AWS, GCP, Azure all support NGC containers"
            )
        else:
            return (
                "ARM64 system - Recommended to use cloud instances\n"
                "Cloud Options:\n"
                "  1. AWS EC2 with NGC containers\n"
                "  2. GCP with NGC containers\n"
                "  3. Azure with NGC containers\n"
                "  4. NVIDIA NGC Cloud (DGX)\n"
                "\n"
                "All provide native AMD64 execution with full GPU support"
            )
    
    def install_ngc_cli(self) -> bool:
        """
        Install NGC CLI tool.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if already installed
            result = subprocess.run(
                ['ngc', '--version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                if self.verbose:
                    print("NGC CLI already installed")
                return True
        except FileNotFoundError:
            pass
        
        if self.verbose:
            print("NGC CLI not found. Installing...")
        
        # Download and install NGC CLI
        # This is a simplified version - actual implementation would vary by OS
        install_script = """
        wget --content-disposition https://ngc.nvidia.com/downloads/ngccli_linux.zip
        unzip -o ngccli_linux.zip
        chmod u+x ngc-cli/ngc
        echo 'export PATH="$PATH:$(pwd)/ngc-cli"' >> ~/.bashrc
        """
        
        if self.verbose:
            print("To install NGC CLI manually, run:")
            print(install_script)
        
        return False
    
    def setup_ngc_registry(self) -> bool:
        """
        Setup NGC registry authentication.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.ngc_api_key:
            if self.verbose:
                print("NGC API key not set. Please set NGC_CLI_API_KEY environment variable")
                print("Get your API key from: https://catalog.ngc.nvidia.com/")
            return False
        
        if not self.check_docker_available():
            if self.verbose:
                print("Docker is not available")
            return False
        
        try:
            # Login to NGC registry
            result = subprocess.run(
                ['docker', 'login', 'nvcr.io', 
                 '--username', '$oauthtoken',
                 '--password-stdin'],
                input=self.ngc_api_key.encode(),
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                if self.verbose:
                    print("Successfully logged into NGC registry")
                return True
            else:
                if self.verbose:
                    print(f"Failed to login to NGC registry: {result.stderr.decode()}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            if self.verbose:
                print(f"Error setting up NGC registry: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get NGC fallback status.
        
        Returns:
            Dictionary with status information
        """
        import platform
        
        status = {
            'architecture': platform.machine(),
            'is_arm64': platform.machine().lower() in ('aarch64', 'arm64'),
            'docker_available': self.check_docker_available(),
            'ngc_api_key_set': bool(self.ngc_api_key),
            'ngc_logged_in': False,
            'use_emulation': self.use_emulation,
            'recommendation': self.get_recommendation(),
        }
        
        if status['docker_available'] and status['ngc_api_key_set']:
            status['ngc_logged_in'] = self.check_ngc_login()
        
        return status


def setup_ngc_fallback(use_emulation: bool = True, verbose: bool = True) -> NGCFallback:
    """
    Setup NGC fallback with automatic configuration.
    
    Args:
        use_emulation: Use Docker emulation on ARM64
        verbose: Print status messages
        
    Returns:
        Configured NGCFallback instance
    """
    fallback = NGCFallback(use_emulation=use_emulation, verbose=verbose)
    
    if verbose:
        print("Setting up NGC fallback...")
        
    status = fallback.get_status()
    
    if verbose:
        print(f"\nNGC Fallback Status:")
        print(f"  Architecture: {status['architecture']}")
        print(f"  Docker Available: {status['docker_available']}")
        print(f"  NGC API Key Set: {status['ngc_api_key_set']}")
        print(f"  NGC Logged In: {status['ngc_logged_in']}")
        print(f"\n{status['recommendation']}")
    
    return fallback
