"""
Hardware Info - Hardware fingerprinting and version tracking.

This module collects hardware, git, and version information used by the
benchmark system to ensure benchmarks are comparable across sessions.

Functions:
- get_hardware_info: Collect GPU/JAX hardware fingerprint
- get_git_info: Get git branch/commit for the bamcmc repo
- get_bamcmc_version: Get installed bamcmc package version
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import jax


def get_hardware_info() -> Dict[str, Any]:
    """
    Collect hardware information for benchmark context.

    This helps identify if benchmarks were run on different hardware,
    which would make comparisons invalid.
    """
    info = {
        'jax_backend': str(jax.default_backend()),
        'jax_devices': [str(d) for d in jax.devices()],
        'jax_version': jax.__version__,
    }

    # Try to get GPU info via nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 3:
                info['gpu_name'] = parts[0]
                info['gpu_memory'] = parts[1]
                info['driver_version'] = parts[2]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try to get CUDA version from nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        # nvidia-smi also shows CUDA version in full output
        result2 = subprocess.run(
            ['nvidia-smi'],
            capture_output=True, text=True, timeout=5
        )
        if result2.returncode == 0:
            for line in result2.stdout.split('\n'):
                if 'CUDA Version' in line:
                    # Extract CUDA version from line like "| NVIDIA-SMI 580.119.02    Driver Version: 580.119.02    CUDA Version: 13.0  |"
                    parts = line.split('CUDA Version:')
                    if len(parts) > 1:
                        cuda_ver = parts[1].strip().split()[0]
                        info['cuda_version'] = cuda_ver
                    break
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return info


def get_git_info(repo_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Get git repository information for the bamcmc package.

    This helps track which version of the backend code was used.
    """
    info = {}

    if repo_path is None:
        repo_path = Path(__file__).parent

    try:
        # Get current branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=repo_path
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()

        # Get current commit (short hash)
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=repo_path
        )
        if result.returncode == 0:
            info['commit'] = result.stdout.strip()

        # Get full commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=repo_path
        )
        if result.returncode == 0:
            info['commit_full'] = result.stdout.strip()

        # Check if working directory is clean
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5,
            cwd=repo_path
        )
        if result.returncode == 0:
            info['dirty'] = len(result.stdout.strip()) > 0

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return info


def get_bamcmc_version() -> str:
    """Get bamcmc package version."""
    try:
        from bamcmc import __version__
        return __version__
    except (ImportError, AttributeError):
        return "unknown"
