"""
Persistent posterior-specific benchmarking system.

This module provides:
1. Hashing of posterior systems (code + data) for consistent identification
2. Persistent storage of benchmark results per posterior configuration
3. Loading cached benchmarks to avoid re-running
4. Hardware fingerprinting for reproducibility tracking

The key insight is that the same posterior+data should always produce the same
hash, allowing benchmarks to be cached and reused across sessions.
"""

import hashlib
import inspect
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np


def get_function_source_hash(func) -> str:
    """
    Get a hash of a function's source code.

    This captures the actual implementation, so code changes will produce
    different hashes even if the function signature is the same.
    """
    if func is None:
        return "none"

    try:
        source = inspect.getsource(func)
        return hashlib.sha256(source.encode()).hexdigest()[:16]
    except (TypeError, OSError):
        # Can't get source (built-in, C extension, lambda, etc.)
        # Fall back to a representation-based hash
        return hashlib.sha256(repr(func).encode()).hexdigest()[:16]


def get_data_hash(data: Dict) -> str:
    """
    Get a hash of the data structure.

    Hashes both shapes and content for exact reproducibility.
    The same data will always produce the same hash.
    """
    hasher = hashlib.sha256()

    # Hash static values
    if 'static' in data:
        static_vals = data['static']
        if isinstance(static_vals, (list, tuple)):
            hasher.update(f"static:{tuple(static_vals)}".encode())
        else:
            hasher.update(f"static:{static_vals}".encode())

    # Hash int arrays (shapes and content)
    if 'int' in data:
        for i, arr in enumerate(data['int']):
            arr_np = np.asarray(arr)
            hasher.update(f"int_{i}_shape:{arr_np.shape}_dtype:{arr_np.dtype}".encode())
            hasher.update(arr_np.tobytes())

    # Hash float arrays (shapes and content)
    if 'float' in data:
        for i, arr in enumerate(data['float']):
            arr_np = np.asarray(arr)
            hasher.update(f"float_{i}_shape:{arr_np.shape}_dtype:{arr_np.dtype}".encode())
            # Round floats to avoid floating point representation noise
            arr_rounded = np.round(arr_np.astype(np.float64), decimals=10)
            hasher.update(arr_rounded.tobytes())

    return hasher.hexdigest()[:16]


def get_posterior_hash(posterior_id: str, model_config: Dict, data: Dict, num_chains: int = None) -> str:
    """
    Compute a unique hash for a posterior configuration.

    Combines:
    - Posterior ID (name)
    - Source code of all registered functions
    - Data structure (shapes and content)
    - Number of chains (affects benchmark timing)

    The same posterior code + same data + same chains = same hash, guaranteed.

    Args:
        posterior_id: The registered name of the posterior
        model_config: The model config dict from the registry
        data: The data dict with 'static', 'int', 'float' keys
        num_chains: Number of MCMC chains (included in hash for benchmark accuracy)

    Returns:
        16-character hex hash that uniquely identifies this configuration
    """
    hasher = hashlib.sha256()

    # Posterior identifier
    hasher.update(f"posterior_id:{posterior_id}".encode())

    # Number of chains (important for benchmark timing)
    if num_chains is not None:
        hasher.update(f"num_chains:{num_chains}".encode())

    # Hash each registered function's source code
    function_keys = [
        'log_posterior',
        'direct_sampler',
        'generated_quantities',
        'initial_vector',
        'batch_type',
        'get_num_gq'
    ]

    for key in function_keys:
        func = model_config.get(key)
        func_hash = get_function_source_hash(func)
        hasher.update(f"{key}:{func_hash}".encode())

    # Hash data
    data_hash = get_data_hash(data)
    hasher.update(f"data:{data_hash}".encode())

    return hasher.hexdigest()[:16]


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


def get_default_benchmark_dir() -> str:
    """
    Get the default benchmark directory.

    Priority:
    1. BAMCMC_BENCHMARKDIR environment variable
    2. ./posterior_benchmarks (in current working directory)
    """
    env_dir = os.environ.get('BAMCMC_BENCHMARKDIR')
    if env_dir:
        return env_dir
    return os.path.join(os.getcwd(), 'posterior_benchmarks')


class PosteriorBenchmarkManager:
    """
    Manages persistent benchmark storage and retrieval for specific posteriors.

    Each posterior+data combination gets its own benchmark file, identified
    by a hash of the posterior code and data content.

    Benchmark files contain:
    - The hash (for verification)
    - Posterior ID
    - bamcmc version and git info
    - Hardware info
    - Benchmark results (compile time, iteration time)
    - Timestamp

    The default storage location is ./posterior_benchmarks in the current
    working directory. This can be overridden with the BAMCMC_BENCHMARKDIR
    environment variable.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize benchmark manager.

        Args:
            cache_dir: Directory for storing benchmarks.
                      Defaults to BAMCMC_BENCHMARKDIR env var, or
                      ./posterior_benchmarks in current working directory.
        """
        if cache_dir is None:
            cache_dir = get_default_benchmark_dir()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_benchmark_path(self, posterior_hash: str) -> Path:
        """Get path to benchmark file for a given hash."""
        return self.cache_dir / f"benchmark_{posterior_hash}.json"

    def load_benchmark(self, posterior_hash: str) -> Optional[Dict[str, Any]]:
        """
        Load benchmark for a posterior hash if it exists.

        Args:
            posterior_hash: Hash identifying the posterior configuration

        Returns:
            Benchmark dict if found, None otherwise
        """
        path = self._get_benchmark_path(posterior_hash)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    benchmark = json.load(f)
                    # Verify hash matches filename
                    if benchmark.get('posterior_hash') == posterior_hash:
                        return benchmark
                    else:
                        print(f"Warning: Hash mismatch in {path}, ignoring cached benchmark")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load benchmark {path}: {e}")
        return None

    def save_benchmark(self,
                       posterior_hash: str,
                       posterior_id: str,
                       num_chains: int,
                       fresh_compile_time: float,
                       iteration_time: float,
                       benchmark_iterations: int,
                       cached_compile_time: Optional[float] = None,
                       extra_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Save benchmark results.

        Args:
            posterior_hash: Hash identifying the posterior configuration
            posterior_id: Human-readable posterior name
            num_chains: Number of MCMC chains
            fresh_compile_time: Time for fresh compilation (seconds)
            iteration_time: Average time per iteration (seconds)
            benchmark_iterations: Number of iterations used for timing
            cached_compile_time: Time for cached compilation (seconds), if measured
            extra_info: Additional info to store

        Returns:
            The saved benchmark dict
        """
        benchmark = {
            'posterior_hash': posterior_hash,
            'posterior_id': posterior_id,
            'bamcmc_version': get_bamcmc_version(),
            'git': get_git_info(),
            'hardware': get_hardware_info(),
            'config': {
                'num_chains': num_chains,
                'benchmark_iterations': benchmark_iterations,
            },
            'results': {
                'fresh_compile_time_s': round(fresh_compile_time, 4),
                'cached_compile_time_s': round(cached_compile_time, 4) if cached_compile_time else None,
                'iteration_time_s': round(iteration_time, 6),
            },
            'timestamp': datetime.now().isoformat(),
        }

        if extra_info:
            benchmark['extra'] = extra_info

        path = self._get_benchmark_path(posterior_hash)
        with open(path, 'w') as f:
            json.dump(benchmark, f, indent=2)

        return benchmark

    def check_hardware_match(self, benchmark: Dict[str, Any]) -> bool:
        """
        Check if stored benchmark was run on matching hardware.

        Returns True if hardware matches, False otherwise.
        """
        current_hw = get_hardware_info()
        cached_hw = benchmark.get('hardware', {})

        # Key fields that must match for valid comparison
        return (
            current_hw.get('gpu_name') == cached_hw.get('gpu_name') and
            current_hw.get('jax_backend') == cached_hw.get('jax_backend')
        )

    def get_cached_benchmark(self,
                             posterior_hash: str,
                             require_hardware_match: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get a cached benchmark if available and valid.

        Args:
            posterior_hash: Hash identifying the posterior configuration
            require_hardware_match: If True, only return if hardware matches

        Returns:
            Benchmark dict if valid cache exists, None otherwise
        """
        benchmark = self.load_benchmark(posterior_hash)

        if benchmark is None:
            return None

        if require_hardware_match and not self.check_hardware_match(benchmark):
            return None

        return benchmark

    def list_benchmarks(self) -> list:
        """List all stored benchmarks."""
        benchmarks = []
        for path in self.cache_dir.glob("benchmark_*.json"):
            try:
                with open(path, 'r') as f:
                    benchmarks.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                pass
        return sorted(benchmarks, key=lambda x: x.get('timestamp', ''), reverse=True)

    def print_benchmark(self, benchmark: Dict[str, Any], verbose: bool = True):
        """Print a human-readable benchmark summary."""
        results = benchmark.get('results', {})
        config = benchmark.get('config', {})
        hw = benchmark.get('hardware', {})
        git = benchmark.get('git', {})

        print(f"Posterior: {benchmark.get('posterior_id', 'unknown')}")
        print(f"  Hash: {benchmark.get('posterior_hash', '?')[:12]}...")

        if verbose:
            print(f"  Version: bamcmc {benchmark.get('bamcmc_version', '?')}")
            if git:
                dirty = " (dirty)" if git.get('dirty') else ""
                print(f"  Git: {git.get('branch', '?')}@{git.get('commit', '?')}{dirty}")
            if hw.get('gpu_name'):
                print(f"  GPU: {hw['gpu_name']}")

        print(f"  Compile (fresh): {results.get('fresh_compile_time_s', 0):.2f}s")
        if results.get('cached_compile_time_s'):
            print(f"  Compile (cached): {results['cached_compile_time_s']:.2f}s")
        print(f"  Per-iteration: {results.get('iteration_time_s', 0):.4f}s")
        print(f"  Chains: {config.get('num_chains', '?')}")

        if verbose:
            print(f"  Benchmark iters: {config.get('benchmark_iterations', '?')}")
            print(f"  Timestamp: {benchmark.get('timestamp', '?')}")


    def compare_benchmark(self,
                          posterior_hash: str,
                          new_iteration_time: float,
                          new_compile_time: float,
                          posterior_id: str = None) -> Dict[str, Any]:
        """
        Compare a new benchmark against cached results.

        Args:
            posterior_hash: Hash identifying the posterior configuration
            new_iteration_time: New iteration time in seconds
            new_compile_time: New compile time in seconds
            posterior_id: Optional posterior ID for display

        Returns:
            Dict with comparison results including deltas and percentages
        """
        cached = self.load_benchmark(posterior_hash)

        comparison = {
            'has_cached': cached is not None,
            'new_iteration_time': new_iteration_time,
            'new_compile_time': new_compile_time,
            'cached_iteration_time': None,
            'cached_compile_time': None,
            'iteration_delta': None,
            'iteration_pct_change': None,
            'compile_delta': None,
            'compile_pct_change': None,
            'cached_git': None,
            'current_git': get_git_info(),
        }

        if cached:
            cached_results = cached.get('results', {})
            cached_iter = cached_results.get('iteration_time_s')
            cached_compile = cached_results.get('fresh_compile_time_s')

            comparison['cached_iteration_time'] = cached_iter
            comparison['cached_compile_time'] = cached_compile
            comparison['cached_git'] = cached.get('git', {})

            if cached_iter:
                delta = new_iteration_time - cached_iter
                pct = (delta / cached_iter) * 100
                comparison['iteration_delta'] = delta
                comparison['iteration_pct_change'] = pct

            if cached_compile:
                delta = new_compile_time - cached_compile
                pct = (delta / cached_compile) * 100
                comparison['compile_delta'] = delta
                comparison['compile_pct_change'] = pct

        return comparison

    def print_comparison(self, comparison: Dict[str, Any], posterior_id: str = None):
        """Print a human-readable benchmark comparison."""
        print("\n" + "=" * 60)
        print("BENCHMARK COMPARISON")
        print("=" * 60)

        if posterior_id:
            print(f"Posterior: {posterior_id}")

        current_git = comparison.get('current_git', {})
        cached_git = comparison.get('cached_git', {})

        if current_git:
            print(f"Current:  {current_git.get('branch', '?')}@{current_git.get('commit', '?')}")
        if cached_git:
            print(f"Cached:   {cached_git.get('branch', '?')}@{cached_git.get('commit', '?')}")

        print("-" * 60)

        # Iteration time comparison
        new_iter = comparison['new_iteration_time']
        cached_iter = comparison['cached_iteration_time']

        print(f"Iteration Time:")
        print(f"  New:    {new_iter:.6f} s")
        if cached_iter:
            print(f"  Cached: {cached_iter:.6f} s")
            delta = comparison['iteration_delta']
            pct = comparison['iteration_pct_change']
            sign = "+" if delta > 0 else ""
            status = "SLOWER" if delta > 0 else "FASTER" if delta < 0 else "SAME"
            color_status = f"({status})"
            print(f"  Delta:  {sign}{delta:.6f} s ({sign}{pct:.1f}%) {color_status}")
        else:
            print(f"  Cached: (no previous benchmark)")

        # Compile time comparison
        new_compile = comparison['new_compile_time']
        cached_compile = comparison['cached_compile_time']

        print(f"\nCompile Time:")
        print(f"  New:    {new_compile:.4f} s")
        if cached_compile:
            print(f"  Cached: {cached_compile:.4f} s")
            delta = comparison['compile_delta']
            pct = comparison['compile_pct_change']
            sign = "+" if delta > 0 else ""
            status = "SLOWER" if delta > 0 else "FASTER" if delta < 0 else "SAME"
            print(f"  Delta:  {sign}{delta:.4f} s ({sign}{pct:.1f}%) ({status})")
        else:
            print(f"  Cached: (no previous benchmark)")

        # Total time predictions for practical iteration counts
        print("-" * 60)
        print("Total Time Predictions (compile + iterations):")
        print(f"{'Iterations':<12} {'New':>12} {'Cached':>12} {'Delta':>12} {'Status':>10}")
        print("-" * 60)

        for n_iters in [1000, 5000, 10000]:
            new_total = new_compile + (new_iter * n_iters)

            if cached_iter and cached_compile:
                cached_total = cached_compile + (cached_iter * n_iters)
                delta = new_total - cached_total
                pct = (delta / cached_total) * 100
                sign = "+" if delta > 0 else ""
                status = "SLOWER" if delta > 0 else "FASTER" if delta < 0 else "SAME"

                print(f"{n_iters:<12} {self._format_time(new_total):>12} "
                      f"{self._format_time(cached_total):>12} "
                      f"{sign}{pct:>+5.1f}%      {status:>10}")
            else:
                print(f"{n_iters:<12} {self._format_time(new_total):>12} "
                      f"{'N/A':>12} {'N/A':>12} {'N/A':>10}")

        print("=" * 60)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


def get_manager(cache_dir: Optional[str] = None) -> PosteriorBenchmarkManager:
    """
    Get a benchmark manager for the specified or default directory.

    Note: This creates a new manager each time to respect the current
    working directory and BAMCMC_BENCHMARKDIR environment variable.

    Args:
        cache_dir: Optional explicit directory. If None, uses
                  BAMCMC_BENCHMARKDIR env var or ./posterior_benchmarks
    """
    return PosteriorBenchmarkManager(cache_dir)


def compute_posterior_hash(posterior_id: str, model_config: Dict, data: Dict, num_chains: int = None) -> str:
    """
    Convenience function to compute posterior hash.

    See get_posterior_hash for details.
    """
    return get_posterior_hash(posterior_id, model_config, data, num_chains)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(
    mcmc_config: Dict[str, Any],
    data: Dict[str, Any],
    benchmark_iterations: int = 100,
    compare: bool = True,
    update_cache: bool = False,
) -> Dict[str, Any]:
    """
    Run benchmark on a posterior model and optionally compare against cached results.

    This is a lightweight alternative to rmcmc() for performance testing only.
    No samples are collected or saved.

    Args:
        mcmc_config: MCMC configuration dict (needs POSTERIOR_ID, chain config, etc.)
        data: Data dict with 'static', 'int', 'float' arrays
        benchmark_iterations: Number of iterations for timing (default 100)
        compare: If True, compare against cached benchmark and print results
        update_cache: If True, save new benchmark as the cached baseline

    Returns:
        Dict with benchmark results:
            - iteration_time: Average time per iteration (seconds)
            - compile_time: Time to compile kernel (seconds)
            - posterior_hash: Hash identifying the posterior configuration
            - comparison: Comparison dict if compare=True, else None
            - user_config: Clean serializable config
    """
    from .mcmc.config import configure_mcmc_system, initialize_mcmc_system
    from .mcmc.compile import compile_mcmc_kernel, benchmark_mcmc_sampler

    # Ensure benchmark-only settings
    mcmc_config = mcmc_config.copy()
    mcmc_config['num_collect'] = 0
    mcmc_config['burn_iter'] = 0
    mcmc_config['save_likelihoods'] = False
    mcmc_config.setdefault('thin_iteration', 1)

    # Configure system
    print("Configuring MCMC system...")
    user_config, runtime_ctx, model_ctx = configure_mcmc_system(mcmc_config, data)

    run_params = model_ctx['run_params']
    block_arrays = model_ctx['block_arrays']

    # Compute posterior hash (includes num_chains for accurate benchmark caching)
    posterior_id = user_config['posterior_id']
    num_chains = user_config['num_chains']
    posterior_hash = get_posterior_hash(
        posterior_id,
        model_ctx['model_config'],
        runtime_ctx['data'],
        num_chains
    )
    print(f"Posterior hash: {posterior_hash}")

    # Initialize
    print("Generating initial vector...")
    initial_vector_np = model_ctx['initial_vector_fn'](user_config)

    initial_carry, user_config = initialize_mcmc_system(
        initial_vector_np,
        user_config,
        runtime_ctx,
        num_gq=run_params.NUM_GQ,
        num_collect=run_params.NUM_COLLECT,
        num_blocks=block_arrays.num_blocks
    )

    print(f"JAX backend: {jax.default_backend()}")

    # Compile kernel
    compiled_chunk, compile_time = compile_mcmc_kernel(
        user_config, runtime_ctx, block_arrays, run_params, initial_carry
    )

    # Run benchmark
    bench_results = benchmark_mcmc_sampler(
        compiled_chunk, initial_carry, benchmark_iterations,
        chunk_size=run_params.CHUNK_SIZE
    )
    iteration_time = bench_results['avg_time']

    # Compare and optionally update cache
    benchmark_mgr = get_manager()
    comparison = None

    if compare:
        comparison = benchmark_mgr.compare_benchmark(
            posterior_hash,
            new_iteration_time=iteration_time,
            new_compile_time=compile_time,
            posterior_id=posterior_id
        )
        benchmark_mgr.print_comparison(comparison, posterior_id=posterior_id)

    if update_cache:
        print("\nUpdating cached benchmark...")
        benchmark_mgr.save_benchmark(
            posterior_hash=posterior_hash,
            posterior_id=posterior_id,
            num_chains=user_config['num_chains'],
            fresh_compile_time=compile_time,
            iteration_time=iteration_time,
            benchmark_iterations=benchmark_iterations,
        )
        print(f"Benchmark saved (hash: {posterior_hash[:8]}...)")
    elif compare:
        print("\n(Use update_cache=True to save these results as the new baseline)")

    return {
        'iteration_time': iteration_time,
        'compile_time': compile_time,
        'posterior_hash': posterior_hash,
        'comparison': comparison,
        'user_config': user_config,
    }
