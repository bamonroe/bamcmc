"""
MCMC Kernel Compilation and Caching.

This module handles JAX compilation of the MCMC kernel:
- _run_mcmc_chunk: Module-level MCMC chunk runner for cache-stable tracing
- _compute_cache_key: Compute in-memory cache key for compiled kernels
- benchmark_mcmc_sampler: Run benchmarks on compiled kernel
- COMPILED_KERNEL_CACHE: In-memory cache for compiled kernels
"""

import jax
import jax.numpy as jnp
import jax.lax
import time
from functools import partial
from typing import Dict, Any, Tuple

from .registry import get_posterior
from .mcmc_types import BlockArrays, RunParams
from .mcmc_scan import mcmc_scan_body_offload


# --- CONSTANTS ---
CHUNK_SIZE = 100

# --- COMPILED FUNCTION CACHE ---
# Cache compiled MCMC kernels by configuration hash (in-memory, within session)
_COMPILED_KERNEL_CACHE = {}


def _compute_cache_key(mcmc_config: Dict[str, Any], data: Dict[str, Any], block_arrays: BlockArrays) -> Tuple:
    """
    Compute a cache key for the compiled MCMC kernel.

    The key captures everything that affects the compiled function:
    - Posterior ID
    - Data shapes (not values)
    - Number of chains
    - Number of parameters
    - Block structure
    - Dtype settings
    """
    # Extract shapes from data tuples
    int_shapes = tuple(arr.shape for arr in data['int'])
    float_shapes = tuple(arr.shape for arr in data['float'])
    static_shape = tuple(data['static']) if 'static' in data else ()

    key = (
        mcmc_config['POSTERIOR_ID'],
        mcmc_config['NUM_CHAINS'],
        mcmc_config['USE_DOUBLE'],
        mcmc_config.get('NUM_COLLECT', 0),
        mcmc_config.get('SAVE_LIKELIHOODS', False),
        block_arrays.num_blocks,
        block_arrays.max_size,
        int_shapes,
        float_shapes,
        static_shape,
    )
    return key


def get_compiled_kernel_cache() -> Dict:
    """Get reference to the compiled kernel cache."""
    return _COMPILED_KERNEL_CACHE


def _run_mcmc_chunk(carry, data_int, data_float, data_static, block_arrays,
                    run_params, posterior_id):
    """
    Module-level MCMC chunk runner for cache-stable tracing.

    By defining this at module level and passing data as explicit arguments
    (not captured via closure), JAX's disk cache can recognize identical
    traces across Python sessions.

    Args:
        carry: MCMC state tuple
        data_int: Tuple of integer data arrays (traced)
        data_float: Tuple of float data arrays (traced)
        data_static: Tuple of static values like n_subjects (static)
        block_arrays: BlockArrays dataclass (traced - contains arrays)
        run_params: RunParams frozen dataclass (static - no arrays)
        posterior_id: String identifier for posterior (static)

    Returns:
        Updated carry tuple and scan outputs
    """
    # Reconstruct data dict inside trace
    data = {
        'int': data_int,
        'float': data_float,
        'static': data_static
    }

    # Get posterior functions from registry
    # These are the same module-level functions each session
    model_config = get_posterior(posterior_id)
    log_post_fn = model_config['log_posterior']
    direct_sampler_fn = model_config['direct_sampler']
    gq_fn = model_config.get('generated_quantities')

    # Create scan body with data bound inside trace
    # Using partial inside the traced function makes the data binding
    # part of the trace, not a closure
    scan_body = partial(
        mcmc_scan_body_offload,
        log_post_fn=partial(log_post_fn, data=data),
        direct_sampler_fn=partial(direct_sampler_fn, data=data),
        gq_fn=partial(gq_fn, data=data) if gq_fn else None,
        block_arrays=block_arrays,
        run_params=run_params
    )

    chunk_range = jnp.arange(CHUNK_SIZE)
    return jax.lax.scan(scan_body, carry, chunk_range)


def benchmark_mcmc_sampler(compiled_chunk_fn, initial_carry, benchmark_iters: int) -> Dict[str, float]:
    """
    Run benchmark iterations on compiled MCMC kernel.

    Args:
        compiled_chunk_fn: Compiled MCMC chunk function
        initial_carry: Initial MCMC state
        benchmark_iters: Number of iterations to benchmark

    Returns:
        Dict with 'avg_time' key containing average time per iteration
    """
    print("\n--- BENCHMARKING ---")
    print(f"Running benchmark ({benchmark_iters} iterations)...", flush=True)
    num_chunks = (benchmark_iters + CHUNK_SIZE - 1) // CHUNK_SIZE
    start_bench = time.perf_counter()
    current_carry = initial_carry
    for _ in range(num_chunks):
        current_carry, _ = compiled_chunk_fn(current_carry)
        jax.block_until_ready(current_carry)
    end_bench = time.perf_counter()
    total_time = end_bench - start_bench
    avg_time = total_time / (num_chunks * CHUNK_SIZE)
    print("--- Benchmark Results (per iteration) ---")
    print(f"  Avg: {avg_time:.6f} s")
    return {'avg_time': avg_time}


def compile_mcmc_kernel(
    mcmc_config: Dict[str, Any],
    data: Dict[str, Any],
    block_arrays: BlockArrays,
    run_params: RunParams,
    initial_carry: Tuple
) -> Tuple[Any, float]:
    """
    Compile the MCMC kernel, using cache if available.

    Args:
        mcmc_config: MCMC configuration dict
        data: Data dict with 'int', 'float', 'static' keys
        block_arrays: BlockArrays with block configuration
        run_params: RunParams with run configuration
        initial_carry: Initial MCMC state for tracing

    Returns:
        Tuple of (compiled_chunk_fn, compile_time)
    """
    # Check in-memory cache for compiled kernel
    cache_key = _compute_cache_key(mcmc_config, data, block_arrays)
    compiled_chunk = _COMPILED_KERNEL_CACHE.get(cache_key)

    # Prepare data for passing as traced arguments (enables cross-session caching)
    data_int = data['int']
    data_float = data['float']
    data_static = tuple(data['static'])
    posterior_id = mcmc_config['POSTERIOR_ID']

    if compiled_chunk is not None:
        print("Using cached kernel (in-memory)", flush=True)
        return compiled_chunk, 0.0

    # Module-level function approach for reliable compilation.
    # Note: Closure-based approach would be faster per-iteration but hangs
    # during compilation for complex kernels like this MCMC sampler.

    # Create JIT wrapper with static arguments for cache-stable compilation
    run_chunk_jit = jax.jit(
        _run_mcmc_chunk,
        static_argnames=('data_static', 'run_params', 'posterior_id')
    )

    print("Compiling kernel... ", end="", flush=True)
    compile_start = time.perf_counter()

    # AOT compilation with explicit arguments
    compiled_fn = run_chunk_jit.lower(
        initial_carry, data_int, data_float, data_static,
        block_arrays, run_params, posterior_id
    ).compile()

    compile_end = time.perf_counter()
    compile_time = compile_end - compile_start
    print(f"Done ({compile_time:.4f}s)")

    # Create wrapper that binds data arguments (they don't change during run)
    _cf = compiled_fn
    _di, _df, _ba = data_int, data_float, block_arrays
    def compiled_chunk(carry):
        return _cf(carry, _di, _df, _ba)

    # Cache the compiled kernel for in-memory reuse
    _COMPILED_KERNEL_CACHE[cache_key] = compiled_chunk
    print(f"Kernel cached (key: {posterior_id}, {mcmc_config['NUM_CHAINS']} chains)")

    return compiled_chunk, compile_time
