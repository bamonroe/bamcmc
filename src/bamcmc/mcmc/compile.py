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

from ..registry import get_posterior
from .types import BlockArrays, RunParams
from .scan import mcmc_scan_body_offload


# --- CONSTANTS ---
CHUNK_SIZE = 100

# --- COMPILED FUNCTION CACHE ---
# Cache compiled MCMC kernels by configuration hash (in-memory, within session)
_COMPILED_KERNEL_CACHE = {}


def _compute_cache_key(user_config: Dict[str, Any], data: Dict[str, Any], block_arrays: BlockArrays, run_params: RunParams) -> Tuple:
    """
    Compute a cache key for the compiled MCMC kernel.

    The key captures everything that affects the compiled function:
    - Posterior ID
    - Data shapes (not values)
    - Number of chains
    - Number of parameters
    - Block structure
    - Dtype settings
    - RunParams values (all static args including START_ITERATION)
    """
    # Extract shapes from data tuples
    int_shapes = tuple(arr.shape for arr in data['int'])
    float_shapes = tuple(arr.shape for arr in data['float'])
    static_shape = tuple(data['static']) if 'static' in data else ()

    # Note: START_ITERATION is intentionally excluded from the cache key.
    # The kernel always uses START_ITERATION=0, and we reset current_iteration
    # to 0 at the start of each run (including resume). Global iteration tracking
    # is handled separately via iteration_offset in the carry/checkpoint.
    key = (
        user_config['posterior_id'],
        user_config['num_chains'],
        user_config['use_double'],
        run_params.NUM_COLLECT,
        run_params.THIN_ITERATION,
        run_params.BURN_ITER,
        run_params.SAVE_LIKELIHOODS,
        run_params.NUM_GQ,
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
    coupled_transform_fn = model_config.get('coupled_transform_dispatch')
    gq_fn = model_config.get('generated_quantities')

    # Bind data to log_post_fn, keeping beta as a passable argument
    # The posterior signature is: log_posterior(chain_state, param_indices, data, beta=1.0)
    # We bind data but leave beta unbound so callers can pass it for tempering
    def log_post_fn_bound(chain_state, param_indices, beta=1.0):
        return log_post_fn(chain_state, param_indices, data, beta)

    # Create gradient function for MALA proposals
    # jax.grad creates the derivative function (no computation yet)
    # Gradient is w.r.t. first argument (chain_state)
    # Note: gradient is computed at beta=1.0 (full posterior) for proposal direction
    def log_post_fn_for_grad(chain_state, param_indices):
        return log_post_fn(chain_state, param_indices, data, 1.0)
    grad_log_post_fn = jax.grad(log_post_fn_for_grad, argnums=0)

    # Bind data to coupled_transform_fn if provided
    coupled_transform_fn_bound = None
    if coupled_transform_fn is not None:
        coupled_transform_fn_bound = partial(coupled_transform_fn, data=data)

    # Create scan body with data bound inside trace
    # Using partial inside the traced function makes the data binding
    # part of the trace, not a closure
    scan_body = partial(
        mcmc_scan_body_offload,
        log_post_fn=log_post_fn_bound,
        grad_log_post_fn=grad_log_post_fn,
        direct_sampler_fn=partial(direct_sampler_fn, data=data),
        coupled_transform_fn=coupled_transform_fn_bound,
        gq_fn=partial(gq_fn, data=data) if gq_fn else None,
        block_arrays=block_arrays,
        run_params=run_params
    )

    # Use fori_loop since we don't need accumulated outputs (scan_body returns None)
    # This is cleaner and makes it explicit that we only care about the final carry
    def fori_body(i, c):
        new_carry, _ = scan_body(c, i)
        return new_carry

    final_carry = jax.lax.fori_loop(0, CHUNK_SIZE, fori_body, carry)
    return final_carry, None


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
    user_config: Dict[str, Any],
    runtime_ctx: Dict[str, Any],
    block_arrays: BlockArrays,
    run_params: RunParams,
    initial_carry: Tuple
) -> Tuple[Any, float]:
    """
    Compile the MCMC kernel, using cache if available.

    Args:
        user_config: User configuration dict (serializable, no JAX types)
        runtime_ctx: Runtime context dict with JAX-dependent objects (dtypes, data, keys)
        block_arrays: BlockArrays with block configuration
        run_params: RunParams with run configuration
        initial_carry: Initial MCMC state for tracing

    Returns:
        Tuple of (compiled_chunk_fn, compile_time)
    """
    data = runtime_ctx['data']

    # Check in-memory cache for compiled kernel
    cache_key = _compute_cache_key(user_config, data, block_arrays, run_params)
    compiled_chunk = _COMPILED_KERNEL_CACHE.get(cache_key)

    # Prepare data for passing as traced arguments (enables cross-session caching)
    data_int = data['int']
    data_float = data['float']
    data_static = tuple(data['static'])
    posterior_id = user_config['posterior_id']

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
    print(f"Kernel cached (key: {posterior_id}, {user_config['num_chains']} chains)")

    return compiled_chunk, compile_time
