"""
MCMC Backend - Main Entry Point.

This module provides the main rmcmc() function for running MCMC sampling.
The implementation is split across several modules for maintainability:

- mcmc_types: Data structures (BlockArrays, RunParams)
- mcmc_config: Configuration and initialization
- mcmc_sampling: Proposal and sampling functions
- mcmc_scan: Scan body and block statistics
- mcmc_compile: Kernel compilation and caching
- mcmc_diagnostics: Convergence diagnostics
"""

import gc
import jax
import jax.numpy as jnp
import numpy as np
import time
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

# Import from submodules
from .types import BlockArrays, RunParams, build_block_arrays
from .config import (
    configure_mcmc_system,
    initialize_mcmc_system,
    validate_mcmc_inputs,
    gen_rng_keys,
)
from .compile import (
    compile_mcmc_kernel,
    benchmark_mcmc_sampler,
    CHUNK_SIZE,
)
from .diagnostics import (
    compute_nested_rhat,
    compute_and_print_rhat,
    print_acceptance_summary,
    print_swap_acceptance_summary,
)

# Import batch specs for sampler types
from ..batch_specs import SamplerType

# Import error handling for validation
from ..error_handling import validate_mcmc_config, diagnose_sampler_issues, print_diagnostics

# Import checkpoint utilities
from ..checkpoint_helpers import (
    load_checkpoint,
    initialize_from_checkpoint,
    scan_checkpoints,
    get_latest_checkpoint,
)

# Import posterior benchmarking system
from ..posterior_benchmark import (
    get_manager as get_benchmark_manager,
    get_posterior_hash,
)

# Public API for this module
__all__ = [
    'rmcmc',
    'rmcmc_single',
    'BlockArrays',
    'RunParams',
    'build_block_arrays',
    'configure_mcmc_system',
    'initialize_mcmc_system',
    'validate_mcmc_inputs',
    'compute_nested_rhat',
]


# =============================================================================
# RMCMC HELPER FUNCTIONS
# =============================================================================

def _validate_checkpoint_compatibility(
    checkpoint: Dict[str, Any],
    block_arrays: BlockArrays,
    user_config: Dict[str, Any],
) -> None:
    """
    Validate that a checkpoint is compatible with the current model configuration.

    Args:
        checkpoint: Loaded checkpoint dict
        block_arrays: Block array specifications from current model
        user_config: Current user configuration

    Raises:
        ValueError: If checkpoint is incompatible with current configuration
    """
    # Check parameter count matches
    expected_params = block_arrays.total_params
    checkpoint_params = checkpoint['num_params']

    if expected_params != checkpoint_params:
        raise ValueError(
            f"Checkpoint parameter count mismatch: checkpoint has {checkpoint_params} parameters, "
            f"but current model expects {expected_params}. "
            f"This can happen if the model definition changed or you're loading a checkpoint "
            f"from a different model."
        )

    # Posterior ID is already checked in initialize_from_checkpoint, but check here
    # for better error messages before we start processing
    if checkpoint['posterior_id'] != user_config['posterior_id']:
        raise ValueError(
            f"Checkpoint posterior mismatch: checkpoint is for '{checkpoint['posterior_id']}', "
            f"but current config specifies '{user_config['posterior_id']}'."
        )


def _initialize_chains(
    resume_from: Optional[str],
    reset_from: Optional[str],
    reset_noise_scale: float,
    user_config: Dict[str, Any],
    runtime_ctx: Dict[str, Any],
    model_ctx: Dict[str, Any],
    run_params: RunParams,
    block_arrays: BlockArrays,
    init_from: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, Any], RunParams]:
    """
    Initialize chain states from one of four sources.

    Args:
        resume_from: Path to checkpoint for exact resume
        reset_from: Path to checkpoint for reset with noise
        reset_noise_scale: Noise scale for reset mode
        user_config: User configuration dict
        runtime_ctx: Runtime context with data
        model_ctx: Model context with initial_vector_fn
        run_params: Run parameters (may be modified for resume)
        block_arrays: Block array specifications
        init_from: Custom initial vector (e.g., from prior samples)

    Returns:
        initial_carry: JAX carry tuple for MCMC loop
        user_config: Possibly updated config
        run_params: Possibly updated with START_ITERATION

    Raises:
        ValueError: If multiple init sources specified
    """
    # Count how many init sources are specified
    init_sources = sum([
        resume_from is not None,
        reset_from is not None,
        init_from is not None,
    ])
    if init_sources > 1:
        raise ValueError("Cannot specify multiple initialization sources (resume_from, reset_from, init_from)")

    if resume_from is not None:
        # --- RESUME MODE: Continue from exact checkpoint state ---
        print(f"Loading checkpoint from {resume_from}...")
        checkpoint = load_checkpoint(resume_from)
        _validate_checkpoint_compatibility(checkpoint, block_arrays, user_config)
        initial_carry, user_config = initialize_from_checkpoint(
            checkpoint,
            user_config,
            runtime_ctx,
            num_gq=run_params.NUM_GQ,
            num_collect=run_params.NUM_COLLECT,
            num_blocks=block_arrays.num_blocks
        )
        # Store iteration offset for checkpoint saving (global iteration tracking)
        # The kernel always uses START_ITERATION=0, current_iteration starts at 0
        user_config['iteration_offset'] = checkpoint['iteration']

    elif reset_from is not None:
        # --- RESET MODE: Start fresh from cross-chain mean with noise ---
        from ..reset_utils import generate_reset_vector
        print(f"Resetting chains from checkpoint {reset_from}...")

        checkpoint = load_checkpoint(reset_from)
        _validate_checkpoint_compatibility(checkpoint, block_arrays, user_config)
        n_subjects = runtime_ctx['data']['static'][0]
        K = user_config.get('num_superchains', user_config['num_chains'])
        M = user_config['num_chains'] // K

        print(f"  Source iteration: {checkpoint['iteration']}")
        print(f"  Generating {K} reset starting points (noise_scale={reset_noise_scale})")

        initial_vector_np = generate_reset_vector(
            checkpoint,
            model_type=user_config['posterior_id'],
            n_subjects=n_subjects,
            K=K,
            M=M,
            noise_scale=reset_noise_scale,
            rng_seed=user_config.get('rng_seed', None)
        )

        initial_carry, user_config = initialize_mcmc_system(
            initial_vector_np,
            user_config,
            runtime_ctx,
            num_gq=run_params.NUM_GQ,
            num_collect=run_params.NUM_COLLECT,
            num_blocks=block_arrays.num_blocks
        )
        # Reset starts fresh (iteration 0)
        print(f"  Reset complete - starting fresh from iteration 0")

    elif init_from is not None:
        # --- INIT FROM PRIOR: Use provided initial vector ---
        print("Using provided initial vector (e.g., from prior samples)...", flush=True)

        initial_carry, user_config = initialize_mcmc_system(
            init_from,
            user_config,
            runtime_ctx,
            num_gq=run_params.NUM_GQ,
            num_collect=run_params.NUM_COLLECT,
            num_blocks=block_arrays.num_blocks
        )

    else:
        # --- FRESH MODE: Generate new initial values ---
        print("Generating initial vector...", flush=True)
        initial_vector_np = model_ctx['initial_vector_fn'](user_config)

        initial_carry, user_config = initialize_mcmc_system(
            initial_vector_np,
            user_config,
            runtime_ctx,
            num_gq=run_params.NUM_GQ,
            num_collect=run_params.NUM_COLLECT,
            num_blocks=block_arrays.num_blocks
        )

    return initial_carry, user_config, run_params


def _run_mcmc_iterations(
    compiled_chunk,
    initial_carry,
    total_iterations: int,
    user_config: Dict[str, Any],
    block_specs: list,
    avg_time: Optional[float],
) -> Tuple[Any, float]:
    """
    Execute the main MCMC sampling loop.

    Args:
        compiled_chunk: Compiled JAX MCMC kernel
        initial_carry: Initial carry state
        total_iterations: Total iterations to run (burn + collect)
        user_config: User configuration dict
        block_specs: List of block specifications (for acceptance rate labels)
        avg_time: Estimated time per iteration (for progress estimate)

    Returns:
        final_carry: Final carry state after all iterations (arrays still on device)
        wall_time: Total wall clock time for sampling
    """
    if total_iterations <= 0:
        return initial_carry, 0.0

    print("\n--- MCMC RUN ---", flush = True)

    # Print time estimate if available
    if avg_time:
        compute_time_sec = avg_time * total_iterations
        finish_time = datetime.now() + timedelta(seconds=compute_time_sec)
        print(f"Estimated computation time: {timedelta(seconds=int(compute_time_sec))}")
        print(f"Estimated completion: {finish_time.strftime('%Y-%m-%d %I:%M:%S %p')}", flush=True)

    # Run iterations in chunks
    start_run_time = time.perf_counter()
    current_carry = initial_carry
    num_chunks = (total_iterations + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(num_chunks):
        current_carry, _ = compiled_chunk(current_carry)
        if i % max(1, num_chunks // 10) == 0:
            print(f"  Chunk {i+1}/{num_chunks}...", flush=True)

    jax.block_until_ready(current_carry)
    end_run_time = time.perf_counter()
    wall_time = end_run_time - start_run_time

    # Compute and display acceptance rates
    # Carry indices (15-element structure):
    #   7: acceptance_counts, 8: current_iteration, 9: temperature_ladder
    #   12: swap_accepts, 13: swap_attempts
    final_acceptance_counts = current_carry[7]
    total_iterations_run = current_carry[8]
    num_chains = user_config["num_chains"]
    total_attempts_per_block = total_iterations_run * num_chains
    acceptance_rates = final_acceptance_counts / total_attempts_per_block
    acceptance_rates_host = jax.device_get(acceptance_rates)

    print_acceptance_summary(block_specs, acceptance_rates_host)

    # Print swap acceptance rates for parallel tempering
    n_temperatures = user_config.get('n_temperatures', 1)
    if n_temperatures > 1:
        temperature_ladder = jax.device_get(current_carry[9])
        swap_accepts = jax.device_get(current_carry[12])
        swap_attempts = jax.device_get(current_carry[13])
        print_swap_acceptance_summary(temperature_ladder, swap_accepts, swap_attempts)

    print(f"\n--- MCMC Run Summary ---")
    print(f"  Total Wall Time: {timedelta(seconds=int(wall_time))} ({wall_time:.2f}s)")

    return current_carry, wall_time


def _transfer_to_host(
    final_carry,
    save_likelihoods: bool,
    n_temperatures: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Transfer history arrays from device to host.

    Args:
        final_carry: Final JAX carry tuple (15 elements) with arrays on device
        save_likelihoods: Whether to transfer likelihood history
        n_temperatures: Number of temperature levels (for deciding temp_history)

    Returns:
        host_history: Sample history array on host
        host_temp_history: Temperature history array (or None if not tempering)
        host_lik_history: Likelihood history (or None)
    """
    print("Transferring history to Host...", flush=True)
    host_history = jax.device_get(final_carry[4])

    # Transfer temperature history if using parallel tempering
    host_temp_history = None
    if n_temperatures > 1:
        print("Transferring temperature history to Host...", flush=True)
        host_temp_history = jax.device_get(final_carry[6])

    host_lik_history = None
    if save_likelihoods:
        print("Transferring likelihood history to Host...", flush=True)
        host_lik_history = jax.device_get(final_carry[5])

    return host_history, host_temp_history, host_lik_history


def _build_checkpoint(
    final_carry,
    posterior_id: str,
    user_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build checkpoint dict from final MCMC carry state.

    Args:
        final_carry: Final JAX carry tuple (15 elements with index process tempering)
        posterior_id: Posterior model identifier
        user_config: User configuration dict

    Returns:
        Checkpoint dict ready for saving

    Carry tuple indices (15 elements):
        0: states_A, 1: keys_A, 2: states_B, 3: keys_B
        4: history, 5: lik_history, 6: temp_history
        7: acceptance_counts, 8: current_iteration
        9: temperature_ladder, 10: temp_assignments_A, 11: temp_assignments_B
        12: swap_accepts, 13: swap_attempts, 14: swap_parity
    """
    # Run iteration (relative to this run's start)
    run_iteration = int(jax.device_get(final_carry[8]))  # Index 8 in 15-element tuple
    # Add offset to get global iteration (for resume tracking)
    iteration_offset = user_config.get('iteration_offset', 0)
    global_iteration = run_iteration + iteration_offset

    checkpoint = {
        'states_A': np.asarray(jax.device_get(final_carry[0])),
        'states_B': np.asarray(jax.device_get(final_carry[2])),
        'keys_A': np.asarray(jax.device_get(final_carry[1])),
        'keys_B': np.asarray(jax.device_get(final_carry[3])),
        'iteration': global_iteration,
        'acceptance_counts': np.asarray(jax.device_get(final_carry[7])),  # Index 7
        'posterior_id': posterior_id,
        'num_params': user_config['num_params'],
        'num_chains_a': user_config['num_chains_a'],
        'num_chains_b': user_config['num_chains_b'],
        'num_superchains': user_config.get('num_superchains', 0),
        'subchains_per_super': user_config.get('subchains_per_super', 0),
    }

    # Add tempering state if using parallel tempering (index process)
    n_temperatures = user_config.get('n_temperatures', 1)
    if n_temperatures > 1:
        checkpoint['n_temperatures'] = n_temperatures
        checkpoint['temperature_ladder'] = np.asarray(jax.device_get(final_carry[9]))
        checkpoint['temp_assignments_A'] = np.asarray(jax.device_get(final_carry[10]))
        checkpoint['temp_assignments_B'] = np.asarray(jax.device_get(final_carry[11]))
        checkpoint['swap_accepts'] = np.asarray(jax.device_get(final_carry[12]))
        checkpoint['swap_attempts'] = np.asarray(jax.device_get(final_carry[13]))
        checkpoint['swap_parity'] = int(jax.device_get(final_carry[14]))  # NEW: DEO parity

    return checkpoint


def _build_results(
    host_history: Optional[np.ndarray],
    host_temp_history: Optional[np.ndarray],
    host_lik_history: Optional[np.ndarray],
    nrhat_values: Optional[np.ndarray],
    user_config: Dict[str, Any],
    compile_time: float,
    wall_time: float,
    avg_time: Optional[float],
    total_iterations: int,
    final_iteration: int,
) -> Dict[str, Any]:
    """
    Build final results dict from sampling outputs.

    Args:
        host_history: Sample history array (all chains)
        host_temp_history: Temperature history array (or None if not tempering)
        host_lik_history: Likelihood history (or None)
        nrhat_values: R-hat values (or None)
        user_config: User configuration
        compile_time: Kernel compilation time
        wall_time: Total sampling wall time
        avg_time: Average time per iteration
        total_iterations: Total iterations run
        final_iteration: Final iteration number

    Returns:
        Results dict with all sampling outputs including temperature_history
    """
    # Build diagnostics dict
    diagnostics = {
        'rhat': nrhat_values,
        'K': user_config['num_superchains'],
        'M': user_config['subchains_per_super'],
        'compile_time': compile_time,
        'wall_time': wall_time,
        'avg_iter_time': avg_time,
        'total_iterations': total_iterations,
    }

    # Compute iteration numbers for each sample
    thin = user_config['thin_iteration']
    num_samples = host_history.shape[0] if host_history is not None else 0
    if num_samples > 0:
        start_iter = final_iteration - num_samples * thin
        iterations = start_iter + (np.arange(num_samples) + 1) * thin - 1
    else:
        iterations = np.array([], dtype=np.int64)

    return {
        'history': host_history,
        'temperature_history': host_temp_history,  # NEW: temperature index per chain per sample
        'iterations': iterations,
        'diagnostics': diagnostics,
        'mcmc_config': user_config,
        'likelihoods': host_lik_history,
        'K': user_config['num_superchains'],
        'M': user_config['subchains_per_super'],
        'thin_iteration': thin,
    }


# =============================================================================
# SINGLE-RUN SAMPLING FUNCTION
# =============================================================================

def rmcmc_single(
    mcmc_config: Dict[str, Any],
    data: Dict[str, Any],
    calculate_rhat: bool = True,
    resume_from: Optional[str] = None,
    reset_from: Optional[str] = None,
    reset_noise_scale: float = 1.0,
    save_initial_to: Optional[str] = None,
    init_from: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run a single MCMC sampling iteration with Unified Nested R-hat support.

    This is the low-level function for a single MCMC run. For multi-run workflows
    with automatic checkpoint management, use rmcmc() instead.

    Args:
        mcmc_config: MCMC configuration dict
        data: Data dict with 'static', 'int', 'float' arrays
        calculate_rhat: Whether to compute R-hat diagnostics
        resume_from: Optional path to checkpoint file to resume from (exact state)
        reset_from: Optional path to checkpoint file to reset from (cross-chain mean + noise)
        reset_noise_scale: Scale factor for noise when resetting (default 1.0 for full posterior spread)
        save_initial_to: Optional path to save initial state checkpoint (iteration=0) before sampling
        init_from: Optional numpy array with custom initial values (e.g., from prior samples).
                   Shape should be (K*M*n_params,) where K=num_superchains, M=subchains_per_super.
                   Use bamcmc.init_from_prior() to generate this from a prior checkpoint.

    Returns:
        results: Dict containing all sampling results:
            - history: Sample array (num_collect, num_chains, num_params + num_gq)
            - iterations: Iteration number for each sample
            - diagnostics: Dict with R-hat values and timing info
            - mcmc_config: Clean serializable config dict (no JAX types)
            - likelihoods: Likelihood history if SAVE_LIKELIHOODS=True, else None
            - K: Number of superchains
            - M: Subchains per superchain
            - thin_iteration: Thinning interval used
        checkpoint: Dict with final chain states for saving/resuming

    Notes:
        - resume_from: Continues sampling from exact checkpoint state
        - reset_from: Resets chains to cross-chain mean with small noise, useful for
          rescuing stuck/straggler chains while preserving learned posterior location
        - init_from: Uses provided array as initial values (e.g., from prior-only run)
        - save_initial_to: Saves the initial chain state before any iterations run,
          useful for tracing chains from their true starting point
    """
    # --- 1. VALIDATE CONFIGURATION ---
    print("Validating MCMC configuration...")
    try:
        validate_mcmc_config(mcmc_config)
        print("Configuration is valid\n")
    except ValueError as e:
        print(f"Invalid configuration:\n{e}")
        raise

    # --- 2. CONFIGURE SYSTEM ---
    user_config, runtime_ctx, model_ctx = configure_mcmc_system(mcmc_config, data)
    run_params = model_ctx['run_params']
    block_arrays = model_ctx['block_arrays']
    block_specs = model_ctx['block_specs']

    # Get posterior hash for benchmark caching
    posterior_id = user_config['posterior_id']
    posterior_hash = get_posterior_hash(
        posterior_id,
        model_ctx['model_config'],
        runtime_ctx['data'],
        user_config['num_chains']
    )
    benchmark_mgr = get_benchmark_manager()
    cached_benchmark = benchmark_mgr.get_cached_benchmark(posterior_hash)

    # Validate inputs
    print("Validating inputs...", flush=True)
    try:
        validate_mcmc_inputs(user_config, runtime_ctx['data'], block_specs)
        print("[OK] Validation passed", flush=True)
    except ValueError as e:
        print(f"[FAIL] Validation failed:\n{e}", flush=True)
        raise

    print(f"Starting sampling for {posterior_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # --- 3. INITIALIZE CHAINS ---
    initial_carry, user_config, run_params = _initialize_chains(
        resume_from=resume_from,
        reset_from=reset_from,
        reset_noise_scale=reset_noise_scale,
        user_config=user_config,
        runtime_ctx=runtime_ctx,
        model_ctx=model_ctx,
        run_params=run_params,
        block_arrays=block_arrays,
        init_from=init_from,
    )

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Likelihood Saving: {'ENABLED' if user_config['save_likelihoods'] else 'DISABLED'}")

    # --- 3b. SAVE INITIAL STATE (if requested) ---
    if save_initial_to is not None:
        initial_checkpoint = _build_checkpoint(initial_carry, posterior_id, user_config)
        np.savez_compressed(save_initial_to, **initial_checkpoint)
        print(f"Initial state saved: {save_initial_to} (iteration {initial_checkpoint['iteration']})")

    # --- 4. COMPILE KERNEL ---
    compiled_chunk, compile_time = compile_mcmc_kernel(
        user_config, runtime_ctx, block_arrays, run_params, initial_carry
    )

    # --- 5. HANDLE BENCHMARKING ---
    benchmark_iters = user_config.get('benchmark', 0)
    avg_time = None

    if cached_benchmark is not None:
        # Use cached benchmark values
        cached_results = cached_benchmark.get('results', {})
        avg_time = cached_results.get('iteration_time_s')
        if avg_time:
            print(f"\n--- Using Cached Benchmark (hash: {posterior_hash[:8]}...) ---")
            print(f"  Per-iteration: {avg_time:.4f}s (from cache)")
            cached_git = cached_benchmark.get('git', {})
            if cached_git:
                print(f"  Cached from: {cached_git.get('branch', '?')}@{cached_git.get('commit', '?')}")
    elif benchmark_iters > 0:
        # Run benchmark and save results
        bench_results = benchmark_mcmc_sampler(compiled_chunk, initial_carry, benchmark_iters)
        avg_time = bench_results['avg_time']

        benchmark_mgr.save_benchmark(
            posterior_hash=posterior_hash,
            posterior_id=posterior_id,
            num_chains=user_config['num_chains'],
            fresh_compile_time=compile_time,
            iteration_time=avg_time,
            benchmark_iterations=benchmark_iters,
        )
        print(f"  Benchmark saved (hash: {posterior_hash[:8]}...)")

    # --- 6. RUN MCMC ITERATIONS ---
    total_iterations = run_params.BURN_ITER + user_config["num_iterations"]

    # Free any intermediate objects to reduce GPU fragmentation
    gc.collect()

    if total_iterations > 0 and run_params.NUM_COLLECT > 0:
        final_carry, wall_time = _run_mcmc_iterations(
            compiled_chunk=compiled_chunk,
            initial_carry=initial_carry,
            total_iterations=total_iterations,
            user_config=user_config,
            block_specs=block_specs,
            avg_time=avg_time,
        )

        # --- 7. COMPUTE R-HAT (on device) ---
        nrhat_values = None
        if calculate_rhat:
            nrhat_values = compute_and_print_rhat(final_carry[4], user_config)

        # --- 8. TRANSFER TO HOST ---
        n_temperatures = user_config.get('n_temperatures', 1)
        host_history, host_temp_history, host_lik_history = _transfer_to_host(
            final_carry, user_config['save_likelihoods'], n_temperatures
        )
    else:
        # No iterations to run
        print("\nSkipping main run.", flush=True)
        final_carry = initial_carry
        wall_time = 0.0
        nrhat_values = None
        n_temperatures = user_config.get('n_temperatures', 1)
        host_history, host_temp_history, host_lik_history = _transfer_to_host(
            initial_carry, user_config['save_likelihoods'], n_temperatures
        )

    # --- 9. POST-RUN DIAGNOSTICS ---
    # Build base diagnostics dict
    diagnostics = {
        'rhat': nrhat_values,
        'K': user_config['num_superchains'],
        'M': user_config['subchains_per_super'],
        'compile_time': compile_time,
        'wall_time': wall_time,
        'avg_iter_time': avg_time,
        'total_iterations': total_iterations,
    }

    print("\n--- Post-Run Diagnostics ---")
    diagnostics = diagnose_sampler_issues(host_history, user_config, diagnostics)
    print_diagnostics(diagnostics)

    if diagnostics['issues']:
        print("\n  Warning: Issues detected during sampling!")
        print("Review diagnostics above before using results.")

    # --- 10. BUILD OUTPUTS ---
    final_checkpoint = _build_checkpoint(final_carry, posterior_id, user_config)
    final_iteration = final_checkpoint['iteration']

    results = _build_results(
        host_history=host_history,
        host_temp_history=host_temp_history,
        host_lik_history=host_lik_history,
        nrhat_values=nrhat_values,
        user_config=user_config,
        compile_time=compile_time,
        wall_time=wall_time,
        avg_time=avg_time,
        total_iterations=total_iterations,
        final_iteration=final_iteration,
    )

    # Post-run diagnostics are already in the results diagnostics dict
    results['diagnostics'] = diagnostics

    return results, final_checkpoint


# =============================================================================
# MULTI-RUN SAMPLING FUNCTION
# =============================================================================

from typing import List

def rmcmc(
    mcmc_config: Dict[str, Any],
    data: Dict[str, Any],
    output_dir: str,
    run_schedule: List[Tuple[str, int]],
    calculate_rhat: bool = True,
    burn_in_fresh: bool = True,
    reset_noise_scale: float = 1.0,
    use_nested_structure: bool = True,
) -> Dict[str, Any]:
    """
    Run MCMC sampling with automatic multi-run and checkpoint management.

    This is the main entry point for MCMC sampling. It handles:
    - Automatic detection of existing checkpoints
    - Multiple sequential runs with resume/reset modes
    - Fresh runs when no checkpoint exists
    - Optional burn-in only for fresh runs

    Args:
        mcmc_config: MCMC configuration dict (see rmcmc_single for details)
        data: Data dict with 'static', 'int', 'float' arrays
        output_dir: Directory for checkpoint and history files
        run_schedule: List of (mode, count) tuples specifying the run sequence.
            Modes:
            - "reset": Reset chains to cross-chain mean + noise (or fresh if no checkpoint)
            - "resume": Resume from exact checkpoint state (or fresh if no checkpoint)
            Examples:
            - [("reset", 10)]: 10 reset runs (first is fresh if no checkpoint)
            - [("resume", 5)]: 5 resume runs (first is fresh if no checkpoint)
            - [("reset", 10), ("resume", 5)]: 10 reset runs then 5 resume runs
        calculate_rhat: Whether to compute R-hat diagnostics each run
        burn_in_fresh: If True, only apply burn_iter to fresh runs (default True)
        reset_noise_scale: Scale factor for noise when resetting (default 1.0)
        use_nested_structure: If True, use nested directory structure (default True):
            {output_dir}/{model}/checkpoints/ and {output_dir}/{model}/history/full/

    Returns:
        Summary dict with:
            - output_dir: Output directory path
            - model_name: Model name from config
            - checkpoint_files: List of (run_index, path) tuples
            - history_files: List of (run_index, path) tuples
            - latest_run_index: Index of the last completed run
            - latest_checkpoint: Path to the most recent checkpoint
            - total_runs_completed: Number of runs completed in this session
            - final_iteration: Final iteration number
            - run_log: List of dicts with info about each run

    Example:
        from bamcmc import rmcmc

        summary = rmcmc(
            mcmc_config,
            data,
            output_dir='./output',
            run_schedule=[("reset", 10), ("resume", 5)],
        )
        print(f"Completed {summary['total_runs_completed']} runs")
        print(f"History files: {summary['history_files']}")
    """
    from pathlib import Path
    from ..checkpoint_helpers import ensure_model_dirs, get_model_paths

    model_name = mcmc_config['posterior_id']

    # Set up directory structure and path constructors
    if use_nested_structure:
        # Create nested directories: {output_dir}/{model}/checkpoints/, history/full/
        paths = ensure_model_dirs(output_dir, model_name)
        checkpoint_dir = paths['checkpoints']
        history_dir = paths['history_full']

        def checkpoint_path(num):
            return str(checkpoint_dir / f"checkpoint_{num:03d}.npz")

        def history_path(num):
            return str(history_dir / f"history_{num:03d}.npz")
    else:
        # Legacy flat structure
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        def checkpoint_path(num):
            return f"{output_dir}/{model_name}_checkpoint{num}.npz"

        def history_path(num):
            return f"{output_dir}/{model_name}_history_{num:03d}.npz"

    # Scan for existing checkpoints
    scan = scan_checkpoints(output_dir, model_name)
    existing_run_index = scan['latest_run_index']
    existing_checkpoint = scan['latest_checkpoint']

    if existing_checkpoint:
        print(f"Found existing checkpoint: {existing_checkpoint} (run {existing_run_index})")
    else:
        print(f"No existing checkpoints found for {model_name}")

    # Expand run schedule into list of (run_index, mode) pairs
    run_list = []
    for mode, count in run_schedule:
        if mode not in ('reset', 'resume'):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'reset' or 'resume'.")
        for _ in range(count):
            run_list.append(mode)

    if not run_list:
        print("Empty run schedule, nothing to do")
        return {
            'output_dir': output_dir,
            'model_name': model_name,
            **scan,
            'total_runs_completed': 0,
            'final_iteration': None,
            'run_log': [],
        }

    # Determine starting run index
    start_run_index = existing_run_index + 1
    total_runs = len(run_list)

    # Format schedule for display
    schedule_parts = [f"{count} {mode}" for mode, count in run_schedule]
    schedule_str = " + ".join(schedule_parts)

    print(f"\n{'='*60}")
    print(f"MULTI-RUN MCMC: {model_name}")
    print(f"Schedule: {schedule_str} ({total_runs} total runs)")
    print(f"Starting from run index {start_run_index}")
    print(f"{'='*60}\n")

    # Track run information
    run_log = []
    total_runs_completed = 0
    final_iteration = None
    original_burn_iter = mcmc_config.get('burn_iter', 0)

    # New indexing scheme:
    # - checkpoint_0 = initial state (iteration=0), saved for fresh/reset runs
    # - checkpoint_N = state after Nth sampling run
    # - run_index in schedule = which sampling run (1-indexed for checkpoints)

    for i, scheduled_mode in enumerate(run_list):
        sampling_run_num = start_run_index + i  # 0-indexed sampling run number
        output_checkpoint_num = sampling_run_num + 1  # checkpoint_1 is after 1st sampling run

        # Check if this run already completed (output checkpoint exists)
        output_checkpoint_file = checkpoint_path(output_checkpoint_num)
        if Path(output_checkpoint_file).exists():
            print(f"\n--- Run {i + 1}/{total_runs} (overall #{sampling_run_num + 1}) already complete (checkpoint_{output_checkpoint_num:03d} exists), skipping ---")
            run_log.append({
                'run_index': sampling_run_num,
                'mode': 'skipped',
                'checkpoint': output_checkpoint_file,
            })
            continue

        # Determine input checkpoint (what to resume/reset from)
        input_checkpoint_num = output_checkpoint_num - 1  # checkpoint_0 for first run
        input_checkpoint_file = checkpoint_path(input_checkpoint_num)
        has_input_checkpoint = Path(input_checkpoint_file).exists()

        # If input checkpoint doesn't exist, search for latest available
        if not has_input_checkpoint and input_checkpoint_num > 0:
            for idx in range(input_checkpoint_num - 1, -1, -1):
                cp = checkpoint_path(idx)
                if Path(cp).exists():
                    input_checkpoint_file = cp
                    has_input_checkpoint = True
                    print(f"  Note: Using checkpoint_{idx:03d} (checkpoint_{input_checkpoint_num:03d} not found)")
                    break

        # Determine mode and whether we need to save initial state
        save_initial_to = None
        if not has_input_checkpoint:
            actual_mode = 'fresh'
            resume_from = None
            reset_from = None
            # Save initial state as checkpoint_000
            save_initial_to = checkpoint_path(0)
        elif scheduled_mode == 'resume':
            actual_mode = 'resume'
            resume_from = input_checkpoint_file
            reset_from = None
        else:  # reset
            actual_mode = 'reset'
            resume_from = None
            reset_from = input_checkpoint_file
            # Save initial (reset) state before sampling
            # For nested structure, just overwrite the checkpoint; for flat, add _reset suffix
            if use_nested_structure:
                save_initial_to = None  # Don't save reset state separately in nested mode
            else:
                save_initial_to = f"{output_dir}/{model_name}_checkpoint{input_checkpoint_num}_reset.npz"

        print(f"\n{'='*60}")
        print(f"Sampling run {i + 1}/{total_runs} (overall #{sampling_run_num + 1}, mode={actual_mode})")
        print(f"  Output: checkpoint_{output_checkpoint_num}")
        print(f"{'='*60}")

        # Handle burn_in_fresh option
        run_config = mcmc_config.copy()
        if burn_in_fresh and actual_mode != 'fresh':
            run_config['burn_iter'] = 0
            if original_burn_iter > 0:
                print(f"  burn_in_fresh=True: skipping burn-in ({actual_mode} run)")
        else:
            run_config['burn_iter'] = original_burn_iter

        # Run single iteration
        try:
            results, checkpoint = rmcmc_single(
                run_config,
                data,
                calculate_rhat=calculate_rhat,
                resume_from=resume_from,
                reset_from=reset_from,
                reset_noise_scale=reset_noise_scale,
                save_initial_to=save_initial_to,
            )
        except Exception as e:
            print(f"\nError during run {i + 1}/{total_runs} (overall #{sampling_run_num + 1}): {e}")
            raise

        # Save output checkpoint
        np.savez_compressed(output_checkpoint_file, **checkpoint)
        final_iteration = checkpoint['iteration']
        print(f"\nCheckpoint saved: {output_checkpoint_file} (iteration {final_iteration})")

        # Save history if we collected samples
        history = results.get('history')
        history_file = None
        if history is not None and len(history) > 0:
            history_file = history_path(sampling_run_num)
            np.savez_compressed(history_file, **results, run_index=sampling_run_num)
            print(f"History saved: {history_file}")

        run_log.append({
            'run_index': sampling_run_num,
            'mode': actual_mode,
            'checkpoint': output_checkpoint_file,
            'initial_checkpoint': save_initial_to,
            'history': history_file,
            'iteration': final_iteration,
            'rhat_max': np.nanmax(results['diagnostics'].get('rhat', [np.nan])),
        })
        total_runs_completed += 1

        # Clean up memory between runs
        gc.collect()

    print(f"\n{'='*60}")
    print(f"MULTI-RUN COMPLETE: {total_runs_completed} runs")
    print(f"{'='*60}\n")

    # Scan for final state
    final_scan = scan_checkpoints(output_dir, model_name)

    return {
        'output_dir': output_dir,
        'model_name': model_name,
        **final_scan,
        'total_runs_completed': total_runs_completed,
        'final_iteration': final_iteration,
        'run_log': run_log,
    }
