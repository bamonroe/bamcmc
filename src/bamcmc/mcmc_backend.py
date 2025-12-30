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

import jax
import jax.numpy as jnp
import numpy as np
import time
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

# Import from submodules
from .mcmc_types import BlockArrays, RunParams, build_block_arrays
from .mcmc_config import (
    configure_mcmc_system,
    initialize_mcmc_system,
    validate_mcmc_inputs,
    gen_rng_keys,
)
from .mcmc_compile import (
    compile_mcmc_kernel,
    benchmark_mcmc_sampler,
    CHUNK_SIZE,
)
from .mcmc_diagnostics import compute_nested_rhat

# Import batch specs for sampler types
from .batch_specs import SamplerType

# Import error handling for validation
from .error_handling import validate_mcmc_config, diagnose_sampler_issues, print_diagnostics

# Import checkpoint utilities
from .checkpoint_helpers import (
    load_checkpoint,
    initialize_from_checkpoint,
)

# Import posterior benchmarking system
from .posterior_benchmark import (
    get_manager as get_benchmark_manager,
    get_posterior_hash,
)

# Re-export for backward compatibility
__all__ = [
    'rmcmc',
    'run_benchmark',
    'BlockArrays',
    'RunParams',
    'build_block_arrays',
    'configure_mcmc_system',
    'initialize_mcmc_system',
    'validate_mcmc_inputs',
    'compute_nested_rhat',
]


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

    # Compute posterior hash
    posterior_id = user_config['posterior_id']
    posterior_hash = get_posterior_hash(
        posterior_id,
        model_ctx['model_config'],
        runtime_ctx['data']
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
    bench_results = benchmark_mcmc_sampler(compiled_chunk, initial_carry, benchmark_iterations)
    iteration_time = bench_results['avg_time']

    # Compare and optionally update cache
    benchmark_mgr = get_benchmark_manager()
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


def rmcmc(
    mcmc_config: Dict[str, Any],
    data: Dict[str, Any],
    calculate_rhat: bool = True,
    resume_from: Optional[str] = None,
    reset_from: Optional[str] = None,
    reset_noise_scale: float = 0.1
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run MCMC sampling with Unified Nested R-hat support.

    Args:
        mcmc_config: MCMC configuration dict
        data: Data dict with 'static', 'int', 'float' arrays
        calculate_rhat: Whether to compute R-hat diagnostics
        resume_from: Optional path to checkpoint file to resume from (exact state)
        reset_from: Optional path to checkpoint file to reset from (cross-chain mean + noise)
        reset_noise_scale: Scale factor for noise when resetting (default 0.1)

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
    """

    # Validate configuration before running
    print("Validating MCMC configuration...")
    try:
        validate_mcmc_config(mcmc_config)
        print("Configuration is valid\n")
    except ValueError as e:
        print(f"Invalid configuration:\n{e}")
        raise

    user_config, runtime_ctx, model_ctx = configure_mcmc_system(mcmc_config, data)
    run_params = model_ctx['run_params']

    block_arrays = model_ctx['block_arrays']
    block_specs = model_ctx['block_specs']

    # Compute posterior hash for persistent benchmarking
    posterior_id = user_config['posterior_id']
    posterior_hash = get_posterior_hash(
        posterior_id,
        model_ctx['model_config'],
        runtime_ctx['data']
    )
    benchmark_mgr = get_benchmark_manager()
    cached_benchmark = benchmark_mgr.get_cached_benchmark(posterior_hash)

    print("Validating inputs...", flush=True)
    try:
        validate_mcmc_inputs(user_config, runtime_ctx['data'], block_specs)
        print("✓ Validation passed", flush=True)
    except ValueError as e:
        print(f"✗ Validation failed:\n{e}", flush=True)
        raise

    print(f"Starting sampling for {posterior_id}...", flush=True)

    # Initialize either from checkpoint, reset, or fresh
    if resume_from is not None and reset_from is not None:
        raise ValueError("Cannot specify both resume_from and reset_from")

    if resume_from is not None:
        print(f"Loading checkpoint from {resume_from}...")
        checkpoint = load_checkpoint(resume_from)
        initial_carry, user_config = initialize_from_checkpoint(
            checkpoint,
            user_config,
            runtime_ctx,
            num_gq=run_params.NUM_GQ,
            num_collect=run_params.NUM_COLLECT,
            num_blocks=block_arrays.num_blocks
        )
        # Set START_ITERATION for resumed runs so collection indices are computed correctly
        # Use replace() since RunParams is frozen
        run_params = replace(run_params, START_ITERATION=checkpoint['iteration'])

    elif reset_from is not None:
        # Reset chains to cross-chain mean with noise
        from .reset_utils import generate_reset_vector
        print(f"Resetting chains from checkpoint {reset_from}...")

        checkpoint = load_checkpoint(reset_from)
        n_subjects = runtime_ctx['data']['static'][0]  # Number of subjects from data
        K = user_config.get('num_superchains', user_config['num_chains'])
        M = user_config['num_chains'] // K

        print(f"  Source iteration: {checkpoint['iteration']}")
        print(f"  Generating {K} reset starting points (noise_scale={reset_noise_scale})")

        initial_vector_np = generate_reset_vector(
            checkpoint,
            model_type=posterior_id,
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
        # Reset starts fresh (iteration 0), not from checkpoint iteration
        # run_params already has START_ITERATION=0 from configure_mcmc_system
        print(f"  Reset complete - starting fresh from iteration 0")

    else:
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

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Likelihood Saving: {'ENABLED' if user_config['save_likelihoods'] else 'DISABLED'}")

    # Compile MCMC kernel
    compiled_chunk, compile_time = compile_mcmc_kernel(
        user_config, runtime_ctx, block_arrays, run_params, initial_carry
    )

    # --- BENCHMARKING ---
    # Check for cached benchmark first, then run if needed
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

        # Save benchmark for future use
        benchmark_mgr.save_benchmark(
            posterior_hash=posterior_hash,
            posterior_id=posterior_id,
            num_chains=user_config['num_chains'],
            fresh_compile_time=compile_time,
            iteration_time=avg_time,
            benchmark_iterations=benchmark_iters,
        )
        print(f"  Benchmark saved (hash: {posterior_hash[:8]}...)")

    total_iterations_to_run = run_params.BURN_ITER + user_config["num_iterations"]
    host_history = None
    host_lik_history = None
    end_run_time = 0.0

    if total_iterations_to_run > 0 and run_params.NUM_COLLECT > 0:
        print("\n--- MCMC RUN ---")
        if avg_time:
            compute_time_sec = avg_time * total_iterations_to_run
            finish_time = datetime.now() + timedelta(seconds=compute_time_sec)
            print(f"Estimated computation time: {timedelta(seconds=int(compute_time_sec))}")
            print(f"Estimated completion: {finish_time.strftime('%Y-%m-%d %I:%M:%S %p')}", flush=True)

        start_run_time = time.perf_counter()
        current_carry = initial_carry
        num_chunks = (total_iterations_to_run + CHUNK_SIZE - 1) // CHUNK_SIZE

        for i in range(num_chunks):
            current_carry, _ = compiled_chunk(current_carry)
            if i % max(1, num_chunks // 10) == 0:
                print(f"  Chunk {i+1}/{num_chunks}...", flush=True)

        jax.block_until_ready(current_carry)
        end_run_time = time.perf_counter()

        final_history_device = current_carry[4]
        final_lik_device = current_carry[5]
        final_acceptance_counts = current_carry[6]

        total_iterations = current_carry[7]
        num_chains = user_config["num_chains"]

        total_attempts_per_block = total_iterations * num_chains
        acceptance_rates = final_acceptance_counts / total_attempts_per_block

        acceptance_rates_host = jax.device_get(acceptance_rates)

        print("\n--- Acceptance Rates by Block ---")
        for i, (spec, rate) in enumerate(zip(block_specs, acceptance_rates_host)):
            label = spec.label if spec.label else f"Block {i}"
            if spec.sampler_type == SamplerType.METROPOLIS_HASTINGS:
                print(f"  {label}: {rate:.1%} ({spec.proposal_type})")
            else:
                print(f"  {label}: N/A (Direct sampler)")

        nrhat_values = None
        if calculate_rhat and final_history_device.shape[1] > 1:
            print("\n--- Computing Unified Nested R-hat (GPU) ---")
            rhat_start = time.perf_counter()

            K = user_config['num_superchains']
            M = user_config['subchains_per_super']

            # Use unified function for both Nested and Standard cases
            nrhat_device = compute_nested_rhat(final_history_device, K, M)
            nrhat_values = jax.device_get(nrhat_device)

            rhat_end = time.perf_counter()
            print(f"Diagnostics complete in {rhat_end - rhat_start:.4f}s")

            # Display diagnostics
            # Threshold logic from Margossian et al. (2022)
            tau = 1e-4
            threshold = jnp.sqrt(1 + 1/M + tau)

            if M > 1:
                label = f"Nested R̂ ({K}x{M})"
            else:
                label = f"Standard R̂ ({K} chains)"

            print(f"\n--- {label} Results ---")
            print(f"  Max: {jnp.max(nrhat_values):.4f}")
            print(f"  Median: {jnp.median(nrhat_values):.4f}")
            print(f"  Threshold: {threshold:.4f} (tau={tau:.0e})")

            if jnp.max(nrhat_values) < threshold:
                print(f"  Converged (max < {threshold:.4f})")
            else:
                print(f"  Not Converged (max = {jnp.max(nrhat_values):.4f} >= {threshold:.4f})")

        print("Transferring history to Host...", flush=True)
        host_history = jax.device_get(final_history_device)

        if user_config['save_likelihoods']:
            print("Transferring likelihood history to Host...", flush=True)
            host_lik_history = jax.device_get(final_lik_device)

        print("\n--- MCMC Run Summary ---")
        print(f"  Total Wall Time: {end_run_time - start_run_time:.4f} s")
    else:
        print("\nSkipping main run.", flush=True)
        host_history = jax.device_get(initial_carry[4])
        if user_config['save_likelihoods']:
             host_lik_history = jax.device_get(initial_carry[5])
        nrhat_values = None
        current_carry = initial_carry
        start_run_time = 0.0

    # Unified diagnostics return
    diagnostics = {
        'rhat': nrhat_values,        # Primary metric (Nested or Standard)
        'K': user_config['num_superchains'],
        'M': user_config['subchains_per_super'],
        # Timing info for benchmarking
        'compile_time': compile_time,
        'wall_time': end_run_time - start_run_time if total_iterations_to_run > 0 else 0.0,
        'avg_iter_time': avg_time,
        'total_iterations': total_iterations_to_run,
    }

    # --- 4. POST-RUN DIAGNOSTICS ---
    print("\n--- Post-Run Diagnostics ---")
    diagnostics = diagnose_sampler_issues(host_history, user_config, diagnostics)
    print_diagnostics(diagnostics)

    if diagnostics['issues']:
        print("\n  Warning: Issues detected during sampling!")
        print("Review diagnostics above before using results.")

    # Build checkpoint from final carry
    final_carry = current_carry if total_iterations_to_run > 0 else initial_carry
    final_iteration = int(jax.device_get(final_carry[7]))

    final_checkpoint = {
        'states_A': np.asarray(jax.device_get(final_carry[0])),
        'states_B': np.asarray(jax.device_get(final_carry[2])),
        'keys_A': np.asarray(jax.device_get(final_carry[1])),
        'keys_B': np.asarray(jax.device_get(final_carry[3])),
        'iteration': final_iteration,
        'acceptance_counts': np.asarray(jax.device_get(final_carry[6])),
        'posterior_id': posterior_id,
        'num_params': user_config['num_params'],
        'num_chains_a': user_config['num_chains_a'],
        'num_chains_b': user_config['num_chains_b'],
        'num_superchains': user_config.get('num_superchains', 0),
        'subchains_per_super': user_config.get('subchains_per_super', 0),
    }

    # Compute iteration numbers for each sample
    thin = user_config['thin_iteration']
    num_samples = host_history.shape[0] if host_history is not None else 0
    if num_samples > 0:
        start_iter = final_iteration - num_samples * thin
        iterations = start_iter + (np.arange(num_samples) + 1) * thin - 1
    else:
        iterations = np.array([], dtype=np.int64)

    # Build results dict with clean serializable config
    results = {
        'history': host_history,
        'iterations': iterations,
        'diagnostics': diagnostics,
        'mcmc_config': user_config,  # Clean, no JAX types
        'likelihoods': host_lik_history,
        'K': user_config['num_superchains'],
        'M': user_config['subchains_per_super'],
        'thin_iteration': thin,
    }

    return results, final_checkpoint
