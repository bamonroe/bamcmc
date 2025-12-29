"""
MCMC Configuration and Initialization.

This module handles setting up and validating MCMC configurations:
- configure_mcmc_system: Main configuration entry point
- initialize_mcmc_system: Initialize chain states and data structures
- validate_mcmc_inputs: Validate inputs before sampling
- gen_rng_keys: Generate JAX random keys
"""

import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
from typing import List, Dict, Tuple, Any

from .registry import get_posterior
from .mcmc_utils import clean_config
from .batch_specs import BlockSpec, validate_block_specs
from .mcmc_types import BlockArrays, RunParams, build_block_arrays


def gen_rng_keys(mcmc_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate and store JAX random keys in config."""
    rng_seed = mcmc_config.setdefault("rng_seed", 42)
    mkey = jax.random.PRNGKey(rng_seed)
    mcmc_config["master_key"], mcmc_config["init_key"] = random.split(mkey, 2)
    return mcmc_config


def validate_mcmc_inputs(config: Dict[str, Any], data: Dict[str, Any], specs: List[BlockSpec]) -> bool:
    """Validate MCMC inputs before starting sampling."""
    errors = []

    if 'BURN_ITER' in config and config['BURN_ITER'] < 0:
        errors.append(f"BURN_ITER must be >= 0, got {config['BURN_ITER']}")

    if 'SAMPLE_ITER' in config and config['SAMPLE_ITER'] < 1:
        errors.append(f"SAMPLE_ITER must be >= 1, got {config['SAMPLE_ITER']}")

    if 'NUM_CHAINS' in config:
        if config['NUM_CHAINS'] < 1:
            errors.append(f"NUM_CHAINS must be >= 1, got {config['NUM_CHAINS']}")
        if config['NUM_CHAINS'] % 2 != 0:
            errors.append(f"NUM_CHAINS must be even for parallel groups, got {config['NUM_CHAINS']}")

    if specs:
        total_params = sum(spec.size for spec in specs)
        if total_params < 1:
            errors.append(f"Total parameters must be >= 1, got {total_params}")

    if errors:
        error_msg = "MCMC Input Validation Failed:\n  " + "\n  ".join(errors)
        raise ValueError(error_msg)

    return True


def configure_mcmc_system(
    mcmc_config: Dict[str, Any],
    data: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Configure the MCMC system from config and data.

    Returns:
        mcmc_config: Updated configuration dict
        data: Data with arrays converted to JAX
        model_context: Dict with log_posterior_fn, blocks, run_params, etc.
    """
    mcmc_config = clean_config(mcmc_config)
    mcmc_config.setdefault('SAVE_LIKELIHOODS', False)

    # JAX persistent cache is configured via environment variables in jax_config.py
    # JAX_COMPILATION_CACHE_DIR -> ~/.cache/jax/bamcmc_cache/

    use_double = mcmc_config.get('USE_DOUBLE')
    posterior_id = mcmc_config.get('POSTERIOR_ID')

    num_chains_a = mcmc_config.get('NUM_CHAINS_A')
    num_chains_b = mcmc_config.get('NUM_CHAINS_B')
    mcmc_config["NUM_CHAINS"] = num_chains_a + num_chains_b

    thin_iteration = mcmc_config.get('THIN_ITERATION')
    num_collect = mcmc_config.get('NUM_COLLECT')
    burn_iter = mcmc_config.get('BURN_ITER')
    mcmc_config["NUM_ITERATIONS"] = thin_iteration * num_collect

    if use_double:
        jax.config.update("jax_enable_x64", True)
        jnp_float_dtype = jnp.float64
        jnp_int_dtype   = jnp.int64
    else:
        jax.config.update("jax_enable_x64", False)
        jnp_float_dtype = jnp.float32
        jnp_int_dtype   = jnp.int32

    mcmc_config["jnp_float_dtype"] = jnp_float_dtype
    mcmc_config["jnp_int_dtype"]   = jnp_int_dtype

    data["int"]   = tuple(jnp.asarray(i, dtype=jnp_int_dtype)   for i in data["int"])
    data["float"] = tuple(jnp.asarray(i, dtype=jnp_float_dtype) for i in data["float"])

    model_config = get_posterior(posterior_id)
    log_posterior_fn  = partial(model_config['log_posterior'], data=data)
    direct_sampler_fn = partial(model_config['direct_sampler'], data=data)

    init_vec_fn = model_config.get('initial_vector')
    if init_vec_fn:
        initial_vector_fn = partial(init_vec_fn, data=data)
    else:
        raise ValueError(f"No 'initial_vector' function defined for {posterior_id}")

    gq_fn = None
    num_gq = 0
    if model_config.get('generated_quantities'):
        gq_fn = partial(model_config['generated_quantities'], data=data)
        num_gq = model_config['get_num_gq'](mcmc_config, data)

    raw_batch_specs = model_config['batch_type'](mcmc_config, data)

    if not isinstance(raw_batch_specs[0], BlockSpec):
        raise TypeError(f"Posterior '{posterior_id}' must return BlockSpec objects.")

    validate_block_specs(raw_batch_specs, posterior_id)

    # Build unified BlockArrays structure
    block_arrays = build_block_arrays(raw_batch_specs)

    # Create RunParams as frozen dataclass for JAX static argument compatibility
    run_params = RunParams(
        BURN_ITER=burn_iter,
        NUM_COLLECT=num_collect,
        THIN_ITERATION=thin_iteration,
        NUM_GQ=num_gq,
        START_ITERATION=0,  # Set to checkpoint iteration when resuming
        SAVE_LIKELIHOODS=mcmc_config['SAVE_LIKELIHOODS'],
    )

    model_context = {
        'log_posterior_fn': log_posterior_fn,
        'direct_sampler_fn': direct_sampler_fn,
        'initial_vector_fn': initial_vector_fn,
        'generated_quantities_fn': gq_fn,
        'block_arrays': block_arrays,
        'block_specs': raw_batch_specs,  # Keep for labels/debugging
        'run_params': run_params,
        'model_config': model_config,  # For posterior hash computation
    }

    return mcmc_config, data, model_context


def initialize_mcmc_system(initial_vector_np, mcmc_config, num_gq, num_collect, num_blocks):
    """
    Initialize MCMC system using the Unified Nested R-hat structure.

    Algorithm:
    1. Determine K (Superchains) and M (Subchains).
       - If K is not provided, K = Total Chains (implies M=1, Standard R-hat).
    2. Extract K distinct initial states from the provided initialization vector.
    3. Replicate each of these K states M times to form the full population.
       (If M=1, this is an identity operation).
    """
    num_chains = mcmc_config["NUM_CHAINS"]
    initial_vector = jnp.asarray(initial_vector_np, dtype = mcmc_config["jnp_float_dtype"])
    num_params = initial_vector.size // num_chains
    mcmc_config["num_params"] = num_params

    # Reshape linear vector to (NumChains, NumParams)
    all_initial_states = initial_vector.reshape(num_chains, num_params)

    # --- UNIFIED NESTED INIT STRATEGY ---
    # Default: 1 Superchain per Chain (Standard R-hat mode)
    K = mcmc_config.get('NUM_SUPERCHAINS', num_chains)

    # Validation
    if num_chains % K != 0:
        raise ValueError(
            f"NUM_CHAINS ({num_chains}) must be divisible by NUM_SUPERCHAINS ({K})"
        )

    M = num_chains // K
    mcmc_config['NUM_SUPERCHAINS'] = K
    mcmc_config['SUBCHAINS_PER_SUPER'] = M

    if M > 1:
        print(f"Structure: {K} Superchains x {M} Subchains (Nested R-hat Mode)")
        # Take the first K distinct states as the 'roots' for the superchains
        base_states = all_initial_states[:K]  # (K, num_params)

        # Replicate roots M times to form the full population of starting points
        # This ensures all M subchains in a group start at the exact same point
        all_initial_states = jnp.repeat(base_states, M, axis=0)
    else:
        print(f"Structure: {K} Independent Chains (Standard R-hat Mode)")
        # If M=1, we just use the K (which equals num_chains) states directly.
        pass

    # Split for Red-Black sampler
    all_keys_for_each_chain = random.split(mcmc_config["init_key"], num_chains)
    initial_states_A, initial_states_B = jnp.split(all_initial_states,  [mcmc_config["NUM_CHAINS_A"]], axis=0)
    initial_keys_A, initial_keys_B = jnp.split(all_keys_for_each_chain, [mcmc_config["NUM_CHAINS_A"]], axis=0)

    total_cols = num_params + num_gq
    initial_history = jnp.empty((num_collect, num_chains, total_cols), dtype = mcmc_config["jnp_float_dtype"])

    if mcmc_config['SAVE_LIKELIHOODS']:
        # 2D array (Iter, Chain) - Summed over blocks
        initial_lik_history = jnp.empty((num_collect, num_chains), dtype=mcmc_config["jnp_float_dtype"])
    else:
        initial_lik_history = jnp.empty((1,), dtype=mcmc_config["jnp_float_dtype"])

    acceptance_counts = jnp.zeros(num_blocks, dtype=jnp.int32)
    current_iteration = jnp.array(0, dtype=jnp.int32)

    initial_carry = (initial_states_A, initial_keys_A, initial_states_B, initial_keys_B,
                     initial_history, initial_lik_history, acceptance_counts, current_iteration)

    return initial_carry, mcmc_config
