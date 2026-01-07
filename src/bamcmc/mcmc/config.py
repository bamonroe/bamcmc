"""
MCMC Configuration and Initialization.

This module handles setting up and validating MCMC configurations:
- configure_mcmc_system: Main configuration entry point
- initialize_mcmc_system: Initialize chain states and data structures
- validate_mcmc_inputs: Validate inputs before sampling
- gen_rng_keys: Generate JAX random keys

Configuration is split into two parts:
- user_config: Serializable config that can be saved/loaded without JAX
- runtime_ctx: JAX-dependent objects that exist only during execution

All config keys use lowercase with underscores (e.g., 'num_chains', 'posterior_id').
"""

import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
from typing import List, Dict, Tuple, Any

from ..registry import get_posterior
from .utils import clean_config
from ..batch_specs import BlockSpec, validate_block_specs, ProposalType
from .types import BlockArrays, RunParams, build_block_arrays


def get_discrete_param_indices(block_specs: List[BlockSpec]) -> List[int]:
    """
    Extract parameter indices that correspond to discrete (multinomial) blocks.

    These parameters should be excluded from R-hat calculations since they
    have different convergence behavior than continuous parameters.

    Args:
        block_specs: List of BlockSpec objects

    Returns:
        List of parameter indices for discrete parameters
    """
    discrete_indices = []
    param_offset = 0

    for spec in block_specs:
        if spec.proposal_type == ProposalType.MULTINOMIAL:
            # Add all param indices for this block
            discrete_indices.extend(range(param_offset, param_offset + spec.size))
        param_offset += spec.size

    return discrete_indices


def gen_rng_keys(rng_seed: int) -> Tuple[Any, Any]:
    """Generate JAX random keys from seed.

    Returns:
        (master_key, init_key): Tuple of JAX PRNGKeys
    """
    mkey = jax.random.PRNGKey(rng_seed)
    master_key, init_key = random.split(mkey, 2)
    return master_key, init_key


def validate_mcmc_inputs(config: Dict[str, Any], data: Dict[str, Any], specs: List[BlockSpec]) -> bool:
    """Validate MCMC inputs before starting sampling."""
    errors = []

    if 'burn_iter' in config and config['burn_iter'] < 0:
        errors.append(f"burn_iter must be >= 0, got {config['burn_iter']}")

    if 'num_collect' in config and config['num_collect'] < 0:
        errors.append(f"num_collect must be >= 0, got {config['num_collect']}")

    if 'num_chains' in config:
        if config['num_chains'] < 1:
            errors.append(f"num_chains must be >= 1, got {config['num_chains']}")
        if config['num_chains'] % 2 != 0:
            errors.append(f"num_chains must be even for parallel groups, got {config['num_chains']}")

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

    Splits configuration into:
    - user_config: Serializable values (can be saved to disk without JAX)
    - runtime_ctx: JAX-dependent objects (keys, dtypes, converted arrays)

    Args:
        mcmc_config: Input configuration dict with keys like 'posterior_id', 'num_chains_a', etc.
        data: Data dict with 'static', 'int', 'float' keys

    Returns:
        user_config: Clean config dict with user values + derived ints
        runtime_ctx: Dict with JAX keys, dtypes, converted data
        model_context: Dict with log_posterior_fn, blocks, run_params, etc.
    """
    mcmc_config = clean_config(mcmc_config)

    # Extract user-provided values (all lowercase)
    use_double = mcmc_config.get('use_double', True)
    posterior_id = mcmc_config.get('posterior_id')
    rng_seed = mcmc_config.get('rng_seed', 42)

    num_chains_a = mcmc_config.get('num_chains_a')
    num_chains_b = mcmc_config.get('num_chains_b')
    thin_iteration = mcmc_config.get('thin_iteration', 1)
    num_collect = mcmc_config.get('num_collect', 0)
    burn_iter = mcmc_config.get('burn_iter', 0)
    save_likelihoods = mcmc_config.get('save_likelihoods', False)

    # Compute derived values (as plain Python ints)
    num_chains = num_chains_a + num_chains_b
    num_iterations = thin_iteration * num_collect

    # Build user_config with user values + derived ints (all lowercase)
    user_config = {
        'posterior_id': posterior_id,
        'use_double': use_double,
        'rng_seed': rng_seed,
        'num_chains_a': num_chains_a,
        'num_chains_b': num_chains_b,
        'thin_iteration': thin_iteration,
        'num_collect': num_collect,
        'burn_iter': burn_iter,
        'save_likelihoods': save_likelihoods,
        # Derived values
        'num_chains': num_chains,
        'num_iterations': num_iterations,
        # These will be set later: num_params, num_superchains, subchains_per_super
    }

    # Copy over optional user config items
    if 'num_superchains' in mcmc_config:
        user_config['num_superchains'] = mcmc_config['num_superchains']
    if 'benchmark' in mcmc_config:
        user_config['benchmark'] = mcmc_config['benchmark']
    if 'n_subjects' in mcmc_config:
        user_config['n_subjects'] = mcmc_config['n_subjects']

    # Configure JAX precision
    if use_double:
        jax.config.update("jax_enable_x64", True)
        jnp_float_dtype = jnp.float64
        jnp_int_dtype = jnp.int64
    else:
        jax.config.update("jax_enable_x64", False)
        jnp_float_dtype = jnp.float32
        jnp_int_dtype = jnp.int32

    # Generate RNG keys
    master_key, init_key = gen_rng_keys(rng_seed)

    # Convert data arrays to JAX
    data_jax = {
        "static": data["static"],
        "int": tuple(jnp.asarray(i, dtype=jnp_int_dtype) for i in data["int"]),
        "float": tuple(jnp.asarray(i, dtype=jnp_float_dtype) for i in data["float"]),
    }

    # Build runtime context (JAX-dependent, not serializable)
    runtime_ctx = {
        'jnp_float_dtype': jnp_float_dtype,
        'jnp_int_dtype': jnp_int_dtype,
        'master_key': master_key,
        'init_key': init_key,
        'data': data_jax,
    }

    # Get posterior functions
    model_config = get_posterior(posterior_id)
    log_posterior_fn = partial(model_config['log_posterior'], data=data_jax)
    direct_sampler_fn = partial(model_config['direct_sampler'], data=data_jax)

    # Get coupled transform dispatch if defined (for COUPLED_TRANSFORM blocks)
    coupled_transform_fn = None
    if model_config.get('coupled_transform_dispatch'):
        coupled_transform_fn = partial(
            model_config['coupled_transform_dispatch'], data=data_jax
        )

    init_vec_fn = model_config.get('initial_vector')
    if init_vec_fn:
        initial_vector_fn = partial(init_vec_fn, data=data)  # Use original data for init
    else:
        raise ValueError(f"No 'initial_vector' function defined for {posterior_id}")

    gq_fn = None
    num_gq = 0
    if model_config.get('generated_quantities'):
        gq_fn = partial(model_config['generated_quantities'], data=data_jax)
        num_gq = model_config['get_num_gq'](user_config, data)

    raw_batch_specs = model_config['batch_type'](user_config, data)

    if not isinstance(raw_batch_specs[0], BlockSpec):
        raise TypeError(f"Posterior '{posterior_id}' must return BlockSpec objects.")

    validate_block_specs(raw_batch_specs, posterior_id)

    # Build unified BlockArrays structure
    block_arrays = build_block_arrays(raw_batch_specs)

    # Identify discrete parameters (excluded from R-hat calculations)
    discrete_param_indices = get_discrete_param_indices(raw_batch_specs)
    user_config['discrete_param_indices'] = discrete_param_indices
    if discrete_param_indices:
        print(f"Discrete parameters: {len(discrete_param_indices)} (excluded from R-hat)")

    # Create RunParams as frozen dataclass for JAX static argument compatibility
    run_params = RunParams(
        BURN_ITER=burn_iter,
        NUM_COLLECT=num_collect,
        THIN_ITERATION=thin_iteration,
        NUM_GQ=num_gq,
        START_ITERATION=0,  # Set to checkpoint iteration when resuming
        SAVE_LIKELIHOODS=save_likelihoods,
    )

    model_context = {
        'log_posterior_fn': log_posterior_fn,
        'direct_sampler_fn': direct_sampler_fn,
        'coupled_transform_fn': coupled_transform_fn,  # For COUPLED_TRANSFORM blocks
        'initial_vector_fn': initial_vector_fn,
        'generated_quantities_fn': gq_fn,
        'block_arrays': block_arrays,
        'block_specs': raw_batch_specs,  # Keep for labels/debugging
        'run_params': run_params,
        'model_config': model_config,  # For posterior hash computation
    }

    return user_config, runtime_ctx, model_context


def initialize_mcmc_system(
    initial_vector_np,
    user_config: Dict[str, Any],
    runtime_ctx: Dict[str, Any],
    num_gq: int,
    num_collect: int,
    num_blocks: int
) -> Tuple[Any, Dict[str, Any]]:
    """
    Initialize MCMC system using the Unified Nested R-hat structure.

    Algorithm:
    1. Determine K (Superchains) and M (Subchains).
       - If K is not provided, K = Total Chains (implies M=1, Standard R-hat).
    2. Extract K distinct initial states from the provided initialization vector.
    3. Replicate each of these K states M times to form the full population.
       (If M=1, this is an identity operation).

    Returns:
        initial_carry: Tuple of initial JAX arrays for the scan
        user_config: Updated with num_params, num_superchains, subchains_per_super
    """
    num_chains = user_config["num_chains"]
    jnp_float_dtype = runtime_ctx['jnp_float_dtype']
    init_key = runtime_ctx['init_key']

    initial_vector = jnp.asarray(initial_vector_np, dtype=jnp_float_dtype)
    num_params = initial_vector.size // num_chains

    # Update user_config with derived value
    user_config = user_config.copy()
    user_config["num_params"] = int(num_params)

    # Reshape linear vector to (NumChains, NumParams)
    all_initial_states = initial_vector.reshape(num_chains, num_params)

    # --- UNIFIED NESTED INIT STRATEGY ---
    # Default: 1 Superchain per Chain (Standard R-hat mode)
    K = user_config.get('num_superchains', num_chains)

    # Validation
    if num_chains % K != 0:
        raise ValueError(
            f"num_chains ({num_chains}) must be divisible by num_superchains ({K})"
        )

    M = num_chains // K
    user_config['num_superchains'] = int(K)
    user_config['subchains_per_super'] = int(M)

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
    num_chains_a = user_config["num_chains_a"]
    all_keys_for_each_chain = random.split(init_key, num_chains)
    initial_states_A, initial_states_B = jnp.split(all_initial_states, [num_chains_a], axis=0)
    initial_keys_A, initial_keys_B = jnp.split(all_keys_for_each_chain, [num_chains_a], axis=0)

    total_cols = num_params + num_gq
    initial_history = jnp.zeros((num_collect, num_chains, total_cols), dtype=jnp_float_dtype)

    if user_config['save_likelihoods']:
        # 2D array (Iter, Chain) - Summed over blocks
        initial_lik_history = jnp.empty((num_collect, num_chains), dtype=jnp_float_dtype)
    else:
        initial_lik_history = jnp.empty((1,), dtype=jnp_float_dtype)

    acceptance_counts = jnp.zeros(num_blocks, dtype=jnp.int32)
    current_iteration = jnp.array(0, dtype=jnp.int32)

    initial_carry = (initial_states_A, initial_keys_A, initial_states_B, initial_keys_B,
                     initial_history, initial_lik_history, acceptance_counts, current_iteration)

    return initial_carry, user_config
