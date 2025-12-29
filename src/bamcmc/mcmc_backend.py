import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, replace
import numpy as np
import time
from datetime import datetime, timedelta

from .registry import get_posterior
from .mcmc_utils import clean_config

# Import modular system
from .batch_specs import BlockSpec, SamplerType, validate_block_specs

# Import settings system
from .settings import build_settings_matrix

# Import error handling for validation
from .error_handling import validate_mcmc_config, diagnose_sampler_issues, print_diagnostics

# Import static proposal dispatch table (created once at import time)
from .proposals import PROPOSAL_DISPATCH_TABLE

# Import checkpoint and batch utilities
from .checkpoint_helpers import (
    save_checkpoint,
    load_checkpoint,
    initialize_from_checkpoint,
    combine_batch_histories,
    apply_burnin,
    compute_rhat_from_history,
)

# Import posterior benchmarking system
from .posterior_benchmark import (
    get_manager as get_benchmark_manager,
    compute_posterior_hash,
    get_posterior_hash,
)

# --- CONSTANTS ---
CHUNK_SIZE = 100
NUGGET = 1e-5

# --- COMPILED FUNCTION CACHE ---
# Cache compiled MCMC kernels by configuration hash (in-memory, within session)
_COMPILED_KERNEL_CACHE = {}


def _compute_cache_key(mcmc_config, data, block_arrays):
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


# --- MODULE-LEVEL MCMC FUNCTIONS FOR CROSS-SESSION CACHING ---
# These functions are defined at module level (not inside rmcmc) so that
# JAX's disk cache can recognize identical traces across Python sessions.

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
    # Note: mcmc_scan_body_offload is defined later in this file
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


# --- DATA STRUCTURES ---

@dataclass(frozen=True)
class BlockArrays:
    """
    Pre-parsed block specification arrays for MCMC backend.

    Groups related block data to reduce function parameter counts.
    All arrays are JAX arrays ready for use in the sampling loop.

    This is a frozen dataclass for immutability and JAX compatibility.
    Registered as a JAX pytree to enable tracing through JIT.
    """
    indices: jnp.ndarray         # (n_blocks, max_block_size) - parameter indices per block
    types: jnp.ndarray           # (n_blocks,) - SamplerType for each block
    masks: jnp.ndarray           # (n_blocks, max_block_size) - valid parameter mask
    proposal_types: jnp.ndarray  # (n_blocks,) - ProposalType for MH blocks
    settings_matrix: jnp.ndarray # (n_blocks, MAX_SETTINGS) - per-block settings
    max_size: int                # Maximum block size
    num_blocks: int              # Number of blocks
    total_params: int            # Total parameter count


def _block_arrays_flatten(ba):
    """Flatten BlockArrays for JAX pytree."""
    # Arrays are children (traced), scalars are auxiliary data (static)
    children = (ba.indices, ba.types, ba.masks, ba.proposal_types, ba.settings_matrix)
    aux_data = (ba.max_size, ba.num_blocks, ba.total_params)
    return children, aux_data


def _block_arrays_unflatten(aux_data, children):
    """Unflatten BlockArrays from JAX pytree."""
    indices, types, masks, proposal_types, settings_matrix = children
    max_size, num_blocks, total_params = aux_data
    return BlockArrays(
        indices=indices,
        types=types,
        masks=masks,
        proposal_types=proposal_types,
        settings_matrix=settings_matrix,
        max_size=max_size,
        num_blocks=num_blocks,
        total_params=total_params
    )


# Register BlockArrays as a JAX pytree
jax.tree_util.register_pytree_node(
    BlockArrays,
    _block_arrays_flatten,
    _block_arrays_unflatten
)


@dataclass(frozen=True)
class RunParams:
    """
    Immutable run parameters for JAX static argument compatibility.

    This frozen dataclass allows run parameters to be passed as static
    arguments to JIT-compiled functions, enabling cross-session caching.
    """
    BURN_ITER: int
    NUM_COLLECT: int
    THIN_ITERATION: int
    NUM_GQ: int
    START_ITERATION: int
    SAVE_LIKELIHOODS: bool


def build_block_arrays(specs: List[BlockSpec], start_idx: int = 0) -> BlockArrays:
    """
    Build BlockArrays from a list of BlockSpec objects.

    Args:
        specs: List of BlockSpec objects defining parameter blocks
        start_idx: Starting parameter index (default 0)

    Returns:
        BlockArrays with all arrays ready for MCMC backend
    """
    if not specs:
        raise ValueError("Empty block specifications")

    max_size = max(spec.size for spec in specs)
    num_blocks = len(specs)

    # Build index and mask arrays
    indices = np.full((num_blocks, max_size), -1, dtype=np.int32)
    types = np.zeros(num_blocks, dtype=np.int32)
    masks = np.zeros((num_blocks, max_size), dtype=np.float32)

    current_param = start_idx
    for i, spec in enumerate(specs):
        types[i] = int(spec.sampler_type)
        block_idxs = np.arange(current_param, current_param + spec.size)
        indices[i, :spec.size] = block_idxs
        masks[i, :spec.size] = 1.0
        current_param += spec.size

    # Build proposal info
    proposal_types = []
    for spec in specs:
        if spec.is_mh_sampler():
            proposal_types.append(int(spec.proposal_type))
        else:
            proposal_types.append(0)

    return BlockArrays(
        indices=jnp.array(indices),
        types=jnp.array(types),
        masks=jnp.array(masks),
        proposal_types=jnp.array(proposal_types, dtype=jnp.int32),
        settings_matrix=build_settings_matrix(specs),
        max_size=max_size,
        num_blocks=num_blocks,
        total_params=current_param,
    )


# --- VALIDATION ---

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

# --- SETUP AND UTILITY FUNCTIONS ---

def gen_rng_keys(mcmc_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate and store JAX random keys in config."""
    rng_seed = mcmc_config.setdefault("rng_seed", 42)
    mkey = jax.random.PRNGKey(rng_seed)
    mcmc_config["master_key"], mcmc_config["init_key"] = random.split(mkey, 2)
    return mcmc_config


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
    # JAX_COMPILATION_CACHE_DIR → ~/.cache/jax/bamcmc_cache/

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
        print(f"Structure: {K} Superchains × {M} Subchains (Nested R-hat Mode)")
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


def benchmark_mcmc_sampler(compiled_chunk_fn, initial_carry, benchmark_iters):
    print("\n--- BENCHMARKING ---")
    print(f"Running benchmark ({benchmark_iters} iterations)...", flush = True)
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


# --- MCMC CORE FUNCTIONS ---

def propose_multivariate_block(key, current_block, block_mean, block_cov, coupled_blocks, block_mask, proposal_type, settings):
    """
    Generate a proposal for a parameter block.

    Args:
        key: JAX random key
        current_block: Current parameter values (block_size,)
        block_mean: Precomputed mean from coupled blocks (block_size,)
        block_cov: Precomputed covariance from coupled blocks (block_size, block_size)
        coupled_blocks: Raw states from opposite group (n_chains, block_size) - for discrete proposals
        block_mask: Mask for valid parameters
        proposal_type: Integer index into dispatch table
        settings: JAX array of proposal-specific settings

    Returns:
        proposal: Proposed parameter values
        log_ratio: Log Hastings ratio
        new_key: Updated random key
    """
    # Pack arguments into operand tuple for static dispatch
    operand = (key, current_block, block_mean, block_cov, coupled_blocks, block_mask, settings)

    # Use static dispatch table - no closure creation during tracing
    proposal, log_ratio, new_key = jax.lax.switch(
        proposal_type,
        PROPOSAL_DISPATCH_TABLE,
        operand
    )
    return proposal, log_ratio, new_key

def metropolis_block_step(operand, log_post_fn):
    """
    Perform one Metropolis-Hastings step for a single block.

    Args:
        operand: Tuple of (key, chain_state, block_idx_vec, block_mean, block_cov,
                          coupled_blocks, block_mask, proposal_type, settings)
        log_post_fn: Log posterior function

    Returns:
        next_state, new_key, chosen_lp, accepted
    """
    key, chain_state, block_idx_vec, block_mean, block_cov, coupled_blocks, block_mask, proposal_type, settings = operand

    safe_indices = jnp.clip(block_idx_vec, 0, chain_state.shape[0] - 1)
    current_block_values = jnp.take(chain_state, safe_indices)

    proposed_block_values, log_den_ratio, key = propose_multivariate_block(
        key, current_block_values, block_mean, block_cov, coupled_blocks, block_mask, proposal_type, settings
    )

    actual_proposal_values = jnp.where(block_mask, proposed_block_values, current_block_values)

    # FIX: Avoid repeated index writes which cause nondeterministic behavior under vmap.
    # Instead of using .at[safe_indices].set(actual_proposal_values) which writes to
    # repeated indices (e.g., [0,1,2,3,4,5,0,0,0,...] when padding clips -1 to 0),
    # we use vectorized operations to identify which chain_state positions need updating.
    #
    # For each chain_state index, check if any valid block position points to it.
    # matches[i, j] = True if block position j (with mask=1) maps to chain_state index i
    chain_indices = jnp.arange(chain_state.shape[0])
    matches = (block_idx_vec[None, :] == chain_indices[:, None]) & (block_mask[None, :] > 0)

    # update_needed[i] = True if any valid block position maps to chain_state index i
    update_needed = jnp.any(matches, axis=1)

    # For indices that need updating, get the value from the first matching block position
    # argmax returns first True index; for rows with no True, returns 0 (but update_needed=False handles that)
    first_match_idx = jnp.argmax(matches, axis=1)
    values_from_block = actual_proposal_values[first_match_idx]

    # Apply update: where update_needed, use values_from_block; otherwise keep chain_state
    proposed_state = jnp.where(update_needed, values_from_block, chain_state)

    # Check if proposal contains NaN/Inf - if so, force rejection
    proposal_is_finite = jnp.all(jnp.isfinite(actual_proposal_values))

    lp_current  = log_post_fn(chain_state,  block_idx_vec)
    lp_proposed = log_post_fn(proposed_state, block_idx_vec)

    safe_lp_current  = jnp.nan_to_num(lp_current,  nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf)
    safe_lp_proposed = jnp.nan_to_num(lp_proposed, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf)

    raw_ratio = log_den_ratio + safe_lp_proposed - safe_lp_current
    # Force rejection if proposal contains NaN/Inf
    safe_ratio = jnp.where(proposal_is_finite,
                           jnp.nan_to_num(raw_ratio, nan=-jnp.inf),
                           -jnp.inf)

    new_key, accept_key = random.split(key)
    log_uniform = jnp.log(random.uniform(accept_key, shape=()))

    accept = log_uniform < safe_ratio
    next_state = jnp.where(accept, proposed_state, chain_state)
    chosen_lp = jnp.where(accept, safe_lp_proposed, safe_lp_current)
    accepted = jnp.float32(accept)

    return next_state, new_key, chosen_lp, accepted

def direct_block_step(operand, direct_sampler_fn):
    key, chain_state, block_idx_vec, _, _, _, _, _, _ = operand
    new_state, new_key = direct_sampler_fn(key, chain_state, block_idx_vec)
    # Safeguard: replace any NaN/Inf values with original state values
    # This prevents bad samples from propagating through the chain
    is_finite = jnp.isfinite(new_state)
    new_state = jnp.where(is_finite, new_state, chain_state)
    accepted = jnp.float32(1.0)
    return new_state, new_key, 0.0, accepted

def full_chain_iteration(key, chain_state, block_means, block_covs, coupled_blocks,
                         block_arrays: BlockArrays,
                         log_post_fn, direct_sampler_fn):
    """
    Run one full Gibbs iteration over all blocks.

    Args:
        key: JAX random key
        chain_state: Current state of this chain (n_params,)
        block_means: Precomputed means for each block (n_blocks, max_block_size)
        block_covs: Precomputed covariances (n_blocks, max_block_size, max_block_size)
        coupled_blocks: Raw block data for discrete proposals (n_blocks, n_chains, max_block_size)
        block_arrays: BlockArrays with indices, types, masks, proposal_types, settings
        log_post_fn: Log posterior function
        direct_sampler_fn: Direct sampler function
    """
    metro_step = partial(metropolis_block_step, log_post_fn=log_post_fn)
    direct_step = partial(direct_block_step, direct_sampler_fn=direct_sampler_fn)

    def scan_body(carry_state, block_i):
        current_state, current_key = carry_state

        b_type    = block_arrays.types[block_i]
        b_indices = block_arrays.indices[block_i]
        b_mask    = block_arrays.masks[block_i]
        b_mean    = block_means[block_i]
        b_cov     = block_covs[block_i]
        b_coupled = coupled_blocks[block_i]
        b_proposal_type = block_arrays.proposal_types[block_i]
        b_settings = block_arrays.settings_matrix[block_i]

        (updated_state, new_key, lp_val, accepted) = jax.lax.cond(
            b_type == SamplerType.DIRECT_CONJUGATE,
            direct_step,
            metro_step,
            (current_key, current_state, b_indices, b_mean, b_cov, b_coupled, b_mask, b_proposal_type, b_settings)
        )
        return (updated_state, new_key), (lp_val, accepted)

    (final_state, final_key), (block_lps, block_accepts) = jax.lax.scan(
        scan_body,
        (chain_state, key),
        jnp.arange(block_arrays.num_blocks)
    )
    return final_state, final_key, block_lps, block_accepts


@partial(jax.vmap, in_axes=(0, 0, None, None, None, None, None, None), out_axes=(0, 0, 0, 0))
def parallel_gibbs_iteration(keys, chain_states, block_means, block_covs, coupled_blocks,
                             block_arrays: BlockArrays,
                             log_post_fn, direct_sampler_fn):
    """
    Run Gibbs iteration for all chains in parallel.

    Args:
        keys: Random keys for each chain (n_chains,)
        chain_states: States of all chains in this group (n_chains, n_params)
        block_means: Precomputed means (n_blocks, max_block_size) - shared across chains
        block_covs: Precomputed covariances (n_blocks, max_block_size, max_block_size) - shared
        coupled_blocks: Raw block data (n_blocks, n_chains, max_block_size) - shared
        block_arrays: BlockArrays with all block configuration - shared across chains
        log_post_fn: Log posterior function
        direct_sampler_fn: Direct sampler function
    """
    return full_chain_iteration(keys, chain_states, block_means, block_covs, coupled_blocks,
                                block_arrays,
                                log_post_fn, direct_sampler_fn)

def compute_block_statistics(
    coupled_states: jnp.ndarray,
    block_indices: jnp.ndarray,
    block_masks: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Precompute mean and covariance for all blocks from coupled states.

    This is called ONCE before vmapping across chains, avoiding redundant computation.

    Args:
        coupled_states: States from opposite group (n_chains, n_params)
        block_indices: Block index array (n_blocks, max_block_size)
        block_masks: Parameter masks (n_blocks, max_block_size)

    Returns:
        block_means: (n_blocks, max_block_size)
        block_covs: (n_blocks, max_block_size, max_block_size)
        coupled_blocks: (n_blocks, n_chains, max_block_size) - raw data for multinomial
    """
    def get_stats_for_one_block(b_indices, b_mask):
        safe_indices = jnp.clip(b_indices, 0, coupled_states.shape[1] - 1)
        block_data = jnp.take(coupled_states, safe_indices, axis=1)

        # Compute mean
        mean = jnp.mean(block_data, axis=0) * b_mask

        # Compute covariance with regularization
        cov = jnp.cov(block_data, rowvar=False)
        cov = jnp.atleast_2d(cov)
        n = cov.shape[0]
        cov_reg = cov + NUGGET * jnp.eye(n)

        # Apply mask
        eye = jnp.eye(n)
        mask_2d = jnp.outer(b_mask, b_mask)
        final_cov = jnp.where(mask_2d, cov_reg, eye)

        return mean, final_cov, block_data

    means, covs, coupled_blocks = jax.vmap(get_stats_for_one_block)(block_indices, block_masks)
    return means, covs, coupled_blocks


def mcmc_scan_body_offload(carry, step_idx, log_post_fn, direct_sampler_fn, gq_fn,
                           block_arrays: BlockArrays, run_params):
    """
    One iteration of the MCMC scan body.

    This function updates both groups A and B in sequence.
    Block statistics (mean, cov) are precomputed ONCE before vmapping across chains.
    """
    states_A, keys_A, states_B, keys_B, history_array, lik_history_array, acceptance_counts, current_iteration = carry

    # Precompute block statistics ONCE from states_B (for updating A)
    means_A, covs_A, coupled_blocks_A = compute_block_statistics(
        states_B, block_arrays.indices, block_arrays.masks
    )

    # Update Group A
    next_states_A, next_keys_A, lps_A, accepts_A = parallel_gibbs_iteration(
        keys_A, states_A, means_A, covs_A, coupled_blocks_A,
        block_arrays,
        log_post_fn, direct_sampler_fn
    )

    # Precompute block statistics ONCE from updated states_A (for updating B)
    means_B, covs_B, coupled_blocks_B = compute_block_statistics(
        next_states_A, block_arrays.indices, block_arrays.masks
    )

    # Update Group B
    next_states_B, next_keys_B, lps_B, accepts_B = parallel_gibbs_iteration(
        keys_B, states_B, means_B, covs_B, coupled_blocks_B,
        block_arrays,
        log_post_fn, direct_sampler_fn
    )

    combined_accepts = jnp.concatenate([accepts_A, accepts_B], axis=0)
    block_accept_counts_iter = jnp.sum(combined_accepts, axis=0).astype(jnp.int32)
    updated_acceptance_counts = acceptance_counts + block_accept_counts_iter

    # Compute iteration relative to this run's start (for resume support)
    # Support both dict and RunParams dataclass for backwards compatibility
    if isinstance(run_params, dict):
        start_iter = run_params['START_ITERATION']
        burn_iter = run_params['BURN_ITER']
        thin_iter = run_params['THIN_ITERATION']
        num_collect = run_params['NUM_COLLECT']
        num_gq = run_params['NUM_GQ']
    else:
        start_iter = run_params.START_ITERATION
        burn_iter = run_params.BURN_ITER
        thin_iter = run_params.THIN_ITERATION
        num_collect = run_params.NUM_COLLECT
        num_gq = run_params.NUM_GQ

    run_iteration = current_iteration - start_iter
    is_after_burn_in = (run_iteration >= burn_iter)
    collection_iteration = run_iteration - burn_iter
    is_thin_iter = (collection_iteration + 1) % thin_iter == 0

    next_history = history_array
    next_lik_history = lik_history_array

    if num_collect > 0:
        thin_idx = collection_iteration // thin_iter
        should_save = is_after_burn_in & is_thin_iter & (thin_idx >= 0) & (thin_idx < num_collect)

        def save_params(h_array):
             return h_array.at[thin_idx].set(
                _save_with_gq(next_states_A, next_states_B, gq_fn, num_gq)
            )

        next_history = jax.lax.cond(should_save, save_params, lambda h: h, history_array)

        if lik_history_array.size > 1:
            def save_lik(l_array):
                total_lps_A = jnp.sum(lps_A, axis=1)
                total_lps_B = jnp.sum(lps_B, axis=1)
                full_lps = jnp.concatenate((total_lps_A, total_lps_B), axis=0)
                return l_array.at[thin_idx].set(full_lps)

            next_lik_history = jax.lax.cond(should_save, save_lik, lambda h: h, lik_history_array)

    next_iteration = current_iteration + 1

    return (next_states_A, next_keys_A, next_states_B, next_keys_B, next_history, next_lik_history, updated_acceptance_counts, next_iteration), None

def _save_with_gq(states_A, states_B, gq_fn, num_gq):
    full_state = jnp.concatenate((states_A, states_B), axis=0)
    if num_gq > 0 and gq_fn is not None:
        gq_values = jax.vmap(gq_fn)(full_state)
        return jnp.concatenate((full_state, gq_values), axis=1)
    else:
        return full_state


@partial(jax.jit, static_argnums=(1, 2))
def compute_nested_rhat(history: jnp.ndarray, K: int, M: int) -> jnp.ndarray:
    """
    Unified Nested R-hat Diagnostic (Margossian et al., 2022).

    Calculates the Convergence Diagnostic for MCMC chains using the "Superchain" concept.

    Structure:
      - K: Number of Superchains (independent starting points).
      - M: Number of Subchains per Superchain (initialized identically to root).

    Cases:
      1. M > 1 (Nested): Measures convergence using many short chains.
         Variance reduction comes from M scaling.
      2. M = 1 (Standard): Measures standard Gelman-Rubin R-hat.
         Reduces to comparing K independent chains.

    Returns:
        nrhat: (n_params,) array of R-hat values.
    """
    n_samples, n_chains, n_params = history.shape

    # 1. Reshape to Superchain structure: (n_samples, K, M, n_params)
    # If M=1, this is (n_samples, K, 1, n_params)
    history_nested = history.reshape(n_samples, K, M, n_params)

    # 2. Compute Superchain Means (averaging over M subchains)
    # If M=1, this is just the chain value itself
    superchain_means_over_time = jnp.mean(history_nested, axis=2) # (n_samples, K, n_params)

    # 3. Compute Grand Means per Superchain (averaging over time)
    superchain_means = jnp.mean(superchain_means_over_time, axis=0) # (K, n_params)

    # 4. Between-Superchain Variance (B)
    # This captures how far apart the K distinct starting groups are.
    # B = n * var(chain_means) in Gelman-Rubin notation
    B = n_samples * jnp.var(superchain_means, axis=0, ddof=1)

    # 5. Within-Superchain Variance (W)
    # This captures local mixing.

    # Component A: Between-subchain variance (only exists if M > 1)
    # Measures how much subchains spread out from their identical start point.
    if M > 1:
        # Variance across M subchains at each time step, averaged over time
        subchain_means_t = jnp.mean(history_nested, axis=0) # (K, M, n_params)
        B_within = jnp.var(subchain_means_t, axis=1, ddof=1) # (K, n_params)
    else:
        # If M=1, there is no variance between subchains (there is only 1).
        B_within = jnp.zeros((K, n_params))

    # Component B: Within-subchain variance (variance over time)
    if n_samples > 1:
        # Variance over time for each individual chain
        W_within_raw = jnp.var(history_nested, axis=0, ddof=1) # (K, M, n_params)
        # Averaged over all M subchains
        W_within = jnp.mean(W_within_raw, axis=1) # (K, n_params)
    else:
        W_within = jnp.zeros((K, n_params))

    # Total Within Variance: Average over all K superchains
    # W = average within-chain variance (standard Gelman-Rubin notation)
    W = jnp.mean(B_within + W_within, axis=0)

    # 6. Final Ratio using Gelman-Rubin formula
    # V_hat = (n-1)/n * W + B/n + B/(m*n)  [full formula]
    # For large n, this simplifies to: R-hat ≈ sqrt(1 + B/(n*W))
    # Using the full formula for better accuracy:
    m = K  # number of chains/superchains
    n = n_samples
    V_hat = ((n - 1) / n) * W + B / n + B / (m * n)
    nrhat = jnp.sqrt(V_hat / W)

    return nrhat

# --- MAIN ENTRY POINT ---

def rmcmc(
    mcmc_config: Dict[str, Any],
    data: Dict[str, Any],
    calculate_rhat: bool = True,
    resume_from: Optional[str] = None,
    reset_from: Optional[str] = None,
    reset_noise_scale: float = 0.1
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], Optional[np.ndarray], Dict[str, Any]]:
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
        history: Sample history array (num_collect, num_chains, num_params + num_gq)
        diagnostics: Dict with R-hat values and timing info
        mcmc_config: Updated configuration dict
        lik_history: Likelihood history if SAVE_LIKELIHOODS=True, else None
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

    mcmc_config, data, model_ctx = configure_mcmc_system(mcmc_config, data)
    mcmc_config = gen_rng_keys(mcmc_config)
    run_params = model_ctx['run_params']

    block_arrays = model_ctx['block_arrays']
    block_specs = model_ctx['block_specs']

    # Compute posterior hash for persistent benchmarking
    posterior_id = mcmc_config['POSTERIOR_ID']
    posterior_hash = get_posterior_hash(
        posterior_id,
        model_ctx['model_config'],
        data
    )
    benchmark_mgr = get_benchmark_manager()
    cached_benchmark = benchmark_mgr.get_cached_benchmark(posterior_hash)

    print("Validating inputs...", flush=True)
    try:
        validate_mcmc_inputs(mcmc_config, data, block_specs)
        print("✓ Validation passed", flush=True)
    except ValueError as e:
        print(f"✗ Validation failed:\n{e}", flush=True)
        raise

    print(f"Starting sampling for {mcmc_config['POSTERIOR_ID']}...", flush=True)

    # Initialize either from checkpoint, reset, or fresh
    if resume_from is not None and reset_from is not None:
        raise ValueError("Cannot specify both resume_from and reset_from")

    if resume_from is not None:
        print(f"Loading checkpoint from {resume_from}...")
        checkpoint = load_checkpoint(resume_from)
        initial_carry, mcmc_config = initialize_from_checkpoint(
            checkpoint,
            mcmc_config,
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
        n_subjects = data['static'][0]  # Number of subjects from data
        K = mcmc_config.get('NUM_SUPERCHAINS', mcmc_config['NUM_CHAINS'])
        M = mcmc_config['NUM_CHAINS'] // K

        print(f"  Source iteration: {checkpoint['iteration']}")
        print(f"  Generating {K} reset starting points (noise_scale={reset_noise_scale})")

        initial_vector_np = generate_reset_vector(
            checkpoint,
            model_type=mcmc_config['POSTERIOR_ID'],
            n_subjects=n_subjects,
            K=K,
            M=M,
            noise_scale=reset_noise_scale,
            rng_seed=mcmc_config.get('rng_seed', None)
        )

        initial_carry, mcmc_config = initialize_mcmc_system(
            initial_vector_np,
            mcmc_config,
            num_gq=run_params.NUM_GQ,
            num_collect=run_params.NUM_COLLECT,
            num_blocks=block_arrays.num_blocks
        )
        # Reset starts fresh (iteration 0), not from checkpoint iteration
        # run_params already has START_ITERATION=0 from configure_mcmc_system
        print(f"  Reset complete - starting fresh from iteration 0")

    else:
        print("Generating initial vector...", flush=True)
        initial_vector_np = model_ctx['initial_vector_fn'](mcmc_config)

        initial_carry, mcmc_config = initialize_mcmc_system(
            initial_vector_np,
            mcmc_config,
            num_gq=run_params.NUM_GQ,
            num_collect=run_params.NUM_COLLECT,
            num_blocks=block_arrays.num_blocks
        )

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Likelihood Saving: {'ENABLED' if mcmc_config['SAVE_LIKELIHOODS'] else 'DISABLED'}")

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
        # No compilation needed - set time to 0
        compile_start = time.perf_counter()
        compile_end = compile_start  # Zero compile time
    else:
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
        print(f"Done ({compile_end - compile_start:.4f}s)")

        # Create wrapper that binds data arguments (they don't change during run)
        _cf = compiled_fn
        _di, _df, _ba = data_int, data_float, block_arrays
        def compiled_chunk(carry):
            return _cf(carry, _di, _df, _ba)

        # Cache the compiled kernel for in-memory reuse
        _COMPILED_KERNEL_CACHE[cache_key] = compiled_chunk
        print(f"Kernel cached (key: {posterior_id}, {mcmc_config['NUM_CHAINS']} chains)")

    # --- BENCHMARKING ---
    # Check for cached benchmark first, then run if needed
    benchmark_iters = mcmc_config.get('BENCHMARK', 0)
    avg_time = None
    fresh_compile_time = compile_end - compile_start

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
        results = benchmark_mcmc_sampler(compiled_chunk, initial_carry, benchmark_iters)
        avg_time = results['avg_time']

        # Save benchmark for future use
        benchmark_mgr.save_benchmark(
            posterior_hash=posterior_hash,
            posterior_id=posterior_id,
            num_chains=mcmc_config['NUM_CHAINS'],
            fresh_compile_time=fresh_compile_time,
            iteration_time=avg_time,
            benchmark_iterations=benchmark_iters,
        )
        print(f"  Benchmark saved (hash: {posterior_hash[:8]}...)")

    total_iterations_to_run = run_params.BURN_ITER + mcmc_config["NUM_ITERATIONS"]
    host_history = None
    host_lik_history = None

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
        num_chains = mcmc_config["NUM_CHAINS"]

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

            K = mcmc_config['NUM_SUPERCHAINS']
            M = mcmc_config['SUBCHAINS_PER_SUPER']

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
            print(f"  Threshold: {threshold:.4f} (τ={tau:.0e})")

            if jnp.max(nrhat_values) < threshold:
                print(f"  ✓ Converged (max < {threshold:.4f})")
            else:
                print(f"  ⚠ Not Converged (max = {jnp.max(nrhat_values):.4f} >= {threshold:.4f})")

        print("Transferring history to Host...", flush=True)
        host_history = jax.device_get(final_history_device)

        if mcmc_config['SAVE_LIKELIHOODS']:
            print("Transferring likelihood history to Host...", flush=True)
            host_lik_history = jax.device_get(final_lik_device)

        print("\n--- MCMC Run Summary ---")
        print(f"  Total Wall Time: {end_run_time - start_run_time:.4f} s")
    else:
        print("\nSkipping main run.", flush = True)
        host_history = jax.device_get(initial_carry[4])
        if mcmc_config['SAVE_LIKELIHOODS']:
             host_lik_history = jax.device_get(initial_carry[5])
        nrhat_values = None

    # Unified diagnostics return
    diagnostics = {
        'rhat': nrhat_values,        # Primary metric (Nested or Standard)
        'K': mcmc_config['NUM_SUPERCHAINS'],
        'M': mcmc_config['SUBCHAINS_PER_SUPER'],
        # Timing info for benchmarking
        'compile_time': compile_end - compile_start,
        'wall_time': end_run_time - start_run_time if total_iterations_to_run > 0 else 0.0,
        'avg_iter_time': avg_time,
        'total_iterations': total_iterations_to_run,
    }

    # --- 4. POST-RUN DIAGNOSTICS ---
    print("\n--- Post-Run Diagnostics ---")
    diagnostics = diagnose_sampler_issues(host_history, mcmc_config, diagnostics)
    print_diagnostics(diagnostics)

    if diagnostics['issues']:
        print("\n⚠️  Warning: Issues detected during sampling!")
        print("Review diagnostics above before using results.")

    # Build checkpoint from final carry
    final_carry = current_carry if total_iterations_to_run > 0 else initial_carry
    final_checkpoint = {
        'states_A': np.asarray(jax.device_get(final_carry[0])),
        'states_B': np.asarray(jax.device_get(final_carry[2])),
        'keys_A': np.asarray(jax.device_get(final_carry[1])),
        'keys_B': np.asarray(jax.device_get(final_carry[3])),
        'iteration': int(jax.device_get(final_carry[7])),
        'acceptance_counts': np.asarray(jax.device_get(final_carry[6])),
        'posterior_id': mcmc_config['POSTERIOR_ID'],
        'num_params': mcmc_config['num_params'],
        'num_chains_a': mcmc_config['NUM_CHAINS_A'],
        'num_chains_b': mcmc_config['NUM_CHAINS_B'],
        'num_superchains': mcmc_config.get('NUM_SUPERCHAINS', 0),
        'subchains_per_super': mcmc_config.get('SUBCHAINS_PER_SUPER', 0),
    }

    return host_history, diagnostics, mcmc_config, host_lik_history, final_checkpoint
