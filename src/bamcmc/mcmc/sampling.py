"""
MCMC Sampling Functions.

Core sampling functions for the MCMC backend:
- propose_multivariate_block: Generate proposals for parameter blocks
- metropolis_block_step: Metropolis-Hastings step for a single block
- direct_block_step: Direct sampling step for conjugate blocks
- full_chain_iteration: Full Gibbs iteration over all blocks
- parallel_gibbs_iteration: Vmapped version for parallel chains
"""

import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial

from ..batch_specs import SamplerType
from ..proposals import PROPOSAL_DISPATCH_TABLE
from ..settings import SettingSlot
from .types import BlockArrays

# Regularization for covariance inversion in MALA
MALA_COV_NUGGET = 1e-6


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
    """
    Perform direct sampling step for conjugate blocks.

    Args:
        operand: Tuple of (key, chain_state, block_idx_vec, ...)
        direct_sampler_fn: Direct sampling function

    Returns:
        next_state, new_key, log_prob (0.0), accepted (1.0)
    """
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
