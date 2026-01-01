"""
MCMC Scan Body and Block Statistics.

This module contains the main scan loop components:
- compute_block_statistics: Precompute mean/covariance for all blocks
- mcmc_scan_body_offload: One iteration of the MCMC scan
- _save_with_gq: Helper to save states with generated quantities
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from .types import BlockArrays, RunParams
from .sampling import parallel_gibbs_iteration


# Regularization constant for covariance matrices
NUGGET = 1e-5


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


def _save_with_gq(states_A, states_B, gq_fn, num_gq):
    """Combine states and optionally compute generated quantities."""
    full_state = jnp.concatenate((states_A, states_B), axis=0)
    if num_gq > 0 and gq_fn is not None:
        gq_values = jax.vmap(gq_fn)(full_state)
        return jnp.concatenate((full_state, gq_values), axis=1)
    else:
        return full_state


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
