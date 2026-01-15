"""
MCMC Scan Body and Block Statistics.

This module contains the main scan loop components:
- compute_block_statistics: Precompute mean/covariance for all blocks
- mcmc_scan_body_offload: One iteration of the MCMC scan
- _save_with_gq: Helper to save states with generated quantities
- attempt_temperature_swaps: Replica exchange swap moves for parallel tempering
"""

import jax
import jax.numpy as jnp
import jax.random as random
from typing import Tuple

from .types import BlockArrays, RunParams
from .sampling import parallel_gibbs_iteration


# Regularization constant for covariance matrices
NUGGET = 1e-5


def compute_block_statistics(
    coupled_states: jnp.ndarray,
    block_indices: jnp.ndarray,
    block_masks: jnp.ndarray,
    log_post_fn=None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Precompute mean, covariance, and mode for all blocks from coupled states.

    This is called ONCE before vmapping across chains, avoiding redundant computation.

    Args:
        coupled_states: States from opposite group (n_chains, n_params)
        block_indices: Block index array (n_blocks, max_block_size)
        block_masks: Parameter masks (n_blocks, max_block_size)
        log_post_fn: Optional log posterior function for computing mode.
                     If provided, evaluates log posterior for each coupled chain
                     to find the mode (chain with highest density).

    Returns:
        block_means: (n_blocks, max_block_size)
        block_covs: (n_blocks, max_block_size, max_block_size)
        coupled_blocks: (n_blocks, n_chains, max_block_size) - raw data for multinomial
        block_modes: (n_blocks, max_block_size) - mode chain values for each block
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

    # Compute mode: find chain with highest log posterior
    if log_post_fn is not None:
        # Evaluate full log posterior for each coupled chain
        # Use a dummy "all parameters" index for full evaluation
        all_indices = jnp.arange(coupled_states.shape[1])
        coupled_log_posts = jax.vmap(lambda s: log_post_fn(s, all_indices))(coupled_states)

        # Find mode chain (highest log posterior)
        mode_idx = jnp.argmax(coupled_log_posts)
        mode_state = coupled_states[mode_idx]

        # Extract mode values for each block
        def get_mode_for_block(b_indices, b_mask):
            safe_indices = jnp.clip(b_indices, 0, mode_state.shape[0] - 1)
            mode_block = jnp.take(mode_state, safe_indices) * b_mask
            return mode_block

        block_modes = jax.vmap(get_mode_for_block)(block_indices, block_masks)
    else:
        # If no log_post_fn, use mean as fallback for mode
        block_modes = means

    return means, covs, coupled_blocks, block_modes


def _save_with_gq(states_A, states_B, gq_fn, num_gq):
    """Combine states and optionally compute generated quantities."""
    full_state = jnp.concatenate((states_A, states_B), axis=0)
    if num_gq > 0 and gq_fn is not None:
        gq_values = jax.vmap(gq_fn)(full_state)
        return jnp.concatenate((full_state, gq_values), axis=1)
    else:
        return full_state


def attempt_temperature_swaps(
    key,
    states_A, states_B,
    temperature_ladder,
    log_post_fn,
    swap_accepts, swap_attempts
):
    """
    Attempt replica exchange swaps between adjacent temperature levels.

    For parallel tempering, after each MCMC iteration we attempt to swap states
    between chains at adjacent temperatures. This allows high-temperature chains
    (with flattened posteriors) to explore broadly, then swap good configurations
    to low-temperature chains.

    Swap acceptance probability between temperatures β_i and β_j (β_i > β_j):
        α = min(1, exp((β_i - β_j) × (log_π(θ_j) - log_π(θ_i))))

    where log_π is the untempered log-posterior.

    Args:
        key: JAX random key
        states_A: Chain states for group A (n_chains_a, n_params)
        states_B: Chain states for group B (n_chains_b, n_params)
        temperature_ladder: Temperature values (n_temperatures,)
        log_post_fn: Untempered log posterior function
        swap_accepts: Running count of accepted swaps per temp pair
        swap_attempts: Running count of attempted swaps per temp pair

    Returns:
        Updated (states_A, states_B, key, swap_accepts, swap_attempts)
    """
    n_temperatures = temperature_ladder.shape[0]

    # If only one temperature, no swaps needed
    if n_temperatures <= 1:
        return states_A, states_B, key, swap_accepts, swap_attempts

    # Combine A and B states for swapping
    all_states = jnp.concatenate([states_A, states_B], axis=0)
    n_chains = all_states.shape[0]
    n_chains_a = states_A.shape[0]

    # Chains are assigned to temperatures in order: chains_per_temp chains per temperature
    # E.g., with 32 chains and 4 temps: chains 0-7 at temp 0, 8-15 at temp 1, etc.
    chains_per_temp = n_chains // n_temperatures

    # Parameter indices for log posterior evaluation
    all_param_indices = jnp.arange(all_states.shape[1])

    def swap_one_pair(carry, temp_idx):
        """Attempt swap between temperatures temp_idx and temp_idx+1."""
        current_states, current_key, accepts, attempts = carry

        # Split key for this swap attempt
        current_key, swap_key, select_key_i, select_key_j = random.split(current_key, 4)

        # Compute start indices for chains at each temperature
        # Chains are arranged: [temp0 chains][temp1 chains][temp2 chains]...
        start_i = temp_idx * chains_per_temp
        start_j = (temp_idx + 1) * chains_per_temp

        # Select random chain from each temperature group
        offset_i = random.randint(select_key_i, (), 0, chains_per_temp)
        offset_j = random.randint(select_key_j, (), 0, chains_per_temp)

        chain_i = start_i + offset_i
        chain_j = start_j + offset_j

        # Get temperatures
        beta_i = temperature_ladder[temp_idx]
        beta_j = temperature_ladder[temp_idx + 1]

        # Compute log posteriors only for the two selected chains (not all chains)
        log_pi_i = log_post_fn(current_states[chain_i], all_param_indices)
        log_pi_j = log_post_fn(current_states[chain_j], all_param_indices)

        # Swap acceptance ratio
        # α = exp((β_i - β_j) × (log_π(θ_j) - log_π(θ_i)))
        # Note: β_i > β_j (colder temp has higher beta), so β_i - β_j > 0
        log_alpha = (beta_i - beta_j) * (log_pi_j - log_pi_i)
        log_alpha = jnp.nan_to_num(log_alpha, nan=-jnp.inf)

        # Accept/reject
        log_uniform = jnp.log(random.uniform(swap_key))
        accept = log_uniform < log_alpha

        # Swap states if accepted
        state_i = current_states[chain_i]
        state_j = current_states[chain_j]
        new_states = jax.lax.cond(
            accept,
            lambda s: s.at[chain_i].set(state_j).at[chain_j].set(state_i),
            lambda s: s,
            current_states
        )

        # Update counts
        new_accepts = accepts.at[temp_idx].add(jnp.int32(accept))
        new_attempts = attempts.at[temp_idx].add(jnp.int32(1))

        return (new_states, current_key, new_accepts, new_attempts), None

    # Attempt swaps for all adjacent temperature pairs
    init_carry = (all_states, key, swap_accepts, swap_attempts)
    (final_states, final_key, final_accepts, final_attempts), _ = jax.lax.scan(
        swap_one_pair,
        init_carry,
        jnp.arange(n_temperatures - 1)
    )

    # Split back into A and B groups
    new_states_A = final_states[:n_chains_a]
    new_states_B = final_states[n_chains_a:]

    return new_states_A, new_states_B, final_key, final_accepts, final_attempts


def mcmc_scan_body_offload(carry, step_idx, log_post_fn, grad_log_post_fn, direct_sampler_fn,
                           coupled_transform_fn, gq_fn, block_arrays: BlockArrays, run_params):
    """
    One iteration of the MCMC scan body.

    This function updates both groups A and B in sequence.
    Block statistics (mean, cov) are precomputed ONCE before vmapping across chains.

    Carry tuple structure (13 elements):
        0: states_A - Chain states for group A
        1: keys_A - RNG keys for group A
        2: states_B - Chain states for group B
        3: keys_B - RNG keys for group B
        4: history_array - Collected samples
        5: lik_history_array - Likelihood history (if enabled)
        6: acceptance_counts - Per-block acceptance counts
        7: current_iteration - Current iteration number
        8: temperature_ladder - Temperature values (n_temperatures,)
        9: temp_assignments_A - Temperature indices for group A chains
        10: temp_assignments_B - Temperature indices for group B chains
        11: swap_accepts - Swap acceptance counts per temperature pair
        12: swap_attempts - Swap attempt counts per temperature pair
    """
    (states_A, keys_A, states_B, keys_B, history_array, lik_history_array,
     acceptance_counts, current_iteration,
     temperature_ladder, temp_assignments_A, temp_assignments_B,
     swap_accepts, swap_attempts) = carry

    # Precompute block statistics ONCE from states_B (for updating A)
    # Pass log_post_fn to compute mode for MODE_WEIGHTED proposals
    means_A, covs_A, coupled_blocks_A, modes_A = compute_block_statistics(
        states_B, block_arrays.indices, block_arrays.masks, log_post_fn
    )

    # Compute beta values for each chain based on temperature assignments
    # betas_A[i] = temperature_ladder[temp_assignments_A[i]]
    betas_A = temperature_ladder[temp_assignments_A]
    betas_B = temperature_ladder[temp_assignments_B]

    # Update Group A
    next_states_A, next_keys_A, lps_A, accepts_A = parallel_gibbs_iteration(
        keys_A, states_A, means_A, covs_A, coupled_blocks_A, modes_A,
        block_arrays,
        log_post_fn, grad_log_post_fn, direct_sampler_fn, coupled_transform_fn,
        betas_A
    )

    # Precompute block statistics ONCE from updated states_A (for updating B)
    means_B, covs_B, coupled_blocks_B, modes_B = compute_block_statistics(
        next_states_A, block_arrays.indices, block_arrays.masks, log_post_fn
    )

    # Update Group B
    next_states_B, next_keys_B, lps_B, accepts_B = parallel_gibbs_iteration(
        keys_B, states_B, means_B, covs_B, coupled_blocks_B, modes_B,
        block_arrays,
        log_post_fn, grad_log_post_fn, direct_sampler_fn, coupled_transform_fn,
        betas_B
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

    # Attempt temperature swaps for parallel tempering
    # This allows high-temperature chains to share good configurations with cold chains
    n_temperatures = temperature_ladder.shape[0]

    def do_swaps(operand):
        """Perform temperature swaps."""
        states_a, states_b, keys_a, s_accepts, s_attempts = operand
        # Use keys_A[0] for swap randomness, split to get new key
        swap_key, new_key_for_chain_0 = random.split(keys_a[0])
        keys_a = keys_a.at[0].set(new_key_for_chain_0)

        new_states_a, new_states_b, _, new_s_accepts, new_s_attempts = attempt_temperature_swaps(
            swap_key,
            states_a, states_b,
            temperature_ladder,
            log_post_fn,
            s_accepts, s_attempts
        )
        return new_states_a, new_states_b, keys_a, new_s_accepts, new_s_attempts

    def skip_swaps(operand):
        """No swaps needed for single temperature."""
        return operand

    # Use lax.cond to handle both cases in JAX-traceable way
    next_states_A, next_states_B, next_keys_A, swap_accepts, swap_attempts = jax.lax.cond(
        n_temperatures > 1,
        do_swaps,
        skip_swaps,
        (next_states_A, next_states_B, next_keys_A, swap_accepts, swap_attempts)
    )

    # Return extended carry tuple with temperature arrays
    next_carry = (
        next_states_A, next_keys_A, next_states_B, next_keys_B,
        next_history, next_lik_history, updated_acceptance_counts, next_iteration,
        temperature_ladder, temp_assignments_A, temp_assignments_B,
        swap_accepts, swap_attempts
    )

    return next_carry, None
