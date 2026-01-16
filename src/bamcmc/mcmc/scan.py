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


def _save_with_gq(states_A, states_B, temp_assigns_A, temp_assigns_B, gq_fn, num_gq):
    """Combine states, compute GQ for all chains, return with temperature indices.

    INDEX PROCESS: Save ALL chains (no filtering). Users filter to beta=1 post-hoc
    using the returned temperature assignments.

    Args:
        states_A: Chain states for group A (n_chains_a, n_params)
        states_B: Chain states for group B (n_chains_b, n_params)
        temp_assigns_A: Temperature index per chain in A (n_chains_a,)
        temp_assigns_B: Temperature index per chain in B (n_chains_b,)
        gq_fn: Generated quantities function (or None)
        num_gq: Number of generated quantities

    Returns:
        states_with_gq: All chain states with GQ (n_chains, n_params + num_gq)
        temp_indices: Temperature index per chain (n_chains,)
    """
    full_state = jnp.concatenate((states_A, states_B), axis=0)
    full_temp_assigns = jnp.concatenate((temp_assigns_A, temp_assigns_B), axis=0)

    if num_gq > 0 and gq_fn is not None:
        gq_values = jax.vmap(gq_fn)(full_state)
        states_with_gq = jnp.concatenate((full_state, gq_values), axis=1)
    else:
        states_with_gq = full_state

    return states_with_gq, full_temp_assigns


def attempt_temperature_swaps(
    key,
    states_A, states_B,
    temp_assignments_A, temp_assignments_B,
    temperature_ladder,
    log_post_fn,
    swap_accepts, swap_attempts,
    swap_parity
):
    """
    Index process parallel tempering with DEO scheme (Syed et al. 2021).

    KEY DIFFERENCE from standard PT: We swap temperature ASSIGNMENTS, not chain states.
    Each chain maintains its continuous parameter trace; only its temperature label changes.

    DEO (Deterministic Even-Odd) Scheme:
    - Even round (parity=0): attempt swaps for pairs (0,1), (2,3), (4,5), ...
    - Odd round (parity=1): attempt swaps for pairs (1,2), (3,4), (5,6), ...
    - This creates deterministic "conveyor belt" round trips through temperature space
    - Reduces round-trip time from O(N²) to O(N) compared to stochastic SEO

    Swap acceptance probability between temperatures β_i and β_j (β_i > β_j):
        α = min(1, exp((β_i - β_j) × (log_π(θ_j) - log_π(θ_i))))

    where log_π is the untempered log-posterior.

    Args:
        key: JAX random key
        states_A: Chain states for group A (n_chains_a, n_params) - NOT modified
        states_B: Chain states for group B (n_chains_b, n_params) - NOT modified
        temp_assignments_A: Temperature index per chain in A (n_chains_a,)
        temp_assignments_B: Temperature index per chain in B (n_chains_b,)
        temperature_ladder: Temperature values (n_temperatures,)
        log_post_fn: Untempered log posterior function
        swap_accepts: Running count of accepted swaps per temp pair
        swap_attempts: Running count of attempted swaps per temp pair
        swap_parity: DEO parity (0 = even pairs, 1 = odd pairs)

    Returns:
        (temp_assignments_A, temp_assignments_B, key, swap_accepts, swap_attempts, next_parity)

        Note: States are NOT returned - they don't change in the index process!
    """
    n_temperatures = temperature_ladder.shape[0]

    # If only one temperature, no swaps needed
    if n_temperatures <= 1:
        return temp_assignments_A, temp_assignments_B, key, swap_accepts, swap_attempts, swap_parity

    # Combine for unified processing
    all_states = jnp.concatenate([states_A, states_B], axis=0)
    all_temp_assigns = jnp.concatenate([temp_assignments_A, temp_assignments_B], axis=0)
    n_chains = all_states.shape[0]
    n_chains_a = states_A.shape[0]

    # Compute log posteriors for ALL chains in parallel via vmap (GPU-efficient)
    all_param_indices = jnp.arange(all_states.shape[1])
    all_log_posts = jax.vmap(lambda s: log_post_fn(s, all_param_indices))(all_states)

    def swap_one_pair(carry, temp_idx):
        """Attempt swap between temperatures temp_idx and temp_idx+1 (DEO-gated)."""
        curr_temp_assigns, curr_key, accepts, attempts = carry

        # Split key for this swap attempt
        curr_key, swap_key = random.split(curr_key, 2)

        # DEO: only process if this pair matches current parity
        # Even pairs (0, 2, 4, ...) on parity=0, odd pairs (1, 3, 5, ...) on parity=1
        pair_parity = temp_idx % 2
        is_active = (pair_parity == swap_parity)

        # Get temperatures for this pair
        beta_cold = temperature_ladder[temp_idx]      # Higher beta = colder
        beta_hot = temperature_ladder[temp_idx + 1]   # Lower beta = hotter

        # Find chains at each temperature
        # INDEX PROCESS: chains don't move, temperature assignments do
        at_cold = (curr_temp_assigns == temp_idx)
        at_hot = (curr_temp_assigns == temp_idx + 1)

        # Select one chain from each temperature (use argmax to get first match)
        # Safe to use - if temperature is unpopulated, acceptance will fail anyway
        cold_chain_idx = jnp.argmax(at_cold.astype(jnp.int32))
        hot_chain_idx = jnp.argmax(at_hot.astype(jnp.int32))

        # Check we have chains at both temperatures
        has_cold = jnp.any(at_cold)
        has_hot = jnp.any(at_hot)
        should_attempt = is_active & has_cold & has_hot

        # Get log posteriors (untempered)
        log_pi_cold = all_log_posts[cold_chain_idx]
        log_pi_hot = all_log_posts[hot_chain_idx]

        # Swap acceptance ratio
        # α = exp((β_cold - β_hot) × (log_π(θ_hot) - log_π(θ_cold)))
        # If hot chain has higher log posterior, swap is favored
        log_alpha = (beta_cold - beta_hot) * (log_pi_hot - log_pi_cold)
        log_alpha = jnp.nan_to_num(log_alpha, nan=-jnp.inf)

        # Accept/reject
        log_uniform = jnp.log(random.uniform(swap_key))
        accept = should_attempt & (log_uniform < log_alpha)

        # INDEX PROCESS: swap temperature assignments, NOT states!
        # Cold chain gets hot temp assignment, hot chain gets cold temp assignment
        new_temp_assigns = jax.lax.cond(
            accept,
            lambda ta: ta.at[cold_chain_idx].set(temp_idx + 1).at[hot_chain_idx].set(temp_idx),
            lambda ta: ta,
            curr_temp_assigns
        )

        # Update counts (only if we actually attempted)
        new_accepts = jax.lax.cond(
            accept,
            lambda a: a.at[temp_idx].add(jnp.int32(1)),
            lambda a: a,
            accepts
        )
        new_attempts = jax.lax.cond(
            should_attempt,
            lambda a: a.at[temp_idx].add(jnp.int32(1)),
            lambda a: a,
            attempts
        )

        return (new_temp_assigns, curr_key, new_accepts, new_attempts), None

    # Process all temperature pairs (inactive ones gated by DEO parity)
    init_carry = (all_temp_assigns, key, swap_accepts, swap_attempts)
    (final_temp_assigns, final_key, final_accepts, final_attempts), _ = jax.lax.scan(
        swap_one_pair,
        init_carry,
        jnp.arange(n_temperatures - 1)
    )

    # Split temp assignments back into A and B groups
    new_temp_assignments_A = final_temp_assigns[:n_chains_a]
    new_temp_assignments_B = final_temp_assigns[n_chains_a:]

    # Toggle parity for next round (DEO: E→O→E→O→...)
    next_parity = 1 - swap_parity

    return (new_temp_assignments_A, new_temp_assignments_B, final_key,
            final_accepts, final_attempts, next_parity)


def mcmc_scan_body_offload(carry, step_idx, log_post_fn, grad_log_post_fn, direct_sampler_fn,
                           coupled_transform_fn, gq_fn, block_arrays: BlockArrays, run_params):
    """
    One iteration of the MCMC scan body.

    This function updates both groups A and B in sequence.
    Block statistics (mean, cov) are precomputed ONCE before vmapping across chains.

    INDEX PROCESS: Temperature swaps modify temp_assignments, not states.
    Each chain maintains a continuous parameter trace.

    Carry tuple structure (15 elements):
        0: states_A - Chain states for group A
        1: keys_A - RNG keys for group A
        2: states_B - Chain states for group B
        3: keys_B - RNG keys for group B
        4: history_array - Collected samples (all chains)
        5: lik_history_array - Likelihood history (if enabled)
        6: temp_history_array - Temperature index per chain per saved iteration
        7: acceptance_counts - Per-block acceptance counts
        8: current_iteration - Current iteration number
        9: temperature_ladder - Temperature values (n_temperatures,)
        10: temp_assignments_A - Temperature indices for group A chains
        11: temp_assignments_B - Temperature indices for group B chains
        12: swap_accepts - Swap acceptance counts per temperature pair
        13: swap_attempts - Swap attempt counts per temperature pair
        14: swap_parity - DEO parity (0=even pairs, 1=odd pairs)
    """
    (states_A, keys_A, states_B, keys_B,
     history_array, lik_history_array, temp_history_array,
     acceptance_counts, current_iteration,
     temperature_ladder, temp_assignments_A, temp_assignments_B,
     swap_accepts, swap_attempts, swap_parity) = carry

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
        n_chains_to_save = run_params.get('N_CHAINS_TO_SAVE', states_A.shape[0] + states_B.shape[0])
    else:
        start_iter = run_params.START_ITERATION
        burn_iter = run_params.BURN_ITER
        thin_iter = run_params.THIN_ITERATION
        num_collect = run_params.NUM_COLLECT
        num_gq = run_params.NUM_GQ
        n_chains_to_save = run_params.N_CHAINS_TO_SAVE

    run_iteration = current_iteration - start_iter
    is_after_burn_in = (run_iteration >= burn_iter)
    collection_iteration = run_iteration - burn_iter
    is_thin_iter = (collection_iteration + 1) % thin_iter == 0

    next_history = history_array
    next_lik_history = lik_history_array
    next_temp_history = temp_history_array

    if num_collect > 0:
        thin_idx = collection_iteration // thin_iter
        should_save = is_after_burn_in & is_thin_iter & (thin_idx >= 0) & (thin_idx < num_collect)

        def save_history(h_array):
            """Save states and GQ for all chains."""
            states_with_gq, _ = _save_with_gq(
                next_states_A, next_states_B,
                temp_assignments_A, temp_assignments_B,
                gq_fn, num_gq
            )
            return h_array.at[thin_idx].set(states_with_gq)

        next_history = jax.lax.cond(should_save, save_history, lambda h: h, history_array)

        # Save temperature history only if tempering is active (array has proper shape)
        if temp_history_array.size > 1:
            def save_temp(th_array):
                _, temp_indices = _save_with_gq(
                    next_states_A, next_states_B,
                    temp_assignments_A, temp_assignments_B,
                    gq_fn, num_gq
                )
                return th_array.at[thin_idx].set(temp_indices)

            next_temp_history = jax.lax.cond(should_save, save_temp, lambda h: h, temp_history_array)

        if lik_history_array.size > 1:
            def save_lik(l_array):
                total_lps_A = jnp.sum(lps_A, axis=1)
                total_lps_B = jnp.sum(lps_B, axis=1)
                # Save likelihoods for ALL chains (index process: no filtering)
                full_lps = jnp.concatenate((total_lps_A, total_lps_B), axis=0)
                return l_array.at[thin_idx].set(full_lps)

            next_lik_history = jax.lax.cond(should_save, save_lik, lambda h: h, lik_history_array)

    next_iteration = current_iteration + 1

    # Attempt temperature swaps for parallel tempering (INDEX PROCESS with DEO)
    # Swaps modify temperature assignments, not states - chains keep their traces
    n_temperatures = temperature_ladder.shape[0]

    def do_swaps(operand):
        """Perform temperature swaps (index process: swap assignments, not states)."""
        temp_a, temp_b, keys_a, s_accepts, s_attempts, parity = operand
        # Use keys_A[0] for swap randomness, split to get new key
        swap_key, new_key_for_chain_0 = random.split(keys_a[0])
        keys_a = keys_a.at[0].set(new_key_for_chain_0)

        # Index process: states passed in but NOT modified, only temp assignments change
        new_temp_a, new_temp_b, _, new_s_accepts, new_s_attempts, new_parity = attempt_temperature_swaps(
            swap_key,
            next_states_A, next_states_B,  # States for log posterior eval only
            temp_a, temp_b,
            temperature_ladder,
            log_post_fn,
            s_accepts, s_attempts, parity
        )
        return new_temp_a, new_temp_b, keys_a, new_s_accepts, new_s_attempts, new_parity

    def skip_swaps(operand):
        """No swaps needed for single temperature."""
        return operand

    # Use lax.cond to handle both cases in JAX-traceable way
    # INDEX PROCESS: states don't change, only temperature assignments
    (temp_assignments_A, temp_assignments_B, next_keys_A,
     swap_accepts, swap_attempts, swap_parity) = jax.lax.cond(
        n_temperatures > 1,
        do_swaps,
        skip_swaps,
        (temp_assignments_A, temp_assignments_B, next_keys_A,
         swap_accepts, swap_attempts, swap_parity)
    )

    # Return extended carry tuple (15 elements) with index process state
    next_carry = (
        next_states_A, next_keys_A, next_states_B, next_keys_B,
        next_history, next_lik_history, next_temp_history,
        updated_acceptance_counts, next_iteration,
        temperature_ladder, temp_assignments_A, temp_assignments_B,
        swap_accepts, swap_attempts, swap_parity
    )

    return next_carry, None
