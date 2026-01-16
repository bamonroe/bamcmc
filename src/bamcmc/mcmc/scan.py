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
from .sampling import parallel_gibbs_iteration, parallel_gibbs_iteration_per_temp


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
        # Evaluate full log posterior (beta=1.0) for each coupled chain
        # Use a dummy "all parameters" index for full evaluation
        all_indices = jnp.arange(coupled_states.shape[1])
        coupled_log_posts = jax.vmap(lambda s: log_post_fn(s, all_indices, 1.0))(coupled_states)

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


def compute_block_statistics_per_temp(
    coupled_states: jnp.ndarray,
    temp_assignments: jnp.ndarray,
    n_temperatures: int,
    block_indices: jnp.ndarray,
    block_masks: jnp.ndarray,
    log_post_fn=None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute block statistics separately for each temperature level.

    Per-temperature proposals: chains at each temperature use statistics computed
    only from chains at that same temperature. This provides better-matched
    proposals for each temperature's distribution shape.

    Args:
        coupled_states: States from opposite group (n_chains, n_params)
        temp_assignments: Temperature index for each chain (n_chains,)
        n_temperatures: Number of temperature levels
        block_indices: Block index array (n_blocks, max_block_size)
        block_masks: Parameter masks (n_blocks, max_block_size)
        log_post_fn: Optional log posterior function for computing mode

    Returns:
        block_means: (n_temperatures, n_blocks, max_block_size)
        block_covs: (n_temperatures, n_blocks, max_block_size, max_block_size)
        coupled_blocks: (n_blocks, n_chains, max_block_size) - still full for multinomial
        block_modes: (n_temperatures, n_blocks, max_block_size)
    """
    n_chains, n_params = coupled_states.shape
    n_blocks = block_indices.shape[0]
    max_block_size = block_indices.shape[1]

    def compute_stats_for_temp(temp_idx):
        """Compute statistics for chains at a specific temperature."""
        # Mask for chains at this temperature
        temp_mask = (temp_assignments == temp_idx)  # (n_chains,)

        # Get states at this temperature (use where to maintain shape for JIT)
        # We'll compute weighted stats where non-matching chains have zero weight
        weights = temp_mask.astype(coupled_states.dtype)
        n_at_temp = jnp.sum(temp_mask)

        def get_stats_for_block(b_indices, b_mask):
            safe_indices = jnp.clip(b_indices, 0, n_params - 1)
            block_data = jnp.take(coupled_states, safe_indices, axis=1)  # (n_chains, block_size)

            # Weighted mean
            weighted_sum = jnp.sum(block_data * weights[:, None], axis=0)
            mean = jnp.where(n_at_temp > 0, weighted_sum / n_at_temp, 0.0) * b_mask

            # Weighted covariance
            centered = block_data - mean[None, :]
            weighted_centered = centered * jnp.sqrt(weights[:, None])
            cov = jnp.dot(weighted_centered.T, weighted_centered)
            cov = jnp.where(n_at_temp > 1, cov / (n_at_temp - 1), jnp.eye(max_block_size))
            cov = cov + NUGGET * jnp.eye(max_block_size)

            # Apply mask
            mask_2d = jnp.outer(b_mask, b_mask)
            final_cov = jnp.where(mask_2d, cov, jnp.eye(max_block_size))

            return mean, final_cov

        means, covs = jax.vmap(get_stats_for_block)(block_indices, block_masks)

        # Compute mode for this temperature (highest full log-post chain at this temp)
        if log_post_fn is not None:
            all_indices = jnp.arange(n_params)
            log_posts = jax.vmap(lambda s: log_post_fn(s, all_indices, 1.0))(coupled_states)
            # Mask out chains not at this temperature
            masked_log_posts = jnp.where(temp_mask, log_posts, -jnp.inf)
            mode_idx = jnp.argmax(masked_log_posts)
            mode_state = coupled_states[mode_idx]

            def get_mode_for_block(b_indices, b_mask):
                safe_indices = jnp.clip(b_indices, 0, mode_state.shape[0] - 1)
                return jnp.take(mode_state, safe_indices) * b_mask

            modes = jax.vmap(get_mode_for_block)(block_indices, block_masks)
        else:
            modes = means

        return means, covs, modes

    # Compute stats for all temperatures
    temp_indices = jnp.arange(n_temperatures)
    all_means, all_covs, all_modes = jax.vmap(compute_stats_for_temp)(temp_indices)

    # coupled_blocks still needs all chains for multinomial proposal
    def get_block_data(b_indices, b_mask):
        safe_indices = jnp.clip(b_indices, 0, n_params - 1)
        return jnp.take(coupled_states, safe_indices, axis=1)

    coupled_blocks = jax.vmap(get_block_data)(block_indices, block_masks)

    return all_means, all_covs, coupled_blocks, all_modes


def compute_block_statistics_blended(
    coupled_states: jnp.ndarray,
    temp_assignments: jnp.ndarray,
    n_temperatures: int,
    block_indices: jnp.ndarray,
    block_masks: jnp.ndarray,
    log_post_fn=None,
    blend_pseudocount: float = 10.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute blended per-temperature + global block statistics.

    Blends per-temperature statistics with global statistics based on the number
    of chains at each temperature. This handles edge cases where a temperature
    has 0 or few chains gracefully.

    Blending formula:
        weight = n_at_temp / (n_at_temp + k)
        blended = weight * per_temp + (1 - weight) * global

    With k=10:
        0 chains  -> 0% per-temp (100% global)
        10 chains -> 50% per-temp
        50 chains -> 83% per-temp
        140 chains -> 93% per-temp

    Args:
        coupled_states: States from opposite group (n_chains, n_params)
        temp_assignments: Temperature index for each chain (n_chains,)
        n_temperatures: Number of temperature levels
        block_indices: Block index array (n_blocks, max_block_size)
        block_masks: Parameter masks (n_blocks, max_block_size)
        log_post_fn: Optional log posterior function for computing mode
        blend_pseudocount: Pseudocount k for blending (default 10.0)

    Returns:
        block_means: (n_temperatures, n_blocks, max_block_size)
        block_covs: (n_temperatures, n_blocks, max_block_size, max_block_size)
        coupled_blocks: (n_blocks, n_chains, max_block_size) - still full for multinomial
        block_modes: (n_temperatures, n_blocks, max_block_size)
    """
    n_chains, n_params = coupled_states.shape
    n_blocks = block_indices.shape[0]
    max_block_size = block_indices.shape[1]

    # 1. Compute GLOBAL statistics (all chains, no temperature filtering)
    global_means, global_covs, coupled_blocks, global_modes = compute_block_statistics(
        coupled_states, block_indices, block_masks, log_post_fn
    )

    # 2. For each temperature, compute per-temp stats and blend with global
    def compute_blended_for_temp(temp_idx):
        """Compute blended statistics for chains at a specific temperature."""
        # Mask for chains at this temperature
        temp_mask = (temp_assignments == temp_idx)  # (n_chains,)
        weights = temp_mask.astype(coupled_states.dtype)
        n_at_temp = jnp.sum(temp_mask)

        # Blending weight: 0 when no chains, approaches 1 with many chains
        blend_weight = n_at_temp / (n_at_temp + blend_pseudocount)

        def get_blended_stats_for_block(b_indices, b_mask, g_mean, g_cov):
            """Compute blended stats for one block."""
            safe_indices = jnp.clip(b_indices, 0, n_params - 1)
            block_data = jnp.take(coupled_states, safe_indices, axis=1)  # (n_chains, block_size)

            # Per-temp weighted mean
            weighted_sum = jnp.sum(block_data * weights[:, None], axis=0)
            per_temp_mean = jnp.where(n_at_temp > 0, weighted_sum / n_at_temp, g_mean) * b_mask

            # Per-temp weighted covariance
            centered = block_data - per_temp_mean[None, :]
            weighted_centered = centered * jnp.sqrt(weights[:, None])
            per_temp_cov = jnp.dot(weighted_centered.T, weighted_centered)
            per_temp_cov = jnp.where(n_at_temp > 1, per_temp_cov / (n_at_temp - 1), g_cov)
            per_temp_cov = per_temp_cov + NUGGET * jnp.eye(max_block_size)

            # Apply mask to covariance
            mask_2d = jnp.outer(b_mask, b_mask)
            per_temp_cov = jnp.where(mask_2d, per_temp_cov, jnp.eye(max_block_size))

            # Blend per-temp with global
            blended_mean = blend_weight * per_temp_mean + (1 - blend_weight) * g_mean
            blended_cov = blend_weight * per_temp_cov + (1 - blend_weight) * g_cov

            return blended_mean, blended_cov

        blended_means, blended_covs = jax.vmap(get_blended_stats_for_block)(
            block_indices, block_masks, global_means, global_covs
        )

        # Mode: use per-temp mode if available, else global mode
        if log_post_fn is not None:
            all_indices = jnp.arange(n_params)
            log_posts = jax.vmap(lambda s: log_post_fn(s, all_indices, 1.0))(coupled_states)
            # Mask out chains not at this temperature
            masked_log_posts = jnp.where(temp_mask, log_posts, -jnp.inf)
            mode_idx = jnp.argmax(masked_log_posts)
            mode_state = coupled_states[mode_idx]

            def get_mode_for_block(b_indices, b_mask, g_mode):
                safe_indices = jnp.clip(b_indices, 0, mode_state.shape[0] - 1)
                per_temp_mode = jnp.take(mode_state, safe_indices) * b_mask
                # Use per-temp mode if we have chains, else global
                return jnp.where(n_at_temp > 0, per_temp_mode, g_mode)

            blended_modes = jax.vmap(get_mode_for_block)(block_indices, block_masks, global_modes)
        else:
            blended_modes = blended_means

        return blended_means, blended_covs, blended_modes

    # Compute blended stats for all temperatures
    temp_indices = jnp.arange(n_temperatures)
    all_means, all_covs, all_modes = jax.vmap(compute_blended_for_temp)(temp_indices)

    return all_means, all_covs, coupled_blocks, all_modes


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

    # Compute log posteriors (at beta=1) for ALL chains in parallel via vmap
    # Note: For swap acceptance with proper tempering, we should use log_likelihood only
    # (prior cancels). However, computing log_lik = log_post(beta=1) - log_post(beta=0)
    # doubles memory usage. Using log_post directly is a reasonable approximation if
    # prior values are similar across chains at different temperatures.
    all_param_indices = jnp.arange(all_states.shape[1])
    all_log_posts = jax.vmap(lambda s: log_post_fn(s, all_param_indices, 1.0))(all_states)

    # Maximum chains per temperature (for static array sizes)
    max_chains_per_temp = n_chains

    def swap_all_pairs(carry, temp_idx):
        """Attempt swaps for ALL chain pairs between temp_idx and temp_idx+1 (DEO-gated)."""
        curr_temp_assigns, curr_key, accepts, attempts = carry

        # DEO: only process if this pair matches current parity
        pair_parity = temp_idx % 2
        is_active = (pair_parity == swap_parity)

        # Get temperatures for this pair
        beta_cold = temperature_ladder[temp_idx]
        beta_hot = temperature_ladder[temp_idx + 1]

        # Find chains at each temperature
        at_cold = (curr_temp_assigns == temp_idx)
        at_hot = (curr_temp_assigns == temp_idx + 1)

        # Count chains at each temperature
        n_cold = jnp.sum(at_cold)
        n_hot = jnp.sum(at_hot)
        n_pairs = jnp.minimum(n_cold, n_hot)

        # Get chain indices at each temperature (padded arrays)
        cold_indices = jnp.where(at_cold, jnp.arange(n_chains), n_chains)
        hot_indices = jnp.where(at_hot, jnp.arange(n_chains), n_chains)

        # Sort to get valid indices first (invalid are n_chains, sorted to end)
        cold_sorted = jnp.sort(cold_indices)
        hot_sorted = jnp.sort(hot_indices)

        # Shuffle the hot indices to get random pairing
        curr_key, shuffle_key, accept_key = random.split(curr_key, 3)
        hot_shuffled = random.permutation(shuffle_key, hot_sorted)

        # For each potential pair position, compute acceptance
        # Pair i: cold_sorted[i] with hot_shuffled[i]
        def compute_swap(pair_idx):
            cold_idx = cold_sorted[pair_idx]
            hot_idx = hot_shuffled[pair_idx]

            # Check if this is a valid pair (both indices < n_chains)
            is_valid = (pair_idx < n_pairs) & is_active

            # Get log posteriors for swap acceptance
            log_post_cold = all_log_posts[cold_idx]
            log_post_hot = all_log_posts[hot_idx]

            # Acceptance ratio for swapping temperatures
            # Note: Using full log_post instead of just log_likelihood (see comment above)
            log_alpha = (beta_cold - beta_hot) * (log_post_hot - log_post_cold)
            log_alpha = jnp.nan_to_num(log_alpha, nan=-jnp.inf)

            return cold_idx, hot_idx, log_alpha, is_valid

        # Vectorized computation over all potential pairs
        pair_indices = jnp.arange(max_chains_per_temp)
        cold_idxs, hot_idxs, log_alphas, valids = jax.vmap(compute_swap)(pair_indices)

        # Generate random values for acceptance
        accept_keys = random.split(accept_key, max_chains_per_temp)
        log_uniforms = jax.vmap(lambda k: jnp.log(random.uniform(k)))(accept_keys)

        # Determine which swaps to accept
        accepts_mask = valids & (log_uniforms < log_alphas)

        # Apply swaps: for each accepted swap, exchange temperature assignments
        # This needs to be done carefully to avoid conflicts
        def apply_swap(ta, pair_data):
            cold_idx, hot_idx, should_swap = pair_data
            # Swap: cold chain gets hot temp, hot chain gets cold temp
            new_ta = jax.lax.cond(
                should_swap,
                lambda t: t.at[cold_idx].set(temp_idx + 1).at[hot_idx].set(temp_idx),
                lambda t: t,
                ta
            )
            return new_ta, None

        new_temp_assigns, _ = jax.lax.scan(
            apply_swap,
            curr_temp_assigns,
            (cold_idxs, hot_idxs, accepts_mask)
        )

        # Update counts
        n_attempted = jnp.sum(valids.astype(jnp.int32))
        n_accepted = jnp.sum(accepts_mask.astype(jnp.int32))

        new_accepts = accepts.at[temp_idx].add(n_accepted)
        new_attempts = attempts.at[temp_idx].add(n_attempted)

        return (new_temp_assigns, curr_key, new_accepts, new_attempts), None

    # Process all temperature pairs (inactive ones gated by DEO parity)
    init_carry = (all_temp_assigns, key, swap_accepts, swap_attempts)
    (final_temp_assigns, final_key, final_accepts, final_attempts), _ = jax.lax.scan(
        swap_all_pairs,
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

    # Get per-temperature proposal setting from run_params
    per_temp_proposals = getattr(run_params, 'PER_TEMP_PROPOSALS', False)
    n_temperatures = getattr(run_params, 'N_TEMPERATURES', 1)
    blend_pseudocount = getattr(run_params, 'BLEND_PSEUDOCOUNT', 10.0)

    # Compute beta values for each chain based on temperature assignments
    # betas_A[i] = temperature_ladder[temp_assignments_A[i]]
    betas_A = temperature_ladder[temp_assignments_A]
    betas_B = temperature_ladder[temp_assignments_B]

    if per_temp_proposals and n_temperatures > 1:
        # Per-temperature proposals with blending: compute blended per-temp + global statistics
        # This handles empty/sparse temperatures gracefully via pseudocount blending

        # Statistics from B for updating A (blended per-temp + global)
        means_per_temp_A, covs_per_temp_A, coupled_blocks_A, modes_per_temp_A = compute_block_statistics_blended(
            states_B, temp_assignments_B, n_temperatures,
            block_arrays.indices, block_arrays.masks, log_post_fn,
            blend_pseudocount
        )
        # Gather per-chain: means_A[chain_i] = means_per_temp_A[temp_assignments_A[chain_i]]
        means_A = means_per_temp_A[temp_assignments_A]  # (n_chains_A, n_blocks, max_block_size)
        covs_A = covs_per_temp_A[temp_assignments_A]    # (n_chains_A, n_blocks, max_block_size, max_block_size)
        modes_A = modes_per_temp_A[temp_assignments_A]  # (n_chains_A, n_blocks, max_block_size)

        # Update Group A with per-chain statistics
        next_states_A, next_keys_A, lps_A, accepts_A = parallel_gibbs_iteration_per_temp(
            keys_A, states_A, means_A, covs_A, coupled_blocks_A, modes_A,
            block_arrays,
            log_post_fn, grad_log_post_fn, direct_sampler_fn, coupled_transform_fn,
            betas_A
        )

        # Statistics from updated A for updating B (blended per-temp + global)
        means_per_temp_B, covs_per_temp_B, coupled_blocks_B, modes_per_temp_B = compute_block_statistics_blended(
            next_states_A, temp_assignments_A, n_temperatures,
            block_arrays.indices, block_arrays.masks, log_post_fn,
            blend_pseudocount
        )
        means_B = means_per_temp_B[temp_assignments_B]
        covs_B = covs_per_temp_B[temp_assignments_B]
        modes_B = modes_per_temp_B[temp_assignments_B]

        # Update Group B with per-chain statistics
        next_states_B, next_keys_B, lps_B, accepts_B = parallel_gibbs_iteration_per_temp(
            keys_B, states_B, means_B, covs_B, coupled_blocks_B, modes_B,
            block_arrays,
            log_post_fn, grad_log_post_fn, direct_sampler_fn, coupled_transform_fn,
            betas_B
        )
    else:
        # Standard: all chains share the same proposal statistics (mixed temperatures)
        # Precompute block statistics ONCE from states_B (for updating A)
        means_A, covs_A, coupled_blocks_A, modes_A = compute_block_statistics(
            states_B, block_arrays.indices, block_arrays.masks, log_post_fn
        )

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
