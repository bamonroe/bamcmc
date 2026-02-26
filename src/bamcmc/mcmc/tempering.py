"""
MCMC Tempering - Parallel tempering temperature swap logic.

This module implements index-process parallel tempering with optional
DEO (Deterministic Even-Odd) scheme (Syed et al. 2021).

Key difference from standard PT: temperature ASSIGNMENTS are swapped,
not chain states. Each chain maintains its continuous parameter trace;
only its temperature label changes.

Functions:
- attempt_temperature_swaps: Perform replica exchange swap moves
"""

import jax
import jax.numpy as jnp
import jax.random as random


def attempt_temperature_swaps(
    key,
    states_A, states_B,
    temp_assignments_A, temp_assignments_B,
    temperature_ladder,
    log_post_fn,
    swap_accepts, swap_attempts,
    swap_parity,
    use_deo=True
):
    """
    Index process parallel tempering with optional DEO scheme (Syed et al. 2021).

    KEY DIFFERENCE from standard PT: We swap temperature ASSIGNMENTS, not chain states.
    Each chain maintains its continuous parameter trace; only its temperature label changes.

    DEO (Deterministic Even-Odd) Scheme (when use_deo=True):
    - Even round (parity=0): attempt swaps for pairs (0,1), (2,3), (4,5), ...
    - Odd round (parity=1): attempt swaps for pairs (1,2), (3,4), (5,6), ...
    - This creates deterministic "conveyor belt" round trips through temperature space
    - Reduces round-trip time from O(N^2) to O(N) compared to stochastic SEO

    When use_deo=False: All pairs attempt swaps every iteration (no parity gating).

    Swap acceptance probability between temperatures beta_i and beta_j (beta_i > beta_j):
        alpha = min(1, exp((beta_i - beta_j) * (log_pi(theta_j) - log_pi(theta_i))))

    where log_pi is the untempered log-posterior.

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
        use_deo: If True, use DEO scheme; if False, all pairs swap every iteration

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

    # Compute log likelihoods for swap acceptance
    # PT swap acceptance should use log_likelihood only (priors cancel out):
    #   log_alpha = (beta_cold - beta_hot) * (log_lik_hot - log_lik_cold)
    # We compute log_lik = log_post(beta=1) - log_post(beta=0) inside vmap for memory efficiency
    all_param_indices = jnp.arange(all_states.shape[1])

    def compute_log_lik(state):
        """Compute log likelihood = log_post(beta=1) - log_post(beta=0)."""
        log_post_1 = log_post_fn(state, all_param_indices, 1.0)
        log_post_0 = log_post_fn(state, all_param_indices, 0.0)
        return log_post_1 - log_post_0

    all_log_liks = jax.vmap(compute_log_lik)(all_states)

    # Maximum chains per temperature (for static array sizes)
    max_chains_per_temp = n_chains

    def swap_all_pairs(carry, temp_idx):
        """Attempt swaps for ALL chain pairs between temp_idx and temp_idx+1 (DEO-gated)."""
        curr_temp_assigns, curr_key, accepts, attempts = carry

        # DEO: only process if this pair matches current parity (when use_deo=True)
        # When use_deo=False, all pairs are active every iteration
        pair_parity = temp_idx % 2
        is_active = jax.lax.cond(
            use_deo,
            lambda: pair_parity == swap_parity,
            lambda: True
        )

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
            """Compute swap acceptance for one cold-hot chain pair."""
            cold_idx = cold_sorted[pair_idx]
            hot_idx = hot_shuffled[pair_idx]

            # Check if this is a valid pair:
            # 1. pair_idx < n_pairs: we have enough chains at both temperatures
            # 2. is_active: DEO parity allows this temperature pair
            # 3. Both indices are valid (not sentinels from padding)
            #    After shuffling, sentinels (n_chains) can end up anywhere in hot_shuffled.
            #    Without this check, swaps with sentinels cause one-way transfers
            #    (chain moves to new temp, but no chain moves back - sentinel update is no-op).
            is_valid = (pair_idx < n_pairs) & is_active & (cold_idx < n_chains) & (hot_idx < n_chains)

            # Get log likelihoods for swap acceptance (use safe indexing)
            safe_cold_idx = jnp.minimum(cold_idx, n_chains - 1)
            safe_hot_idx = jnp.minimum(hot_idx, n_chains - 1)
            log_lik_cold = all_log_liks[safe_cold_idx]
            log_lik_hot = all_log_liks[safe_hot_idx]

            # Acceptance ratio for swapping temperatures (using log_likelihood only)
            log_alpha = (beta_cold - beta_hot) * (log_lik_hot - log_lik_cold)
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
            """Apply a single swap by exchanging temperature assignments."""
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

    # Toggle parity for next round (DEO: E->O->E->O->...)
    next_parity = 1 - swap_parity

    return (new_temp_assignments_A, new_temp_assignments_B, final_key,
            final_accepts, final_attempts, next_parity)
