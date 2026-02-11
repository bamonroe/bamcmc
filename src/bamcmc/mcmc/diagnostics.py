"""
MCMC Diagnostics.

Convergence diagnostics for MCMC chains:
- compute_nested_rhat: Unified Nested R-hat diagnostic (Margossian et al., 2022)
- print_rhat_summary: Print R-hat statistics with convergence check
- print_acceptance_summary: Print MH acceptance rate statistics
"""

import time
from functools import partial
from typing import Dict, Any, Optional, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..batch_specs import SamplerType


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

    Args:
        history: Sample history array (n_samples, n_chains, n_params)
        K: Number of superchains
        M: Number of subchains per superchain

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
    # For large n, this simplifies to: R-hat ~ sqrt(1 + B/(n*W))
    # Using the full formula for better accuracy:
    m = K  # number of chains/superchains
    n = n_samples
    V_hat = ((n - 1) / n) * W + B / n + B / (m * n)

    # No special handling for zero variance - stuck continuous params will show as NaN/inf
    # Discrete params should be filtered out before reporting (see discrete_param_indices)
    nrhat = jnp.sqrt(V_hat / W)

    return nrhat


def compute_and_print_rhat(
    history_device,
    user_config: Dict[str, Any],
) -> Optional[np.ndarray]:
    """
    Compute Nested R-hat and print summary statistics.

    INDEX PROCESS NOTE: With index process parallel tempering, ALL chains are saved
    and chains traverse all temperatures. R-hat is computed on full traces.
    For β=1-filtered analysis, use filter_beta1_samples() post-hoc with temp_history.

    Args:
        history_device: History array on device (JAX array)
        user_config: User configuration with K, M, and discrete_param_indices

    Returns:
        R-hat values array (numpy), or None if single chain
    """
    if history_device.shape[1] <= 1:
        return None

    print("\n--- Computing Unified Nested R-hat (GPU) ---")
    rhat_start = time.perf_counter()

    K = user_config['num_superchains']
    M = user_config['subchains_per_super']

    # Index process: all chains are saved, no K adjustment needed
    n_temperatures = user_config.get('n_temperatures', 1)
    if n_temperatures > 1:
        print(f"  (Index process: R-hat on full traces, {K} superchains × {M} subchains)")
        print(f"  Use filter_beta1_samples() with temp_history for β=1-only analysis")

    nrhat_device = compute_nested_rhat(history_device, K, M)
    nrhat_values = jax.device_get(nrhat_device)

    rhat_end = time.perf_counter()
    print(f"Diagnostics complete in {rhat_end - rhat_start:.4f}s")

    # Filter out discrete parameters for statistics
    discrete_indices = user_config.get('discrete_param_indices', [])
    n_params = len(nrhat_values)
    continuous_mask = np.ones(n_params, dtype=bool)
    continuous_mask[discrete_indices] = False
    continuous_rhat = nrhat_values[continuous_mask]

    # Display results for continuous params only
    # Small regularization constant for nested R-hat threshold
    # From Margossian et al. (2022), prevents division instability when M is large
    TAU_NESTED_RHAT = 1e-4
    threshold = np.sqrt(1 + 1/M + TAU_NESTED_RHAT)

    if M > 1:
        label = f"Nested R̂ ({K}x{M})"
    else:
        label = f"Standard R̂ ({K} chains)"

    n_continuous = len(continuous_rhat)
    n_discrete = len(discrete_indices)

    print(f"\n--- {label} Results ({n_continuous} continuous params, {n_discrete} discrete excluded) ---")
    print(f"  Max: {np.nanmax(continuous_rhat):.4f}")
    print(f"  Median: {np.nanmedian(continuous_rhat):.4f}")
    print(f"  Threshold: {threshold:.4f} (tau={TAU_NESTED_RHAT:.0e})")

    # Check for NaN/Inf (stuck continuous params)
    n_nan = np.sum(~np.isfinite(continuous_rhat))
    if n_nan > 0:
        print(f"  WARNING: {n_nan} continuous params have NaN/Inf R-hat (stuck chains)")

    if np.nanmax(continuous_rhat) < threshold:
        print(f"  Converged (max < {threshold:.4f})")
    else:
        print(f"  Not Converged (max = {np.nanmax(continuous_rhat):.4f} >= {threshold:.4f})")

    return nrhat_values


def print_acceptance_summary(block_specs: List, acceptance_rates_host: np.ndarray) -> None:
    """
    Print summary statistics for MH acceptance rates.

    Args:
        block_specs: List of BlockSpec objects
        acceptance_rates_host: Acceptance rates array (numpy, on host)
    """
    mh_rates = []
    mh_labels = []
    for i, (spec, rate) in enumerate(zip(block_specs, acceptance_rates_host)):
        if spec.sampler_type == SamplerType.METROPOLIS_HASTINGS:
            mh_rates.append(rate)
            mh_labels.append(spec.label if spec.label else f"Block {i}")

    if not mh_rates:
        return

    mh_rates = np.array(mh_rates)
    print(f"\n--- MH Acceptance Rates ({len(mh_rates)} blocks) ---")
    print(f"  Mean: {np.mean(mh_rates):.1%}  Median: {np.median(mh_rates):.1%}  "
          f"Min: {np.min(mh_rates):.1%}  Max: {np.max(mh_rates):.1%}")

    # Warn about low acceptance rates
    low_rate_mask = mh_rates < 0.10
    if np.any(low_rate_mask):
        low_count = np.sum(low_rate_mask)
        low_labels = [lbl for lbl, is_low in zip(mh_labels, low_rate_mask) if is_low]
        print(f"  WARNING: {low_count} block(s) have acceptance rate < 10%")
        if low_count <= 10:
            print(f"    Low blocks: {', '.join(low_labels)}")


def print_swap_acceptance_summary(
    temperature_ladder: np.ndarray,
    swap_accepts: np.ndarray,
    swap_attempts: np.ndarray
) -> None:
    """
    Print summary statistics for parallel tempering swap acceptance rates.

    Args:
        temperature_ladder: Temperature values (n_temperatures,)
        swap_accepts: Number of accepted swaps per temperature pair
        swap_attempts: Number of attempted swaps per temperature pair
    """
    n_temperatures = len(temperature_ladder)
    if n_temperatures <= 1:
        return

    print(f"\n--- Parallel Tempering Swap Rates ({n_temperatures} temperatures) ---")
    print(f"  Temperature ladder: {', '.join(f'{t:.3f}' for t in temperature_ladder)}")

    n_pairs = n_temperatures - 1
    swap_rates = np.zeros(n_pairs)
    for i in range(n_pairs):
        if swap_attempts[i] > 0:
            swap_rates[i] = swap_accepts[i] / swap_attempts[i]
        else:
            swap_rates[i] = 0.0

        beta_i = temperature_ladder[i]
        beta_j = temperature_ladder[i + 1]
        print(f"  Pair ({beta_i:.3f} <-> {beta_j:.3f}): "
              f"{swap_rates[i]:.1%} ({swap_accepts[i]}/{swap_attempts[i]})")

    if n_pairs > 0:
        print(f"  Mean swap rate: {np.mean(swap_rates):.1%}")

    # Warn about low swap rates (should be 20-40% for good mixing)
    low_swap_mask = (swap_rates < 0.10) & (swap_attempts > 0)
    if np.any(low_swap_mask):
        print(f"  WARNING: Some swap rates are < 10% - consider adjusting temperature spacing")


# =============================================================================
# INDEX PROCESS UTILITIES
# =============================================================================

def filter_beta1_samples(
    history: np.ndarray,
    temp_history: np.ndarray,
    target_temp_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter history to only samples where chain was at target temperature.

    INDEX PROCESS: With the index process, chains traverse all temperatures.
    This function extracts only the samples where each chain was at the target
    temperature (typically β=1 for posterior inference).

    Args:
        history: Sample history array (n_samples, n_chains, n_params)
        temp_history: Temperature index per chain per sample (n_samples, n_chains)
        target_temp_idx: Target temperature index (0 = β=1 coldest, default)

    Returns:
        filtered_history: Samples where chain was at target temp (variable shape)
        sample_counts: Number of β=1 samples per chain (n_chains,)

    Usage:
        results = rmcmc_single(config, data)
        history = results['history']
        temp_history = results['temperature_history']
        if temp_history is not None:
            beta1_samples, counts = filter_beta1_samples(history, temp_history)
            # beta1_samples is a list of arrays, one per chain
    """
    if temp_history is None:
        return history, np.full(history.shape[1], history.shape[0])

    n_samples, n_chains, n_params = history.shape
    mask = (temp_history == target_temp_idx)  # (n_samples, n_chains)

    # Count samples per chain at target temperature
    sample_counts = np.sum(mask, axis=0)  # (n_chains,)

    # For uniform output, find minimum count and truncate each chain to that
    min_count = np.min(sample_counts)

    if min_count == 0:
        print(f"WARNING: Some chains have 0 samples at temperature index {target_temp_idx}")
        print(f"  Sample counts range: {np.min(sample_counts)} to {np.max(sample_counts)}")
        return None, sample_counts

    # Build filtered array with uniform sample count per chain
    filtered = np.zeros((min_count, n_chains, n_params), dtype=history.dtype)
    for c in range(n_chains):
        chain_mask = mask[:, c]
        chain_samples = history[chain_mask, c, :]  # (count, n_params)
        filtered[:, c, :] = chain_samples[:min_count]

    return filtered, sample_counts


def compute_round_trip_rate(
    temp_history: np.ndarray,
    n_temperatures: int,
) -> Tuple[float, np.ndarray]:
    """
    Compute round-trip rate from temperature history.

    A round trip is defined as: β=0 (temp_idx=N-1) → β=1 (temp_idx=0) → β=0.
    This measures how effectively chains are exchanging information between
    the hot (reference) and cold (target) distributions.

    Args:
        temp_history: Temperature index per chain per sample (n_samples, n_chains)
        n_temperatures: Total number of temperature levels

    Returns:
        mean_rate: Mean round-trip rate across chains (round trips per sample)
        per_chain_trips: Number of complete round trips per chain
    """
    if temp_history is None or n_temperatures <= 1:
        return 0.0, np.array([])

    n_samples, n_chains = temp_history.shape

    # Count round trips for each chain
    per_chain_trips = np.zeros(n_chains, dtype=np.int32)

    for c in range(n_chains):
        trace = temp_history[:, c]

        # Track direction: are we going toward hot (increasing temp idx) or cold (decreasing)?
        # A round trip completes when we touch both extremes
        touched_cold = False
        touched_hot = False
        trips = 0

        for temp_idx in trace:
            if temp_idx == 0:  # Coldest (β=1)
                if touched_hot:
                    trips += 1  # Completed a trip from hot to cold
                    touched_hot = False
                touched_cold = True
            elif temp_idx == n_temperatures - 1:  # Hottest (β=β_min)
                if touched_cold:
                    # Went from cold to hot, half a round trip
                    pass
                touched_hot = True

        per_chain_trips[c] = trips

    mean_rate = np.mean(per_chain_trips) / n_samples if n_samples > 0 else 0.0

    return mean_rate, per_chain_trips


def print_round_trip_summary(
    temp_history: np.ndarray,
    n_temperatures: int,
) -> None:
    """
    Print summary statistics for index process round-trip behavior.

    Args:
        temp_history: Temperature index per chain per sample (n_samples, n_chains)
        n_temperatures: Total number of temperature levels
    """
    if temp_history is None or n_temperatures <= 1:
        return

    mean_rate, per_chain_trips = compute_round_trip_rate(temp_history, n_temperatures)
    n_samples, n_chains = temp_history.shape

    print(f"\n--- Index Process Round-Trip Summary ({n_temperatures} temperatures) ---")
    print(f"  Chains: {n_chains}")
    print(f"  Samples: {n_samples}")
    print(f"  Total round trips: {np.sum(per_chain_trips)}")
    print(f"  Mean trips per chain: {np.mean(per_chain_trips):.1f}")
    print(f"  Round-trip rate: {mean_rate:.4f} trips/sample")

    if np.mean(per_chain_trips) < 1:
        print(f"  WARNING: Low round-trip count - chains may not be mixing through temperatures")
