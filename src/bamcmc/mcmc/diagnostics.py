"""
MCMC Diagnostics.

Convergence diagnostics for MCMC chains:
- compute_nested_rhat: Unified Nested R-hat diagnostic (Margossian et al., 2022)
- print_rhat_summary: Print R-hat statistics with convergence check
- print_acceptance_summary: Print MH acceptance rate statistics
"""

import time
from functools import partial
from typing import Dict, Any, Optional, List

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

    # Adjust K for parallel tempering: only beta=1 chains are in history
    n_temperatures = user_config.get('n_temperatures', 1)
    if n_temperatures > 1:
        K_effective = K // n_temperatures
        print(f"  (Tempering active: using K={K_effective} of {K} superchains for beta=1 chains)")
        K = K_effective

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
    tau = 1e-4
    threshold = np.sqrt(1 + 1/M + tau)

    if M > 1:
        label = f"Nested R̂ ({K}x{M})"
    else:
        label = f"Standard R̂ ({K} chains)"

    n_continuous = len(continuous_rhat)
    n_discrete = len(discrete_indices)

    print(f"\n--- {label} Results ({n_continuous} continuous params, {n_discrete} discrete excluded) ---")
    print(f"  Max: {np.nanmax(continuous_rhat):.4f}")
    print(f"  Median: {np.nanmedian(continuous_rhat):.4f}")
    print(f"  Threshold: {threshold:.4f} (tau={tau:.0e})")

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
