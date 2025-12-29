"""
MCMC Diagnostics.

Convergence diagnostics for MCMC chains:
- compute_nested_rhat: Unified Nested R-hat diagnostic (Margossian et al., 2022)
"""

import jax
import jax.numpy as jnp
from functools import partial


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
    nrhat = jnp.sqrt(V_hat / W)

    return nrhat
