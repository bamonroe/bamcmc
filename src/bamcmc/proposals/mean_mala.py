"""
Mean-MALA (Chain-Mean Metropolis-Adjusted Langevin Algorithm) Proposal

A hybrid proposal combining Chain-Mean and MALA:
- Like Chain-Mean: proposes independently of current state, centered on population mean
- Like MALA: uses gradient to shift the center toward higher density

Proposal center: c = μ + (cov_mult/2) * Σ * ∇log p(μ)
Proposal distribution: q(x'|x) = N(x' | c, cov_mult * Σ)

Key properties:
- Only ONE gradient evaluation needed (at μ), shared across all chains
- Proposal doesn't depend on current state x (independent proposal)
- Gradient at μ identifies direction the population should move
- Helps stuck chains escape by proposing from population center

Hastings ratio:
Since both forward q(x'|x) and reverse q(x|x') are centered on c:
log α = log p(x') - log p(x) + (1/2ε²)[||x'-c||²_Σ - ||x-c||²_Σ]

where ||y||²_Σ = y^T Σ^{-1} y is the squared Mahalanobis distance.
"""

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot

# Regularization for covariance matrix inversion
COV_NUGGET = 1e-6


def mean_mala_proposal(operand):
    """
    Chain-Mean MALA proposal.

    Proposes from a distribution centered on μ + drift(μ), where μ is
    the coupled mean and drift uses the gradient evaluated at μ.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov,
                          coupled_blocks, block_mask, settings, grad_fn)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean from coupled chains (block_size,)
                      This is μ, the center for the proposal
            step_cov: Precomputed covariance matrix (block_size, block_size)
                     Used as the preconditioning/mass matrix
            coupled_blocks: Raw states (unused)
            block_mask: Mask for valid parameters (1.0 = active, 0.0 = inactive)
            settings: JAX array of settings
                [COV_MULT] - covariance multiplier (default 1.0)
                            Proposal variance = cov_mult * Σ
            grad_fn: Function that takes block values and returns gradient
                     of log posterior w.r.t. those values

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn = operand
    new_key, proposal_key = random.split(key)

    # Use cov_mult for consistent interface with other proposals
    cov_mult = settings[SettingSlot.COV_MULT]
    epsilon = jnp.sqrt(cov_mult)

    d = current_block.shape[0]

    # Regularize covariance for numerical stability
    cov_reg = step_cov + COV_NUGGET * jnp.eye(d)

    # Cholesky decomposition for sampling and density computation
    L = jnp.linalg.cholesky(cov_reg)

    # Gradient at coupled mean (NOT at current state)
    # This is the key difference from standard MALA
    mean_grad = grad_fn(step_mean)

    # Preconditioned drift: (cov_mult/2) * Σ * ∇log p(μ)
    drift = 0.5 * cov_mult * (cov_reg @ mean_grad)

    # Proposal center: c = μ + drift
    center = step_mean + drift

    # Noise term: √cov_mult * L * z = ε * L * z
    noise = random.normal(proposal_key, shape=current_block.shape)
    diffusion = epsilon * (L @ noise)

    # Proposal: x' = c + noise (masked)
    # Note: proposal is centered on c, NOT on current_block
    proposal = (center + diffusion) * block_mask + current_block * (1 - block_mask)

    # --- Hastings ratio computation ---
    # Both q(x'|x) and q(x|x') are N(· | c, cov_mult * Σ)
    # They're centered on the SAME point c, so:
    # log q(x|x') - log q(x'|x) = (1/2ε²)[||x'-c||²_Σ - ||x-c||²_Σ]

    # Compute Mahalanobis distances using solve_triangular for stability
    # ||y||²_Σ = y^T Σ^{-1} y = ||L^{-1} y||²

    diff_current = (current_block - center) * block_mask
    diff_proposal = (proposal - center) * block_mask

    # L^{-1} * diff, then scale by 1/ε for the full (1/cov_mult) factor
    y_current = jax.scipy.linalg.solve_triangular(L, diff_current, lower=True) / epsilon
    y_proposal = jax.scipy.linalg.solve_triangular(L, diff_proposal, lower=True) / epsilon

    # Squared Mahalanobis distances (scaled by 1/cov_mult)
    dist_sq_current = jnp.sum(y_current**2)
    dist_sq_proposal = jnp.sum(y_proposal**2)

    # Hastings correction: penalize moves away from center
    # log q(x|x') - log q(x'|x) = 0.5 * (dist_sq_proposal - dist_sq_current)
    log_hastings_ratio = 0.5 * (dist_sq_proposal - dist_sq_current)

    return proposal, log_hastings_ratio, new_key
