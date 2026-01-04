"""
Mean-Weighted Proposal: Adaptive Interpolation Between Self-Mean and Chain-Mean

This proposal smoothly interpolates between a local random walk (self_mean) and
a global jump toward the population mean (chain_mean), based on the Mahalanobis
distance from the current state to the population mean.

Proposal mean:
    μ_prop = α * x + (1 - α) * μ

where:
    - x is the current state
    - μ is the coupled mean (step_mean from other chain group)
    - α = d / (d + k) is the interpolation weight
    - d is the Mahalanobis distance from x to μ
    - k = 4 * sqrt(ndim) is a dimension-dependent constant

Behavior:
    - When far from μ (large d): α → 1, proposal ≈ random walk centered on x
    - When near μ (small d): α → 0, proposal ≈ chain_mean centered on μ
    - At equilibrium (d ≈ √ndim): α ≈ 0.2, proposals pulled 80% toward μ

This avoids failure modes of pure strategies:
    - Pure random walk: slow to reach high-density region when starting far away
    - Pure chain_mean: severe Hastings penalty when current state is far from μ

The adaptive approach stays safe when far (local moves) and efficient when near
(leverages population information).

Hastings Ratio:
    Because the proposal mean depends on the current state, we need the full
    Hastings correction. For forward proposal y given x, and reverse x given y:

    log q(x|y) - log q(y|x) = -0.5/ε² * [||x - μ_y||²_Σ - ||y - μ_x||²_Σ]

    where μ_x = α_x * x + (1-α_x) * μ and μ_y = α_y * y + (1-α_y) * μ
"""

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot

# Regularization for covariance matrix inversion
COV_NUGGET = 1e-6


def mean_weighted_proposal(operand):
    """
    Adaptive mean-weighted proposal.

    Interpolates between random walk (self_mean) and global jump (chain_mean)
    based on Mahalanobis distance from current state to coupled mean.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov,
                          coupled_blocks, block_mask, settings, grad_fn)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean from coupled chains (block_size,)
            step_cov: Precomputed covariance matrix (block_size, block_size)
            coupled_blocks: Raw states (unused)
            block_mask: Mask for valid parameters (1.0 = active, 0.0 = inactive)
            settings: JAX array of settings
                [COV_MULT] - covariance multiplier (default 1.0)
            grad_fn: Gradient function (unused by this proposal)

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn = operand
    new_key, proposal_key = random.split(key)

    cov_mult = settings[SettingSlot.COV_MULT]
    epsilon = jnp.sqrt(cov_mult)

    # Get effective dimension (number of active parameters)
    ndim = jnp.sum(block_mask)

    # Regularize covariance for numerical stability
    d = current_block.shape[0]
    cov_reg = step_cov + COV_NUGGET * jnp.eye(d)

    # Cholesky decomposition for sampling and Mahalanobis distance
    L = jnp.linalg.cholesky(cov_reg)

    # --- Compute alpha for current state ---
    # Mahalanobis distance from current state to coupled mean
    diff_current = (current_block - step_mean) * block_mask
    # Solve L * y = diff to get y, then ||y|| is Mahalanobis distance
    y_current = jax.scipy.linalg.solve_triangular(L, diff_current, lower=True)
    d_current = jnp.sqrt(jnp.sum(y_current**2))

    # Interpolation weight: α = d / (d + k) where k = 4 * sqrt(ndim)
    k = 4.0 * jnp.sqrt(ndim)
    alpha_current = d_current / (d_current + k + 1e-10)  # Avoid division by zero

    # --- Compute proposal mean for forward direction ---
    # μ_prop = α * x + (1 - α) * step_mean
    prop_mean_current = alpha_current * current_block + (1.0 - alpha_current) * step_mean

    # --- Sample proposal ---
    noise = random.normal(proposal_key, shape=current_block.shape)
    diffusion = epsilon * (L @ noise)
    proposal = (prop_mean_current + diffusion) * block_mask + current_block * (1 - block_mask)

    # --- Compute alpha for proposed state (for Hastings ratio) ---
    diff_proposal = (proposal - step_mean) * block_mask
    y_proposal = jax.scipy.linalg.solve_triangular(L, diff_proposal, lower=True)
    d_proposal = jnp.sqrt(jnp.sum(y_proposal**2))
    alpha_proposal = d_proposal / (d_proposal + k + 1e-10)

    # --- Compute proposal mean for reverse direction ---
    prop_mean_proposal = alpha_proposal * proposal + (1.0 - alpha_proposal) * step_mean

    # --- Hastings ratio ---
    # q(y|x) = N(y | μ_x, ε²Σ) where μ_x = α_x * x + (1-α_x) * μ
    # q(x|y) = N(x | μ_y, ε²Σ) where μ_y = α_y * y + (1-α_y) * μ
    #
    # log q(x|y) - log q(y|x) = -0.5/ε² * [||x - μ_y||²_Σ - ||y - μ_x||²_Σ]

    # Forward: y - μ_x (proposal minus forward proposal mean)
    diff_forward = (proposal - prop_mean_current) * block_mask
    y_forward = jax.scipy.linalg.solve_triangular(L, diff_forward, lower=True) / epsilon

    # Reverse: x - μ_y (current minus reverse proposal mean)
    diff_reverse = (current_block - prop_mean_proposal) * block_mask
    y_reverse = jax.scipy.linalg.solve_triangular(L, diff_reverse, lower=True) / epsilon

    # Squared Mahalanobis distances (already scaled by 1/ε)
    dist_sq_forward = jnp.sum(y_forward**2)
    dist_sq_reverse = jnp.sum(y_reverse**2)

    # log q(x|y) - log q(y|x) = -0.5 * (dist_sq_reverse - dist_sq_forward)
    log_hastings_ratio = -0.5 * (dist_sq_reverse - dist_sq_forward)

    return proposal, log_hastings_ratio, new_key
