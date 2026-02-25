"""
Mode-Weighted Proposal: Adaptive Interpolation Toward Mode (Highest Density Chain)

This proposal smoothly interpolates between a local random walk (self_mean) and
a global jump toward the mode (chain with highest log posterior in coupled group),
based on the Mahalanobis distance from the current state to the mode.

Proposal mean:
    μ_prop = α * x + (1 - α) * m

where:
    - x is the current state
    - m is the mode (state of coupled chain with highest log posterior)
    - α = d / (d + k) is the interpolation weight
    - d is the Mahalanobis distance from x to m
    - k = 4 * sqrt(ndim) is a dimension-dependent constant

Behavior:
    - When far from mode (large d): α → 1, proposal ≈ random walk centered on x
    - When near mode (small d): α → 0, proposal ≈ centered on mode m
    - At equilibrium (d ≈ √ndim): α ≈ 0.2, proposals pulled 80% toward mode

Advantages over MEAN_WEIGHTED:
    - Mode is an actual high-density point (mean might fall in low-density region)
    - Better for multimodal or skewed posteriors where mean ≠ mode
    - Directly targets the highest probability chain state

Hastings Ratio:
    Because the proposal mean depends on the current state, we need the full
    Hastings correction. For forward proposal y given x, and reverse x given y:

    log q(x|y) - log q(y|x) = -0.5/ε² * [||x - μ_y||²_Σ - ||y - μ_x||²_Σ]

    where μ_x = α_x * x + (1-α_x) * m and μ_y = α_y * y + (1-α_y) * m
"""

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot
from .common import (unpack_operand, regularize_covariance, sample_diffusion,
                     compute_alpha_linear, hastings_ratio_fixed_cov)


def mode_weighted_proposal(operand):
    """
    Adaptive mode-weighted proposal.

    Interpolates between random walk (self_mean) and global jump toward mode
    based on Mahalanobis distance from current state to mode.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov,
                          coupled_blocks, block_mask, settings, grad_fn, block_mode)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean from coupled chains (unused - we use mode instead)
            step_cov: Precomputed covariance matrix (block_size, block_size)
            coupled_blocks: Raw states (unused)
            block_mask: Mask for valid parameters (1.0 = active, 0.0 = inactive)
            settings: JAX array of settings
                [COV_MULT] - covariance multiplier (default 1.0)
            grad_fn: Gradient function (unused by this proposal)
            block_mode: Mode chain values (block_size,) - from chain with highest log posterior

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    op = unpack_operand(operand)
    new_key, proposal_key = random.split(op.key)

    cov_mult = op.settings[SettingSlot.COV_MULT]
    epsilon = jnp.sqrt(cov_mult)

    # Get effective dimension (number of active parameters)
    ndim = jnp.sum(op.block_mask)

    # Regularize covariance (unscaled — epsilon applied separately)
    cov_reg = regularize_covariance(op.step_cov)
    L = jnp.linalg.cholesky(cov_reg)

    # Interpolation constant
    k = 4.0 * jnp.sqrt(ndim)

    # --- Compute alpha for current state ---
    # Mahalanobis distance from current state to MODE (not mean)
    diff_current = (op.current_block - op.block_mode) * op.block_mask
    y_current = jax.scipy.linalg.solve_triangular(L, diff_current, lower=True)
    d_current = jnp.sqrt(jnp.sum(y_current**2))
    alpha_current = compute_alpha_linear(d_current, k)

    # --- Compute proposal mean for forward direction ---
    prop_mean_current = alpha_current * op.current_block + (1.0 - alpha_current) * op.block_mode

    # --- Sample proposal ---
    diffusion = sample_diffusion(proposal_key, L, op.current_block.shape, scale=epsilon)
    proposal = (prop_mean_current + diffusion) * op.block_mask + op.current_block * (1 - op.block_mask)

    # --- Compute alpha for proposed state (for Hastings ratio) ---
    diff_proposal = (proposal - op.block_mode) * op.block_mask
    y_proposal = jax.scipy.linalg.solve_triangular(L, diff_proposal, lower=True)
    d_proposal = jnp.sqrt(jnp.sum(y_proposal**2))
    alpha_proposal = compute_alpha_linear(d_proposal, k)

    # --- Compute proposal mean for reverse direction ---
    prop_mean_proposal = alpha_proposal * proposal + (1.0 - alpha_proposal) * op.block_mode

    # --- Hastings ratio ---
    log_hastings_ratio = hastings_ratio_fixed_cov(
        L, epsilon, proposal, op.current_block,
        prop_mean_current, prop_mean_proposal, op.block_mask)

    return proposal, log_hastings_ratio, new_key
