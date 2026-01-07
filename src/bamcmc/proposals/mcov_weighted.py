"""
Mean-Covariance Weighted Proposal (MCOV_WEIGHTED)

This proposal extends MEAN_WEIGHTED by also scaling the proposal covariance
based on distance to the coupled mean. The key insight is that the covariance
scaling creates a new metric, and we use distance in that new metric for the
mean interpolation - creating a coherent, geometrically-motivated proposal.

Algorithm:
    1. Scale base covariance: Σ_base = ε² * Σ  (where ε² = cov_mult)
    2. Compute d1 = Mahalanobis distance from current state to coupled mean using Σ_base
    3. Scale covariance: Σ_weighted = g(d1) * Σ_base  where g(d) = 1 + β*d/(d+k)
    4. Compute d2 = d1 / sqrt(g(d1)) - distance in the weighted metric
    5. Compute α = d2 / (d2 + k) for mean interpolation
    6. Proposal mean: μ_prop = α * x + (1 - α) * m
    7. Sample: y ~ N(μ_prop, Σ_weighted)

Behavior:
    - When far from mean (large d1):
        * g(d1) large → covariance expands (bigger steps)
        * d2 = d1/sqrt(g) smaller → α smaller → mean pulled MORE toward coupled mean
        * Combined effect: take big, directed jumps toward high-density region
    - When near mean (small d1):
        * g(d1) ≈ 1 → covariance unchanged
        * d2 ≈ d1 → standard mean interpolation
        * Combined effect: normal local exploration

Parameters:
    - COV_MULT (ε²): Base step size multiplier
    - COV_BETA (β): Covariance scaling strength. β=0 reduces to MEAN_WEIGHTED.
      Higher β → more aggressive expansion when far from mean.

Hastings Ratio:
    Because both the proposal mean and covariance depend on the current state,
    we need the full correction including the log-determinant term:

    log q(x|y) - log q(y|x) = -0.5 * dim * [log g(d1_y) - log g(d1_x)]
                             - 0.5 * [||x - μ_y||²_{Σ_base}/g_y - ||y - μ_x||²_{Σ_base}/g_x]

    where g_x = g(d1_x) and g_y = g(d1_y). The cov_mult factor cancels between
    forward and reverse directions since it's state-independent.
"""

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot

# Regularization for covariance matrix inversion
COV_NUGGET = 1e-6


def mcov_weighted_proposal(operand):
    """
    Mean-Covariance weighted proposal with sequential distance computation.

    Scales both the proposal covariance and mean interpolation based on distance
    to coupled mean, using the weighted covariance to define the metric for
    mean interpolation.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov,
                          coupled_blocks, block_mask, settings, grad_fn, block_mode)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean from coupled chains (block_size,)
            step_cov: Precomputed covariance matrix (block_size, block_size)
            coupled_blocks: Raw states (unused)
            block_mask: Mask for valid parameters (1.0 = active, 0.0 = inactive)
            settings: JAX array of settings
                [COV_MULT] - base covariance multiplier (ε²)
                [COV_BETA] - covariance scaling strength (β)
            grad_fn: Gradient function (unused by this proposal)
            block_mode: Mode chain values (unused by this proposal)

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode = operand
    del grad_fn, block_mode, coupled_blocks  # Unused

    new_key, proposal_key = random.split(key)

    cov_mult = settings[SettingSlot.COV_MULT]
    cov_beta = settings[SettingSlot.COV_BETA]

    # Get effective dimension (number of active parameters)
    ndim = jnp.sum(block_mask)

    # Scale covariance by cov_mult FIRST, then regularize
    # This makes d1 measure distance in "proposal scale" units
    d = current_block.shape[0]
    cov_scaled = cov_mult * step_cov + COV_NUGGET * jnp.eye(d)

    # Cholesky decomposition of scaled covariance for sampling and Mahalanobis distance
    L = jnp.linalg.cholesky(cov_scaled)

    # Dimension-dependent constant for interpolation
    k = 4.0 * jnp.sqrt(ndim)

    # === STEP 1: Compute d1 (distance with scaled covariance = cov_mult * Σ) ===
    diff_current = (current_block - step_mean) * block_mask
    y_current = jax.scipy.linalg.solve_triangular(L, diff_current, lower=True)
    d1_current = jnp.sqrt(jnp.sum(y_current**2))

    # === STEP 2: Compute covariance scale factor g(d1) ===
    # g(d) = 1 + beta * d / (d + k)
    # When beta=0: g=1, reduces to MEAN_WEIGHTED
    # When beta>0: g>1 when d>0, covariance expands with distance
    g_current = 1.0 + cov_beta * d1_current / (d1_current + k + 1e-10)

    # === STEP 3: Compute d2 (distance in weighted metric) ===
    # d2 = d1 / sqrt(g) because Σ_weighted = g * Σ
    d2_current = d1_current / jnp.sqrt(g_current + 1e-10)

    # === STEP 4: Compute alpha using d2 ===
    alpha_current = d2_current / (d2_current + k + 1e-10)

    # === STEP 5: Compute proposal mean ===
    prop_mean_current = alpha_current * current_block + (1.0 - alpha_current) * step_mean

    # === STEP 6: Sample from N(μ, g * Σ_scaled) where Σ_scaled = cov_mult * Σ ===
    # Since L is from Σ_scaled, final covariance is g * L @ L.T
    # So we sample: μ + sqrt(g) * L @ z where z ~ N(0, I)
    noise = random.normal(proposal_key, shape=current_block.shape)
    scale_current = jnp.sqrt(g_current)
    diffusion = scale_current * (L @ noise)
    proposal = (prop_mean_current + diffusion) * block_mask + current_block * (1 - block_mask)

    # === Compute quantities for reverse direction (proposal -> current) ===
    diff_proposal = (proposal - step_mean) * block_mask
    y_proposal = jax.scipy.linalg.solve_triangular(L, diff_proposal, lower=True)
    d1_proposal = jnp.sqrt(jnp.sum(y_proposal**2))

    g_proposal = 1.0 + cov_beta * d1_proposal / (d1_proposal + k + 1e-10)
    d2_proposal = d1_proposal / jnp.sqrt(g_proposal + 1e-10)
    alpha_proposal = d2_proposal / (d2_proposal + k + 1e-10)

    prop_mean_proposal = alpha_proposal * proposal + (1.0 - alpha_proposal) * step_mean

    # === Hastings ratio ===
    # q(y|x) = N(y | μ_x, g_x * Σ_scaled)  where Σ_scaled = cov_mult * Σ
    # q(x|y) = N(x | μ_y, g_y * Σ_scaled)
    #
    # log q(x|y) - log q(y|x) = -0.5 * dim * [log(g_y) - log(g_x)]
    #                         - 0.5 * [||x - μ_y||²_{Σ_scaled} / g_y - ||y - μ_x||²_{Σ_scaled} / g_x]

    # Log-determinant term from different covariance scales
    log_det_term = -0.5 * ndim * (jnp.log(g_proposal + 1e-10) - jnp.log(g_current + 1e-10))

    # Forward: y - μ_x (proposal minus forward proposal mean)
    diff_forward = (proposal - prop_mean_current) * block_mask
    y_forward = jax.scipy.linalg.solve_triangular(L, diff_forward, lower=True)
    dist_sq_forward = jnp.sum(y_forward**2) / g_current  # Divide by g_x

    # Reverse: x - μ_y (current minus reverse proposal mean)
    diff_reverse = (current_block - prop_mean_proposal) * block_mask
    y_reverse = jax.scipy.linalg.solve_triangular(L, diff_reverse, lower=True)
    dist_sq_reverse = jnp.sum(y_reverse**2) / g_proposal  # Divide by g_y

    # Combined Hastings ratio (cov_mult is baked into L, cancels between directions)
    log_hastings_ratio = log_det_term - 0.5 * (dist_sq_reverse - dist_sq_forward)

    return proposal, log_hastings_ratio, new_key
