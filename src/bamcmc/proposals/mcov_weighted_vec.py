"""
Vectorized Mean-Covariance Weighted Proposal (MCOV_WEIGHTED_VEC)

Like MCOV_WEIGHTED, but with per-parameter distance computation instead of
a single scalar Mahalanobis distance. This allows each parameter to have
its own interpolation strength toward the coupled mean.

Key difference from MCOV_WEIGHTED:
    - d1 is a vector (one distance per parameter) instead of scalar
    - g(d) is computed per-parameter
    - alpha is per-parameter, so interpolation is element-wise
    - Parameters far from their marginal mean stay closer to current value
    - Parameters near their marginal mean get pulled toward coupled mean

Design goals:
    1. At equilibrium (near coupled mean): behave like CHAIN_MEAN proposal
       for fast mixing - alpha ≈ 0, g ≈ 1
    2. Far from equilibrium: cautiously track current state to catch up -
       alpha → 1, g shrinks (with negative beta)
    3. Higher dimensions should track current state more (jumping is riskier)

Algorithm:
    1. Scale base covariance: Sigma_base = cov_mult * Sigma
    2. Compute whitened difference: y = L^{-1} @ (x - m)
    3. Per-parameter distance: d1_vec = |y|  (element-wise absolute value)
    4. Dimension-scaled constant: k = c / sqrt(ndim)  [higher dim → smaller k]
    5. Per-parameter cov scale (quadratic): g_vec = 1 + beta * d1² / (d1² + k²)
    6. Per-parameter metric distance: d2_vec = d1_vec / sqrt(g_vec)
    7. Per-parameter interpolation (quadratic): alpha_vec = d2² / (d2² + k²)
    8. Element-wise proposal mean: mu_prop = alpha_vec * x + (1 - alpha_vec) * m
    9. Sample with per-param scaling: y ~ N(mu_prop, G @ Sigma_base @ G)
       where G = diag(sqrt(g_vec))

The quadratic formulation ensures alpha and g are flat near d=0, recovering
CHAIN_MEAN behavior at equilibrium. The inverted k scaling makes higher
dimensional blocks more conservative (larger alpha for same distance).

Parameters:
    - COV_MULT: Base step size multiplier (scalar)
    - COV_BETA: Covariance scaling strength (scalar). Negative values shrink
      variance when far from mean (aids catching up). beta=0 reduces to
      quadratic MEAN_WEIGHTED.
"""

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot

# Regularization for covariance matrix inversion
COV_NUGGET = 1e-6


def mcov_weighted_vec_proposal(operand):
    """
    Vectorized mean-covariance weighted proposal.

    Each parameter gets its own distance, covariance scaling, and interpolation
    strength based on how far it is from its marginal mean (in whitened space).

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov,
                          coupled_blocks, block_mask, settings, grad_fn, block_mode)

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

    # Get effective dimension
    ndim = jnp.sum(block_mask)
    d = current_block.shape[0]

    # Scale covariance by cov_mult, then regularize
    cov_scaled = cov_mult * step_cov + COV_NUGGET * jnp.eye(d)

    # Cholesky decomposition
    L = jnp.linalg.cholesky(cov_scaled)

    # Dimension-dependent constant for interpolation (inverted scaling)
    # Higher dimensions -> smaller k -> larger alpha (more conservative)
    k = 2.0 / jnp.sqrt(jnp.maximum(ndim, 1.0))
    k_sq = k * k

    # === STEP 1: Compute per-parameter whitened distance d1_vec ===
    diff_current = (current_block - step_mean) * block_mask
    y_current = jax.scipy.linalg.solve_triangular(L, diff_current, lower=True)
    d1_vec_current = jnp.abs(y_current)  # Per-parameter distance
    d1_sq_current = d1_vec_current * d1_vec_current

    # === STEP 2: Per-parameter covariance scale g_vec (quadratic) ===
    # g_i = 1 + beta * d1_i^2 / (d1_i^2 + k^2)
    # Quadratic ensures g is flat near d=0 (chain_mean-like at equilibrium)
    g_vec_current = 1.0 + cov_beta * d1_sq_current / (d1_sq_current + k_sq + 1e-10)

    # === STEP 3: Per-parameter distance in weighted metric ===
    # d2_i = d1_i / sqrt(g_i)
    sqrt_g_current = jnp.sqrt(jnp.maximum(g_vec_current, 1e-10))
    d2_vec_current = d1_vec_current / sqrt_g_current
    d2_sq_current = d2_vec_current * d2_vec_current

    # === STEP 4: Per-parameter alpha (quadratic) ===
    # alpha_i = d2_i^2 / (d2_i^2 + k^2)
    # Quadratic ensures alpha is flat near d=0 (chain_mean-like at equilibrium)
    alpha_vec_current = d2_sq_current / (d2_sq_current + k_sq + 1e-10)

    # === STEP 5: Element-wise proposal mean ===
    prop_mean_current = alpha_vec_current * current_block + (1.0 - alpha_vec_current) * step_mean

    # === STEP 6: Sample from N(mu, G @ Sigma_base @ G) ===
    # where G = diag(sqrt(g_vec))
    # Sample: mu + G @ L @ z = mu + sqrt_g_vec * (L @ z)
    noise = random.normal(proposal_key, shape=current_block.shape)
    L_noise = L @ noise
    diffusion = sqrt_g_current * L_noise  # Element-wise scaling
    proposal = (prop_mean_current + diffusion) * block_mask + current_block * (1 - block_mask)

    # === Compute quantities for reverse direction ===
    diff_proposal = (proposal - step_mean) * block_mask
    y_proposal = jax.scipy.linalg.solve_triangular(L, diff_proposal, lower=True)
    d1_vec_proposal = jnp.abs(y_proposal)
    d1_sq_proposal = d1_vec_proposal * d1_vec_proposal

    # Quadratic g for reverse direction
    g_vec_proposal = 1.0 + cov_beta * d1_sq_proposal / (d1_sq_proposal + k_sq + 1e-10)
    sqrt_g_proposal = jnp.sqrt(jnp.maximum(g_vec_proposal, 1e-10))
    d2_vec_proposal = d1_vec_proposal / sqrt_g_proposal
    d2_sq_proposal = d2_vec_proposal * d2_vec_proposal

    # Quadratic alpha for reverse direction
    alpha_vec_proposal = d2_sq_proposal / (d2_sq_proposal + k_sq + 1e-10)

    prop_mean_proposal = alpha_vec_proposal * proposal + (1.0 - alpha_vec_proposal) * step_mean

    # === Hastings ratio ===
    # q(y|x) = N(y | mu_x, G_x @ Sigma_base @ G_x)
    # q(x|y) = N(x | mu_y, G_y @ Sigma_base @ G_y)
    #
    # Log-determinant term (G is diagonal, so det = prod of diagonal):
    # log det(G @ Sigma @ G) = log det(Sigma) + 2*sum(log(sqrt_g)) = log det(Sigma) + sum(log(g))
    # The log det(Sigma) cancels between forward and reverse.
    log_det_term = -0.5 * (jnp.sum(jnp.log(g_vec_proposal + 1e-10)) -
                           jnp.sum(jnp.log(g_vec_current + 1e-10)))

    # Quadratic form for N(y | mu, G @ Sigma @ G):
    # ||y - mu||^2_{(G @ Sigma @ G)^{-1}} = ||G^{-1} @ (y - mu)||^2_{Sigma^{-1}}
    #                                     = ||L^{-1} @ (G^{-1} @ (y - mu))||^2
    #                                     = ||L^{-1} @ ((y - mu) / sqrt_g)||^2

    # Forward: proposal given current
    diff_forward = ((proposal - prop_mean_current) / sqrt_g_current) * block_mask
    y_forward = jax.scipy.linalg.solve_triangular(L, diff_forward, lower=True)
    dist_sq_forward = jnp.sum(y_forward**2)

    # Reverse: current given proposal
    diff_reverse = ((current_block - prop_mean_proposal) / sqrt_g_proposal) * block_mask
    y_reverse = jax.scipy.linalg.solve_triangular(L, diff_reverse, lower=True)
    dist_sq_reverse = jnp.sum(y_reverse**2)

    log_hastings_ratio = log_det_term - 0.5 * (dist_sq_reverse - dist_sq_forward)

    return proposal, log_hastings_ratio, new_key
