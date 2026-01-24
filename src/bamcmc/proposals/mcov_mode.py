"""
Mode-Targeting Mean-Covariance Proposal (MCOV_MODE) - Scalar Distance

Like MCOV_MODE_VEC, but uses a single scalar Mahalanobis distance instead of
per-parameter distances. This gives uniform behavior across all parameters.

The key insight: when the posterior has multiple modes, the coupled mean sits
between them, creating asymmetric dynamics that trap chains in lower-density
modes. By targeting the highest LP chain (the mode), all chains are pulled
toward the best region of parameter space.

Uses scalar distance for unified control:
- Single Mahalanobis distance d from mode determines both α and g
- All parameters share the same α (uniform interpolation)
- Simpler dynamics, may be more stable for correlated parameters

At equilibrium (d ≈ 0):
    - α ≈ 0: proposal centered on mode
    - g ≈ 1: full step size variance
    - Behavior: fast mixing around mode

Far from mode (d large):
    - α → 1: track current state
    - g → 0: small, cautious steps
    - Behavior: gentle guidance back toward mode

Algorithm:
    1. Compute Mahalanobis distance from MODE: d = sqrt((x - mode)' Σ^{-1} (x - mode))
    2. Compute smooth parameters:
       g = 1 / (1 + (d / k_g)²)
       d2 = d / sqrt(g)
       α = d2² / (d2² + k_α²)
    3. Proposal mean: μ_prop = α × x + (1 - α) × mode
    4. Proposal covariance: Σ_prop = g × Σ_base
    5. Sample: y ~ N(μ_prop, Σ_prop)

Parameters:
    - COV_MULT: Base step size multiplier (default 1.0)
    - K_G: Controls g decay rate (default 10.0)
    - K_ALPHA: Controls α rise rate (default 3.0)

Example behavior:
    d=0:   α=0.00, g=1.00  (pure mode-targeting)
    d=1:   α=0.10, g=0.99  (gentle pull begins)
    d=2:   α=0.31, g=0.96  (moderate blend)
    d=3:   α=0.50, g=0.92  (balanced)
    d=5:   α=0.74, g=0.80  (mostly tracking)
    d=10:  α=0.92, g=0.50  (strong tracking)
    d=20:  α=0.98, g=0.20  (full rescue mode)
"""

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot
from .common import COV_NUGGET, compute_alpha_g_scalar


def mcov_mode_proposal(operand):
    """
    Mode-targeting mean-covariance proposal with scalar α.

    Like MCOV_MODE_VEC but uses a single Mahalanobis distance for all parameters,
    giving uniform interpolation behavior across the block.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov,
                          coupled_blocks, block_mask, settings, grad_fn, block_mode)

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode = operand
    del grad_fn, coupled_blocks, step_mean  # Unused - we use block_mode instead of step_mean

    new_key, proposal_key = random.split(key)

    cov_mult = settings[SettingSlot.COV_MULT]
    k_g = settings[SettingSlot.K_G]
    k_alpha = settings[SettingSlot.K_ALPHA]

    # Get dimensions
    ndim = jnp.sum(block_mask)
    block_size = current_block.shape[0]

    # Scale covariance by cov_mult, then regularize
    cov_scaled = cov_mult * step_cov + COV_NUGGET * jnp.eye(block_size)

    # Cholesky decomposition for sampling and Mahalanobis distance
    L = jnp.linalg.cholesky(cov_scaled)

    # === STEP 1: Compute Mahalanobis distance from MODE ===
    diff_current = (current_block - block_mode) * block_mask
    # d² = diff' Σ^{-1} diff = ||L^{-1} diff||²
    y_current = jax.scipy.linalg.solve_triangular(L, diff_current, lower=True)
    d_current = jnp.sqrt(jnp.sum(y_current ** 2) + 1e-10)

    # === STEP 2: Compute scalar alpha and g for current state ===
    alpha_current, g_current = compute_alpha_g_scalar(d_current, k_g, k_alpha)
    sqrt_g_current = jnp.sqrt(jnp.maximum(g_current, 1e-10))

    # === STEP 3: Compute proposal mean (uniform interpolation toward MODE) ===
    prop_mean_current = alpha_current * current_block + (1.0 - alpha_current) * block_mode

    # === STEP 4: Sample from N(μ_prop, g × Σ_base) ===
    # Sample: μ + sqrt(g) × L × z
    noise = random.normal(proposal_key, shape=current_block.shape)
    L_noise = L @ noise
    diffusion = sqrt_g_current * L_noise
    proposal = (prop_mean_current + diffusion) * block_mask + current_block * (1 - block_mask)

    # === STEP 5: Compute quantities for reverse direction ===
    diff_proposal = (proposal - block_mode) * block_mask
    y_proposal = jax.scipy.linalg.solve_triangular(L, diff_proposal, lower=True)
    d_proposal = jnp.sqrt(jnp.sum(y_proposal ** 2) + 1e-10)

    alpha_proposal, g_proposal = compute_alpha_g_scalar(d_proposal, k_g, k_alpha)
    sqrt_g_proposal = jnp.sqrt(jnp.maximum(g_proposal, 1e-10))

    prop_mean_proposal = alpha_proposal * proposal + (1.0 - alpha_proposal) * block_mode

    # === STEP 6: Hastings ratio ===
    # q(y|x) = N(y | μ_x, g_x × Σ_base)
    # q(x|y) = N(x | μ_y, g_y × Σ_base)
    #
    # Log-determinant term:
    # log det(g × Σ) = ndim × log(g) + log det(Σ)
    # The log det(Σ) cancels between forward and reverse.
    log_det_term = -0.5 * ndim * (jnp.log(g_proposal + 1e-10) - jnp.log(g_current + 1e-10))

    # Quadratic form for N(y | μ, g × Σ):
    # ||y - μ||²_{(g × Σ)^{-1}} = (1/g) × ||y - μ||²_{Σ^{-1}}
    #                           = (1/g) × ||L^{-1}(y - μ)||²

    # Forward: proposal given current
    diff_forward = (proposal - prop_mean_current) * block_mask
    y_forward = jax.scipy.linalg.solve_triangular(L, diff_forward, lower=True)
    dist_sq_forward = jnp.sum(y_forward ** 2) / (g_current + 1e-10)

    # Reverse: current given proposal
    diff_reverse = (current_block - prop_mean_proposal) * block_mask
    y_reverse = jax.scipy.linalg.solve_triangular(L, diff_reverse, lower=True)
    dist_sq_reverse = jnp.sum(y_reverse ** 2) / (g_proposal + 1e-10)

    log_hastings_ratio = log_det_term - 0.5 * (dist_sq_reverse - dist_sq_forward)

    return proposal, log_hastings_ratio, new_key
