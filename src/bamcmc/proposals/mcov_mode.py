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

import jax.numpy as jnp

from ..settings import SettingSlot
from .common import (prepare_proposal, sample_diffusion, compute_mahalanobis,
                     compute_alpha_g_scalar, hastings_ratio_scalar_g)


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
    ps = prepare_proposal(operand)
    op = ps.op

    k_g = op.settings[SettingSlot.K_G]
    k_alpha = op.settings[SettingSlot.K_ALPHA]

    # Get dimensions
    ndim = jnp.sum(op.block_mask)

    # === STEP 1: Compute Mahalanobis distance from MODE ===
    diff_current = (op.current_block - op.block_mode) * op.block_mask
    _, d_current = compute_mahalanobis(diff_current, ps.L)

    # === STEP 2: Compute scalar alpha and g for current state ===
    alpha_current, g_current = compute_alpha_g_scalar(d_current, k_g, k_alpha)
    sqrt_g_current = jnp.sqrt(jnp.maximum(g_current, 1e-10))

    # === STEP 3: Compute proposal mean (uniform interpolation toward MODE) ===
    prop_mean_current = alpha_current * op.current_block + (1.0 - alpha_current) * op.block_mode

    # === STEP 4: Sample from N(μ_prop, g × Σ_base) ===
    diffusion = sample_diffusion(ps.proposal_key, ps.L, op.current_block.shape, scale=sqrt_g_current)
    proposal = (prop_mean_current + diffusion) * op.block_mask + op.current_block * (1 - op.block_mask)

    # === STEP 5: Compute quantities for reverse direction ===
    diff_proposal = (proposal - op.block_mode) * op.block_mask
    _, d_proposal = compute_mahalanobis(diff_proposal, ps.L)

    alpha_proposal, g_proposal = compute_alpha_g_scalar(d_proposal, k_g, k_alpha)

    prop_mean_proposal = alpha_proposal * proposal + (1.0 - alpha_proposal) * op.block_mode

    # === STEP 6: Hastings ratio ===
    log_hastings_ratio = hastings_ratio_scalar_g(
        ps.L, ndim, proposal, op.current_block,
        prop_mean_current, prop_mean_proposal,
        g_current, g_proposal, op.block_mask)

    return proposal, log_hastings_ratio, ps.new_key
