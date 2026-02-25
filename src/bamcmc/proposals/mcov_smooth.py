"""
Smooth Mean-Covariance Proposal (MCOV_SMOOTH) - Vectorized

A proposal distribution with smooth continuous transition from chain_mean behavior
at equilibrium to pulling/tracking behavior when far from the coupled mean.

Uses per-parameter distances and α values for fine-grained control:
- Each parameter gets its own α based on its individual distance from the mean
- Covariance scaling g uses minimum across parameters (cautious when any param is far)

Key insight: chains naturally accelerate as they approach equilibrium because
both α and g change favorably - more pull toward m AND larger step sizes.

At equilibrium (d_i ≈ 0 for all i):
    - α_i ≈ 0: proposal centered on coupled mean m
    - g ≈ 1: full step size variance
    - Behavior: fast mixing like chain_mean proposal

Far from equilibrium (any d_i large):
    - α_i → 1 for far parameters: track current state
    - g → 0: small, cautious steps (driven by farthest parameter)
    - Behavior: gentle guidance back toward mean

Algorithm:
    1. Compute per-parameter distances: d_i = |x_i - m_i| / σ_i
    2. Compute smooth parameters per-parameter:
       k_g, k_α: not dimension-scaled (per-param is 1D)
       g_i = 1 / (1 + (d_i / k_g)²)
       d2_i = d_i / sqrt(g_i)
       α_i = d2_i² / (d2_i² + k_α²)
    3. Scalar g = min(g_i) for covariance scaling
    4. Proposal mean: μ_prop_i = α_i × x_i + (1 - α_i) × m_i
    5. Proposal covariance: Σ_prop = g × Σ_base
    6. Sample: y ~ N(μ_prop, Σ_prop)

Parameters:
    - COV_MULT: Base step size multiplier (default 1.0)
    - K_G: Controls g decay rate per parameter (default 10.0)
    - K_ALPHA: Controls α rise rate per parameter (default 3.0)

Example behavior per parameter:
    d=0:   α=0.00, g=1.00  (pure chain_mean)
    d=1:   α=0.10, g=0.99  (gentle pull begins)
    d=2:   α=0.31, g=0.96  (moderate blend)
    d=3:   α=0.50, g=0.92  (balanced)
    d=5:   α=0.74, g=0.80  (mostly tracking)
    d=10:  α=0.92, g=0.50  (strong tracking)
    d=20:  α=0.98, g=0.20  (full rescue mode)
"""

import jax.numpy as jnp

from ..settings import SettingSlot
from .common import (prepare_proposal, sample_diffusion, compute_alpha_g_vec,
                     hastings_ratio_scalar_g, COV_NUGGET, NUMERICAL_EPS)


def mcov_smooth_proposal(operand):
    """
    Smooth mean-covariance proposal with per-parameter α (vectorized).

    Provides optimal behavior with smooth blending:
    - Each parameter gets its own α based on its distance from mean
    - Covariance scaling g driven by farthest parameter (cautious)
    - No hard boundaries: symmetric Hastings ratios for better acceptance

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

    # Extract diagonal standard deviations for per-parameter distances
    cov_scaled = ps.cov_mult * op.step_cov + COV_NUGGET * jnp.eye(op.current_block.shape[0])
    sigma_diag = jnp.sqrt(jnp.diag(cov_scaled))

    # === STEP 1: Compute per-parameter distances from coupled mean ===
    diff_current = (op.current_block - op.step_mean) * op.block_mask
    d_vec_current = jnp.abs(diff_current) / (sigma_diag + NUMERICAL_EPS)

    # === STEP 2: Compute per-param alpha and scalar g for current state ===
    alpha_vec_current, g_current = compute_alpha_g_vec(d_vec_current, op.block_mask, k_g, k_alpha)
    sqrt_g_current = jnp.sqrt(jnp.maximum(g_current, NUMERICAL_EPS))

    # === STEP 3: Compute proposal mean (per-parameter interpolation) ===
    prop_mean_current = alpha_vec_current * op.current_block + (1.0 - alpha_vec_current) * op.step_mean

    # === STEP 4: Sample from N(μ_prop, g × Σ_base) ===
    diffusion = sample_diffusion(ps.proposal_key, ps.L, op.current_block.shape, scale=sqrt_g_current)
    proposal = (prop_mean_current + diffusion) * op.block_mask + op.current_block * (1 - op.block_mask)

    # === STEP 5: Compute quantities for reverse direction ===
    diff_proposal = (proposal - op.step_mean) * op.block_mask
    d_vec_proposal = jnp.abs(diff_proposal) / (sigma_diag + NUMERICAL_EPS)

    alpha_vec_proposal, g_proposal = compute_alpha_g_vec(d_vec_proposal, op.block_mask, k_g, k_alpha)

    prop_mean_proposal = alpha_vec_proposal * proposal + (1.0 - alpha_vec_proposal) * op.step_mean

    # === STEP 6: Hastings ratio ===
    log_hastings_ratio = hastings_ratio_scalar_g(
        ps.L, ndim, proposal, op.current_block,
        prop_mean_current, prop_mean_proposal,
        g_current, g_proposal, op.block_mask)

    return proposal, log_hastings_ratio, ps.new_key
