"""
Mode-Targeting Mean-Covariance Proposal (MCOV_MODE_VEC) - Vectorized

Like MCOV_SMOOTH, but targets the MODE (highest log-posterior chain) instead of
the coupled mean. This breaks the feedback loop where the coupled mean is
dominated by lower-density modes.

The key insight: when the posterior has multiple modes, the coupled mean sits
between them, creating asymmetric dynamics that trap chains in lower-density
modes. By targeting the highest LP chain (the mode), all chains are pulled
toward the best region of parameter space.

Uses per-parameter distances and α values for fine-grained control:
- Each parameter gets its own α based on its individual distance from the mode
- Covariance scaling g uses minimum across parameters (cautious when any param is far)

At equilibrium (d_i ≈ 0 for all i):
    - α_i ≈ 0: proposal centered on mode
    - g ≈ 1: full step size variance
    - Behavior: fast mixing around mode

Far from mode (any d_i large):
    - α_i → 1 for far parameters: track current state
    - g → 0: small, cautious steps (driven by farthest parameter)
    - Behavior: gentle guidance back toward mode

Algorithm:
    1. Compute per-parameter distances from MODE: d_i = |x_i - mode_i| / σ_i
    2. Compute smooth parameters per-parameter:
       g_i = 1 / (1 + (d_i / k_g)²)
       d2_i = d_i / sqrt(g_i)
       α_i = d2_i² / (d2_i² + k_α²)
    3. Scalar g = min(g_i) for covariance scaling
    4. Proposal mean: μ_prop_i = α_i × x_i + (1 - α_i) × mode_i
    5. Proposal covariance: Σ_prop = g × Σ_base
    6. Sample: y ~ N(μ_prop, Σ_prop)

Parameters:
    - COV_MULT: Base step size multiplier (default 1.0)
    - K_G: Controls g decay rate per parameter (default 10.0)
    - K_ALPHA: Controls α rise rate per parameter (default 3.0)

Example behavior per parameter:
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
from .common import (prepare_proposal, sample_diffusion, compute_alpha_g_vec,
                     hastings_ratio_scalar_g)


def mcov_mode_vec_proposal(operand):
    """
    Mode-targeting mean-covariance proposal with per-parameter α (vectorized).

    Like MCOV_SMOOTH but targets the mode (highest LP chain) instead of the
    coupled mean. This prevents the feedback loop where chains get trapped
    in lower-density modes.

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
    cov_scaled = ps.cov_mult * op.step_cov + 1e-6 * jnp.eye(op.current_block.shape[0])
    sigma_diag = jnp.sqrt(jnp.diag(cov_scaled))

    # === STEP 1: Compute per-parameter distances from MODE ===
    diff_current = (op.current_block - op.block_mode) * op.block_mask
    d_vec_current = jnp.abs(diff_current) / (sigma_diag + 1e-10)

    # === STEP 2: Compute per-param alpha and scalar g for current state ===
    alpha_vec_current, g_current = compute_alpha_g_vec(d_vec_current, op.block_mask, k_g, k_alpha)
    sqrt_g_current = jnp.sqrt(jnp.maximum(g_current, 1e-10))

    # === STEP 3: Compute proposal mean (per-parameter interpolation toward MODE) ===
    prop_mean_current = alpha_vec_current * op.current_block + (1.0 - alpha_vec_current) * op.block_mode

    # === STEP 4: Sample from N(μ_prop, g × Σ_base) ===
    diffusion = sample_diffusion(ps.proposal_key, ps.L, op.current_block.shape, scale=sqrt_g_current)
    proposal = (prop_mean_current + diffusion) * op.block_mask + op.current_block * (1 - op.block_mask)

    # === STEP 5: Compute quantities for reverse direction ===
    diff_proposal = (proposal - op.block_mode) * op.block_mask
    d_vec_proposal = jnp.abs(diff_proposal) / (sigma_diag + 1e-10)

    alpha_vec_proposal, g_proposal = compute_alpha_g_vec(d_vec_proposal, op.block_mask, k_g, k_alpha)

    prop_mean_proposal = alpha_vec_proposal * proposal + (1.0 - alpha_vec_proposal) * op.block_mode

    # === STEP 6: Hastings ratio ===
    log_hastings_ratio = hastings_ratio_scalar_g(
        ps.L, ndim, proposal, op.current_block,
        prop_mean_current, prop_mean_proposal,
        g_current, g_proposal, op.block_mask)

    return proposal, log_hastings_ratio, ps.new_key
