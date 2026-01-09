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

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot

# Regularization for covariance matrix inversion
COV_NUGGET = 1e-6


def compute_alpha_g_vec(d_vec, block_mask, k_g, k_alpha):
    """
    Compute per-parameter α and scalar g with smooth transition.

    No hard boundaries - parameters change continuously from d=0:
    - At d=0: α=0 (pure chain_mean), g=1 (full variance)
    - As d→∞: α→1 (track current), g→0 (cautious steps)

    Args:
        d_vec: Per-parameter distances from coupled mean (vector)
        block_mask: Mask for valid parameters
        k_g: Controls g decay rate (higher = maintain larger steps further from mean)
        k_alpha: Controls α rise rate (higher = stay with chain_mean longer)

    Returns:
        alpha_vec: Per-parameter interpolation weights (vector)
        g: Scalar variance scaling factor (minimum across valid params)
    """
    # g per parameter: decays from 1 toward 0 as d increases
    g_vec = 1.0 / (1.0 + (d_vec / k_g) ** 2)

    # d2 per parameter: distance in proposal metric
    sqrt_g_vec = jnp.sqrt(jnp.maximum(g_vec, 1e-10))
    d2_vec = d_vec / (sqrt_g_vec + 1e-10)

    # α per parameter: rises from 0 toward 1 as d2 increases
    d2_sq = d2_vec ** 2
    k_alpha_sq = k_alpha ** 2
    alpha_vec = d2_sq / (d2_sq + k_alpha_sq + 1e-10)

    # Scalar g: use minimum across valid parameters (cautious approach)
    # When any parameter is far, use small steps
    g_masked = jnp.where(block_mask > 0, g_vec, 1.0)  # Set masked to 1 so they don't affect min
    g = jnp.min(g_masked)

    # Zero out alpha for masked parameters
    alpha_vec = alpha_vec * block_mask

    return alpha_vec, g


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
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode = operand
    del grad_fn, block_mode, coupled_blocks  # Unused

    new_key, proposal_key = random.split(key)

    cov_mult = settings[SettingSlot.COV_MULT]
    k_g = settings[SettingSlot.K_G]
    k_alpha = settings[SettingSlot.K_ALPHA]

    # Get dimensions
    ndim = jnp.sum(block_mask)
    block_size = current_block.shape[0]

    # Scale covariance by cov_mult, then regularize
    cov_scaled = cov_mult * step_cov + COV_NUGGET * jnp.eye(block_size)

    # Extract diagonal standard deviations for per-parameter distances
    sigma_diag = jnp.sqrt(jnp.diag(cov_scaled))

    # Cholesky decomposition for sampling
    L = jnp.linalg.cholesky(cov_scaled)

    # === STEP 1: Compute per-parameter distances from coupled mean ===
    diff_current = (current_block - step_mean) * block_mask
    d_vec_current = jnp.abs(diff_current) / (sigma_diag + 1e-10)

    # === STEP 2: Compute per-param alpha and scalar g for current state ===
    alpha_vec_current, g_current = compute_alpha_g_vec(d_vec_current, block_mask, k_g, k_alpha)
    sqrt_g_current = jnp.sqrt(jnp.maximum(g_current, 1e-10))

    # === STEP 3: Compute proposal mean (per-parameter interpolation) ===
    prop_mean_current = alpha_vec_current * current_block + (1.0 - alpha_vec_current) * step_mean

    # === STEP 4: Sample from N(μ_prop, g × Σ_base) ===
    # Sample: μ + sqrt(g) × L × z
    noise = random.normal(proposal_key, shape=current_block.shape)
    L_noise = L @ noise
    diffusion = sqrt_g_current * L_noise
    proposal = (prop_mean_current + diffusion) * block_mask + current_block * (1 - block_mask)

    # === STEP 5: Compute quantities for reverse direction ===
    diff_proposal = (proposal - step_mean) * block_mask
    d_vec_proposal = jnp.abs(diff_proposal) / (sigma_diag + 1e-10)

    alpha_vec_proposal, g_proposal = compute_alpha_g_vec(d_vec_proposal, block_mask, k_g, k_alpha)
    sqrt_g_proposal = jnp.sqrt(jnp.maximum(g_proposal, 1e-10))

    prop_mean_proposal = alpha_vec_proposal * proposal + (1.0 - alpha_vec_proposal) * step_mean

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
