"""
Common utilities for proposal distributions.

This module provides shared functions and constants used across multiple
proposal implementations to reduce code duplication.

Constants:
    COV_NUGGET: Small regularization constant for covariance matrix inversion

Functions:
    regularize_covariance: Add nugget regularization to covariance matrix
    compute_mahalanobis: Compute Mahalanobis distance using Cholesky factor
    compute_alpha_g_scalar: Compute scalar alpha and g for mode-targeting proposals
    compute_alpha_g_vec: Compute per-parameter alpha and scalar g (vectorized)
    compute_log_det_ratio: Compute log-determinant ratio for Hastings correction
"""

import jax.numpy as jnp
import jax.scipy.linalg


# Regularization constant for covariance matrix inversion
# Small enough to not affect well-conditioned matrices,
# large enough to prevent numerical issues with ill-conditioned ones
COV_NUGGET = 1e-6


def regularize_covariance(cov, cov_mult=1.0, nugget=COV_NUGGET):
    """
    Scale and regularize a covariance matrix for numerical stability.

    Args:
        cov: Input covariance matrix (n, n)
        cov_mult: Covariance multiplier (proposal variance = cov_mult * cov)
        nugget: Small constant added to diagonal (default: COV_NUGGET)

    Returns:
        Regularized covariance matrix: cov_mult * cov + nugget * I
    """
    n = cov.shape[0]
    return cov_mult * cov + nugget * jnp.eye(n)


def compute_mahalanobis(diff, L):
    """
    Compute Mahalanobis distance using pre-computed Cholesky factor.

    The Mahalanobis distance d² = diff' Σ^{-1} diff = ||L^{-1} diff||²

    Args:
        diff: Difference vector (n,)
        L: Lower Cholesky factor of covariance matrix (n, n)

    Returns:
        y: Whitened vector L^{-1} diff (n,)
        d: Mahalanobis distance (scalar)
    """
    y = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    d = jnp.sqrt(jnp.sum(y ** 2) + 1e-10)
    return y, d


def compute_alpha_linear(d, k):
    """
    Compute linear interpolation weight alpha = d / (d + k).

    Used by MEAN_WEIGHTED and MODE_WEIGHTED proposals.

    Behavior:
        - d=0: alpha=0 (pure target-centering)
        - d=k: alpha=0.5 (balanced blend)
        - d→∞: alpha→1 (track current state)

    Args:
        d: Distance (Mahalanobis or standardized)
        k: Interpolation constant (typically 4 * sqrt(ndim))

    Returns:
        alpha: Interpolation weight in [0, 1]
    """
    return d / (d + k + 1e-10)


def compute_alpha_g_scalar(d, k_g, k_alpha):
    """
    Compute scalar alpha and g with smooth transition based on distance.

    Used by MCOV_MODE proposal.

    At d=0: alpha=0 (pure mode-targeting), g=1 (full variance)
    As d→∞: alpha→1 (track current), g→0 (cautious steps)

    Args:
        d: Scalar distance (Mahalanobis from target)
        k_g: Controls g decay rate (higher = maintain larger steps further from target)
        k_alpha: Controls alpha rise rate (higher = stay with target-centering longer)

    Returns:
        alpha: Scalar interpolation weight in [0, 1]
        g: Scalar variance scaling factor in (0, 1]
    """
    # g: decays from 1 toward 0 as d increases
    g = 1.0 / (1.0 + (d / k_g) ** 2)

    # d2: distance in proposal metric
    sqrt_g = jnp.sqrt(jnp.maximum(g, 1e-10))
    d2 = d / (sqrt_g + 1e-10)

    # alpha: rises from 0 toward 1 as d2 increases
    d2_sq = d2 ** 2
    k_alpha_sq = k_alpha ** 2
    alpha = d2_sq / (d2_sq + k_alpha_sq + 1e-10)

    return alpha, g


def compute_alpha_g_vec(d_vec, block_mask, k_g, k_alpha):
    """
    Compute per-parameter alpha and scalar g with smooth transition (vectorized).

    Used by MCOV_SMOOTH and MCOV_MODE_VEC proposals.

    At d=0: alpha=0 (pure target-centering), g=1 (full variance)
    As d→∞: alpha→1 (track current), g→0 (cautious steps)

    The scalar g uses the minimum across valid parameters (cautious approach).

    Args:
        d_vec: Per-parameter distances from target (vector)
        block_mask: Mask for valid parameters (1.0 = active, 0.0 = inactive)
        k_g: Controls g decay rate (higher = maintain larger steps further from target)
        k_alpha: Controls alpha rise rate (higher = stay with target-centering longer)

    Returns:
        alpha_vec: Per-parameter interpolation weights (vector)
        g: Scalar variance scaling factor (minimum across valid params)
    """
    # g per parameter: decays from 1 toward 0 as d increases
    g_vec = 1.0 / (1.0 + (d_vec / k_g) ** 2)

    # d2 per parameter: distance in proposal metric
    sqrt_g_vec = jnp.sqrt(jnp.maximum(g_vec, 1e-10))
    d2_vec = d_vec / (sqrt_g_vec + 1e-10)

    # alpha per parameter: rises from 0 toward 1 as d2 increases
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


def compute_log_det_ratio(g_proposal, g_current, ndim):
    """
    Compute log-determinant ratio for Hastings correction when covariance scales.

    For proposals where Sigma_prop = g * Sigma_base, the determinant ratio is:
        det(g_proposal * Sigma) / det(g_current * Sigma) = (g_proposal / g_current)^ndim

    The log-determinant term in the Hastings ratio (log q(x|y) - log q(y|x)) is:
        -0.5 * ndim * (log g_proposal - log g_current)

    Args:
        g_proposal: Covariance scale factor at proposed state
        g_current: Covariance scale factor at current state
        ndim: Number of dimensions (effective, accounting for mask)

    Returns:
        Log-determinant contribution to Hastings ratio
    """
    return -0.5 * ndim * (jnp.log(g_proposal + 1e-10) - jnp.log(g_current + 1e-10))
