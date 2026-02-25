"""
Common utilities for proposal distributions.

This module provides shared functions and constants used across multiple
proposal implementations to reduce code duplication.

Constants:
    COV_NUGGET: Small regularization constant for covariance matrix inversion

Functions:
    unpack_operand: Unpack the 9-element operand tuple into a named struct
    prepare_proposal: Common setup (unpack, split key, regularize cov, Cholesky)
    sample_diffusion: Generate diffusion noise from Cholesky factor
    regularize_covariance: Add nugget regularization to covariance matrix
    compute_mahalanobis: Compute Mahalanobis distance using Cholesky factor
    compute_alpha_g_scalar: Compute scalar alpha and g for mode-targeting proposals
    compute_alpha_g_vec: Compute per-parameter alpha and scalar g (vectorized)
    compute_log_det_ratio: Compute log-determinant ratio for Hastings correction
    hastings_ratio_fixed_cov: Hastings ratio for proposals with state-dependent mean, fixed cov
    hastings_ratio_scalar_g: Hastings ratio for proposals with scalar g covariance scaling
"""

from collections import namedtuple

import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg

from ..settings import SettingSlot


# Regularization constant for covariance matrix inversion
# Small enough to not affect well-conditioned matrices,
# large enough to prevent numerical issues with ill-conditioned ones
COV_NUGGET = 1e-6


# Named tuple for unpacked operand fields
Operand = namedtuple('Operand', [
    'key', 'current_block', 'step_mean', 'step_cov',
    'coupled_blocks', 'block_mask', 'settings', 'grad_fn', 'block_mode',
])

# Named tuple for common proposal setup
ProposalSetup = namedtuple('ProposalSetup', [
    'op', 'new_key', 'proposal_key', 'cov_mult', 'L',
])


def unpack_operand(operand):
    """
    Unpack the 9-element operand tuple into a named struct.

    Every proposal receives the same (key, current_block, step_mean, step_cov,
    coupled_blocks, block_mask, settings, grad_fn, block_mode) tuple. This
    helper avoids repeating the destructuring line in each proposal.

    Args:
        operand: 9-element tuple passed to proposal functions.

    Returns:
        Operand namedtuple with named fields.
    """
    return Operand(*operand)


def prepare_proposal(operand):
    """
    Common proposal setup: unpack, split key, extract cov_mult, regularize, Cholesky.

    Used by proposals that need a regularized, scaled covariance and its Cholesky
    factor. Combines the 4-5 lines that ~10 proposals repeat identically.

    Args:
        operand: 9-element tuple passed to proposal functions.

    Returns:
        ProposalSetup namedtuple with fields:
            op: Operand namedtuple
            new_key: Updated random key for next step
            proposal_key: Key for sampling this proposal
            cov_mult: Covariance multiplier from settings
            L: Lower Cholesky factor of (cov_mult * step_cov + nugget * I)
    """
    op = unpack_operand(operand)
    new_key, proposal_key = random.split(op.key)
    cov_mult = op.settings[SettingSlot.COV_MULT]
    cov_reg = regularize_covariance(op.step_cov, cov_mult)
    L = jnp.linalg.cholesky(cov_reg)
    return ProposalSetup(op=op, new_key=new_key, proposal_key=proposal_key,
                         cov_mult=cov_mult, L=L)


def sample_diffusion(proposal_key, L, shape, scale=1.0):
    """
    Generate diffusion noise: scale * (L @ z) where z ~ N(0, I).

    Args:
        proposal_key: JAX random key for sampling.
        L: Lower Cholesky factor (n, n).
        shape: Shape for normal samples (n,).
        scale: Scalar multiplier (e.g. sqrt(g) for covariance scaling).

    Returns:
        Diffusion vector (n,).
    """
    noise = random.normal(proposal_key, shape=shape)
    return scale * (L @ noise)


def hastings_ratio_fixed_cov(L, epsilon, proposal, current_block,
                             prop_mean_fwd, prop_mean_rev, block_mask):
    """
    Hastings ratio for proposals with state-dependent mean but fixed covariance.

    Used by MEAN_WEIGHTED and MODE_WEIGHTED. The covariance is the same in both
    directions so only the quadratic forms differ (no log-determinant term).

    log q(x|y) - log q(y|x) = -0.5 * [||x - mu_y||^2 - ||y - mu_x||^2] / eps^2

    where ||.||^2 is measured with Sigma^{-1} (via Cholesky solve).

    Args:
        L: Lower Cholesky factor of covariance matrix.
        epsilon: sqrt(cov_mult), scales the precision.
        proposal: Proposed state.
        current_block: Current state.
        prop_mean_fwd: Forward proposal mean (mu_x).
        prop_mean_rev: Reverse proposal mean (mu_y).
        block_mask: Parameter mask.

    Returns:
        Log Hastings ratio (scalar).
    """
    diff_forward = (proposal - prop_mean_fwd) * block_mask
    y_forward = jax.scipy.linalg.solve_triangular(L, diff_forward, lower=True) / epsilon

    diff_reverse = (current_block - prop_mean_rev) * block_mask
    y_reverse = jax.scipy.linalg.solve_triangular(L, diff_reverse, lower=True) / epsilon

    return -0.5 * (jnp.sum(y_reverse**2) - jnp.sum(y_forward**2))


def hastings_ratio_scalar_g(L, ndim, proposal, current_block,
                            prop_mean_fwd, prop_mean_rev,
                            g_fwd, g_rev, block_mask):
    """
    Hastings ratio for proposals with scalar g covariance scaling.

    Used by MCOV_SMOOTH, MCOV_MODE, and MCOV_MODE_VEC. The covariance scales
    as g * Sigma_base, producing both a log-determinant term and scaled
    quadratic forms.

    log q(x|y) - log q(y|x) = -0.5 * ndim * [log g_rev - log g_fwd]
                             - 0.5 * [||x - mu_y||^2/g_rev - ||y - mu_x||^2/g_fwd]

    Args:
        L: Lower Cholesky factor of base covariance.
        ndim: Effective number of dimensions (sum of block_mask).
        proposal: Proposed state.
        current_block: Current state.
        prop_mean_fwd: Forward proposal mean.
        prop_mean_rev: Reverse proposal mean.
        g_fwd: Scalar covariance scale at current state.
        g_rev: Scalar covariance scale at proposed state.
        block_mask: Parameter mask.

    Returns:
        Log Hastings ratio (scalar).
    """
    log_det_term = compute_log_det_ratio(g_rev, g_fwd, ndim)

    diff_forward = (proposal - prop_mean_fwd) * block_mask
    y_forward = jax.scipy.linalg.solve_triangular(L, diff_forward, lower=True)
    dist_sq_forward = jnp.sum(y_forward ** 2) / (g_fwd + 1e-10)

    diff_reverse = (current_block - prop_mean_rev) * block_mask
    y_reverse = jax.scipy.linalg.solve_triangular(L, diff_reverse, lower=True)
    dist_sq_reverse = jnp.sum(y_reverse ** 2) / (g_rev + 1e-10)

    return log_det_term - 0.5 * (dist_sq_reverse - dist_sq_forward)


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
