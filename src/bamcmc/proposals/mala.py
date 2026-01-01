"""
MALA (Metropolis-Adjusted Langevin Algorithm) Proposal

Gradient-based proposal that uses the score function to bias proposals
toward higher density regions. Preconditioned using the empirical
covariance from coupled chains.

Proposal: x' = x + (ε²/2) * Σ * ∇log p(x) + ε * L * z
where:
    - ε is the step size
    - Σ is the preconditioning matrix (from coupled chains)
    - L is the Cholesky factor of Σ
    - z ~ N(0, I)
"""

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot

# Regularization for covariance matrix inversion
COV_NUGGET = 1e-6


def mala_proposal(operand):
    """
    Preconditioned MALA proposal.

    Uses the gradient of log posterior to construct a drift toward
    higher density, with the covariance matrix from coupled chains
    as the preconditioning/mass matrix.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov,
                          coupled_blocks, block_mask, settings, grad_fn)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean (unused by MALA)
            step_cov: Precomputed covariance matrix (block_size, block_size)
                     Used as the preconditioning/mass matrix
            coupled_blocks: Raw states (unused by MALA)
            block_mask: Mask for valid parameters (1.0 = active, 0.0 = inactive)
            settings: JAX array of settings
                [EPSILON] - step size (default 0.1)
            grad_fn: Function that takes block values and returns gradient
                     of log posterior w.r.t. those values

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn = operand
    new_key, proposal_key = random.split(key)

    epsilon = settings[SettingSlot.EPSILON]
    d = current_block.shape[0]

    # Regularize covariance for numerical stability
    cov_reg = step_cov + COV_NUGGET * jnp.eye(d)

    # Cholesky decomposition for sampling and density computation
    L = jnp.linalg.cholesky(cov_reg)

    # Gradient at current state
    current_grad = grad_fn(current_block)

    # Preconditioned drift: (ε²/2) * Σ * ∇log p(x)
    drift_current = 0.5 * epsilon**2 * (cov_reg @ current_grad)

    # Noise term: ε * L * z
    noise = random.normal(proposal_key, shape=current_block.shape)
    diffusion = epsilon * (L @ noise)

    # Proposal: x' = x + drift + noise (masked)
    proposal = current_block + (drift_current + diffusion) * block_mask

    # --- Hastings ratio computation ---
    # q(x'|x) = N(x' | x + drift(x), ε²Σ)
    # q(x|x') = N(x | x' + drift(x'), ε²Σ)
    # log ratio = log q(x|x') - log q(x'|x)

    # Gradient at proposed state
    proposed_grad = grad_fn(proposal)

    # Drift at proposed state
    drift_proposed = 0.5 * epsilon**2 * (cov_reg @ proposed_grad)

    # Precision matrix scaled by 1/ε²
    # For N(μ, ε²Σ), the quadratic form is (1/ε²) * (x-μ)ᵀ Σ⁻¹ (x-μ)
    # We use solve_triangular for numerical stability

    # Forward: x' given x
    # Mean of q(x'|x) is x + drift_current
    diff_forward = (proposal - current_block - drift_current) * block_mask
    y_forward = jax.scipy.linalg.solve_triangular(L, diff_forward, lower=True) / epsilon

    # Reverse: x given x'
    # Mean of q(x|x') is x' + drift_proposed
    diff_reverse = (current_block - proposal - drift_proposed) * block_mask
    y_reverse = jax.scipy.linalg.solve_triangular(L, diff_reverse, lower=True) / epsilon

    # Log densities (up to normalizing constant which cancels)
    log_q_forward = -0.5 * jnp.sum(y_forward**2)
    log_q_reverse = -0.5 * jnp.sum(y_reverse**2)

    log_hastings_ratio = log_q_reverse - log_q_forward

    return proposal, log_hastings_ratio, new_key
