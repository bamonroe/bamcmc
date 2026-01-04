"""
MALA (Metropolis-Adjusted Langevin Algorithm) Proposal

Gradient-based proposal that uses the score function to bias proposals
toward higher density regions. Preconditioned using the empirical
covariance from coupled chains.

Proposal: x' = x + (cov_mult/2) * Σ * ∇log p(x) + √cov_mult * L * z
where:
    - cov_mult scales the proposal variance (same as other proposals)
    - Σ is the preconditioning matrix (from coupled chains)
    - L is the Cholesky factor of Σ
    - z ~ N(0, I)

The proposal distribution is: q(x'|x) = N(x + drift, cov_mult * Σ)

Using cov_mult (instead of epsilon) makes the interface consistent with
MIXTURE and SELF_MEAN proposals, where cov_mult=1.0 means the proposal
variance equals the coupled-chain covariance.
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
                          coupled_blocks, block_mask, settings, grad_fn, block_mode)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean (unused by MALA)
            step_cov: Precomputed covariance matrix (block_size, block_size)
                     Used as the preconditioning/mass matrix
            coupled_blocks: Raw states (unused by MALA)
            block_mask: Mask for valid parameters (1.0 = active, 0.0 = inactive)
            settings: JAX array of settings
                [COV_MULT] - covariance multiplier (default 1.0)
                            Proposal variance = cov_mult * Σ
                            Same interface as MIXTURE and SELF_MEAN
            grad_fn: Function that takes block values and returns gradient
                     of log posterior w.r.t. those values
            block_mode: Mode chain values (unused by MALA)

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode = operand
    del block_mode  # Unused by this proposal
    new_key, proposal_key = random.split(key)

    # Use cov_mult for consistent interface with other proposals
    # Internally: epsilon = sqrt(cov_mult), so variance = epsilon² * Σ = cov_mult * Σ
    cov_mult = settings[SettingSlot.COV_MULT]
    epsilon = jnp.sqrt(cov_mult)

    d = current_block.shape[0]

    # Regularize covariance for numerical stability
    cov_reg = step_cov + COV_NUGGET * jnp.eye(d)

    # Cholesky decomposition for sampling and density computation
    L = jnp.linalg.cholesky(cov_reg)

    # Gradient at current state
    current_grad = grad_fn(current_block)

    # Preconditioned drift: (cov_mult/2) * Σ * ∇log p(x)
    # Equivalently: (ε²/2) * Σ * ∇log p(x)
    drift_current = 0.5 * cov_mult * (cov_reg @ current_grad)

    # Noise term: √cov_mult * L * z = ε * L * z
    noise = random.normal(proposal_key, shape=current_block.shape)
    diffusion = epsilon * (L @ noise)

    # Proposal: x' = x + drift + noise (masked)
    proposal = current_block + (drift_current + diffusion) * block_mask

    # --- Hastings ratio computation ---
    # q(x'|x) = N(x' | x + drift(x), cov_mult * Σ)
    # q(x|x') = N(x | x' + drift(x'), cov_mult * Σ)
    # log ratio = log q(x|x') - log q(x'|x)

    # Gradient at proposed state
    proposed_grad = grad_fn(proposal)

    # Drift at proposed state
    drift_proposed = 0.5 * cov_mult * (cov_reg @ proposed_grad)

    # Precision matrix scaled by 1/cov_mult
    # For N(μ, cov_mult*Σ), quadratic form is (1/cov_mult) * (x-μ)ᵀ Σ⁻¹ (x-μ)
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
