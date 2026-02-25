"""
Chain-Mean Proposal for MCMC Sampling

Asymmetric (independent) proposal centered on the coupled-chain population
mean, using the empirical covariance from coupled chains.

Proposal: x' ~ N(μ_pop, Σ)
where:
    - μ_pop is the mean of the coupled chains (step_mean)
    - Σ is the empirical covariance from coupled chains (step_cov)

Hastings ratio: log q(x|μ,Σ) - log q(x'|μ,Σ) = 0.5 * (d(x')² - d(x)²)
where d(x)² = (x - μ)ᵀ Σ⁻¹ (x - μ) is the Mahalanobis distance.

This is an independent proposal — the proposal distribution does not depend
on the current state x, only on the coupled-chain statistics. It works well
when chains are well-mixed (population mean is representative) but can have
low acceptance rates when chains are far from the mean.

Best suited for low-dimensional blocks or as a component in MIXTURE proposals.
No additional settings are used; the covariance is not scaled by COV_MULT.
"""

import jax
import jax.numpy as jnp
import jax.random as random

from .common import unpack_operand, sample_diffusion


def chain_mean_proposal(operand):
    """
    Chain-mean proposal: center on population mean.

    Proposal: x' ~ N(μ, Σ)
    where μ and Σ are precomputed from coupled_blocks.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean (block_size,)
            step_cov: Precomputed covariance matrix (block_size, block_size)
            coupled_blocks: Raw states (unused for continuous proposals)
            block_mask: Mask for valid parameters
            settings: JAX array of settings (unused by this proposal)
            grad_fn: Gradient function (unused by chain-mean)
            block_mode: Mode chain values (unused by chain-mean)

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    op = unpack_operand(operand)
    new_key, proposal_key = random.split(op.key)

    L = jnp.linalg.cholesky(op.step_cov)
    perturbation = sample_diffusion(proposal_key, L, op.current_block.shape)

    proposal = op.step_mean + (perturbation * op.block_mask)

    # Hastings ratio: log(q(x_curr | μ)) - log(q(x_prop | μ))
    diff_curr = (op.current_block - op.step_mean) * op.block_mask
    diff_prop = (proposal - op.step_mean) * op.block_mask

    y_curr = jax.scipy.linalg.solve_triangular(L, diff_curr, lower=True)
    y_prop = jax.scipy.linalg.solve_triangular(L, diff_prop, lower=True)

    log_hastings_ratio = 0.5 * (jnp.sum(y_prop**2) - jnp.sum(y_curr**2))

    return proposal, log_hastings_ratio, new_key
