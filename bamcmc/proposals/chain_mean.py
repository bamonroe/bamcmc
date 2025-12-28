"""
Chain-Mean Proposal for MCMC Sampling

Asymmetric proposal centered on population mean.
Also known as "independent" proposal.
"""

import jax
import jax.numpy as jnp
import jax.random as random


def chain_mean_proposal(operand):
    """
    Chain-mean proposal: center on population mean.

    Proposal: x' ~ N(μ, Σ)
    where μ and Σ are precomputed from coupled_blocks.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean (block_size,)
            step_cov: Precomputed covariance matrix (block_size, block_size)
            coupled_blocks: Raw states (unused for continuous proposals)
            block_mask: Mask for valid parameters
            settings: JAX array of settings (unused by this proposal)

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings = operand
    new_key, proposal_key = random.split(key)

    L = jnp.linalg.cholesky(step_cov)
    noise = random.normal(proposal_key, shape=current_block.shape)
    perturbation = L @ noise

    proposal = step_mean + (perturbation * block_mask)

    # Hastings ratio: log(q(x_curr | μ)) - log(q(x_prop | μ))
    diff_curr = (current_block - step_mean) * block_mask
    diff_prop = (proposal - step_mean) * block_mask

    y_curr = jax.scipy.linalg.solve_triangular(L, diff_curr, lower=True)
    y_prop = jax.scipy.linalg.solve_triangular(L, diff_prop, lower=True)

    dist_curr = jnp.sum(y_curr**2)
    dist_prop = jnp.sum(y_prop**2)

    log_hastings_ratio = 0.5 * (dist_prop - dist_curr)

    return proposal, log_hastings_ratio, new_key
