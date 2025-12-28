"""
Mixture Proposal for MCMC Sampling

Combines independent and random walk proposals with configurable mixing weight.
"""

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot


def mixture_proposal(operand):
    """
    Mixture proposal: with probability alpha use independent, else random walk.

    Proposal:
        With prob alpha: x' ~ N(μ_pop, Σ)  [independent]
        With prob 1-alpha: x' ~ N(x_curr, Σ)  [random walk]

    where μ and Σ are precomputed from coupled_blocks.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean (block_size,)
            step_cov: Precomputed covariance matrix (block_size, block_size)
            coupled_blocks: Raw states (unused for continuous proposals)
            block_mask: Mask for valid parameters
            settings: JAX array of settings
                [ALPHA] - mixture weight (probability of using independent proposal)

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings = operand
    alpha = settings[SettingSlot.ALPHA]

    new_key, choice_key, proposal_key = random.split(key, 3)

    use_chain_mean = random.uniform(choice_key) < alpha

    L = jnp.linalg.cholesky(step_cov)
    noise = random.normal(proposal_key, shape=current_block.shape)
    perturbation = L @ noise

    proposal_chain = step_mean + (perturbation * block_mask)
    proposal_self = current_block + (perturbation * block_mask)

    proposal = jnp.where(use_chain_mean, proposal_chain, proposal_self)

    # Hastings ratio for mixture:
    # q(x'|x) = alpha * N(x'|μ,Σ) + (1-alpha) * N(x'|x,Σ)
    # q(x|x') = alpha * N(x|μ,Σ) + (1-alpha) * N(x|x',Σ)

    def log_density(y, center):
        diff = (y - center) * block_mask
        solved = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
        return -0.5 * jnp.sum(solved**2)

    log_q_ind_forward = log_density(proposal, step_mean)
    log_q_rw_forward = log_density(proposal, current_block)
    log_q_forward = jnp.logaddexp(
        jnp.log(alpha) + log_q_ind_forward,
        jnp.log(1 - alpha) + log_q_rw_forward
    )

    log_q_ind_backward = log_density(current_block, step_mean)
    log_q_rw_backward = log_density(current_block, proposal)
    log_q_backward = jnp.logaddexp(
        jnp.log(alpha) + log_q_ind_backward,
        jnp.log(1 - alpha) + log_q_rw_backward
    )

    log_hastings_ratio = log_q_backward - log_q_forward

    return proposal, log_hastings_ratio, new_key
