"""
Mixture Proposal for MCMC Sampling

Combines the chain_mean (independent) and self_mean (random walk) proposals
into a single proposal using a configurable mixing weight.

Proposal:
    q(x'|x) = p * N(x'|μ_pop, Σ) + (1-p) * N(x'|x, cov_mult * Σ)
where:
    - p = chain_prob (mixing weight, SettingSlot.CHAIN_PROB)
    - μ_pop and Σ are from coupled chains
    - cov_mult scales the self_mean component (SettingSlot.COV_MULT)

Hastings ratio: log q(x|x') - log q(x'|x), computed via logaddexp
to maintain numerical stability across the two mixture components.

The mixture inherits the strengths of both components: the self_mean
component ensures reasonable acceptance rates even when far from the
mean, while the chain_mean component enables large jumps toward the
population center. Typical chain_prob values are 0.3–0.7.

Settings used:
    CHAIN_PROB - Probability of using chain_mean component (default 0.5)
    COV_MULT   - Variance multiplier for self_mean component (default 1.0)
"""

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot
from .common import unpack_operand


def mixture_proposal(operand):
    """
    Mixture proposal: with probability chain_prob use chain_mean, else self_mean.

    Proposal:
        With prob chain_prob: x' ~ N(μ_pop, Σ)  [chain_mean]
        With prob 1-chain_prob: x' ~ N(x_curr, cov_mult * Σ)  [self_mean with scaled cov]

    where μ and Σ are precomputed from coupled_blocks.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean (block_size,)
            step_cov: Precomputed covariance matrix (block_size, block_size)
            coupled_blocks: Raw states (unused for continuous proposals)
            block_mask: Mask for valid parameters
            settings: JAX array of settings
                [CHAIN_PROB] - probability of using chain_mean proposal
                [COV_MULT] - covariance multiplier for self_mean component (default 1.0)
            grad_fn: Gradient function (unused by mixture)
            block_mode: Mode chain values (unused by mixture)

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    op = unpack_operand(operand)
    chain_prob = op.settings[SettingSlot.CHAIN_PROB]
    cov_mult = op.settings[SettingSlot.COV_MULT]

    new_key, choice_key, proposal_key = random.split(op.key, 3)

    use_chain_mean = random.uniform(choice_key) < chain_prob

    # Cholesky for chain_mean (unscaled)
    L = jnp.linalg.cholesky(op.step_cov)

    # Scaled Cholesky for self_mean
    scaled_L = L * jnp.sqrt(cov_mult)

    noise = random.normal(proposal_key, shape=op.current_block.shape)

    # chain_mean uses unscaled L, self_mean uses scaled_L
    perturbation_chain = L @ noise
    perturbation_self = scaled_L @ noise

    proposal_chain = op.step_mean + (perturbation_chain * op.block_mask)
    proposal_self = op.current_block + (perturbation_self * op.block_mask)

    proposal = jnp.where(use_chain_mean, proposal_chain, proposal_self)

    # Hastings ratio for mixture:
    # q(x'|x) = chain_prob * N(x'|μ,Σ) + (1-chain_prob) * N(x'|x, cov_mult*Σ)
    # q(x|x') = chain_prob * N(x|μ,Σ) + (1-chain_prob) * N(x|x', cov_mult*Σ)

    def log_density_chain(y, center):
        """Log density for chain_mean (unscaled covariance)."""
        diff = (y - center) * op.block_mask
        solved = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
        return -0.5 * jnp.sum(solved**2)

    def log_density_self(y, center):
        """Log density for self_mean (scaled covariance)."""
        diff = (y - center) * op.block_mask
        solved = jax.scipy.linalg.solve_triangular(scaled_L, diff, lower=True)
        return -0.5 * jnp.sum(solved**2)

    log_q_ind_forward = log_density_chain(proposal, op.step_mean)
    log_q_rw_forward = log_density_self(proposal, op.current_block)
    log_q_forward = jnp.logaddexp(
        jnp.log(chain_prob) + log_q_ind_forward,
        jnp.log(1 - chain_prob) + log_q_rw_forward
    )

    log_q_ind_backward = log_density_chain(op.current_block, op.step_mean)
    log_q_rw_backward = log_density_self(op.current_block, proposal)
    log_q_backward = jnp.logaddexp(
        jnp.log(chain_prob) + log_q_ind_backward,
        jnp.log(1 - chain_prob) + log_q_rw_backward
    )

    log_hastings_ratio = log_q_backward - log_q_forward

    return proposal, log_hastings_ratio, new_key
