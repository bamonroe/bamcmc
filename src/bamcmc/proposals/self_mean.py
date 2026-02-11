"""
Self-Mean Proposal for MCMC Sampling

Symmetric random-walk proposal centered on the current state, using the
empirical covariance from coupled chains as the proposal covariance.

Proposal: x' ~ N(x_current, cov_mult * Σ)
where:
    - Σ is the empirical covariance from coupled chains (step_cov)
    - cov_mult scales the proposal variance (SettingSlot.COV_MULT, default 1.0)

Hastings ratio: 0 (symmetric proposal, q(x'|x) = q(x|x'))

Unlike RAND_WALK, this proposal adapts to the local geometry via the
coupled-chain covariance. Unlike CHAIN_MEAN, it centers on the current
state rather than the population mean, making it conservative but robust.

Settings used:
    COV_MULT - Proposal variance multiplier (default 1.0).
               Values < 1 give smaller steps (higher acceptance, slower mixing).
               Values > 1 give larger steps (lower acceptance, faster exploration).
"""

import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot


def self_mean_proposal(operand):
    """
    Self-mean proposal: center on current state.

    Proposal: x' ~ N(x_current, cov_mult * Σ)
    where Σ is precomputed from coupled_blocks.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean (unused for self-mean)
            step_cov: Precomputed covariance matrix (block_size, block_size)
            coupled_blocks: Raw states (unused for continuous proposals)
            block_mask: Mask for valid parameters (1.0 = active, 0.0 = inactive)
            settings: JAX array of settings
                [COV_MULT] - covariance multiplier (default 1.0)
            grad_fn: Gradient function (unused by self-mean)
            block_mode: Mode chain values (unused by self-mean)

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: 0.0 (symmetric proposal)
        new_key: Updated random key
    """
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode = operand
    del grad_fn, block_mode  # Unused by this proposal
    new_key, proposal_key = random.split(key)

    cov_mult = settings[SettingSlot.COV_MULT]

    # Scale covariance: sqrt(cov_mult) scales the Cholesky factor
    # so that L @ L.T = cov_mult * Σ
    L = jnp.linalg.cholesky(step_cov)
    scaled_L = L * jnp.sqrt(cov_mult)

    noise = random.normal(proposal_key, shape=current_block.shape)
    perturbation = scaled_L @ noise

    proposal = current_block + (perturbation * block_mask)
    log_hastings_ratio = 0.0

    return proposal, log_hastings_ratio, new_key
