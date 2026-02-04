"""
Random Walk Proposal for MCMC Sampling

Simple random walk proposal with identity covariance matrix.
Does NOT use coupled chain statistics - completely independent proposal.
"""

import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot


def rand_walk_proposal(operand):
    """
    Random walk proposal with identity covariance (no coupled chain stats).

    Proposal: x' ~ N(x_current, cov_mult * I)

    This is the simplest possible proposal - isotropic Gaussian noise
    centered on the current state. It does not use any information from
    the coupled chains (step_mean, step_cov, coupled_blocks are all ignored).

    Use cases:
    - Baseline comparison against adaptive proposals
    - When coupled chain statistics are unreliable (early burn-in)
    - Simple models where adaptive proposals are not needed

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode)
            key: JAX random key
            current_block: Current parameter values (block_size,)
            step_mean: Precomputed mean (UNUSED - ignored)
            step_cov: Precomputed covariance matrix (UNUSED - ignored)
            coupled_blocks: Raw states (UNUSED - ignored)
            block_mask: Mask for valid parameters (1.0 = active, 0.0 = inactive)
            settings: JAX array of settings
                [COV_MULT] - covariance multiplier (default 1.0), controls step size
            grad_fn: Gradient function (UNUSED - ignored)
            block_mode: Mode chain values (UNUSED - ignored)

    Returns:
        proposal: Proposed parameter values
        log_hastings_ratio: 0.0 (symmetric proposal)
        new_key: Updated random key
    """
    key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode = operand
    del step_mean, step_cov, coupled_blocks, grad_fn, block_mode  # Unused

    new_key, proposal_key = random.split(key)

    cov_mult = settings[SettingSlot.COV_MULT]

    # Simple isotropic Gaussian noise with variance = cov_mult
    # Standard deviation = sqrt(cov_mult)
    noise = random.normal(proposal_key, shape=current_block.shape)
    perturbation = noise * jnp.sqrt(cov_mult)

    # Apply mask to only perturb valid parameters
    proposal = current_block + (perturbation * block_mask)

    # Symmetric proposal: q(x'|x) = q(x|x'), so log ratio = 0
    log_hastings_ratio = 0.0

    return proposal, log_hastings_ratio, new_key
