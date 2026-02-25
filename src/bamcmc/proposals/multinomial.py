"""
Multinomial Proposal for MCMC Sampling

Discrete proposal for categorical parameters on an integer grid [1, n_categories].
Proposes new values by sampling from a mixture of empirical and uniform distributions,
computed independently for each dimension in the block.

Proposal distribution per dimension:
    q(x'_d) = w * Uniform(1, K) + (1-w) * Empirical_d
where:
    - w = uniform_weight (SettingSlot.UNIFORM_WEIGHT)
    - K = n_categories (SettingSlot.N_CATEGORIES)
    - Empirical_d = frequency distribution of dimension d across coupled chains

Hastings ratio: Σ_d mask_d * [log q(x_d) - log q(x'_d)]
(product of per-dimension ratios, only over active dimensions).

Uses a fixed MAX_CATEGORIES=10 for all array allocations to avoid JAX
recompilation when n_categories changes. Invalid categories beyond
n_categories are masked to zero probability.

Settings used:
    N_CATEGORIES   - Number of valid categories, values in [1, K] (required)
    UNIFORM_WEIGHT - Weight of uniform component in mixture (default 0.4).
                     Higher values increase exploration of rare categories.
"""

import jax
import jax.numpy as jnp
import jax.random as random

from ..settings import SettingSlot
from .common import unpack_operand


def multinomial_proposal(operand):
    """
    Multinomial proposal for discrete parameters on a grid.

    Proposes values by sampling from a mixture of empirical and uniform distributions.
    Each dimension is sampled independently based on observed frequencies.

    Uses fixed MAX_CATEGORIES=10 for array allocations to avoid recompilation,
    then masks out invalid categories based on the actual n_categories setting.

    Args:
        operand: Tuple of (key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode)
            key: JAX random key
            current_block: Current parameter values (block_size,) - discrete values in [1, n_categories]
            step_mean: Precomputed mean (unused for multinomial)
            step_cov: Precomputed covariance (unused for multinomial)
            coupled_blocks: States from opposite group (n_chains, block_size)
            block_mask: Mask for valid parameters
            settings: JAX array of settings
                [UNIFORM_WEIGHT] - weight of uniform vs empirical (proposal = w*Uniform + (1-w)*Empirical)
                [N_CATEGORIES] - number of valid categories (values 1 to n_categories)
            grad_fn: Gradient function (unused by multinomial - discrete parameters)
            block_mode: Mode chain values (unused by multinomial)

    Returns:
        proposal: Proposed parameter values (discrete)
        log_hastings_ratio: Log density ratio for MH acceptance
        new_key: Updated random key
    """
    op = unpack_operand(operand)

    # Fixed maximum for array shapes - avoids recompilation
    MAX_CATEGORIES = 10
    GRID_MIN = 1

    # Get settings from array
    uniform_weight = op.settings[SettingSlot.UNIFORM_WEIGHT]
    n_categories = op.settings[SettingSlot.N_CATEGORIES].astype(jnp.int32)

    block_size = op.current_block.shape[0]

    new_key, proposal_key = random.split(op.key)

    # Convert to 0-indexed categories, clip to valid range
    coupled_indices = (op.coupled_blocks - GRID_MIN).astype(jnp.int32)
    coupled_indices = jnp.clip(coupled_indices, 0, n_categories - 1)

    # Create category validity mask: 1 for valid categories, 0 for invalid
    category_mask = jnp.arange(MAX_CATEGORIES) < n_categories  # (MAX_CATEGORIES,)

    # Uniform distribution over valid categories only
    uniform_probs = jnp.where(category_mask, 1.0 / n_categories, 0.0)

    def compute_probs_for_dim(dim_idx):
        """Compute probabilities for one dimension."""
        dim_values = coupled_indices[:, dim_idx]
        # Count occurrences using one-hot (fixed size MAX_CATEGORIES)
        one_hot = jax.nn.one_hot(dim_values, MAX_CATEGORIES)
        counts = jnp.sum(one_hot, axis=0)
        # Mask out invalid categories before normalizing
        counts = jnp.where(category_mask, counts, 0.0)
        total = jnp.sum(counts)
        # Avoid division by zero
        empirical_probs = jnp.where(total > 0, counts / total, uniform_probs)
        # Mix with uniform (both already masked to valid categories)
        final_probs = uniform_weight * uniform_probs + (1 - uniform_weight) * empirical_probs
        return final_probs

    # Compute probability distributions for each dimension
    all_probs = jax.vmap(compute_probs_for_dim)(jnp.arange(block_size))
    # all_probs shape: (block_size, MAX_CATEGORIES)

    # Sample from each dimension's distribution
    def sample_one_dim(key_and_probs):
        k, probs = key_and_probs
        # Sample category index (invalid categories have 0 probability)
        cat_idx = random.categorical(k, jnp.log(probs + 1e-10))
        # Convert back to grid value
        return (cat_idx + GRID_MIN).astype(op.current_block.dtype)

    dim_keys = random.split(proposal_key, block_size)
    proposal_values = jax.vmap(sample_one_dim)((dim_keys, all_probs))

    # Apply mask: use proposal where mask is 1, current where mask is 0
    proposal = jnp.where(op.block_mask > 0.5, proposal_values, op.current_block)

    # Compute Hastings ratio
    current_indices = (op.current_block - GRID_MIN).astype(jnp.int32)
    current_indices = jnp.clip(current_indices, 0, n_categories - 1)

    proposal_indices = (proposal - GRID_MIN).astype(jnp.int32)
    proposal_indices = jnp.clip(proposal_indices, 0, n_categories - 1)

    def get_log_prob(dim_idx):
        probs = all_probs[dim_idx]
        log_p_current = jnp.log(probs[current_indices[dim_idx]] + 1e-10)
        log_p_proposal = jnp.log(probs[proposal_indices[dim_idx]] + 1e-10)
        # Only count active dimensions
        mask_val = op.block_mask[dim_idx]
        return mask_val * (log_p_current - log_p_proposal)

    log_hastings_ratio = jnp.sum(jax.vmap(get_log_prob)(jnp.arange(block_size)))

    return proposal, log_hastings_ratio, new_key
