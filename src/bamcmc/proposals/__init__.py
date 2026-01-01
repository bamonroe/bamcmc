"""
Proposal Distributions for MCMC Sampling

This package implements proposal distributions for Metropolis-Hastings sampling.
ProposalType enum is defined in batch_specs.py.

To add a new proposal:
1. Add enum value to ProposalType in batch_specs.py
2. Create new file in proposals/ directory with proposal function
3. Add to PROPOSAL_REGISTRY in mcmc/sampling.py
4. Export from this __init__.py

Each proposal function computes its own Hastings ratio - there's no separate
symmetric/asymmetric handling needed in the backend.

Continuous proposals (self_mean, chain_mean, mixture) receive precomputed
mean and covariance to avoid redundant computation across chains.
Discrete proposals (multinomial) receive raw coupled_blocks for counting.
Gradient-based proposals (mala) use the grad_fn for computing proposal drift.

All proposal functions accept a single operand tuple:
    (key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn)

Settings are passed as a JAX array with values accessed by position using SettingSlot.
The grad_fn is a function that maps block values to gradients of log posterior.
Non-MALA proposals ignore grad_fn; MALA uses it for Langevin drift computation.

Note: The dispatch table is built dynamically in sampling.py based on which
proposals are actually used by the model. This ensures JAX only traces the
proposals needed, avoiding memory overhead from unused proposals.
"""

from .self_mean import self_mean_proposal
from .chain_mean import chain_mean_proposal
from .mixture import mixture_proposal
from .multinomial import multinomial_proposal
from .mala import mala_proposal

__all__ = [
    'self_mean_proposal',
    'chain_mean_proposal',
    'mixture_proposal',
    'multinomial_proposal',
    'mala_proposal',
]
