"""
Static Dispatch Table for Proposal Functions

Provides a static dispatch table for use with jax.lax.switch in the MCMC backend.
The table is created once at import time to avoid repeated closure creation during tracing.
"""

from .self_mean import self_mean_proposal
from .chain_mean import chain_mean_proposal
from .mixture import mixture_proposal
from .multinomial import multinomial_proposal


# Static dispatch table - created once at import time
# ORDER MATTERS: Index must match ProposalType enum value!
# SELF_MEAN = 0, CHAIN_MEAN = 1, MIXTURE = 2, MULTINOMIAL = 3
PROPOSAL_DISPATCH_TABLE = [
    self_mean_proposal,      # 0 = ProposalType.SELF_MEAN
    chain_mean_proposal,     # 1 = ProposalType.CHAIN_MEAN
    mixture_proposal,        # 2 = ProposalType.MIXTURE
    multinomial_proposal,    # 3 = ProposalType.MULTINOMIAL
]
