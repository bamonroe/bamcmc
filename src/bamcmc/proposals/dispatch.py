"""
Proposal Dispatch (Legacy)

NOTE: The static dispatch table is no longer used. The dispatch table is now
created dynamically in sampling.py to capture grad_fn via closure, enabling
a unified interface where all proposals receive grad_fn (used by MALA, ignored
by other proposals).

This file is kept for backwards compatibility but the table is not used.
"""

from .self_mean import self_mean_proposal
from .chain_mean import chain_mean_proposal
from .mixture import mixture_proposal
from .multinomial import multinomial_proposal
from .mala import mala_proposal


# Legacy dispatch table - kept for reference but not used
# The actual dispatch table is created dynamically in sampling.py
# to capture grad_fn via closure for the unified proposal interface
PROPOSAL_DISPATCH_TABLE = [
    self_mean_proposal,      # 0 = ProposalType.SELF_MEAN
    chain_mean_proposal,     # 1 = ProposalType.CHAIN_MEAN
    mixture_proposal,        # 2 = ProposalType.MIXTURE
    multinomial_proposal,    # 3 = ProposalType.MULTINOMIAL
    mala_proposal,           # 4 = ProposalType.MALA
]
