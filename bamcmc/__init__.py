"""
bamcmc - Bayesian MCMC Sampling Package

Public API:
    Registration:
        register_posterior - Register a posterior model
        get_posterior - Retrieve a registered posterior
        list_posteriors - List all registered posteriors

    Block Specifications:
        BlockSpec - Dataclass for parameter block configuration
        SamplerType - Enum for sampler types (METROPOLIS_HASTINGS, DIRECT_CONJUGATE, etc.)
        ProposalType - Enum for proposal types (SELF_MEAN, CHAIN_MEAN)
        create_subject_blocks - Helper to create subject-level blocks

    Settings:
        SettingSlot - IntEnum for proposal setting indices (ALPHA, N_CATEGORIES, etc.)

    Checkpointing & Batch Utilities:
        save_checkpoint - Save MCMC state to disk for resuming
        load_checkpoint - Load MCMC state from disk
        combine_batch_histories - Combine multiple batch history files
        apply_burnin - Drop samples before a minimum iteration
        compute_rhat_from_history - Compute nested R-hat on combined history

    Chain Reset Utilities:
        generate_reset_states - Generate K new starting points from checkpoint
        generate_reset_vector - Generate full initial vector with K*M chains
        reset_from_checkpoint - High-level reset from checkpoint file
        print_reset_summary - Print diagnostic info before resetting

Example:
    from bamcmc import register_posterior, BlockSpec, SamplerType

    register_posterior('my_model', {
        'log_posterior': my_log_posterior,
        'batch_type': lambda cfg, data: [BlockSpec(size=2, sampler_type=SamplerType.METROPOLIS_HASTINGS)],
        'initial_vector': my_init_fn,
    })

    # To reset stuck chains:
    from bamcmc import reset_from_checkpoint
    init_vector, info = reset_from_checkpoint('checkpoint.npz', 'my_model', n_subjects=100, K=50, M=20)
"""

from .registry import register_posterior, get_posterior, list_posteriors
from .batch_specs import BlockSpec, SamplerType, ProposalType, create_subject_blocks
from .settings import SettingSlot
from .checkpoint_helpers import (
    save_checkpoint,
    load_checkpoint,
    combine_batch_histories,
    apply_burnin,
    compute_rhat_from_history,
)
from .reset_utils import (
    generate_reset_states,
    generate_reset_vector,
    reset_from_checkpoint,
    print_reset_summary,
)
