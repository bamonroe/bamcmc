"""
bamcmc - Bayesian MCMC Sampling Package

Public API:
    Registration:
        register_posterior - Register a posterior model
        get_posterior - Retrieve a registered posterior
        list_posteriors - List all registered posteriors

    Block Specifications:
        BlockSpec - Dataclass for parameter block configuration
        ProposalGroup - Dataclass for mixed-proposal sub-groups within a block
        SamplerType - Enum for sampler types (METROPOLIS_HASTINGS, DIRECT_CONJUGATE, etc.)
        ProposalType - Enum for proposal types (SELF_MEAN, CHAIN_MEAN, MULTINOMIAL, etc.)
        create_subject_blocks - Helper to create subject-level blocks
        create_mixed_subject_blocks - Helper for blocks with mixed proposal types
        MAX_PROPOSAL_GROUPS - Maximum number of proposal groups per block

    Settings:
        SettingSlot - IntEnum for proposal setting indices (ALPHA, N_CATEGORIES, etc.)

    Checkpointing & Batch Utilities:
        save_checkpoint - Save MCMC state to disk for resuming
        load_checkpoint - Load MCMC state from disk
        initialize_from_checkpoint - Initialize MCMC carry from loaded checkpoint
        combine_batch_histories - Combine multiple batch history files
        apply_burnin - Drop samples before a minimum iteration
        compute_rhat_from_history - Compute nested R-hat on combined history

    Chain Reset Utilities:
        generate_reset_states - Generate K new starting points from checkpoint
        generate_reset_vector - Generate full initial vector with K*M chains
        reset_from_checkpoint - High-level reset from checkpoint file
        print_reset_summary - Print diagnostic info before resetting
        select_diverse_states - Select K diverse states from checkpoint chains
        init_from_prior - Initialize posterior from prior-only samples

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
# CRITICAL: Import jax_config FIRST to set environment variables before JAX loads
from . import jax_config  # noqa: F401

# Import mcmc subpackage to register BlockArrays pytree
from . import mcmc as _mcmc  # noqa: F401

from .registry import register_posterior, get_posterior, list_posteriors
from .batch_specs import (
    BlockSpec,
    ProposalGroup,
    SamplerType,
    ProposalType,
    create_subject_blocks,
    create_mixed_subject_blocks,
    MAX_PROPOSAL_GROUPS,
)
from .settings import SettingSlot
from .mcmc.types import MCMCData
from .checkpoint_helpers import (
    save_checkpoint,
    load_checkpoint,
    initialize_from_checkpoint,
    combine_batch_histories,
    apply_burnin,
    compute_rhat_from_history,
    scan_checkpoints,
    get_latest_checkpoint,
    get_model_paths,
    ensure_model_dirs,
    clean_model_files,
    split_history_by_subject,
    postprocess_all_histories,
    save_prior_config,
    load_prior_config,
)
from .reset_utils import (
    generate_reset_states,
    generate_reset_vector,
    reset_from_checkpoint,
    print_reset_summary,
    select_diverse_states,
    init_from_prior,
)
from .posterior_benchmark import (
    PosteriorBenchmarkManager,
    get_manager as get_benchmark_manager,
    compute_posterior_hash,
    run_benchmark,
)

# Main MCMC entry points
from .mcmc import (
    rmcmc,
    rmcmc_single,
)
