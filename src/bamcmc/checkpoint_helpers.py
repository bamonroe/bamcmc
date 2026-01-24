"""
Checkpoint and batch history utilities for MCMC runs.

This module re-exports functions from submodules for backward compatibility.
For new code, prefer importing directly from the submodules:
    - checkpoint_io: save_checkpoint, load_checkpoint, initialize_from_checkpoint
    - history_processing: combine_batch_histories, apply_burnin, compute_rhat_from_history,
                          split_history_by_subject, postprocess_all_histories
    - output_management: scan_checkpoints, get_model_paths, ensure_model_dirs,
                         get_latest_checkpoint, clean_model_files
    - prior_config: save_prior_config, load_prior_config
"""

# Checkpoint I/O
from .checkpoint_io import (
    save_checkpoint,
    load_checkpoint,
    initialize_from_checkpoint,
)

# History processing
from .history_processing import (
    combine_batch_histories,
    apply_burnin,
    compute_rhat_from_history,
    split_history_by_subject,
    postprocess_all_histories,
)

# Output directory management
from .output_management import (
    scan_checkpoints,
    get_model_paths,
    ensure_model_dirs,
    get_latest_checkpoint,
    clean_model_files,
)

# Prior configuration
from .prior_config import (
    save_prior_config,
    load_prior_config,
)

__all__ = [
    # Checkpoint I/O
    'save_checkpoint',
    'load_checkpoint',
    'initialize_from_checkpoint',
    # History processing
    'combine_batch_histories',
    'apply_burnin',
    'compute_rhat_from_history',
    'split_history_by_subject',
    'postprocess_all_histories',
    # Output management
    'scan_checkpoints',
    'get_model_paths',
    'ensure_model_dirs',
    'get_latest_checkpoint',
    'clean_model_files',
    # Prior config
    'save_prior_config',
    'load_prior_config',
]
