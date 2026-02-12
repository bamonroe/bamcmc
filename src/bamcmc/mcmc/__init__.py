"""
MCMC Subpackage - Core MCMC sampling implementation.

This package contains the core MCMC sampling logic:
- backend: Multi-run orchestrator (rmcmc)
- single_run: Single-run engine (rmcmc_single) and helpers
- compile: Kernel compilation and caching
- config: Configuration and initialization
- diagnostics: Convergence diagnostics (R-hat) and tempering utilities
- tempering: Parallel tempering temperature swap logic
- sampling: Proposal and MH sampling functions
- scan: JAX scan body and block statistics
- types: Core data structures (BlockArrays, RunParams)
- utils: Miscellaneous utilities
"""

# Import types first (needed by other modules)
from .types import BlockArrays, RunParams, build_block_arrays

# Import main entry points
from .backend import rmcmc
from .single_run import rmcmc_single

# Import commonly used functions
from .config import (
    configure_mcmc_system,
    initialize_mcmc_system,
    validate_mcmc_inputs,
)
from .diagnostics import (
    compute_nested_rhat,
    compute_and_print_rhat,
    print_acceptance_summary,
    print_swap_acceptance_summary,
    filter_beta1_samples,
    compute_round_trip_rate,
    print_round_trip_summary,
)
from .compile import compile_mcmc_kernel, benchmark_mcmc_sampler

__all__ = [
    # Main entry points
    'rmcmc',
    'rmcmc_single',
    # Types
    'BlockArrays',
    'RunParams',
    'build_block_arrays',
    # Config
    'configure_mcmc_system',
    'initialize_mcmc_system',
    'validate_mcmc_inputs',
    # Diagnostics
    'compute_nested_rhat',
    'compute_and_print_rhat',
    'print_acceptance_summary',
    'print_swap_acceptance_summary',
    'filter_beta1_samples',
    'compute_round_trip_rate',
    'print_round_trip_summary',
    # Compile
    'compile_mcmc_kernel',
    'benchmark_mcmc_sampler',
]
