"""
MCMC Subpackage - Core MCMC sampling implementation.

This package contains the core MCMC sampling logic:
- backend: Main entry points (rmcmc, rmcmc_single)
- compile: Kernel compilation and caching
- config: Configuration and initialization
- diagnostics: Convergence diagnostics (R-hat)
- sampling: Proposal and MH sampling functions
- scan: JAX scan body and block statistics
- types: Core data structures (BlockArrays, RunParams)
- utils: Miscellaneous utilities
"""

# Import types first (needed by other modules)
from .types import BlockArrays, RunParams, build_block_arrays

# Import main entry points
from .backend import rmcmc, rmcmc_single

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
    # Compile
    'compile_mcmc_kernel',
    'benchmark_mcmc_sampler',
]
