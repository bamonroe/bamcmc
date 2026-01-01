"""
Error Handling and Validation Utilities for MCMC Backend

This module provides validation functions and diagnostic tools for MCMC sampling.
"""

import numpy as np


def validate_mcmc_config(mcmc_config):
    """
    Validates that MCMC configuration is sensible.

    Args:
        mcmc_config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    errors = []

    # Check required keys (all lowercase)
    required_keys = ['posterior_id', 'num_chains_a', 'num_chains_b',
                     'thin_iteration', 'num_collect']
    for key in required_keys:
        if key not in mcmc_config:
            errors.append(f"Missing required config key: '{key}'")

    # Check numeric values
    if 'num_chains_a' in mcmc_config:
        if mcmc_config['num_chains_a'] < 1:
            errors.append("num_chains_a must be >= 1")

    if 'num_chains_b' in mcmc_config:
        if mcmc_config['num_chains_b'] < 1:
            errors.append("num_chains_b must be >= 1")

    if 'thin_iteration' in mcmc_config:
        if mcmc_config['thin_iteration'] < 1:
            errors.append("thin_iteration must be >= 1")

    if 'num_collect' in mcmc_config:
        if mcmc_config['num_collect'] < 0:
            errors.append("num_collect must be >= 0")

    if 'burn_iter' in mcmc_config:
        if mcmc_config['burn_iter'] < 0:
            errors.append("burn_iter must be >= 0")

    if errors:
        raise ValueError("Invalid MCMC configuration:\n  " + "\n  ".join(errors))


def diagnose_sampler_issues(history, mcmc_config, diagnostics):
    """
    Analyzes MCMC history to identify common issues.

    Args:
        history: MCMC history array (n_samples, n_chains, n_params)
        mcmc_config: Configuration dict
        diagnostics: Existing diagnostics dict to extend

    Returns:
        diagnostics: Dictionary with issues, warnings, and info
    """
    diagnostics = diagnostics | {
        'issues': [],
        'warnings': [],
        'info': []
    }

    # Check for NaN/Inf in history
    if not np.all(np.isfinite(history)):
        diagnostics['issues'].append(
            "History contains NaN or Inf values - sampler became unstable"
        )

    # Check for stuck chains (variance near zero)
    chain_vars = np.var(history, axis=0)
    stuck_chains = np.sum(np.all(chain_vars < 1e-10, axis=1))
    if stuck_chains > 0:
        diagnostics['warnings'].append(
            f"{stuck_chains} chain(s) appear stuck (near-zero variance)"
        )

    # Summary info
    diagnostics['info'].append(f"Total samples: {history.shape[0] * history.shape[1]}")
    diagnostics['info'].append(f"Number of chains: {history.shape[1]}")
    diagnostics['info'].append(f"Number of parameters: {history.shape[2]}")

    return diagnostics


def print_diagnostics(diagnostics):
    """Pretty-print diagnostics from diagnose_sampler_issues."""
    if diagnostics['issues']:
        print("\n🔴 ISSUES:")
        for issue in diagnostics['issues']:
            print(f"  - {issue}")

    if diagnostics['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in diagnostics['warnings']:
            print(f"  - {warning}")

    if diagnostics['info']:
        print("\n📊 INFO:")
        for info in diagnostics['info']:
            print(f"  - {info}")

    if not diagnostics['issues'] and not diagnostics['warnings']:
        print("\n✅ No issues detected")
