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

    # Check required keys
    required_keys = ['POSTERIOR_ID', 'NUM_CHAINS_A', 'NUM_CHAINS_B',
                     'THIN_ITERATION', 'NUM_COLLECT']
    for key in required_keys:
        if key not in mcmc_config:
            errors.append(f"Missing required config key: '{key}'")

    # Check numeric values
    if 'NUM_CHAINS_A' in mcmc_config:
        if mcmc_config['NUM_CHAINS_A'] < 1:
            errors.append("NUM_CHAINS_A must be >= 1")

    if 'NUM_CHAINS_B' in mcmc_config:
        if mcmc_config['NUM_CHAINS_B'] < 1:
            errors.append("NUM_CHAINS_B must be >= 1")

    if 'THIN_ITERATION' in mcmc_config:
        if mcmc_config['THIN_ITERATION'] < 1:
            errors.append("THIN_ITERATION must be >= 1")

    if 'NUM_COLLECT' in mcmc_config:
        if mcmc_config['NUM_COLLECT'] < 0:
            errors.append("NUM_COLLECT must be >= 0")

    if 'BURN_ITER' in mcmc_config:
        if mcmc_config['BURN_ITER'] < 0:
            errors.append("BURN_ITER must be >= 0")

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

    # Check for multimodality (looking at chain means)
    if history.shape[1] > 1:  # Multiple chains
        chain_means = np.mean(history, axis=0)
        param_stds = np.std(chain_means, axis=0)
        within_chain_stds = np.mean(np.std(history, axis=0), axis=0)

        # If between-chain variance >> within-chain variance, possible multimodality
        ratios = param_stds / (within_chain_stds + 1e-10)
        multimodal_params = np.where(ratios > 3.0)[0]

        if len(multimodal_params) > 0:
            diagnostics['warnings'].append(
                f"Possible multimodality detected in parameter(s): {multimodal_params.tolist()}"
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
