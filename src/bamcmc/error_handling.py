"""
Error Handling and Validation Utilities for MCMC Backend

This module provides validation functions and diagnostic tools for MCMC sampling.
"""

from typing import Any, Dict

import numpy as np

import logging
logger = logging.getLogger('bamcmc')


def validate_mcmc_config(mcmc_config: Dict[str, Any]) -> None:
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

    if 'num_superchains' in mcmc_config:
        if mcmc_config['num_superchains'] < 1:
            errors.append("num_superchains must be >= 1")

    # Parallel tempering validation
    if 'n_temperatures' in mcmc_config:
        n_temps = mcmc_config['n_temperatures']
        if n_temps < 1:
            errors.append("n_temperatures must be >= 1")

        # Check that we have enough chains for the requested temperatures
        total_chains = mcmc_config.get('num_chains_a', 0) + mcmc_config.get('num_chains_b', 0)
        if total_chains > 0 and n_temps > total_chains:
            errors.append(
                f"n_temperatures ({n_temps}) cannot exceed total chains ({total_chains})"
            )

    if 'beta_min' in mcmc_config:
        beta = mcmc_config['beta_min']
        if beta <= 0 or beta > 1:
            errors.append(f"beta_min must be in (0, 1], got {beta}")

    if errors:
        raise ValueError("Invalid MCMC configuration:\n  " + "\n  ".join(errors))


def diagnose_sampler_issues(history: np.ndarray, mcmc_config: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
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


def print_diagnostics(diagnostics: Dict[str, Any]) -> None:
    """Pretty-print diagnostics from diagnose_sampler_issues."""
    if diagnostics['issues']:
        logger.error("\n[ERROR] ISSUES:")
        for issue in diagnostics['issues']:
            logger.error(f"  - {issue}")

    if diagnostics['warnings']:
        logger.warning("\n[WARN] WARNINGS:")
        for warning in diagnostics['warnings']:
            logger.warning(f"  - {warning}")

    if diagnostics['info']:
        logger.info("\n[INFO] INFO:")
        for info in diagnostics['info']:
            logger.info(f"  - {info}")

    if not diagnostics['issues'] and not diagnostics['warnings']:
        logger.info("\n[OK] No issues detected")
