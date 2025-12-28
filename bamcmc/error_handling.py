"""
Error Handling and Validation Utilities for MCMC Backend

This module provides safety checks and validation functions to catch common
issues early and provide helpful error messages.

These checks can be disabled in production for maximum performance by setting
MCMC_SAFETY_CHECKS=False in the config.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import wraps


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default safety check level
# 0 = No checks (maximum performance)
# 1 = Critical checks only (recommended for production)
# 2 = All checks including expensive ones (recommended for development)
DEFAULT_SAFETY_LEVEL = 1


# ============================================================================
# MATRIX VALIDATION
# ============================================================================

def check_covariance_matrix(cov, name="covariance", nugget=1e-5, min_eigenvalue=None):
    """
    Validates that a covariance matrix is suitable for sampling.

    Args:
        cov: Covariance matrix to check
        name: Name for error messages
        nugget: Regularization amount to suggest if matrix is singular
        min_eigenvalue: Minimum acceptable eigenvalue (default: nugget)

    Returns:
        is_valid: Boolean indicating if matrix is valid
        message: Error message if invalid, empty string if valid
    """
    if min_eigenvalue is None:
        min_eigenvalue = nugget

    cov_np = np.array(cov) if hasattr(cov, 'block_until_ready') else cov

    # Check 1: Shape
    if len(cov_np.shape) != 2:
        return False, f"{name} must be 2D, got shape {cov_np.shape}"

    if cov_np.shape[0] != cov_np.shape[1]:
        return False, f"{name} must be square, got shape {cov_np.shape}"

    # Check 2: Symmetry
    max_asymmetry = np.max(np.abs(cov_np - cov_np.T))
    if max_asymmetry > 1e-8:
        return False, f"{name} is not symmetric (max asymmetry: {max_asymmetry:.2e})"

    # Check 3: Positive definiteness via eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(cov_np)
        min_eig = np.min(eigenvalues)

        if min_eig < min_eigenvalue:
            return False, (
                f"{name} is not positive definite. "
                f"Minimum eigenvalue: {min_eig:.2e}, threshold: {min_eigenvalue:.2e}. "
                f"Consider increasing NUGGET (current: {nugget:.2e}) or checking for "
                f"degenerate parameters."
            )

        # Check condition number
        max_eig = np.max(eigenvalues)
        condition_number = max_eig / min_eig if min_eig > 0 else np.inf

        if condition_number > 1e10:
            return False, (
                f"{name} is poorly conditioned. "
                f"Condition number: {condition_number:.2e}. "
                f"This may cause numerical instability. "
                f"Consider reparameterizing your model or increasing regularization."
            )

    except np.linalg.LinAlgError as e:
        return False, f"Failed to compute eigenvalues of {name}: {str(e)}"

    return True, ""


def safe_cholesky(cov, nugget=1e-5, name="covariance"):
    """
    Computes Cholesky decomposition with safety checks and helpful errors.

    Args:
        cov: Covariance matrix
        nugget: Regularization to add if decomposition fails
        name: Name for error messages

    Returns:
        L: Lower triangular Cholesky factor

    Raises:
        ValueError: If matrix is invalid even after regularization
    """
    # First attempt
    try:
        L = jnp.linalg.cholesky(cov)
        return L
    except:
        pass

    # Validation check
    is_valid, message = check_covariance_matrix(cov, name, nugget)
    if not is_valid:
        raise ValueError(f"Cholesky decomposition failed for {name}. {message}")

    # Try with additional regularization
    eye = jnp.eye(cov.shape[0])
    cov_reg = cov + eye * (nugget * 10)

    try:
        L = jnp.linalg.cholesky(cov_reg)
        print(f"Warning: Added extra regularization to {name} (10x NUGGET)")
        return L
    except:
        raise ValueError(
            f"Cholesky decomposition failed for {name} even after regularization. "
            f"Matrix may be severely ill-conditioned. Consider reparameterizing."
        )


# ============================================================================
# LOG PROBABILITY VALIDATION
# ============================================================================

def check_log_probability(lp, name="log_posterior", allow_neginf=True):
    """
    Validates that a log probability is finite and reasonable.

    Args:
        lp: Log probability value
        name: Name for error messages
        allow_neginf: If True, -inf is acceptable (e.g., for rejecting invalid parameters)

    Returns:
        is_valid: Boolean
        message: Error message if invalid
    """
    lp_val = float(lp) if hasattr(lp, 'block_until_ready') else lp

    if np.isnan(lp_val):
        return False, (
            f"{name} returned NaN. "
            f"Check for: (1) log of negative numbers, (2) 0/0, (3) inf - inf"
        )

    if np.isposinf(lp_val):
        return False, (
            f"{name} returned +inf. "
            f"This usually indicates a bug in the posterior calculation."
        )

    if np.isneginf(lp_val) and not allow_neginf:
        return False, (
            f"{name} returned -inf. "
            f"This may indicate: (1) parameter out of support, (2) numerical underflow"
        )

    # Check for extremely large values that might cause overflow
    if np.abs(lp_val) > 1e10 and not np.isinf(lp_val):
        return False, (
            f"{name} returned very large value: {lp_val:.2e}. "
            f"This may cause numerical instability. "
            f"Consider rescaling your likelihood or using better numerical practices."
        )

    return True, ""


# ============================================================================
# PARAMETER STATE VALIDATION
# ============================================================================

def check_parameter_state(state, name="chain_state"):
    """
    Validates that parameter state array is finite.

    Args:
        state: Parameter state array
        name: Name for error messages

    Returns:
        is_valid: Boolean
        message: Error message if invalid
    """
    state_np = np.array(state) if hasattr(state, 'block_until_ready') else state

    if not np.all(np.isfinite(state_np)):
        n_nan = np.sum(np.isnan(state_np))
        n_inf = np.sum(np.isinf(state_np))

        # Find first invalid index
        invalid_idx = np.where(~np.isfinite(state_np))[0]
        first_invalid = invalid_idx[0] if len(invalid_idx) > 0 else -1

        return False, (
            f"{name} contains non-finite values: "
            f"{n_nan} NaN(s), {n_inf} Inf(s). "
            f"First invalid at index {first_invalid}. "
            f"This indicates numerical instability in the sampler or posterior."
        )

    return True, ""


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

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

    # Check proposal type
    if 'PROPOSAL' in mcmc_config:
        valid_proposals = ['self_mean', 'chain_mean']
        if mcmc_config['PROPOSAL'] not in valid_proposals:
            errors.append(
                f"PROPOSAL must be one of {valid_proposals}, "
                f"got '{mcmc_config['PROPOSAL']}'"
            )

    if errors:
        raise ValueError("Invalid MCMC configuration:\n  " + "\n  ".join(errors))


def validate_batch_specs(batch_specs, model_name=""):
    """
    Validates batch specifications from a posterior model.

    Args:
        batch_specs: List of (size, type) tuples
        model_name: Name of model for error messages

    Raises:
        ValueError: If batch specs are invalid
    """
    errors = []

    if not isinstance(batch_specs, (list, tuple)):
        errors.append("batch_specs must be a list or tuple")
        raise ValueError("\n  ".join(errors))

    for i, spec in enumerate(batch_specs):
        if not isinstance(spec, (list, tuple)):
            errors.append(f"Batch spec {i} must be a tuple, got {type(spec)}")
            continue

        if len(spec) != 2:
            errors.append(
                f"Batch spec {i} must have 2 elements (size, type), got {len(spec)}"
            )
            continue

        size, btype = spec

        if not isinstance(size, (int, np.integer)):
            errors.append(f"Batch spec {i}: size must be an integer, got {type(size)}")
        elif size < 1:
            errors.append(f"Batch spec {i}: size must be >= 1, got {size}")

        if not isinstance(btype, (int, np.integer)):
            errors.append(f"Batch spec {i}: type must be an integer, got {type(btype)}")
        elif btype not in [0, 1]:
            errors.append(
                f"Batch spec {i}: type must be 0 (MH) or 1 (Direct), got {btype}"
            )

    if errors:
        prefix = f"Invalid batch specs for model '{model_name}'" if model_name else "Invalid batch specs"
        raise ValueError(prefix + ":\n  " + "\n  ".join(errors))


# ============================================================================
# DECORATORS FOR OPTIONAL CHECKING
# ============================================================================

def with_safety_checks(safety_level=1):
    """
    Decorator that adds safety checks to a function based on config.

    Usage:
        @with_safety_checks(safety_level=1)
        def my_function(mcmc_config, ...):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract mcmc_config if present (usually first arg)
            mcmc_config = None
            if len(args) > 0 and isinstance(args[0], dict):
                mcmc_config = args[0]
            elif 'mcmc_config' in kwargs:
                mcmc_config = kwargs['mcmc_config']

            # Check if safety checks are enabled
            if mcmc_config:
                check_level = mcmc_config.get('SAFETY_CHECKS', DEFAULT_SAFETY_LEVEL)
                if check_level < safety_level:
                    # Skip checks, run function directly
                    return func(*args, **kwargs)

            # Run function with exception handling
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Enhance error message with context
                import traceback
                tb = traceback.format_exc()
                enhanced_msg = (
                    f"Error in {func.__name__}: {str(e)}\n"
                    f"Consider enabling SAFETY_CHECKS=2 for detailed diagnostics.\n"
                    f"Traceback:\n{tb}"
                )
                raise type(e)(enhanced_msg) from e

        return wrapper
    return decorator


# ============================================================================
# RUNTIME DIAGNOSTICS
# ============================================================================

def diagnose_sampler_issues(history, mcmc_config, diagnostics):
    """
    Analyzes MCMC history to identify common issues.

    Args:
        history: MCMC history array
        mcmc_config: Configuration dict

    Returns:
        diagnostics: Dictionary of diagnostic results
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

    # Check acceptance rate (if likelihood history available)
    # This would require tracking acceptances, which we'd add separately

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
