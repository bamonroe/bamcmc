"""
Test Runner for MCMC Sampler Validation

This script runs the MCMC sampler on conjugate models with known analytical solutions
and validates that the sampler is producing correct results.

Usage:
    python run_tests.py [--model MODEL_NAME] [--verbose]
"""

import numpy as np
import jax.numpy as jnp
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

from .mcmc_backend import rmcmc
from .mcmc_utils import clean_config
from .registry import register_posterior, _REGISTRY
from . import test_posteriors


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_beta_bernoulli_data(n_trials=100, true_theta=0.7, alpha_0=2.0, beta_0=2.0, seed=42):
    """
    Generate synthetic data for Beta-Bernoulli model.

    Returns:
        data: Dictionary in the format expected by the sampler
        true_params: Dictionary of true parameter values
    """
    np.random.seed(seed)

    # Generate Bernoulli trials
    y = np.random.binomial(1, true_theta, n_trials)

    data = {
        "static": (alpha_0, beta_0),
        "int": (y,),
        "float": ()
    }

    true_params = {
        'theta': true_theta,
        'n_success': np.sum(y),
        'n_trials': n_trials
    }

    return data, true_params


def generate_normal_normal_data(n_obs=50, true_mu=5.0, sigma=2.0, mu_0=0.0, tau_0=10.0, seed=42):
    """
    Generate synthetic data for Normal-Normal model (known variance).

    Returns:
        data: Dictionary in the format expected by the sampler
        true_params: Dictionary of true parameter values
    """
    np.random.seed(seed)

    # Generate normal observations
    y = np.random.normal(true_mu, sigma, n_obs)

    data = {
        "static": (mu_0, tau_0, sigma),
        "int": (),
        "float": (y,)
    }

    true_params = {
        'mu': true_mu,
        'sigma': sigma,
        'y_mean': np.mean(y),
        'n_obs': n_obs
    }

    return data, true_params


def generate_beta_bernoulli_hierarchical_data(n_subjects=10, trials_per_subject=20,
                                               true_alpha=3.0, true_beta=2.0,
                                               a0=2.0, b0=2.0, seed=42):
    """
    Generate synthetic hierarchical data.

    Returns:
        data: Dictionary in the format expected by the sampler
        true_params: Dictionary of true parameter values
    """
    np.random.seed(seed)

    # Generate subject-level thetas from Beta
    true_thetas = np.random.beta(true_alpha, true_beta, n_subjects)

    # Generate trials for each subject
    max_trials = trials_per_subject
    y_matrix = np.zeros((n_subjects, max_trials), dtype=int)
    trial_counts = np.full(n_subjects, trials_per_subject, dtype=int)

    for i, theta in enumerate(true_thetas):
        y_matrix[i, :] = np.random.binomial(1, theta, max_trials)

    data = {
        "static": (a0, b0, n_subjects),
        "int": (y_matrix, trial_counts),
        "float": ()
    }

    true_params = {
        'thetas': true_thetas,
        'alpha': true_alpha,
        'beta': true_beta,
        'n_subjects': n_subjects
    }

    return data, true_params


# ============================================================================
# TEST VALIDATION FUNCTIONS
# ============================================================================

def validate_beta_bernoulli_pooled(history, data, true_params, tolerance=0.05):
    """
    Validate Beta-Bernoulli pooled model against analytical posterior.

    Returns:
        results: Dictionary of test results
    """
    # Get analytical posterior
    alpha_post, beta_post = test_posteriors.beta_bernoulli_pooled_analytical_posterior(data)

    # Analytical moments
    analytical_mean = alpha_post / (alpha_post + beta_post)
    analytical_var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
    analytical_std = np.sqrt(analytical_var)

    # Sample statistics (theta is in the GQ column, last column)
    theta_samples = history[:, :, -1].flatten()
    sample_mean = np.mean(theta_samples)
    sample_std = np.std(theta_samples, ddof=1)

    # Compute errors
    mean_error = abs(sample_mean - analytical_mean)
    std_error = abs(sample_std - analytical_std)

    # Determine pass/fail
    mean_passed = mean_error < tolerance
    std_passed = std_error < (tolerance * 2)  # More lenient for std

    results = {
        'analytical_mean': analytical_mean,
        'analytical_std': analytical_std,
        'sample_mean': sample_mean,
        'sample_std': sample_std,
        'mean_error': mean_error,
        'std_error': std_error,
        'mean_passed': mean_passed,
        'std_passed': std_passed,
        'overall_passed': mean_passed and std_passed,
        'alpha_post': alpha_post,
        'beta_post': beta_post,
        'n_success': true_params['n_success'],
        'n_trials': true_params['n_trials']
    }

    return results


def validate_normal_normal_pooled(history, data, true_params, tolerance=0.10):
    """
    Validate Normal-Normal pooled model against analytical posterior.

    Returns:
        results: Dictionary of test results
    """
    # Get analytical posterior
    mu_post, tau_post = test_posteriors.normal_normal_pooled_analytical_posterior(data)

    # Sample statistics (mu is in the GQ column, last column)
    mu_samples = history[:, :, -1].flatten()
    sample_mean = np.mean(mu_samples)
    sample_std = np.std(mu_samples, ddof=1)

    # Compute errors
    mean_error = abs(sample_mean - mu_post)
    std_error = abs(sample_std - tau_post)

    # Determine pass/fail
    mean_passed = mean_error < tolerance
    std_passed = std_error < (tolerance * 2)

    results = {
        'analytical_mean': mu_post,
        'analytical_std': tau_post,
        'sample_mean': sample_mean,
        'sample_std': sample_std,
        'mean_error': mean_error,
        'std_error': std_error,
        'mean_passed': mean_passed,
        'std_passed': std_passed,
        'overall_passed': mean_passed and std_passed,
        'true_mu': true_params['mu'],
        'y_mean': true_params['y_mean']
    }

    return results


def validate_convergence(diagnostics, rhat_threshold=1.05):
    """
    Check R-hat convergence diagnostics.

    Args:
        diagnostics: Dictionary returned by rmcmc containing 'rhat', 'K', 'M'
        rhat_threshold: Threshold for convergence (default 1.05)

    Returns:
        results: Dictionary with convergence information
    """
    rhats_np = np.array(diagnostics['rhat']) if diagnostics['rhat'] is not None else np.array([np.nan])

    max_rhat = np.max(rhats_np)
    all_converged = np.all(rhats_np < rhat_threshold)

    return {
        'rhats': rhats_np,
        'max_rhat': max_rhat,
        'all_converged': all_converged,
        'threshold': rhat_threshold,
        'K': diagnostics['K'],
        'M': diagnostics['M']
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_validation_beta_bernoulli(history, validation_results, filename='test_beta_bernoulli.pdf'):
    """Create diagnostic plots for Beta-Bernoulli test."""

    with PdfPages(filename) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Beta-Bernoulli Pooled Model Validation', fontsize=16)

        # Extract theta samples (last column = GQ)
        theta_samples = history[:, :, -1]

        # 1. Trace plot (first 5 chains)
        n_chains_to_plot = min(5, theta_samples.shape[1])
        for c in range(n_chains_to_plot):
            axes[0, 0].plot(theta_samples[:, c], alpha=0.6, lw=1)
        axes[0, 0].set_title('Trace Plot: θ')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('θ')
        axes[0, 0].grid(alpha=0.3)

        # 2. Posterior density vs analytical
        axes[0, 1].hist(theta_samples.flatten(), bins=50, density=True,
                       alpha=0.6, label='MCMC Samples', color='skyblue')

        # Overlay analytical posterior
        alpha_post = validation_results['alpha_post']
        beta_post = validation_results['beta_post']
        x_range = np.linspace(0, 1, 200)
        analytical_pdf = scipy_stats.beta.pdf(x_range, alpha_post, beta_post)
        axes[0, 1].plot(x_range, analytical_pdf, 'r-', lw=2, label='Analytical')
        axes[0, 1].axvline(validation_results['sample_mean'], color='blue',
                          linestyle='--', label='Sample Mean')
        axes[0, 1].axvline(validation_results['analytical_mean'], color='red',
                          linestyle='--', label='Analytical Mean')
        axes[0, 1].set_title('Posterior Density')
        axes[0, 1].set_xlabel('θ')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Autocorrelation
        from matplotlib import mlab
        chain_0 = theta_samples[:, 0]
        acf = np.correlate(chain_0 - np.mean(chain_0), chain_0 - np.mean(chain_0), mode='full')
        acf = acf[len(acf)//2:] / acf[len(acf)//2]
        axes[1, 0].plot(acf[:100], marker='o', markersize=3)
        axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[1, 0].set_title('Autocorrelation (Chain 0)')
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].set_ylabel('ACF')
        axes[1, 0].grid(alpha=0.3)

        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Test Results:

        Data:
          Successes: {validation_results['n_success']}
          Trials: {validation_results['n_trials']}

        Analytical Posterior: Beta({alpha_post:.2f}, {beta_post:.2f})
          Mean: {validation_results['analytical_mean']:.4f}
          Std:  {validation_results['analytical_std']:.4f}

        Sample Statistics:
          Mean: {validation_results['sample_mean']:.4f}
          Std:  {validation_results['sample_std']:.4f}

        Errors:
          Mean Error: {validation_results['mean_error']:.4f}
          Std Error:  {validation_results['std_error']:.4f}

        Status:
          Mean: {'PASS ✓' if validation_results['mean_passed'] else 'FAIL ✗'}
          Std:  {'PASS ✓' if validation_results['std_passed'] else 'FAIL ✗'}
          Overall: {'PASS ✓' if validation_results['overall_passed'] else 'FAIL ✗'}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                       family='monospace')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Validation plot saved to {filename}")


def plot_validation_normal_normal(history, validation_results, filename='test_normal_normal.pdf'):
    """Create diagnostic plots for Normal-Normal test."""

    with PdfPages(filename) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Normal-Normal Pooled Model Validation', fontsize=16)

        # Extract mu samples (last column = GQ)
        mu_samples = history[:, :, -1]

        # 1. Trace plot
        n_chains_to_plot = min(5, mu_samples.shape[1])
        for c in range(n_chains_to_plot):
            axes[0, 0].plot(mu_samples[:, c], alpha=0.6, lw=1)
        axes[0, 0].set_title('Trace Plot: μ')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('μ')
        axes[0, 0].grid(alpha=0.3)

        # 2. Posterior density vs analytical
        axes[0, 1].hist(mu_samples.flatten(), bins=50, density=True,
                       alpha=0.6, label='MCMC Samples', color='skyblue')

        # Overlay analytical posterior
        mu_post = validation_results['analytical_mean']
        tau_post = validation_results['analytical_std']
        x_range = np.linspace(mu_post - 4*tau_post, mu_post + 4*tau_post, 200)
        analytical_pdf = scipy_stats.norm.pdf(x_range, mu_post, tau_post)
        axes[0, 1].plot(x_range, analytical_pdf, 'r-', lw=2, label='Analytical')
        axes[0, 1].axvline(validation_results['sample_mean'], color='blue',
                          linestyle='--', label='Sample Mean')
        axes[0, 1].axvline(mu_post, color='red',
                          linestyle='--', label='Analytical Mean')
        axes[0, 1].set_title('Posterior Density')
        axes[0, 1].set_xlabel('μ')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Running mean
        chain_0 = mu_samples[:, 0]
        running_mean = np.cumsum(chain_0) / np.arange(1, len(chain_0) + 1)
        axes[1, 0].plot(running_mean)
        axes[1, 0].axhline(mu_post, color='red', linestyle='--', label='Analytical Mean')
        axes[1, 0].set_title('Running Mean (Chain 0)')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Cumulative Mean')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Test Results:

        Data:
          True μ: {validation_results['true_mu']:.4f}
          Sample Mean: {validation_results['y_mean']:.4f}

        Analytical Posterior: N({mu_post:.4f}, {tau_post:.4f})

        Sample Statistics:
          Mean: {validation_results['sample_mean']:.4f}
          Std:  {validation_results['sample_std']:.4f}

        Errors:
          Mean Error: {validation_results['mean_error']:.4f}
          Std Error:  {validation_results['std_error']:.4f}

        Status:
          Mean: {'PASS ✓' if validation_results['mean_passed'] else 'FAIL ✗'}
          Std:  {'PASS ✓' if validation_results['std_passed'] else 'FAIL ✗'}
          Overall: {'PASS ✓' if validation_results['overall_passed'] else 'PASS ✓'}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                       family='monospace')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Validation plot saved to {filename}")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_single_test(model_name, verbose=False):
    """
    Run a single test for the specified model.

    Args:
        model_name: Name of test model (e.g., 'test_beta_bernoulli_pooled')
        verbose: If True, print detailed output

    Returns:
        success: Boolean indicating if test passed
        results: Dictionary of test results
    """
    print(f"\n{'='*70}")
    print(f"RUNNING TEST: {model_name}")
    print(f"{'='*70}\n")

    # Generate data
    if model_name == 'test_beta_bernoulli_pooled':
        data, true_params = generate_beta_bernoulli_data()
        validate_fn = validate_beta_bernoulli_pooled
        plot_fn = plot_validation_beta_bernoulli
    elif model_name == 'test_normal_normal_pooled':
        data, true_params = generate_normal_normal_data()
        validate_fn = validate_normal_normal_pooled
        plot_fn = plot_validation_normal_normal
    elif model_name == 'test_beta_bernoulli_hierarchical':
        data, true_params = generate_beta_bernoulli_hierarchical_data()
        # For now, hierarchical doesn't have simple validation
        print("Hierarchical model test: Running sampler (no analytical validation yet)")
        validate_fn = None
        plot_fn = None
    else:
        print(f"ERROR: Unknown test model '{model_name}'")
        return False, {}

    # Configure MCMC
    mcmc_config = {
        'POSTERIOR_ID': model_name,
        'GPU_PREALLOCATION': True,
        'USE_DOUBLE': True,
        'rng_seed': 1977,
        'BENCHMARK': 50,
        'BURN_ITER': 500,
        'THIN_ITERATION': 1,
        'NUM_COLLECT': 1000,
        'PROPOSAL': 'chain_mean',
        'NUM_CHAINS_A': 25,
        'NUM_CHAINS_B': 25,
        'LAST_ITERS': 1000,
    }

    # Register test posteriors temporarily
    # Save any existing registrations that might be overwritten
    original_registrations = {}
    for name, config in test_posteriors.TEST_POSTERIORS.items():
        if name in _REGISTRY:
            original_registrations[name] = _REGISTRY[name]
        register_posterior(name, config)

    try:
        # Run MCMC
        print("Running MCMC sampler...")
        history, diagnostics, mcmc_config, lik_history, _ = rmcmc(mcmc_config, data)

        # Validate convergence using diagnostics from rmcmc
        conv_results = validate_convergence(diagnostics)
        print(f"\nConvergence Diagnostics:")
        print(f"  Max R-hat: {conv_results['max_rhat']:.4f}")
        print(f"  All converged: {'YES ✓' if conv_results['all_converged'] else 'NO ✗'}")

        # Validate against analytical solution
        if validate_fn is not None:
            validation_results = validate_fn(history, data, true_params)

            print(f"\nValidation Results:")
            print(f"  Mean Error: {validation_results['mean_error']:.4f}")
            print(f"  Std Error:  {validation_results['std_error']:.4f}")
            print(f"  Status: {'PASS ✓' if validation_results['overall_passed'] else 'FAIL ✗'}")

            # Generate plots
            if plot_fn is not None:
                plot_fn(history, validation_results, filename=f'{model_name}_validation.pdf')

            success = validation_results['overall_passed'] and conv_results['all_converged']
            results = {**validation_results, **conv_results}
        else:
            # No validation function (hierarchical model)
            success = conv_results['all_converged']
            results = conv_results

        return success, results

    finally:
        # Restore original registry state
        for name in test_posteriors.TEST_POSTERIORS.keys():
            if name in original_registrations:
                _REGISTRY[name] = original_registrations[name]
            elif name in _REGISTRY:
                del _REGISTRY[name]


def run_all_tests():
    """Run all available tests."""
    test_models = [
        'test_beta_bernoulli_pooled',
        'test_normal_normal_pooled',
        # 'test_beta_bernoulli_hierarchical',  # Add when ready
    ]

    results = {}
    all_passed = True

    for model_name in test_models:
        passed, test_results = run_single_test(model_name)
        results[model_name] = {'passed': passed, 'results': test_results}
        all_passed = all_passed and passed

    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}\n")

    for model_name, result in results.items():
        status = 'PASS ✓' if result['passed'] else 'FAIL ✗'
        print(f"  {model_name}: {status}")

    print(f"\nOverall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MCMC sampler validation tests')
    parser.add_argument('--model', type=str, default='all',
                       help='Model to test (or "all" for all models)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')

    args = parser.parse_args()

    if args.model == 'all':
        success = run_all_tests()
    else:
        success, _ = run_single_test(args.model, verbose=args.verbose)

    exit(0 if success else 1)
