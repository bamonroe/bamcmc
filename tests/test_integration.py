"""
Integration Tests for MCMC Sampler Validation

Runs the MCMC sampler on conjugate models with known analytical solutions
and validates that the sampler is producing correct results.

Run with: pytest tests/test_integration.py -v
"""

import numpy as np
import jax.numpy as jnp
from scipy import stats as scipy_stats
import pytest

from bamcmc.mcmc_backend import rmcmc
from bamcmc.mcmc_utils import clean_config
from bamcmc.registry import register_posterior, _REGISTRY
from bamcmc import test_posteriors


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_beta_bernoulli_data(n_trials=100, true_theta=0.7, alpha_0=2.0, beta_0=2.0, seed=42):
    """Generate synthetic data for Beta-Bernoulli model."""
    np.random.seed(seed)
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
    """Generate synthetic data for Normal-Normal model (known variance)."""
    np.random.seed(seed)
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
    """Generate synthetic hierarchical data."""
    np.random.seed(seed)
    true_thetas = np.random.beta(true_alpha, true_beta, n_subjects)

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
# VALIDATION FUNCTIONS
# ============================================================================

def validate_beta_bernoulli_pooled(history, data, true_params, tolerance=0.05):
    """Validate Beta-Bernoulli pooled model against analytical posterior."""
    alpha_post, beta_post = test_posteriors.beta_bernoulli_pooled_analytical_posterior(data)

    analytical_mean = alpha_post / (alpha_post + beta_post)
    analytical_var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
    analytical_std = np.sqrt(analytical_var)

    theta_samples = history[:, :, -1].flatten()
    sample_mean = np.mean(theta_samples)
    sample_std = np.std(theta_samples, ddof=1)

    mean_error = abs(sample_mean - analytical_mean)
    std_error = abs(sample_std - analytical_std)

    mean_passed = mean_error < tolerance
    std_passed = std_error < (tolerance * 2)

    return {
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


def validate_normal_normal_pooled(history, data, true_params, tolerance=0.10):
    """Validate Normal-Normal pooled model against analytical posterior."""
    mu_post, tau_post = test_posteriors.normal_normal_pooled_analytical_posterior(data)

    mu_samples = history[:, :, -1].flatten()
    sample_mean = np.mean(mu_samples)
    sample_std = np.std(mu_samples, ddof=1)

    mean_error = abs(sample_mean - mu_post)
    std_error = abs(sample_std - tau_post)

    mean_passed = mean_error < tolerance
    std_passed = std_error < (tolerance * 2)

    return {
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


def validate_convergence(diagnostics, rhat_threshold=1.05):
    """Check R-hat convergence diagnostics."""
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
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def register_test_posteriors():
    """Register test posteriors for the test module."""
    original_registrations = {}
    for name, config in test_posteriors.TEST_POSTERIORS.items():
        if name in _REGISTRY:
            original_registrations[name] = _REGISTRY[name]
        register_posterior(name, config)

    yield

    for name in test_posteriors.TEST_POSTERIORS.keys():
        if name in original_registrations:
            _REGISTRY[name] = original_registrations[name]
        elif name in _REGISTRY:
            del _REGISTRY[name]


@pytest.fixture
def mcmc_config():
    """Default MCMC configuration for integration tests."""
    return {
        'gpu_preallocation': True,
        'use_double': True,
        'rng_seed': 1977,
        'benchmark': 50,
        'burn_iter': 500,
        'thin_iteration': 1,
        'num_collect': 1000,
        'proposal': 'chain_mean',
        'num_chains_a': 25,
        'num_chains_b': 25,
        'last_iters': 1000,
    }


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestBetaBernoulliPooled:
    """Integration tests for Beta-Bernoulli pooled model."""

    def test_sampler_matches_analytical_posterior(self, register_test_posteriors, mcmc_config):
        """Test that MCMC samples match the known analytical posterior."""
        data, true_params = generate_beta_bernoulli_data()

        mcmc_config['posterior_id'] = 'test_beta_bernoulli_pooled'

        results, _ = rmcmc(mcmc_config, data)
        history = results['history']
        diagnostics = results['diagnostics']

        # Validate convergence
        conv_results = validate_convergence(diagnostics)
        assert conv_results['all_converged'], f"Chains did not converge. Max R-hat: {conv_results['max_rhat']:.4f}"

        # Validate against analytical solution
        validation = validate_beta_bernoulli_pooled(history, data, true_params)
        assert validation['mean_passed'], f"Mean error {validation['mean_error']:.4f} exceeds tolerance"
        assert validation['std_passed'], f"Std error {validation['std_error']:.4f} exceeds tolerance"


class TestNormalNormalPooled:
    """Integration tests for Normal-Normal pooled model."""

    def test_sampler_matches_analytical_posterior(self, register_test_posteriors, mcmc_config):
        """Test that MCMC samples match the known analytical posterior."""
        data, true_params = generate_normal_normal_data()

        mcmc_config['posterior_id'] = 'test_normal_normal_pooled'

        results, _ = rmcmc(mcmc_config, data)
        history = results['history']
        diagnostics = results['diagnostics']

        # Validate convergence
        conv_results = validate_convergence(diagnostics)
        assert conv_results['all_converged'], f"Chains did not converge. Max R-hat: {conv_results['max_rhat']:.4f}"

        # Validate against analytical solution
        validation = validate_normal_normal_pooled(history, data, true_params)
        assert validation['mean_passed'], f"Mean error {validation['mean_error']:.4f} exceeds tolerance"
        assert validation['std_passed'], f"Std error {validation['std_error']:.4f} exceeds tolerance"


class TestConvergenceDiagnostics:
    """Tests for convergence diagnostics."""

    def test_rhat_computed_correctly(self, register_test_posteriors, mcmc_config):
        """Test that R-hat is computed and returned correctly."""
        data, _ = generate_beta_bernoulli_data()

        mcmc_config['posterior_id'] = 'test_beta_bernoulli_pooled'

        results, _ = rmcmc(mcmc_config, data)
        diagnostics = results['diagnostics']

        assert 'rhat' in diagnostics
        assert diagnostics['rhat'] is not None
        assert 'K' in diagnostics
        assert 'M' in diagnostics

    def test_timing_info_recorded(self, register_test_posteriors, mcmc_config):
        """Test that timing information is recorded."""
        data, _ = generate_beta_bernoulli_data()

        mcmc_config['posterior_id'] = 'test_beta_bernoulli_pooled'

        results, _ = rmcmc(mcmc_config, data)
        diagnostics = results['diagnostics']

        assert 'compile_time' in diagnostics
        assert 'wall_time' in diagnostics
        assert diagnostics['compile_time'] >= 0
        assert diagnostics['wall_time'] >= 0


# ============================================================================
# CLI RUNNER (for backwards compatibility)
# ============================================================================

def run_single_test(model_name, verbose=False):
    """
    Run a single test for the specified model.
    For backwards compatibility with old CLI usage.
    """
    print(f"\n{'='*70}")
    print(f"RUNNING TEST: {model_name}")
    print(f"{'='*70}\n")

    if model_name == 'test_beta_bernoulli_pooled':
        data, true_params = generate_beta_bernoulli_data()
        validate_fn = validate_beta_bernoulli_pooled
    elif model_name == 'test_normal_normal_pooled':
        data, true_params = generate_normal_normal_data()
        validate_fn = validate_normal_normal_pooled
    elif model_name == 'test_beta_bernoulli_hierarchical':
        data, true_params = generate_beta_bernoulli_hierarchical_data()
        print("Hierarchical model test: Running sampler (no analytical validation yet)")
        validate_fn = None
    else:
        print(f"ERROR: Unknown test model '{model_name}'")
        return False, {}

    mcmc_config = {
        'posterior_id': model_name,
        'gpu_preallocation': True,
        'use_double': True,
        'rng_seed': 1977,
        'benchmark': 50,
        'burn_iter': 500,
        'thin_iteration': 1,
        'num_collect': 1000,
        'proposal': 'chain_mean',
        'num_chains_a': 25,
        'num_chains_b': 25,
        'last_iters': 1000,
    }

    # Register test posteriors
    original_registrations = {}
    for name, config in test_posteriors.TEST_POSTERIORS.items():
        if name in _REGISTRY:
            original_registrations[name] = _REGISTRY[name]
        register_posterior(name, config)

    try:
        print("Running MCMC sampler...")
        results, _ = rmcmc(mcmc_config, data)
        history = results['history']
        diagnostics = results['diagnostics']

        conv_results = validate_convergence(diagnostics)
        print(f"\nConvergence Diagnostics:")
        print(f"  Max R-hat: {conv_results['max_rhat']:.4f}")
        print(f"  All converged: {'YES' if conv_results['all_converged'] else 'NO'}")

        if validate_fn is not None:
            validation_results = validate_fn(history, data, true_params)

            print(f"\nValidation Results:")
            print(f"  Mean Error: {validation_results['mean_error']:.4f}")
            print(f"  Std Error:  {validation_results['std_error']:.4f}")
            print(f"  Status: {'PASS' if validation_results['overall_passed'] else 'FAIL'}")

            success = validation_results['overall_passed'] and conv_results['all_converged']
            test_results = {**validation_results, **conv_results}
        else:
            success = conv_results['all_converged']
            test_results = conv_results

        return success, test_results

    finally:
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
    ]

    results = {}
    all_passed = True

    for model_name in test_models:
        passed, test_results = run_single_test(model_name)
        results[model_name] = {'passed': passed, 'results': test_results}
        all_passed = all_passed and passed

    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}\n")

    for model_name, result in results.items():
        status = 'PASS' if result['passed'] else 'FAIL'
        print(f"  {model_name}: {status}")

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    import argparse

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
