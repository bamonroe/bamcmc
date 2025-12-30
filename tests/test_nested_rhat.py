"""
Tests for nested R-hat implementation.

Tests that nested R-hat works with a simple Gaussian target.
Run with: pytest tests/test_nested_rhat.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random

from bamcmc.mcmc_backend import compute_nested_rhat


class TestComputeNestedRhat:
    """Test the compute_nested_rhat function with synthetic data."""

    def test_converged_superchains(self):
        """Test nested R-hat with synthetic converged data."""
        K = 4  # 4 superchains
        M = 10  # 10 subchains per superchain
        n_samples = 20
        n_params = 5

        key = random.PRNGKey(42)

        # Each superchain converges to slightly different mean
        superchain_means = random.normal(key, (K, n_params))

        # Generate samples
        history_list = []
        for k in range(K):
            key, subkey = random.split(key)
            subchain_samples = random.normal(subkey, (n_samples, M, n_params)) * 0.1
            subchain_samples = subchain_samples + superchain_means[k]
            history_list.append(subchain_samples)

        # Stack: (n_samples, K*M, n_params)
        history = jnp.concatenate(history_list, axis=1)

        assert history.shape == (n_samples, K * M, n_params)

        nrhat = compute_nested_rhat(history, K, M)

        assert nrhat.shape == (n_params,)
        assert jnp.all(jnp.isfinite(nrhat))

    def test_threshold_calculation(self):
        """Test R-hat against theoretical threshold."""
        K = 4
        M = 10
        n_samples = 100
        n_params = 3

        key = random.PRNGKey(42)

        # Well-mixed chains from same distribution
        history = random.normal(key, (n_samples, K * M, n_params))

        nrhat = compute_nested_rhat(history, K, M)

        # Threshold for convergence (Margossian et al., 2022)
        tau = 1e-4
        threshold = jnp.sqrt(1 + 1 / M + tau)

        # All parameters should be below threshold for well-mixed chains
        assert jnp.all(nrhat < threshold), f"Max nR-hat {jnp.max(nrhat):.4f} >= threshold {threshold:.4f}"

    def test_detects_divergent_superchains(self):
        """Test that nested R-hat detects when superchains have diverged."""
        K = 4
        M = 5
        n_samples = 50
        n_params = 1

        key = random.PRNGKey(42)

        # Create superchains with very different means
        history_list = []
        for k in range(K):
            key, subkey = random.split(key)
            # Each superchain at mean k*100
            samples = random.normal(subkey, (n_samples, M, n_params)) + k * 100
            history_list.append(samples)

        history = jnp.concatenate(history_list, axis=1)

        nrhat = compute_nested_rhat(history, K, M)

        # Should strongly indicate non-convergence
        assert nrhat[0] > 5.0, f"Expected high R-hat for divergent chains, got {nrhat[0]:.4f}"


def test_perturbation_scale_note():
    """Note about perturbation testing.

    The create_perturbed_superchains function was planned but initialization
    is now handled directly in initialize_mcmc_system.
    """
    pass
