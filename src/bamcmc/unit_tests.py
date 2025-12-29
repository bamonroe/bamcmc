"""
Unit Tests for MCMC Backend Utilities (Refactored for Unified Nested R-hat)

Tests individual components of the MCMC backend in isolation.
Run with: python unit_tests.py
"""

import numpy as np
import jax.numpy as jnp
import unittest
import jax

# Import functions to test
from .mcmc_backend import (
    compute_nested_rhat,           # Unified diagnostic
    initialize_mcmc_system,        # Initialization logic
    build_block_arrays,            # Block array builder
    gen_rng_keys,                  # For initializing RNG keys
)
from .batch_specs import BlockSpec, SamplerType, ProposalType
from .settings import SettingSlot, MAX_SETTINGS, SETTING_DEFAULTS, build_settings_matrix
from .error_handling import validate_mcmc_config


def make_settings_array(alpha=None, n_categories=None):
    """
    Create a settings array for testing proposal functions.

    Args:
        alpha: Mixture weight (default: 0.5)
        n_categories: Number of categories for multinomial (default: 4)

    Returns:
        JAX array of shape (MAX_SETTINGS,) with specified values
    """
    settings = np.zeros(MAX_SETTINGS, dtype=np.float32)
    for slot, default in SETTING_DEFAULTS.items():
        settings[slot] = default
    if alpha is not None:
        settings[SettingSlot.ALPHA] = alpha
    if n_categories is not None:
        settings[SettingSlot.N_CATEGORIES] = n_categories
    return jnp.array(settings)

class TestUnifiedNestedRhat(unittest.TestCase):
    """Test the unified compute_nested_rhat function."""

    def test_perfect_convergence(self):
        """If all chains are identical, R-hat should be close to 1.0."""
        # Shape: (Samples=100, Chains=4, Params=1)
        # Create identical history for all chains
        n_samples = 100
        chain_data = np.random.randn(n_samples, 1)
        history = np.repeat(chain_data[:, np.newaxis, :], 4, axis=1)

        # Test Standard Mode (K=4, M=1)
        rhat = compute_nested_rhat(jnp.array(history), K=4, M=1)
        # When B=0, R-hat = sqrt((n-1)/n) ≈ sqrt(99/100) ≈ 0.995 for n=100
        expected_rhat = np.sqrt((n_samples - 1) / n_samples)
        self.assertAlmostEqual(float(rhat[0]), expected_rhat, places=4)

    def test_standard_rhat_logic_M1(self):
        """Test the math for M=1 (Standard R-hat case)."""
        # Case: 2 Chains, 10 Samples.
        # Chain 1: Mean ~0, small variance
        # Chain 2: Mean ~10, small variance
        # This should give a high R-hat since chains haven't mixed

        N = 10
        m = 2  # number of chains
        history = np.zeros((N, m, 1))
        np.random.seed(123)
        history[:, 0, 0] = 0.0 + np.random.normal(0, 0.01, N)
        history[:, 1, 0] = 10.0 + np.random.normal(0, 0.01, N)

        # Calculate expected using correct Gelman-Rubin formula
        # B = n * var(chain_means)
        # W = average within-chain variance
        # V_hat = (n-1)/n * W + B/n + B/(m*n)
        # R-hat = sqrt(V_hat / W)
        means = np.mean(history, axis=0)  # (2, 1)
        var_means = np.var(means, axis=0, ddof=1)
        B = N * var_means

        vars_within = np.var(history, axis=0, ddof=1)  # (2, 1)
        W = np.mean(vars_within)

        V_hat = ((N - 1) / N) * W + B / N + B / (m * N)
        expected_rhat = np.sqrt(V_hat / W)

        # Run Function
        computed_rhat = compute_nested_rhat(jnp.array(history), K=2, M=1)

        # Use relative tolerance for comparison
        np.testing.assert_allclose(float(computed_rhat[0]), float(expected_rhat[0]), rtol=1e-4)

    def test_nested_rhat_logic_M_gt_1(self):
        """Test the math for M>1 (Nested R-hat case)."""
        # Case: K=2 Superchains, M=2 Subchains (Total 4 chains)
        # Superchain A (Chains 0,1): Centered at 0
        # Superchain B (Chains 2,3): Centered at 10
        N = 100
        K = 2
        M = 2
        history = np.zeros((N, K*M, 1))

        # Group A
        history[:, 0, 0] = np.random.normal(0, 1, N)
        history[:, 1, 0] = np.random.normal(0, 1, N)
        # Group B
        history[:, 2, 0] = np.random.normal(10, 1, N)
        history[:, 3, 0] = np.random.normal(10, 1, N)

        # The logic inside the function:
        # 1. Reshape (N, K, M, P)
        # 2. Superchain Means (avg over M and N) -> Should be approx [0, 10]
        # 3. nB = N * var([0, 10]) -> Large
        # 4. nW = Average local variance -> Approx 1.0 (since we used std=1)

        rhat = compute_nested_rhat(jnp.array(history), K=K, M=M)

        # Expectation: High R-hat because superchains disagree
        self.assertTrue(rhat[0] > 1.5)

        # Verify it runs without error
        self.assertEqual(rhat.shape, (1,))

    def test_nested_matches_manual_calculation(self):
        """Verify JAX implementation matches manual Gelman-Rubin calculation."""
        np.random.seed(42)
        K, M = 10, 5
        n_samples = 100
        n_params = 2
        history = np.random.normal(0, 1, (n_samples, K * M, n_params))

        # JAX implementation
        rhat_jax = compute_nested_rhat(jnp.array(history), K=K, M=M)

        # Manual nested calculation
        history_nested = history.reshape(n_samples, K, M, n_params)

        superchain_means_over_time = np.mean(history_nested, axis=2)
        superchain_means = np.mean(superchain_means_over_time, axis=0)
        B = n_samples * np.var(superchain_means, axis=0, ddof=1)

        W_within_raw = np.var(history_nested, axis=0, ddof=1)
        W_within = np.mean(W_within_raw, axis=1)
        subchain_means = np.mean(history_nested, axis=0)
        B_within = np.var(subchain_means, axis=1, ddof=1)
        W = np.mean(B_within + W_within, axis=0)

        n, m = n_samples, K
        V_hat = ((n - 1) / n) * W + B / n + B / (m * n)
        rhat_manual = np.sqrt(V_hat / W)

        np.testing.assert_allclose(np.array(rhat_jax), rhat_manual, rtol=1e-5)

    def test_standard_matches_manual_calculation(self):
        """Verify M=1 case matches standard Gelman-Rubin calculation."""
        np.random.seed(42)
        K, M = 50, 1
        n_samples = 100
        n_params = 2
        history = np.random.normal(0, 1, (n_samples, K, n_params))

        # JAX implementation
        rhat_jax = compute_nested_rhat(jnp.array(history), K=K, M=M)

        # Manual standard R-hat
        chain_means = np.mean(history, axis=0)
        B = n_samples * np.var(chain_means, axis=0, ddof=1)
        W = np.mean(np.var(history, axis=0, ddof=1), axis=0)

        n, m = n_samples, K
        V_hat = ((n - 1) / n) * W + B / n + B / (m * n)
        rhat_manual = np.sqrt(V_hat / W)

        np.testing.assert_allclose(np.array(rhat_jax), rhat_manual, rtol=1e-5)

    def test_detects_nonconverged_superchains(self):
        """Verify nested R-hat detects when superchains haven't converged."""
        np.random.seed(42)
        K, M = 5, 20
        n_samples = 100
        history = np.zeros((n_samples, K * M, 1))

        # Each superchain has a very different mean
        for k in range(K):
            for m_idx in range(M):
                history[:, k * M + m_idx, 0] = np.random.normal(k * 10, 1.0, n_samples)

        rhat = compute_nested_rhat(jnp.array(history), K=K, M=M)

        # Should strongly indicate non-convergence
        self.assertTrue(rhat[0] > 5.0)

    def test_well_mixed_chains_converge(self):
        """Verify R-hat is near 1.0 for well-mixed chains from same distribution."""
        np.random.seed(42)
        n_samples = 500
        K, M = 20, 5
        # All chains from the same distribution
        history = np.random.normal(0, 1, (n_samples, K * M, 1))

        rhat = compute_nested_rhat(jnp.array(history), K=K, M=M)

        # Should be very close to 1.0
        self.assertLess(rhat[0], 1.05)


class TestInitializationLogic(unittest.TestCase):
    """Test the initialization and superchain/subchain splitting logic."""

    def test_standard_init_replication(self):
        """Test standard initialization (M=1) - No replication."""
        config = {
            'NUM_CHAINS': 4,
            'NUM_CHAINS_A': 2,
            'NUM_CHAINS_B': 2,
            'NUM_SUPERCHAINS': 4, # Explicit M=1
            'jnp_float_dtype': jnp.float32,
            'rng_seed': 42,
            'SAVE_LIKELIHOODS': False
        }
        config = gen_rng_keys(config)  # Initialize RNG keys

        # Init vector: [0, 1, 2, 3]
        init_vec = np.array([0, 1, 2, 3], dtype=np.float32)

        carry, new_config = initialize_mcmc_system(
            init_vec, config, num_gq=0, num_collect=10, num_blocks=1
        )

        states_A = carry[0]
        states_B = carry[2]

        # Should be exactly the input (no replication)
        self.assertTrue(np.array_equal(states_A.flatten(), np.array([0, 1])))
        self.assertTrue(np.array_equal(states_B.flatten(), np.array([2, 3])))
        self.assertEqual(new_config['SUBCHAINS_PER_SUPER'], 1)

    def test_nested_init_replication(self):
        """Test nested initialization (M>1) - Should replicate superchains."""
        config = {
            'NUM_CHAINS': 4,
            'NUM_CHAINS_A': 2,
            'NUM_CHAINS_B': 2,
            'NUM_SUPERCHAINS': 2, # Implies M = 4/2 = 2
            'jnp_float_dtype': jnp.float32,
            'rng_seed': 42,
            'SAVE_LIKELIHOODS': False
        }
        config = gen_rng_keys(config)  # Initialize RNG keys

        # Init vector provided for 4 chains: [10, 20, 30, 40]
        # But for Nested R-hat, it should take the first K=2 (10, 20)
        # And replicate them M=2 times.
        # Expected Result:
        # Superchain 0: [10, 10]
        # Superchain 1: [20, 20]
        # Total: [10, 10, 20, 20]

        init_vec = np.array([10, 20, 30, 40], dtype=np.float32)

        carry, new_config = initialize_mcmc_system(
            init_vec, config, num_gq=0, num_collect=10, num_blocks=1
        )

        states_A = carry[0] # First half [10, 10]
        states_B = carry[2] # Second half [20, 20]

        self.assertEqual(new_config['SUBCHAINS_PER_SUPER'], 2)

        # Verify Group A (Superchain 0)
        self.assertEqual(states_A[0,0], 10.0)
        self.assertEqual(states_A[1,0], 10.0)

        # Verify Group B (Superchain 1)
        self.assertEqual(states_B[0,0], 20.0)
        self.assertEqual(states_B[1,0], 20.0)

class TestBuildBlockArrays(unittest.TestCase):
    """Test the build_block_arrays function."""

    def test_single_block(self):
        """Test with a single parameter block."""
        batch_specs = [
            BlockSpec(
                size=3,
                sampler_type=SamplerType.METROPOLIS_HASTINGS,
                proposal_type=ProposalType.SELF_MEAN,
                label='block0'
            )
        ]
        result = build_block_arrays(batch_specs, start_idx=0)

        self.assertEqual(result.num_blocks, 1)
        self.assertEqual(result.max_size, 3)
        self.assertEqual(result.total_params, 3)

    def test_multiple_blocks(self):
        """Test with multiple blocks of different types."""
        batch_specs = [
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.SELF_MEAN),
            BlockSpec(1, SamplerType.DIRECT_CONJUGATE, direct_sampler_fn=lambda x: x)
        ]
        result = build_block_arrays(batch_specs, start_idx=0)

        self.assertEqual(result.num_blocks, 2)
        self.assertEqual(result.total_params, 3)
        self.assertTrue(jnp.array_equal(result.types, jnp.array([0, 1])))


class TestProposalSettings(unittest.TestCase):
    """Test the proposal settings extraction and MIXTURE proposal."""

    # Expected default alpha (must match SETTING_DEFAULTS in settings.py)
    DEFAULT_ALPHA = 0.5

    def test_build_settings_matrix_default(self):
        """Test that default settings are extracted correctly as JAX arrays."""
        batch_specs = [
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.SELF_MEAN),
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.CHAIN_MEAN),
        ]
        settings_matrix = build_settings_matrix(batch_specs)

        # API returns settings_matrix with shape (n_blocks, MAX_SETTINGS)
        self.assertEqual(settings_matrix.shape[0], 2)
        # Default alpha values should be applied
        alpha_values = settings_matrix[:, SettingSlot.ALPHA]
        np.testing.assert_array_almost_equal(alpha_values, [self.DEFAULT_ALPHA, self.DEFAULT_ALPHA])

    def test_build_settings_matrix_with_alpha(self):
        """Test that alpha setting is extracted for MIXTURE proposal."""
        batch_specs = [
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.MIXTURE,
                     settings={'alpha': 0.3}),
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.MIXTURE,
                     settings={'alpha': 0.7}),
        ]
        settings_matrix = build_settings_matrix(batch_specs)

        self.assertEqual(settings_matrix.shape[0], 2)
        # User-provided alpha values should be extracted
        alpha_values = settings_matrix[:, SettingSlot.ALPHA]
        np.testing.assert_array_almost_equal(alpha_values, [0.3, 0.7])

    def test_build_settings_matrix_direct_sampler(self):
        """Test that direct samplers get default settings (for JAX array compatibility)."""
        batch_specs = [
            BlockSpec(2, SamplerType.DIRECT_CONJUGATE, direct_sampler_fn=lambda x: x),
        ]
        settings_matrix = build_settings_matrix(batch_specs)

        # Direct samplers also get default settings for JAX array structure
        alpha_values = settings_matrix[:, SettingSlot.ALPHA]
        np.testing.assert_array_almost_equal(alpha_values, [self.DEFAULT_ALPHA])

    def test_mixture_proposal_type_enum(self):
        """Test that MIXTURE is properly defined in ProposalType enum."""
        self.assertEqual(int(ProposalType.MIXTURE), 2)
        self.assertEqual(ProposalType.MIXTURE.name, 'MIXTURE')


class TestMixtureProposal(unittest.TestCase):
    """Test the mixture proposal function directly."""

    def _make_coupled_data(self, mean, cov_scale, n_chains=50):
        """Create coupled_blocks and precomputed stats with specified mean and covariance."""
        key = jax.random.PRNGKey(999)
        noise = jax.random.normal(key, (n_chains, len(mean)))
        coupled_blocks = mean + noise * jnp.sqrt(cov_scale)
        # Precompute mean and covariance (mimics what backend does)
        step_mean = jnp.mean(coupled_blocks, axis=0)
        step_cov = jnp.cov(coupled_blocks, rowvar=False)
        step_cov = jnp.atleast_2d(step_cov) + 1e-5 * jnp.eye(len(mean))
        return coupled_blocks, step_mean, step_cov

    def test_mixture_proposal_runs(self):
        """Test that mixture_proposal runs without error."""
        from .proposals import mixture_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([1.0, 2.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([0.0, 0.0]), 0.1)
        block_mask = jnp.array([1.0, 1.0])

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(alpha=0.5))
        proposal, log_ratio, new_key = mixture_proposal(operand)

        self.assertEqual(proposal.shape, (2,))
        self.assertIsInstance(float(log_ratio), float)

    def test_mixture_proposal_alpha_zero(self):
        """Test mixture with alpha=0 (should behave like self_mean)."""
        from .proposals import mixture_proposal

        current_block = jnp.array([1.0, 2.0])
        # Coupled blocks centered far from current
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([10.0, 10.0]), 0.1)
        block_mask = jnp.array([1.0, 1.0])

        # With alpha=0, should always use self_mean (random walk)
        # The proposal should be centered around current_block, not coupled mean
        proposals = []
        for i in range(100):
            key = jax.random.PRNGKey(i)
            operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                       make_settings_array(alpha=0.0))
            prop, _, _ = mixture_proposal(operand)
            proposals.append(prop)

        proposals = jnp.stack(proposals)
        mean_proposal = jnp.mean(proposals, axis=0)

        # Mean should be close to current_block, not coupled mean
        self.assertTrue(jnp.allclose(mean_proposal, current_block, atol=0.5))

    def test_mixture_proposal_alpha_one(self):
        """Test mixture with alpha=1 (should behave like chain_mean)."""
        from .proposals import mixture_proposal

        current_block = jnp.array([1.0, 2.0])
        coupled_mean = jnp.array([10.0, 10.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(coupled_mean, 0.1)
        block_mask = jnp.array([1.0, 1.0])

        # With alpha=1, should always use chain_mean (independent)
        proposals = []
        for i in range(100):
            key = jax.random.PRNGKey(i)
            operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                       make_settings_array(alpha=1.0))
            prop, _, _ = mixture_proposal(operand)
            proposals.append(prop)

        proposals = jnp.stack(proposals)
        mean_proposal = jnp.mean(proposals, axis=0)

        # Mean should be close to coupled_mean, not current_block
        self.assertTrue(jnp.allclose(mean_proposal, coupled_mean, atol=0.5))

    def test_mixture_hastings_ratio_is_finite(self):
        """Test that Hastings ratio is finite and reasonable."""
        from .proposals import mixture_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([5.0, 5.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([5.0, 5.0]), 0.1)
        block_mask = jnp.array([1.0, 1.0])

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(alpha=0.5))
        _, log_ratio, _ = mixture_proposal(operand)

        # Log ratio should be finite
        self.assertTrue(jnp.isfinite(log_ratio))
        # Log ratio should be reasonable (not too extreme)
        self.assertTrue(abs(float(log_ratio)) < 100)


class TestMultinomialProposal(unittest.TestCase):
    """Test the multinomial proposal function for discrete parameters."""

    def _make_dummy_stats(self, block_size):
        """Create dummy precomputed stats (unused by multinomial, but required for signature)."""
        step_mean = jnp.zeros(block_size)
        step_cov = jnp.eye(block_size)
        return step_mean, step_cov

    def test_multinomial_proposal_runs(self):
        """Test that multinomial_proposal runs without error."""
        from .proposals import multinomial_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([2.0, 3.0])  # Discrete values
        # Coupled blocks with discrete values on grid 1-10
        coupled_blocks = jnp.array([
            [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [2.0, 2.0], [1.0, 3.0]
        ])
        block_mask = jnp.array([1.0, 1.0])
        step_mean, step_cov = self._make_dummy_stats(2)

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(alpha=0.05, n_categories=10))
        proposal, log_ratio, new_key = multinomial_proposal(operand)

        self.assertEqual(proposal.shape, (2,))
        # Proposal should be on the grid [1, 10]
        self.assertTrue(jnp.all(proposal >= 1))
        self.assertTrue(jnp.all(proposal <= 10))
        self.assertTrue(jnp.isfinite(log_ratio))

    def test_multinomial_proposal_samples_from_distribution(self):
        """Test that multinomial samples from empirical distribution."""
        from .proposals import multinomial_proposal

        current_block = jnp.array([3.0])
        # Coupled blocks heavily weighted toward value 2
        coupled_blocks = jnp.array([[2.0]] * 10 + [[1.0]] * 2 + [[3.0]] * 1)
        block_mask = jnp.array([1.0])
        step_mean, step_cov = self._make_dummy_stats(1)

        proposals = []
        for i in range(200):
            key = jax.random.PRNGKey(i)
            operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                       make_settings_array(alpha=0.01, n_categories=10))  # Low alpha = track empirical closely
            prop, _, _ = multinomial_proposal(operand)
            proposals.append(float(prop[0]))

        # Value 2 should be most common
        count_2 = sum(1 for p in proposals if p == 2.0)
        count_1 = sum(1 for p in proposals if p == 1.0)
        self.assertGreater(count_2, count_1)

    def test_multinomial_hastings_ratio(self):
        """Test that Hastings ratio is computed correctly."""
        from .proposals import multinomial_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([2.0])
        # Uniform coupled blocks (values 1-5)
        coupled_blocks = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        block_mask = jnp.array([1.0])
        step_mean, step_cov = self._make_dummy_stats(1)

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(alpha=0.05, n_categories=5))
        _, log_ratio, _ = multinomial_proposal(operand)

        # For uniform distribution, ratio should be finite
        self.assertTrue(jnp.isfinite(log_ratio))
        self.assertTrue(abs(float(log_ratio)) < 10)

class TestCheckpointHelpers(unittest.TestCase):
    """Test checkpoint and batch history utilities."""

    def test_apply_burnin_filters_correctly(self):
        """Test that apply_burnin removes samples before min_iteration."""
        from .checkpoint_helpers import apply_burnin

        # Create test data: 10 samples, 4 chains, 2 params
        history = np.random.randn(10, 4, 2)
        iterations = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        likelihoods = np.random.randn(10, 4)

        # Filter with min_iteration=500
        h_filtered, i_filtered, l_filtered = apply_burnin(
            history, iterations, likelihoods, min_iteration=500
        )

        # Should keep iterations >= 500 (6 samples: 500, 600, ..., 1000)
        self.assertEqual(h_filtered.shape[0], 6)
        self.assertEqual(i_filtered.shape[0], 6)
        self.assertEqual(l_filtered.shape[0], 6)
        self.assertEqual(i_filtered[0], 500)
        self.assertEqual(i_filtered[-1], 1000)

    def test_apply_burnin_no_likelihoods(self):
        """Test apply_burnin works when likelihoods is None."""
        from .checkpoint_helpers import apply_burnin

        history = np.random.randn(10, 4, 2)
        iterations = np.arange(10) * 100

        h_filtered, i_filtered, l_filtered = apply_burnin(
            history, iterations, likelihoods=None, min_iteration=300
        )

        self.assertEqual(h_filtered.shape[0], 7)  # iterations 300-900
        self.assertIsNone(l_filtered)

    def test_apply_burnin_keeps_all_if_min_zero(self):
        """Test apply_burnin keeps all samples when min_iteration=0."""
        from .checkpoint_helpers import apply_burnin

        history = np.random.randn(10, 4, 2)
        iterations = np.arange(10) * 100

        h_filtered, i_filtered, _ = apply_burnin(
            history, iterations, min_iteration=0
        )

        self.assertEqual(h_filtered.shape[0], 10)

    def test_compute_rhat_from_history_converged(self):
        """Test compute_rhat_from_history returns ~1.0 for well-mixed chains."""
        from .checkpoint_helpers import compute_rhat_from_history

        np.random.seed(42)
        K, M = 4, 5
        n_samples = 200
        # All chains from same distribution = should converge
        history = np.random.randn(n_samples, K * M, 3)

        rhat = compute_rhat_from_history(history, K=K, M=M)

        self.assertEqual(rhat.shape, (3,))
        # Should be close to 1.0 for well-mixed chains
        self.assertTrue(np.all(rhat < 1.1))

    def test_compute_rhat_from_history_not_converged(self):
        """Test compute_rhat_from_history detects non-convergence."""
        from .checkpoint_helpers import compute_rhat_from_history

        np.random.seed(42)
        K, M = 4, 5
        n_samples = 100
        history = np.zeros((n_samples, K * M, 1))

        # Each superchain has different mean
        for k in range(K):
            for m in range(M):
                chain_idx = k * M + m
                history[:, chain_idx, 0] = np.random.randn(n_samples) + k * 10

        rhat = compute_rhat_from_history(history, K=K, M=M)

        # Should indicate non-convergence (R-hat >> 1)
        self.assertTrue(rhat[0] > 2.0)

    def test_compute_rhat_validates_chain_count(self):
        """Test compute_rhat_from_history raises error on chain mismatch."""
        from .checkpoint_helpers import compute_rhat_from_history

        history = np.random.randn(100, 20, 2)  # 20 chains

        # K*M = 4*4 = 16 != 20
        with self.assertRaises(ValueError):
            compute_rhat_from_history(history, K=4, M=4)

    def test_combine_batch_histories_concatenates(self):
        """Test combine_batch_histories combines multiple batches."""
        from .checkpoint_helpers import combine_batch_histories
        import tempfile
        import os

        # Create temp batch files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Batch 0: iterations 0-99
            batch0 = {
                'history': np.random.randn(10, 4, 2),
                'iterations': np.arange(10) * 10,
                'likelihoods': np.random.randn(10, 4),
                'K': 2, 'M': 2,
                'mcmc_config': {'test': True},
                'thin_iteration': 10,
            }
            path0 = os.path.join(tmpdir, 'batch_000.npz')
            np.savez_compressed(path0, **batch0)

            # Batch 1: iterations 100-199
            batch1 = {
                'history': np.random.randn(10, 4, 2),
                'iterations': np.arange(10, 20) * 10,
                'likelihoods': np.random.randn(10, 4),
                'K': 2, 'M': 2,
                'mcmc_config': {'test': True},
                'thin_iteration': 10,
            }
            path1 = os.path.join(tmpdir, 'batch_001.npz')
            np.savez_compressed(path1, **batch1)

            # Combine
            history, iterations, likelihoods, metadata = combine_batch_histories([path0, path1])

            # Verify concatenation
            self.assertEqual(history.shape[0], 20)  # 10 + 10 samples
            self.assertEqual(iterations.shape[0], 20)
            self.assertEqual(likelihoods.shape[0], 20)
            self.assertEqual(iterations[0], 0)
            self.assertEqual(iterations[-1], 190)
            self.assertEqual(metadata['K'], 2)
            self.assertEqual(metadata['M'], 2)

    def test_save_load_checkpoint_roundtrip(self):
        """Test save_checkpoint and load_checkpoint preserve data."""
        from .checkpoint_helpers import save_checkpoint, load_checkpoint
        import tempfile
        import os

        # Create mock carry tuple
        states_A = np.random.randn(10, 5).astype(np.float32)
        keys_A = np.random.randint(0, 2**31, (10, 2), dtype=np.uint32)
        states_B = np.random.randn(10, 5).astype(np.float32)
        keys_B = np.random.randint(0, 2**31, (10, 2), dtype=np.uint32)
        history = np.zeros((100, 20, 10))  # Placeholder
        lik_history = np.zeros((100, 20))
        acceptance_counts = np.array([100, 200, 300], dtype=np.int32)
        iteration = 500

        carry = (states_A, keys_A, states_B, keys_B,
                 history, lik_history, acceptance_counts, iteration)

        mcmc_config = {
            'POSTERIOR_ID': 'test_model',
            'num_params': 5,
            'NUM_CHAINS_A': 10,
            'NUM_CHAINS_B': 10,
            'NUM_SUPERCHAINS': 4,
            'SUBCHAINS_PER_SUPER': 5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.npz')

            # Save
            save_checkpoint(path, carry, mcmc_config)

            # Load
            loaded = load_checkpoint(path)

            # Verify
            self.assertEqual(loaded['iteration'], 500)
            self.assertEqual(loaded['posterior_id'], 'test_model')
            self.assertEqual(loaded['num_params'], 5)
            self.assertEqual(loaded['num_chains_a'], 10)
            self.assertEqual(loaded['num_chains_b'], 10)
            np.testing.assert_array_equal(loaded['states_A'], states_A)
            np.testing.assert_array_equal(loaded['states_B'], states_B)
            np.testing.assert_array_equal(loaded['acceptance_counts'], acceptance_counts)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests(verbosity=2):
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(import_module_from_string())
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()

def import_module_from_string():
    import sys
    return sys.modules[__name__]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run MCMC backend unit tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()

    verbosity = 2 if args.verbose else 1

    # Run tests
    unittest.main(verbosity=verbosity)
