"""
Unit Tests for MCMC Backend Utilities

Tests individual components of the MCMC backend in isolation.
Run with: pytest tests/test_unit.py -v
"""

import numpy as np
import jax.numpy as jnp
import jax
import jax.random
import pytest

from bamcmc.mcmc import compute_nested_rhat
from bamcmc.mcmc.config import gen_rng_keys, initialize_mcmc_system
from bamcmc.mcmc.types import build_block_arrays
from bamcmc.batch_specs import BlockSpec, SamplerType, ProposalType
from bamcmc.settings import SettingSlot, build_settings_matrix
from bamcmc.error_handling import validate_mcmc_config

from .conftest import make_settings_array, dummy_grad_fn


# ============================================================================
# NESTED R-HAT TESTS
# ============================================================================

class TestUnifiedNestedRhat:
    """Test the unified compute_nested_rhat function."""

    def test_perfect_convergence(self):
        """If all chains are identical, R-hat should be close to 1.0."""
        n_samples = 100
        chain_data = np.random.randn(n_samples, 1)
        history = np.repeat(chain_data[:, np.newaxis, :], 4, axis=1)

        rhat = compute_nested_rhat(jnp.array(history), K=4, M=1)
        expected_rhat = np.sqrt((n_samples - 1) / n_samples)
        assert abs(float(rhat[0]) - expected_rhat) < 1e-4

    def test_standard_rhat_logic_M1(self):
        """Test the math for M=1 (Standard R-hat case)."""
        N = 10
        m = 2
        history = np.zeros((N, m, 1))
        np.random.seed(123)
        history[:, 0, 0] = 0.0 + np.random.normal(0, 0.01, N)
        history[:, 1, 0] = 10.0 + np.random.normal(0, 0.01, N)

        means = np.mean(history, axis=0)
        var_means = np.var(means, axis=0, ddof=1)
        B = N * var_means
        vars_within = np.var(history, axis=0, ddof=1)
        W = np.mean(vars_within)
        V_hat = ((N - 1) / N) * W + B / N + B / (m * N)
        expected_rhat = np.sqrt(V_hat / W)

        computed_rhat = compute_nested_rhat(jnp.array(history), K=2, M=1)
        np.testing.assert_allclose(float(computed_rhat[0]), float(expected_rhat[0]), rtol=1e-4)

    def test_nested_rhat_logic_M_gt_1(self):
        """Test the math for M>1 (Nested R-hat case)."""
        N = 100
        K = 2
        M = 2
        history = np.zeros((N, K*M, 1))

        history[:, 0, 0] = np.random.normal(0, 1, N)
        history[:, 1, 0] = np.random.normal(0, 1, N)
        history[:, 2, 0] = np.random.normal(10, 1, N)
        history[:, 3, 0] = np.random.normal(10, 1, N)

        rhat = compute_nested_rhat(jnp.array(history), K=K, M=M)
        assert rhat[0] > 1.5
        assert rhat.shape == (1,)

    def test_nested_matches_manual_calculation(self):
        """Verify JAX implementation matches manual Gelman-Rubin calculation."""
        np.random.seed(42)
        K, M = 10, 5
        n_samples = 100
        n_params = 2
        history = np.random.normal(0, 1, (n_samples, K * M, n_params))

        rhat_jax = compute_nested_rhat(jnp.array(history), K=K, M=M)

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

        rhat_jax = compute_nested_rhat(jnp.array(history), K=K, M=M)

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

        for k in range(K):
            for m_idx in range(M):
                history[:, k * M + m_idx, 0] = np.random.normal(k * 10, 1.0, n_samples)

        rhat = compute_nested_rhat(jnp.array(history), K=K, M=M)
        assert rhat[0] > 5.0

    def test_well_mixed_chains_converge(self):
        """Verify R-hat is near 1.0 for well-mixed chains from same distribution."""
        np.random.seed(42)
        n_samples = 500
        K, M = 20, 5
        history = np.random.normal(0, 1, (n_samples, K * M, 1))

        rhat = compute_nested_rhat(jnp.array(history), K=K, M=M)
        assert rhat[0] < 1.05


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestInitializationLogic:
    """Test the initialization and superchain/subchain splitting logic."""

    def test_standard_init_replication(self):
        """Test standard initialization (M=1) - No replication."""
        user_config = {
            'num_chains': 4,
            'num_chains_a': 2,
            'num_chains_b': 2,
            'num_superchains': 4,
            'save_likelihoods': False
        }
        master_key, init_key = gen_rng_keys(42)
        runtime_ctx = {
            'jnp_float_dtype': jnp.float32,
            'init_key': init_key,
            'master_key': master_key,
        }

        init_vec = np.array([0, 1, 2, 3], dtype=np.float32)

        carry, new_config = initialize_mcmc_system(
            init_vec, user_config, runtime_ctx, num_gq=0, num_collect=10, num_blocks=1
        )

        states_A = carry[0]
        states_B = carry[2]

        assert np.array_equal(states_A.flatten(), np.array([0, 1]))
        assert np.array_equal(states_B.flatten(), np.array([2, 3]))
        assert new_config['subchains_per_super'] == 1

    def test_nested_init_replication(self):
        """Test nested initialization (M>1) - Should replicate superchains."""
        user_config = {
            'num_chains': 4,
            'num_chains_a': 2,
            'num_chains_b': 2,
            'num_superchains': 2,
            'save_likelihoods': False
        }
        master_key, init_key = gen_rng_keys(42)
        runtime_ctx = {
            'jnp_float_dtype': jnp.float32,
            'init_key': init_key,
            'master_key': master_key,
        }

        init_vec = np.array([10, 20, 30, 40], dtype=np.float32)

        carry, new_config = initialize_mcmc_system(
            init_vec, user_config, runtime_ctx, num_gq=0, num_collect=10, num_blocks=1
        )

        states_A = carry[0]
        states_B = carry[2]

        assert new_config['subchains_per_super'] == 2
        assert states_A[0, 0] == 10.0
        assert states_A[1, 0] == 10.0
        assert states_B[0, 0] == 20.0
        assert states_B[1, 0] == 20.0


# ============================================================================
# BLOCK ARRAYS TESTS
# ============================================================================

class TestBuildBlockArrays:
    """Test the build_block_arrays function."""

    def test_single_block(self, single_block_spec):
        """Test with a single parameter block."""
        result = build_block_arrays(single_block_spec, start_idx=0)

        assert result.num_blocks == 1
        assert result.max_size == 3
        assert result.total_params == 3

    def test_multiple_blocks(self, multi_block_specs):
        """Test with multiple blocks of different types."""
        result = build_block_arrays(multi_block_specs, start_idx=0)

        assert result.num_blocks == 2
        assert result.total_params == 3
        assert jnp.array_equal(result.types, jnp.array([0, 1]))


# ============================================================================
# PROPOSAL SETTINGS TESTS
# ============================================================================

class TestProposalSettings:
    """Test the proposal settings extraction and MIXTURE proposal."""

    DEFAULT_CHAIN_PROB = 0.5

    def test_build_settings_matrix_default(self):
        """Test that default settings are extracted correctly as JAX arrays."""
        batch_specs = [
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.SELF_MEAN),
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.CHAIN_MEAN),
        ]
        settings_matrix = build_settings_matrix(batch_specs)

        assert settings_matrix.shape[0] == 2
        chain_prob_values = settings_matrix[:, SettingSlot.CHAIN_PROB]
        np.testing.assert_array_almost_equal(chain_prob_values, [self.DEFAULT_CHAIN_PROB, self.DEFAULT_CHAIN_PROB])

    def test_build_settings_matrix_with_chain_prob(self):
        """Test that chain_prob setting is extracted for MIXTURE proposal."""
        batch_specs = [
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.MIXTURE,
                     settings={'chain_prob': 0.3}),
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.MIXTURE,
                     settings={'chain_prob': 0.7}),
        ]
        settings_matrix = build_settings_matrix(batch_specs)

        assert settings_matrix.shape[0] == 2
        chain_prob_values = settings_matrix[:, SettingSlot.CHAIN_PROB]
        np.testing.assert_array_almost_equal(chain_prob_values, [0.3, 0.7])

    def test_build_settings_matrix_direct_sampler(self):
        """Test that direct samplers get default settings."""
        batch_specs = [
            BlockSpec(2, SamplerType.DIRECT_CONJUGATE, direct_sampler_fn=lambda x: x),
        ]
        settings_matrix = build_settings_matrix(batch_specs)

        chain_prob_values = settings_matrix[:, SettingSlot.CHAIN_PROB]
        np.testing.assert_array_almost_equal(chain_prob_values, [self.DEFAULT_CHAIN_PROB])

    def test_mixture_proposal_type_enum(self):
        """Test that MIXTURE is properly defined in ProposalType enum."""
        assert int(ProposalType.MIXTURE) == 2
        assert ProposalType.MIXTURE.name == 'MIXTURE'


# ============================================================================
# MIXTURE PROPOSAL TESTS
# ============================================================================

class TestMixtureProposal:
    """Test the mixture proposal function directly."""

    def _make_coupled_data(self, mean, cov_scale, n_chains=50):
        """Create coupled_blocks and precomputed stats."""
        key = jax.random.PRNGKey(999)
        noise = jax.random.normal(key, (n_chains, len(mean)))
        coupled_blocks = mean + noise * jnp.sqrt(cov_scale)
        step_mean = jnp.mean(coupled_blocks, axis=0)
        step_cov = jnp.cov(coupled_blocks, rowvar=False)
        step_cov = jnp.atleast_2d(step_cov) + 1e-5 * jnp.eye(len(mean))
        return coupled_blocks, step_mean, step_cov

    def test_mixture_proposal_runs(self):
        """Test that mixture_proposal runs without error."""
        from bamcmc.proposals import mixture_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([1.0, 2.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([0.0, 0.0]), 0.1)
        block_mask = jnp.array([1.0, 1.0])
        block_mode = step_mean  # Dummy mode for testing

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(chain_prob=0.5), dummy_grad_fn, block_mode)
        proposal, log_ratio, new_key = mixture_proposal(operand)

        assert proposal.shape == (2,)
        assert isinstance(float(log_ratio), float)

    def test_mixture_proposal_chain_prob_zero(self):
        """Test mixture with chain_prob=0 (should behave like self_mean)."""
        from bamcmc.proposals import mixture_proposal

        current_block = jnp.array([1.0, 2.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([10.0, 10.0]), 0.1)
        block_mask = jnp.array([1.0, 1.0])
        block_mode = step_mean  # Dummy mode for testing

        proposals = []
        for i in range(100):
            key = jax.random.PRNGKey(i)
            operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                       make_settings_array(chain_prob=0.0), dummy_grad_fn, block_mode)
            prop, _, _ = mixture_proposal(operand)
            proposals.append(prop)

        proposals = jnp.stack(proposals)
        mean_proposal = jnp.mean(proposals, axis=0)
        assert jnp.allclose(mean_proposal, current_block, atol=0.5)

    def test_mixture_proposal_chain_prob_one(self):
        """Test mixture with chain_prob=1 (should behave like chain_mean)."""
        from bamcmc.proposals import mixture_proposal

        current_block = jnp.array([1.0, 2.0])
        coupled_mean = jnp.array([10.0, 10.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(coupled_mean, 0.1)
        block_mask = jnp.array([1.0, 1.0])
        block_mode = step_mean  # Dummy mode for testing

        proposals = []
        for i in range(100):
            key = jax.random.PRNGKey(i)
            operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                       make_settings_array(chain_prob=1.0), dummy_grad_fn, block_mode)
            prop, _, _ = mixture_proposal(operand)
            proposals.append(prop)

        proposals = jnp.stack(proposals)
        mean_proposal = jnp.mean(proposals, axis=0)
        assert jnp.allclose(mean_proposal, coupled_mean, atol=0.5)

    def test_mixture_hastings_ratio_is_finite(self):
        """Test that Hastings ratio is finite and reasonable."""
        from bamcmc.proposals import mixture_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([5.0, 5.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([5.0, 5.0]), 0.1)
        block_mask = jnp.array([1.0, 1.0])
        block_mode = step_mean  # Dummy mode for testing

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(chain_prob=0.5), dummy_grad_fn, block_mode)
        _, log_ratio, _ = mixture_proposal(operand)

        assert jnp.isfinite(log_ratio)
        assert abs(float(log_ratio)) < 100


# ============================================================================
# MULTINOMIAL PROPOSAL TESTS
# ============================================================================

class TestMultinomialProposal:
    """Test the multinomial proposal function for discrete parameters."""

    def _make_dummy_stats(self, block_size):
        """Create dummy precomputed stats."""
        step_mean = jnp.zeros(block_size)
        step_cov = jnp.eye(block_size)
        return step_mean, step_cov

    def test_multinomial_proposal_runs(self):
        """Test that multinomial_proposal runs without error."""
        from bamcmc.proposals import multinomial_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([2.0, 3.0])
        coupled_blocks = jnp.array([
            [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [2.0, 2.0], [1.0, 3.0]
        ])
        block_mask = jnp.array([1.0, 1.0])
        step_mean, step_cov = self._make_dummy_stats(2)
        block_mode = step_mean  # Dummy mode for testing

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(uniform_weight=0.05, n_categories=10), dummy_grad_fn, block_mode)
        proposal, log_ratio, new_key = multinomial_proposal(operand)

        assert proposal.shape == (2,)
        assert jnp.all(proposal >= 1)
        assert jnp.all(proposal <= 10)
        assert jnp.isfinite(log_ratio)

    def test_multinomial_proposal_samples_from_distribution(self):
        """Test that multinomial samples from empirical distribution."""
        from bamcmc.proposals import multinomial_proposal

        current_block = jnp.array([3.0])
        coupled_blocks = jnp.array([[2.0]] * 10 + [[1.0]] * 2 + [[3.0]] * 1)
        block_mask = jnp.array([1.0])
        step_mean, step_cov = self._make_dummy_stats(1)
        block_mode = step_mean  # Dummy mode for testing

        proposals = []
        for i in range(200):
            key = jax.random.PRNGKey(i)
            operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                       make_settings_array(uniform_weight=0.01, n_categories=10), dummy_grad_fn, block_mode)
            prop, _, _ = multinomial_proposal(operand)
            proposals.append(float(prop[0]))

        count_2 = sum(1 for p in proposals if p == 2.0)
        count_1 = sum(1 for p in proposals if p == 1.0)
        assert count_2 > count_1

    def test_multinomial_hastings_ratio(self):
        """Test that Hastings ratio is computed correctly."""
        from bamcmc.proposals import multinomial_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([2.0])
        coupled_blocks = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        block_mask = jnp.array([1.0])
        step_mean, step_cov = self._make_dummy_stats(1)
        block_mode = step_mean  # Dummy mode for testing

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(uniform_weight=0.05, n_categories=5), dummy_grad_fn, block_mode)
        _, log_ratio, _ = multinomial_proposal(operand)

        assert jnp.isfinite(log_ratio)
        assert abs(float(log_ratio)) < 10


# ============================================================================
# MCOV_WEIGHTED PROPOSAL TESTS
# ============================================================================

class TestMcovWeightedProposal:
    """Test MCOV_WEIGHTED proposal (Mean-Covariance Weighted)."""

    def _make_coupled_data(self, mean, std=0.1):
        """Helper to create coupled_blocks, step_mean, step_cov."""
        n_chains = 10
        dim = mean.shape[0]
        coupled_blocks = jnp.zeros((n_chains, dim))
        for i in range(n_chains):
            coupled_blocks = coupled_blocks.at[i].set(mean + std * jax.random.normal(jax.random.PRNGKey(i), (dim,)))
        step_mean = jnp.mean(coupled_blocks, axis=0)
        step_cov = jnp.eye(dim) * std**2
        return coupled_blocks, step_mean, step_cov

    def test_mcov_weighted_proposal_runs(self):
        """Test that mcov_weighted_proposal runs without error."""
        from bamcmc.proposals import mcov_weighted_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([1.0, 2.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([0.0, 0.0]), 0.1)
        block_mask = jnp.array([1.0, 1.0])
        block_mode = step_mean

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(cov_beta=1.0), dummy_grad_fn, block_mode)
        proposal, log_ratio, new_key = mcov_weighted_proposal(operand)

        assert proposal.shape == (2,)
        assert isinstance(float(log_ratio), float)
        assert jnp.isfinite(log_ratio)

    def test_mcov_weighted_beta_zero_similar_to_mean_weighted(self):
        """Test that beta=0 produces similar behavior to MEAN_WEIGHTED."""
        from bamcmc.proposals import mcov_weighted_proposal, mean_weighted_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([5.0, 5.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([0.0, 0.0]), 1.0)
        block_mask = jnp.array([1.0, 1.0])
        block_mode = step_mean

        # With beta=0, MCOV_WEIGHTED should behave similarly to MEAN_WEIGHTED
        operand_mcov = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                        make_settings_array(cov_beta=0.0), dummy_grad_fn, block_mode)
        operand_mean = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                        make_settings_array(), dummy_grad_fn, block_mode)

        prop_mcov, ratio_mcov, _ = mcov_weighted_proposal(operand_mcov)
        prop_mean, ratio_mean, _ = mean_weighted_proposal(operand_mean)

        # Both proposals use the same RNG key, so should be identical when beta=0
        assert jnp.allclose(prop_mcov, prop_mean, atol=1e-5)
        assert jnp.allclose(ratio_mcov, ratio_mean, atol=1e-5)

    def test_mcov_weighted_covariance_expands_when_far(self):
        """Test that covariance effectively expands when far from mean."""
        from bamcmc.proposals import mcov_weighted_proposal

        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([0.0, 0.0]), 1.0)
        block_mask = jnp.array([1.0, 1.0])
        block_mode = step_mean

        # Current state far from mean
        current_far = jnp.array([10.0, 10.0])
        # Current state near mean
        current_near = jnp.array([0.1, 0.1])

        proposals_far = []
        proposals_near = []

        for i in range(100):
            key = jax.random.PRNGKey(i)

            operand_far = (key, current_far, step_mean, step_cov, coupled_blocks, block_mask,
                           make_settings_array(cov_beta=2.0), dummy_grad_fn, block_mode)
            prop_far, _, _ = mcov_weighted_proposal(operand_far)
            proposals_far.append(prop_far)

            operand_near = (key, current_near, step_mean, step_cov, coupled_blocks, block_mask,
                            make_settings_array(cov_beta=2.0), dummy_grad_fn, block_mode)
            prop_near, _, _ = mcov_weighted_proposal(operand_near)
            proposals_near.append(prop_near)

        proposals_far = jnp.stack(proposals_far)
        proposals_near = jnp.stack(proposals_near)

        # Variance of proposals when far should be larger than when near
        var_far = jnp.var(proposals_far, axis=0)
        var_near = jnp.var(proposals_near, axis=0)

        assert jnp.all(var_far > var_near)

    def test_mcov_weighted_mean_pulled_toward_coupled_mean_when_far(self):
        """Test that proposals are pulled toward coupled mean when far away."""
        from bamcmc.proposals import mcov_weighted_proposal

        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([0.0, 0.0]), 0.5)
        block_mask = jnp.array([1.0, 1.0])
        block_mode = step_mean

        # Current state far from mean
        current_far = jnp.array([20.0, 20.0])

        proposals = []
        for i in range(200):
            key = jax.random.PRNGKey(i)
            operand = (key, current_far, step_mean, step_cov, coupled_blocks, block_mask,
                       make_settings_array(cov_beta=2.0), dummy_grad_fn, block_mode)
            prop, _, _ = mcov_weighted_proposal(operand)
            proposals.append(prop)

        proposals = jnp.stack(proposals)
        mean_proposal = jnp.mean(proposals, axis=0)

        # Mean of proposals should be pulled TOWARD coupled mean (0, 0)
        # i.e., mean_proposal should be between current_far and step_mean
        # This means mean_proposal < current_far (element-wise)
        assert jnp.all(mean_proposal < current_far)
        # And mean_proposal should be noticeably different from current_far
        # (showing the pull effect)
        dist_moved = jnp.linalg.norm(mean_proposal - current_far)
        assert dist_moved > 1.0  # Should have moved at least 1 unit toward mean

    def test_mcov_weighted_hastings_ratio_is_finite(self):
        """Test that Hastings ratio is finite and reasonable."""
        from bamcmc.proposals import mcov_weighted_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([5.0, 5.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([0.0, 0.0]), 1.0)
        block_mask = jnp.array([1.0, 1.0])
        block_mode = step_mean

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(cov_beta=1.0), dummy_grad_fn, block_mode)
        _, log_ratio, _ = mcov_weighted_proposal(operand)

        assert jnp.isfinite(log_ratio)
        assert abs(float(log_ratio)) < 100

    def test_mcov_weighted_respects_mask(self):
        """Test that proposal respects the block mask."""
        from bamcmc.proposals import mcov_weighted_proposal

        key = jax.random.PRNGKey(42)
        current_block = jnp.array([1.0, 2.0, 3.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([0.0, 0.0, 0.0]), 0.5)
        # Mask: only first two dimensions active
        block_mask = jnp.array([1.0, 1.0, 0.0])
        block_mode = step_mean

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(cov_beta=1.0), dummy_grad_fn, block_mode)
        proposal, _, _ = mcov_weighted_proposal(operand)

        # Third dimension should be unchanged
        assert proposal[2] == current_block[2]
        # First two may have changed
        assert proposal.shape == (3,)


# ============================================================================
# CHECKPOINT HELPER TESTS
# ============================================================================

class TestCheckpointHelpers:
    """Test checkpoint and batch history utilities."""

    def test_apply_burnin_filters_correctly(self):
        """Test that apply_burnin removes samples before min_iteration."""
        from bamcmc.checkpoint_helpers import apply_burnin

        history = np.random.randn(10, 4, 2)
        iterations = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        likelihoods = np.random.randn(10, 4)

        h_filtered, i_filtered, l_filtered = apply_burnin(
            history, iterations, likelihoods, min_iteration=500
        )

        assert h_filtered.shape[0] == 6
        assert i_filtered.shape[0] == 6
        assert l_filtered.shape[0] == 6
        assert i_filtered[0] == 500
        assert i_filtered[-1] == 1000

    def test_apply_burnin_no_likelihoods(self):
        """Test apply_burnin works when likelihoods is None."""
        from bamcmc.checkpoint_helpers import apply_burnin

        history = np.random.randn(10, 4, 2)
        iterations = np.arange(10) * 100

        h_filtered, i_filtered, l_filtered = apply_burnin(
            history, iterations, likelihoods=None, min_iteration=300
        )

        assert h_filtered.shape[0] == 7
        assert l_filtered is None

    def test_apply_burnin_keeps_all_if_min_zero(self):
        """Test apply_burnin keeps all samples when min_iteration=0."""
        from bamcmc.checkpoint_helpers import apply_burnin

        history = np.random.randn(10, 4, 2)
        iterations = np.arange(10) * 100

        h_filtered, i_filtered, _ = apply_burnin(
            history, iterations, min_iteration=0
        )

        assert h_filtered.shape[0] == 10

    def test_compute_rhat_from_history_converged(self):
        """Test compute_rhat_from_history returns ~1.0 for well-mixed chains."""
        from bamcmc.checkpoint_helpers import compute_rhat_from_history

        np.random.seed(42)
        K, M = 4, 5
        n_samples = 200
        history = np.random.randn(n_samples, K * M, 3)

        rhat = compute_rhat_from_history(history, K=K, M=M)

        assert rhat.shape == (3,)
        assert np.all(rhat < 1.1)

    def test_compute_rhat_from_history_not_converged(self):
        """Test compute_rhat_from_history detects non-convergence."""
        from bamcmc.checkpoint_helpers import compute_rhat_from_history

        np.random.seed(42)
        K, M = 4, 5
        n_samples = 100
        history = np.zeros((n_samples, K * M, 1))

        for k in range(K):
            for m in range(M):
                chain_idx = k * M + m
                history[:, chain_idx, 0] = np.random.randn(n_samples) + k * 10

        rhat = compute_rhat_from_history(history, K=K, M=M)

        assert rhat[0] > 2.0

    def test_compute_rhat_validates_chain_count(self):
        """Test compute_rhat_from_history raises error on chain mismatch."""
        from bamcmc.checkpoint_helpers import compute_rhat_from_history

        history = np.random.randn(100, 20, 2)

        with pytest.raises(ValueError):
            compute_rhat_from_history(history, K=4, M=4)

    def test_combine_batch_histories_concatenates(self):
        """Test combine_batch_histories combines multiple batches."""
        from bamcmc.checkpoint_helpers import combine_batch_histories
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
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

            history, iterations, likelihoods, metadata = combine_batch_histories([path0, path1])

            assert history.shape[0] == 20
            assert iterations.shape[0] == 20
            assert likelihoods.shape[0] == 20
            assert iterations[0] == 0
            assert iterations[-1] == 190
            assert metadata['K'] == 2
            assert metadata['M'] == 2

    def test_save_load_checkpoint_roundtrip(self):
        """Test save_checkpoint and load_checkpoint preserve data."""
        from bamcmc.checkpoint_helpers import save_checkpoint, load_checkpoint
        import tempfile
        import os

        states_A = np.random.randn(10, 5).astype(np.float32)
        keys_A = np.random.randint(0, 2**31, (10, 2), dtype=np.uint32)
        states_B = np.random.randn(10, 5).astype(np.float32)
        keys_B = np.random.randint(0, 2**31, (10, 2), dtype=np.uint32)
        history = np.zeros((100, 20, 10))
        lik_history = np.zeros((100, 20))
        temp_history = np.zeros((1,), dtype=np.int32)
        acceptance_counts = np.array([100, 200, 300], dtype=np.int32)
        iteration = 500

        carry = (states_A, keys_A, states_B, keys_B,
                 history, lik_history, temp_history,
                 acceptance_counts, iteration,
                 np.array([1.0], dtype=np.float32),
                 np.zeros(10, dtype=np.int32),
                 np.zeros(10, dtype=np.int32),
                 np.zeros(1, dtype=np.int32),
                 np.zeros(1, dtype=np.int32),
                 0)

        mcmc_config = {
            'posterior_id': 'test_model',
            'num_params': 5,
            'num_chains_a': 10,
            'num_chains_b': 10,
            'num_superchains': 4,
            'subchains_per_super': 5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.npz')

            save_checkpoint(path, carry, mcmc_config)
            loaded = load_checkpoint(path)

            assert loaded['iteration'] == 500
            assert loaded['posterior_id'] == 'test_model'
            assert loaded['num_params'] == 5
            assert loaded['num_chains_a'] == 10
            assert loaded['num_chains_b'] == 10
            np.testing.assert_array_equal(loaded['states_A'], states_A)
            np.testing.assert_array_equal(loaded['states_B'], states_B)
            np.testing.assert_array_equal(loaded['acceptance_counts'], acceptance_counts)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_block_single_param(self):
        """Test with a single block containing a single parameter."""
        batch_specs = [
            BlockSpec(1, SamplerType.METROPOLIS_HASTINGS, ProposalType.SELF_MEAN)
        ]
        result = build_block_arrays(batch_specs, start_idx=0)

        assert result.num_blocks == 1
        assert result.max_size == 1
        assert result.total_params == 1
        assert result.indices.shape == (1, 1)

    def test_many_small_blocks(self):
        """Test with many small blocks (typical hierarchical model)."""
        n_subjects = 100
        batch_specs = [
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.CHAIN_MEAN,
                     label=f"subject_{i}")
            for i in range(n_subjects)
        ]
        result = build_block_arrays(batch_specs, start_idx=0)

        assert result.num_blocks == n_subjects
        assert result.total_params == n_subjects * 2

    def test_mixed_block_sizes(self):
        """Test with varying block sizes."""
        batch_specs = [
            BlockSpec(1, SamplerType.METROPOLIS_HASTINGS, ProposalType.SELF_MEAN),
            BlockSpec(5, SamplerType.METROPOLIS_HASTINGS, ProposalType.CHAIN_MEAN),
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.MIXTURE),
        ]
        result = build_block_arrays(batch_specs, start_idx=0)

        assert result.num_blocks == 3
        assert result.max_size == 5  # Largest block
        assert result.total_params == 8

    def test_block_with_custom_start_idx(self):
        """Test that start_idx offsets parameter indices correctly."""
        batch_specs = [
            BlockSpec(3, SamplerType.METROPOLIS_HASTINGS, ProposalType.SELF_MEAN)
        ]

        # Start at index 100
        result = build_block_arrays(batch_specs, start_idx=100)

        # First valid indices should be 100, 101, 102
        assert result.indices[0, 0] == 100
        assert result.indices[0, 1] == 101
        assert result.indices[0, 2] == 102

    def test_nested_rhat_single_sample(self):
        """Test R-hat with minimal samples (edge case)."""
        # With only 2 samples, R-hat should still compute (though unreliable)
        n_samples = 2
        K, M = 2, 2
        history = np.random.randn(n_samples, K * M, 1)

        # Should not crash
        rhat = compute_nested_rhat(jnp.array(history), K=K, M=M)
        # Result may be NaN or extreme but should not error
        assert rhat.shape == (1,)

    def test_nested_rhat_many_chains_few_samples(self):
        """Test R-hat with many chains but few samples."""
        n_samples = 10
        K, M = 50, 1  # 50 chains, no nesting
        history = np.random.randn(n_samples, K * M, 2)

        rhat = compute_nested_rhat(jnp.array(history), K=K, M=M)
        assert rhat.shape == (2,)

    def test_validation_rejects_negative_chains(self):
        """Test that validation catches invalid chain counts."""
        config = {
            'posterior_id': 'test',
            'num_chains_a': -5,  # Invalid!
            'num_chains_b': 10,
            'num_superchains': 1,
        }

        with pytest.raises(ValueError):
            validate_mcmc_config(config)

    def test_validation_rejects_zero_superchains(self):
        """Test that validation catches zero superchains."""
        config = {
            'posterior_id': 'test',
            'num_chains_a': 10,
            'num_chains_b': 10,
            'num_superchains': 0,  # Invalid!
        }

        with pytest.raises(ValueError):
            validate_mcmc_config(config)


# ============================================================================
# DIRECT SAMPLER TESTS
# ============================================================================

class TestDirectSampler:
    """Test direct (conjugate) sampler functionality."""

    def test_direct_sampler_block_spec_creation(self):
        """Test creating a BlockSpec with direct sampler."""
        def dummy_sampler(key, state, indices):
            # Simple direct sampler that returns mean of uniform
            new_val = jax.random.uniform(key, (len(indices),))
            new_state = state.at[indices].set(new_val)
            _, new_key = jax.random.split(key)
            return new_state, new_key

        spec = BlockSpec(
            size=2,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=dummy_sampler,
            label="conjugate_block"
        )

        assert spec.sampler_type == SamplerType.DIRECT_CONJUGATE
        assert spec.direct_sampler_fn is not None
        assert spec.is_direct_sampler()
        assert not spec.is_mh_sampler()

    def test_direct_sampler_requires_function(self):
        """Test that DIRECT_CONJUGATE requires direct_sampler_fn."""
        with pytest.raises(ValueError, match="direct_sampler_fn"):
            BlockSpec(
                size=2,
                sampler_type=SamplerType.DIRECT_CONJUGATE,
                # Missing direct_sampler_fn!
            )

    def test_mh_sampler_defaults_proposal_type(self):
        """Test that MH sampler defaults to SELF_MEAN if proposal_type not specified."""
        # This should NOT raise - proposal_type defaults to SELF_MEAN
        spec = BlockSpec(
            size=2,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
        )
        assert spec.proposal_type == ProposalType.SELF_MEAN

    def test_build_block_arrays_with_direct_sampler(self):
        """Test building BlockArrays with mixed sampler types."""
        def dummy_sampler(key, state, indices):
            return state, key

        batch_specs = [
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.CHAIN_MEAN),
            BlockSpec(1, SamplerType.DIRECT_CONJUGATE, direct_sampler_fn=dummy_sampler),
            BlockSpec(3, SamplerType.METROPOLIS_HASTINGS, ProposalType.SELF_MEAN),
        ]

        result = build_block_arrays(batch_specs, start_idx=0)

        assert result.num_blocks == 3
        # Check sampler types are correct
        assert result.types[0] == SamplerType.METROPOLIS_HASTINGS
        assert result.types[1] == SamplerType.DIRECT_CONJUGATE
        assert result.types[2] == SamplerType.METROPOLIS_HASTINGS


# ============================================================================
# COUPLED TRANSFORM SAMPLER TESTS
# ============================================================================

class TestCoupledTransformSpec:
    """Test COUPLED_TRANSFORM sampler type specifications."""

    def test_coupled_transform_allows_central_dispatch(self):
        """Test that COUPLED_TRANSFORM allows no per-block callbacks (uses central dispatch)."""
        # COUPLED_TRANSFORM can work without per-block callbacks if the model
        # registers a central coupled_transform_dispatch function
        spec = BlockSpec(
            size=2,
            sampler_type=SamplerType.COUPLED_TRANSFORM,
            proposal_type=ProposalType.CHAIN_MEAN,
            label="hyperparameters"
        )
        assert spec.sampler_type == SamplerType.COUPLED_TRANSFORM

    def test_coupled_transform_partial_callbacks_raises(self):
        """Test that partial per-block callbacks raises ValueError."""
        def dummy_indices(state, data):
            return jnp.array([0, 1])

        # Providing only some callbacks should raise
        with pytest.raises(ValueError, match="partial per-block callbacks"):
            BlockSpec(
                size=2,
                sampler_type=SamplerType.COUPLED_TRANSFORM,
                proposal_type=ProposalType.CHAIN_MEAN,
                coupled_indices_fn=dummy_indices,
                # Missing forward_transform_fn, log_jacobian_fn, coupled_log_prior_fn
            )

    def test_coupled_transform_full_callbacks_valid(self):
        """Test creating valid COUPLED_TRANSFORM spec with all callbacks."""
        def dummy_indices(state, data):
            return jnp.array([10, 11, 12])

        def dummy_transform(state, primary_idx, proposed, coupled_idx, data):
            return jnp.array([1.0, 2.0, 3.0])

        def dummy_jacobian(state, proposed, primary_idx, coupled_idx, data):
            return 0.0

        def dummy_prior(values):
            return -0.5 * jnp.sum(values**2)

        spec = BlockSpec(
            size=2,
            sampler_type=SamplerType.COUPLED_TRANSFORM,
            proposal_type=ProposalType.MCOV_SMOOTH,
            coupled_indices_fn=dummy_indices,
            forward_transform_fn=dummy_transform,
            log_jacobian_fn=dummy_jacobian,
            coupled_log_prior_fn=dummy_prior,
            label="hyperparameters"
        )

        assert spec.sampler_type == SamplerType.COUPLED_TRANSFORM
        assert spec.proposal_type == ProposalType.MCOV_SMOOTH
        assert spec.forward_transform_fn is not None
        assert spec.coupled_indices_fn is not None

    def test_coupled_transform_defaults_proposal_type(self):
        """Test that COUPLED_TRANSFORM defaults to SELF_MEAN proposal."""
        spec = BlockSpec(
            size=2,
            sampler_type=SamplerType.COUPLED_TRANSFORM,
        )
        assert spec.proposal_type == ProposalType.SELF_MEAN


# ============================================================================
# SETTINGS VALIDATION TESTS
# ============================================================================

class TestSettingsValidation:
    """Test settings handling and edge cases."""

    def test_default_settings_populated(self):
        """Test that missing settings get defaults."""
        batch_specs = [
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.MIXTURE)
            # No explicit settings - should use defaults
        ]
        settings_matrix = build_settings_matrix(batch_specs)

        # Should have default chain_prob
        assert settings_matrix[0, SettingSlot.CHAIN_PROB] == 0.5  # Default

    def test_custom_settings_override_defaults(self):
        """Test that custom settings override defaults."""
        batch_specs = [
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.MIXTURE,
                     settings={'chain_prob': 0.8, 'cov_mult': 2.0})
        ]
        settings_matrix = build_settings_matrix(batch_specs)

        assert settings_matrix[0, SettingSlot.CHAIN_PROB] == 0.8
        assert settings_matrix[0, SettingSlot.COV_MULT] == 2.0

    def test_multiple_blocks_different_settings(self):
        """Test that each block can have different settings."""
        batch_specs = [
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.MIXTURE,
                     settings={'chain_prob': 0.3}),
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.MIXTURE,
                     settings={'chain_prob': 0.7}),
            BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.MIXTURE,
                     settings={'chain_prob': 0.9}),
        ]
        settings_matrix = build_settings_matrix(batch_specs)

        assert settings_matrix[0, SettingSlot.CHAIN_PROB] == 0.3
        assert settings_matrix[1, SettingSlot.CHAIN_PROB] == 0.7
        assert settings_matrix[2, SettingSlot.CHAIN_PROB] == 0.9


# ============================================================================
# PARALLEL TEMPERING TESTS
# ============================================================================

class TestParallelTempering:
    """Test parallel tempering configuration and initialization."""

    def test_temperature_ladder_geometric_spacing(self):
        """Test that temperature ladder uses correct geometric spacing."""
        user_config = {
            'num_chains': 8,
            'num_chains_a': 4,
            'num_chains_b': 4,
            'num_superchains': 8,
            'save_likelihoods': False,
            'n_temperatures': 4,
            'beta_min': 0.1,
            'swap_every': 1,
        }
        master_key, init_key = gen_rng_keys(42)
        runtime_ctx = {
            'jnp_float_dtype': jnp.float32,
            'init_key': init_key,
            'master_key': master_key,
        }

        init_vec = np.random.randn(8).astype(np.float32)

        carry, new_config = initialize_mcmc_system(
            init_vec, user_config, runtime_ctx, num_gq=0, num_collect=10, num_blocks=1
        )

        # Temperature ladder is element 9 of carry tuple (15-element index process structure)
        temperature_ladder = np.array(carry[9])

        # Expected: beta_min^(i/(n_temps-1)) for i=0,1,2,3
        # [1.0, 0.1^(1/3), 0.1^(2/3), 0.1]
        expected = np.array([0.1 ** (i / 3) for i in range(4)])

        np.testing.assert_allclose(temperature_ladder, expected, rtol=1e-5)

        # First should be 1.0 (coldest), last should be beta_min (hottest)
        assert temperature_ladder[0] == 1.0
        np.testing.assert_allclose(temperature_ladder[-1], 0.1, rtol=1e-5)

    def test_temperature_assignments_split_across_groups(self):
        """Test that temperature assignments are correctly split across A/B groups."""
        user_config = {
            'num_chains': 8,
            'num_chains_a': 4,
            'num_chains_b': 4,
            'num_superchains': 8,
            'save_likelihoods': False,
            'n_temperatures': 4,
            'beta_min': 0.1,
            'swap_every': 1,
        }
        master_key, init_key = gen_rng_keys(42)
        runtime_ctx = {
            'jnp_float_dtype': jnp.float32,
            'init_key': init_key,
            'master_key': master_key,
        }

        init_vec = np.random.randn(8).astype(np.float32)

        carry, new_config = initialize_mcmc_system(
            init_vec, user_config, runtime_ctx, num_gq=0, num_collect=10, num_blocks=1
        )

        # temp_assignments_A is element 10, temp_assignments_B is element 11 (15-element structure)
        temp_assignments_A = np.array(carry[10])
        temp_assignments_B = np.array(carry[11])

        # With 8 chains total, 4 temperatures, we have 2 chains per temperature
        # Interleaved pattern: temp_assignments = [0, 1, 2, 3, 0, 1, 2, 3]
        # This ensures both A and B groups have all temperatures
        # After split at num_chains_a=4:
        #   Group A (first 4): [0, 1, 2, 3]
        #   Group B (last 4): [0, 1, 2, 3]
        expected_A = np.array([0, 1, 2, 3])
        expected_B = np.array([0, 1, 2, 3])

        np.testing.assert_array_equal(temp_assignments_A, expected_A)
        np.testing.assert_array_equal(temp_assignments_B, expected_B)

    def test_no_tempering_when_n_temperatures_is_one(self):
        """Test that n_temperatures=1 gives flat temperature ladder."""
        user_config = {
            'num_chains': 4,
            'num_chains_a': 2,
            'num_chains_b': 2,
            'num_superchains': 4,
            'save_likelihoods': False,
            'n_temperatures': 1,  # No tempering
            'beta_min': 0.1,
            'swap_every': 1,
        }
        master_key, init_key = gen_rng_keys(42)
        runtime_ctx = {
            'jnp_float_dtype': jnp.float32,
            'init_key': init_key,
            'master_key': master_key,
        }

        init_vec = np.random.randn(4).astype(np.float32)

        carry, new_config = initialize_mcmc_system(
            init_vec, user_config, runtime_ctx, num_gq=0, num_collect=10, num_blocks=1
        )

        temperature_ladder = np.array(carry[9])  # Index 9 in 15-element structure

        # Should be just [1.0] when n_temperatures=1
        assert len(temperature_ladder) == 1
        assert temperature_ladder[0] == 1.0

    def test_swap_counters_initialized_to_zero(self):
        """Test that swap accept/attempt counters are initialized to zero."""
        user_config = {
            'num_chains': 8,
            'num_chains_a': 4,
            'num_chains_b': 4,
            'num_superchains': 8,
            'save_likelihoods': False,
            'n_temperatures': 4,
            'beta_min': 0.1,
            'swap_every': 1,
        }
        master_key, init_key = gen_rng_keys(42)
        runtime_ctx = {
            'jnp_float_dtype': jnp.float32,
            'init_key': init_key,
            'master_key': master_key,
        }

        init_vec = np.random.randn(8).astype(np.float32)

        carry, new_config = initialize_mcmc_system(
            init_vec, user_config, runtime_ctx, num_gq=0, num_collect=10, num_blocks=1
        )

        # swap_accepts is element 12, swap_attempts is element 13 (15-element structure)
        swap_accepts = np.array(carry[12])
        swap_attempts = np.array(carry[13])

        # Should have n_temperatures - 1 pairs
        assert len(swap_accepts) == 3
        assert len(swap_attempts) == 3

        # All should be zero at initialization
        np.testing.assert_array_equal(swap_accepts, np.zeros(3))
        np.testing.assert_array_equal(swap_attempts, np.zeros(3))

    def test_tempering_config_defaults(self):
        """Test that tempering config uses correct defaults when not specified."""
        user_config = {
            'num_chains': 4,
            'num_chains_a': 2,
            'num_chains_b': 2,
            'num_superchains': 4,
            'save_likelihoods': False,
            # No tempering config specified - should use defaults
        }
        master_key, init_key = gen_rng_keys(42)
        runtime_ctx = {
            'jnp_float_dtype': jnp.float32,
            'init_key': init_key,
            'master_key': master_key,
        }

        init_vec = np.random.randn(4).astype(np.float32)

        carry, new_config = initialize_mcmc_system(
            init_vec, user_config, runtime_ctx, num_gq=0, num_collect=10, num_blocks=1
        )

        # Should default to n_temperatures=1 (no tempering)
        assert new_config['n_temperatures'] == 1

        # Temperature ladder should be just [1.0] (index 9 in 15-element structure)
        temperature_ladder = np.array(carry[9])
        assert len(temperature_ladder) == 1
        assert temperature_ladder[0] == 1.0

    def test_tempering_validation_rejects_invalid_n_temperatures(self):
        """Test that validation catches invalid n_temperatures."""
        config = {
            'posterior_id': 'test',
            'num_chains_a': 8,
            'num_chains_b': 8,
            'num_superchains': 4,
            'thin_iteration': 10,
            'num_collect': 100,
            'n_temperatures': 0,  # Invalid!
        }

        with pytest.raises(ValueError, match="n_temperatures must be >= 1"):
            validate_mcmc_config(config)

    def test_tempering_validation_rejects_invalid_beta_min(self):
        """Test that validation catches invalid beta_min."""
        config = {
            'posterior_id': 'test',
            'num_chains_a': 8,
            'num_chains_b': 8,
            'num_superchains': 4,
            'thin_iteration': 10,
            'num_collect': 100,
            'n_temperatures': 4,
            'beta_min': 1.5,  # Invalid! Must be in (0, 1]
        }

        with pytest.raises(ValueError, match="beta_min must be in"):
            validate_mcmc_config(config)

    def test_tempering_validation_rejects_too_many_temperatures(self):
        """Test that validation catches more temperatures than chains."""
        config = {
            'posterior_id': 'test',
            'num_chains_a': 2,
            'num_chains_b': 2,
            'num_superchains': 4,
            'thin_iteration': 10,
            'num_collect': 100,
            'n_temperatures': 8,  # Invalid! More temps than chains
        }

        with pytest.raises(ValueError, match="n_temperatures.*cannot exceed"):
            validate_mcmc_config(config)

    def test_checkpoint_roundtrip_with_tempering(self):
        """Test save_checkpoint and load_checkpoint preserve tempering state."""
        from bamcmc.checkpoint_helpers import save_checkpoint, load_checkpoint
        import tempfile
        import os

        states_A = np.random.randn(4, 5).astype(np.float32)
        keys_A = np.random.randint(0, 2**31, (4, 2), dtype=np.uint32)
        states_B = np.random.randn(4, 5).astype(np.float32)
        keys_B = np.random.randint(0, 2**31, (4, 2), dtype=np.uint32)
        history = np.zeros((100, 8, 10))
        lik_history = np.zeros((100, 8))
        temp_history = np.zeros((100, 8), dtype=np.int32)
        acceptance_counts = np.array([100, 200, 300], dtype=np.int32)
        iteration = 500

        # Tempering state
        temperature_ladder = np.array([1.0, 0.464, 0.215, 0.1], dtype=np.float32)
        temp_assignments_A = np.array([0, 1, 2, 3], dtype=np.int32)
        temp_assignments_B = np.array([0, 1, 2, 3], dtype=np.int32)
        swap_accepts = np.array([50, 40, 30], dtype=np.int32)
        swap_attempts = np.array([100, 100, 100], dtype=np.int32)
        swap_parity = 0

        # 15-element carry tuple
        carry = (
            states_A, keys_A, states_B, keys_B,
            history, lik_history, temp_history,
            acceptance_counts, iteration,
            temperature_ladder, temp_assignments_A, temp_assignments_B,
            swap_accepts, swap_attempts, swap_parity
        )

        mcmc_config = {
            'posterior_id': 'test_model',
            'num_params': 5,
            'num_chains_a': 4,
            'num_chains_b': 4,
            'num_superchains': 4,
            'subchains_per_super': 2,
            'n_temperatures': 4,
            'beta_min': 0.1,
            'swap_every': 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.npz')

            save_checkpoint(path, carry, mcmc_config)
            loaded = load_checkpoint(path)

            assert loaded['iteration'] == 500
            assert loaded['n_temperatures'] == 4
            np.testing.assert_array_almost_equal(
                loaded['temperature_ladder'], temperature_ladder
            )
            np.testing.assert_array_equal(loaded['temp_assignments_A'], temp_assignments_A)
            np.testing.assert_array_equal(loaded['temp_assignments_B'], temp_assignments_B)
            np.testing.assert_array_equal(loaded['swap_accepts'], swap_accepts)
            np.testing.assert_array_equal(loaded['swap_attempts'], swap_attempts)

    def test_output_filtering_beta1_chains(self):
        """Test that history array is sized for beta=1 chains only when tempering."""
        user_config = {
            'num_chains': 16,
            'num_chains_a': 8,
            'num_chains_b': 8,
            'num_superchains': 4,
            'save_likelihoods': False,
            'n_temperatures': 4,
            'beta_min': 0.3,
            'swap_every': 1,
            'n_chains_to_save': 4,  # 16 chains / 4 temps = 4 beta=1 chains
        }
        master_key, init_key = gen_rng_keys(42)
        runtime_ctx = {
            'jnp_float_dtype': jnp.float32,
            'init_key': init_key,
            'master_key': master_key,
        }

        init_vec = np.random.randn(16).astype(np.float32)
        num_collect = 10

        carry, new_config = initialize_mcmc_system(
            init_vec, user_config, runtime_ctx, num_gq=0, num_collect=num_collect, num_blocks=1
        )

        # History array is element 4 in carry
        history = carry[4]

        # Should have shape (num_collect, n_chains_to_save, num_params)
        # With 16 chains and 4 temps, only 4 beta=1 chains should be saved
        assert history.shape[0] == num_collect
        assert history.shape[1] == 4  # Only beta=1 chains
        assert new_config['n_chains_to_save'] == 4

    def test_output_filtering_no_tempering(self):
        """Test that all chains are saved when not using tempering."""
        user_config = {
            'num_chains': 8,
            'num_chains_a': 4,
            'num_chains_b': 4,
            'num_superchains': 4,
            'save_likelihoods': False,
            'n_temperatures': 1,  # No tempering
            'n_chains_to_save': 8,  # All chains saved
        }
        master_key, init_key = gen_rng_keys(42)
        runtime_ctx = {
            'jnp_float_dtype': jnp.float32,
            'init_key': init_key,
            'master_key': master_key,
        }

        init_vec = np.random.randn(8).astype(np.float32)
        num_collect = 10

        carry, new_config = initialize_mcmc_system(
            init_vec, user_config, runtime_ctx, num_gq=0, num_collect=num_collect, num_blocks=1
        )

        # History array is element 4 in carry
        history = carry[4]

        # Should have shape (num_collect, num_chains, num_params) - all chains
        assert history.shape[0] == num_collect
        assert history.shape[1] == 8  # All chains saved
        assert new_config['n_chains_to_save'] == 8
