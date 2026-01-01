"""
Unit Tests for MCMC Backend Utilities

Tests individual components of the MCMC backend in isolation.
Run with: pytest tests/test_unit.py -v
"""

import numpy as np
import jax.numpy as jnp
import jax
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

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(chain_prob=0.5), dummy_grad_fn)
        proposal, log_ratio, new_key = mixture_proposal(operand)

        assert proposal.shape == (2,)
        assert isinstance(float(log_ratio), float)

    def test_mixture_proposal_chain_prob_zero(self):
        """Test mixture with chain_prob=0 (should behave like self_mean)."""
        from bamcmc.proposals import mixture_proposal

        current_block = jnp.array([1.0, 2.0])
        coupled_blocks, step_mean, step_cov = self._make_coupled_data(jnp.array([10.0, 10.0]), 0.1)
        block_mask = jnp.array([1.0, 1.0])

        proposals = []
        for i in range(100):
            key = jax.random.PRNGKey(i)
            operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                       make_settings_array(chain_prob=0.0), dummy_grad_fn)
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

        proposals = []
        for i in range(100):
            key = jax.random.PRNGKey(i)
            operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                       make_settings_array(chain_prob=1.0), dummy_grad_fn)
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

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(chain_prob=0.5), dummy_grad_fn)
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

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(uniform_weight=0.05, n_categories=10), dummy_grad_fn)
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

        proposals = []
        for i in range(200):
            key = jax.random.PRNGKey(i)
            operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                       make_settings_array(uniform_weight=0.01, n_categories=10), dummy_grad_fn)
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

        operand = (key, current_block, step_mean, step_cov, coupled_blocks, block_mask,
                   make_settings_array(uniform_weight=0.05, n_categories=5), dummy_grad_fn)
        _, log_ratio, _ = multinomial_proposal(operand)

        assert jnp.isfinite(log_ratio)
        assert abs(float(log_ratio)) < 10


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
        acceptance_counts = np.array([100, 200, 300], dtype=np.int32)
        iteration = 500

        carry = (states_A, keys_A, states_B, keys_B,
                 history, lik_history, acceptance_counts, iteration)

        mcmc_config = {
            'posterior_id': 'test_model',
            'num_params': 5,
            'num_chains_a': 10,
            'num_chains_b': 10,
            'num_superchains': 4,
            'SUBCHAINS_PER_SUPER': 5,
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
