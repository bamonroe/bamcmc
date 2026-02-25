"""
Unit Tests for MCMC Backend Utilities

Tests individual components of the MCMC backend in isolation:
- Nested R-hat computation
- Block array construction
- Proposal settings extraction
- Edge cases and boundary conditions
- Direct sampler and coupled transform specifications
- Settings validation

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
        n_samples = 2
        K, M = 2, 2
        history = np.random.randn(n_samples, K * M, 1)

        rhat = compute_nested_rhat(jnp.array(history), K=K, M=M)
        assert rhat.shape == (1,)

    def test_nested_rhat_many_chains_few_samples(self):
        """Test R-hat with many chains but few samples."""
        n_samples = 10
        K, M = 50, 1
        history = np.random.randn(n_samples, K * M, 2)

        rhat = compute_nested_rhat(jnp.array(history), K=K, M=M)
        assert rhat.shape == (2,)

    def test_single_chain_initialization(self):
        """Test that num_chains=1 initializes correctly."""
        user_config = {
            'num_chains': 2,
            'num_chains_a': 1,
            'num_chains_b': 1,
            'num_superchains': 2,
            'save_likelihoods': False,
        }
        master_key, init_key = gen_rng_keys(42)
        runtime_ctx = {
            'jnp_float_dtype': jnp.float32,
            'init_key': init_key,
            'master_key': master_key,
        }

        init_vec = np.array([10, 20], dtype=np.float32)

        carry, new_config = initialize_mcmc_system(
            init_vec, user_config, runtime_ctx, num_gq=0, num_collect=10, num_blocks=1
        )

        states_A = carry[0]
        states_B = carry[2]

        assert states_A.shape[0] == 1
        assert states_B.shape[0] == 1

    def test_empty_block_spec_list(self):
        """Test that empty block specs raises ValueError."""
        with pytest.raises(ValueError, match="Empty block specifications"):
            build_block_arrays([], start_idx=0)

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
            )

    def test_mh_sampler_defaults_proposal_type(self):
        """Test that MH sampler defaults to SELF_MEAN if proposal_type not specified."""
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

        with pytest.raises(ValueError, match="partial per-block callbacks"):
            BlockSpec(
                size=2,
                sampler_type=SamplerType.COUPLED_TRANSFORM,
                proposal_type=ProposalType.CHAIN_MEAN,
                coupled_indices_fn=dummy_indices,
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
        ]
        settings_matrix = build_settings_matrix(batch_specs)

        assert settings_matrix[0, SettingSlot.CHAIN_PROB] == 0.5

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
# LOGGING CONFIGURATION TESTS
# ============================================================================

class TestLoggingConfig:
    """Test that the bamcmc logger is properly configured."""

    def test_logger_exists(self):
        """The bamcmc logger should be configured on import."""
        import logging
        logger = logging.getLogger('bamcmc')
        assert logger.handlers, "bamcmc logger should have at least one handler"
        assert logger.level == logging.INFO

    def test_logger_output_captured(self, caplog):
        """Logger output should be capturable via standard logging."""
        import logging
        logger = logging.getLogger('bamcmc')
        with caplog.at_level(logging.INFO, logger='bamcmc'):
            logger.info("test message")
        assert "test message" in caplog.text

    def test_logger_verbosity_control(self, caplog):
        """Setting level to WARNING should suppress INFO messages."""
        import logging
        logger = logging.getLogger('bamcmc')
        original_level = logger.level
        try:
            logger.setLevel(logging.WARNING)
            with caplog.at_level(logging.WARNING, logger='bamcmc'):
                logger.info("should not appear")
                logger.warning("should appear")
            assert "should not appear" not in caplog.text
            assert "should appear" in caplog.text
        finally:
            logger.setLevel(original_level)
