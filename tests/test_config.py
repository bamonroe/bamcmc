"""
Configuration and Initialization Tests

Tests MCMC initialization logic and parallel tempering configuration:
- Standard vs nested chain initialization
- Temperature ladder geometry
- Temperature assignment splitting
- Tempering validation

Run with: pytest tests/test_config.py -v
"""

import numpy as np
import jax.numpy as jnp
import jax
import pytest

from bamcmc.mcmc.config import gen_rng_keys, initialize_mcmc_system
from bamcmc.error_handling import validate_mcmc_config


# ============================================================================
# INITIALIZATION TESTS (moved from test_unit.py)
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
# PARALLEL TEMPERING TESTS (moved from test_unit.py)
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

        temperature_ladder = np.array(carry[9])

        expected = np.array([0.1 ** (i / 3) for i in range(4)])

        np.testing.assert_allclose(temperature_ladder, expected, rtol=1e-5)

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

        temp_assignments_A = np.array(carry[10])
        temp_assignments_B = np.array(carry[11])

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
            'n_temperatures': 1,
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

        temperature_ladder = np.array(carry[9])

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

        swap_accepts = np.array(carry[12])
        swap_attempts = np.array(carry[13])

        assert len(swap_accepts) == 3
        assert len(swap_attempts) == 3

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

        assert new_config['n_temperatures'] == 1

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
            'n_temperatures': 0,
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
            'beta_min': 1.5,
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
            'n_temperatures': 8,
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

        temperature_ladder = np.array([1.0, 0.464, 0.215, 0.1], dtype=np.float32)
        temp_assignments_A = np.array([0, 1, 2, 3], dtype=np.int32)
        temp_assignments_B = np.array([0, 1, 2, 3], dtype=np.int32)
        swap_accepts = np.array([50, 40, 30], dtype=np.int32)
        swap_attempts = np.array([100, 100, 100], dtype=np.int32)
        swap_parity = 0

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
            'n_chains_to_save': 4,
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

        history = carry[4]

        assert history.shape[0] == num_collect
        assert history.shape[1] == 4
        assert new_config['n_chains_to_save'] == 4

    def test_output_filtering_no_tempering(self):
        """Test that all chains are saved when not using tempering."""
        user_config = {
            'num_chains': 8,
            'num_chains_a': 4,
            'num_chains_b': 4,
            'num_superchains': 4,
            'save_likelihoods': False,
            'n_temperatures': 1,
            'n_chains_to_save': 8,
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

        history = carry[4]

        assert history.shape[0] == num_collect
        assert history.shape[1] == 8
        assert new_config['n_chains_to_save'] == 8
