"""
Checkpoint Tests - Compatibility Validation and Edge Cases

Tests checkpoint save/load functionality:
- Shape mismatch detection
- Parameter count validation
- Model compatibility checks

Run with: pytest tests/test_checkpoints.py -v
"""

import numpy as np
import jax.numpy as jnp
import pytest
import tempfile
import os

from bamcmc.checkpoint_helpers import (
    save_checkpoint,
    load_checkpoint,
    initialize_from_checkpoint,
    scan_checkpoints,
    clean_model_files,
)
from bamcmc.mcmc.config import gen_rng_keys


# ============================================================================
# CHECKPOINT COMPATIBILITY TESTS
# ============================================================================

class TestCheckpointCompatibility:
    """Test checkpoint validation catches incompatible configurations."""

    def _make_checkpoint_data(self, n_chains_a=10, n_chains_b=10, n_params=5,
                               iteration=100, posterior_id='test_model'):
        """Create checkpoint data for testing."""
        states_A = np.random.randn(n_chains_a, n_params).astype(np.float32)
        states_B = np.random.randn(n_chains_b, n_params).astype(np.float32)
        keys_A = np.random.randint(0, 2**31, (n_chains_a, 2), dtype=np.uint32)
        keys_B = np.random.randint(0, 2**31, (n_chains_b, 2), dtype=np.uint32)

        num_blocks = 3
        history = np.zeros((100, n_chains_a + n_chains_b, n_params))
        lik_history = np.zeros((100, n_chains_a + n_chains_b))
        acceptance_counts = np.array([100, 200, 300], dtype=np.int32)

        carry = (states_A, keys_A, states_B, keys_B,
                 history, lik_history, acceptance_counts, iteration)

        user_config = {
            'posterior_id': posterior_id,
            'num_params': n_params,
            'num_chains_a': n_chains_a,
            'num_chains_b': n_chains_b,
            'num_superchains': 4,
            'subchains_per_super': 5,
        }

        return carry, user_config

    def test_save_load_roundtrip(self):
        """Test that save/load preserves all data correctly."""
        carry, user_config = self._make_checkpoint_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_checkpoint.npz')

            save_checkpoint(path, carry, user_config)
            loaded = load_checkpoint(path)

            assert loaded['iteration'] == 100
            assert loaded['posterior_id'] == 'test_model'
            assert loaded['num_params'] == 5
            assert loaded['num_chains_a'] == 10
            assert loaded['num_chains_b'] == 10
            np.testing.assert_array_equal(loaded['states_A'], carry[0])
            np.testing.assert_array_equal(loaded['states_B'], carry[2])

    def test_posterior_id_mismatch_raises(self):
        """Test that mismatched posterior_id raises ValueError."""
        carry, user_config = self._make_checkpoint_data(posterior_id='model_A')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.npz')
            save_checkpoint(path, carry, user_config)
            checkpoint = load_checkpoint(path)

            # Try to initialize with different model
            wrong_config = user_config.copy()
            wrong_config['posterior_id'] = 'model_B'  # Different model!
            wrong_config['num_chains'] = 20
            wrong_config['save_likelihoods'] = False

            master_key, init_key = gen_rng_keys(42)
            runtime_ctx = {
                'jnp_float_dtype': jnp.float32,
                'init_key': init_key,
                'master_key': master_key,
            }

            with pytest.raises(ValueError, match="posterior.*doesn't match"):
                initialize_from_checkpoint(checkpoint, wrong_config, runtime_ctx,
                                          num_gq=0, num_collect=100, num_blocks=3)

    def test_chain_count_mismatch_raises(self):
        """Test that mismatched chain counts raise ValueError."""
        carry, user_config = self._make_checkpoint_data(n_chains_a=10, n_chains_b=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.npz')
            save_checkpoint(path, carry, user_config)
            checkpoint = load_checkpoint(path)

            # Try to initialize with different chain count
            wrong_config = user_config.copy()
            wrong_config['num_chains_a'] = 20  # Different!
            wrong_config['num_chains'] = 30
            wrong_config['save_likelihoods'] = False

            master_key, init_key = gen_rng_keys(42)
            runtime_ctx = {
                'jnp_float_dtype': jnp.float32,
                'init_key': init_key,
                'master_key': master_key,
            }

            with pytest.raises(ValueError, match="A-chains"):
                initialize_from_checkpoint(checkpoint, wrong_config, runtime_ctx,
                                          num_gq=0, num_collect=100, num_blocks=3)

    def test_b_chain_count_mismatch_raises(self):
        """Test that mismatched B-chain counts raise ValueError."""
        carry, user_config = self._make_checkpoint_data(n_chains_a=10, n_chains_b=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.npz')
            save_checkpoint(path, carry, user_config)
            checkpoint = load_checkpoint(path)

            wrong_config = user_config.copy()
            wrong_config['num_chains_b'] = 5  # Different!
            wrong_config['num_chains'] = 15
            wrong_config['save_likelihoods'] = False

            master_key, init_key = gen_rng_keys(42)
            runtime_ctx = {
                'jnp_float_dtype': jnp.float32,
                'init_key': init_key,
                'master_key': master_key,
            }

            with pytest.raises(ValueError, match="B-chains"):
                initialize_from_checkpoint(checkpoint, wrong_config, runtime_ctx,
                                          num_gq=0, num_collect=100, num_blocks=3)

    def test_valid_checkpoint_initializes_successfully(self):
        """Test that valid checkpoint initializes without error."""
        carry, user_config = self._make_checkpoint_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.npz')
            save_checkpoint(path, carry, user_config)
            checkpoint = load_checkpoint(path)

            config = user_config.copy()
            config['num_chains'] = 20
            config['save_likelihoods'] = False

            master_key, init_key = gen_rng_keys(42)
            runtime_ctx = {
                'jnp_float_dtype': jnp.float32,
                'init_key': init_key,
                'master_key': master_key,
            }

            # Should not raise
            initial_carry, updated_config = initialize_from_checkpoint(
                checkpoint, config, runtime_ctx,
                num_gq=0, num_collect=100, num_blocks=3
            )

            # Verify carry structure (13 elements with tempering state)
            assert len(initial_carry) == 13
            assert initial_carry[0].shape == (10, 5)  # states_A
            assert initial_carry[2].shape == (10, 5)  # states_B
            # Elements 8-12 are tempering state (temp_ladder, temp_assign_A/B, swap_accepts/attempts)

    def test_missing_checkpoint_raises(self):
        """Test that loading non-existent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint('/nonexistent/path/checkpoint.npz')


# ============================================================================
# CHECKPOINT SCANNING TESTS
# ============================================================================

class TestCheckpointScanning:
    """Test checkpoint file scanning functionality."""

    def test_scan_empty_directory(self):
        """Test scanning an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = scan_checkpoints(tmpdir, 'test_model')

            assert result['checkpoint_files'] == []
            assert result['history_files'] == []
            assert result['latest_run_index'] == -1
            assert result['latest_checkpoint'] is None

    def test_scan_finds_checkpoints(self):
        """Test that scanning finds checkpoint files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some checkpoint files
            for i in [1, 3, 5]:
                path = os.path.join(tmpdir, f'my_model_checkpoint{i}.npz')
                np.savez_compressed(path, dummy=np.array([1]))

            result = scan_checkpoints(tmpdir, 'my_model')

            assert len(result['checkpoint_files']) == 3
            assert result['latest_run_index'] == 5
            assert 'checkpoint5.npz' in result['latest_checkpoint']

    def test_scan_finds_history_files(self):
        """Test that scanning finds history files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some history files
            for i in [0, 1, 2]:
                path = os.path.join(tmpdir, f'my_model_history_{i:03d}.npz')
                np.savez_compressed(path, dummy=np.array([1]))

            result = scan_checkpoints(tmpdir, 'my_model')

            assert len(result['history_files']) == 3

    def test_scan_ignores_other_models(self):
        """Test that scanning ignores files from other models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files for different models
            np.savez_compressed(os.path.join(tmpdir, 'model_A_checkpoint1.npz'), d=1)
            np.savez_compressed(os.path.join(tmpdir, 'model_B_checkpoint1.npz'), d=1)
            np.savez_compressed(os.path.join(tmpdir, 'model_A_history_000.npz'), d=1)

            result = scan_checkpoints(tmpdir, 'model_A')

            assert len(result['checkpoint_files']) == 1
            assert len(result['history_files']) == 1


# ============================================================================
# CLEAN FILES TESTS
# ============================================================================

class TestCleanModelFiles:
    """Test checkpoint/history file cleaning."""

    def test_clean_all_deletes_everything(self):
        """Test that mode='all' deletes all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            for i in range(3):
                np.savez_compressed(os.path.join(tmpdir, f'model_checkpoint{i}.npz'), d=1)
                np.savez_compressed(os.path.join(tmpdir, f'model_history_{i:03d}.npz'), d=1)

            result = clean_model_files(tmpdir, 'model', mode='all')

            assert len(result['deleted_checkpoints']) == 3
            assert len(result['deleted_histories']) == 3
            assert result['kept_checkpoint'] is None

            # Verify files are gone
            scan = scan_checkpoints(tmpdir, 'model')
            assert scan['checkpoint_files'] == []
            assert scan['history_files'] == []

    def test_clean_keep_latest_preserves_checkpoint(self):
        """Test that mode='keep_latest' preserves most recent checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            for i in range(3):
                np.savez_compressed(os.path.join(tmpdir, f'model_checkpoint{i}.npz'), d=1)
                np.savez_compressed(os.path.join(tmpdir, f'model_history_{i:03d}.npz'), d=1)

            result = clean_model_files(tmpdir, 'model', mode='keep_latest')

            assert len(result['deleted_checkpoints']) == 2
            assert len(result['deleted_histories']) == 3  # All histories deleted
            assert result['kept_checkpoint'] is not None
            assert 'checkpoint2.npz' in result['kept_checkpoint']

            # Verify only latest checkpoint remains
            scan = scan_checkpoints(tmpdir, 'model')
            assert len(scan['checkpoint_files']) == 1
            assert scan['latest_run_index'] == 2


# ============================================================================
# METADATA TESTS
# ============================================================================

class TestCheckpointMetadata:
    """Test checkpoint metadata handling."""

    def test_metadata_saved_and_loaded(self):
        """Test that optional metadata is preserved."""
        carry, user_config = TestCheckpointCompatibility()._make_checkpoint_data()

        metadata = {
            'custom_field': 'test_value',
            'numeric_field': 42,
            'list_field': [1, 2, 3],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.npz')
            save_checkpoint(path, carry, user_config, metadata=metadata)
            loaded = load_checkpoint(path)

            assert 'metadata' in loaded
            assert loaded['metadata']['custom_field'] == 'test_value'
            assert loaded['metadata']['numeric_field'] == 42

    def test_checkpoint_without_metadata(self):
        """Test loading checkpoint without metadata works."""
        carry, user_config = TestCheckpointCompatibility()._make_checkpoint_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.npz')
            save_checkpoint(path, carry, user_config)  # No metadata
            loaded = load_checkpoint(path)

            # Should not have metadata key or it should be None-ish
            assert 'metadata' not in loaded or loaded.get('metadata') is None
