"""
Reset Utils Tests - Chain Reset and Recovery Utilities

Tests the chain reset mechanism:
- compute_chain_statistics: mean/std computation
- get_discrete_param_indices: legacy model lookups
- generate_reset_states: state generation with discrete/continuous handling
- generate_reset_vector: flat vector generation with superchain replication
- select_diverse_states: evenly-spaced chain selection
- print_reset_summary: summary output (smoke test)
- reset_from_checkpoint / init_from_prior: high-level wrappers via disk I/O

Run with: pytest tests/test_reset_utils.py -v
"""

import numpy as np
import pytest
import tempfile
import os

from bamcmc.reset_utils import (
    compute_chain_statistics,
    get_discrete_param_indices,
    generate_reset_states,
    generate_reset_vector,
    select_diverse_states,
    print_reset_summary,
    reset_from_checkpoint,
    init_from_prior,
    _get_legacy_special_indices,
)


# ============================================================================
# HELPERS
# ============================================================================

def _make_checkpoint(n_chains_a=4, n_chains_b=4, n_params=10,
                     iteration=100, posterior_id='test_model', rng_seed=42):
    """Create a minimal checkpoint dict for testing."""
    rng = np.random.RandomState(rng_seed)
    return {
        'states_A': rng.randn(n_chains_a, n_params).astype(np.float64),
        'states_B': rng.randn(n_chains_b, n_params).astype(np.float64),
        'keys_A': rng.randint(0, 2**31, (n_chains_a, 2), dtype=np.uint32),
        'keys_B': rng.randint(0, 2**31, (n_chains_b, 2), dtype=np.uint32),
        'iteration': iteration,
        'acceptance_counts': np.array([50, 60, 70], dtype=np.int32),
        'posterior_id': posterior_id,
        'num_params': n_params,
        'num_chains_a': n_chains_a,
        'num_chains_b': n_chains_b,
        'num_superchains': 4,
        'subchains_per_super': 2,
    }


def _save_checkpoint_to_disk(checkpoint, directory):
    """Save a checkpoint dict to a .npz file, return the path."""
    filepath = os.path.join(directory, 'checkpoint.npz')
    np.savez_compressed(filepath, **checkpoint)
    return filepath


# ============================================================================
# compute_chain_statistics
# ============================================================================

class TestComputeChainStatistics:
    """Test cross-chain mean and std computation."""

    def test_basic(self):
        """Mean and std should be computed across all chains."""
        states_A = np.array([[1.0, 2.0], [3.0, 4.0]])
        states_B = np.array([[5.0, 6.0], [7.0, 8.0]])

        stats = compute_chain_statistics(states_A, states_B)

        expected_mean = np.mean([[1, 2], [3, 4], [5, 6], [7, 8]], axis=0)
        expected_std = np.std([[1, 2], [3, 4], [5, 6], [7, 8]], axis=0)

        np.testing.assert_allclose(stats['mean'], expected_mean)
        np.testing.assert_allclose(stats['std'], expected_std)
        assert stats['n_chains'] == 4
        assert stats['n_params'] == 2

    def test_single_chain_per_group(self):
        """Should work with one chain per group."""
        states_A = np.array([[1.0, 2.0]])
        states_B = np.array([[3.0, 4.0]])

        stats = compute_chain_statistics(states_A, states_B)

        assert stats['n_chains'] == 2
        np.testing.assert_allclose(stats['mean'], [2.0, 3.0])

    def test_unequal_groups(self):
        """Should work with different numbers of chains in A and B."""
        states_A = np.array([[1.0], [2.0], [3.0]])
        states_B = np.array([[4.0]])

        stats = compute_chain_statistics(states_A, states_B)

        assert stats['n_chains'] == 4
        np.testing.assert_allclose(stats['mean'], [2.5])


# ============================================================================
# get_discrete_param_indices / _get_legacy_special_indices
# ============================================================================

class TestGetDiscreteParamIndices:
    """Test discrete parameter index lookup."""

    def test_unknown_model_returns_empty(self):
        """Unknown model should return empty list."""
        indices = get_discrete_param_indices('unknown_model', 10)
        assert indices == []

    def test_mixture_3model_z_indices(self):
        """3-model mixture should have z indices at expected positions."""
        n_subjects = 5
        indices = get_discrete_param_indices('mixture_3model_bhm', n_subjects)

        # z is at offset 19 in each 20-param subject block
        expected = [s * 20 + 19 for s in range(n_subjects)]
        assert indices == expected

    def test_mixture_4model_z_indices(self):
        """4-model mixture should have z indices at expected positions."""
        n_subjects = 3
        indices = get_discrete_param_indices('mixture_4model_bhm', n_subjects)

        # z is at offset 26 in each 27-param subject block
        expected = [s * 27 + 26 for s in range(n_subjects)]
        assert indices == expected

    def test_mixture_2model_z_indices(self):
        """2-model mixture should have z indices at expected positions."""
        n_subjects = 4
        indices = get_discrete_param_indices('mixture_2model_bhm', n_subjects)

        # z is at offset 13 in each 14-param subject block
        expected = [s * 14 + 13 for s in range(n_subjects)]
        assert indices == expected

    def test_legacy_fallback_emits_deprecation_warning(self):
        """Legacy hardcoded path should emit DeprecationWarning."""
        import warnings as _w
        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            get_discrete_param_indices('mixture_3model_bhm', 2)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "hardcoded" in str(deprecation_warnings[0].message).lower()


class TestLegacySpecialIndices:
    """Test the legacy special indices function directly."""

    def test_3model_keys(self):
        """Should return z, pi, and r indices for 3-model."""
        result = _get_legacy_special_indices('mixture_3model_bhm', 2)
        assert 'z_indices' in result
        assert 'pi_indices' in result
        assert 'r_indices' in result
        assert len(result['z_indices']) == 2
        assert len(result['pi_indices']) == 3  # 3 models -> 3 pi values

    def test_4model_keys(self):
        """Should return z, pi, and r indices for 4-model."""
        result = _get_legacy_special_indices('mixture_4model_bhm', 2)
        assert len(result['z_indices']) == 2
        assert len(result['pi_indices']) == 4  # 4 models -> 4 pi values

    def test_unknown_model_empty(self):
        """Unknown models should return empty lists for all keys."""
        result = _get_legacy_special_indices('some_other_model', 10)
        assert result['z_indices'] == []
        assert result['pi_indices'] == []
        assert result['r_indices'] == []


# ============================================================================
# generate_reset_states
# ============================================================================

class TestGenerateResetStates:
    """Test reset state generation."""

    def test_output_shape(self):
        """Output should be (K, n_params)."""
        cp = _make_checkpoint(n_params=10)
        states = generate_reset_states(cp, 'unknown_model', 0, K=3, rng_seed=42)
        assert states.shape == (3, 10)

    def test_reproducibility(self):
        """Same seed should give same results."""
        cp = _make_checkpoint()
        s1 = generate_reset_states(cp, 'unknown_model', 0, K=4, rng_seed=99)
        s2 = generate_reset_states(cp, 'unknown_model', 0, K=4, rng_seed=99)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        """Different seeds should give different results."""
        cp = _make_checkpoint()
        s1 = generate_reset_states(cp, 'unknown_model', 0, K=4, rng_seed=1)
        s2 = generate_reset_states(cp, 'unknown_model', 0, K=4, rng_seed=2)
        assert not np.allclose(s1, s2)

    def test_states_near_mean(self):
        """Generated states should be centered near the cross-chain mean."""
        cp = _make_checkpoint(n_chains_a=20, n_chains_b=20, n_params=5, rng_seed=42)
        states = generate_reset_states(cp, 'unknown_model', 0, K=100, rng_seed=42)

        all_chains = np.vstack([cp['states_A'], cp['states_B']])
        chain_mean = np.mean(all_chains, axis=0)

        # Mean of many generated states should be close to the chain mean
        np.testing.assert_allclose(np.mean(states, axis=0), chain_mean, atol=0.3)

    def test_noise_scale_affects_spread(self):
        """Larger noise_scale should produce more spread."""
        cp = _make_checkpoint(n_chains_a=10, n_chains_b=10)

        states_small = generate_reset_states(cp, 'unknown_model', 0, K=200,
                                             noise_scale=0.1, rng_seed=42)
        states_large = generate_reset_states(cp, 'unknown_model', 0, K=200,
                                             noise_scale=2.0, rng_seed=42)

        var_small = np.var(states_small, axis=0).mean()
        var_large = np.var(states_large, axis=0).mean()

        assert var_large > var_small

    def test_discrete_params_are_integers(self):
        """Discrete params should be sampled as integers from empirical dist."""
        n_subjects = 2
        n_params = 2 * 20 + 3 + 12 + 14 + 12  # 2 subjects + hypers for 3-model
        cp = _make_checkpoint(n_chains_a=4, n_chains_b=4, n_params=n_params)

        # Set z-params (at offsets 19 and 39) to integer values across chains
        for arr in [cp['states_A'], cp['states_B']]:
            arr[:, 19] = np.random.choice([1, 2, 3], size=arr.shape[0])
            arr[:, 39] = np.random.choice([1, 2, 3], size=arr.shape[0])

        states = generate_reset_states(cp, 'mixture_3model_bhm', n_subjects,
                                       K=20, rng_seed=42)

        # z-params should be integers from {1, 2, 3}
        for z_idx in [19, 39]:
            z_vals = states[:, z_idx]
            assert all(v in [1, 2, 3] for v in z_vals), \
                f"z values at index {z_idx} should be from {{1,2,3}}, got {np.unique(z_vals)}"


# ============================================================================
# generate_reset_vector
# ============================================================================

class TestGenerateResetVector:
    """Test flat initial vector generation."""

    def test_output_shape(self):
        """Output should be flat with K * M * n_params elements."""
        cp = _make_checkpoint(n_params=5)
        vec = generate_reset_vector(cp, 'unknown_model', 0, K=3, M=2, rng_seed=42)
        assert vec.shape == (3 * 2 * 5,)

    def test_subchain_replication(self):
        """Each superchain's M subchains should have identical states."""
        cp = _make_checkpoint(n_params=4)
        K, M = 3, 4
        vec = generate_reset_vector(cp, 'unknown_model', 0, K=K, M=M, rng_seed=42)

        # Reshape to (K*M, n_params)
        states = vec.reshape(K * M, 4)

        for k in range(K):
            base = states[k * M]
            for m in range(1, M):
                np.testing.assert_array_equal(states[k * M + m], base,
                    err_msg=f"Subchain {m} of superchain {k} should match base")

    def test_superchains_differ(self):
        """Different superchains should have different states."""
        cp = _make_checkpoint(n_params=10)
        K, M = 4, 2
        vec = generate_reset_vector(cp, 'unknown_model', 0, K=K, M=M, rng_seed=42)
        states = vec.reshape(K * M, 10)

        # Compare first subchain of each superchain
        for k1 in range(K):
            for k2 in range(k1 + 1, K):
                assert not np.allclose(states[k1 * M], states[k2 * M]), \
                    f"Superchains {k1} and {k2} should differ"


# ============================================================================
# select_diverse_states
# ============================================================================

class TestSelectDiverseStates:
    """Test diverse state selection from checkpoint chains."""

    def test_output_shape(self):
        """Output should be (K, n_params)."""
        cp = _make_checkpoint(n_chains_a=5, n_chains_b=5, n_params=3)
        states = select_diverse_states(cp, K=4, rng_seed=42)
        assert states.shape == (4, 3)

    def test_select_all_chains(self):
        """K == n_chains should return all chains."""
        cp = _make_checkpoint(n_chains_a=3, n_chains_b=3, n_params=2)
        states = select_diverse_states(cp, K=6, rng_seed=42)
        assert states.shape == (6, 2)

    def test_k_exceeds_chains_raises(self):
        """K > n_chains should raise ValueError."""
        cp = _make_checkpoint(n_chains_a=2, n_chains_b=2)
        with pytest.raises(ValueError, match="Cannot select"):
            select_diverse_states(cp, K=10)

    def test_returns_actual_chain_states(self):
        """Selected states should be actual chain states (not interpolated)."""
        cp = _make_checkpoint(n_chains_a=4, n_chains_b=4, n_params=3, rng_seed=42)
        all_chains = np.vstack([cp['states_A'], cp['states_B']])

        states = select_diverse_states(cp, K=3, rng_seed=42)

        # Each selected state should exactly match some chain
        for i in range(states.shape[0]):
            matches = np.any(np.all(np.isclose(all_chains, states[i]), axis=1))
            assert matches, f"Selected state {i} doesn't match any chain"

    def test_no_duplicates(self):
        """Selected states should be unique."""
        cp = _make_checkpoint(n_chains_a=10, n_chains_b=10, n_params=5, rng_seed=42)
        states = select_diverse_states(cp, K=8, rng_seed=42)

        # Check all pairs are different
        for i in range(states.shape[0]):
            for j in range(i + 1, states.shape[0]):
                assert not np.allclose(states[i], states[j]), \
                    f"States {i} and {j} are duplicates"


# ============================================================================
# print_reset_summary (smoke test)
# ============================================================================

class TestPrintResetSummary:
    """Smoke test for print_reset_summary."""

    def test_runs_without_error(self, caplog):
        """Should print summary without raising."""
        import logging
        cp = _make_checkpoint(n_params=10)
        with caplog.at_level(logging.INFO, logger='bamcmc'):
            print_reset_summary(cp, 'unknown_model', 0)

        assert 'Reset Summary' in caplog.text
        assert 'Parameters: 10' in caplog.text

    def test_with_mixture_model(self, caplog):
        """Should print z distribution for mixture models."""
        import logging
        n_subjects = 2
        n_params = 2 * 20 + 3 + 12 + 14 + 12
        cp = _make_checkpoint(n_params=n_params)

        # Set z values to known integers
        for arr in [cp['states_A'], cp['states_B']]:
            arr[:, 19] = 1.0
            arr[:, 39] = 2.0

        with caplog.at_level(logging.INFO, logger='bamcmc'):
            print_reset_summary(cp, 'mixture_3model_bhm', n_subjects)

        assert 'Discrete parameter distribution' in caplog.text


# ============================================================================
# reset_from_checkpoint (disk I/O)
# ============================================================================

class TestResetFromCheckpoint:
    """Test high-level reset from checkpoint file."""

    def test_roundtrip(self):
        """Should load checkpoint from disk and generate reset vector."""
        cp = _make_checkpoint(n_params=5, posterior_id='test_model')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_checkpoint_to_disk(cp, tmpdir)
            init_vec, info = reset_from_checkpoint(
                path, 'test_model', 0, K=2, M=3, rng_seed=42
            )

        assert init_vec.shape == (2 * 3 * 5,)
        assert info['source_iteration'] == 100
        assert info['reset_K'] == 2
        assert info['reset_M'] == 3

    def test_model_mismatch_warns(self, caplog):
        """Should warn when checkpoint model differs from requested."""
        import logging
        cp = _make_checkpoint(posterior_id='model_A')

        with caplog.at_level(logging.WARNING, logger='bamcmc'):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = _save_checkpoint_to_disk(cp, tmpdir)
                reset_from_checkpoint(path, 'model_B', 0, K=2, M=2, rng_seed=42)

        assert 'Checkpoint model' in caplog.text


# ============================================================================
# init_from_prior (disk I/O)
# ============================================================================

class TestInitFromPrior:
    """Test initialization from prior checkpoint."""

    def test_roundtrip(self):
        """Should load prior checkpoint and generate init vector."""
        cp = _make_checkpoint(n_chains_a=10, n_chains_b=10, n_params=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_checkpoint_to_disk(cp, tmpdir)
            init_vec, info = init_from_prior(path, K=4, M=3, rng_seed=42)

        assert init_vec.shape == (4 * 3 * 5,)
        assert info['K'] == 4
        assert info['M'] == 3
        assert info['source_iteration'] == 100

    def test_states_are_actual_chains(self):
        """Init states should be actual prior chain states, not synthetic."""
        cp = _make_checkpoint(n_chains_a=10, n_chains_b=10, n_params=3, rng_seed=42)
        all_chains = np.vstack([cp['states_A'], cp['states_B']])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_checkpoint_to_disk(cp, tmpdir)
            init_vec, _ = init_from_prior(path, K=4, M=2, rng_seed=42)

        states = init_vec.reshape(4 * 2, 3)

        # Each unique superchain state should match an actual chain
        for k in range(4):
            state = states[k * 2]  # first subchain of each superchain
            matches = np.any(np.all(np.isclose(all_chains, state), axis=1))
            assert matches, f"Superchain {k} state doesn't match any prior chain"
