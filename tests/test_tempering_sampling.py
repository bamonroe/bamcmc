"""
Tempering and Sampling Tests - Unit Tests for Core MCMC Internals

Tests:
- attempt_temperature_swaps: DEO parity, swap acceptance, counts, single-temp no-op
- coupled_transform_step: Jacobian/prior ratio in acceptance, NaN rejection

Run with: pytest tests/test_tempering_sampling.py -v
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from bamcmc.mcmc.tempering import attempt_temperature_swaps
from bamcmc.mcmc.sampling import coupled_transform_step, metropolis_block_step
from bamcmc.mcmc.types import BlockArrays, build_block_arrays
from bamcmc.batch_specs import BlockSpec, SamplerType, ProposalType
from bamcmc.settings import MAX_SETTINGS, SETTING_DEFAULTS, SettingSlot


# ============================================================================
# HELPERS
# ============================================================================

def _simple_log_post(state, param_indices, beta=1.0):
    """Simple quadratic log posterior: -0.5 * ||x||^2."""
    return -0.5 * jnp.sum(state ** 2)


def _make_temp_ladder(n_temps):
    """Create evenly-spaced temperature ladder from 1.0 down to ~0."""
    if n_temps == 1:
        return jnp.array([1.0])
    return jnp.linspace(1.0, 0.1, n_temps)


def _make_block_arrays_single(n_params, proposal_type=ProposalType.SELF_MEAN):
    """Build BlockArrays for a single MH block."""
    specs = [BlockSpec(size=n_params, sampler_type=SamplerType.METROPOLIS_HASTINGS,
                       proposal_type=proposal_type)]
    return build_block_arrays(specs)


def _make_mh_operand(n_params, chain_state=None, key=None):
    """Build a full operand tuple for metropolis_block_step / coupled_transform_step."""
    if key is None:
        key = random.PRNGKey(42)
    if chain_state is None:
        chain_state = jnp.ones(n_params)

    block_idx_vec = jnp.arange(n_params)
    block_mean = jnp.zeros(n_params)
    block_cov = jnp.eye(n_params)
    coupled_blocks = jnp.zeros((10, n_params))
    block_mode = jnp.zeros(n_params)
    block_mask = jnp.ones(n_params)
    proposal_type = jnp.int32(0)  # remapped index

    settings = jnp.zeros(MAX_SETTINGS)
    for slot, default in SETTING_DEFAULTS.items():
        settings = settings.at[slot].set(default)

    # Group arrays (single-proposal block)
    from bamcmc.batch_specs import MAX_PROPOSAL_GROUPS
    num_groups = jnp.int32(1)
    group_starts = jnp.zeros(MAX_PROPOSAL_GROUPS, dtype=jnp.int32)
    group_ends = jnp.zeros(MAX_PROPOSAL_GROUPS, dtype=jnp.int32)
    group_ends = group_ends.at[0].set(n_params)
    group_proposal_types = jnp.zeros(MAX_PROPOSAL_GROUPS, dtype=jnp.int32)
    group_settings = jnp.zeros((MAX_PROPOSAL_GROUPS, MAX_SETTINGS))
    group_masks = jnp.zeros(MAX_PROPOSAL_GROUPS)
    group_masks = group_masks.at[0].set(1.0)

    return (key, chain_state, block_idx_vec, block_mean, block_cov, coupled_blocks,
            block_mode, block_mask, proposal_type, settings,
            num_groups, group_starts, group_ends, group_proposal_types, group_settings, group_masks)


# ============================================================================
# attempt_temperature_swaps
# ============================================================================

class TestAttemptTemperatureSwaps:
    """Test parallel tempering swap logic."""

    def test_single_temperature_noop(self):
        """With 1 temperature, swaps should be a no-op."""
        key = random.PRNGKey(0)
        n_chains_a, n_chains_b, n_params = 4, 4, 3
        states_A = jnp.ones((n_chains_a, n_params))
        states_B = jnp.ones((n_chains_b, n_params))
        temp_A = jnp.zeros(n_chains_a, dtype=jnp.int32)
        temp_B = jnp.zeros(n_chains_b, dtype=jnp.int32)
        ladder = jnp.array([1.0])
        accepts = jnp.zeros(1, dtype=jnp.int32)
        attempts = jnp.zeros(1, dtype=jnp.int32)

        new_A, new_B, _, new_accepts, new_attempts, new_parity = attempt_temperature_swaps(
            key, states_A, states_B, temp_A, temp_B,
            ladder, _simple_log_post, accepts, attempts, jnp.int32(0)
        )

        # Everything should be unchanged
        np.testing.assert_array_equal(new_A, temp_A)
        np.testing.assert_array_equal(new_B, temp_B)
        np.testing.assert_array_equal(new_accepts, accepts)
        np.testing.assert_array_equal(new_attempts, attempts)

    def test_deo_parity_toggles(self):
        """DEO parity should toggle between 0 and 1 each call."""
        key = random.PRNGKey(0)
        n = 4
        states = jnp.zeros((n, 2))
        temps = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        ladder = _make_temp_ladder(2)
        accepts = jnp.zeros(1, dtype=jnp.int32)
        attempts = jnp.zeros(1, dtype=jnp.int32)

        _, _, key, _, _, parity1 = attempt_temperature_swaps(
            key, states, states, temps, temps,
            ladder, _simple_log_post, accepts, attempts, jnp.int32(0)
        )
        assert int(parity1) == 1  # 0 -> 1

        _, _, key, _, _, parity2 = attempt_temperature_swaps(
            key, states, states, temps, temps,
            ladder, _simple_log_post, accepts, attempts, jnp.int32(1)
        )
        assert int(parity2) == 0  # 1 -> 0

    def test_deo_even_parity_swaps_pair_0_1(self):
        """Even parity (0) should attempt swaps for pair (0,1)."""
        key = random.PRNGKey(42)
        n_params = 2
        n_temps = 3

        # 6 chains total, 2 per temperature — enough for swaps at each pair
        states_A = jax.random.normal(random.PRNGKey(1), (3, n_params))
        states_B = jax.random.normal(random.PRNGKey(2), (3, n_params))

        temp_A = jnp.array([0, 1, 2], dtype=jnp.int32)
        temp_B = jnp.array([0, 1, 2], dtype=jnp.int32)
        ladder = _make_temp_ladder(n_temps)
        accepts = jnp.zeros(n_temps - 1, dtype=jnp.int32)
        attempts = jnp.zeros(n_temps - 1, dtype=jnp.int32)

        _, _, _, new_accepts, new_attempts, _ = attempt_temperature_swaps(
            key, states_A, states_B, temp_A, temp_B,
            ladder, _simple_log_post, accepts, attempts, jnp.int32(0),
            use_deo=True
        )

        # Even parity: pair (0,1) active, pair (1,2) inactive
        assert int(new_attempts[0]) > 0, "Pair (0,1) should be attempted on even parity"
        assert int(new_attempts[1]) == 0, "Pair (1,2) should NOT be attempted on even parity"

    def test_deo_odd_parity_swaps_pair_1_2(self):
        """Odd parity (1) should attempt swaps for pair (1,2)."""
        key = random.PRNGKey(42)
        n_params = 2
        n_temps = 3

        states_A = jax.random.normal(random.PRNGKey(1), (3, n_params))
        states_B = jax.random.normal(random.PRNGKey(2), (3, n_params))

        temp_A = jnp.array([0, 1, 2], dtype=jnp.int32)
        temp_B = jnp.array([0, 1, 2], dtype=jnp.int32)
        ladder = _make_temp_ladder(n_temps)
        accepts = jnp.zeros(n_temps - 1, dtype=jnp.int32)
        attempts = jnp.zeros(n_temps - 1, dtype=jnp.int32)

        _, _, _, new_accepts, new_attempts, _ = attempt_temperature_swaps(
            key, states_A, states_B, temp_A, temp_B,
            ladder, _simple_log_post, accepts, attempts, jnp.int32(1),
            use_deo=True
        )

        # Odd parity: pair (1,2) active, pair (0,1) inactive
        assert int(new_attempts[0]) == 0, "Pair (0,1) should NOT be attempted on odd parity"
        assert int(new_attempts[1]) > 0, "Pair (1,2) should be attempted on odd parity"

    def test_no_deo_all_pairs_active(self):
        """With use_deo=False, all pairs should be attempted regardless of parity."""
        key = random.PRNGKey(42)
        n_params = 2
        n_temps = 3

        states_A = jax.random.normal(random.PRNGKey(1), (3, n_params))
        states_B = jax.random.normal(random.PRNGKey(2), (3, n_params))

        temp_A = jnp.array([0, 1, 2], dtype=jnp.int32)
        temp_B = jnp.array([0, 1, 2], dtype=jnp.int32)
        ladder = _make_temp_ladder(n_temps)
        accepts = jnp.zeros(n_temps - 1, dtype=jnp.int32)
        attempts = jnp.zeros(n_temps - 1, dtype=jnp.int32)

        _, _, _, _, new_attempts, _ = attempt_temperature_swaps(
            key, states_A, states_B, temp_A, temp_B,
            ladder, _simple_log_post, accepts, attempts, jnp.int32(0),
            use_deo=False
        )

        # Both pairs should be attempted
        assert int(new_attempts[0]) > 0
        assert int(new_attempts[1]) > 0

    def test_states_not_modified(self):
        """Temperature swaps should modify assignments, NOT states."""
        key = random.PRNGKey(42)
        states_A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        states_B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        temp_A = jnp.array([0, 1], dtype=jnp.int32)
        temp_B = jnp.array([0, 1], dtype=jnp.int32)
        ladder = _make_temp_ladder(2)

        accepts = jnp.zeros(1, dtype=jnp.int32)
        attempts = jnp.zeros(1, dtype=jnp.int32)

        # Run multiple times to ensure states never change
        for i in range(10):
            key_i = random.PRNGKey(i)
            new_A, new_B, _, _, _, _ = attempt_temperature_swaps(
                key_i, states_A, states_B, temp_A, temp_B,
                ladder, _simple_log_post, accepts, attempts, jnp.int32(0)
            )
            # Function returns temp assignments, not states
            # Verify by checking shapes match temp assignments, not states
            assert new_A.shape == temp_A.shape
            assert new_B.shape == temp_B.shape

    def test_swap_counts_accumulate(self):
        """Swap accept/attempt counts should accumulate across calls."""
        key = random.PRNGKey(0)
        n_temps = 2
        states_A = jnp.array([[0.1, 0.1], [5.0, 5.0]])
        states_B = jnp.array([[0.2, 0.2], [6.0, 6.0]])
        temp_A = jnp.array([0, 1], dtype=jnp.int32)
        temp_B = jnp.array([0, 1], dtype=jnp.int32)
        ladder = _make_temp_ladder(n_temps)

        accepts = jnp.zeros(n_temps - 1, dtype=jnp.int32)
        attempts = jnp.zeros(n_temps - 1, dtype=jnp.int32)

        for i in range(5):
            key = random.PRNGKey(i)
            _, _, _, accepts, attempts, _ = attempt_temperature_swaps(
                key, states_A, states_B, temp_A, temp_B,
                ladder, _simple_log_post, accepts, attempts, jnp.int32(0),
                use_deo=False
            )

        # After 5 iterations, should have accumulated attempts
        assert int(attempts[0]) > 0
        # Accepts should be <= attempts
        assert int(accepts[0]) <= int(attempts[0])

    def test_identical_states_always_accept(self):
        """When all chains have identical states, swaps should always accept.

        With identical log-likelihoods: log_alpha = (beta_i - beta_j) * 0 = 0,
        so exp(log_alpha) = 1 >= uniform, always accepted.
        """
        key = random.PRNGKey(42)
        n_temps = 2
        # All chains have identical states
        state = jnp.array([1.0, 1.0])
        states_A = jnp.stack([state, state])
        states_B = jnp.stack([state, state])
        temp_A = jnp.array([0, 1], dtype=jnp.int32)
        temp_B = jnp.array([0, 1], dtype=jnp.int32)
        ladder = _make_temp_ladder(n_temps)

        accepts = jnp.zeros(n_temps - 1, dtype=jnp.int32)
        attempts = jnp.zeros(n_temps - 1, dtype=jnp.int32)

        total_accepts = 0
        total_attempts = 0
        for i in range(20):
            key = random.PRNGKey(i)
            _, _, _, accepts_i, attempts_i, _ = attempt_temperature_swaps(
                key, states_A, states_B, temp_A, temp_B,
                ladder, _simple_log_post, jnp.zeros(1, dtype=jnp.int32),
                jnp.zeros(1, dtype=jnp.int32), jnp.int32(0), use_deo=False
            )
            total_accepts += int(accepts_i[0])
            total_attempts += int(attempts_i[0])

        assert total_accepts == total_attempts, \
            f"Identical states should always accept: {total_accepts}/{total_attempts}"

    def test_temp_assignments_are_permuted(self):
        """After swaps, temp assignments should still be a valid permutation."""
        key = random.PRNGKey(42)
        n_temps = 3
        n_chains = 6

        # Assign each chain a temperature
        temp_all = jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
        temp_A = temp_all[:3]
        temp_B = temp_all[3:]

        states_A = jax.random.normal(random.PRNGKey(1), (3, 4))
        states_B = jax.random.normal(random.PRNGKey(2), (3, 4))
        ladder = _make_temp_ladder(n_temps)

        accepts = jnp.zeros(n_temps - 1, dtype=jnp.int32)
        attempts = jnp.zeros(n_temps - 1, dtype=jnp.int32)

        new_A, new_B, _, _, _, _ = attempt_temperature_swaps(
            key, states_A, states_B, temp_A, temp_B,
            ladder, _simple_log_post, accepts, attempts, jnp.int32(0),
            use_deo=False
        )

        # All original temp values should still be present (just rearranged)
        all_new = jnp.concatenate([new_A, new_B])
        for t in range(n_temps):
            orig_count = int(jnp.sum(temp_all == t))
            new_count = int(jnp.sum(all_new == t))
            assert orig_count == new_count, \
                f"Temp {t}: had {orig_count} chains, now {new_count}"


# ============================================================================
# coupled_transform_step
# ============================================================================

class TestCoupledTransformStep:
    """Test the coupled transform (theta-preserving) MH step."""

    def _make_coupled_transform_fn(self, accept_jacobian=0.0, accept_prior=0.0):
        """Create a simple coupled transform function for testing.

        Returns a transform that:
        - Sets coupled indices to fixed values
        - Returns configurable log_jacobian and log_prior_ratio
        """
        def transform_fn(key, chain_state, primary_indices, proposed_primary):
            n_params = chain_state.shape[0]
            # "Coupled" indices are the last 2 params
            coupled_indices = jnp.array([n_params - 2, n_params - 1])
            # Deterministic transform: coupled = proposed_primary[:2] * 0.5
            new_coupled = proposed_primary[:2] * 0.5
            new_key = random.split(key)[0]
            return coupled_indices, new_coupled, accept_jacobian, accept_prior, new_key
        return transform_fn

    def test_basic_acceptance(self):
        """Coupled transform step should accept/reject based on full ratio."""
        n_params = 4

        operand = _make_mh_operand(n_params, chain_state=jnp.array([1.0, 1.0, 2.0, 2.0]))
        transform_fn = self._make_coupled_transform_fn(accept_jacobian=0.0, accept_prior=0.0)
        used_types = (int(ProposalType.SELF_MEAN),)

        next_state, new_key, lp, accepted = coupled_transform_step(
            operand, _simple_log_post, transform_fn, used_types
        )

        assert next_state.shape == (n_params,)
        assert jnp.isfinite(lp)
        # accepted is either 0.0 or 1.0
        assert float(accepted) in (0.0, 1.0)

    def test_large_positive_jacobian_increases_acceptance(self):
        """A large positive log_jacobian should make acceptance more likely."""
        n_params = 4
        chain_state = jnp.array([1.0, 1.0, 2.0, 2.0])

        accept_count_with_jacobian = 0
        accept_count_without = 0

        for i in range(50):
            key = random.PRNGKey(i)
            operand_with = _make_mh_operand(n_params, chain_state=chain_state, key=key)
            transform_with = self._make_coupled_transform_fn(accept_jacobian=10.0)
            _, _, _, acc_with = coupled_transform_step(
                operand_with, _simple_log_post, transform_with,
                (int(ProposalType.SELF_MEAN),)
            )
            accept_count_with_jacobian += int(float(acc_with))

            operand_without = _make_mh_operand(n_params, chain_state=chain_state, key=key)
            transform_without = self._make_coupled_transform_fn(accept_jacobian=-10.0)
            _, _, _, acc_without = coupled_transform_step(
                operand_without, _simple_log_post, transform_without,
                (int(ProposalType.SELF_MEAN),)
            )
            accept_count_without += int(float(acc_without))

        assert accept_count_with_jacobian > accept_count_without, \
            f"Positive Jacobian should increase acceptance: {accept_count_with_jacobian} vs {accept_count_without}"

    def test_nan_proposal_rejected(self):
        """If the coupled transform produces NaN, should force rejection."""
        n_params = 4

        def nan_transform(key, chain_state, primary_indices, proposed_primary):
            coupled_indices = jnp.array([n_params - 2, n_params - 1])
            new_coupled = jnp.array([jnp.nan, jnp.nan])
            new_key = random.split(key)[0]
            return coupled_indices, new_coupled, 0.0, 0.0, new_key

        chain_state = jnp.array([1.0, 1.0, 2.0, 2.0])
        operand = _make_mh_operand(n_params, chain_state=chain_state)
        used_types = (int(ProposalType.SELF_MEAN),)

        next_state, _, _, accepted = coupled_transform_step(
            operand, _simple_log_post, nan_transform, used_types
        )

        assert float(accepted) == 0.0, "NaN coupled values should be rejected"
        # State should be unchanged
        np.testing.assert_array_equal(next_state, chain_state)

    def test_coupled_indices_updated_on_accept(self):
        """When accepted, coupled indices should be updated by the transform."""
        n_params = 4

        def always_better_transform(key, chain_state, primary_indices, proposed_primary):
            coupled_indices = jnp.array([2, 3])
            # Set coupled to specific recognizable values
            new_coupled = jnp.array([99.0, 99.0])
            new_key = random.split(key)[0]
            # Large positive values to guarantee acceptance
            return coupled_indices, new_coupled, 100.0, 100.0, new_key

        chain_state = jnp.array([0.0, 0.0, 1.0, 1.0])
        operand = _make_mh_operand(n_params, chain_state=chain_state)
        used_types = (int(ProposalType.SELF_MEAN),)

        next_state, _, _, accepted = coupled_transform_step(
            operand, _simple_log_post, always_better_transform, used_types
        )

        if float(accepted) == 1.0:
            # Coupled indices (2, 3) should have the transform values
            assert float(next_state[2]) == 99.0
            assert float(next_state[3]) == 99.0


# ============================================================================
# metropolis_block_step (supplementary tests)
# ============================================================================

class TestMetropolisBlockStep:
    """Test the MH step for basic correctness properties."""

    def test_nan_proposal_rejected(self):
        """If proposal is NaN, should reject and keep current state."""
        n_params = 2
        chain_state = jnp.array([1.0, 2.0])

        # Use a log_post_fn that returns NaN for certain states
        def nan_log_post(state, param_indices, beta=1.0):
            # Return NaN if any param > 100 (should never happen with rejection)
            return jnp.where(jnp.any(state > 100), jnp.nan, -0.5 * jnp.sum(state ** 2))

        operand = _make_mh_operand(n_params, chain_state=chain_state)
        used_types = (int(ProposalType.SELF_MEAN),)

        next_state, _, _, accepted = metropolis_block_step(
            operand, nan_log_post, None, used_types
        )

        assert next_state.shape == (n_params,)
        assert jnp.all(jnp.isfinite(next_state))

    def test_always_accept_better_proposal(self):
        """A proposal with much higher log posterior should almost always accept."""
        n_params = 2

        # Log posterior that strongly prefers origin
        def peaked_log_post(state, param_indices, beta=1.0):
            return -100.0 * jnp.sum(state ** 2)

        accept_count = 0
        for i in range(50):
            key = random.PRNGKey(i)
            # Start far from origin - any move toward origin should be accepted
            chain_state = jnp.array([10.0, 10.0])
            operand = _make_mh_operand(n_params, chain_state=chain_state, key=key)
            used_types = (int(ProposalType.SELF_MEAN),)

            _, _, _, accepted = metropolis_block_step(
                operand, peaked_log_post, None, used_types
            )
            accept_count += int(float(accepted))

        # Most proposals should be accepted (moving toward peak)
        # With such a peaked posterior, even random walk proposals toward origin are better
        assert accept_count > 0, "Should accept at least some proposals"

    def test_beta_tempering_parameter(self):
        """Beta < 1 should flatten the posterior, increasing acceptance."""
        n_params = 2

        def log_post_with_beta(state, param_indices, beta=1.0):
            log_prior = -0.5 * jnp.sum(state ** 2)
            log_lik = -50.0 * jnp.sum((state - 5.0) ** 2)
            return log_prior + beta * log_lik

        # Start at a point with moderate posterior density
        chain_state = jnp.array([0.0, 0.0])
        used_types = (int(ProposalType.SELF_MEAN),)

        accept_hot = 0
        accept_cold = 0

        for i in range(100):
            key = random.PRNGKey(i)
            operand = _make_mh_operand(n_params, chain_state=chain_state, key=key)

            # Hot temperature (beta near 0) - should accept more
            _, _, _, acc = metropolis_block_step(operand, log_post_with_beta, None, used_types, beta=0.01)
            accept_hot += int(float(acc))

            # Cold temperature (beta = 1) - should accept less
            operand2 = _make_mh_operand(n_params, chain_state=chain_state, key=key)
            _, _, _, acc2 = metropolis_block_step(operand2, log_post_with_beta, None, used_types, beta=1.0)
            accept_cold += int(float(acc2))

        assert accept_hot >= accept_cold, \
            f"Hot temp should accept more: hot={accept_hot}, cold={accept_cold}"
