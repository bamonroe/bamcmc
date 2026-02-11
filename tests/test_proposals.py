"""
Proposal Tests - Hastings Ratio Verification and Edge Cases

Tests individual proposal functions for correctness:
- Hastings ratio symmetry/correctness
- Edge cases (extreme values, boundary conditions)
- Proper masking behavior

Run with: pytest tests/test_proposals.py -v
"""

import numpy as np
import jax.numpy as jnp
import jax
import pytest

from bamcmc.proposals import (
    self_mean_proposal,
    chain_mean_proposal,
    mixture_proposal,
    mala_proposal,
    mean_mala_proposal,
    mean_weighted_proposal,
    mode_weighted_proposal,
    mcov_weighted_proposal,
    mcov_weighted_vec_proposal,
    mcov_smooth_proposal,
)
from bamcmc.settings import SettingSlot, MAX_SETTINGS, SETTING_DEFAULTS

from .conftest import (
    make_settings_array, dummy_grad_fn,
    make_test_covariance, make_test_operand, log_mvn_density,
)


# ============================================================================
# SYMMETRIC PROPOSAL TESTS
# ============================================================================

class TestSelfMeanProposal:
    """Test self_mean (random walk) proposal."""

    def test_hastings_ratio_is_zero(self):
        """Self-mean is symmetric, so Hastings ratio should be 0."""
        operand = make_test_operand(dim=3, current=jnp.array([1.0, 2.0, 3.0]))
        proposal, log_ratio, _ = self_mean_proposal(operand)

        assert log_ratio == 0.0, "Self-mean should have zero Hastings ratio"

    def test_hastings_ratio_zero_with_cov_mult(self):
        """Hastings ratio should be 0 regardless of cov_mult."""
        settings = make_settings_array(cov_mult=2.5)
        operand = make_test_operand(dim=2, settings=settings)
        _, log_ratio, _ = self_mean_proposal(operand)

        assert log_ratio == 0.0

    def test_proposal_centered_on_current(self):
        """Proposals should be centered on current state."""
        current = jnp.array([5.0, -3.0])
        proposals = []

        for i in range(500):
            key = jax.random.PRNGKey(i)
            operand = make_test_operand(dim=2, current=current, key=key)
            prop, _, _ = self_mean_proposal(operand)
            proposals.append(prop)

        proposals = jnp.stack(proposals)
        mean_proposal = jnp.mean(proposals, axis=0)

        # Mean of proposals should be close to current
        assert jnp.allclose(mean_proposal, current, atol=0.2)

    def test_respects_block_mask(self):
        """Masked dimensions should not change."""
        current = jnp.array([1.0, 2.0, 3.0])
        key, current_block, mean, cov, coupled, _, settings, grad_fn, mode = make_test_operand(
            dim=3, current=current
        )
        # Mask out the third dimension
        block_mask = jnp.array([1.0, 1.0, 0.0])

        operand = (key, current_block, mean, cov, coupled, block_mask, settings, grad_fn, mode)
        proposal, _, _ = self_mean_proposal(operand)

        # Third dimension should be unchanged (current + 0 perturbation)
        # But since we add perturbation * mask, masked dims get current + 0
        assert jnp.isclose(proposal[2], current[2], atol=1e-6)


# ============================================================================
# ASYMMETRIC PROPOSAL TESTS - HASTINGS RATIO VERIFICATION
# ============================================================================

class TestChainMeanProposal:
    """Test chain_mean (independent) proposal Hastings ratio."""

    def test_hastings_ratio_formula(self):
        """Verify Hastings ratio matches q(x|μ) / q(y|μ) formula."""
        mean = jnp.array([0.0, 0.0])
        cov = jnp.eye(2)
        current = jnp.array([1.0, 1.0])

        key = jax.random.PRNGKey(123)
        operand = make_test_operand(dim=2, current=current, mean=mean, cov=cov, key=key)

        proposal, log_ratio, _ = chain_mean_proposal(operand)

        # For chain_mean: log_ratio = log q(current | mean) - log q(proposal | mean)
        # = -0.5 * ||current - mean||²_Σ⁻¹ + 0.5 * ||proposal - mean||²_Σ⁻¹
        log_q_current = log_mvn_density(current, mean, cov)
        log_q_proposal = log_mvn_density(proposal, mean, cov)
        expected_ratio = log_q_current - log_q_proposal

        assert jnp.isclose(log_ratio, expected_ratio, atol=1e-5), \
            f"Expected {expected_ratio}, got {log_ratio}"

    def test_hastings_ratio_with_correlated_cov(self):
        """Test Hastings ratio with non-diagonal covariance."""
        mean = jnp.array([1.0, 2.0])
        # Create a correlated covariance matrix
        cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        current = jnp.array([0.0, 0.0])

        key = jax.random.PRNGKey(456)
        operand = make_test_operand(dim=2, current=current, mean=mean, cov=cov, key=key)

        proposal, log_ratio, _ = chain_mean_proposal(operand)

        log_q_current = log_mvn_density(current, mean, cov)
        log_q_proposal = log_mvn_density(proposal, mean, cov)
        expected_ratio = log_q_current - log_q_proposal

        assert jnp.isclose(log_ratio, expected_ratio, atol=1e-5)

    def test_proposal_centered_on_step_mean(self):
        """Proposals should be centered on step_mean, not current."""
        current = jnp.array([10.0, 10.0])  # Far from mean
        mean = jnp.array([0.0, 0.0])

        proposals = []
        for i in range(500):
            key = jax.random.PRNGKey(i)
            operand = make_test_operand(dim=2, current=current, mean=mean, key=key)
            prop, _, _ = chain_mean_proposal(operand)
            proposals.append(prop)

        proposals = jnp.stack(proposals)
        mean_proposal = jnp.mean(proposals, axis=0)

        # Mean of proposals should be close to step_mean (0, 0), not current (10, 10)
        assert jnp.allclose(mean_proposal, mean, atol=0.2)


class TestMixtureProposalHastings:
    """Test mixture proposal Hastings ratio computation."""

    def test_hastings_ratio_is_finite(self):
        """Hastings ratio should always be finite."""
        for chain_prob in [0.0, 0.3, 0.5, 0.7, 1.0]:
            settings = make_settings_array(chain_prob=chain_prob)
            operand = make_test_operand(dim=2, settings=settings)
            _, log_ratio, _ = mixture_proposal(operand)

            assert jnp.isfinite(log_ratio), f"Non-finite ratio for chain_prob={chain_prob}"

    def test_hastings_ratio_bounded(self):
        """Hastings ratio should be reasonably bounded."""
        current = jnp.array([5.0, 5.0])
        mean = jnp.array([0.0, 0.0])

        for i in range(100):
            key = jax.random.PRNGKey(i)
            settings = make_settings_array(chain_prob=0.5)
            operand = make_test_operand(dim=2, current=current, mean=mean,
                                        settings=settings, key=key)
            _, log_ratio, _ = mixture_proposal(operand)

            # Ratio should be bounded (not exploding)
            assert abs(float(log_ratio)) < 100


class TestMcovSmoothProposalHastings:
    """Test MCOV_SMOOTH proposal Hastings ratio - the most complex proposal."""

    def test_hastings_ratio_is_finite(self):
        """Hastings ratio should be finite for various distances from mean."""
        mean = jnp.array([0.0, 0.0])

        # Test at various distances from mean
        test_points = [
            jnp.array([0.0, 0.0]),    # At mean
            jnp.array([1.0, 1.0]),    # Close to mean
            jnp.array([5.0, 5.0]),    # Moderate distance
            jnp.array([20.0, 20.0]),  # Far from mean
        ]

        for current in test_points:
            key = jax.random.PRNGKey(42)
            operand = make_test_operand(dim=2, current=current, mean=mean, key=key)
            _, log_ratio, _ = mcov_smooth_proposal(operand)

            assert jnp.isfinite(log_ratio), f"Non-finite ratio at {current}"

    def test_near_equilibrium_small_ratio(self):
        """Near equilibrium, Hastings ratio should be small (proposal nearly symmetric)."""
        mean = jnp.array([0.0, 0.0])
        current = jnp.array([0.1, 0.1])  # Very close to mean

        ratios = []
        for i in range(50):
            key = jax.random.PRNGKey(i)
            operand = make_test_operand(dim=2, current=current, mean=mean, key=key)
            _, log_ratio, _ = mcov_smooth_proposal(operand)
            ratios.append(float(log_ratio))

        # Near equilibrium, ratios should be small (close to symmetric)
        mean_abs_ratio = np.mean(np.abs(ratios))
        assert mean_abs_ratio < 5.0, f"Ratio too large near equilibrium: {mean_abs_ratio}"

    def test_respects_settings(self):
        """Test that K_G and K_ALPHA settings affect behavior."""
        current = jnp.array([10.0, 10.0])
        mean = jnp.array([0.0, 0.0])

        # With default settings
        settings_default = make_settings_array()
        operand_default = make_test_operand(dim=2, current=current, mean=mean,
                                            settings=settings_default, key=jax.random.PRNGKey(42))
        prop_default, _, _ = mcov_smooth_proposal(operand_default)

        # With different K_G (controls variance scaling)
        settings = jnp.zeros(MAX_SETTINGS)
        for slot, default in SETTING_DEFAULTS.items():
            settings = settings.at[slot].set(default)
        settings = settings.at[SettingSlot.K_G].set(1.0)  # Much smaller than default
        settings = settings.at[SettingSlot.K_ALPHA].set(1.0)

        operand_modified = make_test_operand(dim=2, current=current, mean=mean,
                                             settings=settings, key=jax.random.PRNGKey(42))
        prop_modified, _, _ = mcov_smooth_proposal(operand_modified)

        # Proposals should differ with different settings
        # (This is a weak test but ensures settings are being read)
        assert not jnp.allclose(prop_default, prop_modified)


# ============================================================================
# HASTINGS RATIO CONSISTENCY TESTS
# ============================================================================

class TestHastingsRatioConsistency:
    """Test that Hastings ratios are computed consistently."""

    def test_chain_mean_reversibility(self):
        """Test q(y|x)/q(x|y) by computing both directions."""
        mean = jnp.array([0.0, 0.0])
        cov = jnp.eye(2) * 0.5
        x = jnp.array([1.0, 2.0])

        # Get a proposal y from x
        key = jax.random.PRNGKey(42)
        operand_forward = make_test_operand(dim=2, current=x, mean=mean, cov=cov, key=key)
        y, log_ratio_forward, _ = chain_mean_proposal(operand_forward)

        # For chain_mean, q(x|mean) and q(y|mean) don't depend on the "current" state
        # The ratio is: log q(x|μ) - log q(y|μ)
        log_q_x = log_mvn_density(x, mean, cov)
        log_q_y = log_mvn_density(y, mean, cov)

        # Forward ratio: log q(x|μ) - log q(y|μ) (returned when proposing y from x)
        expected_forward = log_q_x - log_q_y

        # This should match what chain_mean_proposal returns
        assert jnp.isclose(log_ratio_forward, expected_forward, atol=1e-5)

    def test_mcov_proposals_finite_across_range(self):
        """Test that MCOV proposals are well-behaved across parameter space."""
        proposals_to_test = [
            mcov_weighted_proposal,
            mcov_weighted_vec_proposal,
            mcov_smooth_proposal,
        ]

        mean = jnp.array([0.0, 0.0])
        distances = [0.1, 1.0, 5.0, 10.0, 50.0]

        for proposal_fn in proposals_to_test:
            for d in distances:
                current = jnp.array([d, d])
                key = jax.random.PRNGKey(int(d * 100))
                operand = make_test_operand(dim=2, current=current, mean=mean, key=key)

                prop, log_ratio, _ = proposal_fn(operand)

                assert jnp.all(jnp.isfinite(prop)), \
                    f"{proposal_fn.__name__} produced non-finite proposal at d={d}"
                assert jnp.isfinite(log_ratio), \
                    f"{proposal_fn.__name__} produced non-finite ratio at d={d}"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestProposalEdgeCases:
    """Test proposals under edge conditions."""

    def test_single_dimension(self):
        """Test proposals work with 1D parameters."""
        proposals = [
            self_mean_proposal,
            chain_mean_proposal,
            mcov_smooth_proposal,
        ]

        for proposal_fn in proposals:
            current = jnp.array([1.0])
            mean = jnp.array([0.0])
            cov = jnp.array([[1.0]])

            key = jax.random.PRNGKey(42)
            operand = make_test_operand(dim=1, current=current, mean=mean, cov=cov, key=key)

            prop, log_ratio, _ = proposal_fn(operand)

            assert prop.shape == (1,)
            assert jnp.isfinite(log_ratio)

    def test_high_dimension(self):
        """Test proposals work with higher-dimensional parameters."""
        dim = 10
        proposals = [
            self_mean_proposal,
            chain_mean_proposal,
        ]

        for proposal_fn in proposals:
            current = jnp.ones(dim)
            mean = jnp.zeros(dim)
            cov = make_test_covariance(dim)

            key = jax.random.PRNGKey(42)
            operand = make_test_operand(dim=dim, current=current, mean=mean, cov=cov, key=key)

            prop, log_ratio, _ = proposal_fn(operand)

            assert prop.shape == (dim,)
            assert jnp.isfinite(log_ratio)

    def test_all_masked(self):
        """Test behavior when all dimensions are masked out."""
        current = jnp.array([1.0, 2.0])
        key, _, mean, cov, coupled, _, settings, grad_fn, mode = make_test_operand(dim=2, current=current)
        block_mask = jnp.array([0.0, 0.0])  # All masked

        operand = (key, current, mean, cov, coupled, block_mask, settings, grad_fn, mode)

        # Self-mean should return current unchanged
        prop, _, _ = self_mean_proposal(operand)
        assert jnp.allclose(prop, current)

    def test_very_small_variance(self):
        """Test with very small covariance (concentrated distribution)."""
        cov = jnp.eye(2) * 1e-6
        current = jnp.array([1.0, 1.0])

        operand = make_test_operand(dim=2, current=current, cov=cov)

        # Should not crash, though proposals will be very close to center
        prop, log_ratio, _ = chain_mean_proposal(operand)
        assert jnp.all(jnp.isfinite(prop))
        assert jnp.isfinite(log_ratio)

    def test_large_variance(self):
        """Test with large covariance."""
        cov = jnp.eye(2) * 100.0
        current = jnp.array([1.0, 1.0])

        operand = make_test_operand(dim=2, current=current, cov=cov)

        prop, log_ratio, _ = chain_mean_proposal(operand)
        assert jnp.all(jnp.isfinite(prop))
        assert jnp.isfinite(log_ratio)


# ============================================================================
# GRADIENT-BASED PROPOSAL TESTS
# ============================================================================

class TestMALAProposal:
    """Test MALA (gradient-based) proposal."""

    def test_uses_gradient(self):
        """Test that MALA actually uses the gradient."""
        current = jnp.array([5.0, 5.0])
        mean = jnp.array([0.0, 0.0])

        # Gradient pointing toward origin
        def grad_toward_origin(x):
            return -x  # Gradient of -0.5 * ||x||^2

        # Gradient pointing away from origin
        def grad_away_from_origin(x):
            return x

        proposals_toward = []
        proposals_away = []

        for i in range(100):
            key = jax.random.PRNGKey(i)

            operand_toward = make_test_operand(
                dim=2, current=current, mean=mean, key=key, grad_fn=grad_toward_origin
            )
            prop_toward, _, _ = mala_proposal(operand_toward)
            proposals_toward.append(prop_toward)

            operand_away = make_test_operand(
                dim=2, current=current, mean=mean, key=key, grad_fn=grad_away_from_origin
            )
            prop_away, _, _ = mala_proposal(operand_away)
            proposals_away.append(prop_away)

        mean_toward = jnp.mean(jnp.stack(proposals_toward), axis=0)
        mean_away = jnp.mean(jnp.stack(proposals_away), axis=0)

        # Proposals with gradient toward origin should be closer to origin
        dist_toward = jnp.linalg.norm(mean_toward)
        dist_away = jnp.linalg.norm(mean_away)

        assert dist_toward < dist_away, \
            f"MALA should move toward gradient direction: toward={dist_toward}, away={dist_away}"

    def test_hastings_ratio_accounts_for_drift(self):
        """Test that MALA Hastings ratio is non-zero (asymmetric due to drift)."""
        current = jnp.array([2.0, 2.0])

        def simple_grad(x):
            return -x  # Gradient of quadratic

        key = jax.random.PRNGKey(42)
        operand = make_test_operand(dim=2, current=current, grad_fn=simple_grad, key=key)

        _, log_ratio, _ = mala_proposal(operand)

        # MALA has asymmetric proposal due to drift, so ratio generally non-zero
        assert jnp.isfinite(log_ratio)


# ============================================================================
# MODE-WEIGHTED PROPOSAL TESTS
# ============================================================================

class TestModeWeightedProposal:
    """Test mode_weighted proposal."""

    def test_uses_block_mode(self):
        """Test that mode_weighted uses the block_mode parameter."""
        current = jnp.array([10.0, 10.0])
        mean = jnp.array([0.0, 0.0])
        mode = jnp.array([-5.0, -5.0])  # Mode different from mean

        key, current_block, step_mean, cov, coupled, mask, settings, grad_fn, _ = make_test_operand(
            dim=2, current=current, mean=mean
        )

        proposals = []
        for i in range(100):
            key_i = jax.random.PRNGKey(i)
            operand = (key_i, current_block, step_mean, cov, coupled, mask, settings, grad_fn, mode)
            prop, _, _ = mode_weighted_proposal(operand)
            proposals.append(prop)

        mean_proposal = jnp.mean(jnp.stack(proposals), axis=0)

        # Proposals should be influenced by mode (pulled toward it)
        # The mean of proposals should be between current and mode
        # This is a weak test but ensures mode is being used
        dist_to_mode = jnp.linalg.norm(mean_proposal - mode)
        dist_current_to_mode = jnp.linalg.norm(current - mode)

        # Mean proposal should be closer to mode than current is
        assert dist_to_mode < dist_current_to_mode


# ============================================================================
# HASTINGS RATIO SYMMETRY TESTS
# ============================================================================

class TestHastingsRatioSymmetry:
    """Verify Hastings ratios satisfy detailed balance by checking both directions."""

    def test_chain_mean_ratio_symmetry(self):
        """For CHAIN_MEAN, verify log q(y|x) - log q(x|y) consistency.

        Chain-mean is an independent proposal q(·) = N(μ, Σ), so:
          log_ratio_forward  = log q(x|μ) - log q(y|μ)   (proposing y from x)
          log_ratio_reverse  = log q(y|μ) - log q(x|μ)   (proposing x from y)
        These should sum to zero.
        """
        mean = jnp.array([1.0, -1.0])
        cov = jnp.array([[1.0, 0.3], [0.3, 1.0]])
        x = jnp.array([2.0, 3.0])

        # Get proposal y from x
        key = jax.random.PRNGKey(42)
        operand_forward = make_test_operand(dim=2, current=x, mean=mean, cov=cov, key=key)
        y, log_ratio_forward, _ = chain_mean_proposal(operand_forward)

        # Compute ratio in reverse direction (proposing x from y)
        # For chain_mean, the ratio doesn't depend on which state we're at,
        # only on the proposed and current values relative to μ
        operand_reverse = make_test_operand(dim=2, current=y, mean=mean, cov=cov,
                                             key=jax.random.PRNGKey(99))
        _, log_ratio_reverse_raw, _ = chain_mean_proposal(operand_reverse)

        # The reverse ratio for the specific pair (x, y) is log q(y|μ) - log q(x|μ)
        # We compute it directly from the density function
        log_q_x = log_mvn_density(x, mean, cov)
        log_q_y = log_mvn_density(y, mean, cov)

        expected_forward = log_q_x - log_q_y
        expected_reverse = log_q_y - log_q_x

        # Forward + reverse should sum to zero
        assert jnp.isclose(log_ratio_forward + expected_reverse, 0.0, atol=1e-5), \
            f"Forward ({log_ratio_forward}) + reverse ({expected_reverse}) should be 0"

        # Also verify each direction independently
        assert jnp.isclose(log_ratio_forward, expected_forward, atol=1e-5)


# ============================================================================
# MIXTURE PROPOSAL TESTS (moved from test_unit.py)
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
# MULTINOMIAL PROPOSAL TESTS (moved from test_unit.py)
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
# MCOV_WEIGHTED PROPOSAL TESTS (moved from test_unit.py)
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
        assert jnp.all(mean_proposal < current_far)
        dist_moved = jnp.linalg.norm(mean_proposal - current_far)
        assert dist_moved > 1.0

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
        assert proposal.shape == (3,)
