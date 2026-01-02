"""
Pytest configuration and shared fixtures for bamcmc tests.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from bamcmc.batch_specs import BlockSpec, SamplerType, ProposalType
from bamcmc.settings import SettingSlot, MAX_SETTINGS, SETTING_DEFAULTS
from bamcmc.registry import register_posterior, _REGISTRY
from bamcmc import test_posteriors


@pytest.fixture
def rng_seed():
    """Default RNG seed for reproducible tests."""
    return 42


@pytest.fixture
def basic_mcmc_config():
    """Basic MCMC configuration for tests."""
    return {
        'NUM_CHAINS': 4,
        'num_chains_a': 2,
        'num_chains_b': 2,
        'num_superchains': 4,
        'jnp_float_dtype': jnp.float32,
        'rng_seed': 42,
        'save_likelihoods': False,
    }


@pytest.fixture
def nested_mcmc_config():
    """MCMC configuration with nested R-hat (M > 1)."""
    return {
        'NUM_CHAINS': 4,
        'num_chains_a': 2,
        'num_chains_b': 2,
        'num_superchains': 2,  # M = 4/2 = 2
        'jnp_float_dtype': jnp.float32,
        'rng_seed': 42,
        'save_likelihoods': False,
    }


@pytest.fixture
def single_block_spec():
    """Single parameter block for simple tests."""
    return [
        BlockSpec(
            size=3,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=ProposalType.SELF_MEAN,
            label='block0'
        )
    ]


@pytest.fixture
def multi_block_specs():
    """Multiple blocks of different types."""
    return [
        BlockSpec(2, SamplerType.METROPOLIS_HASTINGS, ProposalType.SELF_MEAN),
        BlockSpec(1, SamplerType.DIRECT_CONJUGATE, direct_sampler_fn=lambda x: x)
    ]


@pytest.fixture
def register_test_posteriors():
    """
    Fixture to register test posteriors and clean up after test.

    Usage:
        def test_something(register_test_posteriors):
            # Test posteriors are now registered
            ...
    """
    # Save any existing registrations
    original_registrations = {}
    for name, config in test_posteriors.TEST_POSTERIORS.items():
        if name in _REGISTRY:
            original_registrations[name] = _REGISTRY[name]
        register_posterior(name, config)

    yield  # Run the test

    # Restore original registry state
    for name in test_posteriors.TEST_POSTERIORS.keys():
        if name in original_registrations:
            _REGISTRY[name] = original_registrations[name]
        elif name in _REGISTRY:
            del _REGISTRY[name]


def make_settings_array(chain_prob=None, n_categories=None, cov_mult=None, uniform_weight=None):
    """
    Create a settings array for testing proposal functions.

    Args:
        chain_prob: Probability of using chain_mean in mixture proposal (default: 0.5)
        n_categories: Number of categories for multinomial (default: 4)
        cov_mult: Covariance multiplier for proposal variance (default: 1.0)
            Used by MIXTURE, SELF_MEAN, and MALA proposals.
        uniform_weight: Weight of uniform distribution in multinomial proposal (default: 0.4)

    Returns:
        JAX array of shape (MAX_SETTINGS,) with specified values
    """
    settings = np.zeros(MAX_SETTINGS, dtype=np.float32)
    for slot, default in SETTING_DEFAULTS.items():
        settings[slot] = default
    if chain_prob is not None:
        settings[SettingSlot.CHAIN_PROB] = chain_prob
    if n_categories is not None:
        settings[SettingSlot.N_CATEGORIES] = n_categories
    if cov_mult is not None:
        settings[SettingSlot.COV_MULT] = cov_mult
    if uniform_weight is not None:
        settings[SettingSlot.UNIFORM_WEIGHT] = uniform_weight
    return jnp.array(settings)


def dummy_grad_fn(block_values):
    """
    Dummy gradient function for testing proposals that don't use gradients.

    Returns zeros with the same shape as input.
    """
    return jnp.zeros_like(block_values)
