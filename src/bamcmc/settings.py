"""
Proposal settings configuration.

This module defines the canonical ordering of proposal settings and provides
utilities to build a settings matrix from BlockSpecs.

Settings are stored in a JAX array of shape (n_blocks, MAX_SETTINGS) for
efficient O(1) access during MCMC iterations. Each proposal function receives
a row of this matrix and accesses settings by position using the SettingSlot enum.

To add a new setting:
1. Add it to SettingSlot enum
2. Add default value to SETTING_DEFAULTS
3. Use it in your proposal: settings[SettingSlot.NEW_SETTING]
4. Specify in BlockSpec: settings={'new_setting': value}
"""

from enum import IntEnum
import numpy as np
import jax.numpy as jnp


class SettingSlot(IntEnum):
    """
    Canonical slot indices for proposal settings.

    These map setting names to positions in the settings array.
    IntEnum values compile to simple integers - no runtime overhead.
    """
    CHAIN_PROB = 0      # Probability of using chain_mean in mixture proposal (0-1)
    N_CATEGORIES = 1    # Number of categories for MULTINOMIAL proposal
    COV_MULT = 2        # Covariance multiplier for proposal variance (used by MIXTURE, SELF_MEAN, MALA)
    UNIFORM_WEIGHT = 3  # Weight of uniform distribution in MULTINOMIAL proposal (0-1)
    # Future settings:
    # BOUNDS_LOW = 4
    # BOUNDS_HIGH = 5


# Default values for each setting
SETTING_DEFAULTS = {
    SettingSlot.CHAIN_PROB: 0.5,
    SettingSlot.N_CATEGORIES: 4.0,  # Stored as float for JAX compatibility
    SettingSlot.COV_MULT: 1.0,      # Proposal variance = cov_mult * Σ (no scaling by default)
    SettingSlot.UNIFORM_WEIGHT: 0.4,  # Mix of uniform and empirical for multinomial
}

# Total number of settings (determines matrix width)
MAX_SETTINGS = len(SettingSlot)


def build_settings_matrix(specs):
    """
    Convert BlockSpec settings dicts into a JAX matrix.

    Args:
        specs: List of BlockSpec objects

    Returns:
        JAX array of shape (n_blocks, MAX_SETTINGS) containing all settings
    """
    n_blocks = len(specs)

    # Initialize with defaults
    matrix = np.zeros((n_blocks, MAX_SETTINGS), dtype=np.float32)
    for slot, default in SETTING_DEFAULTS.items():
        matrix[:, slot] = default

    # Map string keys to slots
    key_to_slot = {
        'chain_prob': SettingSlot.CHAIN_PROB,
        'n_categories': SettingSlot.N_CATEGORIES,
        'cov_mult': SettingSlot.COV_MULT,
        'uniform_weight': SettingSlot.UNIFORM_WEIGHT,
    }

    # Fill in specified values
    for i, spec in enumerate(specs):
        if spec.settings:
            for key, value in spec.settings.items():
                if key in key_to_slot:
                    matrix[i, key_to_slot[key]] = float(value)
                # Unknown keys are silently ignored

    return jnp.array(matrix)
