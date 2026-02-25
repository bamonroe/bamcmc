"""
Checkpoint I/O utilities for saving and loading MCMC state.

This module provides functions for:
- Saving MCMC checkpoints to disk for resumable runs
- Loading checkpoints to resume or reset sampling
- Initializing MCMC carry from loaded checkpoints
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from pathlib import Path

import logging
logger = logging.getLogger('bamcmc')


def save_checkpoint(filepath: str, carry: tuple, user_config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save MCMC checkpoint to disk for resuming later.

    Args:
        filepath: Path to save checkpoint (.npz file)
        carry: Current MCMC carry tuple from run (13 elements with tempering)
        user_config: User configuration dict (serializable, no JAX types)
        metadata: Optional dict of additional metadata

    Saves:
        - Chain states (A and B groups)
        - Random keys for each chain
        - Iteration count
        - Acceptance counts per block
        - Config needed for validation on resume
        - Tempering state (if n_temperatures > 1)
    """
    # Unpack carry tuple (15 elements for index process parallel tempering)
    # Structure: states_A, keys_A, states_B, keys_B, history, lik_history, temp_history,
    #            acceptance_counts, iteration, temperature_ladder, temp_A, temp_B,
    #            swap_accepts, swap_attempts, swap_parity
    states_A = carry[0]
    keys_A = carry[1]
    states_B = carry[2]
    keys_B = carry[3]
    acceptance_counts = carry[7]
    iteration = carry[8]

    checkpoint = {
        'states_A': np.asarray(states_A),
        'states_B': np.asarray(states_B),
        'keys_A': np.asarray(keys_A),
        'keys_B': np.asarray(keys_B),
        'iteration': int(iteration),
        'acceptance_counts': np.asarray(acceptance_counts),
        # Validation metadata
        'posterior_id': user_config['posterior_id'],
        'num_params': user_config['num_params'],
        'num_chains_a': user_config['num_chains_a'],
        'num_chains_b': user_config['num_chains_b'],
        'num_superchains': user_config.get('num_superchains', 0),
        'subchains_per_super': user_config.get('subchains_per_super', 0),
    }

    # Add tempering state if using parallel tempering
    n_temperatures = user_config.get('n_temperatures', 1)
    if n_temperatures > 1 and len(carry) >= 15:
        checkpoint['n_temperatures'] = n_temperatures
        checkpoint['temperature_ladder'] = np.asarray(carry[9])
        checkpoint['temp_assignments_A'] = np.asarray(carry[10])
        checkpoint['temp_assignments_B'] = np.asarray(carry[11])
        checkpoint['swap_accepts'] = np.asarray(carry[12])
        checkpoint['swap_attempts'] = np.asarray(carry[13])
        checkpoint['swap_parity'] = int(carry[14])

    if metadata:
        checkpoint['metadata'] = metadata

    filepath = Path(filepath)
    np.savez_compressed(filepath, **checkpoint)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """
    Load MCMC checkpoint from disk.

    Args:
        filepath: Path to checkpoint file (.npz)

    Returns:
        Dict with checkpoint data including states, keys, iteration, etc.
        Also includes tempering state if present in checkpoint.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # Use context manager to ensure NpzFile is closed after loading
    # Copy arrays to avoid keeping memory-mapped file references
    with np.load(filepath, allow_pickle=True) as data:
        checkpoint = {
            'states_A': data['states_A'].copy(),
            'states_B': data['states_B'].copy(),
            'keys_A': data['keys_A'].copy(),
            'keys_B': data['keys_B'].copy(),
            'iteration': int(data['iteration']),
            'acceptance_counts': data['acceptance_counts'].copy(),
            'posterior_id': str(data['posterior_id']),
            'num_params': int(data['num_params']),
            'num_chains_a': int(data['num_chains_a']),
            'num_chains_b': int(data['num_chains_b']),
            'num_superchains': int(data['num_superchains']),
            'subchains_per_super': int(data['subchains_per_super']),
        }

        if 'metadata' in data:
            checkpoint['metadata'] = data['metadata'].item()

        # Load tempering state if present
        if 'n_temperatures' in data:
            checkpoint['n_temperatures'] = int(data['n_temperatures'])
            checkpoint['temperature_ladder'] = data['temperature_ladder'].copy()
            checkpoint['temp_assignments_A'] = data['temp_assignments_A'].copy()
            checkpoint['temp_assignments_B'] = data['temp_assignments_B'].copy()
            checkpoint['swap_accepts'] = data['swap_accepts'].copy()
            checkpoint['swap_attempts'] = data['swap_attempts'].copy()

    return checkpoint


def initialize_from_checkpoint(checkpoint: Dict[str, Any], user_config: Dict[str, Any], runtime_ctx: Dict[str, Any], num_gq: int, num_collect: int, num_blocks: int) -> Tuple[tuple, Dict[str, Any]]:
    """
    Initialize MCMC carry from a checkpoint.

    Args:
        checkpoint: Dict from load_checkpoint()
        user_config: User configuration dict (serializable, no JAX types)
        runtime_ctx: Runtime context dict with JAX-dependent objects (dtypes, data, keys)
        num_gq: Number of generated quantities
        num_collect: Number of samples to collect this run
        num_blocks: Number of parameter blocks

    Returns:
        initial_carry: Tuple ready for MCMC scan (13 elements with tempering)
        user_config: Updated config with restored checkpoint values
    """
    # Import JAX here to avoid import at module level (keeps module lightweight)
    import jax.numpy as jnp

    # Validate checkpoint matches current config
    if checkpoint['posterior_id'] != user_config['posterior_id']:
        raise ValueError(
            f"Checkpoint posterior '{checkpoint['posterior_id']}' doesn't match "
            f"current config '{user_config['posterior_id']}'"
        )

    if checkpoint['num_chains_a'] != user_config['num_chains_a']:
        raise ValueError(
            f"Checkpoint has {checkpoint['num_chains_a']} A-chains, "
            f"config has {user_config['num_chains_a']}"
        )

    if checkpoint['num_chains_b'] != user_config['num_chains_b']:
        raise ValueError(
            f"Checkpoint has {checkpoint['num_chains_b']} B-chains, "
            f"config has {user_config['num_chains_b']}"
        )

    dtype = runtime_ctx['jnp_float_dtype']
    num_chains = user_config['num_chains']
    num_chains_a = user_config['num_chains_a']
    num_params = checkpoint['num_params']

    # Update user_config with checkpoint values
    user_config = user_config.copy()
    user_config['num_params'] = int(num_params)
    user_config['num_superchains'] = int(checkpoint['num_superchains'])
    user_config['subchains_per_super'] = int(checkpoint['subchains_per_super'])

    # Convert numpy arrays to JAX arrays
    states_A = jnp.asarray(checkpoint['states_A'], dtype=dtype)
    states_B = jnp.asarray(checkpoint['states_B'], dtype=dtype)
    keys_A = jnp.asarray(checkpoint['keys_A'], dtype=jnp.uint32)
    keys_B = jnp.asarray(checkpoint['keys_B'], dtype=jnp.uint32)

    # Fresh history array for this run (use zeros to avoid uninitialized values)
    # Index process: save ALL chains (users filter to beta=1 post-hoc via temp_history)
    n_chains_to_save = user_config.get('n_chains_to_save', num_chains)
    total_cols = num_params + num_gq
    initial_history = jnp.zeros((num_collect, n_chains_to_save, total_cols), dtype=dtype)

    if user_config['save_likelihoods']:
        initial_lik_history = jnp.empty((num_collect, n_chains_to_save), dtype=dtype)
    else:
        initial_lik_history = jnp.empty((1,), dtype=dtype)

    # Reset acceptance counts for this run (each run reports its own rates)
    acceptance_counts = jnp.zeros(num_blocks, dtype=jnp.int32)

    # Reset iteration to 0 for this run (kernel compiled with START_ITERATION=0)
    # Global iteration tracking is handled via iteration_offset in user_config
    current_iteration = jnp.array(0, dtype=jnp.int32)

    K = checkpoint['num_superchains']
    M = checkpoint['subchains_per_super']
    logger.info(f"Resuming from checkpoint at iteration {checkpoint['iteration']} (resetting run counter to 0)")
    logger.info(f"Structure: {K} Superchains × {M} Subchains")

    # Handle tempering state
    # Use int64 for temp assignments when x64 mode is enabled (dtype is float64)
    int_dtype = jnp.int64 if dtype == jnp.float64 else jnp.int32
    n_temperatures = user_config.get('n_temperatures', 1)

    # Temperature history for index process (tracks temp index per chain per saved iteration)
    if n_temperatures > 1:
        initial_temp_history = jnp.zeros((num_collect, num_chains), dtype=int_dtype)
    else:
        initial_temp_history = jnp.empty((1,), dtype=int_dtype)

    if 'n_temperatures' in checkpoint:
        # Restore tempering state from checkpoint
        temperature_ladder = jnp.asarray(checkpoint['temperature_ladder'], dtype=dtype)
        temp_assignments_A = jnp.asarray(checkpoint['temp_assignments_A'], dtype=int_dtype)
        temp_assignments_B = jnp.asarray(checkpoint['temp_assignments_B'], dtype=int_dtype)
        # Reset swap counts for this run
        swap_accepts = jnp.zeros(max(1, n_temperatures - 1), dtype=jnp.int32)
        swap_attempts = jnp.zeros(max(1, n_temperatures - 1), dtype=jnp.int32)
        # Restore DEO parity if available, else default to 0 (even round)
        swap_parity = jnp.array(checkpoint.get('swap_parity', 0), dtype=jnp.int32)
        logger.info(f"Parallel Tempering: {checkpoint['n_temperatures']} temperatures (DEO parity: {int(swap_parity)})")
    else:
        # Create fresh tempering state (single temperature = no tempering)
        if n_temperatures > 1:
            # User wants tempering but checkpoint doesn't have it
            # Create fresh temperature assignments
            beta_min = user_config.get('beta_min', 0.1)
            temp_indices = jnp.arange(n_temperatures, dtype=int_dtype)
            temperature_ladder = jnp.power(beta_min, temp_indices / (n_temperatures - 1))
            chains_per_temp = num_chains // n_temperatures
            temp_assignments = jnp.repeat(jnp.arange(n_temperatures, dtype=int_dtype), chains_per_temp)
            temp_assignments_A, temp_assignments_B = jnp.split(temp_assignments, [num_chains_a], axis=0)
            logger.info(f"Parallel Tempering: {n_temperatures} temperatures (fresh init)")
        else:
            temperature_ladder = jnp.array([1.0], dtype=dtype)
            temp_assignments_A = jnp.zeros(num_chains_a, dtype=int_dtype)
            temp_assignments_B = jnp.zeros(num_chains - num_chains_a, dtype=int_dtype)
        swap_accepts = jnp.zeros(max(1, n_temperatures - 1), dtype=jnp.int32)
        swap_attempts = jnp.zeros(max(1, n_temperatures - 1), dtype=jnp.int32)
        swap_parity = jnp.array(0, dtype=jnp.int32)  # Start with even round

    # Extended carry tuple with index process tempering state (15 elements)
    initial_carry = (
        states_A, keys_A, states_B, keys_B,
        initial_history, initial_lik_history, initial_temp_history,
        acceptance_counts, current_iteration,
        temperature_ladder, temp_assignments_A, temp_assignments_B,
        swap_accepts, swap_attempts, swap_parity
    )

    return initial_carry, user_config
