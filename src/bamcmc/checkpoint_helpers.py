"""
Checkpoint and batch history utilities for MCMC runs.

This module provides functions for:
- Saving and loading MCMC checkpoints for resumable runs
- Combining batch history files from multiple runs
- Applying burn-in filtering to combined histories
- Computing R-hat diagnostics on combined/filtered data
"""

import numpy as np
from pathlib import Path


def save_checkpoint(filepath, carry, user_config, metadata=None):
    """
    Save MCMC checkpoint to disk for resuming later.

    Args:
        filepath: Path to save checkpoint (.npz file)
        carry: Current MCMC carry tuple from run
        user_config: User configuration dict (serializable, no JAX types)
        metadata: Optional dict of additional metadata

    Saves:
        - Chain states (A and B groups)
        - Random keys for each chain
        - Iteration count
        - Acceptance counts per block
        - Config needed for validation on resume
    """
    states_A, keys_A, states_B, keys_B, _, _, acceptance_counts, iteration = carry

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

    if metadata:
        checkpoint['metadata'] = metadata

    filepath = Path(filepath)
    np.savez_compressed(filepath, **checkpoint)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath):
    """
    Load MCMC checkpoint from disk.

    Args:
        filepath: Path to checkpoint file (.npz)

    Returns:
        Dict with checkpoint data including states, keys, iteration, etc.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    data = np.load(filepath, allow_pickle=True)

    checkpoint = {
        'states_A': data['states_A'],
        'states_B': data['states_B'],
        'keys_A': data['keys_A'],
        'keys_B': data['keys_B'],
        'iteration': int(data['iteration']),
        'acceptance_counts': data['acceptance_counts'],
        'posterior_id': str(data['posterior_id']),
        'num_params': int(data['num_params']),
        'num_chains_a': int(data['num_chains_a']),
        'num_chains_b': int(data['num_chains_b']),
        'num_superchains': int(data['num_superchains']),
        'subchains_per_super': int(data['subchains_per_super']),
    }

    if 'metadata' in data:
        checkpoint['metadata'] = data['metadata'].item()

    return checkpoint


def initialize_from_checkpoint(checkpoint, user_config, runtime_ctx, num_gq, num_collect, num_blocks):
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
        initial_carry: Tuple ready for MCMC scan
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

    # Fresh history array for this run
    total_cols = num_params + num_gq
    initial_history = jnp.empty((num_collect, num_chains, total_cols), dtype=dtype)

    if user_config['save_likelihoods']:
        initial_lik_history = jnp.empty((num_collect, num_chains), dtype=dtype)
    else:
        initial_lik_history = jnp.empty((1,), dtype=dtype)

    # Restore acceptance counts (cumulative across runs)
    acceptance_counts = jnp.asarray(checkpoint['acceptance_counts'], dtype=jnp.int32)

    # Continue from previous iteration count
    current_iteration = jnp.array(checkpoint['iteration'], dtype=jnp.int32)

    K = checkpoint['num_superchains']
    M = checkpoint['subchains_per_super']
    print(f"Resuming from iteration {checkpoint['iteration']}")
    print(f"Structure: {K} Superchains × {M} Subchains")

    initial_carry = (states_A, keys_A, states_B, keys_B,
                     initial_history, initial_lik_history, acceptance_counts, current_iteration)

    return initial_carry, user_config


# --- BATCH HISTORY UTILITIES ---

def combine_batch_histories(batch_paths):
    """
    Combine multiple batch history files into a single dataset.

    Args:
        batch_paths: List of paths to batch history .npz files
                    (e.g., ['history_000.npz', 'history_001.npz', ...])

    Returns:
        history: Combined history array (total_samples, num_chains, num_cols)
        iterations: Combined iteration numbers (total_samples,)
        likelihoods: Combined likelihoods (total_samples, num_chains) or None
        metadata: Dict with K, M, mcmc_config, batch_boundaries from first batch
    """
    if not batch_paths:
        raise ValueError("No batch paths provided")

    histories = []
    iterations_list = []
    likelihoods_list = []
    batch_sizes = []  # Track size of each batch for boundaries
    metadata = None

    for i, path in enumerate(batch_paths):
        print(f"Loading batch {i}: {path}...", flush=True)
        data = np.load(path, allow_pickle=True)

        hist = data['history']
        histories.append(hist)
        batch_sizes.append(hist.shape[0])
        iterations_list.append(data['iterations'])

        if 'likelihoods' in data and data['likelihoods'] is not None:
            lik = data['likelihoods']
            # Handle case where likelihoods might be stored as object
            if hasattr(lik, 'item') and lik.shape == ():
                lik = lik.item()
            if lik is not None and not (isinstance(lik, np.ndarray) and lik.shape == (1,)):
                likelihoods_list.append(lik)

        # Get metadata from first batch
        if metadata is None:
            metadata = {
                'K': int(data['K']),
                'M': int(data['M']),
                'mcmc_config': data['mcmc_config'].item() if 'mcmc_config' in data else None,
                'thin_iteration': int(data['thin_iteration']) if 'thin_iteration' in data else 1,
            }

    # Concatenate along sample axis (axis=0)
    combined_history = np.concatenate(histories, axis=0)
    combined_iterations = np.concatenate(iterations_list, axis=0)

    combined_likelihoods = None
    if likelihoods_list:
        combined_likelihoods = np.concatenate(likelihoods_list, axis=0)

    # Compute batch boundaries (start indices for each batch)
    batch_boundaries = np.cumsum([0] + batch_sizes[:-1]).tolist()
    metadata['batch_boundaries'] = batch_boundaries
    metadata['batch_sizes'] = batch_sizes
    metadata['n_batches'] = len(batch_paths)

    print(f"Combined {len(batch_paths)} batches:")
    print(f"  Total samples: {combined_history.shape[0]}")
    print(f"  Iteration range: {combined_iterations[0]} - {combined_iterations[-1]}")
    print(f"  Batch boundaries: {batch_boundaries}")

    return combined_history, combined_iterations, combined_likelihoods, metadata


def apply_burnin(history, iterations, likelihoods=None, min_iteration=0):
    """
    Drop samples before min_iteration (burn-in removal).

    Args:
        history: Sample history array (num_samples, num_chains, num_cols)
        iterations: Iteration numbers (num_samples,)
        likelihoods: Optional likelihood history (num_samples, num_chains)
        min_iteration: Discard all samples with iteration < min_iteration

    Returns:
        Filtered history, iterations, likelihoods (same structure, fewer samples)
    """
    mask = iterations >= min_iteration
    n_dropped = np.sum(~mask)
    n_kept = np.sum(mask)

    print(f"Burn-in filter (min_iteration={min_iteration}):")
    print(f"  Dropped: {n_dropped} samples (iterations {iterations[0]} - {min_iteration - 1})")
    print(f"  Kept: {n_kept} samples (iterations {min_iteration} - {iterations[-1]})")

    filtered_history = history[mask]
    filtered_iterations = iterations[mask]
    filtered_likelihoods = likelihoods[mask] if likelihoods is not None else None

    return filtered_history, filtered_iterations, filtered_likelihoods


def compute_rhat_from_history(history, K, M):
    """
    Compute nested R-hat on combined/filtered history (CPU version).

    This is a NumPy implementation for post-hoc analysis of saved history.

    Args:
        history: Sample history (num_samples, num_chains, num_cols)
        K: Number of superchains
        M: Number of subchains per superchain

    Returns:
        rhat: R-hat values for each parameter (num_cols,)
    """
    n_samples, n_chains, n_cols = history.shape

    if n_chains != K * M:
        raise ValueError(f"Chain count {n_chains} doesn't match K={K} × M={M} = {K*M}")

    # Reshape to (n_samples, K, M, n_cols)
    reshaped = history.reshape(n_samples, K, M, n_cols)

    # Superchain means: (K, n_cols)
    super_means = np.mean(reshaped, axis=(0, 2))

    # Subchain means: (K, M, n_cols)
    sub_means = np.mean(reshaped, axis=0)

    # Between-superchain variance: B_super
    B_super = n_samples * M * np.var(super_means, axis=0, ddof=1)

    # Between-subchain within-superchain variance: B_sub
    B_sub = n_samples * np.mean(np.var(sub_means, axis=1, ddof=1), axis=0)

    # Within-chain variance: W
    W = np.mean(np.var(history, axis=0, ddof=1), axis=0)

    # Nested R-hat formula
    n = n_samples
    var_hat = ((n - 1) / n) * W + (1 / n) * B_sub + (1 / (n * M)) * B_super

    # R-hat = sqrt(var_hat / W)
    rhat = np.sqrt(var_hat / (W + 1e-10))

    # Handle degenerate cases
    rhat = np.where(np.isfinite(rhat), rhat, 1.0)

    print(f"Nested R-hat ({K}×{M}):")
    print(f"  Max: {np.max(rhat):.4f}")
    print(f"  Median: {np.median(rhat):.4f}")
    print(f"  Min: {np.min(rhat):.4f}")

    # Threshold from Margossian et al. (2022)
    tau = 1e-4
    threshold = np.sqrt(1 + 1/M + tau)
    if np.max(rhat) < threshold:
        print(f"  Converged (max < {threshold:.4f})")
    else:
        print(f"  Not converged (max = {np.max(rhat):.4f} >= {threshold:.4f})")

    return rhat
