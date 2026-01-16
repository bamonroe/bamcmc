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

    # Fresh history array for this run (use zeros to avoid uninitialized values)
    total_cols = num_params + num_gq
    initial_history = jnp.zeros((num_collect, num_chains, total_cols), dtype=dtype)

    if user_config['save_likelihoods']:
        initial_lik_history = jnp.empty((num_collect, num_chains), dtype=dtype)
    else:
        initial_lik_history = jnp.empty((1,), dtype=dtype)

    # Reset acceptance counts for this run (each run reports its own rates)
    acceptance_counts = jnp.zeros(num_blocks, dtype=jnp.int32)

    # Reset iteration to 0 for this run (kernel compiled with START_ITERATION=0)
    # Global iteration tracking is handled via iteration_offset in user_config
    current_iteration = jnp.array(0, dtype=jnp.int32)

    K = checkpoint['num_superchains']
    M = checkpoint['subchains_per_super']
    print(f"Resuming from checkpoint at iteration {checkpoint['iteration']} (resetting run counter to 0)")
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
        # Use context manager to close NpzFile after loading
        with np.load(path, allow_pickle=True) as data:
            hist = data['history'].copy()
            histories.append(hist)
            batch_sizes.append(hist.shape[0])
            iterations_list.append(data['iterations'].copy())

            if 'likelihoods' in data and data['likelihoods'] is not None:
                lik = data['likelihoods']
                # Handle case where likelihoods might be stored as object
                if hasattr(lik, 'item') and lik.shape == ():
                    lik = lik.item()
                if lik is not None and not (isinstance(lik, np.ndarray) and lik.shape == (1,)):
                    likelihoods_list.append(lik.copy() if hasattr(lik, 'copy') else lik)

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
    Matches the GPU implementation in mcmc_diagnostics.py.

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

    # 1. Reshape to Superchain structure: (n_samples, K, M, n_cols)
    history_nested = history.reshape(n_samples, K, M, n_cols)

    # 2. Compute Superchain Means (averaging over M subchains)
    superchain_means_over_time = np.mean(history_nested, axis=2)  # (n_samples, K, n_cols)

    # 3. Compute Grand Means per Superchain (averaging over time)
    superchain_means = np.mean(superchain_means_over_time, axis=0)  # (K, n_cols)

    # 4. Between-Superchain Variance (B)
    B = n_samples * np.var(superchain_means, axis=0, ddof=1)

    # 5. Within-Superchain Variance (W)
    # Component A: Between-subchain variance (only exists if M > 1)
    if M > 1:
        subchain_means_t = np.mean(history_nested, axis=0)  # (K, M, n_cols)
        B_within = np.var(subchain_means_t, axis=1, ddof=1)  # (K, n_cols)
    else:
        B_within = np.zeros((K, n_cols))

    # Component B: Within-subchain variance (variance over time)
    if n_samples > 1:
        W_within_raw = np.var(history_nested, axis=0, ddof=1)  # (K, M, n_cols)
        W_within = np.mean(W_within_raw, axis=1)  # (K, n_cols)
    else:
        W_within = np.zeros((K, n_cols))

    # Total Within Variance: Average over all K superchains
    W = np.mean(B_within + W_within, axis=0)

    # 6. Final Ratio using Gelman-Rubin formula
    m = K
    n = n_samples
    V_hat = ((n - 1) / n) * W + B / n + B / (m * n)

    # R-hat = sqrt(V_hat / W)
    # No special handling - stuck continuous params will show as NaN/inf
    rhat = np.sqrt(V_hat / W)

    print(f"Nested R-hat ({K}×{M}):")
    print(f"  Max: {np.nanmax(rhat):.4f}")
    print(f"  Median: {np.nanmedian(rhat):.4f}")
    print(f"  Min: {np.nanmin(rhat):.4f}")

    # Check for NaN/Inf
    n_nan = np.sum(~np.isfinite(rhat))
    if n_nan > 0:
        print(f"  WARNING: {n_nan} params have NaN/Inf R-hat (zero variance - stuck or discrete)")

    # Threshold from Margossian et al. (2022)
    tau = 1e-4
    threshold = np.sqrt(1 + 1/M + tau)
    if np.nanmax(rhat) < threshold:
        print(f"  Converged (max < {threshold:.4f})")
    else:
        print(f"  Not converged (max = {np.nanmax(rhat):.4f} >= {threshold:.4f})")

    return rhat



def scan_checkpoints(output_dir: str, model_name: str):
    """
    Scan output directory for existing checkpoints and history files.

    Args:
        output_dir: Directory containing checkpoint files
        model_name: Model name prefix for checkpoint files

    Returns:
        Dict with:
            - checkpoint_files: List of (run_index, path) tuples, sorted by index
            - history_files: List of (run_index, path) tuples, sorted by index
            - latest_run_index: Highest run index found (-1 if none)
            - latest_checkpoint: Path to most recent checkpoint (None if none)
    """
    import re

    output_path = Path(output_dir)

    checkpoint_files = []
    history_files = []

    if output_path.exists():
        # Pattern: {model}_checkpoint{N}.npz or {model}_checkpoint{N}_suffix.npz
        # Matches: model_checkpoint0.npz, model_checkpoint1.npz, model_checkpoint0_reset.npz
        checkpoint_pattern = re.compile(rf'^{re.escape(model_name)}_checkpoint(\d+)(_[a-z]+)?\.npz$')
        # Pattern: {model}_history_{NNN}.npz
        history_pattern = re.compile(rf'^{re.escape(model_name)}_history_(\d+)\.npz$')

        for f in output_path.iterdir():
            if f.is_file():
                cp_match = checkpoint_pattern.match(f.name)
                if cp_match:
                    run_idx = int(cp_match.group(1))
                    checkpoint_files.append((run_idx, str(f)))

                hist_match = history_pattern.match(f.name)
                if hist_match:
                    run_idx = int(hist_match.group(1))
                    history_files.append((run_idx, str(f)))

    # Sort by run index
    checkpoint_files.sort(key=lambda x: x[0])
    history_files.sort(key=lambda x: x[0])

    latest_run_index = checkpoint_files[-1][0] if checkpoint_files else -1
    latest_checkpoint = checkpoint_files[-1][1] if checkpoint_files else None

    return {
        'checkpoint_files': checkpoint_files,
        'history_files': history_files,
        'latest_run_index': latest_run_index,
        'latest_checkpoint': latest_checkpoint,
    }


def get_latest_checkpoint(output_dir: str, model_name: str):
    """
    Get path to the most recent checkpoint for a model.

    Args:
        output_dir: Directory containing checkpoint files
        model_name: Model name prefix

    Returns:
        Path to latest checkpoint, or None if no checkpoints exist
    """
    scan = scan_checkpoints(output_dir, model_name)
    return scan['latest_checkpoint']


def clean_model_files(output_dir: str, model_name: str, mode: str = 'all'):
    """
    Delete checkpoint and history files for a model.

    Args:
        output_dir: Directory containing the files
        model_name: Model name prefix
        mode: Cleaning mode:
            - 'all': Delete all checkpoints and histories
            - 'keep_latest': Keep latest checkpoint, delete all else

    Returns:
        dict with 'deleted_checkpoints', 'deleted_histories', 'kept_checkpoint'
    """
    scan = scan_checkpoints(output_dir, model_name)

    deleted_checkpoints = []
    deleted_histories = []
    kept_checkpoint = None

    # Determine which checkpoint to keep (if any)
    if mode == 'keep_latest' and scan['latest_checkpoint']:
        kept_checkpoint = scan['latest_checkpoint']

    # Delete checkpoints
    for run_idx, cp_path in scan['checkpoint_files']:
        if cp_path != kept_checkpoint:
            try:
                Path(cp_path).unlink()
                deleted_checkpoints.append(cp_path)
            except OSError as e:
                print(f"  Warning: Could not delete {cp_path}: {e}")

    # Delete all histories
    for run_idx, hist_path in scan['history_files']:
        try:
            Path(hist_path).unlink()
            deleted_histories.append(hist_path)
        except OSError as e:
            print(f"  Warning: Could not delete {hist_path}: {e}")

    return {
        'deleted_checkpoints': deleted_checkpoints,
        'deleted_histories': deleted_histories,
        'kept_checkpoint': kept_checkpoint,
    }
