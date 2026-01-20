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
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath):
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
    print(f"Resuming from checkpoint at iteration {checkpoint['iteration']} (resetting run counter to 0)")
    print(f"Structure: {K} Superchains × {M} Subchains")

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
        print(f"Parallel Tempering: {checkpoint['n_temperatures']} temperatures (DEO parity: {int(swap_parity)})")
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
            print(f"Parallel Tempering: {n_temperatures} temperatures (fresh init)")
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
        metadata: Dict with K, M, mcmc_config, batch_boundaries, temperature_history
    """
    if not batch_paths:
        raise ValueError("No batch paths provided")

    histories = []
    iterations_list = []
    likelihoods_list = []
    temp_history_list = []
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

            # Load temperature_history if present (index process parallel tempering)
            if 'temperature_history' in data:
                th = data['temperature_history']
                # Check if it's a real temperature history (not placeholder)
                if th is not None and th.ndim == 2 and th.shape[0] > 1:
                    temp_history_list.append(th.copy())

            # Get metadata from first batch
            if metadata is None:
                mcmc_config = data['mcmc_config'].item() if 'mcmc_config' in data else {}
                metadata = {
                    'K': int(data['K']),
                    'M': int(data['M']),
                    'mcmc_config': mcmc_config,
                    'thin_iteration': int(data['thin_iteration']) if 'thin_iteration' in data else 1,
                    'n_temperatures': int(mcmc_config.get('n_temperatures', 1)) if mcmc_config else 1,
                }

    # Concatenate along sample axis (axis=0)
    combined_history = np.concatenate(histories, axis=0)
    combined_iterations = np.concatenate(iterations_list, axis=0)

    combined_likelihoods = None
    if likelihoods_list:
        combined_likelihoods = np.concatenate(likelihoods_list, axis=0)

    # Combine temperature history if present
    combined_temp_history = None
    if temp_history_list:
        combined_temp_history = np.concatenate(temp_history_list, axis=0)
        metadata['temperature_history'] = combined_temp_history

    # Compute batch boundaries (start indices for each batch)
    batch_boundaries = np.cumsum([0] + batch_sizes[:-1]).tolist()
    metadata['batch_boundaries'] = batch_boundaries
    metadata['batch_sizes'] = batch_sizes
    metadata['n_batches'] = len(batch_paths)

    print(f"Combined {len(batch_paths)} batches:")
    print(f"  Total samples: {combined_history.shape[0]}")
    print(f"  Iteration range: {combined_iterations[0]} - {combined_iterations[-1]}")
    print(f"  Batch boundaries: {batch_boundaries}")
    if combined_temp_history is not None:
        print(f"  Temperature history: {combined_temp_history.shape}")

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


def compute_rhat_from_history(history, K, M, n_temperatures=1):
    """
    Compute nested R-hat on combined/filtered history (CPU version).

    This is a NumPy implementation for post-hoc analysis of saved history.
    Matches the GPU implementation in mcmc_diagnostics.py.

    Args:
        history: Sample history (num_samples, num_chains, num_cols)
        K: Number of superchains
        M: Number of subchains per superchain
        n_temperatures: Number of temperature levels (for parallel tempering)
            When n_temperatures > 1, only beta=1 chains are in history,
            so K_effective = K // n_temperatures

    Returns:
        rhat: R-hat values for each parameter (num_cols,)
    """
    n_samples, n_chains, n_cols = history.shape

    # Adjust K for parallel tempering (only beta=1 chains are in history)
    if n_temperatures > 1:
        K_effective = K // n_temperatures
        print(f"  (Tempering active: using K={K_effective} of {K} superchains for beta=1 chains)")
        K = K_effective

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

    Supports two directory structures:
    1. New nested structure (preferred):
       {output_dir}/{model_name}/checkpoints/checkpoint_NNN.npz
       {output_dir}/{model_name}/history/full/history_NNN.npz

    2. Legacy flat structure (for backwards compatibility):
       {output_dir}/{model_name}_checkpoint{N}.npz
       {output_dir}/{model_name}_history_{NNN}.npz

    Args:
        output_dir: Base output directory
        model_name: Model name

    Returns:
        Dict with:
            - checkpoint_files: List of (run_index, path) tuples, sorted by index
            - history_files: List of (run_index, path) tuples, sorted by index
            - latest_run_index: Highest run index found (-1 if none)
            - latest_checkpoint: Path to most recent checkpoint (None if none)
            - structure: 'nested' or 'flat' indicating which structure was found
    """
    import re

    output_path = Path(output_dir)

    checkpoint_files = []
    history_files = []
    structure = 'flat'  # Default to flat structure

    # Check for new nested structure first
    nested_base = output_path / model_name
    nested_checkpoints = nested_base / 'checkpoints'
    nested_history = nested_base / 'history' / 'full'

    if nested_checkpoints.exists() or nested_history.exists():
        structure = 'nested'

        # Scan nested checkpoint directory
        if nested_checkpoints.exists():
            # Pattern: checkpoint_NNN.npz
            checkpoint_pattern = re.compile(r'^checkpoint_(\d+)\.npz$')
            for f in nested_checkpoints.iterdir():
                if f.is_file():
                    cp_match = checkpoint_pattern.match(f.name)
                    if cp_match:
                        run_idx = int(cp_match.group(1))
                        checkpoint_files.append((run_idx, str(f)))

        # Scan nested history/full directory
        if nested_history.exists():
            # Pattern: history_NNN.npz
            history_pattern = re.compile(r'^history_(\d+)\.npz$')
            for f in nested_history.iterdir():
                if f.is_file():
                    hist_match = history_pattern.match(f.name)
                    if hist_match:
                        run_idx = int(hist_match.group(1))
                        history_files.append((run_idx, str(f)))

    # Fall back to legacy flat structure
    if not checkpoint_files and not history_files and output_path.exists():
        structure = 'flat'
        # Pattern: {model}_checkpoint{N}.npz or {model}_checkpoint{N}_suffix.npz
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
        'structure': structure,
    }


def get_model_paths(output_dir: str, model_name: str):
    """
    Get directory paths for the nested structure.

    Args:
        output_dir: Base output directory (e.g., '../data/output/dbar_fed0')
        model_name: Model name (e.g., 'mix2_EH_bhm')

    Returns:
        Dict with paths:
            - base: {output_dir}/{model_name}/
            - checkpoints: {base}/checkpoints/
            - history_full: {base}/history/full/
            - history_per_subject: {base}/history/per_subject/
    """
    base = Path(output_dir) / model_name
    return {
        'base': base,
        'checkpoints': base / 'checkpoints',
        'history_full': base / 'history' / 'full',
        'history_per_subject': base / 'history' / 'per_subject',
    }


def ensure_model_dirs(output_dir: str, model_name: str):
    """
    Create the nested directory structure for a model.

    Args:
        output_dir: Base output directory
        model_name: Model name

    Returns:
        Dict with paths (same as get_model_paths)
    """
    paths = get_model_paths(output_dir, model_name)
    paths['checkpoints'].mkdir(parents=True, exist_ok=True)
    paths['history_full'].mkdir(parents=True, exist_ok=True)
    paths['history_per_subject'].mkdir(parents=True, exist_ok=True)
    return paths


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

    Handles both nested and flat directory structures.

    Args:
        output_dir: Directory containing the files
        model_name: Model name prefix
        mode: Cleaning mode:
            - 'all': Delete all checkpoints and histories
            - 'keep_latest': Keep latest checkpoint, delete all else

    Returns:
        dict with 'deleted_checkpoints', 'deleted_histories', 'kept_checkpoint'
    """
    import shutil

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

    # Delete all histories (from full/)
    for run_idx, hist_path in scan['history_files']:
        try:
            Path(hist_path).unlink()
            deleted_histories.append(hist_path)
        except OSError as e:
            print(f"  Warning: Could not delete {hist_path}: {e}")

    # For nested structure, also clean per_subject directory
    if scan.get('structure') == 'nested':
        paths = get_model_paths(output_dir, model_name)
        per_subject_dir = paths['history_per_subject']
        if per_subject_dir.exists():
            try:
                shutil.rmtree(per_subject_dir)
                per_subject_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"  Warning: Could not clean per_subject directory: {e}")

    return {
        'deleted_checkpoints': deleted_checkpoints,
        'deleted_histories': deleted_histories,
        'kept_checkpoint': kept_checkpoint,
    }


def split_history_by_subject(
    history_path: str,
    output_base: str,
    n_subjects: int,
    n_hyper: int = 32,
    params_per_subject: int = 13,
    hyper_first: bool = False,
    verbose: bool = True,
):
    """
    Split a full history file into per-subject files.

    This is a post-processing step to enable memory-efficient analysis.
    Instead of loading all 44GB at once, we can load ~170MB per subject.

    Args:
        history_path: Path to full history file (e.g., history/full/history_000.npz)
        output_base: Base directory for per-subject output (e.g., history/per_subject)
        n_subjects: Number of subjects in the model
        n_hyper: Number of hyperparameters (saved in a shared file)
        params_per_subject: Number of parameters per subject
        hyper_first: If True, layout is [hyper][subjects]. If False (default),
                     layout is [subjects][hyper] (standard BHM layout).
        verbose: Print progress

    Returns:
        dict with:
            - hyper_path: Path to hyperparameter file
            - subject_paths: List of (subject_idx, path) tuples
            - n_chains: Number of chains in history
            - n_samples: Number of samples per chain

    Output structure:
        {output_base}/
            hyperparameters/history_NNN.npz  - hyperparameters only
            subject_000/history_NNN.npz      - subject 0 params
            subject_001/history_NNN.npz      - subject 1 params
            ...
    """
    import numpy as np
    from pathlib import Path

    history_path = Path(history_path)
    output_base = Path(output_base)

    # Extract run index from filename (e.g., history_010.npz -> 010)
    run_idx = history_path.stem.split('_')[-1]

    if verbose:
        print(f"Splitting {history_path.name}...")

    # Load the full history
    data = np.load(history_path, allow_pickle=True)
    history = data['history']  # Shape: (n_samples, n_chains, n_params)

    n_samples, n_chains, n_params = history.shape

    # Calculate expected params (excluding generated quantities)
    subject_total = n_subjects * params_per_subject
    expected_params = n_hyper + subject_total
    if n_params < expected_params:
        raise ValueError(
            f"History has {n_params} params, expected at least {expected_params} "
            f"({n_hyper} hyper + {n_subjects} × {params_per_subject})"
        )

    # Get metadata from source file - copy all important fields
    iterations = data['iterations'] if 'iterations' in data else None
    likelihoods = data['likelihoods'] if 'likelihoods' in data else None

    # Preserve all top-level metadata fields (K, M, temperature_history, etc.)
    metadata_fields = {}
    for key in ['K', 'M', 'thin_iteration', 'run_index', 'mcmc_config', 'diagnostics',
                'temperature_history', 'n_temperatures']:
        if key in data:
            metadata_fields[key] = data[key]

    subject_paths = []

    # Calculate index ranges based on layout
    if hyper_first:
        # Layout: [hyper(n_hyper)][subjects(n_subjects × params_per_subject)][gq...]
        hyper_start = 0
        hyper_end = n_hyper
        subject_base = n_hyper
    else:
        # Layout: [subjects(n_subjects × params_per_subject)][hyper(n_hyper)][gq...]
        # This is the standard BHM layout
        subject_base = 0
        hyper_start = subject_total
        hyper_end = subject_total + n_hyper

    # Save hyperparameters (with all metadata - only needed in hyper file)
    hyper_dir = output_base / 'hyperparameters'
    hyper_dir.mkdir(parents=True, exist_ok=True)
    hyper_path = hyper_dir / f'history_{run_idx}.npz'

    hyper_history = history[:, :, hyper_start:hyper_end]
    save_dict = {'history': hyper_history}
    if iterations is not None:
        save_dict['iterations'] = iterations
    if likelihoods is not None:
        save_dict['likelihoods'] = likelihoods
    # Add all metadata fields
    save_dict.update(metadata_fields)

    np.savez_compressed(hyper_path, **save_dict)

    if verbose:
        print(f"  Saved hyperparameters: {hyper_path.name} ({hyper_history.nbytes / 1e6:.1f} MB)")

    # Save each subject's parameters
    for subj_idx in range(n_subjects):
        subj_dir = output_base / f'subject_{subj_idx:03d}'
        subj_dir.mkdir(parents=True, exist_ok=True)
        subj_path = subj_dir / f'history_{run_idx}.npz'

        # Calculate subject's parameter indices
        start_idx = subject_base + subj_idx * params_per_subject
        end_idx = start_idx + params_per_subject

        subj_history = history[:, :, start_idx:end_idx]

        save_dict = {'history': subj_history, 'subject_idx': subj_idx}
        if iterations is not None:
            save_dict['iterations'] = iterations
        # Don't duplicate likelihoods or full metadata for each subject (waste of space)

        np.savez_compressed(subj_path, **save_dict)
        subject_paths.append((subj_idx, str(subj_path)))

    if verbose:
        per_subj_mb = n_samples * n_chains * params_per_subject * 8 / 1e6
        print(f"  Saved {n_subjects} subject files (~{per_subj_mb:.1f} MB each)")

    return {
        'hyper_path': str(hyper_path),
        'subject_paths': subject_paths,
        'n_chains': n_chains,
        'n_samples': n_samples,
    }


def postprocess_all_histories(
    output_dir: str,
    model_name: str,
    n_subjects: int,
    n_hyper: int = 32,
    params_per_subject: int = 13,
    hyper_first: bool = False,
    verbose: bool = True,
):
    """
    Post-process all history files for a model, splitting into per-subject files.

    Args:
        output_dir: Base output directory (e.g., ../data/output/dbar_fed0)
        model_name: Model name
        n_subjects: Number of subjects
        n_hyper: Number of hyperparameters
        params_per_subject: Parameters per subject
        hyper_first: If True, layout is [hyper][subjects]. If False (default),
                     layout is [subjects][hyper] (standard BHM layout).

    Returns:
        dict with summary of processed files
    """
    from pathlib import Path

    paths = get_model_paths(output_dir, model_name)
    full_history_dir = paths['history_full']
    per_subject_dir = paths['history_per_subject']

    if not full_history_dir.exists():
        print(f"No history directory found: {full_history_dir}")
        return {'processed': 0}

    # Find all history files
    history_files = sorted(full_history_dir.glob('history_*.npz'))

    if not history_files:
        print(f"No history files found in {full_history_dir}")
        return {'processed': 0}

    if verbose:
        print(f"Post-processing {len(history_files)} history files...")
        total_size = sum(f.stat().st_size for f in history_files)
        print(f"  Total size: {total_size / 1e9:.1f} GB")

    results = []
    for hist_path in history_files:
        result = split_history_by_subject(
            str(hist_path),
            str(per_subject_dir),
            n_subjects=n_subjects,
            n_hyper=n_hyper,
            params_per_subject=params_per_subject,
            hyper_first=hyper_first,
            verbose=verbose,
        )
        results.append(result)

    return {
        'processed': len(results),
        'results': results,
    }
