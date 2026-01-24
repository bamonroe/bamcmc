"""
History processing utilities for MCMC output.

This module provides functions for:
- Combining batch history files from multiple runs
- Applying burn-in filtering
- Computing R-hat diagnostics on combined/filtered data
- Splitting history files for memory-efficient analysis
"""

import numpy as np
from pathlib import Path


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

    print(f"Combined {len(batch_paths)} batches:", flush=True)
    print(f"  Total samples: {combined_history.shape[0]}", flush=True)
    print(f"  Iteration range: {combined_iterations[0]} - {combined_iterations[-1]}", flush=True)
    print(f"  Batch boundaries: {batch_boundaries}", flush=True)
    if combined_temp_history is not None:
        print(f"  Temperature history: {combined_temp_history.shape}", flush=True)

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

    print(f"Burn-in filter (min_iteration={min_iteration}):", flush=True)
    print(f"  Dropped: {n_dropped} samples (iterations {iterations[0]} - {min_iteration - 1})", flush=True)
    print(f"  Kept: {n_kept} samples (iterations {min_iteration} - {iterations[-1]})", flush=True)

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
        print(f"  (Tempering active: using K={K_effective} of {K} superchains for beta=1 chains)", flush=True)
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

    print(f"Nested R-hat ({K}×{M}):", flush=True)
    print(f"  Max: {np.nanmax(rhat):.4f}", flush=True)
    print(f"  Median: {np.nanmedian(rhat):.4f}", flush=True)
    print(f"  Min: {np.nanmin(rhat):.4f}", flush=True)

    # Check for NaN/Inf
    n_nan = np.sum(~np.isfinite(rhat))
    if n_nan > 0:
        print(f"  WARNING: {n_nan} params have NaN/Inf R-hat (zero variance - stuck or discrete)", flush=True)

    # Threshold from Margossian et al. (2022)
    tau = 1e-4
    threshold = np.sqrt(1 + 1/M + tau)
    if np.nanmax(rhat) < threshold:
        print(f"  Converged (max < {threshold:.4f})", flush=True)
    else:
        print(f"  Not converged (max = {np.nanmax(rhat):.4f} >= {threshold:.4f})", flush=True)

    return rhat


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
    history_path = Path(history_path)
    output_base = Path(output_base)

    # Extract run index from filename (e.g., history_010.npz -> 010)
    run_idx = history_path.stem.split('_')[-1]

    if verbose:
        print(f"Splitting {history_path.name}...", flush=True)

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
        print(f"  Saved hyperparameters: {hyper_path.name} ({hyper_history.nbytes / 1e6:.1f} MB)", flush=True)

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
        print(f"  Saved {n_subjects} subject files (~{per_subj_mb:.1f} MB each)", flush=True)

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
    # Import here to avoid circular dependency
    from .output_management import get_model_paths

    paths = get_model_paths(output_dir, model_name)
    full_history_dir = paths['history_full']
    per_subject_dir = paths['history_per_subject']

    if not full_history_dir.exists():
        print(f"No history directory found: {full_history_dir}", flush=True)
        return {'processed': 0}

    # Find all history files
    history_files = sorted(full_history_dir.glob('history_*.npz'))

    if not history_files:
        print(f"No history files found in {full_history_dir}", flush=True)
        return {'processed': 0}

    if verbose:
        print(f"Post-processing {len(history_files)} history files...", flush=True)
        total_size = sum(f.stat().st_size for f in history_files)
        print(f"  Total size: {total_size / 1e9:.1f} GB", flush=True)

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
