"""
Chain Reset Utilities for MCMC Runs.

Provides functions to reset chains to cross-chain mean, bringing
straggler/stuck chains back into the fold while maintaining the
superchain structure.

Usage:
    from bamcmc.reset_utils import generate_reset_states

    # Load checkpoint and generate reset states
    checkpoint = load_checkpoint('checkpoint.npz')
    reset_states = generate_reset_states(
        checkpoint,
        model_type='mixture_3model_bhm',
        n_subjects=250,
        noise_scale=1.0,  # Use full posterior spread (default)
        rng_seed=42
    )
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pathlib import Path


def compute_chain_statistics(states_A: np.ndarray, states_B: np.ndarray) -> Dict[str, Any]:
    """
    Compute cross-chain mean and std for all parameters.

    Args:
        states_A: Array of shape (n_chains_a, n_params)
        states_B: Array of shape (n_chains_b, n_params)

    Returns:
        dict with 'mean' and 'std' arrays of shape (n_params,)
    """
    # Combine all chains
    all_states = np.vstack([states_A, states_B])

    return {
        'mean': np.mean(all_states, axis=0),
        'std': np.std(all_states, axis=0),
        'n_chains': all_states.shape[0],
        'n_params': all_states.shape[1],
    }


def get_discrete_param_indices(model_type: str, n_subjects: int) -> List[int]:
    """
    Get indices of discrete parameters that need special handling during reset.

    Discrete parameters (like model indicators z) must be sampled from their
    empirical distribution rather than using mean + noise like continuous params.

    First checks if the posterior has registered a get_discrete_param_indices function.
    Falls back to legacy get_special_param_indices for backward compatibility.

    Args:
        model_type: Model identifier string
        n_subjects: Number of subjects in the model

    Returns:
        list of discrete parameter indices
    """
    # First, try new simplified interface
    try:
        from .registry import get_posterior
        posterior_config = get_posterior(model_type)
        if 'get_discrete_param_indices' in posterior_config:
            return posterior_config['get_discrete_param_indices'](n_subjects)
        # Fall back to legacy interface
        if 'get_special_param_indices' in posterior_config:
            special = posterior_config['get_special_param_indices'](n_subjects)
            return special.get('z_indices', [])
    except (KeyError, ImportError):
        pass

    # Legacy hardcoded logic - extract just z_indices
    special = _get_legacy_special_indices(model_type, n_subjects)
    return special.get('z_indices', [])


def _get_legacy_special_indices(model_type: str, n_subjects: int) -> Dict[str, List[int]]:
    """Legacy function for backward compatibility with old posteriors."""
    if model_type == 'mixture_3model_bhm':
        # Layout per subject: M1(6) + M2(7) + M3(6) + z(1) = 20 params
        subject_block_size = 20
        z_offset = 19  # z is last param in subject block

        # r parameter offsets within subject block:
        # M1: delta(0), r(1), alpha(2), beta(3), mu_risk(4), mu_disc(5)
        # M2: delta(6), s(7), r(8), alpha(9), beta(10), mu_risk(11), mu_disc(12)
        # M3: k(13), r(14), alpha(15), beta(16), mu_risk(17), mu_disc(18)
        r_offsets = [1, 8, 14]  # r params in M1, M2, M3

        z_indices = [s * subject_block_size + z_offset for s in range(n_subjects)]
        r_indices = []
        for s in range(n_subjects):
            base = s * subject_block_size
            r_indices.extend([base + off for off in r_offsets])

        # Hyperparameters start after all subjects
        hyper_start = n_subjects * subject_block_size
        pi_indices = list(range(hyper_start, hyper_start + 3))

        # Hyperparameter r means (not logsd)
        # M1 hypers: 12 values (6 params x 2), r_mean is at index 2
        # M2 hypers: 14 values (7 params x 2), r_mean is at index 4 (after delta, s)
        # M3 hypers: 12 values (6 params x 2), r_mean is at index 2
        m1_hyper_start = hyper_start + 3
        m2_hyper_start = m1_hyper_start + 12
        m3_hyper_start = m2_hyper_start + 14

        hyper_r_indices = [
            m1_hyper_start + 2,   # M1 r mean
            m2_hyper_start + 4,   # M2 r mean
            m3_hyper_start + 2,   # M3 r mean
        ]
        r_indices.extend(hyper_r_indices)

        return {
            'z_indices': z_indices,
            'pi_indices': pi_indices,
            'r_indices': r_indices,
        }

    elif model_type == 'mixture_4model_bhm':
        # Layout per subject: M1(6) + M2(7) + M3(6) + M4(7) + z(1) = 27 params
        subject_block_size = 27
        z_offset = 26

        r_offsets = [1, 8, 14, 21]  # r params in M1, M2, M3, M4

        z_indices = [s * subject_block_size + z_offset for s in range(n_subjects)]
        r_indices = []
        for s in range(n_subjects):
            base = s * subject_block_size
            r_indices.extend([base + off for off in r_offsets])

        hyper_start = n_subjects * subject_block_size
        pi_indices = list(range(hyper_start, hyper_start + 4))

        return {
            'z_indices': z_indices,
            'pi_indices': pi_indices,
            'r_indices': r_indices,
        }

    elif model_type == 'mixture_2model_bhm':
        subject_block_size = 14  # M1(6) + M2(7) + z(1)
        z_offset = 13

        r_offsets = [1, 8]  # r params in M1, M2

        z_indices = [s * subject_block_size + z_offset for s in range(n_subjects)]
        r_indices = []
        for s in range(n_subjects):
            base = s * subject_block_size
            r_indices.extend([base + off for off in r_offsets])

        hyper_start = n_subjects * subject_block_size
        pi_indices = list(range(hyper_start, hyper_start + 2))

        return {
            'z_indices': z_indices,
            'pi_indices': pi_indices,
            'r_indices': r_indices,
        }

    else:
        # Non-mixture models: no special handling needed
        return {
            'z_indices': [],
            'pi_indices': [],
            'r_indices': [],
        }


def generate_reset_states(checkpoint: Dict[str, Any], model_type: str, n_subjects: int, K: int, noise_scale: float = 1.0, rng_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate K new starting states based on cross-chain mean and std.

    This creates K distinct starting points (one per superchain) by:
    1. Computing cross-chain mean and std for all parameters
    2. For continuous params: new_val = mean + N(0, noise_scale * std)
    3. For discrete params: sample from empirical distribution

    Args:
        checkpoint: Dict from load_checkpoint() with states_A, states_B
        model_type: Model identifier (e.g., 'mixture_3model_bhm')
        n_subjects: Number of subjects in the model
        K: Number of superchains (distinct starting points to generate)
        noise_scale: Scale factor for noise (default 1.0 to preserve full posterior spread)
        rng_seed: Random seed for reproducibility

    Returns:
        Array of shape (K, n_params) with new starting states
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    states_A = checkpoint['states_A']
    states_B = checkpoint['states_B']

    # Compute statistics
    stats = compute_chain_statistics(states_A, states_B)
    mean = stats['mean']
    std = stats['std']
    n_params = stats['n_params']

    # Get discrete parameter indices (only these need special handling)
    discrete_indices = set(get_discrete_param_indices(model_type, n_subjects))

    # Generate K new states
    new_states = np.zeros((K, n_params))

    for k in range(K):
        for i in range(n_params):
            if i in discrete_indices:
                # Discrete parameter: sample from empirical distribution
                all_vals = np.concatenate([states_A[:, i], states_B[:, i]])
                int_vals = all_vals.astype(int)
                unique, counts = np.unique(int_vals, return_counts=True)
                probs = counts / counts.sum()
                new_states[k, i] = np.random.choice(unique, p=probs)
            else:
                # Continuous parameter: mean + noise
                noise = np.random.normal(0, noise_scale * max(std[i], 0.01))
                new_states[k, i] = mean[i] + noise

    return new_states


def generate_reset_vector(checkpoint: Dict[str, Any], model_type: str, n_subjects: int, K: int, M: int, noise_scale: float = 1.0, rng_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate full initial vector with K superchains replicated M times.

    This is the function to use when resetting chains. It produces a flat
    vector suitable for passing to the MCMC backend.

    Args:
        checkpoint: Dict from load_checkpoint()
        model_type: Model identifier
        n_subjects: Number of subjects
        K: Number of superchains
        M: Number of subchains per superchain
        noise_scale: Scale for noise added to mean (default 1.0 for full posterior spread)
        rng_seed: Random seed

    Returns:
        Array of shape (K * M * n_params,) - flat initial vector
    """
    # Generate K distinct starting points
    base_states = generate_reset_states(
        checkpoint, model_type, n_subjects, K, noise_scale, rng_seed
    )

    # Replicate each K times for M subchains
    # The backend will later split this into A and B groups
    num_chains = K * M
    n_params = base_states.shape[1]

    # Replicate: each of K states appears M times consecutively
    all_states = np.repeat(base_states, M, axis=0)  # (K*M, n_params)

    # Flatten to match expected format
    return all_states.flatten()


def reset_from_checkpoint(checkpoint_path: str, model_type: str, n_subjects: int, K: int, M: int,
                          noise_scale: float = 1.0, rng_seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    High-level function to generate reset initial vector from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .npz file
        model_type: Model identifier
        n_subjects: Number of subjects
        K: Number of superchains
        M: Subchains per superchain
        noise_scale: Scale for noise
        rng_seed: Random seed

    Returns:
        tuple of (initial_vector, checkpoint_info)
    """
    from .checkpoint_helpers import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path)

    # Validate model type matches
    if checkpoint['posterior_id'] != model_type:
        print(f"Warning: Checkpoint model ({checkpoint['posterior_id']}) "
              f"differs from requested ({model_type})")

    init_vector = generate_reset_vector(
        checkpoint, model_type, n_subjects, K, M, noise_scale, rng_seed
    )

    info = {
        'source_iteration': checkpoint['iteration'],
        'source_chains': checkpoint['num_chains_a'] + checkpoint['num_chains_b'],
        'reset_K': K,
        'reset_M': M,
        'noise_scale': noise_scale,
    }

    return init_vector, info


def select_diverse_states(checkpoint: Dict[str, Any], K: int, rng_seed: Optional[int] = None) -> np.ndarray:
    """
    Select K diverse states from checkpoint chains.

    Unlike generate_reset_states (which uses mean + noise), this function
    selects actual chain states, ensuring diversity by spacing selections
    evenly across all chains.

    This is useful for initializing posterior chains from prior samples,
    where you want to preserve the actual prior draws rather than
    perturbing around the mean.

    Args:
        checkpoint: Dict from load_checkpoint() with states_A, states_B
        K: Number of diverse states to select
        rng_seed: Random seed for reproducibility (optional)

    Returns:
        Array of shape (K, n_params) with selected states
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    states_A = checkpoint['states_A']
    states_B = checkpoint['states_B']

    # Combine all chains
    all_states = np.vstack([states_A, states_B])
    n_chains = all_states.shape[0]
    n_params = all_states.shape[1]

    if K > n_chains:
        raise ValueError(f"Cannot select {K} states from {n_chains} chains")

    # Select K evenly-spaced chains (with random offset for variety)
    if K == n_chains:
        indices = np.arange(K)
    else:
        # Evenly spaced with random start
        step = n_chains / K
        offset = np.random.uniform(0, step) if rng_seed is None else 0
        indices = np.array([(offset + i * step) % n_chains for i in range(K)], dtype=int)
        indices = np.unique(indices)  # Remove any duplicates

        # If we lost some due to duplicates, fill with random
        while len(indices) < K:
            remaining = np.setdiff1d(np.arange(n_chains), indices)
            indices = np.append(indices, np.random.choice(remaining))

    selected_states = all_states[indices[:K], :]

    print(f"Selected {K} diverse states from {n_chains} chains")
    print(f"  Chain indices: {indices[:K]}")

    return selected_states


def init_from_prior(prior_checkpoint_path: str, K: int, M: int, rng_seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate initialization vector for posterior using prior samples.

    This is the recommended way to initialize posterior chains from a
    prior-only run. It selects K diverse states from the prior checkpoint
    and replicates each M times for the superchain structure.

    Workflow:
        1. Run prior-only model: python get_samples.py --model mix2_exp_weibull_prior_only
        2. Use this function to create init vector for posterior
        3. Pass to posterior sampling

    Args:
        prior_checkpoint_path: Path to prior-only checkpoint .npz file
        K: Number of superchains (distinct starting points)
        M: Subchains per superchain
        rng_seed: Random seed for reproducibility

    Returns:
        tuple of (initial_vector, info_dict)

    Example:
        init_vector, info = init_from_prior(
            'prior_checkpoint.npz',
            K=60,  # 60 superchains
            M=10,  # 10 subchains each = 600 total chains
        )
    """
    from .checkpoint_helpers import load_checkpoint

    checkpoint = load_checkpoint(prior_checkpoint_path)

    print(f"Initializing from prior checkpoint:")
    print(f"  Source: {prior_checkpoint_path}")
    print(f"  Iteration: {checkpoint['iteration']}")
    print(f"  Model: {checkpoint['posterior_id']}")

    # Select K diverse states from prior
    base_states = select_diverse_states(checkpoint, K, rng_seed)

    # Replicate each K state M times for subchains
    all_states = np.repeat(base_states, M, axis=0)  # (K*M, n_params)

    print(f"  Output: {K} superchains × {M} subchains = {K*M} total chains")

    info = {
        'source_checkpoint': prior_checkpoint_path,
        'source_iteration': checkpoint['iteration'],
        'source_model': checkpoint['posterior_id'],
        'source_chains': checkpoint['num_chains_a'] + checkpoint['num_chains_b'],
        'K': K,
        'M': M,
    }

    return all_states.flatten(), info


def print_reset_summary(checkpoint: Dict[str, Any], model_type: str, n_subjects: int) -> None:
    """
    Print summary statistics to help decide on reset parameters.

    Args:
        checkpoint: Dict from load_checkpoint()
        model_type: Model identifier
        n_subjects: Number of subjects
    """
    stats = compute_chain_statistics(checkpoint['states_A'], checkpoint['states_B'])
    discrete_indices = get_discrete_param_indices(model_type, n_subjects)

    print(f"\nReset Summary for {model_type}")
    print("=" * 60)
    print(f"Chains: {stats['n_chains']} total")
    print(f"Parameters: {stats['n_params']}")
    print(f"Source iteration: {checkpoint['iteration']}")

    # Summarize z distribution (discrete parameters)
    if discrete_indices:
        all_z = np.concatenate([
            checkpoint['states_A'][:, discrete_indices],
            checkpoint['states_B'][:, discrete_indices]
        ])
        z_flat = all_z.flatten().astype(int)
        unique, counts = np.unique(z_flat, return_counts=True)
        print(f"\nDiscrete parameter distribution (z indicators):")
        for z, c in zip(unique, counts):
            print(f"  Value {z}: {c/len(z_flat):.1%}")

    # Parameter spread summary
    print(f"\nParameter spread (std across chains):")
    print(f"  Min std:  {stats['std'].min():.4f}")
    print(f"  Max std:  {stats['std'].max():.4f}")
    print(f"  Mean std: {stats['std'].mean():.4f}")

    # Identify potentially stuck parameters (very low std)
    stuck_mask = stats['std'] < 0.001
    if stuck_mask.any():
        n_stuck = stuck_mask.sum()
        print(f"\n  WARNING: {n_stuck} parameters have std < 0.001 (possibly stuck)")
