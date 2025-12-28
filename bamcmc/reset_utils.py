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
        noise_scale=0.1,
        rng_seed=42
    )
"""

import numpy as np
from pathlib import Path


def compute_chain_statistics(states_A, states_B):
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


def get_special_param_indices(model_type, n_subjects):
    """
    Get indices of special parameters that need non-standard handling.

    Args:
        model_type: Model identifier string
        n_subjects: Number of subjects in the model

    Returns:
        dict with:
            'z_indices': list of z indicator indices (discrete)
            'pi_indices': list of pi mixing weight indices (simplex)
            'r_indices': list of r parameter indices (natural scale, not log)
    """
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


def generate_reset_states(checkpoint, model_type, n_subjects, K, noise_scale=0.1, rng_seed=None):
    """
    Generate K new starting states based on cross-chain mean.

    This creates K distinct starting points (one per superchain) by:
    1. Computing cross-chain mean for all parameters
    2. Adding scaled noise: new_val = mean + N(0, noise_scale * std)
    3. Special handling for discrete (z) and simplex (pi) parameters

    Args:
        checkpoint: Dict from load_checkpoint() with states_A, states_B
        model_type: Model identifier (e.g., 'mixture_3model_bhm')
        n_subjects: Number of subjects in the model
        K: Number of superchains (distinct starting points to generate)
        noise_scale: Scale factor for noise (default 0.1)
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

    # Get special parameter indices
    special = get_special_param_indices(model_type, n_subjects)
    z_indices = set(special['z_indices'])
    pi_indices = set(special['pi_indices'])
    r_indices = set(special['r_indices'])

    # Generate K new states
    new_states = np.zeros((K, n_params))

    for k in range(K):
        for i in range(n_params):
            if i in z_indices:
                # Discrete z: sample from empirical distribution
                all_z = np.concatenate([states_A[:, i], states_B[:, i]])
                z_vals = all_z.astype(int)
                # Use empirical probabilities
                unique, counts = np.unique(z_vals, return_counts=True)
                probs = counts / counts.sum()
                new_states[k, i] = np.random.choice(unique, p=probs)

            elif i in pi_indices:
                # Simplex pi: use cross-chain mean (population-level, no noise)
                new_states[k, i] = mean[i]

            elif i in r_indices:
                # r parameters: natural scale, add noise in natural space
                noise = np.random.normal(0, noise_scale * max(std[i], 0.01))
                new_states[k, i] = mean[i] + noise

            else:
                # Standard unconstrained parameter: add noise
                noise = np.random.normal(0, noise_scale * max(std[i], 0.01))
                new_states[k, i] = mean[i] + noise

    # Normalize pi values to sum to 1 (for each superchain)
    if pi_indices:
        pi_list = sorted(pi_indices)
        for k in range(K):
            pi_vals = new_states[k, pi_list]
            pi_vals = np.clip(pi_vals, 0.01, 0.99)  # Ensure valid
            new_states[k, pi_list] = pi_vals / pi_vals.sum()

    return new_states


def generate_reset_vector(checkpoint, model_type, n_subjects, K, M, noise_scale=0.1, rng_seed=None):
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
        noise_scale: Scale for noise added to mean
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


def reset_from_checkpoint(checkpoint_path, model_type, n_subjects, K, M,
                          noise_scale=0.1, rng_seed=None):
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


def print_reset_summary(checkpoint, model_type, n_subjects):
    """
    Print summary statistics to help decide on reset parameters.

    Args:
        checkpoint: Dict from load_checkpoint()
        model_type: Model identifier
        n_subjects: Number of subjects
    """
    stats = compute_chain_statistics(checkpoint['states_A'], checkpoint['states_B'])
    special = get_special_param_indices(model_type, n_subjects)

    print(f"\nReset Summary for {model_type}")
    print("=" * 60)
    print(f"Chains: {stats['n_chains']} total")
    print(f"Parameters: {stats['n_params']}")
    print(f"Source iteration: {checkpoint['iteration']}")

    # Summarize z distribution
    if special['z_indices']:
        all_z = np.concatenate([
            checkpoint['states_A'][:, special['z_indices']],
            checkpoint['states_B'][:, special['z_indices']]
        ])
        z_flat = all_z.flatten().astype(int)
        unique, counts = np.unique(z_flat, return_counts=True)
        print(f"\nZ distribution (model indicators):")
        for z, c in zip(unique, counts):
            print(f"  Model {z}: {c/len(z_flat):.1%}")

    # Summarize pi
    if special['pi_indices']:
        pi_vals = np.concatenate([
            checkpoint['states_A'][:, special['pi_indices']],
            checkpoint['states_B'][:, special['pi_indices']]
        ])
        pi_mean = np.mean(pi_vals, axis=0)
        print(f"\nPi (mixing weights) mean: {pi_mean}")

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
