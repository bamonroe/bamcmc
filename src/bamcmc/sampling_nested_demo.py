"""
Example sampling script demonstrating nested R-hat usage.

This shows how to enable and use nested R-hat for many-short-chains regime.
"""

import pandas as pd
import numpy as np
from .mcmc_backend import rmcmc
from .mcmc_utils import clean_config
from .error_handling import validate_mcmc_config, diagnose_sampler_issues, print_diagnostics

# Note: clean_data is in root, not in package - import from there if needed
# from clean_data import prepare_ahlr

seed = 1977

# Example 1: Standard configuration (no nested R-hat)
standard_config = {
    'POSTERIOR_ID': 'eut_crra_bhm',
    'GPU_PREALLOCATION': True,
    'USE_DOUBLE': True,
    'rng_seed': seed,
    'BENCHMARK': 100,
    'BURN_ITER': 1000,
    'SAVE_LIKELIHOODS': True,
    'THIN_ITERATION': 1,
    'NUM_COLLECT': 100,  # 100 samples per chain
    'NUM_CHAINS_A': 50,
    'NUM_CHAINS_B': 50,
    'LAST_ITERS': 100,
}

# Example 2: Nested R-hat configuration (many short chains)
nested_config = {
    'POSTERIOR_ID': 'eut_crra_bhm',
    'GPU_PREALLOCATION': True,
    'USE_DOUBLE': True,
    'rng_seed': seed,
    'BENCHMARK': 100,
    'BURN_ITER': 1000,
    'SAVE_LIKELIHOODS': True,
    'THIN_ITERATION': 1,
    'NUM_COLLECT': 20,  # Only 20 samples per chain!
    
    # Nested R-hat structure
    'USE_NESTED_RHAT': True,
    'NUM_SUPERCHAINS': 8,  # K = 8 distinct initializations
    
    # Total chains must be divisible by NUM_SUPERCHAINS
    'NUM_CHAINS_A': 400,  # 8 × 50 per group
    'NUM_CHAINS_B': 400,
    
    'LAST_ITERS': 20,
}

# Example 3: Production-scale nested R-hat
production_config = {
    'POSTERIOR_ID': 'eut_crra_bhm',
    'GPU_PREALLOCATION': True,
    'USE_DOUBLE': True,
    'rng_seed': seed,
    'BENCHMARK': 100,
    'BURN_ITER': 1000,
    'SAVE_LIKELIHOODS': True,
    'THIN_ITERATION': 1,
    'NUM_COLLECT': 20,  # Short chains for efficiency
    
    # Large-scale nested structure
    'USE_NESTED_RHAT': True,
    'NUM_SUPERCHAINS': 20,  # K = 20 for robust detection
    
    # 8000 total chains (20 × 400)
    'NUM_CHAINS_A': 4000,  # 200 per superchain
    'NUM_CHAINS_B': 4000,
    
    'LAST_ITERS': 20,
}


def run_with_config(config, description):
    """Run MCMC with given configuration and print summary."""
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    
    # Validate
    print("Validating configuration...")
    try:
        validate_mcmc_config(config)
        print("✅ Configuration valid\n")
    except ValueError as e:
        print(f"❌ Invalid configuration:\n{e}")
        return
    
    # Prepare data
    data = prepare_ahlr("ahlr.csv")
    config['N_SUBJECTS'] = data['static'][2]
    
    # Run MCMC
    print(f"Starting sampling for {config['POSTERIOR_ID']}...")
    history, diagnostics_dict, config, lik_history = rmcmc(config, data)
    print("Sampling complete.")
    
    # Display diagnostics summary
    print("\n--- Diagnostics Summary ---")
    
    num_chains = config['NUM_CHAINS_A'] + config['NUM_CHAINS_B']
    num_samples = history.shape[0]
    num_params = history.shape[2]
    
    print(f"Chains: {num_chains}")
    print(f"Samples per chain: {num_samples}")
    print(f"Parameters: {num_params}")
    print(f"Total samples: {num_chains * num_samples}")
    
    # Standard R-hat
    if diagnostics_dict['rhat'] is not None:
        rhat = diagnostics_dict['rhat']
        print(f"\nStandard R̂:")
        print(f"  Max: {np.max(rhat):.4f}")
        print(f"  Median: {np.median(rhat):.4f}")
        
        if np.max(rhat) < 1.01:
            print("  ✓ Converged (R̂ < 1.01)")
        else:
            print(f"  ⚠ May not have converged (max R̂ = {np.max(rhat):.4f})")
    
    # Nested R-hat
    if diagnostics_dict['nested_rhat'] is not None:
        nrhat = diagnostics_dict['nested_rhat']
        K = config['NUM_SUPERCHAINS']
        M = config['SUBCHAINS_PER_SUPER']
        
        # Threshold
        tau = 1e-4
        threshold = np.sqrt(1 + 1/M + tau)
        
        print(f"\nNested R̂ ({K} superchains × {M} subchains):")
        print(f"  Max: {np.max(nrhat):.4f}")
        print(f"  Median: {np.median(nrhat):.4f}")
        print(f"  Threshold: {threshold:.4f}")
        
        if np.max(nrhat) < threshold:
            print(f"  ✓ Converged (nR̂ < {threshold:.4f})")
        else:
            print(f"  ⚠ May not have converged (max nR̂ = {np.max(nrhat):.4f})")
    
    # Post-run diagnostics
    print("\n--- Additional Diagnostics ---")
    diagnostics = diagnose_sampler_issues(history, config)
    print_diagnostics(diagnostics)
    
    # Save results
    filename = f"{config['POSTERIOR_ID']}_nested.npz" if config.get('USE_NESTED_RHAT') else f"{config['POSTERIOR_ID']}.npz"
    print(f"\nSaving to {filename}...")
    
    save_dict = {
        'history': history,
        'rhat': diagnostics_dict.get('rhat'),
        'mcmc_config': config,
        'likelihoods': lik_history
    }
    
    if diagnostics_dict['nested_rhat'] is not None:
        save_dict['nested_rhat'] = diagnostics_dict['nested_rhat']
    
    np.savez_compressed(filename, **save_dict)
    print("✓ Save complete")
    
    return history, diagnostics_dict


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("NESTED R-HAT DEMONSTRATION")
    print("=" * 70)
    
    # Choose which configuration to run
    print("\nAvailable configurations:")
    print("1. Standard (100 chains × 100 samples)")
    print("2. Nested R-hat - Small (800 chains × 20 samples)")
    print("3. Nested R-hat - Production (8000 chains × 20 samples)")
    
    choice = input("\nSelect configuration (1-3) [default: 2]: ").strip() or "2"
    
    if choice == "1":
        run_with_config(standard_config, "Standard Configuration")
    elif choice == "2":
        run_with_config(nested_config, "Nested R-hat - Small Scale")
    elif choice == "3":
        confirm = input("Production config uses 8000 chains. Continue? (y/n) [n]: ").strip().lower()
        if confirm == 'y':
            run_with_config(production_config, "Nested R-hat - Production Scale")
        else:
            print("Cancelled.")
    else:
        print("Invalid choice. Running default (nested small scale)...")
        run_with_config(nested_config, "Nested R-hat - Small Scale")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
