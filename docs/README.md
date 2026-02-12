# bamcmc Documentation

**bamcmc** is a JAX-based MCMC sampling package designed for Bayesian hierarchical models with advanced features for complex posteriors.

## Quick Links

### API Reference
- [Sampler Types](./samplers.md) - METROPOLIS_HASTINGS, DIRECT_CONJUGATE, COUPLED_TRANSFORM
- [Proposal Types](./proposals.md) - 13 proposal distributions for different scenarios
- [BlockSpec Reference](./block_spec.md) - How to specify parameter blocks
- [Log-Posterior Functions](./log_posterior.md) - Requirements for posterior functions
- [Registering Models](./registration.md) - How to register posteriors with the system
- [Settings Reference](./settings.md) - Available proposal settings
- [Checkpoint Helpers](./checkpoint_helpers.md) - Checkpoints, output management, and post-processing

### Guides
- [Configuration](./configuration.md) - Data format, MCMC config keys, parallel tempering
- [Parallel Tempering](./tempering.md) - Index process PT, DEO scheme, temperature tuning
- [Architecture](./architecture.md) - Coupled chains, data structures, module responsibilities
- [Development](./development.md) - Adding proposals, defining posteriors, minimal template, testing
- [Troubleshooting](./troubleshooting.md) - Common errors, debugging, performance, JAX patterns

## Package Overview

bamcmc provides:

- **Coupled A/B chains**: Two groups of chains share proposal statistics for improved mixing
- **Nested R-hat diagnostics**: Hierarchical convergence checking (Margossian et al., 2022)
- **Block-based sampling**: Parameters organized into blocks with configurable samplers
- **Cross-session compilation caching**: JAX compilations persist across Python sessions
- **Parallel tempering**: Multi-temperature sampling for multimodal posteriors

## Installation

The package is designed to be installed in editable mode:

```bash
pip install -e /path/to/bamcmc
```

## Basic Usage

```python
from bamcmc import (
    register_posterior, rmcmc, rmcmc_single,
    BlockSpec, SamplerType, ProposalType
)

# 1. Define model functions
def log_posterior(chain_state, param_indices, data):
    """Log posterior for a parameter block."""
    params = chain_state[param_indices]
    # ... compute log p(params | data)
    return log_prob

def batch_specs(mcmc_config, data):
    """Define parameter blocks."""
    n_subjects = data['static'][0]
    return [
        BlockSpec(
            size=2,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=ProposalType.CHAIN_MEAN,
            label=f"Subject_{i}"
        )
        for i in range(n_subjects)
    ]

def initial_vector(mcmc_config, data):
    """Initial parameter values."""
    K = mcmc_config['num_superchains']
    return np.random.randn(K, n_params).flatten()

# 2. Register the model
register_posterior('my_model', {
    'log_posterior': log_posterior,
    'batch_type': batch_specs,
    'initial_vector': initial_vector,
})

# 3. Configure MCMC
mcmc_config = {
    'posterior_id': 'my_model',
    'use_double': True,
    'rng_seed': 1977,
    'num_chains_a': 500,
    'num_chains_b': 500,
    'num_superchains': 100,
    'benchmark': 100,
    'burn_iter': 5000,
    'num_collect': 10000,
    'thin_iteration': 10,
}

# 4. Prepare data
data = {
    "static": (n_subjects, hyperprior_a),
    "int": (int_array,),
    "float": (float_array,),
}

# 5. Run MCMC
results, checkpoint = rmcmc_single(mcmc_config, data)
```

## Architecture

### Coupled A/B Chains

The sampler runs two groups of chains (A and B) that share proposal statistics:

1. **Group A chains** propose using statistics computed from Group B
2. **Group B chains** propose using statistics computed from Group A

This cross-coupling improves mixing while maintaining detailed balance.

### Block Structure

Parameters are organized into **blocks** - groups of parameters updated together in one MH step. Each block has:

- **Size**: Number of parameters in the block
- **Sampler type**: How to sample (MH, direct, coupled transform)
- **Proposal type**: For MH samplers, how to construct the proposal distribution
- **Settings**: Tunable parameters for the proposal (step size, mixing weights, etc.)

### Nested R-hat

For hierarchical models, chains are organized into:

- **K superchains**: Independent starting points
- **M subchains per superchain**: Perturbed copies of each start

Nested R-hat checks convergence both within and between superchains, catching issues that standard R-hat might miss.

## Data Format

Data is passed as a dict with three tuple fields:

```python
data = {
    "static": (scalar1, scalar2, ...),  # Hashable scalars
    "int": (int_array1, int_array2),     # Integer arrays
    "float": (float_array1, float_array2), # Float arrays
}
```

The `static` tuple should contain only hashable values (int, float, tuple) since it's used in JAX's static argument hashing.

## Key Functions

### rmcmc_single()

Run a single MCMC sampling session:

```python
results, checkpoint = rmcmc_single(
    mcmc_config,
    data,
    calculate_rhat=True,
    resume_from=None,      # Path to checkpoint to resume
)

# Results contain:
# - history: (num_collect, num_chains, num_params + num_gq)
# - diagnostics: {'rhat': array, 'compile_time': float, ...}
# - K: Number of superchains
# - M: Subchains per superchain
```

### rmcmc()

Run multiple MCMC sessions with automatic checkpointing:

```python
summary = rmcmc(
    mcmc_config,
    data,
    output_dir='./output',
    run_schedule=[("resume", 5)],  # 5 resume runs
    calculate_rhat=True,
)
```

## Configuration Reference

| Key | Type | Description |
|-----|------|-------------|
| `posterior_id` | str | Registered posterior name |
| `use_double` | bool | Use float64 precision |
| `rng_seed` | int | Random seed |
| `num_chains_a` | int | Chains in group A |
| `num_chains_b` | int | Chains in group B |
| `num_superchains` | int | K superchains for nested R-hat |
| `benchmark` | int | Benchmark iterations |
| `burn_iter` | int | Burn-in iterations |
| `num_collect` | int | Collection iterations |
| `thin_iteration` | int | Thinning interval |
| `save_likelihoods` | bool | Save log-likelihood history |

### Parallel Tempering Configuration

| Key | Type | Description |
|-----|------|-------------|
| `n_temperatures` | int | Number of temperature levels (default: 1) |
| `beta_min` | float | Minimum inverse temperature (default: 0.1) |
| `swap_every` | int | Iterations between swap attempts (default: 1) |

## JAX Compilation Caching

bamcmc caches JAX compilations to disk for faster startup on subsequent runs. The cache is stored in `./jax_cache/` by default.

To clear the cache (force recompilation):

```bash
rm -rf ./jax_cache/
```

## See Also

- [Detailed Sampler Documentation](./samplers.md)
- [Proposal Type Reference](./proposals.md)
- [BlockSpec Specification](./block_spec.md)
- [Log-Posterior Requirements](./log_posterior.md)
