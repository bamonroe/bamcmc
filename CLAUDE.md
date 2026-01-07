# bamcmc Package Guide

This document provides a comprehensive overview of the bamcmc (Bayesian Adaptive MCMC) package for Claude or other developers working with this codebase.

## Package Overview

bamcmc is a JAX-based MCMC sampling package designed for Bayesian hierarchical models. Key features:

- **Coupled A/B chains**: Runs two groups of chains (A and B) that share proposal statistics
- **Nested R-hat diagnostics**: Implements Margossian et al. (2022) for hierarchical convergence checking
- **Block-based sampling**: Parameters are organized into blocks with configurable samplers
- **Cross-session compilation caching**: JAX compilations persist across Python sessions
- **Posterior benchmarking**: Performance tracking with hash-based caching

## Environment Setup

The project uses a Python virtual environment at `/workspace/pylib/`:

```bash
# Activate the environment
source /workspace/pylib/bin/activate

# Or run directly with the environment's Python
/workspace/pylib/bin/python your_script.py
```

The environment includes:
- JAX 0.8.2 with CUDA 13 support
- numpy, scipy, matplotlib
- bamcmc installed in editable mode

**Running tests:**
```bash
source /workspace/pylib/bin/activate
cd /workspace/bamcmc
pytest tests/ -v
```

## Directory Structure

```
bamcmc/
├── src/bamcmc/
│   ├── __init__.py           # Public API exports
│   ├── mcmc/                  # Core MCMC implementation
│   │   ├── __init__.py       # Subpackage exports
│   │   ├── backend.py        # Main entry points (rmcmc, rmcmc_single)
│   │   ├── types.py          # Core dataclasses (BlockArrays, RunParams)
│   │   ├── config.py         # Configuration and initialization
│   │   ├── sampling.py       # Proposal and MH sampling logic
│   │   ├── scan.py           # JAX scan body and block statistics
│   │   ├── compile.py        # Kernel compilation and caching
│   │   ├── diagnostics.py    # R-hat computation and acceptance rates
│   │   └── utils.py          # Misc utilities
│   ├── proposals/            # Proposal distribution implementations
│   │   ├── __init__.py
│   │   ├── self_mean.py      # Random walk proposal
│   │   ├── chain_mean.py     # Independent proposal
│   │   ├── mixture.py        # Mixture of self/chain mean
│   │   ├── multinomial.py    # Discrete parameter proposal
│   │   ├── mala.py           # MALA (gradient-based) proposal
    │   │   ├── mean_mala.py      # MEAN_MALA: gradient at coupled mean
    │   │   ├── mean_weighted.py  # MEAN_WEIGHTED: adaptive interpolation
    │   │   ├── mode_weighted.py  # MODE_WEIGHTED: interpolation toward mode
    │   │   ├── mcov_weighted.py      # MCOV_WEIGHTED: covariance-scaled interpolation
    │   │   └── mcov_weighted_vec.py  # MCOV_WEIGHTED_VEC: vectorized per-param MCOV
│   ├── batch_specs.py        # BlockSpec, SamplerType, ProposalType
│   ├── registry.py           # Posterior registration system
│   ├── settings.py           # Per-block settings (SettingSlot)
│   ├── checkpoint_helpers.py # Save/load checkpoints, combine batches
│   ├── reset_utils.py        # Reset stuck chains from checkpoint
│   ├── posterior_benchmark.py# Benchmark caching with posterior hashing
│   ├── error_handling.py     # Validation and diagnostics
│   ├── jax_config.py         # JAX environment configuration
│   └── test_posteriors.py    # Conjugate test models with analytical solutions
├── tests/
│   ├── conftest.py           # Pytest fixtures
│   ├── test_unit.py          # Unit tests (38 tests)
│   ├── test_nested_rhat.py   # Nested R-hat tests
│   └── test_integration.py   # Full MCMC validation tests
└── pyproject.toml
```

## Core Concepts

### 1. Posterior Registration

Models are registered with the system before sampling:

```python
from bamcmc import register_posterior, BlockSpec, SamplerType, ProposalType

register_posterior('my_model', {
    # Required:
    'log_posterior': fn(chain_state, param_indices, data) -> scalar,
    'batch_type': fn(mcmc_config, data) -> List[BlockSpec],
    'initial_vector': fn(mcmc_config, data) -> array,

    # Optional:
    'direct_sampler': fn(key, chain_state, param_indices, data) -> (state, key),
    'generated_quantities': fn(chain_state, data) -> array,
    'get_num_gq': fn(mcmc_config, data) -> int,
})
```

### 2. Block Specifications

Parameters are organized into blocks, each with its own sampling strategy:

```python
BlockSpec(
    size=3,                                    # Number of parameters
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.CHAIN_MEAN,     # For MH samplers
    settings={'alpha': 0.5},                   # Per-block settings
    label="Subject_0_params"                   # For debugging
)
```

**SamplerType options:**
- `METROPOLIS_HASTINGS` (0): Standard MH with proposal distribution
- `DIRECT_CONJUGATE` (1): Direct/Gibbs sampling

**ProposalType options:**
- `SELF_MEAN` (0): Random walk centered on current state
- `CHAIN_MEAN` (1): Independent proposal centered on population mean
- `MIXTURE` (2): With probability alpha use CHAIN_MEAN, else SELF_MEAN
- `MULTINOMIAL` (3): For discrete parameters on integer grid
- `MALA` (4): Metropolis-adjusted Langevin (gradient-based, preconditioned)
- `MEAN_MALA` (5): MALA with gradient computed at coupled mean (independent proposal)
- `MEAN_WEIGHTED` (6): Adaptive interpolation between self and chain mean based on Mahalanobis distance
- `MODE_WEIGHTED` (7): Interpolation toward the mode (highest log-posterior chain)
- `MCOV_WEIGHTED` (8): Mean-covariance weighted interpolation with configurable blend
- `MCOV_WEIGHTED_VEC` (9): Vectorized MCOV with per-parameter distance and interpolation

### 3. Data Format

Data is passed as a dict with three tuple fields:

```python
data = {
    "static": (n_subjects, hyperprior_a, hyperprior_b),  # Scalars (hashable)
    "int": (int_array1, int_array2),                      # Integer arrays
    "float": (float_array1, float_array2),                # Float arrays
}
```

### 4. MCMC Configuration

All config keys use lowercase with underscores:

```python
mcmc_config = {
    'posterior_id': 'my_model',      # Registered posterior name
    'use_double': True,              # float64 precision
    'rng_seed': 1977,                # Random seed

    # Chain configuration
    'num_chains_a': 500,             # Chains in group A
    'num_chains_b': 500,             # Chains in group B
    'num_superchains': 100,          # K superchains (for nested R-hat)
    # M = num_chains / num_superchains subchains per superchain

    # Iteration settings
    'benchmark': 100,                # Benchmark iterations (for timing)
    'burn_iter': 5000,               # Burn-in iterations
    'num_collect': 10000,            # Collection iterations
    'thin_iteration': 10,            # Thinning interval

    # Optional
    'save_likelihoods': False,       # Save log-likelihood history
    'proposal': 'chain_mean',        # Default proposal type
}
```

### 5. Coupled A/B Chain Architecture

The sampler runs two groups of chains that share proposal statistics:

1. **Group A chains** propose using statistics computed from Group B
2. **Group B chains** propose using statistics computed from Group A
3. This coupling improves mixing while maintaining detailed balance

The `step_mean` and `step_cov` for proposals are computed from the "coupled" group.

### 6. Nested R-hat (Margossian et al., 2022)

For hierarchical models, standard R-hat can miss non-convergence. Nested R-hat:

- Organizes chains into K **superchains** with M **subchains** each
- Checks convergence both within and between superchains
- More sensitive to stuck chains in hierarchical posteriors

```
Total chains = K × M
K = NUM_SUPERCHAINS (independent starting points)
M = NUM_CHAINS / NUM_SUPERCHAINS (perturbed copies of each start)
```

## Key Data Structures

### BlockArrays (frozen dataclass, JAX pytree)

Pre-parsed block specifications for efficient JAX operations:

```python
@dataclass(frozen=True)
class BlockArrays:
    indices: jnp.ndarray         # (n_blocks, max_size) parameter indices
    types: jnp.ndarray           # (n_blocks,) SamplerType per block
    masks: jnp.ndarray           # (n_blocks, max_size) valid param mask
    proposal_types: jnp.ndarray  # (n_blocks,) remapped indices into compact dispatch
    settings_matrix: jnp.ndarray # (n_blocks, MAX_SETTINGS)
    max_size: int
    num_blocks: int
    total_params: int
    used_proposal_types: tuple   # Original ProposalType values used (dispatch order)
```

### RunParams (frozen dataclass)

Immutable run parameters for JAX static arguments:

```python
@dataclass(frozen=True)
class RunParams:
    BURN_ITER: int
    NUM_COLLECT: int
    THIN_ITERATION: int
    NUM_GQ: int
    START_ITERATION: int
    SAVE_LIKELIHOODS: bool
```

## Main Entry Points

### rmcmc_single() - Single-Run Sampling

```python
from bamcmc import rmcmc_single  # Or: from bamcmc.mcmc.backend import rmcmc_single

results, checkpoint = rmcmc_single(
    mcmc_config,
    data,
    calculate_rhat=True,
    resume_from=None,      # Path to checkpoint to resume
    reset_from=None,       # Path to checkpoint for reset (new noise)
    reset_noise_scale=0.1,
)

# Returns:
# results dict containing:
#   - history: (num_collect, num_chains, num_params + num_gq)
#   - iterations: Iteration number for each sample
#   - diagnostics: {'rhat': array, 'K': int, 'M': int, 'compile_time': float, ...}
#   - mcmc_config: Clean serializable config dict (no JAX types)
#   - likelihoods: Likelihood history if SAVE_LIKELIHOODS=True, else None
#   - K: Number of superchains
#   - M: Subchains per superchain
#   - thin_iteration: Thinning interval used
#
# checkpoint dict for saving/resuming
```

### rmcmc() - Multi-Run Sampling

```python
from bamcmc import rmcmc  # Or: from bamcmc.mcmc.backend import rmcmc

summary = rmcmc(
    mcmc_config,
    data,
    output_dir='./output',
    run_schedule=[("reset", 3), ("resume", 5)],  # 3 reset runs, then 5 resume runs
    calculate_rhat=True,
    burn_in_fresh=True,    # Only burn-in on fresh/reset runs
    reset_noise_scale=0.1,
)

# summary dict containing:
#   - history_files: List of (run_idx, filepath) tuples
#   - checkpoint_files: List of checkpoint filepaths
#   - run_log: List of dicts with per-run details
#   - final_iteration: Iteration after all runs
#   - total_runs_completed: Number of runs completed
```

### run_benchmark() - Performance Testing

```python
from bamcmc.posterior_benchmark import run_benchmark

results = run_benchmark(
    mcmc_config,
    data,
    benchmark_iterations=100,
    compare=True,          # Compare against cached baseline
    update_cache=False,    # Save as new baseline
)
```

## Module Responsibilities

### mcmc/ subpackage

#### mcmc/backend.py
- `rmcmc()`: Multi-run sampling with checkpoint management
- `rmcmc_single()`: Single MCMC run
- Orchestrates configuration, compilation, and execution

#### mcmc/types.py
- `BlockArrays`: JAX-compatible block specification container
- `RunParams`: Immutable run parameters
- `build_block_arrays()`: Factory function

#### mcmc/config.py
- `configure_mcmc_system()`: Setup model context from config, returns `(user_config, runtime_ctx, model_context)`
  - `user_config`: Serializable config (no JAX types) that can be saved to disk
  - `runtime_ctx`: JAX-dependent objects (dtypes, data arrays, RNG keys)
  - `model_context`: Model functions and block specifications
- `initialize_mcmc_system()`: Create initial chain states
- `validate_mcmc_inputs()`: Validate data shapes
- `gen_rng_keys()`: Generate JAX RNG keys

#### mcmc/sampling.py
- `create_mh_proposal()`: Build proposal for a block
- `metropolis_hastings_step()`: Single MH accept/reject step
- Block update logic using proposal dispatch

#### mcmc/scan.py
- `mcmc_scan_body()`: Inner loop body for jax.lax.scan
- `compute_block_stats()`: Calculate proposal statistics
- Handles collection, thinning, and acceptance counting

#### mcmc/compile.py
- `compile_mcmc_kernel()`: AOT compilation with caching
- `benchmark_mcmc_sampler()`: Time the compiled kernel
- `CHUNK_SIZE`: Iteration chunk size for compilation

#### mcmc/diagnostics.py
- `compute_nested_rhat()`: Nested R-hat calculation (JAX/GPU)
- `compute_and_print_rhat()`: Compute and print R-hat summary
- `print_acceptance_summary()`: Print MH acceptance rates

### proposals/
Each proposal implements:
```python
def proposal_fn(operand) -> (proposal, log_hastings_ratio, new_key):
    key, current_block, step_mean, step_cov, coupled_blocks, mask, settings = operand
    ...
```

### posterior_benchmark.py
- `PosteriorBenchmarkManager`: Caches benchmark results by posterior hash
- `get_posterior_hash()`: Hash of posterior code + data for cache keys
- `compare_benchmark()`: Compare new vs cached performance

### checkpoint_helpers.py
- `save_checkpoint()` / `load_checkpoint()`: Persist chain states
- `combine_batch_histories()`: Merge multiple batch files
- `apply_burnin()`: Remove samples before iteration threshold

### reset_utils.py
- `generate_reset_vector()`: Create new starting points from checkpoint
- Uses cross-chain mean + small noise to rescue stuck chains

## JAX Compilation Caching

The package uses cross-session JAX compilation caching:

1. **Environment variable**: `JAX_COMPILATION_CACHE_DIR` set in `jax_config.py`
2. **Module-level functions**: Sampling functions defined at module level (not closures)
3. **Frozen dataclasses**: `BlockArrays` and `RunParams` are hashable static args
4. **AOT compilation**: Uses `.lower().compile()` for explicit compilation

Cache is stored in `./jax_cache/` by default. Clear it to force recompilation.

## Testing

Run tests with pytest:

```bash
# All tests
pytest tests/ -v

# Unit tests only (fast, no GPU needed)
pytest tests/test_unit.py -v

# Integration tests (runs actual MCMC)
pytest tests/test_integration.py -v
```

Test posteriors in `test_posteriors.py` have analytical solutions for validation:
- `test_beta_bernoulli_pooled`: Beta-Bernoulli conjugate model
- `test_normal_normal_pooled`: Normal-Normal with known variance
- `test_beta_bernoulli_hierarchical`: Hierarchical Beta-Bernoulli

## Common Patterns

### Adding a New Proposal Type

1. Add enum to `batch_specs.py`:
   ```python
   class ProposalType(IntEnum):
       MY_PROPOSAL = 5  # Next available index
   ```

2. Create `proposals/my_proposal.py`:
   ```python
   def my_proposal(operand):
       key, current, mean, cov, coupled, mask, settings, grad_fn = operand
       # ... compute proposal and log ratio
       # grad_fn available for gradient-based proposals (ignore if not needed)
       return proposal, log_ratio, new_key
   ```

3. Add to `PROPOSAL_REGISTRY` in `mcmc/sampling.py`:
   ```python
   PROPOSAL_REGISTRY = {
       int(ProposalType.SELF_MEAN): self_mean_proposal,
       # ... existing entries ...
       int(ProposalType.MY_PROPOSAL): my_proposal,
   }
   ```

4. Export from `proposals/__init__.py`

Note: The dispatch table is built dynamically based on which proposals are
actually used by the model. This ensures JAX only traces needed proposals,
avoiding memory overhead from unused ones.

### Defining a New Posterior

```python
def my_log_posterior(chain_state, param_indices, data):
    """Log posterior for a parameter block."""
    params = chain_state[param_indices]
    # ... compute log posterior
    return log_prob

def my_batch_specs(mcmc_config, data):
    """Define parameter blocks."""
    n_subjects = data['static'][0]
    return [
        BlockSpec(size=2, sampler_type=SamplerType.METROPOLIS_HASTINGS,
                  proposal_type=ProposalType.CHAIN_MEAN, label=f"Subj_{i}")
        for i in range(n_subjects)
    ] + [
        BlockSpec(size=1, sampler_type=SamplerType.DIRECT_CONJUGATE,
                  direct_sampler_fn=my_hyper_sampler, label="Hyperprior")
    ]

def my_initial_vector(mcmc_config, data):
    """Initial values for all chains."""
    K = mcmc_config['NUM_SUPERCHAINS']
    n_params = ...
    return np.random.randn(K, n_params).flatten()

register_posterior('my_model', {
    'log_posterior': my_log_posterior,
    'batch_type': my_batch_specs,
    'initial_vector': my_initial_vector,
})
```

### Resuming from Checkpoint

```python
# Resume exact state
results, checkpoint = rmcmc(config, data, resume_from='checkpoint.npz')
history = results['history']
diagnostics = results['diagnostics']

# Reset with noise (rescue stuck chains)
results, checkpoint = rmcmc(config, data, reset_from='checkpoint.npz',
                            reset_noise_scale=0.1)
```

## Performance Considerations

1. **Block size**: Larger blocks = fewer kernel calls but coarser updates
2. **Chunk size**: `CHUNK_SIZE` in mcmc_compile.py controls iteration batching
3. **Proposal type**: CHAIN_MEAN often mixes faster but needs good initialization
4. **Float precision**: USE_DOUBLE=True is slower but more stable
5. **Compilation**: First run compiles (~3-17s); cached runs are fast (~3s)

## Debugging Tips

1. **Check R-hat**: Values > 1.1 indicate non-convergence
2. **Acceptance rates**: Stored in diagnostics, target 20-40%
3. **Trace plots**: Plot `history[:, chain_idx, param_idx]` over iterations
4. **Block labels**: Use `label` in BlockSpec for clearer error messages
5. **Validation errors**: `validate_mcmc_config()` catches common issues

## Dependencies

- **jax** / **jaxlib**: GPU-accelerated array operations
- **numpy**: Array utilities
- **scipy** (optional): For integration tests only
