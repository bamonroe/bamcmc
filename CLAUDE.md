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
│   ├── proposals/            # 14 proposal distribution implementations
│   │   ├── __init__.py
│   │   ├── common.py         # Shared helpers (COV_NUGGET, compute_mahalanobis, etc.)
│   │   ├── self_mean.py      # SELF_MEAN: random walk proposal
│   │   ├── chain_mean.py     # CHAIN_MEAN: independent proposal
│   │   ├── mixture.py        # MIXTURE: mix of self/chain mean
│   │   ├── multinomial.py    # MULTINOMIAL: discrete parameter proposal
│   │   ├── mala.py           # MALA: gradient-based proposal
│   │   ├── mean_mala.py      # MEAN_MALA: gradient at coupled mean
│   │   ├── mean_weighted.py  # MEAN_WEIGHTED: adaptive interpolation
│   │   ├── mode_weighted.py  # MODE_WEIGHTED: interpolation toward mode
│   │   ├── mcov_weighted.py  # MCOV_WEIGHTED: covariance-scaled interpolation
│   │   ├── mcov_weighted_vec.py  # MCOV_WEIGHTED_VEC: vectorized per-param MCOV
│   │   ├── mcov_smooth.py    # MCOV_SMOOTH: smooth distance-based scaling
│   │   ├── mcov_mode.py      # MCOV_MODE: mode-targeting with scalar distance
│   │   └── mcov_mode_vec.py  # MCOV_MODE_VEC: mode-targeting with per-param distance
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
    'initial_vector': fn(mcmc_config, data) -> flat_array,
    'direct_sampler': fn(key, chain_state, param_indices, data) -> (state, key),

    # Optional:
    'generated_quantities': fn(chain_state, data) -> array,
    'get_num_gq': fn(mcmc_config, data) -> int,
})
```

**CRITICAL: initial_vector sizing**

The `initial_vector` function must return a flat array of size:
```
size = (num_chains_a + num_chains_b) × n_params
```

**NOT** `num_superchains` - that's only for R-hat diagnostics.

```python
def initial_vector(mcmc_config, data):
    n_params = get_n_params(data)
    # CORRECT: use num_chains_a + num_chains_b
    num_chains = mcmc_config['num_chains_a'] + mcmc_config['num_chains_b']
    rng = np.random.default_rng(mcmc_config.get('rng_seed', 42))
    return rng.normal(0, 0.1, size=(num_chains, n_params)).flatten()
```

**CRITICAL: direct_sampler is always required**

Even for MH-only models, `direct_sampler` must be provided and return valid values.
JAX traces all branches of `lax.switch` during compilation, so raising exceptions will fail:

```python
# WRONG - causes compilation error
def direct_sampler(key, chain_state, param_indices, data):
    raise NotImplementedError("MH only")

# CORRECT - placeholder that returns valid values
def direct_sampler(key, chain_state, param_indices, data):
    """Placeholder for MH-only models."""
    return chain_state, key
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
- `COUPLED_TRANSFORM` (2): MH with deterministic coupled parameter transforms (theta-preserving NCP)

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
- `MCOV_SMOOTH` (10): Smooth distance-based covariance and mean scaling
- `MCOV_MODE` (11): Mode-targeting with scalar Mahalanobis distance
- `MCOV_MODE_VEC` (12): Mode-targeting with per-parameter distances

**Settings by Proposal Type:**

| Proposal | Key Settings | Defaults |
|----------|--------------|----------|
| SELF_MEAN | `cov_mult` | 1.0 |
| CHAIN_MEAN | `cov_mult` | 1.0 |
| MIXTURE | `alpha`, `cov_mult` | 0.5, 1.0 |
| MULTINOMIAL | `alpha`, `n_categories` | 0.5, 2 |
| MALA | `cov_mult` | 1.0 |
| MEAN_MALA | `cov_mult` | 1.0 |
| MEAN_WEIGHTED | `cov_mult` | 1.0 |
| MODE_WEIGHTED | `cov_mult` | 1.0 |
| MCOV_WEIGHTED | `cov_mult`, `cov_beta` | 1.0, -0.5 |
| MCOV_WEIGHTED_VEC | `cov_mult`, `cov_beta` | 1.0, -0.5 |
| MCOV_SMOOTH | `cov_mult`, `k_g`, `k_alpha` | 1.0, 10.0, 3.0 |
| MCOV_MODE | `cov_mult`, `k_g`, `k_alpha` | 1.0, 10.0, 3.0 |
| MCOV_MODE_VEC | `cov_mult`, `k_g`, `k_alpha` | 1.0, 10.0, 3.0 |

**Note:** Unrecognized settings keys will trigger a warning (e.g., typo `cov_multt`).

### 3. Data Format

Data is passed as a dict with three tuple fields:

```python
data = {
    "static": (n_subjects, hyperprior_a, hyperprior_b),  # Scalars (hashable)
    "int": (int_array1, int_array2),                      # Integer arrays
    "float": (float_array1, float_array2),                # Float arrays
}
```

**Important constraints:**

1. **static must be hashable Python scalars** (used for JAX compilation caching):
   ```python
   # WRONG - numpy scalars may cause hashing issues
   data["static"] = (np.int64(5), np.float64(1.0))

   # CORRECT - use Python scalars
   data["static"] = (int(5), float(1.0))
   ```

2. **Arrays should be C-contiguous** for performance:
   ```python
   # May cause issues
   data["float"] = (X[:, ::2], y)  # Non-contiguous slice

   # Correct
   data["float"] = (np.ascontiguousarray(X[:, ::2]), y)
   ```

3. **Empty tuples required for unused fields:**
   ```python
   data = {
       "static": (n_obs,),
       "int": (),      # Required even if empty
       "float": (X, y),
   }
   ```

### 4. MCMC Configuration

**All config keys use lowercase with underscores** (not UPPERCASE).

```python
mcmc_config = {
    # === REQUIRED ===
    'posterior_id': 'my_model',      # Registered posterior name
    'num_chains_a': 500,             # Chains in group A
    'num_chains_b': 500,             # Chains in group B

    # === OPTIONAL (with defaults) ===
    'use_double': True,              # float64 precision (default: False)
    'rng_seed': 1977,                # Random seed (default: system time)
    'num_superchains': 100,          # K superchains for R-hat (default: num_chains_a // 5)
    'burn_iter': 5000,               # Burn-in iterations (default: 1000)
    'num_collect': 10000,            # Collection iterations (default: 1000)
    'thin_iteration': 10,            # Thinning interval (default: 1)
    'benchmark': 100,                # Benchmark iterations (default: 0)
    'save_likelihoods': False,       # Save log-likelihood history (default: False)
    'chunk_size': 100,               # Iterations per compiled chunk (default: 100)

    # Parallel tempering (optional)
    'n_temperatures': 1,             # Number of temperature levels (default: 1 = disabled)
    'beta_min': 0.1,                 # Minimum inverse temperature (default: 0.1)
}
```

**Common mistake:** Using `NUM_SUPERCHAINS` (uppercase) will silently use the default value.

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

### 7. Parallel Tempering

Parallel tempering runs chains at different "temperatures" to improve exploration of multimodal posteriors. Higher temperatures flatten the posterior, making it easier for chains to cross barriers between modes.

**Configuration:**
```python
mcmc_config = {
    # ... other config ...
    'n_temperatures': 8,         # Number of temperature levels (1 = disabled)
    'beta_min': 0.1,             # Minimum inverse temperature (hottest chain)
}
```

**Temperature Ladder:**
Temperatures are spaced geometrically between β=1.0 (target posterior) and β=beta_min:
```
β_i = beta_min^(i / (n_temperatures - 1))  for i = 0, 1, ..., n_temperatures-1
```

**DEO (Deterministic Even-Odd) Swap Scheme:**
Adjacent temperature levels swap states using a deterministic alternating pattern:
- Even rounds: attempt swaps (0↔1), (2↔3), (4↔5), ...
- Odd rounds: attempt swaps (1↔2), (3↔4), (5↔6), ...

Swaps are accepted with Metropolis probability based on the log-likelihood difference.

**Output:**
- History contains ALL chains (all temperatures)
- `temperature_history` array tracks which temperature each chain has at each saved iteration
- Filter to β=1 chains post-hoc for posterior inference

## Key Data Structures

### BlockArrays (frozen dataclass, JAX pytree)

Pre-parsed block specifications for efficient JAX operations:

```python
@dataclass(frozen=True)
class BlockArrays:
    # Basic block info
    indices: jnp.ndarray         # (n_blocks, max_size) parameter indices
    types: jnp.ndarray           # (n_blocks,) SamplerType per block
    masks: jnp.ndarray           # (n_blocks, max_size) valid param mask

    # Single-proposal fields (backward compatible)
    proposal_types: jnp.ndarray  # (n_blocks,) remapped indices into compact dispatch
    settings_matrix: jnp.ndarray # (n_blocks, MAX_SETTINGS)

    # Mixed-proposal group fields (for blocks with multiple proposal types)
    group_starts: jnp.ndarray        # (n_blocks, MAX_PROPOSAL_GROUPS)
    group_ends: jnp.ndarray          # (n_blocks, MAX_PROPOSAL_GROUPS)
    group_proposal_types: jnp.ndarray  # (n_blocks, MAX_PROPOSAL_GROUPS)
    group_settings: jnp.ndarray      # (n_blocks, MAX_PROPOSAL_GROUPS, MAX_SETTINGS)
    group_masks: jnp.ndarray         # (n_blocks, MAX_PROPOSAL_GROUPS)
    num_groups: jnp.ndarray          # (n_blocks,)

    # Metadata
    max_size: int
    num_blocks: int
    total_params: int
    used_proposal_types: tuple   # Original ProposalType values used (dispatch order)
    has_mixed_proposals: bool    # True if any block has multiple groups
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

### rmcmc() - Multi-Run Sampling (Recommended)

**This is the recommended entry point for production sampling.** It provides:
- Automatic checkpoint management (save after each run, resume on crash)
- Flexible run scheduling (reset runs, resume runs)
- Organized output directory structure
- Graceful recovery from interruptions

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

### rmcmc_single() - Single-Run Sampling

For custom workflows or single-shot sampling without checkpoint management:

```python
from bamcmc import rmcmc_single

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
# checkpoint dict for saving/resuming (caller manages persistence)
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
- `DEFAULT_CHUNK_SIZE`: Default iteration chunk size (actual value from `run_params.CHUNK_SIZE`)

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

**Checkpoint I/O:**
- `save_checkpoint(filepath, carry, user_config, metadata=None)`: Save chain states to disk
- `load_checkpoint(filepath)`: Load checkpoint, returns dict with states, keys, iteration, etc.
- `initialize_from_checkpoint(checkpoint, user_config, runtime_ctx, ...)`: Create MCMC carry from checkpoint

**History Processing:**
- `combine_batch_histories(batch_paths)`: Merge multiple history files into one
- `apply_burnin(history, iterations, likelihoods=None, min_iteration=0)`: Filter samples before threshold
- `compute_rhat_from_history(history, K, M)`: Compute R-hat on combined history

**Output Management:**
- `scan_checkpoints(output_dir, model_name)`: Find existing checkpoint and history files
- `get_latest_checkpoint(output_dir, model_name)`: Get path to most recent checkpoint
- `clean_model_files(output_dir, model_name, mode='all')`: Delete output files (see below)
- `get_model_paths()`: Get standardized paths for model output
- `ensure_model_dirs()`: Create output directory structure

**Post-Processing (Memory-Efficient Analysis):**
- `get_model_paths()`: Get standardized paths for model output
- `split_history_by_subject()`: Split full history into per-subject files
- `postprocess_all_histories()`: Batch process all history files
- `save_prior_config()`: Save prior parameters to JSON for JAX-free plotting
- `load_prior_config()`: Load prior parameters from JSON

```python
from bamcmc import postprocess_all_histories

# Split 35GB of history files into per-subject files
postprocess_all_histories(
    output_dir='../data/output/dbar_fed0',
    model_name='mix2_EH_bhm',
    n_subjects=245,
    n_hyper=18,           # Number of hyperparameters
    params_per_subject=13,
    hyper_first=False,    # Layout: [subjects][hyper] (standard BHM)
)
```

**Parameter Layout:**
- `hyper_first=False` (default): Subject params first, then hyperparameters
- `hyper_first=True`: Hyperparameters first, then subject params

### reset_utils.py
- `generate_reset_vector()`: Create new starting points from checkpoint
- Uses cross-chain mean + small noise to rescue stuck chains

## Output Directory Management

The package provides utilities for managing checkpoint and history files across multiple runs.

### Directory Structure

When using `rmcmc()` with `use_nested_structure=True` (default):

```
{output_dir}/{model_name}/
├── checkpoints/
│   ├── checkpoint_000.npz    # Initial state (iteration=0)
│   ├── checkpoint_001.npz    # After 1st sampling run
│   ├── checkpoint_002.npz    # After 2nd sampling run
│   └── ...
└── history/
    └── full/
        ├── history_000.npz   # Samples from 1st run
        ├── history_001.npz   # Samples from 2nd run
        └── ...
```

### Scanning for Existing Files

```python
from bamcmc import scan_checkpoints, get_latest_checkpoint

# Find all checkpoints and histories
scan = scan_checkpoints(output_dir, model_name)
# Returns: {
#   'checkpoint_files': [(run_idx, path), ...],
#   'history_files': [(run_idx, path), ...],
#   'latest_checkpoint': path or None,
#   'latest_run_index': int or -1,
# }

# Get just the latest checkpoint path
latest = get_latest_checkpoint(output_dir, model_name)  # path or None
```

### Cleaning Output Files

The `clean_model_files()` function provides two cleaning modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `'all'` | Delete all checkpoints and histories | Complete fresh start |
| `'keep_latest'` | Delete histories and old checkpoints, **keep latest checkpoint** | Normal workflow (allows resume) |

```python
from bamcmc import clean_model_files

# Delete everything (fresh start)
result = clean_model_files(output_dir, model_name, mode='all')

# Keep latest checkpoint for resume (default for iterative workflows)
result = clean_model_files(output_dir, model_name, mode='keep_latest')

# Result contains:
# {
#   'deleted_checkpoints': [paths...],
#   'deleted_histories': [paths...],
#   'kept_checkpoint': path or None,
# }
```

### Recommended Workflow

For iterative MCMC workflows, the recommended pattern is:

1. **First run**: No checkpoint exists, starts fresh
2. **Subsequent runs**: Resume from latest checkpoint
3. **Before each session**: Clean with `mode='keep_latest'` to remove old histories but preserve resume capability

```python
from bamcmc import rmcmc, clean_model_files

# Clean old files but keep latest checkpoint
clean_model_files(output_dir, model_name, mode='keep_latest')

# Run MCMC - will resume if checkpoint exists, else start fresh
summary = rmcmc(
    mcmc_config,
    data,
    output_dir=output_dir,
    run_schedule=[("resume", 5)],  # 5 resume runs
    burn_in_fresh=True,            # Only burn-in on fresh runs
)
```

**Key behavior**: With `mode='keep_latest'`, if no checkpoint exists, `resume` mode automatically starts fresh. This means you rarely need explicit `mode='all'` unless you want to discard learned posterior state.

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
    """Log posterior for a parameter block. Must be JAX-traceable."""
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
        BlockSpec(size=1, sampler_type=SamplerType.METROPOLIS_HASTINGS,
                  proposal_type=ProposalType.CHAIN_MEAN, label="Hyperprior")
    ]

def my_initial_vector(mcmc_config, data):
    """Initial values. MUST return size = (num_chains_a + num_chains_b) * n_params."""
    n_params = data['static'][0] * 2 + 1  # example calculation
    num_chains = mcmc_config['num_chains_a'] + mcmc_config['num_chains_b']  # NOT num_superchains!
    rng = np.random.default_rng(mcmc_config.get('rng_seed', 42))
    return rng.normal(0, 0.1, size=(num_chains, n_params)).flatten()

def my_direct_sampler(key, chain_state, param_indices, data):
    """Placeholder - required even for MH-only models."""
    return chain_state, key

register_posterior('my_model', {
    'log_posterior': my_log_posterior,
    'batch_type': my_batch_specs,
    'initial_vector': my_initial_vector,
    'direct_sampler': my_direct_sampler,  # Required!
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

### Mixed Proposals (Multiple Proposal Types per Block)

When a block contains both continuous and discrete parameters, you can use different
proposal types for different sub-groups within the block:

```python
from bamcmc import BlockSpec, ProposalGroup, SamplerType, ProposalType, create_mixed_subject_blocks

# Define proposal groups for a block of size 13:
# - Params 0-11: continuous (use MCOV_MODE)
# - Param 12: discrete model indicator (use MULTINOMIAL)
groups = [
    ProposalGroup(start=0, end=12, proposal_type=ProposalType.MCOV_MODE,
                  settings={'cov_mult': 1.0}),
    ProposalGroup(start=12, end=13, proposal_type=ProposalType.MULTINOMIAL,
                  settings={'alpha': 0.5, 'n_categories': 2}),
]

# Create block with mixed proposals
spec = BlockSpec(
    size=13,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_groups=groups,
    label="Subject_0"
)

# Or use helper for many identical subject blocks
specs = create_mixed_subject_blocks(245, groups, label_prefix="Subj")
```

**Key points:**
- Groups must be contiguous and cover the entire block
- Maximum 4 groups per block (`MAX_PROPOSAL_GROUPS`)
- Hastings ratios are combined by multiplication: q(θ|θ')/q(θ'|θ) = Π[qᵢ ratio]
- Gradient-based proposals (MALA) cannot be used for discrete groups
- Settings are per-group (each group has its own `alpha`, `cov_mult`, etc.)

**Use case:** Mixture models where each subject has continuous preferences plus a
discrete model assignment indicator. Combining these in one block enables joint
updates that can improve mixing for the discrete indicator.

## Performance Considerations

1. **Block size**: Larger blocks = fewer kernel calls but coarser updates
2. **Chunk size**: `chunk_size` in mcmc_config controls iterations per compiled chunk.
   - Default: 100. Lower values (e.g., 10) reduce compilation time/memory but add ~1-10% runtime overhead
   - Useful for OOM during compilation or faster iteration during development
3. **Proposal type**: CHAIN_MEAN often mixes faster but needs good initialization
4. **Float precision**: USE_DOUBLE=True is slower but more stable
5. **Compilation**: First run compiles (~3-17s); cached runs are fast (~3s)

## Debugging Tips

1. **Check R-hat**: Values > 1.1 indicate non-convergence
2. **Acceptance rates**: Stored in diagnostics, target 20-40%
3. **Trace plots**: Plot `history[:, chain_idx, param_idx]` over iterations
4. **Block labels**: Use `label` in BlockSpec for clearer error messages
5. **Validation errors**: `validate_mcmc_config()` catches common issues

## JAX-Compatible Coding Patterns

When writing `log_posterior` and other functions that JAX traces:

### Module-Level Functions Required

For compilation caching to work, define functions at module level (not closures):

```python
# WRONG - closure won't cache properly
def make_log_posterior(hyperparams):
    def log_posterior(chain_state, param_indices, data):
        return compute(hyperparams)  # Uses closure variable
    return log_posterior

# CORRECT - module-level, get hyperparams from data
def log_posterior(chain_state, param_indices, data):
    hyperparams = data["static"][0]  # From data, not closure
    return compute(hyperparams)
```

### Avoid Python Loops for Large Iterations

Python `for` loops are unrolled during tracing. For large N, use `lax.fori_loop` or `vmap`:

```python
# SLOW - N iterations unrolled, huge trace
def log_posterior(chain_state, param_indices, data):
    n_items = data["static"][0]
    log_lik = 0.0
    for i in range(n_items):  # Unrolled!
        log_lik += item_lik(i, ...)
    return log_lik

# FAST - single traced loop
def log_posterior(chain_state, param_indices, data):
    n_items = data["static"][0]
    def body(i, acc):
        return acc + item_lik(i, ...)
    return jax.lax.fori_loop(0, n_items, body, 0.0)

# FASTEST - vectorized
def log_posterior(chain_state, param_indices, data):
    return jnp.sum(jax.vmap(item_lik)(jnp.arange(n_items), ...))
```

### No Python Control Flow on Traced Values

```python
# WRONG - x is a tracer, not a concrete value
if x > 0:
    return log_prob

# CORRECT - use jnp.where for conditional logic
return jnp.where(x > 0, log_prob, -jnp.inf)
```

## Troubleshooting

### Common Error Messages

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `cannot reshape array of shape (X,) into shape (Y, Z)` | `initial_vector` returns wrong size | Use `(num_chains_a + num_chains_b) * n_params` |
| `KeyError: 'direct_sampler'` | Missing `direct_sampler` in registration | Add placeholder function that returns `(chain_state, key)` |
| `NotImplementedError` during compilation | `direct_sampler` raises exception | Make it return valid values (JAX traces all branches) |
| `Abstract tracer value encountered` | Python control flow on JAX traced values | Use `jnp.where`, `lax.cond`, `lax.switch` |
| `jax.errors.ConcretizationTypeError` | Using traced value as array shape/index | Move shape computation before JAX tracing |
| `cholesky: decomposition failed` | Ill-conditioned covariance matrix | Check for NaN/Inf in parameters, increase `COV_NUGGET` |

### Initialization Issues

**Symptom**: All chains stuck or immediately divergent

**Solutions**:
1. Check `initial_vector` returns sensible parameter values
2. Verify parameters are in prior support (e.g., positive for variance)
3. Use smaller initial variance (`rng.normal(0, 0.1, ...)` not `0, 1`)
4. Check `log_posterior` returns finite values for initial state

### Slow Compilation

**Symptom**: First run takes minutes instead of seconds

**Solutions**:
1. Reduce Python loops in `log_posterior` (use `vmap`/`fori_loop`)
2. Avoid creating arrays inside traced functions
3. Check for accidental recompilation (data shape changes)
4. Clear JAX cache if corrupted: `rm -rf ./jax_cache/`

## Minimal Template

Copy this as a starting point for new posteriors:

```python
"""Minimal bamcmc posterior template."""
import numpy as np
import jax.numpy as jnp
from bamcmc import register_posterior, BlockSpec, SamplerType, ProposalType

MODEL_ID = "my_model"

def log_posterior(chain_state, param_indices, data):
    """Compute log posterior. Must be JAX-traceable."""
    params = chain_state[param_indices]
    n_params = data["static"][0]

    # Example: simple Gaussian prior
    log_prior = -0.5 * jnp.sum(params ** 2)

    # Your log likelihood here
    X, y = data["float"]
    # log_lik = ...

    return log_prior  # + log_lik

def batch_type(mcmc_config, data):
    """Define parameter blocks."""
    n_params = data["static"][0]
    return [
        BlockSpec(
            size=n_params,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=ProposalType.CHAIN_MEAN,
            settings={"cov_mult": 1.0},
            label="params",
        )
    ]

def initial_vector(mcmc_config, data):
    """Initial values. Size = (num_chains_a + num_chains_b) * n_params."""
    n_params = data["static"][0]
    num_chains = mcmc_config["num_chains_a"] + mcmc_config["num_chains_b"]
    rng = np.random.default_rng(mcmc_config.get("rng_seed", 42))
    return rng.normal(0, 0.1, size=(num_chains, n_params)).flatten()

def direct_sampler(key, chain_state, param_indices, data):
    """Placeholder for MH-only models. Must return valid values."""
    return chain_state, key

def register():
    """Call this before sampling."""
    register_posterior(MODEL_ID, {
        "log_posterior": log_posterior,
        "batch_type": batch_type,
        "initial_vector": initial_vector,
        "direct_sampler": direct_sampler,
    })
```

## Dependencies

- **jax** / **jaxlib**: GPU-accelerated array operations
- **numpy**: Array utilities
- **scipy** (optional): For integration tests only

---

## COUPLED_TRANSFORM Sampler Type

### Overview

The `COUPLED_TRANSFORM` sampler type enables **theta-preserving updates** for Non-Centered Parameterization (NCP) in hierarchical models. When updating hyperparameters (μ, σ), it simultaneously adjusts epsilon values to keep θ = μ + σε constant:

```
ε' = (θ - μ') / σ' = (μ + σε - μ') / σ'
```

This makes the **likelihood cancel** from the acceptance ratio, achieving ~40-50% acceptance rates for hyperparameters (vs ~1-2% with standard NCP updates).

### Implementation

**Status**: ✅ Implemented and Working

**Key Files**:
- `batch_specs.py`: `SamplerType.COUPLED_TRANSFORM` enum value
- `mcmc/sampling.py`: `coupled_transform_step()` function
- Model posteriors implement `coupled_transform_dispatch()` function

### Usage

```python
# In batch_type function:
BlockSpec(
    size=2,  # (mean, logsd)
    sampler_type=SamplerType.COUPLED_TRANSFORM,
    proposal_type=ProposalType.MCOV_WEIGHTED_VEC,  # Adaptive!
    settings={'cov_mult': 1.0, 'cov_beta': -0.9},
    label="Hyper_r",
)

# Register posterior with coupled_transform_dispatch:
register_posterior('my_ncp_model', {
    'log_posterior': log_posterior_fn,
    'batch_type': batch_specs_fn,
    'initial_vector': initial_vector_fn,
    'coupled_transform_dispatch': coupled_transform_dispatch_fn,
})
```

### Acceptance Ratio

The acceptance ratio for COUPLED_TRANSFORM excludes the data likelihood:

```
log α = log(proposal_ratio)           # From MCOV_WEIGHTED_VEC
      + N × (log σ - log σ')          # Jacobian correction
      - 0.5 × Σ[(ε'ᵢ)² - εᵢ²]         # Epsilon prior ratio
      + log p(μ', σ') - log p(μ, σ)   # Hyperprior ratio
```

### Example Model

See `/workspace/code/posteriors/mix2_mh_ncp_bhm.py` for a complete implementation.

### Mathematical Documentation

- `/workspace/technical_notes/coupled_transform_sampler.tex` - Complete derivation
- `/workspace/code/docs/theta_preserving_ncp_sampler.tex` - Pedagogical explanation

---

## Known Issues & Improvement Suggestions

This section documents known issues and potential improvements identified during code review.

### High Priority (All Fixed)

#### 1. ~~Emoji in Error Output~~ (FIXED)

**Status**: Fixed. Replaced emojis with text-based indicators (`[ERROR]`, `[WARN]`, `[INFO]`, `[OK]`) in:
- `src/bamcmc/error_handling.py`
- `src/bamcmc/mcmc/backend.py`

#### 2. ~~Silent Failure on Unknown Settings Keys~~ (ALREADY IMPLEMENTED)

**Status**: Already implemented. The code in `settings.py:93-100` already warns on unknown keys.

#### 3. ~~Missing Checkpoint Compatibility Validation~~ (FIXED)

**Status**: Fixed. Added `_validate_checkpoint_compatibility()` in `src/bamcmc/mcmc/backend.py` that:
- Validates parameter count matches before resume/reset
- Checks posterior ID matches
- Provides clear error messages when checkpoint is incompatible

### Medium Priority

#### 4. Undocumented Magic Numbers (diagnostics.py)

**Issue**: The nested R-hat threshold uses an undocumented constant.

**Location**: `src/bamcmc/mcmc/diagnostics.py:142`

**Current Code**:
```python
tau = 1e-4
threshold = np.sqrt(1 + 1/M + tau)
```

**Suggested Fix**: Document the source:
```python
# tau: small regularization constant for nested R-hat threshold
# from Margossian et al. (2022), prevents division instability when M is large
TAU_NESTED_RHAT = 1e-4
threshold = np.sqrt(1 + 1/M + TAU_NESTED_RHAT)
```

#### 5. Missing TypedDict for Data Structure

**Issue**: The `data` dict structure is documented but not typed, making it easy to misuse.

**Suggested Addition** (in `src/bamcmc/mcmc/types.py`):
```python
from typing import TypedDict, Tuple
import numpy as np

class MCMCData(TypedDict):
    static: Tuple[int, ...]           # Scalars (hashable)
    int: Tuple[np.ndarray, ...]       # Integer arrays
    float: Tuple[np.ndarray, ...]     # Float arrays
```

#### 6. Settings Key Mapping Boilerplate (settings.py)

**Issue**: Adding a new setting requires edits in 3 places (SettingSlot enum, SETTING_DEFAULTS, key_to_slot dict).

**Suggested Fix**: Auto-generate the mapping:
```python
class SettingSlot(IntEnum):
    COV_MULT = 0
    ALPHA = 1
    # ... etc

# Auto-generate from enum names (lowercase)
KEY_TO_SLOT = {slot.name.lower(): slot.value for slot in SettingSlot}
```

### Lower Priority

#### 7. Missing Dispatch Table Assertion (sampling.py)

**Issue**: The compact proposal dispatch table is built dynamically but not validated.

**Location**: `src/bamcmc/mcmc/sampling.py:88`

**Suggested Fix**:
```python
dispatch_table = [
    (lambda fn: lambda op: fn((*op, grad_fn, block_mode)))(PROPOSAL_REGISTRY[ptype])
    for ptype in used_proposal_types
]

assert len(dispatch_table) == len(used_proposal_types), \
    f"Dispatch table size mismatch: {len(dispatch_table)} vs {len(used_proposal_types)}"
```

#### 8. Test Coverage Gaps

**Current Gaps**:
- No explicit tests for proposal Hastings ratio symmetry
- No tests for checkpoint resume with mismatched data shapes
- No tests for edge cases: single chain, empty blocks
- Direct sampler interface not explicitly tested
- No performance regression tests

**Suggested Additions**:
```python
# tests/test_proposals.py
def test_hastings_ratio_symmetry():
    """Verify q(y|x) / q(x|y) computed correctly for each proposal."""
    # Sample x, y, compute ratio both directions, verify consistency

def test_checkpoint_shape_mismatch():
    """Verify clear error when checkpoint doesn't match model."""
    # Create checkpoint with N params, try to load with M != N
```

### Strengths (Don't Change)

The following patterns are well-implemented and should be preserved:

1. **JAX pytree registration** - BlockArrays properly registered for JAX transforms
2. **Compact proposal dispatch** - Only traces used proposals (memory efficient)
3. **COUPLED_TRANSFORM implementation** - Theta-preserving sampler is mathematically elegant
4. **Cross-session compilation caching** - Excellent user experience
5. **Registry pattern** - Clean plugin architecture for posteriors
6. **Frozen dataclasses** - Immutability for JAX static args
7. **Comprehensive docstrings** - Most functions well-documented

### Summary Table

| Priority | Issue | File | Status |
|----------|-------|------|--------|
| ~~High~~ | ~~Emoji in diagnostics~~ | error_handling.py, backend.py | **FIXED** |
| ~~High~~ | ~~Silent unknown settings~~ | settings.py | **Already implemented** |
| ~~High~~ | ~~Checkpoint compatibility~~ | backend.py | **FIXED** |
| Medium | Magic number docs | diagnostics.py | Open |
| Medium | TypedDict for data | types.py | Open |
| Medium | Settings auto-mapping | settings.py | Open |
| Low | Dispatch assertion | sampling.py | Open |
| Low | Test coverage | tests/ | Open |
