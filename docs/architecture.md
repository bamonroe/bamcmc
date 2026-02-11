# Architecture

This document describes the internal architecture of bamcmc, including the coupled chain design, key data structures, module responsibilities, and compilation caching.

## Coupled A/B Chain Architecture

The sampler runs two groups of chains that share proposal statistics:

1. **Group A chains** propose using statistics computed from Group B
2. **Group B chains** propose using statistics computed from Group A
3. This coupling improves mixing while maintaining detailed balance

The `step_mean` and `step_cov` for proposals are computed from the "coupled" group.

## Nested R-hat (Margossian et al., 2022)

For hierarchical models, standard R-hat can miss non-convergence. Nested R-hat:

- Organizes chains into K **superchains** with M **subchains** each
- Checks convergence both within and between superchains
- More sensitive to stuck chains in hierarchical posteriors

```
Total chains = K x M
K = NUM_SUPERCHAINS (independent starting points)
M = NUM_CHAINS / NUM_SUPERCHAINS (perturbed copies of each start)
```

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
from bamcmc import rmcmc

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

### Other Modules

- **posterior_benchmark.py**: `PosteriorBenchmarkManager` caches benchmark results by posterior hash; `compare_benchmark()` compares new vs cached performance
- **checkpoint_helpers.py**: Save/load checkpoints, combine histories, output management (see [checkpoint_helpers.md](./checkpoint_helpers.md))
- **reset_utils.py**: `generate_reset_vector()` creates new starting points from checkpoint using cross-chain mean + small noise
- **settings.py**: Per-block settings management (`SettingSlot` enum)
- **registry.py**: Posterior registration system
- **error_handling.py**: Validation and diagnostics

## JAX Compilation Caching

The package uses cross-session JAX compilation caching:

1. **Environment variable**: `JAX_COMPILATION_CACHE_DIR` set in `jax_config.py`
2. **Module-level functions**: Sampling functions defined at module level (not closures)
3. **Frozen dataclasses**: `BlockArrays` and `RunParams` are hashable static args
4. **AOT compilation**: Uses `.lower().compile()` for explicit compilation

Cache is stored in `./jax_cache/` by default. Clear it to force recompilation.
