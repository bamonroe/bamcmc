# Configuration Reference

This document covers data format requirements, MCMC configuration options, and parallel tempering setup.

## Data Format

Data is passed as a dict with three tuple fields:

```python
data = {
    "static": (n_subjects, hyperprior_a, hyperprior_b),  # Scalars (hashable)
    "int": (int_array1, int_array2),                      # Integer arrays
    "float": (float_array1, float_array2),                # Float arrays
}
```

### Important Constraints

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

## MCMC Configuration

**All config keys use lowercase with underscores** (not UPPERCASE).

```python
mcmc_config = {
    # === REQUIRED ===
    'posterior_id': 'my_model',      # Registered posterior name
    'num_chains_a': 500,             # Chains in group A
    'num_chains_b': 500,             # Chains in group B

    # === OPTIONAL (with defaults) ===
    'use_double': True,              # float64 precision (default: True)
    'rng_seed': 1977,                # Random seed (default: 42)
    'num_superchains': 100,          # K superchains for R-hat (default: num_chains)
    'burn_iter': 5000,               # Burn-in iterations (default: 0)
    'num_collect': 10000,            # Collection iterations (default: 10)
    'thin_iteration': 10,            # Thinning interval (default: 100)
    'benchmark': 100,                # Benchmark iterations (default: 10)
    'save_likelihoods': False,       # Save log-likelihood history (default: False)
    'chunk_size': 100,               # Iterations per compiled chunk (default: 100)

    # Parallel tempering (optional)
    'n_temperatures': 1,             # Number of temperature levels (default: 1 = disabled)
    'beta_min': 0.1,                 # Minimum inverse temperature (default: 0.1)
}
```

**Common mistake:** Using `NUM_SUPERCHAINS` (uppercase) will silently use the default value.

### Configuration Key Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `posterior_id` | str | *required* | Registered posterior name |
| `num_chains_a` | int | *required* | Chains in group A |
| `num_chains_b` | int | *required* | Chains in group B |
| `use_double` | bool | `True` | Use float64 precision |
| `rng_seed` | int | `42` | Random seed |
| `num_superchains` | int | `num_chains` | K superchains for nested R-hat |
| `burn_iter` | int | `0` | Burn-in iterations |
| `num_collect` | int | `10` | Collection iterations |
| `thin_iteration` | int | `100` | Thinning interval |
| `benchmark` | int | `10` | Benchmark iterations |
| `save_likelihoods` | bool | `False` | Save log-likelihood history |
| `chunk_size` | int | `100` | Iterations per compiled chunk |
| `resume_runs` | int | `1` | Number of resume runs for `rmcmc()` |
| `reset_runs` | int | `0` | Number of reset runs for `rmcmc()` (scheduled before resume runs) |
| `run_schedule` | list | `None` | Explicit `[(mode, count), ...]` override; ignores resume_runs/reset_runs |
| `n_temperatures` | int | `1` | Temperature levels (1 = disabled) |
| `beta_min` | float | `0.1` | Minimum inverse temperature |

## Parallel Tempering

Parallel tempering runs chains at different "temperatures" to improve exploration of multimodal posteriors. Higher temperatures flatten the posterior, making it easier for chains to cross barriers between modes.

```python
mcmc_config = {
    # ... other config ...
    'n_temperatures': 8,         # Number of temperature levels (1 = disabled)
    'beta_min': 0.1,             # Minimum inverse temperature (hottest chain)
}
```

Results include `temperature_ladder`, `swap_rates`, `round_trip_rate`, and `round_trip_counts` when tempering is active. Use `filter_beta1_samples()` to extract posterior samples.

For full details on the index process, DEO scheme, parameter tuning, and diagnostics interpretation, see [docs/tempering.md](./tempering.md).
