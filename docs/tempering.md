# Parallel Tempering

## Overview

Parallel tempering (PT) runs chains at multiple "temperatures" to improve exploration of multimodal posteriors. Higher temperatures flatten the posterior landscape, allowing chains to cross energy barriers between modes. Information flows from hot (exploratory) chains to cold (target) chains through temperature swaps.

Enable parallel tempering by setting `n_temperatures > 1` in the MCMC config. When `n_temperatures=1` (default), tempering is disabled and all chains sample at the target posterior.

## Index Process vs Standard PT

bamcmc implements **index-process** parallel tempering, which differs from standard PT:

- **Standard PT**: Swaps chain *states* (parameter vectors) between temperatures.
- **Index process**: Swaps temperature *assignments* between chains. Each chain keeps its continuous parameter trace; only its temperature label changes.

Benefits of the index process:
- Each chain maintains a continuous trajectory through parameter space
- Temperature history is tracked per chain, enabling post-hoc filtering to beta=1 samples
- No discontinuities in chain traces from state swaps

## DEO Scheme

The default swap strategy is the **Deterministic Even-Odd (DEO)** scheme (Syed et al., 2021):

- **Even rounds** (parity=0): Attempt swaps for pairs (0,1), (2,3), (4,5), ...
- **Odd rounds** (parity=1): Attempt swaps for pairs (1,2), (3,4), (5,6), ...

This creates a deterministic "conveyor belt" that moves chains through temperature space with **O(N) round-trip time**, compared to O(N^2) for stochastic schemes.

Swap acceptance between temperatures beta_i and beta_j (beta_i > beta_j):

```
alpha = min(1, exp((beta_i - beta_j) * (log_lik(theta_j) - log_lik(theta_i))))
```

where `log_lik` is the log-likelihood (prior terms cancel).

## Temperature Ladder

Temperatures are spaced geometrically between beta=1.0 (target posterior) and beta=beta_min:

```
beta_i = beta_min^(i / (n_temperatures - 1))    for i = 0, 1, ..., n_temperatures-1
```

Example with `n_temperatures=8, beta_min=0.1`:

```
[1.000, 0.720, 0.518, 0.373, 0.268, 0.193, 0.139, 0.100]
```

## Configuration

```python
mcmc_config = {
    'posterior_id': 'my_model',
    'num_chains_a': 500,
    'num_chains_b': 500,
    'num_superchains': 100,
    'burn_iter': 5000,
    'num_collect': 10000,
    'thin_iteration': 10,
    'use_double': True,
    # Tempering options
    'n_temperatures': 8,             # Number of temperature levels (1 = disabled)
    'beta_min': 0.1,                 # Minimum inverse temperature (hottest chain)
    'swap_every': 1,                 # Iterations between swap attempts (default: 1)
    'per_temp_proposals': True,      # Separate proposal stats per temperature (default: True)
    'use_deo': True,                 # Use DEO scheme (default: True)
}

# Use rmcmc for multi-run sampling with automatic checkpointing.
# Each run saves a checkpoint, so subsequent runs resume from the last
# iteration. This keeps per-run memory bounded regardless of total
# sampling length.
summary = rmcmc(
    mcmc_config,
    data,
    output_dir='./output',
    run_schedule=[("resume", 10)],   # 10 resume runs
)
```

For single-run workflows (e.g., testing or custom orchestration), use `rmcmc_single()` instead.

### Chain Count Requirements

When tempering is enabled:
1. `num_chains` (= `num_chains_a + num_chains_b`) must be divisible by `n_temperatures`
2. `chains_per_temperature` (= `num_chains / n_temperatures`) must be even (for A/B groups)

Example: `num_chains_a=500, num_chains_b=500, n_temperatures=8` gives 125 chains per temperature.

## Output & Interpretation

### Results Dict Fields

When `n_temperatures > 1`, the results dict includes:

| Field | Shape | Description |
|-------|-------|-------------|
| `temperature_history` | `(num_collect, num_chains)` | Temperature index per chain per saved iteration |
| `temperature_ladder` | `(n_temperatures,)` | Array of beta values from 1.0 to beta_min |
| `swap_rates` | `(n_temperatures - 1,)` | Acceptance rate for each adjacent temperature pair |
| `round_trip_rate` | float | Mean round trips per sample across chains |
| `round_trip_counts` | `(num_chains,)` | Number of complete round trips per chain |

When `n_temperatures=1`, all these fields are `None`.

### Filtering to Target Temperature

The history contains samples from ALL chains at ALL temperatures. For posterior inference, filter to beta=1 samples.

When using `rmcmc`, load and combine the saved history files, then filter:

```python
from bamcmc import combine_batch_histories, filter_beta1_samples

# After rmcmc completes, combine the per-run history files
history, temp_history, _ = combine_batch_histories(
    summary['history_files'], include_temp_history=True
)

if temp_history is not None:
    filtered, counts = filter_beta1_samples(history, temp_history)
    # filtered: (min_count, n_chains, n_params) - only beta=1 samples
    # counts: (n_chains,) - number of beta=1 samples per chain
```

With `rmcmc_single`, the results dict contains the arrays directly:

```python
results, checkpoint = rmcmc_single(config, data)
if results['temperature_history'] is not None:
    filtered, counts = filter_beta1_samples(
        results['history'], results['temperature_history']
    )
```

### Checkpoint Fields

Checkpoints include the full tempering state for exact resume:

| Field | Description |
|-------|-------------|
| `n_temperatures` | Number of temperature levels |
| `temperature_ladder` | Beta values array |
| `temp_assignments_A` | Temperature index per chain in group A |
| `temp_assignments_B` | Temperature index per chain in group B |
| `swap_accepts` | Accepted swaps per temperature pair |
| `swap_attempts` | Attempted swaps per temperature pair |
| `swap_parity` | DEO parity (0 or 1) |

## Interpreting Diagnostics

### Swap Acceptance Rates

- **Target range**: 20-40% per adjacent pair
- **< 10%**: Temperature spacing is too wide; increase `n_temperatures` or increase `beta_min`
- **> 60%**: Temperatures are too close; decrease `n_temperatures` to save computation

### Round-Trip Rate

A round trip is a chain traversing from beta=1 (coldest) to beta=beta_min (hottest) and back. Higher round-trip rates indicate better mixing through temperature space.

- **Warning sign**: Mean trips per chain < 1 means chains are not completing full round trips
- The DEO scheme should produce round-trip rates close to the theoretical maximum of O(1/N)

### R-hat with Tempering

R-hat is computed on full traces (all temperatures) by default. For beta=1-only convergence analysis, use `filter_beta1_samples()` post-hoc and compute R-hat on the filtered history.

## Choosing Parameters

### n_temperatures

- **4-8**: Good starting range for mildly multimodal posteriors
- **8-16**: For strongly multimodal posteriors with high energy barriers
- More temperatures = better mixing but higher computational cost (linear scaling)

### beta_min

- **0.1** (default): Flattens the likelihood by 10x. Suitable for most cases.
- **0.01**: Very hot chains for strongly multimodal posteriors with deep barriers
- **0.5**: Mild tempering for weakly multimodal posteriors

### swap_every

- **1** (default): Attempt swaps every iteration. Recommended for most cases.
- Higher values reduce swap overhead but slow temperature mixing.

## Disabling Tempering

Set `n_temperatures=1` (the default) to disable parallel tempering entirely. No temperature arrays are allocated and no swap computations occur.

## Computational Cost

- **Per-iteration overhead**: Each swap round evaluates the log-posterior twice per chain (at beta=0 and beta=1) to compute the log-likelihood. This is inherent to the index process design.
- **Memory**: All chains at all temperatures are stored in the history array. Memory scales as `num_collect * num_chains * n_params`. Because tempering multiplies the effective chain count, use `rmcmc` with multiple shorter runs rather than one long `rmcmc_single` run. Each run checkpoints and frees memory, so total sampling length is not bounded by GPU/host RAM.
- **Compilation**: Tempering adds temperature swap logic to the compiled kernel. Recompilation is needed when changing `n_temperatures`.
