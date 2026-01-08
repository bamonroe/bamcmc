# bamcmc

Bayesian MCMC sampling package with coupled A/B chains and nested R-hat diagnostics.

## Features

- **Coupled A/B Sampling**: Chains split into two groups where each group's proposal distribution is informed by the other group's current state
- **Nested R-hat Diagnostics**: Supports superchain/subchain structure for improved convergence diagnostics (Margossian et al., 2022)
- **GPU Acceleration**: Built on JAX for efficient GPU-based sampling
- **Flexible Proposal System**: 11 proposal types for different sampling scenarios
- **COUPLED_TRANSFORM Sampler**: Theta-preserving updates for Non-Centered Parameterization (NCP) in hierarchical models
- **Registry Pattern**: Easy registration of custom posterior models
- **Multi-run Sampling**: Automatic checkpoint management with reset/resume schedules
- **Cross-session Caching**: JAX compilation persists across Python sessions

## Proposal Types

| Type | Description |
|------|-------------|
| `SELF_MEAN` | Random walk centered on current state |
| `CHAIN_MEAN` | Independent proposal centered on population mean |
| `MIXTURE` | Probabilistic mix of SELF_MEAN and CHAIN_MEAN |
| `MULTINOMIAL` | For discrete parameters on integer grid |
| `MALA` | Metropolis-adjusted Langevin (gradient-based) |
| `MEAN_MALA` | MALA with gradient at coupled mean |
| `MEAN_WEIGHTED` | Adaptive interpolation based on Mahalanobis distance |
| `MODE_WEIGHTED` | Interpolation toward mode (highest log-posterior chain) |
| `MCOV_WEIGHTED` | Mean-covariance weighted with configurable blend |
| `MCOV_WEIGHTED_VEC` | Vectorized per-parameter distance and interpolation |
| `MCOV_SMOOTH` | Smooth three-zone transition: chain_mean at equilibrium, tracking when far |

## Sampler Types

| Type | Description |
|------|-------------|
| `METROPOLIS_HASTINGS` | Standard MH with configurable proposal |
| `DIRECT_CONJUGATE` | Direct/Gibbs sampling for conjugate priors |
| `COUPLED_TRANSFORM` | MH with deterministic coupled transforms (theta-preserving NCP) |

## Installation

```bash
pip install -e .
```

Requires JAX with CUDA support for GPU acceleration.

## Quick Start

```python
from bamcmc import register_posterior, BlockSpec, SamplerType, ProposalType, rmcmc

# Register your posterior
register_posterior('my_model', {
    'log_posterior': my_log_posterior_fn,
    'batch_type': my_batch_type_fn,
    'initial_vector': my_initial_vector_fn,
})

# Run MCMC with run schedule
mcmc_config = {
    'posterior_id': 'my_model',
    'num_chains_a': 500,
    'num_chains_b': 500,
    'burn_iter': 1000,
    'num_collect': 5000,
    'thin_iteration': 10,
}

summary = rmcmc(
    mcmc_config,
    data,
    output_dir='./output',
    run_schedule=[('reset', 3), ('resume', 5)],  # 3 reset runs, then 5 resume runs
)

# Or use rmcmc_single for single-run control
from bamcmc import rmcmc_single
results, checkpoint = rmcmc_single(mcmc_config, data)
history = results['history']
diagnostics = results['diagnostics']
```

## Documentation

See `CLAUDE.md` for detailed package documentation including:
- Core concepts (BlockSpec, proposals, coupled chains)
- Data format requirements
- Adding new proposals and posteriors
- Performance considerations

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
