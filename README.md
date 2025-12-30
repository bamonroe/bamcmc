# bamcmc

Bayesian MCMC sampling package with coupled A/B chains and nested R-hat diagnostics.

## Features

- **Coupled A/B Sampling**: Chains split into two groups where each group's proposal distribution is informed by the other group's current state
- **Nested R-hat Diagnostics**: Supports superchain/subchain structure for improved convergence diagnostics
- **GPU Acceleration**: Built on JAX for efficient GPU-based sampling
- **Flexible Proposal System**: Pluggable proposal distributions (self-mean, chain-mean, mixture, multinomial)
- **Registry Pattern**: Easy registration of custom posterior models

## Installation

```bash
pip install -e .
```

Requires JAX with CUDA support for GPU acceleration.

## Quick Start

```python
from bamcmc import register_posterior, BlockSpec, SamplerType, ProposalType

# Register your posterior
register_posterior('my_model', {
    'log_posterior': my_log_posterior_fn,
    'batch_type': my_batch_type_fn,
    'initial_vector': my_initial_vector_fn,
})

# Run MCMC (all config keys are lowercase)
from bamcmc.mcmc_backend import rmcmc

mcmc_config = {
    'posterior_id': 'my_model',
    'num_chains_a': 500,
    'num_chains_b': 500,
    'burn_iter': 1000,
    'num_collect': 5000,
    'thin_iteration': 10,
}

results, checkpoint = rmcmc(mcmc_config, data)
history = results['history']
diagnostics = results['diagnostics']
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
