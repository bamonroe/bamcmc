# Development Guide

This document covers how to add new proposal types, define new posteriors, run tests, and provides a minimal template for getting started.

## Adding a New Proposal Type

1. Add enum to `batch_specs.py`:
   ```python
   class ProposalType(IntEnum):
       MY_PROPOSAL = 13  # Next available index
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

## Defining a New Posterior

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

## Resuming from Checkpoint

```python
# Resume exact state
results, checkpoint = rmcmc(config, data, resume_from='checkpoint.npz')
history = results['history']
diagnostics = results['diagnostics']

# Reset with noise (rescue stuck chains)
results, checkpoint = rmcmc(config, data, reset_from='checkpoint.npz',
                            reset_noise_scale=0.1)
```

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

## Dependencies

- **jax** / **jaxlib**: GPU-accelerated array operations
- **numpy**: Array utilities
- **scipy** (optional): For integration tests only
