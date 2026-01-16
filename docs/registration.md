# Registering Posteriors

Models must be registered with bamcmc before sampling. This documents the registration system.

## Basic Registration

```python
from bamcmc import register_posterior

register_posterior('my_model', {
    # Required
    'log_posterior': log_posterior_fn,
    'batch_type': batch_specs_fn,
    'initial_vector': initial_vector_fn,

    # Optional
    'direct_sampler': direct_sampler_fn,
    'coupled_transform_dispatch': coupled_transform_dispatch_fn,
    'generated_quantities': gq_fn,
    'get_num_gq': num_gq_fn,
    'get_special_param_indices': special_indices_fn,
})
```

---

## Required Functions

### log_posterior

Computes log posterior for a parameter block.

```python
def log_posterior(chain_state, param_indices, data):
    """
    Log posterior for a parameter block.

    Args:
        chain_state: Full parameter vector (1D JAX array)
        param_indices: Indices of parameters in this block (1D JAX array)
        data: Data dict

    Returns:
        Scalar log posterior value
    """
    params = chain_state[param_indices]
    return compute_log_prob(params, data)
```

See [Log-Posterior Requirements](./log_posterior.md) for details.

---

### batch_type

Returns the list of BlockSpecs defining the parameter structure.

```python
def batch_type(mcmc_config, data):
    """
    Define parameter blocks for the model.

    Args:
        mcmc_config: MCMC configuration dict
        data: Data dict

    Returns:
        List of BlockSpec objects
    """
    n_subjects = data['static'][0]

    subject_specs = [
        BlockSpec(size=3, sampler_type=SamplerType.METROPOLIS_HASTINGS,
                  proposal_type=ProposalType.CHAIN_MEAN, label=f"Subj_{i}")
        for i in range(n_subjects)
    ]

    hyper_specs = [
        BlockSpec(size=2, sampler_type=SamplerType.COUPLED_TRANSFORM,
                  proposal_type=ProposalType.MCOV_WEIGHTED_VEC, label="Hyper")
    ]

    return subject_specs + hyper_specs
```

See [BlockSpec Reference](./block_spec.md) for details.

---

### initial_vector

Returns initial parameter values for all chains.

```python
def initial_vector(mcmc_config, data):
    """
    Generate initial parameter values.

    Args:
        mcmc_config: MCMC configuration dict
        data: Data dict

    Returns:
        1D array of initial values, shape (K * n_params,)
        where K = num_superchains
    """
    K = mcmc_config['num_superchains']
    n_subjects = data['static'][0]
    params_per_subject = 3
    n_hyper = 4

    n_params = n_subjects * params_per_subject + n_hyper

    # Generate K different starting points
    rng = np.random.default_rng(mcmc_config['rng_seed'])
    init = rng.normal(size=(K, n_params))

    # Flatten for the sampler
    return init.flatten()
```

#### Notes on Initial Values

1. **Shape**: Return `(K * n_params,)` where K = `num_superchains`
2. **Diversity**: Each of the K superchains should have different starting values
3. **Constraints**: Ensure values satisfy any constraints (e.g., positive variances)
4. **NCP models**: Initialize epsilon values near 0 (standard normal)

---

## Optional Functions

### direct_sampler

For blocks using `SamplerType.DIRECT_CONJUGATE`.

```python
def direct_sampler(key, chain_state, param_indices, data):
    """
    Sample directly from conditional distribution.

    Args:
        key: JAX random key
        chain_state: Full parameter vector
        param_indices: Indices of parameters to sample
        data: Data dict

    Returns:
        new_state: Updated chain state
        new_key: Consumed random key
    """
    # Extract conditioning variables
    # Sample from conditional
    new_values = sample_conditional(key, conditioning_vars, data)

    # Update state
    new_state = chain_state.at[param_indices].set(new_values)
    return new_state, new_key
```

---

### coupled_transform_dispatch

For blocks using `SamplerType.COUPLED_TRANSFORM`.

```python
def coupled_transform_dispatch(key, chain_state, primary_indices,
                               proposed_primary, data):
    """
    Transform coupled parameters for theta-preserving updates.

    Args:
        key: JAX random key
        chain_state: Full parameter vector
        primary_indices: Indices of hyperparameters being proposed
        proposed_primary: Proposed hyperparameter values
        data: Data dict

    Returns:
        coupled_indices: Indices of epsilon parameters to transform
        new_coupled: Transformed epsilon values
        log_jacobian: Log |det(Jacobian)|
        log_prior_ratio: log p(new_eps) - log p(old_eps)
        key: Updated random key
    """
    # Identify which hyperparameter block
    first_idx = primary_indices[0]

    # Get old hyperparameters
    mu_old = chain_state[first_idx]
    sigma_old = jnp.exp(chain_state[first_idx + 1])

    # Get new hyperparameters
    mu_new = proposed_primary[0]
    sigma_new = jnp.exp(proposed_primary[1])

    # Get epsilon indices
    coupled_indices = get_eps_indices(first_idx, data)
    N = len(coupled_indices)

    # Current epsilon
    old_eps = chain_state[coupled_indices]

    # Theta values (preserved)
    theta = mu_old + sigma_old * old_eps

    # New epsilon
    new_eps = (theta - mu_new) / sigma_new

    # Jacobian
    log_jacobian = N * (jnp.log(sigma_old) - jnp.log(sigma_new))

    # Prior ratio (N(0,1) on epsilon)
    log_prior_ratio = -0.5 * (jnp.sum(new_eps**2) - jnp.sum(old_eps**2))

    return coupled_indices, new_eps, log_jacobian, log_prior_ratio, key
```

See [Samplers - COUPLED_TRANSFORM](./samplers.md#coupled_transform) for details.

---

### generated_quantities

Compute derived quantities from the chain state (e.g., transformations, predictions).

```python
def generated_quantities(chain_state, data):
    """
    Compute derived quantities.

    Args:
        chain_state: Full parameter vector
        data: Data dict

    Returns:
        1D array of generated quantities
    """
    n_subjects = data['static'][0]
    gq = []

    for i in range(n_subjects):
        # Transform epsilon to theta
        eps = chain_state[i*3:(i+1)*3]
        mu = chain_state[hyper_start]
        sigma = jnp.exp(chain_state[hyper_start + 1])
        theta = mu + sigma * eps
        gq.extend(theta)

    return jnp.array(gq)
```

---

### get_num_gq

Return the number of generated quantities.

```python
def get_num_gq(mcmc_config, data):
    """
    Number of generated quantities.

    Args:
        mcmc_config: MCMC configuration
        data: Data dict

    Returns:
        Integer count of generated quantities
    """
    n_subjects = data['static'][0]
    return n_subjects * 3  # 3 theta values per subject
```

---

### get_special_param_indices

For models with discrete or constrained parameters that need special handling during reset operations.

```python
def get_special_param_indices(n_subjects):
    """
    Indices of parameters needing special reset handling.

    Returns:
        Dict with:
        - 'z_indices': Discrete model indicators
        - 'pi_indices': Simplex parameters (mixing weights)
        - 'r_indices': Parameters on natural scale
    """
    return {
        'z_indices': [i * 13 + 12 for i in range(n_subjects)],  # z indicators
        'pi_indices': [],  # No simplex parameters
        'r_indices': [],   # No natural-scale parameters
    }
```

---

## Complete Example

```python
import jax.numpy as jnp
import numpy as np
from bamcmc import (
    register_posterior, BlockSpec, SamplerType, ProposalType
)


def log_posterior(chain_state, param_indices, data):
    """Log posterior for normal-normal hierarchical model."""
    n_subjects = data['static'][0]
    subject_end = n_subjects * 2  # 2 params per subject

    first_idx = param_indices[0]

    if first_idx < subject_end:
        # Subject block: epsilon prior + likelihood
        subject_idx = first_idx // 2
        eps = chain_state[param_indices]

        # Prior on epsilon
        log_prior = -0.5 * jnp.sum(eps**2)

        # Get hyperparameters
        mu = chain_state[subject_end]
        sigma = jnp.exp(chain_state[subject_end + 1])

        # Transform to theta
        theta = mu + sigma * eps

        # Likelihood
        y = data['float'][0][subject_idx]
        log_lik = -0.5 * jnp.sum((y - theta)**2)

        return log_prior + log_lik
    else:
        # Hyperparameter block: hyperprior only
        mu = chain_state[param_indices[0]]
        logsd = chain_state[param_indices[1]]

        log_prior_mu = -0.5 * mu**2 / 100  # N(0, 100)
        sigma = jnp.exp(logsd)
        log_prior_sigma = -jnp.log(1 + sigma**2) + logsd  # Half-Cauchy(1)

        return log_prior_mu + log_prior_sigma


def batch_type(mcmc_config, data):
    """Define parameter blocks."""
    n_subjects = data['static'][0]

    subject_specs = [
        BlockSpec(size=2, sampler_type=SamplerType.METROPOLIS_HASTINGS,
                  proposal_type=ProposalType.CHAIN_MEAN, label=f"Subj_{i}")
        for i in range(n_subjects)
    ]

    hyper_spec = BlockSpec(
        size=2,
        sampler_type=SamplerType.COUPLED_TRANSFORM,
        proposal_type=ProposalType.MCOV_WEIGHTED_VEC,
        settings={'cov_mult': 1.0, 'cov_beta': -0.9},
        label="Hyperparameters"
    )

    return subject_specs + [hyper_spec]


def initial_vector(mcmc_config, data):
    """Generate initial values."""
    K = mcmc_config['num_superchains']
    n_subjects = data['static'][0]
    n_params = n_subjects * 2 + 2

    rng = np.random.default_rng(mcmc_config['rng_seed'])

    init = np.zeros((K, n_params))
    # Epsilon values: standard normal
    init[:, :n_subjects*2] = rng.normal(size=(K, n_subjects*2))
    # Hyperparameters: mu=0, logsd=0 (sigma=1)
    init[:, n_subjects*2] = 0.0
    init[:, n_subjects*2 + 1] = 0.0

    return init.flatten()


def coupled_transform_dispatch(key, chain_state, primary_indices,
                               proposed_primary, data):
    """Theta-preserving transform for hyperparameters."""
    n_subjects = data['static'][0]

    # Old hyperparameters
    mu_old = chain_state[primary_indices[0]]
    sigma_old = jnp.exp(chain_state[primary_indices[1]])

    # New hyperparameters
    mu_new = proposed_primary[0]
    sigma_new = jnp.exp(proposed_primary[1])

    # Epsilon indices (all subject epsilon values)
    coupled_indices = jnp.arange(n_subjects * 2)
    N = n_subjects * 2

    # Current epsilon
    old_eps = chain_state[coupled_indices]

    # Theta (preserved)
    theta = mu_old + sigma_old * old_eps

    # New epsilon
    new_eps = (theta - mu_new) / sigma_new

    # Jacobian and prior ratio
    log_jacobian = N * (jnp.log(sigma_old) - jnp.log(sigma_new))
    log_prior_ratio = -0.5 * (jnp.sum(new_eps**2) - jnp.sum(old_eps**2))

    return coupled_indices, new_eps, log_jacobian, log_prior_ratio, key


# Register the model
register_posterior('normal_normal_ncp', {
    'log_posterior': log_posterior,
    'batch_type': batch_type,
    'initial_vector': initial_vector,
    'coupled_transform_dispatch': coupled_transform_dispatch,
})
```

---

## Listing Registered Models

```python
from bamcmc import list_posteriors

print(list_posteriors())
# ['normal_normal_ncp', 'other_model', ...]
```

## Getting a Registered Model

```python
from bamcmc.registry import get_posterior

config = get_posterior('normal_normal_ncp')
log_post_fn = config['log_posterior']
batch_fn = config['batch_type']
```
