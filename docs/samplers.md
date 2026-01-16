# Sampler Types

bamcmc provides three sampler types for different parameter update strategies.

## Overview

| Sampler Type | Use Case | Requires |
|--------------|----------|----------|
| `METROPOLIS_HASTINGS` | Standard parameters | `proposal_type` |
| `DIRECT_CONJUGATE` | Conjugate/Gibbs sampling | `direct_sampler_fn` |
| `COUPLED_TRANSFORM` | Theta-preserving NCP | `coupled_transform_dispatch` |

---

## METROPOLIS_HASTINGS

The standard Metropolis-Hastings sampler with configurable proposal distributions.

### Usage

```python
from bamcmc import BlockSpec, SamplerType, ProposalType

BlockSpec(
    size=3,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.CHAIN_MEAN,
    settings={'cov_mult': 1.0},
    label="Subject_params"
)
```

### Acceptance Ratio

For a proposed move from state `x` to `x'`:

```
log α = log p(x') - log p(x) + log q(x|x') - log q(x'|x)
```

Where:
- `p(·)` is the (tempered) posterior
- `q(·|·)` is the proposal distribution

### Settings

Settings depend on the chosen `proposal_type`. Common settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `cov_mult` | Covariance multiplier | 1.0 |
| `chain_prob` | Probability of chain_mean in MIXTURE | 0.5 |

See [Proposal Types](./proposals.md) for proposal-specific settings.

### Mixed Proposals

A single MH block can use different proposals for different parameter sub-groups:

```python
from bamcmc import BlockSpec, ProposalGroup, ProposalType

# 12 continuous params + 1 discrete indicator
BlockSpec(
    size=13,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_groups=[
        ProposalGroup(start=0, end=12, proposal_type=ProposalType.MCOV_MODE,
                      settings={'cov_mult': 1.0}),
        ProposalGroup(start=12, end=13, proposal_type=ProposalType.MULTINOMIAL,
                      settings={'uniform_weight': 0.4, 'n_categories': 2}),
    ],
    label="Subject_0"
)
```

Requirements for mixed proposals:
- Groups must be contiguous
- Groups must cover the entire block (start at 0, end at block size)
- Maximum 4 groups per block

---

## DIRECT_CONJUGATE

For parameters with conjugate priors where direct sampling from the conditional distribution is possible (Gibbs sampling).

### Usage

```python
def my_direct_sampler(key, chain_state, param_indices, data):
    """
    Sample directly from the conditional distribution.

    Args:
        key: JAX random key
        chain_state: Full parameter vector
        param_indices: Indices of parameters to update
        data: Data dict

    Returns:
        new_state: Updated chain state
        new_key: Consumed random key
    """
    # Extract conditioning variables
    other_params = chain_state[~param_indices]

    # Sample from conditional
    new_key, sample_key = jax.random.split(key)
    new_values = sample_conditional(sample_key, other_params, data)

    # Update state
    new_state = chain_state.at[param_indices].set(new_values)
    return new_state, new_key

BlockSpec(
    size=1,
    sampler_type=SamplerType.DIRECT_CONJUGATE,
    direct_sampler_fn=my_direct_sampler,
    label="Variance_hyperprior"
)
```

### Requirements

1. **direct_sampler_fn**: Required. Must have signature:
   ```python
   (key, chain_state, param_indices, data) -> (new_state, new_key)
   ```

2. The function must:
   - Handle NaN/Inf gracefully (the framework will replace bad values)
   - Return the full chain_state with updated values
   - Consume the random key and return a new one

### Examples

**Beta-Bernoulli conjugate:**

```python
def beta_bernoulli_direct(key, chain_state, param_indices, data):
    """Sample p from Beta posterior given Bernoulli data."""
    successes = data['int'][0].sum()
    failures = data['int'][0].size - successes
    alpha_prior, beta_prior = data['static'][1], data['static'][2]

    new_key, sample_key = jax.random.split(key)
    p = jax.random.beta(sample_key,
                        alpha_prior + successes,
                        beta_prior + failures)

    new_state = chain_state.at[param_indices].set(p)
    return new_state, new_key
```

---

## COUPLED_TRANSFORM

For Non-Centered Parameterization (NCP) models where hyperparameter updates should preserve derived quantities (theta-preserving updates).

### Background

In NCP hierarchical models:
```
ε ~ N(0, 1)          # Latent (epsilon)
θ = μ + σ × ε        # Derived (theta)
y ~ L(θ)             # Data likelihood
```

When updating hyperparameters (μ, σ), a naive MH update would change both hyperparameters AND θ values, causing the likelihood to contribute to the acceptance ratio. This often leads to very low acceptance rates.

**Theta-preserving updates** instead:
1. Propose new (μ', σ')
2. Compute ε' = (θ - μ') / σ' to keep θ constant
3. Accept/reject based only on priors (likelihood cancels!)

### Usage

```python
# In batch_type function:
BlockSpec(
    size=2,  # (mean, logsd)
    sampler_type=SamplerType.COUPLED_TRANSFORM,
    proposal_type=ProposalType.MCOV_WEIGHTED_VEC,
    settings={'cov_mult': 1.0, 'cov_beta': -0.9},
    label="Hyper_r"
)

# Register posterior with coupled_transform_dispatch:
register_posterior('my_ncp_model', {
    'log_posterior': log_posterior_fn,
    'batch_type': batch_specs_fn,
    'initial_vector': initial_vector_fn,
    'coupled_transform_dispatch': coupled_transform_dispatch_fn,
})
```

### coupled_transform_dispatch Function

The `coupled_transform_dispatch` function handles all COUPLED_TRANSFORM blocks:

```python
def coupled_transform_dispatch(key, chain_state, primary_indices, proposed_primary, data):
    """
    Transform coupled parameters to preserve derived quantities.

    Args:
        key: JAX random key
        chain_state: Full parameter vector
        primary_indices: Indices of hyperparameters being proposed
        proposed_primary: Proposed hyperparameter values
        data: Data dict

    Returns:
        coupled_indices: Indices of epsilon parameters to transform
        new_coupled: Transformed epsilon values
        log_jacobian: Log |det(Jacobian)| = N × (log σ_old - log σ_new)
        log_prior_ratio: log p(ε') - log p(ε) = -0.5 × (Σε'² - Σε²)
        key: Updated random key
    """
    # Identify which hyperparameter block this is
    first_idx = primary_indices[0]

    # Get old and new hyperparameters
    mu_old = chain_state[first_idx]
    logsd_old = chain_state[first_idx + 1]
    sigma_old = jnp.exp(logsd_old)

    mu_new = proposed_primary[0]
    logsd_new = proposed_primary[1]
    sigma_new = jnp.exp(logsd_new)

    # Get epsilon indices (depends on model structure)
    coupled_indices = get_epsilon_indices_for_hyper(first_idx)
    N = len(coupled_indices)

    # Current epsilon values
    old_eps = chain_state[coupled_indices]

    # Theta values (preserved)
    theta = mu_old + sigma_old * old_eps

    # New epsilon values (theta-preserving)
    new_eps = (theta - mu_new) / sigma_new

    # Jacobian: N copies of ∂ε'/∂σ' = -θ/σ'² = -(old_sigma * old_eps)/sigma_new²
    # But using the simplified form: |J| = (σ_old/σ_new)^N
    log_jacobian = N * (jnp.log(sigma_old) - jnp.log(sigma_new))

    # Prior ratio for N(0,1) on epsilon
    log_prior_ratio = -0.5 * (jnp.sum(new_eps**2) - jnp.sum(old_eps**2))

    return coupled_indices, new_eps, log_jacobian, log_prior_ratio, key
```

### Acceptance Ratio

For COUPLED_TRANSFORM, the acceptance ratio is:

```
log α = log_proposal_ratio           # From adaptive proposal
      + log_jacobian                  # N × (log σ - log σ')
      + log_prior_ratio              # -0.5 × (Σε'² - Σε²)
      + log p(μ', σ') - log p(μ, σ)  # Hyperprior ratio
```

Note: The **likelihood cancels** because θ is preserved.

### Benefits

- **Higher acceptance rates**: ~40-50% vs ~1-2% with naive NCP updates
- **Better mixing**: Hyperparameters can move freely without fighting likelihood
- **Mathematically correct**: Detailed balance maintained

---

## Choosing a Sampler

| Scenario | Recommended Sampler |
|----------|-------------------|
| Standard continuous parameters | METROPOLIS_HASTINGS |
| Discrete parameters | METROPOLIS_HASTINGS with MULTINOMIAL |
| Conjugate distributions | DIRECT_CONJUGATE |
| NCP hyperparameters | COUPLED_TRANSFORM |
| Mixed continuous + discrete | METROPOLIS_HASTINGS with mixed proposals |
