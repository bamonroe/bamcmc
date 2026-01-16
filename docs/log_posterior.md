# Log-Posterior Function Requirements

The `log_posterior` function is the core of your model - it computes the log posterior density for parameter blocks.

## Basic Signature

```python
def log_posterior(chain_state, param_indices, data):
    """
    Compute log posterior for a parameter block.

    Args:
        chain_state: Full parameter vector (1D array of all parameters)
        param_indices: Indices of parameters in this block (1D array)
        data: Data dict with 'static', 'int', 'float' fields

    Returns:
        Scalar log posterior value (float)
    """
    # Extract block parameters
    params = chain_state[param_indices]

    # Compute log posterior
    log_prior = compute_prior(params)
    log_lik = compute_likelihood(params, data)

    return log_prior + log_lik
```

## Requirements

### 1. Pure Function

The function must be a pure function (no side effects):

```python
# GOOD - pure function
def log_posterior(chain_state, param_indices, data):
    return compute_log_prob(chain_state, param_indices, data)

# BAD - modifies global state
counter = 0
def log_posterior(chain_state, param_indices, data):
    global counter
    counter += 1  # Side effect!
    return compute_log_prob(chain_state, param_indices, data)
```

### 2. JAX Compatible

The function must work with JAX arrays:

```python
import jax.numpy as jnp

def log_posterior(chain_state, param_indices, data):
    params = chain_state[param_indices]

    # Use jnp, not np
    log_prob = -0.5 * jnp.sum(params**2)  # GOOD
    # log_prob = -0.5 * np.sum(params**2)  # BAD

    return log_prob
```

### 3. Differentiable (for MALA)

If using gradient-based proposals (MALA, MEAN_MALA), the function must be differentiable:

```python
# GOOD - differentiable
def log_posterior(chain_state, param_indices, data):
    x = chain_state[param_indices]
    return -0.5 * jnp.sum(x**2)

# BAD - non-differentiable control flow
def log_posterior(chain_state, param_indices, data):
    x = chain_state[param_indices]
    if x[0] > 0:  # Non-differentiable!
        return -x[0]
    else:
        return x[0]

# GOOD - differentiable alternative
def log_posterior(chain_state, param_indices, data):
    x = chain_state[param_indices]
    return jnp.where(x[0] > 0, -x[0], x[0])  # jnp.where is differentiable
```

### 4. Handle All Block Types

The same `log_posterior` function handles all blocks. Use the indices to determine which type:

```python
def log_posterior(chain_state, param_indices, data):
    first_idx = param_indices[0]
    n_subjects = data['static'][0]

    # Determine block type from first index
    subject_end = n_subjects * params_per_subject

    if first_idx < subject_end:
        # Subject block
        return subject_log_posterior(chain_state, param_indices, data)
    else:
        # Hyperparameter block
        return hyper_log_posterior(chain_state, param_indices, data)
```

### 5. Return Scalar

Always return a single scalar value:

```python
# GOOD
def log_posterior(chain_state, param_indices, data):
    return jnp.sum(log_probs)  # Scalar

# BAD - returns array
def log_posterior(chain_state, param_indices, data):
    return log_probs  # Array, not scalar!
```

---

## Common Patterns

### Subject-Level Posterior

For hierarchical models, subject-level blocks typically include:
- Prior on epsilon (NCP) or theta (CP)
- Likelihood of subject's data

```python
def subject_log_posterior(chain_state, param_indices, data):
    """Log posterior for one subject's parameters."""
    # Get subject index
    params_per_subject = 3
    subject_idx = param_indices[0] // params_per_subject

    # Extract parameters
    eps_r, eps_alpha, eps_beta = chain_state[param_indices]

    # Get hyperparameters
    hyper_start = data['static'][0] * params_per_subject
    mu_r = chain_state[hyper_start]
    sigma_r = jnp.exp(chain_state[hyper_start + 1])
    # ... etc

    # Transform epsilon to theta (NCP)
    r = mu_r + sigma_r * eps_r
    # ... etc

    # Prior on epsilon (N(0,1))
    log_prior = -0.5 * (eps_r**2 + eps_alpha**2 + eps_beta**2)

    # Likelihood
    log_lik = compute_likelihood(r, alpha, beta, data, subject_idx)

    return log_prior + log_lik
```

### Hyperparameter Posterior

For hyperparameter blocks:

```python
def hyper_log_posterior(chain_state, param_indices, data):
    """Log hyperprior for hyperparameters."""
    mu = chain_state[param_indices[0]]
    logsd = chain_state[param_indices[1]]

    # Hyperprior on mu: N(mu_0, tau_0²)
    mu_0, tau_0_sq = 0.0, 10.0
    log_prior_mu = -0.5 * (mu - mu_0)**2 / tau_0_sq

    # Hyperprior on sigma: Half-Cauchy or Inverse-Gamma
    sigma = jnp.exp(logsd)
    scale = 2.5
    log_prior_sigma = -jnp.log(1 + (sigma/scale)**2) + logsd  # Half-Cauchy with Jacobian

    return log_prior_mu + log_prior_sigma
```

### COUPLED_TRANSFORM Blocks

For theta-preserving hyperparameter updates, the log_posterior only evaluates the **hyperprior** (likelihood cancels):

```python
def hyper_log_posterior(chain_state, param_indices, data):
    """
    Log hyperprior for COUPLED_TRANSFORM blocks.

    Note: This function is called for the PRIMARY parameters only.
    The epsilon prior ratio is computed separately by coupled_transform_dispatch.
    """
    mu = chain_state[param_indices[0]]
    logsd = chain_state[param_indices[1]]

    # Return ONLY the hyperprior (not epsilon prior, not likelihood)
    return hyperprior_mean_logsd(mu, logsd)
```

---

## Debugging

### Check for NaN/Inf

The sampler handles NaN/Inf values, but they indicate problems:

```python
def log_posterior(chain_state, param_indices, data):
    result = compute_log_prob(...)

    # Debug: check for numerical issues
    # (remove in production)
    if not jnp.isfinite(result):
        print(f"Warning: log_posterior = {result}")
        print(f"  params = {chain_state[param_indices]}")

    return result
```

### Test Individually

Test your log_posterior function before running MCMC:

```python
import jax.numpy as jnp

# Create test state
n_params = 100
chain_state = jnp.zeros(n_params)

# Test subject block
subject_indices = jnp.array([0, 1, 2])
result = log_posterior(chain_state, subject_indices, data)
print(f"Subject log_posterior: {result}")
assert jnp.isfinite(result), "Got non-finite result!"

# Test hyperparameter block
hyper_indices = jnp.array([90, 91])
result = log_posterior(chain_state, hyper_indices, data)
print(f"Hyper log_posterior: {result}")
assert jnp.isfinite(result), "Got non-finite result!"
```

### Test Gradients

For MALA proposals, verify gradients work:

```python
import jax

def test_gradients():
    chain_state = jnp.zeros(n_params)
    param_indices = jnp.array([0, 1, 2])

    # Define gradient function
    def lp(params):
        state = chain_state.at[param_indices].set(params)
        return log_posterior(state, param_indices, data)

    # Compute gradient
    grad_fn = jax.grad(lp)
    grads = grad_fn(chain_state[param_indices])

    print(f"Gradients: {grads}")
    assert jnp.all(jnp.isfinite(grads)), "Non-finite gradients!"

test_gradients()
```

---

## Performance Tips

### 1. Vectorize Likelihood Computation

```python
# SLOW - loop over observations
def compute_likelihood_slow(params, data):
    log_lik = 0.0
    for i in range(len(data['float'][0])):
        log_lik += single_obs_likelihood(params, data, i)
    return log_lik

# FAST - vectorized
def compute_likelihood_fast(params, data):
    obs = data['float'][0]
    log_liks = vmap_likelihood(params, obs)  # Use jax.vmap
    return jnp.sum(log_liks)
```

### 2. Pre-compute Constants

```python
# Put constants in data['static'] to avoid recomputation
data = {
    'static': (n_subjects, prior_mean, prior_var),
    'float': (observations,),
    'int': (),
}
```

### 3. Use jax.lax.cond for Branching

```python
# GOOD - JAX-compatible branching
def log_posterior(chain_state, param_indices, data):
    first_idx = param_indices[0]

    return jax.lax.cond(
        first_idx < subject_end,
        lambda _: subject_log_posterior(chain_state, param_indices, data),
        lambda _: hyper_log_posterior(chain_state, param_indices, data),
        operand=None
    )
```

---

## Common Mistakes

### 1. Using NumPy Instead of JAX

```python
# WRONG
import numpy as np
return np.sum(x**2)

# RIGHT
import jax.numpy as jnp
return jnp.sum(x**2)
```

### 2. Non-static Shapes

```python
# WRONG - dynamic shape based on value
mask = x > 0
y = x[mask]  # Shape depends on data!

# RIGHT - use where for static shapes
y = jnp.where(x > 0, x, 0.0)
```

### 3. Python Control Flow

```python
# WRONG - Python if
if x[0] > 0:
    return -x[0]

# RIGHT - JAX control flow
return jnp.where(x[0] > 0, -x[0], x[0])
```

### 4. Forgetting Jacobian for Transformations

```python
# WRONG - missing Jacobian for log transform
sigma = jnp.exp(logsd)
log_prior = -sigma  # Half-exponential prior on sigma

# RIGHT - include Jacobian
sigma = jnp.exp(logsd)
log_prior = -sigma + logsd  # + logsd is the Jacobian
```
