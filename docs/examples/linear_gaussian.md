# Example: Linear Gaussian (Bayesian Linear Regression)

This walkthrough estimates a Bayesian linear regression model with known noise variance using bamcmc. Because the model is conjugate, we can compare MCMC output against the analytical posterior.

## Model

Likelihood (known variance):

```
y | X, beta, sigma^2  ~  N(X @ beta, sigma^2 * I)
```

Prior:

```
beta  ~  N(0, tau^2 * I)
```

Analytical posterior:

```
beta | y  ~  N(mu_post, Sigma_post)

Sigma_post = (X'X / sigma^2 + I / tau^2)^{-1}
mu_post    = Sigma_post @ (X'y / sigma^2)
```

## Full Code

### 1. Imports

```python
import numpy as np
import jax.numpy as jnp
import jax.scipy.stats as stats
from bamcmc import (
    register_posterior, rmcmc,
    BlockSpec, SamplerType, ProposalType,
)
```

### 2. Generate Synthetic Data

```python
np.random.seed(42)

n_obs = 200
n_predictors = 3
sigma = 1.0       # known noise std dev
tau = 5.0         # prior std dev

beta_true = np.array([1.5, -0.8, 0.3])
X = np.random.randn(n_obs, n_predictors)
y = X @ beta_true + np.random.randn(n_obs) * sigma
```

### 3. Compute the Analytical Posterior (for verification)

```python
sigma_sq = sigma ** 2
tau_sq = tau ** 2

Sigma_post = np.linalg.inv(X.T @ X / sigma_sq + np.eye(n_predictors) / tau_sq)
mu_post = Sigma_post @ (X.T @ y / sigma_sq)

print("Analytical posterior mean:", mu_post)
print("Analytical posterior std: ", np.sqrt(np.diag(Sigma_post)))
```

### 4. Pack the Data

```python
data = {
    "static": (int(n_obs), int(n_predictors), float(sigma), float(tau)),
    "int": (),
    "float": (X, y),
}
```

`static` values must be Python scalars (not numpy scalars) because JAX hashes them for compilation caching. Empty tuples are required for unused fields.

### 5. Define Model Functions

```python
def log_posterior(chain_state, param_indices, data, beta=1.0):
    """Log posterior for the coefficient block."""
    params = chain_state[param_indices]

    n_obs = data["static"][0]
    n_predictors = data["static"][1]
    sigma = data["static"][2]
    tau = data["static"][3]

    X = data["float"][0]   # (n_obs, n_predictors)
    y = data["float"][1]   # (n_obs,)

    # Prior: beta_j ~ N(0, tau^2)
    log_prior = jnp.sum(stats.norm.logpdf(params, loc=0.0, scale=tau))

    # Likelihood: y ~ N(X @ beta, sigma^2 I)
    residuals = y - X @ params
    log_lik = jnp.sum(stats.norm.logpdf(residuals, loc=0.0, scale=sigma))

    return log_prior + beta * log_lik


def batch_specs(mcmc_config, data):
    """One block containing all coefficients."""
    n_predictors = data["static"][1]
    return [
        BlockSpec(
            size=n_predictors,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=ProposalType.CHAIN_MEAN,
            label="beta",
        )
    ]


def initial_vector(mcmc_config, data):
    """Initial coefficient values for all chains."""
    n_predictors = data["static"][1]
    num_chains = mcmc_config["num_chains_a"] + mcmc_config["num_chains_b"]
    K = mcmc_config.get("num_superchains", num_chains)

    rng = np.random.default_rng(mcmc_config.get("rng_seed", 42))
    init = rng.normal(0, 0.5, size=(K, n_predictors))

    if K < num_chains:
        full_init = np.zeros((num_chains, n_predictors))
        full_init[:K] = init
        return full_init.flatten()
    return init.flatten()


def direct_sampler(key, chain_state, param_indices, data):
    """No-op placeholder (required even for MH-only models)."""
    return chain_state, key
```

**Notes on `log_posterior`:**
- The `beta` argument (inverse temperature) multiplies the likelihood. This is required for parallel tempering support; set `beta=1.0` as default.
- Extract parameters via `chain_state[param_indices]` -- never assume fixed positions, since the backend assigns indices per block.
- All operations must be JAX-traceable (use `jnp`, not `np`).

**Notes on `initial_vector`:**
- Must return a flat array of size `num_chains * n_params` (where `num_chains = num_chains_a + num_chains_b`), **not** `num_superchains * n_params`.
- Generate `K = num_superchains` distinct starting points; the backend replicates them across subchains.

### 6. Register and Run

```python
register_posterior("linear_gaussian", {
    "log_posterior": log_posterior,
    "batch_type": batch_specs,
    "initial_vector": initial_vector,
    "direct_sampler": direct_sampler,
})

mcmc_config = {
    "posterior_id": "linear_gaussian",
    "use_double": True,
    "rng_seed": 1977,
    "num_chains_a": 500,
    "num_chains_b": 500,
    "num_superchains": 100,
    "burn_iter": 2000,
    "num_collect": 5000,
    "thin_iteration": 5,
    "benchmark": 100,
    "resume_runs": 3,
}

summary = rmcmc(mcmc_config, data, output_dir="./output_linear_gaussian")
```

`resume_runs` controls how many sequential sampling runs are performed; each run resumes from the previous checkpoint. This keeps per-run memory bounded.

### 7. Check Results

```python
from bamcmc import load_checkpoint, combine_batch_histories

# Load combined history across all runs
history_files = summary["history_files"]
paths = [p for _, p in history_files]
combined = combine_batch_histories(paths)
# combined shape: (total_samples, num_chains, n_params)

# Posterior means across all chains and samples
mcmc_means = combined.mean(axis=(0, 1))

print("True coefficients:     ", beta_true)
print("Analytical posterior:   ", mu_post)
print("MCMC posterior means:   ", mcmc_means)
```

With 3 resume runs of 5000 collected iterations each (thinned by 5), you should see the MCMC means closely match the analytical solution.

## Key Takeaways

1. **Data format** -- `static` for hashable scalars, `int` and `float` for arrays, empty tuples for unused fields.
2. **`log_posterior` signature** -- always includes `beta=1.0` for tempering support; uses `chain_state[param_indices]` to extract the current block's parameters.
3. **`initial_vector` size** -- `(num_chains_a + num_chains_b) * n_params`, with `K` distinct starting points.
4. **`direct_sampler` placeholder** -- required by the registry even when all blocks use MH.
5. **`rmcmc`** is the recommended entry point for production sampling; it handles checkpointing and multi-run scheduling automatically.
