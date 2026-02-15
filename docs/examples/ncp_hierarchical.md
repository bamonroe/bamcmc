# Example: NCP Hierarchical Model with COUPLED_TRANSFORM

This walkthrough estimates a hierarchical normal model using the Non-Centered Parameterization (NCP) with bamcmc's `COUPLED_TRANSFORM` sampler. The theta-preserving update is a distinctive feature of bamcmc: when proposing new hyperparameters, epsilon values are deterministically adjusted so that the derived quantities (theta) stay constant, causing the likelihood to cancel from the acceptance ratio.

## Model

### Hierarchical Normal (NCP)

Each subject *i* has *n_obs* observations drawn from a subject-specific mean:

```
eps_ij  ~ N(0, 1)                 # latent (one per obs per subject)
theta_ij = mu + sigma * eps_ij    # derived
y_ij    ~ N(theta_ij, sigma_obs^2)  # data (known observation noise)
```

Hyperpriors:

```
mu     ~ N(0, 10)
sigma  ~ Half-Cauchy(1)           # sampled on log scale (log_sigma)
```

### Why NCP?

In the standard (centered) parameterization, `theta` is sampled directly and `mu`, `sigma` appear in its prior. This creates a *funnel*: when `sigma` is small, `theta` values must cluster tightly around `mu`, making the posterior geometry difficult for standard MH. NCP decorrelates `epsilon` from the hyperparameters by construction.

### The theta-preserving idea

Even with NCP, a naive update of `(mu, sigma)` changes all `theta = mu + sigma * eps` values, so the likelihood fights the proposal and acceptance rates collapse. The theta-preserving trick avoids this:

1. Propose new hyperparameters `(mu', sigma')` using an adaptive proposal
2. Set `eps' = (theta - mu') / sigma'` so that `theta' = mu' + sigma' * eps' = theta` (unchanged)
3. Because theta is preserved, the likelihood terms cancel in the MH ratio

The acceptance ratio reduces to:

```
log alpha = log_proposal_ratio              # from adaptive proposal
          + N * (log sigma - log sigma')    # Jacobian of the transform
          + -0.5 * (sum(eps'^2) - sum(eps^2))  # epsilon prior ratio
          + log p(mu', sigma') - log p(mu, sigma)  # hyperprior ratio
```

No likelihood evaluation is needed for the hyperparameter update. In practice this gives ~40-50% acceptance rates vs ~1-2% with naive updates.

## Full Code

### 1. Imports

```python
import numpy as np
import jax.numpy as jnp
from bamcmc import (
    register_posterior, rmcmc,
    BlockSpec, SamplerType, ProposalType,
)
```

### 2. Generate Synthetic Data

```python
np.random.seed(42)

n_subjects = 8
n_obs = 20          # observations per subject
sigma_obs = 1.0     # known observation noise

mu_true = 3.0
sigma_true = 2.0

# Subject-level means
theta_true = np.random.randn(n_subjects) * sigma_true + mu_true

# Observations: Y[i, j] ~ N(theta_true[i], sigma_obs^2)
Y = np.zeros((n_subjects, n_obs))
for i in range(n_subjects):
    Y[i] = np.random.randn(n_obs) * sigma_obs + theta_true[i]
```

### 3. Pack the Data

```python
data = {
    "static": (int(n_subjects), int(n_obs), float(sigma_obs)),
    "int": (),
    "float": (Y,),
}
```

`static` values must be Python scalars (not numpy scalars) because JAX hashes them for compilation caching. Empty tuples are required for unused fields.

### 4. Define Model Functions

#### log_posterior

A single function handles both subject blocks and the hyperparameter block. The backend calls it once per block, passing the relevant `param_indices`.

```python
def log_posterior(chain_state, param_indices, data, beta=1.0):
    """Log posterior for subject blocks and hyperparameter block."""
    n_subjects = data["static"][0]
    n_obs = data["static"][1]
    sigma_obs = data["static"][2]
    subject_end = n_subjects  # one epsilon per subject

    first_idx = param_indices[0]

    if first_idx < subject_end:
        # --- Subject block: one epsilon parameter ---
        eps = chain_state[param_indices]  # shape (1,)

        # Prior: eps ~ N(0, 1)
        log_prior = -0.5 * jnp.sum(eps ** 2)

        # Get hyperparameters from state
        mu = chain_state[subject_end]
        sigma = jnp.exp(chain_state[subject_end + 1])

        # Derived theta
        theta = mu + sigma * eps  # shape (1,)

        # Likelihood: y_ij ~ N(theta_i, sigma_obs^2) for j in 1..n_obs
        subject_idx = first_idx
        y_i = data["float"][0][subject_idx]  # shape (n_obs,)
        log_lik = -0.5 * jnp.sum((y_i - theta) ** 2) / (sigma_obs ** 2)

        return log_prior + beta * log_lik
    else:
        # --- Hyperparameter block: hyperprior only ---
        mu = chain_state[param_indices[0]]
        logsd = chain_state[param_indices[1]]

        # mu ~ N(0, 10)
        log_prior_mu = -0.5 * mu ** 2 / 100.0

        # sigma ~ Half-Cauchy(1), sampled as log_sigma
        # p(sigma) = 2 / (pi * (1 + sigma^2)), Jacobian |d sigma / d logsd| = sigma
        sigma = jnp.exp(logsd)
        log_prior_sigma = -jnp.log(1.0 + sigma ** 2) + logsd

        return log_prior_mu + log_prior_sigma
```

**Notes:**
- `beta` (inverse temperature) multiplies the likelihood only. It is required for parallel tempering support; defaults to `1.0`.
- The hyperparameter block returns only the hyperprior. The likelihood is not needed here because the `COUPLED_TRANSFORM` sampler handles it through the theta-preserving mechanism.
- `sigma` is stored on the log scale (`logsd`) to allow unconstrained sampling.

#### batch_type

```python
def batch_type(mcmc_config, data):
    """One block per subject (epsilon) + one COUPLED_TRANSFORM block (mu, log_sigma)."""
    n_subjects = data["static"][0]

    subject_specs = [
        BlockSpec(
            size=1,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=ProposalType.CHAIN_MEAN,
            label=f"Subj_{i}",
        )
        for i in range(n_subjects)
    ]

    hyper_spec = BlockSpec(
        size=2,
        sampler_type=SamplerType.COUPLED_TRANSFORM,
        proposal_type=ProposalType.MCOV_WEIGHTED_VEC,
        settings={"cov_mult": 1.0, "cov_beta": -0.9},
        label="Hyperparameters",
    )

    return subject_specs + [hyper_spec]
```

The hyperparameter block uses `COUPLED_TRANSFORM` with the `MCOV_WEIGHTED_VEC` proposal, which provides adaptive covariance scaling per parameter. The `cov_beta` setting controls how aggressively the proposal interpolates toward the coupled mean.

#### initial_vector

```python
def initial_vector(mcmc_config, data):
    """Initial parameter values for all chains."""
    n_subjects = data["static"][0]
    n_params = n_subjects + 2  # n_subjects epsilons + mu + log_sigma

    num_chains = mcmc_config["num_chains_a"] + mcmc_config["num_chains_b"]
    K = mcmc_config.get("num_superchains", num_chains)

    rng = np.random.default_rng(mcmc_config.get("rng_seed", 42))

    init = np.zeros((K, n_params))
    init[:, :n_subjects] = rng.normal(size=(K, n_subjects))  # epsilon ~ N(0,1)
    init[:, n_subjects] = 0.0      # mu = 0
    init[:, n_subjects + 1] = 0.0  # log_sigma = 0 (sigma = 1)

    if K < num_chains:
        full_init = np.zeros((num_chains, n_params))
        full_init[:K] = init
        return full_init.flatten()
    return init.flatten()
```

Must return a flat array of size `num_chains * n_params` (where `num_chains = num_chains_a + num_chains_b`). Generate `K = num_superchains` distinct starting points; the backend replicates them across subchains.

#### coupled_transform_dispatch

This is the function that makes `COUPLED_TRANSFORM` work. It is called by the sampler after proposing new hyperparameters, and returns the deterministically transformed epsilon values that keep theta constant.

```python
def coupled_transform_dispatch(key, chain_state, primary_indices,
                                proposed_primary, data):
    """
    Theta-preserving transform for hyperparameters.

    Given proposed (mu', log_sigma'), compute new epsilon values so that
    theta = mu + sigma * eps = mu' + sigma' * eps' is preserved.

    Returns:
        coupled_indices: Indices of epsilon parameters to update
        new_eps:         Transformed epsilon values
        log_jacobian:    log |det J| = N * (log sigma_old - log sigma_new)
        log_prior_ratio: log p(eps') - log p(eps) for N(0,1) prior
        key:             Updated random key
    """
    n_subjects = data["static"][0]

    # Old hyperparameters (from current state)
    mu_old = chain_state[primary_indices[0]]
    sigma_old = jnp.exp(chain_state[primary_indices[1]])

    # New hyperparameters (proposed)
    mu_new = proposed_primary[0]
    sigma_new = jnp.exp(proposed_primary[1])

    # Epsilon indices: the first n_subjects entries in the state vector
    coupled_indices = jnp.arange(n_subjects)
    N = n_subjects

    # Current epsilon values
    old_eps = chain_state[coupled_indices]

    # Theta (the quantity we preserve)
    theta = mu_old + sigma_old * old_eps

    # New epsilon values that keep theta constant
    new_eps = (theta - mu_new) / sigma_new

    # Jacobian: |d eps' / d sigma'| accumulated over N epsilon values
    # |J| = (sigma_old / sigma_new)^N
    log_jacobian = N * (jnp.log(sigma_old) - jnp.log(sigma_new))

    # Prior ratio for eps ~ N(0, 1)
    log_prior_ratio = -0.5 * (jnp.sum(new_eps ** 2) - jnp.sum(old_eps ** 2))

    return coupled_indices, new_eps, log_jacobian, log_prior_ratio, key
```

### 5. Register and Run

```python
register_posterior("ncp_hierarchical", {
    "log_posterior": log_posterior,
    "batch_type": batch_type,
    "initial_vector": initial_vector,
    "coupled_transform_dispatch": coupled_transform_dispatch,
})

mcmc_config = {
    "posterior_id": "ncp_hierarchical",
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

summary = rmcmc(mcmc_config, data, output_dir="./output_ncp_hierarchical")
```

Note: no `direct_sampler` is needed. Unlike MH-only models (which require a no-op placeholder), models with `COUPLED_TRANSFORM` blocks and no `DIRECT_CONJUGATE` blocks can omit it.

### 6. Check Results

```python
from bamcmc import load_checkpoint, combine_batch_histories

# Load combined history across all runs
history_files = summary["history_files"]
paths = [p for _, p in history_files]
combined = combine_batch_histories(paths)
# combined shape: (total_samples, num_chains, n_params)

n_params = n_subjects + 2

# Posterior means across all chains and samples
mcmc_means = combined.mean(axis=(0, 1))

# Hyperparameters
mu_mcmc = mcmc_means[n_subjects]
sigma_mcmc = np.exp(mcmc_means[n_subjects + 1])

print(f"True mu:     {mu_true:.3f}")
print(f"MCMC mu:     {mu_mcmc:.3f}")
print()
print(f"True sigma:  {sigma_true:.3f}")
print(f"MCMC sigma:  {sigma_mcmc:.3f}")
print()

# Recovered subject-level theta values
eps_mcmc = mcmc_means[:n_subjects]
theta_mcmc = mu_mcmc + sigma_mcmc * eps_mcmc

print("Subject  theta_true  theta_mcmc")
for i in range(n_subjects):
    print(f"  {i:4d}    {theta_true[i]:8.3f}    {theta_mcmc[i]:8.3f}")
```

With 3 resume runs of 5000 collected iterations each (thinned by 5), you should see the MCMC posterior means for `mu` and `sigma` close to their true values, and the recovered subject-level `theta` values tracking the truth.

## Key Takeaways

1. **`COUPLED_TRANSFORM` vs standard MH** -- The hyperparameter block uses `SamplerType.COUPLED_TRANSFORM` instead of `METROPOLIS_HASTINGS`. This tells the backend to call `coupled_transform_dispatch` after proposing hyperparameters, rather than evaluating the full likelihood.
2. **`coupled_transform_dispatch`** -- Returns five values: the indices of coupled parameters (epsilon), their new values, the log Jacobian, the log prior ratio, and the updated key. The backend combines these with the hyperprior ratio and proposal ratio to form the acceptance probability.
3. **Likelihood cancellation** -- Because `theta = mu + sigma * eps` is held constant, the data likelihood `p(y | theta)` is the same for the current and proposed states, so it drops out of the MH ratio entirely.
4. **No `direct_sampler` needed** -- Models that use only MH and COUPLED_TRANSFORM blocks do not need a `direct_sampler` placeholder.
5. **Log-scale sigma** -- Sampling `log(sigma)` instead of `sigma` directly allows unconstrained proposals. The Half-Cauchy prior includes the Jacobian term (`+ logsd`).
