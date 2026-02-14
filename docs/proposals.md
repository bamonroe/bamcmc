# Proposal Types

bamcmc provides 13 proposal distributions for MH sampling, ranging from simple random walks to adaptive gradient-based methods. Every proposal receives the same operand from the backend:

```
(key, current_block, step_mean, step_cov, coupled_blocks, block_mask, settings, grad_fn, block_mode)
```

- `current_block` -- current parameter values for this block
- `step_mean` / `step_cov` -- empirical mean and covariance of the *coupled* chains (Group A sees Group B statistics and vice versa)
- `coupled_blocks` -- raw parameter values from all coupled chains (used by MULTINOMIAL)
- `block_mask` -- binary mask for active parameters within the block
- `settings` -- per-block settings array (7 slots, see [Settings Slots](#settings-slots))
- `grad_fn` -- gradient of log-posterior (only used by MALA / MEAN_MALA)
- `block_mode` -- state of the coupled chain with the highest log-posterior value

Every proposal returns `(proposal, log_hastings_ratio, new_key)`.

## Settings Slots

Settings are stored in a fixed-size array indexed by `SettingSlot`. User-facing keys are lowercase versions of the enum names.

| Slot | Key | Default | Used by |
|------|-----|---------|---------|
| 0 | `chain_prob` | 0.5 | MIXTURE |
| 1 | `n_categories` | 4.0 | MULTINOMIAL |
| 2 | `cov_mult` | 1.0 | All except CHAIN_MEAN and MULTINOMIAL |
| 3 | `uniform_weight` | 0.4 | MULTINOMIAL |
| 4 | `cov_beta` | 1.0 | MCOV_WEIGHTED, MCOV_WEIGHTED_VEC |
| 5 | `k_g` | 10.0 | MCOV_SMOOTH, MCOV_MODE, MCOV_MODE_VEC |
| 6 | `k_alpha` | 3.0 | MCOV_SMOOTH, MCOV_MODE, MCOV_MODE_VEC |

## Quick Reference

| Proposal | Strategy | Target | Distance metric | Covariance scaling | Settings |
|----------|----------|--------|-----------------|-------------------|----------|
| `SELF_MEAN` | Random walk | Current state | -- | Fixed | `cov_mult` |
| `CHAIN_MEAN` | Independent | Population mean | -- | Fixed | -- |
| `MIXTURE` | Blend of above | Mean or current | -- | Fixed | `chain_prob`, `cov_mult` |
| `MULTINOMIAL` | Discrete | Empirical freqs | -- | -- | `n_categories`, `uniform_weight` |
| `MALA` | Gradient RW | Current + drift | -- | Fixed | `cov_mult` |
| `MEAN_MALA` | Gradient indep. | Mean + drift | -- | Fixed | `cov_mult` |
| `MEAN_WEIGHTED` | Adaptive interp. | Population mean | Mahalanobis (scalar) | Fixed | `cov_mult` |
| `MODE_WEIGHTED` | Adaptive interp. | Mode | Mahalanobis (scalar) | Fixed | `cov_mult` |
| `MCOV_WEIGHTED` | Adaptive interp. + cov | Population mean | Mahalanobis (scalar) | Scalar g(d) | `cov_mult`, `cov_beta` |
| `MCOV_WEIGHTED_VEC` | Per-param adaptive | Population mean | Whitened per-param | Per-param g(d) | `cov_mult`, `cov_beta` |
| `MCOV_SMOOTH` | Smooth 3-zone | Population mean | Marginal per-param | Scalar min(g_i) | `cov_mult`, `k_g`, `k_alpha` |
| `MCOV_MODE` | Smooth mode-target | Mode | Mahalanobis (scalar) | Scalar g(d) | `cov_mult`, `k_g`, `k_alpha` |
| `MCOV_MODE_VEC` | Smooth mode-target | Mode | Marginal per-param | Scalar min(g_i) | `cov_mult`, `k_g`, `k_alpha` |

---

## Basic Proposals

### SELF_MEAN (Random Walk)

The simplest proposal -- a symmetric random walk centered on the current state, using the empirical covariance from the coupled chains.

**Proposal distribution:**

```
L = cholesky(cov_mult * Sigma + nugget * I)
x' = x + L @ z,    z ~ N(0, I)
```

**Hastings ratio:** `0` (symmetric -- `q(x'|x) = q(x|x')`)

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.SELF_MEAN,
    settings={'cov_mult': 1.0}
)
```

**Settings:**
- `cov_mult` (default 1.0) -- scales the proposal covariance. Reduce for higher acceptance; increase for larger jumps.

**When to use:** Low-dimensional blocks, or when you want purely local exploration.

---

### CHAIN_MEAN (Independent)

Independent proposal centered on the coupled-chain population mean. The proposal does not depend on the current state, which means it can make large jumps.

**Proposal distribution:**

```
L = cholesky(Sigma + nugget * I)
x' = mu + L @ z,    z ~ N(0, I)
```

**Hastings ratio:**

Because the proposal is independent, the forward and reverse densities differ:

```
log q(x'|x) = -0.5 * ||L^{-1}(x' - mu)||^2   (+ constant)
log q(x|x') = -0.5 * ||L^{-1}(x  - mu)||^2   (+ constant)

log_hastings = 0.5 * (||L^{-1}(x' - mu)||^2 - ||L^{-1}(x - mu)||^2)
```

This penalises proposals that land far from the mean and rewards moving the chain closer to the mean.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.CHAIN_MEAN,
)
```

**Settings:** None (does not use `cov_mult`; covariance is unscaled).

**When to use:** Good default for low-dimensional blocks. Excellent at rescuing stuck chains since it always proposes from the population center.

---

### MIXTURE

Randomly selects between CHAIN_MEAN and SELF_MEAN on each iteration.

**Proposal distribution:**

```
q(x'|x) = p * N(x' | mu, Sigma) + (1-p) * N(x' | x, cov_mult * Sigma)
```

where `p = chain_prob`.

**Hastings ratio:**

The full mixture density is evaluated in both directions using `logaddexp` for numerical stability:

```
log_forward  = logaddexp(log(p) + log N(x'|mu, Sigma),
                         log(1-p) + log N(x'|x, cov_mult * Sigma))
log_backward = logaddexp(log(p) + log N(x|mu, Sigma),
                         log(1-p) + log N(x|x', cov_mult * Sigma))
log_hastings = log_backward - log_forward
```

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MIXTURE,
    settings={'chain_prob': 0.5, 'cov_mult': 1.0}
)
```

**Settings:**
- `chain_prob` (default 0.5) -- probability of using the CHAIN_MEAN component.
- `cov_mult` (default 1.0) -- covariance multiplier applied only to the SELF_MEAN component.

**When to use:** Good general-purpose choice. Combines global mixing (CHAIN_MEAN) with local exploration (SELF_MEAN).

---

### MULTINOMIAL (Discrete)

For discrete parameters taking integer values in `{1, 2, ..., K}`. Each dimension is independently sampled from a mixture of the empirical frequency distribution and a uniform distribution.

**Proposal distribution (per dimension d):**

```
q(x'_d = k) = (1 - w) * f_d(k) + w * (1/K)
```

where `f_d(k)` is the empirical frequency of category `k` across coupled chains for dimension `d`, `w = uniform_weight`, and `K = n_categories`.

**Hastings ratio:**

```
log_hastings = sum_d [ log q(x_d) - log q(x'_d) ]
```

summed over active dimensions (masked by `block_mask`).

```python
BlockSpec(
    size=1,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MULTINOMIAL,
    settings={'n_categories': 3, 'uniform_weight': 0.4}
)
```

**Settings:**
- `n_categories` (default 4) -- number of valid categories. Values must be 1-indexed integers in `[1, K]`.
- `uniform_weight` (default 0.4) -- weight of the uniform component. Higher values prevent the proposal from getting stuck at consensus; lower values exploit the empirical distribution more aggressively.

**When to use:** The only proposal suitable for discrete/categorical parameters.

---

## Gradient-Based Proposals

### MALA (Metropolis-Adjusted Langevin)

Gradient-based proposal that uses the score function to bias moves toward higher density regions, preconditioned by the coupled-chain covariance.

**Proposal distribution:**

```
Sigma_reg = Sigma + nugget * I
drift(x)  = (cov_mult / 2) * Sigma_reg @ grad log p(x)
L         = cholesky(Sigma_reg)
epsilon   = sqrt(cov_mult)

x' = x + drift(x) + epsilon * L @ z,    z ~ N(0, I)
```

Equivalently: `q(x'|x) = N(x' | x + drift(x), cov_mult * Sigma_reg)`

**Hastings ratio:**

Two gradient evaluations are required (at `x` and at `x'`):

```
y_forward = L^{-1}(x' - x - drift(x)) / epsilon
y_reverse = L^{-1}(x  - x' - drift(x')) / epsilon
log_hastings = -0.5 * (||y_reverse||^2 - ||y_forward||^2)
```

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MALA,
    settings={'cov_mult': 0.1}
)
```

**Settings:**
- `cov_mult` (default 1.0) -- controls step size. Smaller values give more accurate drift and higher acceptance rates. Typical values: 0.01--1.0.

**Requirements:** `log_posterior` must be differentiable w.r.t. the block parameters. Do NOT use with discrete parameters.

---

### MEAN_MALA (Chain-Mean MALA)

Evaluates the gradient at the population mean instead of the current state, producing an independent proposal shifted by a gradient-based drift.

**Proposal distribution:**

```
drift  = (cov_mult / 2) * Sigma_reg @ grad log p(mu)
center = mu + drift

q(x'|x) = N(x' | center, cov_mult * Sigma)
```

The proposal center does not depend on `x`, so only one gradient evaluation is needed (at `mu`), shared across all chains.

**Hastings ratio:**

Since the forward and reverse proposals share the same center:

```
log_hastings = 0.5 * (||L^{-1}(x' - center) / epsilon||^2
                     - ||L^{-1}(x  - center) / epsilon||^2)
```

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MEAN_MALA,
    settings={'cov_mult': 0.1}
)
```

**Settings:**
- `cov_mult` (default 1.0) -- controls step size.

**When to use:** When gradients are available but chains may be stuck. Cheaper than MALA (one gradient call vs two) and can rescue outlier chains.

---

## Adaptive Proposals

All adaptive proposals share a common pattern: they compute a *distance* `d` between the current state and a target (population mean or mode), then use `d` to control two quantities:

- **alpha(d)** -- interpolation weight. When alpha=0 the proposal mean equals the target; when alpha=1 it equals the current state.
- **g(d)** -- covariance scale factor. Controls how large the proposal steps are.

The key difference between the proposals in this family is *which* formulas they use for alpha and g, *what* they target (mean vs mode), and whether distance is computed as a scalar or per-parameter.

### Shared Building Blocks

The following helper functions (in `proposals/common.py`) are referenced throughout:

**Linear alpha** (used by MEAN_WEIGHTED, MODE_WEIGHTED, MCOV_WEIGHTED):

```
alpha(d) = d / (d + k)        k = 4 * sqrt(ndim)
```

**Quadratic alpha with coupled g** (used by MCOV_SMOOTH, MCOV_MODE, MCOV_MODE_VEC):

```
g(d) = 1 / (1 + (d / k_g)^2)
d2   = d / sqrt(g)                     # distance in proposal metric
alpha(d) = d2^2 / (d2^2 + k_alpha^2)
```

Here `k_g` controls how fast the covariance shrinks and `k_alpha` controls how fast alpha rises. Because `d2 > d` when `g < 1`, the alpha response is *steeper* than a naive `d^2/(d^2+k^2)` -- chains that are far enough for `g` to kick in are also strongly pulled toward the target.

**Mahalanobis distance:**

```
y = L^{-1}(x - target)
d = sqrt(sum(y^2))
```

**Marginal per-parameter distance:**

```
d_i = |x_i - target_i| / sqrt(diag(Sigma_base)_i)
```

---

### MEAN_WEIGHTED

Interpolates between SELF_MEAN and CHAIN_MEAN based on scalar Mahalanobis distance from the population mean.

**Algorithm:**

```
d = Mahalanobis(x, mu; Sigma_base)      # scalar distance
k = 4 * sqrt(ndim)
alpha = d / (d + k)                      # 0 near mean, 1 far away

mu_prop = alpha * x + (1 - alpha) * mu
x' ~ N(mu_prop, cov_mult * Sigma)
```

**Hastings ratio:**

Both alpha values (at current and proposed state) must be computed:

```
alpha_x at d(x, mu),  alpha_y at d(x', mu)
mu_x = alpha_x * x  + (1 - alpha_x) * mu
mu_y = alpha_y * x' + (1 - alpha_y) * mu

log_hastings = -0.5 * (||L^{-1}(x - mu_y) / epsilon||^2
                      - ||L^{-1}(x' - mu_x) / epsilon||^2)
```

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MEAN_WEIGHTED,
    settings={'cov_mult': 1.0}
)
```

**Settings:**
- `cov_mult` (default 1.0) -- base covariance multiplier.

**When to use:** When chains are recovering from different distances to equilibrium. No covariance scaling, so simpler dynamics than MCOV variants.

---

### MODE_WEIGHTED

Same as MEAN_WEIGHTED but interpolates toward the **mode** (coupled chain with highest log-posterior) instead of the population mean.

**Algorithm:**

```
d = Mahalanobis(x, mode; Sigma_base)
alpha = d / (d + k)

mu_prop = alpha * x + (1 - alpha) * mode
x' ~ N(mu_prop, cov_mult * Sigma)
```

**Hastings ratio:** Same structure as MEAN_WEIGHTED, with `mode` replacing `mu`.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MODE_WEIGHTED,
    settings={'cov_mult': 1.0}
)
```

**Settings:**
- `cov_mult` (default 1.0) -- base covariance multiplier.

**When to use:** Multimodal or skewed posteriors where the mean sits between modes and targeting it would be counterproductive.

---

### MCOV_WEIGHTED

Extends MEAN_WEIGHTED by also **scaling the proposal covariance** based on distance. Both mean interpolation and covariance scaling are driven by the same scalar Mahalanobis distance.

**Algorithm:**

```
d  = Mahalanobis(x, mu; Sigma_base)
k  = 4 * sqrt(ndim)

g  = clamp(1 + beta * d / (d + k), 0.1, 10.0)     # covariance scale
d2 = d / sqrt(g)                                     # distance in scaled metric
alpha = d2 / (d2 + k)                                # mean interpolation

mu_prop = alpha * x + (1 - alpha) * mu
x' ~ N(mu_prop, g * Sigma_base)
```

**How `cov_beta` works:**

| `cov_beta` | Effect when far from mean | Typical use |
|------------|--------------------------|-------------|
| `> 0` | g > 1: larger steps, broader exploration | Multimodal exploration |
| `= 0` | g = 1 always: reduces to MEAN_WEIGHTED | When no cov scaling needed |
| `< 0` | g < 1: smaller steps, cautious approach | Catching up after reset |

The covariance scale `g` is clamped to `[0.1, 10.0]` for numerical safety.

**Hastings ratio (includes log-determinant correction):**

```
log_det = -0.5 * ndim * (log(g_y) - log(g_x))
log_hastings = log_det - 0.5 * (||L^{-1}(x - mu_y)||^2 / g_y
                                - ||L^{-1}(x' - mu_x)||^2 / g_x)
```

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MCOV_WEIGHTED,
    settings={'cov_mult': 1.0, 'cov_beta': 1.0}
)
```

**Settings:**
- `cov_mult` (default 1.0) -- base covariance multiplier (epsilon^2).
- `cov_beta` (default 1.0) -- covariance scaling strength. See table above.

---

### MCOV_WEIGHTED_VEC (Vectorized)

Per-parameter version of MCOV_WEIGHTED. Each parameter gets its own alpha and g based on its individual whitened distance from the population mean.

**Algorithm:**

```
y      = L^{-1}(x - mu)                              # whitened coordinates
d1_i   = |y_i|                                        # per-parameter distance
k      = 2 / sqrt(ndim)

g_i    = clamp(1 + beta * d1_i^2 / (d1_i^2 + k^2), 0.1, 10.0)   # per-param cov scale
d2_i   = d1_i / sqrt(g_i)
alpha_i = d2_i^2 / (d2_i^2 + k^2)                    # per-param mean interpolation

mu_prop_i = alpha_i * x_i + (1 - alpha_i) * mu_i
x' = mu_prop + diag(sqrt(g_vec)) @ L @ z
```

Note the quadratic formulations (`d^2/(d^2+k^2)` instead of `d/(d+k)`) which make alpha and g flat near `d=0`, recovering pure CHAIN_MEAN behavior at equilibrium.

**Hastings ratio:**

```
log_det = -0.5 * (sum(log g_proposal) - sum(log g_current))
log_hastings = log_det - 0.5 * (dist_sq_reverse - dist_sq_forward)
```

where the distances account for element-wise g-scaling.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.COUPLED_TRANSFORM,
    proposal_type=ProposalType.MCOV_WEIGHTED_VEC,
    settings={'cov_mult': 1.0, 'cov_beta': -0.9}
)
```

**Settings:**
- `cov_mult` (default 1.0) -- base step size.
- `cov_beta` (default 1.0) -- per-parameter covariance scaling. Recommended: `-0.9` for NCP hyperparameters (smaller steps when far from mean).

**When to use:** Blocks where parameters have different scales or convergence rates. Recommended for hyperparameters with `COUPLED_TRANSFORM`.

---

### MCOV_SMOOTH

Smooth three-zone proposal with per-parameter distance but scalar covariance scaling. Uses the `k_g` / `k_alpha` parameterisation for direct control over the transition profile.

**Algorithm:**

```
sigma_i = sqrt(diag(Sigma_base)_i)         # marginal std dev per parameter
d_i     = |x_i - mu_i| / sigma_i           # standardised per-parameter distance

# Per-parameter computations:
g_i     = 1 / (1 + (d_i / k_g)^2)          # decay toward 0 as d grows
d2_i    = d_i / sqrt(g_i)
alpha_i = d2_i^2 / (d2_i^2 + k_alpha^2)    # rise toward 1 as d grows

# Scalar covariance scale (cautious -- driven by farthest parameter):
g = min(g_i)    over active parameters

mu_prop_i = alpha_i * x_i + (1 - alpha_i) * mu_i
x' ~ N(mu_prop, g * Sigma_base)
```

**What `k_g` and `k_alpha` control:**

`k_g` sets the *distance scale for covariance shrinkage*. When any parameter is `k_g` standard deviations from the mean, `g` drops to 0.5 and proposal steps shrink to ~71% of normal. Larger `k_g` means the proposal tolerates bigger deviations before shrinking.

`k_alpha` sets the *distance scale for mean interpolation*. When a parameter is `k_alpha` (adjusted) standard deviations from the mean, `alpha` rises to ~0.5 and the proposal center sits halfway between the current state and the population mean. Smaller `k_alpha` means the proposal starts pulling chains toward the mean sooner.

The two parameters give independent control over the two adaptive behaviors:

| Parameter | Controls | Effect of increasing |
|-----------|----------|---------------------|
| `k_g` | When steps shrink | Steps stay full-size to greater distances |
| `k_alpha` | When chains get pulled toward mean | Chains track their current state to greater distances |

**Behavior at different distances** (with defaults `k_g=10`, `k_alpha=3`):

| d (std devs) | alpha | g | Proposal behavior |
|:---:|:---:|:---:|---|
| 0 | 0.00 | 1.00 | Pure CHAIN_MEAN, full-size steps |
| 1 | 0.10 | 0.99 | Mostly CHAIN_MEAN, steps unchanged |
| 3 | 0.50 | 0.92 | Balanced -- halfway between tracking and pulling |
| 10 | 0.92 | 0.50 | Mostly tracks current state, half-size steps |
| 20 | 0.98 | 0.20 | Nearly tracks current state, small cautious steps |

**Natural acceleration:** As a chain approaches equilibrium, alpha decreases (stronger pull toward mean) *and* g increases (larger steps). The chain accelerates as it gets closer. Conversely, a chain far from equilibrium takes small steps centered near its current position, avoiding wild jumps that would be rejected.

**Hastings ratio:**

```
log_det = -0.5 * ndim * (log(g_proposal) - log(g_current))
log_hastings = log_det - 0.5 * (||L^{-1}(x - mu_y)||^2 / g_y
                                - ||L^{-1}(x' - mu_x)||^2 / g_x)
```

```python
BlockSpec(
    size=4,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MCOV_SMOOTH,
    settings={'cov_mult': 1.0, 'k_g': 10.0, 'k_alpha': 3.0}
)
```

**Settings:**
- `cov_mult` (default 1.0) -- base step size multiplier.
- `k_g` (default 10.0) -- distance (in marginal std devs) where g drops to 0.5.
- `k_alpha` (default 3.0) -- distance (adjusted by g) where alpha rises to ~0.5.

**When to use:** Complex recovery scenarios where you want fine-grained control over the transition from exploration to tracking mode.

---

## Mode-Targeting Proposals

The mean-targeting proposals (MEAN_WEIGHTED, MCOV_WEIGHTED, MCOV_SMOOTH) pull chains toward the population mean. In multimodal posteriors the mean can sit between modes in a low-density valley, making it a poor target. The mode-targeting variants replace the mean with the **state of the coupled chain that has the highest log-posterior value**, avoiding this problem.

### MCOV_MODE

Same adaptive structure as MCOV_SMOOTH but with scalar Mahalanobis distance and targeting the mode.

**Algorithm:**

```
d = Mahalanobis(x, mode; Sigma_base)      # scalar distance to mode

g     = 1 / (1 + (d / k_g)^2)
d2    = d / sqrt(g)
alpha = d2^2 / (d2^2 + k_alpha^2)

mu_prop = alpha * x + (1 - alpha) * mode
x' ~ N(mu_prop, g * Sigma_base)
```

**Hastings ratio:** Same structure as MCOV_SMOOTH.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MCOV_MODE,
    settings={'cov_mult': 1.0, 'k_g': 10.0, 'k_alpha': 3.0}
)
```

**Settings:**
- `cov_mult` (default 1.0) -- base step size.
- `k_g` (default 10.0) -- distance where g drops to 0.5.
- `k_alpha` (default 3.0) -- distance where alpha rises to ~0.5.

**When to use:** Multimodal posteriors. The scalar distance produces a uniform alpha across all parameters, which is simpler and can be more stable for correlated parameters.

---

### MCOV_MODE_VEC

Per-parameter version of MCOV_MODE. Each parameter gets its own alpha based on how far it is from the mode in marginal standard deviations, while covariance scaling uses the cautious `min(g_i)` rule.

**Algorithm:**

```
sigma_i = sqrt(diag(Sigma_base)_i)
d_i     = |x_i - mode_i| / sigma_i

g_i     = 1 / (1 + (d_i / k_g)^2)
d2_i    = d_i / sqrt(g_i)
alpha_i = d2_i^2 / (d2_i^2 + k_alpha^2)
g       = min(g_i)

mu_prop_i = alpha_i * x_i + (1 - alpha_i) * mode_i
x' ~ N(mu_prop, g * Sigma_base)
```

**Hastings ratio:** Same structure as MCOV_SMOOTH.

```python
BlockSpec(
    size=4,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MCOV_MODE_VEC,
    settings={'cov_mult': 1.0, 'k_g': 10.0, 'k_alpha': 3.0}
)
```

**Settings:**
- `cov_mult` (default 1.0) -- base step size.
- `k_g` (default 10.0) -- per-parameter distance where g_i drops to 0.5.
- `k_alpha` (default 3.0) -- per-parameter distance where alpha_i rises to ~0.5.

**When to use:** Multimodal posteriors where different parameters converge at different rates toward the mode.

---

## Choosing a Proposal

### By Parameter Type

| Parameter Type | Recommended Proposals |
|----------------|----------------------|
| Standard continuous | MIXTURE, MCOV_WEIGHTED |
| Hyperparameters (NCP) | MCOV_WEIGHTED_VEC with COUPLED_TRANSFORM |
| Discrete indicators | MULTINOMIAL |
| Smooth, unimodal | MALA, MEAN_MALA |
| Multimodal | MCOV_MODE, MODE_WEIGHTED |
| Stuck chains | MEAN_MALA, CHAIN_MEAN |

### By Situation

| Situation | Recommendation |
|-----------|---------------|
| Starting fresh | MIXTURE or MCOV_WEIGHTED |
| Chains recovering from reset | MCOV_SMOOTH or MCOV_WEIGHTED with negative `cov_beta` |
| Low acceptance rates | Reduce `cov_mult` |
| Stuck at consensus | Increase `uniform_weight` for MULTINOMIAL |
| Gradient available | MALA (if smooth) or MEAN_MALA |

### General Guidelines

1. **Start simple**: MIXTURE or CHAIN_MEAN are good defaults.
2. **Use MULTINOMIAL** for discrete parameters.
3. **Use COUPLED_TRANSFORM** with MCOV_WEIGHTED_VEC for NCP hyperparameters.
4. **Tune `cov_mult`** if acceptance rates are too low (<10%) or too high (>70%).
5. **Avoid MALA** for non-differentiable or discrete parameters.
6. **Use `k_g`/`k_alpha`** (MCOV_SMOOTH, MCOV_MODE, MCOV_MODE_VEC) when you want direct control over the transition profile. Increase `k_g` to keep full-size steps at greater distances; decrease `k_alpha` to start pulling chains toward the target sooner.
