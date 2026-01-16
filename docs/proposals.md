# Proposal Types

bamcmc provides 13 proposal distributions for MH sampling, ranging from simple random walks to adaptive gradient-based methods.

## Quick Reference

| Proposal Type | Description | Use Case | Settings |
|---------------|-------------|----------|----------|
| `SELF_MEAN` | Random walk | Simple, low-dimensional | `cov_mult` |
| `CHAIN_MEAN` | Independent | Well-mixing chains | none |
| `MIXTURE` | Blend of above | General purpose | `chain_prob`, `cov_mult` |
| `MULTINOMIAL` | Discrete | Categorical parameters | `uniform_weight`, `n_categories` |
| `MALA` | Gradient-based | Smooth posteriors | `cov_mult` |
| `MEAN_MALA` | Population gradient | Stuck chains with gradients | `cov_mult` |
| `MEAN_WEIGHTED` | Adaptive mean | Recovering chains | `cov_mult`, `cov_beta` |
| `MODE_WEIGHTED` | Target highest | Multimodal | `cov_mult`, `cov_beta` |
| `MCOV_WEIGHTED` | Adaptive mean+cov | General purpose | `cov_mult`, `cov_beta` |
| `MCOV_WEIGHTED_VEC` | Per-param adaptive | Hyperparameters | `cov_mult`, `cov_beta` |
| `MCOV_SMOOTH` | Smooth 3-zone | Complex recovery | `cov_mult`, `k_g`, `k_alpha` |
| `MCOV_MODE` | Mode-targeting | Multimodal (scalar) | `cov_mult`, `cov_beta` |
| `MCOV_MODE_VEC` | Mode-targeting | Multimodal (per-param) | `cov_mult`, `cov_beta` |

---

## Basic Proposals

### SELF_MEAN (Random Walk)

**Proposal:** `x' ~ N(x, cov_mult × Σ)`

The simplest proposal - a random walk centered on the current state with covariance from the coupled chains.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.SELF_MEAN,
    settings={'cov_mult': 1.0}
)
```

**Properties:**
- Symmetric: Hastings ratio = 0
- Local exploration only
- Good for low-dimensional, smooth posteriors

**Settings:**
- `cov_mult` (default: 1.0): Scales the proposal covariance

---

### CHAIN_MEAN (Independent)

**Proposal:** `x' ~ N(μ, Σ)`

Independent proposal centered on the population mean. Proposes the same regardless of current state.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.CHAIN_MEAN
)
```

**Properties:**
- Asymmetric: Hastings ratio accounts for distance to mean
- Good global mixing when chains are well-behaved
- Can rescue stuck chains (proposes from population center)

**Settings:** None

---

### MIXTURE

**Proposal:** With probability `p` use CHAIN_MEAN, else use SELF_MEAN

Combines the global mixing of CHAIN_MEAN with the local exploration of SELF_MEAN.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MIXTURE,
    settings={'chain_prob': 0.5, 'cov_mult': 1.0}
)
```

**Properties:**
- Flexible tradeoff between local and global moves
- Good general-purpose choice

**Settings:**
- `chain_prob` (default: 0.5): Probability of using CHAIN_MEAN
- `cov_mult` (default: 1.0): Covariance multiplier for SELF_MEAN component

---

### MULTINOMIAL (Discrete)

**Proposal:** Sample from mixture of empirical and uniform distributions

For discrete parameters on a grid `{1, 2, ..., n_categories}`.

```python
BlockSpec(
    size=1,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MULTINOMIAL,
    settings={'n_categories': 3, 'uniform_weight': 0.4}
)
```

**Properties:**
- Independent: proposes based on population frequencies
- Uniform weight prevents getting stuck at consensus

**Settings:**
- `n_categories` (default: 4): Number of valid categories (1 to n)
- `uniform_weight` (default: 0.4): Weight of uniform distribution in mixture

**Note:** Discrete parameters should use integer values `1, 2, ..., n_categories` (1-indexed).

---

## Gradient-Based Proposals

### MALA (Metropolis-Adjusted Langevin)

**Proposal:** `x' = x + (cov_mult/2) × Σ × ∇log p(x) + √cov_mult × L × z`

Uses the gradient to bias proposals toward higher density regions.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MALA,
    settings={'cov_mult': 0.1}
)
```

**Properties:**
- Gradient-informed: moves toward posterior mode
- Preconditioned by empirical covariance
- Requires differentiable log-posterior

**Settings:**
- `cov_mult` (default: 1.0): Controls step size (smaller = more accurate)

**Requirements:**
- The `log_posterior` function must be differentiable w.r.t. the block parameters
- Do NOT use with discrete parameters

---

### MEAN_MALA (Chain-Mean MALA)

**Proposal:** `x' ~ N(μ + drift(μ), cov_mult × Σ)`

Evaluates gradient at population mean instead of current state - more stable for stuck chains.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MEAN_MALA,
    settings={'cov_mult': 0.1}
)
```

**Properties:**
- Single gradient evaluation at μ (shared across chains)
- Independent of current state
- Good for rescuing stuck chains with gradient guidance

**Settings:**
- `cov_mult` (default: 1.0): Controls step size

---

## Adaptive Proposals

These proposals adapt their behavior based on how far the chain is from the population mean.

### MEAN_WEIGHTED

**Proposal:** Interpolates between SELF_MEAN and CHAIN_MEAN based on distance

```
α(d) = d² / (d² + k²)
μ_prop = α × x + (1-α) × μ
x' ~ N(μ_prop, Σ)
```

Where `d` is the Mahalanobis distance from the coupled mean.

**Behavior:**
- Near equilibrium (d ≈ 0): α ≈ 0, behaves like CHAIN_MEAN
- Far from equilibrium (d large): α → 1, tracks current state

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MEAN_WEIGHTED,
    settings={'cov_mult': 1.0, 'cov_beta': 1.0}
)
```

---

### MCOV_WEIGHTED

**Proposal:** Adaptive mean AND covariance scaling

Extends MEAN_WEIGHTED by also scaling the proposal covariance based on distance:

```
g(d) = 1 + β × d / (d + k)    # Covariance scale
α(d) = d² / (d² + k²)          # Mean interpolation
```

**Settings:**
- `cov_mult` (default: 1.0): Base covariance multiplier
- `cov_beta` (default: 1.0): Covariance scaling strength
  - Positive: increase variance when far (explore more)
  - Negative: decrease variance when far (cautious catching up)
  - Zero: no covariance scaling (reduces to MEAN_WEIGHTED)

---

### MCOV_WEIGHTED_VEC (Vectorized)

**Recommended for hyperparameters**

Per-parameter version of MCOV_WEIGHTED - each parameter gets its own α and g based on its individual distance from the marginal mean.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.COUPLED_TRANSFORM,
    proposal_type=ProposalType.MCOV_WEIGHTED_VEC,
    settings={'cov_mult': 1.0, 'cov_beta': -0.9}
)
```

**Properties:**
- Handles parameters with different scales
- Parameters far from their mean stay closer to current value
- Parameters near their mean get pulled toward coupled mean

**Settings:**
- `cov_mult` (default: 1.0): Base step size
- `cov_beta` (default: 1.0): Per-parameter covariance scaling
  - Recommended: `-0.9` for catching up (smaller steps when far)

---

### MCOV_SMOOTH

Three-zone smooth transition proposal with customizable parameters.

```
d → g(d): 1.0 at d=0 → 0.5 at d=k_g → 0 as d→∞
d → α(d): 0.0 at d=0 → 0.5 at d=k_alpha → 1 as d→∞
```

**Settings:**
- `cov_mult` (default: 1.0): Base step size
- `k_g` (default: 10.0): Distance where g = 0.5 (half-size steps)
- `k_alpha` (default: 3.0): Distance where α = 0.5 (balanced blend)

**Behavior at different distances:**

| Distance | α | g | Behavior |
|----------|---|---|----------|
| d=0 | 0.00 | 1.00 | Pure CHAIN_MEAN |
| d=1 | 0.10 | 0.99 | Gentle pull begins |
| d=3 | 0.50 | 0.92 | Balanced blend |
| d=10 | 0.92 | 0.50 | Strong tracking |
| d=20 | 0.98 | 0.20 | Full rescue mode |

---

## Mode-Targeting Proposals

### MCOV_MODE

Targets the mode (chain with highest log-posterior) instead of the mean.

```
μ_prop = α × x + (1-α) × mode
```

Where `mode` is the state of the chain with highest posterior value.

```python
BlockSpec(
    size=2,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.MCOV_MODE,
    settings={'cov_mult': 1.0, 'cov_beta': 1.0}
)
```

**Use case:** Multimodal posteriors where you want chains to move toward the dominant mode.

---

### MCOV_MODE_VEC

Per-parameter version of MCOV_MODE - each parameter can have different interpolation strength toward the mode.

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
| Chains recovering from reset | MCOV_SMOOTH with negative cov_beta |
| Low acceptance rates | Reduce cov_mult |
| Stuck at consensus | Increase uniform_weight for MULTINOMIAL |
| Gradient available | MALA (if smooth) or MEAN_MALA |

### General Guidelines

1. **Start simple**: MIXTURE or CHAIN_MEAN are good defaults
2. **Use MULTINOMIAL** for discrete parameters
3. **Use COUPLED_TRANSFORM** with MCOV_WEIGHTED_VEC for NCP hyperparameters
4. **Tune cov_mult** if acceptance rates are too low (<10%) or too high (>70%)
5. **Avoid MALA** for non-differentiable or discrete parameters
