# Settings Reference

Settings control proposal behavior. They are specified in `BlockSpec.settings` or `ProposalGroup.settings`.

## Available Settings

| Setting | Type | Default | Used By | Description |
|---------|------|---------|---------|-------------|
| `cov_mult` | float | 1.0 | Most proposals | Covariance multiplier |
| `chain_prob` | float | 0.5 | MIXTURE | Probability of chain_mean |
| `cov_beta` | float | 1.0 | MCOV_* proposals | Covariance scaling strength |
| `n_categories` | int | 4 | MULTINOMIAL | Number of discrete categories |
| `uniform_weight` | float | 0.4 | MULTINOMIAL | Weight of uniform distribution |
| `k_g` | float | 10.0 | MCOV_SMOOTH | Controls g decay rate |
| `k_alpha` | float | 3.0 | MCOV_SMOOTH | Controls alpha rise rate |

---

## Detailed Descriptions

### cov_mult

**Type:** float
**Default:** 1.0
**Used by:** SELF_MEAN, MIXTURE, MALA, MEAN_MALA, MEAN_WEIGHTED, MODE_WEIGHTED, MCOV_*

Scales the proposal covariance matrix:
```
Σ_proposal = cov_mult × Σ_coupled
```

**Guidelines:**
- Decrease if acceptance rate < 10%
- Increase if acceptance rate > 70%
- Typical range: 0.1 to 2.0

```python
settings={'cov_mult': 0.5}  # Smaller steps, higher acceptance
settings={'cov_mult': 2.0}  # Larger steps, lower acceptance
```

---

### chain_prob

**Type:** float (0 to 1)
**Default:** 0.5
**Used by:** MIXTURE

Probability of using CHAIN_MEAN (independent) vs SELF_MEAN (random walk):

```python
settings={'chain_prob': 0.8}  # 80% independent, 20% random walk
settings={'chain_prob': 0.2}  # 20% independent, 80% random walk
```

**Guidelines:**
- Higher values: Better global mixing, may lower acceptance for far chains
- Lower values: More local exploration, may slow mixing

---

### cov_beta

**Type:** float
**Default:** 1.0
**Used by:** MCOV_WEIGHTED, MCOV_WEIGHTED_VEC, MCOV_SMOOTH, MCOV_MODE, MCOV_MODE_VEC

Controls how proposal covariance scales with distance from coupled mean:

```
g(d) = 1 + β × f(d)    # f(d) depends on proposal type
Σ_proposal = g(d) × cov_mult × Σ_coupled
```

**Values:**
- `β > 0`: Increase variance when far (explore more aggressively)
- `β = 0`: No distance-based scaling
- `β < 0`: Decrease variance when far (cautious catching up)

**Recommended:**
- For COUPLED_TRANSFORM hyperparameters: `cov_beta=-0.9`
- For general use: `cov_beta=0.0` to `1.0`

```python
# Cautious catching up (recommended for NCP hyperparameters)
settings={'cov_mult': 1.0, 'cov_beta': -0.9}

# Aggressive exploration when far
settings={'cov_mult': 1.0, 'cov_beta': 1.5}
```

---

### n_categories

**Type:** int
**Default:** 4
**Used by:** MULTINOMIAL

Number of valid discrete categories. Parameters take values `{1, 2, ..., n_categories}`.

```python
# Binary indicator (z=1 or z=2)
settings={'n_categories': 2}

# 3-way model selection
settings={'n_categories': 3}
```

**Note:** Values are 1-indexed. A parameter with `n_categories=2` takes values 1 or 2, not 0 or 1.

---

### uniform_weight

**Type:** float (0 to 1)
**Default:** 0.4
**Used by:** MULTINOMIAL

Weight of uniform distribution in the proposal mixture:

```
q(k) = w × Uniform(1, n_categories) + (1-w) × Empirical(k)
```

Where `w = uniform_weight` and Empirical is based on coupled chain frequencies.

**Guidelines:**
- Higher values: More exploration, prevents stuck at consensus
- Lower values: More exploitation, faster convergence when chains agree

```python
# More exploration
settings={'n_categories': 2, 'uniform_weight': 0.5}

# More exploitation
settings={'n_categories': 2, 'uniform_weight': 0.2}
```

---

### k_g

**Type:** float
**Default:** 10.0
**Used by:** MCOV_SMOOTH

Controls how quickly the covariance scaling `g` decays with distance:

```
g(d) = 1 / (1 + (d / k_g)²)
```

- At `d = 0`: g = 1.0 (full covariance)
- At `d = k_g`: g = 0.5 (half covariance)
- As `d → ∞`: g → 0 (minimal steps)

**Guidelines:**
- Larger `k_g`: Maintain larger steps further from mean
- Smaller `k_g`: Start shrinking steps sooner

```python
# Keep full steps until d=10
settings={'k_g': 10.0, 'k_alpha': 3.0}

# Start shrinking at d=5
settings={'k_g': 5.0, 'k_alpha': 3.0}
```

---

### k_alpha

**Type:** float
**Default:** 3.0
**Used by:** MCOV_SMOOTH

Controls how quickly the interpolation weight `α` rises with distance:

```
α(d) = d² / (d² + k_α²)
```

Where `α` blends between chain_mean (α=0) and self_mean (α=1).

- At `d = 0`: α = 0 (pure chain_mean)
- At `d = k_alpha`: α = 0.5 (50/50 blend)
- As `d → ∞`: α → 1 (pure self_mean)

**Guidelines:**
- Larger `k_alpha`: Stay with chain_mean longer
- Smaller `k_alpha`: Transition to tracking sooner

```python
# Balanced at d=3
settings={'k_g': 10.0, 'k_alpha': 3.0}

# Transition at d=5
settings={'k_g': 10.0, 'k_alpha': 5.0}
```

---

## Settings by Proposal Type

### SELF_MEAN
```python
settings={'cov_mult': 1.0}
```

### CHAIN_MEAN
No configurable settings.

### MIXTURE
```python
settings={
    'chain_prob': 0.5,
    'cov_mult': 1.0
}
```

### MULTINOMIAL
```python
settings={
    'n_categories': 3,
    'uniform_weight': 0.4
}
```

### MALA / MEAN_MALA
```python
settings={'cov_mult': 0.1}  # Small steps for MALA
```

### MCOV_WEIGHTED / MCOV_WEIGHTED_VEC
```python
settings={
    'cov_mult': 1.0,
    'cov_beta': -0.9  # Cautious catching up
}
```

### MCOV_SMOOTH
```python
settings={
    'cov_mult': 1.0,
    'k_g': 10.0,
    'k_alpha': 3.0
}
```

### MCOV_MODE / MCOV_MODE_VEC
```python
settings={
    'cov_mult': 1.0,
    'cov_beta': 1.0
}
```

---

## Default Values

If a setting is not specified, the default is used:

```python
from bamcmc.settings import SETTING_DEFAULTS

print(SETTING_DEFAULTS)
# {
#     SettingSlot.CHAIN_PROB: 0.5,
#     SettingSlot.N_CATEGORIES: 4.0,
#     SettingSlot.COV_MULT: 1.0,
#     SettingSlot.UNIFORM_WEIGHT: 0.4,
#     SettingSlot.COV_BETA: 1.0,
#     SettingSlot.K_G: 10.0,
#     SettingSlot.K_ALPHA: 3.0,
# }
```

---

## Tuning Guidelines

### Low Acceptance Rate (< 10%)

1. Decrease `cov_mult`: `settings={'cov_mult': 0.5}`
2. For MULTINOMIAL: Increase `uniform_weight`
3. For MALA: Use smaller `cov_mult` (e.g., 0.01)

### High Acceptance Rate (> 70%)

1. Increase `cov_mult`: `settings={'cov_mult': 2.0}`
2. Consider switching to CHAIN_MEAN

### Stuck Chains

1. Use CHAIN_MEAN or MIXTURE with high `chain_prob`
2. For MCOV_*: Use negative `cov_beta` for cautious catching up
3. For MULTINOMIAL: Increase `uniform_weight`

### Multimodal Posterior

1. Use MCOV_MODE or MODE_WEIGHTED
2. Consider parallel tempering (`n_temperatures > 1`)
3. Increase `uniform_weight` for discrete indicators
