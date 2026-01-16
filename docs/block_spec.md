# BlockSpec Reference

`BlockSpec` defines how parameter blocks are structured and sampled.

## Basic Structure

```python
from bamcmc import BlockSpec, SamplerType, ProposalType

BlockSpec(
    # Required
    size=3,                                    # Number of parameters
    sampler_type=SamplerType.METROPOLIS_HASTINGS,

    # For MH samplers
    proposal_type=ProposalType.CHAIN_MEAN,
    settings={'cov_mult': 1.0},

    # Metadata
    label="Subject_0_params"
)
```

## Required Fields

### size (int)

Number of parameters in this block. Must be >= 1.

```python
BlockSpec(size=5, ...)  # Block with 5 parameters
```

### sampler_type (SamplerType)

How to sample this block:

```python
from bamcmc import SamplerType

# Standard Metropolis-Hastings
sampler_type=SamplerType.METROPOLIS_HASTINGS

# Direct/Gibbs sampling from conjugate conditional
sampler_type=SamplerType.DIRECT_CONJUGATE

# MH with theta-preserving transforms (for NCP)
sampler_type=SamplerType.COUPLED_TRANSFORM
```

## Optional Fields

### proposal_type (ProposalType)

For `METROPOLIS_HASTINGS` and `COUPLED_TRANSFORM` samplers, specifies the proposal distribution:

```python
from bamcmc import ProposalType

proposal_type=ProposalType.SELF_MEAN      # Random walk
proposal_type=ProposalType.CHAIN_MEAN     # Independent
proposal_type=ProposalType.MIXTURE        # Blend
proposal_type=ProposalType.MULTINOMIAL    # Discrete
proposal_type=ProposalType.MALA           # Gradient-based
# ... see proposals.md for all options
```

Default: `ProposalType.SELF_MEAN`

### settings (dict)

Proposal-specific settings:

```python
settings={
    'cov_mult': 1.0,       # Covariance multiplier
    'chain_prob': 0.5,     # For MIXTURE: prob of chain_mean
    'n_categories': 3,     # For MULTINOMIAL: number of categories
    'uniform_weight': 0.4, # For MULTINOMIAL: uniform mixture weight
    'cov_beta': 0.0,       # For MCOV_*: covariance scaling
    'k_g': 10.0,           # For MCOV_SMOOTH: g decay rate
    'k_alpha': 3.0,        # For MCOV_SMOOTH: alpha rise rate
}
```

### label (str)

Human-readable identifier for debugging:

```python
label="Subject_42_preferences"
```

Labels appear in error messages and diagnostic output.

### direct_sampler_fn (Callable)

For `DIRECT_CONJUGATE` sampler, the function that samples from the conditional:

```python
def my_sampler(key, chain_state, param_indices, data):
    # Sample from conditional distribution
    new_state = chain_state.at[param_indices].set(new_values)
    return new_state, new_key

BlockSpec(
    size=1,
    sampler_type=SamplerType.DIRECT_CONJUGATE,
    direct_sampler_fn=my_sampler
)
```

### metadata (dict)

Arbitrary metadata (not used by sampler):

```python
metadata={'subject_id': 42, 'group': 'control'}
```

---

## Mixed Proposals

For blocks containing both continuous and discrete parameters, use `proposal_groups`:

```python
from bamcmc import BlockSpec, ProposalGroup, ProposalType

BlockSpec(
    size=13,  # 12 continuous + 1 discrete
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_groups=[
        ProposalGroup(
            start=0,
            end=12,
            proposal_type=ProposalType.MCOV_MODE,
            settings={'cov_mult': 1.0}
        ),
        ProposalGroup(
            start=12,
            end=13,
            proposal_type=ProposalType.MULTINOMIAL,
            settings={'n_categories': 2, 'uniform_weight': 0.4}
        ),
    ],
    label="Subject_0"
)
```

### ProposalGroup Fields

| Field | Type | Description |
|-------|------|-------------|
| `start` | int | Starting index within block (0-indexed, inclusive) |
| `end` | int | Ending index within block (exclusive) |
| `proposal_type` | ProposalType | Proposal for this sub-group |
| `settings` | dict | Settings for this sub-group |

### Requirements

1. Groups must be **contiguous** (no gaps)
2. First group must start at 0
3. Last group must end at block size
4. Maximum **4 groups** per block

---

## Common Patterns

### Subject Blocks

Create many identical blocks for subjects:

```python
from bamcmc import create_subject_blocks

specs = create_subject_blocks(
    n_subjects=100,
    params_per_subject=3,
    sampler_type=SamplerType.METROPOLIS_HASTINGS,
    proposal_type=ProposalType.CHAIN_MEAN,
    settings={'cov_mult': 1.0},
    label_prefix="Subj"
)
# Creates 100 BlockSpecs with labels "Subj_0", "Subj_1", ...
```

### Mixed Subject Blocks

Create subject blocks with mixed proposals:

```python
from bamcmc import create_mixed_subject_blocks, ProposalGroup

groups = [
    ProposalGroup(0, 12, ProposalType.MCOV_MODE, settings={'cov_mult': 1.0}),
    ProposalGroup(12, 13, ProposalType.MULTINOMIAL, settings={'n_categories': 2}),
]

specs = create_mixed_subject_blocks(
    n_subjects=100,
    proposal_groups=groups,
    label_prefix="Subj"
)
```

### Hyperparameter Blocks with COUPLED_TRANSFORM

```python
# For NCP hyperparameters (mean, logsd pairs)
hyper_specs = [
    BlockSpec(
        size=2,
        sampler_type=SamplerType.COUPLED_TRANSFORM,
        proposal_type=ProposalType.MCOV_WEIGHTED_VEC,
        settings={'cov_mult': 1.0, 'cov_beta': -0.9},
        label="Hyper_r"
    ),
    BlockSpec(
        size=2,
        sampler_type=SamplerType.COUPLED_TRANSFORM,
        proposal_type=ProposalType.MCOV_WEIGHTED_VEC,
        settings={'cov_mult': 1.0, 'cov_beta': -0.9},
        label="Hyper_alpha"
    ),
]
```

---

## batch_type Function

The `batch_type` function returns all block specifications for a model:

```python
def batch_type(mcmc_config, data):
    """
    Define parameter blocks for the model.

    Args:
        mcmc_config: MCMC configuration dict
        data: Data dict with 'static', 'int', 'float' fields

    Returns:
        List of BlockSpec objects defining all parameter blocks
    """
    n_subjects = data['static'][0]

    # Subject-level parameters
    subject_specs = create_subject_blocks(
        n_subjects=n_subjects,
        params_per_subject=3,
        proposal_type=ProposalType.CHAIN_MEAN,
        label_prefix="Subj"
    )

    # Hyperparameters
    hyper_specs = [
        BlockSpec(size=2, sampler_type=SamplerType.COUPLED_TRANSFORM,
                  proposal_type=ProposalType.MCOV_WEIGHTED_VEC,
                  label="Hyper_mean_sd"),
    ]

    return subject_specs + hyper_specs
```

### Block Order

The order of blocks in the returned list determines:

1. **Parameter indices**: Parameters are assigned consecutive indices based on block order
2. **Update order**: Blocks are updated in the order listed (per iteration)

Typical pattern: subject blocks first, then hyperparameter blocks.

---

## Validation

BlockSpecs are validated at creation time:

```python
# This will raise ValueError:
BlockSpec(size=0, ...)  # size must be >= 1

# This will raise ValueError:
BlockSpec(
    size=2,
    sampler_type=SamplerType.DIRECT_CONJUGATE
    # Missing direct_sampler_fn
)

# This will raise ValueError (groups don't cover block):
BlockSpec(
    size=10,
    proposal_groups=[
        ProposalGroup(0, 5, ProposalType.CHAIN_MEAN)
        # Missing coverage for indices 5-10
    ]
)
```

Use `validate_block_specs()` to check a list of specs:

```python
from bamcmc.batch_specs import validate_block_specs

try:
    validate_block_specs(my_specs, model_name="my_model")
except ValueError as e:
    print(f"Invalid specs: {e}")
```

---

## Debugging

Use `summarize_blocks()` for an overview:

```python
from bamcmc.batch_specs import summarize_blocks

print(summarize_blocks(my_specs))
```

Output:
```
Block Specification Summary:
  Total blocks: 102
  Total parameters: 308

Sampler breakdown:
  Metropolis Hastings: 100 block(s)
  Coupled Transform: 2 block(s)

Labeled blocks:
  Block 0: Subj_0 (size=3)
  Block 1: Subj_1 (size=3)
  ...
  Block 100: Hyper_mean (size=2)
  Block 101: Hyper_sd (size=2)
```
