# bamcmc Package Guide

bamcmc is a JAX-based MCMC sampling package designed for Bayesian hierarchical models. Key features:

- **Coupled A/B chains**: Runs two groups of chains (A and B) that share proposal statistics
- **Nested R-hat diagnostics**: Implements Margossian et al. (2022) for hierarchical convergence checking
- **Block-based sampling**: Parameters are organized into blocks with configurable samplers
- **Cross-session compilation caching**: JAX compilations persist across Python sessions
- **Posterior benchmarking**: Performance tracking with hash-based caching

Full documentation: [docs/README.md](docs/README.md)

## Environment Setup

The project uses a Python virtual environment at `/workspace/pylib/`:

```bash
# Activate the environment
source /workspace/pylib/bin/activate

# Or run directly with the environment's Python
/workspace/pylib/bin/python your_script.py
```

The environment includes:
- JAX 0.8.2 with CUDA 13 support
- numpy, scipy, matplotlib
- bamcmc installed in editable mode

**Running tests:**
```bash
source /workspace/pylib/bin/activate
cd /workspace/bamcmc
pytest tests/ -v
```

## Directory Structure

```
bamcmc/
├── src/bamcmc/
│   ├── __init__.py           # Public API exports
│   ├── mcmc/                  # Core MCMC implementation
│   │   ├── __init__.py       # Subpackage exports
│   │   ├── backend.py        # Main entry points (rmcmc, rmcmc_single)
│   │   ├── types.py          # Core dataclasses (BlockArrays, RunParams)
│   │   ├── config.py         # Configuration and initialization
│   │   ├── sampling.py       # Proposal and MH sampling logic
│   │   ├── scan.py           # JAX scan body and block statistics
│   │   ├── compile.py        # Kernel compilation and caching
│   │   ├── diagnostics.py    # R-hat computation and acceptance rates
│   │   └── utils.py          # Misc utilities
│   ├── proposals/            # 14 proposal distribution implementations
│   │   ├── __init__.py
│   │   ├── common.py         # Shared helpers (COV_NUGGET, compute_mahalanobis, etc.)
│   │   ├── self_mean.py      # SELF_MEAN: random walk proposal
│   │   ├── chain_mean.py     # CHAIN_MEAN: independent proposal
│   │   ├── mixture.py        # MIXTURE: mix of self/chain mean
│   │   ├── multinomial.py    # MULTINOMIAL: discrete parameter proposal
│   │   ├── mala.py           # MALA: gradient-based proposal
│   │   ├── mean_mala.py      # MEAN_MALA: gradient at coupled mean
│   │   ├── mean_weighted.py  # MEAN_WEIGHTED: adaptive interpolation
│   │   ├── mode_weighted.py  # MODE_WEIGHTED: interpolation toward mode
│   │   ├── mcov_weighted.py  # MCOV_WEIGHTED: covariance-scaled interpolation
│   │   ├── mcov_weighted_vec.py  # MCOV_WEIGHTED_VEC: vectorized per-param MCOV
│   │   ├── mcov_smooth.py    # MCOV_SMOOTH: smooth distance-based scaling
│   │   ├── mcov_mode.py      # MCOV_MODE: mode-targeting with scalar distance
│   │   └── mcov_mode_vec.py  # MCOV_MODE_VEC: mode-targeting with per-param distance
│   ├── batch_specs.py        # BlockSpec, SamplerType, ProposalType
│   ├── registry.py           # Posterior registration system
│   ├── settings.py           # Per-block settings (SettingSlot)
│   ├── checkpoint_helpers.py # Save/load checkpoints, combine batches
│   ├── reset_utils.py        # Reset stuck chains from checkpoint
│   ├── posterior_benchmark.py# Benchmark caching with posterior hashing
│   ├── error_handling.py     # Validation and diagnostics
│   ├── jax_config.py         # JAX environment configuration
│   └── test_posteriors.py    # Conjugate test models with analytical solutions
├── tests/
│   ├── conftest.py           # Pytest fixtures
│   ├── test_unit.py          # Unit tests (38 tests)
│   ├── test_nested_rhat.py   # Nested R-hat tests
│   └── test_integration.py   # Full MCMC validation tests
└── pyproject.toml
```

## Core Concepts (Quick Reference)

### Posterior Registration

Models are registered before sampling. Required functions: `log_posterior`, `batch_type`, `initial_vector`, `direct_sampler`.

**CRITICAL**: `initial_vector` must return size `(num_chains_a + num_chains_b) * n_params` (NOT `num_superchains`). `direct_sampler` must return valid values even for MH-only models (JAX traces all branches).

-> See [docs/registration.md](docs/registration.md) for full details.

### Block Specifications

Parameters are organized into blocks with a sampler type and proposal type.

**SamplerType**: `METROPOLIS_HASTINGS` (0), `DIRECT_CONJUGATE` (1), `COUPLED_TRANSFORM` (2)

**ProposalType**: `SELF_MEAN` (0), `CHAIN_MEAN` (1), `MIXTURE` (2), `MULTINOMIAL` (3), `MALA` (4), `MEAN_MALA` (5), `MEAN_WEIGHTED` (6), `MODE_WEIGHTED` (7), `MCOV_WEIGHTED` (8), `MCOV_WEIGHTED_VEC` (9), `MCOV_SMOOTH` (10), `MCOV_MODE` (11), `MCOV_MODE_VEC` (12)

-> See [docs/block_spec.md](docs/block_spec.md), [docs/samplers.md](docs/samplers.md), [docs/proposals.md](docs/proposals.md), [docs/settings.md](docs/settings.md)

### Data Format

Dict with three tuple fields: `"static"` (hashable scalars), `"int"` (integer arrays), `"float"` (float arrays). Static values must be Python scalars, arrays should be C-contiguous, empty tuples required for unused fields.

-> See [docs/configuration.md](docs/configuration.md)

### MCMC Configuration

All config keys use **lowercase with underscores**. Required: `posterior_id`, `num_chains_a`, `num_chains_b`. Common mistake: `NUM_SUPERCHAINS` (uppercase) silently uses the default.

-> See [docs/configuration.md](docs/configuration.md) for full key reference.

### Coupled A/B Chains

Group A proposes using Group B statistics and vice versa. This coupling improves mixing while maintaining detailed balance.

-> See [docs/architecture.md](docs/architecture.md)

### Nested R-hat

Organizes chains into K superchains with M subchains each. More sensitive to stuck chains than standard R-hat.

-> See [docs/architecture.md](docs/architecture.md)

### Parallel Tempering

Runs chains at different temperatures for multimodal posteriors. Set `n_temperatures > 1` to enable.

-> See [docs/configuration.md](docs/configuration.md)

### Mixed Proposals

Blocks can have multiple proposal types for different parameter sub-groups (e.g., continuous + discrete). Use `ProposalGroup` and `proposal_groups` in BlockSpec.

-> See [docs/block_spec.md](docs/block_spec.md)

### COUPLED_TRANSFORM Sampler

Theta-preserving updates for NCP hierarchical models. When updating hyperparameters (mu, sigma), simultaneously adjusts epsilon to keep theta = mu + sigma*epsilon constant. Achieves ~40-50% acceptance rates vs ~1-2% with standard NCP.

Key files: `batch_specs.py` (enum), `mcmc/sampling.py` (`coupled_transform_step()`), model posteriors (`coupled_transform_dispatch()`).

-> See [docs/samplers.md](docs/samplers.md)
-> Mathematical docs: `/workspace/technical_notes/coupled_transform_sampler.tex`, `/workspace/code/docs/theta_preserving_ncp_sampler.tex`

## API Entry Points

- **`rmcmc()`**: Multi-run sampling with automatic checkpointing (recommended for production)
- **`rmcmc_single()`**: Single MCMC run for custom workflows
- **`run_benchmark()`**: Performance testing with cached baselines

-> See [docs/architecture.md](docs/architecture.md) for full signatures and return values.

## Output & Checkpoints

-> See [docs/checkpoint_helpers.md](docs/checkpoint_helpers.md) for checkpoint I/O, output directory management, scanning, cleaning, and post-processing.

## Development

-> See [docs/development.md](docs/development.md) for adding proposals, defining posteriors, minimal template, and testing.

## Troubleshooting

-> See [docs/troubleshooting.md](docs/troubleshooting.md) for common errors, debugging tips, performance tuning, and JAX-compatible coding patterns.

---

## Known Issues & Improvement Suggestions

This section documents known issues and potential improvements identified during code review.

### High Priority (All Fixed)

#### 1. ~~Emoji in Error Output~~ (FIXED)

**Status**: Fixed. Replaced emojis with text-based indicators (`[ERROR]`, `[WARN]`, `[INFO]`, `[OK]`) in:
- `src/bamcmc/error_handling.py`
- `src/bamcmc/mcmc/backend.py`

#### 2. ~~Silent Failure on Unknown Settings Keys~~ (ALREADY IMPLEMENTED)

**Status**: Already implemented. The code in `settings.py:93-100` already warns on unknown keys.

#### 3. ~~Missing Checkpoint Compatibility Validation~~ (FIXED)

**Status**: Fixed. Added `_validate_checkpoint_compatibility()` in `src/bamcmc/mcmc/backend.py` that:
- Validates parameter count matches before resume/reset
- Checks posterior ID matches
- Provides clear error messages when checkpoint is incompatible

### Medium Priority

#### 4. Undocumented Magic Numbers (diagnostics.py)

**Issue**: The nested R-hat threshold uses an undocumented constant.

**Location**: `src/bamcmc/mcmc/diagnostics.py:142`

**Current Code**:
```python
tau = 1e-4
threshold = np.sqrt(1 + 1/M + tau)
```

**Suggested Fix**: Document the source:
```python
# tau: small regularization constant for nested R-hat threshold
# from Margossian et al. (2022), prevents division instability when M is large
TAU_NESTED_RHAT = 1e-4
threshold = np.sqrt(1 + 1/M + TAU_NESTED_RHAT)
```

#### 5. Missing TypedDict for Data Structure

**Issue**: The `data` dict structure is documented but not typed, making it easy to misuse.

**Suggested Addition** (in `src/bamcmc/mcmc/types.py`):
```python
from typing import TypedDict, Tuple
import numpy as np

class MCMCData(TypedDict):
    static: Tuple[int, ...]           # Scalars (hashable)
    int: Tuple[np.ndarray, ...]       # Integer arrays
    float: Tuple[np.ndarray, ...]     # Float arrays
```

#### 6. Settings Key Mapping Boilerplate (settings.py)

**Issue**: Adding a new setting requires edits in 3 places (SettingSlot enum, SETTING_DEFAULTS, key_to_slot dict).

**Suggested Fix**: Auto-generate the mapping:
```python
class SettingSlot(IntEnum):
    COV_MULT = 0
    ALPHA = 1
    # ... etc

# Auto-generate from enum names (lowercase)
KEY_TO_SLOT = {slot.name.lower(): slot.value for slot in SettingSlot}
```

### Lower Priority

#### 7. Missing Dispatch Table Assertion (sampling.py)

**Issue**: The compact proposal dispatch table is built dynamically but not validated.

**Location**: `src/bamcmc/mcmc/sampling.py:88`

**Suggested Fix**:
```python
dispatch_table = [
    (lambda fn: lambda op: fn((*op, grad_fn, block_mode)))(PROPOSAL_REGISTRY[ptype])
    for ptype in used_proposal_types
]

assert len(dispatch_table) == len(used_proposal_types), \
    f"Dispatch table size mismatch: {len(dispatch_table)} vs {len(used_proposal_types)}"
```

#### 8. Test Coverage Gaps

**Current Gaps**:
- No explicit tests for proposal Hastings ratio symmetry
- No tests for checkpoint resume with mismatched data shapes
- No tests for edge cases: single chain, empty blocks
- Direct sampler interface not explicitly tested
- No performance regression tests

**Suggested Additions**:
```python
# tests/test_proposals.py
def test_hastings_ratio_symmetry():
    """Verify q(y|x) / q(x|y) computed correctly for each proposal."""
    # Sample x, y, compute ratio both directions, verify consistency

def test_checkpoint_shape_mismatch():
    """Verify clear error when checkpoint doesn't match model."""
    # Create checkpoint with N params, try to load with M != N
```

### Strengths (Don't Change)

The following patterns are well-implemented and should be preserved:

1. **JAX pytree registration** - BlockArrays properly registered for JAX transforms
2. **Compact proposal dispatch** - Only traces used proposals (memory efficient)
3. **COUPLED_TRANSFORM implementation** - Theta-preserving sampler is mathematically elegant
4. **Cross-session compilation caching** - Excellent user experience
5. **Registry pattern** - Clean plugin architecture for posteriors
6. **Frozen dataclasses** - Immutability for JAX static args
7. **Comprehensive docstrings** - Most functions well-documented

### Summary Table

| Priority | Issue | File | Status |
|----------|-------|------|--------|
| ~~High~~ | ~~Emoji in diagnostics~~ | error_handling.py, backend.py | **FIXED** |
| ~~High~~ | ~~Silent unknown settings~~ | settings.py | **Already implemented** |
| ~~High~~ | ~~Checkpoint compatibility~~ | backend.py | **FIXED** |
| Medium | Magic number docs | diagnostics.py | Open |
| Medium | TypedDict for data | types.py | Open |
| Medium | Settings auto-mapping | settings.py | Open |
| Low | Dispatch assertion | sampling.py | Open |
| Low | Test coverage | tests/ | Open |
