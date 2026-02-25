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
│   │   ├── tempering.py      # Parallel tempering swap logic (index process, DEO)
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

Index-process PT with DEO scheme. Swaps temperature assignments (not states), so each chain maintains a continuous trace. Set `n_temperatures > 1` to enable. Results include `temperature_ladder`, `swap_rates`, `round_trip_rate`, and `round_trip_counts`. Use `filter_beta1_samples()` to extract posterior samples. Public API exports: `filter_beta1_samples`, `compute_round_trip_rate`.

-> See [docs/tempering.md](docs/tempering.md) for full details

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

### Medium Priority (All Fixed)

#### 4. ~~Undocumented Magic Numbers~~ (FIXED)

**Status**: Fixed. Named the constant `TAU_NESTED_RHAT` with documentation comments in `src/bamcmc/mcmc/diagnostics.py`.

#### 5. ~~Missing TypedDict for Data Structure~~ (FIXED)

**Status**: Fixed. Added `MCMCData` TypedDict in `src/bamcmc/mcmc/types.py`, exported from `src/bamcmc/__init__.py`.

#### 6. ~~Settings Key Mapping Boilerplate~~ (FIXED)

**Status**: Fixed. Replaced manual `key_to_slot` dict with auto-generated module-level `KEY_TO_SLOT` in `src/bamcmc/settings.py`. Adding a new setting now requires 2 edits instead of 3.

### Lower Priority (All Fixed)

#### 7. ~~Missing Dispatch Table Assertion~~ (FIXED)

**Status**: Fixed. Added assertion after dispatch table construction in `src/bamcmc/mcmc/sampling.py`.

#### 8. ~~Test Coverage Gaps~~ (FIXED)

**Status**: Fixed. Added tests for:
- Hastings ratio symmetry (CHAIN_MEAN) in `tests/test_proposals.py`
- Checkpoint num_params mismatch in `tests/test_checkpoints.py`
- Single chain initialization in `tests/test_unit.py`
- Empty block spec list in `tests/test_unit.py`

### Strengths (Don't Change)

The following patterns are well-implemented and should be preserved:

1. **JAX pytree registration** - BlockArrays properly registered for JAX transforms
2. **Compact proposal dispatch** - Only traces used proposals (memory efficient)
3. **COUPLED_TRANSFORM implementation** - Theta-preserving sampler is mathematically elegant
4. **Cross-session compilation caching** - Excellent user experience
5. **Registry pattern** - Clean plugin architecture for posteriors
6. **Frozen dataclasses** - Immutability for JAX static args
7. **Comprehensive docstrings** - Most functions well-documented

### High Priority

#### 9. ~~Test Coverage for Untested Proposal Types~~ (FIXED)

**Status**: Fixed. Added 29 tests for the 5 previously untested proposal types in `tests/test_proposals.py`:
- `MEAN_MALA`: Hastings ratio formula verification, gradient usage at mean, independence from current state
- `MODE_WEIGHTED`: Finiteness, near-mode/far-from-mode interpolation, block mask
- `MCOV_WEIGHTED_VEC`: Per-parameter alpha behavior, near-mean/far-from-mean dynamics, block mask
- `MCOV_MODE`: Scalar distance targeting, k_g settings effect, near-mode/far-from-mode behavior
- `MCOV_MODE_VEC`: Per-parameter alpha, consistency with MCOV_MODE, cross-family finiteness

All 13 proposals now have test coverage (65 total tests in test_proposals.py).

#### 10. ~~Test Coverage for reset_utils~~ (FIXED)

**Status**: Fixed. Added 30 tests in `tests/test_reset_utils.py` covering all 7 public functions:
- `compute_chain_statistics()`: basic math, single chain, unequal groups
- `get_discrete_param_indices()`: all 3 mixture models + unknown model fallback
- `_get_legacy_special_indices()`: z/pi/r index structure for 3-model and 4-model
- `generate_reset_states()`: shape, reproducibility, near-mean centering, noise_scale, discrete param sampling
- `generate_reset_vector()`: flat shape, subchain replication, superchain diversity
- `select_diverse_states()`: shape, K==n_chains, K>n_chains error, actual chain states, uniqueness
- `print_reset_summary()`: smoke tests with plain and mixture models
- `reset_from_checkpoint()` / `init_from_prior()`: disk I/O roundtrips, model mismatch warning

#### 11. ~~Test Coverage for Tempering and Coupled Transform~~ (FIXED)

**Status**: Fixed. Added 16 tests in `tests/test_tempering_sampling.py`:
- `attempt_temperature_swaps()`: single-temp no-op, DEO parity toggle, even/odd parity gating,
  use_deo=False all-pairs, states-not-modified, count accumulation, identical-states-always-accept,
  assignments-are-permuted (9 tests)
- `coupled_transform_step()`: basic acceptance, Jacobian effect on acceptance, NaN rejection,
  coupled indices updated on accept (4 tests)
- `metropolis_block_step()`: NaN rejection, peaked-posterior acceptance, beta tempering parameter (3 tests)

### Medium Priority

#### 12. ~~Proposal Code Duplication~~ (FIXED)

**Status**: Fixed. Extracted shared helpers into `src/bamcmc/proposals/common.py`:
- `unpack_operand()`: Unpacks 9-element operand tuple into `Operand` namedtuple (used by all 13 proposals)
- `prepare_proposal()`: Common setup — unpack, split key, extract cov_mult, regularize cov, Cholesky (used by ~6 proposals)
- `sample_diffusion()`: Generates `scale * (L @ normal(key, shape))` noise (used by ~10 proposals)
- `hastings_ratio_fixed_cov()`: Hastings ratio for state-dependent mean with fixed covariance (used by mean_weighted, mode_weighted)
- `hastings_ratio_scalar_g()`: Hastings ratio with scalar g covariance scaling (used by mcov_smooth, mcov_mode, mcov_mode_vec)

All 13 proposals refactored to use these helpers. Saves ~5-15 lines per proposal.

#### 13. ~~Inconsistent Epsilon/Nugget Constants~~ (FIXED)

**Status**: Fixed. Consolidated numerical constants across proposals:
- Added `NUMERICAL_EPS = 1e-10` in `proposals/common.py` — replaced all 30+ hardcoded `1e-10` values across 8 proposal files
- Replaced inline `1e-6` in `mcov_smooth.py` and `mcov_mode_vec.py` with `COV_NUGGET`
- Added explanatory comments to `COV_NUGGET` (proposals) and `NUGGET` (scan.py) explaining why they differ (1e-6 vs 1e-5)
- Promoted `TAU_NESTED_RHAT` to module level in `diagnostics.py`, imported in `history_processing.py` instead of duplicating

#### 14. ~~Type Annotations on Public Functions~~ (FIXED)

**Status**: Fixed. Added type annotations to 20 public functions across 4 files:
- `src/bamcmc/checkpoint_io.py`: `save_checkpoint()`, `load_checkpoint()`, `initialize_from_checkpoint()`
- `src/bamcmc/history_processing.py`: `combine_batch_histories()`, `apply_burnin()`, `compute_rhat_from_history()`, `split_history_by_subject()`, `postprocess_all_histories()`
- `src/bamcmc/reset_utils.py`: all 9 public/private functions
- `src/bamcmc/error_handling.py`: `validate_mcmc_config()`, `diagnose_sampler_issues()`, `print_diagnostics()`

#### 15. ~~Test Coverage for Benchmark and Hash Systems~~ (FIXED)

**Status**: Fixed. Added 34 tests in `tests/test_benchmark_hash.py` covering all three modules:
- `posterior_hash.py`: `get_function_source_hash` (4 tests), `get_data_hash` (5 tests), `get_posterior_hash`/`compute_posterior_hash` (4 tests)
- `posterior_benchmark.py`: `PosteriorBenchmarkManager` save/load (5 tests), hardware match (2 tests), cached benchmark (3 tests), compare (2 tests), list/print (3 tests), factory functions (2 tests)
- `prior_config.py`: save/load roundtrip, nonexistent file, numpy array serialization (4 tests)

### Lower Priority

#### 16. Large Functions

**Status**: Open. Several functions exceed 200 lines and could benefit from phase extraction:
- `rmcmc()` in `src/bamcmc/mcmc/backend.py` — 294 lines
- `rmcmc_single()` in `src/bamcmc/mcmc/single_run.py` — 244 lines
- `configure_mcmc_system()` in `src/bamcmc/mcmc/config.py` — 197 lines

Extracting phases (validation, initialization, run-loop, diagnostics) would make them easier to test and reason about.

#### 17. Logging vs Print Statements

**Status**: Open. 127+ `print()` calls used for status output across the codebase. Switching to Python's `logging` module would let users control verbosity without code changes (e.g., silence output in library usage, enable debug in troubleshooting).

Major files: `diagnostics.py` (23), `single_run.py` (24), `posterior_benchmark.py` (25), `backend.py` (18), `history_processing.py` (18).

#### 18. Hardcoded Model-Specific Indices in reset_utils

**Status**: Open. `_get_legacy_special_indices()` in `src/bamcmc/reset_utils.py` (lines 85-176) has hardcoded block sizes and parameter offsets for specific mixture models (e.g., `subject_block_size = 20` for mixture_3model_bhm). This couples a utility module to specific posteriors — ideally posteriors would declare their own discrete indices via the registry.

### Summary Table

| Priority | Issue | File | Status |
|----------|-------|------|--------|
| ~~High~~ | ~~Emoji in diagnostics~~ | error_handling.py, backend.py | **FIXED** |
| ~~High~~ | ~~Silent unknown settings~~ | settings.py | **Already implemented** |
| ~~High~~ | ~~Checkpoint compatibility~~ | backend.py | **FIXED** |
| ~~Medium~~ | ~~Magic number docs~~ | diagnostics.py | **FIXED** |
| ~~Medium~~ | ~~TypedDict for data~~ | types.py | **FIXED** |
| ~~Medium~~ | ~~Settings auto-mapping~~ | settings.py | **FIXED** |
| ~~Low~~ | ~~Dispatch assertion~~ | sampling.py | **FIXED** |
| ~~Low~~ | ~~Test coverage~~ | tests/ | **FIXED** |
| ~~High~~ | ~~Test untested proposals~~ | tests/test_proposals.py | **FIXED** |
| ~~High~~ | ~~Test reset_utils~~ | tests/test_reset_utils.py | **FIXED** |
| ~~High~~ | ~~Test tempering & coupled transform~~ | test_tempering_sampling.py | **FIXED** |
| ~~Medium~~ | ~~Proposal code duplication~~ | proposals/*.py, common.py | **FIXED** |
| ~~Medium~~ | ~~Inconsistent epsilon/nugget constants~~ | scan.py, proposals/common.py | **FIXED** |
| ~~Medium~~ | ~~Type annotations on public API~~ | checkpoint_io.py, history_processing.py, etc. | **FIXED** |
| ~~Medium~~ | ~~Test benchmark/hash systems~~ | test_benchmark_hash.py | **FIXED** |
| Low | Large functions (200+ lines) | backend.py, single_run.py, config.py | Open |
| Low | Logging vs print statements | Multiple files (127+ prints) | Open |
| Low | Hardcoded model indices | reset_utils.py | Open |
