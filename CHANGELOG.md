# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-04-06

First stable release.

### Added
- Coupled A/B chain sampling: two chain groups share proposal statistics to improve mixing while maintaining detailed balance.
- Nested R-hat diagnostics following Margossian et al. (2022) with superchain/subchain structure.
- 13 proposal distributions: `SELF_MEAN`, `CHAIN_MEAN`, `MIXTURE`, `MULTINOMIAL`, `MALA`, `MEAN_MALA`, `MEAN_WEIGHTED`, `MODE_WEIGHTED`, `MCOV_WEIGHTED`, `MCOV_WEIGHTED_VEC`, `MCOV_SMOOTH`, `MCOV_MODE`, `MCOV_MODE_VEC`.
- Three sampler types: `METROPOLIS_HASTINGS`, `DIRECT_CONJUGATE`, and `COUPLED_TRANSFORM` (theta-preserving updates for NCP hierarchical models).
- Mixed proposals within a single block via `ProposalGroup`.
- Parallel tempering with DEO index-process scheme, including `filter_beta1_samples()` and `compute_round_trip_rate()` helpers.
- Cross-session JAX compilation caching.
- Posterior benchmarking with hash-based caching (`run_benchmark`, `PosteriorBenchmarkManager`).
- Checkpoint save/load/resume workflow with compatibility validation.
- Chain reset utilities for recovering from stuck chains (`reset_from_checkpoint`, `generate_reset_states`, `select_diverse_states`, etc.).
- Public `bamcmc.__version__` attribute, sourced from package metadata.
- Comprehensive pytest suite covering all 13 proposals, reset utilities, tempering, coupled-transform sampling, and benchmark/hash subsystems.

[Unreleased]: https://github.com/bamonroe/bamcmc/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/bamonroe/bamcmc/releases/tag/v1.0.0
