# Troubleshooting

This document covers common errors, debugging strategies, performance tuning, and JAX-compatible coding patterns.

## Common Error Messages

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `cannot reshape array of shape (X,) into shape (Y, Z)` | `initial_vector` returns wrong size | Use `(num_chains_a + num_chains_b) * n_params` |
| `KeyError: 'direct_sampler'` | Missing `direct_sampler` in registration | Add placeholder function that returns `(chain_state, key)` |
| `NotImplementedError` during compilation | `direct_sampler` raises exception | Make it return valid values (JAX traces all branches) |
| `Abstract tracer value encountered` | Python control flow on JAX traced values | Use `jnp.where`, `lax.cond`, `lax.switch` |
| `jax.errors.ConcretizationTypeError` | Using traced value as array shape/index | Move shape computation before JAX tracing |
| `cholesky: decomposition failed` | Ill-conditioned covariance matrix | Check for NaN/Inf in parameters, increase `COV_NUGGET` |

## Initialization Issues

**Symptom**: All chains stuck or immediately divergent

**Solutions**:
1. Check `initial_vector` returns sensible parameter values
2. Verify parameters are in prior support (e.g., positive for variance)
3. Use smaller initial variance (`rng.normal(0, 0.1, ...)` not `0, 1`)
4. Check `log_posterior` returns finite values for initial state

## Slow Compilation

**Symptom**: First run takes minutes instead of seconds

**Solutions**:
1. Reduce Python loops in `log_posterior` (use `vmap`/`fori_loop`)
2. Avoid creating arrays inside traced functions
3. Check for accidental recompilation (data shape changes)
4. Clear JAX cache if corrupted: `rm -rf ./jax_cache/`

## Debugging Tips

1. **Check R-hat**: Values > 1.1 indicate non-convergence
2. **Acceptance rates**: Stored in diagnostics, target 20-40%
3. **Trace plots**: Plot `history[:, chain_idx, param_idx]` over iterations
4. **Block labels**: Use `label` in BlockSpec for clearer error messages
5. **Validation errors**: `validate_mcmc_config()` catches common issues

## Performance Considerations

1. **Block size**: Larger blocks = fewer kernel calls but coarser updates
2. **Chunk size**: `chunk_size` in mcmc_config controls iterations per compiled chunk.
   - Default: 100. Lower values (e.g., 10) reduce compilation time/memory but add ~1-10% runtime overhead
   - Useful for OOM during compilation or faster iteration during development
3. **Proposal type**: CHAIN_MEAN often mixes faster but needs good initialization
4. **Float precision**: `use_double=True` is slower but more stable
5. **Compilation**: First run compiles (~3-17s); cached runs are fast (~3s)

## JAX-Compatible Coding Patterns

When writing `log_posterior` and other functions that JAX traces:

### Module-Level Functions Required

For compilation caching to work, define functions at module level (not closures):

```python
# WRONG - closure won't cache properly
def make_log_posterior(hyperparams):
    def log_posterior(chain_state, param_indices, data):
        return compute(hyperparams)  # Uses closure variable
    return log_posterior

# CORRECT - module-level, get hyperparams from data
def log_posterior(chain_state, param_indices, data):
    hyperparams = data["static"][0]  # From data, not closure
    return compute(hyperparams)
```

### Avoid Python Loops for Large Iterations

Python `for` loops are unrolled during tracing. For large N, use `lax.fori_loop` or `vmap`:

```python
# SLOW - N iterations unrolled, huge trace
def log_posterior(chain_state, param_indices, data):
    n_items = data["static"][0]
    log_lik = 0.0
    for i in range(n_items):  # Unrolled!
        log_lik += item_lik(i, ...)
    return log_lik

# FAST - single traced loop
def log_posterior(chain_state, param_indices, data):
    n_items = data["static"][0]
    def body(i, acc):
        return acc + item_lik(i, ...)
    return jax.lax.fori_loop(0, n_items, body, 0.0)

# FASTEST - vectorized
def log_posterior(chain_state, param_indices, data):
    return jnp.sum(jax.vmap(item_lik)(jnp.arange(n_items), ...))
```

### No Python Control Flow on Traced Values

```python
# WRONG - x is a tracer, not a concrete value
if x > 0:
    return log_prob

# CORRECT - use jnp.where for conditional logic
return jnp.where(x > 0, log_prob, -jnp.inf)
```
