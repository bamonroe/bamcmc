# Checkpoint Helpers

The `checkpoint_helpers` module provides utilities for managing MCMC checkpoints, output file management, and post-processing history files for memory-efficient analysis.

## Checkpoint Management

### save_checkpoint / load_checkpoint

Persist and restore chain states for resume functionality:

```python
from bamcmc.checkpoint_helpers import save_checkpoint, load_checkpoint

# Save current state
save_checkpoint(filepath, carry, user_config, metadata=None)

# Load checkpoint
checkpoint = load_checkpoint(filepath)
# Returns dict with states, keys, iteration, etc.

# Resume from checkpoint
results, checkpoint = rmcmc_single(config, data, resume_from='checkpoint_001.npz')
```

### initialize_from_checkpoint

Create MCMC carry state from a loaded checkpoint:

```python
from bamcmc.checkpoint_helpers import initialize_from_checkpoint

carry = initialize_from_checkpoint(checkpoint, user_config, runtime_ctx, ...)
```

### combine_batch_histories

Merge multiple history batch files into one:

```python
from bamcmc.checkpoint_helpers import combine_batch_histories

combined = combine_batch_histories(['history_000.npz', 'history_001.npz'])
```

### apply_burnin

Remove initial samples before a specified iteration:

```python
from bamcmc.checkpoint_helpers import apply_burnin

filtered = apply_burnin(history, iterations, likelihoods=None, min_iteration=5000)
```

### compute_rhat_from_history

Compute R-hat on combined history:

```python
from bamcmc.checkpoint_helpers import compute_rhat_from_history

rhat = compute_rhat_from_history(history, K, M)
```

## Output Directory Management

### Scanning for Existing Files

```python
from bamcmc import scan_checkpoints, get_latest_checkpoint

# Find all checkpoints and histories
scan = scan_checkpoints(output_dir, model_name)
# Returns: {
#   'checkpoint_files': [(run_idx, path), ...],
#   'history_files': [(run_idx, path), ...],
#   'latest_checkpoint': path or None,
#   'latest_run_index': int or -1,
# }

# Get just the latest checkpoint path
latest = get_latest_checkpoint(output_dir, model_name)  # path or None
```

### Cleaning Output Files

The `clean_model_files()` function provides two cleaning modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `'all'` | Delete all checkpoints and histories | Complete fresh start |
| `'keep_latest'` | Delete histories and old checkpoints, **keep latest checkpoint** | Normal workflow (allows resume) |

```python
from bamcmc import clean_model_files

# Delete everything (fresh start)
result = clean_model_files(output_dir, model_name, mode='all')

# Keep latest checkpoint for resume (default for iterative workflows)
result = clean_model_files(output_dir, model_name, mode='keep_latest')

# Result contains:
# {
#   'deleted_checkpoints': [paths...],
#   'deleted_histories': [paths...],
#   'kept_checkpoint': path or None,
# }
```

### Recommended Workflow

For iterative MCMC workflows, the recommended pattern is:

1. **First run**: No checkpoint exists, starts fresh
2. **Subsequent runs**: Resume from latest checkpoint
3. **Before each session**: Clean with `mode='keep_latest'` to remove old histories but preserve resume capability

```python
from bamcmc import rmcmc, clean_model_files

# Clean old files but keep latest checkpoint
clean_model_files(output_dir, model_name, mode='keep_latest')

# Run MCMC - will resume if checkpoint exists, else start fresh
mcmc_config['resume_runs'] = 5  # 5 resume runs
summary = rmcmc(
    mcmc_config,
    data,
    output_dir=output_dir,
    burn_in_fresh=True,            # Only burn-in on fresh runs
)
```

**Key behavior**: With `mode='keep_latest'`, if no checkpoint exists, `resume` mode automatically starts fresh. This means you rarely need explicit `mode='all'` unless you want to discard learned posterior state.

## Output Directory Structure

The `get_model_paths()` function provides standardized paths for a model's output:

```python
from bamcmc.checkpoint_helpers import get_model_paths

paths = get_model_paths('../data/output/dbar_fed0', 'mix2_EH_bhm')
# Returns dict with:
#   'checkpoints': Path to checkpoint directory
#   'history_full': Path to full history files
#   'history_per_subject': Path to split per-subject files
```

**Directory Structure:**

```
{output_dir}/{model_name}/
├── checkpoints/
│   └── checkpoint_NNN.npz
└── history/
    ├── full/
    │   └── history_NNN.npz
    └── per_subject/
        ├── hyperparameters/
        │   └── history_NNN.npz
        ├── subject_000/
        │   └── history_NNN.npz
        └── subject_NNN/
```

## Post-Processing for Memory-Efficient Analysis

For large models (e.g., 245 subjects with 800 chains), full history files can be 35GB+. Post-processing splits these into per-subject files for memory-efficient analysis.

### split_history_by_subject

Split a single history file:

```python
from bamcmc.checkpoint_helpers import split_history_by_subject

result = split_history_by_subject(
    history_path='history/full/history_000.npz',
    output_base='history/per_subject',
    n_subjects=245,
    n_hyper=18,
    params_per_subject=13,
    hyper_first=False,  # Standard BHM layout
)
```

### postprocess_all_histories

Batch process all history files for a model:

```python
from bamcmc import postprocess_all_histories

result = postprocess_all_histories(
    output_dir='../data/output/dbar_fed0',
    model_name='mix2_EH_bhm',
    n_subjects=245,
    n_hyper=18,
    params_per_subject=13,
    hyper_first=False,
)
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `output_dir` | str | Base output directory |
| `model_name` | str | Model name |
| `n_subjects` | int | Number of subjects |
| `n_hyper` | int | Number of hyperparameters |
| `params_per_subject` | int | Parameters per subject |
| `hyper_first` | bool | Parameter layout (see below) |

## Parameter Layout

The `hyper_first` parameter controls the expected layout of the state vector:

### `hyper_first=False` (Default - Standard BHM Layout)

```
[subject_0 params][subject_1 params]...[subject_N params][hyperparameters][generated_quantities]
```

This is the standard layout for Bayesian Hierarchical Models where each subject's parameters are stored first, followed by population-level hyperparameters.

**Example for mix2_EH_bhm (245 subjects):**
- Indices 0-3184: Subject params (245 x 13)
- Indices 3185-3202: Hyperparameters (18)
- Indices 3203+: Generated quantities

### `hyper_first=True`

```
[hyperparameters][subject_0 params][subject_1 params]...[subject_N params][generated_quantities]
```

Use this layout if your model stores hyperparameters at the beginning of the state vector.

## Output Files

After post-processing, per-subject files contain:

**Hyperparameter files** (`hyperparameters/history_NNN.npz`):
- `history`: Shape `(n_samples, n_chains, n_hyper)`
- `iterations`: Iteration numbers
- `K`, `M`: Superchain configuration
- `mcmc_config`: Full MCMC configuration
- `temperature_history`: For parallel tempering

**Subject files** (`subject_NNN/history_NNN.npz`):
- `history`: Shape `(n_samples, n_chains, params_per_subject)`
- `iterations`: Iteration numbers
- `subject_idx`: Subject index

## Memory Savings

For a typical mix2_EH_bhm run with 245 subjects, 800 chains, 20 batches:

| File Type | Per-File Size | Total |
|-----------|--------------|-------|
| Full history | ~1.8 GB | ~36 GB |
| Hyperparameter split | ~12 MB | ~240 MB |
| Per-subject split | ~8 MB | ~40 GB |

The total split size is similar, but analysis can load just what's needed:
- Hyperparameter analysis: ~240 MB instead of ~36 GB
- Per-subject analysis: ~160 MB per subject instead of ~36 GB

## Integration with Analysis

After post-processing, use `run_analysis_from_split()` in `plot_code.common`:

```python
from plot_code.common import run_analysis_from_split

run_analysis_from_split(
    output_dir='../data/output/dbar_fed0',
    model_name='mix2_EH_bhm',
    n_subjects=245,
    post_id='mix2_EH_bhm',
    min_iteration=0,
    skip_hyper=False,
    skip_subjects=False,
    dataset='dbar_fed0',
)
```

This loads hyperparameters once (~240 MB), then processes each subject one at a time, keeping peak memory usage under 2 GB.
