"""
MCMC Backend - Multi-run orchestrator.

This module provides the main rmcmc() function for running multi-run MCMC
sampling with automatic checkpoint management. The single-run engine
(rmcmc_single) and its helpers live in mcmc.single_run.

Module layout:
- backend: Multi-run orchestrator (rmcmc)
- single_run: Single-run engine (rmcmc_single) and helpers
- compile: Kernel compilation and caching
- config: Configuration and initialization
- diagnostics: Convergence diagnostics
- sampling: Proposal and MH sampling functions
- scan: JAX scan body and block statistics
- types: Core data structures (BlockArrays, RunParams)
"""

import gc
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Import single-run engine and its public helpers
from .single_run import rmcmc_single, _validate_checkpoint_compatibility

# Import types (re-exported for backwards compatibility)
from .types import BlockArrays, RunParams, build_block_arrays

# Import commonly used config functions (re-exported)
from .config import (
    configure_mcmc_system,
    initialize_mcmc_system,
    validate_mcmc_inputs,
)

# Import diagnostics (re-exported)
from .diagnostics import compute_nested_rhat

# Import checkpoint utilities
from ..checkpoint_helpers import (
    scan_checkpoints,
    ensure_model_dirs,
    get_model_paths,
)

# Public API for this module
__all__ = [
    'rmcmc',
    'rmcmc_single',
    '_validate_checkpoint_compatibility',
    'BlockArrays',
    'RunParams',
    'build_block_arrays',
    'configure_mcmc_system',
    'initialize_mcmc_system',
    'validate_mcmc_inputs',
    'compute_nested_rhat',
]


# =============================================================================
# MULTI-RUN SAMPLING FUNCTION
# =============================================================================

def rmcmc(
    mcmc_config: Dict[str, Any],
    data: Dict[str, Any],
    output_dir: str,
    run_schedule: List[Tuple[str, int]],
    calculate_rhat: bool = True,
    burn_in_fresh: bool = True,
    reset_noise_scale: float = 1.0,
    use_nested_structure: bool = True,
) -> Dict[str, Any]:
    """
    Run MCMC sampling with automatic multi-run and checkpoint management.

    This is the main entry point for MCMC sampling. It handles:
    - Automatic detection of existing checkpoints
    - Multiple sequential runs with resume/reset modes
    - Fresh runs when no checkpoint exists
    - Optional burn-in only for fresh runs

    Args:
        mcmc_config: MCMC configuration dict (see rmcmc_single for details)
        data: Data dict with 'static', 'int', 'float' arrays
        output_dir: Directory for checkpoint and history files
        run_schedule: List of (mode, count) tuples specifying the run sequence.
            Modes:
            - "reset": Reset chains to cross-chain mean + noise (or fresh if no checkpoint)
            - "resume": Resume from exact checkpoint state (or fresh if no checkpoint)
            Examples:
            - [("reset", 10)]: 10 reset runs (first is fresh if no checkpoint)
            - [("resume", 5)]: 5 resume runs (first is fresh if no checkpoint)
            - [("reset", 10), ("resume", 5)]: 10 reset runs then 5 resume runs
        calculate_rhat: Whether to compute R-hat diagnostics each run
        burn_in_fresh: If True, only apply burn_iter to fresh runs (default True)
        reset_noise_scale: Scale factor for noise when resetting (default 1.0)
        use_nested_structure: If True, use nested directory structure (default True):
            {output_dir}/{model}/checkpoints/ and {output_dir}/{model}/history/full/

    Returns:
        Summary dict with:
            - output_dir: Output directory path
            - model_name: Model name from config
            - checkpoint_files: List of (run_index, path) tuples
            - history_files: List of (run_index, path) tuples
            - latest_run_index: Index of the last completed run
            - latest_checkpoint: Path to the most recent checkpoint
            - total_runs_completed: Number of runs completed in this session
            - final_iteration: Final iteration number
            - run_log: List of dicts with info about each run

    Example:
        from bamcmc import rmcmc

        summary = rmcmc(
            mcmc_config,
            data,
            output_dir='./output',
            run_schedule=[("reset", 10), ("resume", 5)],
        )
        print(f"Completed {summary['total_runs_completed']} runs")
        print(f"History files: {summary['history_files']}")
    """
    model_name = mcmc_config['posterior_id']

    # Set up directory structure and path constructors
    if use_nested_structure:
        # Create nested directories: {output_dir}/{model}/checkpoints/, history/full/
        paths = ensure_model_dirs(output_dir, model_name)
        checkpoint_dir = paths['checkpoints']
        history_dir = paths['history_full']

        def checkpoint_path(num):
            return str(checkpoint_dir / f"checkpoint_{num:03d}.npz")

        def history_path(num):
            return str(history_dir / f"history_{num:03d}.npz")
    else:
        # Legacy flat structure
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        def checkpoint_path(num):
            return f"{output_dir}/{model_name}_checkpoint{num}.npz"

        def history_path(num):
            return f"{output_dir}/{model_name}_history_{num:03d}.npz"

    # Scan for existing checkpoints
    scan = scan_checkpoints(output_dir, model_name)
    existing_run_index = scan['latest_run_index']
    existing_checkpoint = scan['latest_checkpoint']

    if existing_checkpoint:
        print(f"Found existing checkpoint: {existing_checkpoint} (run {existing_run_index})")
    else:
        print(f"No existing checkpoints found for {model_name}")

    # Expand run schedule into list of (run_index, mode) pairs
    run_list = []
    for mode, count in run_schedule:
        if mode not in ('reset', 'resume'):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'reset' or 'resume'.")
        for _ in range(count):
            run_list.append(mode)

    if not run_list:
        print("Empty run schedule, nothing to do")
        return {
            'output_dir': output_dir,
            'model_name': model_name,
            **scan,
            'total_runs_completed': 0,
            'final_iteration': None,
            'run_log': [],
        }

    # Determine starting run index
    start_run_index = existing_run_index + 1
    total_runs = len(run_list)

    # Format schedule for display
    schedule_parts = [f"{count} {mode}" for mode, count in run_schedule]
    schedule_str = " + ".join(schedule_parts)

    print(f"\n{'='*60}")
    print(f"MULTI-RUN MCMC: {model_name}")
    print(f"Schedule: {schedule_str} ({total_runs} total runs)")
    print(f"Starting from run index {start_run_index}")
    print(f"{'='*60}\n")

    # Track run information
    run_log = []
    total_runs_completed = 0
    final_iteration = None
    original_burn_iter = mcmc_config.get('burn_iter', 0)

    # New indexing scheme:
    # - checkpoint_0 = initial state (iteration=0), saved for fresh/reset runs
    # - checkpoint_N = state after Nth sampling run
    # - run_index in schedule = which sampling run (1-indexed for checkpoints)

    for i, scheduled_mode in enumerate(run_list):
        sampling_run_num = start_run_index + i  # 0-indexed sampling run number
        output_checkpoint_num = sampling_run_num + 1  # checkpoint_1 is after 1st sampling run

        # Check if this run already completed (output checkpoint exists)
        output_checkpoint_file = checkpoint_path(output_checkpoint_num)
        if Path(output_checkpoint_file).exists():
            print(f"\n--- Run {i + 1}/{total_runs} (overall #{sampling_run_num + 1}) already complete (checkpoint_{output_checkpoint_num:03d} exists), skipping ---")
            run_log.append({
                'run_index': sampling_run_num,
                'mode': 'skipped',
                'checkpoint': output_checkpoint_file,
            })
            continue

        # Determine input checkpoint (what to resume/reset from)
        input_checkpoint_num = output_checkpoint_num - 1  # checkpoint_0 for first run
        input_checkpoint_file = checkpoint_path(input_checkpoint_num)
        has_input_checkpoint = Path(input_checkpoint_file).exists()

        # If input checkpoint doesn't exist, search for latest available
        if not has_input_checkpoint and input_checkpoint_num > 0:
            for idx in range(input_checkpoint_num - 1, -1, -1):
                cp = checkpoint_path(idx)
                if Path(cp).exists():
                    input_checkpoint_file = cp
                    has_input_checkpoint = True
                    print(f"  Note: Using checkpoint_{idx:03d} (checkpoint_{input_checkpoint_num:03d} not found)")
                    break

        # Determine mode and whether we need to save initial state
        save_initial_to = None
        if not has_input_checkpoint:
            actual_mode = 'fresh'
            resume_from = None
            reset_from = None
            # Save initial state as checkpoint_000
            save_initial_to = checkpoint_path(0)
        elif scheduled_mode == 'resume':
            actual_mode = 'resume'
            resume_from = input_checkpoint_file
            reset_from = None
        else:  # reset
            actual_mode = 'reset'
            resume_from = None
            reset_from = input_checkpoint_file
            # Save initial (reset) state before sampling
            # For nested structure, just overwrite the checkpoint; for flat, add _reset suffix
            if use_nested_structure:
                save_initial_to = None  # Don't save reset state separately in nested mode
            else:
                save_initial_to = f"{output_dir}/{model_name}_checkpoint{input_checkpoint_num}_reset.npz"

        print(f"\n{'='*60}")
        print(f"Sampling run {i + 1}/{total_runs} (overall #{sampling_run_num + 1}, mode={actual_mode})")
        print(f"  Output: checkpoint_{output_checkpoint_num}")
        print(f"{'='*60}")

        # Handle burn_in_fresh option
        run_config = mcmc_config.copy()
        if burn_in_fresh and actual_mode != 'fresh':
            run_config['burn_iter'] = 0
            if original_burn_iter > 0:
                print(f"  burn_in_fresh=True: skipping burn-in ({actual_mode} run)")
        else:
            run_config['burn_iter'] = original_burn_iter

        # Run single iteration
        try:
            results, checkpoint = rmcmc_single(
                run_config,
                data,
                calculate_rhat=calculate_rhat,
                resume_from=resume_from,
                reset_from=reset_from,
                reset_noise_scale=reset_noise_scale,
                save_initial_to=save_initial_to,
            )
        except Exception as e:
            print(f"\nError during run {i + 1}/{total_runs} (overall #{sampling_run_num + 1}): {e}")
            raise

        # Save output checkpoint
        np.savez_compressed(output_checkpoint_file, **checkpoint)
        final_iteration = checkpoint['iteration']
        print(f"\nCheckpoint saved: {output_checkpoint_file} (iteration {final_iteration})")

        # Save history if we collected samples
        history = results.get('history')
        history_file = None
        if history is not None and len(history) > 0:
            history_file = history_path(sampling_run_num)
            np.savez_compressed(history_file, **results, run_index=sampling_run_num)
            print(f"History saved: {history_file}")

        run_log.append({
            'run_index': sampling_run_num,
            'mode': actual_mode,
            'checkpoint': output_checkpoint_file,
            'initial_checkpoint': save_initial_to,
            'history': history_file,
            'iteration': final_iteration,
            'rhat_max': np.nanmax(results['diagnostics'].get('rhat', [np.nan])),
        })
        total_runs_completed += 1

        # Clean up memory between runs
        gc.collect()

    print(f"\n{'='*60}")
    print(f"MULTI-RUN COMPLETE: {total_runs_completed} runs")
    print(f"{'='*60}\n")

    # Scan for final state
    final_scan = scan_checkpoints(output_dir, model_name)

    return {
        'output_dir': output_dir,
        'model_name': model_name,
        **final_scan,
        'total_runs_completed': total_runs_completed,
        'final_iteration': final_iteration,
        'run_log': run_log,
    }
