"""
Output directory and file management utilities.

This module provides functions for:
- Scanning for existing checkpoints and history files
- Managing model output directory structure
- Cleaning up model output files
"""

import re
from pathlib import Path

import logging
logger = logging.getLogger('bamcmc')


def scan_checkpoints(output_dir: str, model_name: str):
    """
    Scan output directory for existing checkpoints and history files.

    Supports two directory structures:
    1. New nested structure (preferred):
       {output_dir}/{model_name}/checkpoints/checkpoint_NNN.npz
       {output_dir}/{model_name}/history/full/history_NNN.npz

    2. Legacy flat structure (for backwards compatibility):
       {output_dir}/{model_name}_checkpoint{N}.npz
       {output_dir}/{model_name}_history_{NNN}.npz

    Args:
        output_dir: Base output directory
        model_name: Model name

    Returns:
        Dict with:
            - checkpoint_files: List of (run_index, path) tuples, sorted by index
            - history_files: List of (run_index, path) tuples, sorted by index
            - latest_run_index: Highest run index found (-1 if none)
            - latest_checkpoint: Path to most recent checkpoint (None if none)
            - structure: 'nested' or 'flat' indicating which structure was found
    """
    output_path = Path(output_dir)

    checkpoint_files = []
    history_files = []
    structure = 'flat'  # Default to flat structure

    # Check for new nested structure first
    nested_base = output_path / model_name
    nested_checkpoints = nested_base / 'checkpoints'
    nested_history = nested_base / 'history' / 'full'

    if nested_checkpoints.exists() or nested_history.exists():
        structure = 'nested'

        # Scan nested checkpoint directory
        if nested_checkpoints.exists():
            # Pattern: checkpoint_NNN.npz
            checkpoint_pattern = re.compile(r'^checkpoint_(\d+)\.npz$')
            for f in nested_checkpoints.iterdir():
                if f.is_file():
                    cp_match = checkpoint_pattern.match(f.name)
                    if cp_match:
                        run_idx = int(cp_match.group(1))
                        checkpoint_files.append((run_idx, str(f)))

        # Scan nested history/full directory
        if nested_history.exists():
            # Pattern: history_NNN.npz
            history_pattern = re.compile(r'^history_(\d+)\.npz$')
            for f in nested_history.iterdir():
                if f.is_file():
                    hist_match = history_pattern.match(f.name)
                    if hist_match:
                        run_idx = int(hist_match.group(1))
                        history_files.append((run_idx, str(f)))

    # Fall back to legacy flat structure
    if not checkpoint_files and not history_files and output_path.exists():
        structure = 'flat'
        # Pattern: {model}_checkpoint{N}.npz or {model}_checkpoint{N}_suffix.npz
        checkpoint_pattern = re.compile(rf'^{re.escape(model_name)}_checkpoint(\d+)(_[a-z]+)?\.npz$')
        # Pattern: {model}_history_{NNN}.npz
        history_pattern = re.compile(rf'^{re.escape(model_name)}_history_(\d+)\.npz$')

        for f in output_path.iterdir():
            if f.is_file():
                cp_match = checkpoint_pattern.match(f.name)
                if cp_match:
                    run_idx = int(cp_match.group(1))
                    checkpoint_files.append((run_idx, str(f)))

                hist_match = history_pattern.match(f.name)
                if hist_match:
                    run_idx = int(hist_match.group(1))
                    history_files.append((run_idx, str(f)))

    # Sort by run index
    checkpoint_files.sort(key=lambda x: x[0])
    history_files.sort(key=lambda x: x[0])

    latest_run_index = checkpoint_files[-1][0] if checkpoint_files else -1
    latest_checkpoint = checkpoint_files[-1][1] if checkpoint_files else None

    return {
        'checkpoint_files': checkpoint_files,
        'history_files': history_files,
        'latest_run_index': latest_run_index,
        'latest_checkpoint': latest_checkpoint,
        'structure': structure,
    }


def get_model_paths(output_dir: str, model_name: str):
    """
    Get directory paths for the nested structure.

    Args:
        output_dir: Base output directory (e.g., '../data/output/dbar_fed0')
        model_name: Model name (e.g., 'mix2_EH_bhm')

    Returns:
        Dict with paths:
            - base: {output_dir}/{model_name}/
            - checkpoints: {base}/checkpoints/
            - history_full: {base}/history/full/
            - history_per_subject: {base}/history/per_subject/
    """
    base = Path(output_dir) / model_name
    return {
        'base': base,
        'checkpoints': base / 'checkpoints',
        'history_full': base / 'history' / 'full',
        'history_per_subject': base / 'history' / 'per_subject',
    }


def ensure_model_dirs(output_dir: str, model_name: str):
    """
    Create the nested directory structure for a model.

    Args:
        output_dir: Base output directory
        model_name: Model name

    Returns:
        Dict with paths (same as get_model_paths)
    """
    paths = get_model_paths(output_dir, model_name)
    paths['checkpoints'].mkdir(parents=True, exist_ok=True)
    paths['history_full'].mkdir(parents=True, exist_ok=True)
    paths['history_per_subject'].mkdir(parents=True, exist_ok=True)
    return paths


def get_latest_checkpoint(output_dir: str, model_name: str):
    """
    Get path to the most recent checkpoint for a model.

    Args:
        output_dir: Directory containing checkpoint files
        model_name: Model name prefix

    Returns:
        Path to latest checkpoint, or None if no checkpoints exist
    """
    scan = scan_checkpoints(output_dir, model_name)
    return scan['latest_checkpoint']


def clean_model_files(output_dir: str, model_name: str, mode: str = 'all'):
    """
    Delete checkpoint and history files for a model.

    Handles both nested and flat directory structures.

    Args:
        output_dir: Directory containing the files
        model_name: Model name prefix
        mode: Cleaning mode:
            - 'all': Delete all checkpoints and histories
            - 'keep_latest': Keep latest checkpoint, delete all else

    Returns:
        dict with 'deleted_checkpoints', 'deleted_histories', 'kept_checkpoint'
    """
    import shutil

    scan = scan_checkpoints(output_dir, model_name)

    deleted_checkpoints = []
    deleted_histories = []
    kept_checkpoint = None

    # Determine which checkpoint to keep (if any)
    if mode == 'keep_latest' and scan['latest_checkpoint']:
        kept_checkpoint = scan['latest_checkpoint']

    # Delete checkpoints
    for run_idx, cp_path in scan['checkpoint_files']:
        if cp_path != kept_checkpoint:
            try:
                Path(cp_path).unlink()
                deleted_checkpoints.append(cp_path)
            except OSError as e:
                logger.warning(f"Could not delete {cp_path}: {e}")

    # Delete all histories (from full/)
    for run_idx, hist_path in scan['history_files']:
        try:
            Path(hist_path).unlink()
            deleted_histories.append(hist_path)
        except OSError as e:
            logger.warning(f"Could not delete {hist_path}: {e}")

    # For nested structure, also clean per_subject directory
    if scan.get('structure') == 'nested':
        paths = get_model_paths(output_dir, model_name)
        per_subject_dir = paths['history_per_subject']
        if per_subject_dir.exists():
            try:
                shutil.rmtree(per_subject_dir)
                per_subject_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(f"Could not clean per_subject directory: {e}")

    return {
        'deleted_checkpoints': deleted_checkpoints,
        'deleted_histories': deleted_histories,
        'kept_checkpoint': kept_checkpoint,
    }
