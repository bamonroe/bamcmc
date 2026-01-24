"""
Prior configuration utilities for JAX-free plotting.

This module provides functions for saving and loading prior configuration
to/from JSON files, enabling plotting code to work without importing
posterior modules (which would trigger JAX initialization).
"""

import json
from pathlib import Path


def save_prior_config(output_dir: str, model_name: str, prior_config: dict):
    """
    Save prior configuration to JSON for JAX-free plotting.

    This allows plotting code to load prior parameters without importing
    the posterior module (which would trigger JAX initialization and GPU
    memory allocation).

    Args:
        output_dir: Base output directory (e.g., ../data/output/dbar_fed0)
        model_name: Model name
        prior_config: Dict containing prior parameters. Expected structure:
            {
                'shared_risk_priors': {...},
                'discount_priors': {...},
                'mixing_prior': {...},
            }

    Returns:
        Path to saved config file
    """
    # Import here to avoid circular dependency
    from .output_management import get_model_paths

    paths = get_model_paths(output_dir, model_name)
    config_path = paths['base'] / 'prior_config.json'

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        return obj

    serializable_config = convert_for_json(prior_config)

    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)

    print(f"Prior config saved: {config_path}", flush=True)
    return str(config_path)


def load_prior_config(output_dir: str, model_name: str):
    """
    Load prior configuration from JSON (JAX-free).

    Args:
        output_dir: Base output directory
        model_name: Model name

    Returns:
        Dict with prior config, or None if file doesn't exist
    """
    # Import here to avoid circular dependency
    from .output_management import get_model_paths

    paths = get_model_paths(output_dir, model_name)
    config_path = paths['base'] / 'prior_config.json'

    if not config_path.exists():
        return None

    with open(config_path, 'r') as f:
        return json.load(f)
