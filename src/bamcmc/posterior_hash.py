"""
Posterior Hashing - Unique identification of posterior configurations.

This module provides functions to compute content-based hashes for posterior
models. A hash uniquely identifies a (posterior code + data + chain count)
combination, enabling benchmark caching across sessions.

Functions:
- get_function_source_hash: Hash a function's source code
- get_data_hash: Hash the MCMCData structure (shapes + content)
- get_posterior_hash: Hash a full posterior configuration
- compute_posterior_hash: Convenience alias for get_posterior_hash
"""

import hashlib
import inspect
from typing import Dict, Optional

import numpy as np


def get_function_source_hash(func) -> str:
    """
    Get a hash of a function's source code.

    This captures the actual implementation, so code changes will produce
    different hashes even if the function signature is the same.
    """
    if func is None:
        return "none"

    try:
        source = inspect.getsource(func)
        return hashlib.sha256(source.encode()).hexdigest()[:16]
    except (TypeError, OSError):
        # Can't get source (built-in, C extension, lambda, etc.)
        # Fall back to a representation-based hash
        return hashlib.sha256(repr(func).encode()).hexdigest()[:16]


def get_data_hash(data: Dict) -> str:
    """
    Get a hash of the data structure.

    Hashes both shapes and content for exact reproducibility.
    The same data will always produce the same hash.
    """
    hasher = hashlib.sha256()

    # Hash static values
    if 'static' in data:
        static_vals = data['static']
        if isinstance(static_vals, (list, tuple)):
            hasher.update(f"static:{tuple(static_vals)}".encode())
        else:
            hasher.update(f"static:{static_vals}".encode())

    # Hash int arrays (shapes and content)
    if 'int' in data:
        for i, arr in enumerate(data['int']):
            arr_np = np.asarray(arr)
            hasher.update(f"int_{i}_shape:{arr_np.shape}_dtype:{arr_np.dtype}".encode())
            hasher.update(arr_np.tobytes())

    # Hash float arrays (shapes and content)
    if 'float' in data:
        for i, arr in enumerate(data['float']):
            arr_np = np.asarray(arr)
            hasher.update(f"float_{i}_shape:{arr_np.shape}_dtype:{arr_np.dtype}".encode())
            # Round floats to avoid floating point representation noise
            arr_rounded = np.round(arr_np.astype(np.float64), decimals=10)
            hasher.update(arr_rounded.tobytes())

    return hasher.hexdigest()[:16]


def get_posterior_hash(posterior_id: str, model_config: Dict, data: Dict, num_chains: int = None) -> str:
    """
    Compute a unique hash for a posterior configuration.

    Combines:
    - Posterior ID (name)
    - Source code of all registered functions
    - Data structure (shapes and content)
    - Number of chains (affects benchmark timing)

    The same posterior code + same data + same chains = same hash, guaranteed.

    Args:
        posterior_id: The registered name of the posterior
        model_config: The model config dict from the registry
        data: The data dict with 'static', 'int', 'float' keys
        num_chains: Number of MCMC chains (included in hash for benchmark accuracy)

    Returns:
        16-character hex hash that uniquely identifies this configuration
    """
    hasher = hashlib.sha256()

    # Posterior identifier
    hasher.update(f"posterior_id:{posterior_id}".encode())

    # Number of chains (important for benchmark timing)
    if num_chains is not None:
        hasher.update(f"num_chains:{num_chains}".encode())

    # Hash each registered function's source code
    function_keys = [
        'log_posterior',
        'direct_sampler',
        'generated_quantities',
        'initial_vector',
        'batch_type',
        'get_num_gq'
    ]

    for key in function_keys:
        func = model_config.get(key)
        func_hash = get_function_source_hash(func)
        hasher.update(f"{key}:{func_hash}".encode())

    # Hash data
    data_hash = get_data_hash(data)
    hasher.update(f"data:{data_hash}".encode())

    return hasher.hexdigest()[:16]


def compute_posterior_hash(posterior_id: str, model_config: Dict, data: Dict, num_chains: int = None) -> str:
    """
    Convenience function to compute posterior hash.

    See get_posterior_hash for details.
    """
    return get_posterior_hash(posterior_id, model_config, data, num_chains)
