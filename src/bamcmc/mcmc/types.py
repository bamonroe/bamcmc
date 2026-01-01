"""
MCMC Data Structures and Type Definitions.

This module contains the core data structures used by the MCMC backend:
- BlockArrays: Pre-parsed block specification arrays
- RunParams: Immutable run parameters for JAX static arguments
- build_block_arrays: Factory function for BlockArrays
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import List

from ..batch_specs import BlockSpec
from ..settings import build_settings_matrix


@dataclass(frozen=True)
class BlockArrays:
    """
    Pre-parsed block specification arrays for MCMC backend.

    Groups related block data to reduce function parameter counts.
    All arrays are JAX arrays ready for use in the sampling loop.

    This is a frozen dataclass for immutability and JAX compatibility.
    Registered as a JAX pytree to enable tracing through JIT.

    Note on proposal_types: These are REMAPPED indices into a compact dispatch
    table containing only the proposals actually used by this model. This ensures
    JAX only traces the proposals that are needed, not all available proposals.
    The used_proposal_types tuple stores the original ProposalType enum values
    in dispatch order.
    """
    indices: jnp.ndarray         # (n_blocks, max_block_size) - parameter indices per block
    types: jnp.ndarray           # (n_blocks,) - SamplerType for each block
    masks: jnp.ndarray           # (n_blocks, max_block_size) - valid parameter mask
    proposal_types: jnp.ndarray  # (n_blocks,) - REMAPPED indices into compact dispatch table
    settings_matrix: jnp.ndarray # (n_blocks, MAX_SETTINGS) - per-block settings
    max_size: int                # Maximum block size
    num_blocks: int              # Number of blocks
    total_params: int            # Total parameter count
    used_proposal_types: tuple   # Original ProposalType values actually used (in dispatch order)


def _block_arrays_flatten(ba):
    """Flatten BlockArrays for JAX pytree."""
    # Arrays are children (traced), scalars/tuples are auxiliary data (static)
    children = (ba.indices, ba.types, ba.masks, ba.proposal_types, ba.settings_matrix)
    aux_data = (ba.max_size, ba.num_blocks, ba.total_params, ba.used_proposal_types)
    return children, aux_data


def _block_arrays_unflatten(aux_data, children):
    """Unflatten BlockArrays from JAX pytree."""
    indices, types, masks, proposal_types, settings_matrix = children
    max_size, num_blocks, total_params, used_proposal_types = aux_data
    return BlockArrays(
        indices=indices,
        types=types,
        masks=masks,
        proposal_types=proposal_types,
        settings_matrix=settings_matrix,
        max_size=max_size,
        num_blocks=num_blocks,
        total_params=total_params,
        used_proposal_types=used_proposal_types,
    )


# Register BlockArrays as a JAX pytree
jax.tree_util.register_pytree_node(
    BlockArrays,
    _block_arrays_flatten,
    _block_arrays_unflatten
)


@dataclass(frozen=True)
class RunParams:
    """
    Immutable run parameters for JAX static argument compatibility.

    This frozen dataclass allows run parameters to be passed as static
    arguments to JIT-compiled functions, enabling cross-session caching.
    """
    BURN_ITER: int
    NUM_COLLECT: int
    THIN_ITERATION: int
    NUM_GQ: int
    START_ITERATION: int
    SAVE_LIKELIHOODS: bool


def build_block_arrays(specs: List[BlockSpec], start_idx: int = 0) -> BlockArrays:
    """
    Build BlockArrays from a list of BlockSpec objects.

    Args:
        specs: List of BlockSpec objects defining parameter blocks
        start_idx: Starting parameter index (default 0)

    Returns:
        BlockArrays with all arrays ready for MCMC backend
    """
    if not specs:
        raise ValueError("Empty block specifications")

    max_size = max(spec.size for spec in specs)
    num_blocks = len(specs)

    # Build index and mask arrays
    indices = np.full((num_blocks, max_size), -1, dtype=np.int32)
    types = np.zeros(num_blocks, dtype=np.int32)
    masks = np.zeros((num_blocks, max_size), dtype=np.float32)

    current_param = start_idx
    for i, spec in enumerate(specs):
        types[i] = int(spec.sampler_type)
        block_idxs = np.arange(current_param, current_param + spec.size)
        indices[i, :spec.size] = block_idxs
        masks[i, :spec.size] = 1.0
        current_param += spec.size

    # Collect unique proposal types actually used by MH blocks
    used_types_set = set()
    for spec in specs:
        if spec.is_mh_sampler():
            used_types_set.add(int(spec.proposal_type))

    # Create sorted list for consistent ordering (becomes the compact dispatch table order)
    used_types_list = tuple(sorted(used_types_set))

    # Create mapping: original ProposalType value -> compact index
    type_to_compact = {t: i for i, t in enumerate(used_types_list)}

    # Build remapped proposal_types (indices into compact dispatch table)
    proposal_types = []
    for spec in specs:
        if spec.is_mh_sampler():
            original_type = int(spec.proposal_type)
            proposal_types.append(type_to_compact[original_type])
        else:
            proposal_types.append(0)  # Unused for direct samplers

    return BlockArrays(
        indices=jnp.array(indices),
        types=jnp.array(types),
        masks=jnp.array(masks),
        proposal_types=jnp.array(proposal_types, dtype=jnp.int32),
        settings_matrix=build_settings_matrix(specs),
        max_size=max_size,
        num_blocks=num_blocks,
        total_params=current_param,
        used_proposal_types=used_types_list,
    )
