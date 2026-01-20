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

from ..batch_specs import BlockSpec, MAX_PROPOSAL_GROUPS
from ..settings import build_settings_matrix, MAX_SETTINGS, SETTING_DEFAULTS, SettingSlot


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

    Mixed Proposals:
    For blocks with proposal_groups, the group_* arrays store per-group info.
    Each block has up to MAX_PROPOSAL_GROUPS groups. Invalid groups have
    group_masks[block, group] = 0.

    The proposal dispatch computes proposals for each group independently,
    then combines the Hastings ratios by multiplication (addition in log space).
    """
    # Basic block info
    indices: jnp.ndarray         # (n_blocks, max_block_size) - parameter indices per block
    types: jnp.ndarray           # (n_blocks,) - SamplerType for each block
    masks: jnp.ndarray           # (n_blocks, max_block_size) - valid parameter mask

    # Single-proposal fields (for backward compatibility and simple blocks)
    proposal_types: jnp.ndarray  # (n_blocks,) - REMAPPED indices for first/only group
    settings_matrix: jnp.ndarray # (n_blocks, MAX_SETTINGS) - settings for first/only group

    # Mixed-proposal group fields
    group_starts: jnp.ndarray        # (n_blocks, MAX_PROPOSAL_GROUPS) - start indices within block
    group_ends: jnp.ndarray          # (n_blocks, MAX_PROPOSAL_GROUPS) - end indices within block
    group_proposal_types: jnp.ndarray  # (n_blocks, MAX_PROPOSAL_GROUPS) - remapped proposal types
    group_settings: jnp.ndarray      # (n_blocks, MAX_PROPOSAL_GROUPS, MAX_SETTINGS)
    group_masks: jnp.ndarray         # (n_blocks, MAX_PROPOSAL_GROUPS) - valid group mask
    num_groups: jnp.ndarray          # (n_blocks,) - number of valid groups per block

    # Metadata
    max_size: int                # Maximum block size
    num_blocks: int              # Number of blocks
    total_params: int            # Total parameter count
    used_proposal_types: tuple   # Original ProposalType values actually used (in dispatch order)
    has_mixed_proposals: bool    # True if any block has multiple proposal groups


def _block_arrays_flatten(ba):
    """Flatten BlockArrays for JAX pytree."""
    # Arrays are children (traced), scalars/tuples/bools are auxiliary data (static)
    children = (
        ba.indices, ba.types, ba.masks, ba.proposal_types, ba.settings_matrix,
        ba.group_starts, ba.group_ends, ba.group_proposal_types, ba.group_settings,
        ba.group_masks, ba.num_groups
    )
    aux_data = (ba.max_size, ba.num_blocks, ba.total_params, ba.used_proposal_types,
                ba.has_mixed_proposals)
    return children, aux_data


def _block_arrays_unflatten(aux_data, children):
    """Unflatten BlockArrays from JAX pytree."""
    (indices, types, masks, proposal_types, settings_matrix,
     group_starts, group_ends, group_proposal_types, group_settings,
     group_masks, num_groups) = children
    (max_size, num_blocks, total_params, used_proposal_types,
     has_mixed_proposals) = aux_data
    return BlockArrays(
        indices=indices,
        types=types,
        masks=masks,
        proposal_types=proposal_types,
        settings_matrix=settings_matrix,
        group_starts=group_starts,
        group_ends=group_ends,
        group_proposal_types=group_proposal_types,
        group_settings=group_settings,
        group_masks=group_masks,
        num_groups=num_groups,
        max_size=max_size,
        num_blocks=num_blocks,
        total_params=total_params,
        used_proposal_types=used_proposal_types,
        has_mixed_proposals=has_mixed_proposals,
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
    N_CHAINS_TO_SAVE: int  # Number of chains to save (beta=1 chains when tempering)
    PER_TEMP_PROPOSALS: bool = True  # Per-temperature proposal statistics
    N_TEMPERATURES: int = 1  # Number of temperature levels
    USE_DEO: bool = True  # Use DEO scheme for parallel tempering swaps


def build_block_arrays(specs: List[BlockSpec], start_idx: int = 0) -> BlockArrays:
    """
    Build BlockArrays from a list of BlockSpec objects.

    Args:
        specs: List of BlockSpec objects defining parameter blocks
        start_idx: Starting parameter index (default 0)

    Returns:
        BlockArrays with all arrays ready for MCMC backend

    Handles both single-proposal blocks (backward compatible) and
    mixed-proposal blocks (using proposal_groups).
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

    # Collect unique proposal types from ALL groups (including mixed proposals)
    used_types_set = set()
    has_mixed = False
    for spec in specs:
        if spec.is_mh_sampler():
            groups = spec.get_effective_groups()
            if len(groups) > 1:
                has_mixed = True
            for group in groups:
                used_types_set.add(int(group.proposal_type))

    # Create sorted list for consistent ordering (becomes the compact dispatch table order)
    used_types_list = tuple(sorted(used_types_set))

    # Create mapping: original ProposalType value -> compact index
    type_to_compact = {t: i for i, t in enumerate(used_types_list)} if used_types_list else {}

    # Build arrays for groups
    group_starts = np.zeros((num_blocks, MAX_PROPOSAL_GROUPS), dtype=np.int32)
    group_ends = np.zeros((num_blocks, MAX_PROPOSAL_GROUPS), dtype=np.int32)
    group_proposal_types = np.zeros((num_blocks, MAX_PROPOSAL_GROUPS), dtype=np.int32)
    group_settings = np.zeros((num_blocks, MAX_PROPOSAL_GROUPS, MAX_SETTINGS), dtype=np.float32)
    group_masks = np.zeros((num_blocks, MAX_PROPOSAL_GROUPS), dtype=np.float32)
    num_groups_arr = np.zeros(num_blocks, dtype=np.int32)

    # Build remapped proposal_types and group arrays
    proposal_types = []
    for i, spec in enumerate(specs):
        if spec.is_mh_sampler():
            groups = spec.get_effective_groups()
            num_groups_arr[i] = len(groups)

            # First group's proposal type goes in legacy proposal_types array
            first_group = groups[0]
            proposal_types.append(type_to_compact[int(first_group.proposal_type)])

            # Fill in all group arrays
            for g, group in enumerate(groups):
                if g >= MAX_PROPOSAL_GROUPS:
                    break
                group_starts[i, g] = group.start
                group_ends[i, g] = group.end
                group_proposal_types[i, g] = type_to_compact[int(group.proposal_type)]
                group_masks[i, g] = 1.0

                # Build settings for this group
                for key, value in group.settings.items():
                    if hasattr(SettingSlot, key.upper()):
                        slot = getattr(SettingSlot, key.upper())
                        group_settings[i, g, slot] = float(value)
                # Fill defaults for unset settings
                for slot, default in SETTING_DEFAULTS.items():
                    if group_settings[i, g, slot] == 0.0:
                        group_settings[i, g, slot] = default
        else:
            proposal_types.append(0)  # Unused for direct samplers
            num_groups_arr[i] = 0

    return BlockArrays(
        indices=jnp.array(indices),
        types=jnp.array(types),
        masks=jnp.array(masks),
        proposal_types=jnp.array(proposal_types, dtype=jnp.int32),
        settings_matrix=build_settings_matrix(specs),
        group_starts=jnp.array(group_starts),
        group_ends=jnp.array(group_ends),
        group_proposal_types=jnp.array(group_proposal_types, dtype=jnp.int32),
        group_settings=jnp.array(group_settings),
        group_masks=jnp.array(group_masks),
        num_groups=jnp.array(num_groups_arr),
        max_size=max_size,
        num_blocks=num_blocks,
        total_params=current_param,
        used_proposal_types=used_types_list,
        has_mixed_proposals=has_mixed,
    )
