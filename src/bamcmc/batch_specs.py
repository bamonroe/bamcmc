"""
Batch Specification System

This module defines a clean, self-documenting way to specify parameter blocks
and their sampling strategies for MCMC.

Key improvements over the old (size, type) tuple system:
1. Self-documenting: BlockSpec clearly shows what each field means
2. Extensible: Easy to add new fields without breaking existing code
3. Type-safe: Using dataclasses catches errors at definition time
4. Flexible: Supports per-block settings and metadata
5. Mixed proposals: Different proposal types for sub-groups within a block
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from enum import IntEnum
import numpy as np


# Maximum number of proposal groups per block (for JAX fixed-shape arrays)
MAX_PROPOSAL_GROUPS = 4


# ============================================================================
# SAMPLER TYPE ENUMERATION
# ============================================================================

class SamplerType(IntEnum):
    """
    Enumeration of available sampler types.

    This replaces the old integer-based batch type system with named constants.
    """
    # Core samplers (currently implemented)
    METROPOLIS_HASTINGS = 0  # Standard MH with proposal distribution
    DIRECT_CONJUGATE = 1     # Direct sampling from conditional (e.g., Gibbs)
    COUPLED_TRANSFORM = 2    # MH with deterministic coupled parameter updates

    # Future samplers (reserved for implementation)
    ADAPTIVE_MH = 3          # MH with adaptive covariance tuning
    HMC = 4                  # Hamiltonian Monte Carlo
    NUTS = 5                 # No-U-Turn Sampler
    ELLIPTICAL_SLICE = 6     # For Gaussian priors
    SLICE_SAMPLER = 7        # Slice sampling
    DELAYED_ACCEPTANCE = 8   # Two-stage MH for expensive likelihoods

    def __str__(self):
        return self.name.replace('_', ' ').title()


# ============================================================================
# PROPOSAL TYPE ENUMERATION
# ============================================================================

class ProposalType(IntEnum):
    """
    Enumeration of proposal distribution strategies.

    This controls how the proposal distribution is constructed for MH samplers.
    All proposal implementations live in proposals.py.

    To add a new proposal:
    1. Add enum value here (e.g., LANGEVIN = 4)
    2. Go to proposals.py and implement the proposal function
    3. Add it to the dispatch table in create_proposal_dispatch_table()
    """
    # Currently implemented
    SELF_MEAN = 0      # Random walk: center proposal on current state
    CHAIN_MEAN = 1     # Independent: center proposal on population mean
    MIXTURE = 2        # Mixture: with prob alpha use CHAIN_MEAN, else SELF_MEAN
    MULTINOMIAL = 3    # Discrete: sample from empirical distribution on grid
    MALA = 4           # Metropolis-adjusted Langevin (gradient-based, preconditioned)
    MEAN_MALA = 5      # Chain-mean MALA: gradient at coupled mean, independent proposal
    MEAN_WEIGHTED = 6  # Adaptive interpolation between self_mean and chain_mean
    MODE_WEIGHTED = 7  # Adaptive interpolation toward mode (highest log posterior chain)
    MCOV_WEIGHTED = 8      # Mean-Cov weighted: covariance scales with distance, affects mean interpolation
    MCOV_WEIGHTED_VEC = 9  # Vectorized MCOV: per-parameter distance, interpolation, and cov scaling
    MCOV_SMOOTH = 10       # Three-zone smoothstep: chain_mean near equilibrium, tracking mid-range, rescue far
    MCOV_MODE = 11         # Mode-targeting with scalar Mahalanobis distance (uniform α across params)
    MCOV_MODE_VEC = 12     # Mode-targeting with per-parameter distances (individual α per param)

    # Future proposals - add new enum values here, implement in proposals.py
    # ADAPTIVE = 13       # Adaptive covariance during burn-in
    # PRECONDITIONED = 13 # Use custom preconditioning matrix

    def __str__(self):
        return self.name.replace('_', ' ').title()


# ============================================================================
# PROPOSAL GROUP (for mixed proposals within a block)
# ============================================================================

@dataclass
class ProposalGroup:
    """
    Specification for a proposal sub-group within a block.

    Used when different parameters within a single MH block require different
    proposal types. For example, continuous parameters (0-11) using MCOV_MODE
    and a discrete indicator (12) using MULTINOMIAL.

    The indices are relative to the block, not the full parameter vector.
    Groups must be contiguous and non-overlapping, covering the entire block.

    Fields:
        start: Starting index within block (0-indexed, inclusive)
        end: Ending index within block (exclusive)
        proposal_type: ProposalType for this sub-group
        settings: Per-group settings (alpha, cov_mult, etc.)

    Example:
        # Block of size 13: params 0-11 continuous, param 12 discrete
        ProposalGroup(start=0, end=12, proposal_type=ProposalType.MCOV_MODE,
                      settings={'cov_mult': 1.0})
        ProposalGroup(start=12, end=13, proposal_type=ProposalType.MULTINOMIAL,
                      settings={'alpha': 0.5, 'n_categories': 2})
    """
    start: int
    end: int
    proposal_type: 'ProposalType'
    settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f"ProposalGroup start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"ProposalGroup end ({self.end}) must be > start ({self.start})")
        if not isinstance(self.proposal_type, (ProposalType, int)):
            raise ValueError(f"proposal_type must be ProposalType or int, got {type(self.proposal_type)}")
        # Convert int to ProposalType if needed
        if isinstance(self.proposal_type, int):
            object.__setattr__(self, 'proposal_type', ProposalType(self.proposal_type))

    @property
    def size(self) -> int:
        """Number of parameters in this group."""
        return self.end - self.start


# ============================================================================
# BLOCK SPECIFICATION
# ============================================================================

@dataclass
class BlockSpec:
    """
    Specification for a single parameter block.

    A "block" is a group of parameters that are updated together in one step.

    Required fields:
        size: Number of parameters in this block
        sampler_type: How to sample this block (MH, direct, coupled_transform, etc.)

    Optional fields:
        proposal_type: For MH samplers, which proposal strategy to use (single proposal)
        proposal_groups: For mixed proposals, list of ProposalGroup objects defining
                         different proposal types for contiguous sub-groups
        direct_sampler_fn: For direct samplers, the sampling function
        label: Human-readable name for debugging/logging
        settings: Dict of sampler-specific settings (used when proposal_type is set)
        metadata: Additional info (not used by sampler, for user reference)

    For COUPLED_TRANSFORM sampler type (theta-preserving updates):
        coupled_indices_fn: Function(chain_state, data) -> Array of coupled param indices
        forward_transform_fn: Function(chain_state, primary_indices, proposed_primary,
                                       coupled_indices, data) -> new coupled values
        log_jacobian_fn: Function(chain_state, proposed_state, primary_indices,
                                  coupled_indices, data) -> log |det J|
        coupled_log_prior_fn: Function(values) -> log prior for coupled params

    Mixed Proposals:
        When proposal_groups is specified, different parts of the block use different
        proposal types. This is useful for blocks containing both continuous and discrete
        parameters. The groups must be contiguous and cover the entire block.

        The Hastings ratio for mixed proposals is the product of individual ratios:
            q(θ|θ')/q(θ'|θ) = Π[qᵢ(θᵢ|θ'ᵢ)/qᵢ(θ'ᵢ|θᵢ)]

        Note: Gradient-based proposals (MALA) cannot be used for discrete parameters.

    Examples:
        # Simple MH block with random walk
        BlockSpec(size=2, sampler_type=SamplerType.METROPOLIS_HASTINGS)

        # MH block with independent proposal
        BlockSpec(size=3, sampler_type=SamplerType.METROPOLIS_HASTINGS,
                  proposal_type=ProposalType.CHAIN_MEAN)

        # Mixed proposal block: continuous (0-11) + discrete (12)
        BlockSpec(size=13, sampler_type=SamplerType.METROPOLIS_HASTINGS,
                  proposal_groups=[
                      ProposalGroup(start=0, end=12, proposal_type=ProposalType.MCOV_MODE,
                                    settings={'cov_mult': 1.0}),
                      ProposalGroup(start=12, end=13, proposal_type=ProposalType.MULTINOMIAL,
                                    settings={'alpha': 0.5, 'n_categories': 2}),
                  ],
                  label="Subject_0")

        # Direct sampler block
        BlockSpec(size=1, sampler_type=SamplerType.DIRECT_CONJUGATE,
                  direct_sampler_fn=my_conjugate_sampler,
                  label="Hyperparameter: variance")

        # Coupled transform block (theta-preserving NCP updates)
        BlockSpec(size=2, sampler_type=SamplerType.COUPLED_TRANSFORM,
                  proposal_type=ProposalType.MCOV_WEIGHTED_VEC,
                  coupled_indices_fn=get_epsilon_indices,
                  forward_transform_fn=theta_preserving_transform,
                  log_jacobian_fn=compute_jacobian,
                  coupled_log_prior_fn=epsilon_log_prior,
                  label="Hyper_r")
    """
    # Required
    size: int
    sampler_type: SamplerType

    # Optional - for MH samplers (single proposal type for entire block)
    proposal_type: Optional[ProposalType] = ProposalType.SELF_MEAN

    # Optional - for MH samplers (mixed proposals: different types for sub-groups)
    proposal_groups: Optional[List['ProposalGroup']] = None

    # Optional - for direct samplers
    direct_sampler_fn: Optional[Callable] = None

    # Optional - for COUPLED_TRANSFORM samplers
    coupled_indices_fn: Optional[Callable] = None       # (chain_state, data) -> indices
    forward_transform_fn: Optional[Callable] = None     # Transform coupled params
    log_jacobian_fn: Optional[Callable] = None          # Log Jacobian of transform
    coupled_log_prior_fn: Optional[Callable] = None     # Log prior for coupled params

    # Optional - metadata
    label: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the specification after initialization."""
        if self.size < 1:
            raise ValueError(f"Block size must be >= 1, got {self.size}")

        if not isinstance(self.sampler_type, (SamplerType, int)):
            raise ValueError(f"sampler_type must be SamplerType or int, got {type(self.sampler_type)}")

        # Convert to SamplerType if int was provided (for backward compatibility)
        if isinstance(self.sampler_type, int):
            object.__setattr__(self, 'sampler_type', SamplerType(self.sampler_type))

        if self.proposal_type is not None and isinstance(self.proposal_type, int):
            object.__setattr__(self, 'proposal_type', ProposalType(self.proposal_type))

        # Validate proposal_groups if provided
        if self.proposal_groups is not None:
            self._validate_proposal_groups()

        # Validate sampler-specific requirements
        if self.sampler_type == SamplerType.DIRECT_CONJUGATE:
            if self.direct_sampler_fn is None:
                raise ValueError(
                    "DIRECT_CONJUGATE sampler requires direct_sampler_fn to be specified"
                )

        # Validate COUPLED_TRANSFORM requirements
        # Per-block callbacks are optional if using central coupled_transform_dispatch
        # The model can register a coupled_transform_dispatch function that handles
        # all COUPLED_TRANSFORM blocks by inspecting the block label/indices
        if self.sampler_type == SamplerType.COUPLED_TRANSFORM:
            # If per-block callbacks are provided, require all of them
            has_per_block = (
                self.coupled_indices_fn is not None or
                self.forward_transform_fn is not None or
                self.log_jacobian_fn is not None or
                self.coupled_log_prior_fn is not None
            )
            if has_per_block:
                missing = []
                if self.coupled_indices_fn is None:
                    missing.append("coupled_indices_fn")
                if self.forward_transform_fn is None:
                    missing.append("forward_transform_fn")
                if self.log_jacobian_fn is None:
                    missing.append("log_jacobian_fn")
                if self.coupled_log_prior_fn is None:
                    missing.append("coupled_log_prior_fn")
                if missing:
                    raise ValueError(
                        f"COUPLED_TRANSFORM with partial per-block callbacks missing: {', '.join(missing)}"
                    )
            # If no per-block callbacks, the model must register coupled_transform_dispatch
            # This is validated at registration time by config.py

    def _validate_proposal_groups(self):
        """Validate proposal_groups specification."""
        groups = self.proposal_groups

        if len(groups) == 0:
            raise ValueError("proposal_groups cannot be empty if specified")

        if len(groups) > MAX_PROPOSAL_GROUPS:
            raise ValueError(
                f"Too many proposal groups ({len(groups)}), maximum is {MAX_PROPOSAL_GROUPS}"
            )

        # Check groups cover entire block contiguously
        sorted_groups = sorted(groups, key=lambda g: g.start)

        # First group must start at 0
        if sorted_groups[0].start != 0:
            raise ValueError(
                f"First proposal group must start at 0, got {sorted_groups[0].start}"
            )

        # Last group must end at block size
        if sorted_groups[-1].end != self.size:
            raise ValueError(
                f"Last proposal group must end at block size ({self.size}), "
                f"got {sorted_groups[-1].end}"
            )

        # Groups must be contiguous (no gaps or overlaps)
        for i in range(len(sorted_groups) - 1):
            if sorted_groups[i].end != sorted_groups[i + 1].start:
                raise ValueError(
                    f"Proposal groups must be contiguous: group ending at "
                    f"{sorted_groups[i].end} followed by group starting at "
                    f"{sorted_groups[i + 1].start}"
                )

        # Validate no gradient-based proposals on discrete parameters
        # (MULTINOMIAL indicates discrete; MALA/MEAN_MALA need gradients)
        gradient_proposals = {ProposalType.MALA, ProposalType.MEAN_MALA}
        discrete_proposals = {ProposalType.MULTINOMIAL}

        for group in groups:
            if group.proposal_type in gradient_proposals:
                # Check if any other group in this block uses MULTINOMIAL
                # (indicates the block has discrete parameters)
                has_discrete = any(g.proposal_type in discrete_proposals for g in groups)
                if has_discrete:
                    # This is fine - gradient proposal is on a different group
                    pass
                # Additional checks could be added here for future discrete proposal types

    def is_mh_sampler(self):
        """Check if this block uses a Metropolis-Hastings sampler."""
        return self.sampler_type in [
            SamplerType.METROPOLIS_HASTINGS,
            SamplerType.COUPLED_TRANSFORM,  # Uses MH accept/reject with transform
            SamplerType.ADAPTIVE_MH,
            SamplerType.HMC,
            SamplerType.NUTS
        ]

    def is_direct_sampler(self):
        """Check if this block uses direct sampling."""
        return self.sampler_type in [
            SamplerType.DIRECT_CONJUGATE,
            SamplerType.SLICE_SAMPLER,
            SamplerType.ELLIPTICAL_SLICE
        ]

    def has_mixed_proposals(self):
        """Check if this block uses multiple proposal types."""
        return self.proposal_groups is not None and len(self.proposal_groups) > 0

    def get_effective_groups(self) -> List['ProposalGroup']:
        """
        Get the effective proposal groups for this block.

        If proposal_groups is specified, returns them directly.
        Otherwise, returns a single group covering the entire block
        using proposal_type and settings.

        Returns:
            List of ProposalGroup objects
        """
        if self.has_mixed_proposals():
            return list(self.proposal_groups)
        else:
            # Single implicit group covering entire block
            return [ProposalGroup(
                start=0,
                end=self.size,
                proposal_type=self.proposal_type,
                settings=dict(self.settings)
            )]

    def __repr__(self):
        """Pretty string representation for debugging."""
        parts = [f"BlockSpec(size={self.size}, sampler={self.sampler_type}"]
        if self.has_mixed_proposals():
            parts.append(f"mixed_proposals={len(self.proposal_groups)} groups")
        elif self.proposal_type is not None and self.is_mh_sampler():
            parts.append(f"proposal={self.proposal_type}")
        if self.label:
            parts.append(f'label="{self.label}"')
        if self.settings and not self.has_mixed_proposals():
            parts.append(f"settings={self.settings}")
        return ", ".join(parts) + ")"


# ============================================================================
# VALIDATION
# ============================================================================

def validate_block_specs(specs: List[BlockSpec], model_name: str = "") -> None:
    """
    Validate a list of block specifications.

    Args:
        specs: List of BlockSpec objects
        model_name: Name of model (for error messages)

    Raises:
        ValueError: If specs are invalid
    """
    if not isinstance(specs, list):
        raise ValueError(f"Block specs must be a list, got {type(specs)}")

    if len(specs) == 0:
        raise ValueError("Block specs list cannot be empty")

    errors = []

    for i, spec in enumerate(specs):
        if not isinstance(spec, BlockSpec):
            errors.append(
                f"Block {i}: Expected BlockSpec object, got {type(spec)}"
            )
            continue

        # Validate size
        if spec.size < 1:
            errors.append(f"Block {i} ({spec.label or 'unlabeled'}): size must be >= 1")

        # Validate direct sampler function
        if spec.sampler_type == SamplerType.DIRECT_CONJUGATE:
            if spec.direct_sampler_fn is None:
                errors.append(
                    f"Block {i} ({spec.label or 'unlabeled'}): "
                    f"DIRECT_CONJUGATE requires direct_sampler_fn"
                )

        # Validate COUPLED_TRANSFORM requirements
        # Per-block callbacks are optional if using central coupled_transform_dispatch
        if spec.sampler_type == SamplerType.COUPLED_TRANSFORM:
            has_per_block = (
                spec.coupled_indices_fn is not None or
                spec.forward_transform_fn is not None or
                spec.log_jacobian_fn is not None or
                spec.coupled_log_prior_fn is not None
            )
            if has_per_block:
                missing = []
                if spec.coupled_indices_fn is None:
                    missing.append("coupled_indices_fn")
                if spec.forward_transform_fn is None:
                    missing.append("forward_transform_fn")
                if spec.log_jacobian_fn is None:
                    missing.append("log_jacobian_fn")
                if spec.coupled_log_prior_fn is None:
                    missing.append("coupled_log_prior_fn")
                if missing:
                    errors.append(
                        f"Block {i} ({spec.label or 'unlabeled'}): "
                        f"COUPLED_TRANSFORM with partial per-block callbacks missing {', '.join(missing)}"
                    )

    if errors:
        prefix = f"Invalid block specs for '{model_name}'" if model_name else "Invalid block specs"
        raise ValueError(f"{prefix}:\n  " + "\n  ".join(errors))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_subject_blocks(n_subjects: int,
                         params_per_subject: int,
                         sampler_type: SamplerType = SamplerType.METROPOLIS_HASTINGS,
                         proposal_type: ProposalType = ProposalType.SELF_MEAN,
                         settings : Dict = {},
                         label_prefix: str = "Subject") -> List[BlockSpec]:
    """
    Convenience function to create many identical subject blocks.

    Args:
        n_subjects: Number of subjects
        params_per_subject: Parameters per subject
        sampler_type: Sampler to use
        proposal_type: Proposal type for MH samplers
        label_prefix: Prefix for block labels

    Returns:
        List of BlockSpec objects, one per subject

    Example:
        >>> specs = create_subject_blocks(10, 2, label_prefix="Subj")
        >>> # Creates 10 blocks of size 2 with labels "Subj_0", "Subj_1", ...
    """
    return [
        BlockSpec(
            size=params_per_subject,
            sampler_type=sampler_type,
            proposal_type=proposal_type if sampler_type == SamplerType.METROPOLIS_HASTINGS else None,
            settings=settings,
            label=f"{label_prefix}_{i}"
        )
        for i in range(n_subjects)
    ]


def create_mixed_subject_blocks(
    n_subjects: int,
    proposal_groups: List[ProposalGroup],
    sampler_type: SamplerType = SamplerType.METROPOLIS_HASTINGS,
    label_prefix: str = "Subject"
) -> List[BlockSpec]:
    """
    Convenience function to create subject blocks with mixed proposals.

    Each subject block uses the same proposal_groups configuration,
    allowing different proposal types for different parameters within
    each subject's block.

    Args:
        n_subjects: Number of subjects
        proposal_groups: List of ProposalGroup objects defining the proposal
                         structure. The total size is inferred from groups.
        sampler_type: Sampler to use (default METROPOLIS_HASTINGS)
        label_prefix: Prefix for block labels

    Returns:
        List of BlockSpec objects, one per subject

    Example:
        # Subject blocks with 12 continuous params (MCOV_MODE) + 1 discrete (MULTINOMIAL)
        groups = [
            ProposalGroup(start=0, end=12, proposal_type=ProposalType.MCOV_MODE,
                          settings={'cov_mult': 1.0}),
            ProposalGroup(start=12, end=13, proposal_type=ProposalType.MULTINOMIAL,
                          settings={'alpha': 0.5, 'n_categories': 2}),
        ]
        specs = create_mixed_subject_blocks(245, groups, label_prefix="Subj")
    """
    if not proposal_groups:
        raise ValueError("proposal_groups cannot be empty")

    # Infer block size from groups
    block_size = max(g.end for g in proposal_groups)

    return [
        BlockSpec(
            size=block_size,
            sampler_type=sampler_type,
            proposal_groups=proposal_groups,
            label=f"{label_prefix}_{i}"
        )
        for i in range(n_subjects)
    ]


# ============================================================================
# SUMMARY UTILITIES
# ============================================================================

def summarize_blocks(specs: List[BlockSpec]) -> str:
    """
    Create a human-readable summary of block specifications.

    Args:
        specs: List of BlockSpec objects

    Returns:
        Formatted string summary
    """
    total_params = sum(spec.size for spec in specs)
    sampler_counts = {}

    for spec in specs:
        stype = str(spec.sampler_type)
        sampler_counts[stype] = sampler_counts.get(stype, 0) + 1

    lines = [
        f"Block Specification Summary:",
        f"  Total blocks: {len(specs)}",
        f"  Total parameters: {total_params}",
        f"",
        f"Sampler breakdown:"
    ]

    for stype, count in sorted(sampler_counts.items()):
        lines.append(f"  {stype}: {count} block(s)")

    if any(spec.label for spec in specs):
        lines.append(f"\nLabeled blocks:")
        for i, spec in enumerate(specs):
            if spec.label:
                lines.append(f"  Block {i}: {spec.label} (size={spec.size})")

    return "\n".join(lines)


def print_block_summary(specs: List[BlockSpec]) -> None:
    """Print a summary of block specifications."""
    print(summarize_blocks(specs))
