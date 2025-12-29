"""
Posterior Registration System

This module provides a registry for posterior models that can be used with the MCMC backend.
User code registers posteriors via register_posterior(), and the backend retrieves them via get_posterior().

Example usage:
    from bamcmc import register_posterior, BlockSpec, SamplerType

    def my_log_posterior(chain_state, param_indices, data):
        ...

    def my_batch_specs(mcmc_config, data):
        return [BlockSpec(size=2, sampler_type=SamplerType.METROPOLIS_HASTINGS)]

    register_posterior('my_model', {
        'log_posterior': my_log_posterior,
        'batch_type': my_batch_specs,
        'initial_vector': my_init_fn,
        # optional:
        'direct_sampler': my_direct_sampler,
        'generated_quantities': my_gq_fn,
        'get_num_gq': my_num_gq_fn,
    })
"""

_REGISTRY = {}


def register_posterior(name, config):
    """
    Register a posterior model with the MCMC system.

    Args:
        name: Unique model identifier string (e.g., 'eut_crra_bhm')
        config: Dict containing model functions with keys:

            Required:
                log_posterior: fn(chain_state, param_indices, data) -> scalar
                    Log posterior density for a parameter block.

                batch_type: fn(mcmc_config, data) -> List[BlockSpec]
                    Returns block specifications defining parameter structure.

                initial_vector: fn(mcmc_config, data) -> array
                    Returns initial parameter values for all chains.

            Optional:
                direct_sampler: fn(key, chain_state, param_indices, data) -> (state, key)
                    Direct/Gibbs sampler for blocks with SamplerType.DIRECT_CONJUGATE.

                generated_quantities: fn(chain_state, data) -> array
                    Computes derived quantities from chain state.

                get_num_gq: fn(mcmc_config, data) -> int
                    Returns number of generated quantities.

    Raises:
        ValueError: If required keys are missing or name is already registered.

    Example:
        register_posterior('my_model', {
            'log_posterior': my_log_posterior_fn,
            'batch_type': my_batch_specs_fn,
            'initial_vector': my_init_fn,
        })
    """
    if name in _REGISTRY:
        raise ValueError(f"Posterior '{name}' is already registered")

    required_keys = ['log_posterior', 'batch_type', 'initial_vector']
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required keys for posterior '{name}': {missing}")

    _REGISTRY[name] = config


def get_posterior(name):
    """
    Get a registered posterior configuration by name.

    Args:
        name: The posterior identifier

    Returns:
        Dict containing the posterior model functions

    Raises:
        KeyError: If the posterior is not registered
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"Unknown posterior '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_posteriors():
    """
    List all registered posterior names.

    Returns:
        List of registered posterior name strings
    """
    return list(_REGISTRY.keys())


def clear_registry():
    """
    Clear all registered posteriors. Primarily for testing.
    """
    _REGISTRY.clear()
