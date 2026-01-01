import os

# Suppress CUDA/XLA C++ warnings (GPU interconnect, NUMA, cuDNN factories)
# Must be set before JAX import; does not affect JAX compilation time messages
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

def clean_config(mcmc_config):
    """
    Cleans the config file and sets defaults.
    All config keys use lowercase with underscores.
    """

    # Define Defaults and retrieve values from dictionary (all lowercase)
    mcmc_config.setdefault('gpu_preallocation', False)
    mcmc_config.setdefault('use_double', False)
    mcmc_config.setdefault('posterior_id', 'normal_10_5')
    mcmc_config.setdefault('thin_iteration', 100)
    mcmc_config.setdefault('num_collect', 10)
    mcmc_config.setdefault('burn_iter', 0)
    mcmc_config.setdefault('num_chains_a', 500)
    mcmc_config.setdefault('num_chains_b', 500)
    mcmc_config.setdefault('benchmark', 10)
    mcmc_config.setdefault('proposal', 'chain_mean')

    if type(mcmc_config["gpu_preallocation"]) != bool:
        print("'gpu_preallocation' must be 'True' or 'False'")

    # 1. Environment Setup
    if 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ:
        if mcmc_config["gpu_preallocation"]:
            print("Pre-allocating GPU memory.")
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
        else:
            print("WARNING: Disabling GPU memory pre-allocation.")
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    return mcmc_config
