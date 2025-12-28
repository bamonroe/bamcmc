import os

# Suppress CUDA/XLA C++ warnings (GPU interconnect, NUMA, cuDNN factories)
# Must be set before JAX import; does not affect JAX compilation time messages
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

def clean_config(mcmc_config):
    """
    Cleans the config file and sets defaults
    """

    # Define Defaults and retrieve values from dictionary
    mcmc_config.setdefault('GPU_PREALLOCATION', False)
    mcmc_config.setdefault('USE_DOUBLE', False)
    mcmc_config.setdefault('POSTERIOR_ID', 'normal_10_5')
    mcmc_config.setdefault('THIN_ITERATION', 100)
    mcmc_config.setdefault('NUM_COLLECT', 10)
    mcmc_config.setdefault('BURN_ITER', 0)
    mcmc_config.setdefault('NUM_CHAINS_A', 500)
    mcmc_config.setdefault('NUM_CHAINS_B', 500)
    mcmc_config.setdefault('BENCHMARK', 10)
    mcmc_config.setdefault('PROPOSAL', 'chain_mean')

    if type(mcmc_config["GPU_PREALLOCATION"]) != bool:
        print("'GPU_PREALLOCATION' must be 'True' or 'False'")

    # 1. Environment Setup
    if 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ:
        if mcmc_config["GPU_PREALLOCATION"]:
            print("Pre-allocating GPU memory.")
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
        else:
            print("WARNING: Disabling GPU memory pre-allocation.")
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    return mcmc_config
