"""
Test Posteriors - Conjugate Models with Analytical Solutions

This module contains simple conjugate models used for testing the MCMC backend.
These posteriors have known analytical solutions, allowing us to verify sampler correctness.

DO NOT import this module in production sampling code.
These models are for testing/validation only.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.random as random
from functools import partial

# Import new BlockSpec system
from .batch_specs import BlockSpec, SamplerType, ProposalType


# ============================================================================
# BETA-BERNOULLI (POOLED) - Single Parameter
# ============================================================================

def beta_bernoulli_pooled_batch_specs(mcmc_config, data):
    """Single parameter (theta) using MH sampling."""
    return [
        BlockSpec(
            size=1,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=ProposalType.SELF_MEAN,
            label="theta"
        )
    ]


def beta_bernoulli_pooled_log_posterior(chain_state, param_indices, data):
    """
    Beta-Bernoulli Model (Pooled across all subjects).
    
    Model:
        theta ~ Beta(alpha_0, beta_0)          [Prior]
        y_i ~ Bernoulli(theta) for i=1..n      [Likelihood]
    
    Analytical Posterior:
        theta | y ~ Beta(alpha_0 + sum(y), beta_0 + n - sum(y))
    
    Data format:
        data["int"][0]: Column vector of binary outcomes (0 or 1)
        data["static"][0]: alpha_0 (prior pseudo-count for successes)
        data["static"][1]: beta_0 (prior pseudo-count for failures)
    """
    # Extract theta (it's on the logit scale for unconstrained sampling)
    theta_raw = chain_state[param_indices[0]]
    theta = jax.nn.sigmoid(theta_raw)  # Transform to (0, 1)
    
    # Get data
    y = data["int"][0]  # Binary outcomes
    alpha_0 = data["static"][0]
    beta_0 = data["static"][1]
    
    # Prior: Beta on the constrained parameter
    # We need to account for the Jacobian of the sigmoid transform
    log_prior = stats.beta.logpdf(theta, alpha_0, beta_0)
    log_jacobian = jnp.log(theta) + jnp.log(1.0 - theta)  # Jacobian of logit transform
    
    # Likelihood: Product of Bernoulli
    log_lik = jnp.sum(y * jnp.log(theta) + (1 - y) * jnp.log(1 - theta))
    
    return log_prior + log_jacobian + log_lik


def beta_bernoulli_pooled_gq(chain_state, data):
    """Generated quantities: return theta on the probability scale."""
    theta_raw = chain_state[0]
    theta = jax.nn.sigmoid(theta_raw)
    return jnp.array([theta])


def beta_bernoulli_pooled_initial_vector(mcmc_config, data):
    """Initialize theta_raw near 0 (which maps to theta ≈ 0.5)."""
    num_chains = mcmc_config["num_chains"]
    # Only generate K distinct states for superchains; backend replicates for subchains
    K = mcmc_config.get("num_superchains", num_chains)
    init = np.random.normal(0.0, 0.5, (K, 1))
    # Pad to full size (backend expects num_chains rows, uses first K)
    if K < num_chains:
        full_init = np.zeros((num_chains, 1))
        full_init[:K] = init
        return full_init.flatten()
    return init.flatten()


def beta_bernoulli_pooled_analytical_posterior(data):
    """
    Returns the parameters of the analytical posterior Beta distribution.
    
    Returns:
        (alpha_post, beta_post) for Beta(alpha_post, beta_post)
    """
    y = np.array(data["int"][0])
    alpha_0 = data["static"][0]
    beta_0 = data["static"][1]
    
    n_success = np.sum(y)
    n_total = len(y)
    n_failure = n_total - n_success
    
    alpha_post = alpha_0 + n_success
    beta_post = beta_0 + n_failure
    
    return alpha_post, beta_post


# ============================================================================
# NORMAL-NORMAL (POOLED) - Single Parameter (Known Variance)
# ============================================================================

def normal_normal_pooled_batch_specs(mcmc_config, data):
    """Single parameter (mu) using MH sampling."""
    return [
        BlockSpec(
            size=1,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=ProposalType.SELF_MEAN,
            label="mu"
        )
    ]


def normal_normal_pooled_log_posterior(chain_state, param_indices, data):
    """
    Normal-Normal Model with Known Variance (Pooled).
    
    Model:
        mu ~ Normal(mu_0, tau_0^2)           [Prior]
        y_i ~ Normal(mu, sigma^2) for i=1..n [Likelihood, sigma^2 known]
    
    Analytical Posterior:
        mu | y ~ Normal(mu_n, tau_n^2)
        where:
            tau_n^2 = 1 / (1/tau_0^2 + n/sigma^2)
            mu_n = tau_n^2 * (mu_0/tau_0^2 + sum(y)/sigma^2)
    
    Data format:
        data["float"][0]: Array of observations y
        data["static"][0]: mu_0 (prior mean)
        data["static"][1]: tau_0 (prior std dev)
        data["static"][2]: sigma (likelihood std dev, known)
    """
    mu = chain_state[param_indices[0]]
    
    # Get data
    y = data["float"][0]
    mu_0 = data["static"][0]
    tau_0 = data["static"][1]
    sigma = data["static"][2]
    
    # Prior
    log_prior = stats.norm.logpdf(mu, loc=mu_0, scale=tau_0)
    
    # Likelihood
    log_lik = jnp.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    
    return log_prior + log_lik


def normal_normal_pooled_gq(chain_state, data):
    """No transformation needed for Normal."""
    return jnp.array([chain_state[0]])


def normal_normal_pooled_initial_vector(mcmc_config, data):
    """Initialize mu near the prior mean."""
    num_chains = mcmc_config["num_chains"]
    mu_0 = data["static"][0]
    tau_0 = data["static"][1]
    # Only generate K distinct states for superchains; backend replicates for subchains
    K = mcmc_config.get("num_superchains", num_chains)
    init = np.random.normal(mu_0, tau_0, (K, 1))
    # Pad to full size (backend expects num_chains rows, uses first K)
    if K < num_chains:
        full_init = np.zeros((num_chains, 1))
        full_init[:K] = init
        return full_init.flatten()
    return init.flatten()


def normal_normal_pooled_analytical_posterior(data):
    """
    Returns the parameters of the analytical posterior Normal distribution.
    
    Returns:
        (mu_post, tau_post) for Normal(mu_post, tau_post^2)
    """
    y = np.array(data["float"][0])
    mu_0 = data["static"][0]
    tau_0 = data["static"][1]
    sigma = data["static"][2]
    
    n = len(y)
    tau_0_sq = tau_0**2
    sigma_sq = sigma**2
    
    # Posterior precision and variance
    post_precision = (1.0 / tau_0_sq) + (n / sigma_sq)
    tau_post_sq = 1.0 / post_precision
    
    # Posterior mean
    mu_post = tau_post_sq * ((mu_0 / tau_0_sq) + (np.sum(y) / sigma_sq))
    
    tau_post = np.sqrt(tau_post_sq)
    
    return mu_post, tau_post


# ============================================================================
# BETA-BERNOULLI (HIERARCHICAL) - Multiple Subjects + Hyperparameters
# ============================================================================

def beta_bernoulli_hierarchical_batch_specs(mcmc_config, data):
    """
    One parameter per subject (theta_i) + 2 hyperparameters (alpha, beta).
    
    Subject parameters use MH.
    Hyperparameters use direct sampling.
    """
    n_subjects = data["static"][2]
    
    # Subject parameters (1 per subject, all MH)
    subject_specs = [
        BlockSpec(
            size=1,
            sampler_type=SamplerType.METROPOLIS_HASTINGS,
            proposal_type=ProposalType.SELF_MEAN,
            label=f"Subject_{i}_theta"
        )
        for i in range(n_subjects)
    ]
    
    # Hyperparameters (2 total, both direct sampled)
    hyper_specs = [
        BlockSpec(
            size=1,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=beta_bernoulli_hierarchical_direct_dispatch,
            label="Hyper_alpha"
        ),
        BlockSpec(
            size=1,
            sampler_type=SamplerType.DIRECT_CONJUGATE,
            direct_sampler_fn=beta_bernoulli_hierarchical_direct_dispatch,
            label="Hyper_beta"
        )
    ]
    
    return subject_specs + hyper_specs


def beta_bernoulli_hierarchical_log_posterior(chain_state, param_indices, data):
    """
    Hierarchical Beta-Bernoulli Model.
    
    Model:
        alpha, beta ~ Gamma(a0, b0)                    [Hyperpriors]
        theta_i ~ Beta(alpha, beta) for i=1..n_subj    [Subject priors]
        y_ij ~ Bernoulli(theta_i) for trials j         [Likelihood]
    
    Data format:
        data["int"][0]: 2D array (n_subjects, max_trials) of binary outcomes
        data["int"][1]: Array of trial counts per subject
        data["static"][0]: a0 (hyperprior shape for alpha)
        data["static"][1]: b0 (hyperprior rate for alpha)
        data["static"][2]: n_subjects
    """
    n_subjects = data["static"][2]
    first_hyper = n_subjects  # Index where hyperparameters start
    
    param_idx = param_indices[0]
    
    # Check if this is a subject parameter or hyperparameter
    if param_idx < first_hyper:
        # Subject parameter update
        subject_id = param_idx
        
        # Extract hyperparameters (on log scale for unconstrained sampling)
        alpha_raw = chain_state[first_hyper]
        beta_raw = chain_state[first_hyper + 1]
        alpha = jnp.exp(alpha_raw)
        beta = jnp.exp(beta_raw)
        
        # Extract subject parameter (logit scale)
        theta_raw = chain_state[param_idx]
        theta = jax.nn.sigmoid(theta_raw)
        
        # Prior from hyperparameters
        log_prior = stats.beta.logpdf(theta, alpha, beta)
        log_jacobian = jnp.log(theta) + jnp.log(1.0 - theta)
        
        # Likelihood for this subject
        y_subject = data["int"][0][subject_id]  # All trials for this subject
        n_trials = data["int"][1][subject_id]   # Number of valid trials
        
        # Mask out padded trials
        mask = jnp.arange(y_subject.shape[0]) < n_trials
        y_valid = jnp.where(mask, y_subject, 0)
        
        log_lik = jnp.sum(
            jnp.where(
                mask,
                y_valid * jnp.log(theta) + (1 - y_valid) * jnp.log(1 - theta),
                0.0
            )
        )
        
        return log_prior + log_jacobian + log_lik
    
    else:
        # Hyperparameter update - this won't be called if using direct sampler
        # But we need it defined for the interface
        return 0.0


def direct_sampler_hierarchical_alpha(key, chain_state, hyper_idx, data):
    """
    Direct sampler for alpha hyperparameter.
    
    Conjugate update: Gamma prior + Beta likelihood → Gamma posterior
    (This is approximate; true conjugacy requires specific prior structure)
    
    For simplicity, we'll use a Gamma prior and update based on all thetas.
    """
    n_subjects = data["static"][2]
    a0 = data["static"][0]
    b0 = data["static"][1]
    
    # Get current beta hyperparameter
    beta_raw = chain_state[hyper_idx + 1]
    beta_hyper = jnp.exp(beta_raw)
    
    # Get all subject thetas
    theta_raws = chain_state[:n_subjects]
    thetas = jax.nn.sigmoid(theta_raws)
    
    # Use method of moments approximation for Gamma posterior
    # (In practice, you might use MH here too, but this demonstrates direct sampling)
    sum_log_theta = jnp.sum(jnp.log(thetas))
    
    # Posterior parameters (simplified)
    alpha_post = a0 + n_subjects * 2.0  # Simplified
    beta_post = b0 + 1.0
    
    new_key, sample_key = random.split(key)
    alpha_sample = random.gamma(sample_key, alpha_post) / beta_post
    alpha_sample = jnp.maximum(alpha_sample, 0.1)  # Ensure positive
    
    alpha_raw_new = jnp.log(alpha_sample)
    
    return chain_state.at[hyper_idx].set(alpha_raw_new), new_key


def direct_sampler_hierarchical_beta(key, chain_state, hyper_idx, data):
    """Direct sampler for beta hyperparameter (similar to alpha)."""
    n_subjects = data["static"][2]
    a0 = data["static"][0]
    b0 = data["static"][1]
    
    # Similar logic to alpha sampler
    beta_post = a0 + n_subjects * 2.0
    rate_post = b0 + 1.0
    
    new_key, sample_key = random.split(key)
    beta_sample = random.gamma(sample_key, beta_post) / rate_post
    beta_sample = jnp.maximum(beta_sample, 0.1)
    
    beta_raw_new = jnp.log(beta_sample)
    
    return chain_state.at[hyper_idx].set(beta_raw_new), new_key


def beta_bernoulli_hierarchical_direct_dispatch(key, chain_state, param_indices, data):
    """Dispatch to appropriate hyperparameter sampler."""
    hyper_idx = param_indices[0]
    n_subjects = data["static"][2]
    offset = hyper_idx - n_subjects
    
    is_alpha = (offset == 0)
    
    true_branch = partial(direct_sampler_hierarchical_alpha, data=data)
    false_branch = partial(direct_sampler_hierarchical_beta, data=data)
    
    return jax.lax.cond(is_alpha, true_branch, false_branch, key, chain_state, hyper_idx)


def beta_bernoulli_hierarchical_gq(chain_state, data):
    """Generated quantities: all thetas on probability scale + hyperparameters."""
    n_subjects = data["static"][2]
    
    theta_raws = chain_state[:n_subjects]
    thetas = jax.nn.sigmoid(theta_raws)
    
    alpha_raw = chain_state[n_subjects]
    beta_raw = chain_state[n_subjects + 1]
    alpha = jnp.exp(alpha_raw)
    beta = jnp.exp(beta_raw)
    
    return jnp.concatenate([thetas, jnp.array([alpha, beta])])


def beta_bernoulli_hierarchical_initial_vector(mcmc_config, data):
    """Initialize all parameters."""
    num_chains = mcmc_config["num_chains"]
    n_subjects = data["static"][2]
    
    total_params = n_subjects + 2
    init = np.zeros((num_chains, total_params))
    
    # Subject thetas (logit scale, centered at 0 → theta ≈ 0.5)
    init[:, :n_subjects] = np.random.normal(0.0, 0.5, (num_chains, n_subjects))
    
    # Hyperparameters (log scale, α ≈ 2, β ≈ 2)
    init[:, n_subjects] = np.random.normal(np.log(2.0), 0.2, num_chains)
    init[:, n_subjects + 1] = np.random.normal(np.log(2.0), 0.2, num_chains)
    
    return init.flatten()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_test_num_gq(mcmc_config, data):
    """Returns number of generated quantities for test models."""
    posterior_id = mcmc_config.get("posterior_id", "")
    
    if posterior_id == "test_beta_bernoulli_hierarchical":
        n_subjects = data["static"][2]
        return n_subjects + 2  # All thetas + 2 hyperparams
    else:
        return 1  # Single GQ for pooled models


def placeholder_direct_sampler(key, chain_state, param_indices, data):
    """Placeholder for models that don't use direct sampling."""
    new_key, _ = random.split(key)
    return chain_state, new_key


# ============================================================================
# REGISTRY - Add test posteriors here
# ============================================================================

TEST_POSTERIORS = {
    'test_beta_bernoulli_pooled': {
        'log_posterior': beta_bernoulli_pooled_log_posterior,
        'direct_sampler': placeholder_direct_sampler,
        'generated_quantities': beta_bernoulli_pooled_gq,
        'batch_type': beta_bernoulli_pooled_batch_specs,
        'initial_vector': beta_bernoulli_pooled_initial_vector,
        'get_num_gq': get_test_num_gq,
        'analytical_posterior': beta_bernoulli_pooled_analytical_posterior,
    },
    'test_normal_normal_pooled': {
        'log_posterior': normal_normal_pooled_log_posterior,
        'direct_sampler': placeholder_direct_sampler,
        'generated_quantities': normal_normal_pooled_gq,
        'batch_type': normal_normal_pooled_batch_specs,
        'initial_vector': normal_normal_pooled_initial_vector,
        'get_num_gq': get_test_num_gq,
        'analytical_posterior': normal_normal_pooled_analytical_posterior,
    },
    'test_beta_bernoulli_hierarchical': {
        'log_posterior': beta_bernoulli_hierarchical_log_posterior,
        'direct_sampler': beta_bernoulli_hierarchical_direct_dispatch,
        'generated_quantities': beta_bernoulli_hierarchical_gq,
        'batch_type': beta_bernoulli_hierarchical_batch_specs,
        'initial_vector': beta_bernoulli_hierarchical_initial_vector,
        'get_num_gq': get_test_num_gq,
        # Note: Hierarchical model doesn't have simple analytical posterior
    },
}
