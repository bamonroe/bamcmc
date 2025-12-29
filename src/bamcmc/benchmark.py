"""
Performance Benchmark for MCMC Sampler

Runs a moderately-sized posterior (~20 seconds) and saves timing metadata to JSON.
Used to detect performance regressions (>10% wall time increase).

Usage:
    python -m bamcmc.benchmark              # Run benchmark
    python -m bamcmc.benchmark --compare    # Compare against baseline
    python -m bamcmc.benchmark --save       # Save as new baseline
"""

import json
import os
import time
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import jax

from .mcmc_backend import rmcmc
from .registry import register_posterior, _REGISTRY
from . import test_posteriors


# Benchmark configuration - designed to take ~15-20 seconds
# Uses Normal-Normal pooled model with high iteration count
BENCHMARK_CONFIG = {
    'POSTERIOR_ID': 'test_normal_normal_pooled',
    'GPU_PREALLOCATION': True,
    'USE_DOUBLE': True,
    'rng_seed': 2024,
    'BENCHMARK': 100,
    'BURN_ITER': 50000,
    'THIN_ITERATION': 1,
    'NUM_COLLECT': 100000,
    'PROPOSAL': 'chain_mean',
    'NUM_CHAINS_A': 100,
    'NUM_CHAINS_B': 100,
    'NUM_SUPERCHAINS': 40,  # 40 superchains x 5 subchains = 200 chains
    'LAST_ITERS': 100000,
}

# Data configuration for normal-normal model
BENCHMARK_DATA_CONFIG = {
    'n_obs': 500,
    'true_mu': 5.0,
    'sigma': 2.0,
    'mu_0': 0.0,
    'tau_0': 10.0,
    'seed': 42,
}

BASELINE_FILE = Path(__file__).parent / 'benchmark_baseline.json'
REGRESSION_THRESHOLD = 0.10  # 10% regression threshold


def generate_benchmark_data():
    """Generate data for normal-normal pooled model."""
    cfg = BENCHMARK_DATA_CONFIG
    np.random.seed(cfg['seed'])

    # Generate normal observations
    y = np.random.normal(cfg['true_mu'], cfg['sigma'], cfg['n_obs'])

    data = {
        "static": (cfg['mu_0'], cfg['tau_0'], cfg['sigma']),
        "int": (),
        "float": (y,)
    }

    return data


def get_system_info():
    """Collect system information for benchmark context."""
    return {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'python_version': platform.python_version(),
        'jax_version': jax.__version__,
        'jax_backend': jax.default_backend(),
        'device_count': jax.device_count(),
        'devices': [str(d) for d in jax.devices()],
    }


def run_benchmark(verbose=True):
    """
    Run the performance benchmark.

    Returns:
        dict: Benchmark results including timing and metadata
    """
    if verbose:
        print("=" * 70)
        print("PERFORMANCE BENCHMARK")
        print("=" * 70)
        print()

    # Register test posteriors
    original_registrations = {}
    for name, config in test_posteriors.TEST_POSTERIORS.items():
        if name in _REGISTRY:
            original_registrations[name] = _REGISTRY[name]
        register_posterior(name, config)

    try:
        # Generate data
        if verbose:
            print("Generating benchmark data...")
        data = generate_benchmark_data()

        # Run MCMC
        if verbose:
            print(f"Running benchmark posterior: {BENCHMARK_CONFIG['POSTERIOR_ID']}")
            print(f"  Observations: {BENCHMARK_DATA_CONFIG['n_obs']}")
            print(f"  Chains: {BENCHMARK_CONFIG['NUM_CHAINS_A'] + BENCHMARK_CONFIG['NUM_CHAINS_B']}")
            print(f"  Iterations: {BENCHMARK_CONFIG['BURN_ITER']} burn + {BENCHMARK_CONFIG['NUM_COLLECT']} collect")
            print()

        overall_start = time.perf_counter()
        history, diagnostics, mcmc_config, _ = rmcmc(BENCHMARK_CONFIG.copy(), data)
        overall_end = time.perf_counter()

        # Collect results
        results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': get_system_info(),
            'config': {
                'posterior_id': BENCHMARK_CONFIG['POSTERIOR_ID'],
                'n_obs': BENCHMARK_DATA_CONFIG['n_obs'],
                'n_chains': BENCHMARK_CONFIG['NUM_CHAINS_A'] + BENCHMARK_CONFIG['NUM_CHAINS_B'],
                'burn_iter': BENCHMARK_CONFIG['BURN_ITER'],
                'num_collect': BENCHMARK_CONFIG['NUM_COLLECT'],
            },
            'timing': {
                'compile_time': diagnostics.get('compile_time', 0.0),
                'wall_time': diagnostics.get('wall_time', 0.0),
                'avg_iter_time': diagnostics.get('avg_iter_time', 0.0),
                'total_iterations': diagnostics.get('total_iterations', 0),
                'overall_time': overall_end - overall_start,
            },
            'diagnostics': {
                'max_rhat': float(np.max(diagnostics['rhat'])) if diagnostics['rhat'] is not None else None,
                'converged': bool(np.max(diagnostics['rhat']) < 1.1) if diagnostics['rhat'] is not None else None,
            }
        }

        if verbose:
            print()
            print("=" * 70)
            print("BENCHMARK RESULTS")
            print("=" * 70)
            print(f"  Compile Time:     {results['timing']['compile_time']:.2f} s")
            print(f"  Wall Time:        {results['timing']['wall_time']:.2f} s")
            print(f"  Overall Time:     {results['timing']['overall_time']:.2f} s")
            print(f"  Avg Iter Time:    {results['timing']['avg_iter_time']*1000:.3f} ms")
            print(f"  Max R-hat:        {results['diagnostics']['max_rhat']:.4f}")
            print(f"  Converged:        {'Yes' if results['diagnostics']['converged'] else 'No'}")

        return results

    finally:
        # Restore original registry state
        for name in test_posteriors.TEST_POSTERIORS.keys():
            if name in original_registrations:
                _REGISTRY[name] = original_registrations[name]
            elif name in _REGISTRY:
                del _REGISTRY[name]


def save_baseline(results):
    """Save benchmark results as the new baseline."""
    with open(BASELINE_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline saved to: {BASELINE_FILE}")


def load_baseline():
    """Load the baseline benchmark results."""
    if not BASELINE_FILE.exists():
        return None
    with open(BASELINE_FILE, 'r') as f:
        return json.load(f)


def compare_to_baseline(results, baseline):
    """
    Compare current results to baseline.

    Returns:
        tuple: (passed, comparison_dict)
    """
    if baseline is None:
        return True, {'message': 'No baseline found - skipping comparison'}

    # Compare key metrics
    current_wall = results['timing']['wall_time']
    baseline_wall = baseline['timing']['wall_time']
    wall_diff = (current_wall - baseline_wall) / baseline_wall

    current_compile = results['timing']['compile_time']
    baseline_compile = baseline['timing']['compile_time']
    compile_diff = (current_compile - baseline_compile) / baseline_compile if baseline_compile > 0 else 0

    current_iter = results['timing']['avg_iter_time'] or 0
    baseline_iter = baseline['timing']['avg_iter_time'] or 0
    iter_diff = (current_iter - baseline_iter) / baseline_iter if baseline_iter > 0 else 0

    comparison = {
        'wall_time': {
            'baseline': baseline_wall,
            'current': current_wall,
            'diff_pct': wall_diff * 100,
            'regression': wall_diff > REGRESSION_THRESHOLD,
        },
        'compile_time': {
            'baseline': baseline_compile,
            'current': current_compile,
            'diff_pct': compile_diff * 100,
            'regression': compile_diff > REGRESSION_THRESHOLD,
        },
        'avg_iter_time': {
            'baseline': baseline_iter,
            'current': current_iter,
            'diff_pct': iter_diff * 100,
            'regression': iter_diff > REGRESSION_THRESHOLD,
        },
        'baseline_timestamp': baseline['timestamp'],
    }

    # Overall pass/fail based on wall time (the primary metric)
    passed = not comparison['wall_time']['regression']

    return passed, comparison


def print_comparison(comparison):
    """Print comparison results."""
    print()
    print("=" * 70)
    print("COMPARISON TO BASELINE")
    print("=" * 70)

    if 'message' in comparison:
        print(f"  {comparison['message']}")
        return

    print(f"  Baseline from: {comparison['baseline_timestamp']}")
    print()

    for metric, data in comparison.items():
        if metric == 'baseline_timestamp':
            continue

        status = "REGRESSION" if data['regression'] else "OK"
        sign = "+" if data['diff_pct'] > 0 else ""

        metric_name = metric.replace('_', ' ').title()
        print(f"  {metric_name}:")
        print(f"    Baseline: {data['baseline']:.4f} s")
        print(f"    Current:  {data['current']:.4f} s")
        print(f"    Change:   {sign}{data['diff_pct']:.1f}% [{status}]")
        print()


def run_benchmark_test(save=False, compare=True, verbose=True):
    """
    Run benchmark test with optional save and comparison.

    Args:
        save: If True, save results as new baseline
        compare: If True, compare against existing baseline
        verbose: If True, print detailed output

    Returns:
        tuple: (passed, results, comparison)
    """
    results = run_benchmark(verbose=verbose)

    passed = True
    comparison = None

    if compare:
        baseline = load_baseline()
        passed, comparison = compare_to_baseline(results, baseline)
        if verbose:
            print_comparison(comparison)

    if save:
        save_baseline(results)

    if verbose:
        print()
        if passed:
            print("BENCHMARK PASSED")
        else:
            print(f"BENCHMARK FAILED - Wall time regression > {REGRESSION_THRESHOLD*100:.0f}%")

    return passed, results, comparison


def main():
    """Main entry point for benchmark script."""
    import argparse

    parser = argparse.ArgumentParser(description='Run MCMC performance benchmark')
    parser.add_argument('--save', action='store_true',
                       help='Save results as new baseline')
    parser.add_argument('--compare', action='store_true', default=True,
                       help='Compare against existing baseline (default: True)')
    parser.add_argument('--no-compare', action='store_false', dest='compare',
                       help='Skip comparison')
    parser.add_argument('--json', type=str, default=None,
                       help='Save results to specified JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output (only essential info)')

    args = parser.parse_args()

    passed, results, comparison = run_benchmark_test(
        save=args.save,
        compare=args.compare,
        verbose=not args.quiet
    )

    if args.json:
        output = {
            'results': results,
            'comparison': comparison,
            'passed': passed,
        }
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.json}")

    exit(0 if passed else 1)


if __name__ == '__main__':
    main()
