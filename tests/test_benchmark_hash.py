"""
Benchmark and Hash System Tests - Posterior hashing, benchmark management, prior config I/O.

Tests the untested benchmarking/caching system:
- posterior_hash.py: get_function_source_hash, get_data_hash, get_posterior_hash, compute_posterior_hash
- posterior_benchmark.py: PosteriorBenchmarkManager (save/load/compare/list/print), get_default_benchmark_dir, get_manager
- prior_config.py: save_prior_config, load_prior_config

Run with: pytest tests/test_benchmark_hash.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bamcmc.posterior_hash import (
    get_function_source_hash,
    get_data_hash,
    get_posterior_hash,
    compute_posterior_hash,
)
from bamcmc.posterior_benchmark import (
    PosteriorBenchmarkManager,
    get_default_benchmark_dir,
    get_manager,
)
from bamcmc.prior_config import save_prior_config, load_prior_config


# ============================================================================
# HELPERS
# ============================================================================

def _make_data(static=(10, 0.5), int_arrays=None, float_arrays=None):
    """Create a minimal MCMCData dict for testing."""
    if int_arrays is None:
        int_arrays = (np.array([1, 2, 3]),)
    if float_arrays is None:
        float_arrays = (np.array([1.0, 2.0, 3.0]),)
    return {"static": static, "int": int_arrays, "float": float_arrays}


def _make_model_config(**overrides):
    """Create a minimal model config dict with dummy functions."""
    config = {
        "log_posterior": _dummy_log_posterior,
        "direct_sampler": _dummy_direct_sampler,
        "generated_quantities": None,
        "initial_vector": _dummy_initial_vector,
        "batch_type": _dummy_batch_type,
        "get_num_gq": None,
    }
    config.update(overrides)
    return config


def _dummy_log_posterior(x):
    return -0.5 * np.sum(x**2)


def _dummy_direct_sampler(x):
    return x


def _dummy_initial_vector(config):
    return np.zeros(10)


def _dummy_batch_type():
    return "test"


def _dummy_different_function(x):
    return x + 1


def _save_benchmark_helper(mgr, posterior_hash="abc123", posterior_id="test_model"):
    """Save a benchmark and return it."""
    return mgr.save_benchmark(
        posterior_hash=posterior_hash,
        posterior_id=posterior_id,
        num_chains=8,
        fresh_compile_time=5.0,
        iteration_time=0.001,
        benchmark_iterations=100,
        cached_compile_time=1.5,
    )


# ============================================================================
# POSTERIOR HASH TESTS
# ============================================================================


class TestGetFunctionSourceHash:
    """Tests for get_function_source_hash."""

    def test_consistent_hash(self):
        """Same function produces same hash across calls."""
        h1 = get_function_source_hash(_dummy_log_posterior)
        h2 = get_function_source_hash(_dummy_log_posterior)
        assert h1 == h2

    def test_different_functions_different_hash(self):
        """Different functions produce different hashes."""
        h1 = get_function_source_hash(_dummy_log_posterior)
        h2 = get_function_source_hash(_dummy_different_function)
        assert h1 != h2

    def test_none_returns_none_string(self):
        """None input returns 'none'."""
        assert get_function_source_hash(None) == "none"

    def test_builtin_does_not_crash(self):
        """Built-in function (no source) returns a hash without crashing."""
        h = get_function_source_hash(len)
        assert isinstance(h, str)
        assert len(h) == 16


class TestGetDataHash:
    """Tests for get_data_hash."""

    def test_deterministic(self):
        """Same data produces same hash."""
        data = _make_data()
        h1 = get_data_hash(data)
        h2 = get_data_hash(data)
        assert h1 == h2

    def test_different_float_data(self):
        """Different float data produces different hash."""
        data1 = _make_data(float_arrays=(np.array([1.0, 2.0]),))
        data2 = _make_data(float_arrays=(np.array([3.0, 4.0]),))
        assert get_data_hash(data1) != get_data_hash(data2)

    def test_different_int_data(self):
        """Different int data produces different hash."""
        data1 = _make_data(int_arrays=(np.array([1, 2]),))
        data2 = _make_data(int_arrays=(np.array([3, 4]),))
        assert get_data_hash(data1) != get_data_hash(data2)

    def test_different_static(self):
        """Different static values produce different hash."""
        data1 = _make_data(static=(10, 0.5))
        data2 = _make_data(static=(20, 0.5))
        assert get_data_hash(data1) != get_data_hash(data2)

    def test_empty_data(self):
        """Empty data dict returns a hash without crashing."""
        h = get_data_hash({})
        assert isinstance(h, str)
        assert len(h) == 16


class TestGetPosteriorHash:
    """Tests for get_posterior_hash and compute_posterior_hash."""

    def test_deterministic(self):
        """Same inputs produce same hash."""
        data = _make_data()
        config = _make_model_config()
        h1 = get_posterior_hash("test", config, data, num_chains=4)
        h2 = get_posterior_hash("test", config, data, num_chains=4)
        assert h1 == h2

    def test_different_posterior_id(self):
        """Different posterior_id produces different hash."""
        data = _make_data()
        config = _make_model_config()
        h1 = get_posterior_hash("model_a", config, data)
        h2 = get_posterior_hash("model_b", config, data)
        assert h1 != h2

    def test_different_num_chains(self):
        """Different num_chains produces different hash."""
        data = _make_data()
        config = _make_model_config()
        h1 = get_posterior_hash("test", config, data, num_chains=4)
        h2 = get_posterior_hash("test", config, data, num_chains=8)
        assert h1 != h2

    def test_compute_delegates_to_get(self):
        """compute_posterior_hash returns same result as get_posterior_hash."""
        data = _make_data()
        config = _make_model_config()
        h1 = get_posterior_hash("test", config, data, num_chains=4)
        h2 = compute_posterior_hash("test", config, data, num_chains=4)
        assert h1 == h2


# ============================================================================
# POSTERIOR BENCHMARK MANAGER TESTS
# ============================================================================


class TestBenchmarkManagerSaveLoad:
    """Tests for save_benchmark and load_benchmark."""

    def test_save_returns_expected_keys(self):
        """save_benchmark returns dict with expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            result = _save_benchmark_helper(mgr)
            expected_keys = {
                "posterior_hash", "posterior_id", "bamcmc_version",
                "git", "hardware", "config", "results", "timestamp",
            }
            assert expected_keys.issubset(result.keys())

    def test_save_creates_file(self):
        """save_benchmark creates a JSON file on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            _save_benchmark_helper(mgr, posterior_hash="xyz789")
            path = Path(tmpdir) / "benchmark_xyz789.json"
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["posterior_hash"] == "xyz789"

    def test_load_roundtrip(self):
        """load_benchmark round-trips saved data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            saved = _save_benchmark_helper(mgr, posterior_hash="rt001")
            loaded = mgr.load_benchmark("rt001")
            assert loaded is not None
            assert loaded["posterior_hash"] == "rt001"
            assert loaded["results"]["iteration_time_s"] == saved["results"]["iteration_time_s"]

    def test_load_nonexistent(self):
        """load_benchmark returns None for nonexistent hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            assert mgr.load_benchmark("nonexistent") is None

    def test_load_corrupted_json(self):
        """load_benchmark returns None for corrupted JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            path = Path(tmpdir) / "benchmark_corrupt.json"
            path.write_text("{invalid json content")
            assert mgr.load_benchmark("corrupt") is None


class TestBenchmarkManagerHardware:
    """Tests for check_hardware_match."""

    def test_hardware_match_true(self):
        """Returns True when hardware matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            fake_hw = {"gpu_name": "RTX 4090", "jax_backend": "gpu"}
            benchmark = {"hardware": fake_hw}
            with patch("bamcmc.posterior_benchmark.get_hardware_info", return_value=fake_hw):
                assert mgr.check_hardware_match(benchmark) is True

    def test_hardware_match_false_gpu(self):
        """Returns False when GPU name differs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            cached_hw = {"gpu_name": "RTX 3090", "jax_backend": "gpu"}
            current_hw = {"gpu_name": "RTX 4090", "jax_backend": "gpu"}
            benchmark = {"hardware": cached_hw}
            with patch("bamcmc.posterior_benchmark.get_hardware_info", return_value=current_hw):
                assert mgr.check_hardware_match(benchmark) is False


class TestBenchmarkManagerCachedBenchmark:
    """Tests for get_cached_benchmark."""

    def test_returns_benchmark_when_hw_matches(self):
        """Returns benchmark when exists and hardware matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            saved = _save_benchmark_helper(mgr, posterior_hash="hw001")
            hw = saved["hardware"]
            with patch("bamcmc.posterior_benchmark.get_hardware_info", return_value=hw):
                result = mgr.get_cached_benchmark("hw001", require_hardware_match=True)
            assert result is not None
            assert result["posterior_hash"] == "hw001"

    def test_returns_none_when_hw_mismatch_required(self):
        """Returns None when hardware doesn't match and require_hardware_match=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            _save_benchmark_helper(mgr, posterior_hash="hw002")
            different_hw = {"gpu_name": "totally_different", "jax_backend": "cpu"}
            with patch("bamcmc.posterior_benchmark.get_hardware_info", return_value=different_hw):
                result = mgr.get_cached_benchmark("hw002", require_hardware_match=True)
            assert result is None

    def test_returns_benchmark_when_hw_mismatch_not_required(self):
        """Returns benchmark when hardware doesn't match but require_hardware_match=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            _save_benchmark_helper(mgr, posterior_hash="hw003")
            different_hw = {"gpu_name": "totally_different", "jax_backend": "cpu"}
            with patch("bamcmc.posterior_benchmark.get_hardware_info", return_value=different_hw):
                result = mgr.get_cached_benchmark("hw003", require_hardware_match=False)
            assert result is not None


class TestBenchmarkManagerCompare:
    """Tests for compare_benchmark."""

    def test_no_cached_benchmark(self):
        """No cached benchmark: has_cached=False, deltas are None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            result = mgr.compare_benchmark("noexist", 0.002, 6.0)
            assert result["has_cached"] is False
            assert result["iteration_delta"] is None
            assert result["compile_delta"] is None

    def test_with_cached_benchmark(self):
        """With cached benchmark: computes correct deltas and percentages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            _save_benchmark_helper(mgr, posterior_hash="cmp001")
            # Saved: iteration_time=0.001, compile_time=5.0
            result = mgr.compare_benchmark("cmp001", 0.002, 6.0)
            assert result["has_cached"] is True
            assert result["cached_iteration_time"] == pytest.approx(0.001)
            assert result["cached_compile_time"] == pytest.approx(5.0)
            # iteration: 0.002 - 0.001 = 0.001
            assert result["iteration_delta"] == pytest.approx(0.001)
            assert result["iteration_pct_change"] == pytest.approx(100.0)
            # compile: 6.0 - 5.0 = 1.0
            assert result["compile_delta"] == pytest.approx(1.0)
            assert result["compile_pct_change"] == pytest.approx(20.0)


class TestBenchmarkManagerListPrint:
    """Tests for list_benchmarks, print_benchmark, print_comparison."""

    def test_list_benchmarks_sorted(self):
        """list_benchmarks returns saved benchmarks sorted by timestamp (most recent first)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            _save_benchmark_helper(mgr, posterior_hash="list001", posterior_id="first")
            _save_benchmark_helper(mgr, posterior_hash="list002", posterior_id="second")
            benchmarks = mgr.list_benchmarks()
            assert len(benchmarks) == 2
            # Most recent should be first
            assert benchmarks[0]["posterior_id"] == "second"

    def test_print_benchmark_smoke(self, capsys):
        """print_benchmark runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            bm = _save_benchmark_helper(mgr)
            mgr.print_benchmark(bm)
            captured = capsys.readouterr()
            assert "test_model" in captured.out

    def test_print_comparison_smoke(self, capsys):
        """print_comparison runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PosteriorBenchmarkManager(cache_dir=tmpdir)
            _save_benchmark_helper(mgr, posterior_hash="prt001")
            comparison = mgr.compare_benchmark("prt001", 0.002, 6.0)
            mgr.print_comparison(comparison, posterior_id="test_model")
            captured = capsys.readouterr()
            assert "BENCHMARK COMPARISON" in captured.out


class TestBenchmarkManagerFactory:
    """Tests for get_default_benchmark_dir and get_manager."""

    def test_default_dir_respects_env_var(self):
        """get_default_benchmark_dir respects BAMCMC_BENCHMARKDIR env var."""
        with patch.dict(os.environ, {"BAMCMC_BENCHMARKDIR": "/tmp/custom_bench"}):
            assert get_default_benchmark_dir() == "/tmp/custom_bench"

    def test_get_manager_returns_instance(self):
        """get_manager returns a PosteriorBenchmarkManager instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = get_manager(cache_dir=tmpdir)
            assert isinstance(mgr, PosteriorBenchmarkManager)


# ============================================================================
# PRIOR CONFIG TESTS
# ============================================================================


def _mock_get_model_paths(output_dir, model_name):
    """Return paths dict with 'base' pointing to output_dir/model_name."""
    base = Path(output_dir) / model_name
    base.mkdir(parents=True, exist_ok=True)
    return {"base": base}


class TestPriorConfig:
    """Tests for save_prior_config and load_prior_config."""

    def test_save_creates_file(self):
        """save_prior_config creates JSON file and returns path string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("bamcmc.output_management.get_model_paths", side_effect=_mock_get_model_paths):
                path = save_prior_config(tmpdir, "test_model", {"a": 1, "b": [2, 3]})
            assert isinstance(path, str)
            assert Path(path).exists()

    def test_roundtrip(self):
        """load_prior_config round-trips saved data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"shared_risk_priors": {"mu": 0.0, "sigma": 1.0}}
            with patch("bamcmc.output_management.get_model_paths", side_effect=_mock_get_model_paths):
                save_prior_config(tmpdir, "test_model", config)
                loaded = load_prior_config(tmpdir, "test_model")
            assert loaded == config

    def test_load_nonexistent(self):
        """load_prior_config returns None when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("bamcmc.output_management.get_model_paths", side_effect=_mock_get_model_paths):
                assert load_prior_config(tmpdir, "no_such_model") is None

    def test_numpy_array_serialization(self):
        """save_prior_config handles numpy arrays (converted to lists)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"params": np.array([1.0, 2.0, 3.0])}
            with patch("bamcmc.output_management.get_model_paths", side_effect=_mock_get_model_paths):
                save_prior_config(tmpdir, "test_model", config)
                loaded = load_prior_config(tmpdir, "test_model")
            assert loaded["params"] == [1.0, 2.0, 3.0]
