"""
JAX Configuration - MUST be imported before any JAX imports.

This module sets environment variables for JAX configuration including:
- Persistent compilation cache directory
- Minimum compile time threshold for caching
- GPU memory allocator settings
"""
import os
from pathlib import Path

# --- GPU MEMORY ALLOCATOR ---
# Use async allocator to reduce fragmentation (recommended by JAX for large models)
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

# --- PERSISTENT COMPILATION CACHE ---
# Enables cross-session caching of compiled kernels
_JAX_CACHE_DIR = Path.home() / ".cache" / "jax" / "bamcmc_cache"
_JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_JAX_CACHE_DIR))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1.0")
