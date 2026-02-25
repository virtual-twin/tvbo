# conftest.py — early environment setup for tests
#
# Must run before any JAX import to force CPU backend (avoids jax-metal
# XLA errors on Apple Silicon) and to set up virtual XLA devices for
# tests that use pmap.

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
n_devices = min(os.cpu_count() or 2, 8)
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={n_devices}")
