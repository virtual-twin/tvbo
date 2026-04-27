# conftest.py — early environment setup for tests
#
# Must run before any JAX import to force CPU backend (avoids jax-metal
# XLA errors on Apple Silicon) and to set up virtual XLA devices for
# tests that use pmap.

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
n_devices = min(os.cpu_count() or 2, 8)
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={n_devices}")


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (e.g. tvboptim interop docs)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="slow test — pass --run-slow to include")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
