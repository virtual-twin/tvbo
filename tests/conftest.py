"""Early environment setup for the test suite, plus helpers shared across test modules.

The environment part must run before any JAX import: it forces the CPU backend (jax-metal
raises XLA errors on Apple Silicon) and sets up the virtual XLA devices the pmap tests need.
"""

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


@pytest.fixture(autouse=True)
def _isolate_tvbo_logging():
    """Restore the ``tvbo`` logger after each test.

    ``tvbo.log.configure_logging`` installs a stream handler and sets
    ``propagate = False`` so the CLI owns its output and does not double-print through a
    host application's root logger. That is right for a CLI and wrong to leave behind in a
    test process: it is global and sticky, so once any test invokes the CLI, every later
    ``caplog`` assertion reads empty — caplog's handler sits on the root logger, which the
    records no longer reach — and the captured stream it kept is closed by then, so the
    handler raises ``I/O operation on closed file`` on the way past.

    Autouse because the pollution is invisible at the point it bites: the failing test is
    never the one that configured logging.
    """
    import logging

    logger = logging.getLogger("tvbo")
    handlers, propagate, level = list(logger.handlers), logger.propagate, logger.level
    try:
        yield
    finally:
        logger.handlers[:] = handlers
        logger.propagate = propagate
        logger.setLevel(level)


@pytest.fixture
def unwrapped():
    """``fn(code)`` -> *code* with all whitespace removed, for substring checks on codegen.

    Generated Python is black-formatted, so a long statement is split across lines at a
    column black chooses. Asserting on the statement's text rather than on its layout keeps
    a codegen test about what the emitter produces, not about how it was wrapped. A fixture
    rather than an importable helper because ``tests/`` is not a package.
    """
    return lambda code: "".join(code.split())
