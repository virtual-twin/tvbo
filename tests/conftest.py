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
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help=(
            "Re-baseline the golden corpora instead of asserting against them. "
            "Re-baselining changes what TVBO promises to emit — review the diff and "
            "commit it on its own."
        ),
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="slow test — pass --run-slow to include")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def regenerate(pytestconfig) -> bool:
    """Whether the golden corpora are being re-baselined rather than asserted."""
    return bool(pytestconfig.getoption("--regenerate-golden"))


def pytest_sessionfinish(session, exitstatus):
    """Never let a re-baselining run report success.

    Regeneration asserts nothing — it overwrites every reference with whatever the current
    code produces. A green run would be indistinguishable from a suite that passed, which
    is exactly how an unreviewed re-baseline reaches main.

    A run that actually failed keeps its own status. Overriding that too would hide the
    case that matters most: if a model raised while regenerating, its reference was never
    written, and reporting that identically to a clean regeneration is how a corpus with a
    hole in it gets committed.
    """
    if not session.config.getoption("--regenerate-golden", default=False):
        return
    if exitstatus not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        return

    session.exitstatus = pytest.ExitCode.USAGE_ERROR
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep(
            "=",
            "golden corpora REGENERATED — nothing was asserted; review the diff and commit it on its own",
            red=True,
        )


@pytest.fixture
def unwrapped():
    """``fn(code)`` -> *code* with all whitespace removed, for substring checks on codegen.

    Generated Python is black-formatted, so a long statement is split across lines at a
    column black chooses. Asserting on the statement's text rather than on its layout keeps
    a codegen test about what the emitter produces, not about how it was wrapped. A fixture
    rather than an importable helper because ``tests/`` is not a package.
    """
    return lambda code: "".join(code.split())
