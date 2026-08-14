"""Early environment setup for the test suite, plus helpers shared across test modules.

The environment part must run before any JAX import: it forces the CPU backend (jax-metal
raises XLA errors on Apple Silicon) and sets up the virtual XLA devices the pmap tests need.

It also gives each xdist worker its own ``TVB_USER_HOME``. TVB derives its storage from
that variable — including a log folder it ``os.makedirs`` without ``exist_ok`` on import —
so parallel workers importing tvb race on that mkdir and the loser raises FileExistsError.
"""

import os
import tempfile

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
n_devices = min(os.cpu_count() or 2, 8)
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={n_devices}")

_tvb_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _tvb_worker:
    os.environ["TVB_USER_HOME"] = os.path.join(tempfile.gettempdir(), f"tvb-home-{_tvb_worker}")
else:
    os.environ.setdefault("TVB_USER_HOME", os.path.join(tempfile.gettempdir(), "tvb-home-main"))


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


@pytest.fixture(scope="session")
def icosphere():
    """``fn(subdivisions, radius)`` -> a closed triangulated sphere ``(vertices, faces)``.

    The reference geometry for surface work: closed, so it has no boundary at all, and the
    only curved surface whose Laplace-Beltrami spectrum is known in closed form
    (``l(l+1)/R**2`` with multiplicity ``2l+1``). A fixture rather than an importable
    helper because ``tests/`` is not a package.
    """
    import numpy as np

    def build(subdivisions: int = 3, radius: float = 1.0):
        t = (1 + 5**0.5) / 2
        vertices = np.array(
            [
                [-1, t, 0],
                [1, t, 0],
                [-1, -t, 0],
                [1, -t, 0],
                [0, -1, t],
                [0, 1, t],
                [0, -1, -t],
                [0, 1, -t],
                [t, 0, -1],
                [t, 0, 1],
                [-t, 0, -1],
                [-t, 0, 1],
            ],
            float,
        )
        faces = np.array(
            [
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                [4, 9, 5],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ]
        )
        for _ in range(subdivisions):
            midpoints, split, points = {}, [], list(vertices)

            def midpoint(a, b):
                key = (min(a, b), max(a, b))
                if key not in midpoints:
                    midpoints[key] = len(points)
                    points.append((np.asarray(points[a]) + points[b]) / 2)
                return midpoints[key]

            for a, b, c in faces:
                ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
                split += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            vertices, faces = np.array(points), np.array(split)
        return radius * vertices / np.linalg.norm(vertices, axis=1, keepdims=True), faces

    return build


@pytest.fixture
def unwrapped():
    """``fn(code)`` -> *code* with all whitespace removed, for substring checks on codegen.

    Generated Python is black-formatted, so a long statement is split across lines at a
    column black chooses. Asserting on the statement's text rather than on its layout keeps
    a codegen test about what the emitter produces, not about how it was wrapped. A fixture
    rather than an importable helper because ``tests/`` is not a package.
    """
    return lambda code: "".join(code.split())
