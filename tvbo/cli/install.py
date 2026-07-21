"""``tvbo install`` — provision optional components that pip alone cannot place.

Some extras have a half that is not distributed on PyPI. ``pip install tvbo[…]``
installs the Python wrapper; this verb installs the native component and links it
onto the current environment so imports resolve.

Currently one target:

* ``tvbo install auto7p`` — the AUTO-07p bifurcation-continuation engine used
  through :mod:`pycobi`. AUTO-07p is a Fortran program: this command locates an
  existing build (or builds one from source with ``--build``), then drops a
  ``.pth`` link to its ``python/`` front-end into the active environment's
  site-packages, so ``import auto`` resolves for every process using this venv.
"""
from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import typer

from . import _common

app = typer.Typer(name="install", no_args_is_help=True)

AUTO_REPO = "https://github.com/auto-07p/auto-07p"
PTH_NAME = "auto-07p.pth"

# Locations searched for an existing AUTO-07p install, in order, before falling
# back to ``--build``. Each is tested with :func:`_is_auto_dir`.
_DEFAULT_AUTO_LOCATIONS = (
    "/opt/auto-07p",
    "/usr/local/auto-07p",
    "/Applications/auto-07p",
    "~/auto-07p",
    "~/.local/share/auto-07p",
)


def _site_packages() -> Path:
    """The site-packages of the running interpreter (venv-aware)."""
    return Path(sysconfig.get_paths()["purelib"])


def _is_auto_dir(path: Path) -> bool:
    """True if *path* is an AUTO-07p tree whose Python front-end is present."""
    return (path / "python" / "auto" / "__init__.py").is_file()


def _search_auto_dir() -> Path | None:
    """Locate an AUTO-07p install from ``$AUTO_DIR``, then the known locations."""
    candidates: list[str] = []
    if os.environ.get("AUTO_DIR"):
        candidates.append(os.environ["AUTO_DIR"])
    candidates.extend(_DEFAULT_AUTO_LOCATIONS)
    for c in candidates:
        p = Path(c).expanduser()
        if _is_auto_dir(p):
            return p.resolve()
    return None


def _module_importable(module: str) -> bool:
    """Whether *module* imports in a fresh interpreter using this venv.

    A subprocess is used so a ``.pth`` written earlier in this run — which the
    already-initialised parent process would not see — is honoured.
    """
    return (
        subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
        ).returncode
        == 0
    )


def _run_step(cmd: list[str], *, cwd: str | None = None, what: str) -> None:
    """Run a build step, converting a non-zero exit into a clean ``die``.

    Keeps a failed clone/configure/make from surfacing as a raw
    ``CalledProcessError`` traceback — the reason is reported the same way as
    every other fatal path in this command.
    """
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        _common.die(f"{what} failed (exit {result.returncode}): {' '.join(cmd)}")


def _build_auto(prefix: Path) -> Path:
    """Clone and build AUTO-07p under *prefix*, returning the build directory.

    Requires a source toolchain (git, make, a Fortran compiler). Missing tools
    are reported up front, and a build-step failure is reported cleanly rather
    than as an opaque traceback.
    """
    import shutil

    prefix = prefix.expanduser().resolve()
    if _is_auto_dir(prefix):
        _common.info(f"AUTO-07p already built at {prefix} — skipping build")
        return prefix

    missing = [t for t in ("git", "make") if shutil.which(t) is None]
    if shutil.which("gfortran") is None and shutil.which("gcc") is None:
        missing.append("gfortran (or gcc)")
    if missing:
        _common.die(
            "cannot build AUTO-07p — missing build tools: "
            + ", ".join(missing)
            + ". Install them (e.g. `brew install gcc make git` or "
            "`apt install gfortran make git`) and retry, or point --auto-dir at "
            "an existing build."
        )

    if (prefix / ".git").is_dir():
        _common.info(f"updating AUTO-07p source in {prefix}")
        _run_step(["git", "-C", str(prefix), "pull", "--ff-only"], what="git pull")
    elif prefix.exists() and any(prefix.iterdir()):
        _common.die(
            f"--prefix {prefix} already exists and is not an AUTO-07p checkout; "
            "remove it or choose another --prefix."
        )
    else:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        _common.info(f"cloning AUTO-07p into {prefix}")
        _run_step(
            ["git", "clone", "--depth", "1", AUTO_REPO, str(prefix)], what="git clone"
        )
    _common.info("configuring AUTO-07p")
    _run_step(["./configure"], cwd=str(prefix), what="configure")
    _common.info("building AUTO-07p (make) — this can take a few minutes")
    _run_step(["make"], cwd=str(prefix), what="make")
    if not _is_auto_dir(prefix):
        _common.die(f"build finished but {prefix}/python/auto is still missing")
    return prefix


def _write_link(auto_dir: Path, *, force: bool) -> tuple[Path, bool]:
    """Point the ``.pth`` link at ``auto_dir/python``. Returns (path, changed)."""
    target = str(auto_dir / "python")
    pth = _site_packages() / PTH_NAME
    if pth.exists() and pth.read_text().strip() == target and not force:
        return pth, False
    pth.write_text(target + "\n")
    return pth, True


@app.command("auto7p")
def auto7p(
    auto_dir: str = typer.Option(
        None,
        "--auto-dir",
        metavar="PATH",
        help="Path to an existing AUTO-07p install. Defaults to $AUTO_DIR, then "
        "common locations.",
    ),
    build: bool = typer.Option(
        False,
        "--build",
        help="Build AUTO-07p from source when no install is found.",
    ),
    prefix: str = typer.Option(
        "~/.local/share/auto-07p",
        "--prefix",
        metavar="PATH",
        help="Where to clone + build AUTO-07p when --build is used.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Rewrite the .pth link even if it is already correct."
    ),
    uninstall: bool = typer.Option(
        False, "--uninstall", help="Remove the AUTO-07p link from this environment."
    ),
) -> None:
    """Install AUTO-07p and link it onto this environment for pycobi continuation.

    The pip half (pycobi) comes from the ``auto7p`` extra; this command
    provides the native AUTO-07p engine, which is not on PyPI. It locates an
    existing build (``--auto-dir`` / ``$AUTO_DIR`` / common paths), or builds one
    with ``--build``, then links its ``python/`` front-end into site-packages so
    ``import auto`` resolves. Safe to re-run — this is the step to repeat after
    recreating the virtualenv, since the link lives inside it. Pass ``--uninstall``
    to remove the link (the AUTO-07p build itself is left untouched).
    """
    if uninstall:
        pth = _site_packages() / PTH_NAME
        if pth.exists():
            pth.unlink()
            _common.info(f"removed AUTO-07p link → {pth}")
        else:
            _common.info(f"no AUTO-07p link to remove ({pth})")
        return

    if auto_dir:
        # An explicit path is honoured strictly: a typo must fail loudly rather
        # than fall through to a different auto-detected install.
        given = Path(auto_dir).expanduser()
        if not _is_auto_dir(given):
            _common.die(
                f"--auto-dir {auto_dir} is not an AUTO-07p install "
                "(no python/auto/__init__.py under it)."
            )
        resolved = given.resolve()
    else:
        resolved = _search_auto_dir()
        if resolved is None:
            if build:
                resolved = _build_auto(Path(prefix))
            else:
                _common.die(
                    "no AUTO-07p install found (searched $AUTO_DIR and "
                    f"{', '.join(_DEFAULT_AUTO_LOCATIONS)}). Re-run with --build to "
                    f"build it from {AUTO_REPO}, or pass --auto-dir PATH."
                )

    pth, changed = _write_link(resolved, force=force)
    _common.info(
        f"linked AUTO-07p ({resolved}) → {pth}"
        if changed
        else f"AUTO-07p already linked → {pth}"
    )

    if not _module_importable("auto"):
        _common.die(
            f"linked {resolved}/python but `import auto` still fails — the build "
            "may be incomplete (no compiled module). Re-run with --build, or "
            "rebuild AUTO-07p in place."
        )
    if not _module_importable("pycobi"):
        _common.warn(
            "pycobi is not installed — the Python wrapper is required to drive "
            'AUTO-07p. Install it with: pip install "tvbo[auto7p]"'
        )

    if os.environ.get("AUTO_DIR") != str(resolved):
        _common.warn(
            f"set AUTO_DIR for continuation at runtime: export AUTO_DIR={resolved} "
            "(add it to your shell profile). pycobi reads it to find the auto "
            "command-line tools."
        )
    _common.info("auto7p ready — `import auto, pycobi` resolve in this environment.")
