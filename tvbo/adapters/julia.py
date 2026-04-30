"""Python ↔ Julia bridge using juliacall (PythonCall.jl).

Replaces the legacy pyjulia (``julia`` package) which is unmaintained
and emits spurious MainInclude warnings on Julia ≥ 1.3.

Julia package management is handled automatically by juliapkg via
``tvbo/juliapkg.json``. When juliacall is first imported, juliapkg
will install Julia (if needed) and all declared packages.
"""

import re
from typing import Any, Optional

_julia_main = None
_installed_packages = set()


def get_julia(compiled_modules=True):
    """Return the Julia Main module (lazily initialized).

    On first call, juliacall/juliapkg will automatically:
    - Install Julia if not present
    - Resolve and install packages declared in tvbo/juliapkg.json
    - Precompile packages

    Parameters
    ----------
    compiled_modules : bool
        Ignored (kept for API compatibility). juliacall always uses
        precompiled modules.

    Returns
    -------
    (None, Main)
        Tuple of ``(None, Main)`` for backward compatibility.
        The first element was the pyjulia ``Julia`` instance;
        with juliacall it is not needed.
    """
    global _julia_main
    if _julia_main is None:
        try:
            from juliacall import Main
        except ImportError:
            raise ImportError("juliacall package not installed. Run: pip install juliacall")
        _julia_main = Main
    return None, _julia_main


def install_julia_package(package_name: str, Main: Optional[Any] = None, update: bool = False):
    """Install a Julia package if not already installed.

    Args:
        package_name: Name of the Julia package to install
        Main: Julia Main module (will be retrieved if None)
        update: If True, update the package to the latest version
    """
    if Main is None:
        _, Main = get_julia()

    if package_name in _installed_packages and not update:
        return

    print(f"{'Updating' if update else 'Installing'} Julia package: {package_name}...")
    try:
        if update:
            Main.seval(f'import Pkg; Pkg.update("{package_name}")')
        else:
            Main.seval(f'import Pkg; Pkg.add("{package_name}")')
        _installed_packages.add(package_name)
        print(f"Successfully {'updated' if update else 'installed'} {package_name}")
    except Exception as e:
        print(f"Warning: Failed to {'update' if update else 'install'} {package_name}: {e}")
        raise


def eval_with_auto_install(code, max_retries=3):
    """Evaluate Julia code, automatically installing missing packages if needed."""
    _, Main = get_julia()

    for attempt in range(max_retries):
        try:
            return Main.seval(code)
        except Exception as e:
            error_msg = str(e)
            # Check if it's a missing package error
            match = re.search(r"Package (\w+) not found", error_msg)
            if match and attempt < max_retries - 1:
                package_name = match.group(1)
                try:
                    install_julia_package(package_name, Main)
                    # Retry after installing
                    continue
                except Exception:
                    print(f"Failed to auto-install {package_name}, re-raising original error")
                    raise e
            else:
                # Not a package error or max retries reached
                raise
