from importlib import import_module
import re
import sys

_julia_instance = None
_julia_main = None
_installed_packages = set()

# tvbo/adapters/julia.py

def install():
    import julia
    julia.install()

def get_julia(compiled_modules=True):
    """Initialize Julia runtime. Use compiled_modules=True to use precompiled packages."""
    global _julia_instance, _julia_main
    if _julia_instance is None or _julia_main is None:
        try:
            jl = import_module("julia")
        except ImportError:
            raise ImportError("PyJulia (julia) package not installed. Run: pip install julia")

        import os
        os.environ['JULIA_NUM_THREADS'] = '1'

        # Suppress Julia's MainInclude warnings (written to fd 2 by Julia C runtime)
        stderr_fd = sys.stderr.fileno()
        saved_fd = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)

        try:
            _init_julia_instance(jl, compiled_modules)
            _julia_main = import_module("julia.Main")
        finally:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)

    return _julia_instance, _julia_main


def _init_julia_instance(jl, compiled_modules):
    """Try multiple strategies to initialize the Julia runtime."""
    global _julia_instance
    try:
        _julia_instance = jl.Julia(
            compiled_modules=compiled_modules,
            debug=False,
            init_julia=True,
            runtime='/opt/homebrew/Cellar/julia/1.12.2/bin/julia'
        )
    except jl.UnsupportedPythonError:
        try:
            jl.install()
        except Exception:
            pass
        _julia_instance = jl.Julia(
            compiled_modules=compiled_modules,
            debug=False,
            init_julia=True,
        )
    except FileNotFoundError:
        _julia_instance = jl.Julia(
            compiled_modules=compiled_modules,
            debug=False,
            init_julia=True,
        )
    except Exception:
        _julia_instance = jl.Julia(compiled_modules=False, debug=False)

def install_julia_package(package_name: str, Main=None, update: bool = False):
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
            Main.eval(f'import Pkg; Pkg.update("{package_name}")')
        else:
            Main.eval(f'import Pkg; Pkg.add("{package_name}")')
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
            return Main.eval(code)
        except Exception as e:
            error_msg = str(e)
            # Check if it's a missing package error
            match = re.search(r'Package (\w+) not found', error_msg)
            if match and attempt < max_retries - 1:
                package_name = match.group(1)
                try:
                    install_julia_package(package_name, Main)
                    # Retry after installing
                    continue
                except Exception as install_error:
                    print(f"Failed to auto-install {package_name}, re-raising original error")
                    raise e
            else:
                # Not a package error or max retries reached
                raise
