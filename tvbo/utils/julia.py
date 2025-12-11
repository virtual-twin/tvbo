from importlib import import_module
import re

_julia_instance = None
_julia_main = None
_installed_packages = set()

# tvbo/utils/julia.py

def install():
    import julia
    julia.install()

def get_julia(compiled_modules=False):
    """Initialize Julia runtime. Use compiled_modules=False by default for stability."""
    global _julia_instance, _julia_main
    if _julia_instance is None or _julia_main is None:
        try:
            jl = import_module("julia")
        except ImportError:
            raise ImportError("PyJulia (julia) package not installed. Run: pip install julia")

        # Silence Julia warnings during startup
        import os
        os.environ['JULIA_NUM_THREADS'] = '1'

        try:
            # Use minimal Julia configuration for stability
            _julia_instance = jl.Julia(
                compiled_modules=compiled_modules,
                debug=False,
                init_julia=True,
                runtime='/opt/homebrew/Cellar/julia/1.12.2/bin/julia'  # explicit path for macOS homebrew
            )
        except jl.UnsupportedPythonError:
            # PyJulia is not configured for this Python environment, configure it now
            print("Configuring PyJulia for the current Python environment...")
            try:
                jl.install()
            except Exception as install_err:
                print(f"Julia install failed: {install_err}")
                print("Trying alternative initialization...")
            # Retry after installation
            _julia_instance = jl.Julia(
                compiled_modules=compiled_modules,
                debug=False,
                init_julia=True
            )
        except FileNotFoundError:
            # Julia binary not found, try without explicit path
            _julia_instance = jl.Julia(
                compiled_modules=compiled_modules,
                debug=False,
                init_julia=True
            )
        except Exception as e:
            print(f"Error initializing Julia: {e}")
            print("Retrying with minimal configuration...")
            try:
                _julia_instance = jl.Julia(compiled_modules=False, debug=False)
            except Exception as retry_err:
                raise RuntimeError(
                    f"Failed to initialize Julia: {retry_err}. "
                    "Try restarting the kernel or reinstalling: pip install julia && python -c 'import julia; julia.install()'"
                )

        try:
            _julia_main = import_module("julia.Main")
        except Exception as e:
            raise RuntimeError(f"Failed to import julia.Main: {e}")

    return _julia_instance, _julia_main

def install_julia_package(package_name, Main=None, update=False):
    """Install a Julia package if not already installed.

    Args:
        package_name: Name of the Julia package to install
        Main: Julia Main module (will be retrieved if None)
        update: If True, update the package to the latest version
    """
    global _installed_packages

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
