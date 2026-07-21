#
# Module: __init__.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""
Templates
=========

Templates for generating code.
"""

import os
import subprocess
import tempfile
from mako.lookup import TemplateLookup
from os.path import join, abspath, dirname

root = abspath(dirname(__file__))


def _mako_module_dir() -> str:
    """A writable directory for mako's compiled-template cache.

    Mako compiles each template to a ``.py`` module and caches it here. The obvious
    home — a ``modules/`` dir beside the templates — is read-only whenever tvbo is
    installed read-only, most sharply inside a container: the first render at runtime
    then dies with ``OSError: [Errno 30] Read-only file system`` (and only at runtime,
    since the package dir is writable during local dev). Default to a per-user dir
    under the system temp instead, which is writable and, under SLURM, node-local so
    concurrent jobs on different nodes don't share it. ``TVBO_MAKO_CACHE`` overrides.
    """
    override = os.environ.get("TVBO_MAKO_CACHE")
    if override:
        return override
    return join(tempfile.gettempdir(), f"tvbo-mako-{os.getuid()}")


lookup = TemplateLookup(
    directories=[
        root,
        join(root, "base"),
        join(root, "autodiff"),
        join(root, "tvb"),
        join(root, "julia"),
        join(root, "generic_python"),
        join(root, "report"),
        join(root, "numcont"),
        join(root, "pde"),
        join(root, "pyrates"),
        join(root, "tvboptim"),
        join(root, "networkdynamics"),
        join(root, "modelingtoolkit"),
        join(root, "neuroml"),
        join(root, "brian2"),
        join(root, "bsplot"),
        join(root, "workflow", "snakemake"),
    ],
    module_directory=_mako_module_dir(),
)


def run_julia(julia_code, verbose=0):
    """
    Run Julia code as a string using subprocess with manual logging.

    Parameters:
    julia_code (str): The Julia code to execute.
    verbose (int): Verbosity level (0 for silent, 1 for basic output, 2 for detailed output).

    Returns:
    dict: A dictionary with 'stdout', 'stderr', and 'returncode' from the subprocess.
    """
    command = ["julia", "-e", julia_code]

    # Print basic info if verbose
    if verbose > 0:
        print("Starting Julia execution...")
    try:
        if verbose == 0:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
            )
        elif verbose == 1:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            print("Julia execution completed.")
        else:  # verbose == 2
            result = subprocess.run(command, text=True, check=True)
            print("Julia execution completed with detailed output.")

        # Return output, errors, and return code
        return {
            "stdout": result.stdout if verbose > 0 else None,
            "stderr": result.stderr if verbose > 0 else None,
            "command": command if verbose > 0 else None,
            "returncode": result.returncode,
        }

    except subprocess.CalledProcessError as e:
        # Print error messages
        print(f"Error: Julia execution failed with return code {e.returncode}.")
        if verbose > 0:
            print(f"Error Output: {e.stderr}")
        return {"stdout": e.stdout, "stderr": e.stderr, "returncode": e.returncode}
