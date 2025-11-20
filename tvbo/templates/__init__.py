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
import subprocess
from mako.lookup import TemplateLookup
from os.path import join, abspath, dirname

root = abspath(dirname(__file__))
lookup = TemplateLookup(
    directories=[
        root,
        join(root, "autodiff"),
        join(root, "tvb"),
        join(root, "julia"),
        join(root, "generic_python"),
        join(root, "report"),
        join(root, "numcont"),
        join(root, "pde"),
    ],
    module_directory=join(root, "modules"),
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
