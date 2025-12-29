#
# Module: __init__.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""
TVB-O Export
============

Provides export functionality for TVBO models to various formats:

- TVB (The Virtual Brain)
- JAX (autodiff)
- Julia (DifferentialEquations.jl, BifurcationKit.jl)
- PyRates (YAML templates)
- LEMS (NeuroML)
- PDE-FEM (scikit-fem)

"""

from tvbo.export.pyrates import (
    # Complete experiment (model + network, ready to run)
    to_pyrates_yaml_string,
    # Modular exports
    to_pyrates_model_yaml,      # OperatorTemplate only (dynamics)
    to_pyrates_network_yaml,    # NodeTemplate + CircuitTemplate (topology)
    # Import
    from_pyrates_yaml,
    # Alias
    network_to_pyrates_yaml_string,
)

from tvbo.export.functions import (
    # Standalone function code generation
    generate_function,
    generate_loss_function,
    generate_indexed_function,
    generate_callable_function,
    generate_inline_function,
    function_to_callable,
    # Convenience alias
    generate,
)
