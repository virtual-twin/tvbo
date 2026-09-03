# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Function Classes.

Extended Function and LossFunction classes with code generation methods.
Inherits from the LinkML datamodel and adds rendering/execution capabilities.

Usage
-----

From YAML file:

    from tvbo import Function, LossFunction

    func = Function.from_file("correlation.yaml")
    code = func.render_code(format='jax')
    callable_fn = func.to_callable()

From YAML string:

    func = Function.from_string(yaml_string)

From datamodel object:

    from tvbo.datamodel import schema as tvbo_datamodel
    dm_func = tvbo_datamodel.Function(name='sigmoid', ...)
    func = Function.from_datamodel(dm_func)
"""

from tvbo.datamodel import schema as tvbo_datamodel

Function = tvbo_datamodel.Function
LossFunction = tvbo_datamodel.LossFunction
"""The generated classes themselves. What a function does — resolve to a callable, render as code,
answer for its own symbols — lives in :mod:`tvbo.behaviour.function`, attached where the classes are
generated. One declared function now has one class, however it was built and whichever module the
caller imported."""
