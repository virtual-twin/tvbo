# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Shared codegen helpers the Mako templates and the export registry read.

Answers the questions a backend asks about a model before emitting it — which observations are derived, which equations read the integrator's time symbol, which base parameters a derived-parameter block consumes — plus the one namespace generated modules are `exec`-ed into and the formatting entry point every component-level render routes through.
"""

import logging
from typing import Any

import sympy as sp

logger = logging.getLogger(__name__)

exec_globals = {}
"""Globals a generated module is `exec`-ed into, shared so one render's imports serve the next."""


def is_derived(obs: Any, experiment: Any) -> bool:
    """Return True if ``obs`` derives from other observations in ``experiment``.

    An Observation is derived when any item in its multivalued ``source`` slot names ANOTHER observation in the same experiment. Source entries may be bare strings, objects with a ``name`` attribute, or inlined Observation/StateVariable instances.

    A SELF-reference (an observation whose ``source`` names itself — e.g. an observation ``r_A`` with ``source: [r_A]`` that simply observes the model variable ``r_A``) is NOT derived: an observation cannot derive from itself.
    Without this exclusion such observations are mis-routed to the derived path, where they have no pipeline and are never computed, so the generated ``observations.r_A = _all_obs.r_A`` extraction raises AttributeError.
    """
    obs_names = set((getattr(experiment, "observations", {}) or {}).keys())
    if not obs_names:
        return False
    self_name = getattr(obs, "name", None)
    for s in getattr(obs, "source", None) or []:
        name = getattr(s, "name", None) or s
        if isinstance(name, str) and name in obs_names and name != self_name:
            return True
    return False


def source_observations(obs: Any, experiment: Any) -> list:
    """Return the source names of ``obs`` that resolve to other observations.

    A filtered view of ``obs.source`` keeping only entries whose name matches a key in ``experiment.observations``.
    """
    obs_names = set((getattr(experiment, "observations", {}) or {}).keys())
    if not obs_names:
        return []
    self_name = getattr(obs, "name", None)
    out = []
    for s in getattr(obs, "source", None) or []:
        name = getattr(s, "name", None) or s
        if isinstance(name, str) and name in obs_names and name != self_name:
            out.append(name)
    return out


COMPONENT_LANGUAGES = {
    "python": "python",
    "numpy": "python",
    "scipy": "python",
    "autodiff": "python",
    "pyrates": "yaml",
}
"""Component-level ``format`` aliases, which name no export format, to their language."""


def source_language(format: str) -> str:
    """Return the output language of *format*, or ``""`` when it emits none.

    Resolves through the export registry so a backend declares its language once, on its :class:`~tvbo.export.registry.ExportFormat`. The component-level aliases in :data:`COMPONENT_LANGUAGES` are not registered formats and are mapped here.
    """
    if format in COMPONENT_LANGUAGES:
        return COMPONENT_LANGUAGES[format]
    from tvbo.export import registry

    try:
        return registry.resolve(format).language
    except ValueError:
        return ""


def format_code(code: str, format: str = "python", use_black: bool = True) -> str:
    """Format generated *code* for the backend named by *format*.

    Component-level renders (a Dynamics, a Coupling, an Observation) come through here; whole-experiment renders are formatted by :func:`tvbo.export.registry.render`. Both resolve the language the same way and both route to :mod:`tvbo.codegen.style`, so they cannot drift apart.

    Args:
        code: Source code string to format
        format: Backend key or component-level alias (python, jax, numpy, tvboptim…)
        use_black: Set False to return *code* untouched

    Raises:
        tvbo.codegen.style.GeneratedSourceError: *code* does not parse as its language.
    """
    from tvbo.codegen.style import format_source

    return format_source(code, source_language(format)) if use_black else code


def time_dependent_equations(model) -> list[str]:
    """Names whose equation reads the time symbol ``t``, sorted.

    A backend whose derivative signature carries no time — TVB's ``Model.dfun`` — cannot express these, and emitting the term anyway yields an unbound name. The equations are the ground truth rather than the ``autonomous`` slot, which is author-declared and can disagree with them.

    A model that declares a symbol of its own named ``t`` — a time constant, a threshold — reads no time at all: there the name means that symbol, and flagging it would block a valid autonomous export.
    """
    t = sp.Symbol("t")
    # Only integrated and derived quantities: a `functions:` entry taking an argument named `t` binds it as a parameter, so its rhs naming `t` is not time dependence.
    scoped = set(model.state_variables) | set(model.derived_variables) | set(model.derived_parameters)
    # `t` is integrator time only when the model declares no symbol of its own by that name.
    if "t" in scoped | set(model.parameters):
        return []
    return sorted(name for name, eq in (model.get_equations() or {}).items() if name in scoped and t in eq.rhs.free_symbols)


def derived_parameter_inputs(model) -> list[str]:
    """Base parameter names the derived-parameter expressions read, in model order.

    A backend that computes derived parameters must first unpack the base parameters they depend on — ``ReducedSetHindmarshRose`` derives twelve of them from ``a``, ``b``, ``sigma`` and friends, so dropping the unpack breaks the model. Unpacking
    *every* parameter instead leaves the unread ones as dead bindings, so this returns
    exactly the ones consumed.

    Returns an empty list when the model derives no parameters, which is the case where the whole unpack is dead.
    """
    derived = getattr(model, "derived_parameters", None) or {}
    if not derived:
        return []
    equations = model.get_equations() or {}
    consumed = set()
    for name in derived:
        eq = equations.get(name)
        if eq is not None:
            consumed |= {str(sym) for sym in eq.rhs.free_symbols}
    return [name for name in model.parameters if name in consumed]


### Integrator ###
def get_integrator_info(integrator):
    """Collect scheme metadata for an integrator into a dict.

    Args:
        integrator: Ontology integrator class describing the scheme.

    Returns:
        A dict with the integrator `class_name`, the number of derivative stages
        `n_dx`, its `intermediate_steps`, and the `dX_expr` update expression
        (`None` when unset).
    """
    n_dx = len(integrator.intermediate_steps) + 1
    intermediade_steps = integrator.intermediate_steps if n_dx > 1 else []
    dX = integrator.dX.first() if integrator.dX else None

    info = {
        "class_name": integrator.name,
        "n_dx": n_dx,
        "intermediate_steps": intermediade_steps,
        "dX_expr": dX,
    }
    return info
