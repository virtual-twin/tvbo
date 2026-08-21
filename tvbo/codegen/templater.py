#
# Module: templater.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Generate TVBO Python classes from ontology definitions via Mako templates.

Provides helpers that read model, parameter, state-variable, equation,
coupling and integrator metadata from the ontology and render it into
executable TVBO/TVB source using the templates in `tvbo.templates`.
"""
import logging
from dataclasses import dataclass
from typing import Any

import sympy as sp

from tvbo import templates
from tvbo.templates.base.utils import get_func_name

logger = logging.getLogger(__name__)


def _log_source(rendered_code: str) -> None:
    """Emit rendered code with line numbers when ``print_source`` is requested."""
    if not logger.isEnabledFor(logging.INFO):
        return
    numbered = "\n".join(
        f"{i}\t{line}" for i, line in enumerate(rendered_code.split("\n"), start=1)
    )
    logger.info("rendered source:\n%s", numbered)

exec_globals = {}
TEMPLATES = templates.root


def is_derived(obs: Any, experiment: Any) -> bool:
    """Return True if ``obs`` derives from other observations in ``experiment``.

    An Observation is derived when any item in its multivalued ``source``
    slot names ANOTHER observation in the same experiment. Source entries
    may be bare strings, objects with a ``name`` attribute, or inlined
    Observation/StateVariable instances.

    A SELF-reference (an observation whose ``source`` names itself — e.g. an
    observation ``r_A`` with ``source: [r_A]`` that simply observes the model
    variable ``r_A``) is NOT derived: an observation cannot derive from itself.
    Without this exclusion such observations are mis-routed to the derived path,
    where they have no pipeline and are never computed, so the generated
    ``observations.r_A = _all_obs.r_A`` extraction raises AttributeError.
    """
    obs_names = set((getattr(experiment, "observations", {}) or {}).keys())
    if not obs_names:
        return False
    self_name = getattr(obs, "name", None)
    for s in (getattr(obs, "source", None) or []):
        name = getattr(s, "name", None) or s
        if isinstance(name, str) and name in obs_names and name != self_name:
            return True
    return False


def source_observations(obs: Any, experiment: Any) -> list:
    """Return the source names of ``obs`` that resolve to other observations.

    A filtered view of ``obs.source`` keeping only entries whose name
    matches a key in ``experiment.observations``.
    """
    obs_names = set((getattr(experiment, "observations", {}) or {}).keys())
    if not obs_names:
        return []
    self_name = getattr(obs, "name", None)
    out = []
    for s in (getattr(obs, "source", None) or []):
        name = getattr(s, "name", None) or s
        if isinstance(name, str) and name in obs_names and name != self_name:
            out.append(name)
    return out


@dataclass(frozen=True)
class CodeFormat:
    """How one component-level code format is rendered, formatted and re-entered.

    A component format renders a single `Dynamics` rather than a whole experiment, so
    it names no :class:`~tvbo.export.registry.ExportFormat` and declares itself here
    instead. `template` is the Mako template that emits it, or empty for the formats an
    adapter builds. `entry` is the module-level name the emitted code binds its callable
    to; empty means the template names it after the model, falling back to `unnamed_entry`
    for a model with no name of its own. `language` selects the normaliser in
    :func:`format_code`, is empty for output returned verbatim, and is what says whether
    the output can be executed at all.
    """

    template: str = ""
    language: str = ""
    entry: str = ""
    unnamed_entry: str = "dfun"
    render_kwargs: tuple = ()


_PYTHON_MODEL = "tvbo-python-model.py.mako"
_JAX_DFUNS = "tvbo-jax-dfuns.py.mako"
_AUTO7P = "tvbo-auto7p.py.mako"
_PDE_FEM = "tvbo-pde-fem.py.mako"

CODE_FORMATS: dict[str, CodeFormat] = {
    "tvb": CodeFormat("tvbo-tvb-model.py.mako", "python", unnamed_entry="GeneratedDynamics"),
    "tvboptim": CodeFormat(
        "tvbo-tvboptim-dynamics.py.mako", "python", unnamed_entry="GeneratedDynamics"
    ),
    "scipy": CodeFormat(_PYTHON_MODEL, "python"),
    "python": CodeFormat(_PYTHON_MODEL, "python"),
    "jax-python": CodeFormat(_PYTHON_MODEL, "python"),
    "python-jax": CodeFormat(_PYTHON_MODEL, "python"),
    "python-network": CodeFormat(
        _PYTHON_MODEL, "python", render_kwargs=(("coupling_as_argument", True),)
    ),
    "jax": CodeFormat(_JAX_DFUNS, "python", entry="dfun"),
    "numpy": CodeFormat(_JAX_DFUNS, "python", entry="dfun"),
    "autodiff": CodeFormat(_JAX_DFUNS, "python", entry="dfun"),
    "julia": CodeFormat("tvbo-julia-DifferentialEquations.jl.mako", "julia"),
    "bifurcation-numcont": CodeFormat(_AUTO7P),
    "bifurcation-auto7p": CodeFormat(_AUTO7P),
    "pde-fem": CodeFormat(_PDE_FEM, "python"),
    "pde-python": CodeFormat(_PDE_FEM, "python"),
    "pde": CodeFormat(_PDE_FEM, "python"),
    "pyrates": CodeFormat(language="yaml"),
}
"""Every component-level code format: template, output language and entry point."""


def code_format(format: str) -> CodeFormat | None:
    """The :class:`CodeFormat` declared for *format*, or ``None`` if it declares none."""
    return CODE_FORMATS.get(str(format).lower())


def source_language(format: str) -> str:
    """Return the output language of *format*, or ``""`` when it emits none.

    A backend declares its language once: component formats on their
    :data:`CODE_FORMATS` entry, experiment-level export backends on their
    :class:`~tvbo.export.registry.ExportFormat`.
    """
    spec = code_format(format)
    if spec is not None:
        return spec.language
    from tvbo.export import registry

    try:
        return registry.resolve(format).language
    except ValueError:
        return ""


def entry_point_name(model, format: str) -> str:
    """The name generated code for *format* binds its callable to.

    The template and [`Dynamics.execute`](#tvbo.behaviour.dynamics_runtime.DynamicsRuntime.execute)
    read this one declaration — including the fallback for a model with no name of its
    own — so handing a rendered dfun to a custom JAX or NumPy workflow never depends on
    guessing which name the template chose.

    Raises:
        ValueError: If *format* is not declared, or emits something other than Python.
            A Julia module or a YAML document has no Python callable to bind, and
            `exec`-ing one raises `SyntaxError` from inside the generated text rather
            than naming the format that could never have worked.
    """
    spec = code_format(format)
    if spec is None:
        raise ValueError(
            f"Format '{format}' declares no entry point in CODE_FORMATS, "
            "so its output cannot be executed."
        )
    if spec.language != "python":
        raise ValueError(
            f"Format '{format}' emits {spec.language or 'unformatted'} output, not Python, "
            "so there is no callable to execute. Render it and hand it to that toolchain."
        )
    return spec.entry or get_func_name(model, fallback=spec.unnamed_entry)


def format_code(code: str, format: str = "python", use_black: bool = True) -> str:
    """Format generated *code* for the backend named by *format*.

    Component-level renders (a Dynamics, a Coupling, an Observation) come through
    here; whole-experiment renders are formatted by
    :func:`tvbo.export.registry.render`. Both resolve the language the same way and
    both route to :mod:`tvbo.codegen.style`, so they cannot drift apart.

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

    A backend whose derivative signature carries no time — TVB's ``Model.dfun`` —
    cannot express these, and emitting the term anyway yields an unbound name. The
    equations are the ground truth rather than the ``autonomous`` slot, which is
    author-declared and can disagree with them.
    """
    t = sp.Symbol("t")
    # Only integrated and derived quantities: a `functions:` entry taking an argument
    # named `t` binds it as a parameter, so its rhs naming `t` is not time dependence.
    scoped = set(model.state_variables) | set(model.derived_variables) | set(model.derived_parameters)
    # `t` denotes integrator time ONLY when the model does not itself declare a symbol named
    # `t`. A model with a parameter/state literally called `t` (a time constant, threshold, …)
    # means that symbol, not the reserved time — flagging it would block a valid, autonomous export.
    if "t" in scoped | set(model.parameters):
        return []
    return sorted(
        name for name, eq in (model.get_equations() or {}).items()
        if name in scoped and t in eq.rhs.free_symbols
    )


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


