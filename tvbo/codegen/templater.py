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
from typing import Any

import black

from tvbo import templates

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


def format_code(code: str, format: str = "python", use_black: bool = True, **kwargs: Any) -> str:
    """Format code using black for Python variants.

    Args:
        code: Source code string to format
        format: Language/variant (python, jax, numpy, scipy, tvboptim)
        use_black: Whether to apply black formatting (default True)
        **kwargs: Additional black.FileMode options (line_length, etc.)
    """
    if format in ["tvb", "python", "autodiff", "jax", "numpy", "scipy", "tvboptim"]:
        if use_black:
            code = black.format_str(code, mode=black.FileMode(**kwargs))
    return code


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


