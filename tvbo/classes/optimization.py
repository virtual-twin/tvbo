"""What an optimization stage fits and what its loss calls, resolved once for every reader.

A stage names the parameters it fits in ``free_parameters``. Each entry fits its parameter at the parameter's own layout, its declared ``shape`` or, with ``heterogeneous: true``, one value per node, and writes ``shape`` or ``heterogeneous`` only to change that layout; ``heterogeneous: false`` or ``shape: "()"`` narrows it to one scalar. An entry that writes the layout its parameter already declares is a restatement and is refused, so the layout has one home. :func:`fitted_parameters` resolves every entry of a stage to the parameter, its owner and the layout it is fitted at; the tvboptim stage marking reads it. :func:`loss_callee` names what an optimization's loss calls and the module it is imported from, which the generated loss and the check refusing a loss nothing defines both read.
"""

from __future__ import annotations

from typing import Any

from tvbo.codegen.dims import N_NODES, shape_terms, terms_code
from tvbo.utils import network_couplings


def optimization_stages(optimization: Any) -> list:
    """The stages of *optimization*; one written without stages is its own single stage."""
    return list(optimization.stages or []) or ([optimization] if optimization.free_parameters else [])


def loss_callee(loss: Any) -> tuple[str, str | None]:
    """The name *loss* calls and the module that name is imported from.

    A ``function:`` reference names a Function the experiment declares, so it imports nothing and its module is ``None``. A ``callable:`` is called by its ``name`` (else ``qualname``) and imported from its ``module``, ``None`` when it states none. A loss stating neither calls ``loss``.
    """
    function = getattr(loss, "function", None)
    if function:
        return (function if isinstance(function, str) else getattr(function, "name", None) or str(function)), None
    call = getattr(loss, "callable", None)
    if call:
        return getattr(call, "name", None) or getattr(call, "qualname", None) or "loss", getattr(call, "module", None) or None
    return "loss", None


def layout(parameter: Any) -> tuple:
    """The layout *parameter* declares, as shape terms: its ``shape``, else ``(n_nodes,)`` when it is heterogeneous, else a scalar."""
    terms = shape_terms(getattr(parameter, "shape", None))
    if terms is not None:
        return terms
    return (N_NODES,) if getattr(parameter, "heterogeneous", False) else ()


def _written_layout(entry: dict) -> tuple | None:
    """The layout a parsed stage entry writes: its ``shape``, else ``(n_nodes,)`` for ``heterogeneous: true`` and a scalar for ``heterogeneous: false``, or None when it writes neither and inherits its parameter's."""
    if entry.get("shape") is not None:
        return shape_terms(entry["shape"])
    if entry.get("heterogeneous") is None:
        return None
    return (N_NODES,) if entry["heterogeneous"] else ()


def _declared(entry: dict, model: Any, couplings: dict) -> Any:
    """The Parameter a parsed stage entry names on its owner, or None for one tvbo does not declare (an event's, the noise's)."""
    owner = couplings.get(entry["coupling_key"]) if entry.get("coupling_key") else model
    return (getattr(owner, "parameters", None) or {}).get(entry["name"])


def fitted_parameters(stage: Any, experiment: Any, couplings: dict | None = None) -> list[dict]:
    """Every parameter *stage* fits, each as the parsed entry with its layout after inheritance.

    Each entry is parsed by :func:`tvbo.templates.tvboptim.utils.parse_free_param` against the experiment's dynamics and *couplings* (by default the network's), then given the layout it is fitted at: the one it writes, or else its parameter's own. ``heterogeneous`` and ``shape`` in the result state that layout. A written layout equal to the parameter's own is refused, naming the stage and the parameter.
    """
    from tvbo.templates.tvboptim.utils import parse_free_param

    couplings = network_couplings(experiment.network) if couplings is None else couplings
    model = experiment.dynamics
    fitted = []
    for raw in stage.free_parameters or []:
        entry = parse_free_param(raw, set(couplings), model, couplings)
        if entry is None:
            continue
        parameter = _declared(entry, model, couplings)
        written = _written_layout(entry)
        if parameter is not None:
            own = layout(parameter)
            if written == own:
                raise ValueError(
                    f"optimization stage {getattr(stage, 'name', None)!r}: the entry for {entry['name']!r} writes the layout {terms_code(own) if own else 'of a scalar'} that {entry['name']!r} declares; "
                    f"an entry fits its parameter at the parameter's own layout, so write `{{parameter: ...}}` without `shape` or `heterogeneous`, or write them only to change the layout"
                )
            if written is None:
                written = own
                entry["shape"] = str(parameter.shape) if shape_terms(getattr(parameter, "shape", None)) else None
        entry["heterogeneous"] = bool(written)
        if not written:
            entry["shape"] = None
        fitted.append(entry)
    return fitted
