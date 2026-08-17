"""Resolve a :class:`DataRef` to a labelled array — the one cross-container reference path.

A ``DataRef`` (``schema/common.yaml``) points at one labelled array in another experiment's result, a dataset, or a curated entity: WHERE (``experiment`` id or
``iri``), WHICH (``output``), SLICE (``sel``), and an optional node ``reconcile``.
It is the single primitive behind four authoring surfaces — a figure ``Layer.used``, a sourced ``Argument.used``, a sourced ``Parameter.used``, and (via the same
container + label semantics) the ``initial_state.from_experiment`` seed. This module is their shared resolver, so "find the source container, take the array, slice it,
reconcile it by label" lives in exactly one place.

The resolver is deliberately backend-independent and free of JAX: it returns a plain
``xarray.DataArray`` and takes its network context (label alias map + model node order) by injection, so both the run-time experiment resolvers and the figure codegen
adapter can reuse the same primitives without dragging in each other's dependencies.
Every selection is keyed by label, never positional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

# --------------------------------------------------------------------------- WHERE


def experiment_id(iri) -> Optional[str]:
    """Experiment id of an ``iri`` whose last segment IS an experiment token, else ``None``.

    ``tvbo:exp/<study>/exp-32`` / ``exp-32`` / ``exp32`` / a bare ``32`` -> ``"32"``. A curated / dataset iri whose last segment merely *contains* digits
    (``rec-avgMatrix_atlas-HCPMMP1``) returns ``None`` — so it is not misread as an experiment (which would silently bind to ``exp-1``); ``locate_container`` then treats
    it as a path / curated reference instead.
    """
    if not iri:
        return None
    import re

    key = re.split(r"[:/#]", str(iri))[-1]
    m = re.fullmatch(r"(?:exp-?)?(\d+)", key)
    return m.group(1) if m else None


def _source_id_int(value) -> int:
    """Numeric experiment id from an Experiment, an ``exp-N``/``N`` string, or an int.

    Normalises the spellings the workflow planner accepts (``exp-3`` as written in a recipe, a bare ``3``, an Experiment object with an ``id``) to the integer
    ``locate_exp_container`` globs by, so resolve-time matches plan-time (which reads the id via ``experiment_id``/``str``) instead of raising on ``int('exp-3')``.
    """
    eid = experiment_id(getattr(value, "id", value))
    if eid is None:
        raise ValueError(
            f"cross-experiment sourcing: cannot read an experiment id from {value!r} "
            f"(expected an experiment, an 'exp-N'/'N' string, or an integer)."
        )
    return int(eid)


def locate_exp_container(results_root, source_id) -> Path:
    """Path to experiment ``source_id``'s saved result HDF5 under ``results_root``.

    Globs by the ``exp-<id>_`` file stem, skipping the ``*network*`` sidecar, so the output-directory layout (``results/2``, ``output/nc/exp2``, flat BIDS files, …)
    does not matter. Raises when the source has not been run yet — the actionable
    "run experiment N first" error shared by every consumer.

    Raises when the matches are DIFFERENT RUNS of the same experiment, because no rule here
    can say which one a spec meant. Taking the first sorted hit is the silent-wrong-answer
    version of that: a root holding a dozen retrieved kit archives beside the canonical
    result would bind whichever path sorts first, and a fit score an order of magnitude off
    reads as a finding rather than as a lookup error.

    A per-subject COHORT is not that case. ``ExperimentResult._save_per_subject`` writes one
    ``sub-<id>_exp-<N>_…_result.h5`` shard per subject into a single directory, so the glob
    legitimately matches many files that differ only in their ``sub-`` entity; those collapse
    to one stem and the first shard is returned as before. Two files sharing a name across
    directories, or two distinct stems, are the ambiguity that raises.
    """
    import re

    root = Path(results_root) if results_root else Path.cwd()
    cands = sorted(p for p in root.glob(f"**/*exp-{source_id}_*.h5") if "network" not in p.name)
    if not cands:
        raise FileNotFoundError(
            f"cross-experiment sourcing: no saved result for experiment {source_id} "
            f"under {root} (looked for '*exp-{source_id}_*.h5'). Run experiment "
            f"{source_id} first so its result is available."
        )
    stems = {re.sub(r"^sub-[A-Za-z0-9]+_", "", p.name) for p in cands}
    is_cohort = len(stems) == 1 and len({p.name for p in cands}) == len(cands)
    if len(cands) > 1 and not is_cohort:
        listed = "\n  ".join(str(p) for p in cands[:10])
        more = f"\n  … and {len(cands) - 10} more" if len(cands) > 10 else ""
        raise FileNotFoundError(
            f"cross-experiment sourcing: {len(cands)} saved results for experiment "
            f"{source_id} under {root}, which are different runs of the same experiment:"
            f"\n  {listed}{more}\n"
            f"Point the results root at the ONE canonical container for this study (or move "
            f"the archived runs out of it). Picking one here would bind a figure or a warm "
            f"start to whichever path sorts first."
        )
    return cands[0]


def analysis_container_path(results_root, name) -> Path:
    """``<results_root>/results/<name>/result.h5`` — the analysis-container layout, once."""
    root = Path(results_root) if results_root else Path.cwd()
    return root / "results" / str(name) / "result.h5"


def locate_analysis_container(results_root, name) -> Path:
    """Path to the container a study analysis named ``name`` writes under ``results_root``.

    ``<results_root>/results/<name>/result.h5`` — the one place the convention is spelled, so the writer (:mod:`tvbo.data.analysis_io`), the run-time resolver and the figure
    adapter cannot disagree about where an analysis result lives. Raises when it has not been produced yet, with the command that produces it.
    """
    path = analysis_container_path(results_root, name)
    if not path.is_file():
        raise FileNotFoundError(
            f"analysis sourcing: no saved result for analysis {str(name)!r} (looked for "
            f"{path}). Run the study (`tvbo run <Study>.yaml`) so its analyses execute."
        )
    return path


def is_local_ref(ref) -> bool:
    """A ``DataRef`` with no WHERE (neither ``experiment`` nor ``iri``).

    Such a reference names one of *this* experiment's own outputs (the local end of the reference spectrum, subsuming an ``Argument.value: "observations.x"``); it is
    resolved by the in-run observation machinery, not by opening a sibling container.
    Consumers test this to route a local reference to the right resolver.
    """
    return not any(getattr(ref, w, None) for w in ("experiment", "analysis", "iri"))


def locate_container(ref, *, results_root=None, fallback_experiment=None) -> Path:
    """Resolve a ``DataRef``'s WHERE to a result-container path.

    Precedence ladder (matches the design's one rule): an explicit ``experiment`` id, an ``analysis`` name, or an ``iri`` carrying a trailing experiment number resolves
    against ``results_root``; a filesystem ``iri`` that exists is taken as-is (a curated / external container); a reference with no WHERE falls back to
    ``fallback_experiment`` (the enclosing ``initial_state.source_experiment``, so the warm-start ergonomic of naming the sibling once is preserved). Raises when none
    applies.
    """
    exp = getattr(ref, "experiment", None)
    if exp is not None:
        return locate_exp_container(results_root, _source_id_int(exp))

    ana = getattr(ref, "analysis", None)
    if ana is not None:
        return locate_analysis_container(results_root, ana)

    iri = getattr(ref, "iri", None)
    if iri:
        p = Path(str(iri))
        if p.exists():
            return p
        eid = experiment_id(iri)
        if eid is not None:
            return locate_exp_container(results_root, int(eid))
        raise FileNotFoundError(
            f"cross-experiment sourcing: could not resolve DataRef iri {iri!r} to a "
            "container (not an existing path, and no trailing experiment number)."
        )

    if fallback_experiment is not None:
        return locate_exp_container(results_root, _source_id_int(fallback_experiment))

    raise ValueError(
        "cross-experiment sourcing: DataRef has neither 'experiment' nor 'iri' and no "
        "source_experiment fallback — a local reference must be resolved in-run, not here."
    )


# --------------------------------------------------------------------------- WHICH


def match_output(keys: Iterable[str], output: str, prefer: Iterable[str] = ()) -> str:
    """Container data-var key for ``output`` — exact, or the ``<producer>__<name>`` suffix.

    A recorded state variable matches by name; a declared observation / estimate is stored ``observation__<name>`` / ``estimate__<name>``, so the trailing-``__``
    suffix matches too. An EXACT match always wins: a container holding both a recorded
    ``power`` and an ``observation__power`` would otherwise resolve by dict iteration order, so one spec could bind different arrays in different containers. Shared by
    every consumer (figure layers, warm-start parameters, state seeds) so ``output`` addresses them all identically.

    An AMBIGUOUS bare name raises. A run with several algorithms records one copy of every
    observation per algorithm (``algorithm__fic__mean_H_e`` beside ``algorithm__fic_eib__mean_H_e``),
    so a bare name can have more than one suffix match, and returning the first by iteration order
    is the very failure the exact-match rule above exists to prevent — one spec binding different
    arrays in different containers. For a state seed it silently picks the wrong endpoint and
    nothing fails. *prefer* names producers in priority order and is how a caller RESOLVES such an
    ambiguity rather than merely refusing it: the first matching producer wins. A single candidate
    needs neither, so *prefer* is inert when the name is unambiguous.
    """
    keys = [str(k) for k in keys]
    for k in keys:
        if k == output:
            return k
    candidates = [k for k in keys if k.endswith(f"__{output}")]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        for producer in prefer:
            hit = [k for k in candidates if f"__{producer}__" in k or k.startswith(f"{producer}__")]
            if hit:
                return hit[0]
        raise KeyError(
            f"cross-experiment sourcing: '{output}' is recorded by {len(candidates)} producers "
            f"({sorted(candidates)})"
            + (f" and none of them is one of {list(prefer)}" if prefer else "")
            + ". Qualify the output with its producer so the choice is declared, not guessed."
        )
    raise KeyError(
        f"cross-experiment sourcing: source container does not hold '{output}' "
        f"(looked for an exact match or a '*__{output}' observation/estimate; "
        f"have {sorted(keys)[:12]}{' …' if len(keys) > 12 else ''})."
    )


# --------------------------------------------------------------------------- SLICE


def _is_numeric(value) -> bool:
    vals = value if isinstance(value, (list, tuple)) else [value]
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)


def resolve_dim(da, dim: str) -> str:
    """The axis *dim* names on *da*, seeing through the container's per-variable prefix.

    A saved ``ExperimentResult`` renames an axis to ``<variable>__<axis>`` whenever two of
    its variables carry same-named axes at different sizes (a Dataset cannot hold both). That
    prefix is a storage detail: a spec says ``sel: {node: PFC}`` about the quantity, and must
    keep saying it whether or not a sibling observation happened to force the rename.

    Searches dims AND non-dimension coordinates, because :func:`select_labeled` selects on
    either: a branch-point array is dimmed by ``branch_point`` with ``K`` a 1-D coordinate
    along it, and the container prefixes that coordinate by the same collision rule it
    applies to axes. Resolves only when exactly one name carries the suffix; two would make
    the reference ambiguous, and guessing between them is how a selection silently reads the
    wrong axis. Returns *dim* unchanged when it is already an axis or coordinate, or when
    nothing matches, so the caller's own error still reports the real dims.
    """
    dims = [str(d) for d in getattr(da, "dims", ())]
    coords = [str(c) for c in getattr(da, "coords", {})]
    if dim in dims or dim in coords:
        return dim
    suffix = f"__{dim}"
    hits = list(dict.fromkeys(d for d in dims + coords if d.endswith(suffix)))
    if len(hits) == 1:
        return str(hits[0])
    if hits:
        raise KeyError(
            f"selection key {dim!r} matches more than one axis of the sourced array "
            f"({sorted(str(h) for h in hits)}); name the one you mean."
        )
    return dim


def resolve_sel_keys(da, sel: Optional[Mapping[str, object]]) -> dict:
    """*sel* with every key resolved to the axis it names on *da* (see :func:`resolve_dim`).

    For a caller that applies ``.sel`` itself and only needs the keys corrected — the emitted
    figure script, which passes its own ``method``. Values pass through untouched.
    """
    return {resolve_dim(da, str(k)): v for k, v in (sel or {}).items()}


def select_labeled(da, sel: Optional[Mapping[str, object]]):
    """Apply a label-keyed ``.sel`` to ``da``, never positional.

    Each entry selects along a dimension by coordinate label (``method="nearest"`` for a numeric selection, e.g. the sampled K on a continuous sweep; exact for a label).
    A selection key that is a *non-dimension* coordinate defined along a single dimension is honoured too (the operating-point-out-of-a-branch case: the container
    is dimmed by ``branch_point`` with ``K`` a coordinate along it) — the nearest index along that coordinate is taken. Returns ``da`` unchanged for an empty sel.
    """
    if not sel:
        return da
    import numpy as np

    for dim, value in sel.items():
        dim = resolve_dim(da, dim)
        if dim in da.dims:
            da = da.sel({dim: value}, method="nearest" if _is_numeric(value) else None)
        elif dim in da.coords:
            coord = da.coords[dim]
            if coord.ndim != 1:
                raise ValueError(
                    f"cross-experiment sourcing: cannot select on coordinate '{dim}' "
                    f"with {coord.ndim} dimensions; sel supports 1-D coordinates only."
                )
            axis = coord.dims[0]
            if _is_numeric(value):
                # Nearest match along the coordinate — for a scalar (drops the axis) AND for a list (keeps the axis, one nearest index per target), consistent with the dimension path's method="nearest". Exact np.isin would silently miss on a continuous sweep whose sampled values never match exactly.
                cvals = np.asarray(coord.values, dtype=float)
                if isinstance(value, (list, tuple)):
                    da = da.isel({axis: [int(np.abs(cvals - float(t)).argmin()) for t in value]})
                else:
                    da = da.isel({axis: int(np.abs(cvals - float(value)).argmin())})
            else:
                wanted = value if isinstance(value, (list, tuple)) else [value]
                mask = np.isin(np.asarray(coord.values), np.asarray(wanted))
                da = da.isel({axis: np.flatnonzero(mask)})
        else:
            raise KeyError(
                f"cross-experiment sourcing: selection key '{dim}' is neither a "
                f"dimension nor a coordinate of the sourced array (dims {tuple(da.dims)}, "
                f"coords {tuple(da.coords)})."
            )
    return da


# --------------------------------------------------------------------------- RECONCILE


def reconcile_by_label(
    da, alias_map: Mapping[str, str], model_labels: Sequence[str], node_dims: Optional[Sequence[str]] = None
):
    """Align every labelled node axis of ``da`` to the model's node order, by label.

    A node axis is any dimension carrying string coordinates. Each is relabelled source -> canonical through ``alias_map`` (alias-aware, so a divergent nomenclature
    or a hemisphere-swapped convention still matches) then restricted to
    ``model_labels`` in the model's order — on *both* axes of a per-edge matrix.
    Unlabelled axes are left untouched (assumed already in model order). Pass
    ``node_dims`` to restrict reconciliation to a known set of node axes (so a labelled *non-node* dimension is not mistaken for one); the default reconciles
    every string-coordinate axis. This is the ``reconcile: by_label`` path; ``none`` skips it.
    """
    import numpy as np

    model_set = set(str(m) for m in model_labels)
    for d in list(da.dims):
        if node_dims is not None and d not in node_dims:
            continue
        if d not in da.coords:
            continue
        vals = np.asarray(da.coords[d].values)
        if vals.dtype.kind not in ("U", "S", "O"):
            continue
        mapped = [alias_map.get(str(v), str(v)) for v in vals]
        # Without an explicit node_dims restriction, only reconcile an axis that is actually a node axis — its labels overlap the model's node labels. A labelled
        # NON-node axis (e.g. a 'population'/'variable' coord) is left untouched rather than forced through `.sel(model_labels)` (which would raise).
        if node_dims is None and not (model_set & set(mapped)):
            continue
        da = da.assign_coords({d: mapped})
        da = da.sel({d: list(model_labels)})
    return da


# --------------------------------------------------------------------------- sel extraction


def sel_dict(ref) -> dict:
    """``DataRef.sel`` (a collection of ``Argument``) as a plain ``{dim: value}`` mapping.

    The Argument name is the dimension/coordinate; its ``value`` the coordinate label.
    Keyed by name, never positional. Empty when the reference carries no ``sel``.

    Both spellings resolve: the keyed dict a study writes (``sel: {variable: phi}``) and the list of Arguments a dataclass build produces.
    """
    sel = getattr(ref, "sel", None) or []
    items = sel.items() if hasattr(sel, "items") else [(getattr(a, "name", None), a) for a in sel]
    return {str(name): getattr(arg, "value", arg) for name, arg in items if name is not None}


def reconcile_mode(ref) -> str:
    """The reference's node-reconcile mode as a plain string (``'by_label'`` / ``'none'``)."""
    mode = getattr(ref, "reconcile", None)
    return str(mode) if mode is not None else "none"


# --------------------------------------------------------------------------- full pipeline


def resolve_dataref(
    ref,
    *,
    results_root=None,
    fallback_experiment=None,
    alias_map: Optional[Mapping[str, str]] = None,
    model_labels: Optional[Sequence[str]] = None,
):
    """Resolve a container-backed ``DataRef`` to a labelled :class:`xarray.DataArray`.

    Runs the four steps in order — WHERE (:func:`locate_container`), WHICH (:func:`match_output`), SLICE (:func:`select_labeled`), RECONCILE
    (:func:`reconcile_by_label`, only when the reference asks for ``by_label`` and a network context is supplied) — and returns the array detached from the source
    file. ``alias_map`` / ``model_labels`` are the consuming network's
    ``region_alias_map()`` and node order, injected so this stays network-agnostic.

    For a *local* reference (no WHERE) raises via :func:`locate_container`; callers test :func:`is_local_ref` first and route those to the in-run resolver.
    """
    import xarray as xr

    path = locate_container(ref, results_root=results_root, fallback_experiment=fallback_experiment)
    ds = xr.open_dataset(path, engine="h5netcdf")
    try:
        da = ds[match_output(ds.data_vars, getattr(ref, "output", None))]
        da = select_labeled(da, sel_dict(ref))
        da = da.load()
    finally:
        ds.close()

    da = apply_transform(da, getattr(ref, "transform", None))
    if reconcile_mode(ref) == "by_label" and alias_map is not None and model_labels is not None:
        da = reconcile_by_label(da, alias_map, model_labels)
    return da


def apply_transform(da, name: Optional[str]):
    """Apply a named ``fn(da) -> da`` reduction, or return ``da`` unchanged for no transform.

    Resolves the name against the shared ``bsplot`` transform registry — the same one a figure ``Layer.transform`` and a study's ``code_modules`` use — so a sourced array and
    a figure layer name the same transforms. Imported lazily to keep this module free of an adapter dependency; a miss raises the registry's actionable "not registered" error.
    """
    if not name:
        return da
    from tvbo.adapters.bsplot import TRANSFORMS, registered

    return registered(TRANSFORMS, name, "transform")(da)
