"""Figure -> distributed-workflow emitter.

Wires TVBO's declarative :class:`~tvbo.datamodel.pydantic.Figure` codegen into the
HPC/workflow emission so that figures render as their own scheduler jobs, siblings
of the experiment rules ``tvbo workflow snakemake`` already emits.

The idea in one line: *a figure's PROV ``used`` edges are its workflow dependency
edges.* Every layer of a figure binds to an experiment result container (via
``bsplot._container_path``); those containers are exactly the render rule's
``input:``, so the rule schedules after the experiments that produce them. The
per-figure resource request comes from ``Figure.workflow_overrides`` (a
``WorkflowConfig``) merged over the study-level ``workflow`` — the same override
pattern experiments use.

Resolution (used->inputs, workflow_overrides->resources, unit conversion) lives here
in Python; the rule *structure* lives in
``tvbo/templates/workflow/snakemake/tvbo-figure-rule.smk.mako`` (the house codegen
rule). ``emit_figure_rules`` returns the Snakemake rule text; ``write_figure_kit``
also freezes each figure's self-contained ``plot.py`` and the ``.smk`` snippet to disk.
"""
from __future__ import annotations

import datetime as _dt
import re
import shlex
from pathlib import Path

from tvbo.adapters import bsplot
from tvbo.cli import _workflow as _wf
from tvbo.templates import lookup
from tvbo.utils import as_list, deep_merge, sanitize_name

_RULE_TEMPLATE = "tvbo-figure-rule.smk.mako"


# --------------------------------------------------------------------------- helpers

def _figure_block(workflow, overrides, engine: str = "snakemake"):
    """Merge ``figure.workflow_overrides`` over the study ``workflow`` -> (spec, block).

    Reuses the ``_workflow`` merge machinery so the semantics match the experiment
    emitter exactly: name-keyed engine slots (env/options) merge by name, and the
    engine block inherits the engine-agnostic resource keys (and, for Snakemake, the
    SLURM scheduler identity) from the ``slurm`` block when it does not set them —
    unset falls back, an override wins only where it names a key.
    """
    base = _wf._canonicalize_engine_maps(_wf._as_plain_dict(workflow))
    over = _wf._canonicalize_engine_maps(_wf._as_plain_dict(overrides))
    spec = deep_merge(base, over)

    block = dict(spec.get(engine) or {})
    shared = ["cpus_per_task", "mem", "time", "modules", "venv", "env", "setup"]
    if engine == "snakemake":
        shared += ["partition", "account", "gres"]
    slurm = spec.get("slurm") or {}
    for k in shared:
        if k not in block and k in slurm:
            block[k] = slurm[k]
    if "env" in block:
        block["env"] = _wf._normalize_env(block["env"])
    if "options" in block:
        block["options"] = _wf._normalize_directives(block["options"])
    if "setup" in block:
        block["setup"] = _wf._as_lines(block["setup"])
    return spec, block


def _rule_resources(block: dict) -> dict:
    """Lower a merged engine block into a Snakemake ``resources:`` map.

    Returns ``{key: python-literal-string}`` (already repr'd so the template emits
    ``key=<literal>`` verbatim). ``cpus_per_task``/``mem_mb``/``runtime`` map across
    engines; the SLURM scheduler identity (partition/account/gres) is surfaced as the
    executor's ``slurm_partition``/``slurm_account``/``slurm_extra`` resources so a
    per-figure override lands on the rule itself. ``options`` pass through verbatim.
    """
    r: dict = {}
    if block.get("cpus_per_task"):
        r["cpus_per_task"] = str(int(block["cpus_per_task"]))
    # `is not None`, not truthiness: a declared 0 is a value the figure chose, and
    # dropping it silently hands the job the partition default instead.
    mb = _wf.mem_mb(block.get("mem"))
    if mb is not None:
        r["mem_mb"] = str(mb)
    rt = _wf.runtime_minutes(block.get("time"))
    if rt is not None:
        r["runtime"] = str(rt)
    if block.get("partition"):
        r["slurm_partition"] = repr(str(block["partition"]))
    if block.get("account"):
        r["slurm_account"] = repr(str(block["account"]))
    if block.get("gres"):
        r["slurm_extra"] = repr("--gres=" + str(block["gres"]))
    for opt in (block.get("options") or []):
        v = str(opt["value"])
        # numeric -> bare int literal; else a repr'd (safely escaped) string literal
        r[opt["name"]] = v if v.lstrip("-").isdigit() else repr(v)
    return r


def _exp_key_of(iri, keys) -> str | None:
    """The kit experiment key a figure ``used.iri`` points at, or ``None`` for an
    external reference. Mirrors ``bsplot._container_path``'s segment/digit extraction so
    the IRI a figure renders from also resolves to the rule that produces its cells."""
    if not iri:
        return None
    last = re.split(r"[:/#]", str(iri))[-1]
    digits = re.sub(r"\D", "", last)
    for cand in ([last, digits, f"exp{digits}", f"exp-{digits}"] if digits else [last]):
        if cand in keys:
            return cand
    return None


def _figure_inputs(figure, base_dir: Path, exp_plans_by_key: dict) -> list[dict]:
    """This figure's ``used`` result dependencies, deduped, as ``{value, raw}`` items.

    A ``used`` edge to a KIT experiment resolves to that experiment's own output files —
    the ``expand()`` over its fanned grid (so the figure waits for EVERY cell) or its single
    group-run path — emitted RAW so Snakemake evaluates the ``expand``. A ``used`` edge to
    something the kit does not produce (an external/author-time container) falls back to the
    single resolved container path via the same ``bsplot._container_path`` the ``plot.py``
    reads, emitted as a quoted string. Unresolved edges are dropped — a rule cannot depend
    on a file that does not exist.
    """
    inputs, seen = [], set()
    for panel in as_list(getattr(figure, "panels", None)):
        for layer in (getattr(panel, "layers", None) or []):
            iri = getattr(getattr(layer, "used", None), "iri", None)
            key = _exp_key_of(iri, exp_plans_by_key)
            if key is not None:
                item = {"value": _wf.fan_input_expr(exp_plans_by_key[key]), "raw": True}
            else:
                container = bsplot._container_path(iri, Path(base_dir))
                if not container:
                    continue
                item = {"value": container, "raw": False}
            dedup = (item["raw"], item["value"])
            if dedup not in seen:
                seen.add(dedup)
                inputs.append(item)
    return inputs


def _figure_context(figure, base_dir, workflow, exp_plans_by_key, bundled_code) -> dict:
    """Resolve one ``Figure`` into the template context for its render rule."""
    base_dir = Path(base_dir)
    name = figure.name or "figure"
    fmt = (figure.format or "png").lstrip(".")
    _spec, block = _figure_block(workflow, getattr(figure, "workflow_overrides", None))
    return {
        "name": name,
        "rule_name": "fig_" + sanitize_name(name),
        "inputs": _figure_inputs(figure, base_dir, exp_plans_by_key),
        "output": f"figures/{name}.{fmt}",
        "figures_dir": "figures",
        "script": f"figures/plot_{sanitize_name(name)}.py",
        # The figure's custom-panel code_modules are bundled into the kit's code/, so put it
        # on PYTHONPATH exactly as the experiment rules do when the kit carries bundled code.
        "pythonpath_code": bool(bundled_code),
        "threads": int(block.get("cpus_per_task") or block.get("cores") or 1),
        "resources": _rule_resources(block),
        "container": _spec.get("container"),
        # Activation runs in the rule shell before plot.py, exactly as the experiment rules
        # do: a fresh compute-node shell inherits nothing, so `modules`/`venv` must be put in
        # place or `python`/`bsplot` is the wrong (system) interpreter and the render fails.
        "activation": _activation_lines(block),
        "env": block.get("env") or [],
    }


def _activation_lines(block: dict) -> list[str]:
    """Shell lines that put the declared environment in place (modules, then venv, then the
    verbatim ``setup`` lines) — mirrors the experiment rules' ``_activation`` so a figure
    renders in the same interpreter its data was produced with."""
    lines = ["module load %s" % m for m in (block.get("modules") or [])]
    if block.get("venv"):
        lines.append("source %s/bin/activate" % shlex.quote(str(block["venv"])))
    return lines + list(block.get("setup") or [])


def figure_contexts(figures, base_dir=".", workflow=None, exp_plans=None,
                    bundled_code=False) -> list[dict]:
    """Per-figure template contexts (fan-aware inputs). ``exp_plans`` are the emitter's
    per-experiment dicts; without them (author-time render) inputs fall back to
    ``output/nc`` containers. Public so the study emitter can read the figure outputs it
    must add to the default target before it renders the Snakefile."""
    keys = {ep["key"]: ep for ep in (exp_plans or [])}
    return [_figure_context(f, base_dir, workflow, keys, bundled_code) for f in figures]


# --------------------------------------------------------------------------- emit

def emit_figure_rules(figures, base_dir=".", workflow=None, kit_dir="kit",
                      include_all: bool = False, exp_plans=None,
                      bundled_code: bool = False) -> str:
    """Render Snakemake render rules for *figures* — one rule per figure.

    Args:
        figures: An iterable of ``Figure`` objects (e.g. ``study.figures``).
        base_dir: Root the experiment result containers live under; each figure's
            ``used`` IRIs resolve to ``<base_dir>/output/nc/<exp>/*.h5``.
        workflow: The study-level ``WorkflowConfig`` (or ``None``); each figure's
            ``workflow_overrides`` merges over it for that figure's resources.
        kit_dir: Directory the companion :func:`write_figure_kit` writes to (kept for
            API symmetry; rule paths are kit-relative and independent of it).
        include_all: When True, prepend an aggregate ``all_figures`` target rule so
            the snippet is runnable standalone (``snakemake -s figures.smk``).
        exp_plans: The emitter's per-experiment dicts; a figure ``used`` edge to one of
            them becomes an ``expand()`` over that experiment's fanned cells (the whole
            grid), so the render waits for the sweep. Without them, inputs fall back to
            author-time ``output/nc`` containers.
        bundled_code: Whether the kit carries a ``code/`` dir the figure's custom-panel
            modules were bundled into (put on the rule's ``PYTHONPATH``).

    Returns:
        The Snakemake rule text. Each rule's ``input:`` is the figure's ``used``
        dependencies and its ``resources:`` reflect ``workflow_overrides`` over
        *workflow*; the rule runs the figure's frozen ``plot.py``.
    """
    fig_ctxs = figure_contexts(figures, base_dir, workflow, exp_plans, bundled_code)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    return lookup.get_template(_RULE_TEMPLATE).render(
        figures=fig_ctxs, now=now, include_all=include_all)


def write_figure_kit(figures, base_dir=".", out_dir="kit", workflow=None,
                     include_all: bool = True, exp_plans=None,
                     bundled_code: bool = False) -> Path:
    """Freeze a figure workflow kit to disk: per-figure ``plot.py`` + the ``.smk`` snippet.

    Layout::

        out_dir/
          figures.smk                # the render rules (from emit_figure_rules)
          figures/plot_<name>.py     # self-contained bsplot script per figure

    Each ``plot_<name>.py`` is ``bsplot.render_code(figure, base_dir, outfile=…)`` with
    ``outfile`` set to the rule's declared ``output`` (``figures/<name>.<fmt>``), so
    running ``python figures/plot_<name>.py`` from the kit root produces exactly what
    the rule promises. Returns the kit directory.
    """
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    for figure in figures:
        name = figure.name or "figure"
        fmt = (figure.format or "png").lstrip(".")
        script = out_dir / "figures" / f"plot_{sanitize_name(name)}.py"
        code = bsplot.render_code(figure, base_dir=base_dir,
                                  outfile=f"figures/{name}.{fmt}")
        script.write_text(code, encoding="utf-8")
    rules = emit_figure_rules(figures, base_dir, workflow=workflow,
                              kit_dir=str(out_dir), include_all=include_all,
                              exp_plans=exp_plans, bundled_code=bundled_code)
    (out_dir / "figures.smk").write_text(rules, encoding="utf-8")
    return out_dir
