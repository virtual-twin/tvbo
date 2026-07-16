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
from pathlib import Path

from tvbo.adapters import bsplot
from tvbo.cli import _workflow as _wf
from tvbo.templates import lookup
from tvbo.utils import as_list, deep_merge, sanitize_name

_RULE_TEMPLATE = "tvbo-figure-rule.smk.mako"


# --------------------------------------------------------------------------- helpers

def _mem_mb(mem):
    """'8G'/'8GB'/'512M'/'2000' -> integer megabytes (Snakemake ``mem_mb``)."""
    if not mem:
        return None
    s = str(mem).strip().upper().rstrip("B")
    try:
        if s.endswith("G"):
            return int(float(s[:-1]) * 1000)
        if s.endswith("M"):
            return int(float(s[:-1]))
        if s.endswith("K"):
            return max(1, int(float(s[:-1]) / 1000))
        return int(float(s))
    except ValueError:
        return None


def _runtime_min(t):
    """'02:00:00' (HH:MM:SS) or '120' -> integer minutes (Snakemake ``runtime``)."""
    if not t:
        return None
    s = str(t).strip()
    try:
        if ":" in s:
            p = [int(x) for x in s.split(":")]
            if len(p) == 3:
                return p[0] * 60 + p[1] + (1 if p[2] else 0)
            if len(p) == 2:
                return p[0] + (1 if p[1] else 0)
        return int(float(s))
    except ValueError:
        return None


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
    mb = _mem_mb(block.get("mem"))
    if mb:
        r["mem_mb"] = str(mb)
    rt = _runtime_min(block.get("time"))
    if rt:
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


def _figure_inputs(figure, base_dir: Path) -> list[str]:
    """Deduped result-container paths this figure's layers ``used`` (first-seen order).

    Each layer's ``used.iri`` is resolved to its container with the same
    ``bsplot._container_path`` the emitted ``plot.py`` reads, so the rule's inputs
    are exactly the files the render will open. Unresolved (missing) containers are
    dropped — a rule cannot depend on a file that does not exist.
    """
    inputs, seen = [], set()
    for panel in as_list(getattr(figure, "panels", None)):
        for layer in (getattr(panel, "layers", None) or []):
            used = getattr(layer, "used", None)
            container = bsplot._container_path(getattr(used, "iri", None), base_dir)
            if container and container not in seen:
                seen.add(container)
                inputs.append(container)
    return inputs


def _figure_context(figure, base_dir, workflow) -> dict:
    """Resolve one ``Figure`` into the template context for its render rule."""
    base_dir = Path(base_dir)
    name = figure.name or "figure"
    fmt = (figure.format or "png").lstrip(".")
    _spec, block = _figure_block(workflow, getattr(figure, "workflow_overrides", None))
    return {
        "name": name,
        "rule_name": "fig_" + sanitize_name(name),
        "inputs": _figure_inputs(figure, base_dir),
        "output": f"figures/{name}.{fmt}",
        "figures_dir": "figures",
        "script": f"figures/plot_{sanitize_name(name)}.py",
        "threads": int(block.get("cpus_per_task") or block.get("cores") or 1),
        "resources": _rule_resources(block),
        "container": _spec.get("container"),
        "setup": block.get("setup") or [],
        "env": block.get("env") or [],
    }


# --------------------------------------------------------------------------- emit

def emit_figure_rules(figures, base_dir=".", workflow=None, kit_dir="kit",
                      include_all: bool = False) -> str:
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

    Returns:
        The Snakemake rule text. Each rule's ``input:`` is the figure's ``used``
        containers and its ``resources:`` reflect ``workflow_overrides`` over
        *workflow*; the rule runs the figure's frozen ``plot.py``.
    """
    fig_ctxs = [_figure_context(f, base_dir, workflow) for f in figures]
    now = _dt.datetime.now().isoformat(timespec="seconds")
    return lookup.get_template(_RULE_TEMPLATE).render(
        figures=fig_ctxs, now=now, include_all=include_all)


def write_figure_kit(figures, base_dir=".", out_dir="kit", workflow=None,
                     include_all: bool = True) -> Path:
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
                              kit_dir=str(out_dir), include_all=include_all)
    (out_dir / "figures.smk").write_text(rules, encoding="utf-8")
    return out_dir
