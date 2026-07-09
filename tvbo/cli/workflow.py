"""``tvbo workflow`` — emit Slurm / Snakemake / Nextflow artefacts from a Study."""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any

import typer
from mako.template import Template

from . import _common
from . import _workflow as _wf
from ._backends import list_backends, resolve_backend
from .run import _execute_engine_artefact


app = typer.Typer(name="workflow", no_args_is_help=True)


_TEMPLATES = Path(__file__).resolve().parent.parent / "templates" / "workflow"


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``--slurm.account=foo`` / ``--container=img`` style strings.

    Accepts both ``key=value`` and the same with a leading ``--`` (for
    convenience when users pipe things in). Dotted keys are nested.
    """
    out: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for raw in items:
        s = raw.lstrip("-")
        if "=" not in s:
            raise typer.BadParameter(f"override {raw!r} must be of the form key=value")
        k, _, v = s.partition("=")
        records.append({"key": k, "value": v, "source": "flag"})
        target = out
        parts = k.split(".")
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        # Try numeric/bool coercion
        coerced: Any = v
        if v.lower() in {"true", "false"}:
            coerced = v.lower() == "true"
        else:
            try:
                coerced = int(v)
            except ValueError:
                try:
                    coerced = float(v)
                except ValueError:
                    coerced = v
        target[parts[-1]] = coerced
    return {"merged": out, "records": records}


def _resolve_study_and_experiment(spec: str, experiment_arg: str | None):
    """Load *spec* and pick the experiment to plan against.

    Returns ``(study_obj_or_none, experiment_obj, study_key)``.
    """
    kind, obj = _common.resolve_spec(spec)
    if kind == "study":
        exps = getattr(obj, "experiments", None) or getattr(obj, "simulation_experiments", None) or []
        items = list(exps.values()) if hasattr(exps, "values") else list(exps)
        if not items:
            _common.die(f"Study {spec!r} has no experiments.")
        if experiment_arg is not None:
            wanted = [
                e for e in items
                if (getattr(e, "key", None) == experiment_arg
                    or getattr(e, "name", None) == experiment_arg
                    or getattr(e, "label", None) == experiment_arg)
            ]
            if not wanted:
                _common.die(f"No experiment named {experiment_arg!r} in study.")
            exp = wanted[0]
        else:
            exp = items[0]
        # Prefer the *runtime* experiment (has render/render_code/render_yaml) over the
        # datamodel object, so the kit can freeze the backend script + YAML snapshot.
        if hasattr(obj, "get_experiment"):
            sel = getattr(exp, "id", None)
            if sel is None:
                sel = getattr(exp, "key", None) or getattr(exp, "name", None) or getattr(exp, "label", None)
            try:
                exp = obj.get_experiment(sel)
            except Exception:
                pass
        return obj, exp, getattr(obj, "key", None) or "study"

    if kind == "experiment":
        return None, obj, getattr(obj, "key", None) or "experiment"

    _common.die(f"`tvbo workflow` requires a Study or Experiment SPEC; got kind={kind!r}.")


def _build_plan(spec: str, *, engine: str, backend: str,
                experiment: str | None, overrides: list[str]):
    """Return ``(plan, experiment_obj)``."""
    study, exp, study_key = _resolve_study_and_experiment(spec, experiment)
    base = _wf.merge_workflow_spec(study, exp)
    parsed = _parse_overrides(overrides)
    spec_dict = _deep_merge(base, parsed["merged"])
    plan = _wf.plan(
        study_key=str(study_key),
        experiment=exp,
        backend=backend,
        engine=engine,
        workflow_spec=spec_dict,
        overrides=parsed["records"],
        source_spec=spec,
        experiment_selector=experiment,
    )
    return plan, exp


from tvbo.utils import deep_merge as _deep_merge  # noqa: E402  (shared recursive merge)


def _render_template(rel: str, **ctx) -> str:
    tpl = Template(filename=str(_TEMPLATES / rel),
                   strict_undefined=False,
                   imports=["import os"])
    return tpl.render(now=_dt.datetime.now().isoformat(timespec="seconds"), **ctx)


def _write_or_stdout(text: str, output: Path | None) -> None:
    if output is None:
        typer.echo(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    _common.info(f"wrote {output}")


_ARTEFACT_NAME = {
    "slurm": "run.sbatch",
    "snakemake": "Snakefile",
    "nextflow": "main.nf",
}

_TEMPLATE_PATH = {
    "slurm": "slurm/run.sbatch.mako",
    "snakemake": "snakemake/study.smk.mako",
    "nextflow": "nextflow/main.nf.mako",
}


def _network_has_matrices(net) -> bool:
    """True when a network carries a real connectome (more than a placeholder node).

    A metadata-only experiment has ``number_of_nodes in (None, 0, 1)`` and no
    weights; such a network round-trips fine as inline YAML and needs no
    companion file.
    """
    if net is None:
        return False
    n = getattr(net, "number_of_nodes", None)
    if n and n > 1:
        return True
    try:
        import numpy as _np

        return _np.asarray(net.weights).size > 1
    except Exception:
        return False


def _freeze_spec_yaml(experiment, spec_dir: Path) -> str:
    """Render the experiment as a self-contained YAML spec next to *spec_dir*.

    When the experiment has a connectome, its matrices are saved as an HDF5
    companion (``network.h5``) with a YAML sidecar (``network.yaml``) and the
    rendered spec references them through ``network.data_file`` while preserving
    any inline coupling / transforms / parameters. Without a connectome the plain
    metadata render already round-trips, so it is returned unchanged.
    """
    from tvbo import datamodel as dm
    from tvbo.classes.network import Network

    net = getattr(experiment, "network", None)
    if not _network_has_matrices(net):
        return experiment.render(format="yaml")

    spec_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(net, Network):
        net.__class__ = Network
    net.save(spec_dir / "network.yaml", binary_format="h5")
    _common.info("wrote spec/network.yaml + spec/network.h5")

    # Compact network reference: data_file + inline coupling/transforms/parameters,
    # so the rendered spec loads the companion connectome rather than a stub.
    ref = dm.Network(data_file="network.h5")
    if getattr(net, "coupling", None):
        for k, v in dict(net.coupling).items():
            ref.coupling[k] = v
    if getattr(net, "transforms", None):
        ref.transforms = list(net.transforms)
    if getattr(net, "parameters", None):
        for k, v in dict(net.parameters).items():
            ref.parameters[k] = v

    original = experiment.network
    experiment.network = ref
    try:
        return experiment.to_yaml()
    finally:
        experiment.network = original


def _emit_kit(*, engine: str, plan, experiment, out_dir: Path) -> Path:
    """Write a self-contained reproducibility kit under *out_dir*.

    Layout::

        out_dir/
          <artefact>            # Snakefile / run.sbatch / main.nf
          scripts/<exp>.<ext>   # frozen backend code from experiment.render(backend)
          spec/<exp>.yaml       # frozen YAML snapshot of the experiment
          README.md             # provenance + how-to-run
    """
    from tvbo import export as _export

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scripts").mkdir(exist_ok=True)
    (out_dir / "spec").mkdir(exist_ok=True)

    # 1) Frozen YAML spec snapshot (self-contained run target when it round-trips).
    #    A connectome-backed experiment cannot be frozen as inline YAML: the
    #    metadata render drops the matrices, so the reloaded network collapses to
    #    a single node. Instead the network is written as an HDF5 companion
    #    (network.h5) + YAML sidecar (network.yaml) and referenced from the spec
    #    via ``network.data_file`` — the mechanism resolve_spec loads on the node.
    spec_relpath = None
    try:
        spec_dir = out_dir / "spec"
        yaml_text = _freeze_spec_yaml(experiment, spec_dir)
        spec_path = spec_dir / f"{plan.experiment_key}.yaml"
        spec_path.write_text(yaml_text, encoding="utf-8")
        spec_relpath = str(spec_path.relative_to(out_dir))
    except Exception as exc:
        _common.info(f"(could not snapshot YAML spec: {exc})")

    # 2) Frozen backend script with __main__ component
    script_path = None
    try:
        fmt = _export.resolve(plan.backend.name)
        ext = (fmt.extension or ".py").lstrip(".")
        code = experiment.render(format=plan.backend.name)
        script_path = out_dir / "scripts" / f"{plan.experiment_key}.{ext}"
        script_path.write_text(code, encoding="utf-8")
        _common.info(f"wrote {script_path.relative_to(out_dir)}")
    except Exception as exc:
        _common.info(f"(could not render backend script for {plan.backend.name!r}: {exc})")

    # 3) Workflow artefact
    artefact = out_dir / _ARTEFACT_NAME[engine]
    text = _render_template(
        _TEMPLATE_PATH[engine],
        plan=plan,
        block=plan.engine_block,
        script_relpath=str(script_path.relative_to(out_dir)) if script_path else None,
        spec_relpath=spec_relpath,
    )
    artefact.write_text(text, encoding="utf-8")
    _common.info(f"wrote {artefact.relative_to(out_dir)}")

    # 3b) Environment files (pip + conda) rendered via Mako from the experiment's
    #     declared environment.requirements, so the kit provisions the right env.
    if plan.pip_specs:
        (out_dir / "requirements.txt").write_text(
            _render_template("requirements.txt.mako", plan=plan), encoding="utf-8")
        (out_dir / "environment.yml").write_text(
            _render_template("environment.yml.mako", plan=plan), encoding="utf-8")
        _common.info("wrote requirements.txt + environment.yml")

    # 4) README
    _write_readme(out_dir, engine=engine, plan=plan, script_relpath=
                  str(script_path.relative_to(out_dir)) if script_path else None)
    return out_dir


def _write_readme(out_dir: Path, *, engine: str, plan, script_relpath: str | None) -> None:
    lines = [
        f"# {plan.study_key} / {plan.experiment_key}",
        "",
        f"Generated by `tvbo workflow {engine}` on {_dt.datetime.now().isoformat(timespec='seconds')}.",
        "",
        "## Layout",
        "",
        f"- `{_ARTEFACT_NAME[engine]}` — workflow artefact (run with the engine below)",
    ]
    if script_relpath:
        lines.append(f"- `{script_relpath}` — frozen backend code (`{plan.backend.name}`); has a `__main__` block, runnable with `python`")
    lines += [
        f"- `spec/{plan.experiment_key}.yaml` — frozen YAML snapshot of the experiment",
        "",
        "## Run",
        "",
        "```bash",
    ]
    if engine == "slurm":
        lines.append(f"sbatch {_ARTEFACT_NAME[engine]}")
    elif engine == "snakemake":
        lines.append("snakemake --cores all")
    elif engine == "nextflow":
        lines.append("nextflow run main.nf")
    lines += [
        "```", "", "## Reproducibility", "",
        f"- backend       : `{plan.backend.name}` ({plan.backend.label})",
        f"- container     : `{plan.container or '(none)'}`",
        f"- workflow cells: {plan.n_workflow_cells} (chunk={plan.chunk}, array tasks={plan.n_array_tasks})",
        f"- vectorize_axes: {[ax.name for ax in plan.vectorize_axes]}",
        f"- workflow_axes : {[ax.name for ax in plan.workflow_axes]}",
        "",
    ]
    if plan.overrides:
        lines.append("### CLI overrides")
        lines.append("")
        for o in plan.overrides:
            lines.append(f"- `{o['key']}` = `{o['value']}` ({o['source']})")
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("plan", help="Show the resolved workflow plan (no artefact emitted).")
def plan_cmd(
    spec: str = typer.Argument(..., help="Path, CURIE, or DB name (Study or Experiment)."),
    backend: str = typer.Option("tvboptim", "--backend", "-b"),
    engine: str = typer.Option("local", "--engine", "-e"),
    experiment: str = typer.Option(None, "--experiment"),
    override: list[str] = typer.Option(
        [], "--set", help="Override a workflow spec key, e.g. ``--set slurm.account=foo`` (repeatable)."
    ),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Show the resolved workflow plan for *spec* without emitting any artefacts.

    Reports the chosen study/experiment, backend, engine, vectorized vs
    workflow-fanned axes, total cell count, chunking, and applied overrides.
    """
    plan, _exp = _build_plan(spec, engine=engine, backend=backend,
                             experiment=experiment, overrides=override)

    if json:
        import json as _json
        payload = {
            "study": plan.study_key,
            "experiment": plan.experiment_key,
            "backend": plan.backend.name,
            "engine": plan.engine,
            "container": plan.container,
            "out_dir": plan.out_dir,
            "vectorize_axes": [
                {"name": ax.name, "parameter": ax.parameter, "kind": ax.kind, "n": ax.n}
                for ax in plan.vectorize_axes
            ],
            "workflow_axes": [
                {"name": ax.name, "parameter": ax.parameter, "kind": ax.kind, "n": ax.n}
                for ax in plan.workflow_axes
            ],
            "n_workflow_cells": plan.n_workflow_cells,
            "n_array_tasks": plan.n_array_tasks,
            "chunk": plan.chunk,
            "engine_block": plan.engine_block,
            "overrides": plan.overrides,
        }
        typer.echo(_json.dumps(payload, default=str, indent=2))
        return

    typer.echo(f"study      : {plan.study_key}")
    typer.echo(f"experiment : {plan.experiment_key}")
    typer.echo(f"backend    : {plan.backend.name} ({plan.backend.label})")
    typer.echo(f"engine     : {plan.engine}")
    typer.echo(f"container  : {plan.container or '(none)'}")
    typer.echo(f"out_dir    : {plan.out_dir}")
    typer.echo("")
    typer.echo("vectorized inside backend (1 job covers all):")
    for ax in plan.vectorize_axes:
        typer.echo(f"  - {ax.name:<12} {ax.parameter}  kind={ax.kind}  n={ax.n}")
    typer.echo("")
    typer.echo(f"workflow-fanned axes (engine spawns 1 task per cell):")
    for ax in plan.workflow_axes:
        typer.echo(f"  - {ax.name:<12} {ax.parameter}  kind={ax.kind}  n={ax.n}")
    typer.echo("")
    typer.echo(f"total workflow cells : {plan.n_workflow_cells}")
    typer.echo(f"chunk                : {plan.chunk}  →  {plan.n_array_tasks} array task(s)")
    if plan.overrides:
        typer.echo("\noverrides:")
        for o in plan.overrides:
            typer.echo(f"  - {o['key']}={o['value']!r}  ({o['source']})")


def _emit(engine: str, *, spec: str, backend: str, experiment: str | None,
          output: Path | None, override: list[str], stdout: bool) -> None:
    plan, exp = _build_plan(spec, engine=engine, backend=backend,
                            experiment=experiment, overrides=override)
    if stdout:
        text = _render_template(_TEMPLATE_PATH[engine], plan=plan,
                                block=plan.engine_block, script_relpath=None)
        typer.echo(text)
        return None
    out_dir = output or Path("out") / plan.study_key / plan.experiment_key / engine
    _emit_kit(engine=engine, plan=plan, experiment=exp, out_dir=out_dir)
    return out_dir


def _execute_emitted(engine: str, out_dir: Path, *, slurm_array: str | None = None) -> None:
    """Execute a generated workflow artefact inside *out_dir*."""
    artefact = out_dir / _ARTEFACT_NAME[engine]
    _execute_engine_artefact(engine, artefact, slurm_array=slurm_array)


@app.command("slurm", help="Emit a self-contained sbatch kit (artefact + scripts + spec).")
def slurm(
    spec: str = typer.Argument(...),
    backend: str = typer.Option("tvboptim", "--backend", "-b"),
    experiment: str = typer.Option(None, "--experiment"),
    output: Path = typer.Option(None, "-o", "--output", help="Output directory."),
    override: list[str] = typer.Option([], "--set"),
    stdout: bool = typer.Option(False, "--stdout", help="Print artefact only; do not write a kit."),
) -> None:
    """Emit a self-contained sbatch kit (`run.sbatch` + scripts + frozen spec)."""
    _emit("slurm", spec=spec, backend=backend, experiment=experiment,
          output=output, override=override, stdout=stdout)


@app.command("snakemake", help="Emit a self-contained Snakemake kit (Snakefile + scripts + spec).")
def snakemake(
    spec: str = typer.Argument(...),
    backend: str = typer.Option("tvboptim", "--backend", "-b"),
    experiment: str = typer.Option(None, "--experiment"),
    output: Path = typer.Option(None, "-o", "--output", help="Output directory."),
    override: list[str] = typer.Option([], "--set"),
    stdout: bool = typer.Option(False, "--stdout", help="Print artefact only; do not write a kit."),
) -> None:
    """Emit a self-contained Snakemake kit (`Snakefile` + scripts + frozen spec)."""
    _emit("snakemake", spec=spec, backend=backend, experiment=experiment,
          output=output, override=override, stdout=stdout)


@app.command("nextflow", help="Emit a self-contained Nextflow kit (main.nf + scripts + spec).")
def nextflow(
    spec: str = typer.Argument(...),
    backend: str = typer.Option("tvboptim", "--backend", "-b"),
    experiment: str = typer.Option(None, "--experiment"),
    output: Path = typer.Option(None, "-o", "--output", help="Output directory."),
    override: list[str] = typer.Option([], "--set"),
    stdout: bool = typer.Option(False, "--stdout", help="Print artefact only; do not write a kit."),
) -> None:
    """Emit a self-contained Nextflow kit (`main.nf` + scripts + frozen spec)."""
    _emit("nextflow", spec=spec, backend=backend, experiment=experiment,
          output=output, override=override, stdout=stdout)


@app.command("run", help="Emit a workflow kit and execute it with the selected engine.")
def run_workflow(
    engine: str = typer.Argument(..., help="Execution engine: slurm | snakemake | nextflow."),
    spec: str = typer.Argument(..., help="Path, CURIE, or DB name (Study or Experiment)."),
    backend: str = typer.Option("tvboptim", "--backend", "-b"),
    experiment: str = typer.Option(None, "--experiment"),
    output: Path = typer.Option(None, "-o", "--output", help="Output directory."),
    override: list[str] = typer.Option([], "--set"),
    array: str = typer.Option(
        None, "--array",
        help="Slurm array index or range to submit (e.g. '0' for smoke, '0-3' for four tasks). Ignored for non-Slurm engines.",
    ),
    array_throttle: int = typer.Option(
        None,
        "--array-throttle",
        min=1,
        help="Limit concurrent Slurm array tasks when using --array, e.g. 1 for one GPU at a time.",
    ),
) -> None:
    """Emit a self-contained kit then execute it (or submit for Slurm).

    Use ``--array 0`` to submit only the first array task as a quick smoke test
    without changing the experiment spec. Use ``--array-throttle`` to cap how
    many Slurm array tasks run at once, for example ``--array 0-39 --array-throttle 1``
    to keep one GPU busy at a time.
    """
    engine = engine.lower()
    if engine not in {"slurm", "snakemake", "nextflow"}:
        _common.die("`tvbo workflow run` expects engine one of: slurm, snakemake, nextflow")
    effective_overrides = list(override)
    if engine == "slurm" and array is not None:
        override_keys = {
            s.lstrip("-").split("=", 1)[0]
            for s in effective_overrides
            if "=" in s
        }
        chunk_keys = {"distribute.chunk", "chunk", "slurm.array_chunk"}
        if not (override_keys & chunk_keys):
            plan_preview, _exp = _build_plan(
                spec,
                engine=engine,
                backend=backend,
                experiment=experiment,
                overrides=effective_overrides,
            )
            # Smoke submissions should not run the full vectorized sweep as task 0/1.
            if (not plan_preview.workflow_axes
                    and plan_preview.n_array_tasks == 1
                    and plan_preview.n_vectorize_cells > 1):
                effective_overrides.append(
                    f"slurm.array_chunk={plan_preview.n_vectorize_cells}"
                )
                _common.info(
                    "auto smoke chunking: set slurm.array_chunk="
                    f"{plan_preview.n_vectorize_cells} for --array run"
                )
    out_dir = _emit(engine, spec=spec, backend=backend, experiment=experiment,
                    output=output, override=effective_overrides, stdout=False)
    if out_dir is None:
        _common.die("failed to emit workflow kit")
    if array is not None and array_throttle is not None:
        array = f"{array}%{array_throttle}"
    _execute_emitted(engine, out_dir, slurm_array=array)


@app.command("backends", help="List backends and their ontology-derived capabilities.")
def backends(
    json: bool = typer.Option(False, "--json"),
) -> None:
    """List execution backends and their ontology-derived capabilities (continuous/spiking/jit/etc.)."""
    rows = []
    for spec in list_backends():
        rows.append({
            "name": spec.name,
            "label": spec.label,
            "tasks": sorted(spec.tasks),
            "capabilities": sorted(spec.capabilities),
            "vectorize_axes": sorted(spec.vectorize_axes),
        })
    if json:
        import json as _json
        typer.echo(_json.dumps(rows, indent=2))
        return
    for r in rows:
        typer.echo(f"{r['name']}  ({r['label']})")
        typer.echo(f"  tasks         : {', '.join(r['tasks'])}")
        typer.echo(f"  capabilities  : {', '.join(r['capabilities'])}")
        typer.echo(f"  vectorize_axes: {', '.join(r['vectorize_axes']) or '(none — workflow handles all)'}")
        typer.echo("")
