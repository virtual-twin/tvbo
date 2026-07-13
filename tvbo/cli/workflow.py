"""``tvbo workflow`` — emit Slurm / Snakemake / Nextflow artefacts from a Study."""
from __future__ import annotations

import datetime as _dt
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import typer
from mako.template import Template

from . import _common
from . import _workflow as _wf
from ._backends import list_backends, resolve_backend


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
            wanted = [e for e in items if experiment_arg in _common.experiment_ids(e)]
            if not wanted:
                _common.die(f"No experiment named {experiment_arg!r} in study.")
            exp = wanted[0]
        else:
            exp = items[0]
        # Prefer the *runtime* experiment (has render/render_code/render_yaml) over the
        # datamodel object, so the kit can freeze the backend script + YAML snapshot.
        if not hasattr(exp, "render") and hasattr(obj, "get_experiment"):
            sel = (getattr(exp, "id", None) or getattr(exp, "key", None)
                   or getattr(exp, "name", None) or getattr(exp, "label", None))
            try:
                exp = obj.get_experiment(sel)
            except Exception as e:
                _common.die(
                    f"Could not resolve experiment {sel!r} to a runnable object: {e}\n"
                    "If the recipe references custom modules, make them importable "
                    "(run from their directory or set PYTHONPATH)."
                )
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


def _freeze_spec_yaml(experiment, spec_dir: Path, *, workflow_spec: dict | None = None) -> str:
    """Render the experiment as a self-contained YAML spec next to *spec_dir*.

    When the experiment has a connectome, its matrices are saved as an HDF5
    companion (``network.h5``) with a YAML sidecar (``network.yaml``) and the
    rendered spec references them through ``network.data_file`` while preserving
    any inline coupling / transforms / parameters. Without a connectome the plain
    metadata render already round-trips.

    *workflow_spec* is the effective merged workflow config (study < experiment <
    ``--set``). When given, the frozen spec's ``workflow`` block is rewritten to it,
    so the spec records exactly what ran and re-emits identically without the flags.
    """
    from tvbo import datamodel as dm
    from tvbo.classes.network import Network

    net = getattr(experiment, "network", None)
    has_net = _network_has_matrices(net)

    original_net = getattr(experiment, "network", None)
    original_wf = getattr(experiment, "workflow", None)
    if workflow_spec:
        effective = _wf.workflow_config_from_spec(workflow_spec)
        if effective is not None:
            experiment.workflow = effective
    try:
        if not has_net:
            return experiment.render(format="yaml")

        spec_dir.mkdir(parents=True, exist_ok=True)
        if not isinstance(net, Network):
            net.__class__ = Network
        # Bake real node labels into the frozen connectome so the kit is
        # self-contained and label reconciliation works on reload (the bids- block
        # that would hydrate them is dropped from the frozen spec).
        if hasattr(experiment, "bake_real_node_labels"):
            experiment.bake_real_node_labels()
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

        experiment.network = ref
        return experiment.to_yaml()
    finally:
        experiment.network = original_net
        if workflow_spec:
            experiment.workflow = original_wf


def _bundle_callable_modules(spec_yaml_text: str, out_dir: Path) -> bool:
    """Copy the recipe's custom callable/builder modules into the kit's ``code/``.

    A recipe references user code by bare module name (``callable: {module:
    my_analysis}`` / ``builder: {module: my_networks}``). Those modules are
    importable at emit time (on the author's ``PYTHONPATH``) but are not installed
    packages, so the frozen spec cannot resolve them on a compute node unless they
    travel with the kit. Each referenced module that resolves to a **local** ``.py``
    file (not under ``site-packages``/``dist-packages`` — installed deps are
    provisioned via ``requirements.txt``/``environment.yml`` instead) is copied into
    ``code/``. Returns True if anything was bundled (the sbatch then puts ``code`` on
    ``PYTHONPATH``).
    """
    import importlib
    import re
    import shutil

    names = set(re.findall(r'module:\s*["\']?([A-Za-z_][\w.]*)', spec_yaml_text))
    bundled: list[str] = []
    code_dir = out_dir / "code"
    for name in sorted(names):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        f = getattr(mod, "__file__", None) or ""
        if not f.endswith(".py") or "site-packages" in f or "dist-packages" in f:
            continue  # installed package — comes from the emitted requirements, not bundled
        code_dir.mkdir(exist_ok=True)
        shutil.copy2(f, code_dir / Path(f).name)
        bundled.append(Path(f).name)
    if bundled:
        _common.info(f"bundled callable modules → code/: {', '.join(bundled)}")
    return bool(bundled)


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
    bundled_code = False
    try:
        spec_dir = out_dir / "spec"
        yaml_text = _freeze_spec_yaml(experiment, spec_dir, workflow_spec=plan.workflow_spec)
        spec_path = spec_dir / f"{plan.experiment_key}.yaml"
        spec_path.write_text(yaml_text, encoding="utf-8")
        spec_relpath = str(spec_path.relative_to(out_dir))
        bundled_code = _bundle_callable_modules(yaml_text, out_dir)
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
    # BIDS result stem (no subject) so the engine can declare the exact output a
    # per-subject `tvbo run` writes (sub-<subject>_<stem>.h5), not a bare result.h5.
    try:
        result_stem = experiment.get_result_stem()
    except Exception:
        result_stem = "result"
    text = _render_template(
        _TEMPLATE_PATH[engine],
        plan=plan,
        block=plan.engine_block,
        script_relpath=str(script_path.relative_to(out_dir)) if script_path else None,
        spec_relpath=spec_relpath,
        bundled_code=bundled_code,
        result_stem=result_stem,
    )
    artefact.write_text(text, encoding="utf-8")
    _common.info(f"wrote {artefact.relative_to(out_dir)}")

    # 3a) Gather job — reassembles the array's shard outputs into one result per
    #     experiment (identical to a local run). Submitted with a dependency by
    #     `tvbo workflow run`, so no manual post-processing is needed. A single-task
    #     array (e.g. chunk=1 on one GPU) has nothing to reassemble — it writes the
    #     canonical result directly — so no gather job is emitted.
    if engine == "slurm" and plan.n_array_tasks > 1:
        # BIDS-style result stem (pybids), matching what a local ExperimentResult.save writes.
        try:
            result_stem = experiment.get_result_stem()
        except Exception:
            result_stem = "result"
        finalize = out_dir / "finalize.sbatch"
        finalize.write_text(
            _render_template("slurm/finalize.sbatch.mako", plan=plan,
                             block=plan.engine_block, result_stem=result_stem),
            encoding="utf-8")
        _common.info(f"wrote {finalize.relative_to(out_dir)}")

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


def _study_experiments(spec: str, experiment: str | None):
    """Resolve the runtime experiments a study/experiment SPEC fans out over.

    Returns ``(study_or_none, [runtime_experiments], study_key)``. A study yields
    all its experiments (or the ``--experiment`` id/label subset); a bare
    experiment SPEC yields a single-item list.
    """
    kind, obj = _common.resolve_spec(spec)
    if kind != "study":
        _study, exp, study_key = _resolve_study_and_experiment(spec, experiment)
        return None, [exp], study_key
    raw = getattr(obj, "experiments", None) or getattr(obj, "simulation_experiments", None) or []
    items = list(raw.values()) if hasattr(raw, "values") else list(raw)
    if experiment is not None:
        wanted = {s.strip() for s in str(experiment).split(",") if s.strip()}
        items = [e for e in items if wanted & _common.experiment_ids(e)]
        if not items:
            _common.die(f"No experiment(s) matching {experiment!r} in study.")
    resolved = []
    for e in items:
        if not hasattr(e, "render") and hasattr(obj, "get_experiment"):
            sel = (getattr(e, "id", None) or getattr(e, "key", None)
                   or getattr(e, "name", None) or getattr(e, "label", None))
            try:
                e = obj.get_experiment(sel)
            except Exception as exc:
                _common.die(f"Could not resolve experiment {sel!r} to a runnable object: {exc}")
        resolved.append(e)
    return obj, resolved, getattr(obj, "key", None) or "study"


def _emit_snakemake_study(*, spec: str, backend: str, experiment: str | None,
                          output: Path | None, override: list[str], stdout: bool = False):
    """Emit one Snakefile that fans every experiment (and, per experiment, every
    subject / sweep cell) into its own job. In kit mode each experiment is frozen
    into a self-contained ``spec/<key>/`` so a rule runs ``tvbo run
    spec/<key>/experiment.yaml``; in ``--stdout`` mode nothing is written and the
    rules run ``tvbo run <source-spec> --experiment <id>``."""
    study, experiments, study_key = _study_experiments(spec, experiment)
    out_dir = output or Path("output").joinpath(str(study_key), "snakemake")
    if not stdout:
        out_dir.mkdir(parents=True, exist_ok=True)
    parsed = _parse_overrides(override)

    def _san(s):
        return "".join(c if (c.isalnum() or c in ".-") else "_" for c in str(s))

    exp_plans, block, last_plan, bundled_code = [], {}, None, False
    for exp in experiments:
        key = _san(_common.experiment_key(exp))
        base = _wf.merge_workflow_spec(study, exp)
        spec_dict = _deep_merge(base, parsed["merged"])
        plan = _wf.plan(study_key=str(study_key), experiment=exp, backend=backend,
                        engine="snakemake", workflow_spec=spec_dict,
                        overrides=parsed["records"], source_spec=spec, experiment_selector=key)
        block = plan.engine_block or {}
        last_plan = plan
        try:
            result_stem = exp.get_result_stem()
        except Exception:
            result_stem = "result"
        if stdout:
            spec_relpath, select = spec, key
        else:
            edir = out_dir / "spec" / key
            yaml_text = _freeze_spec_yaml(exp, edir, workflow_spec=spec_dict)
            (edir / "experiment.yaml").write_text(yaml_text, encoding="utf-8")
            # Custom callable/builder modules the recipe references travel with the
            # kit (shared code/ dir), so `tvbo run` resolves them on the node.
            bundled_code = _bundle_callable_modules(yaml_text, out_dir) or bundled_code
            spec_relpath, select = f"spec/{key}/experiment.yaml", None
            _common.info(f"froze experiment {key} ({len(plan.workflow_axes)} fan-out axes)")
        exp_plans.append({
            "key": key,
            "rule_name": "exp_" + key.replace("-", "_").replace(".", "_"),
            "spec_relpath": spec_relpath,
            "select": select,
            "backend": backend,
            "out_dir": plan.out_dir,
            "result_stem": result_stem,
            "container": plan.container,
            "axes": [{"name": ax.name, "parameter": ax.parameter, "values": list(ax.values)}
                     for ax in plan.workflow_axes],
        })

    text = _render_template("snakemake/study.smk.mako", exp_plans=exp_plans,
                            block=block, bundled_code=bundled_code)
    if stdout:
        typer.echo(text)
        return None
    (out_dir / "Snakefile").write_text(text, encoding="utf-8")
    _common.info(f"wrote Snakefile ({len(exp_plans)} experiment rule(s))")
    if last_plan is not None:
        _write_readme(out_dir, engine="snakemake", plan=last_plan, script_relpath=None)
    return out_dir


def _emit(engine: str, *, spec: str, backend: str, experiment: str | None,
          output: Path | None, override: list[str], stdout: bool) -> None:
    if engine == "snakemake":
        return _emit_snakemake_study(spec=spec, backend=backend, experiment=experiment,
                                     output=output, override=override, stdout=stdout)
    plan, exp = _build_plan(spec, engine=engine, backend=backend,
                            experiment=experiment, overrides=override)
    if stdout:
        text = _render_template(_TEMPLATE_PATH[engine], plan=plan,
                                block=plan.engine_block, script_relpath=None)
        typer.echo(text)
        return None
    if output:
        out_dir = output
    else:
        # A standalone experiment has study_key == experiment_key (same fallback),
        # so collapse the redundant level: out/<experiment>/<engine> not …/x/x/….
        parts = ([plan.experiment_key] if plan.study_key == plan.experiment_key
                 else [plan.study_key, plan.experiment_key])
        out_dir = Path("output").joinpath(*parts, engine)
    _emit_kit(engine=engine, plan=plan, experiment=exp, out_dir=out_dir)
    return out_dir


def _execute_engine_artefact(engine: str, artefact: Path, *, slurm_array: str | None = None) -> None:
    """Submit/execute a rendered workflow artefact for *engine*.

    Runs from the artefact's own directory so the generated script can use the
    relative ``spec/`` and ``scripts/`` paths of an emitted kit. *slurm_array*
    restricts a Slurm submission to an index or range (``'0'`` for a single
    smoke task, ``'0-3'`` for four); ignored for non-Slurm engines.
    """
    if engine == "slurm":
        cmd = ["sbatch"]
        if slurm_array is not None:
            cmd.append(f"--array={slurm_array}")
        cmd.append(artefact.name)
    elif engine == "snakemake":
        cmd = ["snakemake", "--cores", "all"]
    elif engine == "nextflow":
        cmd = ["nextflow", "run", artefact.name]
    else:
        _common.die(f"unsupported engine {engine!r}; expected {'|'.join(_ARTEFACT_NAME)}")
    _common.info("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=artefact.parent)


def _execute_emitted(engine: str, out_dir: Path, *, slurm_array: str | None = None) -> None:
    """Execute a generated workflow artefact inside *out_dir*.

    For Slurm this submits the array job and then chains the gather job
    (``finalize.sbatch``) with an ``afterok`` dependency, so the run converges to
    one reassembled result with no manual step.
    """
    if engine == "slurm":
        _submit_slurm_chain(out_dir, slurm_array=slurm_array)
    else:
        _execute_engine_artefact(engine, out_dir / _ARTEFACT_NAME[engine], slurm_array=slurm_array)


def _submit_slurm_chain(out_dir: Path, *, slurm_array: str | None = None) -> None:
    """Submit ``run.sbatch`` (array), then ``finalize.sbatch`` with a dependency.

    ``sbatch --parsable`` returns the array job id; the gather job is submitted
    ``--dependency=afterok`` on it and told where the shards landed via
    ``TVBO_SHARD_DIR``, so it reassembles them into one result once every task
    succeeds.
    """
    cmd = ["sbatch", "--parsable"]
    if slurm_array is not None:
        cmd.append(f"--array={slurm_array}")
    cmd.append(_ARTEFACT_NAME["slurm"])
    _common.info("$ " + " ".join(shlex.quote(c) for c in cmd))
    res = subprocess.run(cmd, check=True, cwd=out_dir, capture_output=True, text=True)
    job_id = (res.stdout or "").strip().split(";")[0]
    _common.info(f"submitted array job {job_id}")

    finalize = out_dir / "finalize.sbatch"
    if finalize.exists() and job_id:
        base = job_id.split("_")[0]
        fcmd = ["sbatch", f"--dependency=afterok:{base}",
                f"--export=ALL,TVBO_SHARD_DIR=results/{base}", finalize.name]
        _common.info("$ " + " ".join(shlex.quote(c) for c in fcmd))
        subprocess.run(fcmd, check=True, cwd=out_dir)
        _common.info(f"submitted gather job (afterok:{base}) — one result when the array finishes")


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


@app.command("finalize", help="Reassemble an array run's shard outputs into one keyed result artifact.")
def finalize(
    shards_dir: str = typer.Argument("results", help="Directory holding the array tasks' shard outputs."),
    output: Path = typer.Option(Path("results"), "-o", "--output", help="Where to write the reassembled result."),
    spec: Path = typer.Option(None, "--spec", help="Frozen spec YAML to attach as the result sidecar (auto-detected in spec/ when omitted)."),
    stem: str = typer.Option("result", "--stem", help="Basename of the result artifact (<stem>.h5 + <stem>.yaml)."),
    compress: bool = typer.Option(True, "--compress/--no-compress",
                                  help="gzip-deflate the reassembled HDF5 (default on)."),
) -> None:
    """Gather sharded HPC outputs into one self-describing ``ExperimentResult``.

    Concatenates each array task's slice by parameter value into the full grid a
    local run produces, and writes it as ``<stem>.h5`` (keyed groups) plus a
    ``<stem>.yaml`` sidecar (the frozen, fully-overridden spec) — the same
    HDF5-plus-YAML layout as a network. No manual post-processing is needed;
    emitted kits submit this automatically as a dependent gather job.
    """
    from tvbo.data.types import reassemble_experiment_results

    sidecar = spec
    if sidecar is None:
        specs = sorted(p for p in Path("spec").glob("*.yaml")
                       if p.name != "network.yaml") if Path("spec").is_dir() else []
        sidecar = specs[0] if specs else None
    written = reassemble_experiment_results(shards_dir, str(output), stem=stem,
                                            sidecar=sidecar, compress=compress)
    for w in written:
        _common.info(f"wrote {w}")


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
    if engine not in _ARTEFACT_NAME:
        _common.die(f"`tvbo workflow run` expects engine one of: {', '.join(_ARTEFACT_NAME)}")
    # A comma-separated --experiment ("2,3,20,30") submits each as its own kit/job;
    # on the cluster they run in parallel. Each gets its own output subdir.
    _exp_ids = [e.strip() for e in str(experiment).split(",") if e.strip()] if experiment else []
    if len(_exp_ids) > 1:
        for _eid in _exp_ids:
            _out_i = (output / f"exp{_eid}") if output else Path("output") / f"exp{_eid}"
            _common.info(f"── experiment {_eid} → {_out_i}")
            run_workflow(engine, spec, backend, _eid, _out_i, override, array, array_throttle)
        return
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
