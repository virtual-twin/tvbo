"""``tvbo workflow`` — emit Slurm / Snakemake / Nextflow artefacts from a Study."""

from __future__ import annotations

import datetime as _dt
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional

import typer
from mako.template import Template

from . import _common
from . import _workflow as _wf
from ._backends import list_backends


app = typer.Typer(name="workflow", no_args_is_help=True)


_TEMPLATES = Path(__file__).resolve().parent.parent / "templates" / "workflow"


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``--slurm.account=foo`` / ``--container=img`` style strings.

    Accepts both ``key=value`` and the same with a leading ``--`` (for convenience when users pipe things in). Dotted keys are nested.
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
        # Coerce the value: JSON for a list/object literal (e.g. a `setup` command list), else bool/int/float, else the raw string.
        coerced: Any = v
        if v.lstrip()[:1] in ("[", "{"):
            try:
                coerced = json.loads(v)
            except ValueError:
                coerced = v
        elif v.lower() in {"true", "false"}:
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
        elif len(items) == 1:
            exp = items[0]
        else:
            ids = ", ".join(sorted(_common.experiment_key(e) for e in items))
            _common.die(
                f"Study {spec!r} has {len(items)} experiments ({ids}); a single-kit "
                "engine will not silently pick the first.\n"
                "Emit the WHOLE study as one Snakemake DAG with "
                "`tvbo workflow snakemake <spec>` (no --experiment), or pass "
                "`--experiment <id>` to emit exactly one."
            )
        # Prefer the *runtime* experiment (has render/render_code/render_yaml) over the datamodel object, so the kit can freeze the backend script + YAML snapshot.
        if not hasattr(exp, "render") and hasattr(obj, "get_experiment"):
            sel = (
                getattr(exp, "id", None)
                or getattr(exp, "key", None)
                or getattr(exp, "name", None)
                or getattr(exp, "label", None)
            )
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


def _build_plan(spec: str, *, engine: str, backend: str, experiment: str | None, overrides: list[str]):
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


def _build_plans(spec: str, *, engine: str, backend: str, experiment: str | None, overrides: list[str]):
    """Return ``(study_or_none, [(plan, experiment_obj), ...])``.

    A study SPEC with no ``--experiment`` plans EVERY experiment (the whole study) — mirroring how the snakemake kit emits one rule per experiment, never silently collapsing to the first. An explicit ``--experiment`` (comma list) subsets; a bare experiment SPEC yields a single plan.
    """
    study, experiments, study_key = _study_experiments(spec, experiment)
    parsed = _parse_overrides(overrides)
    built = []
    for exp in experiments:
        base = _wf.merge_workflow_spec(study, exp)
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
        built.append((plan, exp))
    return study, built


def _plan_payload(plan) -> dict:
    """The JSON-serialisable view of one resolved plan (shared by ``plan --json``)."""
    return {
        "study": plan.study_key,
        "experiment": plan.experiment_key,
        "backend": plan.backend.name,
        "engine": plan.engine,
        "container": plan.container,
        "out_dir": plan.out_dir,
        "vectorize_axes": [
            {"name": ax.name, "parameter": ax.parameter, "kind": ax.kind, "n": ax.n} for ax in plan.vectorize_axes
        ],
        "workflow_axes": [
            {"name": ax.name, "parameter": ax.parameter, "kind": ax.kind, "n": ax.n} for ax in plan.workflow_axes
        ],
        "n_workflow_cells": plan.n_workflow_cells,
        "n_array_tasks": plan.n_array_tasks,
        "chunk": plan.chunk,
        "engine_block": plan.engine_block,
        "overrides": plan.overrides,
    }


def _print_plan_block(plan, *, show_study: bool = True) -> None:
    """Print one plan's human-readable block. ``show_study`` prints the study line (suppressed in whole-study mode where it heads the whole listing once)."""
    if show_study:
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
    typer.echo("workflow-fanned axes (engine spawns 1 task per cell):")
    for ax in plan.workflow_axes:
        typer.echo(f"  - {ax.name:<12} {ax.parameter}  kind={ax.kind}  n={ax.n}")
    typer.echo("")
    typer.echo(f"total workflow cells : {plan.n_workflow_cells}")
    typer.echo(f"chunk                : {plan.chunk}  →  {plan.n_array_tasks} array task(s)")
    if plan.overrides:
        typer.echo("\noverrides:")
        for o in plan.overrides:
            typer.echo(f"  - {o['key']}={o['value']!r}  ({o['source']})")


def _render_template(rel: str, **ctx) -> str:
    tpl = Template(filename=str(_TEMPLATES / rel), strict_undefined=False, imports=["import os"])
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

    A metadata-only experiment has ``number_of_nodes in (None, 0, 1)`` and no weights; such a network round-trips fine as inline YAML and needs no companion file.
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


def _freeze_spec_yaml(
    experiment, spec_dir: Path, *, workflow_spec: dict | None = None, dataset_bids_root: str | None = None
) -> str:
    """Render the experiment as a self-contained YAML spec next to *spec_dir*.

    When the experiment has a connectome, its matrices are saved as an HDF5 companion (``network.h5``) with a YAML sidecar (``network.yaml``) and the rendered spec references them through ``network.data_file`` while preserving any inline coupling / transforms / parameters. Without a connectome the plain metadata render already round-trips.

    *workflow_spec* is the effective merged workflow config (study < experiment <
    ``--set``). When given, the frozen spec's ``workflow`` block is rewritten to it, so the spec records exactly what ran and re-emits identically without the flags.

    *dataset_bids_root* rewrites the frozen ``dataset.bids_root`` (e.g. to a relative
    ``dataset`` once the per-subject data is bundled under ``spec/dataset``), so the spec points at the bundled tree instead of the author's machine-specific path.
    """
    from tvbo import datamodel as dm
    from tvbo.classes.network import Network

    net = getattr(experiment, "network", None)
    has_net = _network_has_matrices(net)

    original_net = getattr(experiment, "network", None)
    original_wf = getattr(experiment, "workflow", None)
    ds = getattr(experiment, "dataset", None)
    original_ds_root = getattr(ds, "bids_root", None) if ds is not None else None
    if dataset_bids_root is not None and ds is not None:
        ds.bids_root = dataset_bids_root
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
        # Bake real node labels into the frozen connectome so the kit is self-contained and label reconciliation works on reload (the bids- block that would hydrate them is dropped from the frozen spec).
        if hasattr(experiment, "bake_real_node_labels"):
            experiment.bake_real_node_labels()
        net.save(spec_dir / "network.yaml", binary_format="h5")
        _common.info("wrote spec/network.yaml + spec/network.h5")

        # Compact network reference: data_file + inline coupling/transforms/parameters, so the rendered spec loads the companion connectome rather than a stub.
        ref = dm.Network(data_file="network.h5")
        if getattr(net, "coupling", None):
            for k, v in dict(net.coupling).items():
                ref.coupling[k] = v
        if getattr(net, "transforms", None):
            ref.transforms = list(net.transforms)
        if getattr(net, "parameters", None):
            for k, v in dict(net.parameters).items():
                ref.parameters[k] = v
        # Every non-None scalar, so the next such field survives the round-trip without an edit here.
        for _f in (
            "label",
            "descriptor",
            "number_of_nodes",
            "distance_unit",
            "time_unit",
            "structural_measures",
            "observational_measures",
        ):
            _v = getattr(net, _f, None)
            if _v is not None:
                setattr(ref, _f, list(_v) if isinstance(_v, (list, tuple)) else _v)

        experiment.network = ref
        return experiment.to_yaml()
    finally:
        experiment.network = original_net
        if workflow_spec:
            experiment.workflow = original_wf
        if dataset_bids_root is not None and ds is not None:
            ds.bids_root = original_ds_root


def _bundle_callable_modules(spec_yaml_text: str, out_dir: Path) -> bool:
    """Copy the recipe's custom callable/builder modules into the kit's ``code/``.

    A recipe references user code by bare module name (``callable: {module:
    my_analysis}`` / ``builder: {module: my_networks}``). Those modules are importable at emit time (on the author's ``PYTHONPATH``) but are not installed packages, so the frozen spec cannot resolve them on a compute node unless they travel with the kit. Each referenced module that resolves to a **local** ``.py`` file (not under ``site-packages``/``dist-packages`` — installed deps are provisioned via ``requirements.txt``/``environment.yml`` instead) is copied into ``code/``. Returns True if anything was bundled (the sbatch then puts ``code`` on ``PYTHONPATH``).
    """
    import re

    names = set(re.findall(r'module:\s*["\']?([A-Za-z_][\w.]*)', spec_yaml_text))
    bundled = _bundle_modules(names, out_dir)
    if bundled:
        _common.info(f"bundled callable modules → code/: {', '.join(bundled)}")
    return bool(bundled)


def _bundle_modules(names, out_dir: Path, *, seen: set[str] | None = None) -> list[str]:
    """Copy each LOCAL ``.py`` module in *names* (and its transitive local imports) into the kit's ``code/``; return the bundled filenames.

    A local module (not under ``site-packages``/``dist-packages`` — those are provisioned via ``requirements``) is a study helper that must travel with the kit. *seen* threads across calls so a study bundles its experiment callables and its figure ``code_modules`` into one ``code/`` without copying a shared helper twice.
    """
    import importlib
    import shutil

    seen = seen if seen is not None else set()
    bundled: list[str] = []
    code_dir = out_dir / "code"

    def _copy(name: str, f: str) -> None:
        code_dir.mkdir(exist_ok=True)
        shutil.copy2(f, code_dir / Path(f).name)
        bundled.append(Path(f).name)
        seen.add(name)

    for name in sorted(set(names)):
        if name in seen:
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        f = getattr(mod, "__file__", None) or ""
        if not f.endswith(".py") or "site-packages" in f or "dist-packages" in f:
            continue  # installed package — comes from the emitted requirements, not bundled
        _copy(name, f)
        # A callable often imports a LOCAL helper of its own (e.g. Koller's wave_detection_methods, pulled in via a runtime sys.path insert) — not a `module:` reference and not an installed package, so nothing else carries it and the kit is not self-contained. Follow those transitive local, standalone imports too.
        for dep_name, dep_f in _local_module_deps(mod, seen):
            _copy(dep_name, dep_f)
    return bundled


def _local_module_deps(mod, seen: set[str]):
    """Yield (name, file) for the LOCAL, STANDALONE modules ``mod`` transitively imports.

    Resolves each imported name through ``sys.modules`` (``mod`` is already imported, so its dependencies are populated) and keeps only those backed by a plain local ``.py`` — NOT an installed package (``site-packages``), NOT a package with an ``__init__`` (those ship via requirements), and NOT ``tvbo`` itself. This is exactly the shape of a recipe's own helper module, so following it makes the emitted kit self-contained without vendoring the helper into the recipe tree.
    """
    import ast
    import sys

    src = getattr(mod, "__file__", None)
    if not src or not src.endswith(".py"):
        return
    try:
        tree = ast.parse(Path(src).read_text(encoding="utf-8"))
    except Exception:
        return
    # Everything under the interpreter's own roots is stdlib or an installed dep (both live in the Python install / its site-packages); a recipe's helper lives in the study tree, outside them. That single check excludes the whole stdlib AND every installed package, so only genuine local helpers survive.
    py_roots = tuple(str(Path(p).resolve()) for p in {sys.base_prefix, sys.prefix, sys.exec_prefix})
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    for name in sorted(imported):
        if name in seen or name == "tvbo" or name in sys.stdlib_module_names:
            continue
        dep = sys.modules.get(name)
        f = getattr(dep, "__file__", None) if dep is not None else None
        if not f or not f.endswith(".py") or Path(f).name == "__init__.py":
            continue  # missing, a C-extension, or a package (provisioned via requirements)
        rf = str(Path(f).resolve())
        if rf.startswith(py_roots) or "site-packages" in rf or "dist-packages" in rf:
            continue  # stdlib or installed — not a local recipe helper
        seen.add(name)
        yield name, f
        yield from _local_module_deps(dep, seen)


def _parse_bundle_select(items: list[str]) -> dict[str, str]:
    """Parse ``--bundle-select atlas=HCPMMP1`` entries into a BIDS-entity dict.

    Each key is a BIDS entity as it appears in the target filename (``atlas``, ``desc``, ``cohort``, ``tpl`` …) or ``suffix``; the pairs pin exactly which per-subject file a bundle copies when a subject directory holds several variants.
    """
    out: dict[str, str] = {}
    for raw in items:
        s = raw.lstrip("-")
        if "=" not in s:
            raise typer.BadParameter(f"--bundle-select {raw!r} must be KEY=VALUE (e.g. atlas=HCPMMP1).")
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip()
    return out


def _bundle_selection(experiment, cli_select: dict | None) -> dict | None:
    """Resolve whether to bundle this experiment's dataset, and with what selection.

    Bundling is requested either on the command line (``--bundle-dataset``, which passes at least ``{}``) or declaratively in the recipe (``dataset.bundle: true``).
    The metadata flag makes a self-contained kit the recipe's own intent, so the packaging command needs no bundle flag. Returns the entity-override dict to pass to :func:`_bundle_dataset` (``{}`` = resolve purely from the observation's BIDS query), or ``None`` when neither source requests a bundle.
    """
    if cli_select is not None:
        return cli_select
    ds = getattr(experiment, "dataset", None)
    return {} if (ds is not None and getattr(ds, "bundle", None)) else None


def _bundle_dataset(experiment, dest_dir: Path, entity_overrides: dict | None) -> str | None:
    """Copy the fan-out's per-subject dataset files into the kit; return the new root.

    Resolves each enumerated subject's empirical target (sidecar + payload) through the experiment's dataset query — tightened by *entity_overrides* — and copies it under
    *dest_dir* as ``sub-<id>/<file>``, so a kit carries exactly the data its fan-out
    consumes and nothing else. *dest_dir* is a sibling of the frozen spec, so its bare name is the relative ``dataset.bids_root`` to record. Returns that name, or None when there is no dataset-sourced target to bundle.

    A *requested* bundle that cannot be resolved (a missing file, an over-tight ``--bundle-select``) is a hard error: silently keeping the machine-specific root would ship a kit that fails on every node — the exact hazard this removes.
    """
    import shutil

    try:
        manifest = experiment.dataset_bundle_files(entity_overrides)
    except (FileNotFoundError, ValueError) as exc:
        _common.die(f"--bundle-dataset: {exc}")
    if not manifest:
        _common.warn("--bundle-dataset: experiment has no dataset-sourced target to bundle.")
        return None
    n_files = n_bytes = 0
    for subject, files in manifest.items():
        subdir = dest_dir / f"sub-{subject}"
        subdir.mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = subdir / f.name
            shutil.copytree(f, dst, dirs_exist_ok=True) if f.is_dir() else shutil.copy2(f, dst)
            n_files += 1
            n_bytes += sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) if dst.is_dir() else dst.stat().st_size
    _common.info(
        f"bundled dataset: {len(manifest)} subject(s), {n_files} file(s), "
        f"{n_bytes / 1e6:.1f} MB → {dest_dir.name}/ "
        f"(dataset.bids_root rewritten to relative '{dest_dir.name}')"
    )
    return dest_dir.name


def _emit_kit(*, engine: str, plan, experiment, out_dir: Path, bundle_select: dict | None = None) -> Path:
    """Write a self-contained reproducibility kit under *out_dir*.

    Layout::

        out_dir/
          <artefact>            # Snakefile / run.sbatch / main.nf
          scripts/<exp>.<ext>   # frozen backend code from experiment.render(backend)
          spec/<exp>.yaml       # frozen YAML snapshot of the experiment
          README.md             # provenance + how-to-run

    The slurm array shards a sweep and lets the backend vectorize each shard (``--shard``); it has NO per-cell ``--pin`` fan-out. An experiment that EXPLICITLY declares ``distribute.workflow`` over model/coupling parameters asked for per-cell fan-out (e.g. a non-jittable host observation computed once per cell) — slurm would silently vectorize it, tracing that host observation inside the vmap (TracerArrayConversionError). Such an experiment is rejected here with a pointer to ``tvbo workflow snakemake``, which fans one ``--pin`` per cell (see ``_emit_snakemake_study``'s fan-out note).
    """
    from tvbo import export as _export

    # Explicit per-cell distribute.workflow fan-out is snakemake-only (see docstring).
    _explicit_wf = set((plan.workflow_spec.get("distribute") or {}).get("workflow") or [])
    _fanned_params = [
        ax
        for ax in plan.workflow_axes
        if ax.kind == "parameters" and (ax.parameter in _explicit_wf or ax.name in _explicit_wf)
    ]
    if engine == "slurm" and _fanned_params:
        _axes = ", ".join(ax.parameter for ax in _fanned_params)
        _common.die(
            f"Experiment {plan.experiment_key!r} declares distribute.workflow over parameter "
            f"axis(es) [{_axes}]: a per-cell fan-out the slurm emitter cannot serve — it shards + "
            f"vectorizes, so a non-jittable per-cell observation (e.g. a host wave metric) traces "
            f"inside the vmap and crashes. Emit with `tvbo workflow snakemake` (one --pin per cell)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scripts").mkdir(exist_ok=True)
    (out_dir / "spec").mkdir(exist_ok=True)

    spec_dir = out_dir / "spec"
    # Before the error-swallowing spec freeze, so a bundling failure is a hard error the user sees.
    _sel = _bundle_selection(experiment, bundle_select)
    bundle_root = _bundle_dataset(experiment, spec_dir / "dataset", _sel) if _sel is not None else None
    spec_relpath = None
    bundled_code = False
    try:
        yaml_text = _freeze_spec_yaml(experiment, spec_dir, workflow_spec=plan.workflow_spec, dataset_bids_root=bundle_root)
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
        _staged = _bundle_script_constants(code, out_dir)
        if _staged:
            _common.info(f"staged {_staged} producer constant(s) → constants/")
    except Exception as exc:
        _common.info(f"(could not render backend script for {plan.backend.name!r}: {exc})")

    # 3) Workflow artefact
    artefact = out_dir / _ARTEFACT_NAME[engine]
    # BIDS result stem (no subject) so the engine can declare the exact output a per-subject `tvbo run` writes (sub-<subject>_<stem>.h5), not a bare result.h5.
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

    # A single-task array writes the canonical result directly, so it needs no gather job.
    if engine == "slurm" and plan.n_array_tasks > 1:
        # BIDS-style result stem (pybids), matching what a local ExperimentResult.save writes.
        try:
            result_stem = experiment.get_result_stem()
        except Exception:
            result_stem = "result"
        finalize = out_dir / "finalize.sbatch"
        finalize.write_text(
            _render_template("slurm/finalize.sbatch.mako", plan=plan, block=plan.engine_block, result_stem=result_stem),
            encoding="utf-8",
        )
        _common.info(f"wrote {finalize.relative_to(out_dir)}")

    # 3b) Environment files (pip + conda) rendered via Mako from the experiment's declared environment.requirements, so the kit provisions the right env.
    if plan.pip_specs:
        (out_dir / "requirements.txt").write_text(_render_template("requirements.txt.mako", plan=plan), encoding="utf-8")
        (out_dir / "environment.yml").write_text(_render_template("environment.yml.mako", plan=plan), encoding="utf-8")
        _common.info("wrote requirements.txt + environment.yml")
    # 3c) When a container AND requirements are both declared, emit a one-time setup.sh that layers the declared deps onto the base image (see needs_container_layer) — so a study adds `igl` without rebuilding the SIF. Engine-independent: the Slurm run.sbatch and every Snakemake rule both prepend the layer to PYTHONPATH.
    if plan.needs_container_layer:
        setup = out_dir / "setup.sh"
        setup.write_text(_render_template("setup.sh.mako", plan=plan), encoding="utf-8")
        setup.chmod(0o755)
        _common.info("wrote setup.sh (layers declared requirements onto the container)")

    # 4) README
    _write_readme(
        out_dir,
        engine=engine,
        plans=[plan],
        script_relpath=str(script_path.relative_to(out_dir)) if script_path else None,
        spec_layout=spec_relpath or f"spec/{plan.experiment_key}.yaml",
    )
    return out_dir


def _write_readme(out_dir: Path, *, engine: str, plans, script_relpath: str | None, spec_layout: str) -> None:
    """Write the kit's README covering every experiment frozen into it.

    *plans* holds one plan per frozen experiment, and *spec_layout* is where the
    emitter actually put the frozen specs — ``spec/<key>.yaml`` for a one-file kit, ``spec/<experiment>/experiment.yaml`` for the per-experiment directories the snakemake emitter writes (it uses those for a single experiment too, so the layout follows the emitter, not the plan count). Provenance is summed over the whole list, so a study kit reports its real totals rather than whichever experiment happened to be frozen last.
    """
    plans = list(plans)
    text = _render_template(
        "readme.md.mako",
        engine=engine,
        artefact=_ARTEFACT_NAME[engine],
        plans=plans,
        head=plans[0],
        study=len(plans) > 1,
        script_relpath=script_relpath,
        spec_layout=spec_layout,
    )
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# Commands


@app.command("plan", help="Show the resolved workflow plan (no artefact emitted).")
def plan_cmd(
    spec: str = typer.Argument(..., help="Path, CURIE, or DB name (Study or Experiment)."),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Execution backend; default: the experiment's declared execution.backend, else tvboptim.",
    ),
    engine: str = typer.Option("local", "--engine", "-e"),
    experiment: str = typer.Option(None, "--experiment"),
    override: list[str] = typer.Option(
        [], "--set", help="Override a workflow spec key, e.g. ``--set slurm.account=foo`` (repeatable)."
    ),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Show the resolved workflow plan for *spec* without emitting any artefacts.

    Reports the chosen study/experiment, backend, engine, vectorized vs workflow-fanned axes, total cell count, chunking, and applied overrides.
    """
    study, built = _build_plans(spec, engine=engine, backend=backend, experiment=experiment, overrides=override)
    plans = [p for p, _ in built]

    if json:
        import json as _json

        payloads = [_plan_payload(p) for p in plans]
        typer.echo(_json.dumps(payloads[0] if len(payloads) == 1 else payloads, default=str, indent=2))
        return

    multi = len(plans) > 1
    scope = "whole study" if experiment is None else "selected"
    if multi:
        typer.echo(f"study      : {plans[0].study_key}   ({len(plans)} experiments — {scope})\n")
    total_cells = total_tasks = 0
    for plan in plans:
        if multi:
            bar = "─" * max(3, 46 - len(str(plan.experiment_key)))
            typer.echo(f"── experiment {plan.experiment_key} {bar}")
        _print_plan_block(plan, show_study=not multi)
        total_cells += plan.n_workflow_cells or 0
        total_tasks += plan.n_array_tasks or 0
        if multi:
            typer.echo("")
    if multi:
        typer.echo(f"total ({scope}) : {len(plans)} experiments, {total_cells} workflow cells, {total_tasks} array task(s)")


def _study_experiments(spec: str, experiment: str | None):
    """Resolve the runtime experiments a study/experiment SPEC fans out over.

    Returns ``(study_or_none, [runtime_experiments], study_key)``. A study yields all its experiments (or the ``--experiment`` id/label subset); a bare experiment SPEC yields a single-item list.
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
            sel = getattr(e, "id", None) or getattr(e, "key", None) or getattr(e, "name", None) or getattr(e, "label", None)
            try:
                e = obj.get_experiment(sel)
            except Exception as exc:
                _common.die(f"Could not resolve experiment {sel!r} to a runnable object: {exc}")
        resolved.append(e)
    return obj, resolved, getattr(obj, "key", None) or "study"


def _study_figures(study) -> list:
    """The study's ``Figure`` objects as a list, or ``[]`` when it declares none.

    A bare-experiment SPEC (``study is None``) never carries figures, so the figure emission is skipped and existing experiment-rule emission is untouched.
    """
    if study is None:
        return []
    from tvbo.utils import as_list

    return as_list(getattr(study, "figures", None))


def _figure_code_modules(figs) -> set[str]:
    """The ``code_modules`` every figure declares — modules whose import registers the figures' custom panels/transforms. Bundled into the kit's ``code/`` so a figure's ``plot.py`` can ``import`` them on a compute node (they are local study helpers, not installed packages)."""
    names: set[str] = set()
    for fig in figs:
        names.update(str(m) for m in (getattr(fig, "code_modules", None) or []))
    return names


def _figure_base_dir(study, out_dir: Path) -> str:
    """Root the figures' ``used`` containers resolve against.

    Prefers the study's source-file directory (where ``output/nc/`` lives at author time); falls back to the kit dir. ``bsplot`` resolves each layer's
    IRI to ``<base>/output/nc/<exp>/*.h5``.
    """
    src = getattr(study, "_source_file", None)
    return str(Path(src).parent) if src else str(out_dir)


def _bundle_script_constants(code: str, out_dir: Path) -> int:
    """Copy every producer/sourced constant a frozen backend script loads into the kit's ``constants/`` dir, so ``--rendered`` execution finds it on a node that lacks the author's ``~/.tvbo/constants``. The rendered ``_load_constant`` resolves a missing absolute path by basename against ``$TVBO_CONSTANTS_DIR`` or the run dir's ``constants/`` (see the observation template), so a frozen kit carries its operators with it. Returns the number staged."""
    import re
    import shutil

    dest = out_dir / "constants"
    staged: dict[str, Path] = {}
    missing: list[str] = []
    for p in sorted(set(re.findall(r'_load_constant\(\s*["\']([^"\']+)["\']', code))):
        src = Path(p)
        if not src.is_file():
            missing.append(p)
            continue
        # The rendered `_load_constant` resolves a missing path BY BASENAME, so two constants sharing one would fit against whichever was copied last.
        prior = staged.get(src.name)
        if prior is not None and prior != src.resolve():
            _common.die(
                f"two constants are named {src.name!r} but come from different files:\n"
                f"  - {prior}\n  - {src.resolve()}\n"
                "The kit resolves them by basename, so one would silently stand in for "
                "the other. Rename one, or point both at the same file."
            )
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / src.name)
        staged[src.name] = src.resolve()
    if missing:
        _common.warn(
            "these constants were not found on this machine and are NOT in the kit; the "
            "job will fail on the node unless they exist there:\n  - " + "\n  - ".join(missing)
        )
    return len(staged)


def _freeze_backend_script(experiment, out_dir: Path, backend_name: str, key: str) -> str | None:
    """Freeze *experiment*'s pre-rendered backend script under ``out_dir/scripts/<key>.<ext>``.

    Mirrors the single-experiment kit's script freeze (`_emit_kit` step 2): the rendered tvboptim/jax/… code imports only stable tvbo runtime modules (never codegen), so a rule can execute it as-is via ``tvbo run … --rendered`` — no code generation on the node.
    Returns the kit-relative path, or ``None`` when the render fails, in which case the rule falls back to re-rendering from the frozen spec.
    """
    from tvbo import export as _export

    try:
        fmt = _export.resolve(backend_name)
        ext = (fmt.extension or ".py").lstrip(".")
        code = experiment.render(format=backend_name)
        scripts_dir = out_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        path = scripts_dir / f"{key}.{ext}"
        path.write_text(code, encoding="utf-8")
        _bundle_script_constants(code, out_dir)
        rel = str(path.relative_to(out_dir))
        _common.info(f"wrote {rel}")
        return rel
    except Exception as exc:
        _common.info(f"(could not render backend script for {backend_name!r}: {exc})")
        return None


def _emit_snakemake_study(
    *,
    spec: str,
    backend: str,
    experiment: str | None,
    output: Path | None,
    override: list[str],
    stdout: bool = False,
    bundle_select: dict | None = None,
    code_source: str = "spec",
):
    """Emit one Snakefile that fans every experiment (and, per experiment, every subject / sweep cell) into its own job. In kit mode each experiment is frozen
    BOTH as a self-contained ``spec/<key>/experiment.yaml`` (re-rendered at run time)
    AND as a pre-rendered ``scripts/<key>.<ext>`` (run as-is, no codegen). Each rule picks between them at run time from ``$TVBO_CODE_SOURCE`` (default *code_source*, ``'spec'`` for back-compat), so ONE kit runs either way; ``--stdout`` writes nothing and its rules run ``tvbo run <source-spec> --experiment <id>``.

    Fan-out note: an experiment that fans a ``parameters`` axis over the workflow (one ``--pin`` per cell, e.g. a per-cell host observation) is emitted spec-mode ONLY, with
    NO frozen script. A frozen script bakes the model/coupling/network parameters at a single point and hardcodes the whole grid, so the per-cell ``--pin`` can never reach them — pins collapse the exploration on the experiment OBJECT, which only a spec-mode re-render reads. Skipping the (invalid) frozen script also means a run forcing ``--code-source frozen`` still falls back to spec for these rules. Subject / seed / IC fans keep their frozen script: their per-cell value reaches the frozen run at call time (``--subject``, seed / initial-condition kwargs)."""
    study, experiments, study_key = _study_experiments(spec, experiment)
    out_dir = output or Path("output").joinpath(str(study_key), "snakemake")
    if not stdout:
        out_dir.mkdir(parents=True, exist_ok=True)
    parsed = _parse_overrides(override)

    def _san(s):
        return "".join(c if (c.isalnum() or c in ".-") else "_" for c in str(s))

    # Every experiment identifier (id / key / name) -> its sanitized workflow key, so a from_experiment dependency (recorded by id in plan.depends_on) resolves to the source experiment's rule and output dir even when that experiment carries an explicit ``key`` that differs from its id.
    _key_of = {}
    for _e in experiments:
        _k = _san(_common.experiment_key(_e))
        for _ref in (getattr(_e, "id", None), getattr(_e, "key", None), getattr(_e, "name", None)):
            if _ref is not None:
                _key_of[str(_ref)] = _k

    exp_plans, block, plans, bundled_code = [], {}, [], False
    for exp in experiments:
        key = _san(_common.experiment_key(exp))
        base = _wf.merge_workflow_spec(study, exp)
        spec_dict = _deep_merge(base, parsed["merged"])
        # A run modifier, not a workflow-block field, so it is popped before the plan and freeze.
        _max_iter = spec_dict.pop("max_iterations", None)
        if spec_dict.pop("smoke", False) and _max_iter is None:
            _max_iter = 1
        # Engine-native benchmarking: each rule carries Snakemake's `benchmark:` directive (near-zero-overhead resource TSV). ON by default; --no-benchmark / --set benchmark=false opts out. A run modifier, not a workflow-block field, so pop it before the plan/freeze.
        _benchmark = bool(spec_dict.pop("benchmark", True))
        plan = _wf.plan(
            study_key=str(study_key),
            experiment=exp,
            backend=backend,
            engine="snakemake",
            workflow_spec=spec_dict,
            overrides=parsed["records"],
            source_spec=spec,
            experiment_selector=key,
        )
        # Study-level block for the shipped profile: the cluster identity (partition/account) is a property of the run, not of one experiment, so take the first experiment that declares one — matching how the Snakefile's global `container:` keys off exp_plans[0]. Per-rule resources come from each plan's own block (see exp_plans below).
        block = block or (plan.engine_block or {})
        plans.append(plan)
        try:
            result_stem = exp.get_result_stem()
        except Exception:
            result_stem = "result"
        # A fanned `parameters` sweep can't be frozen (see the docstring's fan-out note).
        _fanned_parameter = any(ax.kind == "parameters" for ax in plan.workflow_axes)
        scripts_relpath = None
        if stdout:
            spec_relpath, select = spec, key
        else:
            edir = out_dir / "spec" / key
            edir.mkdir(parents=True, exist_ok=True)  # non-connectome freeze doesn't create it
            _sel = _bundle_selection(exp, bundle_select)
            bundle_root = _bundle_dataset(exp, edir / "dataset", _sel) if _sel is not None else None
            yaml_text = _freeze_spec_yaml(exp, edir, workflow_spec=spec_dict, dataset_bids_root=bundle_root)
            (edir / "experiment.yaml").write_text(yaml_text, encoding="utf-8")
            # Custom callable/builder modules the recipe references travel with the kit (shared code/ dir), so `tvbo run` resolves them on the node.
            bundled_code = _bundle_callable_modules(yaml_text, out_dir) or bundled_code
            spec_relpath, select = f"spec/{key}/experiment.yaml", None
            # Freeze the pre-rendered backend script ALONGSIDE the spec, so the SAME kit runs either way: `--code-source frozen` runs `scripts/<key>.<ext>` with no codegen on the node. A render failure is non-fatal — the spec path still works; the rule falls back to it when the script is absent.
            if not _fanned_parameter:
                scripts_relpath = _freeze_backend_script(exp, out_dir, plan.backend.name, key)
                _common.info(f"froze experiment {key} ({len(plan.workflow_axes)} fan-out axes)")
            else:
                _common.info(
                    f"experiment {key}: {len(plan.workflow_axes)} fanned parameter axis(es) → spec-mode per cell (no frozen script)"
                )
        exp_plans.append(
            {
                "key": key,
                "rule_name": "exp_" + key.replace("-", "_").replace(".", "_"),
                "spec_relpath": spec_relpath,
                # Pre-rendered backend script (frozen alongside the spec); None in --stdout mode or if the render failed, in which case the rule always uses the spec.
                "scripts_relpath": scripts_relpath,
                # Emit-time default code source baked into the rule (overridable at run time via $TVBO_CODE_SOURCE); a fanned-parameter experiment is spec-only.
                "code_source": "spec" if _fanned_parameter else code_source,
                "select": select,
                # The plan resolves an unset backend to the experiment's execution.backend (else tvboptim); emit that resolved name, never the raw None — otherwise the rule renders `--backend=None` and every cell dies at backend resolution.
                "backend": plan.backend.name,
                # Smoke iteration cap threaded to the rule's `tvbo run --max-iterations` (None => uncapped).
                "max_iterations": _max_iter,
                # Engine-native benchmarking: attach Snakemake's `benchmark:` directive to the rule.
                "benchmark": _benchmark,
                "out_dir": plan.out_dir,
                "result_stem": result_stem,
                "container": plan.container,
                # Whether this rule's `tvbo run` must prepend the requirements venv (setup.sh built it — native, or on the image) to PYTHONPATH — see needs_env_layer.
                "needs_env_layer": plan.needs_env_layer,
                "extras_venv": plan.container_extras_venv,
                # Resources are declared per experiment (a per-subject fit and a group analysis need different walltime/memory), so each rule carries its own block rather than sharing one study-wide dict.
                "block": plan.engine_block or {},
                # workflow.retries: re-run a failed cell N times, each attempt raising host time/mem (CPU) or shrinking the on-device batch (GPU).
                "retries": int(getattr(plan, "retries", 0) or 0),
                "axes": [{"name": ax.name, "parameter": ax.parameter, "values": list(ax.values)} for ax in plan.workflow_axes],
                # on_device cohort: the subjects this single job produces one result each for.
                "cohort_subjects": list(plan.cohort_subjects),
                "cohort_result_files": list(plan.cohort_result_files),
                "depends_on": [_key_of.get(str(d), _san(str(d))) for d in plan.depends_on],
            }
        )

    # Figures are resolved BEFORE the Snakefile renders: their outputs join the default target (so `tvbo workflow submit` renders them right after the grid) and their custom-panel code_modules bundle into code/ (so plot.py imports resolve on a node).
    figs = _study_figures(study)
    fig_base = _figure_base_dir(study, out_dir)
    # Figures inherit the study workflow WITH the `--set` overrides merged in (same effective config the experiment rules get), so a `--set slurm.venv=…`/partition/etc. reaches the render rule too — otherwise the figure runs in the wrong (system) interpreter.
    fig_workflow = _deep_merge(_wf._as_plain_dict(getattr(study, "workflow", None)), parsed["merged"] or {})
    figure_outputs: list[str] = []
    if figs:
        from tvbo.adapters import figure_workflow

        if not stdout:
            fig_mods = _figure_code_modules(figs)
            fig_bundled = _bundle_modules(fig_mods, out_dir) if fig_mods else []
            if fig_bundled:
                bundled_code = True
                _common.info(f"bundled figure modules → code/: {', '.join(fig_bundled)}")
        fig_ctxs = figure_workflow.figure_contexts(
            figs, base_dir=fig_base, workflow=fig_workflow, exp_plans=exp_plans, bundled_code=bundled_code
        )
        figure_outputs = [c["output"] for c in fig_ctxs]

    text = _render_template(
        "snakemake/study.smk.mako", exp_plans=exp_plans, block=block, bundled_code=bundled_code, figure_outputs=figure_outputs
    )
    if stdout:
        typer.echo(text)
        if figs:
            typer.echo(
                figure_workflow.emit_figure_rules(
                    figs,
                    base_dir=fig_base,
                    workflow=fig_workflow,
                    include_all=True,
                    exp_plans=exp_plans,
                    bundled_code=bundled_code,
                )
            )
        return None
    (out_dir / "Snakefile").write_text(text, encoding="utf-8")
    _common.info(f"wrote Snakefile ({len(exp_plans)} experiment rule(s))")
    # The study path builds its own artefacts rather than going through _emit_kit, so this mirrors it.
    _kit0 = plans[0] if plans else None
    if _kit0 is not None and _kit0.pip_specs:
        (out_dir / "requirements.txt").write_text(_render_template("requirements.txt.mako", plan=_kit0), encoding="utf-8")
        (out_dir / "environment.yml").write_text(_render_template("environment.yml.mako", plan=_kit0), encoding="utf-8")
        _common.info("wrote requirements.txt + environment.yml")
    if _kit0 is not None and _kit0.needs_env_layer:
        setup = out_dir / "setup.sh"
        setup.write_text(_render_template("setup.sh.mako", plan=_kit0), encoding="utf-8")
        setup.chmod(0o755)
        _common.info("wrote setup.sh (provisions declared requirements into a venv)")
    # Match the Snakefile's global `container:` directive (keyed on the first experiment): when it is emitted, enable Apptainer in the profile so the run needs no extra flag; when it is not, leave the profile container-free.
    _kit_plan = plans[0] if plans else None
    # A differing per-experiment image becomes that rule's own `container:` directive, but binds reach Apptainer through Snakemake's single `--apptainer-args`, which is per-run and cannot vary per rule. Say so rather than drop it silently: a task missing a bind fails at import time, far from the declaration that was ignored.
    if _kit_plan is not None:
        _divergent = sorted({p.experiment_key for p in plans[1:] if p.container_exec_flags != _kit_plan.container_exec_flags})
        if _divergent:
            _common.warn(
                f"experiments {', '.join(_divergent)} declare container_binds/container_args "
                f"differing from {_kit_plan.experiment_key}'s; Snakemake applies one "
                f"--apptainer-args per run, so {_kit_plan.experiment_key}'s are used for all "
                "rules. Emit those experiments as their own kit if they need different binds."
            )
    _write_snakemake_profile(out_dir, block, plan=_kit_plan)
    if plans:
        _write_readme(
            out_dir, engine="snakemake", plans=plans, script_relpath=None, spec_layout="spec/<experiment>/experiment.yaml"
        )
    if figs:
        # A study is its experiments PLUS the figures that read their results: freeze each figure's self-contained plot.py + the render rules, then wire them in. The figure outputs were already added to `rule all` (see figure_outputs above), so the default target renders them right after the grid — a fanned figure's `input:` is the expand() over its experiment's cells, so it waits for the whole sweep.
        figure_workflow.write_figure_kit(
            figs,
            base_dir=fig_base,
            out_dir=out_dir,
            workflow=fig_workflow,
            include_all=True,
            exp_plans=exp_plans,
            bundled_code=bundled_code,
        )
        with (out_dir / "Snakefile").open("a", encoding="utf-8") as fh:
            fh.write('\n\ninclude: "figures.smk"\n')
        _common.info(f"wrote figures.smk + {len(figs)} figure plot script(s) (run: `snakemake all_figures`)")
    return out_dir


def _write_snakemake_profile(out_dir: Path, block: dict, plan=None) -> None:
    """Ship a SLURM profile so the kit runs from a login node with one command.

    Snakemake 8+/9 submits to the scheduler via an executor plugin: the lightweight ``snakemake`` process runs on the login node and dispatches each rule as its own
    SLURM job (with the per-rule ``resources:`` in the Snakefile). The profile carries the compute-environment settings — ``executor: slurm``, the concurrent-jobs cap, and the cluster-identity default-resources (partition/account) that don't belong in the workflow definition. Run: ``snakemake --profile profile`` from the kit dir.
    """
    _container_args = getattr(plan, "container_exec_flags", "") or ""
    text = _render_template(
        "snakemake/profile.yaml.mako",
        jobs=100,
        container=getattr(plan, "container", None),
        container_args=_container_args,
        # YAML 1.2 is a JSON superset, so a JSON string literal is always a valid — and correctly escaped — YAML scalar. container_args is free-form, so it may contain the quote that would otherwise terminate the scalar early.
        container_args_yaml=json.dumps(_container_args),
        partition=block.get("partition"),
        account=block.get("account"),
        retries=1,
    )
    prof = out_dir / "profile"
    prof.mkdir(parents=True, exist_ok=True)
    (prof / "config.yaml").write_text(text, encoding="utf-8")
    _common.info("wrote profile/config.yaml (SLURM executor — `snakemake --profile profile`)")


def _pack_kit(out_dir: Path) -> Path:
    """Archive the kit into ``<out_dir>.tar.gz`` and remove the loose directory.

    The tarball IS the shippable artifact: ``tvbo workflow submit <archive>`` (and any run) re-extracts it, so keeping the uncompressed directory beside it is pure clutter. The archive holds the kit directory at top level, so submit extracts and runs it directly (see :func:`_resolve_kit_dir`). Emit without ``--pack`` when the loose directory is what you want (e.g. to ``sbatch`` it in place).
    """
    import shutil

    out_dir = Path(out_dir)
    archive = shutil.make_archive(str(out_dir), "gztar", root_dir=str(out_dir.parent), base_dir=out_dir.name)
    shutil.rmtree(out_dir)
    _common.info(f"packed {Path(archive).name} (removed loose {out_dir.name}/)")
    return Path(archive)


def _warn_machine_specific_bids_root(out_dir: Path) -> None:
    """Warn when a kit bakes an absolute ``dataset.bids_root`` into its frozen spec.

    A reproducibility kit is meant to travel. A per-subject dataset fan-out points ``dataset.bids_root`` at a data tree that is almost always machine-specific, so the baked absolute path will not resolve on the target host and every task fails to load its per-subject target. Surface it (engine-agnostic — read from the frozen ``spec/*.yaml``) with the exact override, so a packed kit is never shipped with a silently-wrong data root.
    """
    import re

    spec_dir = out_dir / "spec"
    if not spec_dir.is_dir():
        return
    seen: set[str] = set()
    for spec_file in sorted(spec_dir.glob("*.yaml")):
        try:
            text = spec_file.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in re.finditer(r"^\s*bids_root:\s*(\S.*?)\s*$", text, re.MULTILINE):
            root = m.group(1).strip().strip("'\"")
            if root.startswith("/"):
                seen.add(root)
    for root in sorted(seen):
        _common.warn(
            f"kit bakes an absolute dataset.bids_root ({root}) — this data tree is "
            f"machine-specific; verify it exists on the target host (a per-subject "
            f"dataset fan-out fails to resolve each subject's target if it does not). "
            f"The slurm launcher reads it from $TVBO_BIDS_ROOT when set (just "
            f"`export TVBO_BIDS_ROOT=<cluster path>`); otherwise override at submit "
            f"time with `--set dataset.bids_root=<cluster path>`."
        )


def _finalize_kit(out_dir: Path, *, pack: bool) -> Path:
    """Warn on portability hazards, optionally pack, and return the artifact path.

    Returns the ``<kit>.tar.gz`` when *pack* (the loose dir is removed), else the kit directory — so the caller always gets a path that exists.
    """
    _warn_machine_specific_bids_root(out_dir)
    return _pack_kit(out_dir) if pack else out_dir


def _emit(
    engine: str,
    *,
    spec: str,
    backend: str,
    experiment: str | None,
    output: Path | None,
    override: list[str],
    stdout: bool,
    pack: bool = False,
    bundle_select: dict | None = None,
    code_source: str = "spec",
) -> None:
    if engine == "snakemake":
        out_dir = _emit_snakemake_study(
            spec=spec,
            backend=backend,
            experiment=experiment,
            output=output,
            override=override,
            stdout=stdout,
            bundle_select=bundle_select,
            code_source=code_source,
        )
        return _finalize_kit(out_dir, pack=pack) if out_dir is not None else None
    plan, exp = _build_plan(spec, engine=engine, backend=backend, experiment=experiment, overrides=override)
    if stdout:
        text = _render_template(_TEMPLATE_PATH[engine], plan=plan, block=plan.engine_block, script_relpath=None)
        typer.echo(text)
        return None
    if output:
        out_dir = output
    else:
        # A standalone experiment has study_key == experiment_key (same fallback), so collapse the redundant level: out/<experiment>/<engine> not …/x/x/….
        parts = [plan.experiment_key] if plan.study_key == plan.experiment_key else [plan.study_key, plan.experiment_key]
        out_dir = Path("output").joinpath(*parts, engine)
    _emit_kit(engine=engine, plan=plan, experiment=exp, out_dir=out_dir, bundle_select=bundle_select)
    return _finalize_kit(out_dir, pack=pack)


_LAUNCHER = {"slurm": "sbatch", "snakemake": "snakemake", "nextflow": "nextflow"}


def _resolve_launcher(name: str) -> str | None:
    """Find an engine launcher, preferring the environment tvbo itself runs in.

    ``snakemake`` is normally installed alongside ``tvbo`` in the same venv, and a cluster user runs the CLI by absolute path (``.venv/bin/tvbo …``) rather than activating it — which leaves that venv's ``bin`` off ``PATH``, so a bare
    :func:`shutil.which` misses a launcher sitting right next to the running interpreter. Look there first, then fall back to ``PATH``. Returns the resolved path, or ``None`` when the launcher genuinely is not installed.
    """
    import shutil
    import sys

    sibling = Path(sys.executable).parent / name
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which(name)


def _experiment_targets(kit_dir: Path, experiment: str) -> list:
    """Map a ``--experiment`` selector (``'41,50'``) to a study kit's Snakemake rule targets (``exp_41 exp_50``), so ONE full-study pack runs any subset of its experiments at submit time. Validated against the kit's Snakefile — a typo or an experiment not in the pack fails here, not with an opaque Snakemake ``MissingRuleException`` mid-run."""
    import re

    snakefile = kit_dir / _ARTEFACT_NAME["snakemake"]
    if not snakefile.is_file():
        _common.die(f"--experiment needs the kit's {snakefile.name} to validate against, and {kit_dir} has none.")
    rules = set(re.findall(r"^rule\s+(exp_[A-Za-z0-9_]+)\s*:", snakefile.read_text(), re.M))
    targets, missing = [], []
    for tok in (t.strip() for t in str(experiment).split(",") if t.strip()):
        rule = "exp_" + re.sub(r"[^0-9A-Za-z]", "_", tok)
        (targets if rule in rules else missing).append(rule)
    if missing:
        _common.die(
            f"--experiment: the kit has no rule for "
            f"{[r[len('exp_') :] for r in missing]}; its experiments are "
            f"{sorted(r[len('exp_') :] for r in rules)}."
        )
    return targets


def _execute_engine_artefact(
    engine: str,
    artefact: Path,
    *,
    slurm_array: str | None = None,
    dry_run: bool = False,
    profile: str | None = None,
    cores: str | None = None,
    code_source: str | None = None,
    experiment: str | None = None,
) -> None:
    """Submit/execute a rendered workflow artefact for *engine*.

    Runs from the artefact's own directory so the generated script can use the relative ``spec/`` and ``scripts/`` paths of an emitted kit. *slurm_array* restricts a Slurm submission to an index or range (``'0'`` for a single smoke task, ``'0-3'`` for four); ignored for non-Slurm engines. *dry_run* asks the engine to resolve and report the work without running or queueing it — each engine spells that differently, so it maps to the engine's own flag.
    *profile* (Snakemake only) overrides the kit's shipped ``profile/`` with a named
    or path profile — e.g. a site profile like ``cubi-v1`` that carries the cluster's canonical executor config; the Snakefile's per-rule ``resources:`` apply on top of whichever profile is used. *cores* (Snakemake only) forces a
    LOCAL run on that many cores (``'all'`` for every core), overriding only the profile's executor — its container/bind/retry settings still apply — the native local-testing path; the SAME kit submits to the scheduler on HPC when *cores* is unset.
    """
    # Resolve to an absolute launcher so a venv-installed console script is found even when that venv's bin/ is not on PATH (see :func:`_resolve_launcher`).
    exe = _resolve_launcher(_LAUNCHER.get(engine, "")) or _LAUNCHER.get(engine, "")
    if experiment and engine != "snakemake":
        _common.die(f"--experiment selects experiments from a snakemake study kit; this is a {engine!r} kit.")
    if engine == "slurm":
        cmd = [exe]
        if dry_run:
            cmd.append("--test-only")
        if slurm_array is not None:
            cmd.append(f"--array={slurm_array}")
        cmd.append(artefact.name)
    elif engine == "snakemake":
        has_scheduler = bool(_resolve_launcher("sbatch"))  # checks the venv sibling AND $PATH
        use_profile = profile or ("profile" if (artefact.parent / "profile" / "config.yaml").exists() else None)
        run_local = cores is not None or not has_scheduler
        if run_local and cores is None:
            _common.info("no SLURM scheduler found (no `sbatch`); running locally")
        cmd = [exe]
        if use_profile:
            cmd += ["--profile", use_profile]
        if run_local:
            cmd += ["--executor", "local"]
        if run_local or not use_profile:
            cmd += ["--cores", cores or "all"]
        if dry_run:
            cmd.append("--dry-run")
        if experiment:
            cmd += _experiment_targets(artefact.parent, experiment)
    elif engine == "nextflow":
        cmd = [exe, "run", artefact.name]
        if dry_run:
            cmd.append("-preview")
    else:
        _common.die(f"unsupported engine {engine!r}; expected {'|'.join(_ARTEFACT_NAME)}")
    # Select the frozen-vs-spec code source for this run by exporting TVBO_CODE_SOURCE into the engine's environment; each rule's shell reads it (default = the kit's emit-time default). Inherited by a local run and by any executor that forwards the environment.
    env = None
    if code_source is not None:
        env = {**os.environ, "TVBO_CODE_SOURCE": code_source}
        _common.info(f"code source: TVBO_CODE_SOURCE={code_source}")
    _common.info("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=artefact.parent, env=env)


def _execute_emitted(
    engine: str,
    out_dir: Path,
    *,
    slurm_array: str | None = None,
    dry_run: bool = False,
    profile: str | None = None,
    cores: str | None = None,
    code_source: str | None = None,
    experiment: str | None = None,
) -> None:
    """Execute a generated workflow artefact inside *out_dir*.

    For Slurm this submits the array job and then chains the gather job (``finalize.sbatch``) with an ``afterok`` dependency, so the run converges to one reassembled result with no manual step. With *dry_run* nothing is queued:
    the engine only reports the work it would do, so the Slurm chain is skipped (there is no array job id to depend on). *profile* overrides the Snakemake profile and *cores* forces a local Snakemake run (see
    :func:`_execute_engine_artefact`).
    """
    if engine == "slurm" and not dry_run:
        # The chain submits the whole array; it has no per-experiment target, so an ignored selector would burn the study's allocation on work nobody asked for.
        if experiment:
            _common.die(f"--experiment selects experiments from a snakemake study kit; this is a {engine!r} kit.")
        _submit_slurm_chain(out_dir, slurm_array=slurm_array, code_source=code_source)
    else:
        _execute_engine_artefact(
            engine,
            out_dir / _ARTEFACT_NAME[engine],
            slurm_array=slurm_array,
            dry_run=dry_run,
            profile=profile,
            cores=cores,
            code_source=code_source,
            experiment=experiment,
        )


def _submit_slurm_chain(out_dir: Path, *, slurm_array: str | None = None, code_source: str | None = None) -> None:
    """Submit ``run.sbatch`` (array), then ``finalize.sbatch`` with a dependency.

    ``sbatch --parsable`` returns the array job id; the gather job is submitted ``--dependency=afterok`` on it and told where the shards landed via ``TVBO_SHARD_DIR``, so it reassembles them into one result once every task succeeds. When *code_source* is set it is exported into the submit environment as ``TVBO_CODE_SOURCE`` (``sbatch`` forwards it to the job via its default ``--export=ALL``), selecting the frozen-vs-spec code source per submission.
    """
    env = None
    cmd = ["sbatch", "--parsable"]
    if slurm_array is not None:
        cmd.append(f"--array={slurm_array}")
    if code_source is not None:
        cmd.append(f"--export=ALL,TVBO_CODE_SOURCE={code_source}")
        env = {**os.environ, "TVBO_CODE_SOURCE": code_source}
        _common.info(f"code source: TVBO_CODE_SOURCE={code_source}")
    cmd.append(_ARTEFACT_NAME["slurm"])
    _common.info("$ " + " ".join(shlex.quote(c) for c in cmd))
    res = subprocess.run(cmd, check=True, cwd=out_dir, capture_output=True, text=True, env=env)
    job_id = (res.stdout or "").strip().split(";")[0]
    _common.info(f"submitted array job {job_id}")

    finalize = out_dir / "finalize.sbatch"
    if finalize.exists() and job_id:
        base = job_id.split("_")[0]
        fcmd = ["sbatch", f"--dependency=afterok:{base}", f"--export=ALL,TVBO_SHARD_DIR=results/{base}", finalize.name]
        _common.info("$ " + " ".join(shlex.quote(c) for c in fcmd))
        subprocess.run(fcmd, check=True, cwd=out_dir)
        _common.info(f"submitted gather job (afterok:{base}) — one result when the array finishes")


def _detect_engine_from_kit(kit_dir: Path) -> str | None:
    """Infer an already-emitted kit's engine from its artefact file.

    A kit carries exactly one engine artefact (``run.sbatch`` / ``Snakefile`` / ``main.nf``); its presence names the engine to submit with.
    """
    for engine, artefact in _ARTEFACT_NAME.items():
        if (kit_dir / artefact).exists():
            return engine
    return None


@app.command("slurm", help="Emit a self-contained sbatch kit (artefact + scripts + spec).")
def slurm(
    spec: str = typer.Argument(...),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Execution backend; default: the experiment's declared execution.backend, else tvboptim.",
    ),
    experiment: str = typer.Option(None, "--experiment"),
    output: Path = typer.Option(None, "-o", "--output", help="Output directory."),
    override: list[str] = typer.Option([], "--set"),
    stdout: bool = typer.Option(False, "--stdout", help="Print artefact only; do not write a kit."),
    pack: bool = typer.Option(
        False, "--pack", help="Emit ONLY <kit>.tar.gz (remove the loose kit dir), ready to scp + `tvbo workflow submit`."
    ),
    bundle_dataset: bool = typer.Option(
        False,
        "--bundle-dataset",
        help="Copy the fan-out's per-subject dataset files into the kit (spec/dataset/) "
        "and point dataset.bids_root at them, so the kit is self-contained — no "
        "separate FC upload or $TVBO_BIDS_ROOT needed. Scope subjects via dataset.subjects.",
    ),
    bundle_select: list[str] = typer.Option(
        [],
        "--bundle-select",
        help="Override: add a BIDS entity to disambiguate when a subject directory holds "
        "several files matching the observation's query (not needed when the query "
        "already names one file). Repeatable. Implies --bundle-dataset.",
    ),
) -> None:
    """Emit a self-contained sbatch kit (`run.sbatch` + scripts + frozen spec)."""
    sel = _parse_bundle_select(bundle_select) if (bundle_dataset or bundle_select) else None
    _emit(
        "slurm",
        spec=spec,
        backend=backend,
        experiment=experiment,
        output=output,
        override=override,
        stdout=stdout,
        pack=pack,
        bundle_select=sel,
    )


def _validate_code_source(value):
    """Reject a mistyped ``--code-source`` up front — a silently-unmatched value would fall through to the spec path with no error. ``None`` (submit's 'use the kit default') is allowed."""
    if value is not None and value not in ("spec", "frozen"):
        raise typer.BadParameter("must be 'spec' or 'frozen'")
    return value


@app.command("snakemake", help="Emit a self-contained Snakemake kit (Snakefile + scripts + spec).")
def snakemake(
    spec: str = typer.Argument(...),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Execution backend; default: the experiment's declared execution.backend, else tvboptim.",
    ),
    experiment: str = typer.Option(None, "--experiment"),
    output: Path = typer.Option(None, "-o", "--output", help="Output directory."),
    override: list[str] = typer.Option([], "--set"),
    stdout: bool = typer.Option(False, "--stdout", help="Print artefact only; do not write a kit."),
    pack: bool = typer.Option(
        False, "--pack", help="Emit ONLY <kit>.tar.gz (remove the loose kit dir), ready to scp + `tvbo workflow submit`."
    ),
    benchmark: Optional[bool] = typer.Option(
        None,
        "--benchmark/--no-benchmark",
        help="Attach Snakemake's native `benchmark:` directive to every rule: a per-cell TSV "
        "(wall time, max_rss/max_vms/max_uss/max_pss MB, CPU time, I/O) written next to "
        "each output — locally or as a SLURM job. ON by default (a 30 s psutil sampler, "
        "near-zero overhead); pass --no-benchmark to skip. Sugar for --set benchmark=<bool>.",
    ),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Cap every rule's `tvbo run` to one tuning iteration (reach the post-tuning "
        "evaluation fast, e.g. to verify a fit streams within memory). --set smoke=true.",
    ),
    max_iterations: int = typer.Option(
        None,
        "--max-iterations",
        min=1,
        help="Cap every rule's `tvbo run` to N tuning iterations. --set max_iterations=N.",
    ),
    code_source: str = typer.Option(
        "spec",
        "--code-source",
        help="Emit-time DEFAULT code source baked into each rule: 'spec' re-renders backend "
        "code from the frozen spec at run time (back-compat); 'frozen' runs the "
        "pre-rendered scripts/<key> as-is (no codegen — needs only the node's tvbo "
        "runtime). BOTH artefacts are always emitted, so the SAME kit runs either way; "
        "override per run with $TVBO_CODE_SOURCE (see `tvbo workflow submit --code-source`).",
        callback=_validate_code_source,
    ),
    bundle_dataset: bool = typer.Option(
        False,
        "--bundle-dataset",
        help="Copy the fan-out's per-subject dataset files into the kit (spec/<exp>/dataset/) "
        "and point dataset.bids_root at them, so the kit is self-contained — no "
        "separate FC upload or $TVBO_BIDS_ROOT needed. Scope subjects via dataset.subjects.",
    ),
    bundle_select: list[str] = typer.Option(
        [],
        "--bundle-select",
        help="Override: add a BIDS entity to disambiguate when a subject directory holds "
        "several files matching the observation's query (not needed when the query "
        "already names one file). Repeatable. Implies --bundle-dataset.",
    ),
) -> None:
    """Emit a self-contained Snakemake kit (`Snakefile` + scripts + frozen spec)."""
    sel = _parse_bundle_select(bundle_select) if (bundle_dataset or bundle_select) else None
    # Run-modifier flags are sugar for the equivalent `--set` overrides (threaded into the rule at emit): keep the kit the single source of truth, no separate config.
    override = [
        *override,
        *([f"benchmark={'true' if benchmark else 'false'}"] if benchmark is not None else []),
        *(["smoke=true"] if smoke else []),
        *([f"max_iterations={max_iterations}"] if max_iterations is not None else []),
    ]
    _emit(
        "snakemake",
        spec=spec,
        backend=backend,
        experiment=experiment,
        output=output,
        override=override,
        stdout=stdout,
        pack=pack,
        bundle_select=sel,
        code_source=code_source,
    )


@app.command("nextflow", help="Emit a self-contained Nextflow kit (main.nf + scripts + spec).")
def nextflow(
    spec: str = typer.Argument(...),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Execution backend; default: the experiment's declared execution.backend, else tvboptim.",
    ),
    experiment: str = typer.Option(None, "--experiment"),
    output: Path = typer.Option(None, "-o", "--output", help="Output directory."),
    override: list[str] = typer.Option([], "--set"),
    stdout: bool = typer.Option(False, "--stdout", help="Print artefact only; do not write a kit."),
    pack: bool = typer.Option(
        False, "--pack", help="Emit ONLY <kit>.tar.gz (remove the loose kit dir), ready to scp + `tvbo workflow submit`."
    ),
    bundle_dataset: bool = typer.Option(
        False,
        "--bundle-dataset",
        help="Copy the fan-out's per-subject dataset files into the kit (spec/dataset/) "
        "and point dataset.bids_root at them, so the kit is self-contained — no "
        "separate FC upload or $TVBO_BIDS_ROOT needed. Scope subjects via dataset.subjects.",
    ),
    bundle_select: list[str] = typer.Option(
        [],
        "--bundle-select",
        help="Override: add a BIDS entity to disambiguate when a subject directory holds "
        "several files matching the observation's query (not needed when the query "
        "already names one file). Repeatable. Implies --bundle-dataset.",
    ),
) -> None:
    """Emit a self-contained Nextflow kit (`main.nf` + scripts + frozen spec)."""
    sel = _parse_bundle_select(bundle_select) if (bundle_dataset or bundle_select) else None
    _emit(
        "nextflow",
        spec=spec,
        backend=backend,
        experiment=experiment,
        output=output,
        override=override,
        stdout=stdout,
        pack=pack,
        bundle_select=sel,
    )


@app.command("finalize", help="Reassemble an array run's shard outputs into one keyed result artifact.")
def finalize(
    shards_dir: str = typer.Argument("results", help="Directory holding the array tasks' shard outputs."),
    output: Path = typer.Option(Path("results"), "-o", "--output", help="Where to write the reassembled result."),
    spec: Path = typer.Option(
        None, "--spec", help="Frozen spec YAML to attach as the result sidecar (auto-detected in spec/ when omitted)."
    ),
    stem: str = typer.Option("result", "--stem", help="Basename of the result artifact (<stem>.h5 + <stem>.yaml)."),
    compress: bool = typer.Option(True, "--compress/--no-compress", help="gzip-deflate the reassembled HDF5 (default on)."),
) -> None:
    """Gather sharded HPC outputs into one self-describing ``ExperimentResult``.

    Concatenates each array task's slice by parameter value into the full grid a local run produces, and writes it as ``<stem>.h5`` (keyed groups) plus a ``<stem>.yaml`` sidecar (the frozen, fully-overridden spec) — the same
    HDF5-plus-YAML layout as a network. No manual post-processing is needed;
    emitted kits submit this automatically as a dependent gather job.
    """
    from tvbo.data.types import reassemble_experiment_results

    sidecar = spec
    if sidecar is None:
        specs = sorted(p for p in Path("spec").glob("*.yaml") if p.name != "network.yaml") if Path("spec").is_dir() else []
        sidecar = specs[0] if specs else None
    written = reassemble_experiment_results(shards_dir, str(output), stem=stem, sidecar=sidecar, compress=compress)
    for w in written:
        _common.info(f"wrote {w}")


@app.command("run", help="Emit a workflow kit and execute it with the selected engine.")
def run_workflow(
    engine: str = typer.Argument(..., help="Execution engine: slurm | snakemake | nextflow."),
    spec: str = typer.Argument(..., help="Path, CURIE, or DB name (Study or Experiment)."),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Execution backend; default: the experiment's declared execution.backend, else tvboptim.",
    ),
    experiment: str = typer.Option(None, "--experiment"),
    output: Path = typer.Option(None, "-o", "--output", help="Output directory."),
    override: list[str] = typer.Option([], "--set"),
    array: str = typer.Option(
        None,
        "--array",
        help="Slurm array index or range to submit (e.g. '0' for smoke, '0-3' for four tasks). Ignored for non-Slurm engines.",
    ),
    array_throttle: int = typer.Option(
        None,
        "--array-throttle",
        min=1,
        help="Limit concurrent Slurm array tasks when using --array, e.g. 1 for one GPU at a time.",
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Snakemake only: run with this profile (name or path) instead of the kit's shipped "
        "'profile/' — e.g. a site profile carrying the cluster's executor config like BIH "
        "CUBI's 'cubi-v1'. Per-rule resources in the Snakefile apply on top of it.",
    ),
    cores: str = typer.Option(
        None,
        "--cores",
        help="Snakemake only: run LOCALLY on this many cores (an integer, or 'all'). Overrides "
        "only the profile's executor — the kit still runs in the container its profile "
        "declares. The SAME kit submits to the scheduler on HPC when --cores is omitted; "
        "on a machine with no `sbatch` a bare run falls back to local automatically.",
    ),
    code_source: str = typer.Option(
        "spec",
        "--code-source",
        help="Snakemake only: code source for the emitted-and-run kit — 'spec' re-renders "
        "backend code at run time (default), 'frozen' runs the pre-rendered scripts/<key> "
        "(no codegen). Baked in as the kit's default AND exported for this run; BOTH "
        "artefacts are emitted, so the kit stays submittable either way afterwards.",
        callback=_validate_code_source,
    ),
) -> None:
    """Emit a self-contained kit then execute it (or submit for Slurm).

    Use ``--array 0`` to submit only the first array task as a quick smoke test without changing the experiment spec. Use ``--array-throttle`` to cap how many Slurm array tasks run at once, for example ``--array 0-39 --array-throttle 1`` to keep one GPU busy at a time.
    """
    engine = engine.lower()
    if engine not in _ARTEFACT_NAME:
        _common.die(f"`tvbo workflow run` expects engine one of: {', '.join(_ARTEFACT_NAME)}")
    # Each id becomes its own kit, job and output subdir, so they run in parallel on the cluster.
    _exp_ids = [e.strip() for e in str(experiment).split(",") if e.strip()] if experiment else []
    if len(_exp_ids) > 1:
        for _eid in _exp_ids:
            _out_i = (output / f"exp{_eid}") if output else Path("output") / f"exp{_eid}"
            _common.info(f"── experiment {_eid} → {_out_i}")
            run_workflow(engine, spec, backend, _eid, _out_i, override, array, array_throttle, profile, cores, code_source)
        return
    effective_overrides = list(override)
    if engine == "slurm" and array is not None:
        override_keys = {s.lstrip("-").split("=", 1)[0] for s in effective_overrides if "=" in s}
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
            if not plan_preview.workflow_axes and plan_preview.n_array_tasks == 1 and plan_preview.n_vectorize_cells > 1:
                effective_overrides.append(f"slurm.array_chunk={plan_preview.n_vectorize_cells}")
                _common.info(f"auto smoke chunking: set slurm.array_chunk={plan_preview.n_vectorize_cells} for --array run")
    out_dir = _emit(
        engine,
        spec=spec,
        backend=backend,
        experiment=experiment,
        output=output,
        override=effective_overrides,
        stdout=False,
        code_source=code_source,
    )
    if out_dir is None:
        _common.die("failed to emit workflow kit")
    if array is not None and array_throttle is not None:
        array = f"{array}%{array_throttle}"
    # The emitted kit already defaults to code_source; export it too so a non-default choice reaches the job even on an executor that only forwards the environment.
    _execute_emitted(
        engine,
        out_dir,
        slurm_array=array,
        profile=profile,
        cores=cores,
        code_source=code_source if code_source != "spec" else None,
    )


def _tar_extractall_safe(tar, dest: Path) -> None:
    """Extract *tar* into *dest*, using the ``data`` filter where available.

    The ``filter='data'`` guard (Python >= 3.12) rejects members with absolute or ``..`` paths; older interpreters fall back to a plain extract. Kits are emitted by tvbo, so this is defensive rather than a trust boundary.
    """
    try:
        tar.extractall(dest, filter="data")
    except TypeError:
        tar.extractall(dest)


def _resolve_kit_dir(kit: Path, force: bool = False, dest_override: Path | None = None) -> Path:
    """Resolve a kit path that may be a directory or a packaged archive.

    A directory is returned as-is. A ``.tar.gz`` / ``.tgz`` / ``.tar`` / ``.zip`` archive is extracted into *dest_override* (default: next to itself) and the kit root inside it (the directory holding the engine artefact) is returned, so a shipped kit runs without a manual unzip. Point *dest_override* at a fresh directory to run a second copy in parallel without touching an in-progress kit's ``results/`` + ``logs/``. An already-extracted kit at the destination is reused rather than re-extracted, so a re-submit never clobbers an in-progress ``results/`` — unless *force*, which re-extracts a freshly re-uploaded archive over the stale kit files (Snakefile/spec/code/profile) while leaving ``results/`` and ``.snakemake/`` (not in the archive) untouched, so the run resumes anew.
    """
    if kit.is_dir():
        return kit
    if not kit.is_file():
        _common.die(f"not a kit directory or archive: {kit}")
    name = kit.name.lower()
    is_tar = name.endswith((".tar.gz", ".tgz", ".tar"))
    is_zip = name.endswith(".zip")
    if not (is_tar or is_zip):
        _common.die(f"{kit} is neither a kit directory nor a .tar.gz/.tgz/.tar/.zip archive.")
    import tarfile
    import zipfile

    dest = dest_override or kit.parent
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(kit) if is_zip else tarfile.open(kit) as arc:
        names = arc.namelist() if is_zip else arc.getnames()
        tops = sorted({n.split("/", 1)[0] for n in names if n and not n.startswith("/")})
        if not force:
            for t in tops:  # reuse an already-extracted kit; keep its results
                d = dest / t
                if d.is_dir() and _detect_engine_from_kit(d):
                    _common.info(f"kit already extracted → {d} (reusing; pass --force to re-extract)")
                    return d
        # results/ and .snakemake/ are not in the archive, so they survive and the run resumes.
        arc.extractall(dest) if is_zip else _tar_extractall_safe(arc, dest)
    for t in tops:
        d = dest / t
        if d.is_dir() and _detect_engine_from_kit(d):
            _common.info(f"extracted kit → {d}")
            return d
    if _detect_engine_from_kit(dest):  # artefact sat at the archive root
        return dest
    _common.die(f"{kit} did not contain an engine artefact ({', '.join(_ARTEFACT_NAME.values())}).")


@app.command("submit", help="Submit an already-emitted kit (directory or .tar.gz/.zip) to its engine.")
def submit_kit(
    kit: Path = typer.Argument(
        ...,
        help="Path to an emitted kit directory OR a .tar.gz/.tgz/.tar/.zip archive of one (holds run.sbatch / Snakefile / main.nf).",
    ),
    engine: str = typer.Option(None, "--engine", "-e", help="Engine to submit with; auto-detected from the kit when omitted."),
    experiment: str = typer.Option(
        None,
        "--experiment",
        help="Snakemake study kit only: run only these experiments from the full pack (e.g. "
        "'41,50' -> the exp_41 and exp_50 rules). Validated against the kit's Snakefile; "
        "omit to run the whole study (the `all` target).",
    ),
    array: str = typer.Option(
        None,
        "--array",
        help="Slurm array index or range to submit (e.g. '0' for a smoke task, '0-3' for four). Ignored for non-Slurm engines.",
    ),
    array_throttle: int = typer.Option(
        None,
        "--array-throttle",
        min=1,
        help="Limit concurrent Slurm array tasks when using --array (e.g. 1 for one GPU at a time).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-extract a re-uploaded archive over an already-extracted kit, refreshing "
        "the Snakefile/spec/code/profile. Leaves results/ and .snakemake/ intact so the "
        "run resumes with the new definition (no need to rm the kit dir first).",
    ),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Directory to extract the archive into (default: next to the archive). Point it "
        "at a fresh dir to run a second copy in parallel — its results/ + logs/ stay "
        "isolated from an in-progress kit. Ignored when the kit is already a directory.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Resolve and report the work without running or queueing anything — the engine's "
        "own dry run. Validates the kit (DAG, wildcards, inputs) before a large submission.",
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Snakemake only: use this profile (a name resolved against the Snakemake config "
        "search path, or a path) instead of the kit's shipped 'profile/'. Use a site profile "
        "that carries the cluster's canonical executor config — e.g. BIH CUBI's 'cubi-v1' "
        "(`--profile cubi-v1`). The Snakefile's per-rule resources apply on top of it.",
    ),
    cores: str = typer.Option(
        None,
        "--cores",
        help="Snakemake only: run the kit LOCALLY on this many cores (an integer, or 'all'). "
        "Overrides only the profile's executor — the kit still runs in the container its "
        "profile declares. The SAME kit submits to the scheduler on HPC when --cores is "
        "omitted; on a machine with no `sbatch`, a bare submit falls back to local.",
    ),
    code_source: str = typer.Option(
        None,
        "--code-source",
        help="Override the kit's baked-in code source for THIS submission by exporting "
        "TVBO_CODE_SOURCE into the run environment: 'frozen' runs the pre-rendered "
        "scripts/<key> (no codegen); 'spec' re-renders from the frozen spec. Omit to use "
        "the kit's emit-time default (`tvbo workflow snakemake --code-source`). Lets ONE "
        "kit be submitted both ways for verification. Forwarded to the job by a local run "
        "and by any executor that exports its environment (Slurm `--export=ALL`).",
        callback=_validate_code_source,
    ),
) -> None:
    """Submit a kit already emitted by ``tvbo workflow slurm|snakemake|nextflow``.

    Runs only the *execute* half of ``tvbo workflow run`` against an existing kit — no recipe, no re-emit. The kit may be a directory or a packaged ``.tar.gz`` / ``.zip`` archive of one; an archive is extracted next to itself first, so a shipped kit runs without a manual unzip. For Slurm this submits ``run.sbatch`` (the array job) and chains ``finalize.sbatch`` with an ``afterok`` dependency, so you get one reassembled result without touching ``sbatch`` yourself. The engine is inferred from the kit's artefact file unless ``--engine`` is given.
    Run it from a login node (Slurm) or wherever the engine's launcher is available.
    """
    kit = _resolve_kit_dir(kit.expanduser().resolve(), force=force, dest_override=out.expanduser().resolve() if out else None)
    eng = (engine or "").lower() or _detect_engine_from_kit(kit)
    if eng is None:
        _common.die(f"could not detect an engine in {kit} (expected one of: {', '.join(_ARTEFACT_NAME.values())}).")
    if eng not in _ARTEFACT_NAME:
        _common.die(f"unknown engine {eng!r}; expected one of: {', '.join(_ARTEFACT_NAME)}")
    if not (kit / _ARTEFACT_NAME[eng]).exists():
        _common.die(f"{kit} has no {_ARTEFACT_NAME[eng]} (needed for engine {eng!r}).")
    launcher = _LAUNCHER[eng]
    if _resolve_launcher(launcher) is None:
        _common.die(
            f"{launcher!r} was not found next to this interpreter or on PATH — "
            f"{eng} kits are launched with {launcher}. "
            f"Install it in the same environment as tvbo, or run this where "
            f"{launcher} is available" + (" (a Slurm login node)." if eng == "slurm" else ".")
        )
    if array is not None and array_throttle is not None:
        array = f"{array}%{array_throttle}"
    # Before submitting, so a containerized kit needs no manual setup step; idempotent and cheap.
    _provision_env_layer(kit, dry_run=dry_run)
    _execute_emitted(
        eng,
        kit,
        slurm_array=array,
        dry_run=dry_run,
        profile=profile,
        cores=cores,
        code_source=code_source,
        experiment=experiment,
    )


def _provision_env_layer(kit: Path, *, dry_run: bool) -> None:
    """Run the kit's one-time setup.sh (provisions declared requirements into a venv).

    No-op when the kit declares no layer (no setup.sh). On a dry run it only reports the step. A failure is fatal: without the layer, every task crashes on the first import of a declared dependency, far from here — better to stop at submit with a clear message.
    """
    setup = kit / "setup.sh"
    if not setup.exists():
        return
    if dry_run:
        _common.info(f"[dry-run] would run {setup.name} (provision declared requirements into a venv)")
        return
    _common.info("provisioning the declared-requirements venv (setup.sh)…")
    result = subprocess.run(["bash", setup.name], cwd=str(kit))
    if result.returncode != 0:
        _common.die(
            f"setup.sh failed (exit {result.returncode}). The requirements venv is not in "
            f"place, so every task would crash on its first import — refusing to submit. "
            f"For a containerized kit, check the image and `apptainer`/`singularity` are available here."
        )


@app.command("backends", help="List backends and their ontology-derived capabilities.")
def backends(
    json: bool = typer.Option(False, "--json"),
) -> None:
    """List execution backends and their ontology-derived capabilities (continuous/spiking/jit/etc.)."""
    rows = []
    for spec in list_backends():
        rows.append(
            {
                "name": spec.name,
                "label": spec.label,
                "tasks": sorted(spec.tasks),
                "capabilities": sorted(spec.capabilities),
                "vectorize_axes": sorted(spec.vectorize_axes),
            }
        )
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
