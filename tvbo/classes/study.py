"""User-facing `SimulationStudy` class for grouping related experiments.

Provides a thin wrapper around the generated datamodel that adds loading helpers (YAML files, the tvbo database, openMINDS JSON-LD), citation formatting, and access to individual `SimulationExperiment`s.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tvbo import templates
from tvbo.classes import experiment
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.utils import report, yaml_loader


@dataclass(frozen=True)
class FigureImage:
    """One figure a study run rendered: the image, the self-contained script that drew it, and an inline representation.

    ``_repr_png_`` / ``_repr_svg_`` are what a notebook or Quarto cell calls, so the cell that ran the study also shows what it drew and a page never writes its own image reference. The object is a :class:`os.PathLike`, so it still passes anywhere a path does.
    """

    name: str
    path: Path
    script: Path | None = None
    caption: str | None = None

    def __fspath__(self) -> str:
        return str(self.path)

    def _repr_png_(self):
        return self.path.read_bytes() if self.path.suffix.lower() == ".png" and self.path.is_file() else None

    def _repr_svg_(self):
        return self.path.read_text(encoding="utf-8") if self.path.suffix.lower() == ".svg" and self.path.is_file() else None


class StudyResult(Mapping):
    """What one :meth:`SimulationStudy.run` produced — the study-level counterpart of the :class:`~tvbo.data.types.ExperimentResult` a single experiment's ``run()`` returns.

    A mapping from experiment to that experiment's outputs, read back from the same container the study's figures read, so a page and the figure it shows cannot report different numbers. Keys are the ``exp-<id>`` spelling the recipe's ``id:`` gives each experiment, which is also the name of the container on disk; the experiment's ``label`` reaches it too, and so does the bare id a figure layer writes (``used: {experiment: 2}``). A bare integer is accepted for that last reason and no other: it is the id, never a position in the list.

    Each entry is an :class:`xarray.DataTree` in the recipe's own shape rather than the container's flat one, so an output is reached along the path it was declared at::

        results["exp-1"].optimizations.spectral_gradient_fit.observations.peak_frequencies

    Being a tree rather than a bag of attributes is what keeps the run an xarray object end to end: the node labels are declared once at the root and inherited by every group, one ``.sel`` reaches every per-node output at once, and an analysis can write back out what it derived.

    :meth:`dataset` hands back the underlying :class:`xarray.Dataset` for anyone who wants the flat names, :meth:`analysis` reaches an analysis container by name, :meth:`figure` a rendered figure by name, and :meth:`report` the study's Methods section. Containers are read on first access and cached.
    """

    def __init__(self, study, root: Path, results_root: Path, experiments, figures, members=None):
        from tvbo.data.dataref import experiment_id

        self.study = study
        self.members = dict(members or {})
        """Each member study's own result, keyed by the label the recipe gives it — empty unless this is a study-of-studies.

        Nested rather than merged into this mapping, because that is the shape the recipe has and the shape the results have on disk: a member keeps its own results root, and two members both declaring ``exp-1`` are two different experiments.
        """
        self.root = Path(root)
        self.results_root = Path(results_root)
        self._figures = {f.name: f for f in figures}
        self._open: dict[str, object] = {}
        self._keys: list[str] = []
        self._by_alias: dict[str, str] = {}
        for exp in experiments or []:
            eid = experiment_id(getattr(exp, "id", exp))
            if eid is None:
                continue
            key = f"exp-{eid}"
            self._keys.append(key)
            for alias in (key, eid, getattr(exp, "label", None)):
                if alias:
                    self._by_alias[str(alias)] = key

    def _dataset(self, path: Path):
        """The container at *path*, read into memory once and kept.

        Read rather than opened: an open HDF5 handle holds a lock, and a result object that outlives its cell — which every notebook one does — would block the next run from rewriting the container it is holding.
        """
        import xarray as xr

        key = str(path)
        if key not in self._open:
            with xr.open_dataset(path, engine="h5netcdf") as ds:
                self._open[key] = ds.load()
        return self._open[key]

    def _container(self, key) -> Path:
        """Path to the container of the experiment *key* names, in any spelling the recipe uses."""
        from tvbo.data.dataref import locate_exp_container

        resolved = self._by_alias.get(str(getattr(key, "id", key)))
        if resolved is None:
            raise KeyError(
                f"{key!r} names no experiment in this study. It declares {self._keys}, "
                f"reachable by that key, by its label, or by the bare id a figure layer writes."
            )
        return locate_exp_container(self.results_root, resolved.removeprefix("exp-"))

    def __getitem__(self, key):
        """Experiment *key*'s outputs as an :class:`xarray.DataTree` shaped like the recipe that declared them."""
        from tvbo.data.experiment_result_io import result_tree

        return result_tree(self._dataset(self._container(key)))

    def dataset(self, key):
        """The raw :class:`xarray.Dataset` behind experiment *key*, with the container's flat variable names."""
        return self._dataset(self._container(key))

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def analysis(self, name: str):
        """The container the analysis *name* wrote, as a labelled :class:`xarray.Dataset`."""
        from tvbo.data.dataref import locate_analysis_container

        return self._dataset(locate_analysis_container(self.results_root, name))

    @property
    def figures(self) -> dict:
        """The rendered figures, keyed by their declared ``name``."""
        return dict(self._figures)

    def _undrawn(self) -> str:
        """The declared figures this run did not draw, named so a reader is not left to conclude the recipe declared none."""
        from tvbo.utils import as_list

        declared = {str(f.name) for f in as_list(getattr(self.study, "figures", None)) if getattr(f, "name", None)}
        missing = sorted(declared - set(self._figures))
        if not missing:
            return ""
        return f"; declared but not drawn: {', '.join(missing)} — rendering failed, and the run logged why at WARNING"

    def figure(self, name: str | None = None) -> FigureImage:
        """The rendered figure *name*, or the only one when the study declares a single figure."""
        if name is None:
            if len(self._figures) != 1:
                raise KeyError(
                    f"this run drew {len(self._figures)} figures; name one of: {sorted(self._figures)}{self._undrawn()}"
                )
            return next(iter(self._figures.values()))
        try:
            return self._figures[name]
        except KeyError:
            raise KeyError(
                f"no figure {name!r} was rendered; this run drew: {sorted(self._figures)}{self._undrawn()}"
            ) from None

    def report(self, *args, **kwargs) -> str:
        """The study's Methods section — :meth:`SimulationStudy.report`, reached from the run that produced the numbers it describes."""
        return self.study.report(*args, **kwargs)

    def __repr__(self) -> str:
        # Relative where that is shorter: this repr is cell output on a docs page, and an absolute path would publish one machine's directory layout and change under every other.
        root = str(self.results_root)
        try:
            rel = os.path.relpath(self.results_root)
            root = rel if len(rel) < len(root) else root
        except ValueError:
            pass
        members = f", members={sorted(self.members)}" if self.members else ""
        return f"StudyResult(experiments={self._keys}, figures={sorted(self._figures)}{members}, results_root={root!r})"


class SimulationStudy(tvbo_datamodel.SimulationStudy):
    """A collection of related `SimulationExperiment`s with shared provenance.

    Aggregates the experiments behind a published paper or analysis (model, DOI, year, citation, dataset) into one declarative YAML/Pydantic object.
    Load with `from_db(name)` for curated studies, `from_file(path)` for local YAML, or `from_openminds(...)` for JSON-LD provenance graphs.

    A study that declares `members` is a study-of-studies: it aggregates other studies' recipes and owns whatever experiments, analyses and figures belong to no member. Recursion runs through the pointer, so a member's own recipe may declare members of its own.

    The most-used entry points are [`get_experiment(id)`](#tvbo.classes.study.SimulationStudy.get_experiment) to materialise a single run, [`cite()`](#tvbo.classes.study.SimulationStudy.cite) for the formatted citation, and [`to_openminds(...)`](#tvbo.classes.study.SimulationStudy.to_openminds) for JSON-LD export.
    """

    def member_recipes(self, base=None, *, include_optional: bool = True) -> list[tuple[str, "Path"]]:
        """The member study recipes as ``(label, resolved_path)`` pairs.

        A study with ``members`` is a study-of-studies. Each ``recipe`` is resolved relative to *base* (this study's own directory) when it is not an IRI or an absolute path, so a member keeps its own directory and therefore its own results root. ``optional`` members are dropped when *include_optional* is False (a ``--skip``-style light run).
        """
        base = Path(base) if base is not None else Path(getattr(self, "_source_file", ".")).resolve().parent
        out: list[tuple[str, Path]] = []
        for m in getattr(self, "members", None) or []:
            if getattr(m, "optional", None) and not include_optional:
                continue
            recipe = str(getattr(m, "recipe", ""))
            label = getattr(m, "label", None) or Path(recipe).stem
            path = Path(recipe)
            resolved = path if path.is_absolute() or "://" in recipe else (base / recipe)
            out.append((str(label), resolved))
        return out

    def __repr__(self) -> str:
        key = self.key or "?"
        title = self.title or "Untitled Study"
        year = getattr(self, "year", "n.d.")
        doi = self.doi or "n/a"
        exps = getattr(self, "experiments", {}) or {}
        n_exp = len(exps)
        model_name = str(self.model) if self.model else "n/a"
        n_members = len(getattr(self, "members", None) or [])
        members = f", members={n_members}" if n_members else ""
        return (
            f"SimulationStudy(\n"
            f"  key={key!r},\n"
            f"  title={title!r},\n"
            f"  year={year}, doi={doi!r},\n"
            f"  model={model_name!r}, experiments={n_exp}{members}\n"
            f")"
        )

    @classmethod
    def from_file(cls, filepath):
        """Load a study from a local YAML file.

        The resolved absolute path is stored on the returned instance so that experiments materialised later can locate sibling data files.

        Args:
            filepath: Path to the study YAML file.

        Returns:
            A `SimulationStudy` parsed from the file.
        """
        from tvbo.utils import register_recipe_code_paths

        study = yaml_loader.load(filepath, cls)
        study._source_file = str(Path(filepath).resolve())
        # Make the recipe's callable code importable so custom builders/callables resolve by bare module name without a PYTHONPATH prefix: an explicit code_source (local dir or git repo) when declared, else the code/ subdir beside the YAML.
        register_recipe_code_paths(study._source_file, getattr(study, "code_source", None))
        # Keep the raw (anchor- and !include-resolved) experiment dicts so experiments can be materialised through SimulationExperiment.from_string — that path iri-sources dynamics/coupling from the registry, which loading the datamodel object directly does not. Extract them with the SAME loader as the LinkML load above (yaml_loader, NOT a plain yaml.safe_load) so the two load paths cannot diverge: load_as_dict resolves `!include` fragments and folds slot aliases identically, so a modular (!include-split) study materialises exactly like a monolithic one. A plain safe_load chokes on the `!include` tag and would silently empty this, dropping every experiment to the iri-unaware from_datamodel fallback.
        try:
            _raw = yaml_loader.load_as_dict(filepath) or {}
            _raw_exps = {e.get("id"): e for e in (_raw.get("experiments") or []) if isinstance(e, dict)}
        except Exception:
            _raw_exps = {}
        # Store as a plain dict (bypass the JsonObj setattr that would wrap it).
        object.__setattr__(study, "_raw_experiments", _raw_exps)
        return study

    @classmethod
    def from_datamodel(cls, datamodel: tvbo_datamodel.SimulationStudy):
        """Wrap a generated datamodel instance as a `SimulationStudy`.

        Args:
            datamodel: A datamodel-level study whose fields are copied into the
                user-facing class.

        Returns:
            A `SimulationStudy` with the same field values as `datamodel`.
        """
        return cls(**datamodel._as_dict)

    @classmethod
    def from_db(cls, name: str) -> "SimulationStudy":
        """Load a SimulationStudy by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("SimulationStudy", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available studies in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("SimulationStudy")

    def cite(self):
        """Return the formatted citation for this study.

        Returns:
            The citation string resolved from the study's `key`.
        """
        return report.get_citation(self.key)

    def experiment_ids(self) -> list:
        """The declared experiment ids, in recipe order."""
        return [getattr(e, "id", None) for e in (getattr(self, "experiments", None) or [])]

    def run(self, root=None, *, backend: str | None = None, figures: bool = True) -> "StudyResult":
        """Run the whole study in process — every experiment, its analyses, its declared figures — and return what it produced.

        The Python entry point to what ``tvbo run <recipe>.yaml`` does from a shell, on the same orchestration, so a notebook and the CLI cannot drift apart. Use it wherever the containers are wanted as objects rather than as files: a docs page, a notebook, an embedding application.

        Args:
            root: The study root this run writes into — containers land in its results directory, figures in its figures directory, and every layer's ``used:`` resolves against it. Defaults to the recipe's own directory, which is what the CLI uses; point it elsewhere to keep a run's outputs out of the tree the recipe is committed in.
            backend: Backend for each experiment that does not declare its own.
            figures: Render the study's declared ``figures:`` once the experiments finish.

        A study-of-studies runs through the same branch the CLI takes: every member first, in its own directory, then this study's own content, then the results manifest — and each member's own `StudyResult` comes back under `StudyResult.members`.

        Returns:
            A [`StudyResult`](#tvbo.classes.study.StudyResult) — the experiments' containers by id, the analyses by name, the rendered figures, and any member studies.
        """
        from tvbo.cli.run import _has_members, _run_whole_study, _run_with_members, member_root

        spec = getattr(self, "_source_file", None)
        if not spec:
            raise ValueError(
                "run() needs the recipe this study was loaded from, to resolve its relative paths; "
                "load it with SimulationStudy.from_file(...) or from_db(...) rather than constructing it in memory."
            )
        recipe_dir = Path(spec).resolve().parent
        base = Path(root).resolve() if root is not None else recipe_dir
        base.mkdir(parents=True, exist_ok=True)

        if not _has_members(self):
            _run_whole_study(self, str(spec), None, backend=backend, figures=figures, base=base)
            return self._collect(base)

        _run_with_members(self, str(spec), None, backend=backend, figures=figures, base=base)
        members = {}
        for label, recipe in self.member_recipes(recipe_dir):
            member = SimulationStudy.from_file(str(recipe))
            members[label] = member._collect(member_root(base, recipe_dir, recipe) or Path(recipe).resolve().parent)
        return self._collect(base, members=members)

    def _collect(self, base: Path, members=None) -> "StudyResult":
        """The `StudyResult` describing what a completed run left under *base*.

        Reading the outcome is separated from causing it so that a member study, whose run this object did not drive, is described by exactly the same code as the study that owns it.
        """
        from tvbo.adapters.bsplot import compose_caption
        from tvbo.cli.figures import figure_outputs
        from tvbo.cli.run import study_path_for
        from tvbo.utils import as_list

        fig_dir = study_path_for("figures", base)
        drawn = []
        for fig in as_list(getattr(self, "figures", None)):
            name, image, script = figure_outputs(fig, fig_dir)
            if image.is_file():
                drawn.append(
                    FigureImage(
                        name=name,
                        path=image,
                        script=script if script.is_file() else None,
                        caption=compose_caption(fig) or None,
                    )
                )
        return StudyResult(
            self,
            base,
            study_path_for("results", base),
            getattr(self, "experiments", None) or [],
            drawn,
            members=members,
        )

    def report(
        self,
        format: str = "markdown",
        part: str = "main",
        level: int = 2,
        equations: str = "semantic",
        orient: str = "auto",
        experiments=None,
        outputfile: str | None = None,
        derivative_notation: str = "dot",
        mul_symbol: str | None = None,
    ) -> str:
        r"""Render one Methods section for the whole study.

        Experiments that share a model share its equations and its symbol table; a model that merely varies a sibling contributes only its delta. Everything the experiments hold in common is stated once, and the comparison table carries only what actually differs — so a seven-experiment study stops emitting seven copies of the same six equations and three copies of the same parameter table.

        Args:
            format: ``markdown`` / ``md`` (``\\tag`` numbering), ``qmd`` (Quarto
                ``{#eq-…}`` / ``{#tbl-…}`` anchors), or ``pdf``.
            part: ``main``, ``supplementary`` or ``all`` — which experiments carry their
                full paragraph, read from each experiment's declared ``part``. Every
                experiment appears in the comparison table regardless, so a demoted one
                is still visible; ``part`` never changes what runs.
            level: Heading depth of the model sections, so the block nests under the
                section that hosts it; experiments sit one level deeper.
            equations: ``semantic`` anchors on model and variable (stable when an
                experiment is inserted), ``sequential`` anchors on the number, ``none``
                leaves equations unnumbered.
            orient: ``auto`` keeps the experiment table narrow, or pin it with ``rows`` /
                ``columns`` (where the *experiments* go) so the Methods keeps its shape.
            experiments: Optional explicit ids to describe; defaults to all of them.
            outputfile: Write the render here; the extension (``.md`` / ``.qmd`` /
                ``.pdf``) overrides ``format``.
            derivative_notation: ``dot`` for :math:`\\dot x`, anything else for ``dx/dt``.
            mul_symbol: Passed to ``sympy.latex``.
        """
        ext_format = {".md": "markdown", ".markdown": "markdown", ".qmd": "qmd", ".pdf": "pdf"}
        if outputfile:
            ext = os.path.splitext(outputfile)[1].lower()
            if ext not in ext_format:
                raise ValueError("outputfile extension must be one of: .md, .qmd, .pdf")
            format = ext_format[ext]
        if format not in ("markdown", "md", "qmd", "pdf"):
            raise ValueError("format must be one of: markdown, md, qmd, pdf")
        if part not in ("main", "supplementary", "all"):
            raise ValueError("part must be one of: main, supplementary, all")

        ids = self.experiment_ids() if experiments is None else list(experiments)
        render = templates.lookup.get_template("report/tvbo-report-study.md.mako").render(
            experiments=[self.get_experiment(i) for i in ids],
            part=part,
            level=level,
            fmt="qmd" if format == "qmd" else "markdown",
            eqs=report.Equations(equations, "qmd" if format == "qmd" else "markdown"),
            orient=orient,
            derivative_notation=derivative_notation,
            mul_symbol=mul_symbol,
        )

        if outputfile:
            if format == "pdf":
                report.to_pdf(render, outputfile)
            else:
                with open(outputfile, "w", encoding="utf-8") as fh:
                    fh.write(render)
        return render

    def get_experiment(self, experiment_id):
        """Retrieve a single experiment by its declared id."""
        exps = getattr(self, "experiments", None) or []
        source_file = getattr(self, "_source_file", None)
        raw_experiments = getattr(self, "_raw_experiments", None) or {}
        for exp_dm in exps:
            if getattr(exp_dm, "id", None) == experiment_id:
                if source_file:
                    experiment.SimulationExperiment._pending_source_file = source_file
                try:
                    # Materialise through the YAML construction path so that iri-sourced components (dynamics, coupling) are merged from the registry — exactly as SimulationExperiment.from_file does. from_datamodel alone skips that resolution and would leave an iri-only dynamics unpopulated. Prefer the raw authored experiment dict (minimal, anchor-resolved) so the merge behaves identically to loading a standalone spec.
                    raw = raw_experiments.get(experiment_id)
                    if isinstance(raw, dict):
                        import yaml

                        exp = experiment.SimulationExperiment.from_string(yaml.safe_dump(raw))
                    else:
                        from linkml_runtime.dumpers import yaml_dumper

                        exp = experiment.SimulationExperiment.from_string(yaml_dumper.dumps(exp_dm))
                    if source_file:
                        exp._source_file = source_file
                finally:
                    experiment.SimulationExperiment._pending_source_file = None
                return exp
        available = [getattr(e, "id", None) for e in exps]
        raise KeyError(f"Experiment {experiment_id!r} not found. Available: {available}")

    # ---- OpenMINDS JSON-LD conversion ----
    def to_openminds(
        self,
        filepath: str | None = None,
        base_id: str | None = None,
        include_context: bool = True,
    ) -> dict:
        """Export study to openMINDS JSON-LD format.

        Parameters
        ----------
        filepath : str, optional
            If provided, write JSON-LD to this file path.
        base_id : str, optional
            Base URI for generating @id values (e.g., "https://example.org/studies").
            If not provided and study has a DOI, uses the DOI as @id.
        include_context : bool
            Whether to include the @context in the output. Default True.

        Returns:
        -------
        dict
            OpenMINDS-compatible JSON-LD dictionary.

        Example:
        -------
        >>> study = SimulationStudy.from_file("study.yaml")
        >>> jsonld = study.to_openminds()
        >>> study.to_openminds("output.jsonld", base_id="https://example.org")
        """
        from tvbo.adapters.openminds import save_openminds, study_to_openminds

        result = study_to_openminds(self, base_id=base_id, include_context=include_context)

        if filepath:
            save_openminds(self, filepath, base_id=base_id)

        return result

    @classmethod
    def from_openminds(cls, source: str | dict) -> "SimulationStudy":
        """Create a SimulationStudy from openMINDS JSON-LD.

        Parameters
        ----------
        source : str or dict
            Either a file path to a JSON-LD file, or a dict containing
            JSON-LD data.

        Returns:
        -------
        SimulationStudy
            New instance constructed from the openMINDS data.

        Example:
        -------
        >>> study = SimulationStudy.from_openminds("study.jsonld")
        >>> study = SimulationStudy.from_openminds({"@type": "tvbo:SimulationStudy", ...})
        """
        from tvbo.adapters.openminds import load_openminds, study_from_openminds

        if isinstance(source, str):
            # It's a file path
            data = load_openminds(source)
        elif isinstance(source, dict):
            data = study_from_openminds(source)
        else:
            raise TypeError(f"Expected str or dict, got {type(source)}")

        return cls(**data)
