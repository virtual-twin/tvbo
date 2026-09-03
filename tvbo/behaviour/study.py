#
# Module: behaviour/study.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""What a ``SimulationStudy`` record does: load itself, walk what it nests, run, report and cite.

A study that declares ``studies`` is a study-of-studies — it aggregates other studies and owns whatever experiments, analyses and figures belong to none of them — so every operation here recurses. The nested studies are records of the same class, which is what makes the nesting self-similar to any depth, and is why nothing here has to promote a nested study to a different class before it can run.
"""

from __future__ import annotations

import os
from pathlib import Path

from jsonasobj2 import as_dict


def _wire_nested(study, raw: dict | None) -> None:
    """Give every study nested under *study* the identity its own recipe gives it, recursively.

    ``!include`` splices a sub-study's content into the parent document and the parent's constructor builds it as a record like any other, which knows neither which file it came from nor which raw experiment dicts are its own. This walks the constructed tree beside the raw document it was built from — the raw mapping is where the loader recorded the origin — and hands each nested study its source file, the code directory that file implies, and its raw experiments. Reloading each recipe would produce the same objects by parsing the whole tree a second time.
    """
    from tvbo.utils import as_list, register_recipe_code_paths
    from tvbo.utils.yaml_loader import include_source

    raw_nested = (raw or {}).get("studies") or []
    wired = []
    for index, child in enumerate(as_list(getattr(study, "studies", None))):
        nested_raw = raw_nested[index] if index < len(raw_nested) else None
        source = include_source(nested_raw)
        if source:
            child._source_file = str(Path(source).resolve())
            register_recipe_code_paths(child._source_file, getattr(child, "code_source", None))
        elif getattr(child, "_source_file", None) is None:
            child._source_file = getattr(study, "_source_file", None)
        object.__setattr__(
            child,
            "_raw_experiments",
            {e.get("id"): e for e in (nested_raw or {}).get("experiments") or [] if isinstance(e, dict)},
        )
        _wire_nested(child, nested_raw)
        wired.append(child)
    object.__setattr__(study, "_nested", wired)


class SimulationStudyBehaviour:
    """A collection of related `SimulationExperiment`s with shared provenance.

    Aggregates the experiments behind a published paper or analysis (model, DOI, year, citation, dataset) into one declarative YAML/Pydantic object.
    Load with `from_db(name)` for curated studies, `from_file(path)` for local YAML, or `from_openminds(...)` for JSON-LD provenance graphs.

    A study that declares `studies` is a study-of-studies: it aggregates other studies and owns whatever experiments, analyses and figures belong to none of them. The nested studies are `SimulationStudy` objects like this one, so the nesting is self-similar and recurses to any depth.

    The most-used entry points are [`get_experiment(id)`](#tvbo.classes.study.SimulationStudy.get_experiment) to materialise a single run, [`cite()`](#tvbo.classes.study.SimulationStudy.cite) for the formatted citation, and [`to_openminds(...)`](#tvbo.classes.study.SimulationStudy.to_openminds) for JSON-LD export.
    """

    def study_label(self) -> str:
        """The name this study is addressed by — in a `--skip` list, a `count:` binding, or a run's progress line.

        The citekey first, then the recipe's own filename, and only then ``label``: a study's ``label`` is a human-readable title and is frequently a whole sentence, which no one can type into ``--skip`` and which reads as noise in a progress line.
        """
        for attr in ("citekey", "key"):
            value = getattr(self, attr, None)
            if value:
                return str(value)
        source = getattr(self, "_source_file", None)
        if source:
            return Path(source).stem
        return str(self.label) if getattr(self, "label", None) else "study"

    def nested_studies(self) -> list[tuple[str, SimulationStudyBehaviour]]:
        """This study's immediate sub-studies as ``(label, study)`` pairs.

        A sub-study spliced in by ``!include`` carries the file it came from, so it keeps its own ``_source_file`` and therefore its own results root, code directory and relative references — which is the whole point of nesting by pointer rather than by copy. A sub-study written inline has no file of its own and inherits this study's.

        Wired once when the study is loaded, so the tree is parsed exactly once however often it is walked.
        """
        from tvbo.utils import as_list

        wired = getattr(self, "_nested", None)
        if wired is None:
            wired = [self._adopt_inline(nested) for nested in as_list(getattr(self, "studies", None))]
            object.__setattr__(self, "_nested", wired)
        return [(study.study_label(), study) for study in wired]

    def _adopt_inline(self, nested):
        """An inline sub-study, sharing this one's source file when its own recipe named none.

        A nested study is already a study — the class is the same one — so this hands down the source file and nothing else.
        """
        study = nested
        if getattr(study, "_source_file", None) is None:
            study._source_file = getattr(self, "_source_file", None)
        return study

    def walk_studies(self, *, include_self: bool = True) -> list[tuple[str, SimulationStudyBehaviour]]:
        """Every study in this tree, each sub-study before the study that holds it.

        Depth first with the holder last, because that is the order the content depends in: a study's own analyses and figures may read what its sub-studies produced, never the other way round. *include_self* drops the outermost study, which is how a caller runs the tree without re-running the study it started from.
        """
        out: list[tuple[str, SimulationStudyBehaviour]] = []
        for label, nested in self.nested_studies():
            out.extend(nested.walk_studies(include_self=False))
            out.append((label, nested))
        if include_self:
            out.append((self.study_label(), self))
        return out

    def __repr__(self) -> str:
        key = self.key or "?"
        title = self.title or "Untitled Study"
        year = getattr(self, "year", "n.d.")
        doi = self.doi or "n/a"
        exps = getattr(self, "experiments", {}) or {}
        n_exp = len(exps)
        model_name = str(self.model) if self.model else "n/a"
        from tvbo.utils import as_list

        n_nested = len(as_list(getattr(self, "studies", None)))
        members = f", studies={n_nested}" if n_nested else ""
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
        from tvbo.utils import register_recipe_code_paths, yaml_loader

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
        # Wire the nested studies from the content already parsed, rather than reloading each recipe: `!include` has read every one of them, and a tree of twenty-odd studies parsed twice costs seconds on every command that only wants to know what it contains.
        _wire_nested(study, _raw)
        return study

    @classmethod
    def from_datamodel(cls, datamodel):
        """Wrap a generated datamodel instance as a `SimulationStudy`.

        Args:
            datamodel: A datamodel-level study whose fields are copied into the
                user-facing class.

        Returns:
            A `SimulationStudy` with the same field values as `datamodel`.
        """
        return cls(**as_dict(datamodel))

    @classmethod
    def from_db(cls, name: str) -> SimulationStudyBehaviour:
        """Load a SimulationStudy by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("SimulationStudyBehaviour", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available studies in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("SimulationStudyBehaviour")

    def cite(self):
        """Return the formatted citation for this study.

        Returns:
            The citation string resolved from the study's `key`.
        """
        from tvbo.utils import report

        return report.get_citation(self.key)

    def experiment_ids(self) -> list:
        """The declared experiment ids, in recipe order."""
        return [getattr(e, "id", None) for e in (getattr(self, "experiments", None) or [])]

    def run(self, root=None, *, backend: str | None = None, figures: bool = True):
        """Run the whole study in process — every experiment, its analyses, its declared figures — and return what it produced.

        The Python entry point to what ``tvbo run <recipe>.yaml`` does from a shell, on the same orchestration, so a notebook and the CLI cannot drift apart. Use it wherever the containers are wanted as objects rather than as files: a docs page, a notebook, an embedding application.

        Args:
            root: The study root this run writes into — containers land in its results directory, figures in its figures directory, and every layer's ``used:`` resolves against it. Defaults to the recipe's own directory, which is what the CLI uses; point it elsewhere to keep a run's outputs out of the tree the recipe is committed in.
            backend: Backend for each experiment that does not declare its own.
            figures: Render the study's declared ``figures:`` once the experiments finish.

        A study-of-studies runs through the same branch the CLI takes: every nested study first, in its own directory, then this study's own content, then the results manifest — and each nested study's own `StudyResult` comes back under `StudyResult.studies`.

        Returns:
            A [`StudyResult`](#tvbo.classes.study.StudyResult) — the experiments' containers by id, the analyses by name, the rendered figures, and any nested studies.
        """
        from tvbo.cli.run import _has_nested, _run_tree, _run_whole_study, nested_root

        spec = getattr(self, "_source_file", None)
        if not spec:
            raise ValueError(
                "run() needs the recipe this study was loaded from, to resolve its relative paths; "
                "load it with SimulationStudy.from_file(...) or from_db(...) rather than constructing it in memory."
            )
        recipe_dir = Path(spec).resolve().parent
        base = Path(root).resolve() if root is not None else recipe_dir
        base.mkdir(parents=True, exist_ok=True)

        if not _has_nested(self):
            _run_whole_study(self, str(spec), None, backend=backend, figures=figures, base=base)
            return self._collect(base)

        _run_tree(self, str(spec), None, backend=backend, figures=figures, base=base)
        nested_results = {}
        for label, nested in self.nested_studies():
            recipe = Path(getattr(nested, "_source_file", None) or spec)
            nested_results[label] = nested._collect(nested_root(base, recipe_dir, label) or recipe.resolve().parent)
        return self._collect(base, studies=nested_results)

    def _collect(self, base: Path, studies=None):
        """The `StudyResult` describing what a completed run left under *base*.

        Reading the outcome is separated from causing it so that a nested study, whose run this object did not drive, is described by exactly the same code as the study that owns it.
        """
        from tvbo.adapters.bsplot import compose_caption
        from tvbo.cli.figures import figure_outputs
        from tvbo.cli.run import study_path_for
        from tvbo.utils import as_list

        fig_dir = study_path_for("figures", base)
        from tvbo.classes.study import FigureImage, StudyResult

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
            studies=studies,
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
        from tvbo import templates
        from tvbo.utils import report

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
        from tvbo.classes import experiment

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
    def from_openminds(cls, source: str | dict) -> SimulationStudyBehaviour:
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
