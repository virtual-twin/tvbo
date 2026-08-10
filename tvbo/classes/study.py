"""User-facing `SimulationStudy` class for grouping related experiments.

Provides a thin wrapper around the generated datamodel that adds loading
helpers (YAML files, the tvbo database, openMINDS JSON-LD), citation
formatting, and access to individual `SimulationExperiment`s.
"""

import os

from tvbo.utils import yaml_loader

from tvbo import templates
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.classes import experiment
from tvbo.utils import report


class SimulationStudy(tvbo_datamodel.SimulationStudy):
    """A collection of related `SimulationExperiment`s with shared provenance.

    Aggregates the experiments behind a published paper or analysis (model,
    DOI, year, citation, dataset) into one declarative YAML/Pydantic object.
    Load with `from_db(name)` for curated studies, `from_file(path)` for
    local YAML, or `from_openminds(...)` for JSON-LD provenance graphs.

    The most-used entry points are
    [`get_experiment(id)`](#tvbo.classes.study.SimulationStudy.get_experiment)
    to materialise a single run, [`cite()`](#tvbo.classes.study.SimulationStudy.cite)
    for the formatted citation, and
    [`to_openminds(...)`](#tvbo.classes.study.SimulationStudy.to_openminds) for
    JSON-LD export.
    """

    def __repr__(self) -> str:
        key = self.key or "?"
        title = self.title or "Untitled Study"
        year = getattr(self, "year", "n.d.")
        doi = self.doi or "n/a"
        exps = getattr(self, "experiments", {}) or {}
        n_exp = len(exps)
        model_name = str(self.model) if self.model else "n/a"
        return (
            f"SimulationStudy(\n"
            f"  key={key!r},\n"
            f"  title={title!r},\n"
            f"  year={year}, doi={doi!r},\n"
            f"  model={model_name!r}, experiments={n_exp}\n"
            f")"
        )

    @classmethod
    def from_file(cls, filepath):
        """Load a study from a local YAML file.

        The resolved absolute path is stored on the returned instance so that
        experiments materialised later can locate sibling data files.

        Args:
            filepath: Path to the study YAML file.

        Returns:
            A `SimulationStudy` parsed from the file.
        """
        from pathlib import Path
        from tvbo.utils import register_recipe_code_paths

        study = yaml_loader.load(filepath, cls)
        study._source_file = str(Path(filepath).resolve())
        # Make the recipe's callable code importable so custom builders/callables
        # resolve by bare module name without a PYTHONPATH prefix: an explicit
        # code_source (local dir or git repo) when declared, else the code/
        # subdir beside the YAML.
        register_recipe_code_paths(study._source_file, getattr(study, "code_source", None))
        # Keep the raw (anchor- and !include-resolved) experiment dicts so experiments
        # can be materialised through SimulationExperiment.from_string — that path
        # iri-sources dynamics/coupling from the registry, which loading the datamodel
        # object directly does not. Extract them with the SAME loader as the LinkML load
        # above (yaml_loader, NOT a plain yaml.safe_load) so the two load paths cannot
        # diverge: load_as_dict resolves `!include` fragments and folds slot aliases
        # identically, so a modular (!include-split) study materialises exactly like a
        # monolithic one. A plain safe_load chokes on the `!include` tag and would silently
        # empty this, dropping every experiment to the iri-unaware from_datamodel fallback.
        try:
            _raw = yaml_loader.load_as_dict(filepath) or {}
            _raw_exps = {
                e.get("id"): e for e in (_raw.get("experiments") or []) if isinstance(e, dict)
            }
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
        """Render one Methods section for the whole study.

        Experiments that share a model share its equations and its symbol table; a model
        that merely varies a sibling contributes only its delta. Everything the
        experiments hold in common is stated once, and the comparison table carries only
        what actually differs — so a seven-experiment study stops emitting seven copies
        of the same six equations and three copies of the same parameter table.

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
                    # Materialise through the YAML construction path so that
                    # iri-sourced components (dynamics, coupling) are merged from
                    # the registry — exactly as SimulationExperiment.from_file
                    # does. from_datamodel alone skips that resolution and would
                    # leave an iri-only dynamics unpopulated. Prefer the raw
                    # authored experiment dict (minimal, anchor-resolved) so the
                    # merge behaves identically to loading a standalone spec.
                    raw = raw_experiments.get(experiment_id)
                    if isinstance(raw, dict):
                        import yaml

                        exp = experiment.SimulationExperiment.from_string(
                            yaml.safe_dump(raw)
                        )
                    else:
                        from linkml_runtime.dumpers import yaml_dumper

                        exp = experiment.SimulationExperiment.from_string(
                            yaml_dumper.dumps(exp_dm)
                        )
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

        Returns
        -------
        dict
            OpenMINDS-compatible JSON-LD dictionary.

        Example
        -------
        >>> study = SimulationStudy.from_file("study.yaml")
        >>> jsonld = study.to_openminds()
        >>> study.to_openminds("output.jsonld", base_id="https://example.org")
        """
        from tvbo.adapters.openminds import study_to_openminds, save_openminds

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

        Returns
        -------
        SimulationStudy
            New instance constructed from the openMINDS data.

        Example
        -------
        >>> study = SimulationStudy.from_openminds("study.jsonld")
        >>> study = SimulationStudy.from_openminds({"@type": "tvbo:SimulationStudy", ...})
        """
        from tvbo.adapters.openminds import study_from_openminds, load_openminds

        if isinstance(source, str):
            # It's a file path
            data = load_openminds(source)
        elif isinstance(source, dict):
            data = study_from_openminds(source)
        else:
            raise TypeError(f"Expected str or dict, got {type(source)}")

        return cls(**data)


class StudyCollection(SimulationStudy, tvbo_datamodel.StudyCollection):
    """A whole manuscript as one runnable specification.

    Aggregates the member studies a paper reports (`members`) and owns the
    paper's own demonstration experiments, analyses and figures (inherited from
    `SimulationStudy`). `tvbo run` walks the members and the owned content, emits
    every reported number to a results manifest (`results`) and every figure with
    its composed caption, then packages the run as a COMBINE/OMEX archive
    (`archive`) — so the paper becomes an instance of the reproducibility it
    argues for. Load with `from_file(path)` (inherited).
    """

    def __repr__(self) -> str:
        title = self.title or "Untitled StudyCollection"
        n_members = len(getattr(self, "members", None) or [])
        n_results = len(getattr(self, "results", None) or [])
        n_figures = len(getattr(self, "figures", None) or [])
        return (
            f"StudyCollection(\n"
            f"  title={title!r},\n"
            f"  members={n_members}, results={n_results}, figures={n_figures}\n"
            f")"
        )

    def member_recipes(self, base=None, *, include_optional: bool = True) -> list[tuple[str, "Path"]]:
        """The member study recipes as ``(label, resolved_path)`` pairs.

        Each ``recipe`` is resolved relative to *base* (the StudyCollection file's
        directory) when it is not an IRI or an absolute path. ``optional`` members
        are dropped when *include_optional* is False (a ``--skip``-style light run).
        """
        from pathlib import Path

        base = Path(base) if base is not None else Path(getattr(self, "_source_file", ".")).resolve().parent
        out: list[tuple[str, Path]] = []
        for m in (getattr(self, "members", None) or []):
            if getattr(m, "optional", None) and not include_optional:
                continue
            recipe = str(getattr(m, "recipe", ""))
            label = getattr(m, "label", None) or Path(recipe).stem
            p = Path(recipe)
            resolved = p if p.is_absolute() or "://" in recipe else (base / recipe)
            out.append((str(label), resolved))
        return out
