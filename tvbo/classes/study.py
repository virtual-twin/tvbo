from tvbo.utils import yaml_loader

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
        from pathlib import Path
        study = yaml_loader.load(filepath, cls)
        study._source_file = str(Path(filepath).resolve())
        return study

    @classmethod
    def from_datamodel(cls, datamodel: tvbo_datamodel.SimulationStudy):
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
        return report.get_citation(self.key)

    def get_experiment(self, experiment_id):
        """Retrieve a single experiment by its declared id."""
        exps = getattr(self, "experiments", None) or []
        source_file = getattr(self, "_source_file", None)
        for exp_dm in exps:
            if getattr(exp_dm, "id", None) == experiment_id:
                if source_file:
                    experiment.SimulationExperiment._pending_source_file = source_file
                try:
                    exp = experiment.SimulationExperiment.from_datamodel(exp_dm)
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
