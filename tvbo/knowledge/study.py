from copy import deepcopy

from linkml_runtime.loaders import yaml_loader

from tvbo.datamodel import tvbo_datamodel
from tvbo.export import experiment, report
from tvbo.knowledge import Dynamics


class SimulationStudy(tvbo_datamodel.SimulationStudy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = Dynamics.from_datamodel(self.model) if self.model else None

    def __repr__(self) -> str:
        md = self
        key = md.key or "?"
        title = md.title or "Untitled Study"
        year = getattr(md, "year", "n.d.")
        doi = md.doi or "n/a"
        n_exp = len(getattr(md, "simulation_experiments", []) or [])
        model_name = getattr(getattr(md, "model", None), "name", None) or "n/a"
        dyns = sorted(
            {
                d
                for exp in (getattr(md, "simulation_experiments", []) or [])
                for d in (getattr(exp, "dynamics", []) or [])
            }
        )
        dyn_str = ", ".join(dyns) if dyns else "—"

        # Multi-line but compact formatting
        return (
            f"SimulationStudy(\n"
            f"  key={key!r},\n"
            f"  title={title!r},\n"
            f"  year={year}, doi={doi!r},\n"
            f"  model={model_name!r}, experiments={n_exp},\n"
            f"  dynamics=[{dyn_str}]\n"
            f")"
        )

    @classmethod
    def from_file(cls, filepath):
        return yaml_loader.load(filepath, cls)

    @classmethod
    def from_datamodel(cls, datamodel: tvbo_datamodel.SimulationStudy):
        return cls(**datamodel._as_dict)

    def cite(self):
        return report.get_citation(self.key)

    def update_experiment_model(self, exp):
        """
        Ensure each experiment has a concrete model instance by filling only missing
        fields from the study-level base model. This keeps experiment models independent
        and avoids metadata duplication.
        """

        base_model = getattr(self, "model", None)
        if base_model is None:
            # Nothing to inherit
            return

        # If experiment has no model, clone the base model
        if getattr(exp, "model", None) is None:
            exp.model = deepcopy(base_model)
            return

        # Merge only missing fields using the dict representations
        base_dict = deepcopy(base_model._as_dict)
        exp_dict = deepcopy(exp.model._as_dict)

        def fill_missing(target: dict, source: dict):
            for k, sv in source.items():
                if k not in target or target[k] in (None, {}, [], ""):
                    target[k] = deepcopy(sv)
                else:
                    tv = target[k]
                    if isinstance(tv, dict) and isinstance(sv, dict):
                        fill_missing(tv, sv)
                    elif isinstance(tv, list) and isinstance(sv, list):
                        if not tv:
                            target[k] = deepcopy(sv)
                    # else: keep existing explicit override
            return target

        merged_dict = fill_missing(exp_dict, base_dict)

        # Reconstruct an instance of the same class with merged content
        exp.model = type(exp.model)(**merged_dict)

    @property
    def experiments(self):
        """
        Computed mapping of experiment id -> SimulationExperiment instance with
        the study's base model merged into each experiment (non-metadata runtime view).
        """
        exps = {}
        for exp in getattr(self, "simulation_experiments", []) or []:
            self.update_experiment_model(exp)
            exps[exp.id] = experiment.SimulationExperiment.from_datamodel(exp)
        return exps

    def get_experiment(self, experiment_id):
        """
        Retrieve a single experiment strictly by its declared id,
        ensuring the model inheritance/override is applied.
        """
        exp_dm = None
        exps = getattr(self, "simulation_experiments", []) or []

        # Lookup by declared id only (no index fallback)
        for e in exps:
            if getattr(e, "id", None) == experiment_id:
                exp_dm = e
                break

        if exp_dm is None:
            raise KeyError(f"Experiment {experiment_id!r} not found")

        # Apply base model merge
        self.update_experiment_model(exp_dm)
        return experiment.SimulationExperiment.from_datamodel(exp_dm)
