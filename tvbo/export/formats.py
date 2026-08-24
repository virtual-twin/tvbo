"""Built-in export-format registrations.

Importing this module populates :mod:`tvbo.export.registry` with every backend that ships with TVBO. Third-party packages can register their own formats the same way (preferably in their own module's import-time code).

Each renderer is a thin closure ``(experiment, **kwargs) -> str``. Heavy adapter imports happen *inside* the closure so that simply listing formats does not pull in optional dependencies (Mako templates, NeuroML, …).
"""

from __future__ import annotations

from .registry import ExportFormat, register

# Serialisation (LinkML YAML / openMINDS JSON-LD)


def _render_yaml(exp, **kw) -> str:
    return exp.to_yaml(filepath=kw.get("filepath"))


def _render_pyrates_yaml(exp, **kw) -> str:
    return exp.to_yaml(filepath=kw.get("filepath"), format="pyrates")


def _render_openminds(exp, **kw) -> str:
    import json

    from tvbo.adapters.openminds import experiment_to_openminds

    indent = kw.pop("indent", 2)
    return json.dumps(experiment_to_openminds(exp, **kw), indent=indent, default=str)


# Reports (markdown / pdf)


def _render_markdown(exp, **kw) -> str:
    return exp.report(format="markdown", **kw)


def _render_pdf(exp, **kw) -> str:
    # report() writes the PDF to *outputfile*; return value is the path/empty
    return exp.report(format="pdf", **kw) or ""


# Code generation (delegates to existing render_code branches via templates)


def _render_tvb(exp, **kw):
    from tvbo.classes.experiment import templates

    template = templates.lookup.get_template("tvbo-tvb-SimulationExperiment.py.mako")
    return template.render(experiment=exp, **kw)


def _render_jax(exp, **kw):
    from tvbo.adapters.observation_sampling import resolve_observation_sampling
    from tvbo.classes.experiment import templates

    template = templates.lookup.get_template("autodiff/tvbo-jax-sim.py.mako")
    # The same shared resolver the tvboptim runtime uses, so every Python backend agrees.
    observations = getattr(exp, "observations", None) or {}
    dt = exp.integration.step_size
    kw.setdefault(
        "obs_sampling",
        {name: resolve_observation_sampling(obs, dt) for name, obs in observations.items()},
    )
    return template.render(experiment=exp, **kw)


def _render_tvboptim(exp, **kw):
    from tvbo.adapters.tvboptim import TvboptimAdapter
    from tvbo.classes.experiment import templates

    template = templates.lookup.get_template("tvboptim/tvbo-tvboptim-experiment.py.mako")
    # Which couplings this experiment has is the adapter's answer, not the template's.
    kw.setdefault("all_couplings", TvboptimAdapter(exp).resolve_couplings())
    # Resolve network- and dataset-sourced observation pointers once, in Python, and hand the {obs_name: measure} mapping to the template (which only emits code). Both bind at run_experiment time via _bind_network_observations.
    measures = dict(exp.network_observation_measures)
    measures.update(exp.dataset_observation_targets)
    kw.setdefault("network_obs_measures", measures)
    # Model-side gather (keyed by label, never positional) that aligns a simulated observable to a by_label empirical target's shared nodes in the loss.
    kw.setdefault("dataset_reconcile_indices", exp.dataset_reconcile_indices())
    return template.render(experiment=exp, **kw)


def _render_pde(exp, **kw):
    from tvbo.classes.experiment import templates

    template = templates.lookup.get_template("tvbo-pde-fem.py.mako")
    return template.render(experiment=exp, **kw)


def _render_julia(exp, **kw):
    from tvbo.classes.experiment import templates

    template = templates.lookup.get_template("tvbo-julia-DifferentialEquations.jl.mako")
    return template.render(experiment=exp, model=exp.dynamics, **kw)


def _render_networkdynamics(exp, **kw):
    from tvbo.adapters.networkdynamics import NetworkDynamicsAdapter

    return NetworkDynamicsAdapter(exp).render_code(**kw)


def _render_modelingtoolkit(exp, **kw):
    from tvbo.adapters.modelingtoolkit import ModelingToolkitAdapter

    return ModelingToolkitAdapter(exp).render_code(**kw)


def _render_bifurcationkit(exp, **kw):
    from tvbo.adapters.bifurcationkit import BifurcationKitAdapter

    return BifurcationKitAdapter(exp).render_code(**kw)


def _render_pyrates_bifurcation(exp, **kw):
    from tvbo.adapters.pyrates_bifurcation import PyRatesBifurcationAdapter

    return PyRatesBifurcationAdapter(exp).render_code(**kw)


def _render_neuroml(exp, **kw):
    from tvbo.adapters.neuroml import NeuroMLAdapter

    kw.setdefault("use_standard_types", True)
    return NeuroMLAdapter(exp).render_code(**kw)


def _render_lems(exp, **kw):
    from tvbo.adapters.neuroml import NeuroMLAdapter

    kw.setdefault("use_standard_types", False)
    return NeuroMLAdapter(exp).render_code(**kw)


def _render_brian2(exp, **kw):
    from tvbo.adapters.brian2 import Brian2Adapter

    return Brian2Adapter(exp).render_code(**kw)


def _render_rateml_python(exp, **kw):
    from tvbo.classes.experiment import templates

    template = templates.lookup.get_template("rateml/tvbo-rateml-python.py.mako")
    return template.render(model=exp.dynamics, experiment=exp, **kw)


def _render_rateml_cuda(exp, **kw):
    from tvbo.classes.experiment import templates

    template = templates.lookup.get_template("rateml/tvbo-rateml-cuda.c.mako")
    return template.render(model=exp.dynamics, experiment=exp, **kw)


def _render_rateml_driver(exp, **kw):
    from tvbo.classes.experiment import templates

    template = templates.lookup.get_template("rateml/tvbo-rateml-driver.py.mako")
    return template.render(model=exp.dynamics, experiment=exp, **kw)


# Registration

_BUILTINS = [
    # serialisation
    ExportFormat(
        "yaml",
        "TVBO YAML",
        ".yaml",
        "application/x-yaml",
        _render_yaml,
        aliases=("tvbo", "tvbo-yaml"),
        supports_with_data=True,
    ),
    # No language: `to_yaml(filepath=...)` writes the file, so normalising would diverge.
    ExportFormat(
        "pyrates-yaml", "PyRates YAML", ".yaml", "application/x-yaml", _render_pyrates_yaml, aliases=("pyrates_yaml",)
    ),
    ExportFormat(
        "openminds", "openMINDS JSON-LD", ".jsonld", "application/ld+json", _render_openminds, aliases=("jsonld", "json-ld")
    ),
    # reports
    ExportFormat("markdown", "Markdown report", ".md", "text/markdown", _render_markdown, aliases=("md", "report")),
    ExportFormat("pdf", "PDF report", ".pdf", "application/pdf", _render_pdf),
    # standards
    ExportFormat(
        "neuroml",
        "NeuroML (standard IRI components)",
        ".nml",
        "application/xml",
        _render_neuroml,
        aliases=("nml",),
        language="xml",
    ),
    ExportFormat("lems", "LEMS (custom components)", ".xml", "application/xml", _render_lems, language="xml"),
    # code: python
    ExportFormat("tvb", "TVB Python", ".py", "text/x-python", _render_tvb, language="python"),
    ExportFormat(
        "tvboptim", "tvboptim Python", ".py", "text/x-python", _render_tvboptim, aliases=("tvb-optim",), language="python"
    ),
    ExportFormat("jax", "JAX Python", ".py", "text/x-python", _render_jax, aliases=("autodiff",), language="python"),
    ExportFormat(
        "pde", "PDE-FEM Python", ".py", "text/x-python", _render_pde, aliases=("pde-fem", "pde-python"), language="python"
    ),
    ExportFormat(
        "brian2", "Brian2 Python (spiking)", ".py", "text/x-python", _render_brian2, aliases=("brian",), language="python"
    ),
    # code: julia
    ExportFormat(
        "julia",
        "Julia (DifferentialEquations.jl)",
        ".jl",
        "text/plain",
        _render_julia,
        aliases=("diffeq", "differentialequations"),
        language="julia",
    ),
    ExportFormat(
        "networkdynamics",
        "NetworkDynamics.jl",
        ".jl",
        "text/plain",
        _render_networkdynamics,
        aliases=("nd", "networkdynamics.jl"),
        language="julia",
    ),
    ExportFormat(
        "modelingtoolkit",
        "ModelingToolkit.jl",
        ".jl",
        "text/plain",
        _render_modelingtoolkit,
        aliases=("mtk", "modelingtoolkit.jl"),
        language="julia",
    ),
    ExportFormat(
        "bifurcationkit",
        "BifurcationKit.jl",
        ".jl",
        "text/plain",
        _render_bifurcationkit,
        aliases=("bifurcationkit.jl", "bifurcation", "bifurcation-julia"),
        language="julia",
    ),
    ExportFormat(
        "pyrates-bifurcation",
        "PyRates / AUTO-07p bifurcation",
        ".py",
        "text/x-python",
        _render_pyrates_bifurcation,
        aliases=("pyrates-bif", "pycobi", "bifurcation-pyrates", "auto", "auto-07p"),
        language="python",
    ),
    # code: rateml
    ExportFormat(
        "rateml",
        "RateML Python (Numba gufunc)",
        ".py",
        "text/x-python",
        _render_rateml_python,
        aliases=("rateml-python",),
        language="python",
    ),
    ExportFormat("rateml-cuda", "RateML CUDA kernel", ".c", "text/x-c", _render_rateml_cuda, aliases=("cuda",), language="c"),
    ExportFormat("rateml-driver", "RateML PyCUDA driver", ".py", "text/x-python", _render_rateml_driver, language="python"),
]


def _register_builtins() -> None:
    for fmt in _BUILTINS:
        register(fmt, overwrite=True)


_register_builtins()
