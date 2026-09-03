"""Backend capability registry for the workflow planner.

This module is the **single source of truth in code** for what each TVB-O backend can do. The values here mirror the OWL axioms in ``ontology/tvb-o-axioms.ttl`` (§4.1). The mapping is intentionally typed as a plain Python table so the CLI does not pull in ``rdflib``/``owlready2`` at import time. A round-trip test (``tests/test_cli_backends_match_ontology.py``) keeps the two in sync when the ontology changes.

Ontology vocabulary used here
-----------------------------
* ``tvbo:Backend``                — JAX, TVB, PyRates, tvboptim,
  NetworkDynamics, BifurcationKit, NumPy.
* ``tvbo:SimulationTask``         — ODE/DDE/SDE/SDDE/RDEIntegration,
  EventDrivenIntegration, NumericalContinuation, BifurcationAnalysis, ParameterExploration, GradientBasedOptimization.
* ``tvbo:BackendCapability``      — Autodiff, JITCompilation, GPUSupport,
  VectorizedRNG, NumPyExecution, BuiltinModelLibrary, CodeGeneration, NetworkXTopology, JuliaJIT, DiffEqIntegrators, ContinuationSolver, DelayHistoryBuffer, StochasticSolver, StiffSolver.

The workflow planner (``tvbo.cli._workflow``) consults ``BACKENDS[name].vectorize_axes`` to decide which sweep axes can stay inside a single backend invocation (vmap / EnsembleProblem / batched solve) and which must be fanned out as workflow tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

# Sweep-axis kinds the planner understands. These are intentionally coarse: a study may declare richer axes (e.g. specific parameter names) but at the planner level we only care about *what kind* of axis it is so we can match it against backend capabilities.
AXIS_KINDS = (
    "parameters",  # any model / coupling / integrator parameter
    "initial_conditions",
    "noise_seed",
    "subjects",  # subject / sample axis (per-subject sims)
)


@dataclass(frozen=True)
class BackendSpec:
    """Static description of a backend's capabilities."""

    name: str  # canonical key, lowercase
    label: str  # human label
    tasks: frozenset[str]  # tvbo:SimulationTask member labels
    capabilities: frozenset[str]  # tvbo:BackendCapability member labels
    vectorize_axes: frozenset[str] = frozenset()  # axes the backend can pack inside one job
    aliases: tuple[str, ...] = ()

    def can_vectorize(self, axis_kind: str) -> bool:
        return axis_kind in self.vectorize_axes

    def supports_task(self, task: str) -> bool:
        return task in self.tasks


# Mirrors the backend declarations in ontology/tvb-o-axioms.ttl §4.1, vectorize-axis sets included.
BACKENDS: dict[str, BackendSpec] = {
    "jax": BackendSpec(
        name="jax",
        label="JAX",
        tasks=frozenset({"ODEIntegration", "SDEIntegration", "ParameterExploration", "GradientBasedOptimization"}),
        capabilities=frozenset({"Autodiff", "JITCompilation", "GPUSupport", "VectorizedRNG", "CodeGeneration"}),
        vectorize_axes=frozenset({"parameters", "initial_conditions", "noise_seed"}),
    ),
    "tvboptim": BackendSpec(
        name="tvboptim",
        label="tvboptim",
        tasks=frozenset({"ODEIntegration", "SDEIntegration", "GradientBasedOptimization"}),
        capabilities=frozenset({"Autodiff", "JITCompilation", "StochasticSolver", "CodeGeneration"}),
        vectorize_axes=frozenset({"parameters", "initial_conditions", "noise_seed", "subjects"}),
    ),
    "pyrates": BackendSpec(
        name="pyrates",
        label="PyRates",
        tasks=frozenset({"ODEIntegration", "DDEIntegration"}),
        capabilities=frozenset({"CodeGeneration", "NetworkXTopology", "DelayHistoryBuffer"}),
        vectorize_axes=frozenset({"parameters"}),
    ),
    "tvb": BackendSpec(
        name="tvb",
        label="TVB",
        tasks=frozenset({"ODEIntegration", "DDEIntegration", "SDEIntegration", "SDDEIntegration", "ParameterExploration"}),
        capabilities=frozenset({"NumPyExecution", "BuiltinModelLibrary", "DelayHistoryBuffer", "StochasticSolver"}),
        vectorize_axes=frozenset(),  # everything fans out
    ),
    "networkdynamics": BackendSpec(
        name="networkdynamics",
        label="NetworkDynamics.jl",
        tasks=frozenset({"ODEIntegration", "DDEIntegration", "SDEIntegration", "SDDEIntegration", "ParameterExploration"}),
        capabilities=frozenset({"JuliaJIT", "DiffEqIntegrators", "DelayHistoryBuffer", "StochasticSolver", "StiffSolver"}),
        vectorize_axes=frozenset({"parameters"}),
        aliases=("nd",),
    ),
    "bifurcationkit": BackendSpec(
        name="bifurcationkit",
        label="BifurcationKit.jl",
        tasks=frozenset({"NumericalContinuation", "BifurcationAnalysis"}),
        capabilities=frozenset({"ContinuationSolver", "JuliaJIT"}),
        vectorize_axes=frozenset({"parameters"}),
    ),
    "numpy": BackendSpec(
        name="numpy",
        label="NumPy",
        tasks=frozenset({"ODEIntegration", "SDEIntegration"}),
        capabilities=frozenset({"NumPyExecution"}),
        vectorize_axes=frozenset(),
    ),
    "brian2": BackendSpec(
        name="brian2",
        label="Brian2",
        tasks=frozenset({"ODEIntegration", "EventDrivenIntegration"}),
        capabilities=frozenset({"SpikingSimulation", "NumPyExecution", "CodeGeneration"}),
        vectorize_axes=frozenset(),  # a sweep fans out into per-cell runs
        aliases=("brian",),
    ),
}


def resolve_backend(name: str) -> BackendSpec:
    """Look up a backend by canonical key or alias."""
    key = name.lower()
    if key in BACKENDS:
        return BACKENDS[key]
    for spec in BACKENDS.values():
        if key in spec.aliases:
            return spec
    raise ValueError(f"Unknown backend {name!r}. Known: {', '.join(sorted(BACKENDS))}.")


def list_backends() -> list[BackendSpec]:
    return list(BACKENDS.values())


def axis_kind_of(parameter_path: str) -> str:
    """Classify an exploration axis by its dotted parameter path.

    Examples:
    --------
    >>> axis_kind_of("ReducedWongWang.G")
    'parameters'
    >>> axis_kind_of("integrator.noise_seed")
    'noise_seed'
    >>> axis_kind_of("initial_conditions.x")
    'initial_conditions'
    >>> axis_kind_of("sample.subject_id")
    'subjects'
    """
    p = parameter_path.lower()
    if "noise_seed" in p or p.endswith(".seed"):
        return "noise_seed"
    if "initial_condition" in p or p.startswith("initial_conditions"):
        return "initial_conditions"
    if "subject" in p or "sample" in p:
        return "subjects"
    return "parameters"
