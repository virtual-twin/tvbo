"""What each backend can do, as the workflow planner reads it.

`BACKENDS` is the table `tvbo.run.workflow.plan` consults: `BackendSpec.vectorize_axes` decides which sweep axes stay inside one backend invocation (vmap, EnsembleProblem, batched solve) and which a workflow engine fans out as separate tasks, and `tvbo run --shard` slices only the axes a backend vectorises. It is a plain Python table, so importing it pulls in neither ``rdflib`` nor ``owlready2``.

Task and capability names are the ``tvbo:SimulationTask`` and ``tvbo:BackendCapability`` individuals of ``ontology/tvb-o-axioms.ttl`` (section 4.1). The table is written by hand rather than read from that file, and the ontology declares no vectorisable axes, so the ontology is the vocabulary here and not the source of the values.
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
        """Whether a sweep axis of *axis_kind* (one of `AXIS_KINDS`) can stay inside one invocation of this backend."""
        return axis_kind in self.vectorize_axes

    def supports_task(self, task: str) -> bool:
        """Whether this backend performs *task*, a ``tvbo:SimulationTask`` name."""
        return task in self.tasks


# Task and capability names are ontology/tvb-o-axioms.ttl §4.1 individuals; vectorize_axes are the planner's own.
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


def effective_backend(experiment, requested: str | None = None) -> str:
    """The backend that runs *experiment*: *requested* when given, else its declared ``execution.backend``, else ``tvboptim``.

    An explicit ``--backend`` wins for the whole run; otherwise each experiment self-selects (a spiking network declares ``brian2``), so one study can mix a mean-field sweep and a spiking column and run each on the right engine.
    """
    return requested or getattr(getattr(experiment, "execution", None), "backend", None) or "tvboptim"


def list_backends() -> list[BackendSpec]:
    """Every registered backend, in registration order."""
    return list(BACKENDS.values())


def axis_kind_of(parameter_path: str) -> str:
    """Classify an exploration axis by its dotted parameter path.

    Examples:
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
