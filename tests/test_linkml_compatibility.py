"""LinkML compatibility-contract smoke tests.

Per `dev/OntologicalRestructuring/plan.md` §2.8 (Phase 0 governance baseline): these tests guard the 13 generated LinkML classes that are subclassed by runtime knowledge classes. Removing or renaming any of these (or their critical fields) silently breaks YAML loading and the tvbo-platform Pydantic contract.

Run:

    pytest tests/test_linkml_compatibility.py -x -n0 -v

If any test fails, the LinkML schema change is in the FORBIDDEN category (per main proposal Appendix E) and must be reverted or migrated with deprecation aliases.
"""

import importlib

import pytest

# Generated dataclass module -> list of class names that runtime code subclasses.
SUBCLASSED_GENERATED = {
    "tvbo.datamodel.tvbo_datamodel": [
        "Dynamics",
        "Coupling",
        "Integrator",
        "Noise",
        "Function",
        "LossFunction",
        "Stimulus",
        "Continuation",
        "SimulationExperiment",
        "SimulationStudy",
        "Network",
        "BrainAtlas",
    ],
}

# Pydantic mirror module - used by tvbo-platform.
PYDANTIC_REQUIRED = [
    "Dynamics",
    "Coupling",
    "Integrator",
    "Network",
    "SimulationExperiment",
    "SimulationStudy",
]

CRITICAL_FIELDS = {
    "Dynamics": ["name", "parameters", "state_variables", "derived_variables"],
    "Parameter": ["name"],
    "StateVariable": ["name"],
    "Coupling": ["name"],
    "Integrator": ["method"],
    "SimulationExperiment": ["id"],
    "SimulationStudy": ["key"],
}
"""Fields that MUST exist on the generated dataclass for the runtime ``__post_init__()`` ``_normalize_inlined_as_dict`` calls (main proposal Appendix A.2). Changing the ``name`` key here silently drops YAML data on load."""


@pytest.mark.parametrize(
    "module,name",
    [(m, n) for m, names in SUBCLASSED_GENERATED.items() for n in names],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_generated_class_importable(module, name):
    """Every subclassed LinkML class must be importable."""
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{module} is missing class {name!r}"
    cls = getattr(mod, name)
    assert isinstance(cls, type), f"{module}.{name} is not a class"


@pytest.mark.parametrize("name", PYDANTIC_REQUIRED)
def test_pydantic_class_importable(name):
    """tvbo-platform imports these Pydantic mirrors. They must exist."""
    mod = importlib.import_module("tvbo.datamodel.tvbopydantic")
    assert hasattr(mod, name), f"tvbo.datamodel.tvbopydantic missing {name!r}"


@pytest.mark.parametrize("name,fields", list(CRITICAL_FIELDS.items()))
def test_critical_fields_present(name, fields):
    """LinkML _normalize_inlined_as_dict and runtime access depend on these."""
    mod = importlib.import_module("tvbo.datamodel.tvbo_datamodel")
    cls = getattr(mod, name)
    annotations = getattr(cls, "__annotations__", {}) or {}
    # Walk MRO to tolerate inherited fields.
    declared = set(annotations)
    for base in cls.__mro__:
        declared |= set(getattr(base, "__annotations__", {}) or {})
    missing = [f for f in fields if f not in declared]
    assert not missing, f"{name} is missing critical fields: {missing}"


def test_runtime_subclasses_resolve():
    """Each runtime subclass must still inherit from its LinkML generated parent."""
    pairs = [
        ("tvbo.classes.dynamics", "Dynamics", "Dynamics"),
        ("tvbo.classes.coupling", "Coupling", "Coupling"),
        ("tvbo.classes.noise", "Integrator", "Integrator"),
        ("tvbo.classes.noise", "Noise", "Noise"),
        ("tvbo.classes.function", "Function", "Function"),
        ("tvbo.classes.function", "LossFunction", "LossFunction"),
        ("tvbo.classes.observation", "Function", "Function"),
        ("tvbo.classes.perturbation", "Stimulus", "Stimulus"),
        ("tvbo.classes.continuation", "Continuation", "Continuation"),
        ("tvbo.classes.experiment", "SimulationExperiment", "SimulationExperiment"),
        ("tvbo.classes.study", "SimulationStudy", "SimulationStudy"),
        ("tvbo.classes.network", "Network", "Network"),
        ("tvbo.classes.atlas", "Atlas", "BrainAtlas"),
    ]
    base_mod_name = "tvbo.datamodel.tvbo_datamodel"
    bm = importlib.import_module(base_mod_name)
    failures = []
    for sub_mod, sub_name, base_name in pairs:
        try:
            sm = importlib.import_module(sub_mod)
        except ImportError as e:
            failures.append(f"import failed for {sub_mod}: {e}")
            continue
        if not hasattr(sm, sub_name):
            failures.append(f"{sub_mod} missing {sub_name}")
            continue
        if not hasattr(bm, base_name):
            failures.append(f"{base_mod_name} missing {base_name}")
            continue
        sub = getattr(sm, sub_name)
        base = getattr(bm, base_name)
        if not issubclass(sub, base):
            failures.append(f"{sub_mod}.{sub_name} no longer inherits from {base_mod_name}.{base_name}")
    assert not failures, "\n".join(failures)
