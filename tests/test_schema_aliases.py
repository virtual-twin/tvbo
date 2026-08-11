"""Schema-declared ``aliases:`` work at load time, and only where they apply.

LinkML ``aliases:`` are metadata — its loaders key on the canonical slot name — so a
declared alias is inert and raises ``unexpected keyword argument`` unless something
resolves it. Resolution happens in each generated class's ``__init__``
(``hatch_build._alias_support``), where the kwargs are known to belong to that class:
no document traversal, and a free-form key can never be mistaken for a slot.
"""

import pytest

from tvbo import Dynamics, Network, SimulationExperiment
from tvbo.datamodel.dialect import SCALAR_SHORTCUTS, SLOT_ALIASES

_BASE = "id: 1\nlabel: t\ndynamics: {name: Generic2dOscillator}\n"


def _declared_aliases():
    """``[(class name, alias, canonical slot)]`` for every alias in the datamodel."""
    import inspect

    from pydantic import BaseModel

    from tvbo.datamodel import pydantic as dm

    out = []
    for obj in vars(dm).values():
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
            for slot, field in obj.model_fields.items():
                extra = field.json_schema_extra or {}
                meta = extra.get("linkml_meta", {}) if isinstance(extra, dict) else {}
                for alias in meta.get("aliases", []) or []:
                    if alias != slot:
                        out.append((obj.__name__, alias, slot))
    return out


# ── completeness: no declared alias is silently inert ────────────────


# `range`/`boundaries` -> `domain` carry extra semantics (clamp, distribution) and are
# resolved by yaml_loader._fold_state_variable_domains instead.
_SEMANTIC = {"range", "boundaries"}


def test_every_declared_alias_is_resolvable():
    """Each ``(class, alias)`` pair is either folded by that class or owned by a
    dedicated pass — never simply ignored.

    Keyed on the pair, not the alias name: ``range`` is exempt on ``StateVariable``
    but that must not silently exempt it everywhere else.
    """
    unresolved = [
        (cls, alias)
        for cls, alias, slot in _declared_aliases()
        if alias not in _SEMANTIC and SLOT_ALIASES.get(cls, {}).get(alias) != slot
    ]
    assert not unresolved, f"declared but never resolved: {unresolved}"


def test_semantic_aliases_are_never_plain_renamed():
    """``boundaries`` reaching the generic fold would strip the ``enforce: clamp``
    that its own pass adds."""
    assert not any(set(amap) & _SEMANTIC for amap in SLOT_ALIASES.values())


def test_an_alias_is_scoped_to_the_class_that_declares_it():
    """``target_variable`` aliases ``Edge.target_var`` but is canonical on ``Event``,
    so it must fold for one and not the other."""
    assert SLOT_ALIASES["Edge"]["target_variable"] == "target_var"
    assert "target_variable" not in SLOT_ALIASES.get("Event", {})


# ── scoping: a fold applies to its class, and nothing else ───────────


def test_a_user_key_that_collides_with_an_alias_is_left_alone():
    """The reason the fold is class-scoped rather than context-free.

    ``dt`` aliases ``Integrator.step_size`` and ``components`` aliases
    ``Dynamics.modes``, but both are ordinary strings a recipe may use as a parameter
    name or a free-form key. A context-free rename silently rewrites those.
    """
    exp = SimulationExperiment.from_string(
        _BASE + "integration: {dt: 0.05}\n"
        "network: {number_of_nodes: 1, nodes: [{id: 0}]}\n"
    )
    assert exp.integration.step_size == 0.05


def test_a_model_parameter_named_dt_survives_a_real_load():
    dyn = Dynamics.from_string(
        "name: M\nparameters: {dt: {value: 0.25}}\n"
        "state_variables:\n  x: {equation: {rhs: '-dt*x'}, initial_value: 0.1}\n"
    )
    assert "dt" in dyn.parameters and dyn.parameters["dt"].value == 0.25


# ── the aliases that were declared but inert ─────────────────────────


def test_dt_is_accepted_for_integrator_step_size():
    exp = SimulationExperiment.from_string(_BASE + "integration: {dt: 0.05}")
    assert exp.integration.step_size == 0.05


def test_righthandside_is_accepted():
    dyn = Dynamics.from_string(
        "name: M\nstate_variables:\n  x: {equation: {righthandside: '-x'}, initial_value: 0.1}\n"
    )
    assert dyn.state_variables["x"].equation.rhs == "-x"


def test_number_of_regions_is_accepted():
    assert Network(**{"number_of_regions": 3}).number_of_nodes == 3


# ── the scalar shortcut also lifts ARRAY literals ────────────────────


def test_scalar_shortcut_lifts_an_array_literal():
    """``sel: {time: [0.006, 0.016]}`` is a coordinate LIST, and the slot it lifts into
    holds arrays as well as scalars. Lifting only scalars left the list to be built as an
    Argument positionally, where it landed in ``description`` and the selection silently
    vanished — a sourced argument then arrived unsliced."""
    from tvbo.datamodel.schema import DataRef

    ref = DataRef(experiment="1", output="integration",
                  sel={"variable": "phi", "time": [0.006, 0.016]})
    assert ref.sel["variable"].value == "phi"
    assert list(ref.sel["time"].value) == [0.006, 0.016]
    assert ref.sel["time"].description is None


def test_scalar_shortcut_leaves_a_collection_list_alone():
    """A list of MAPPINGS is the list spelling of a keyed collection, not a literal."""
    from tvbo.datamodel.dialect import lift_scalar

    members = [{"name": "a", "value": 1}, {"name": "b", "value": 2}]
    assert lift_scalar(members, "value", True) == members
    assert lift_scalar([[1, 2], [3, 4]], "value", False) == {"value": [[1, 2], [3, 4]]}


def test_scalar_shortcut_keeps_keyed_list_scalars_as_identifiers():
    """``arguments: [v]`` is the list spelling of a NAME-KEYED collection, so ``v`` is the
    argument's name — not a value to wrap. Lifting it to ``{value: v}`` mislabelled ``v`` as
    ``value`` and stranded the real name in ``description``, generating ``def Sigm(value)``
    with a body that still referenced ``v`` (``NameError: name 'v' is not defined``). A
    non-keyed list (``additional_equations``) still lifts each element."""
    from tvbo.datamodel.dialect import lift_scalar
    from tvbo.datamodel.schema import Function

    fn = Function(name="Sigm", arguments=["v"])
    assert list(fn.arguments) == ["v"]
    assert fn.arguments["v"].name == "v" and fn.arguments["v"].value is None

    assert lift_scalar(["v"], "value", True, keyed=True) == ["v"]
    assert lift_scalar(["x = -x"], "rhs", True, keyed=False) == [{"rhs": "x = -x"}]


# ── the collisions still resolve to the right slot ───────────────────


def test_stimulus_event_target_variable_is_not_rewritten():
    """An ``Event``'s ``target_variable`` is canonical, while it aliases ``Edge.target_var``."""
    exp = SimulationExperiment.from_string(
        _BASE + "events: {s: {event_type: stimulus, target_variable: y0, regions: [0], weighting: [1.0]}}"
    )
    ev = exp.events["s"]
    assert ev.target_variable == "y0"
    assert list(ev.nodes) == [0] and [float(w) for w in ev.weights] == [1.0]


def test_edge_source_and_target_variable_fold_to_the_edge_slots():
    net = SimulationExperiment.from_string(
        _BASE + "network: {number_of_nodes: 2, edges: [{source: 0, target: 1, "
        "source_variable: V, target_variable: W}]}"
    ).network
    assert (net.edges[0].source_var, net.edges[0].target_var) == ("V", "W")


def test_boundaries_still_implies_clamp_and_domain_still_does_not():
    """Clamping is never a default: only ``enforce: clamp`` and the legacy
    ``boundaries`` spelling (a hard clamp in TVB) constrain a trajectory."""
    from tvbo.utils import domain_enforcement

    def enforce(spec):
        d = Dynamics.from_string(
            "name: M\nstate_variables:\n  x: {equation: {rhs: '-x'}, initial_value: 0.1, " + spec + "}\n"
        )
        return domain_enforcement(d.state_variables["x"].domain)

    assert enforce("domain: {lo: 0.0, hi: 1.0}") == "none"
    assert enforce("range: {lo: 0.0, hi: 1.0}") == "none"
    assert enforce("boundaries: {lo: 0.0, hi: 1.0}") == "clamp"
    assert enforce("domain: {lo: 0.0, hi: 1.0, enforce: clamp}") == "clamp"


# ── one `dt`, one meaning ────────────────────────────────────────────


def test_pde_solver_is_a_solver_and_spells_its_step_the_same_way():
    """``PDESolver.dt`` used to be its own slot, so ``dt`` named two different things."""
    from tvbo.datamodel.schema import PDESolver, Solver

    assert issubclass(PDESolver, Solver)
    fields = set(PDESolver.__dataclass_fields__)
    assert {"step_size", "method", "abs_tol", "rel_tol"} <= fields
    assert not ({"dt", "time_integrator", "tolerances"} & fields)
    assert PDESolver(step_size=0.001).step_size == 0.001


def test_conflicting_alias_and_canonical_keeps_the_canonical():
    with pytest.warns(UserWarning, match="both 'dt' and its canonical"):
        exp = SimulationExperiment.from_string(_BASE + "integration: {dt: 0.05, step_size: 0.01}")
    assert exp.integration.step_size == 0.01


# ── the same dialect folds on the pydantic validation path ────────────


def _pyd(yaml_text, target="SimulationExperiment"):
    """Load *yaml_text* through the strict pydantic models.

    The dataclasses fold the dialect in ``__init__``; the pydantic models
    (``extra='forbid'``) fold it in a ``mode="before"`` validator. Both call the one
    implementation in ``tvbo.datamodel.dialect``, so the validator cannot reject a
    document the dataclass loader accepts. It could before: the two paths carried
    separate copies, and the pydantic copy had the aliases but not the scalar shortcuts.
    """
    from tvbo.utils import pydantic_loader

    return pydantic_loader.loads(yaml_text, target_class=target)


def test_pydantic_loader_accepts_dt_righthandside_and_number_of_regions():
    exp = _pyd(
        _BASE + "integration: {dt: 0.05}\n"
        "network: {number_of_regions: 1, nodes: [{id: 0}]}\n"
    )
    assert exp.integration.step_size == 0.05
    assert exp.network.number_of_nodes == 1


def test_pydantic_loader_folds_are_class_scoped():
    """Same scoping as the dataclass path: ``Edge`` folds ``source_variable`` /
    ``target_variable``, a stimulus ``Event`` keeps ``target_variable`` canonical."""
    exp = _pyd(
        _BASE + "network: {number_of_nodes: 2, edges: [{source: 0, target: 1, "
        "source_variable: V, target_variable: W}]}\n"
        "events: {s: {event_type: stimulus, target_variable: y0, regions: [0], weighting: [1.0]}}"
    )
    assert (exp.network.edges[0].source_var, exp.network.edges[0].target_var) == ("V", "W")
    assert exp.events["s"].target_variable == "y0"


def test_pydantic_loader_conflict_keeps_the_canonical():
    with pytest.warns(UserWarning, match="both 'dt' and its canonical"):
        exp = _pyd(_BASE + "integration: {dt: 0.05, step_size: 0.01}")
    assert exp.integration.step_size == 0.01


def test_pydantic_validator_folds_every_alias_the_dataclass_loader_does():
    """Parity guard: every ``SLOT_ALIASES`` entry the dataclass path folds is also
    folded on the pydantic path, so the validator never rejects a loader-valid key."""
    from tvbo.utils import pydantic_loader

    for cls, amap in SLOT_ALIASES.items():
        for alias, canonical in amap.items():
            folded = pydantic_loader.normalize({alias: "x"}, cls)
            assert alias not in folded and canonical in folded, (cls, alias, canonical)


def test_pydantic_lifts_every_scalar_shortcut_the_dataclass_loader_does():
    """Parity guard for the other half of the dialect.

    The aliases had this guard and stayed in step; the scalar shortcuts had none and
    silently diverged — they were applied only on the dataclass path, so ``omega: 0.0628``
    loaded through one entry point and was rejected by the other.
    """
    from tvbo.datamodel import pydantic as dm
    from tvbo.utils import pydantic_loader

    for cls, lifts in SCALAR_SHORTCUTS.items():
        if getattr(dm, cls, None) is None:
            continue
        for slot, (target, multivalued, _keyed) in lifts.items():
            if multivalued:
                continue  # collections lift per member; covered by the keyed-dict tests
            assert pydantic_loader.normalize({slot: "x"}, cls)[slot] == {target: "x"}, (
                cls,
                slot,
                target,
            )


def test_a_bare_scalar_lifts_on_both_paths():
    """The dialect the README leads with: ``omega: 0.0628`` is a Parameter of that value."""
    from tvbo.datamodel import schema
    from tvbo.utils import yaml_loader

    src = "name: probe\nparameters: {omega: 0.0628}\n"
    assert yaml_loader.loads(src, schema.Dynamics).parameters["omega"].value == 0.0628
    assert _pyd(src, "Dynamics").parameters["omega"].value == 0.0628


def test_the_model_validator_folds_the_dialect_without_the_loader():
    """``model_validate`` alone folds it: the dialect rides on the models, not the loader.

    This is what lets the platform hand a raw dict straight to a generated model.
    """
    from tvbo.datamodel.pydantic import Equation, Solver

    assert Solver.model_validate({"dt": 0.01}).step_size == 0.01
    equation = Equation.model_validate({"lefthandside": "x", "righthandside": "y"})
    assert (equation.lhs, equation.rhs) == ("x", "y")
