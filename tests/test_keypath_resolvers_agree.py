"""The two keypath resolvers must answer alike, and neither may grow a scope alone.

`parameter_keypath` classifies a reference still held as raw dotted text; `axis_keypath` reads
an exploration axis whose scope the classifier already resolved into flags. They are reached
from different call paths — a fitted free parameter, an inference prior and a working-point ramp
go through the first, the grid binding and the adiabatic sweep through the second — so a
divergence between them is silent by construction: the reference lands on a same-named leaf in
the wrong sub-object, or on a leaf nothing reads, and the run completes with a plausible number.

A row-by-row table cannot catch the way this actually decays, which is a seventh scope added to
one resolver and forgotten in the other: the missing scope simply isn't in the table, so the
table still passes. `test_every_reserved_scope_is_wired_on_both_sides` therefore derives the
scopes from the module's own constants and fails on the omission rather than on a row.
"""

import pytest

from tvbo.templates.tvboptim import utils as U

_CI_KEY = {"LinCoupling": "ci_lin"}.get

# scope -> (raw reference, the axis dict the classifier builds for it, expected keypath).
# Adding a reserved scope to the module means adding it here; the scope-set test below fails
# until it is, and it is what stops a scope existing on only one of the two resolvers.
_SCOPES = {
    U._NOISE_SCOPE: ("noise.sigma", {"name": "sigma", "is_noise": True}, "noise.sigma"),
    U._NETWORK_SCOPE: (
        "network.conduction_speed",
        {"name": "conduction_speed", "is_network": True, "graph_leaf": "speed"},
        "graph.speed",
    ),
    U._INITIAL_CONDITIONS_SCOPE: (
        "initial_conditions.theta",
        {"name": "theta", "is_ic": True},
        "dynamics._ic_theta",
    ),
    U._RANDOM_SEED_SCOPE: (
        "execution.random_seed",
        {"name": "random_seed", "is_seed": True},
        "dynamics._noise_seed",
    ),
}


@pytest.mark.parametrize("scope", sorted(_SCOPES))
def test_both_resolvers_agree_on_each_reserved_scope(scope):
    ref, ax, expected = _SCOPES[scope]
    assert U.parameter_keypath(ref) == expected
    assert U.axis_keypath(ax) == expected


def test_every_reserved_scope_is_wired_on_both_sides():
    """A scope added to one resolver and not the other must fail HERE, not in a run.

    The module names its reserved scopes as constants, so the set is discoverable. Anything in
    it that this file does not exercise is a scope whose two spellings have never been compared.
    """
    declared = {
        v
        for k, v in vars(U).items()
        if k.startswith("_") and k.endswith("_SCOPE") and isinstance(v, str)
    }
    missing = declared - set(_SCOPES)
    assert not missing, (
        f"reserved scope(s) {sorted(missing)} are declared in tvbo.templates.tvboptim.utils but "
        f"are not exercised against BOTH resolvers. Add them to _SCOPES — a scope wired into "
        f"only one of parameter_keypath/axis_keypath binds the wrong leaf silently."
    )


def test_the_coupling_scope_agrees_when_the_experiment_declares_it():
    """A coupling prefix is a scope only because the experiment declares that coupling.

    `parameter_keypath` therefore needs the declared set, while `axis_keypath` reads a flag the
    classifier already set — the one place the two take genuinely different inputs.
    """
    got = U.parameter_keypath("LinCoupling.G", couplings={"LinCoupling"}, coupling_key=_CI_KEY)
    assert got == "coupling.ci_lin.G"
    assert U.axis_keypath({"name": "G", "is_coupling": True, "coupling_key": "ci_lin"}) == got


def test_the_external_scope_agrees():
    """An external input is addressable as a fitted parameter and as a swept axis alike."""
    got = U.parameter_keypath("stimulus.amplitude", external={"stimulus"})
    assert got == "external.stimulus.amplitude"
    ax = {"name": "amplitude", "is_external": True, "external_key": "stimulus"}
    assert U.axis_keypath(ax) == got


def test_a_bare_dynamics_reference_agrees():
    assert U.parameter_keypath("w") == "dynamics.w"
    assert U.parameter_keypath("ReducedWongWang.w") == "dynamics.w"
    assert U.axis_keypath({"name": "w"}) == "dynamics.w"


@pytest.mark.parametrize(
    "ref,leaf",
    [("network.edges.weight", "weights"), ("network.edges.length", "lengths"), ("network.edges.delay", "delays")],
)
def test_every_sweepable_edge_attribute_agrees(ref, leaf):
    """Each edge attribute with a live graph leaf must resolve the same way from both sides.

    `delay` is the newest and the reason this is parametrised rather than written once: the raw
    path picks it up through `network_axis_leaf`, so a leaf added to that table reaches
    `parameter_keypath` for free while `axis_keypath` needs the classifier to set `graph_leaf`.
    """
    assert U.parameter_keypath(ref) == f"graph.{leaf}"
    assert U.axis_keypath({"name": ref, "is_network": True, "graph_leaf": leaf}) == f"graph.{leaf}"


def test_a_flag_only_axis_resolves_without_any_declared_text():
    """`axis_keypath` must never recover a scope by re-parsing the reference.

    Delegating it to `parameter_keypath` was tried and reverted for exactly this: an axis dict
    carrying only flags has no `label` to read, so the text-based path returned `dynamics.sigma`
    for a noise axis — re-deriving a scope that was already known, and getting it wrong when the
    text is absent, which is the failure the shared resolver exists to remove.
    """
    assert U.axis_keypath({"name": "sigma", "is_noise": True}) == "noise.sigma"
    assert U.axis_keypath({"name": "theta", "is_ic": True}) == "dynamics._ic_theta"
