"""A coupling names its global coupling scaling by declaring a parameter an instance of ``tvbo:GlobalCouplingScaling``, and a coupling that declares no post-expression gets that parameter times ``gx`` on every backend.

Two Linear nodes joined by edges of weight 1 start equal, so each obeys ``dx/dt = (gamma + k) x`` with ``k`` the factor the coupling applies to the neighbour's state, and the trace's decay rate gives ``k`` back. A parameter called ``G`` that no declaration names the global scaling is an ordinary parameter, a drive or a gain: the default does not read it and no warning is keyed to its name. NetworkDynamics.jl is compared on the post-expression it renders rather than on a run, because a Julia run from an overlay re-resolves the shared Julia environment.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy
import xarray as xr
import yaml

from tvbo import Coupling, SimulationExperiment

pytest.importorskip("tvboptim")
pytestmark = pytest.mark.backend_tvboptim

GAMMA = -10.0
"""Linear's damping, so the uncoupled trace decays at rate 10 per ms."""

STEP = 0.01
"""Euler step in ms; the factor is recovered exactly from Euler's per-step ratio."""

SCALING = "tvbo:GlobalCouplingScaling"
"""The class membership that names a parameter the coupling's global scaling."""

NAME = "FactorCoupling"


def parameter(name: str, value: float, *classes: str) -> dict:
    """A coupling parameter *name* of *value*, an instance of each of *classes*."""
    return {name: {"name": name, "value": value, **({"instance_of": list(classes)} if classes else {})}}


def coupling_spec(pre: str | None, post: str | None, parameters: dict | None = None, name: str = NAME) -> dict:
    """The coupling *name* with pre-expression *pre*, post-expression *post* and *parameters*; each is left out when ``None``."""
    spec = {"name": name, "delayed": False, "incoming_states": ["x"]}
    if pre is not None:
        spec["pre_expression"] = {"rhs": pre}
    if post is not None:
        spec["post_expression"] = {"rhs": post}
    if parameters:
        spec["parameters"] = parameters
    return spec


def experiment(coupling: dict, key: str = NAME, events: dict | None = None) -> SimulationExperiment:
    """Two Linear nodes joined by *coupling* under *key*, with *events* when given."""
    edges = [
        {"source": i, "target": 1 - i, "directed": True, "parameters": {"weight": {"name": "weight", "value": 1.0}}}
        for i in (0, 1)
    ]
    network = {"number_of_nodes": 2, "nodes": [{"id": 0}, {"id": 1}], "edges": edges, "coupling": {key: coupling}}
    dynamics = {"name": "Linear", "iri": "tvbo:Linear", "parameters": {"gamma": {"name": "gamma", "value": GAMMA}}}
    return SimulationExperiment(
        dynamics=dynamics,
        network=network,
        integration={"method": "Euler", "step_size": STEP, "duration": 1.0},
        **({"events": events} if events else {}),
    )


def resolve(coupling: dict, key: str = NAME, events: dict | None = None):
    """The couplings a backend reads from the experiment :func:`experiment` builds, where a parameter nothing reads is refused."""
    from tvbo.adapters.base import BaseAdapter

    return BaseAdapter(experiment(coupling, key, events)).resolve_couplings()


def trip(target: str | None) -> dict:
    """An event on *target* that reads and sets only ``limit``, or one on the experiment where *target* is ``None``."""
    event = {
        "event_type": "continuous",
        "condition": {"rhs": "t - p[:limit]"},
        "condition_parameters": ["limit"],
        "affect": {"rhs": "p[:limit] = 0"},
        "affect_parameters": ["limit"],
    }
    return {"trip": event | ({"target_component": target} if target else {})}


def effective_factor(result) -> float:
    """The factor ``k`` the coupling applied, from the decay of node 0's trace: Euler gives ``x[n+1] = (1 + STEP (gamma + k)) x[n]``."""
    data = next(
        d
        for c in (result, getattr(result, "integration", None))
        for d in (c, getattr(c, "data", None))
        if isinstance(d, xr.DataArray)
    )
    if "mode" in data.dims:
        data = data.isel(mode=0)
    trace = data.transpose("time", "variable", "node").values[:, 0, 0].astype(float)
    times = np.asarray(data["time"].values, dtype=float)
    steps = round((times[-1] - times[0]) / STEP)
    per_step = (trace[-1] / trace[0]) ** (1.0 / steps)
    return (per_step - 1.0) / STEP - GAMMA


def warned(action, name: str = NAME) -> tuple:
    """What *action* returns, and the messages of the warnings it raises that name the coupling *name*."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = action()
    return result, [str(w.message) for w in caught if repr(name) in str(w.message)]


def run(pre, post, parameters=None):
    """The effective factor of a tvboptim run, and the warnings raised on the way that name the coupling."""
    result, messages = warned(lambda: experiment(coupling_spec(pre, post, parameters)).run("tvboptim", quiet=True))
    return effective_factor(result), messages


def put_after(spec: dict) -> SimulationExperiment:
    """An experiment built on the identity coupling of *spec*'s pre-expression, with *spec* put on its network once it is built."""
    exp = experiment(coupling_spec(spec["pre_expression"]["rhs"], "gx"))
    exp.network.coupling[NAME] = Coupling(**spec)
    return exp


READERS = {
    "post": lambda coupling: coupling.post,
    "symbolic": lambda coupling: coupling.symbolic(),
    "render_code": lambda coupling: rendered_post(coupling.render_code("tvb"), "tvb"),
}
"""The readers of a coupling outside any experiment, each answering with a sympy expression that reads its post-expression."""


def rendered_post(code: str, fmt: str) -> sympy.Expr:
    """The post-expression *code* renders for ``FactorCoupling`` on *fmt*, as a sympy expression."""
    pattern = {
        "tvboptim": r"def post\(self, summed_inputs, local_states, params\):(?:\n {8}.*)*?\n {8}return (.+)",
        "networkdynamics": r"FactorCoupling_post\(gx, \(.*?\)\) = (.+)",
        "jax": r"\ndef cfun\(.*\):(?:\n.*)*?\n {4}return (.+)",
        "tvb": r"class FactorCoupling\(Coupling\):(?:\n.*)*?\n {8}post = (.+)",
    }[fmt]
    found = re.search(pattern, code)
    assert found, f"{fmt} renders no post-expression for FactorCoupling"
    return sympy.sympify(found.group(1).replace(".*", "*"))


@pytest.mark.parametrize(
    "pre, post, parameters, factor",
    [
        ("x_j", None, parameter("K", 2.0, SCALING), 2.0),
        ("G * x_j", None, parameter("G", 2.0), 2.0),
        ("G * x_j", "gx", parameter("G", 1.0, SCALING), 1.0),
        ("G * x_j", "gx", parameter("G", 2.0, SCALING), 2.0),
    ],
    ids=[
        "the default post-expression scales by the declared scaling whatever its name",
        "an undeclared G in the pre-expression",
        "a declared scaling of 1 in the pre-expression under the identity post",
        "a declared scaling of 2 in the pre-expression under the identity post",
    ],
)
def test_a_scaling_that_acts_once_runs_at_its_factor_without_a_warning(pre, post, parameters, factor):
    """A scaling written once, in the default post-expression ``K * gx`` or where the pre-expression reads it beside the identity ``gx``, runs at ``k`` equal to its value and raises nothing; an undeclared ``G`` acts where it is written."""
    k, messages = run(pre, post, parameters)
    assert k == pytest.approx(factor, rel=1e-6)
    assert not messages


REFUSAL = r"FactorCoupling.*parameter {}.*none of its expressions and no event on the network's edges reads.*instance_of: \[tvbo:GlobalCouplingScaling\]"
"""The refusal of an unread coupling parameter: it names the coupling and the parameter and says how to make the parameter act."""


@pytest.mark.parametrize(
    "pre, post, parameters, unread",
    [
        ("x_j", None, parameter("G", 2.0), "'G'"),
        ("x_j", "gx", parameter("K", 2.0, SCALING), "'K'"),
        ("a * x_j", "gx + c", parameter("a", 1.0) | parameter("b", 0.5) | parameter("c", 0.1), "'b'"),
    ],
    ids=["undeclared G under the default gx", "declared scaling under a declared post", "one of three"],
)
def test_a_parameter_no_expression_reads_loads_and_is_refused_where_a_backend_reads_it(pre, post, parameters, unread):
    """A coupling parameter that neither the pre-expression nor the post-expression reads, the default ``gx`` included, leaves the experiment loadable, so a saved one can still be inspected, and is refused where a backend reads the couplings, naming the coupling and that parameter alone."""
    experiment(coupling_spec(pre, post, parameters))
    with pytest.raises(ValueError, match=REFUSAL.format(unread)) as refused:
        resolve(coupling_spec(pre, post, parameters))
    assert "'a'" not in str(refused.value) and "'c'" not in str(refused.value)


@pytest.mark.parametrize("backend, source", [("tvboptim", "x_j"), ("python", "x")])
def test_a_parameter_no_expression_reads_is_refused_on_run_when_put_on_the_network_afterwards(backend, source):
    """A coupling with an unread ``G`` put on the network once the experiment is built is refused when the experiment runs, on the pure-Python runner as on tvboptim."""
    exp = put_after(coupling_spec(source, None, parameter("G", 2.0)))
    with pytest.raises(ValueError, match=REFUSAL.format("'G'")):
        exp.run(backend, quiet=True)


def start(name: str, rhs: str) -> dict:
    """A coupling parameter *name* whose start is the equation *rhs*."""
    return {name: {"name": name, "equation": {"rhs": rhs}}}


@pytest.mark.parametrize(
    "pre, parameters",
    [
        ("w * x_j", parameter("s", 1.0) | {"w": {"name": "w", "value": 2.0, "equation": {"rhs": "2 * s"}}}),
        ("a * x_j", start("a", "2 * b") | start("b", "3 * c") | parameter("c", 1.0)),
    ],
    ids=["w = 2 * s reads s", "a = 2 * b and b = 3 * c read b and c"],
)
def test_a_parameter_the_equation_of_a_read_parameter_reads_is_read(pre, parameters):
    """A parameter's equation is an expression of the coupling once the parameter is read, so ``s``, which only ``w = 2 * s`` reads, and the chain ``a = 2 * b``, ``b = 3 * c`` from a read ``a`` fill without a refusal."""
    assert experiment(coupling_spec(pre, "gx", parameters)).coupling.post_expression.rhs == "gx"


@pytest.mark.parametrize(
    "parameters, unread",
    [
        (start("w", "2 * s") | parameter("s", 0.75), "'w', 's'"),
        (start("p", "p + 1"), "'p'"),
        (start("u", "v") | start("v", "u"), "'u', 'v'"),
    ],
    ids=[
        "the equation of an unread parameter",
        "an equation reading its own parameter",
        "a cycle neither end of which is read",
    ],
)
def test_a_start_equation_reads_only_for_a_parameter_that_is_read(parameters, unread):
    """A parameter's equation reads the names in it only once the parameter itself is read, and never toward itself, so one refusal names every parameter left unread."""
    with pytest.raises(ValueError, match=rf"coupling 'FactorCoupling' has the parameters? {unread}, which none"):
        resolve(coupling_spec("x_j", "gx", parameters))


def test_a_parameter_taken_from_the_curated_function_is_named_as_the_coupling_s():
    """Linear's ``a`` and ``b``, which the recipe never wrote, are refused as parameters the coupling has from ``tvbo:Linear`` once its own post-expression ``gx`` drops them."""
    with pytest.raises(
        ValueError, match=r"coupling 'Linear', built from `tvbo:Linear`, has the parameters 'b', 'a', which none"
    ):
        resolve(
            {"iri": "tvbo:Linear", "delayed": False, "pre_expression": {"rhs": "x_j"}, "post_expression": {"rhs": "gx"}},
            key="Linear",
        )


@pytest.mark.parametrize(
    "target, refused",
    [
        ("all_edges", False),
        ("edge_1", False),
        ("edge_2", False),
        ("edge_0", True),
        ("edge_3", True),
        ("all_vertices", True),
        (None, True),
    ],
)
def test_only_an_event_on_the_edges_reads_a_coupling_parameter(target, refused):
    """An event on the edges, all of them or one of the two listed, reads the coupling's ``limit``; one on ``edge_0``, on an edge past the last listed, on the nodes or on the experiment names parameters of the dynamics or of itself, so ``limit`` stays unread and is refused."""
    spec = coupling_spec("x_j", "gx", parameter("limit", 1.0))
    if refused:
        with pytest.raises(ValueError, match=REFUSAL.format("'limit'")):
            resolve(spec, events=trip(target))
    else:
        assert resolve(spec, events=trip(target))[NAME].post_expression.rhs == "gx"


CASCADING = Path(__file__).resolve().parents[1] / "docs/Interoperability/NetworkDynamics.jl/yaml/cascading_failure.yaml"
"""The docs power grid, whose line coupling ``ActivePower`` declares a ``limit`` that only the line-tripping event ``edge_trip`` reads."""


def cascading(tmp_path, edit) -> SimulationExperiment:
    """The docs power grid loaded from a copy in *tmp_path* of its spec with *edit* applied."""
    spec = yaml.safe_load(CASCADING.read_text())
    edit(spec)
    (tmp_path / CASCADING.name).write_text(yaml.safe_dump(spec, sort_keys=False))
    return SimulationExperiment.from_file(str(tmp_path / CASCADING.name))


def move_edge_trip(collection: str):
    """An edit moving ``edge_trip``, which targets ``all_edges``, into the own ``events`` of the first of the network's *collection*."""
    return lambda spec: spec["network"][collection][0].update(events={"edge_trip": spec["events"].pop("edge_trip")})


EDGE_TRIP_PLACES = {
    "the experiment": lambda spec: None,
    "an edge's own events": move_edge_trip("edges"),
    "a node's own events": move_edge_trip("nodes"),
}
"""Where ``edge_trip`` is declared: on the experiment, as the docs recipe has it, or in the first edge's or the first node's own ``events``."""


@pytest.mark.parametrize("place", sorted(EDGE_TRIP_PLACES))
def test_a_parameter_only_an_event_reads_is_used(place, tmp_path):
    """``limit``, which no expression of ``ActivePower`` reads and ``edge_trip`` names in its ``condition_parameters``, is accepted wherever the event targeting the edges is declared, and the NetworkDynamics.jl edge carries it for the callback."""
    code = cascading(tmp_path, EDGE_TRIP_PLACES[place]).render_code(format="networkdynamics")
    assert "psym = [:K => 1.63, :limit => 1.0, :w => 1.0]," in code
    assert "edge_trip_cond = ComponentCondition([:P], [:limit])" in code


def test_the_same_parameter_without_its_event_is_refused(tmp_path):
    """With ``edge_trip`` removed, no event reads ``limit``, and rendering the experiment for NetworkDynamics.jl is refused, naming ``ActivePower`` and ``limit``."""
    exp = cascading(tmp_path, lambda spec: spec["events"].pop("edge_trip"))
    with pytest.raises(
        ValueError, match=r"ActivePower.*parameter 'limit'.*none of its expressions and no event on the network's edges reads"
    ):
        exp.render_code(format="networkdynamics")


@pytest.mark.parametrize("reader", sorted(READERS))
def test_a_coupling_read_on_its_own_is_filled_without_the_refusal(reader):
    """A coupling outside any experiment has no events to consult, so `post`, `symbolic` and `render_code` give an unread ``G`` the default ``gx`` rather than refusing it."""
    coupling = Coupling(**coupling_spec("x_j", None, parameter("G", 2.0)))
    READERS[reader](coupling)
    assert coupling.post_expression.rhs == "gx"


def test_a_parameter_read_only_in_a_branch_is_read():
    """A parameter read only by a branch of a conditional pre-expression, which leaves the right-hand side empty, is not refused by a fill that no event reads for."""
    spec = coupling_spec(None, "gx", parameter("theta", 0.5))
    spec["pre_expression"] = {
        "conditionals": [{"condition": "x_j > theta", "expression": "x_j"}, {"condition": "True", "expression": "0"}]
    }
    assert Coupling(**spec).fill_post_expression().refuse_unread(set()).post_expression.rhs == "gx"


def test_a_declared_scaling_is_read_by_the_default_before_the_fill():
    """`Coupling.refuse_unread` on a coupling that declares ``K`` the scaling and no post-expression counts the default ``K * gx`` as reading ``K``, also before `Coupling.fill_post_expression` has given it one."""
    coupling = Coupling(**coupling_spec("x_j", None, parameter("K", 2.0, SCALING)))
    assert coupling.refuse_unread(set()).post_expression is None


def test_a_declared_scaling_in_the_pre_expression_and_the_default_post_expression_acts_twice_and_is_warned_about():
    """``G * x_j`` with ``G = 2`` declared the global scaling and the default post-expression runs at ``k = 4``, and the warning names the coupling and ``G``."""
    k, messages = run("G * x_j", None, parameter("G", 2.0, SCALING))
    assert k == pytest.approx(4.0, rel=1e-6)
    assert messages and "`G`" in messages[0], f"G = 2 acting twice ran at {k:.6g} with no warning naming the coupling and `G`"


def test_a_declared_scaling_in_a_branch_of_the_pre_expression_is_warned_about():
    """A scaling read only inside a branch of a conditional pre-expression acts twice under the default post-expression too, and is warned about."""
    spec = coupling_spec(None, None, parameter("K", 2.0, SCALING))
    spec["pre_expression"] = {
        "conditionals": [{"condition": "x_j > 0", "expression": "K * x_j"}, {"condition": "True", "expression": "0"}]
    }
    _, messages = warned(lambda: Coupling(**spec).fill_post_expression())
    assert messages and "`K`" in messages[0]


def test_a_gain_named_g_raises_no_warning():
    """The curated PreSigmoidal reads its gain ``G`` inside the pre-expression and raises nothing."""
    _, messages = warned(
        lambda: experiment({"iri": "tvbo:PreSigmoidal"}, key="PreSigmoidal").render("tvboptim"), "PreSigmoidal"
    )
    assert not messages


def test_two_declared_global_scalings_are_refused():
    """A coupling declaring two parameters its global scaling is refused, naming both."""
    with pytest.raises(ValueError, match=r"FactorCoupling.*\bA\b.*\bB\b"):
        experiment(coupling_spec("x_j", None, parameter("A", 1.0, SCALING) | parameter("B", 2.0, SCALING)))


@pytest.mark.parametrize("term", ["tvbo:NoSuchClass", "tvbo:coupling_functions/Linear"])
def test_an_instance_of_naming_no_ontology_class_is_refused(term):
    """A coupling parameter declared an instance of a ``tvbo:`` term that is not a class of the ontology, a scoped record's IRI included, is refused, naming the coupling, the parameter and the term."""
    with pytest.raises(ValueError, match=rf"FactorCoupling.*\bK\b.*{re.escape(term)}"):
        experiment(coupling_spec("x_j", None, parameter("K", 2.0, term)))


@pytest.mark.parametrize(
    "classes",
    [
        ["tvbo:GlobalCouplingScaling", "tvbo:GlobalCouplingScaling"],
        ["https://w3id.org/tvbo/GlobalCouplingScaling"],
        ["tvbo:GlobalCouplingScaling", "neuroml:pointCellCondBased"],
    ],
)
def test_the_scaling_is_recognised_however_it_is_written(classes):
    """A parameter is the scaling once, whether it lists the class twice, writes it in full or lists a class of another namespace beside it."""
    coupling = Coupling(**coupling_spec("x_j", None, parameter("K", 2.0, *classes)))
    assert coupling.global_scaling() == "K"


@pytest.mark.parametrize("other", ["neuroml:pointCellCondBased", "GO:0007268"])
def test_a_class_of_another_namespace_is_neither_the_scaling_nor_refused(other):
    """Only a ``tvbo:`` class is checked against the TVB-O ontology; a class of another namespace is a membership the rule leaves alone."""
    exp = experiment(coupling_spec("K * x_j", None, parameter("K", 2.0, other)))
    assert exp.coupling.global_scaling() is None
    assert exp.coupling.post_expression.rhs == "gx"


def test_naming_the_global_scaling_never_loads_the_ontology(monkeypatch):
    """A build whose curated coupling declares only ``tvbo:GlobalCouplingScaling`` recognises it by its IRI and never reads the ontology; the integrator's own ontology fill is stubbed out, since it reads the ontology for its own reasons."""
    import traceback

    from tvbo.behaviour.integrator import IntegratorBehaviour
    from tvbo.ontology import owl

    loads = []
    real = owl._load_ontology
    monkeypatch.setattr(owl, "_load_ontology", lambda: loads.append("".join(traceback.format_stack(limit=8))) or real())
    monkeypatch.setattr(IntegratorBehaviour, "enrich", lambda self, source=None: self)
    exp = experiment({"iri": "tvbo:Linear"}, key="Linear")
    assert exp.coupling.global_scaling() == "a"
    assert not loads, f"the build read the ontology:\n{loads[0]}"


@pytest.mark.parametrize("fmt", ["tvboptim", "networkdynamics", "jax", "tvb"])
@pytest.mark.parametrize("when", ["declared", "put on the network after construction"])
def test_every_backend_renders_the_default_post_expression(fmt, when):
    """A coupling declaring ``K`` the global scaling and no post-expression renders ``K * gx`` on every backend, whether it was declared with the experiment or put on its network afterwards."""
    spec = coupling_spec("x_j", None, parameter("K", 2.0, SCALING))
    exp = experiment(spec) if when == "declared" else put_after(spec)
    assert rendered_post(exp.render(fmt), fmt) == sympy.sympify("K * gx")


def test_the_default_post_expression_is_written_into_the_record_and_its_report():
    """The experiment's YAML and its Markdown report state the post-expression every backend runs."""
    exp = experiment(coupling_spec("x_j", None, parameter("K", 2.0, SCALING)))
    assert re.search(r"post_expression:\n\s+rhs: K \* gx", exp.render("yaml"))
    assert r"**Post-synaptic:** $c_{\text{post}} = K \cdot gx$" in exp.render("markdown")


def test_a_coupling_rendered_on_its_own_gets_the_default():
    """`Coupling.render_code` on a coupling declaring ``K`` the global scaling and no post-expression renders ``K * gx``."""
    coupling = Coupling(**coupling_spec("x_j", None, parameter("K", 2.0, SCALING)))
    assert READERS["render_code"](coupling) == sympy.sympify("K * gx")


@pytest.mark.parametrize("reader", ["post", "symbolic"])
def test_a_coupling_on_its_own_reads_its_default_post_expression(reader):
    """`Coupling.post` and `Coupling.symbolic` on a coupling declaring no post-expression read the default, not an error."""
    coupling = Coupling(**coupling_spec("x_j", None, parameter("K", 2.0, SCALING)))
    assert sympy.Symbol("K") in READERS[reader](coupling).free_symbols
    assert coupling.post_expression.rhs == "K * gx"


def test_a_curated_function_keeps_its_own_post_expression():
    """The default fills only what a coupling still lacks once it is built from the function it names."""
    exp = experiment({"iri": "tvbo:Sigmoidal"}, key="Sigmoidal")
    assert str(exp.coupling.post_expression.rhs) == str(Coupling.from_db("Sigmoidal").post_expression.rhs)


@pytest.mark.parametrize("when", ["declared", "put on the network after construction"])
def test_the_python_backend_runs_the_default_post_expression(when):
    """The pure-Python network runner reads the coupling through the same fill: ``x`` with ``K = 2`` declared the scaling and no post-expression runs at ``k = 2``."""
    spec = coupling_spec("x", None, parameter("K", 2.0, SCALING))
    exp = experiment(spec) if when == "declared" else put_after(spec)
    assert effective_factor(exp.run("python")) == pytest.approx(2.0, rel=1e-6)


def lems_coupling(coupling: dict, key: str = NAME, after: bool = False) -> tuple[str, str, str]:
    """The post-expression `build_lems_context` hands the LEMS templates for *coupling* on two Linear nodes counted rather than listed, the network form the LEMS Coupling ComponentType is emitted for, then the ``gx`` and ``post`` that ComponentType renders; with *after*, the coupling is put on the network once the experiment is built."""
    from tvbo.adapters.neuroml import NeuroMLAdapter, build_lems_context

    network = {"number_of_nodes": 2, "coupling": {key: coupling_spec("x_j", "gx", name=key) if after else coupling}}
    exp = SimulationExperiment(
        dynamics={"iri": "tvbo:Linear"}, network=network, integration={"method": "Euler", "step_size": STEP, "duration": 1.0}
    )
    if after:
        exp.network.coupling[key] = Coupling(**coupling)
    context = build_lems_context(exp)["coupling_post_rhs"]
    code = NeuroMLAdapter(exp).render_code()
    assert "global_coupling" not in code
    found = dict(re.findall(r'<DerivedVariable name="(gx|post)"\s+dimension="none" value="([^"]+)"/>', code))
    return context, found["gx"], found["post"]


@pytest.mark.parametrize(
    "coupling, key, after, post",
    [
        (coupling_spec("x_j", None, parameter("K", 2.0, SCALING)), NAME, False, ("K * gx", "K*gx")),
        (coupling_spec("x_j", None, parameter("K", 2.0, SCALING)), NAME, True, ("K * gx", "K*gx")),
        ({"iri": "tvbo:Linear", "delayed": False}, "Linear", False, ("a*gx + b", "b + a*gx")),
    ],
    ids=["declared", "put on the network after construction", "curated Linear"],
)
def test_the_lems_coupling_applies_the_declared_scaling_once(coupling, key, after, post):
    """The LEMS Coupling ComponentType sums without a factor of its own and applies the post-expression, so the declared scaling acts once."""
    declared, rendered = post
    assert lems_coupling(coupling, key, after) == (declared, "pre", rendered)


@pytest.mark.parametrize("events, unread", [(None, "'G'"), (trip("all_edges"), None)], ids=["unread", "read by an edge event"])
def test_the_lems_context_refuses_what_nothing_reads_and_counts_an_edge_event(events, unread):
    """`build_lems_context` fills the coupling with the names the experiment's edge events read: a coupling put on the network after construction with an unread ``G`` is refused there, and a ``limit`` an edge event reads is not."""
    from tvbo.adapters.neuroml import build_lems_context

    exp = experiment(coupling_spec("x_j", "gx"), events=events)
    exp.network.coupling[NAME] = Coupling(**coupling_spec("x_j", "gx", parameter("G" if unread else "limit", 2.0)))
    if unread:
        with pytest.raises(ValueError, match=REFUSAL.format(unread)):
            build_lems_context(exp)
    else:
        assert build_lems_context(exp)["coupling_post_rhs"] == "gx"


def test_the_lems_context_refuses_a_post_expression_written_by_branches_alone():
    """The LEMS Coupling ComponentType takes one expression for ``post``, so a post-expression declared only by branches is refused, naming the coupling, rather than rendered as ``None``."""
    from tvbo.adapters.neuroml import build_lems_context

    spec = coupling_spec("x_j", None, parameter("K", 2.0, SCALING))
    spec["post_expression"] = {
        "conditionals": [{"condition": "gx > 0", "expression": "K * gx"}, {"condition": "True", "expression": "0"}]
    }
    exp = SimulationExperiment(
        dynamics={"iri": "tvbo:Linear"},
        network={"number_of_nodes": 2, "coupling": {NAME: spec}},
        integration={"method": "Euler", "step_size": STEP, "duration": 1.0},
    )
    with pytest.raises(NotImplementedError, match=r"FactorCoupling.*by branches alone"):
        build_lems_context(exp)


def test_a_list_pre_expression_into_one_input_without_recombination_is_refused():
    """Two pre-expression terms summed into Linear's one coupling input need a post-expression that recombines ``gx_0`` and ``gx_1``; without one the run is refused, naming the coupling."""
    with pytest.raises(ValueError, match=r"FactorCoupling.*gx_0"):
        experiment(coupling_spec("[x_j, x_j]", None)).run("tvboptim", quiet=True)


CURATED_SCALINGS = {
    "Difference": "a",
    "FastLinearCoupling": "G",
    "HyperbolicTangent": None,
    "KuramotoCoupling": "a",
    "Linear": "a",
    "PreSigmoidal": "H",
    "Scaling": "a",
    "Sigmoidal": None,
    "SigmoidalJansenRit": "a",
}
"""Each curated coupling's global scaling: the parameter scaling the summed input that its definition calls a rescaling of the connection strength or a global factor. HyperbolicTangent's ``a`` scales the summed input but is defined as the sigmoid's minimum, and Sigmoidal's ``a`` is the sigmoid's steepness, so neither declares one."""


@pytest.mark.parametrize("name", sorted(CURATED_SCALINGS))
def test_the_curated_couplings_declare_their_global_scaling(name):
    """Every curated coupling names its global scaling, or none where its definition gives none."""
    assert Coupling.from_db(name).global_scaling() == CURATED_SCALINGS[name]
