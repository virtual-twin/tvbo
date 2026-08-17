"""`network.`-scoped exploration axes resolve to the graph leaf they sweep.

The scope is declarative: an axis names a network attribute (`network.edges.weight`,
`network.conduction_speed`) and codegen resolves it to the live backend-graph leaf,
so every cell sees its own value without a Network or graph rebuild.

The canonical form is `network.edges.<label>`; `network.weight(s)`/`network.length(s)`
are shortcuts. Both go through `edge_label`, the same resolver observations use, so an
axis and an observation can never disagree about which matrix a reference means.
"""

import pytest

from tvbo.templates.tvboptim.utils import network_axis_leaf


@pytest.mark.parametrize(
    "ref, leaf",
    [
        # Canonical `network.edges.<label>` form.
        ("network.edges.weight", "weights"),
        ("network.edges.length", "lengths"),
        # Shortcuts resolve to the same leaf as the canonical form — singular and
        # plural alike, so a recipe cannot half-work on spelling.
        ("network.weight", "weights"),
        ("network.weights", "weights"),
        ("network.length", "lengths"),
        ("network.lengths", "lengths"),
        # The network's own scalars.
        ("network.conduction_speed", "speed"),
    ],
)
def test_network_axis_resolves_to_graph_leaf(ref, leaf):
    assert network_axis_leaf(ref) == leaf


@pytest.mark.parametrize(
    "ref",
    [
        "MurrayWangDM.mu",  # dynamics-scoped
        "EIBLinearCoupling.wLRE",  # coupling-scoped
        "execution.random_seed",  # execution-scoped
        "conduction_speed",  # bare name is not a scoped reference
        "",
        None,
        3.0,
    ],
)
def test_non_network_refs_return_none(ref):
    """Out-of-scope references route through the dynamics/coupling path untouched."""
    assert network_axis_leaf(ref) is None


def test_edges_weight_is_not_misparsed_as_dynamics():
    """`network.edges.weight` must not rsplit into prefix 'network.edges'.

    Splitting a scoped path on the LAST dot leaves prefix='network.edges', which
    matches neither the network scope nor a coupling key, so the axis silently
    falls through and is emitted as a dynamics parameter — a wrong-scope write
    that sweeps nothing and yields identical cells rather than an error.
    """
    assert network_axis_leaf("network.edges.weight") == "weights"


def test_unsweepable_edge_attribute_raises():
    """An edge attribute with no graph leaf fails at codegen, not silently.

    A matrix like `fc` is legitimately referenceable by an observation but has no
    live leaf to sweep, so naming it as an axis is a recipe error.
    """
    with pytest.raises(ValueError, match="no graph leaf to sweep"):
        network_axis_leaf("network.edges.fc")


def test_unknown_network_attribute_raises():
    with pytest.raises(ValueError, match="unknown network attribute"):
        network_axis_leaf("network.bogus")


def test_error_names_the_sweepable_set():
    """The error tells the author what they may sweep instead."""
    with pytest.raises(ValueError) as e:
        network_axis_leaf("network.bogus")
    msg = str(e.value)
    assert "network.conduction_speed" in msg
    assert "network.edges.weight" in msg


# --- codegen: the delay history buffer ---------------------------------------
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

_DELAY_SPEC = """
id: 9
dynamics:
  name: Kuramoto
  label: Kuramoto
  parameters:
    omega: {name: omega, value: 0.0628, unit: rad_per_ms}
  coupling_inputs:
    c: {name: c, description: "coupling"}
  state_variables:
    theta:
      name: theta
      unit: rad
      initial_value: 0.1
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + c"}
      variable_of_interest: true
      coupling_variable: true
  output: [theta]
  number_of_modes: 1
network:
  number_of_nodes: 2
  parameters:
    conduction_speed: {name: conduction_speed, value: 3.0, unit: mm_per_ms}
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - {source: 0, target: 1, weight: 0.5, distance: 10.0}
    - {source: 1, target: 0, weight: 0.5, distance: 10.0}
coupling:
  name: KuramotoCoupling
  label: KuramotoCoupling
  parameters:
    a: {name: a, value: 0.01}
    N: {name: N, value: 1.0}
  pre_expression: {rhs: "sin(theta_j - theta_i)"}
  post_expression: {rhs: "a * gx / N"}
  incoming_states: [theta]
  local_states: [theta]
integration:
  method: Heun
  duration: 40.0
  step_size: 1.0
  transient_time: 0.0
explorations:
  delay_sweep:
    name: delay_sweep
    mode: product
    space:
      - parameter: PARAMETER
        AXIS
"""

_LENGTH_BUILDER = "import numpy as np\n\n\ndef longer_tracts(n):\n    return [np.full((2, 2), 10.0 * (i + 1)) for i in range(int(n))]\n"


def _render_delay_sweep(tmp_path, parameter, axis):
    spec = _DELAY_SPEC.replace("PARAMETER", parameter).replace("        AXIS\n", axis)
    p = tmp_path / "delay.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    return exp.render_code("tvboptim")


def test_a_swept_speed_rebuilds_the_base_graph(tmp_path):
    """The slowest swept speed sizes the history buffer, so the base graph is rebuilt.

    `_v_build` marks the rebuild: `DenseLengthGraph(` and `max_delay_bound` are emitted for
    the base network whether or not anything is swept, so neither distinguishes it.
    """
    code = _render_delay_sweep(tmp_path, "network.conduction_speed", "        explored_values: [1.0, 3.0]\n")
    assert "_v_build" in code
    assert "min(_v_build, 1.0)" in code, "the buffer must be sized for the SLOWEST swept speed"


def test_a_swept_tract_length_also_rebuilds_the_base_graph(tmp_path, monkeypatch):
    """`delay = length / speed`, so a swept LENGTH sizes the buffer exactly as a speed does.

    prepare() sizes the history buffer once from the base graph and never re-reads it, so a
    length axis whose matrices exceed the base network's leaves every over-long cell
    integrating against a truncated history — a silently wrong trajectory, not an error.
    """
    import sys

    (tmp_path / "length_builder.py").write_text(_LENGTH_BUILDER)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("length_builder", None)

    axis = (
        "        builder:\n"
        "          callable: {name: longer_tracts, module: length_builder}\n"
        "          arguments: {n: {value: 3}}\n"
    )
    code = _render_delay_sweep(tmp_path, "network.edges.length", axis)
    assert "_v_build" in code, "a swept length must rebuild the base graph"
    assert "_bound_lengths" in code, "the buffer must be sized from the LONGEST swept tracts"
    assert code.count("longer_tracts(") == 1, (
        "the builder must be called ONCE — it may read base observations or cross-experiment "
        "data, and two calls could disagree about the values the grid then sweeps"
    )


def test_a_domain_declared_length_axis_bounds_the_buffer_at_its_hi(tmp_path):
    """A length axis may be declared `domain:` rather than by explicit points or a builder.

    Reading `values` unconditionally is a bare KeyError at RENDER time for that spelling —
    the sweep never runs at all. The longest tract a `domain:` axis reaches is its `hi`, so
    that is what the buffer must cover.
    """
    code = _render_delay_sweep(tmp_path, "network.edges.length", "        domain: {lo: 5.0, hi: 40.0, n: 4}\n")
    assert "jnp.full_like(_lengths, 40.0)" in code
    assert "_bound_lengths" in code


def test_a_swept_weight_leaves_the_buffer_alone(tmp_path):
    """No delay depends on the weights, so sweeping them needs no rebuild."""
    code = _render_delay_sweep(tmp_path, "network.edges.weight", "        explored_values: [0.1, 0.5]\n")
    assert "_bound_lengths" not in code
    assert "_v_build" not in code


_TWO_AXIS_SPACE = """      - parameter: network.conduction_speed
        explored_values: [1.0, 3.0]
      - parameter: network.edges.length
        builder:
          callable: {name: longer_tracts, module: tract_builder}
          arguments: {n: {value: 2}}
"""


def _run_two_network_axes(tmp_path, monkeypatch):
    import sys

    (tmp_path / "tract_builder.py").write_text(_LENGTH_BUILDER)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("tract_builder", None)
    spec = _DELAY_SPEC.replace(
        "      - parameter: PARAMETER\n        AXIS\n", _TWO_AXIS_SPACE
    )
    p = tmp_path / "two_axes.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    return exp.run("tvboptim", mode="exploration")


def test_two_network_axes_are_each_keyed_by_their_own_declared_name(tmp_path, monkeypatch):
    """Two `network.` axes in one exploration must both reach the container.

    Graph-leaf columns are POSITIONAL (`graph.1`, `graph.2`) because the graph pytree carries no
    field names, so resolving them through a single network label lets the last axis win and the
    other's coordinate arrive under a name matching no declared axis. The container then refuses
    to place the observation — after the whole compute has finished.
    """
    r = _run_two_network_axes(tmp_path, monkeypatch)
    expl = r.explorations.delay_sweep
    names = {str(getattr(a, "name", "")) for a in expl.axes}
    assert names == {"network.conduction_speed", "network.edges.length"}, names

    grid = expl.as_grid()
    assert "network.conduction_speed" in grid.dims
    assert "network.edges.length" in grid.dims
    assert grid.sizes["network.conduction_speed"] == 2
    assert grid.sizes["network.edges.length"] == 2


def test_a_matrix_valued_axis_is_keyed_by_point_index(tmp_path, monkeypatch):
    """A builder axis of whole matrices rounds trips: its coordinate is the point index.

    A matrix cannot be an xarray coordinate, so codegen declares `arange(n)` while the per-cell
    column carries the matrices. Converting the cells where the materialised points still exist
    is what lets the container place them at all.
    """
    import numpy as np

    r = _run_two_network_axes(tmp_path, monkeypatch)
    grid = r.explorations.delay_sweep.as_grid()
    np.testing.assert_array_equal(np.asarray(grid.coords["network.edges.length"]), [0, 1])
