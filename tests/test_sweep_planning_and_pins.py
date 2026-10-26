"""How a sweep is planned, pinned and placed before any backend runs it.

The workflow planner sizes each axis: one swept per node, by a ``builder`` or from a branch has no cell count before the run, so a kit splits it into ``chunk`` array tasks and no workflow engine fans it out. ``--pin`` holds one element of a per-node dynamics parameter and refuses anything else by name, a ``--limit`` sample gets a directory of its own, and TVB, which runs no sweep in-process, refuses one. None of this needs tvboptim, so these run wherever tvbo does.
"""

import re

import pytest

from tvbo import SimulationExperiment
from tvbo.run import study as study_run
from tvbo.run.workflow import plan

K0 = (0.1, 0.5, 0.9)
K1 = (0.2, 0.4, 0.6, 0.8)
ELEMENT_0 = "ElemDecay.k[0]"
ELEMENT_1 = "ElemDecay.k[1]"

MODEL = """
id: 1
label: Decay rate per node
dynamics:
  name: ElemDecay
  parameters:
    k: {parameter}
  state_variables:
    x: {{equation: {{rhs: "-k*x"}}, initial_value: 0.7}}
network:
  label: Pair
  number_of_nodes: 2
integration: {{method: euler, step_size: 0.5, duration: 4.0, unit: ms}}
observations:
  x_last:
    label: Final x
    source: [x]
    aggregation: last
explorations:
  rates:
    label: Decay rate
    space:
      ElemDecay.k:
{axis}
"""

PER_NODE_K = '{value: 0.5, heterogeneous: true, shape: "(n_nodes,)"}'
SHARED_DOMAIN = "        domain: {lo: 0.1, hi: 0.9, n: 4}"

GRID = MODEL.format(
    parameter=PER_NODE_K,
    axis=f"        element_domains:\n          - {{element: 0, explored_values: {list(K0)}}}\n          - {{element: 1, explored_values: {list(K1)}}}",
)

BUILT = MODEL.format(
    parameter="{value: 0.5}",
    axis="        builder:\n          callable: {name: rates, module: rate_builder}\n          arguments: {n: {value: 5}}",
)

SCALAR = MODEL.format(parameter="{value: 0.5}", axis=SHARED_DOMAIN)

PER_NODE = MODEL.format(parameter=PER_NODE_K, axis=SHARED_DOMAIN)

NODE_SET = SCALAR.replace(
    "  number_of_nodes: 2\n", "  number_of_nodes: 2\n  nodes:\n    - {id: 0, parameters: {k: {value: 0.3}}}\n    - {id: 1}\n"
)

LABELLED = GRID.replace(
    "  number_of_nodes: 2\n", "  number_of_nodes: 2\n  nodes:\n    - {id: 0, label: V1}\n    - {id: 1, label: V2}\n"
)

COUPLED = GRID.replace("      ElemDecay.k:\n", "      Lin.k:\n").replace(
    "  number_of_nodes: 2\n",
    "  number_of_nodes: 2\n  coupling:\n    Lin:\n      parameters:\n        k: {value: 0.01}\n      pre_expression: {rhs: x_j}\n",
)


def _pin(spec: str, *pins) -> None:
    """Apply *pins* to the experiment *spec* declares, as ``tvbo run --pin`` does before anything runs."""
    study_run.apply_axis_pins(SimulationExperiment.from_string(spec), pins)


NODE_NAMED = SCALAR.replace(
    "  number_of_nodes: 2\n", "  number_of_nodes: 2\n  nodes:\n    - {id: 0, parameters: {k: {unit: ms}}}\n    - {id: 1}\n"
)


@pytest.mark.parametrize(
    ("spec", "sized_at_run_time"),
    [(GRID, True), (BUILT, True), (PER_NODE, True), (NODE_SET, True), (SCALAR, False), (NODE_NAMED, False)],
    ids=["element_domains", "builder", "per-node-shape", "per-node-by-a-node", "lockstep", "named-by-a-node-without-a-value"],
)
def test_a_kit_sizes_an_axis_the_grid_expands_at_run_time(spec, sized_at_run_time):
    """An axis on a per-node parameter (declared per node, or per node because a node sets its own value) or one a builder fills has no cell count before the run: the kit submits ``chunk`` array tasks that each slice the grid the backend builds. A lockstep axis keeps its static count, which caps the tasks."""
    experiment = SimulationExperiment.from_string(spec)
    built = plan(study_key="s", experiment=experiment, backend="tvboptim", workflow_spec={"distribute": {"chunk": 5}})
    (axis,) = built.vectorize_axes
    assert axis.runtime_sized is sized_at_run_time
    assert built.n_array_tasks == (5 if sized_at_run_time else 4)


@pytest.mark.parametrize("axis", ["network.edges.EI.weight", "network.global_coupling"])
def test_an_axis_tvboptim_cannot_bind_is_planned_for_the_backend_that_sweeps_it(axis):
    """A reference tvboptim's keypath resolver cannot bind (a Brian2 projection's weight, a network parameter a pin writes) names no per-node dynamics parameter, so the planner fans it out by its values and leaves the refusal to the codegen that cannot sweep it."""
    spec = SCALAR.replace("      ElemDecay.k:\n", f"      {axis}:\n")
    (planned,) = plan(study_key="s", experiment=SimulationExperiment.from_string(spec), backend="tvb").workflow_axes
    assert not planned.runtime_sized and len(planned.values) == 4


@pytest.mark.parametrize("spec", [GRID, BUILT, PER_NODE], ids=["element_domains", "builder", "per-node"])
def test_a_workflow_engine_refuses_to_fan_out_an_axis_sized_at_run_time(spec):
    """TVB fans every axis out from the workflow engine, which has no values for an axis the grid expands at run time; a per-cell pin would set every node to one value where the grid sweeps each node."""
    with pytest.raises(ValueError, match="exist only at run time"):
        plan(study_key="s", experiment=SimulationExperiment.from_string(spec), backend="tvb")


@pytest.mark.parametrize("limit", [3, 4, 100])
@pytest.mark.parametrize("spec", [SCALAR, GRID], ids=["lockstep", "element_domains"])
def test_tvb_refuses_a_limited_sweep(tmp_path, monkeypatch, spec, limit):
    """TVB runs no sweep in-process, so ``--limit`` is refused whether or not it would cut the sweep."""
    ran = []
    monkeypatch.setattr(study_run, "_exec_one", lambda *args, **kwargs: ran.append(kwargs))
    with pytest.raises(study_run.StudyRunError, match="does not vectorise"):
        study_run._run_one(
            SimulationExperiment.from_string(spec), "tvb", tmp_path, study_run.RunOptions(backend="tvb", limit=limit), None
        )
    assert ran == []


def test_tvb_refuses_an_experiment_whose_sweep_it_would_skip():
    """A TVB run of an experiment that declares an exploration names it and the workflow kit that runs it, rather than running the base simulation alone."""
    with pytest.raises(ValueError, match=r"runs no exploration in-process, and 'Decay rate per node' declares rates"):
        SimulationExperiment.from_string(SCALAR).run(format="tvb")


def test_a_limited_run_has_a_directory_beside_the_results(tmp_path):
    """A ``--limit`` sample lands beside the results directory, never inside it, since the container lookups glob it recursively; a shard ignores the limit and stays in the results directory."""
    whole = study_run.results_root("recipe.yaml", None, base=tmp_path)
    assert study_run.results_root("recipe.yaml", None, base=tmp_path, sample=4) == whole.with_name(f"{whole.name}-limit-4")
    assert study_run.RunOptions(limit=4).sample == 4
    assert study_run.RunOptions(limit=4, shard=(0, 2)).sample is None


def test_a_sample_s_figures_read_the_sample(tmp_path):
    """Inside ``reading_results`` a figure finds its study's containers in the sample directory, and outside it in the study's own results."""
    from tvbo import datamodel as dm
    from tvbo.adapters import bsplot

    sample = study_run.results_root("recipe.yaml", None, base=tmp_path, sample=4)
    sample.mkdir(parents=True)
    container = sample / "exp-1_result.h5"
    container.write_bytes(b"")
    layer = bsplot._UsedOnly(dm.DataRef(iri="exp-1"))
    assert bsplot._resolve_layer(layer, "cartesian", tmp_path)["container"] == ""
    with bsplot.reading_results(sample):
        assert bsplot._resolve_layer(layer, "cartesian", tmp_path)["container"] == str(container.resolve())


@pytest.mark.parametrize(
    ("element", "spec", "said"),
    [
        ("V9", LABELLED, "no node labelled 'V9'"),
        ("V1", GRID, "no node labelled 'V1'"),
        ("2", GRID, "outside the network's 2 nodes"),
    ],
)
def test_pin_refuses_an_element_the_network_does_not_have(element, spec, said):
    """An unknown label, a label the generated node names do not carry and an index past the last node are refused by name, before anything runs."""
    with pytest.raises(study_run.StudyRunError, match=re.escape(said)):
        _pin(spec, (f"ElemDecay.k[{element}]", 0.9))


def test_pin_refuses_an_element_of_an_axis_a_builder_supplies():
    """A builder hands over every node's value at each point, which no element domain overrides, so holding one element is refused rather than silently ignored."""
    with pytest.raises(study_run.StudyRunError, match="takes its values from a builder"):
        _pin(BUILT.replace("k: {value: 0.5}", f"k: {PER_NODE_K}"), (ELEMENT_0, 0.9))


def test_pin_refuses_an_element_of_a_parameter_shared_by_every_node():
    """A scalar parameter swept in lockstep has no element to pin: pinning one would split its four cells into an axis per node, sixteen on two nodes and four to the power of the node count beyond."""
    with pytest.raises(study_run.StudyRunError, match="holds one value for every node"):
        _pin(SCALAR, (ELEMENT_0, 0.9))


def test_pin_refuses_a_path_that_binds_a_coupling():
    """``Lin.k`` names the coupling's ``k``, which has no value per node, though the model beside it declares a per-node ``k`` of the same name."""
    with pytest.raises(study_run.StudyRunError, match=re.escape("it binds 'coupling.Lin.k'")):
        _pin(COUPLED, ("Lin.k[0]", 0.9))


def test_pin_refuses_a_value_that_is_not_a_number():
    """A malformed value is a one-line refusal naming the pin, not a traceback."""
    with pytest.raises(study_run.StudyRunError, match=re.escape(f"--pin {ELEMENT_0}: could not convert")):
        _pin(GRID, (ELEMENT_0, "abc"))


@pytest.mark.parametrize("raw", ["ElemDecay.k[0]=0.9", "ElemDecay.k[V1]=0.9"])
def test_the_command_line_reads_both_element_pin_forms(raw):
    """``tvbo run --pin`` hands an element pin to the runner whole, its value a number."""
    from tvbo.cli.run import _assignments

    ((key, value),) = _assignments([raw], "--pin", "parameter=value")
    assert (key, value) == (raw.split("=")[0], 0.9)


def test_a_domain_without_n_counts_as_the_grid_builds_it():
    """A domain declaring neither ``n`` nor ``step`` spans 50 points in the workflow plan, as in the generated grid."""
    from types import SimpleNamespace

    from tvbo.run.workflow import _axis_values

    axis = SimpleNamespace(explored_values=None, domain=SimpleNamespace(lo=0.0, hi=1.0, n=None, step=None, log_scale=False))
    assert len(_axis_values(axis)) == 50
    axis.domain.step = 0.25
    assert _axis_values(axis) == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_a_sample_reads_what_it_produces_and_the_whole_run_for_the_rest(tmp_path):
    """Inside ``reading_sample`` an experiment or analysis the sample run produces is read from its own directory and any other from the whole run, even where an older sample of it lies in the sample's directory, and a missing cohort shard is reported where it was looked for; outside, a lookup reads the directory it is given."""
    from tvbo.data.dataref import analysis_container_path, locate_analysis_container, locate_exp_container, reading_sample

    whole = study_run.results_root("recipe.yaml", None, base=tmp_path)
    own = study_run.results_root("recipe.yaml", None, base=tmp_path, sample=4)
    files = {
        "whole-3": whole / "exp-3_result.h5",
        "own-3": own / "exp-3_result.h5",
        "own-5": own / "exp-5_result.h5",
        "own-a": analysis_container_path(own, "a"),
        "whole-b": analysis_container_path(whole, "b"),
        "whole-7": whole / "sub-01_exp-7_result.h5",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    with reading_sample(own, whole, experiments=["exp-5"], analyses=["a"]):
        assert locate_exp_container(own, "3") == files["whole-3"], (
            "an experiment this sample does not run is read from the whole run"
        )
        assert locate_exp_container(own, "5") == files["own-5"]
        assert locate_analysis_container(own, "a") == files["own-a"]
        assert locate_analysis_container(own, "b") == files["whole-b"]
        with pytest.raises(FileNotFoundError, match=re.escape(f"shard(s) under {whole.resolve()})")):
            locate_exp_container(own, "7", subject="02")
    assert locate_exp_container(own, "3") == files["own-3"]
