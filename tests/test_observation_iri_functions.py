"""A curated observation model referenced by `iri` merges its helper functions.

An observation model whose pipeline calls helper functions by name (an HRF kernel, a downsample, a strided convolution) ships those functions in a `functions:` block.
`Observation` has no `functions` slot — codegen reads them from `experiment.functions`
— so `populate_observation_from_iri` hands them to a `functions_sink` for the experiment to merge. Without that, the pipeline's `function:` refs resolve to nothing
and the step degrades to a passthrough.
"""

from tvbo import datamodel as dm
from tvbo.classes.observation import populate_observation_from_iri


def test_bold_tvb_pipeline_steps_are_function_calls():
    """Merged pipeline steps are FunctionCall (they may reference a function by name)."""
    obs = dm.Observation(name="bold", iri="tvbo:BOLD_TVB", source=["S_e"])
    assert populate_observation_from_iri(obs) is True
    assert obs.pipeline, "curated pipeline was not merged"
    assert all(isinstance(s, dm.FunctionCall) for s in obs.pipeline)


def test_strided_model_merges_helper_functions_into_sink():
    """BOLD_HRF_strided ships strided_hrf + HRF helpers; they land in the sink."""
    obs = dm.Observation(name="bold", iri="tvbo:BOLD_HRF_strided", source=["S_e"])
    sink: dict = {}
    assert populate_observation_from_iri(obs, functions_sink=sink) is True

    assert "strided_hrf" in sink, "the fused strided convolution function must merge"
    for helper in ("hrf_kernel", "subsample", "prepend_history"):
        assert helper in sink, f"pipeline calls {helper} by name; it must merge"
    assert all(isinstance(f, dm.Function) for f in sink.values())

    # The pipeline references strided_hrf, so the fusion actually reaches codegen.
    names = [getattr(s, "function", None) or getattr(s, "name", None) for s in obs.pipeline]
    assert "strided_hrf" in names


def test_sink_does_not_override_existing_functions():
    """A function already present in the sink (experiment-defined) wins."""
    sentinel = dm.Function(name="hrf_kernel", description="experiment override")
    sink = {"hrf_kernel": sentinel}
    obs = dm.Observation(name="bold", iri="tvbo:BOLD_HRF_strided", source=["S_e"])
    populate_observation_from_iri(obs, functions_sink=sink)
    assert sink["hrf_kernel"] is sentinel, "curated helper must not clobber an existing one"


def test_no_sink_is_safe():
    """Merging without a sink (a model that needs no helpers) does not error."""
    obs = dm.Observation(name="bold", iri="tvbo:BOLD_HRF_strided", source=["S_e"])
    assert populate_observation_from_iri(obs) is True  # functions simply not collected


def test_experiment_with_no_functions_merges_and_renders(tmp_path):
    """An experiment with no `functions:` block that references a curated model still renders — the model's helpers merge as a plain dict, not a JsonObj.

    The merge must not go through the LinkML functions setter (which re-wraps a dict assignment into a JsonObj whose values are raw dicts, breaking
    `dict(experiment.functions)` in codegen). Guards that regression end to end.
    """
    from tvbo.classes.experiment import SimulationExperiment

    spec = tmp_path / "no_funcs.yaml"
    spec.write_text(
        "id: 999\n"
        "label: none-functions iri test\n"
        "dynamics: {name: Generic2dOscillator, iri: tvbo:Generic2dOscillator}\n"
        "integration: {duration: 2000, step_size: 4.0, transient_time: 0.0}\n"
        "observations:\n"
        "  bold: {iri: tvbo:BOLD_HRF_strided, source: [V]}\n"
    )
    e = SimulationExperiment.from_file(str(spec))

    assert isinstance(e.functions, dict), "merged functions must be a plain dict"
    assert "strided_hrf" in e.functions and "hrf_kernel" in e.functions

    code = e.render_code(format="tvboptim")
    import ast

    ast.parse(code)  # must not raise (the JsonObj form broke dict(experiment.functions))
    assert "def strided_hrf" in code and "tensordot" in code
