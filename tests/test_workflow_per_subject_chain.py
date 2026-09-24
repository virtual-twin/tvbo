"""A per-subject experiment that reads another per-subject experiment waits for, and finds, that subject's result.

Three gaps kept a chain of per-subject experiments (fit -> forward run -> driven circuit) from running as a workflow: an event parameter's `used:` edge registered no dependency, so the driven experiment could start before the recording it plays existed; a dependency on a subject-fanned source waited for the whole cohort instead of the one subject; and a rule looked for its sources under its own output directory, where they never are.
"""

from __future__ import annotations

from tvbo.classes.experiment import SimulationExperiment
from tvbo.cli import _workflow
from tvbo.cli.workflow import _render_template

_EVENT_SOURCED = """
id: 61
dynamics:
  name: Integrator
  parameters:
    drive: {value: 0.0}
  state_variables:
    x: {equation: {rhs: "drive"}, initial_value: 0.0}
  output: [x]
network:
  number_of_nodes: 2
integration: {method: euler, duration: 10.0, step_size: 1.0, transient_time: 0.0}
events:
  drive:
    event_type: stimulus
    parameters:
      data:
        used: {experiment: 60, output: recording}
"""


def test_a_stimulus_playing_another_run_depends_on_it():
    exp = SimulationExperiment.from_string(_EVENT_SOURCED)
    assert _workflow.plan(study_key="s", experiment=exp, backend="tvboptim").depends_on == ["60"]


def _ep(key, axes, depends_on=()):
    return {
        "key": key,
        "rule_name": f"exp_{key}",
        "spec_relpath": f"spec/{key}/experiment.yaml",
        "select": None,
        "backend": "tvboptim",
        "out_dir": "results",
        "result_stem": f"exp-{key}_result",
        "container": None,
        "block": {},
        "axes": axes,
        "depends_on": list(depends_on),
    }


_SUBJECTS = [{"name": "subject", "parameter": "dataset.active_subject", "values": ["01", "02"]}]


def test_a_subject_waits_for_its_own_source_and_searches_the_shared_root():
    smk = _render_template(
        "snakemake/study.smk.mako",
        exp_plans=[_ep("60", _SUBJECTS), _ep("61", _SUBJECTS, depends_on=["60"])],
        block={},
        bundled_code=False,
    )
    rule = smk[smk.index("rule exp_61:") :]
    inputs = rule[rule.index("input:") : rule.index("output:")]
    assert 'f"{OUT_DIR}/60/sub-{{subject}}_exp-60_result.h5"' in inputs
    assert "expand(" not in inputs
    assert 'f"--results-root {OUT_DIR} "' in rule


def test_a_source_outside_the_kit_leaves_no_empty_input_block():
    """A warm start from a result staged by hand has no rule to wait for; an `input:` with nothing under it is a Snakefile syntax error."""
    smk = _render_template(
        "snakemake/study.smk.mako", exp_plans=[_ep("30", _SUBJECTS, depends_on=["34"])], block={}, bundled_code=False
    )
    rule = smk[smk.index("rule exp_30:") :]
    assert "input:" not in rule
    assert 'f"--results-root {OUT_DIR} "' in rule


def test_a_grid_source_is_still_awaited_whole():
    grid = [{"name": "k", "parameter": "Model.k", "values": [1.0, 2.0]}]
    smk = _render_template(
        "snakemake/study.smk.mako",
        exp_plans=[_ep("5", grid), _ep("61", _SUBJECTS, depends_on=["5"])],
        block={},
        bundled_code=False,
    )
    rule = smk[smk.index("rule exp_61:") :]
    assert "expand(" in rule[rule.index("input:") : rule.index("output:")]
