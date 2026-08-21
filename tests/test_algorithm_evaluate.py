"""Whether an algorithm's post-tuning evaluation is materialised.

An algorithm that another one depends on contributes only its tuned state — the algorithm
that follows re-tunes from it and supersedes its observations. Its post-tuning evaluation
runs anyway, and it is not a diagnostic-sized run: it simulates the experiment's full
declared duration. On Schirner2023's per-subject cohort that is a 50 000-TR fold costing
about 10 h on each of 1096 subject jobs, producing `algorithm__fic__*` observations that
only the group experiment's Fig 3b panel reads.

`Algorithm.evaluate` is where that is declared. It defaults to True, so an experiment that
does not set it renders exactly as before.
"""

import re

import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment, database_path
from tvbo.datamodel import Algorithm, AlgorithmStage, Parameter

_EXP = database_path / "experiments" / "EI_Tuning_FIC_EIB_Optimization.yaml"


def _experiment(n_iter=3, staged=True):
    """FIC (plain) + FIC_EIB (optionally staged), the canonical depends_on pair."""
    exp = SimulationExperiment.from_file(str(_EXP))
    exp.configure()
    fe, fic = exp.algorithms["fic_eib"], exp.algorithms["fic"]
    if staged:
        fe.stages = [
            AlgorithmStage(n_iterations=n_iter, arguments=[Parameter(name="eta", value=0.10)]),
            AlgorithmStage(n_iterations=n_iter, arguments=[Parameter(name="eta", value=0.05)]),
        ]
    fe.n_iterations = fic.n_iterations = n_iter
    return exp


def _call_site(code, algo):
    """The `run_<algo>(...)` invocation, whitespace-collapsed so black's wrapping cannot break a match."""
    marker = f'if algorithm_name == "{algo}":'
    start = code.index(marker)
    nxt = code.find('if algorithm_name == "', start + len(marker))
    return " ".join(code[start : nxt if nxt != -1 else len(code)].split())


def _stage_guard(call_site, flag):
    """`run_post_tuning=<flag> and (_si == last)`, however black chose to wrap it."""
    return re.search(rf"run_post_tuning={flag} and \( ?_si == len\(_stage_defs\) - 1 ?\)", call_site)


def test_the_slot_defaults_to_evaluating():
    """Absent from the YAML, an algorithm still records its post-tuning observations."""
    assert Algorithm(name="fic").evaluate
    assert not Algorithm(name="fic", evaluate=False).evaluate


def test_by_default_every_algorithm_folds():
    code = _experiment().render_code("tvboptim")
    assert "run_post_tuning=True" in _call_site(code, "fic")


def test_declaring_evaluate_false_drops_only_that_algorithms_fold():
    """The dependency stops folding; the algorithm that consumes its state does not.

    Skipping both would drop the deliverable observations the fit exists to produce.
    """
    exp = _experiment()
    exp.algorithms["fic"].evaluate = False
    code = exp.render_code("tvboptim")
    assert "run_post_tuning=False" in _call_site(code, "fic")
    assert "run_post_tuning=False" not in _call_site(code, "fic_eib")


def test_a_staged_algorithm_folds_once_not_once_per_stage():
    """The per-stage guard is the reason a six-stage schedule pays one fold, not six."""
    code = _experiment().render_code("tvboptim")
    assert _stage_guard(_call_site(code, "fic_eib"), "True")


def test_evaluate_false_on_a_staged_algorithm_drops_every_stage():
    exp = _experiment()
    exp.algorithms["fic_eib"].evaluate = False
    code = exp.render_code("tvboptim")
    assert _stage_guard(_call_site(code, "fic_eib"), "False")


def test_the_completion_log_separates_tuning_from_the_fold():
    """The log separates the two phases.

    `fic complete! (tuning …s)` once covered the fold too, which read as a 287x per-iteration anomaly in a phase that was really a full-duration simulation.
    """
    code = _experiment().render_code("tvboptim")
    assert "_tune_t1 = time.perf_counter()" in code
    assert "tuning {_tune_t1 - _algo_t0:.1f}s" in code
    assert "post-tuning eval {time.perf_counter() - _tune_t1:.1f}s" in code
