"""An `includes:` argument override has to reach the rule it overrides.

A combined loop runs an included algorithm's update rule alongside its own. `get_all_hyperparams` merges both sets into ONE variable per name and lets the enclosing algorithm's value win a collision, so an included rule whose hyperparameter shares a name — `eta` under an algorithm that also has an `eta` — read the outer number. The declared override was computed and then dropped at the call site.

It is a silent divergence, not a crash: the combined FIC+EIB loop ran its feedback-inhibition step at the EIB rate of 0.005 instead of the declared 0.1, twenty times too weak to hold excitatory activity at target. Mean S_e drifted from 0.23 to 0.49 over the loop and the fitted FC scored 0.26 against the reference workflow's 0.70.
"""

from __future__ import annotations

import pytest

from tvbo.classes.experiment import SimulationExperiment

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def code():
    exp = SimulationExperiment.from_db("EI_Tuning_FIC_EIB_Optimization")
    return exp.render_code("tvboptim")


def _call(code: str, rule: str, *, occurrence: int) -> str:
    """The *occurrence*-th emitted call to *rule*, up to its closing paren."""
    parts = code.split(f"new_{rule} = {rule}_update(")
    return parts[occurrence].split(")")[0]


def test_the_included_rule_is_called_with_the_declared_override(code):
    """`includes: [{algorithm: fic, arguments: [{name: eta, value: 0.1}]}]` — 0.1 is what FIC must run at."""
    combined = _call(code, "J_i", occurrence=2)
    assert "0.1," in combined, combined


def test_the_enclosing_algorithms_own_rules_still_take_the_traced_value(code):
    """Only the shadowed name is pinned; a staged schedule keeps varying the outer rate."""
    assert "eta," in _call(code, "wLRE", occurrence=1)
    assert "eta," in _call(code, "wFFI", occurrence=1)


def test_the_standalone_algorithm_is_untouched(code):
    """Run on its own, FIC has no enclosing scope to collide with and reads its own hyperparameter."""
    assert "eta," in _call(code, "J_i", occurrence=1)
