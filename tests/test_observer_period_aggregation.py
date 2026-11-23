"""An observer that declares both a sampling ``period`` and a time ``aggregation`` is refused.

An observer with a ``period`` is a monitor: it emits its output every period as a time series. An ``aggregation`` reduces the run to one value per node. The resolver honoured whichever it read first and dropped the other without a word: ``mean`` with a period emitted the series, ``median`` reduced every step through its histogram and ignored the period (G206). Until the pair is given a meaning (a reduction over the sampled frames), it is refused by name.
"""

import pytest

from tvbo.datamodel import DerivedVariable, Dynamics, Equation, Observation, Range, StateVariable
from tvbo.templates.tvboptim.utils import resolve_reduction


def _observer(**slots):
    """A running-sum observer of source ``x`` with the given observation slots."""
    return Observation(
        name="obs",
        source=["x"],
        dynamics=Dynamics(
            name="observer",
            state_variables={"acc": StateVariable(name="acc", equation=Equation(rhs="acc + x"), equation_type="recurrence")},
            derived_variables={"out": DerivedVariable(name="out", equation=Equation(rhs="x"))},
            output=["out"],
        ),
        **slots,
    )


@pytest.mark.parametrize(
    "slots",
    [
        {"period": 5.0, "aggregation": "mean"},
        {"period": 5.0, "aggregation": "median", "histogram": Range(lo=-5.0, hi=55.0, n=512)},
    ],
    ids=["mean", "median"],
)
def test_a_period_with_an_aggregation_is_refused_by_name(slots):
    """Either aggregation beside a period raises, naming the observation, instead of honouring one slot and dropping the other."""
    with pytest.raises(ValueError, match=r"'obs' declares both `period` and `aggregation"):
        resolve_reduction(_observer(**slots))


def test_a_period_alone_and_an_aggregation_alone_still_resolve():
    """The refusal is the pair only: a monitor and a whole-run median resolve as before."""
    assert resolve_reduction(_observer(period=5.0)) is not None
    median = resolve_reduction(_observer(aggregation="median", histogram=Range(lo=-5.0, hi=55.0, n=512)))
    assert median["statistic"] == "median"
