"""A time-varying stochastic parameter under a trial ensemble draws noise for the settle too.

The emitter pre-draws one noise vector per stochastic parameter per trial, and its length has to
cover the whole scan. The scan opens at ``-transient_time``, so a settle is part of that length --
and the branch that was supposed to add it referenced a name the template never defined, so every
experiment combining a settle, ``n_trials`` and a time-varying parameter distribution died at run
time with ``NameError: name '_t_total' is not defined``. Nothing caught it because the combination
appears in no other test and only a documentation notebook exercised it.
"""

from __future__ import annotations

import copy
import re

import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

BASE = {
    "id": 1,
    "label": "stochastic-parameter ensemble",
    "dynamics": {
        "name": "MiniOsc",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {
            "a": {
                "value": 1.0,
                "distribution": {
                    "name": "normal",
                    "parameters": {"loc": {"value": 1.0}, "scale": {"value": 0.1}},
                    "domain": {"lo": 0.5, "hi": 1.5},
                    "axis": "time",
                    "seed": 5,
                },
            }
        },
        "state_variables": {"x": {"equation": {"rhs": "-a * x"}, "initial_value": 0.1}},
    },
    "integration": {"method": "heun", "step_size": 0.1, "duration": 1.0, "transient_time": 2.0, "unit": "s"},
    "explorations": {
        "ens": {
            "name": "ens",
            "mode": "product",
            "n_trials": 3,
            "record": ["x"],
            "space": [{"parameter": "MiniOsc.a", "explored_values": [1.0]}],
        }
    },
}


def _render(transient: float) -> str:
    spec = copy.deepcopy(BASE)
    spec["integration"]["transient_time"] = transient
    return SimulationExperiment(**spec).render_code("tvboptim")


def _steps(code: str) -> str:
    hit = re.search(r"_n_steps_stoch\s*=\s*(.+)", code)
    assert hit, "the trial ensemble emitted no stochastic-parameter draw at all"
    return hit.group(1).strip()


@pytest.mark.parametrize("transient,total", [(2.0, 3.0), (0.0, 1.0)])
def test_the_draw_spans_the_settle_and_the_window(transient, total):
    expr = _steps(_render(transient))
    assert eval(expr) == int(total / 0.1) + 2  # noqa: S307 — the emitted expression is the thing under test


def test_the_emitted_draw_names_nothing_the_module_never_defines():
    """The failure this pins is a NameError at run time, which no rendering test sees."""
    code = _render(2.0)
    assert "_t_total" not in code
    expr = _steps(code)
    assert not re.search(r"[A-Za-z_]\w*", expr.replace("int", "").replace("float", "")), (
        f"the draw length must be a literal expression, got {expr!r}: a name here is only bound if "
        "some other branch happened to define it"
    )
