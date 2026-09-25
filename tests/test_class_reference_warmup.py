"""A declared `warmup` must survive the swap from a TVB monitor to its tvboptim equivalent.

`ClassReference.warmup` exists because an external class is a black box: how much signal it convolves before its first output means anything cannot be inferred from the length of what it returns, since a class that pads or trims would make that inference wrong and still return a plausible-looking series. So the recipe declares it, and codegen picks the settle convention from it — present means kernel-bearing (cut the settle at the INPUT, leave the support in front of t = 0 for the class to eat), absent means memoryless (drop the settle's own output samples).

`adapt_class_reference_for_tvboptim` rebuilds the reference when it points at a TVB monitor that has a tvboptim equivalent, and a rebuild that forgets a declared field silently demotes a kernel-bearing monitor to the memoryless convention — which opens the kernel on zeros and is wrong by the whole warm-up over its entire support. Swapping which class runs must never change how much signal that class eats.
"""

import pytest

from tvbo.templates.tvboptim.utils import adapt_class_reference_for_tvboptim


def _reference(module, name, **constructor_args):
    return {
        "name": name,
        "module": module,
        "constructor_args": dict(constructor_args),
        "constructor_arg_codes": {},
        "call_args": {},
        "accepts_voi": False,
        "extra_imports": [],
        "warmup_steps": 5000,
    }


@pytest.mark.parametrize(
    "module,name,args",
    [
        ("tvb.simulator.monitors", "Bold", {"hrf_kernel": "FirstOrderVolterra", "period": 720.0}),
        ("tvb.simulator.monitors", "TemporalAverage", {"period": 4.0}),
        ("tvb.simulator.monitors", "SubSample", {"period": 4.0}),
    ],
)
def test_an_adapted_tvb_monitor_keeps_its_declared_warmup(module, name, args):
    adapted = adapt_class_reference_for_tvboptim(_reference(module, name, **args), obs=None, dt=1.0)
    assert adapted is not None, f"{name} was not adapted at all"
    assert adapted["warmup_steps"] == 5000, f"{name} lost its declared warm-up: {adapted.get('warmup_steps')!r}"


@pytest.mark.parametrize(
    "module,name",
    [("tvboptim.observations.tvb_monitors.bold", "HRFBold"), ("my.study.monitors", "CustomMonitor")],
)
def test_a_reference_that_is_not_rebuilt_keeps_it_too(module, name):
    """The control: these never went through the rebuild, so they pin that the test measures the rebuild."""
    adapted = adapt_class_reference_for_tvboptim(_reference(module, name), obs=None, dt=1.0)
    assert adapted["warmup_steps"] == 5000


def test_a_reference_declaring_no_warmup_is_memoryless():
    """Absent means memoryless, and must not become a truthy default through the rebuild."""
    ref = _reference("tvb.simulator.monitors", "TemporalAverage", period=4.0)
    del ref["warmup_steps"]
    adapted = adapt_class_reference_for_tvboptim(ref, obs=None, dt=1.0)
    assert not adapted["warmup_steps"]
