"""TVB <-> tvbo state-variable range/boundary fidelity and drift round-trips.

For every concrete TVB simulator model that tvbo mirrors (matched by the YAML ``name:``), this asserts:

* ``tvbo -> TVB`` codegen reproduces TVB's ``state_variable_range`` and ``state_variable_boundaries`` exactly — the IC-sampling range stays finite (via the sampling ``distribution``) while a half-open clamp stays ``inf``;
* ``TVB -> tvbo -> TVB`` round-trips those losslessly;
* the tvbo-generated TVB model's drift (``dfun``) matches the original.
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.backend_tvb

ABSTRACT = {"Model", "ModelNumbaDfun", "ReducedSetBase"}
MODELS_DIR = Path(__file__).resolve().parent.parent / "tvbo" / "database" / "models"


def _tvb_classes():
    tvb_models = pytest.importorskip("tvb.simulator.models")
    from tvb.simulator.models.base import Model

    out = {}
    for mi in pkgutil.iter_modules(tvb_models.__path__):
        try:
            mod = importlib.import_module(f"tvb.simulator.models.{mi.name}")
        except Exception:
            continue
        for nm, obj in vars(mod).items():
            if inspect.isclass(obj) and issubclass(obj, Model) and nm not in ABSTRACT and not inspect.isabstract(obj):
                out[nm] = obj
    return out


def _tvbo_model_names():
    names = set()
    for f in MODELS_DIR.glob("*.y*ml"):
        m = re.search(r"(?m)^name:\s*(\S+)", f.read_text())
        if m:
            names.add(m.group(1))
    return names


def _matched():
    try:
        tvb = _tvb_classes()
    except Exception:
        return []
    return sorted(set(tvb) & _tvbo_model_names())


def _ground_truth(cls):
    """DECLARED (pre-configure) range/boundaries; None bound -> +/-inf."""
    m = cls()

    def conv(d):
        out = {}
        for k, v in (d or {}).items():
            lo = float(v[0]) if v[0] is not None else -np.inf
            hi = float(v[1]) if v[1] is not None else np.inf
            out[k] = (lo, hi)
        return out

    return conv(getattr(m, "state_variable_range", None)), conv(getattr(m, "state_variable_boundaries", None))


MATCHED = _matched()

KNOWN_DFUN_GAPS = {}
"""Models whose drift cannot match TVB, mapped to the reason — marked xfail rather than skipped, so a fix that closes a gap turns the test green (xpass) and flags the stale entry.

Empty: the discrete regime traits (Hopfield ``dynamic``, Epileptor ``modification``, EpileptorCodim3 ``N``) are expressed as a Piecewise on the parameter, so every default regime matches TVB.
"""


@pytest.mark.skipif(not MATCHED, reason="TVB not installed / no matched models")
@pytest.mark.parametrize("name", MATCHED)
def test_range_boundary_roundtrip(name):
    import types

    from tvbo.adapters.tvb import (
        _extract_dynamics,
        tvb_state_variable_boundaries,
        tvb_state_variable_ranges,
    )
    from tvbo.classes.dynamics import Dynamics

    cls = _tvb_classes()[name]
    rng, bnd = _ground_truth(cls)

    # tvbo (stored YAML) -> TVB code: must reproduce TVB's declared dicts exactly.
    m = Dynamics.from_db(name)
    gen_r = {k: (float(a), float(b)) for k, (a, b) in tvb_state_variable_ranges(m).items()}
    gen_b = {k: (float(a), float(b)) for k, (a, b) in tvb_state_variable_boundaries(m).items()}
    assert gen_r == rng, f"{name} state_variable_range: {gen_r} != TVB {rng}"
    assert gen_b == bnd, f"{name} state_variable_boundaries: {gen_b} != TVB {bnd}"

    # TVB -> tvbo -> TVB: ingest the live TVB model and re-export losslessly.
    dyn = _extract_dynamics(types.SimpleNamespace(model=cls(), initial_conditions=None))
    rt_r = {k: (float(a), float(b)) for k, (a, b) in tvb_state_variable_ranges(dyn).items()}
    rt_b = {k: (float(a), float(b)) for k, (a, b) in tvb_state_variable_boundaries(dyn).items()}
    assert rt_r == rng and rt_b == bnd, f"{name} ingest round-trip diverged"


@pytest.mark.skipif(not MATCHED, reason="TVB not installed / no matched models")
@pytest.mark.parametrize("name", MATCHED)
def test_generated_dfun_matches_tvb(name):
    """The tvbo-generated TVB model's drift must equal the original TVB model's.

    State variables are compared by NAME (tvbo and TVB may order them differently — an internal layout choice, not a dynamics difference), with the same per-variable state fed to both models. Coupling is a uniform constant so the comparison is independent of each backend's coupling-array ordering while still exercising the coupling terms.
    """
    if name in KNOWN_DFUN_GAPS:
        pytest.xfail(KNOWN_DFUN_GAPS[name])

    from tvbo.classes.dynamics import Dynamics

    cls = _tvb_classes()[name]
    code = Dynamics.from_db(name).render_code("tvb")
    ns = {}
    exec(compile(code, f"<gen:{name}>", "exec"), ns)
    GenCls = ns.get(name)
    assert GenCls is not None, f"generated class {name} not found"

    orig = cls()
    orig.configure()
    gen = GenCls()
    gen.configure()
    nmodes = int(getattr(orig, "number_of_modes", 1) or 1)
    nnodes = 4
    rng = getattr(orig, "state_variable_range", {})

    # One random state per variable name, inside its declared range.
    rs = np.random.RandomState(0)
    svals = {}
    for sv in orig.state_variables:
        a = rs.random((nnodes, nmodes))
        if sv in rng:
            lo, hi = float(rng[sv][0]), float(rng[sv][1])
            if np.isfinite(lo) and np.isfinite(hi):
                a = lo + a * (hi - lo)
        svals[sv] = a

    def assemble(model):
        return np.array([svals[sv] for sv in model.state_variables])

    def coup(model, n_default):
        """Uniform coupling sized to *model*'s own coupling array, or *n_default* terms if it declares none — the two backends may expose a different number."""
        n = len(getattr(model, "coupling_terms", []) or []) or n_default
        return np.full((n, nnodes, nmodes), 0.05)

    ncvar = len(np.atleast_1d(orig.cvar))
    d_orig = np.asarray(orig.dfun(assemble(orig), coup(orig, ncvar), local_coupling=0.0))
    d_gen = np.asarray(gen.dfun(assemble(gen), coup(gen, ncvar), local_coupling=0.0))
    oi = {n: i for i, n in enumerate(orig.state_variables)}
    gi = {n: i for i, n in enumerate(gen.state_variables)}
    for sv in orig.state_variables:
        np.testing.assert_allclose(
            d_gen[gi[sv]],
            d_orig[oi[sv]],
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{name} generated dfun for '{sv}' != TVB",
        )
