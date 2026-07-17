"""Resolving a Parameter's value from its declared provenance.

A parameter is a literal (``value:``), obtained (``source:`` + ``measure:``) or derived
(``producer:``). These pin the contract the resolver and codegen share: a literal is
materialised, anything else resolves lazily and the Parameter object itself is never
touched — which is what lets the generated datamodel stay untouched.
"""

import sys

import numpy as np
import pytest

from tvbo.data import param_io
from tvbo.datamodel.schema import Callable, FunctionCall, Parameter

pytest.importorskip("h5py")


@pytest.fixture(autouse=True)
def _clear():
    param_io.clear_cache()
    yield
    param_io.clear_cache()


@pytest.fixture
def store(tmp_path):
    """A binary store holding two named arrays."""
    import h5py

    path = tmp_path / "operators.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("ops/grad_op", data=np.arange(6.0).reshape(2, 3))
        f.create_dataset("ops/boundary", data=np.array([True, False]))
    return path


# --------------------------------------------------------------------- literal values

def test_a_literal_value_is_returned_untouched():
    assert param_io.resolve(Parameter(name="a", value=0.5)) == 0.5


def test_a_literal_is_not_lazy():
    """The rule codegen branches on: literals inline, everything else resolves."""
    assert not param_io.is_lazy(Parameter(name="a", value=0.5))


def test_a_parameter_with_no_value_at_all_resolves_to_none():
    """A free parameter declares no value; that is the caller's business, not an error."""
    assert param_io.resolve(Parameter(name="free_one")) is None


# ---------------------------------------------------------------------------- sources

def test_a_sourced_array_is_read_from_the_store(store, tmp_path):
    p = Parameter(name="grad_op", source=store.name, measure="ops/grad_op")

    got = param_io.resolve(p, source_dir=tmp_path)

    np.testing.assert_array_equal(got, np.arange(6.0).reshape(2, 3))


def test_a_relative_source_resolves_against_the_spec_directory(store, tmp_path):
    """A kit carries its companion next to the spec, so relative must mean spec-relative."""
    p = Parameter(name="grad_op", source="operators.h5", measure="ops/grad_op")

    got = param_io.resolve(p, source_dir=tmp_path)

    assert got.shape == (2, 3)


def test_a_sourced_parameter_is_lazy_and_never_materialised(store, tmp_path):
    """The whole point: a 66MB operator must not end up in `value` or in a YAML dump."""
    p = Parameter(name="grad_op", source=store.name, measure="ops/grad_op")
    assert param_io.is_lazy(p)

    param_io.resolve(p, source_dir=tmp_path)

    assert p.value is None


def test_a_missing_source_raises_naming_the_parameter(tmp_path):
    p = Parameter(name="grad_op", source="absent.h5", measure="x")

    with pytest.raises(ValueError, match="grad_op"):
        param_io.resolve(p, source_dir=tmp_path)


def test_a_source_holding_several_arrays_requires_a_measure(store, tmp_path):
    p = Parameter(name="grad_op", source=store.name)

    with pytest.raises(ValueError, match="measure"):
        param_io.resolve(p, source_dir=tmp_path)


# -------------------------------------------------------------------------- producers

_CALLS = []


def _fake_precompute(k_ring=1):
    """Stands in for a study's precompute; records calls so caching is observable."""
    _CALLS.append(k_ring)
    return {"grad_op": np.full((2, 2), float(k_ring)), "boundary": np.zeros(2)}


@pytest.fixture
def producer_module():
    mod = sys.modules[__name__]
    _CALLS.clear()
    yield mod.__name__


def _producer(output, k_ring=2, name="_fake_precompute"):
    return FunctionCall(
        callable=Callable(name=name, module=__name__),
        arguments={"k_ring": {"name": "k_ring", "value": k_ring}},
        output=output,
    )


def test_a_produced_value_is_computed_and_selected_by_output(producer_module):
    p = Parameter(name="grad_op", producer=_producer("grad_op"))

    got = param_io.resolve(p)

    np.testing.assert_array_equal(got, np.full((2, 2), 2.0))


def test_a_producer_runs_once_across_parameters_sharing_it(producer_module):
    """One precompute returning a bundle serves many parameters — it must not re-run."""
    a = Parameter(name="grad_op", producer=_producer("grad_op"))
    b = Parameter(name="boundary", producer=_producer("boundary"))

    param_io.resolve(a)
    param_io.resolve(b)
    param_io.resolve(a)

    assert _CALLS == [2]


def test_differing_arguments_are_computed_separately(producer_module):
    """The cache is content-addressed, so k_ring=2 must not serve k_ring=3."""
    two = param_io.resolve(Parameter(name="g", producer=_producer("grad_op", k_ring=2)))
    three = param_io.resolve(Parameter(name="g", producer=_producer("grad_op", k_ring=3)))

    assert two[0, 0] == 2.0 and three[0, 0] == 3.0
    assert _CALLS == [2, 3]


def test_a_produced_parameter_is_lazy_and_never_materialised(producer_module):
    p = Parameter(name="grad_op", producer=_producer("grad_op"))
    assert param_io.is_lazy(p)

    param_io.resolve(p)

    assert p.value is None


def test_an_unknown_output_raises_listing_what_the_producer_returned(producer_module):
    p = Parameter(name="typo", producer=_producer("no_such_key"))

    with pytest.raises(ValueError, match="no output 'no_such_key'"):
        param_io.resolve(p)


def test_an_unimportable_producer_raises_pointing_at_code_source():
    p = Parameter(
        name="x",
        producer=FunctionCall(callable=Callable(name="nope", module="not_a_module")),
    )

    with pytest.raises(ValueError, match="code_source"):
        param_io.resolve(p)


# ------------------------------------------------- entity references in producer args

def _echo_positions(positions=None, hemi="lh"):
    """Stands in for a producer that needs the network's geometry."""
    return {"pos": np.asarray(positions), "hemi": hemi}


def _network(n=3):
    from tvbo.classes.network import Network
    from tvbo.datamodel.schema import Coordinate, Node

    nodes = [
        Node(id=i, label=f"n{i}", position=Coordinate(x=float(i), y=1.0, z=2.0))
        for i in range(n)
    ]
    return Network(label="net", nodes=nodes)


def _ref_producer(value="network.nodes.position"):
    return FunctionCall(
        callable=Callable(name="_echo_positions", module=__name__),
        arguments={
            "positions": {"name": "positions", "value": value},
            "hemi": {"name": "hemi", "value": "lh"},
        },
        output="pos",
    )


def test_a_producer_argument_may_reference_the_network(producer_module):
    """`positions: network.nodes.position` must arrive as the array, not the string."""
    p = Parameter(name="ops", producer=_ref_producer())

    got = param_io.resolve(p, context=_network(3))

    assert got.shape == (3, 3)
    np.testing.assert_array_equal(got[:, 0], [0.0, 1.0, 2.0])


def test_a_non_reference_argument_stays_a_literal(producer_module):
    """`hemi: lh` is a plain string — only fully-qualified network.* is a reference."""
    p = Parameter(name="ops", producer=_ref_producer())
    p.producer.output = "hemi"

    assert param_io.resolve(p, context=_network()) == "lh"


def test_a_reference_without_context_raises_asking_for_one(producer_module):
    p = Parameter(name="ops", producer=_ref_producer())

    with pytest.raises(ValueError, match="no context was given"):
        param_io.resolve(p)


def test_an_unsupported_reference_lists_the_supported_forms(producer_module):
    p = Parameter(name="ops", producer=_ref_producer(value="network.nonsense"))

    with pytest.raises(ValueError, match="network.nodes.position"):
        param_io.resolve(p, context=_network())


# ------------------------------------------------------------- shared-array immutability

def test_a_resolved_array_is_read_only(store, tmp_path):
    """One buffer is shared, so an in-place write must raise here, not corrupt a later run."""
    p = Parameter(name="grad_op", source=store.name, measure="ops/grad_op")

    got = param_io.resolve(p, source_dir=tmp_path)

    with pytest.raises(ValueError, match="read-only"):
        got *= 2.0


def test_a_produced_array_is_read_only(producer_module):
    p = Parameter(name="grad_op", producer=_producer("grad_op"))

    got = param_io.resolve(p)

    with pytest.raises(ValueError, match="read-only"):
        got[0, 0] = 99.0


def test_rebinding_a_bundle_key_cannot_poison_the_cache(producer_module):
    """A producer read whole (no `output`) hands back a copy: its dict is the cache entry."""
    whole = _producer("grad_op")
    whole.output = None
    p = Parameter(name="ops", producer=whole)

    param_io.resolve(p)["grad_op"] = "junk"

    np.testing.assert_array_equal(param_io.resolve(p)["grad_op"], np.full((2, 2), 2.0))
