"""Resolving a Parameter's value from its declared provenance.

A parameter is a literal (``value:``), obtained (``source:`` + ``measure:``) or derived (``producer:``). These pin the contract the resolver and codegen share: a literal is materialised, anything else resolves lazily and the Parameter object itself is never touched — which is what lets the generated datamodel stay untouched.

An artifact's content address keys on the producing module's source, so editing a callable writes a NEW artifact and deliberately leaves the old one behind: nothing at write time knows whether another study still reads it. The reclaim tests pin what may then be reclaimed, and what may not.
"""

import sys
from types import SimpleNamespace

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

    nodes = [Node(id=i, label=f"n{i}", position=Coordinate(x=float(i), y=1.0, z=2.0)) for i in range(n)]
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


def test_a_producer_argument_resolves_the_bare_network_positions(producer_module):
    """`network.positions` (the observation/pipeline spelling) resolves like the legacy `network.nodes.position`, so a producer and a pipeline step read the same reference."""
    p = Parameter(name="ops", producer=_ref_producer(value="network.positions"))

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

    with pytest.raises(ValueError, match="network.positions.*network.instrength"):
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


_OWNED = np.zeros((2, 2))


def _echo_owned(k_ring=1):
    """A producer that hands back an array the recipe still owns (a module-level cache)."""
    return {"grad_op": _OWNED}


def test_readonly_does_not_freeze_the_producers_own_array():
    """Freezing the returned object in place would break the recipe's own later use of an array it still holds — so resolve must hand back a read-only VIEW, not freeze it."""
    p = Parameter(
        name="g",
        producer=FunctionCall(callable=Callable(name="_echo_owned", module=__name__), output="grad_op"),
    )

    got = param_io.resolve(p)

    assert not got.flags.writeable  # the handed-out view is protected
    assert _OWNED.flags.writeable  # the recipe's own array is untouched
    _OWNED[0, 0] = 7.0  # and still writable by its owner


# ------------------------------------------- materialise: the (file, key) codegen emits


def test_a_sourced_parameter_materialises_to_its_own_file(store, tmp_path):
    """Nothing is written: the bytes already live in the declared store."""
    p = Parameter(name="grad_op", source=store.name, measure="ops/grad_op")

    path, key = param_io.materialise(p, source_dir=tmp_path)

    assert path == store
    assert key == "ops/grad_op"


def test_a_sourced_parameter_without_measure_cannot_be_materialised(store, tmp_path):
    """A backend needs a key to read back; refuse rather than guess."""
    p = Parameter(name="grad_op", source=store.name)

    with pytest.raises(ValueError, match="measure"):
        param_io.materialise(p, source_dir=tmp_path)


def test_a_produced_parameter_is_written_to_a_content_addressed_artifact(producer_module, tmp_path):
    p = Parameter(name="grad_op", producer=_producer("grad_op"))

    path, key = param_io.materialise(p, cache_dir=tmp_path, context=None)

    assert path.exists() and key == "grad_op"
    import h5py

    with h5py.File(path) as f:
        np.testing.assert_array_equal(f["grad_op"][()], np.full((2, 2), 2.0))


def test_the_whole_bundle_is_written_so_a_sibling_output_is_a_cache_hit(producer_module, tmp_path):
    """One precompute emits every operator; siblings must not re-run it."""
    path, _ = param_io.materialise(Parameter(name="grad_op", producer=_producer("grad_op")), cache_dir=tmp_path)
    import h5py

    with h5py.File(path) as f:
        assert sorted(f) == ["boundary", "grad_op"]


def test_a_materialised_artifact_is_reused_across_processes(producer_module, tmp_path):
    """The point of the on-disk cache: the producer does not re-run for a fresh process."""
    p = Parameter(name="grad_op", producer=_producer("grad_op"))
    param_io.materialise(p, cache_dir=tmp_path)
    assert _CALLS == [2]

    param_io.clear_cache()  # a new process has no in-memory cache
    param_io.materialise(p, cache_dir=tmp_path)

    assert _CALLS == [2]  # served from disk, not recomputed


def test_differing_producer_arguments_materialise_to_different_artifacts(producer_module, tmp_path):
    """Content-addressed: an edited argument is a new artifact, never a stale hit."""
    two, _ = param_io.materialise(Parameter(name="g", producer=_producer("grad_op", k_ring=2)), cache_dir=tmp_path)
    three, _ = param_io.materialise(Parameter(name="g", producer=_producer("grad_op", k_ring=3)), cache_dir=tmp_path)

    assert two != three


def _write_producer_module(tmp_path, fill, monkeypatch):
    """A throwaway producer module whose source can be edited between calls.

    Compiled straight from the text rather than imported: two writes a fraction of a second apart share an mtime at CPython's one-second resolution, so any loader path replays the first version's cached bytecode and the test would measure nothing.
    """
    import sys
    import types

    path = tmp_path / "edited_producer.py"
    path.write_text(f"import numpy as np\n\n\ndef precompute():\n    return {{'op': np.full((2, 2), {fill})}}\n")
    mod = types.ModuleType("edited_producer")
    mod.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    monkeypatch.setitem(sys.modules, "edited_producer", mod)
    return FunctionCall(
        callable=Callable(name="precompute", module="edited_producer"),
        arguments={},
        output="op",
    )


def test_editing_the_producers_code_materialises_a_new_artifact(tmp_path, monkeypatch):
    """The key must see a code change, not only an argument change.

    Keyed on `(module, function, kwargs)` alone, an edited callable is invisible: the run reads the array from before the edit while a direct call returns the new value, and nothing raises. Pang2023 drove a whole wave model off that stale artifact.

    `clear_cache` between the two is what a second `tvbo run` is — a fresh process that imports the edited module — which is the scope the invalidation is defined at, since Python re-executes a module on import and never mid-process.
    """
    import h5py

    cache = tmp_path / "constants"
    before, _ = param_io.materialise(
        Parameter(name="op", producer=_write_producer_module(tmp_path, 1.0, monkeypatch)), cache_dir=cache
    )
    param_io.clear_cache()
    after, _ = param_io.materialise(
        Parameter(name="op", producer=_write_producer_module(tmp_path, 7.0, monkeypatch)), cache_dir=cache
    )

    assert before != after
    with h5py.File(after) as f:
        np.testing.assert_array_equal(f["op"][()], np.full((2, 2), 7.0))
    with h5py.File(before) as f:
        np.testing.assert_array_equal(f["op"][()], np.full((2, 2), 1.0))


def test_the_artifact_path_and_the_memory_key_carry_the_same_digest(tmp_path, monkeypatch):
    """Keyed apart, one of the two answers for code the other never saw.

    The digest lived only in the artifact path once, while the in-memory cache keyed on `(module, function, kwargs)`. A process that materialised, had its producer edited underneath it, and materialised again then computed the NEW path from the new source while the cache still answered on the old one — writing pre-edit arrays under a digest asserting they are post-edit. Every later run reads that file and trusts it.
    """
    producer = _write_producer_module(tmp_path, 1.0, monkeypatch)
    digest = param_io._module_source_digest("edited_producer")

    path, _ = param_io.materialise(Parameter(name="op", producer=producer), cache_dir=tmp_path / "constants")
    key = param_io._producer_key("edited_producer", "precompute", {})

    assert digest and digest in key
    assert key in param_io._CACHE
    assert path.name.startswith("edited_producer.precompute.")


def test_the_digest_is_pinned_to_the_loaded_source_not_the_bytes_on_disk(tmp_path, monkeypatch):
    """Re-reading the file would rename the artifact while the stale function fills it.

    Python does not re-execute an imported module, so a digest taken from disk on every lookup describes code that is not running. Pinning it to what was loaded keeps the key honest; the edit lands on the next process, as the edited function itself does.
    """
    _write_producer_module(tmp_path, 1.0, monkeypatch)
    loaded = param_io._module_source_digest("edited_producer")

    (tmp_path / "edited_producer.py").write_text("def precompute():\n    return {}\n")

    assert param_io._module_source_digest("edited_producer") == loaded


def test_an_unchanged_producer_still_hits_its_artifact(tmp_path, monkeypatch):
    """The invalidation must not defeat the cache — identical source, identical artifact."""
    cache = tmp_path / "constants"
    first, _ = param_io.materialise(
        Parameter(name="op", producer=_write_producer_module(tmp_path, 1.0, monkeypatch)), cache_dir=cache
    )
    param_io.clear_cache()
    second, _ = param_io.materialise(
        Parameter(name="op", producer=_write_producer_module(tmp_path, 1.0, monkeypatch)), cache_dir=cache
    )

    assert first == second


def test_a_producer_with_no_source_file_still_materialises(producer_module, tmp_path):
    """A module without readable source contributes no digest rather than raising."""
    assert param_io._module_source_digest("builtins") == ""
    path, _ = param_io.materialise(Parameter(name="grad_op", producer=_producer("grad_op")), cache_dir=tmp_path)
    assert path.exists()


def test_a_literal_parameter_cannot_be_materialised():
    """A literal inlines; asking for a file it does not have is a caller error."""
    with pytest.raises(ValueError, match="no file to read"):
        param_io.materialise(Parameter(name="a", value=0.5))


def test_the_returned_key_names_an_array_that_exists(producer_module, tmp_path):
    """The pair codegen emits must be readable: a key absent from the artifact would fail inside a simulation, far from the declaration that caused it."""
    import h5py

    p = Parameter(name="grad_op", producer=_producer("grad_op"))
    path, key = param_io.materialise(p, cache_dir=tmp_path)

    with h5py.File(path) as f:
        assert key in f


def test_an_outputless_producer_returning_a_bundle_is_refused(producer_module, tmp_path):
    """Ambiguous: the artifact holds several arrays and nothing says which to read."""
    prod = _producer("grad_op")
    prod.output = None
    p = Parameter(name="ops", producer=prod)

    with pytest.raises(ValueError, match="name the one to read"):
        param_io.materialise(p, cache_dir=tmp_path)


def test_a_typo_output_is_caught_at_materialise_not_at_run_time(producer_module, tmp_path):
    p = Parameter(name="y", producer=_producer("TYPO"))

    with pytest.raises(ValueError, match="no array 'TYPO'"):
        param_io.materialise(p, cache_dir=tmp_path)


def test_a_typo_output_is_caught_even_on_a_cache_hit(producer_module, tmp_path):
    """The write is skipped on a hit, so validation must not live in the write path."""
    param_io.materialise(Parameter(name="ok", producer=_producer("grad_op")), cache_dir=tmp_path)
    param_io.clear_cache()

    with pytest.raises(ValueError, match="no array 'TYPO'"):
        param_io.materialise(Parameter(name="y", producer=_producer("TYPO")), cache_dir=tmp_path)


def test_a_typo_measure_is_caught_at_materialise(store, tmp_path):
    p = Parameter(name="x", source=store.name, measure="ops/TYPO")

    with pytest.raises(ValueError, match="no array 'ops/TYPO'"):
        param_io.materialise(p, source_dir=tmp_path)


# ------------------------------------------------ provenance is mutually exclusive


def test_source_and_producer_together_are_refused(producer_module, tmp_path, store):
    """Two claims about where the value comes from; resolve() and materialise() would otherwise each pick their own and hand back different values for one parameter."""
    p = Parameter(name="x", source=store.name, measure="ops/grad_op", producer=_producer("grad_op"))

    with pytest.raises(ValueError, match="mutually exclusive"):
        param_io.resolve(p, source_dir=tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        param_io.materialise(p, source_dir=tmp_path, cache_dir=tmp_path)


def test_value_and_source_together_are_refused(store, tmp_path):
    p = Parameter(name="x", value=1.0, source=store.name, measure="ops/grad_op")

    with pytest.raises(ValueError, match="mutually exclusive"):
        param_io.resolve(p, source_dir=tmp_path)


def test_rebinding_a_bundle_key_cannot_poison_the_cache(producer_module):
    """A producer read whole (no `output`) hands back a copy: its dict is the cache entry."""
    whole = _producer("grad_op")
    whole.output = None
    p = Parameter(name="ops", producer=whole)

    param_io.resolve(p)["grad_op"] = "junk"

    np.testing.assert_array_equal(param_io.resolve(p)["grad_op"], np.full((2, 2), 2.0))


# ------------------------------------------------------- reclaiming superseded artifacts


class _Study:
    """The smallest thing the walk has to find a producer inside of."""

    def __init__(self, *parameters):
        self.experiments = [SimpleNamespace(parameters={p.name: p for p in parameters})]


def test_an_artifact_of_an_unedited_producer_is_never_superseded(tmp_path, monkeypatch):
    cache = tmp_path / "constants"
    p = Parameter(name="op", producer=_write_producer_module(tmp_path, 1.0, monkeypatch))
    param_io.materialise(p, cache_dir=cache)

    assert param_io.superseded_artifacts(_Study(p), cache_dir=cache) == []


def test_editing_the_producer_supersedes_the_artifact_written_before_it(tmp_path, monkeypatch):
    cache = tmp_path / "constants"
    before, _ = param_io.materialise(
        Parameter(name="op", producer=_write_producer_module(tmp_path, 1.0, monkeypatch)), cache_dir=cache
    )
    param_io.clear_cache()  # a second `tvbo run` — a fresh process
    edited = Parameter(name="op", producer=_write_producer_module(tmp_path, 2.0, monkeypatch))
    after, _ = param_io.materialise(edited, cache_dir=cache)
    assert before != after

    dead = param_io.superseded_artifacts(_Study(edited), cache_dir=cache)

    assert dead == [before]


def test_an_artifact_of_a_producer_this_study_never_declares_is_left_alone(tmp_path, monkeypatch):
    """It is very likely ANOTHER study's; reading one spec cannot decide that."""
    cache = tmp_path / "constants"
    param_io.materialise(Parameter(name="op", producer=_write_producer_module(tmp_path, 1.0, monkeypatch)), cache_dir=cache)
    stranger = cache / "someone_else.precompute.0123456789abcdef.h5"
    stranger.write_bytes(b"")

    assert stranger not in param_io.superseded_artifacts(_Study(), cache_dir=cache)
