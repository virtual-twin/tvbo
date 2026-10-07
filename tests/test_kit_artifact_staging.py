"""A packed kit carries every array its frozen script reads, sourced parameters included.

`_bundle_script_artifacts` scanned the rendered source for `_load_constant(...)` only, so a kit shipped the observer operators and left behind the arrays a `Parameter.source` binds.
Both readers bake the author's own path, which resolves nowhere else, so an experiment whose model or coupling parameter is sourced could be packed, transferred and submitted, and only fail on the node when the file it names is not there.

The staging directory is keyed by basename, which is why the duplicate-basename guard is part of the contract rather than a nicety: without it one artifact silently stands in for another.
"""

import pytest
import typer

from tvbo.cli.workflow import _bundle_script_artifacts
from tvbo.data.matrix_io import resolve_staged_path


@pytest.fixture
def store(tmp_path):
    p = tmp_path / "published.h5"
    p.write_bytes(b"\x89HDF\r\n\x1a\n")
    return p


def test_a_sourced_parameter_is_staged_into_the_kit(tmp_path, store):
    """`_load_param` is the reader a sourced model/coupling parameter emits."""
    kit = tmp_path / "kit"
    n = _bundle_script_artifacts(f'w = _load_param("{store}", "wLRE")\n', kit)
    assert n == 1
    assert (kit / "constants" / "published.h5").read_bytes() == store.read_bytes()


def test_an_observer_constant_is_still_staged(tmp_path, store):
    """The original behaviour, kept: widening the scan must not drop what it already caught."""
    kit = tmp_path / "kit"
    assert _bundle_script_artifacts(f'k = _load_constant("{store}", "op")\n', kit) == 1
    assert (kit / "constants" / "published.h5").is_file()


def test_two_artifacts_sharing_a_basename_are_refused(tmp_path):
    """Basename resolution makes a name collision a silent substitution; it must not pack."""
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "same.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + d.name.encode())
    code = f'x = _load_param("{a / "same.h5"}", "w")\ny = _load_constant("{b / "same.h5"}", "op")\n'
    with pytest.raises(typer.Exit):
        _bundle_script_artifacts(code, tmp_path / "kit")


def test_resolve_staged_path_finds_a_staged_artifact_by_basename(tmp_path, monkeypatch, store):
    """On the node the author's path is gone; the kit's staging dir answers for it."""
    kit = tmp_path / "kit"
    _bundle_script_artifacts(f'w = _load_param("{store}", "wLRE")\n', kit)
    store.unlink()
    monkeypatch.setenv("TVBO_CONSTANTS_DIR", str(kit / "constants"))
    assert resolve_staged_path(store) == kit / "constants" / "published.h5"


def test_resolve_staged_path_prefers_an_existing_path(tmp_path, monkeypatch, store):
    """A run on the authoring machine must never be redirected to a same-named stand-in."""
    decoy = tmp_path / "kit" / "constants"
    decoy.mkdir(parents=True)
    (decoy / "published.h5").write_bytes(b"decoy")
    monkeypatch.setenv("TVBO_CONSTANTS_DIR", str(decoy))
    assert resolve_staged_path(store) == store


def test_a_frozen_spec_source_resolves_from_the_staging_dir(tmp_path, monkeypatch, store):
    """Spec-mode is the kit's DEFAULT code source, so it must find what frozen-mode finds.

    The rule re-renders `spec/<id>/experiment.yaml`, whose `Parameter.source` still spells the author's relative path. Only the frozen script's reader knew about the staging dir, so a kit could carry the artifact and still fail on the node it was packed for.
    """
    from tvbo.data.param_io import _resolve_path

    kit = tmp_path / "kit"
    _bundle_script_artifacts(f'w = _load_param("{store}", "wLRE")\n', kit)
    store.unlink()
    spec_dir = kit / "spec" / "35"
    spec_dir.mkdir(parents=True)
    monkeypatch.setenv("TVBO_CONSTANTS_DIR", str(kit / "constants"))
    assert _resolve_path("input/oracle/published.h5", spec_dir) == kit / "constants" / "published.h5"
