"""`scripts/spec_gaps.py`: which reader a file counts as, and what it counts as read."""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import spec_gaps  # noqa: E402


@pytest.mark.backend_core
@pytest.mark.parametrize(
    ("relative", "reader"),
    [
        ("tvbo/adapters/tvb.py", "tvb"),
        ("tvbo/templates/tvb/tvbo-tvb-model.py.mako", "tvb"),
        ("tvbo/adapters/julia_model.py", "julia"),
        ("tvbo/templates/autodiff/tvbo-jax-sim.py.mako", "jax"),
        ("tvbo/adapters/smallscale/lowering.py", "smallscale"),
        ("tvbo/templates/base/dfun.mako", "core"),
        ("tvbo/adapters/bids.py", "core"),
        ("tvbo/classes/dynamics.py", "core"),
        ("scripts/ontology/gen_abox.py", "core"),
        ("tvbo/datamodel/pydantic.py", None),
    ],
)
def test_each_file_reads_for_its_backend_or_for_every_backend(relative: str, reader: str | None):
    assert spec_gaps.reader_of(REPO / relative) == reader


@pytest.mark.backend_core
def test_a_name_in_a_docstring_is_not_a_read(tmp_path: pathlib.Path):
    module = tmp_path / "m.py"
    module.write_text(
        '"""Mentions documented_only."""\n\nPATTERN = r"^keyed_by_regex:"\n\n\ndef f(x):\n    return x.attribute_read, g(keyword_passed=1)\n'
    )

    assert {"keyed_by_regex", "attribute_read", "keyword_passed"} <= spec_gaps.names_in(module)
    assert "documented_only" not in spec_gaps.names_in(module)
