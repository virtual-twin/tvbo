"""Regression tests for ``tvbo.utils.yaml_loader``.

Covers the two YAML extensions the wrapper adds on top of LinkML's
``DupCheckYamlLoader``: standard merge keys (``<<: *anchor``) and
``!include`` directives, with file-local anchor scope.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tvbo.utils import yaml_loader


def _write(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content))
    return path


def test_anchors_and_aliases_round_trip(tmp_path: Path) -> None:
    yaml_file = _write(tmp_path / "a.yaml", """
        defaults: &defaults
          a: 1
          b: 2
        echo: *defaults
    """)
    result = yaml_loader.load_as_dict(yaml_file)
    assert result["defaults"] == {"a": 1, "b": 2}
    assert result["echo"] == {"a": 1, "b": 2}


def test_merge_keys_expand_with_overrides(tmp_path: Path) -> None:
    yaml_file = _write(tmp_path / "a.yaml", """
        defaults: &defaults
          a: 1
          b: 2
        override:
          <<: *defaults
          b: 99
          c: 3
    """)
    result = yaml_loader.load_as_dict(yaml_file)
    assert result["override"] == {"a": 1, "b": 99, "c": 3}


def test_explicit_duplicate_keys_still_rejected(tmp_path: Path) -> None:
    yaml_file = _write(tmp_path / "a.yaml", "a: 1\na: 2\n")
    with pytest.raises(ValueError, match="Duplicate key"):
        yaml_loader.load_as_dict(yaml_file)


def test_include_substitutes_external_file(tmp_path: Path) -> None:
    _write(tmp_path / "frag.yaml", """
        name: Fragment
        value: 42
    """)
    main = _write(tmp_path / "main.yaml", """
        wrapper: !include frag.yaml
    """)
    result = yaml_loader.load_as_dict(main)
    assert result["wrapper"] == {"name": "Fragment", "value": 42}


def test_include_resolves_relative_to_source_file(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(sub / "child.yaml", "id: child\n")
    main = _write(tmp_path / "main.yaml", """
        nested: !include sub/child.yaml
    """)
    result = yaml_loader.load_as_dict(main)
    assert result["nested"] == {"id": "child"}


def test_anchor_scope_is_file_local_across_include(tmp_path: Path) -> None:
    _write(tmp_path / "frag.yaml", "x: *defaults\n")
    main = _write(tmp_path / "main.yaml", """
        defaults: &defaults
          a: 1
        included: !include frag.yaml
    """)
    with pytest.raises(Exception):
        # Anchor &defaults is defined in main; *defaults inside frag.yaml
        # must NOT resolve. Expect a YAML composer error.
        yaml_loader.load_as_dict(main)


def test_nested_includes(tmp_path: Path) -> None:
    _write(tmp_path / "inner.yaml", "value: 7\n")
    _write(tmp_path / "middle.yaml", "deeper: !include inner.yaml\n")
    main = _write(tmp_path / "main.yaml", "wrap: !include middle.yaml\n")
    result = yaml_loader.load_as_dict(main)
    assert result["wrap"] == {"deeper": {"value": 7}}


def test_loads_accepts_yaml_string(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent("""
        defaults: &d
          x: 1
        echo:
          <<: *d
          y: 2
    """)
    result = yaml_loader.load_as_dict(yaml_text)
    assert result["echo"] == {"x": 1, "y": 2}


def test_long_yaml_string_is_not_mistaken_for_path() -> None:
    # Some callers pass full YAML content as a string (e.g. yaml.safe_dump
    # of an in-memory dict). Must not trigger Path.exists() / OSError.
    long_text = "data:\n" + "\n".join(f"  key_{i}: {i}" for i in range(500))
    result = yaml_loader.load_as_dict(long_text)
    assert result["data"]["key_0"] == 0
    assert result["data"]["key_499"] == 499


def test_missing_include_raises_filenotfound(tmp_path: Path) -> None:
    main = _write(tmp_path / "main.yaml", "broken: !include does_not_exist.yaml\n")
    with pytest.raises(FileNotFoundError):
        yaml_loader.load_as_dict(main)
