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
    yaml_file = _write(
        tmp_path / "a.yaml",
        """
        defaults: &defaults
          a: 1
          b: 2
        echo: *defaults
    """,
    )
    result = yaml_loader.load_as_dict(yaml_file)
    assert result["defaults"] == {"a": 1, "b": 2}
    assert result["echo"] == {"a": 1, "b": 2}


def test_merge_keys_expand_with_overrides(tmp_path: Path) -> None:
    yaml_file = _write(
        tmp_path / "a.yaml",
        """
        defaults: &defaults
          a: 1
          b: 2
        override:
          <<: *defaults
          b: 99
          c: 3
    """,
    )
    result = yaml_loader.load_as_dict(yaml_file)
    assert result["override"] == {"a": 1, "b": 99, "c": 3}


def test_explicit_duplicate_keys_still_rejected(tmp_path: Path) -> None:
    yaml_file = _write(tmp_path / "a.yaml", "a: 1\na: 2\n")
    with pytest.raises(ValueError, match="Duplicate key"):
        yaml_loader.load_as_dict(yaml_file)


def test_include_substitutes_external_file(tmp_path: Path) -> None:
    _write(
        tmp_path / "frag.yaml",
        """
        name: Fragment
        value: 42
    """,
    )
    main = _write(
        tmp_path / "main.yaml",
        """
        wrapper: !include frag.yaml
    """,
    )
    result = yaml_loader.load_as_dict(main)
    assert result["wrapper"] == {"name": "Fragment", "value": 42}


def test_include_resolves_relative_to_source_file(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(sub / "child.yaml", "id: child\n")
    main = _write(
        tmp_path / "main.yaml",
        """
        nested: !include sub/child.yaml
    """,
    )
    result = yaml_loader.load_as_dict(main)
    assert result["nested"] == {"id": "child"}


def test_anchor_scope_is_file_local_across_include(tmp_path: Path) -> None:
    _write(tmp_path / "frag.yaml", "x: *defaults\n")
    main = _write(
        tmp_path / "main.yaml",
        """
        defaults: &defaults
          a: 1
        included: !include frag.yaml
    """,
    )
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


def test_study_from_file_materialises_an_included_experiment(tmp_path: Path) -> None:
    """A modular (`!include`-split) study loads and materialises like a monolithic one.

    ``SimulationStudy.from_file`` keeps the raw experiment dicts (``_raw_experiments``) so
    ``get_experiment`` can re-materialise through the iri-aware ``from_string`` path. Both
    the datamodel load AND that raw extraction must go through ``yaml_loader`` — a plain
    ``yaml.safe_load`` chokes on the ``!include`` tag, silently emptying ``_raw_experiments``
    and dropping every experiment to the iri-unaware fallback. This guards that harmonisation.
    """
    import tvbo

    _write(
        tmp_path / "dyn.yaml",
        """
        name: Osc
        system_type: continuous
        output: [x]
        parameters: {a: {value: 1.0}}
        state_variables: {x: {equation: {rhs: '-a*x'}, initial_value: 0.1}}
    """,
    )
    study_yaml = _write(
        tmp_path / "Study.yaml",
        """
        key: T
        experiments:
          - id: 1
            label: e1
            dynamics: !include dyn.yaml
            network: {number_of_nodes: 1}
            integration: {method: heun, step_size: 0.1, duration: 1.0, transient_time: 0.0, unit: s}
    """,
    )

    study = tvbo.SimulationStudy.from_file(str(study_yaml))
    # The included experiment is captured (not emptied by a safe_load choke on !include)...
    assert list(getattr(study, "_raw_experiments", {})) == [1]
    # ...and materialises with the `!include`d Dynamics fully resolved.
    exp = study.get_experiment(1)
    assert exp.dynamics.name == "Osc"


def test_include_merges_into_a_mapping(tmp_path: Path) -> None:
    """An `!include`d fragment merges alongside a mapping's own keys and other anchors.

    Without this the two idioms do not compose — a fragment can only *replace* a whole
    slot — so every consumer of a partial fragment (a haemodynamic cascade shared by two
    models) has to copy it. Explicit keys must still win over merged ones, and an earlier
    merge over a later one, exactly as with plain anchors.
    """
    _write(
        tmp_path / "frag.yaml",
        """
        z: {rhs: 'a - z'}
        f: {rhs: 'z'}
        shared: from_fragment
    """,
    )
    main = _write(
        tmp_path / "main.yaml",
        """
        base: &base
          shared: from_anchor
          only_in_base: 1
        solo:
          <<: !include frag.yaml
          phi: {rhs: 'w'}
          z: {rhs: 'overridden'}
        combined:
          <<: [*base, !include frag.yaml]
    """,
    )
    data = yaml_loader.load_as_dict(main)

    assert set(data["solo"]) == {"z", "f", "shared", "phi"}
    assert data["solo"]["f"] == {"rhs": "z"}
    assert data["solo"]["z"] == {"rhs": "overridden"}  # explicit beats merged
    assert data["combined"]["only_in_base"] == 1
    assert data["combined"]["f"] == {"rhs": "z"}
    assert data["combined"]["shared"] == "from_anchor"  # earlier merge wins


def test_merged_include_anchors_stay_file_local(tmp_path: Path) -> None:
    """A merged fragment's anchors resolve inside the fragment and do not leak out."""
    _write(
        tmp_path / "frag.yaml",
        """
        one: &shape {rhs: 'x'}
        two: *shape
    """,
    )
    main = _write(
        tmp_path / "main.yaml",
        """
        block:
          <<: !include frag.yaml
    """,
    )
    assert yaml_loader.load_as_dict(main)["block"] == {"one": {"rhs": "x"}, "two": {"rhs": "x"}}


def test_merged_include_must_hold_a_mapping(tmp_path: Path) -> None:
    _write(tmp_path / "frag.yaml", "- 1\n- 2\n")
    main = _write(tmp_path / "main.yaml", "block:\n  <<: !include frag.yaml\n")
    with pytest.raises(Exception, match="must hold a mapping"):
        yaml_loader.load_as_dict(main)
