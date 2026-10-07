"""The 1.0 upgrade aid has to survive the shapes real recipes are written in.

``scripts/migrate_experiment_coupling.py`` moves an experiment's ``coupling:`` under its ``network:``. It rewrites lines rather than round-tripping the YAML, because these recipes are hand-authored and a round-trip reflows every one of them; the cost of that choice is that indices shift under the rewrite, and one place got the shift wrong.

Expanding ``network: *anchor`` into a ``<<:`` merge adds a line *above* the coupling, so the coupling's own start and end move down by one. Using the pre-expansion indices left the block's last line behind at the experiment level, where it is still valid YAML — a stray ``local_states: []`` on the network, a stray ``parameters:`` on the experiment — so the file parsed, the study loaded, and the recipe quietly meant something else. Six of nineteen sites in Koller2024 came out that way.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "migrate_experiment_coupling.py"


def _module():
    spec = importlib.util.spec_from_file_location("migrate_experiment_coupling", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALIASED = """experiments:
  - id: 1
    network: &sheet
      number_of_nodes: 900
    coupling: &kuramoto
      name: KuramotoCoupling
      parameters:
        a: {name: a, value: 0.01}
      local_states: [theta]
  - id: 2
    network: *sheet
    coupling:
      <<: *kuramoto
      parameters: {a: {name: a, value: 0.5}}
  - id: 3
    network: *sheet
    coupling: *kuramoto
"""


def _migrated():
    migrated, notes = _module()._migrate_text(ALIASED)
    return migrated, notes, yaml.safe_load(migrated)["experiments"]


def test_every_site_moves():
    _, notes, _ = _migrated()
    assert not [n for n in notes if "SKIPPED" in n]


def test_no_experiment_still_declares_a_coupling():
    """An experiment-level `coupling:` is what raises `TypeError` on load, so none may be left."""
    _, _, experiments = _migrated()
    assert [e for e in experiments if "coupling" in e] == []


def test_nothing_of_the_block_is_left_behind():
    """The off-by-one under alias expansion stranded the block's last line on the experiment."""
    _, _, experiments = _migrated()
    for exp in experiments:
        assert set(exp) == {"id", "network"}, exp


def test_the_declared_values_survive_the_move():
    _, _, experiments = _migrated()
    couplings = [next(iter(e["network"]["coupling"].values())) for e in experiments]
    assert [c["name"] for c in couplings] == ["KuramotoCoupling"] * 3
    assert [c["parameters"]["a"]["value"] for c in couplings] == [0.01, 0.5, 0.01]
    assert couplings[0]["local_states"] == ["theta"]


def test_the_anchor_rides_along():
    """Dropping `&kuramoto` off the re-keyed line leaves every later `*kuramoto` undefined."""
    migrated, _, _ = _migrated()
    assert "KuramotoCoupling: &kuramoto" in migrated


def test_an_included_network_says_where_the_coupling_belongs():
    """The mapping exists, it is just in another file — and that file is the answer."""
    _, notes = _module()._migrate_text(
        "experiments:\n  - id: 1\n    network: !include spec/net.yaml\n    coupling:\n      name: Linear\n"
    )
    assert "!include spec/net.yaml" in notes[0] and "SKIPPED" in notes[0]


def test_a_coupling_already_under_the_network_is_left_alone():
    already = "experiments:\n  - id: 1\n    network:\n      coupling:\n        Linear: {name: Linear}\n"
    migrated, notes = _module()._migrate_text(already)
    assert migrated == already and notes == []
