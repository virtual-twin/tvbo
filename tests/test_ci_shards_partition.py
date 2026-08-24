"""The native CI shards partition the test suite: every file runs, and no file runs twice.

``test-native`` splits the suite so one heavy backend cannot exhaust a runner. Six shards name their files explicitly and a seventh, ``rest``, takes everything they miss — but ``rest`` is spelled as the whole ``tests/`` directory minus a list of ignores, so the claim only holds while that list keeps up with the six. When it fell behind, every named file ran a second time inside ``rest``, at a concurrency its own shard had deliberately avoided, and the heaviest tvboptim test crashed its worker there while passing in the shard that owns it.

These pin both halves of the partition, since either one failing is a silent loss: a file named by no shard is a file CI stopped running, and a file named twice is a run whose resource profile nobody chose.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

pytestmark = pytest.mark.backend_core

REPO = pathlib.Path(__file__).resolve().parents[1]
CI = REPO / ".github" / "workflows" / "ci.yml"


def _shards() -> dict[str, list[str]]:
    """Each ``test-native`` shard's ``paths``, split into tokens, keyed by shard name."""
    workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    include = workflow["jobs"]["test-native"]["strategy"]["matrix"]["include"]
    return {entry["shard"]: str(entry["paths"]).split() for entry in include}


def _named_files(shards: dict[str, list[str]]) -> set[str]:
    """Every test file the explicitly-listed shards run."""
    return {token for shard, tokens in shards.items() if shard != "rest" for token in tokens if not token.startswith("--")}


def test_the_rest_shard_ignores_every_file_another_shard_names():
    """``rest`` is the complement, so a file with its own shard must not run inside it too."""
    shards = _shards()
    ignored = {token.removeprefix("--ignore=") for token in shards["rest"] if token.startswith("--ignore=")}
    duplicated = sorted(_named_files(shards) - ignored)
    assert not duplicated, (
        f"{len(duplicated)} file(s) run in their own shard AND in `rest`: {duplicated}. "
        "Add an --ignore= for each, or `rest` runs them a second time at its own concurrency."
    )


ELSEWHERE = {
    "tests/test_docs.py": "the `docs` job renders the notebooks it drives",
    "tests/test_database_validation.py": "the `Schema validation` job runs it against the LinkML schema",
}
"""Files ``rest`` ignores that no other shard names, each with the job that does run them."""


def test_nothing_is_ignored_into_running_nowhere():
    """The other direction: an ignore that names no shard and no job would drop the file from CI silently."""
    shards = _shards()
    ignored = {token.removeprefix("--ignore=") for token in shards["rest"] if token.startswith("--ignore=")}
    orphaned = sorted(ignored - _named_files(shards) - set(ELSEWHERE))
    assert not orphaned, (
        f"`rest` ignores {orphaned}, which no other shard names. Either give the file a shard, or record "
        "the job that runs it in ELSEWHERE — an ignore alone means it runs nowhere."
    )


def test_every_path_a_shard_names_exists():
    """A shard naming a file that has been renamed or deleted is coverage lost without a failure."""
    shards = _shards()
    tokens = {token.removeprefix("--ignore=") for tokens in shards.values() for token in tokens}
    missing = sorted(path for path in tokens if not (REPO / path).exists())
    assert not missing, f"CI names test path(s) that do not exist: {missing}"
