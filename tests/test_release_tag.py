"""A release has one version, which GitHub, docker, the docs and PyPI must all read the same way.

`tvbo/__init__.py` holds it in PEP 440 (`1.0.0rc1`), the spelling hatch writes into the wheel. The git tag spells it in semver (`v1.0.0-rc.1`), the spelling docker's `type=semver` and the docs image pull parse. A `-` in the tag is what makes a release a pre-release on GitHub, a dry run for the docs, and an image without `latest`. `make print-tag` is the one mapping between the two spellings, and these tests pin it against `packaging`'s PEP 440 and the semver.org grammar.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from packaging.version import Version

pytestmark = pytest.mark.backend_core

REPO = Path(__file__).resolve().parents[1]

SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)
"""The grammar semver.org publishes, which is what docker's metadata action parses a tag with."""

VERSIONS = ("0.5.3", "1.0.0", "1.0.0a1", "1.0.0b2", "1.0.0rc1", "1.0.0rc12", "2.10.3rc1")


def _make(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["make", "-s", *args], cwd=REPO, capture_output=True, text=True)


def _tag(version: str) -> str:
    return _make("print-tag", f"PKG_VERSION={version}").stdout.strip()


def _release_sh(function: str, *args: str) -> str:
    """*function* from `scripts/release.sh`, called on *args* without running the script."""
    source = f'eval "$(sed -n "/^{function}()/,/^}}/p" scripts/release.sh)"; {function} {" ".join(args)}'
    return subprocess.run(["bash", "-c", source], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


@pytest.mark.parametrize("version", VERSIONS)
def test_the_tag_is_the_same_version_in_semver_spelling(version):
    """Docker reads the tag as semver, PyPI normalises it back to the package version, and both see a pre-release exactly when the tag carries a `-`."""
    tag = _tag(version)

    assert tag.startswith("v")
    assert SEMVER.match(tag[1:]), f"{tag} is not semver, so docker would publish no version image"
    assert Version(tag[1:]) == Version(version), f"{tag} is not {version} to PyPI"
    assert ("-" in tag) == Version(version).is_prerelease


def test_the_checked_out_version_passes_its_own_gate():
    """The gate every tag-triggered publish runs accepts the tag `make release` creates, and refuses the PEP 440 spelling."""
    tag = _make("print-tag").stdout.strip()

    assert _make("check-tag", f"TAG={tag}").returncode == 0
    if Version(tag[1:]).is_prerelease:
        assert _make("check-tag", f"TAG=v{_make('print-version').stdout.strip()}").returncode != 0


@pytest.mark.parametrize(("a", "b"), [(a, b) for a in VERSIONS for b in VERSIONS if a != b])
def test_the_release_script_orders_versions_as_pep_440_does(a, b):
    """`sort -V` alone ranks `1.0.0rc1` above `1.0.0`, which would refuse the final release after its candidate."""
    assert _release_sh("highest", a, b) == str(max(Version(a), Version(b)))


NPM_BUMPS = {
    "1.0.0rc1": ("1.0.0", "1.0.0", "1.0.0"),
    "1.2.3rc1": ("1.2.3", "1.3.0", "2.0.0"),
    "1.2.0rc1": ("1.2.0", "1.2.0", "2.0.0"),
    "0.9.0rc1": ("0.9.0", "0.9.0", "1.0.0"),
    "2.0.0a1": ("2.0.0", "2.0.0", "2.0.0"),
    "1.0.1b2": ("1.0.1", "1.1.0", "2.0.0"),
    "1.2.3": ("1.2.4", "1.3.0", "2.0.0"),
    "0.5.3": ("0.5.4", "0.6.0", "1.0.0"),
}
"""What npm's `semver.inc` returns for patch, minor and major on each version's semver spelling."""


@pytest.mark.parametrize("version", NPM_BUMPS)
def test_a_bump_releases_a_candidate_already_at_its_level(version):
    """`BUMP=` on a candidate releases it where the candidate sits at that level, so `0.9.0rc1` is followed by `0.9.0`, never skipped to `0.9.1`."""
    assert tuple(_release_sh("bump_version", version, level) for level in ("patch", "minor", "major")) == NPM_BUMPS[version]
