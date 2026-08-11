"""Tests for ``tvbo install auto7p`` — locating and linking the AUTO-07p engine.

AUTO-07p is not a PyPI package: it is a native build whose ``python/`` front-end must be linked onto the environment for ``import auto`` to resolve. These tests
cover the pure logic — detecting a valid tree, the search order, strict handling of an explicit path, and idempotent linking — without requiring a real build.
"""

import pytest
import typer

from tvbo.cli import install as inst


def _fake_auto_tree(root):
    """Create a minimal directory that :func:`_is_auto_dir` accepts."""
    (root / "python" / "auto").mkdir(parents=True)
    (root / "python" / "auto" / "__init__.py").write_text("")
    return root


def test_is_auto_dir_requires_python_front_end(tmp_path):
    assert not inst._is_auto_dir(tmp_path)
    _fake_auto_tree(tmp_path)
    assert inst._is_auto_dir(tmp_path)


def test_search_prefers_auto_dir_env(tmp_path, monkeypatch):
    tree = _fake_auto_tree(tmp_path / "engine")
    monkeypatch.setenv("AUTO_DIR", str(tree))
    monkeypatch.setattr(inst, "_DEFAULT_AUTO_LOCATIONS", ())
    assert inst._search_auto_dir() == tree.resolve()


def test_search_falls_back_to_known_locations(tmp_path, monkeypatch):
    tree = _fake_auto_tree(tmp_path / "opt-auto")
    monkeypatch.delenv("AUTO_DIR", raising=False)
    monkeypatch.setattr(inst, "_DEFAULT_AUTO_LOCATIONS", (str(tree),))
    assert inst._search_auto_dir() == tree.resolve()


def test_search_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTO_DIR", raising=False)
    monkeypatch.setattr(inst, "_DEFAULT_AUTO_LOCATIONS", (str(tmp_path / "nowhere"),))
    assert inst._search_auto_dir() is None


def test_write_link_is_idempotent(tmp_path, monkeypatch):
    tree = _fake_auto_tree(tmp_path / "engine")
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.setattr(inst, "_site_packages", lambda: site)

    pth, changed = inst._write_link(tree, force=False)
    assert changed is True
    assert pth.read_text().strip() == str(tree / "python")

    _, changed_again = inst._write_link(tree, force=False)
    assert changed_again is False

    _, forced = inst._write_link(tree, force=True)
    assert forced is True


def test_explicit_bad_auto_dir_fails_loudly(tmp_path, monkeypatch):
    """A typo'd --auto-dir must error, not silently auto-detect another install."""
    real = _fake_auto_tree(tmp_path / "real")
    monkeypatch.setenv("AUTO_DIR", str(real))  # a valid install exists elsewhere
    with pytest.raises(typer.Exit) as exc:
        inst.auto7p(
            auto_dir=str(tmp_path / "typo"),
            build=False,
            prefix="x",
            force=False,
            uninstall=False,
        )
    assert exc.value.exit_code != 0


def test_missing_install_without_build_exits(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTO_DIR", raising=False)
    monkeypatch.setattr(inst, "_DEFAULT_AUTO_LOCATIONS", (str(tmp_path / "absent"),))
    with pytest.raises(typer.Exit) as exc:
        inst.auto7p(
            auto_dir=None,
            build=False,
            prefix=str(tmp_path),
            force=False,
            uninstall=False,
        )
    assert exc.value.exit_code != 0


def test_uninstall_removes_link(tmp_path, monkeypatch):
    """--uninstall removes the .pth link and is a no-op when already absent."""
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.setattr(inst, "_site_packages", lambda: site)
    (site / inst.PTH_NAME).write_text("/some/auto/python\n")

    inst.auto7p(auto_dir=None, build=False, prefix="x", force=False, uninstall=True)
    assert not (site / inst.PTH_NAME).exists()

    # Idempotent: removing an absent link does not raise.
    inst.auto7p(auto_dir=None, build=False, prefix="x", force=False, uninstall=True)
