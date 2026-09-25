"""A study run must say when a figure it declared was never drawn.

Rendering is deliberately not fatal — the experiments already succeeded and their containers are on disk — but the failure was reported through ``info``, which the default WARNING level suppresses. So a render that failed said nothing at all, and the next reader hit ``figure()`` and was told the study declared no figures, which was untrue: it declared one that never got drawn. Two documentation pages failed this way in CI with no trace of the cause.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from tvbo.classes.study import StudyResult
from tvbo.cli.run import _render_study_figures


def _study(*names):
    """A stand-in study declaring figures by name and nothing else."""
    return SimpleNamespace(figures=[SimpleNamespace(name=n) for n in names])


def test_a_declared_figure_that_never_drew_is_named(tmp_path):
    """Not "declares 0 figures": the recipe declared one, and saying otherwise sends the reader to the wrong file."""
    result = StudyResult(_study("gradient"), tmp_path, tmp_path, [], [])
    with pytest.raises(KeyError) as excinfo:
        result.figure()
    assert "gradient" in str(excinfo.value)


def test_asking_for_one_by_name_says_it_was_declared(tmp_path):
    result = StudyResult(_study("gradient"), tmp_path, tmp_path, [], [])
    with pytest.raises(KeyError) as excinfo:
        result.figure("gradient")
    assert "declared but not drawn" in str(excinfo.value)


def test_a_study_that_drew_everything_it_declared_adds_no_note(tmp_path):
    """The note exists to explain an absence, so a run with nothing missing must not carry it."""
    drawn = [SimpleNamespace(name="gradient")]
    result = StudyResult(_study("gradient"), tmp_path, tmp_path, [], drawn)
    with pytest.raises(KeyError) as excinfo:
        result.figure("other")
    assert "declared but not drawn" not in str(excinfo.value)


def test_a_render_failure_is_reported_above_the_default_level(tmp_path, monkeypatch, caplog):
    """At WARNING, because INFO is below the default threshold and a failure nobody is told about is a silent one."""
    from tvbo.cli import figures as figures_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("the mesh had no vertices")

    monkeypatch.setattr(figures_mod, "render_figures", _boom)
    with caplog.at_level(logging.WARNING, logger="tvbo.cli"):
        _render_study_figures(_study("gradient"), str(tmp_path / "study.yaml"), base=Path(tmp_path))
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "a failed render logged nothing at WARNING or above"
    assert "the mesh had no vertices" in warnings[0].getMessage()
