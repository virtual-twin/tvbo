"""A slot whose range is a class states its default through the dialect, and the generated class keeps a value as it was written."""

from __future__ import annotations

from tvbo.datamodel import DataRef, Exploration
from tvbo.datamodel.dialect_tables import SLOT_DEFAULTS


def test_n_parallel_has_one_declared_default():
    assert SLOT_DEFAULTS["Exploration"]["n_parallel"] == "auto"
    assert Exploration.model_fields["n_parallel"].default is None


def test_an_absent_n_parallel_is_auto():
    assert Exploration(name="sweep").n_parallel == "auto"


def test_a_stated_width_stays_an_integer():
    assert Exploration(name="sweep", n_parallel=8).n_parallel == 8


def test_an_experiment_handle_keeps_its_type():
    assert DataRef(experiment=3, output="x").experiment == 3
    assert DataRef(experiment="exp_a", output="x").experiment == "exp_a"
