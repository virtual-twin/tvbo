"""A slot whose range is a class states its default through the dialect, and both generated forms keep a value as it was written."""

from __future__ import annotations

import pytest

from tvbo.datamodel import pydantic, schema
from tvbo.datamodel.dialect_tables import SLOT_DEFAULTS

FORMS = [schema, pydantic]


def test_n_parallel_has_one_declared_default():
    assert SLOT_DEFAULTS["Exploration"]["n_parallel"] == "auto"
    assert pydantic.Exploration.model_fields["n_parallel"].default is None


@pytest.mark.parametrize("form", FORMS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_an_absent_n_parallel_is_auto(form):
    assert form.Exploration(name="sweep").n_parallel == "auto"


@pytest.mark.parametrize("form", FORMS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_a_stated_width_stays_an_integer(form):
    assert form.Exploration(name="sweep", n_parallel=8).n_parallel == 8


@pytest.mark.parametrize("form", FORMS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_an_experiment_handle_keeps_its_type(form):
    assert form.DataRef(experiment=3, output="x").experiment == 3
    assert form.DataRef(experiment="exp_a", output="x").experiment == "exp_a"
