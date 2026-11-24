"""A figure's render rule waits for every run and analysis its panels read, a grid cell's, an inset's and a grid template's mesh included.

The rule learns its inputs from the bindings `figure_workflow._panel_bindings` lists, and that walked a panel's own layers, annotations and surface but not the grid cells (`cells:`) and insets the panel draws, nor the mesh a grid's `cell:` template gives every cell. A figure whose only binding to an experiment or an analysis sat in one of them got no input for it, so its rule waited for nothing.

Self-contained: the kit's plan is a dict, and the one container is an empty file that is located, never opened.
"""

from __future__ import annotations

import pytest

import tvbo.datamodel.pydantic as P
from tvbo.adapters import figure_workflow as fw
from tvbo.data.dataref import analysis_container_path
from tvbo.utils.study_layout import study_path

PLAN = {"key": "3", "rule_name": "exp_3", "axes": [], "result_stem": "exp-3_model-M_result"}
"""The kit's plan of experiment 3, one run writing one result."""

LAYER = P.Layer(used=P.DataRef(experiment="3", output="x"), encoding=P.Encoding(x="t", y="x"))


@pytest.mark.parametrize(
    "panel",
    [
        {"kind": "grid", "cells": [{"kind": "cartesian", "layers": [LAYER]}]},
        {"kind": "cartesian", "insets": [{"kind": "cartesian", "bounds": [0.6, 0.6, 0.3, 0.3], "layers": [LAYER]}]},
    ],
    ids=["grid-cell", "inset"],
)
def test_a_run_read_only_in_a_grid_cell_or_an_inset_is_an_input_of_the_figures_rule(tmp_path, panel):
    """The rule's inputs were empty, so the figure did not wait for experiment 3 (G211)."""
    figure = P.Figure(name="fig", layout="a", panels={"a": P.Panel(panel_key="a", **panel)})
    (context,) = fw.figure_contexts([figure], base_dir=tmp_path, exp_plans=[PLAN])
    assert context["inputs"] == [{"value": "rules.exp_3.output", "raw": True}]


def test_the_mesh_a_grid_template_gives_every_cell_is_an_input_of_the_figures_rule(tmp_path):
    """Koller2024 Fig. 4e and f draw every cell on the mesh of `cell: {kind: surface, surface: {used: ...}}`, and the mesh's analysis was no input of the rule (G211). An analysis binding is an input as its resolved container, the route `_figure_inputs` takes for every binding that names no experiment of the kit."""
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    mesh = analysis_container_path(results, "mesh")
    mesh.touch()
    panel = P.Panel(
        panel_key="a",
        kind="grid",
        cell={"kind": "surface", "surface": {"used": {"analysis": "mesh"}}},
        cells=[
            {"kind": "surface", "surface": {"view": "lateral"}, "layers": [{"used": {"analysis": "values", "output": "v"}}]}
        ],
    )
    (context,) = fw.figure_contexts([P.Figure(name="fig", layout="a", panels={"a": panel})], base_dir=tmp_path)
    assert {"value": str(mesh.resolve()), "raw": False} in context["inputs"]
