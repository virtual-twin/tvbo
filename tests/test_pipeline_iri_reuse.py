"""A pipeline step names a curated observation by ``iri`` and states only what differs from it.

The five BOLD variants differ in one thing — their hemodynamic kernel — and repeat the interim average, the convolution and the output stride in each file. That is how one pipeline becomes five copies that drift: a fix to the output stride lands in whichever variant its author was reading. The same reuse an experiment already has (``experiments: [{iri: …, <overrides>}]``) is what a pipeline step lacked.

Splicing rather than gap-filling, because the target is a list: an observation whose pipeline is several steps contributes all of them, in order. So an override has a single step to apply to only when the referenced observation has exactly one, and naming a multi-step observation while also overriding is an error rather than a guess about which step was meant.
"""

from __future__ import annotations

import pytest
import yaml

from tvbo.utils import yaml_loader as Y


def _expand(doc: str) -> dict:
    return Y._normalize_loaded(yaml.safe_load(doc))


def test_a_step_naming_a_single_step_observation_becomes_that_step():
    """The reused step arrives whole — its equation and every declared derivation on it."""
    data = _expand("""
name: Demo
pipeline:
    - iri: tvbo:observation/temporal_average_tvb
""")
    (step,) = data["pipeline"]
    assert step["equation"]["rhs"] == "window_mean(X, window_size)"
    assert step["equation"]["parameters"]["window_size"]["equation"]["rhs"] == "iround(period / dt)"


def test_the_entrys_own_keys_are_merged_over_the_curated_step():
    """The point of the reference: state the difference, not a second copy."""
    data = _expand("""
name: Demo
pipeline:
    - iri: tvbo:observation/temporal_average_tvb
      name: interim_average
""")
    (step,) = data["pipeline"]
    assert step["name"] == "interim_average"
    assert step["equation"]["rhs"] == "window_mean(X, window_size)", "the override replaced more than it named"


def test_a_multi_step_observation_contributes_all_of_its_steps_in_order():
    data = _expand("""
name: Demo
pipeline:
    - iri: tvbo:observation/bold_tvb
""")
    assert [s["name"] for s in data["pipeline"]] == [
        "temporal_average_interim",
        "hemodynamic_response",
        "convolve",
        "subsample_to_period",
        "volterra_transform",
    ]


def test_overriding_a_multi_step_reference_raises_rather_than_guessing():
    """Silently applying the override to one of five steps would be a monitor nobody declared."""
    with pytest.raises(ValueError, match="has 5 steps"):
        _expand("""
name: Demo
pipeline:
    - iri: tvbo:observation/bold_tvb
      name: whoops
""")


def test_a_step_that_names_nothing_is_left_exactly_as_written():
    """The control: this pins that the expansion is keyed on the reference, not on being a pipeline."""
    doc = """
name: Demo
pipeline:
    - name: a
      equation: {rhs: "X + 1"}
    - name: b
      function: subsample
"""
    assert _expand(doc)["pipeline"] == yaml.safe_load(doc)["pipeline"]


def test_the_expansion_reaches_an_observation_written_inline_in_a_recipe():
    """A pipeline nested under `observations:` expands like a curated one, or the reuse works in the database and not in a recipe."""
    data = _expand("""
label: demo experiment
observations:
    my_average:
        pipeline:
            - iri: tvbo:observation/temporal_average_tvb
""")
    (step,) = data["observations"]["my_average"]["pipeline"]
    assert step["equation"]["rhs"] == "window_mean(X, window_size)"


def test_two_recipes_naming_the_same_observation_get_independent_steps():
    """The spliced step is a copy, or one recipe's later edit reaches every other that named it.

    Found by `test_rendering_does_not_modify_the_model`, which failed only when this module ran beside it: handing out the curated document's own step objects let a downstream mutation persist in the shared document for the rest of the process.
    """
    doc = """
name: Demo
pipeline:
    - iri: tvbo:observation/temporal_average_tvb
"""
    first = _expand(doc)["pipeline"][0]
    first["equation"]["rhs"] = "MUTATED"
    second = _expand(doc)["pipeline"][0]
    assert second["equation"]["rhs"] == "window_mean(X, window_size)"
