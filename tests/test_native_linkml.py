"""The schema and every published record are valid LinkML as LinkML's own tools read them, with no TVBO code in the loop.

``test_database_validation`` checks the records against the JSON Schema TVBO ships, which is deliberately relaxed (``additionalProperties`` opened) because TVBO's dialect folds aliases and envelopes before construction. A third party has none of that: they run ``linkml-lint`` and ``linkml-validate``. This module is that third party. It uses the metamodel linter and LinkML's closed JSON Schema validator — the ``linkml-validate`` default — against the schema source, both in their own processes (``native_linkml_worker`` says why), so the result does not depend on what else ran in this interpreter.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .database_corpus import REPO, collect

SCHEMA = REPO / "schema" / "tvbo_datamodel.yaml"
WORKER = Path(__file__).with_name("native_linkml_worker.py")

CASES = collect()
IDS = [str(p.relative_to(REPO)) for p, _ in CASES]

PROBES = {
    "experiment-integer": ({"experiment": 3, "output": "x"}, "DataRef"),
    "experiment-name": ({"experiment": "exp_a", "output": "x"}, "DataRef"),
    "experiment-list": ({"experiment": [1], "output": "x"}, "DataRef"),
    "n_parallel-width": ({"name": "sweep", "n_parallel": 8}, "Exploration"),
    "n_parallel-auto": ({"name": "sweep", "n_parallel": "auto"}, "Exploration"),
    "envelope-own-class": ({"tvbo_class": "tvbo:Network", "schema_version": "tvb-datamodel/0.7.0", "label": "n"}, "Network"),
    "envelope-other-class": ({"tvbo_class": "tvbo:Theme", "label": "n"}, "Network"),
}


@pytest.fixture(scope="module")
def native():
    """Every record and probe validated by LinkML's closed validator, in one fresh interpreter: ``{case id: [message, ...]}``."""
    if importlib.util.find_spec("linkml") is None:
        pytest.skip("linkml is not installed")
    cases = [[str(path), {"path": str(path)}, target] for path, target in CASES]
    cases += [[f"probe:{name}", {"instance": instance}, target] for name, (instance, target) in PROBES.items()]
    run = subprocess.run(
        [sys.executable, str(WORKER)],
        input=json.dumps({"schema": str(SCHEMA), "cases": cases}),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert run.returncode == 0, f"the LinkML validator failed to run:\n{run.stderr[-4000:]}"
    return json.loads(run.stdout)


def test_the_schema_lints_without_errors():
    """``linkml-lint`` reports no error on the schema: the metamodel accepts every class, slot and default as written."""
    lint = shutil.which("linkml-lint") or str(Path(sys.executable).with_name("linkml-lint"))
    if not Path(lint).exists():
        pytest.skip("linkml-lint is not installed")
    run = subprocess.run([lint, "--format", "json", str(SCHEMA)], capture_output=True, text=True, check=False)
    problems = json.loads(run.stdout or "[]")
    errors = [f"{p.get('rule')}: {p.get('message')}" for p in problems if p.get("level") == "error"]
    assert not errors, "linkml-lint errors:\n  - " + "\n  - ".join(errors)


@pytest.mark.parametrize(("path", "target_class"), CASES, ids=IDS)
def test_a_published_record_validates_natively(native, path, target_class):
    messages = native[str(path)]
    assert not messages, f"{path.relative_to(REPO)} is not valid LinkML as {target_class}:\n  - " + "\n  - ".join(messages)


@pytest.mark.parametrize("probe", ["experiment-integer", "experiment-name"])
def test_an_experiment_handle_is_an_integer_or_a_name(native, probe):
    assert not native[f"probe:{probe}"]


def test_an_experiment_handle_is_never_a_list(native):
    assert native["probe:experiment-list"]


@pytest.mark.parametrize("probe", ["n_parallel-width", "n_parallel-auto"])
def test_n_parallel_accepts_a_width_or_auto(native, probe):
    assert not native[f"probe:{probe}"]


def test_the_envelope_is_declared(native):
    assert not native["probe:envelope-own-class"]


def test_an_envelope_naming_another_class_is_refused(native):
    assert native["probe:envelope-other-class"]
