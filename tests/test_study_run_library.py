"""A study run takes one library path whether a shell or a notebook starts it, and records every container it writes.

`tvbo run` and `SimulationStudy.run()` both call :mod:`tvbo.run.study`. These pin that the run record sits in the sidecar of every container a run writes — from Python as from the shell, and for each study a study-of-studies holds — that both entry points record the same runs, and that the library reports a refusal as an exception rather than exiting the process.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tvbo
from tests.test_provenance_records import STUDY
from tvbo.data import provenance
from tvbo.run import study as study_run
from tvbo.utils.study_layout import study_path


def _write_demo(root: Path) -> Path:
    """The provenance smoke study (one two-node Kuramoto experiment, id 3) as ``<root>/Demo.yaml``."""
    root.mkdir(parents=True, exist_ok=True)
    spec = root / "Demo.yaml"
    spec.write_text(STUDY.replace("# WORKFLOW\n", ""), encoding="utf-8")
    return spec


def _write_tree(root: Path) -> Path:
    """A study-of-studies holding the smoke study and one experiment of its own (id 5), with one authored result."""
    _write_demo(root / "demo")
    own = STUDY.split("experiments:\n", 1)[1].replace("  - id: 3\n", "  - id: 5\n")
    spec = root / "collection.yaml"
    spec.write_text(
        "tvbo_class: tvbo:SimulationStudy\ncitekey: Tree\ntitle: A study of studies\n"
        "studies:\n  - !include demo/Demo.yaml\n"
        f"experiments:\n{own}"
        "results:\n  - {key: parcels, value: '2', source: Demo}\n",
        encoding="utf-8",
    )
    return spec


def _recorded(results: Path) -> dict[str, list]:
    """``{container name: provenance activities}`` for every container under *results*."""
    return {
        h5.name: (yaml.safe_load(h5.with_suffix(".yaml").read_text()).get("provenance") or {}).get("activities") or []
        for h5 in sorted(results.glob("*_result.h5"))
    }


@pytest.fixture(autouse=True)
def _needs_tvboptim():
    pytest.importorskip("tvboptim")


def test_a_python_run_records_every_container_it_writes(tmp_path):
    spec = _write_demo(tmp_path / "demo")
    tvbo.SimulationStudy.from_file(str(spec)).run()
    recorded = _recorded(study_path("results", root=spec.parent))
    assert recorded, "the run wrote no container"
    assert all(recorded.values()), f"containers written without a run record: {[n for n, a in recorded.items() if not a]}"
    assert "tvbo:exp/Demo/exp-3" in {a["iri"] for acts in recorded.values() for a in acts}


def test_each_study_of_a_tree_records_its_containers(tmp_path):
    spec = _write_tree(tmp_path / "tree")
    result = tvbo.SimulationStudy.from_file(str(spec)).run()
    nested = _recorded(study_path("results", root=spec.parent / "demo"))
    holder = _recorded(study_path("results", root=spec.parent))
    assert nested and all(nested.values()), f"the nested study's containers carry no run record: {nested}"
    assert holder and all(holder.values()), f"the holder's own containers carry no run record: {holder}"
    assert {a["iri"] for acts in holder.values() for a in acts} == {"tvbo:exp/Tree/exp-5"}
    assert set(result.studies) == {"Demo"}


def test_a_redirected_python_run_records_where_it_wrote(tmp_path):
    spec = _write_demo(tmp_path / "demo")
    build = tmp_path / "_build"
    tvbo.SimulationStudy.from_file(str(spec)).run(root=build)
    recorded = _recorded(study_path("results", root=build))
    assert recorded and all(recorded.values())
    assert not study_path("results", root=spec.parent).exists(), "a redirected run wrote into the recipe's tree"


def test_the_shell_and_python_record_the_same_runs(tmp_path):
    """One path, so one record: the same study run from each entry point yields the same activities over the same outputs."""
    from typer.testing import CliRunner

    from tvbo.cli import app

    shell = _write_demo(tmp_path / "shell")
    res = CliRunner().invoke(app, ["run", str(shell)])
    assert res.exit_code == 0, res.output
    python = _write_demo(tmp_path / "python")
    tvbo.SimulationStudy.from_file(str(python)).run()

    def _runs(root: Path) -> dict:
        return {label: sorted(record["ent"]["outputs"]) for label, record in provenance.read_records(root).items()}

    assert _runs(shell.parent) == _runs(python.parent) != {}


def test_a_refusal_is_an_exception_not_an_exit(tmp_path):
    """The library never exits the process; `tvbo run` turns this exception into its error line and exit code."""
    spec = _write_demo(tmp_path / "demo")
    study = tvbo.SimulationStudy.from_file(str(spec))
    with pytest.raises(study_run.StudyRunError, match="No experiment"):
        study_run.run_study(study, str(spec), experiment="99")
