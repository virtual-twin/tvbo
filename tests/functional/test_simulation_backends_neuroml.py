"""Functional tests for NeuroML/LEMS backend.

Tests:
  1. Render every TVBO database model as LEMS (Dynamics.render_code('lems')) 2. Render every TVBO database model as full experiment (SimulationExperiment.render('lems')) 3. Render NeuroML canonical example experiments from YAML 4. Split-file export (dynamics / network / simulation) 5. PyLEMS validation (where possible)"""

from pathlib import Path

import pytest

from tvbo import database_path
from tvbo.classes.dynamics import Dynamics
from tvbo.classes.experiment import SimulationExperiment
from tests.functional.simulation_backends_shared import (
    MODEL_FILES,
    MODEL_IDS,
    _HAVE_LEMS,
    _HAVE_NEUROML,
    _HAVE_NEURON,
    _HAVE_BRIAN2,
    _HAVE_NETPYNE,
    _HAVE_EDEN,
)


# ── NeuroML experiment YAML files ─────────────────────────────────────

NEUROML_EXPERIMENTS_DIR = database_path / "experiments" / "neuroml"

NEUROML_EXPERIMENT_FILES = sorted(NEUROML_EXPERIMENTS_DIR.glob("*.yaml"))
NEUROML_EXPERIMENT_IDS = [f.stem for f in NEUROML_EXPERIMENT_FILES]


# ── Test classes ──────────────────────────────────────────────────────


@pytest.mark.backend_neuroml
class TestNeuroMLRender:
    """Test LEMS rendering for all TVBO database models."""

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_dynamics_render_lems(self, model_file):
        """Dynamics.render_code('lems') produces non-empty XML for every model."""
        model = Dynamics.from_file(model_file)
        xml = model.render_code("lems")
        assert xml is not None
        assert len(xml) > 100
        assert "<Lems>" in xml
        assert "<ComponentType" in xml

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_experiment_render_lems(self, model_file):
        """SimulationExperiment.render('lems') produces full simulation XML."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)
        xml = exp.render("lems")
        assert xml is not None
        assert len(xml) > 100
        assert "<Lems>" in xml
        assert "<Simulation" in xml

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_render_format_aliases(self, model_file):
        """All format aliases ('lems', 'neuroml', 'nml') produce the same XML."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)
        xml_lems = exp.render("lems")
        xml_neuroml = exp.render("neuroml")
        xml_nml = exp.render("nml")
        assert xml_lems == xml_neuroml == xml_nml


@pytest.mark.backend_neuroml
class TestNeuroMLExperiments:
    """Test LEMS rendering for NeuroML canonical example experiments."""

    @pytest.mark.parametrize(
        "exp_file",
        NEUROML_EXPERIMENT_FILES,
        ids=NEUROML_EXPERIMENT_IDS,
    )
    def test_experiment_from_yaml_renders(self, exp_file):
        """NeuroML experiment YAML files load and render as LEMS."""
        exp = SimulationExperiment.from_file(str(exp_file))
        assert exp.dynamics is not None
        xml = exp.render("lems")
        assert xml is not None
        assert "<Lems>" in xml
        # Standard NeuroML types use <Include file="Cells.xml"/> instead of custom <ComponentType> definitions.
        assert "<ComponentType" in xml or '<Include file="Cells.xml"/>' in xml
        assert "<Simulation" in xml

    @pytest.mark.parametrize(
        "exp_file",
        NEUROML_EXPERIMENT_FILES,
        ids=NEUROML_EXPERIMENT_IDS,
    )
    def test_experiment_has_correct_structure(self, exp_file):
        """Experiment YAML produces XML with expected structural elements."""
        exp = SimulationExperiment.from_file(str(exp_file))
        xml = exp.render("lems")
        dyn = exp.dynamics
        # Model name appears as ComponentType or Component
        assert dyn.name in xml
        # Every state variable appears in the XML
        for sv_name in dyn.state_variables or {}:
            assert sv_name in xml


@pytest.mark.backend_neuroml
class TestNeuroMLSplitExport:
    """Test split-file export (dynamics / network / simulation)."""

    def test_split_export_creates_three_files(self, tmp_path):
        """export(split=True) creates dynamics, network, simulation files."""
        from tvbo.adapters.neuroml import NeuroMLAdapter

        model = Dynamics.from_db("Generic2dOscillator")
        exp = SimulationExperiment(dynamics=model)
        adapter = NeuroMLAdapter(exp)
        paths = adapter.export(str(tmp_path), split=True, validate=False)
        assert "dynamics" in paths
        assert "network" in paths
        assert "simulation" in paths
        for role, fpath in paths.items():
            assert Path(fpath).exists(), f"{role} file missing: {fpath}"
            content = Path(fpath).read_text()
            assert len(content) > 50
            assert "<Lems>" in content

    def test_monolithic_export(self, tmp_path):
        """export(split=False) creates a single simulation file."""
        from tvbo.adapters.neuroml import NeuroMLAdapter

        model = Dynamics.from_db("Generic2dOscillator")
        exp = SimulationExperiment(dynamics=model)
        adapter = NeuroMLAdapter(exp)
        paths = adapter.export(str(tmp_path), split=False, validate=False)
        assert "simulation" in paths
        assert Path(paths["simulation"]).exists()

    def test_render_dynamics_standalone(self):
        """render_dynamics() produces valid standalone LEMS with ComponentType."""
        from tvbo.adapters.neuroml import NeuroMLAdapter

        model = Dynamics.from_db("Generic2dOscillator")
        adapter = NeuroMLAdapter(model)
        xml = adapter.render_dynamics()
        assert "<ComponentType" in xml
        assert "<Simulation" not in xml

    def test_render_network_includes_dynamics_file(self):
        """render_network(dynamics_file=...) adds Include element."""
        from tvbo.adapters.neuroml import NeuroMLAdapter

        model = Dynamics.from_db("Generic2dOscillator")
        exp = SimulationExperiment(dynamics=model)
        adapter = NeuroMLAdapter(exp)
        xml = adapter.render_network(dynamics_file="my_dynamics.xml")
        assert 'file="my_dynamics.xml"' in xml

    def test_render_simulation_includes_network_file(self):
        """render_simulation(network_file=...) adds Include element."""
        from tvbo.adapters.neuroml import NeuroMLAdapter

        model = Dynamics.from_db("Generic2dOscillator")
        exp = SimulationExperiment(dynamics=model)
        adapter = NeuroMLAdapter(exp)
        xml = adapter.render_simulation(network_file="my_network.xml")
        assert 'file="my_network.xml"' in xml


@pytest.mark.backend_neuroml
@pytest.mark.skipif(not _HAVE_LEMS, reason="lems (PyLEMS) not installed")
class TestNeuroMLValidation:
    """Test PyLEMS validation of rendered XML."""

    def test_validate_simple_model(self):
        """Simple 2-variable model validates with PyLEMS.

        Validation may fail for complex expressions because of known PyLEMS ExprParser limitations (v0.6.9), but FitzHughNagumo is within what the parser handles, so a failure here is a real regression.
        """
        from tvbo.adapters.neuroml import NeuroMLAdapter

        exp_file = NEUROML_EXPERIMENTS_DIR / "FitzHughNagumo_Ex9.yaml"
        if not exp_file.exists():
            pytest.skip("FitzHughNagumo_Ex9.yaml not found")
        exp = SimulationExperiment.from_file(str(exp_file))
        adapter = NeuroMLAdapter(exp)
        adapter.validate()

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_validate_all_models(self, model_file):
        """Attempt PyLEMS validation for all models, xfailing the known parser limitations.

        The PyLEMS ExprParser (v0.6.9) cannot handle some compound expressions or certain parameter names (`H`, for instance, clashes with Heaviside), so a failure is xfailed rather than raised; the models are still validated so that a new failure mode shows up as a change in which models xfail.
        """
        from tvbo.adapters.neuroml import NeuroMLAdapter

        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)
        adapter = NeuroMLAdapter(exp)
        xml = adapter.render_code()
        try:
            adapter.validate(xml)
        except Exception as e:
            pytest.xfail(f"PyLEMS validation failed (known limitation): {e}")


# ── Run tests (downstream simulator execution) ───────────────────────

# Use FitzHughNagumo — simple 2-variable model that validates and runs reliably across all backends.
_RUN_EXPERIMENT = str(NEUROML_EXPERIMENTS_DIR / "FitzHughNagumo_Ex9.yaml")


def _run_neuroml_backend(backend):
    """Helper: load FHN experiment, run via NeuroMLAdapter, return result."""
    from tvbo.adapters.neuroml import NeuroMLAdapter

    exp = SimulationExperiment.from_file(_RUN_EXPERIMENT)
    adapter = NeuroMLAdapter(exp)
    return adapter.run(backend=backend)


@pytest.mark.backend_neuroml
@pytest.mark.skipif(not _HAVE_NEUROML, reason="pyNeuroML not installed")
class TestNeuroMLRun:
    """Test LEMS simulation execution via pyNeuroML runners."""

    def test_run_jneuroml(self):
        """Run FitzHughNagumo via jNeuroML reference engine."""
        result = _run_neuroml_backend("jneuroml")
        assert result is not None
        assert result.integration is not None
        assert result.integration.data.sizes["time"] > 0
        assert "V" in result.integration.data.coords["variable"].values

    @pytest.mark.skipif(not _HAVE_NEURON, reason="NEURON not installed")
    def test_run_neuron(self):
        """Run FitzHughNagumo via NEURON (jNeuroML export)."""
        result = _run_neuroml_backend("neuron")
        assert result is not None
        assert result.integration is not None
        assert result.integration.data.sizes["time"] > 0

    @pytest.mark.skipif(not _HAVE_BRIAN2, reason="Brian2 not installed")
    def test_run_brian2(self):
        """Run FitzHughNagumo via Brian2 (jNeuroML export)."""
        result = _run_neuroml_backend("brian2")
        assert result is not None
        assert result.integration is not None
        assert result.integration.data.sizes["time"] > 0

    @pytest.mark.skipif(not _HAVE_NETPYNE, reason="NetPyNE not installed")
    def test_run_netpyne(self):
        """Run FitzHughNagumo via NetPyNE (jNeuroML export)."""
        result = _run_neuroml_backend("netpyne")
        assert result is not None
        assert result.integration is not None
        assert result.integration.data.sizes["time"] > 0

    @pytest.mark.skipif(not _HAVE_EDEN, reason="EDEN not installed")
    def test_run_eden(self):
        """Run FitzHughNagumo via EDEN simulator."""
        result = _run_neuroml_backend("eden")
        assert result is not None
        assert result.integration is not None
        assert result.integration.data.sizes["time"] > 0

    def test_run_invalid_backend(self):
        """ValueError for unsupported backend name."""
        from tvbo.adapters.neuroml import NeuroMLAdapter

        exp = SimulationExperiment.from_file(_RUN_EXPERIMENT)
        adapter = NeuroMLAdapter(exp)
        with pytest.raises(ValueError, match="Unknown NeuroML backend"):
            adapter.run(backend="nonexistent")

    def test_run_via_experiment(self):
        """SimulationExperiment.run('neuroml') routes to adapter.run()."""
        exp = SimulationExperiment.from_file(_RUN_EXPERIMENT)
        result = exp.run("neuroml")
        assert result is not None
        assert result.integration is not None

    def test_run_via_experiment_with_backend(self):
        """SimulationExperiment.run('neuroml', backend='jneuroml') works."""
        exp = SimulationExperiment.from_file(_RUN_EXPERIMENT)
        result = exp.run("neuroml", backend="jneuroml")
        assert result is not None
        assert result.integration is not None
