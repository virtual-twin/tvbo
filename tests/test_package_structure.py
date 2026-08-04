"""Package structure validation tests.

Fast (<5s) tests that verify import paths, public API surface, backward
compatibility shims, and class identity invariants. Run during restructuring
to catch broken imports immediately:

    pytest tests/test_package_structure.py -x -n0 -v

These tests document the CURRENT (pre-v1.0) import contract. During migration,
update expected paths here FIRST, then fix the code to match.
"""

import importlib

import pytest


# ── Section 1: Top-level public API ──────────────────────────────────────────


class TestTopLevelAPI:
    """Everything users should be able to `from tvbo import X`."""

    TOP_LEVEL_NAMES = [
        "Dynamics",
        "SimulationExperiment",
        "SimulationStudy",
        "Network",
        "Connectome",  # deprecated alias, should == Network
        "Atlas",
        "Coupling",
        "Noise",
        "Continuation",
        "Function",
        "LossFunction",
    ]

    @pytest.mark.parametrize("name", TOP_LEVEL_NAMES)
    def test_top_level_import(self, name):
        """Core classes are importable from `tvbo` directly."""
        import tvbo

        obj = getattr(tvbo, name, None)
        assert obj is not None, f"`from tvbo import {name}` should work"

    def test_version_exists(self):
        import tvbo

        assert hasattr(tvbo, "__version__")
        assert isinstance(tvbo.__version__, str)
        assert len(tvbo.__version__) >= 5  # e.g. "0.2.6"


# ── Section 2: Deep import paths (current, pre-restructuring) ───────────────


class TestCurrentImportPaths:
    """Verify all current deep import paths still resolve.

    These are the imports used throughout tvbo code, docs, and tests.
    During restructuring, backward-compat shims must keep these working.
    """

    IMPORT_SPECS = [
        # (module_path, attribute_name)
        ("tvbo.classes.dynamics", "Dynamics"),
        ("tvbo.classes.coupling", "Coupling"),
        ("tvbo.classes.noise", "Noise"),
        ("tvbo.classes.continuation", "Continuation"),
        ("tvbo.classes.observation", "Function"),
        ("tvbo.classes.observation", "ObservationModel"),
        ("tvbo.classes.perturbation", "Stimulus"),
        ("tvbo.classes.equation", None),  # module import
        ("tvbo.classes.function", "Function"),
        ("tvbo.classes.function", "LossFunction"),
        ("tvbo.classes.study", "SimulationStudy"),
        ("tvbo.ontology.owl", "onto"),
        ("tvbo.classes.experiment", "SimulationExperiment"),
        ("tvbo.codegen.code", "parse_eq"),
        ("tvbo.codegen.templater", None),  # module import
        ("tvbo.classes.network", "Network"),
        ("tvbo.classes.network", "Connectome"),
        ("tvbo.classes.atlas", "Atlas"),
        ("tvbo.data.types", "TimeSeries"),
        ("tvbo.datamodel", "tvbo_datamodel"),
        ("tvbo.datamodel.schema", "Dynamics"),
        ("tvbo.datamodel.schema", "Parameter"),
        ("tvbo.datamodel.schema", "Network"),
        ("tvbo.datamodel.schema", "Equation"),
        ("tvbo.datamodel.pydantic", "Dynamics"),
    ]

    @pytest.mark.parametrize("module_path,attr", IMPORT_SPECS, ids=[f"{m}.{a}" if a else m for m, a in IMPORT_SPECS])
    def test_deep_import(self, module_path, attr):
        mod = importlib.import_module(module_path)
        if attr is not None:
            assert hasattr(mod, attr), f"{module_path} should have attribute '{attr}'"


# ── Section 3: Class identity invariants ─────────────────────────────────────


class TestClassIdentity:
    """Ensure aliased classes point to the same object."""

    def test_connectome_is_subclass_of_network(self):
        """Connectome should be a deprecated alias/subclass of Network."""
        from tvbo.classes.network import Connectome, Network

        assert issubclass(Connectome, Network)

    def test_top_level_dynamics_is_same_class(self):
        """tvbo.Dynamics is the same class as localdynamics.Dynamics."""
        import tvbo
        from tvbo.classes.dynamics import Dynamics

        assert tvbo.Dynamics is Dynamics

    def test_top_level_network_is_same_class(self):
        """tvbo.Network is the same class as classes.network.Network."""
        import tvbo
        from tvbo.classes.network import Network

        assert tvbo.Network is Network

    def test_top_level_coupling_is_same_class(self):
        """tvbo.Coupling is the same class from network.py."""
        import tvbo
        from tvbo.classes.coupling import Coupling

        assert tvbo.Coupling is Coupling

    def test_top_level_experiment_is_same_class(self):
        """tvbo.SimulationExperiment is the same class from export."""
        import tvbo
        from tvbo.classes.experiment import SimulationExperiment

        assert tvbo.SimulationExperiment is SimulationExperiment

    def test_dynamics_inherits_from_datamodel(self):
        """Runtime Dynamics should subclass the datamodel Dynamics."""
        from tvbo.classes.dynamics import Dynamics
        from tvbo.datamodel.schema import Dynamics as DmDynamics

        assert issubclass(Dynamics, DmDynamics)

    def test_network_inherits_from_datamodel(self):
        """Runtime Network should subclass the datamodel Network."""
        from tvbo.classes.network import Network
        from tvbo.datamodel.schema import Network as DmNetwork

        assert issubclass(Network, DmNetwork)


# ── Section 4: Datamodel sanity ──────────────────────────────────────────────


class TestDatamodel:
    """Verify datamodel module structure and key classes."""

    EXPECTED_SCHEMA_CLASSES = [
        "Dynamics",
        "Coupling",
        "Network",
        "Parameter",
        "Equation",
        "StateVariable",
        "Node",
        "Edge",
        "Integrator",
        "Noise",
        "Parcellation",
        "BrainAtlas",
        "Matrix",
    ]

    @pytest.mark.parametrize("cls_name", EXPECTED_SCHEMA_CLASSES)
    def test_schema_class_exists(self, cls_name):
        from tvbo.datamodel import tvbo_datamodel as dm

        assert hasattr(dm, cls_name), f"tvbo_datamodel should have class '{cls_name}'"

    def test_namespace_size_bounded(self):
        """Datamodel namespace shouldn't leak unbounded internals."""
        from tvbo.datamodel import tvbo_datamodel as dm

        public = [n for n in dir(dm) if not n.startswith("_")]
        # 301 after Reducer / Assignment / ReducerEmitKind gave the reducers database
        # directory a class. The bound catches *unbounded* growth — a generator leaking
        # internals — so a handful of reviewed classes raises it rather than trips it.
        assert len(public) < 330, f"tvbo_datamodel has {len(public)} public names — check for namespace pollution"

    def test_pydantic_module_importable(self):
        from tvbo.datamodel import tvbopydantic

        assert hasattr(tvbopydantic, "Dynamics")


# ── Section 5: Subpackage modules exist ──────────────────────────────────────


class TestSubpackages:
    """Verify key subpackages are importable."""

    SUBPACKAGES = [
        "tvbo.classes",
        "tvbo.codegen",
        "tvbo.ontology",
        "tvbo.data",
        "tvbo.data.tvbo_data",
        "tvbo.datamodel",
        "tvbo.adapters",
        "tvbo.templates",
        "tvbo.parse",
        "tvbo.plot",
        "tvbo.analysis",
        "tvbo.run",
        "tvbo.api",
    ]

    @pytest.mark.parametrize("pkg", SUBPACKAGES)
    def test_subpackage_importable(self, pkg):
        importlib.import_module(pkg)


# ── Section 6: Future paths (post-restructuring) ────────────────────────────
# Uncomment these after each migration step to verify new paths work.

# class TestNewImportPaths:
#     """Verify new import paths after restructuring."""
#
#     NEW_IMPORT_SPECS = [
#         ("tvbo.models", "Dynamics"),
#         ("tvbo.models", "Coupling"),
#         ("tvbo.classes.dynamics", "Dynamics"),
#         ("tvbo.classes.coupling", "Coupling"),
#         ("tvbo.classes.noise", "Noise"),
#         ("tvbo.classes.continuation", "Continuation"),
#         ("tvbo.classes.observation", "Function"),
#         ("tvbo.classes.function", "Function"),
#         ("tvbo.classes.function", "LossFunction"),
#         ("tvbo.experiment", "SimulationExperiment"),
#         ("tvbo.experiment", "SimulationStudy"),
#         ("tvbo.ontology", None),
#         ("tvbo.ontology.owl", "onto"),
#         ("tvbo.codegen", None),
#         ("tvbo.codegen.code", "parse_eq"),
#         ("tvbo.codegen.templater", None),
#         ("tvbo.data.network", "Network"),
#         ("tvbo.data.atlas", "Atlas"),
#         ("tvbo.bids", None),
#         ("tvbo.bids.models", None),
#         ("tvbo.bids.paths", None),
#         ("tvbo.bids.export", None),
#         ("tvbo.bids.ingest", None),
#         ("tvbo.datamodel.schema", "Dynamics"),
#         ("tvbo.datamodel.pydantic", "Dynamics"),
#     ]
#
#     @pytest.mark.parametrize("module_path,attr", NEW_IMPORT_SPECS,
#                              ids=[f"{m}.{a}" if a else m for m, a in NEW_IMPORT_SPECS])
#     def test_new_import(self, module_path, attr):
#         mod = importlib.import_module(module_path)
#         if attr is not None:
#             assert hasattr(mod, attr), f"{module_path} should have '{attr}'"
