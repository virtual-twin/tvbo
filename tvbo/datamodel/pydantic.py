from __future__ import annotations

import re
import sys
from datetime import (
    date,
    datetime,
    time
)
from decimal import Decimal
from enum import Enum
from typing import (
    TypeVar,
    Union
)
from typing import (
    TypeVar,
    Union
)
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    TypeVar,
    Union
)
from typing import (
    TypeVar,
    Union
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer
)

if sys.version_info.minor >= 12:
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

if sys.version_info.minor >= 12:
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

if sys.version_info.minor >= 12:
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

if sys.version_info.minor >= 12:
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

if sys.version_info.minor >= 12:
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

if sys.version_info.minor >= 12:
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

if sys.version_info.minor >= 12:
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

if sys.version_info.minor >= 12:
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType



metamodel_version = "1.11.0"
version = "0.4.0"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        serialize_by_alias = True,
        validate_by_name = True,
        validate_assignment = True,
        validate_default = True,
        extra = "forbid",
        arbitrary_types_allowed = True,
        use_enum_values = True,
        strict = False,
    )





class LinkMLMeta(RootModel):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key:str):
        return getattr(self.root, key)

    def __getitem__(self, key:str):
        return self.root[key]

    def __setitem__(self, key:str, value):
        self.root[key] = value

    def __contains__(self, key:str) -> bool:
        return key in self.root



_T = TypeVar("_T")

AnyShapeArray = TypeAliasType(
    "AnyShapeArray", list[Union[_T, "AnyShapeArray[_T]"]], type_params=(_T,)
)
linkml_meta = LinkMLMeta({'created_on': '2024-01-01T00:00:00',
     'default_prefix': 'tvbo',
     'default_range': 'string',
     'description': 'Metadata schema for simulation studies using The Virtual '
                    'Brain neuroinformatics platform or other dynamic network '
                    'models of large-scale brain activity.',
     'id': 'https://w3id.org/tvbo',
     'imports': ['linkml:types',
                 'common',
                 'units',
                 'SANDS',
                 'tvbo_study',
                 'tvb_dbs',
                 'software'],
     'license': 'https://spdx.org/licenses/EUPL-1.2',
     'name': 'tvb-datamodel',
     'prefixes': {'CHEBI': {'prefix_prefix': 'CHEBI',
                            'prefix_reference': 'http://purl.obolibrary.org/obo/CHEBI_'},
                  'CL': {'prefix_prefix': 'CL',
                         'prefix_reference': 'http://purl.obolibrary.org/obo/CL_'},
                  'GO': {'prefix_prefix': 'GO',
                         'prefix_reference': 'http://purl.obolibrary.org/obo/GO_'},
                  'MESH': {'prefix_prefix': 'MESH',
                           'prefix_reference': 'http://purl.bioontology.org/ontology/MESH/'},
                  'UBERON': {'prefix_prefix': 'UBERON',
                             'prefix_reference': 'http://purl.obolibrary.org/obo/UBERON_'},
                  'UO': {'prefix_prefix': 'UO',
                         'prefix_reference': 'http://purl.obolibrary.org/obo/UO_'},
                  'bibo': {'prefix_prefix': 'bibo',
                           'prefix_reference': 'http://purl.org/ontology/bibo/'},
                  'biotools': {'prefix_prefix': 'biotools',
                               'prefix_reference': 'https://bio.tools/ontology/'},
                  'dcterms': {'prefix_prefix': 'dcterms',
                              'prefix_reference': 'http://purl.org/dc/terms/'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'nidm': {'prefix_prefix': 'nidm',
                           'prefix_reference': 'http://purl.org/nidash/nidm#'},
                  'oboInOwl': {'prefix_prefix': 'oboInOwl',
                               'prefix_reference': 'http://www.geneontology.org/formats/oboInOwl#'},
                  'prov': {'prefix_prefix': 'prov',
                           'prefix_reference': 'http://www.w3.org/ns/prov#'},
                  'qudt': {'prefix_prefix': 'qudt',
                           'prefix_reference': 'http://qudt.org/vocab/unit/'},
                  'rdfs': {'prefix_prefix': 'rdfs',
                           'prefix_reference': 'http://www.w3.org/2000/01/rdf-schema#'},
                  'schema': {'prefix_prefix': 'schema',
                             'prefix_reference': 'http://schema.org/'},
                  'sio': {'prefix_prefix': 'sio',
                          'prefix_reference': 'http://semanticscience.org/resource/'},
                  'skos': {'prefix_prefix': 'skos',
                           'prefix_reference': 'http://www.w3.org/2004/02/skos/core#'},
                  'tvbo': {'prefix_prefix': 'tvbo',
                           'prefix_reference': 'https://w3id.org/tvbo/'},
                  'wd': {'prefix_prefix': 'wd',
                         'prefix_reference': 'http://www.wikidata.org/entity/'}},
     'source_file': 'schema/tvbo_datamodel.yaml',
     'title': 'The Virtual Brain Data Model'} )

class UnitEnum(str, Enum):
    """
    Physical units of measurement for model parameters, state variables, and integration settings. Uses conventional abbreviations as values, mapped to the QUDT ontology (http://qudt.org/vocab/unit/) with UO cross-references where available.
    """
    s = "s"
    """
    Second
    """
    ms = "ms"
    """
    Millisecond
    """
    us = "us"
    """
    Microsecond
    """
    per_s = "per_s"
    """
    Per second (s⁻¹)
    """
    per_ms = "per_ms"
    """
    Per millisecond (ms⁻¹)
    """
    Hz = "Hz"
    """
    Hertz (s⁻¹)
    """
    kHz = "kHz"
    """
    Kilohertz
    """
    V = "V"
    """
    Volt
    """
    mV = "mV"
    """
    Millivolt
    """
    per_mV = "per_mV"
    """
    Reciprocal millivolt (mV⁻¹)
    """
    mV_per_ms = "mV_per_ms"
    """
    Millivolt per millisecond
    """
    mV_per_s = "mV_per_s"
    """
    Millivolt per second
    """
    A = "A"
    """
    Ampere
    """
    nA = "nA"
    """
    Nanoampere
    """
    pA = "pA"
    """
    Picoampere
    """
    uA_per_cm2 = "uA_per_cm2"
    """
    Microampere per square centimetre (current density)
    """
    pF = "pF"
    """
    Picofarad
    """
    nF = "nF"
    """
    Nanofarad
    """
    uF_per_cm2 = "uF_per_cm2"
    """
    Microfarad per square centimetre (specific capacitance)
    """
    nS = "nS"
    """
    Nanosiemens
    """
    uS = "uS"
    """
    Microsiemens
    """
    pS = "pS"
    """
    Picosiemens
    """
    S_per_cm2 = "S_per_cm2"
    """
    Siemens per square centimetre (conductance density)
    """
    mS_per_cm2 = "mS_per_cm2"
    """
    Millisiemens per square centimetre (conductance density)
    """
    S_per_m2 = "S_per_m2"
    """
    Siemens per square metre (conductance density, SI)
    """
    nS_per_mV = "nS_per_mV"
    """
    Nanosiemens per millivolt
    """
    per_nC = "per_nC"
    """
    Reciprocal nanocoulomb (nC⁻¹)
    """
    per_pC = "per_pC"
    """
    Reciprocal picocoulomb (pC⁻¹)
    """
    mol_per_m3 = "mol_per_m3"
    """
    Mole per cubic metre (mol/m³)
    """
    mol_per_cm3 = "mol_per_cm3"
    """
    Mole per cubic centimetre (mol/cm³)
    """
    mmol_per_m3 = "mmol_per_m3"
    """
    Millimole per cubic metre (mmol/m³ ≈ mM)
    """
    mol_per_m_per_A_per_s = "mol_per_m_per_A_per_s"
    """
    Mole per metre per ampere per second (concentration-current coupling)
    """
    um3 = "um3"
    """
    Cubic micrometre (µm³)
    """
    m = "m"
    """
    Metre
    """
    mm = "mm"
    """
    Millimetre
    """
    cm = "cm"
    """
    Centimetre
    """
    m_per_s = "m_per_s"
    """
    Metre per second
    """
    mm_per_ms = "mm_per_ms"
    """
    Millimetre per millisecond (= m/s)
    """
    Hz_per_nA = "Hz_per_nA"
    """
    Hertz per nanoampere (neural gain)
    """
    S_per_m = "S_per_m"
    """
    Siemens per metre (conductivity)
    """
    H_per_m = "H_per_m"
    """
    Henry per metre (permeability)
    """
    ohm = "ohm"
    """
    Ohm (Ω)
    """
    Mohm = "Mohm"
    """
    Megaohm (MΩ)
    """
    kohm_cm = "kohm_cm"
    """
    Kilo-ohm centimetre (axial resistivity)
    """
    degC = "degC"
    """
    Degree Celsius
    """
    rad_per_ms = "rad_per_ms"
    """
    Radian per millisecond
    """
    dimensionless = "dimensionless"
    """
    Dimensionless (unitless)
    """
    percent = "percent"
    """
    Percent (%)
    """
    arbitrary_unit = "arbitrary_unit"
    """
    Arbitrary units (a.u.)
    """
    kg = "kg"
    """
    Kilogram
    """
    kg_per_s = "kg_per_s"
    """
    Kilogram per second
    """
    m_per_s2 = "m_per_s2"
    """
    Metre per second squared (acceleration)
    """
    N_per_m = "N_per_m"
    """
    Newton per metre (spring constant)
    """
    rad = "rad"
    """
    Radian
    """
    rad_per_s = "rad_per_s"
    """
    Radian per second (angular velocity)
    """
    s2 = "s2"
    """
    Second squared (inertia constant)
    """
    per_unit = "per_unit"
    """
    Per-unit (dimensionless power-systems convention)
    """


class PhysicalDimension(str, Enum):
    """
    Physical dimension categories for LEMS and dimensional analysis. Each dimension decomposes into SI base dimensions (M, L, T, I, K, N).
    """
    none = "none"
    """
    Dimensionless
    """
    time = "time"
    """
    Time [T]
    """
    per_time = "per_time"
    """
    Inverse time [T⁻¹]
    """
    voltage = "voltage"
    """
    Voltage [M L² T⁻³ I⁻¹]
    """
    current = "current"
    """
    Electric current [I]
    """
    capacitance = "capacitance"
    """
    Capacitance [M⁻¹ L⁻² T⁴ I²]
    """
    conductance = "conductance"
    """
    Conductance [M⁻¹ L⁻² T³ I²]
    """
    resistance = "resistance"
    """
    Resistance [M L² T⁻³ I⁻²]
    """
    charge = "charge"
    """
    Electric charge [T I]
    """
    concentration = "concentration"
    """
    Concentration [L⁻³ N]
    """
    substance = "substance"
    """
    Amount of substance [N]
    """
    length = "length"
    """
    Length [L]
    """
    volume = "volume"
    """
    Volume [L³]
    """
    temperature = "temperature"
    """
    Temperature [K]
    """


class SpecimenEnum(str, Enum):
    """
    A set of permissible types for specimens used in brain atlas creation.
    """
    Subject = "Subject"
    SubjectGroup = "SubjectGroup"
    TissueSample = "TissueSample"
    TissueSampleCollection = "TissueSampleCollection"


class Hemisphere(str, Enum):
    left = "left"
    right = "right"
    both = "both"


class SexEnum(str, Enum):
    male = "male"
    """
    Male
    """
    female = "female"
    """
    Female
    """
    other = "other"
    """
    Other or not reported
    """


class SimulationScale(str, Enum):
    """
    Spatial / organizational scale at which a tool operates. Multi-valued: a tool can span multiple scales. Mapped to SIO and Wikidata where possible.
    """
    channel = "channel"
    """
    Ion channel / sub-cellular molecular dynamics.
    """
    neuron = "neuron"
    """
    Single neuron (compartmental or point).
    """
    neural_network = "neural_network"
    """
    Microcircuit / local network of neurons.
    """
    neural_mass = "neural_mass"
    """
    Population-level neural mass or mean-field model.
    """
    network_system = "network_system"
    """
    Whole-brain or large-scale network of regions.
    """
    whole_brain = "whole_brain"
    """
    Whole-brain models targeting cortex-wide dynamics.
    """


class ToolRole(str, Enum):
    """
    Primary function of the tool in a simulation workflow.
    """
    simulator = "simulator"
    """
    Core numerical simulator.
    """
    framework = "framework"
    """
    Multi-paradigm simulation framework.
    """
    backend_runtime = "backend_runtime"
    """
    Optimized execution backend for another simulator.
    """
    optimization_framework = "optimization_framework"
    """
    Parameter optimization / fitting tool.
    """
    specification_language = "specification_language"
    """
    Model description language or data standard.
    """
    workflow_framework = "workflow_framework"
    """
    Orchestration, model-building, or pipeline tool.
    """
    analysis_tool = "analysis_tool"
    """
    Post-processing, signal analysis, or statistics.
    """
    visualization_tool = "visualization_tool"
    """
    Visualization or graphical user interface.
    """
    model_repository = "model_repository"
    """
    Database or repository of published models.
    """
    continuation_tool = "continuation_tool"
    """
    Numerical continuation / bifurcation analysis.
    """
    inference_framework = "inference_framework"
    """
    Probabilistic / Bayesian inference toolkit for model parameters.
    """
    feature_extraction = "feature_extraction"
    """
    Feature library or pipeline for derived signal descriptors.
    """


class ModelParadigm(str, Enum):
    """
    Computational paradigm or modeling approach supported by the tool.
    """
    neural_mass = "neural_mass"
    """
    Phenomenological population-rate / neural-mass models.
    """
    mean_field = "mean_field"
    """
    Mean-field reductions of spiking networks.
    """
    spiking = "spiking"
    """
    Spiking neuron models (LIF, AdEx, Izhikevich, etc.).
    """
    conductance_based = "conductance_based"
    """
    Conductance-based / Hodgkin-Huxley-type models.
    """
    compartmental = "compartmental"
    """
    Multi-compartment morphologically detailed models.
    """
    rate_based = "rate_based"
    """
    Firing-rate models.
    """
    phase_oscillator = "phase_oscillator"
    """
    Phase-reduced or Kuramoto-type oscillator models.
    """
    reaction_diffusion = "reaction_diffusion"
    """
    Stochastic or deterministic reaction-diffusion.
    """
    plasticity = "plasticity"
    """
    Synaptic plasticity (STDP, homeostatic, etc.).
    """
    generic = "generic"
    """
    General-purpose, not specific to neuroscience.
    """
    multiscale = "multiscale"
    """
    Bridging multiple spatial/temporal scales.
    """
    dynamic_mean_field = "dynamic_mean_field"
    """
    Dynamic mean-field approximation (e.g., Deco et al.).
    """
    neural_field = "neural_field"
    """
    Continuous neural field equations (Amari, Wilson-Cowan field).
    """
    data_standard = "data_standard"
    """
    Data format or exchange standard.
    """
    model_description = "model_description"
    """
    Declarative model specification language.
    """
    bifurcation_analysis = "bifurcation_analysis"
    """
    Dynamical systems bifurcation / continuation analysis.
    """


class DevelopmentStatus(str, Enum):
    """
    Development status of the software. Based on repostatus.org categories.
    """
    active = "active"
    """
    Actively developed with regular releases.
    """
    inactive = "inactive"
    """
    No longer actively developed; may still work.
    """
    concept = "concept"
    """
    Minimal or no implementation; ideas / prototypes.
    """
    wip = "wip"
    """
    Work in progress; not yet feature-complete.
    """
    suspended = "suspended"
    """
    Development paused; may resume in future.
    """
    unsupported = "unsupported"
    """
    Released but no longer supported.
    """
    moved = "moved"
    """
    Project has been moved to a different location.
    """


class EcosystemEnum(str, Enum):
    """
    Package ecosystem or registry the software is distributed through.
    """
    pypi = "pypi"
    """
    Python Package Index.
    """
    conda_forge = "conda_forge"
    """
    Conda-Forge community channel.
    """
    cran = "cran"
    """
    Comprehensive R Archive Network.
    """
    julia_registry = "julia_registry"
    """
    Julia General package registry.
    """
    npm = "npm"
    """
    Node Package Manager registry.
    """
    bioconda = "bioconda"
    """
    Bioinformatics Conda channel.
    """
    github = "github"
    """
    Distributed via GitHub releases.
    """
    maven = "maven"
    """
    Maven Central Repository (Java).
    """
    docker = "docker"
    """
    Docker container registry / image distribution.
    """


class ProgrammingLanguageEnum(str, Enum):
    """
    Programming languages relevant to computational neuroscience tools. Mapped to Wikidata identifiers.
    """
    Python = "Python"
    C = "C"
    CPLUS_SIGNPLUS_SIGN = "C++"
    Java = "Java"
    Julia = "Julia"
    MATLAB = "MATLAB"
    R = "R"
    Fortran = "Fortran"
    Haskell = "Haskell"
    Rust = "Rust"
    JavaScript = "JavaScript"
    XML = "XML"
    HOC = "HOC"
    """
    NEURON's high-level interpreted language.
    """
    CNUMBER_SIGN = "C#"


class RequirementRole(str, Enum):
    engine = "engine"
    """
    Primary simulation/processing engine.
    """
    runtime = "runtime"
    """
    General runtime dependency.
    """
    analysis = "analysis"
    """
    Post-processing / analysis tool.
    """
    dev = "dev"
    """
    Development / build dependency.
    """
    optional = "optional"
    """
    Optional or extra feature dependency.
    """


class EnvironmentType(str, Enum):
    conda = "conda"
    """
    Conda environment.
    """
    venv = "venv"
    """
    Python virtual environment.
    """
    docker = "docker"
    """
    Docker container.
    """
    singularity = "singularity"
    """
    Singularity/Apptainer container.
    """


class ParallelMode(str, Enum):
    """
    How a trial / grid-point axis is realised at JAX codegen time. The choice trades peak memory against throughput: vmap batches in parallel (fast, n_trials × working-set memory), lax_map runs sequentially via ``jax.lax.map`` (memory bounded by one trial), pmap shards across devices, auto picks vmap when the estimated batched memory fits and lax_map otherwise.
    """
    auto = "auto"
    """
    Pick vmap when memory permits, lax_map otherwise.
    """
    vmap = "vmap"
    """
    Parallel batched execution (jax.vmap). Fast; high peak memory.
    """
    lax_map = "lax_map"
    """
    Sequential execution via jax.lax.map. Slower; bounded memory.
    """
    pmap = "pmap"
    """
    Cross-device parallel execution (jax.pmap). For multi-GPU/TPU.
    """


class ImagingModality(str, Enum):
    BOLD = "BOLD"
    """
    Blood Oxygen Level Dependent signal.
    """
    EEG = "EEG"
    """
    Electroencephalography.
    """
    MEG = "MEG"
    """
    Magnetoencephalography.
    """
    SEEG = "SEEG"
    """
    Stereoelectroencephalography.
    """
    IEEG = "IEEG"
    """
    Intracranial Electroencephalography.
    """


class ModelType(str, Enum):
    """
    Coarse classification of a Dynamics model by its mathematical/biological origin. Used for filtering and display in list_db().
    """
    mean_field = "mean_field"
    """
    Mathematically derived mean-field models obtained by exact reduction of spiking networks (Ott-Antonsen ansatz, Lorentzian heterogeneity, etc.). Examples: MontbrioPazoRoxin, CoombesByrne, ReducedWongWang, ZerlautAdaptationFirstOrder.
    """
    neural_mass = "neural_mass"
    """
    Phenomenological population-rate / neural-mass models that describe synaptic and firing-rate dynamics without an explicit derivation from single-neuron statistics. Examples: JansenRit, WilsonCowan, LarterBreakspear, TsodyksMarkram.
    """
    phase_oscillator = "phase_oscillator"
    """
    Phase-reduced or Kuramoto-type oscillator models. Examples: Kuramoto, SupHopf.
    """
    phenomenological = "phenomenological"
    """
    Empirical / phenomenological models that capture macroscopic dynamics without direct biophysical derivation. Examples: Epileptor2D, Epileptor5D.
    """
    spiking = "spiking"
    """
    Single-neuron or conductance-based spiking models (HH, AdEx, LIF, Izhikevich, etc.). These can be used as nodes in a network alongside mean-field models.
    """
    generic = "generic"
    """
    Generic / normal-form dynamical systems not specific to neural modelling (e.g. Generic2dOscillator, GenericLinear).
    """
    field = "field"
    """
    Spatially distributed neural-field models described by integro- differential or PDE formulations.
    """


class SystemType(str, Enum):
    continuous = "continuous"
    """
    Continuous-time dynamics (e.g., ODE/SDE).
    """
    discrete = "discrete"
    """
    Discrete-time dynamics (e.g., maps, iterated updates).
    """


class BoundaryConditionType(str, Enum):
    Dirichlet = "Dirichlet"
    Neumann = "Neumann"
    Robin = "Robin"
    Periodic = "Periodic"


class DiscretizationMethod(str, Enum):
    FDM = "FDM"
    """
    Finite Difference Method
    """
    FEM = "FEM"
    """
    Finite Element Method
    """
    FVM = "FVM"
    """
    Finite Volume Method
    """
    Spectral = "Spectral"


class ElementType(str, Enum):
    triangle = "triangle"
    quad = "quad"
    tetrahedron = "tetrahedron"
    hexahedron = "hexahedron"


class OperatorType(str, Enum):
    gradient = "gradient"
    divergence = "divergence"
    laplacian = "laplacian"
    curl = "curl"


class SamplingAxis(str, Enum):
    """
    Dimension along which a distribution is sampled.
    """
    space = "space"
    """
    Sample once per node (heterogeneous parameter or spatially varying IC).
    """
    time = "time"
    """
    Resample every integration timestep (stochastic time-varying input).
    """


class NoiseType(str, Enum):
    gaussian = "gaussian"
    white = "white"
    brown = "brown"
    pink = "pink"


class AggregationType(str, Enum):
    """
    How to aggregate time series data
    """
    mean = "mean"
    """
    Average over time
    """
    last = "last"
    """
    Last value in window
    """
    first = "first"
    """
    First value in window
    """
    window = "window"
    """
    Sliding window aggregation
    """
    none = "none"
    """
    No aggregation
    """


class EventType(str, Enum):
    """
    Type of event triggering mechanism.
    """
    continuous = "continuous"
    """
    Triggered when condition function crosses zero (root-finding). Maps to ContinuousCallback / ContinuousComponentCallback.
    """
    discrete = "discrete"
    """
    Triggered when condition function returns true (checked at each step). Maps to DiscreteCallback / DiscreteComponentCallback.
    """
    preset_time = "preset_time"
    """
    Triggered at predetermined time points. Maps to PresetTimeCallback / PresetTimeComponentCallback.
    """
    stimulus = "stimulus"
    """
    Continuous time-dependent input signal (e.g., external current). Legacy Stimulus behavior.
    """
    stimulation = "stimulation"
    """
    Synonym of 'stimulus' — a continuous time-dependent input signal injected into a target state variable across target regions. The codegen treats 'stimulation' and 'stimulus' identically.
    """


class StandardGraphType(str, Enum):
    """
    Well-known graph generator families with automatic backend mapping. The type field on GraphGenerator is a free string; this enum lists common types that get automatic code generation for Julia (Graphs.jl) and Python (NetworkX).

    """
    BarabasiAlbert = "BarabasiAlbert"
    """
    Barabasi-Albert preferential attachment (params: k)
    """
    WattsStrogatz = "WattsStrogatz"
    """
    Watts-Strogatz small-world (params: k, p)
    """
    ErdosRenyi = "ErdosRenyi"
    """
    Erdos-Renyi random graph (params: p)
    """
    Complete = "Complete"
    """
    Complete graph (all-to-all)
    """
    Cycle = "Cycle"
    """
    Cycle graph (ring)
    """
    Star = "Star"
    """
    Star graph
    """
    RandomRegular = "RandomRegular"
    """
    Random regular graph (params: k)
    """
    Grid = "Grid"
    """
    Grid/lattice graph (params: dims)
    """
    RandomReservoir = "RandomReservoir"
    """
    Sparse random recurrent adjacency with post-hoc spectral- radius rescaling. Parameters: n, sparsity, spectral_radius, weight_distribution, seed. Canonical Echo State Network substrate (Jaeger 2001) for reservoir computing.
    """
    WeightShuffle = "WeightShuffle"
    """
    Derived generator: permute the non-zero entries of a source matrix. Parameters: source (IRI to another Network), preserve (binary_mask | degree | weight_distribution), seed. Used for null-model controls (e.g. shuffled SC).
    """


class DimensionType(str, Enum):
    """
    Dimensions along which operations can be applied
    """
    time = "time"
    """
    Temporal dimension
    """
    state = "state"
    """
    State variable dimension
    """
    node = "node"
    """
    Network node dimension (general graph term)
    """
    region = "region"
    """
    Spatial/regional dimension (alias for node in brain networks)
    """
    mode = "mode"
    """
    Mode dimension (e.g., coupling modes)
    """
    sample = "sample"
    """
    Sample/trial/realization dimension
    """
    batch = "batch"
    """
    Batch dimension (for parallel processing)
    """
    frequency = "frequency"
    """
    Frequency dimension (spectral analysis)
    """


class ReductionType(str, Enum):
    """
    Operations for reducing/aggregating values across dimensions
    """
    mean = "mean"
    """
    Arithmetic mean
    """
    sum = "sum"
    """
    Sum of values
    """
    max = "max"
    """
    Maximum value
    """
    min = "min"
    """
    Minimum value
    """
    none = "none"
    """
    No reduction (return per-element values)
    """


class ContinuationAlgorithm(str, Enum):
    """
    Predictor-corrector algorithm for numerical continuation.
    """
    PALC = "PALC"
    """
    Pseudo-arclength continuation (default). Uses weighted dot product constraint.
    """
    MoorePenrose = "MoorePenrose"
    """
    Moore-Penrose continuation.
    """
    Natural = "Natural"
    """
    Natural parameter continuation. Simple parameter stepping, no arc-length constraint.
    """


class NumericalDiscretizationMethod(str, Enum):
    """
    Numerical discretization method for boundary value problems (periodic orbits, connecting orbits, quasi-periodic tori).
    """
    collocation = "collocation"
    """
    Orthogonal collocation at Gauss points.
    """
    trapezoid = "trapezoid"
    """
    Trapezoidal rule discretization.
    """
    shooting = "shooting"
    """
    Standard multiple shooting.
    """
    poincare = "poincare"
    """
    Poincaré shooting.
    """


class InitialStateMethod(str, Enum):
    """
    Strategy for obtaining the starting equilibrium or periodic orbit.
    """
    time_integration = "time_integration"
    """
    Integrate the ODE forward until convergence (robust, default).
    """
    newton = "newton"
    """
    Use Newton's method to find the nearest fixed point.
    """
    given = "given"
    """
    Use the model's default initial values directly.
    """
    from_branch = "from_branch"
    """
    Start from a point on a previously computed branch.
    """


class SparseFormat(str, Enum):
    dense = "dense"
    """
    Dense N×N array with gzip compression
    """
    csr = "csr"
    """
    Compressed Sparse Row (data, indices, indptr)
    """
    coo = "coo"
    """
    Coordinate list (data, row, col)
    """



class Coordinate(ConfiguredBaseModel):
    """
    A 3D coordinate with X, Y, Z values.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://openminds.ebrains.eu/sands/BrainAtlas'})

    coordinateSpace: Optional[str] = Field(default=None, description="""Common coordinate space (e.g. FSLMNI152, MNI152NLin2009cAsym). Reference by name; the CommonCoordinateSpace must be defined in tvbo/database/coordinate_spaces/ (or future equivalent location).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'BrainAtlas']} })
    x: Optional[float] = Field(default=None, description="""X coordinate""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'Matrix']} })
    y: Optional[float] = Field(default=None, description="""Y coordinate""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'Matrix']} })
    z: Optional[float] = Field(default=None, description="""Z coordinate""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate']} })


class BrainAtlas(ConfiguredBaseModel):
    """
    A schema for representing a version of a brain atlas.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'atom:atlas/Atlas',
         'from_schema': 'https://openminds.ebrains.eu/sands/BrainAtlas'})

    coordinateSpace: Optional[str] = Field(default=None, description="""Common coordinate space (e.g. FSLMNI152, MNI152NLin2009cAsym). Reference by name; the CommonCoordinateSpace must be defined in tvbo/database/coordinate_spaces/ (or future equivalent location).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'BrainAtlas']} })
    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    abbreviation: Optional[str] = Field(default=None, description="""Slot for the abbreviation of a resource.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'CommonCoordinateSpace', 'ParcellationEntity'],
         'slot_uri': 'skos:notation'} })
    author: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'SimulationTool'], 'slot_uri': 'dcterms:creator'} })
    isVersionOf: Optional[str] = Field(default=None, description="""Linked type for the version of a brain atlas or coordinate space.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas'],
         'slot_uri': 'dcterms:isVersionOf',
         'union_of': ['BrainAtlas',
                      'CommonCoordinateSpace',
                      'ParcellationTerminology',
                      'ParcellationEntity']} })
    versionIdentifier: Optional[str] = Field(default=None, description="""Enter the version identifier of this brain atlas or coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'ParcellationEntity', 'ParcellationTerminology'],
         'slot_uri': 'dcterms:hasVersion'} })
    terminology: Optional[ParcellationTerminology] = Field(default=None, description="""Add the parcellation terminology version used for this brain atlas version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas']} })


class CommonCoordinateSpace(ConfiguredBaseModel):
    """
    A schema for representing a version of a common coordinate space.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'atom:atlas/Transformation',
         'from_schema': 'https://openminds.ebrains.eu/sands/BrainAtlas'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    abbreviation: Optional[str] = Field(default=None, description="""Slot for the abbreviation of a resource.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'CommonCoordinateSpace', 'ParcellationEntity'],
         'slot_uri': 'skos:notation'} })
    alternateName: Optional[list[str]] = Field(default=None, description="""Enter any alternate names, including abbreviations, for this entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace', 'ParcellationEntity'],
         'slot_uri': 'atom:atlas/hasName'} })
    unit: Optional[UnitEnum] = Field(default=None, description="""Physical unit of measurement. Values are drawn from the QUDT ontology (http://qudt.org/vocab/unit/) with UO cross-references where available.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField'],
         'slot_uri': 'qudt:unit'} })
    license: Optional[str] = Field(default=None, description="""Linked type for the license of the brain atlas or coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'Provenance'],
         'slot_uri': 'dcterms:license'} })
    anatomicalAxesOrientation: Optional[str] = Field(default=None, description="""Add the axes orientation in standard anatomical terms (XYZ).""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace']} })
    axesOrigin: Optional[str] = Field(default=None, description="""Enter the origin (central point where all axes intersect).""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace']} })
    nativeUnit: Optional[str] = Field(default=None, description="""Add the native unit that is used for this common coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace']} })
    defaultImage: Optional[list[str]] = Field(default=None, description="""Add all image files used as visual representation of this common coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace']} })


class ParcellationEntity(ConfiguredBaseModel):
    """
    A schema for representing a parcellation entity, which is an anatomical location or study target.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'categories': ['anatomicalLocation', 'studyTarget'],
         'class_uri': 'atom:atlas/Region',
         'from_schema': 'https://openminds.ebrains.eu/sands/BrainAtlas'})

    abbreviation: Optional[str] = Field(default=None, description="""Slot for the abbreviation of a resource.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'CommonCoordinateSpace', 'ParcellationEntity'],
         'slot_uri': 'skos:notation'} })
    alternateName: Optional[list[str]] = Field(default=None, description="""Enter any alternate names, including abbreviations, for this entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace', 'ParcellationEntity'],
         'slot_uri': 'atom:atlas/hasName'} })
    lookupLabel: Optional[int] = Field(default=None, description="""Enter the label used for looking up this entity in the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity'], 'slot_uri': 'atom:atlas/lookupLabel'} })
    hasParent: Optional[list[str]] = Field(default=None, description="""Add all anatomical parent structures for this entity as defined within the corresponding brain atlas.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity'], 'slot_uri': 'atom:atlas/hasParent'} })
    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    ontologyIdentifier: Optional[list[str]] = Field(default=None, description="""Enter the internationalized resource identifier (IRI) to the related ontological terms.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity', 'ParcellationTerminology'],
         'slot_uri': 'atom:atlas/hasIlxId'} })
    versionIdentifier: Optional[str] = Field(default=None, description="""Enter the version identifier of this brain atlas or coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'ParcellationEntity', 'ParcellationTerminology'],
         'slot_uri': 'dcterms:hasVersion'} })
    relatedUBERONTerm: Optional[str] = Field(default=None, description="""Add the related anatomical entity as defined by the UBERON ontology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity']} })
    originalLookupLabel: Optional[int] = Field(default=None, description="""Add the original label of this entity as defined in the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity']} })
    hemisphere: Optional[Hemisphere] = Field(default=None, description="""Add the hemisphere of this entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity', 'Electrode']} })
    center: Optional[Coordinate] = Field(default=None, description="""Add the center coordinate of this entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity']} })
    color: Optional[str] = Field(default=None, description="""Add the color code used for visual representation of this entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity']} })


class ParcellationTerminology(ConfiguredBaseModel):
    """
    A schema for representing a parcellation terminology, which consists of parcellation entities.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'atom:parcellationTerminology',
         'from_schema': 'https://openminds.ebrains.eu/sands/BrainAtlas'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })
    ontologyIdentifier: Optional[list[str]] = Field(default=None, description="""Enter the internationalized resource identifier (IRI) to the related ontological terms.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity', 'ParcellationTerminology'],
         'slot_uri': 'atom:atlas/hasIlxId'} })
    versionIdentifier: Optional[str] = Field(default=None, description="""Enter the version identifier of this brain atlas or coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'ParcellationEntity', 'ParcellationTerminology'],
         'slot_uri': 'dcterms:hasVersion'} })
    entities: Optional[dict[str, ParcellationEntity]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology']} })


class Subject(ConfiguredBaseModel):
    """
    A participant in a study. Each subject typically has their own brain network (connectome) and empirical recordings. Corresponds to a BIDS 'sub-' entity.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/study'})

    subject_id: str = Field(default=..., description="""BIDS-compatible subject identifier (without 'sub-' prefix). Examples: '01', 'ctrl03', 'patient17'.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject', 'TimeSeries'],
         'exact_mappings': ['schema:identifier'],
         'slot_uri': 'dcterms:identifier'} })
    label: Optional[str] = Field(default=None, description="""Human-readable label for the subject.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    group: Optional[str] = Field(default=None, description="""Group assignment (e.g., 'control', 'patient', 'healthy'). Maps to participants.tsv 'group' column in BIDS.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    age: Optional[float] = Field(default=None, description="""Age at time of study (years).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    sex: Optional[SexEnum] = Field(default=None, description="""Biological sex.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    sessions: Optional[dict[str, Session]] = Field(default=None, description="""Data collection sessions for this subject. Each session can have its own network, empirical data, and conditions.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    network: Optional[str] = Field(default=None, description="""Path to subject-specific connectome (when not session-dependent). Relative to dataset root or BIDS derivatives. For session-specific networks, use Session.network instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject', 'Session', 'SimulationExperiment']} })
    metadata: Optional[str] = Field(default=None, description="""Additional subject metadata as key-value pairs or path to a sidecar JSON file.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })


class Session(ConfiguredBaseModel):
    """
    A data collection session for a subject. Corresponds to a BIDS 'ses-' entity. Sessions capture longitudinal timepoints (baseline, follow-up), different experimental conditions, or repeated measures.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/study'})

    session_id: str = Field(default=..., description="""BIDS session identifier (without 'ses-' prefix). Examples: 'baseline', '6month', 'pre', 'post'.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Session', 'TimeSeries'], 'slot_uri': 'dcterms:identifier'} })
    label: Optional[str] = Field(default=None, description="""Human-readable session label.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    network: Optional[str] = Field(default=None, description="""Path to session-specific connectome. Overrides Subject.network when set. Relative to dataset root or BIDS derivatives.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject', 'Session', 'SimulationExperiment']} })
    empirical_data: Optional[list[str]] = Field(default=None, description="""Paths to empirical recordings for this session (e.g., BOLD time series, MEG/EEG). Relative to dataset root.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Session']} })
    condition: Optional[str] = Field(default=None, description="""Experimental condition label (e.g., 'rest', 'task-nback'). Maps to BIDS 'task-' entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Session', 'ConditionalBlock', 'Event', 'Case']} })


class Dataset(ConfiguredBaseModel):
    """
    A collection of subjects for a multi-subject study. Provides the subject/session structure needed for workflow rendering. Optionally backed by a BIDS directory layout.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/study'})

    dataset_id: str = Field(default=..., description="""Unique identifier for the dataset.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'Phenotype'], 'slot_uri': 'dcterms:identifier'} })
    subjects: Optional[dict[str, Subject]] = Field(default=None, description="""Subjects in a dataset.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'Phenotype']} })
    label: Optional[str] = Field(default=None, description="""Human-readable dataset name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    bids_root: Optional[str] = Field(default=None, description="""Path to BIDS dataset root directory. When set, subject networks and empirical data paths are resolved relative to this root.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    conditions: Optional[list[str]] = Field(default=None, description="""Global condition labels applied across all subjects (e.g., ['rest', 'task-nback', 'task-motor']).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    reference: Optional[str] = Field(default=None, description="""DOI or citation for this dataset.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'ClinicalScale', 'ClinicalScore', 'Tractogram'],
         'slot_uri': 'dcterms:references'} })


class DBSDataset(Dataset):
    """
    Collection of data related to a specific DBS study.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/dbs',
         'slot_usage': {'subjects': {'name': 'subjects', 'range': 'DBSSubject'}}})

    clinical_scores: Optional[list[ClinicalScore]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSDataset']} })
    coordinate_space: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSDataset',
                       'DBSSubject',
                       'Electrode',
                       'EField',
                       'Network',
                       'SpatialDomain',
                       'Mesh']} })
    dataset_id: str = Field(default=..., description="""Unique identifier for the dataset.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'Phenotype'], 'slot_uri': 'dcterms:identifier'} })
    subjects: Optional[dict[str, DBSSubject]] = Field(default=None, description="""Subjects in a dataset.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'Phenotype']} })
    label: Optional[str] = Field(default=None, description="""Human-readable dataset name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    bids_root: Optional[str] = Field(default=None, description="""Path to BIDS dataset root directory. When set, subject networks and empirical data paths are resolved relative to this root.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    conditions: Optional[list[str]] = Field(default=None, description="""Global condition labels applied across all subjects (e.g., ['rest', 'task-nback', 'task-motor']).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    reference: Optional[str] = Field(default=None, description="""DOI or citation for this dataset.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'ClinicalScale', 'ClinicalScore', 'Tractogram'],
         'slot_uri': 'dcterms:references'} })


class DBSSubject(Subject):
    """
    Human or animal subject receiving DBS.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/dbs'})

    diagnosis: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSSubject']} })
    handedness: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSSubject']} })
    protocols: Optional[list[str]] = Field(default=None, description="""All DBS protocols assigned to this subject.""", json_schema_extra = { "linkml_meta": {'domain_of': ['DBSSubject']} })
    coordinate_space: Optional[str] = Field(default=None, description="""Coordinate space used for this subject's data""", json_schema_extra = { "linkml_meta": {'domain_of': ['DBSDataset',
                       'DBSSubject',
                       'Electrode',
                       'EField',
                       'Network',
                       'SpatialDomain',
                       'Mesh']} })
    subject_id: str = Field(default=..., description="""BIDS-compatible subject identifier (without 'sub-' prefix). Examples: '01', 'ctrl03', 'patient17'.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject', 'TimeSeries'],
         'exact_mappings': ['schema:identifier'],
         'slot_uri': 'dcterms:identifier'} })
    label: Optional[str] = Field(default=None, description="""Human-readable label for the subject.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    group: Optional[str] = Field(default=None, description="""Group assignment (e.g., 'control', 'patient', 'healthy'). Maps to participants.tsv 'group' column in BIDS.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    age: Optional[float] = Field(default=None, description="""Age at time of study (years).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    sex: Optional[SexEnum] = Field(default=None, description="""Biological sex.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    sessions: Optional[dict[str, Session]] = Field(default=None, description="""Data collection sessions for this subject. Each session can have its own network, empirical data, and conditions.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    network: Optional[str] = Field(default=None, description="""Path to subject-specific connectome (when not session-dependent). Relative to dataset root or BIDS derivatives. For session-specific networks, use Session.network instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject', 'Session', 'SimulationExperiment']} })
    metadata: Optional[str] = Field(default=None, description="""Additional subject metadata as key-value pairs or path to a sidecar JSON file.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })


class Electrode(ConfiguredBaseModel):
    """
    Implanted DBS electrode and contact geometry.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/dbs'})

    electrode_id: Optional[str] = Field(default=None, description="""Unique identifier for this electrode""", json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode'], 'slot_uri': 'dcterms:identifier'} })
    manufacturer: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    model: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode', 'SimulationExperiment', 'SimulationStudy']} })
    hemisphere: Optional[str] = Field(default="left", description="""Hemisphere of electrode (left/right)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity', 'Electrode'], 'ifabsent': 'left'} })
    contacts: Optional[list[Contact]] = Field(default=None, description="""List of physical contacts along the electrode""", json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    head: Optional[Coordinate] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    tail: Optional[Coordinate] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    trajectory: Optional[list[Coordinate]] = Field(default=None, description="""The planned trajectory for electrode implantation""", json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    target_structure: Optional[str] = Field(default=None, description="""Anatomical target structure from a brain atlas""", json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    coordinate_space: Optional[str] = Field(default=None, description="""Coordinate space used for implantation planning""", json_schema_extra = { "linkml_meta": {'domain_of': ['DBSDataset',
                       'DBSSubject',
                       'Electrode',
                       'EField',
                       'Network',
                       'SpatialDomain',
                       'Mesh']} })
    recon_path: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })


class Contact(ConfiguredBaseModel):
    """
    Individual contact on a DBS electrode.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/dbs'})

    contact_id: Optional[int] = Field(default=None, description="""Identifier (e.g., 0, 1, 2)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Contact'], 'slot_uri': 'dcterms:identifier'} })
    coordinate: Optional[Coordinate] = Field(default=None, description="""3D coordinate of the contact center in the defined coordinate space""", json_schema_extra = { "linkml_meta": {'domain_of': ['Contact']} })
    label: Optional[str] = Field(default=None, description="""Optional human-readable label (e.g., \"1a\")""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })


class StimulationSetting(ConfiguredBaseModel):
    """
    DBS parameters for a specific session.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/dbs'})

    electrode_reference: Optional[Electrode] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })
    amplitude: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })
    frequency: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })
    pulse_width: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })
    mode: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting', 'Exploration']} })
    active_contacts: Optional[list[int]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })
    efield: Optional[EField] = Field(default=None, description="""Metadata about the E-field result for this setting""", json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })


class DBSProtocol(ConfiguredBaseModel):
    """
    A protocol describing DBS therapy, potentially bilateral or multi-lead.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/dbs'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    electrodes: Optional[list[Electrode]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSProtocol']} })
    settings: Optional[list[StimulationSetting]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSProtocol']} })
    timing_info: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSProtocol']} })
    notes: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSProtocol']} })
    clinical_improvement: Optional[list[ClinicalImprovement]] = Field(default=None, description="""Observed improvement relative to baseline based on a defined score.""", json_schema_extra = { "linkml_meta": {'domain_of': ['DBSProtocol']} })


class ClinicalScale(ConfiguredBaseModel):
    """
    A clinical assessment inventory or structured scale composed of multiple scores or items.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/dbs'})

    acronym: Optional[str] = Field(default=None, description="""Short abbreviation (e.g., UPDRS)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'Observation',
                       'Function',
                       'FunctionCall'],
         'slot_uri': 'skos:notation'} })
    name: Optional[str] = Field(default=None, description="""Full name of the scale (e.g., Unified Parkinson’s Disease Rating Scale)""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    version: Optional[str] = Field(default=None, description="""Version of the instrument (e.g., 3.0)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'SimulationTool',
                       'SoftwareRequirement',
                       'SoftwareEnvironment'],
         'slot_uri': 'schema:softwareVersion'} })
    domain: Optional[str] = Field(default=None, description="""Overall clinical domain (e.g., motor, cognition)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'ExplorationAxis',
                       'FreeParameter',
                       'PDE']} })
    reference: Optional[str] = Field(default=None, description="""DOI, PMID or persistent identifier""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'ClinicalScale', 'ClinicalScore', 'Tractogram'],
         'slot_uri': 'dcterms:references'} })


class ClinicalScore(ConfiguredBaseModel):
    """
    Metadata about a clinical score or scale.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/dbs'})

    acronym: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'Observation',
                       'Function',
                       'FunctionCall'],
         'slot_uri': 'skos:notation'} })
    name: Optional[str] = Field(default=None, description="""Full name of the score (e.g., Unified Parkinson's Disease Rating Scale - Part III)""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    domain: Optional[str] = Field(default=None, description="""Domain assessed (e.g. motor, mood, pain)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'ExplorationAxis',
                       'FreeParameter',
                       'PDE']} })
    reference: Optional[str] = Field(default=None, description="""PubMed ID, DOI, or other reference to the score definition""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'ClinicalScale', 'ClinicalScore', 'Tractogram'],
         'slot_uri': 'dcterms:references'} })
    scale: Optional[ClinicalScale] = Field(default=None, description="""The scale this score belongs to, if applicable""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore', 'SimulationTool', 'BidsEntities']} })
    parent_score: Optional[ClinicalScore] = Field(default=None, description="""If this score is a subscore of a broader composite""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore']} })


class ClinicalImprovement(ConfiguredBaseModel):
    """
    Relative improvement on a defined clinical score.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/dbs'})

    score: Optional[ClinicalScore] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalImprovement']} })
    baseline_value: Optional[float] = Field(default=None, description="""Preoperative baseline value of the score""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalImprovement']} })
    absolute_value: Optional[float] = Field(default=None, description="""Absolute value of the score at the time of assessment""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalImprovement']} })
    percent_change: Optional[float] = Field(default=None, description="""Percent change compared to preoperative baseline (positive = improvement)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalImprovement']} })
    time_post_surgery: Optional[float] = Field(default=None, description="""Timepoint of assessment in days or months after implantation""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalImprovement']} })
    evaluator: Optional[str] = Field(default=None, description="""Who performed the rating (e.g., rater initials, clinician ID, or system)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalImprovement']} })
    timepoint: Optional[str] = Field(default=None, description="""Timepoint of assessment (e.g., \"1 month post-op\", \"6 months post-op\")""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalImprovement']} })


class EField(ConfiguredBaseModel):
    """
    Simulated electric field from DBS modeling.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo/dbs'})

    volume_data: Optional[str] = Field(default=None, description="""Reference to raw or thresholded volume""", json_schema_extra = { "linkml_meta": {'domain_of': ['EField']} })
    coordinate_space: Optional[str] = Field(default=None, description="""Reference to a common coordinate space (e.g. MNI152)""", json_schema_extra = { "linkml_meta": {'domain_of': ['DBSDataset',
                       'DBSSubject',
                       'Electrode',
                       'EField',
                       'Network',
                       'SpatialDomain',
                       'Mesh']} })
    threshold_applied: Optional[float] = Field(default=None, description="""Threshold value applied to the E-field simulation""", json_schema_extra = { "linkml_meta": {'domain_of': ['EField']} })


class SoftwarePackage(ConfiguredBaseModel):
    """
    Identity and metadata for a software package, aligned with schema.org/SoftwareApplication and CodeMeta v3.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'schema:SoftwareApplication',
         'from_schema': 'https://w3id.org/tvbo/software'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    homepage: Optional[str] = Field(default=None, description="""Project homepage URL.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage'], 'slot_uri': 'schema:url'} })
    license: Optional[str] = Field(default=None, description="""SPDX license identifier (e.g., MIT, GPL-3.0-only).""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'Provenance'],
         'slot_uri': 'dcterms:license'} })
    repository: Optional[str] = Field(default=None, description="""Source code repository URL.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage'], 'slot_uri': 'schema:codeRepository'} })
    doi: Optional[str] = Field(default=None, description="""Digital Object Identifier for the software or its reference publication.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage', 'SimulationStudy'], 'slot_uri': 'bibo:doi'} })
    ecosystem: Optional[list[EcosystemEnum]] = Field(default=None, description="""Package ecosystem(s) through which the software is distributed.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage']} })


class SimulationTool(SoftwarePackage):
    """
    A software tool for computational neuroscience simulation, analysis, or model specification. Extends SoftwarePackage with neuroscience-specific controlled vocabularies for scale, paradigm, role, and interoperability. Aligned with CodeMeta v3 and DOAP.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'schema:SoftwareApplication',
         'from_schema': 'https://w3id.org/tvbo/software'})

    application_category: Optional[str] = Field(default=None, description="""High-level category (e.g., simulation, analysis, specification).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'schema:applicationCategory'} })
    scale: Optional[list[SimulationScale]] = Field(default=None, description="""Spatial/organizational scales the tool operates at.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore', 'SimulationTool', 'BidsEntities']} })
    model_paradigm: Optional[list[ModelParadigm]] = Field(default=None, description="""Computational paradigms supported.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool']} })
    tool_role: Optional[list[ToolRole]] = Field(default=None, description="""Primary function(s) in a simulation workflow.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool']} })
    programming_language: Optional[list[ProgrammingLanguageEnum]] = Field(default=None, description="""Implementation languages.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'schema:programmingLanguage'} })
    runtime_platform: Optional[list[str]] = Field(default=None, description="""Execution backends or platforms (e.g., MPI, OpenMP, CUDA, JAX).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'schema:runtimePlatform'} })
    operating_system: Optional[list[str]] = Field(default=None, description="""Supported operating systems (e.g., Linux, macOS, Windows).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'schema:operatingSystem'} })
    interoperates_with: Optional[list[str]] = Field(default=None, description="""Tools this tool can exchange data with or delegate to.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool']} })
    version: Optional[str] = Field(default=None, description="""Current or latest stable version string.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'SimulationTool',
                       'SoftwareRequirement',
                       'SoftwareEnvironment'],
         'slot_uri': 'schema:softwareVersion'} })
    date_created: Optional[date] = Field(default=None, description="""Date the software was first released (YYYY-MM-DD).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool', 'Provenance'],
         'slot_uri': 'schema:dateCreated'} })
    date_modified: Optional[date] = Field(default=None, description="""Date of the most recent release or significant update.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'schema:dateModified'} })
    development_status: Optional[DevelopmentStatus] = Field(default=None, description="""Development status (repostatus.org aligned).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool']} })
    author: Optional[list[str]] = Field(default=None, description="""Original author(s) or creating organization(s).""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'SimulationTool'], 'slot_uri': 'dcterms:creator'} })
    maintainer: Optional[list[str]] = Field(default=None, description="""Current maintainer(s) or responsible organization(s).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'schema:maintainer'} })
    funder: Optional[list[str]] = Field(default=None, description="""Funding bodies or grants (e.g., 'EU H2020 945539', 'NIH R01...').""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'schema:funder'} })
    reference_publication: Optional[str] = Field(default=None, description="""DOI of the primary reference publication for this tool.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'dcterms:references'} })
    citation: Optional[list[str]] = Field(default=None, description="""Additional citation strings or DOIs.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'schema:citation'} })
    keywords: Optional[list[str]] = Field(default=None, description="""Tags for topic-based discovery and clustering.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'schema:keywords'} })
    same_as: Optional[list[str]] = Field(default=None, description="""URIs that unambiguously identify this tool (Wikidata, bio.tools, RRID, SciCrunch).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'], 'slot_uri': 'schema:sameAs'} })
    issue_tracker: Optional[str] = Field(default=None, description="""URL of the bug tracker or issue board.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool']} })
    is_accessible_for_free: Optional[bool] = Field(default=True, description="""Whether the tool is free/open-source.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool'],
         'ifabsent': 'True',
         'slot_uri': 'schema:isAccessibleForFree'} })
    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    homepage: Optional[str] = Field(default=None, description="""Project homepage URL.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage'], 'slot_uri': 'schema:url'} })
    license: Optional[str] = Field(default=None, description="""SPDX license identifier (e.g., MIT, GPL-3.0-only).""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'Provenance'],
         'slot_uri': 'dcterms:license'} })
    repository: Optional[str] = Field(default=None, description="""Source code repository URL.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage'], 'slot_uri': 'schema:codeRepository'} })
    doi: Optional[str] = Field(default=None, description="""Digital Object Identifier for the software or its reference publication.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage', 'SimulationStudy'], 'slot_uri': 'bibo:doi'} })
    ecosystem: Optional[list[EcosystemEnum]] = Field(default=None, description="""Package ecosystem(s) through which the software is distributed.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage']} })


class SoftwareRequirement(ConfiguredBaseModel):
    """
    An individual software requirement binding a package to a version constraint and a role within an environment.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'comments': ['Represents an individual requirement (package/module/library).',
                      "Use 'version_spec' instead of 'version' for semantic clarity."],
         'from_schema': 'https://w3id.org/tvbo/software'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    dataLocation: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })
    package: Optional[str] = Field(default=None, description="""Reference to the software package identity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    version_spec: Optional[str] = Field(default=None, description="""Version or constraint specifier (e.g., '==2.7.3', '>=1.2,<2').""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    role: Optional[RequirementRole] = Field(default=RequirementRole.runtime, json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement'], 'ifabsent': 'runtime'} })
    optional: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement'], 'ifabsent': 'False'} })
    hash: Optional[str] = Field(default=None, description="""Build or artifact hash for exact reproducibility.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement', 'ReferenceFingerprint']} })
    source_url: Optional[str] = Field(default=None, description="""Canonical source or repository URL.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    url: Optional[str] = Field(default=None, description="""(Deprecated) Use source_url.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    license: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'Provenance'],
         'slot_uri': 'dcterms:license'} })
    modules: Optional[list[str]] = Field(default=None, description="""(Deprecated) Use environment.requirements list instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    version: Optional[str] = Field(default=None, description="""(Deprecated) Use version_spec.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'SimulationTool',
                       'SoftwareRequirement',
                       'SoftwareEnvironment']} })


class SoftwareEnvironment(ConfiguredBaseModel):
    """
    A reproducible software environment aggregating one or more SoftwareRequirement entries. Used by SimulationExperiment to specify the execution context.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'comments': ['An environment aggregates one or more SoftwareRequirement '
                      'entries.',
                      'Use SimulationExperiment.environment to reference a reusable '
                      'environment.'],
         'from_schema': 'https://w3id.org/tvbo/software'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    dataLocation: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })
    version: Optional[str] = Field(default=None, description="""Environment definition version (not a package version).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'SimulationTool',
                       'SoftwareRequirement',
                       'SoftwareEnvironment']} })
    platform: Optional[str] = Field(default=None, description="""OS / architecture (e.g., linux-64, macos-arm64).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment']} })
    environment_type: Optional[EnvironmentType] = Field(default=None, description="""Category: conda, venv, docker, etc.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment']} })
    container_image: Optional[str] = Field(default=None, description="""Container image reference (e.g., ghcr.io/org/img:tag@sha256:...).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment']} })
    build_hash: Optional[str] = Field(default=None, description="""Deterministic hash of the resolved dependency set.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment']} })
    requirements: Optional[list[SoftwareRequirement]] = Field(default=None, description="""Constituent software requirements.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment', 'Function', 'PDESolver']} })


class Range(ConfiguredBaseModel):
    """
    Specifies a range for array generation, parameter bounds, or grid exploration.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Range', 'from_schema': 'https://w3id.org/tvbo'})

    lo: Optional[Any] = Field(default=None, description="""Lower bound or starting value. Can be a number or argument name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range']} })
    hi: Optional[Any] = Field(default=None, description="""Upper bound or stopping value. Can be a number or argument name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range']} })
    step: Optional[Any] = Field(default=None, description="""Step size. Can be: number, argument name, or expression.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range']} })
    n: Optional[int] = Field(default=None, description="""Number of points (alternative to step for grid exploration).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range']} })
    log_scale: Optional[bool] = Field(default=False, description="""Whether to use logarithmic spacing.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range'], 'ifabsent': 'False'} })
    explored_values: Optional[AnyShapeArray[float]] = Field(default=None, description="""Explicit explored values for this element. When set on an element_domain entry, overrides the parent parameter's explored_values for this specific element.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range', 'Parameter', 'ExplorationAxis']} })
    element: Optional[int] = Field(default=None, description="""Element/node index this range applies to. Used in element_domains to explicitly link a domain to a specific element of a heterogeneous parameter (e.g., element: 0 for node 0). Required when used in element_domains to avoid ambiguous positional indexing.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range']} })


class Equation(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Equation', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator'],
         'slot_uri': 'skos:definition'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    lhs: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'aliases': ['lefthandside'], 'domain_of': ['Equation']} })
    rhs: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'aliases': ['righthandside'], 'domain_of': ['Equation']} })
    conditionals: Optional[list[ConditionalBlock]] = Field(default=None, description="""Conditional logic for piecewise equations.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    engine: Optional[SoftwareRequirement] = Field(default=None, description="""Primary engine (must appear in environment.requirements; migration target replacing deprecated 'software').""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    pycode: Optional[str] = Field(default=None, description="""Python code for the equation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation', 'Noise']} })
    latex: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation'], 'ifabsent': 'False'} })


class ConditionalBlock(ConfiguredBaseModel):
    """
    A single condition and its corresponding equation segment.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:ConditionalBlock', 'from_schema': 'https://w3id.org/tvbo'})

    condition: Optional[str] = Field(default=None, description="""The condition for this block (e.g., t > onset).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Session', 'ConditionalBlock', 'Event', 'Case']} })
    expression: Optional[str] = Field(default=None, description="""The equation to apply when the condition is met.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ConditionalBlock', 'DifferentialOperator']} })


class Stimulus(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Stimulus', 'from_schema': 'https://w3id.org/tvbo'})

    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })
    duration: Optional[float] = Field(default=1000, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'Event', 'InitialState', 'Integrator'],
         'ifabsent': 'float(1000)'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    regions: Optional[AnyShapeArray[int]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'Event']} })
    weighting: Optional[AnyShapeArray[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'Event']} })
    noise: Optional[Noise] = Field(default=None, description="""Optional stochastic contribution to the stimulus. Mirrors state_variables.noise on the Dynamics side. The stimulus value at each integration step is the sum of `equation`'s evaluation (deterministic, if set) and a draw from this Noise process. When only `noise` is set and `equation` is absent, the stimulus IS the noise (pure stochastic source — e.g. iid uniform driver for memory-capacity benchmarks; signal+noise paradigms for psychophysics).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'StateVariable', 'Integrator']} })


class Event(ConfiguredBaseModel):
    """
    A discrete or continuous event that modifies the system during simulation. Generalizes Stimulus: can represent external inputs (stimulus type), threshold-triggered state changes (continuous/discrete type), or time-scheduled interventions (preset_time type). Attaches to components (nodes/edges) or to the experiment level.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Event', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    event_type: Optional[EventType] = Field(default=EventType.stimulus, description="""Type of event trigger mechanism.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event'], 'ifabsent': 'stimulus'} })
    condition: Optional[Equation] = Field(default=None, description="""Condition function. For continuous events: triggers when expression crosses zero. For discrete events: triggers when expression evaluates to true. Not used for preset_time or stimulus types.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Session', 'ConditionalBlock', 'Event', 'Case']} })
    condition_states: Optional[list[str]] = Field(default=None, description="""State variable symbols accessible in the condition function. For edges, can include source/destination vertex outputs.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event']} })
    condition_parameters: Optional[list[str]] = Field(default=None, description="""Parameter symbols accessible in the condition function.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event']} })
    affect: Optional[Equation] = Field(default=None, description="""Affect function: what happens when the event triggers. Can modify state variables and/or parameters. For stimulus type, this is the stimulus equation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event']} })
    affect_states: Optional[list[str]] = Field(default=None, description="""State variable symbols modifiable in the affect function.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event']} })
    affect_parameters: Optional[list[str]] = Field(default=None, description="""Parameter symbols modifiable in the affect function.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event']} })
    affect_negative: Optional[Equation] = Field(default=None, description="""Affect on downcrossing (continuous events only). If not specified, uses the same affect for both crossings.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event']} })
    trigger_times: Optional[list[float]] = Field(default=None, description="""Predetermined trigger times for preset_time events. The solver will step exactly to these times.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event']} })
    target_component: Optional[str] = Field(default=None, description="""Component to attach this event to. Can be a node label, edge label, or 'all_edges'/'all_vertices' for broadcast. If not specified, event is experiment-level.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event']} })
    equation: Optional[Equation] = Field(default=None, description="""Stimulus equation for stimulus-type events. Legacy compatibility with Stimulus class.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator']} })
    regions: Optional[AnyShapeArray[int]] = Field(default=None, description="""Target regions for stimulus-type events.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'Event']} })
    weighting: Optional[AnyShapeArray[float]] = Field(default=None, description="""Per-region weighting for stimulus-type events (explicit values).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'Event']} })
    weight_distribution: Optional[Distribution] = Field(default=None, description="""Distribution to sample per-region stimulus weights from, as an alternative to the explicit `weighting` array (e.g. Uniform(-1, 1) per node). When set, the codegen samples one weight per target region using the distribution's seed.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event']} })
    target_variable: Optional[str] = Field(default=None, description="""State variable that a stimulus-type event drives — the variable its signal is injected into (e.g. 'y_0' for Jansen-Rit, or a named 'stimulus' input exposed to coupling). If absent, the stimulus adds to the model's default external-input variable.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event', 'TuningObjective']} })
    target_regions: Optional[list[str]] = Field(default=None, description="""Regions a stimulus-type event targets: 'all' (broadcast to every node) or explicit region labels / indices. More expressive than the integer-only `regions` slot; when both are absent the event applies to all regions.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event']} })
    duration: Optional[float] = Field(default=None, description="""Duration of stimulus-type events.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'Event', 'InitialState', 'Integrator']} })


class TemporalApplicableEquation(Equation):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:TemporalApplicableEquation',
         'from_schema': 'https://w3id.org/tvbo'})

    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    time_dependent: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['TemporalApplicableEquation',
                       'SpatialField',
                       'BoundaryCondition'],
         'ifabsent': 'False'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator'],
         'slot_uri': 'skos:definition'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    lhs: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'aliases': ['lefthandside'], 'domain_of': ['Equation']} })
    rhs: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'aliases': ['righthandside'], 'domain_of': ['Equation']} })
    conditionals: Optional[list[ConditionalBlock]] = Field(default=None, description="""Conditional logic for piecewise equations.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    engine: Optional[SoftwareRequirement] = Field(default=None, description="""Primary engine (must appear in environment.requirements; migration target replacing deprecated 'software').""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    pycode: Optional[str] = Field(default=None, description="""Python code for the equation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation', 'Noise']} })
    latex: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation'], 'ifabsent': 'False'} })


class Parcellation(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Parcellation', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    iri: Optional[str] = Field(default=None, description="""Optional stable IRI (or compact URI) for this entity in an external ontology or knowledge base. Used to load metadata from an external source; not required when the entity is fully self-contained (equations, parameters, etc. defined in the file itself).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation',
                       'Tractogram',
                       'ReferenceFingerprint',
                       'Reference',
                       'Dynamics',
                       'Function',
                       'Coupling'],
         'slot_uri': 'dcterms:identifier'} })
    data_source: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation', 'Tractogram', 'Observation']} })
    atlas: Optional[BrainAtlas] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation', 'BidsEntities']} })


class Tractogram(ConfiguredBaseModel):
    """
    Reference to tractography/diffusion MRI data used to derive structural connectivity
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Tractogram', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    iri: Optional[str] = Field(default=None, description="""Optional stable IRI (or compact URI) for this entity in an external ontology or knowledge base. Used to load metadata from an external source; not required when the entity is fully self-contained (equations, parameters, etc. defined in the file itself).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation',
                       'Tractogram',
                       'ReferenceFingerprint',
                       'Reference',
                       'Dynamics',
                       'Function',
                       'Coupling'],
         'slot_uri': 'dcterms:identifier'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    data_source: Optional[str] = Field(default=None, description="""Path or URI to the tractography data file""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation', 'Tractogram', 'Observation']} })
    number_of_subjects: Optional[int] = Field(default=None, description="""Number of subjects in the tractography dataset""", json_schema_extra = { "linkml_meta": {'domain_of': ['Tractogram']} })
    acquisition: Optional[str] = Field(default=None, description="""Acquisition protocol or scanner information""", json_schema_extra = { "linkml_meta": {'domain_of': ['Tractogram', 'BidsEntities']} })
    processing_pipeline: Optional[str] = Field(default=None, description="""Processing pipeline used to generate the tractography""", json_schema_extra = { "linkml_meta": {'domain_of': ['Tractogram']} })
    reference: Optional[str] = Field(default=None, description="""Publication or DOI reference for this tractography dataset""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'ClinicalScale', 'ClinicalScore', 'Tractogram'],
         'slot_uri': 'dcterms:references'} })


class Matrix(ConfiguredBaseModel):
    """
    Adjacency matrix of a network.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Matrix', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })
    x: Optional[BrainRegionSeries] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'Matrix']} })
    y: Optional[BrainRegionSeries] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'Matrix']} })
    values: Optional[list[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'BrainRegionSeries', 'SpatialField']} })
    format: Optional[SparseFormat] = Field(default=None, description="""Storage format in binary companion (dense, csr, coo)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'Edge', 'DataSource']} })
    shape: Optional[list[int]] = Field(default=None, description="""Matrix dimensions [N, M]""", json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'NamedArray', 'Parameter', 'FreeParameter', 'NDArray']} })
    dtype: Optional[str] = Field(default="float32", description="""Data type for matrix values""", json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'NamedArray', 'NDArray'],
         'ifabsent': 'string(float32)'} })


class BrainRegionSeries(ConfiguredBaseModel):
    """
    A series whose values represent latitude
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:BrainRegionSeries', 'from_schema': 'https://w3id.org/tvbo'})

    values: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'BrainRegionSeries', 'SpatialField']} })


class Provenance(ConfiguredBaseModel):
    """
    W3C PROV-O aligned provenance. Reusable on any entity (Network, TimeSeries, Dynamics, etc.).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'prov:Entity', 'from_schema': 'https://w3id.org/tvbo'})

    derived_from: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance', 'SimulationStudy'],
         'slot_uri': 'prov:wasDerivedFrom'} })
    references: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance',
                       'Dynamics',
                       'SimulationExperiment',
                       'SimulationStudy'],
         'slot_uri': 'dcterms:references'} })
    date_created: Optional[str] = Field(default=None, description="""ISO 8601 (prov:generatedAtTime)""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationTool', 'Provenance']} })
    license: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'Provenance']} })
    generated_by: Optional[str] = Field(default=None, description="""Software/agent identifier (prov:wasGeneratedBy)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })
    experiment_yaml_hash: Optional[str] = Field(default=None, description="""SHA-256 hex digest of the normalised experiment YAML that produced this artifact. Used by the cross-experiment cache (see :class:`SimulationStudy.from_file`) to detect when a cached upstream result has gone stale because the experiment's spec was edited. Computed as ``hashlib.sha256(yaml.safe_dump(normalized_yaml).encode()).hexdigest()`` over the fully resolved (post-``!include``) experiment block.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })
    inputs: Optional[list[ReferenceFingerprint]] = Field(default=None, description="""Per-input fingerprints for cache invalidation. Each entry records the IRI + dotted-field path of an ``aux_data`` reference plus the (mtime, size, sha256) of the underlying file at the time the artifact was produced. A downstream cache hit requires every fingerprint to match the current file state.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })


class ReferenceFingerprint(ConfiguredBaseModel):
    """
    Cache-invalidation fingerprint for one ``aux_data`` reference. Captures enough about the upstream artifact that a downstream cache can decide cheaply (via mtime + size) whether to trust the cached result, falling back to a hash recompute on mismatch.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:ReferenceFingerprint',
         'from_schema': 'https://w3id.org/tvbo'})

    iri: str = Field(default=..., description="""IRI of the referenced artifact (Network, ExperimentResult, BehavioralData, ...).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation',
                       'Tractogram',
                       'ReferenceFingerprint',
                       'Reference',
                       'Dynamics',
                       'Function',
                       'Coupling']} })
    field: Optional[str] = Field(default=None, description="""Dotted-path subkey within the referenced entity (optional).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceFingerprint', 'Reference', 'PDE']} })
    mtime: Optional[float] = Field(default=None, description="""File modification time at fingerprinting (POSIX timestamp).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceFingerprint']} })
    size: Optional[int] = Field(default=None, description="""File size in bytes at fingerprinting.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceFingerprint', 'Sample']} })
    hash: Optional[str] = Field(default=None, description="""SHA-256 hex digest of the file at fingerprinting (recomputed lazily when mtime/size differ).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement', 'ReferenceFingerprint']} })


class Phenotype(ConfiguredBaseModel):
    """
    Per-subject phenotype table (BIDS ``phenotype/`` directory convention). Carries cognitive scores, clinical scales, demographic variables, behavioral task outputs, physiological measures, or any other per-subject numeric measurement bundle for a cohort. Sidecar companion to per-subject Network sidecars in multi-subject studies that correlate simulated quantities with empirical scores (e.g. PMAT24_A, g-factor, CardSort, ProcSpeed for Schirner 2023). The yaml carries metadata + the measure list; the h5 carries ``measures/<name>`` 1-D float arrays of length ``len(subjects)``.
    Aligns with the BIDS phenotype standard (https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/phenotypic-and-assessment-data.html) and with NIDM's ``nidm:Phenotype`` concept. Per-measure metadata can optionally carry Cognitive Atlas (https://www.cognitiveatlas.org/) ``cogat:Task`` and ``cogat:Concept`` IRIs via ``measure_specs``.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Phenotype', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    dataset_id: str = Field(default=..., description="""Unique identifier for this phenotype bundle.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'Phenotype'], 'slot_uri': 'dcterms:identifier'} })
    subjects: list[str] = Field(default=..., description="""Ordered list of subject IDs (e.g. HCP-YA 6-digit IDs). Each ``measures/<name>`` array in the h5 companion is indexed by this order.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset', 'Phenotype']} })
    measures: Optional[list[str]] = Field(default=None, description="""Names of the measures present in the h5 companion (one ``measures/<name>`` dataset per name). Example: ``[g_factor, PMAT24_A_CR, PMAT24_A_RTCR, CardSort_Unadj, ProcSpeed_Unadj]``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Phenotype']} })
    measure_specs: Optional[list[MeasureSpec]] = Field(default=None, description="""Optional per-measure metadata (Cognitive Atlas IRIs, units, description). When present, each entry's ``name`` must match an entry in ``measures``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Phenotype']} })
    category: Optional[str] = Field(default="cognitive", description="""Coarse classification of the phenotype bundle, used for filtering in larger study libraries. Canonical values: ``cognitive`` (intelligence / memory / attention scores), ``clinical`` (UPDRS, MMSE, …), ``behavioral`` (raw task outputs), ``demographic``, ``physiological``, ``derived`` (composite scores like g-factor or brain-age). Free string — extend as needed.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Phenotype'], 'ifabsent': 'string(cognitive)'} })
    data_file: str = Field(default=..., description="""Path to the h5 companion (relative to the yaml).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Phenotype', 'Network']} })
    cohort: Optional[str] = Field(default=None, description="""Cohort label (e.g. 'HCPYA', 'PPMI').""", json_schema_extra = { "linkml_meta": {'domain_of': ['Phenotype', 'BidsEntities']} })
    provenance: Optional[Provenance] = Field(default=None, description="""W3C PROV-O metadata for the source of these scores.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Phenotype', 'Network']} })


class MeasureSpec(ConfiguredBaseModel):
    """
    Metadata for one phenotype measure. Optional per-measure entry on ``Phenotype.measure_specs``.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:MeasureSpec', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Must match the measure name in ``Phenotype.measures``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling']} })
    task_iri: Optional[str] = Field(default=None, description="""Cognitive Atlas (``cogat:``) IRI of the assessment instrument / task that produced this measure (e.g. ``cogat:trm_4a3fd79d09cba`` for PMAT24).""", json_schema_extra = { "linkml_meta": {'domain_of': ['MeasureSpec']} })
    concept_iri: Optional[str] = Field(default=None, description="""Cognitive Atlas IRI of the cognitive construct measured (e.g. ``cogat:trm_521ef89ce5e84`` for fluid intelligence).""", json_schema_extra = { "linkml_meta": {'domain_of': ['MeasureSpec']} })
    unit: Optional[str] = Field(default=None, description="""Unit annotation (e.g. 'count', 'ms', 'z-score').""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    measure_type: Optional[str] = Field(default=None, description="""Sub-type within the phenotype (e.g. 'raw', 'unadjusted', 'age-corrected', 'derived', 'RT', 'accuracy').""", json_schema_extra = { "linkml_meta": {'domain_of': ['MeasureSpec']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })


class NamedArray(ConfiguredBaseModel):
    """
    A named numeric array. Used as a sidecar slot value where a schema-typed object (e.g. ``ExperimentResult.parameters``) holds multiple arrays addressable by name (``w_LRE``, ``w_FFI``, ``J_i``, ...). The actual numeric data lives in the companion ``.h5`` at ``parameters/<name>``; the YAML carries only the descriptor.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:NamedArray', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Field name; matches the h5 dataset path under ``parameters/``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling']} })
    shape: Optional[list[int]] = Field(default=None, description="""Array dimensions.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'NamedArray', 'Parameter', 'FreeParameter', 'NDArray']} })
    dtype: Optional[str] = Field(default="float32", description="""Numpy dtype string.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'NamedArray', 'NDArray'],
         'ifabsent': 'string(float32)'} })
    unit: Optional[str] = Field(default=None, description="""Optional unit annotation (e.g. 'nA').""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })


class BidsEntities(ConfiguredBaseModel):
    """
    BIDS filename entities (BEP017-aligned) for provenance and data discovery. Reusable on Network, BrainAtlas, Tractogram, or any dataset with BIDS-conformant naming.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:BidsEntities', 'from_schema': 'https://w3id.org/tvbo'})

    template: Optional[str] = Field(default=None, description="""BIDS tpl- entity (e.g., FSLMNI152, MNI152NLin2009cAsym)""", json_schema_extra = { "linkml_meta": {'domain_of': ['BidsEntities']} })
    cohort: Optional[str] = Field(default=None, description="""BIDS cohort- entity (e.g., HCPYA, PPMI85)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Phenotype', 'BidsEntities']} })
    reconstruction: Optional[str] = Field(default=None, description="""BIDS rec- entity (e.g., dTOR)""", json_schema_extra = { "linkml_meta": {'domain_of': ['BidsEntities']} })
    segmentation: Optional[str] = Field(default=None, description="""BIDS seg- entity (e.g., ordered, ranked, 17Networks)""", json_schema_extra = { "linkml_meta": {'domain_of': ['BidsEntities']} })
    scale: Optional[str] = Field(default=None, description="""BIDS scale- entity (BEP017, e.g., 1000)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore', 'SimulationTool', 'BidsEntities']} })
    atlas: Optional[str] = Field(default=None, description="""BIDS atlas- entity (e.g., Schaefer2018, HCPMMP1)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation', 'BidsEntities']} })
    acquisition: Optional[str] = Field(default=None, description="""BIDS acq- entity (e.g., EEGstandard1005, MEGBrainstorm)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Tractogram', 'BidsEntities']} })
    hemi: Optional[str] = Field(default=None, description="""BIDS hemi- entity (L or R) for hemisphere-specific surface/volume data""", json_schema_extra = { "linkml_meta": {'domain_of': ['BidsEntities']} })


class Network(ConfiguredBaseModel):
    """
    Network specification with nodes, edges, and reusable coupling configurations. Supports both explicit node/edge representation and matrix-based connectivity (Connectome compatibility).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'aliases': ['Network', 'Connectome', 'Graph', 'Connectivity'],
         'class_uri': 'tvbo:Network',
         'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    nodes: Optional[list[Node]] = Field(default=None, description="""List of nodes with individual dynamics (optional, for heterogeneous networks)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    edges: Optional[list[Edge]] = Field(default=None, description="""List of directed edges with coupling references (optional, for explicit edge definition)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    primary_weight: Optional[str] = Field(default=None, description="""Name of the edge group that should serve as the canonical weights matrix when ``weights_matrix`` is queried. Lets a single Network sidecar bundle several connectivity variants (e.g. band-specific NMF reweightings, distance-modulated SC, shuffled controls) under different edge names while still presenting one of them as the active weight to consumers (simulators, observation pipelines). Defaults to ``weight`` / ``weights`` / ``sc`` lookup when not set.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    coupling: Optional[dict[str, Coupling]] = Field(default=None, description="""Reusable coupling configurations referenced by edges (e.g., 'instant', 'delayed', 'inhibitory')""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network', 'Edge', 'SimulationExperiment']} })
    dynamics: Optional[dict[str, Dynamics]] = Field(default=None, description="""Dictionary of dynamics models keyed by name. Nodes reference these by name. For heterogeneous networks with per-node dynamics.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network',
                       'Node',
                       'Edge',
                       'Continuation',
                       'SimulationExperiment']} })
    node_template: Optional[Node] = Field(default=None, description="""Default Node attributes applied to every Node in this Network. Explicit entries in `nodes:` override the template field-by-field (shallow replace, same semantics as `dict.update`). Reserved per-Node slots (id, label, position, region) come from the data backbone (Network.iri-loaded parcellation or procedural generator) and are ignored if set on the template. Lets a homogeneous Network declare its per-Node configuration once without enumerating regions.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    edge_template: Optional[Edge] = Field(default=None, description="""Default Edge attributes applied to every Edge in this Network. Explicit entries in `edges:` override the template field-by-field. Reserved per-Edge slots (source, target, weight, delay, distance) come from the data backbone (loaded matrix or procedural generator) and are ignored if set on the template.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    number_of_nodes: Optional[int] = Field(default=1, description="""Number of nodes in the network (derived from nodes if not set)""", json_schema_extra = { "linkml_meta": {'aliases': ['number_of_nodes', 'number_of_regions'],
         'domain_of': ['Network'],
         'ifabsent': 'integer(1)'} })
    coordinate_space: Optional[CommonCoordinateSpace] = Field(default=None, description="""Coordinate space for node positions (e.g., MNI152NLin2009c). Mirrors BrainAtlas.coordinateSpace so network node positions are unambiguous.""", json_schema_extra = { "linkml_meta": {'domain_of': ['DBSDataset',
                       'DBSSubject',
                       'Electrode',
                       'EField',
                       'Network',
                       'SpatialDomain',
                       'Mesh']} })
    parcellation: Optional[Parcellation] = Field(default=None, description="""Brain parcellation/atlas reference""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network', 'Mesh']} })
    tractogram: Optional[Tractogram] = Field(default=None, description="""Reference to tractography data""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    mesh: Optional[Mesh] = Field(default=None, description="""Optional triangle mesh geometry. When set, the Network carries triangle faces (and optionally normals, curvature) alongside its node coordinates. Used by surface-based observations (cortical wave detection, Helmholtz-Hodge decomposition, spectrospatial mode analysis). Mesh face data lives in the same h5 companion under a ``mesh/`` group.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network', 'SpatialField', 'FieldStateVariable', 'PDE']} })
    transforms: Optional[list[Function]] = Field(default=None, description="""Ordered list of transforms applied to edge property matrices. Each Function's name identifies the target edge property (e.g. 'weight', 'length'). Supports equation-based (symbolic) or callable-based (software) transforms. Multiple transforms on the same target are applied sequentially.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    data_file: Optional[str] = Field(default=None, description="""Path to companion data file. Supported extensions: .h5 (HDF5), .zarr/ (Zarr), .csv (legacy single-matrix). Null if no companion data needed.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Phenotype', 'Network']} })
    descriptor: Optional[str] = Field(default=None, description="""Short alphanumeric identifier for the BIDS desc- filename entity (e.g., SC, FC, EC, SCFC). Classifies the connectivity modality of the network's edge measures.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    bids_dir: Optional[str] = Field(default=None, description="""Path to BEP017-compliant BIDS directory for loading connectivity matrices""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    bids: Optional[BidsEntities] = Field(default=None, description="""BIDS filename entities for this dataset""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    structural_measures: Optional[list[str]] = Field(default=None, description="""BEP017 measure names for structural connectivity (e.g., streamlineCount, tractLength)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    observational_measures: Optional[list[str]] = Field(default=None, description="""BEP017 measure names for observational targets (e.g., BoldCorrelation)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    provenance: Optional[Provenance] = Field(default=None, description="""W3C PROV-O aligned provenance""", json_schema_extra = { "linkml_meta": {'domain_of': ['Phenotype', 'Network']} })
    parent_network: Optional[str] = Field(default=None, description="""Path/URI to parent (coarser) Network. When set, this network is a refinement where each node maps to exactly one parent node via node_mapping.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    node_mapping: Optional[str] = Field(default=None, description="""HDF5 dataset path for node-to-parent mapping. Int32 array of shape (N,) where entry i is the parent node ID. Required when parent_network is set.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    distance_unit: Optional[UnitEnum] = Field(default=UnitEnum.mm, description="""Unit for distances/lengths in the network""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network'], 'ifabsent': 'string(mm)'} })
    time_unit: Optional[UnitEnum] = Field(default=UnitEnum.ms, description="""Default time unit for the network""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network'], 'ifabsent': 'string(ms)'} })
    edge_matrix_files: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    graph_generator: Optional[GraphGenerator] = Field(default=None, description="""Graph generator specification.  When set, overrides explicit edges/nodes for graph construction.  The type field is a free string; StandardGraphType lists well-known types that get automatic code generation across backends.
""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })


class GraphGenerator(ConfiguredBaseModel):
    """
    Backend-agnostic graph generator specification.  Captures the mathematical family (type) and its parameters so that each backend can emit the correct constructor call (Graphs.jl, NetworkX, etc.). The number of nodes is always taken from Network.number_of_nodes.

    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:GraphGenerator', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    type: str = Field(default=..., description="""Graph family name.  Use a StandardGraphType value for automatic backend mapping, or any custom string for documentation purposes.
""", json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator',
                       'File',
                       'Aggregation',
                       'TuningObjective',
                       'Algorithm']} })
    seed: Optional[int] = Field(default=None, description="""Random seed for reproducible graph generation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator', 'Distribution', 'Noise']} })
    directed: Optional[bool] = Field(default=False, description="""Whether to generate a directed graph.""", json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator', 'Edge'], 'ifabsent': 'boolean(false)'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, description="""Generator parameters (e.g. k, p, dims).  Names are matched by the backend mapping to construct the call.
""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    builder: Optional[Callable] = Field(default=None, description="""Optional Python callable that builds the network at YAML load time. When set, ``Network._resolve`` imports ``<module>.<name>`` and invokes it with the ``parameters`` block as keyword arguments. The callable must return either a ``Network`` instance or a tuple ``(weights, lengths, node_params)``. Use the existing ``Callable`` slots (``name`` is the function name, ``module`` is its dotted module path). Reuses the same idiom as TVB monitor class references; no free ``module:function`` strings.
""", json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator']} })


class File(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:File', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    type: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator',
                       'File',
                       'Aggregation',
                       'TuningObjective',
                       'Algorithm']} })
    path: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['File', 'DataSource']} })
    extension: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['File']} })


class Reference(ConfiguredBaseModel):
    """
    A small typed pointer to another TVBO entity (Network, Mesh, Observation, …). The ``iri`` identifies the target via the registry; the optional ``field`` is a dotted-path subkey resolved by attribute walk on the loaded target (e.g. ``field: 'weight_alpha'`` picks the ``weight_alpha`` named edge matrix on a Network; ``field: 'mesh.faces'`` picks the mesh face array). Used uniformly anywhere a TVBO entity needs to point at a sub-array of another entity without inlining the data.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Reference', 'from_schema': 'https://w3id.org/tvbo'})

    iri: str = Field(default=..., description="""CURIE or full IRI of the referenced TVBO entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation',
                       'Tractogram',
                       'ReferenceFingerprint',
                       'Reference',
                       'Dynamics',
                       'Function',
                       'Coupling']} })
    field: Optional[str] = Field(default=None, description="""Optional dotted-path subkey into the referenced entity (e.g. ``weight_alpha`` for a Network's named edge matrix, ``mesh.faces`` for the face array, or ``nodes.ef_alpha`` for a per-node scalar field). Resolved by attribute walk at load time.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceFingerprint', 'Reference', 'PDE']} })


class Node(ConfiguredBaseModel):
    """
    A node in a network with its own dynamics and properties
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Node',
         'from_schema': 'https://w3id.org/tvbo',
         'slot_usage': {'record': {'ifabsent': 'True', 'name': 'record'}}})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    record: Optional[bool] = Field(default=True, description="""Whether to include this element in simulation output files. Applicable to state variables (default true), derived variables (default false), and network nodes (default true). Set false to suppress recording.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'StateVariable', 'DerivedVariable'], 'ifabsent': 'True'} })
    id: int = Field(default=..., description="""Unique node identifier""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'SimulationExperiment'],
         'slot_uri': 'dcterms:identifier'} })
    dynamics: Optional[str] = Field(default=None, description="""Dynamics model governing this node's behavior. Can be a reference (by name) or inline definition. If not provided, uses experiment's dynamics.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network',
                       'Node',
                       'Edge',
                       'Continuation',
                       'SimulationExperiment']} })
    position: Optional[Coordinate] = Field(default=None, description="""Spatial coordinates (x, y, z) of the node""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node']} })
    region: Optional[str] = Field(default=None, description="""Brain region or anatomical label""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'SpatialDomain']} })
    state: Optional[dict[str, StateValue]] = Field(default=None, description="""Per-node initial state variable values, keyed by state variable name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node']} })
    events: Optional[dict[str, Event]] = Field(default=None, description="""Events attached to this node (e.g., threshold-based state changes).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'Edge', 'Dynamics', 'SimulationExperiment']} })
    subnetwork: Optional[Network] = Field(default=None, description="""Optional finer-scale Network that \"lives inside\" this Node (network-of-networks / multi-scale primitive). The subnetwork's nodes map (one-to-many) to this single parent Node. Cross-scale signal flow is declared as ordinary Edges in the subnetwork's `edges` list, using `source_network` / `target_network` with the `parent` sentinel to traverse up the hierarchy. The subnetwork may carry its own `graph_generator` (discrete sub-network — reservoir, spiking population, jaxley compartments) and/or `mesh` (continuous surface field) — one unified slot for reservoirs-per-region, spiking-populations-per-region, cells-per-region, and surface-mesh-patches-per-region. Recursive: a subnetwork's Nodes can themselves carry subnetworks.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node']} })


class StateValue(ConfiguredBaseModel):
    """
    A named state variable value for per-node initialization.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:StateValue', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    value: Optional[Any] = Field(default=None, description="""Numeric, string, or boolean value. ScalarValue accepts any literal primitive type, allowing parameters to carry control flags (e.g., booleans) or symbolic placeholders alongside numeric defaults.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateValue',
                       'Parameter',
                       'Argument',
                       'Option',
                       'BoundaryCondition'],
         'slot_uri': 'schema:value'} })


class Edge(ConfiguredBaseModel):
    """
    An edge in a network. Two modes: explicit (source+target set, scalar parameters in YAML) or template (no source/target, N×N matrix measure in HDF5). Both coexist in the same edges list.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Edge', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    source: Optional[int] = Field(default=None, description="""Source node ID (set for explicit edges, absent for template edges)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge', 'Observation', 'Dynamics', 'Parameter', 'CouplingInput']} })
    target: Optional[int] = Field(default=None, description="""Target node ID (set for explicit edges, absent for template edges)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    weight: Optional[float] = Field(default=None, description="""Connection weight (explicit edges)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    delay: Optional[float] = Field(default=None, description="""Conduction delay (explicit edges, ms)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    distance: Optional[float] = Field(default=None, description="""Edge length / tract distance (explicit edges, mm)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    unit: Optional[str] = Field(default=None, description="""Unit for matrix values (template edges only)""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    format: Optional[SparseFormat] = Field(default=None, description="""Storage format in HDF5 (template edges only)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'Edge', 'DataSource']} })
    weighted: Optional[bool] = Field(default=True, description="""Matrix entries carry weights (not just 0/1)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge'], 'ifabsent': 'boolean(true)'} })
    valid_diagonal: Optional[bool] = Field(default=False, description="""Self-connections are meaningful""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge'], 'ifabsent': 'boolean(false)'} })
    non_negative: Optional[bool] = Field(default=True, description="""All values >= 0""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge'], 'ifabsent': 'boolean(true)'} })
    source_var: Optional[str] = Field(default=None, description="""Output variable from source node to use (e.g., 'x_out'). If not specified, uses first output variable from source dynamics.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    target_var: Optional[str] = Field(default=None, description="""Input variable on target node to connect to (e.g., 'c_in'). If not specified, uses first coupling input from target dynamics.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    coupling: Optional[str] = Field(default=None, description="""Coupling function for this edge. Can be a reference (by name) to coupling or inline definition. If not provided, uses experiment's default coupling.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network', 'Edge', 'SimulationExperiment']} })
    directed: Optional[bool] = Field(default=False, description="""Whether the edge is directed. If false, represents a symmetric/bidirectional connection.""", json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator', 'Edge'], 'ifabsent': 'False'} })
    source_network: Optional[str] = Field(default=None, description="""Path or name of the Network whose nodes define the source endpoints of this Edge. Symmetric counterpart to `target_network`. Together they let an Edge bridge any two Networks: both unset (within-Network edge), one set (this Network ↔ the named one), or both set (peer-to-peer projection between two other Networks). Accepts either an absolute IRI (peer Network) or the relative-path sentinel `parent` / `parent/parent` (cross-scale up/down via the Node.subnetwork hierarchy).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    target_network: Optional[str] = Field(default=None, description="""Path or name of the Network whose nodes define the target endpoints (e.g. the columns of a non-square projection matrix such as a gain matrix region → EEG sensors). Also accepts the relative-path sentinel `parent` / `parent/parent` for cross-scale signal flow up the Node.subnetwork hierarchy. One mechanism handles both projection matrices (absolute IRI) and multi-scale boundaries (sentinel).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    dimension_labels: Optional[list[str]] = Field(default=None, description="""Ordered labels for the matrix columns (dim-1) when the matrix is non-square.  Row labels (dim-0) are the parent Network's node labels.  Stored as HDF5 dimension scales in the companion file.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    dynamics: Optional[str] = Field(default=None, description="""Dynamics model for this edge. When specified, the edge has its own state variables and ODE (EdgeModel with f in ND.jl). Uses the same Dynamics class as nodes — state_variables define edge states, derived_variables define observables, output defines what is visible for plotting/analysis. The coupling_function on Coupling still defines how vertex outputs map to edge outputs for aggregation at vertices.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network',
                       'Node',
                       'Edge',
                       'Continuation',
                       'SimulationExperiment']} })
    events: Optional[dict[str, Event]] = Field(default=None, description="""Events attached to this edge (e.g., threshold-based line tripping).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'Edge', 'Dynamics', 'SimulationExperiment']} })


class Observation(ConfiguredBaseModel):
    """
    Unified class for all observation/measurement specifications. Covers monitors (BOLD, EEG), tuning observables, and derived quantities. Pipeline is a sequence of Functions with input -> output flow.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Observation', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    acronym: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'Observation',
                       'Function',
                       'FunctionCall'],
         'slot_uri': 'skos:notation'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    environment: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'SimulationExperiment', 'PDESolver']} })
    time_scale: Optional[UnitEnum] = Field(default=UnitEnum.ms, description="""Time unit for the integration / simulation. Determines the physical time meaning of one model time-step.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'Integrator'], 'ifabsent': 'ms'} })
    source: Optional[list[str]] = Field(default=None, description="""Ordered list of inputs this observation derives from. Each entry is either a StateVariable reference (raw observation of an integrated trajectory, e.g. ``S_e``) or an Observation reference (derived observation, e.g. ``bold``). Codegen checks each source name against ``experiment.observations``; names matching an existing observation flag the parent as derived.""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'StateVariable'}, {'range': 'Observation'}],
         'domain_of': ['Edge', 'Observation', 'Dynamics', 'Parameter', 'CouplingInput']} })
    aux_data: Optional[list[Reference]] = Field(default=None, description="""Ordered list of auxiliary inputs the observation pipeline consumes. Examples: a Surface mesh's faces array, an empirical FC matrix on a reference Network, or another Observation's output. Resolved load-time eagerly; rendered code is self-contained.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    period: Optional[float] = Field(default=None, description="""Sampling period for monitors (ms). For BOLD: TR in ms.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    downsample_period: Optional[float] = Field(default=None, description="""Intermediate downsampling period (ms). For BOLD: typically matches dt.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    voi: Optional[int] = Field(default=None, description="""Variable of interest index (which state variable to monitor). Default: 0.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    imaging_modality: Optional[ImagingModality] = Field(default=None, description="""Type of imaging modality (BOLD, EEG, MEG, etc.)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    warmup_source: Optional[str] = Field(default=None, description="""Reference to transient simulation result for history initialization (e.g., 'result_init').""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'ClassReference']} })
    data_source: Optional[DataSource] = Field(default=None, description="""Load data from external source (file, database, API). When specified, this observation represents empirical/external data rather than simulated data. Enables unified treatment of all data.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation', 'Tractogram', 'Observation']} })
    skip_t: Optional[int] = Field(default=None, description="""Number of samples to skip at the start (transient removal). For FC: typically 10-20 TRs.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    tail_samples: Optional[int] = Field(default=None, description="""Number of samples from the end to use. Takes the last N samples before aggregation. E.g., tail_samples: 500 means use data[-500:].""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    aggregation: Optional[AggregationType] = Field(default=None, description="""How to aggregate over time""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'Coupling']} })
    window_size: Optional[int] = Field(default=None, description="""Number of samples for windowed aggregation""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    pipeline: Optional[list[FunctionCall]] = Field(default=None, description="""Ordered sequence of Functions. Each step has a unique `name` (used to key step outputs) and transforms input -> output. List form preserves execution order.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    class_reference: Optional[ClassReference] = Field(default=None, description="""Direct class reference (alternative to pipeline). Use for external library classes like tvboptim.Bold, custom monitors, or any callable class. The class is instantiated with constructor_args and called with call_args. Example: {name: Bold, module: tvboptim.observations.tvb_monitors.bold, constructor_args: [{name: period, value: 1000.0}]}""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })


class Dynamics(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'aliases': ['NeuralMassModel'],
         'class_uri': 'tvbo:Dynamics',
         'comments': ['Successor class replacing deprecated NeuralMassModel.'],
         'from_schema': 'https://w3id.org/tvbo',
         'slot_usage': {'name': {'ifabsent': 'Dynamics', 'name': 'name'},
                        'system_type': {'ifabsent': 'continuous',
                                        'name': 'system_type'}}})

    has_reference: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics'], 'slot_uri': 'dcterms:references'} })
    name: str = Field(default="Dynamics", description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'ifabsent': 'Dynamics',
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    iri: Optional[str] = Field(default=None, description="""Optional stable IRI (or compact URI) for this entity in an external ontology or knowledge base. Used to load metadata from an external source; not required when the entity is fully self-contained (equations, parameters, etc. defined in the file itself).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation',
                       'Tractogram',
                       'ReferenceFingerprint',
                       'Reference',
                       'Dynamics',
                       'Function',
                       'Coupling'],
         'slot_uri': 'dcterms:identifier'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    source: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Edge', 'Observation', 'Dynamics', 'Parameter', 'CouplingInput'],
         'slot_uri': 'dcterms:source'} })
    references: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance',
                       'Dynamics',
                       'SimulationExperiment',
                       'SimulationStudy'],
         'slot_uri': 'dcterms:references'} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })
    derived_parameters: Optional[dict[str, DerivedParameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    derived_variables: Optional[dict[str, DerivedVariable]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    coupling_terms: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    coupling_inputs: Optional[dict[str, CouplingInput]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    state_variables: Optional[dict[str, StateVariable]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    modified: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    output: Optional[list[str]] = Field(default=None, description="""Output variable names to include in simulation results. References to state_variables or derived_variables by name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function', 'FunctionCall']} })
    derived_from_model: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    number_of_modes: Optional[int] = Field(default=1, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics'], 'ifabsent': 'integer(1)'} })
    local_coupling_term: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    functions: Optional[dict[str, Function]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Algorithm', 'SimulationExperiment', 'PDE']} })
    stimulus: Optional[Stimulus] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    modes: Optional[dict[str, Dynamics]] = Field(default=None, json_schema_extra = { "linkml_meta": {'aliases': ['components'], 'domain_of': ['Dynamics']} })
    model_type: Optional[ModelType] = Field(default=None, description="""Coarse classification of this model (mean_field, neural_mass, phase_oscillator, phenomenological, spiking, generic, field). Used for filtering in Dynamics.list_db(model_type=...).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    system_type: Optional[SystemType] = Field(default=SystemType.continuous, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics'], 'ifabsent': 'continuous'} })
    autonomous: Optional[bool] = Field(default=True, description="""Whether the system is autonomous (equations do not depend explicitly on time t). Non-autonomous systems have explicit time dependence, e.g. f*cos(omega*t).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics'], 'ifabsent': 'true'} })
    observed: Optional[dict[str, DerivedVariable]] = Field(default=None, description="""Observable functions computed from states, inputs, and parameters after simulation. Unlike derived_variables (which are intermediate algebraic expressions used within the ODE), observed variables are post-hoc quantities recoverable from the solution. Maps to obsf/obssym in ND.jl EdgeModel/VertexModel. Example: absolute force magnitude computed from force components.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Coupling']} })
    events: Optional[dict[str, Event]] = Field(default=None, description="""Discrete state transitions intrinsic to the dynamical system, such as threshold-triggered resets in spiking neuron models. Unlike experiment-level events (stimulation, perturbation), these define the model's own discontinuous behavior.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'Edge', 'Dynamics', 'SimulationExperiment']} })


class StateVariable(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:StateVariable',
         'from_schema': 'https://w3id.org/tvbo',
         'slot_usage': {'record': {'ifabsent': 'True', 'name': 'record'}}})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    symbol: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable',
                       'Parameter',
                       'DerivedParameter',
                       'DerivedVariable']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator'],
         'slot_uri': 'skos:definition'} })
    domain: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'ExplorationAxis',
                       'FreeParameter',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    unit: Optional[UnitEnum] = Field(default=None, description="""Physical unit of measurement. Values are drawn from the QUDT ontology (http://qudt.org/vocab/unit/) with UO cross-references where available.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField'],
         'slot_uri': 'qudt:unit'} })
    record: Optional[bool] = Field(default=True, description="""Whether to include this element in simulation output files. Applicable to state variables (default true), derived variables (default false), and network nodes (default true). Set false to suppress recording.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'StateVariable', 'DerivedVariable'], 'ifabsent': 'True'} })
    grounding: Optional[list[str]] = Field(default=None, description="""External ontology IRIs (typically GO, ChEBI, UBERON, CL, MeSH) that this entity is a surrogate / abstraction / model of. Replaces the legacy OWL pattern `tvbo:surrogate_of` by carrying the link inline with the YAML data instance. Multiple IRIs allowed: a single parameter may abstract several biological processes (e.g. a synaptic conductance grounding both GO:0060079 (excitatory PSP) and GO:0007268 (chemical synaptic transmission)).""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Parameter', 'DerivedVariable'],
         'slot_uri': 'oboInOwl:hasDbXref'} })
    variable_of_interest: Optional[bool] = Field(default=True, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'True'} })
    coupling_variable: Optional[bool] = Field(default=False, description="""Whether this state variable is transmitted to connected nodes through the coupling function. In TVB terms, this determines the cvar indices (state variables extracted from history and fed into the coupling function). The coupling function may override this via its incoming_states attribute.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'False'} })
    equation_type: Optional[str] = Field(default="differential", description="""Type of equation: 'differential' (default) means dx/dt = rhs, 'algebraic' means 0 = rhs or x ~ rhs (DAE constraint). Algebraic equations are used by ModelingToolkit.jl backend.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'string(differential)'} })
    equation_order: Optional[int] = Field(default=1, description="""Order of the time derivative on the LHS. Default 1 means dx/dt = rhs (first-order ODE). Order 2 means d²x/dt² = rhs (second-order ODE), etc. Higher-order ODEs are automatically lowered to coupled first-order systems by backends like ModelingToolkit.jl via mtkcompile.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'int(1)'} })
    noise: Optional[Noise] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'StateVariable', 'Integrator']} })
    stimulation_variable: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })
    boundaries: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })
    initial_value: Optional[float] = Field(default=0.1, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'FreeParameter', 'SpatialField'],
         'ifabsent': 'float(0.1)'} })
    derivative_initial_value: Optional[float] = Field(default=None, description="""Initial value for the first time derivative, used when equation_order > 1. For a second-order ODE d²x/dt² = f, this sets dx/dt(0). Required by ModelingToolkit.jl to fully specify higher-order initial value problems.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })
    distribution: Optional[Distribution] = Field(default=None, description="""Distribution for sampling initial conditions per node. If present, initial_value is used as fallback/mean.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Parameter', 'Noise', 'Coupling']} })
    history: Optional[TimeSeries] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })


class Distribution(ConfiguredBaseModel):
    """
    A probability distribution for sampling parameters or initial conditions. Standard distributions (Uniform, Gaussian) are specified by name and domain/parameters. Custom distributions use a Function for the PDF/sampling rule. Default name is Uniform when only domain is given.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Distribution',
         'from_schema': 'https://w3id.org/tvbo',
         'slot_usage': {'name': {'ifabsent': 'string(Uniform)', 'name': 'name'}}})

    name: str = Field(default="Uniform", description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'ifabsent': 'string(Uniform)',
         'slot_uri': 'schema:name'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    domain: Optional[Range] = Field(default=None, description="""Support of the distribution (sampling bounds). For Uniform this fully defines the distribution.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'ExplorationAxis',
                       'FreeParameter',
                       'PDE']} })
    function: Optional[Function] = Field(default=None, description="""Custom distribution function (PDF or sampling callable). Only needed for non-standard distributions.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Distribution', 'FunctionCall', 'Noise']} })
    seed: Optional[int] = Field(default=None, description="""Random seed for reproducible sampling.""", json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator', 'Distribution', 'Noise']} })
    axis: Optional[SamplingAxis] = Field(default=SamplingAxis.space, description="""Dimension along which the distribution is sampled. 'space' = per-node (default), 'time' = per-timestep (stochastic input).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Distribution'], 'ifabsent': 'space'} })
    correlation: Optional[Matrix] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Distribution']} })


class Parameter(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Parameter', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    symbol: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable',
                       'Parameter',
                       'DerivedParameter',
                       'DerivedVariable']} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator'],
         'slot_uri': 'skos:definition'} })
    value: Optional[Any] = Field(default=None, description="""Numeric, string, or boolean value. ScalarValue accepts any literal primitive type, allowing parameters to carry control flags (e.g., booleans) or symbolic placeholders alongside numeric defaults.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateValue',
                       'Parameter',
                       'Argument',
                       'Option',
                       'BoundaryCondition'],
         'slot_uri': 'schema:value'} })
    default: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter'], 'slot_uri': 'schema:defaultValue'} })
    domain: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'ExplorationAxis',
                       'FreeParameter',
                       'PDE']} })
    reported_optimum: Optional[float] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    unit: Optional[UnitEnum] = Field(default=None, description="""Physical unit of measurement. Values are drawn from the QUDT ontology (http://qudt.org/vocab/unit/) with UO cross-references where available.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField'],
         'slot_uri': 'qudt:unit'} })
    dataset_path: Optional[str] = Field(default=None, description="""Dataset path for array-valued parameters. When set, the parameter value is stored in the binary companion file (HDF5 or Zarr) at this path. The value slot is omitted.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    grounding: Optional[list[str]] = Field(default=None, description="""External ontology IRIs (typically GO, ChEBI, UBERON, CL, MeSH) that this entity is a surrogate / abstraction / model of. Replaces the legacy OWL pattern `tvbo:surrogate_of` by carrying the link inline with the YAML data instance. Multiple IRIs allowed: a single parameter may abstract several biological processes (e.g. a synaptic conductance grounding both GO:0060079 (excitatory PSP) and GO:0007268 (chemical synaptic transmission)).""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Parameter', 'DerivedVariable'],
         'slot_uri': 'oboInOwl:hasDbXref'} })
    comment: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter'], 'slot_uri': 'rdfs:comment'} })
    heterogeneous: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'FreeParameter']} })
    distribution: Optional[Distribution] = Field(default=None, description="""Distribution for heterogeneous per-node parameter sampling. Implies heterogeneous=true.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Parameter', 'Noise', 'Coupling']} })
    source: Optional[str] = Field(default=None, description="""Data source for this parameter's value. When set, the value is loaded from the referenced entity rather than being a YAML literal. The referent is typically a Network with per-node parameters (dscalar pattern) or a flat dataset (HDF5, TSV). Combine with `measure:` when the source exposes multiple named measures. Distinct from the global `iri:` slot, which is reserved for ontology grounding.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge', 'Observation', 'Dynamics', 'Parameter', 'CouplingInput']} })
    measure: Optional[str] = Field(default=None, description="""Selector into the source. When `source` points at a Network with per-node parameters (or a dscalar with multiple maps), picks which named measure to load. Aligns with names listed in Network.structural_measures / Network.observational_measures. Ignored when the source resolves to a scalar/array dataset.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    free: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    shape: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'NamedArray', 'Parameter', 'FreeParameter', 'NDArray']} })
    explored_values: Optional[AnyShapeArray[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Range', 'Parameter', 'ExplorationAxis']} })
    element_domains: Optional[list[Range]] = Field(default=None, description="""Per-element domain overrides for heterogeneous parameters. When specified, element_domains[i] overrides domain for element i during exploration auto-expansion. Length must match parameter shape (e.g., n_nodes for shape \"(n_nodes,)\"). If not set, all elements share the same domain.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'ExplorationAxis']} })


class CouplingInput(ConfiguredBaseModel):
    """
    Specification of a coupling input channel for multi-coupling dynamics
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:CouplingInput', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    source: Optional[str] = Field(default=None, description="""Name of the coupling function that feeds this input. When omitted, resolved automatically: (1) same name as a coupling function → direct match, (2) single coupling input and single coupling function → auto-mapped, (3) multiple of each → positional order.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge', 'Observation', 'Dynamics', 'Parameter', 'CouplingInput']} })
    dimension: Optional[int] = Field(default=1, description="""Dimensionality of the coupling input (number of coupled values)""", json_schema_extra = { "linkml_meta": {'domain_of': ['CouplingInput'], 'ifabsent': 'integer(1)'} })
    keys: Optional[list[str]] = Field(default=None, description="""Named keys for multi-dimensional coupling. When dimension > 1, provides symbolic names for each index (e.g., keys: [lre, ffi] for dimension: 2). Used in equations as variable names.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CouplingInput']} })


class Argument(ConfiguredBaseModel):
    """
    A function argument with explicit value specification. Value can be: literal (number/string), reference to input (input.key), or cross-observation reference (observation_name.output_key).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Argument', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    value: Optional[Any] = Field(default=None, description="""Argument value. Can be: - Literal: 1.0, \"string\", boolean, etc. - Input reference: \"input.frequencies\" (from source_observation outputs) - Cross-observation: \"target_frequencies.peak_freqs\" (from another observation)""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateValue',
                       'Parameter',
                       'Argument',
                       'Option',
                       'BoundaryCondition']} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })


class Function(ConfiguredBaseModel):
    """
    A function with explicit input -> transformation -> output flow. Can be equation-based (symbolic) or software-based (callable). In a pipeline, functions are chained: output of one becomes input of next.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Function', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    acronym: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'Observation',
                       'Function',
                       'FunctionCall'],
         'slot_uri': 'skos:notation'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    iri: Optional[str] = Field(default=None, description="""Optional stable IRI (or compact URI) for this entity in an external ontology or knowledge base. Used to load metadata from an external source; not required when the entity is fully self-contained (equations, parameters, etc. defined in the file itself).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation',
                       'Tractogram',
                       'ReferenceFingerprint',
                       'Reference',
                       'Dynamics',
                       'Function',
                       'Coupling'],
         'slot_uri': 'dcterms:identifier'} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator'],
         'slot_uri': 'skos:definition'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    requirements: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment', 'Function', 'PDESolver']} })
    input: Optional[str] = Field(default=None, description="""Simple input reference: name of previous function's output in pipeline. For multi-argument functions, use arguments with value references instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    output: Optional[str] = Field(default=None, description="""Name for this function's output (referenced by subsequent functions)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function', 'FunctionCall']} })
    arguments: Optional[list[Argument]] = Field(default=None, description="""Variables consumed by the function (referenced in the equation). Each argument has a name and optional metadata (description, default value, unit).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall', 'AlgorithmInclude']} })
    output_equation: Optional[Equation] = Field(default=None, description="""Output transformation equation (if equation-based)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function']} })
    source_code: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    callable: Optional[Callable] = Field(default=None, description="""Software implementation reference (if software-based)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    apply_on_dimension: Optional[DimensionType] = Field(default=None, description="""Which dimension to apply the transformation on""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    aggregate: Optional[Aggregation] = Field(default=None, description="""How to aggregate the result across dimensions. E.g., aggregate.over=node computes per-row (per-node) with keepdims. The type field controls whether to reduce (mean/sum) or keep dimensions (none).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'LossFunction', 'FunctionCall']} })
    time_range: Optional[Range] = Field(default=None, description="""Time range for generated TimeSeries (for kernel generators). Equation is evaluated at each time point.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })


class Aggregation(ConfiguredBaseModel):
    """
    Specifies how to aggregate values across a dimension. Used for loss functions to define per-element loss with reduction.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Aggregation', 'from_schema': 'https://w3id.org/tvbo'})

    over: Optional[DimensionType] = Field(default=None, description="""Dimension to aggregate over (e.g., node, time, state)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Aggregation']} })
    type: Optional[ReductionType] = Field(default=ReductionType.mean, description="""Aggregation operation (mean, sum, max, min, none)""", json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator',
                       'File',
                       'Aggregation',
                       'TuningObjective',
                       'Algorithm'],
         'ifabsent': 'string(mean)'} })


class LossFunction(Function):
    """
    A loss function for optimization with optional aggregation. Extends Function with aggregation specification for per-element losses.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:LossFunction', 'from_schema': 'https://w3id.org/tvbo'})

    aggregate: Optional[Aggregation] = Field(default=None, description="""How to aggregate the loss across dimensions. Example: aggregate.over=node, aggregate.type=mean computes loss per node, then averages.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'LossFunction', 'FunctionCall']} })
    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    acronym: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'Observation',
                       'Function',
                       'FunctionCall'],
         'slot_uri': 'skos:notation'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    iri: Optional[str] = Field(default=None, description="""Optional stable IRI (or compact URI) for this entity in an external ontology or knowledge base. Used to load metadata from an external source; not required when the entity is fully self-contained (equations, parameters, etc. defined in the file itself).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation',
                       'Tractogram',
                       'ReferenceFingerprint',
                       'Reference',
                       'Dynamics',
                       'Function',
                       'Coupling'],
         'slot_uri': 'dcterms:identifier'} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator'],
         'slot_uri': 'skos:definition'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    requirements: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment', 'Function', 'PDESolver']} })
    input: Optional[str] = Field(default=None, description="""Simple input reference: name of previous function's output in pipeline. For multi-argument functions, use arguments with value references instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    output: Optional[str] = Field(default=None, description="""Name for this function's output (referenced by subsequent functions)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function', 'FunctionCall']} })
    arguments: Optional[list[Argument]] = Field(default=None, description="""Variables consumed by the function (referenced in the equation). Each argument has a name and optional metadata (description, default value, unit).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall', 'AlgorithmInclude']} })
    output_equation: Optional[Equation] = Field(default=None, description="""Output transformation equation (if equation-based)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function']} })
    source_code: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    callable: Optional[Callable] = Field(default=None, description="""Software implementation reference (if software-based)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    apply_on_dimension: Optional[DimensionType] = Field(default=None, description="""Which dimension to apply the transformation on""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    time_range: Optional[Range] = Field(default=None, description="""Time range for generated TimeSeries (for kernel generators). Equation is evaluated at each time point.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })


class FunctionCall(ConfiguredBaseModel):
    """
    Invocation of a function in a pipeline. Can reference a defined Function by name, OR inline a callable directly for external library functions, OR inline an equation, OR use class_call for class instantiation. Mirrors Function attributes so pipeline steps can be self-contained. The `name` is an optional step label (used in pipelines for keyed access to step outputs); it is NOT a global identifier (singleton uses like `loss`, `observable` may omit it).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:FunctionCall', 'from_schema': 'https://w3id.org/tvbo'})

    acronym: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'Observation',
                       'Function',
                       'FunctionCall'],
         'slot_uri': 'skos:notation'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    name: Optional[str] = Field(default=None, description="""Optional step label; used in pipelines to key step outputs.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    function: Optional[str] = Field(default=None, description="""Reference to a defined Function (by name)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Distribution', 'FunctionCall', 'Noise']} })
    callable: Optional[Callable] = Field(default=None, description="""Direct callable specification (alternative to function reference)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    class_call: Optional[ClassReference] = Field(default=None, description="""Class instantiation and call (alternative to callable/function). Use for external library classes that need __init__ then __call__. Example: Bold monitor from tvboptim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['FunctionCall']} })
    input: Optional[str] = Field(default=None, description="""Reference to previous function's output in pipeline (by name)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    output: Optional[str] = Field(default=None, description="""Name for this step's output (referenced by subsequent functions)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function', 'FunctionCall']} })
    apply_on_dimension: Optional[DimensionType] = Field(default=None, description="""Dimension to apply function over (generates vmap in code). E.g., 'node' applies per-node.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    aggregate: Optional[Aggregation] = Field(default=None, description="""How to aggregate the result across dimensions. Example: aggregate.over=node, aggregate.type=mean applies function per node, then averages. Used in loss functions.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'LossFunction', 'FunctionCall']} })
    arguments: Optional[list[Argument]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall', 'AlgorithmInclude']} })
    time_range: Optional[Range] = Field(default=None, description="""Time range for generated TimeSeries (for kernel generators)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    source_code: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })


class Callable(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Callable', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    module: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Callable']} })
    software: Optional[SoftwareRequirement] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Callable', 'Continuation', 'SimulationExperiment']} })


class ClassReference(Callable):
    """
    Reference to a class that can be instantiated and called. Used for external library classes (e.g., tvboptim.Bold, custom monitors). The class is instantiated with constructor_args, then called with call_args. Generalizable pattern: works for tvboptim, TVB, or any Python class.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:ClassReference', 'from_schema': 'https://w3id.org/tvbo'})

    constructor_args: Optional[list[Argument]] = Field(default=None, description="""Arguments passed to __init__ when instantiating the class. Example: period=1000.0, downsample_period=4.0 for Bold monitor.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClassReference']} })
    call_args: Optional[list[Argument]] = Field(default=None, description="""Arguments passed when calling the instance (__call__). Usually the input data from simulation result. Example: result (simulation output array).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClassReference']} })
    warmup_source: Optional[str] = Field(default=None, description="""Reference to transient simulation result for history initialization. Some monitors (e.g., Bold) require history from warmup simulation. Value should reference a simulation result name (e.g., 'result_init').""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'ClassReference']} })
    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    module: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Callable']} })
    software: Optional[SoftwareRequirement] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Callable', 'Continuation', 'SimulationExperiment']} })


class Case(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Case', 'from_schema': 'https://w3id.org/tvbo'})

    condition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Session', 'ConditionalBlock', 'Event', 'Case']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator']} })


class DerivedParameter(Parameter):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:DerivedParameter', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    symbol: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable',
                       'Parameter',
                       'DerivedParameter',
                       'DerivedVariable']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    unit: Optional[UnitEnum] = Field(default=None, description="""Physical unit of measurement. Values are drawn from the QUDT ontology (http://qudt.org/vocab/unit/) with UO cross-references where available.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField'],
         'slot_uri': 'qudt:unit'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator'],
         'slot_uri': 'skos:definition'} })
    value: Optional[Any] = Field(default=None, description="""Numeric, string, or boolean value. ScalarValue accepts any literal primitive type, allowing parameters to carry control flags (e.g., booleans) or symbolic placeholders alongside numeric defaults.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateValue',
                       'Parameter',
                       'Argument',
                       'Option',
                       'BoundaryCondition'],
         'slot_uri': 'schema:value'} })
    default: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter'], 'slot_uri': 'schema:defaultValue'} })
    domain: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'ExplorationAxis',
                       'FreeParameter',
                       'PDE']} })
    reported_optimum: Optional[float] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    dataset_path: Optional[str] = Field(default=None, description="""Dataset path for array-valued parameters. When set, the parameter value is stored in the binary companion file (HDF5 or Zarr) at this path. The value slot is omitted.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    grounding: Optional[list[str]] = Field(default=None, description="""External ontology IRIs (typically GO, ChEBI, UBERON, CL, MeSH) that this entity is a surrogate / abstraction / model of. Replaces the legacy OWL pattern `tvbo:surrogate_of` by carrying the link inline with the YAML data instance. Multiple IRIs allowed: a single parameter may abstract several biological processes (e.g. a synaptic conductance grounding both GO:0060079 (excitatory PSP) and GO:0007268 (chemical synaptic transmission)).""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Parameter', 'DerivedVariable'],
         'slot_uri': 'oboInOwl:hasDbXref'} })
    comment: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter'], 'slot_uri': 'rdfs:comment'} })
    heterogeneous: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'FreeParameter']} })
    distribution: Optional[Distribution] = Field(default=None, description="""Distribution for heterogeneous per-node parameter sampling. Implies heterogeneous=true.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Parameter', 'Noise', 'Coupling']} })
    source: Optional[str] = Field(default=None, description="""Data source for this parameter's value. When set, the value is loaded from the referenced entity rather than being a YAML literal. The referent is typically a Network with per-node parameters (dscalar pattern) or a flat dataset (HDF5, TSV). Combine with `measure:` when the source exposes multiple named measures. Distinct from the global `iri:` slot, which is reserved for ontology grounding.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge', 'Observation', 'Dynamics', 'Parameter', 'CouplingInput']} })
    measure: Optional[str] = Field(default=None, description="""Selector into the source. When `source` points at a Network with per-node parameters (or a dscalar with multiple maps), picks which named measure to load. Aligns with names listed in Network.structural_measures / Network.observational_measures. Ignored when the source resolves to a scalar/array dataset.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    free: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    shape: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'NamedArray', 'Parameter', 'FreeParameter', 'NDArray']} })
    explored_values: Optional[AnyShapeArray[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Range', 'Parameter', 'ExplorationAxis']} })
    element_domains: Optional[list[Range]] = Field(default=None, description="""Per-element domain overrides for heterogeneous parameters. When specified, element_domains[i] overrides domain for element i during exploration auto-expansion. Length must match parameter shape (e.g., n_nodes for shape \"(n_nodes,)\"). If not set, all elements share the same domain.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'ExplorationAxis']} })


class DerivedVariable(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:DerivedVariable',
         'from_schema': 'https://w3id.org/tvbo',
         'slot_usage': {'record': {'ifabsent': 'False', 'name': 'record'}}})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    symbol: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable',
                       'Parameter',
                       'DerivedParameter',
                       'DerivedVariable']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    unit: Optional[UnitEnum] = Field(default=None, description="""Physical unit of measurement. Values are drawn from the QUDT ontology (http://qudt.org/vocab/unit/) with UO cross-references where available.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField'],
         'slot_uri': 'qudt:unit'} })
    record: Optional[bool] = Field(default=False, description="""Whether to include this element in simulation output files. Applicable to state variables (default true), derived variables (default false), and network nodes (default true). Set false to suppress recording.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'StateVariable', 'DerivedVariable'], 'ifabsent': 'False'} })
    grounding: Optional[list[str]] = Field(default=None, description="""External ontology IRIs (typically GO, ChEBI, UBERON, CL, MeSH) that this entity is a surrogate / abstraction / model of. Replaces the legacy OWL pattern `tvbo:surrogate_of` by carrying the link inline with the YAML data instance. Multiple IRIs allowed: a single parameter may abstract several biological processes (e.g. a synaptic conductance grounding both GO:0060079 (excitatory PSP) and GO:0007268 (chemical synaptic transmission)).""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Parameter', 'DerivedVariable'],
         'slot_uri': 'oboInOwl:hasDbXref'} })
    conditional: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['DerivedVariable'], 'ifabsent': 'False'} })
    cases: Optional[list[Case]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DerivedVariable']} })


class Noise(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Noise', 'from_schema': 'https://w3id.org/tvbo'})

    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    noise_type: Optional[str] = Field(default="gaussian", json_schema_extra = { "linkml_meta": {'domain_of': ['Noise'], 'ifabsent': 'gaussian'} })
    correlated: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['Noise'], 'ifabsent': 'False'} })
    gaussian: Optional[bool] = Field(default=False, description="""Indicates whether the noise is Gaussian""", json_schema_extra = { "linkml_meta": {'domain_of': ['Noise'], 'ifabsent': 'False'} })
    additive: Optional[bool] = Field(default=True, description="""Indicates whether the noise is additive""", json_schema_extra = { "linkml_meta": {'domain_of': ['Noise'], 'ifabsent': 'True'} })
    seed: Optional[int] = Field(default=42, json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator', 'Distribution', 'Noise'],
         'ifabsent': 'integer(42)'} })
    random_state: Optional[RandomStream] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Noise']} })
    intensity: Optional[Parameter] = Field(default=None, description="""Optional scalar or vector intensity parameter for noise.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Noise']} })
    distribution: Optional[Distribution] = Field(default=None, description="""Optional probability distribution from which the noise draws its samples. When set, takes precedence over the `noise_type` / `gaussian` flags — any Distribution family is accepted (Uniform { lo, hi }, Normal { mean, std }, LogNormal { mu, sigma }, Beta, …). Reuses the existing Distribution class also used by Parameter.distribution.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Parameter', 'Noise', 'Coupling']} })
    function: Optional[Function] = Field(default=None, description="""Optional functional form of the noise (callable specification).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Distribution', 'FunctionCall', 'Noise']} })
    pycode: Optional[str] = Field(default=None, description="""Inline Python code representation of the noise process.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation', 'Noise']} })
    targets: Optional[dict[str, StateVariable]] = Field(default=None, description="""State variables this noise applies to; if omitted, applies globally.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Noise']} })


class RandomStream(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:RandomStream', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })


class DataSource(ConfiguredBaseModel):
    """
    Specification for loading external/empirical data.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:DataSource', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    path: Optional[str] = Field(default=None, description="""File path or URI to the data""", json_schema_extra = { "linkml_meta": {'domain_of': ['File', 'DataSource']} })
    loader: Optional[Callable] = Field(default=None, description="""Callable that loads the data (e.g., load_functional_connectivity)""", json_schema_extra = { "linkml_meta": {'domain_of': ['DataSource']} })
    format: Optional[str] = Field(default=None, description="""Data format: 'npy', 'mat', 'csv', 'nifti', etc.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'Edge', 'DataSource']} })
    key: Optional[str] = Field(default=None, description="""Key/variable name within the file (for .mat, .npz, etc.)""", json_schema_extra = { "linkml_meta": {'domain_of': ['DataSource', 'SimulationStudy']} })
    preprocessing: Optional[Function] = Field(default=None, description="""Optional preprocessing to apply after loading""", json_schema_extra = { "linkml_meta": {'domain_of': ['DataSource']} })


class OptimizationStage(ConfiguredBaseModel):
    """
    A single stage in a multi-stage optimization workflow. Stages run sequentially, with each stage potentially using different parameters, shapes, learning rates, and algorithms.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:OptimizationStage', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    free_parameters: Optional[list[FreeParameter]] = Field(default=None, description="""Parameters to optimize in this stage. Each entry is a FreeParameter that references an existing Parameter by dotted scope (e.g. \"ReducedWongWang.w\" or \"FastLinearCoupling.G\") and supplies optimization-specific metadata (heterogeneous, shape, bounds). No new Parameter is created here.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Continuation']} })
    algorithm: Optional[str] = Field(default="adam", description="""Optimizer for this stage: 'adam', 'adamw', 'sgd', etc.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'AlgorithmInclude', 'Continuation'],
         'ifabsent': 'string(adam)'} })
    learning_rate: Optional[float] = Field(default=0.001, json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Algorithm'], 'ifabsent': 'float(0.001)'} })
    max_iterations: Optional[int] = Field(default=100, json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage'], 'ifabsent': 'integer(100)'} })
    hyperparameters: Optional[list[Parameter]] = Field(default=None, description="""Stage-specific hyperparameters (e.g., b2=0.9999 for adam)""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Algorithm']} })
    freeze_parameters: Optional[list[str]] = Field(default=None, description="""Parameters from previous stages to freeze (keep at optimized value but not update)""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage']} })
    warmup_from: Optional[str] = Field(default=None, description="""Previous stage to initialize from. Final values from that stage become initial values for this stage.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage']} })


class Optimization(OptimizationStage):
    """
    Configuration for parameter optimization. Inherits single-stage fields from OptimizationStage. For multi-stage workflows, use 'stages' (ignores inherited single-stage fields). Loss equation references observations directly by name.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Optimization', 'from_schema': 'https://w3id.org/tvbo'})

    execution: Optional[ExecutionConfig] = Field(default=None, description="""Per-optimization execution configuration (overrides experiment-level defaults). Useful for setting random_seed, precision, or hardware for optimization phase.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization',
                       'Exploration',
                       'Algorithm',
                       'Continuation',
                       'SimulationExperiment']} })
    integration: Optional[Integrator] = Field(default=None, description="""Integration settings for optimization simulations (overrides experiment defaults). If specified, creates a fresh model_fn and state with prepare() before optimization. Can specify different duration, step_size, method than the experiment. If not specified, uses experiment-level integration settings.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization', 'SimulationExperiment']} })
    loss: Optional[FunctionCall] = Field(default=None, description="""Loss function call. Uses FunctionCall to either: 1. Reference existing function: function: rmse 2. Inline callable: callable: {module: ..., name: ...} Arguments specify inputs (simulated_fc, empirical_fc, etc.)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization']} })
    stages: Optional[list[OptimizationStage]] = Field(default=None, description="""Ordered list of optimization stages. Stages run sequentially. Stage n+1 starts from optimized values of stage n. When defined, inherited single-stage fields are ignored.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization']} })
    depends_on: Optional[str] = Field(default=None, description="""Algorithm to use as starting point for optimization. If specified, optimization starts from algorithm's result state. If not specified, optimization starts from initial simulation state.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization', 'Algorithm']} })
    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    free_parameters: Optional[list[FreeParameter]] = Field(default=None, description="""Parameters to optimize in this stage. Each entry is a FreeParameter that references an existing Parameter by dotted scope (e.g. \"ReducedWongWang.w\" or \"FastLinearCoupling.G\") and supplies optimization-specific metadata (heterogeneous, shape, bounds). No new Parameter is created here.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Continuation']} })
    algorithm: Optional[str] = Field(default="adam", description="""Optimizer for this stage: 'adam', 'adamw', 'sgd', etc.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'AlgorithmInclude', 'Continuation'],
         'ifabsent': 'string(adam)'} })
    learning_rate: Optional[float] = Field(default=0.001, json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Algorithm'], 'ifabsent': 'float(0.001)'} })
    max_iterations: Optional[int] = Field(default=100, json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage'], 'ifabsent': 'integer(100)'} })
    hyperparameters: Optional[list[Parameter]] = Field(default=None, description="""Stage-specific hyperparameters (e.g., b2=0.9999 for adam)""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Algorithm']} })
    freeze_parameters: Optional[list[str]] = Field(default=None, description="""Parameters from previous stages to freeze (keep at optimized value but not update)""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage']} })
    warmup_from: Optional[str] = Field(default=None, description="""Previous stage to initialize from. Final values from that stage become initial values for this stage.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage']} })


class Exploration(ConfiguredBaseModel):
    """
    Parameter space exploration (grid search, sweep).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Exploration', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    execution: Optional[ExecutionConfig] = Field(default=None, description="""Per-exploration execution configuration (overrides experiment-level defaults). Useful for setting random_seed, n_workers for parallel grid search.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization',
                       'Exploration',
                       'Algorithm',
                       'Continuation',
                       'SimulationExperiment']} })
    space: Optional[list[ExplorationAxis]] = Field(default=None, description="""Ordered list of exploration axes spanning the search space. Each axis references an existing Parameter (by dotted name, e.g. \"ReducedWongWang.w\" or \"FastLinearCoupling.G\") and supplies the Range to sweep. No new Parameter is created here.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Exploration']} })
    parameters: Optional[list[Parameter]] = Field(default=None, description="""Hyper-parameters of the exploration itself (e.g. tolerances, sampler settings, grid-refinement controls). Distinct from `space`, which defines what is being swept.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    mode: Optional[str] = Field(default="product", description="""Combination mode: 'product' (full grid), 'zip' (paired)""", json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting', 'Exploration'],
         'ifabsent': 'string(product)'} })
    observable: Optional[FunctionCall] = Field(default=None, description="""Observable to compute at each point. Use function: obs_name for simple observation, or function: func_name + arguments for FunctionCall.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Exploration']} })
    n_parallel: Optional[int] = Field(default=1, description="""Parallel evaluations""", json_schema_extra = { "linkml_meta": {'domain_of': ['Exploration'], 'ifabsent': 'integer(1)'} })
    n_trials: Optional[int] = Field(default=1, description="""Number of independent trials per grid point. Each trial uses a different noise seed. Used for averaging stochastic simulations (e.g., VEP = average of 20 trials).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Exploration'], 'ifabsent': 'integer(1)'} })
    average: Optional[str] = Field(default=None, description="""Averaging mode across trials. 'trials' = average over n_trials independent runs (evoked potential paradigm). None = return all trials.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Exploration']} })
    parallel_mode: Optional[ParallelMode] = Field(default=None, description="""How trial-axis parallelism is realised at codegen time. ``vmap`` batches all trials in parallel (fast, peak memory ~n_trials × per-trial working set). ``lax_map`` runs them sequentially via ``jax.lax.map`` (slower, peak memory bounded by one trial). ``pmap`` shards across devices for multi-GPU. ``auto`` (default) picks ``vmap`` when the estimated batched memory fits, ``lax_map`` otherwise.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Exploration']} })
    parallel_batch_size: Optional[int] = Field(default=None, description="""Chunk size for ``lax_map`` and chunked-``vmap`` modes. ``1`` = strictly sequential (minimum memory). Larger values amortise compile overhead across trials at the cost of memory. Ignored when parallel_mode is ``vmap`` (which always uses the full n_trials axis).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Exploration']} })


class ExplorationAxis(ConfiguredBaseModel):
    """
    One axis of a parameter exploration grid. Points to an existing Parameter (by dotted reference, e.g. \"ReducedWongWang.w\" or \"FastLinearCoupling.G\") and supplies the sweep specification (domain, explored_values, or per-element overrides). No new Parameter is created.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:ExplorationAxis', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    parameter: str = Field(default=..., description="""Dotted reference to the Parameter being swept. Format: \"<scope>.<param_name>\" where <scope> is either a Dynamics class name (e.g. \"ReducedWongWang\") or a coupling key (e.g. \"FastLinearCoupling\"). Resolved against the enclosing experiment at runtime.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExplorationAxis', 'FreeParameter']} })
    domain: Optional[Range] = Field(default=None, description="""Sweep range for this axis (lo, hi, n, step, log_scale).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'ExplorationAxis',
                       'FreeParameter',
                       'PDE']} })
    explored_values: Optional[AnyShapeArray[float]] = Field(default=None, description="""Explicit list of values to sweep (overrides domain).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range', 'Parameter', 'ExplorationAxis']} })
    element_domains: Optional[list[Range]] = Field(default=None, description="""Per-element sweep overrides for heterogeneous parameters (e.g. shape \"(n_nodes,)\"). When set, element_domains[i] replaces the shared domain for element i.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'ExplorationAxis']} })
    unit: Optional[str] = Field(default=None, description="""Optional axis unit (defaults to the referenced Parameter's unit).""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })


class FreeParameter(ConfiguredBaseModel):
    """
    One degree of freedom in an OptimizationStage. References an existing Parameter by dotted scope (e.g. \"ReducedWongWang.w\" or \"FastLinearCoupling.G\") and supplies optimization-specific metadata (heterogeneous, shape, bounds, initial value). No new Parameter is created here.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:FreeParameter', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    parameter: str = Field(default=..., description="""Dotted reference to the Parameter to optimize. Format: \"<scope>.<param_name>\" where <scope> is either a Dynamics class name or a coupling key.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExplorationAxis', 'FreeParameter']} })
    heterogeneous: Optional[bool] = Field(default=False, description="""If true, the parameter is optimized per-element (broadcast to `shape`). If false, a single scalar is optimized and broadcast at runtime.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'FreeParameter'], 'ifabsent': 'boolean(false)'} })
    shape: Optional[str] = Field(default=None, description="""Optimization shape as a Python-tuple-style string (e.g. \"(n_nodes,)\" or \"(n_nodes, n_nodes)\"). Required when heterogeneous is true.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'NamedArray', 'Parameter', 'FreeParameter', 'NDArray']} })
    initial_value: Optional[Any] = Field(default=None, description="""Optional initial value for the optimizer (overrides the referenced Parameter's value).""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'FreeParameter', 'SpatialField']} })
    domain: Optional[Range] = Field(default=None, description="""Optional bounds (lo, hi) for constrained optimization.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'ExplorationAxis',
                       'FreeParameter',
                       'PDE']} })


class UpdateRule(ConfiguredBaseModel):
    """
    Defines how a parameter is updated based on observables. Represents iterative learning rules like FIC or EIB updates. Functions from experiment.functions are available in the equation.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:UpdateRule', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    target_parameter: Parameter = Field(default=..., description="""The parameter to update (e.g., J_i, wLRE)""", json_schema_extra = { "linkml_meta": {'domain_of': ['UpdateRule']} })
    equation: Equation = Field(default=..., description="""Update equation (e.g., 'J_i + eta * delta'). Can use functions defined in experiment.functions section.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator']} })
    bounds: Optional[Range] = Field(default=None, description="""Constraints on parameter values after update""", json_schema_extra = { "linkml_meta": {'domain_of': ['UpdateRule']} })
    warmup: Optional[bool] = Field(default=None, description="""Whether to apply learning rate warmup to this update rule. When true, the learning rate (eta) is scaled by (i+1)/n_iterations.""", json_schema_extra = { "linkml_meta": {'domain_of': ['UpdateRule']} })
    requires: Optional[list[str]] = Field(default=None, description="""Observables required by this update rule""", json_schema_extra = { "linkml_meta": {'domain_of': ['UpdateRule']} })


class AlgorithmInclude(ConfiguredBaseModel):
    """
    Reference to an included algorithm with optional argument overrides. Allows combining algorithms with different hyperparameter values.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:AlgorithmInclude', 'from_schema': 'https://w3id.org/tvbo'})

    algorithm: str = Field(default=..., description="""Reference to the algorithm to include""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'AlgorithmInclude', 'Continuation']} })
    arguments: Optional[list[Parameter]] = Field(default=None, description="""Override hyperparameter values for the included algorithm. Maps parameter names to new values.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall', 'AlgorithmInclude']} })


class TuningObjective(ConfiguredBaseModel):
    """
    Defines what the tuning algorithm optimizes for. Can be an activity target (FIC) or a connectivity target (EIB).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:TuningObjective', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    type: Optional[str] = Field(default=None, description="""Type of objective: 'activity_target', 'fc_matching', 'custom'""", json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator',
                       'File',
                       'Aggregation',
                       'TuningObjective',
                       'Algorithm']} })
    target_variable: Optional[str] = Field(default=None, description="""State variable for activity targets (e.g., S_e)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Event', 'TuningObjective']} })
    target_value: Optional[float] = Field(default=None, description="""Target value for activity objectives""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningObjective']} })
    target_data: Optional[str] = Field(default=None, description="""Reference to empirical data observation for matching objectives""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningObjective']} })
    metric: Optional[Equation] = Field(default=None, description="""Metric equation for matching (e.g., correlation, rmse)""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningObjective']} })


class Algorithm(ConfiguredBaseModel):
    """
    A complete specification of an iterative parameter tuning algorithm. Combines update rules, objectives, observations, and hyperparameters.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Algorithm', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    execution: Optional[ExecutionConfig] = Field(default=None, description="""Per-algorithm execution configuration (overrides experiment-level defaults). Useful for setting random_seed per algorithm to ensure reproducibility.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization',
                       'Exploration',
                       'Algorithm',
                       'Continuation',
                       'SimulationExperiment']} })
    type: Optional[str] = Field(default=None, description="""Algorithm type: 'fic', 'eib', 'homeostatic', 'custom'""", json_schema_extra = { "linkml_meta": {'domain_of': ['GraphGenerator',
                       'File',
                       'Aggregation',
                       'TuningObjective',
                       'Algorithm']} })
    includes: Optional[list[AlgorithmInclude]] = Field(default=None, description="""Include update rules from other algorithms with optional argument overrides. Unlike depends_on (sequential), includes means combined execution. Example: includes: [{algorithm: fic, arguments: [{name: eta, value: 0.1}]}]""", json_schema_extra = { "linkml_meta": {'domain_of': ['Algorithm']} })
    objective: Optional[TuningObjective] = Field(default=None, description="""What the algorithm optimizes for""", json_schema_extra = { "linkml_meta": {'domain_of': ['Algorithm']} })
    observations: Optional[list[str]] = Field(default=None, description="""References to observations defined in the observations section. Includes both simulated observations and external data (via data_source).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Algorithm', 'SimulationExperiment']} })
    update_rules: Optional[list[UpdateRule]] = Field(default=None, description="""How parameters are updated each iteration. When using 'includes', update_rules are inherited from included algorithms.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Algorithm']} })
    hyperparameters: Optional[list[Parameter]] = Field(default=None, description="""Additional algorithm-specific parameters""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Algorithm']} })
    learning_rate: Optional[float] = Field(default=None, description="""Learning rate (eta) for the tuning algorithm""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Algorithm']} })
    learning_rate_warmup: Optional[bool] = Field(default=False, description="""Linear warmup of learning rate from 0 to learning_rate over n_iterations. eta_effective = eta * (i+1) / n_iterations""", json_schema_extra = { "linkml_meta": {'domain_of': ['Algorithm'], 'ifabsent': 'boolean(false)'} })
    n_iterations: Optional[int] = Field(default=None, description="""Number of iterations to run""", json_schema_extra = { "linkml_meta": {'domain_of': ['Algorithm']} })
    learning_rate_schedule: Optional[str] = Field(default=None, description="""Learning rate schedule: 'constant', 'linear', 'exponential'""", json_schema_extra = { "linkml_meta": {'domain_of': ['Algorithm']} })
    simulation_period: Optional[float] = Field(default=None, description="""Duration of each simulation step (e.g., one BOLD TR)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Algorithm']} })
    apply_every: Optional[int] = Field(default=1, description="""Apply update every N iterations""", json_schema_extra = { "linkml_meta": {'domain_of': ['Algorithm'], 'ifabsent': 'integer(1)'} })
    functions: Optional[list[FunctionCall]] = Field(default=None, description="""Function calls for tracking progress, computing metrics, etc. Each FunctionCall references a function from the experiment's functions section and specifies arguments for that specific algorithm context.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Algorithm', 'SimulationExperiment', 'PDE']} })
    depends_on: Optional[list[str]] = Field(default=None, description="""Other algorithms that must run first (e.g., EIB depends on FIC)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization', 'Algorithm']} })


class Option(ConfiguredBaseModel):
    """
    A toolkit-specific key-value option (string name + string value). Used for backend settings that are not universal numeric parameters (e.g., solver name, tangent method, jacobian type).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Option',
         'from_schema': 'https://w3id.org/tvbo',
         'slot_usage': {'name': {'description': 'Option name (key).',
                                 'name': 'name',
                                 'required': True}}})

    name: str = Field(default=..., description="""Option name (key).""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    value: str = Field(default=..., description="""Option value.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateValue',
                       'Parameter',
                       'Argument',
                       'Option',
                       'BoundaryCondition']} })


class Discretization(ConfiguredBaseModel):
    """
    Discretization method for boundary value problems in continuation (periodic orbits, connecting orbits, quasi-periodic tori). Specifies the method; method-specific numerics go in parameters.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Discretization', 'from_schema': 'https://w3id.org/tvbo'})

    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    method: Optional[NumericalDiscretizationMethod] = Field(default=NumericalDiscretizationMethod.collocation, description="""Discretization method.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization', 'InitialState', 'Solver', 'Integrator'],
         'ifabsent': 'string(collocation)'} })
    ode_solver: Optional[Solver] = Field(default=None, description="""ODE solver for flow-based methods (shooting, poincaré). Specifies algorithm (e.g. Vern9, Rodas5) and tolerances. Not needed for collocation or trapezoid.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization']} })
    linear_solver: Optional[Solver] = Field(default=None, description="""Linear solver for the Newton bordered system. E.g. COPBLS (collocation), MatrixBLS (shooting/poincaré).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization']} })
    mesh_intervals: Optional[int] = Field(default=50, description="""Number of mesh intervals (time slices) for collocation or trapezoid methods. Collocation: N in PeriodicOrbitOCollProblem(N, m). Trapezoid: M in PeriodicOrbitTrapProblem(M=...).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization'], 'ifabsent': 'int(50)'} })
    degree: Optional[int] = Field(default=4, description="""Polynomial degree per mesh interval for collocation. The m in PeriodicOrbitOCollProblem(N, m).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization'], 'ifabsent': 'int(4)'} })
    n_sections: Optional[int] = Field(default=3, description="""Number of shooting sections for shooting or Poincaré methods.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization'], 'ifabsent': 'int(3)'} })
    options: Optional[list[Option]] = Field(default=None, description="""Toolkit-specific string options (jacobian type, etc.).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization', 'BranchSwitch', 'Continuation']} })


class InitialState(ConfiguredBaseModel):
    """
    How to obtain the starting equilibrium or periodic orbit for continuation. Most robust: time-integrate to steady state.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:InitialState', 'from_schema': 'https://w3id.org/tvbo'})

    method: Optional[InitialStateMethod] = Field(default=InitialStateMethod.time_integration, description="""Strategy for finding the initial state.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization', 'InitialState', 'Solver', 'Integrator'],
         'ifabsent': 'string(time_integration)'} })
    duration: Optional[float] = Field(default=2000.0, description="""Integration duration for time_integration method.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'Event', 'InitialState', 'Integrator'],
         'ifabsent': 'float(2000.0)'} })
    abs_tol: Optional[float] = Field(default=1e-10, description="""Absolute tolerance for ODE integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState', 'Solver'], 'ifabsent': 'float(1e-10)'} })
    rel_tol: Optional[float] = Field(default=1e-10, description="""Relative tolerance for ODE integration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState', 'Solver'], 'ifabsent': 'float(1e-10)'} })
    solver: Optional[Solver] = Field(default=None, description="""ODE solver for time_integration method. Specify method (e.g., Tsit5, Heun, RK4) and tolerances.""", json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState', 'PDE']} })
    source_branch: Optional[str] = Field(default=None, description="""Name of a previously computed branch (for from_branch method).""", json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState']} })
    source_point: Optional[str] = Field(default=None, description="""Which point on the source branch: 'endpoint', 'hopf:1', 'fold:2', a step number, etc.""", json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState', 'BranchSwitch']} })


class BranchSwitch(ConfiguredBaseModel):
    """
    Specification for switching from a detected bifurcation point to a new branch (periodic orbits from Hopf, fold continuation, etc.). Each BranchSwitch says: \"from which special point on the parent branch, continue what kind of object, with what settings.\" Override parent solver settings via the inline continuation field — only explicitly set attributes take effect; everything else is inherited from the parent Continuation.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:BranchSwitch', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    source_point: Optional[str] = Field(default=None, description="""Which bifurcation point to start from. Syntax: - 'hopf:-1' = last Hopf (default) - 'hopf:all' = all Hopf points - 'hopf:1' = first Hopf - 'fold:2' = second fold - integer = specific special point index""", json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState', 'BranchSwitch']} })
    delta_p: Optional[float] = Field(default=None, description="""Initial parameter offset from the bifurcation point.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BranchSwitch']} })
    continuation: Optional[Continuation] = Field(default=None, description="""Override solver settings for this branch. Uses the same Continuation type — only explicitly set attributes override the parent's values.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BranchSwitch']} })
    discretization: Optional[Discretization] = Field(default=None, description="""Discretization method for the branch solution. Required for periodic orbit branches (Hopf → PO). Not needed for codim-2 branches (fold/Hopf continuation).""", json_schema_extra = { "linkml_meta": {'domain_of': ['BranchSwitch', 'PDESolver']} })
    bothside: Optional[bool] = Field(default=None, description="""Continue branch in both directions.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BranchSwitch', 'Continuation']} })
    options: Optional[dict[str, Union[str, Option]]] = Field(default=None, description="""Toolkit-specific string options for this branch (linear solver, etc.).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization', 'BranchSwitch', 'Continuation']} })


class Continuation(ConfiguredBaseModel):
    """
    Complete specification of a numerical continuation / bifurcation analysis. All universal solver settings live directly here. Toolkit-specific string options go in the options slot. When used inside a BranchSwitch, only explicitly set attributes override the parent's values.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Continuation', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    dynamics: Optional[str] = Field(default=None, description="""Reference to the dynamical system model (by name). Resolved from the experiment's dynamics dict at runtime.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network',
                       'Node',
                       'Edge',
                       'Continuation',
                       'SimulationExperiment']} })
    free_parameters: Optional[list[Parameter]] = Field(default=None, description="""Parameters to vary. First parameter is primary (codim-1); second enables codim-2 continuation. Each Parameter has name + domain (Range with lo/hi bounds).""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Continuation']} })
    ds: Optional[float] = Field(default=None, description="""Initial arc-length step size.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    ds_min: Optional[float] = Field(default=None, description="""Minimum adaptive step size.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    ds_max: Optional[float] = Field(default=None, description="""Maximum adaptive step size.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    max_steps: Optional[int] = Field(default=None, description="""Maximum continuation steps.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    newton_tol: Optional[float] = Field(default=None, description="""Absolute tolerance for Newton corrector convergence.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    newton_max_iterations: Optional[int] = Field(default=None, description="""Maximum Newton corrector iterations per step.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    nev: Optional[int] = Field(default=None, description="""Number of eigenvalues to compute. Must be >= number of state variables for reliable Hopf detection.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    tol_stability: Optional[float] = Field(default=None, description="""Tolerance on real part of eigenvalue for stability boundary.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    detect_bifurcation: Optional[int] = Field(default=None, description="""Bifurcation detection level. 0 = off, 1 = eigenvalues only, 2 = detect, 3 = locate precisely.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    detect_fold: Optional[bool] = Field(default=None, description="""Enable fold (limit point) detection.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    n_inversion: Optional[int] = Field(default=None, description="""Number of eigenvalue sign inversions to flag a bifurcation. Must be even. Higher = fewer false positives.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    max_bisection_steps: Optional[int] = Field(default=None, description="""Maximum bisection steps for bifurcation point localization.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    algorithm: Optional[ContinuationAlgorithm] = Field(default=ContinuationAlgorithm.PALC, description="""Predictor-corrector algorithm.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'AlgorithmInclude', 'Continuation'],
         'ifabsent': 'string(PALC)'} })
    initial_state: Optional[InitialState] = Field(default=None, description="""How to obtain the initial equilibrium. Default: time integration to steady state.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    branches: Optional[dict[str, BranchSwitch]] = Field(default=None, description="""Child branches to continue from detected bifurcation points (PO from Hopf, fold continuation, etc.).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Continuation']} })
    bothside: Optional[bool] = Field(default=None, description="""Continue in both directions from the starting point.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BranchSwitch', 'Continuation']} })
    execution: Optional[ExecutionConfig] = Field(default=None, description="""Per-analysis execution configuration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization',
                       'Exploration',
                       'Algorithm',
                       'Continuation',
                       'SimulationExperiment']} })
    software: Optional[SoftwareRequirement] = Field(default=None, description="""Backend engine (BifurcationKit, AUTO-07p, MatCont, etc.).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Callable', 'Continuation', 'SimulationExperiment']} })
    options: Optional[list[Option]] = Field(default=None, description="""Toolkit-specific string options (tangent method, solver name, etc.).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization', 'BranchSwitch', 'Continuation']} })


class Solver(ConfiguredBaseModel):
    """
    Lightweight specification of a numerical ODE solver / integrator. Covers adaptive solvers (Vern9, Rodas5, Tsit5, etc.) used in shooting methods, initial-state integration, and other contexts where only the algorithm and tolerances matter.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Solver', 'from_schema': 'https://w3id.org/tvbo'})

    method: Optional[str] = Field(default="Tsit5", description="""Solver algorithm name (e.g., Vern9, Rodas5, Tsit5, euler, heun, rk4).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization', 'InitialState', 'Solver', 'Integrator'],
         'ifabsent': 'string(Tsit5)'} })
    abs_tol: Optional[float] = Field(default=1e-10, description="""Absolute tolerance for adaptive solvers.""", json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState', 'Solver'], 'ifabsent': 'float(1e-10)'} })
    rel_tol: Optional[float] = Field(default=1e-10, description="""Relative tolerance for adaptive solvers.""", json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState', 'Solver'], 'ifabsent': 'float(1e-10)'} })


class Integrator(Solver):
    """
    Fixed-step or adaptive ODE integrator with TVB-specific extensions (noise, transient time, etc.). Inherits abs_tol, rel_tol from Solver. Overrides method default to 'euler'.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Integrator', 'from_schema': 'https://w3id.org/tvbo'})

    time_scale: Optional[UnitEnum] = Field(default=UnitEnum.ms, description="""Time unit for the integration / simulation. Determines the physical time meaning of one model time-step.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'Integrator'], 'ifabsent': 'ms'} })
    unit: Optional[UnitEnum] = Field(default=None, description="""Physical unit of measurement. Values are drawn from the QUDT ontology (http://qudt.org/vocab/unit/) with UO cross-references where available.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField'],
         'slot_uri': 'qudt:unit'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    duration: Optional[float] = Field(default=1000, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'Event', 'InitialState', 'Integrator'],
         'ifabsent': 'float(1000)'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    method: Optional[str] = Field(default="euler", description="""Integration method (euler, heun, rk4, etc.)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Discretization', 'InitialState', 'Solver', 'Integrator'],
         'ifabsent': 'string(euler)'} })
    step_size: Optional[float] = Field(default=0.01220703125, json_schema_extra = { "linkml_meta": {'aliases': ['dt'],
         'domain_of': ['Integrator'],
         'ifabsent': 'float(0.01220703125)'} })
    steps: Optional[int] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator']} })
    noise: Optional[Noise] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'StateVariable', 'Integrator']} })
    state_wise_sigma: Optional[list[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator']} })
    transient_time: Optional[float] = Field(default=0, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator'], 'ifabsent': 'float(0)'} })
    scipy_ode_base: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator'], 'ifabsent': 'False'} })
    number_of_stages: Optional[int] = Field(default=1, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator'], 'ifabsent': 'integer(1)'} })
    intermediate_expressions: Optional[dict[str, DerivedVariable]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator']} })
    update_expression: Optional[DerivedVariable] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator']} })
    delayed: Optional[bool] = Field(default=True, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator', 'Coupling'], 'ifabsent': 'True'} })
    abs_tol: Optional[float] = Field(default=1e-10, description="""Absolute tolerance for adaptive solvers.""", json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState', 'Solver'], 'ifabsent': 'float(1e-10)'} })
    rel_tol: Optional[float] = Field(default=1e-10, description="""Relative tolerance for adaptive solvers.""", json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState', 'Solver'], 'ifabsent': 'float(1e-10)'} })


class Coupling(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Coupling',
         'from_schema': 'https://w3id.org/tvbo',
         'slot_usage': {'name': {'ifabsent': 'Linear', 'name': 'name'}}})

    name: str = Field(default="Linear", description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'ifabsent': 'Linear',
         'slot_uri': 'schema:name'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    iri: Optional[str] = Field(default=None, description="""Optional stable IRI (or compact URI) for this entity in an external ontology or knowledge base. Used to load metadata from an external source; not required when the entity is fully self-contained (equations, parameters, etc. defined in the file itself).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation',
                       'Tractogram',
                       'ReferenceFingerprint',
                       'Reference',
                       'Dynamics',
                       'Function',
                       'Coupling'],
         'slot_uri': 'dcterms:identifier'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    coupling_function: Optional[Equation] = Field(default=None, description="""Mathematical function defining the coupling""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    sparse: Optional[bool] = Field(default=False, description="""Whether the coupling uses sparse representations""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling'], 'ifabsent': 'False'} })
    pre_expression: Optional[Equation] = Field(default=None, description="""Pre-processing expression applied before coupling""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    post_expression: Optional[Equation] = Field(default=None, description="""Post-processing expression applied after coupling""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    incoming_states: Optional[list[str]] = Field(default=None, description="""References to state variables from connected (source) nodes. Auto-populated from state_variables with coupling_variable=true when omitted. Used by name in pre_expression.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    local_states: Optional[list[str]] = Field(default=None, description="""References to state variables from the local (target) node. Used by name in pre_expression.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    delayed: Optional[bool] = Field(default=True, description="""Whether coupling includes transmission delays""", json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator', 'Coupling'], 'ifabsent': 'True'} })
    symmetry: Optional[str] = Field(default="directed", description="""Edge symmetry type for NetworkDynamics.jl EdgeModel: 'directed' (default), 'antisymmetric', or 'symmetric'. AntiSymmetric edges flip sign for the reverse direction.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling'], 'ifabsent': 'string(directed)'} })
    outsym: Optional[list[str]] = Field(default=None, description="""Output symbol names for the edge model. E.g. ['P'] for a scalar power flow, ['Fx', 'Fy'] for 2D forces. Maps directly to outsym in ND.jl EdgeModel. If not specified, derived from coupling variables of the connected vertex dynamics.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    observed: Optional[dict[str, DerivedVariable]] = Field(default=None, description="""Observable functions computed from edge inputs and parameters after simulation. Maps to obsf/obssym in ND.jl EdgeModel. Example: absolute force magnitude computed from force components.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Coupling']} })
    inner_coupling: Optional[Coupling] = Field(default=None, description="""For hierarchical coupling: inner coupling applied at regional level""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    region_mapping: Optional[RegionMapping] = Field(default=None, description="""For hierarchical coupling: vertex-to-region mapping for aggregation""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    regional_connectivity: Optional[Network] = Field(default=None, description="""For hierarchical coupling: region-to-region connectivity with weights and delays""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    aggregation: Optional[str] = Field(default=None, description="""For hierarchical coupling: aggregation method ('sum', 'mean', 'max') or custom Function""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'string'}, {'range': 'Function'}],
         'domain_of': ['Observation', 'Coupling']} })
    distribution: Optional[str] = Field(default=None, description="""For hierarchical coupling: distribution method ('broadcast', 'weighted') or custom Function""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'string'}, {'range': 'Function'}],
         'domain_of': ['StateVariable', 'Parameter', 'Noise', 'Coupling']} })


class RegionMapping(ConfiguredBaseModel):
    """
    Maps vertices to parent regions for hierarchical/aggregated coupling
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:RegionMapping', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })
    vertex_to_region: Optional[list[int]] = Field(default=None, description="""Array mapping each vertex index to its parent region index. Can use dataLocation instead for large arrays.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RegionMapping']} })
    n_vertices: Optional[int] = Field(default=None, description="""Total number of vertices""", json_schema_extra = { "linkml_meta": {'domain_of': ['RegionMapping']} })
    n_regions: Optional[int] = Field(default=None, description="""Total number of regions""", json_schema_extra = { "linkml_meta": {'domain_of': ['RegionMapping']} })


class Sample(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Sample', 'from_schema': 'https://w3id.org/tvbo'})

    groups: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Sample']} })
    size: Optional[int] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceFingerprint', 'Sample']} })


class ExecutionConfig(ConfiguredBaseModel):
    """
    Configuration for computational execution (parallelization, precision, hardware).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:ExecutionConfig', 'from_schema': 'https://w3id.org/tvbo'})

    n_workers: Optional[int] = Field(default=1, description="""Number of parallel workers (maps to pmap devices in JAX, processes in multiprocessing)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig'], 'ifabsent': 'integer(1)'} })
    n_threads: Optional[int] = Field(default=-1, description="""Number of CPU threads per worker (-1 = auto-detect)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig'], 'ifabsent': 'integer(-1)'} })
    precision: Optional[str] = Field(default="float64", description="""Floating point precision: 'float32' or 'float64'""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig'], 'ifabsent': 'string(float64)'} })
    accelerator: Optional[str] = Field(default="cpu", description="""Hardware accelerator: 'cpu', 'gpu', 'tpu'""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig'], 'ifabsent': 'string(cpu)'} })
    batch_size: Optional[int] = Field(default=None, description="""Batch size for vectorized operations (None = auto)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig']} })
    random_seed: Optional[int] = Field(default=42, description="""Base random seed for reproducibility""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig'], 'ifabsent': 'integer(42)'} })
    find_fixpoint: Optional[bool] = Field(default=False, description="""Whether to find a fixed point (steady state) before time integration. Used as initial condition for ODEProblem. Maps to NLsolve.fixpoint! in ND.jl or similar in other backends.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig'], 'ifabsent': 'False'} })


class SimulationExperiment(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Simulation',
         'from_schema': 'https://w3id.org/tvbo',
         'tree_root': True})

    model: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode', 'SimulationExperiment', 'SimulationStudy']} })
    references: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance',
                       'Dynamics',
                       'SimulationExperiment',
                       'SimulationStudy'],
         'slot_uri': 'dcterms:references'} })
    id: int = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'SimulationExperiment'],
         'slot_uri': 'dcterms:identifier'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    additional_equations: Optional[list[Equation]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    dynamics: Optional[Dynamics] = Field(default=None, description="""Default dynamics model for all nodes. For heterogeneous networks with multiple dynamics, use network.dynamics instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network',
                       'Node',
                       'Edge',
                       'Continuation',
                       'SimulationExperiment']} })
    integration: Optional[Integrator] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization', 'SimulationExperiment']} })
    connectivity: Optional[Network] = Field(default=None, json_schema_extra = { "linkml_meta": {'deprecated': "Use 'network' instead. 'connectivity' is kept for backward "
                       'compatibility only and will be removed in a future version.',
         'domain_of': ['SimulationExperiment']} })
    network: Optional[Network] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Subject', 'Session', 'SimulationExperiment']} })
    coupling: Optional[Coupling] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Network', 'Edge', 'SimulationExperiment']} })
    observations: Optional[dict[str, Observation]] = Field(default=None, description="""All observations on this experiment, keyed by name. An Observation is considered derived (computed from other observations rather than directly from state variables) when any item in its multivalued ``source`` slot names another observation in this same dict.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Algorithm', 'SimulationExperiment']} })
    functions: Optional[dict[str, Function]] = Field(default=None, description="""Reusable function definitions. Referenced by name in observation pipelines. Enables DRY: define compute_fc once, use in both simulated and empirical paths.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Algorithm', 'SimulationExperiment', 'PDE']} })
    stimulation: Optional[Stimulus] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    events: Optional[dict[str, Event]] = Field(default=None, description="""Events that apply at the experiment level. For component-level events, attach them to individual nodes or edges instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'Edge', 'Dynamics', 'SimulationExperiment']} })
    field_dynamics: Optional[PDE] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    optimizations: Optional[dict[str, Optimization]] = Field(default=None, description="""Parameter optimization configurations""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    explorations: Optional[dict[str, Exploration]] = Field(default=None, description="""Parameter exploration/grid search specifications""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    algorithms: Optional[dict[str, Algorithm]] = Field(default=None, description="""Iterative parameter tuning algorithms (FIC, EIB, etc.)""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    continuations: Optional[dict[str, Continuation]] = Field(default=None, description="""Numerical continuation and bifurcation analysis specifications. Each entry defines a continuation experiment (equilibrium branch, codim-2 curve, periodic orbit family, etc.).            # Either reference a full reusable environment (preferred) or, for""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    environment: Optional[SoftwareEnvironment] = Field(default=None, description="""Execution environment (collection of requirements).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'SimulationExperiment', 'PDESolver']} })
    execution: Optional[ExecutionConfig] = Field(default=None, description="""Computational execution configuration (parallelization, devices).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization',
                       'Exploration',
                       'Algorithm',
                       'Continuation',
                       'SimulationExperiment']} })
    software: Optional[SoftwareRequirement] = Field(default=None, description="""(Deprecated) Single software requirement; prefer 'environment' with aggregated requirements.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Callable', 'Continuation', 'SimulationExperiment']} })
    dataset: Optional[Dataset] = Field(default=None, description="""Multi-subject dataset for workflow rendering. When set, render_workflow() uses dataset.subjects/sessions to generate per-subject parallel jobs.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })


class SimulationStudy(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:SimulationStudy', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    derived_from: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance', 'SimulationStudy'],
         'slot_uri': 'prov:wasDerivedFrom'} })
    model: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode', 'SimulationExperiment', 'SimulationStudy']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    references: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance',
                       'Dynamics',
                       'SimulationExperiment',
                       'SimulationStudy'],
         'slot_uri': 'dcterms:references'} })
    key: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DataSource', 'SimulationStudy']} })
    title: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy'], 'slot_uri': 'dcterms:title'} })
    year: Optional[int] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy'], 'slot_uri': 'dcterms:issued'} })
    doi: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage', 'SimulationStudy'], 'slot_uri': 'bibo:doi'} })
    sample: Optional[Sample] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy']} })
    experiments: Optional[list[SimulationExperiment]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy']} })


class TimeSeries(ConfiguredBaseModel):
    """
    Time series data from simulations or measurements. Supports BIDS-compatible export for computational modeling (BEP034).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:TimeSeries',
         'comments': ['Supports BIDS BEP034 computational modeling extension.',
                      'Use to_bids() method in Python class to export as BIDS '
                      'dataset.'],
         'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })
    data: Optional[Matrix] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    time: Optional[Matrix] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    sampling_rate: Optional[float] = Field(default=None, description="""Sampling rate in Hz.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    sampling_period: Optional[float] = Field(default=None, description="""Time between samples (inverse of sampling_rate).""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    sampling_period_unit: Optional[str] = Field(default="ms", description="""Unit of the sampling period (e.g., 'ms', 's').""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries'], 'ifabsent': 'ms'} })
    unit: Optional[str] = Field(default=None, description="""Physical unit of the time series values.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField'],
         'slot_uri': 'qudt:unit'} })
    labels_ordering: Optional[list[str]] = Field(default=None, description="""Ordering of dimensions: Time, State Variable, Space, Mode.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    labels_dimensions: Optional[str] = Field(default=None, description="""Mapping of dimension names to their labels (JSON-encoded dict).""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    source_experiment: Optional[int] = Field(default=None, description="""Reference to the SimulationExperiment that generated this TimeSeries.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    generated_at: Optional[datetime ] = Field(default=None, description="""Timestamp when this TimeSeries was generated.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    software_environment: Optional[str] = Field(default=None, description="""Software environment used to generate this data.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    task_name: Optional[str] = Field(default=None, description="""BIDS task name for the simulation (e.g., 'rest', 'simulation').""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    subject_id: Optional[str] = Field(default=None, description="""BIDS subject identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject', 'TimeSeries'], 'slot_uri': 'dcterms:identifier'} })
    session_id: Optional[str] = Field(default=None, description="""BIDS session identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Session', 'TimeSeries'], 'slot_uri': 'dcterms:identifier'} })
    run_id: Optional[int] = Field(default=None, description="""BIDS run number.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries'], 'slot_uri': 'dcterms:identifier'} })
    modality: Optional[ImagingModality] = Field(default=None, description="""Imaging modality or simulation output type.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    model_equation_ref: Optional[str] = Field(default=None, description="""BIDS ModelEq reference: path to _eq.xml LEMS file.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    model_param_ref: Optional[str] = Field(default=None, description="""BIDS ModelParam reference: path to _param.xml LEMS file.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    connectivity_ref: Optional[str] = Field(default=None, description="""Reference to connectivity data (_conndata-network_connectivity.tsv).""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })


class NDArray(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:NDArray', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    shape: Optional[list[int]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'NamedArray', 'Parameter', 'FreeParameter', 'NDArray']} })
    dtype: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'NamedArray', 'NDArray']} })
    dataLocation: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })


class SpatialDomain(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:SpatialDomain', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    coordinate_space: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSDataset',
                       'DBSSubject',
                       'Electrode',
                       'EField',
                       'Network',
                       'SpatialDomain',
                       'Mesh']} })
    region: Optional[str] = Field(default=None, description="""Optional named region/ROI in the atlas/parcellation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'SpatialDomain']} })
    geometry: Optional[str] = Field(default=None, description="""Optional file for geometry/ROI mask (e.g., NIfTI, GIfTI).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SpatialDomain']} })


class Mesh(ConfiguredBaseModel):
    """
    Triangle (or higher-order) mesh geometry. May stand alone (via ``mesh_file`` pointing at an external GIFTI/VTK/MSH file) OR be inlined on a Network as ``Network.mesh``. In the inlined-on-Network case, the vertices are the parent Network's ``nodes/coordinates`` (so ``coordinates`` here may be left empty), the faces live in the same h5 companion under a path given by ``elements`` (default ``mesh/faces``), and optional per-vertex ``normals`` / ``curvature`` live alongside. The optional ``parcel_map_field`` points at the parent Network's per-vertex parcel-id array (default ``nodes/parent_index`` from the hierarchical-Network pattern, see Network.qmd §7.1).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Mesh', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Stimulus',
                       'Matrix',
                       'Dynamics',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'NDArray',
                       'Mesh']} })
    element_type: Optional[ElementType] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    coordinates: Optional[list[Coordinate]] = Field(default=None, description="""Node coordinates (x,y,z) in the given coordinate space.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    elements: Optional[str] = Field(default=None, description="""Topology indices: either a file reference, or (when inlined on a Network) a dotted path inside the parent Network's h5 companion to the (M, 3) int triangle face array. Default convention: ``mesh/faces``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    coordinate_space: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSDataset',
                       'DBSSubject',
                       'Electrode',
                       'EField',
                       'Network',
                       'SpatialDomain',
                       'Mesh']} })
    mesh_file: Optional[str] = Field(default=None, description="""Path to external mesh file (GIFTI, VTK, MSH, FreeSurfer, etc.).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    mesh_format: Optional[str] = Field(default=None, description="""Explicit format override (gifti, freesurfer, meshio, vtk, gmsh). Auto-detected from extension if null.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    number_of_vertices: Optional[int] = Field(default=None, description="""Number of vertices in the mesh.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    number_of_elements: Optional[int] = Field(default=None, description="""Number of elements (triangles, quads, tetrahedra, etc.).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    parcellation: Optional[Parcellation] = Field(default=None, description="""Brain parcellation this mesh's parcel_map indexes into. Inlined to match Network.parcellation's existing pattern. When the mesh is inlined on a Network that already declares ``parcellation``, this slot is optional and defaults to the parent's value.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network', 'Mesh']} })
    normals: Optional[str] = Field(default=None, description="""Optional path inside the parent Network's h5 (or an external file) to per-vertex normals (N, 3) float. Default convention: ``mesh/normals``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    curvature: Optional[str] = Field(default=None, description="""Optional path to per-vertex curvature (N,) float. Default convention: ``mesh/curvature``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    vertices_field: Optional[str] = Field(default=None, description="""When the mesh is inlined on a Network, the dotted path inside the parent's h5 to vertex coordinates. Default: ``nodes/coordinates``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    parcel_map_field: Optional[str] = Field(default=None, description="""When the mesh is inlined on a Network with a parcellation, the dotted path inside the parent's h5 to the per-vertex parcel id array. Default: ``nodes/parent_index``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })


class SpatialField(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:SpatialField', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    quantity_kind: Optional[str] = Field(default=None, description="""Scalar, vector, or tensor.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SpatialField']} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    mesh: Optional[Mesh] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Network', 'SpatialField', 'FieldStateVariable', 'PDE']} })
    values: Optional[NDArray] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'BrainRegionSeries', 'SpatialField']} })
    time_dependent: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['TemporalApplicableEquation',
                       'SpatialField',
                       'BoundaryCondition'],
         'ifabsent': 'False'} })
    initial_value: Optional[float] = Field(default=0.1, description="""Constant initial value for the field.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'FreeParameter', 'SpatialField'],
         'ifabsent': 'float(0.1)'} })
    initial_expression: Optional[Equation] = Field(default=None, description="""Analytic initial condition for the field.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SpatialField']} })


class FieldStateVariable(StateVariable):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:FieldStateVariable', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    mesh: Optional[Mesh] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Network', 'SpatialField', 'FieldStateVariable', 'PDE']} })
    boundary_conditions: Optional[list[BoundaryCondition]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['FieldStateVariable', 'PDE']} })
    name: str = Field(default=..., description="""Globally unique identifier for the entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Event',
                       'Tractogram',
                       'MeasureSpec',
                       'NamedArray',
                       'GraphGenerator',
                       'File',
                       'StateValue',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'UpdateRule',
                       'Algorithm',
                       'Option',
                       'BranchSwitch',
                       'Continuation',
                       'Coupling'],
         'slot_uri': 'schema:name'} })
    symbol: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable',
                       'Parameter',
                       'DerivedParameter',
                       'DerivedVariable']} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator'],
         'slot_uri': 'skos:definition'} })
    domain: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'ExplorationAxis',
                       'FreeParameter',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    unit: Optional[UnitEnum] = Field(default=None, description="""Physical unit of measurement. Values are drawn from the QUDT ontology (http://qudt.org/vocab/unit/) with UO cross-references where available.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'MeasureSpec',
                       'NamedArray',
                       'Edge',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'ExplorationAxis',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField'],
         'slot_uri': 'qudt:unit'} })
    record: Optional[bool] = Field(default=True, description="""Whether to include this element in simulation output files. Applicable to state variables (default true), derived variables (default false), and network nodes (default true). Set false to suppress recording.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'StateVariable', 'DerivedVariable'], 'ifabsent': 'True'} })
    grounding: Optional[list[str]] = Field(default=None, description="""External ontology IRIs (typically GO, ChEBI, UBERON, CL, MeSH) that this entity is a surrogate / abstraction / model of. Replaces the legacy OWL pattern `tvbo:surrogate_of` by carrying the link inline with the YAML data instance. Multiple IRIs allowed: a single parameter may abstract several biological processes (e.g. a synaptic conductance grounding both GO:0060079 (excitatory PSP) and GO:0007268 (chemical synaptic transmission)).""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Parameter', 'DerivedVariable'],
         'slot_uri': 'oboInOwl:hasDbXref'} })
    variable_of_interest: Optional[bool] = Field(default=True, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'True'} })
    coupling_variable: Optional[bool] = Field(default=False, description="""Whether this state variable is transmitted to connected nodes through the coupling function. In TVB terms, this determines the cvar indices (state variables extracted from history and fed into the coupling function). The coupling function may override this via its incoming_states attribute.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'False'} })
    equation_type: Optional[str] = Field(default="differential", description="""Type of equation: 'differential' (default) means dx/dt = rhs, 'algebraic' means 0 = rhs or x ~ rhs (DAE constraint). Algebraic equations are used by ModelingToolkit.jl backend.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'string(differential)'} })
    equation_order: Optional[int] = Field(default=1, description="""Order of the time derivative on the LHS. Default 1 means dx/dt = rhs (first-order ODE). Order 2 means d²x/dt² = rhs (second-order ODE), etc. Higher-order ODEs are automatically lowered to coupled first-order systems by backends like ModelingToolkit.jl via mtkcompile.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'int(1)'} })
    noise: Optional[Noise] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'StateVariable', 'Integrator']} })
    stimulation_variable: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })
    boundaries: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })
    initial_value: Optional[float] = Field(default=0.1, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'FreeParameter', 'SpatialField'],
         'ifabsent': 'float(0.1)'} })
    derivative_initial_value: Optional[float] = Field(default=None, description="""Initial value for the first time derivative, used when equation_order > 1. For a second-order ODE d²x/dt² = f, this sets dx/dt(0). Required by ModelingToolkit.jl to fully specify higher-order initial value problems.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })
    distribution: Optional[Distribution] = Field(default=None, description="""Distribution for sampling initial conditions per node. If present, initial_value is used as fallback/mean.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Parameter', 'Noise', 'Coupling']} })
    history: Optional[TimeSeries] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })


class DifferentialOperator(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:DifferentialOperator',
         'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator'],
         'slot_uri': 'skos:definition'} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Event',
                       'Observation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    operator_type: Optional[OperatorType] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DifferentialOperator']} })
    coefficient: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DifferentialOperator']} })
    tensor_coefficient: Optional[str] = Field(default=None, description="""Optional anisotropic tensor (e.g., diffusion).""", json_schema_extra = { "linkml_meta": {'domain_of': ['DifferentialOperator']} })
    expression: Optional[Equation] = Field(default=None, description="""Symbolic form (e.g., '-div(D * grad(u))').""", json_schema_extra = { "linkml_meta": {'domain_of': ['ConditionalBlock', 'DifferentialOperator']} })


class BoundaryCondition(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:BoundaryCondition', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    bc_type: Optional[BoundaryConditionType] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['BoundaryCondition']} })
    on_region: Optional[str] = Field(default=None, description="""Mesh/atlas subset where BC applies.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BoundaryCondition']} })
    value: Optional[Equation] = Field(default=None, description="""Constant, parameter, or equation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateValue',
                       'Parameter',
                       'Argument',
                       'Option',
                       'BoundaryCondition']} })
    time_dependent: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['TemporalApplicableEquation',
                       'SpatialField',
                       'BoundaryCondition'],
         'ifabsent': 'False'} })


class PDESolver(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:PDESolver', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    requirements: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment', 'Function', 'PDESolver']} })
    environment: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'SimulationExperiment', 'PDESolver']} })
    discretization: Optional[DiscretizationMethod] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['BranchSwitch', 'PDESolver']} })
    time_integrator: Optional[str] = Field(default=None, description="""e.g., implicit Euler, Crank-Nicolson.""", json_schema_extra = { "linkml_meta": {'domain_of': ['PDESolver']} })
    dt: Optional[float] = Field(default=None, description="""Time step (s).""", json_schema_extra = { "linkml_meta": {'domain_of': ['PDESolver']} })
    tolerances: Optional[str] = Field(default=None, description="""Abs/rel tolerances.""", json_schema_extra = { "linkml_meta": {'domain_of': ['PDESolver']} })
    preconditioner: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['PDESolver']} })


class PDE(ConfiguredBaseModel):
    """
    Partial differential equation problem definition.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:PDE', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Subject',
                       'Session',
                       'Dataset',
                       'Contact',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'FunctionCall',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'TuningObjective',
                       'Continuation',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'rdfs:label'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'ClinicalScore',
                       'SoftwarePackage',
                       'SoftwareRequirement',
                       'SoftwareEnvironment',
                       'Equation',
                       'Stimulus',
                       'Event',
                       'Tractogram',
                       'Matrix',
                       'Phenotype',
                       'MeasureSpec',
                       'NamedArray',
                       'Network',
                       'GraphGenerator',
                       'File',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'FunctionCall',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Exploration',
                       'ExplorationAxis',
                       'FreeParameter',
                       'UpdateRule',
                       'TuningObjective',
                       'Algorithm',
                       'BranchSwitch',
                       'Continuation',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE'],
         'slot_uri': 'dcterms:description'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'Event',
                       'TemporalApplicableEquation',
                       'Network',
                       'GraphGenerator',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Discretization',
                       'BranchSwitch',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    domain: Optional[SpatialDomain] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'ExplorationAxis',
                       'FreeParameter',
                       'PDE']} })
    mesh: Optional[Mesh] = Field(default=None, description="""Shared mesh for all field state variables in this PDE.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network', 'SpatialField', 'FieldStateVariable', 'PDE']} })
    state_variables: Optional[list[FieldStateVariable]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    field: Optional[SpatialField] = Field(default=None, description="""Primary field being solved for (deprecated; use state_variables).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceFingerprint', 'Reference', 'PDE']} })
    operators: Optional[list[DifferentialOperator]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['PDE']} })
    sources: Optional[list[Equation]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['PDE']} })
    boundary_conditions: Optional[list[BoundaryCondition]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['FieldStateVariable', 'PDE']} })
    solver: Optional[PDESolver] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['InitialState', 'PDE']} })
    derived_parameters: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    derived_variables: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    functions: Optional[list[str]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Algorithm', 'SimulationExperiment', 'PDE']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
Coordinate.model_rebuild()
BrainAtlas.model_rebuild()
CommonCoordinateSpace.model_rebuild()
ParcellationEntity.model_rebuild()
ParcellationTerminology.model_rebuild()
Subject.model_rebuild()
Session.model_rebuild()
Dataset.model_rebuild()
DBSDataset.model_rebuild()
DBSSubject.model_rebuild()
Electrode.model_rebuild()
Contact.model_rebuild()
StimulationSetting.model_rebuild()
DBSProtocol.model_rebuild()
ClinicalScale.model_rebuild()
ClinicalScore.model_rebuild()
ClinicalImprovement.model_rebuild()
EField.model_rebuild()
SoftwarePackage.model_rebuild()
SimulationTool.model_rebuild()
SoftwareRequirement.model_rebuild()
SoftwareEnvironment.model_rebuild()
Range.model_rebuild()
Equation.model_rebuild()
ConditionalBlock.model_rebuild()
Stimulus.model_rebuild()
Event.model_rebuild()
TemporalApplicableEquation.model_rebuild()
Parcellation.model_rebuild()
Tractogram.model_rebuild()
Matrix.model_rebuild()
BrainRegionSeries.model_rebuild()
Provenance.model_rebuild()
ReferenceFingerprint.model_rebuild()
Phenotype.model_rebuild()
MeasureSpec.model_rebuild()
NamedArray.model_rebuild()
BidsEntities.model_rebuild()
Network.model_rebuild()
GraphGenerator.model_rebuild()
File.model_rebuild()
Reference.model_rebuild()
Node.model_rebuild()
StateValue.model_rebuild()
Edge.model_rebuild()
Observation.model_rebuild()
Dynamics.model_rebuild()
StateVariable.model_rebuild()
Distribution.model_rebuild()
Parameter.model_rebuild()
CouplingInput.model_rebuild()
Argument.model_rebuild()
Function.model_rebuild()
Aggregation.model_rebuild()
LossFunction.model_rebuild()
FunctionCall.model_rebuild()
Callable.model_rebuild()
ClassReference.model_rebuild()
Case.model_rebuild()
DerivedParameter.model_rebuild()
DerivedVariable.model_rebuild()
Noise.model_rebuild()
RandomStream.model_rebuild()
DataSource.model_rebuild()
OptimizationStage.model_rebuild()
Optimization.model_rebuild()
Exploration.model_rebuild()
ExplorationAxis.model_rebuild()
FreeParameter.model_rebuild()
UpdateRule.model_rebuild()
AlgorithmInclude.model_rebuild()
TuningObjective.model_rebuild()
Algorithm.model_rebuild()
Option.model_rebuild()
Discretization.model_rebuild()
InitialState.model_rebuild()
BranchSwitch.model_rebuild()
Continuation.model_rebuild()
Solver.model_rebuild()
Integrator.model_rebuild()
Coupling.model_rebuild()
RegionMapping.model_rebuild()
Sample.model_rebuild()
ExecutionConfig.model_rebuild()
SimulationExperiment.model_rebuild()
SimulationStudy.model_rebuild()
TimeSeries.model_rebuild()
NDArray.model_rebuild()
SpatialDomain.model_rebuild()
Mesh.model_rebuild()
SpatialField.model_rebuild()
FieldStateVariable.model_rebuild()
DifferentialOperator.model_rebuild()
BoundaryCondition.model_rebuild()
PDESolver.model_rebuild()
PDE.model_rebuild()
