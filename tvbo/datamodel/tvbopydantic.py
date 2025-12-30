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
    Any,
    ClassVar,
    Literal,
    Optional,
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



metamodel_version = "None"
version = "None"


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

    @model_serializer(mode='wrap', when_used='unless-none')
    def treat_empty_lists_as_none(
            self, handler: SerializerFunctionWrapHandler,
            info: SerializationInfo) -> dict[str, Any]:
        if info.exclude_none:
            _instance = self.model_copy()
            for field, field_info in type(_instance).model_fields.items():
                if getattr(_instance, field) == [] and not(
                        field_info.is_required()):
                    setattr(_instance, field, None)
        else:
            _instance = self
        return handler(_instance, info)



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
linkml_meta = LinkMLMeta({'default_prefix': 'tvbo',
     'default_range': 'string',
     'description': 'Metadata schema for simulation studies using The Virtual '
                    'Brain neuroinformatics platform or other dynamic network '
                    'models of large-scale brain activity.',
     'id': 'https://w3id.org/tvbo',
     'imports': ['linkml:types', 'SANDS', 'tvb_dbs'],
     'name': 'tvb-datamodel',
     'prefixes': {'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'prov': {'prefix_prefix': 'prov',
                           'prefix_reference': 'http://www.w3.org/ns/prov#'},
                  'rdfs': {'prefix_prefix': 'rdfs',
                           'prefix_reference': 'http://www.w3.org/2000/01/rdf-schema#'},
                  'tvbo': {'prefix_prefix': 'tvbo',
                           'prefix_reference': 'http://www.thevirtualbrain.org/tvb-o/'}},
     'source_file': 'schema/tvbo_datamodel.yaml',
     'title': 'The Virtual Brain Data Model'} )

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


class RequirementRole(str, Enum):
    engine = "engine"
    """
    Primary simulation/processing engine
    """
    runtime = "runtime"
    """
    General runtime dependency
    """
    analysis = "analysis"
    """
    Post-processing / analysis tool
    """
    dev = "dev"
    """
    Development / build dependency
    """
    optional = "optional"
    """
    Optional or extra feature dependency
    """


class EnvironmentType(str, Enum):
    conda = "conda"


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



class Coordinate(ConfiguredBaseModel):
    """
    A 3D coordinate with X, Y, Z values.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://openminds.ebrains.eu/sands/BrainAtlas'})

    coordinateSpace: Optional[str] = Field(default=None, description="""Add the common coordinate space used for this brain atlas version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'BrainAtlas']} })
    x: Optional[float] = Field(default=None, description="""X coordinate""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'Matrix']} })
    y: Optional[float] = Field(default=None, description="""Y coordinate""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'Matrix']} })
    z: Optional[float] = Field(default=None, description="""Z coordinate""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate']} })


class BrainAtlas(ConfiguredBaseModel):
    """
    A schema for representing a version of a brain atlas.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'atom:atlas/Atlas',
         'from_schema': 'https://openminds.ebrains.eu/sands/BrainAtlas'})

    coordinateSpace: Optional[str] = Field(default=None, description="""Add the common coordinate space used for this brain atlas version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'BrainAtlas']} })
    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    abbreviation: Optional[str] = Field(default=None, description="""Slot for the abbreviation of a resource.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'CommonCoordinateSpace', 'ParcellationEntity']} })
    author: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas']} })
    isVersionOf: Optional[str] = Field(default=None, description="""Linked type for the version of a brain atlas or coordinate space.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas'],
         'union_of': ['BrainAtlas',
                      'CommonCoordinateSpace',
                      'ParcellationTerminology',
                      'ParcellationEntity']} })
    versionIdentifier: Optional[str] = Field(default=None, description="""Enter the version identifier of this brain atlas or coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'ParcellationEntity', 'ParcellationTerminology']} })
    terminology: Optional[ParcellationTerminology] = Field(default=None, description="""Add the parcellation terminology version used for this brain atlas version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas']} })


class CommonCoordinateSpace(ConfiguredBaseModel):
    """
    A schema for representing a version of a common coordinate space.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'atom:atlas/Transformation',
         'from_schema': 'https://openminds.ebrains.eu/sands/BrainAtlas'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    abbreviation: Optional[str] = Field(default=None, description="""Slot for the abbreviation of a resource.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'CommonCoordinateSpace', 'ParcellationEntity']} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    license: Optional[str] = Field(default=None, description="""Linked type for the license of the brain atlas or coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    anatomicalAxesOrientation: Optional[str] = Field(default=None, description="""Add the axes orientation in standard anatomical terms (XYZ).""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace']} })
    axesOrigin: Optional[str] = Field(default=None, description="""Enter the origin (central point where all axes intersect).""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace']} })
    nativeUnit: Optional[str] = Field(default=None, description="""Add the native unit that is used for this common coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace']} })
    defaultImage: Optional[list[str]] = Field(default=[], description="""Add all image files used as visual representation of this common coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace']} })


class ParcellationEntity(ConfiguredBaseModel):
    """
    A schema for representing a parcellation entity, which is an anatomical location or study target.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'categories': ['anatomicalLocation', 'studyTarget'],
         'class_uri': 'atom:atlas/Region',
         'from_schema': 'https://openminds.ebrains.eu/sands/BrainAtlas'})

    abbreviation: Optional[str] = Field(default=None, description="""Slot for the abbreviation of a resource.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'CommonCoordinateSpace', 'ParcellationEntity']} })
    alternateName: Optional[list[str]] = Field(default=[], description="""Enter any alternate names, including abbreviations, for this entity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity'], 'slot_uri': 'atom:atlas/hasName'} })
    lookupLabel: Optional[int] = Field(default=None, description="""Enter the label used for looking up this entity in the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity'], 'slot_uri': 'atom:atlas/lookupLabel'} })
    hasParent: Optional[list[str]] = Field(default=[], description="""Add all anatomical parent structures for this entity as defined within the corresponding brain atlas.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity'], 'slot_uri': 'atom:atlas/hasParent'} })
    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    ontologyIdentifier: Optional[list[str]] = Field(default=[], description="""Enter the internationalized resource identifier (IRI) to the related ontological terms.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity', 'ParcellationTerminology'],
         'slot_uri': 'atom:atlas/hasIlxId'} })
    versionIdentifier: Optional[str] = Field(default=None, description="""Enter the version identifier of this brain atlas or coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'ParcellationEntity', 'ParcellationTerminology']} })
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
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Stimulus',
                       'Matrix',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'NDArray',
                       'Mesh']} })
    ontologyIdentifier: Optional[list[str]] = Field(default=[], description="""Enter the internationalized resource identifier (IRI) to the related ontological terms.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity', 'ParcellationTerminology'],
         'slot_uri': 'atom:atlas/hasIlxId'} })
    versionIdentifier: Optional[str] = Field(default=None, description="""Enter the version identifier of this brain atlas or coordinate space version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas', 'ParcellationEntity', 'ParcellationTerminology']} })
    entities: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology']} })


class Dataset(ConfiguredBaseModel):
    """
    Collection of data related to a specific DBS study.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'http://www.thevirtualbrain.org/tvbo/dbs'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dataset_id: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    subjects: Optional[dict[str, Subject]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    clinical_scores: Optional[list[ClinicalScore]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset']} })
    coordinate_space: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'Subject',
                       'Electrode',
                       'EField',
                       'SpatialDomain',
                       'Mesh']} })


class Subject(ConfiguredBaseModel):
    """
    Human or animal subject receiving DBS.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'http://www.thevirtualbrain.org/tvbo/dbs'})

    subject_id: str = Field(default=..., description="""Unique identifier for a subject within a dataset.""", json_schema_extra = { "linkml_meta": {'aliases': ['subject code', 'subject label'],
         'domain_of': ['Subject', 'TimeSeries'],
         'exact_mappings': ['schema:identifier']} })
    age: Optional[float] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    sex: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    diagnosis: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    handedness: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    protocols: Optional[list[str]] = Field(default=[], description="""All DBS protocols assigned to this subject.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject']} })
    coordinate_space: Optional[str] = Field(default=None, description="""Coordinate space used for this subject's data""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'Subject',
                       'Electrode',
                       'EField',
                       'SpatialDomain',
                       'Mesh']} })


class Electrode(ConfiguredBaseModel):
    """
    Implanted DBS electrode and contact geometry.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'http://www.thevirtualbrain.org/tvbo/dbs'})

    electrode_id: Optional[str] = Field(default=None, description="""Unique identifier for this electrode""", json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    manufacturer: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    model: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode', 'SimulationExperiment', 'SimulationStudy']} })
    hemisphere: Optional[str] = Field(default="left", description="""Hemisphere of electrode (left/right)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationEntity', 'Electrode'], 'ifabsent': 'left'} })
    contacts: Optional[list[Contact]] = Field(default=[], description="""List of physical contacts along the electrode""", json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    head: Optional[Coordinate] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    tail: Optional[Coordinate] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    trajectory: Optional[list[Coordinate]] = Field(default=[], description="""The planned trajectory for electrode implantation""", json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    target_structure: Optional[str] = Field(default=None, description="""Anatomical target structure from a brain atlas""", json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })
    coordinate_space: Optional[str] = Field(default=None, description="""Coordinate space used for implantation planning""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'Subject',
                       'Electrode',
                       'EField',
                       'SpatialDomain',
                       'Mesh']} })
    recon_path: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode']} })


class Contact(ConfiguredBaseModel):
    """
    Individual contact on a DBS electrode.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'http://www.thevirtualbrain.org/tvbo/dbs'})

    contact_id: Optional[int] = Field(default=None, description="""Identifier (e.g., 0, 1, 2)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Contact']} })
    coordinate: Optional[Coordinate] = Field(default=None, description="""3D coordinate of the contact center in the defined coordinate space""", json_schema_extra = { "linkml_meta": {'domain_of': ['Contact']} })
    label: Optional[str] = Field(default=None, description="""Optional human-readable label (e.g., \"1a\")""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })


class StimulationSetting(ConfiguredBaseModel):
    """
    DBS parameters for a specific session.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'http://www.thevirtualbrain.org/tvbo/dbs'})

    electrode_reference: Optional[Electrode] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })
    amplitude: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })
    frequency: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })
    pulse_width: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })
    mode: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting', 'Exploration']} })
    active_contacts: Optional[list[int]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })
    efield: Optional[EField] = Field(default=None, description="""Metadata about the E-field result for this setting""", json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting']} })


class DBSProtocol(ConfiguredBaseModel):
    """
    A protocol describing DBS therapy, potentially bilateral or multi-lead.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'http://www.thevirtualbrain.org/tvbo/dbs'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    electrodes: Optional[list[Electrode]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['DBSProtocol']} })
    settings: Optional[list[StimulationSetting]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['DBSProtocol']} })
    timing_info: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSProtocol']} })
    notes: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DBSProtocol']} })
    clinical_improvement: Optional[list[ClinicalImprovement]] = Field(default=[], description="""Observed improvement relative to baseline based on a defined score.""", json_schema_extra = { "linkml_meta": {'domain_of': ['DBSProtocol']} })


class ClinicalScale(ConfiguredBaseModel):
    """
    A clinical assessment inventory or structured scale composed of multiple scores or items.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'http://www.thevirtualbrain.org/tvbo/dbs'})

    acronym: Optional[str] = Field(default=None, description="""Short abbreviation (e.g., UPDRS)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'ClinicalScore', 'Observation', 'Function']} })
    name: Optional[str] = Field(default=None, description="""Full name of the scale (e.g., Unified Parkinson’s Disease Rating Scale)""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    version: Optional[str] = Field(default=None, description="""Version of the instrument (e.g., 3.0)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'SoftwareEnvironment', 'SoftwareRequirement']} })
    domain: Optional[str] = Field(default=None, description="""Overall clinical domain (e.g., motor, cognition)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Parameter',
                       'PDE']} })
    reference: Optional[str] = Field(default=None, description="""DOI, PMID or persistent identifier""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'ClinicalScore', 'Tractogram']} })


class ClinicalScore(ConfiguredBaseModel):
    """
    Metadata about a clinical score or scale.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'http://www.thevirtualbrain.org/tvbo/dbs'})

    acronym: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'ClinicalScore', 'Observation', 'Function']} })
    name: Optional[str] = Field(default=None, description="""Full name of the score (e.g., Unified Parkinson's Disease Rating Scale - Part III)""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    domain: Optional[str] = Field(default=None, description="""Domain assessed (e.g. motor, mood, pain)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Parameter',
                       'PDE']} })
    reference: Optional[str] = Field(default=None, description="""PubMed ID, DOI, or other reference to the score definition""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'ClinicalScore', 'Tractogram']} })
    scale: Optional[ClinicalScale] = Field(default=None, description="""The scale this score belongs to, if applicable""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore']} })
    parent_score: Optional[ClinicalScore] = Field(default=None, description="""If this score is a subscore of a broader composite""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore']} })


class ClinicalImprovement(ConfiguredBaseModel):
    """
    Relative improvement on a defined clinical score.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'http://www.thevirtualbrain.org/tvbo/dbs'})

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
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'http://www.thevirtualbrain.org/tvbo/dbs'})

    volume_data: Optional[str] = Field(default=None, description="""Reference to raw or thresholded volume""", json_schema_extra = { "linkml_meta": {'domain_of': ['EField']} })
    coordinate_space: Optional[str] = Field(default=None, description="""Reference to a common coordinate space (e.g. MNI152)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'Subject',
                       'Electrode',
                       'EField',
                       'SpatialDomain',
                       'Mesh']} })
    threshold_applied: Optional[float] = Field(default=None, description="""Threshold value applied to the E-field simulation""", json_schema_extra = { "linkml_meta": {'domain_of': ['EField']} })


class Range(ConfiguredBaseModel):
    """
    Specifies a range for array generation, parameter bounds, or grid exploration.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    lo: Optional[str] = Field(default="0", description="""Lower bound or starting value. Can be a number or argument name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range'], 'ifabsent': 'string(0)'} })
    hi: Optional[str] = Field(default=None, description="""Upper bound or stopping value. Can be a number or argument name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range']} })
    step: Optional[str] = Field(default=None, description="""Step size. Can be: number, argument name, or expression.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range']} })
    n: Optional[int] = Field(default=None, description="""Number of points (alternative to step for grid exploration).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range']} })
    log_scale: Optional[bool] = Field(default=False, description="""Whether to use logarithmic spacing.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Range'], 'ifabsent': 'False'} })


class Equation(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator']} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    lhs: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    rhs: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    conditionals: Optional[list[ConditionalBlock]] = Field(default=[], description="""Conditional logic for piecewise equations.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    engine: Optional[SoftwareRequirement] = Field(default=None, description="""Primary engine (must appear in environment.requirements; migration target replacing deprecated 'software').""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    pycode: Optional[str] = Field(default=None, description="""Python code for the equation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation', 'Noise']} })
    latex: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation'], 'ifabsent': 'False'} })


class ConditionalBlock(ConfiguredBaseModel):
    """
    A single condition and its corresponding equation segment.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    condition: Optional[str] = Field(default=None, description="""The condition for this block (e.g., t > onset).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ConditionalBlock', 'Case']} })
    expression: Optional[str] = Field(default=None, description="""The equation to apply when the condition is met.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ConditionalBlock', 'DifferentialOperator']} })


class Stimulus(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Stimulus',
                       'Matrix',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'NDArray',
                       'Mesh']} })
    duration: Optional[float] = Field(default=1000, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'Integrator'], 'ifabsent': 'float(1000)'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    regions: Optional[AnyShapeArray[int]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus']} })
    weighting: Optional[AnyShapeArray[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus']} })


class TemporalApplicableEquation(Equation):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    time_dependent: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['TemporalApplicableEquation',
                       'SpatialField',
                       'BoundaryCondition'],
         'ifabsent': 'False'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    lhs: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    rhs: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    conditionals: Optional[list[ConditionalBlock]] = Field(default=[], description="""Conditional logic for piecewise equations.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    engine: Optional[SoftwareRequirement] = Field(default=None, description="""Primary engine (must appear in environment.requirements; migration target replacing deprecated 'software').""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation']} })
    pycode: Optional[str] = Field(default=None, description="""Python code for the equation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation', 'Noise']} })
    latex: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation'], 'ifabsent': 'False'} })


class Parcellation(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Parcellation', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    region_labels: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation']} })
    center_coordinates: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation']} })
    data_source: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation', 'Tractogram', 'Observation']} })
    atlas: BrainAtlas = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation']} })


class Tractogram(ConfiguredBaseModel):
    """
    Reference to tractography/diffusion MRI data used to derive structural connectivity
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Tractogram', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    data_source: Optional[str] = Field(default=None, description="""Path or URI to the tractography data file""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation', 'Tractogram', 'Observation']} })
    number_of_subjects: Optional[int] = Field(default=None, description="""Number of subjects in the tractography dataset""", json_schema_extra = { "linkml_meta": {'domain_of': ['Tractogram']} })
    acquisition: Optional[str] = Field(default=None, description="""Acquisition protocol or scanner information""", json_schema_extra = { "linkml_meta": {'domain_of': ['Tractogram']} })
    processing_pipeline: Optional[str] = Field(default=None, description="""Processing pipeline used to generate the tractography""", json_schema_extra = { "linkml_meta": {'domain_of': ['Tractogram']} })
    reference: Optional[str] = Field(default=None, description="""Publication or DOI reference for this tractography dataset""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'ClinicalScore', 'Tractogram']} })


class Matrix(ConfiguredBaseModel):
    """
    Adjacency matrix of a network.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Stimulus',
                       'Matrix',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'NDArray',
                       'Mesh']} })
    x: Optional[BrainRegionSeries] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'Matrix']} })
    y: Optional[BrainRegionSeries] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Coordinate', 'Matrix']} })
    values: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'BrainRegionSeries', 'SpatialField']} })


class BrainRegionSeries(ConfiguredBaseModel):
    """
    A series whose values represent latitude
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    values: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'BrainRegionSeries', 'SpatialField']} })


class Network(ConfiguredBaseModel):
    """
    Network specification with nodes, edges, and reusable coupling configurations. Supports both explicit node/edge representation and matrix-based connectivity (Connectome compatibility).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'aliases': ['Connectome', 'Graph', 'Connectivity'],
         'class_uri': 'tvbo:Network',
         'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    nodes: Optional[list[Node]] = Field(default=[], description="""List of nodes with individual dynamics (optional, for heterogeneous networks)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    edges: Optional[list[Edge]] = Field(default=[], description="""List of directed edges with coupling references (optional, for explicit edge definition)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    coupling_library: Optional[dict[str, Coupling]] = Field(default=None, description="""Reusable coupling configurations referenced by edges (e.g., 'instant', 'delayed', 'inhibitory')""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    number_of_regions: Optional[int] = Field(default=1, description="""Number of regions (derived from nodes if not set)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network'], 'ifabsent': 'integer(1)'} })
    number_of_nodes: Optional[int] = Field(default=1, description="""Number of nodes in the network (derived from nodes if not set)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network'], 'ifabsent': 'integer(1)'} })
    parcellation: Optional[Parcellation] = Field(default=None, description="""Brain parcellation/atlas reference""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    tractogram: Optional[str] = Field(default=None, description="""Reference to tractography data""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    normalization: Optional[Equation] = Field(default=None, description="""Normalization equation for connectivity weights""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    global_coupling_strength: Optional[Parameter] = Field(default=None, description="""Global scaling factor for all coupling weights""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    conduction_speed: Optional[Parameter] = Field(default=None, description="""Conduction speed for computing delays from distances""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })
    distance_unit: Optional[str] = Field(default="mm", description="""Unit for distances/lengths in the network (e.g., 'mm', 'm', 'cm')""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network'], 'ifabsent': 'string(mm)'} })
    time_unit: Optional[str] = Field(default="ms", description="""Default time unit for the network (e.g., 'ms', 's')""", json_schema_extra = { "linkml_meta": {'domain_of': ['Network'], 'ifabsent': 'string(ms)'} })
    edge_matrix_files: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Network']} })


class File(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    type: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['File', 'Aggregation', 'TuningObjective', 'TuningAlgorithm']} })
    path: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['File', 'DataSource']} })
    extension: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['File']} })


class Node(ConfiguredBaseModel):
    """
    A node in a network with its own dynamics and properties
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Node', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    id: int = Field(default=..., description="""Unique node identifier""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'SimulationExperiment']} })
    dynamics: Optional[str] = Field(default=None, description="""Dynamics model governing this node's behavior. Can be a reference (by name) or inline definition. If not provided, uses experiment's local_dynamics.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'SimulationExperiment']} })
    position: Optional[Coordinate] = Field(default=None, description="""Spatial coordinates (x, y, z) of the node""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node']} })
    region: Optional[str] = Field(default=None, description="""Brain region or anatomical label""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'SpatialDomain']} })
    parameters: Optional[list[str]] = Field(default=[], description="""Node-specific parameter overrides""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    initial_state: Optional[list[float]] = Field(default=[], description="""Initial values for state variables""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node']} })


class Edge(ConfiguredBaseModel):
    """
    A directed edge in a network with coupling and connectivity properties. Edge properties (weight, delay, distance) are stored in the parameters slot with optional units.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Edge', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    source: int = Field(default=..., description="""Source node ID""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge', 'Observation', 'Dynamics']} })
    target: int = Field(default=..., description="""Target node ID""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    source_var: Optional[str] = Field(default=None, description="""Output variable from source node to use (e.g., 'x_out'). If not specified, uses first output variable from source dynamics.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    target_var: Optional[str] = Field(default=None, description="""Input variable on target node to connect to (e.g., 'c_in'). If not specified, uses first coupling input from target dynamics.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge']} })
    coupling: Optional[str] = Field(default=None, description="""Coupling function for this edge. Can be a reference (by name) to coupling_library or inline definition. If not provided, uses experiment's default coupling.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge', 'SimulationExperiment']} })
    directed: Optional[bool] = Field(default=False, description="""Whether the edge is directed. If false, represents a symmetric/bidirectional connection.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge'], 'ifabsent': 'False'} })


class Observation(ConfiguredBaseModel):
    """
    Unified class for all observation/measurement specifications. Covers monitors (BOLD, EEG), tuning observables, and derived quantities. Pipeline is a sequence of Functions with input → output flow.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Observation', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    acronym: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'ClinicalScore', 'Observation', 'Function']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    environment: Optional[SoftwareEnvironment] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'SimulationExperiment', 'PDESolver']} })
    time_scale: Optional[str] = Field(default="ms", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'Integrator'], 'ifabsent': 'ms'} })
    source: Optional[str] = Field(default=None, description="""State variable to observe (e.g., S_e for excitatory activity)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Edge', 'Observation', 'Dynamics']} })
    source_observation: Optional[str] = Field(default=None, description="""Derive from another observation (e.g., FC derived from Bold)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    period: Optional[float] = Field(default=None, description="""Sampling period for monitors (ms). For BOLD: TR in ms.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    downsample_period: Optional[float] = Field(default=None, description="""Intermediate downsampling period (ms). For BOLD: typically matches dt.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    voi: Optional[int] = Field(default=None, description="""Variable of interest index (which state variable to monitor). Default: 0.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    imaging_modality: Optional[ImagingModality] = Field(default=None, description="""Type of imaging modality (BOLD, EEG, MEG, etc.)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    warmup_source: Optional[str] = Field(default=None, description="""Reference to transient simulation result for history initialization (e.g., 'result_init').""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'ClassReference']} })
    data_source: Optional[DataSource] = Field(default=None, description="""Load data from external source (file, database, API). When specified, this observation represents empirical/external data rather than simulated data. Enables unified treatment of all data.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parcellation', 'Tractogram', 'Observation']} })
    skip_t: Optional[int] = Field(default=None, description="""Number of samples to skip at the start (transient removal). For FC: typically 10-20 TRs.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    aggregation: Optional[AggregationType] = Field(default=None, description="""How to aggregate over time""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'Coupling']} })
    window_size: Optional[int] = Field(default=None, description="""Number of samples for windowed aggregation""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    pipeline: Optional[list[FunctionCall]] = Field(default=[], description="""Ordered sequence of Functions. Each Function transforms input → output.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })
    class_reference: Optional[ClassReference] = Field(default=None, description="""Direct class reference (alternative to pipeline). Use for external library classes like tvboptim.Bold, custom monitors, or any callable class. The class is instantiated with constructor_args and called with call_args. Example: {name: Bold, module: tvboptim.observations.tvb_monitors.bold, constructor_args: [{name: period, value: 1000.0}]}""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation']} })


class Dynamics(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'aliases': ['NeuralMassModel'],
         'class_uri': 'tvbo:Dynamics',
         'comments': ['Successor class replacing deprecated NeuralMassModel.'],
         'from_schema': 'https://w3id.org/tvbo',
         'slot_usage': {'name': {'ifabsent': 'Generic2dOscillator', 'name': 'name'},
                        'system_type': {'ifabsent': 'continuous',
                                        'name': 'system_type'}}})

    has_reference: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    name: str = Field(default="Generic2dOscillator", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage'],
         'ifabsent': 'Generic2dOscillator'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    iri: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function']} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    source: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Edge', 'Observation', 'Dynamics']} })
    references: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'SimulationExperiment']} })
    derived_parameters: Optional[dict[str, DerivedParameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    derived_variables: Optional[dict[str, DerivedVariable]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    coupling_terms: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    coupling_inputs: Optional[dict[str, CouplingInput]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    state_variables: Optional[dict[str, StateVariable]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    modified: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    output: Optional[list[str]] = Field(default=[], description="""Output variable names to include in simulation results. References to state_variables or derived_variables by name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function', 'FunctionCall']} })
    derived_from_model: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    number_of_modes: Optional[int] = Field(default=1, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics'], 'ifabsent': 'integer(1)'} })
    local_coupling_term: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    functions: Optional[dict[str, Function]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'SimulationExperiment', 'PDE']} })
    stimulus: Optional[Stimulus] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    modes: Optional[dict[str, Dynamics]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics']} })
    system_type: Optional[SystemType] = Field(default=SystemType.continuous, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics'], 'ifabsent': 'continuous'} })


class StateVariable(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:StateVariable', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    symbol: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable',
                       'Parameter',
                       'DerivedParameter',
                       'DerivedVariable']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator']} })
    domain: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Parameter',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    variable_of_interest: Optional[bool] = Field(default=True, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'True'} })
    coupling_variable: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'False'} })
    noise: Optional[Noise] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Integrator']} })
    stimulation_variable: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })
    boundaries: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })
    initial_value: Optional[float] = Field(default=0.1, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'SpatialField'], 'ifabsent': 'float(0.1)'} })
    history: Optional[TimeSeries] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })


class Distribution(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    dependencies: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Distribution']} })
    correlation: Optional[Matrix] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Distribution']} })


class Parameter(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Parameter', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    symbol: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable',
                       'Parameter',
                       'DerivedParameter',
                       'DerivedVariable']} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator']} })
    value: Optional[float] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'Argument', 'BoundaryCondition']} })
    default: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    domain: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Parameter',
                       'PDE']} })
    reported_optimum: Optional[float] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    comment: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    heterogeneous: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    free: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    shape: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'NDArray']} })
    explored_values: Optional[AnyShapeArray[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })


class CouplingInput(ConfiguredBaseModel):
    """
    Specification of a coupling input channel for multi-coupling dynamics
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dimension: Optional[int] = Field(default=1, description="""Dimensionality of the coupling input (number of coupled values)""", json_schema_extra = { "linkml_meta": {'domain_of': ['CouplingInput'], 'ifabsent': 'integer(1)'} })


class Argument(ConfiguredBaseModel):
    """
    A function argument with explicit value specification. Value can be: literal (number/string), reference to input (input.key), or cross-observation reference (observation_name.output_key).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Argument', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    value: Optional[Union[float, int, str]] = Field(default=None, description="""Argument value. Can be: - Literal: 1.0, \"string\", etc. - Input reference: \"input.frequencies\" (from source_observation outputs) - Cross-observation: \"target_frequencies.peak_freqs\" (from another observation)""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'float'}, {'range': 'integer'}, {'range': 'string'}],
         'domain_of': ['Parameter', 'Argument', 'BoundaryCondition']} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })


class Function(ConfiguredBaseModel):
    """
    A function with explicit input → transformation → output flow. Can be equation-based (symbolic) or software-based (callable). In a pipeline, functions are chained: output of one becomes input of next.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Function', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    acronym: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'ClinicalScore', 'Observation', 'Function']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
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
                       'DifferentialOperator']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    requirements: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'SoftwareEnvironment', 'PDESolver']} })
    input: Optional[str] = Field(default=None, description="""Simple input reference: name of previous function's output in pipeline. For multi-argument functions, use arguments with value references instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function']} })
    output: Optional[str] = Field(default=None, description="""Name for this function's output (referenced by subsequent functions)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function', 'FunctionCall']} })
    iri: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function']} })
    arguments: Optional[list[Argument]] = Field(default=[], description="""Parameters/arguments for the function""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    output_equation: Optional[Equation] = Field(default=None, description="""Output transformation equation (if equation-based)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function']} })
    source_code: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Function']} })
    callable: Optional[Callable] = Field(default=None, description="""Software implementation reference (if software-based)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    apply_on_dimension: Optional[DimensionType] = Field(default=None, description="""Which dimension to apply the transformation on""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    time_range: Optional[Range] = Field(default=None, description="""Time range for generated TimeSeries (for kernel generators). Equation is evaluated at each time point.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function']} })


class Aggregation(ConfiguredBaseModel):
    """
    Specifies how to aggregate values across a dimension. Used for loss functions to define per-element loss with reduction.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    over: Optional[DimensionType] = Field(default=None, description="""Dimension to aggregate over (e.g., node, time, state)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Aggregation']} })
    type: Optional[ReductionType] = Field(default=ReductionType.mean, description="""Aggregation operation (mean, sum, max, min, none)""", json_schema_extra = { "linkml_meta": {'domain_of': ['File', 'Aggregation', 'TuningObjective', 'TuningAlgorithm'],
         'ifabsent': 'string(mean)'} })


class LossFunction(Function):
    """
    A loss function for optimization with optional aggregation. Extends Function with aggregation specification for per-element losses.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    aggregate: Optional[Aggregation] = Field(default=None, description="""How to aggregate the loss across dimensions. Example: aggregate.over=node, aggregate.type=mean computes loss per node, then averages.""", json_schema_extra = { "linkml_meta": {'domain_of': ['LossFunction', 'FunctionCall']} })
    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    acronym: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'ClinicalScore', 'Observation', 'Function']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
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
                       'DifferentialOperator']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    requirements: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'SoftwareEnvironment', 'PDESolver']} })
    input: Optional[str] = Field(default=None, description="""Simple input reference: name of previous function's output in pipeline. For multi-argument functions, use arguments with value references instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function']} })
    output: Optional[str] = Field(default=None, description="""Name for this function's output (referenced by subsequent functions)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function', 'FunctionCall']} })
    iri: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function']} })
    arguments: Optional[list[Argument]] = Field(default=[], description="""Parameters/arguments for the function""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    output_equation: Optional[Equation] = Field(default=None, description="""Output transformation equation (if equation-based)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function']} })
    source_code: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Function']} })
    callable: Optional[Callable] = Field(default=None, description="""Software implementation reference (if software-based)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    apply_on_dimension: Optional[DimensionType] = Field(default=None, description="""Which dimension to apply the transformation on""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    time_range: Optional[Range] = Field(default=None, description="""Time range for generated TimeSeries (for kernel generators). Equation is evaluated at each time point.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function']} })


class FunctionCall(ConfiguredBaseModel):
    """
    Invocation of a function in a pipeline. Can reference a defined Function by name, OR inline a callable directly for external library functions. OR inline a class_call for classes that need instantiation then calling. Use arguments to specify inputs by name (referencing previous outputs or values).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    function: Optional[str] = Field(default=None, description="""Reference to a defined Function (by name)""", json_schema_extra = { "linkml_meta": {'domain_of': ['FunctionCall', 'Noise']} })
    callable: Optional[Callable] = Field(default=None, description="""Direct callable specification (alternative to function reference)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    class_call: Optional[ClassReference] = Field(default=None, description="""Class instantiation and call (alternative to callable/function). Use for external library classes that need __init__ then __call__. Example: Bold monitor from tvboptim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['FunctionCall']} })
    output: Optional[str] = Field(default=None, description="""Name for this step's output (referenced by subsequent functions)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'Function', 'FunctionCall']} })
    apply_on_dimension: Optional[DimensionType] = Field(default=None, description="""Dimension to apply function over (generates vmap in code). E.g., 'node' applies per-node.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })
    aggregate: Optional[Aggregation] = Field(default=None, description="""How to aggregate the result across dimensions. Example: aggregate.over=node, aggregate.type=mean applies function per node, then averages. Used in loss functions.""", json_schema_extra = { "linkml_meta": {'domain_of': ['LossFunction', 'FunctionCall']} })
    arguments: Optional[list[Argument]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'FunctionCall']} })


class Callable(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    module: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Callable']} })
    software: Optional[SoftwareRequirement] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Callable', 'SimulationExperiment']} })


class ClassReference(Callable):
    """
    Reference to a class that can be instantiated and called. Used for external library classes (e.g., tvboptim.Bold, custom monitors). The class is instantiated with constructor_args, then called with call_args. Generalizable pattern: works for tvboptim, TVB, or any Python class.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    constructor_args: Optional[list[Argument]] = Field(default=[], description="""Arguments passed to __init__ when instantiating the class. Example: period=1000.0, downsample_period=4.0 for Bold monitor.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClassReference']} })
    call_args: Optional[list[Argument]] = Field(default=[], description="""Arguments passed when calling the instance (__call__). Usually the input data from simulation result. Example: result (simulation output array).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClassReference']} })
    warmup_source: Optional[str] = Field(default=None, description="""Reference to transient simulation result for history initialization. Some monitors (e.g., Bold) require history from warmup simulation. Value should reference a simulation result name (e.g., 'result_init').""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'ClassReference']} })
    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    module: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Callable']} })
    software: Optional[SoftwareRequirement] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Callable', 'SimulationExperiment']} })


class Case(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    condition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ConditionalBlock', 'Case']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator']} })


class DerivedParameter(Parameter):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:DerivedParameter', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    symbol: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable',
                       'Parameter',
                       'DerivedParameter',
                       'DerivedVariable']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator']} })
    value: Optional[float] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'Argument', 'BoundaryCondition']} })
    default: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    domain: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Parameter',
                       'PDE']} })
    reported_optimum: Optional[float] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    comment: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    heterogeneous: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    free: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })
    shape: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'NDArray']} })
    explored_values: Optional[AnyShapeArray[float]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter']} })


class DerivedVariable(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:DerivedVariable', 'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    symbol: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable',
                       'Parameter',
                       'DerivedParameter',
                       'DerivedVariable']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    conditional: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['DerivedVariable'], 'ifabsent': 'False'} })
    cases: Optional[list[Case]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['DerivedVariable']} })


class Noise(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Noise', 'from_schema': 'https://w3id.org/tvbo'})

    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
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
    seed: Optional[int] = Field(default=42, json_schema_extra = { "linkml_meta": {'domain_of': ['Noise'], 'ifabsent': 'integer(42)'} })
    random_state: Optional[RandomStream] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Noise']} })
    intensity: Optional[Parameter] = Field(default=None, description="""Optional scalar or vector intensity parameter for noise.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Noise']} })
    function: Optional[Function] = Field(default=None, description="""Optional functional form of the noise (callable specification).""", json_schema_extra = { "linkml_meta": {'domain_of': ['FunctionCall', 'Noise']} })
    pycode: Optional[str] = Field(default=None, description="""Inline Python code representation of the noise process.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation', 'Noise']} })
    targets: Optional[dict[str, StateVariable]] = Field(default=None, description="""State variables this noise applies to; if omitted, applies globally.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Noise']} })


class RandomStream(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:RandomStream', 'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Stimulus',
                       'Matrix',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'NDArray',
                       'Mesh']} })


class DataSource(ConfiguredBaseModel):
    """
    Specification for loading external/empirical data.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    path: Optional[str] = Field(default=None, description="""File path or URI to the data""", json_schema_extra = { "linkml_meta": {'domain_of': ['File', 'DataSource']} })
    loader: Optional[Callable] = Field(default=None, description="""Callable that loads the data (e.g., load_functional_connectivity)""", json_schema_extra = { "linkml_meta": {'domain_of': ['DataSource']} })
    format: Optional[str] = Field(default=None, description="""Data format: 'npy', 'mat', 'csv', 'nifti', etc.""", json_schema_extra = { "linkml_meta": {'domain_of': ['DataSource']} })
    key: Optional[str] = Field(default=None, description="""Key/variable name within the file (for .mat, .npz, etc.)""", json_schema_extra = { "linkml_meta": {'domain_of': ['DataSource', 'SimulationStudy']} })
    preprocessing: Optional[Function] = Field(default=None, description="""Optional preprocessing to apply after loading""", json_schema_extra = { "linkml_meta": {'domain_of': ['DataSource']} })


class OptimizationStage(ConfiguredBaseModel):
    """
    A single stage in a multi-stage optimization workflow. Stages run sequentially, with each stage potentially using different parameters, shapes, learning rates, and algorithms.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    free_parameters: Optional[list[str]] = Field(default=[], description="""Parameters to optimize in this stage. Use 'shape' attribute to specify scalar vs regional. Example: {name: w, shape: \"(n_nodes,)\"} for heterogeneous.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization']} })
    algorithm: Optional[str] = Field(default="adam", description="""Optimizer for this stage: 'adam', 'adamw', 'sgd', etc.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization'], 'ifabsent': 'string(adam)'} })
    learning_rate: Optional[float] = Field(default=0.001, json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization', 'TuningAlgorithm'],
         'ifabsent': 'float(0.001)'} })
    max_iterations: Optional[int] = Field(default=100, json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization'], 'ifabsent': 'integer(100)'} })
    hyperparameters: Optional[dict[str, Parameter]] = Field(default=None, description="""Stage-specific hyperparameters (e.g., b2=0.9999 for adam)""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization', 'TuningAlgorithm']} })
    freeze_parameters: Optional[list[str]] = Field(default=[], description="""Parameters from previous stages to freeze (keep at optimized value but not update)""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage']} })
    warmup_from: Optional[str] = Field(default=None, description="""Previous stage to initialize from. Final values from that stage become initial values for this stage.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage']} })


class Optimization(ConfiguredBaseModel):
    """
    Configuration for parameter optimization. Loss equation references observations directly by name. Both simulated and empirical data are observations (use data_source for empirical).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    loss: Optional[FunctionCall] = Field(default=None, description="""Loss function call. Uses FunctionCall to either: 1. Reference existing function: function: rmse 2. Inline callable: callable: {module: ..., name: ...} Arguments specify inputs (simulated_fc, empirical_fc, etc.)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization']} })
    stages: Optional[dict[str, OptimizationStage]] = Field(default=None, description="""Ordered list of optimization stages. Stages run sequentially. Stage n+1 starts from optimized values of stage n. Example: Stage 1 (global params) -> Stage 2 (regional params).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Optimization']} })
    free_parameters: Optional[list[str]] = Field(default=[], description="""Parameters to optimize (single-stage mode, ignored if stages defined)""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization']} })
    algorithm: Optional[str] = Field(default="adam", description="""Optimizer (single-stage mode): 'adam', 'adamw', 'sgd', 'lbfgs', etc.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization'], 'ifabsent': 'string(adam)'} })
    learning_rate: Optional[float] = Field(default=0.001, description="""Learning rate (single-stage mode)""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization', 'TuningAlgorithm'],
         'ifabsent': 'float(0.001)'} })
    max_iterations: Optional[int] = Field(default=100, description="""Max iterations (single-stage mode)""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization'], 'ifabsent': 'integer(100)'} })
    hyperparameters: Optional[dict[str, Parameter]] = Field(default=None, description="""Algorithm-specific hyperparameters (single-stage mode)""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization', 'TuningAlgorithm']} })


class Exploration(ConfiguredBaseModel):
    """
    Parameter space exploration (grid search, sweep).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    parameters: dict[str, Parameter] = Field(default=..., description="""Parameters with domain ranges to explore (uses domain.lo, domain.hi, domain.n)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    mode: Optional[str] = Field(default="product", description="""Combination mode: 'product' (full grid), 'zip' (paired)""", json_schema_extra = { "linkml_meta": {'domain_of': ['StimulationSetting', 'Exploration'],
         'ifabsent': 'string(product)'} })
    observable: Optional[FunctionCall] = Field(default=None, description="""Observable to compute at each point. Use function: obs_name for simple observation, or function: func_name + arguments for FunctionCall.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Exploration']} })
    n_parallel: Optional[int] = Field(default=1, description="""Parallel evaluations""", json_schema_extra = { "linkml_meta": {'domain_of': ['Exploration'], 'ifabsent': 'integer(1)'} })


class UpdateRule(ConfiguredBaseModel):
    """
    Defines how a parameter is updated based on observables. Represents iterative learning rules like FIC or EIB updates.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    target_parameter: Parameter = Field(default=..., description="""The parameter to update (e.g., J_i, wLRE)""", json_schema_extra = { "linkml_meta": {'domain_of': ['UpdateRule']} })
    equation: Equation = Field(default=..., description="""Update equation (e.g., 'J_i + eta * delta')""", json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator']} })
    bounds: Optional[Range] = Field(default=None, description="""Constraints on parameter values after update""", json_schema_extra = { "linkml_meta": {'domain_of': ['UpdateRule']} })
    requires: Optional[list[str]] = Field(default=[], description="""Observables required by this update rule""", json_schema_extra = { "linkml_meta": {'domain_of': ['UpdateRule']} })


class TuningObjective(ConfiguredBaseModel):
    """
    Defines what the tuning algorithm optimizes for. Can be an activity target (FIC) or a connectivity target (EIB).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    type: Optional[str] = Field(default=None, description="""Type of objective: 'activity_target', 'fc_matching', 'custom'""", json_schema_extra = { "linkml_meta": {'domain_of': ['File', 'Aggregation', 'TuningObjective', 'TuningAlgorithm']} })
    target_variable: Optional[str] = Field(default=None, description="""State variable for activity targets (e.g., S_e)""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningObjective']} })
    target_value: Optional[float] = Field(default=None, description="""Target value for activity objectives""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningObjective']} })
    target_data: Optional[str] = Field(default=None, description="""Reference to empirical data observation for matching objectives""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningObjective']} })
    metric: Optional[Equation] = Field(default=None, description="""Metric equation for matching (e.g., correlation, rmse)""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningObjective']} })


class TuningAlgorithm(ConfiguredBaseModel):
    """
    A complete specification of an iterative parameter tuning algorithm. Combines update rules, objectives, observables, and hyperparameters.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    type: Optional[str] = Field(default=None, description="""Algorithm type: 'fic', 'eib', 'homeostatic', 'custom'""", json_schema_extra = { "linkml_meta": {'domain_of': ['File', 'Aggregation', 'TuningObjective', 'TuningAlgorithm']} })
    objective: Optional[TuningObjective] = Field(default=None, description="""What the algorithm optimizes for""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningAlgorithm']} })
    observables: Optional[dict[str, Observation]] = Field(default=None, description="""Quantities computed from simulation for update rules""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningAlgorithm']} })
    update_rules: dict[str, UpdateRule] = Field(default=..., description="""How parameters are updated each iteration""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningAlgorithm']} })
    hyperparameters: Optional[dict[str, Parameter]] = Field(default=None, description="""Additional algorithm-specific parameters""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization', 'TuningAlgorithm']} })
    learning_rate: Optional[float] = Field(default=None, description="""Learning rate (eta) for the tuning algorithm""", json_schema_extra = { "linkml_meta": {'domain_of': ['OptimizationStage', 'Optimization', 'TuningAlgorithm']} })
    n_iterations: Optional[int] = Field(default=None, description="""Number of iterations to run""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningAlgorithm']} })
    learning_rate_schedule: Optional[str] = Field(default=None, description="""Learning rate schedule: 'constant', 'linear', 'exponential'""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningAlgorithm']} })
    simulation_period: Optional[float] = Field(default=None, description="""Duration of each simulation step (e.g., one BOLD TR)""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningAlgorithm']} })
    apply_every: Optional[int] = Field(default=1, description="""Apply update every N iterations""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningAlgorithm'], 'ifabsent': 'integer(1)'} })
    depends_on: Optional[list[str]] = Field(default=[], description="""Other algorithms that must run first (e.g., EIB depends on FIC)""", json_schema_extra = { "linkml_meta": {'domain_of': ['TuningAlgorithm']} })


class Integrator(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    time_scale: Optional[str] = Field(default="ms", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'Integrator'], 'ifabsent': 'ms'} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    duration: Optional[float] = Field(default=1000, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus', 'Integrator'], 'ifabsent': 'float(1000)'} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    method: Optional[str] = Field(default="euler", description="""Integration method (euler, heun, rk4, etc.)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator'], 'ifabsent': 'string(euler)'} })
    step_size: Optional[float] = Field(default=0.01220703125, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator'], 'ifabsent': 'float(0.01220703125)'} })
    steps: Optional[int] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator']} })
    noise: Optional[Noise] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Integrator']} })
    state_wise_sigma: Optional[list[float]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator']} })
    transient_time: Optional[float] = Field(default=0, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator'], 'ifabsent': 'float(0)'} })
    scipy_ode_base: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator'], 'ifabsent': 'False'} })
    number_of_stages: Optional[int] = Field(default=1, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator'], 'ifabsent': 'integer(1)'} })
    intermediate_expressions: Optional[dict[str, DerivedVariable]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator']} })
    update_expression: Optional[DerivedVariable] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator']} })
    delayed: Optional[bool] = Field(default=True, json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator', 'Coupling'], 'ifabsent': 'True'} })


class Coupling(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Coupling',
         'from_schema': 'https://w3id.org/tvbo',
         'slot_usage': {'name': {'ifabsent': 'Linear', 'name': 'name'}}})

    name: str = Field(default="Linear", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage'],
         'ifabsent': 'Linear'} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    coupling_function: Optional[Equation] = Field(default=None, description="""Mathematical function defining the coupling""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    sparse: Optional[bool] = Field(default=False, description="""Whether the coupling uses sparse representations""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling'], 'ifabsent': 'False'} })
    pre_expression: Optional[Equation] = Field(default=None, description="""Pre-processing expression applied before coupling""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    post_expression: Optional[Equation] = Field(default=None, description="""Post-processing expression applied after coupling""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    incoming_states: Optional[list[str]] = Field(default=[], description="""References to state variables from connected nodes (source)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    local_states: Optional[list[str]] = Field(default=[], description="""References to state variables from local node (target)""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    delayed: Optional[bool] = Field(default=True, description="""Whether coupling includes transmission delays""", json_schema_extra = { "linkml_meta": {'domain_of': ['Integrator', 'Coupling'], 'ifabsent': 'True'} })
    inner_coupling: Optional[Coupling] = Field(default=None, description="""For hierarchical coupling: inner coupling applied at regional level""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    region_mapping: Optional[RegionMapping] = Field(default=None, description="""For hierarchical coupling: vertex-to-region mapping for aggregation""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    regional_connectivity: Optional[Network] = Field(default=None, description="""For hierarchical coupling: region-to-region connectivity with weights and delays""", json_schema_extra = { "linkml_meta": {'domain_of': ['Coupling']} })
    aggregation: Optional[str] = Field(default=None, description="""For hierarchical coupling: aggregation method ('sum', 'mean', 'max') or custom Function""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'string'}, {'range': 'Function'}],
         'domain_of': ['Observation', 'Coupling']} })
    distribution: Optional[str] = Field(default=None, description="""For hierarchical coupling: distribution method ('broadcast', 'weighted') or custom Function""", json_schema_extra = { "linkml_meta": {'any_of': [{'range': 'string'}, {'range': 'Function'}],
         'domain_of': ['Coupling']} })


class RegionMapping(ConfiguredBaseModel):
    """
    Maps vertices to parent regions for hierarchical/aggregated coupling
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Stimulus',
                       'Matrix',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'NDArray',
                       'Mesh']} })
    vertex_to_region: Optional[list[int]] = Field(default=[], description="""Array mapping each vertex index to its parent region index. Can use dataLocation instead for large arrays.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RegionMapping']} })
    n_vertices: Optional[int] = Field(default=None, description="""Total number of vertices""", json_schema_extra = { "linkml_meta": {'domain_of': ['RegionMapping']} })
    n_regions: Optional[int] = Field(default=None, description="""Total number of regions""", json_schema_extra = { "linkml_meta": {'domain_of': ['RegionMapping']} })


class Sample(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    groups: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Sample']} })
    size: Optional[int] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Sample']} })


class ExecutionConfig(ConfiguredBaseModel):
    """
    Configuration for computational execution (parallelization, precision, hardware).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    n_workers: Optional[int] = Field(default=1, description="""Number of parallel workers (maps to pmap devices in JAX, processes in multiprocessing)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig'], 'ifabsent': 'integer(1)'} })
    n_threads: Optional[int] = Field(default=-1, description="""Number of CPU threads per worker (-1 = auto-detect)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig'], 'ifabsent': 'integer(-1)'} })
    precision: Optional[str] = Field(default="float64", description="""Floating point precision: 'float32' or 'float64'""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig'], 'ifabsent': 'string(float64)'} })
    accelerator: Optional[str] = Field(default="cpu", description="""Hardware accelerator: 'cpu', 'gpu', 'tpu'""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig'], 'ifabsent': 'string(cpu)'} })
    batch_size: Optional[int] = Field(default=None, description="""Batch size for vectorized operations (None = auto)""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExecutionConfig']} })


class SimulationExperiment(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'tvbo:Simulation',
         'from_schema': 'https://w3id.org/tvbo',
         'tree_root': True})

    model: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode', 'SimulationExperiment', 'SimulationStudy']} })
    id: int = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'SimulationExperiment']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    additional_equations: Optional[list[Equation]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    local_dynamics: Optional[Dynamics] = Field(default=None, description="""Default dynamics model for all nodes (used when node.dynamics not specified or as fallback)""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    dynamics: Optional[dict[str, Dynamics]] = Field(default=None, description="""Dictionary of dynamics models keyed by name. Nodes reference these by name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'SimulationExperiment']} })
    integration: Optional[Integrator] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    connectivity: Optional[Network] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    network: Optional[Network] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    coupling: Optional[Coupling] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Edge', 'SimulationExperiment']} })
    observations: Optional[dict[str, Observation]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    functions: Optional[dict[str, Function]] = Field(default=None, description="""Reusable function definitions. Referenced by name in observation pipelines. Enables DRY: define compute_fc once, use in both simulated and empirical paths.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'SimulationExperiment', 'PDE']} })
    stimulation: Optional[Stimulus] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    field_dynamics: Optional[PDE] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    optimization: Optional[dict[str, Optimization]] = Field(default=None, description="""Parameter optimization configurations""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    explorations: Optional[dict[str, Exploration]] = Field(default=None, description="""Parameter exploration/grid search specifications""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    tuning_algorithms: Optional[dict[str, TuningAlgorithm]] = Field(default=None, description="""Iterative parameter tuning algorithms (FIC, EIB, etc.)""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    environment: Optional[SoftwareEnvironment] = Field(default=None, description="""Execution environment (collection of requirements).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'SimulationExperiment', 'PDESolver']} })
    execution: Optional[ExecutionConfig] = Field(default=None, description="""Computational execution configuration (parallelization, devices).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationExperiment']} })
    software: Optional[SoftwareRequirement] = Field(default=None, description="""(Deprecated) Single software requirement; prefer 'environment' with aggregated requirements.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Callable', 'SimulationExperiment']} })
    references: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'SimulationExperiment']} })


class SimulationStudy(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    derived_from: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy']} })
    model: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Electrode', 'SimulationExperiment', 'SimulationStudy']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    key: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['DataSource', 'SimulationStudy']} })
    title: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy']} })
    year: Optional[int] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy']} })
    doi: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy', 'SoftwarePackage']} })
    sample: Optional[Sample] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy']} })
    simulation_experiments: Optional[list[SimulationExperiment]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy']} })


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
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Stimulus',
                       'Matrix',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'NDArray',
                       'Mesh']} })
    data: Optional[Matrix] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    time: Optional[Matrix] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    sampling_rate: Optional[float] = Field(default=None, description="""Sampling rate in Hz.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    sampling_period: Optional[float] = Field(default=None, description="""Time between samples (inverse of sampling_rate).""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    sampling_period_unit: Optional[str] = Field(default="ms", description="""Unit of the sampling period (e.g., 'ms', 's').""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries'], 'ifabsent': 'ms'} })
    unit: Optional[str] = Field(default=None, description="""Physical unit of the time series values.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    labels_ordering: Optional[list[str]] = Field(default=[], description="""Ordering of dimensions: Time, State Variable, Space, Mode.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    labels_dimensions: Optional[str] = Field(default=None, description="""Mapping of dimension names to their labels (JSON-encoded dict).""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    source_experiment: Optional[int] = Field(default=None, description="""Reference to the SimulationExperiment that generated this TimeSeries.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    generated_at: Optional[datetime ] = Field(default=None, description="""Timestamp when this TimeSeries was generated.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    software_environment: Optional[SoftwareEnvironment] = Field(default=None, description="""Software environment used to generate this data.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    task_name: Optional[str] = Field(default=None, description="""BIDS task name for the simulation (e.g., 'rest', 'simulation').""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    subject_id: Optional[str] = Field(default=None, description="""BIDS subject identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Subject', 'TimeSeries']} })
    session_id: Optional[str] = Field(default=None, description="""BIDS session identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    run_id: Optional[int] = Field(default=None, description="""BIDS run number.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    modality: Optional[ImagingModality] = Field(default=None, description="""Imaging modality or simulation output type.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    model_equation_ref: Optional[str] = Field(default=None, description="""BIDS ModelEq reference: path to _eq.xml LEMS file.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    model_param_ref: Optional[str] = Field(default=None, description="""BIDS ModelParam reference: path to _param.xml LEMS file.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })
    connectivity_ref: Optional[str] = Field(default=None, description="""Reference to connectivity data (_conndata-network_connectivity.tsv).""", json_schema_extra = { "linkml_meta": {'domain_of': ['TimeSeries']} })


class SoftwareEnvironment(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'comments': ['An environment now aggregates one or more SoftwareRequirement '
                      'entries.',
                      'Use SimulationExperiment.environment to reference a reusable '
                      'environment.',
                      "Field 'name' supersedes previous 'software' attribute for "
                      'clarity.',
                      "'version' here is an environment spec version, not a package "
                      'version.'],
         'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Stimulus',
                       'Matrix',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'NDArray',
                       'Mesh']} })
    name: Optional[str] = Field(default=None, description="""Human-readable environment label/name (deprecated alias was 'software').""", json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    version: Optional[str] = Field(default=None, description="""Optional version tag for the environment definition (not a package version).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'SoftwareEnvironment', 'SoftwareRequirement']} })
    platform: Optional[str] = Field(default=None, description="""OS / architecture description (e.g., linux-64).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment']} })
    environment_type: Optional[EnvironmentType] = Field(default=None, description="""Category: conda, venv, docker, etc.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment']} })
    container_image: Optional[str] = Field(default=None, description="""Container image reference (e.g., ghcr.io/org/img:tag@sha256:...).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment']} })
    build_hash: Optional[str] = Field(default=None, description="""Deterministic hash/fingerprint of the resolved dependency set.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareEnvironment']} })
    requirements: Optional[dict[str, SoftwareRequirement]] = Field(default=None, description="""Constituent software/module requirements that define this environment.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'SoftwareEnvironment', 'PDESolver']} })


class SoftwareRequirement(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'comments': ['Represents an individual requirement (package/module/library).',
                      "Add 'package' to separate identity from requirement expression.",
                      "Use 'version_spec' instead of 'version' for semantic clarity.",
                      "'modules' retained only for backward compatibility and will be "
                      'removed in a future release.',
                      'No pointer back to SoftwareEnvironment; aggregation is one-way '
                      'from SoftwareEnvironment.requirements.'],
         'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Stimulus',
                       'Matrix',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'NDArray',
                       'Mesh']} })
    package: Optional[str] = Field(default=None, description="""Reference to the software package identity.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    version_spec: Optional[str] = Field(default=None, description="""Version or constraint specifier (e.g., '==2.7.3', '>=1.2,<2').""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    role: Optional[RequirementRole] = Field(default=RequirementRole.runtime, json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement'], 'ifabsent': 'runtime'} })
    optional: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement'], 'ifabsent': 'False'} })
    hash: Optional[str] = Field(default=None, description="""Build or artifact hash for exact reproducibility (wheel, sdist, image layer).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    source_url: Optional[str] = Field(default=None, description="""Canonical source or repository URL.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    url: Optional[str] = Field(default=None, description="""(Deprecated) Use source_url.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    license: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    modules: Optional[list[str]] = Field(default=[], description="""(Deprecated) Former ad-hoc list; use environment.requirements list instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwareRequirement']} })
    version: Optional[str] = Field(default=None, description="""(Deprecated) Use version_spec.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale', 'SoftwareEnvironment', 'SoftwareRequirement']} })


class SoftwarePackage(ConfiguredBaseModel):
    """
    Identity information about a software package independent of a specific version requirement.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'comments': ['Distinct from SoftwareRequirement which binds a package to a '
                      'version/role.'],
         'from_schema': 'https://w3id.org/tvbo'})

    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    homepage: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage']} })
    license: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    repository: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage']} })
    doi: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SimulationStudy', 'SoftwarePackage']} })
    ecosystem: Optional[str] = Field(default=None, description="""Package ecosystem or index (e.g., pypi, conda-forge).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SoftwarePackage']} })


class NDArray(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    shape: Optional[list[int]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'NDArray']} })
    dtype: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['NDArray']} })
    dataLocation: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Stimulus',
                       'Matrix',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'NDArray',
                       'Mesh']} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })


class SpatialDomain(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    coordinate_space: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'Subject',
                       'Electrode',
                       'EField',
                       'SpatialDomain',
                       'Mesh']} })
    region: Optional[str] = Field(default=None, description="""Optional named region/ROI in the atlas/parcellation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Node', 'SpatialDomain']} })
    geometry: Optional[str] = Field(default=None, description="""Optional file for geometry/ROI mask (e.g., NIfTI, GIfTI).""", json_schema_extra = { "linkml_meta": {'domain_of': ['SpatialDomain']} })


class Mesh(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    dataLocation: Optional[str] = Field(default=None, description="""Add the location of the data file containing the parcellation terminology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Stimulus',
                       'Matrix',
                       'RandomStream',
                       'RegionMapping',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'NDArray',
                       'Mesh']} })
    element_type: Optional[ElementType] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    coordinates: Optional[list[Coordinate]] = Field(default=[], description="""Node coordinates (x,y,z) in the given coordinate space.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    elements: Optional[str] = Field(default=None, description="""Connectivity (indices) or file reference to topology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Mesh']} })
    coordinate_space: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Dataset',
                       'Subject',
                       'Electrode',
                       'EField',
                       'SpatialDomain',
                       'Mesh']} })


class SpatialField(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    quantity_kind: Optional[str] = Field(default=None, description="""Scalar, vector, or tensor.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SpatialField']} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    mesh: Optional[Mesh] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SpatialField', 'FieldStateVariable', 'PDE']} })
    values: Optional[NDArray] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Matrix', 'BrainRegionSeries', 'SpatialField']} })
    time_dependent: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['TemporalApplicableEquation',
                       'SpatialField',
                       'BoundaryCondition'],
         'ifabsent': 'False'} })
    initial_value: Optional[float] = Field(default=0.1, description="""Constant initial value for the field.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'SpatialField'], 'ifabsent': 'float(0.1)'} })
    initial_expression: Optional[Equation] = Field(default=None, description="""Analytic initial condition for the field.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SpatialField']} })


class FieldStateVariable(StateVariable):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    mesh: Optional[Mesh] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['SpatialField', 'FieldStateVariable', 'PDE']} })
    boundary_conditions: Optional[list[BoundaryCondition]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['FieldStateVariable', 'PDE']} })
    name: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['BrainAtlas',
                       'CommonCoordinateSpace',
                       'ParcellationEntity',
                       'DBSProtocol',
                       'ClinicalScale',
                       'ClinicalScore',
                       'Tractogram',
                       'File',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'CouplingInput',
                       'Argument',
                       'Function',
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningAlgorithm',
                       'Coupling',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage']} })
    symbol: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable',
                       'Parameter',
                       'DerivedParameter',
                       'DerivedVariable']} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator']} })
    domain: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Parameter',
                       'PDE']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
                       'Case',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Noise',
                       'UpdateRule',
                       'DifferentialOperator'],
         'slot_uri': 'tvbo:Equation'} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['CommonCoordinateSpace',
                       'StateVariable',
                       'Parameter',
                       'Argument',
                       'DerivedParameter',
                       'DerivedVariable',
                       'Integrator',
                       'TimeSeries',
                       'NDArray',
                       'SpatialField']} })
    variable_of_interest: Optional[bool] = Field(default=True, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'True'} })
    coupling_variable: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable'], 'ifabsent': 'False'} })
    noise: Optional[Noise] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'Integrator']} })
    stimulation_variable: Optional[bool] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })
    boundaries: Optional[Range] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })
    initial_value: Optional[float] = Field(default=0.1, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable', 'SpatialField'], 'ifabsent': 'float(0.1)'} })
    history: Optional[TimeSeries] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['StateVariable']} })


class DifferentialOperator(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    definition: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DifferentialOperator']} })
    equation: Optional[Equation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Stimulus',
                       'Observation',
                       'StateVariable',
                       'Distribution',
                       'Parameter',
                       'Function',
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
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    bc_type: Optional[BoundaryConditionType] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['BoundaryCondition']} })
    on_region: Optional[str] = Field(default=None, description="""Mesh/atlas subset where BC applies.""", json_schema_extra = { "linkml_meta": {'domain_of': ['BoundaryCondition']} })
    value: Optional[Equation] = Field(default=None, description="""Constant, parameter, or equation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Parameter', 'Argument', 'BoundaryCondition']} })
    time_dependent: Optional[bool] = Field(default=False, json_schema_extra = { "linkml_meta": {'domain_of': ['TemporalApplicableEquation',
                       'SpatialField',
                       'BoundaryCondition'],
         'ifabsent': 'False'} })


class PDESolver(ConfiguredBaseModel):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    requirements: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Function', 'SoftwareEnvironment', 'PDESolver']} })
    environment: Optional[SoftwareEnvironment] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Observation', 'SimulationExperiment', 'PDESolver']} })
    discretization: Optional[DiscretizationMethod] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['PDESolver']} })
    time_integrator: Optional[str] = Field(default=None, description="""e.g., implicit Euler, Crank–Nicolson.""", json_schema_extra = { "linkml_meta": {'domain_of': ['PDESolver']} })
    step_size: Optional[float] = Field(default=None, description="""Time step (s).""", json_schema_extra = { "linkml_meta": {'domain_of': ['PDESolver']} })
    tolerances: Optional[str] = Field(default=None, description="""Abs/rel tolerances.""", json_schema_extra = { "linkml_meta": {'domain_of': ['PDESolver']} })
    preconditioner: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['PDESolver']} })


class PDE(ConfiguredBaseModel):
    """
    Partial differential equation problem definition.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/tvbo'})

    label: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ParcellationTerminology',
                       'Dataset',
                       'Contact',
                       'Equation',
                       'Stimulus',
                       'Parcellation',
                       'Tractogram',
                       'Matrix',
                       'Network',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'StateVariable',
                       'Parameter',
                       'Function',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'TuningObjective',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'DifferentialOperator',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScore',
                       'Equation',
                       'Stimulus',
                       'Tractogram',
                       'Matrix',
                       'Network',
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
                       'Callable',
                       'DerivedParameter',
                       'DerivedVariable',
                       'RandomStream',
                       'DataSource',
                       'OptimizationStage',
                       'Optimization',
                       'Exploration',
                       'UpdateRule',
                       'TuningObjective',
                       'TuningAlgorithm',
                       'Integrator',
                       'Coupling',
                       'RegionMapping',
                       'SimulationExperiment',
                       'SimulationStudy',
                       'TimeSeries',
                       'SoftwareEnvironment',
                       'SoftwareRequirement',
                       'SoftwarePackage',
                       'NDArray',
                       'SpatialDomain',
                       'Mesh',
                       'SpatialField',
                       'FieldStateVariable',
                       'BoundaryCondition',
                       'PDESolver',
                       'PDE']} })
    parameters: Optional[dict[str, Parameter]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Equation',
                       'Stimulus',
                       'TemporalApplicableEquation',
                       'Node',
                       'Edge',
                       'Observation',
                       'Dynamics',
                       'Distribution',
                       'Noise',
                       'Exploration',
                       'Integrator',
                       'Coupling',
                       'PDE']} })
    domain: Optional[SpatialDomain] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClinicalScale',
                       'ClinicalScore',
                       'StateVariable',
                       'Parameter',
                       'PDE']} })
    mesh: Optional[Mesh] = Field(default=None, description="""Shared mesh for all field state variables in this PDE.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SpatialField', 'FieldStateVariable', 'PDE']} })
    state_variables: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    field: Optional[SpatialField] = Field(default=None, description="""Primary field being solved for (deprecated; use state_variables).""", json_schema_extra = { "linkml_meta": {'domain_of': ['PDE']} })
    operators: Optional[list[DifferentialOperator]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['PDE']} })
    sources: Optional[list[Equation]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['PDE']} })
    boundary_conditions: Optional[list[BoundaryCondition]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['FieldStateVariable', 'PDE']} })
    solver: Optional[PDESolver] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['PDE']} })
    derived_parameters: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    derived_variables: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'PDE']} })
    functions: Optional[list[str]] = Field(default=[], json_schema_extra = { "linkml_meta": {'domain_of': ['Dynamics', 'SimulationExperiment', 'PDE']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
Coordinate.model_rebuild()
BrainAtlas.model_rebuild()
CommonCoordinateSpace.model_rebuild()
ParcellationEntity.model_rebuild()
ParcellationTerminology.model_rebuild()
Dataset.model_rebuild()
Subject.model_rebuild()
Electrode.model_rebuild()
Contact.model_rebuild()
StimulationSetting.model_rebuild()
DBSProtocol.model_rebuild()
ClinicalScale.model_rebuild()
ClinicalScore.model_rebuild()
ClinicalImprovement.model_rebuild()
EField.model_rebuild()
Range.model_rebuild()
Equation.model_rebuild()
ConditionalBlock.model_rebuild()
Stimulus.model_rebuild()
TemporalApplicableEquation.model_rebuild()
Parcellation.model_rebuild()
Tractogram.model_rebuild()
Matrix.model_rebuild()
BrainRegionSeries.model_rebuild()
Network.model_rebuild()
File.model_rebuild()
Node.model_rebuild()
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
UpdateRule.model_rebuild()
TuningObjective.model_rebuild()
TuningAlgorithm.model_rebuild()
Integrator.model_rebuild()
Coupling.model_rebuild()
RegionMapping.model_rebuild()
Sample.model_rebuild()
ExecutionConfig.model_rebuild()
SimulationExperiment.model_rebuild()
SimulationStudy.model_rebuild()
TimeSeries.model_rebuild()
SoftwareEnvironment.model_rebuild()
SoftwareRequirement.model_rebuild()
SoftwarePackage.model_rebuild()
NDArray.model_rebuild()
SpatialDomain.model_rebuild()
Mesh.model_rebuild()
SpatialField.model_rebuild()
FieldStateVariable.model_rebuild()
DifferentialOperator.model_rebuild()
BoundaryCondition.model_rebuild()
PDESolver.model_rebuild()
PDE.model_rebuild()
