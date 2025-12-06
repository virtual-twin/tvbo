# Auto generated from tvbo_datamodel.yaml by pythongen.py version: 0.0.1
# Generation date: 2025-12-06T23:30:25
# Schema: tvb-datamodel
#
# id: https://w3id.org/tvbo
# description: Metadata schema for simulation studies using The Virtual Brain neuroinformatics platform or other dynamic network models of large-scale brain activity.
# license: https://creativecommons.org/publicdomain/zero/1.0/

import dataclasses
import re
from dataclasses import dataclass
from datetime import (
    date,
    datetime,
    time
)
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Union
)

from jsonasobj2 import (
    JsonObj,
    as_dict
)
from linkml_runtime.linkml_model.meta import (
    EnumDefinition,
    PermissibleValue,
    PvFormulaOptions
)
from linkml_runtime.utils.curienamespace import CurieNamespace
from linkml_runtime.utils.enumerations import EnumDefinitionImpl
from linkml_runtime.utils.formatutils import (
    camelcase,
    sfx,
    underscore
)
from linkml_runtime.utils.metamodelcore import (
    bnode,
    empty_dict,
    empty_list
)
from linkml_runtime.utils.slot import Slot
from linkml_runtime.utils.yamlutils import (
    YAMLRoot,
    extended_float,
    extended_int,
    extended_str
)
from rdflib import (
    Namespace,
    URIRef
)

from linkml_runtime.linkml_model.types import Boolean, Float, Integer, String
from linkml_runtime.utils.metamodelcore import Bool

metamodel_version = "1.7.0"
version = None

# Namespaces
ATOM = CurieNamespace('atom', 'http://uri.interlex.org/tgbugs/uris/readable/')
LINKML = CurieNamespace('linkml', 'https://w3id.org/linkml/')
PROV = CurieNamespace('prov', 'http://www.w3.org/ns/prov#')
RDFS = CurieNamespace('rdfs', 'http://www.w3.org/2000/01/rdf-schema#')
SCHEMA = CurieNamespace('schema', 'http://schema.org/')
TVBO = CurieNamespace('tvbo', 'http://www.thevirtualbrain.org/tvb-o/')
TVBO_DBS = CurieNamespace('tvbo_dbs', 'http://www.thevirtualbrain.org/tvb-o/dbs/')
DEFAULT_ = TVBO


# Types

# Class references
class ObservationModelName(extended_str):
    pass


class DownsamplingModelName(ObservationModelName):
    pass


class DynamicsName(extended_str):
    pass


class NeuralMassModelName(DynamicsName):
    pass


class StateVariableName(extended_str):
    pass


class DistributionName(extended_str):
    pass


class ParameterName(extended_str):
    pass


class CouplingInputName(extended_str):
    pass


class FunctionName(extended_str):
    pass


class CallableName(extended_str):
    pass


class DerivedParameterName(ParameterName):
    pass


class DerivedVariableName(extended_str):
    pass


class MonitorName(ObservationModelName):
    pass


class CouplingName(extended_str):
    pass


class SimulationExperimentId(extended_int):
    pass


class SoftwareRequirementName(extended_str):
    pass


class SoftwarePackageName(extended_str):
    pass


class FieldStateVariableName(StateVariableName):
    pass


class BrainAtlasName(extended_str):
    pass


class CommonCoordinateSpaceName(extended_str):
    pass


class ParcellationEntityName(extended_str):
    pass


class SubjectSubjectId(extended_str):
    pass


class DBSProtocolName(extended_str):
    pass


@dataclass(repr=False)
class Range(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Range"]
    class_class_curie: ClassVar[str] = "tvbo:Range"
    class_name: ClassVar[str] = "Range"
    class_model_uri: ClassVar[URIRef] = TVBO.Range

    lo: Optional[float] = None
    hi: Optional[float] = None
    step: Optional[float] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.lo is not None and not isinstance(self.lo, float):
            self.lo = float(self.lo)

        if self.hi is not None and not isinstance(self.hi, float):
            self.hi = float(self.hi)

        if self.step is not None and not isinstance(self.step, float):
            self.step = float(self.step)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Equation(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Equation"]
    class_class_curie: ClassVar[str] = "tvbo:Equation"
    class_name: ClassVar[str] = "Equation"
    class_model_uri: ClassVar[URIRef] = TVBO.Equation

    label: Optional[str] = None
    definition: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    lhs: Optional[str] = None
    rhs: Optional[str] = None
    conditionals: Optional[Union[Union[dict, "ConditionalBlock"], list[Union[dict, "ConditionalBlock"]]]] = empty_list()
    engine: Optional[Union[dict, "SoftwareRequirement"]] = None
    pycode: Optional[str] = None
    latex: Optional[Union[bool, Bool]] = False

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.definition is not None and not isinstance(self.definition, str):
            self.definition = str(self.definition)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.lhs is not None and not isinstance(self.lhs, str):
            self.lhs = str(self.lhs)

        if self.rhs is not None and not isinstance(self.rhs, str):
            self.rhs = str(self.rhs)

        if not isinstance(self.conditionals, list):
            self.conditionals = [self.conditionals] if self.conditionals is not None else []
        self.conditionals = [v if isinstance(v, ConditionalBlock) else ConditionalBlock(**as_dict(v)) for v in self.conditionals]

        if self.engine is not None and not isinstance(self.engine, SoftwareRequirement):
            self.engine = SoftwareRequirement(**as_dict(self.engine))

        if self.pycode is not None and not isinstance(self.pycode, str):
            self.pycode = str(self.pycode)

        if self.latex is not None and not isinstance(self.latex, Bool):
            self.latex = Bool(self.latex)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ConditionalBlock(YAMLRoot):
    """
    A single condition and its corresponding equation segment.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["ConditionalBlock"]
    class_class_curie: ClassVar[str] = "tvbo:ConditionalBlock"
    class_name: ClassVar[str] = "ConditionalBlock"
    class_model_uri: ClassVar[URIRef] = TVBO.ConditionalBlock

    condition: Optional[str] = None
    expression: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.condition is not None and not isinstance(self.condition, str):
            self.condition = str(self.condition)

        if self.expression is not None and not isinstance(self.expression, str):
            self.expression = str(self.expression)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Stimulus(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Stimulus"]
    class_class_curie: ClassVar[str] = "tvbo:Stimulus"
    class_name: ClassVar[str] = "Stimulus"
    class_model_uri: ClassVar[URIRef] = TVBO.Stimulus

    equation: Optional[Union[dict, Equation]] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    description: Optional[str] = None
    dataLocation: Optional[str] = None
    duration: Optional[float] = 1000
    label: Optional[str] = None
    regions: Optional[Union[int, list[int]]] = empty_list()
    weighting: Optional[Union[float, list[float]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        if self.duration is not None and not isinstance(self.duration, float):
            self.duration = float(self.duration)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if not isinstance(self.regions, list):
            self.regions = [self.regions] if self.regions is not None else []
        self.regions = [v if isinstance(v, int) else int(v) for v in self.regions]

        if not isinstance(self.weighting, list):
            self.weighting = [self.weighting] if self.weighting is not None else []
        self.weighting = [v if isinstance(v, float) else float(v) for v in self.weighting]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class TemporalApplicableEquation(Equation):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["TemporalApplicableEquation"]
    class_class_curie: ClassVar[str] = "tvbo:TemporalApplicableEquation"
    class_name: ClassVar[str] = "TemporalApplicableEquation"
    class_model_uri: ClassVar[URIRef] = TVBO.TemporalApplicableEquation

    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    time_dependent: Optional[Union[bool, Bool]] = False

    def __post_init__(self, *_: str, **kwargs: Any):
        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.time_dependent is not None and not isinstance(self.time_dependent, Bool):
            self.time_dependent = Bool(self.time_dependent)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Parcellation(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Parcellation"]
    class_class_curie: ClassVar[str] = "tvbo:Parcellation"
    class_name: ClassVar[str] = "Parcellation"
    class_model_uri: ClassVar[URIRef] = TVBO.Parcellation

    atlas: Union[dict, "BrainAtlas"] = None
    label: Optional[str] = None
    region_labels: Optional[Union[str, list[str]]] = empty_list()
    center_coordinates: Optional[Union[float, list[float]]] = empty_list()
    data_source: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.atlas):
            self.MissingRequiredField("atlas")
        if not isinstance(self.atlas, BrainAtlas):
            self.atlas = BrainAtlas(**as_dict(self.atlas))

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if not isinstance(self.region_labels, list):
            self.region_labels = [self.region_labels] if self.region_labels is not None else []
        self.region_labels = [v if isinstance(v, str) else str(v) for v in self.region_labels]

        if not isinstance(self.center_coordinates, list):
            self.center_coordinates = [self.center_coordinates] if self.center_coordinates is not None else []
        self.center_coordinates = [v if isinstance(v, float) else float(v) for v in self.center_coordinates]

        if self.data_source is not None and not isinstance(self.data_source, str):
            self.data_source = str(self.data_source)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Matrix(YAMLRoot):
    """
    Adjacency matrix of a network.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Matrix"]
    class_class_curie: ClassVar[str] = "tvbo:Matrix"
    class_name: ClassVar[str] = "Matrix"
    class_model_uri: ClassVar[URIRef] = TVBO.Matrix

    label: Optional[str] = None
    description: Optional[str] = None
    dataLocation: Optional[str] = None
    x: Optional[Union[dict, "BrainRegionSeries"]] = None
    y: Optional[Union[dict, "BrainRegionSeries"]] = None
    values: Optional[Union[float, list[float]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        if self.x is not None and not isinstance(self.x, BrainRegionSeries):
            self.x = BrainRegionSeries(**as_dict(self.x))

        if self.y is not None and not isinstance(self.y, BrainRegionSeries):
            self.y = BrainRegionSeries(**as_dict(self.y))

        if not isinstance(self.values, list):
            self.values = [self.values] if self.values is not None else []
        self.values = [v if isinstance(v, float) else float(v) for v in self.values]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class BrainRegionSeries(YAMLRoot):
    """
    A series whose values represent latitude
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["BrainRegionSeries"]
    class_class_curie: ClassVar[str] = "tvbo:BrainRegionSeries"
    class_name: ClassVar[str] = "BrainRegionSeries"
    class_model_uri: ClassVar[URIRef] = TVBO.BrainRegionSeries

    values: Optional[Union[str, list[str]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if not isinstance(self.values, list):
            self.values = [self.values] if self.values is not None else []
        self.values = [v if isinstance(v, str) else str(v) for v in self.values]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Network(YAMLRoot):
    """
    Network specification with nodes, edges, and reusable coupling configurations. Supports both explicit node/edge
    representation and matrix-based connectivity (Connectome compatibility).
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Network"]
    class_class_curie: ClassVar[str] = "tvbo:Network"
    class_name: ClassVar[str] = "Network"
    class_model_uri: ClassVar[URIRef] = TVBO.Network

    label: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[Union[Union[dict, "Node"], list[Union[dict, "Node"]]]] = empty_list()
    edges: Optional[Union[Union[dict, "Edge"], list[Union[dict, "Edge"]]]] = empty_list()
    coupling_library: Optional[Union[dict[Union[str, CouplingName], Union[dict, "Coupling"]], list[Union[dict, "Coupling"]]]] = empty_dict()
    number_of_regions: Optional[int] = 1
    number_of_nodes: Optional[int] = 1
    parcellation: Optional[Union[dict, Parcellation]] = None
    tractogram: Optional[str] = None
    weights: Optional[Union[dict, Matrix]] = None
    lengths: Optional[Union[dict, Matrix]] = None
    normalization: Optional[Union[dict, Equation]] = None
    node_labels: Optional[Union[str, list[str]]] = empty_list()
    global_coupling_strength: Optional[Union[dict, "Parameter"]] = None
    conduction_speed: Optional[Union[dict, "Parameter"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if not isinstance(self.nodes, list):
            self.nodes = [self.nodes] if self.nodes is not None else []
        self.nodes = [v if isinstance(v, Node) else Node(**as_dict(v)) for v in self.nodes]

        if not isinstance(self.edges, list):
            self.edges = [self.edges] if self.edges is not None else []
        self.edges = [v if isinstance(v, Edge) else Edge(**as_dict(v)) for v in self.edges]

        self._normalize_inlined_as_dict(slot_name="coupling_library", slot_type=Coupling, key_name="name", keyed=True)

        if self.number_of_regions is not None and not isinstance(self.number_of_regions, int):
            self.number_of_regions = int(self.number_of_regions)

        if self.number_of_nodes is not None and not isinstance(self.number_of_nodes, int):
            self.number_of_nodes = int(self.number_of_nodes)

        if self.parcellation is not None and not isinstance(self.parcellation, Parcellation):
            self.parcellation = Parcellation(**as_dict(self.parcellation))

        if self.tractogram is not None and not isinstance(self.tractogram, str):
            self.tractogram = str(self.tractogram)

        if self.weights is not None and not isinstance(self.weights, Matrix):
            self.weights = Matrix(**as_dict(self.weights))

        if self.lengths is not None and not isinstance(self.lengths, Matrix):
            self.lengths = Matrix(**as_dict(self.lengths))

        if self.normalization is not None and not isinstance(self.normalization, Equation):
            self.normalization = Equation(**as_dict(self.normalization))

        if not isinstance(self.node_labels, list):
            self.node_labels = [self.node_labels] if self.node_labels is not None else []
        self.node_labels = [v if isinstance(v, str) else str(v) for v in self.node_labels]

        if self.global_coupling_strength is not None and not isinstance(self.global_coupling_strength, Parameter):
            self.global_coupling_strength = Parameter(**as_dict(self.global_coupling_strength))

        if self.conduction_speed is not None and not isinstance(self.conduction_speed, Parameter):
            self.conduction_speed = Parameter(**as_dict(self.conduction_speed))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Node(YAMLRoot):
    """
    A node in a network with its own dynamics and properties
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Node"]
    class_class_curie: ClassVar[str] = "tvbo:Node"
    class_name: ClassVar[str] = "Node"
    class_model_uri: ClassVar[URIRef] = TVBO.Node

    id: int = None
    label: Optional[str] = None
    description: Optional[str] = None
    dynamics: Optional[Union[str, DynamicsName]] = None
    position: Optional[Union[dict, "Coordinate"]] = None
    region: Optional[str] = None
    parameters: Optional[Union[Union[str, ParameterName], list[Union[str, ParameterName]]]] = empty_list()
    initial_state: Optional[Union[float, list[float]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, int):
            self.id = int(self.id)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dynamics is not None and not isinstance(self.dynamics, DynamicsName):
            self.dynamics = DynamicsName(self.dynamics)

        if self.position is not None and not isinstance(self.position, Coordinate):
            self.position = Coordinate(**as_dict(self.position))

        if self.region is not None and not isinstance(self.region, str):
            self.region = str(self.region)

        if not isinstance(self.parameters, list):
            self.parameters = [self.parameters] if self.parameters is not None else []
        self.parameters = [v if isinstance(v, ParameterName) else ParameterName(v) for v in self.parameters]

        if not isinstance(self.initial_state, list):
            self.initial_state = [self.initial_state] if self.initial_state is not None else []
        self.initial_state = [v if isinstance(v, float) else float(v) for v in self.initial_state]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Edge(YAMLRoot):
    """
    A directed edge in a network with coupling and connectivity properties
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Edge"]
    class_class_curie: ClassVar[str] = "tvbo:Edge"
    class_name: ClassVar[str] = "Edge"
    class_model_uri: ClassVar[URIRef] = TVBO.Edge

    source: int = None
    target: int = None
    coupling: Union[dict, "Coupling"] = None
    label: Optional[str] = None
    description: Optional[str] = None
    weight: Optional[float] = 1.0
    delay: Optional[float] = 0.0
    distance: Optional[float] = None
    tract_properties: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.source):
            self.MissingRequiredField("source")
        if not isinstance(self.source, int):
            self.source = int(self.source)

        if self._is_empty(self.target):
            self.MissingRequiredField("target")
        if not isinstance(self.target, int):
            self.target = int(self.target)

        if self._is_empty(self.coupling):
            self.MissingRequiredField("coupling")
        if not isinstance(self.coupling, Coupling):
            self.coupling = Coupling(**as_dict(self.coupling))

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.weight is not None and not isinstance(self.weight, float):
            self.weight = float(self.weight)

        if self.delay is not None and not isinstance(self.delay, float):
            self.delay = float(self.delay)

        if self.distance is not None and not isinstance(self.distance, float):
            self.distance = float(self.distance)

        if self.tract_properties is not None and not isinstance(self.tract_properties, str):
            self.tract_properties = str(self.tract_properties)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ObservationModel(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["ObservationModel"]
    class_class_curie: ClassVar[str] = "tvbo:ObservationModel"
    class_name: ClassVar[str] = "ObservationModel"
    class_model_uri: ClassVar[URIRef] = TVBO.ObservationModel

    name: Union[str, ObservationModelName] = None
    acronym: Optional[str] = None
    description: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    environment: Optional[Union[dict, "SoftwareEnvironment"]] = None
    transformation: Optional[Union[dict, "Function"]] = None
    pipeline: Optional[Union[Union[dict, "ProcessingStep"], list[Union[dict, "ProcessingStep"]]]] = empty_list()
    data_injections: Optional[Union[Union[dict, "DataInjection"], list[Union[dict, "DataInjection"]]]] = empty_list()
    argument_mappings: Optional[Union[Union[dict, "ArgumentMapping"], list[Union[dict, "ArgumentMapping"]]]] = empty_list()
    derivatives: Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, "DerivedVariable"]], list[Union[dict, "DerivedVariable"]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, ObservationModelName):
            self.name = ObservationModelName(self.name)

        if self.acronym is not None and not isinstance(self.acronym, str):
            self.acronym = str(self.acronym)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.environment is not None and not isinstance(self.environment, SoftwareEnvironment):
            self.environment = SoftwareEnvironment(**as_dict(self.environment))

        if self.transformation is not None and not isinstance(self.transformation, Function):
            self.transformation = Function(**as_dict(self.transformation))

        if not isinstance(self.pipeline, list):
            self.pipeline = [self.pipeline] if self.pipeline is not None else []
        self.pipeline = [v if isinstance(v, ProcessingStep) else ProcessingStep(**as_dict(v)) for v in self.pipeline]

        if not isinstance(self.data_injections, list):
            self.data_injections = [self.data_injections] if self.data_injections is not None else []
        self.data_injections = [v if isinstance(v, DataInjection) else DataInjection(**as_dict(v)) for v in self.data_injections]

        if not isinstance(self.argument_mappings, list):
            self.argument_mappings = [self.argument_mappings] if self.argument_mappings is not None else []
        self.argument_mappings = [v if isinstance(v, ArgumentMapping) else ArgumentMapping(**as_dict(v)) for v in self.argument_mappings]

        self._normalize_inlined_as_dict(slot_name="derivatives", slot_type=DerivedVariable, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ProcessingStep(YAMLRoot):
    """
    A single processing step in an observation model pipeline or standalone operation
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["ProcessingStep"]
    class_class_curie: ClassVar[str] = "tvbo:ProcessingStep"
    class_name: ClassVar[str] = "ProcessingStep"
    class_model_uri: ClassVar[URIRef] = TVBO.ProcessingStep

    transformation: Union[dict, "Function"] = None
    order: Optional[int] = None
    type: Optional[Union[str, "OperationType"]] = None
    input_mapping: Optional[Union[Union[dict, "ArgumentMapping"], list[Union[dict, "ArgumentMapping"]]]] = empty_list()
    output_alias: Optional[str] = None
    apply_on_dimension: Optional[str] = None
    ensure_shape: Optional[str] = None
    variables_of_interest: Optional[Union[dict[Union[str, StateVariableName], Union[dict, "StateVariable"]], list[Union[dict, "StateVariable"]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.transformation):
            self.MissingRequiredField("transformation")
        if not isinstance(self.transformation, Function):
            self.transformation = Function(**as_dict(self.transformation))

        if self.order is not None and not isinstance(self.order, int):
            self.order = int(self.order)

        if self.type is not None and not isinstance(self.type, OperationType):
            self.type = OperationType(self.type)

        if not isinstance(self.input_mapping, list):
            self.input_mapping = [self.input_mapping] if self.input_mapping is not None else []
        self.input_mapping = [v if isinstance(v, ArgumentMapping) else ArgumentMapping(**as_dict(v)) for v in self.input_mapping]

        if self.output_alias is not None and not isinstance(self.output_alias, str):
            self.output_alias = str(self.output_alias)

        if self.apply_on_dimension is not None and not isinstance(self.apply_on_dimension, str):
            self.apply_on_dimension = str(self.apply_on_dimension)

        if self.ensure_shape is not None and not isinstance(self.ensure_shape, str):
            self.ensure_shape = str(self.ensure_shape)

        self._normalize_inlined_as_dict(slot_name="variables_of_interest", slot_type=StateVariable, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class DataInjection(YAMLRoot):
    """
    External data injected into the observation pipeline
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["DataInjection"]
    class_class_curie: ClassVar[str] = "tvbo:DataInjection"
    class_name: ClassVar[str] = "DataInjection"
    class_model_uri: ClassVar[URIRef] = TVBO.DataInjection

    name: str = None
    data_source: Optional[str] = None
    values: Optional[Union[float, list[float]]] = empty_list()
    shape: Optional[Union[int, list[int]]] = empty_list()
    generation_function: Optional[Union[dict, "Function"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, str):
            self.name = str(self.name)

        if self.data_source is not None and not isinstance(self.data_source, str):
            self.data_source = str(self.data_source)

        if not isinstance(self.values, list):
            self.values = [self.values] if self.values is not None else []
        self.values = [v if isinstance(v, float) else float(v) for v in self.values]

        if not isinstance(self.shape, list):
            self.shape = [self.shape] if self.shape is not None else []
        self.shape = [v if isinstance(v, int) else int(v) for v in self.shape]

        if self.generation_function is not None and not isinstance(self.generation_function, Function):
            self.generation_function = Function(**as_dict(self.generation_function))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ArgumentMapping(YAMLRoot):
    """
    Maps function arguments to pipeline inputs/outputs
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["ArgumentMapping"]
    class_class_curie: ClassVar[str] = "tvbo:ArgumentMapping"
    class_name: ClassVar[str] = "ArgumentMapping"
    class_model_uri: ClassVar[URIRef] = TVBO.ArgumentMapping

    function_argument: str = None
    source: str = None
    constant_value: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.function_argument):
            self.MissingRequiredField("function_argument")
        if not isinstance(self.function_argument, str):
            self.function_argument = str(self.function_argument)

        if self._is_empty(self.source):
            self.MissingRequiredField("source")
        if not isinstance(self.source, str):
            self.source = str(self.source)

        if self.constant_value is not None and not isinstance(self.constant_value, str):
            self.constant_value = str(self.constant_value)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class DownsamplingModel(ObservationModel):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["DownsamplingModel"]
    class_class_curie: ClassVar[str] = "tvbo:DownsamplingModel"
    class_name: ClassVar[str] = "DownsamplingModel"
    class_model_uri: ClassVar[URIRef] = TVBO.DownsamplingModel

    name: Union[str, DownsamplingModelName] = None
    period: Optional[float] = 0.9765625

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, DownsamplingModelName):
            self.name = DownsamplingModelName(self.name)

        if self.period is not None and not isinstance(self.period, float):
            self.period = float(self.period)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Dynamics(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Dynamics"]
    class_class_curie: ClassVar[str] = "tvbo:Dynamics"
    class_name: ClassVar[str] = "Dynamics"
    class_model_uri: ClassVar[URIRef] = TVBO.Dynamics

    name: Union[str, DynamicsName] = "Generic2dOscillator"
    has_reference: Optional[str] = None
    label: Optional[str] = None
    iri: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    description: Optional[str] = None
    source: Optional[str] = None
    references: Optional[Union[str, list[str]]] = empty_list()
    derived_parameters: Optional[Union[dict[Union[str, DerivedParameterName], Union[dict, "DerivedParameter"]], list[Union[dict, "DerivedParameter"]]]] = empty_dict()
    derived_variables: Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, "DerivedVariable"]], list[Union[dict, "DerivedVariable"]]]] = empty_dict()
    coupling_terms: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    coupling_inputs: Optional[Union[dict[Union[str, CouplingInputName], Union[dict, "CouplingInput"]], list[Union[dict, "CouplingInput"]]]] = empty_dict()
    state_variables: Optional[Union[dict[Union[str, StateVariableName], Union[dict, "StateVariable"]], list[Union[dict, "StateVariable"]]]] = empty_dict()
    modified: Optional[Union[bool, Bool]] = None
    output_transforms: Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, "DerivedVariable"]], list[Union[dict, "DerivedVariable"]]]] = empty_dict()
    derived_from_model: Optional[Union[str, NeuralMassModelName]] = None
    number_of_modes: Optional[int] = 1
    local_coupling_term: Optional[Union[str, ParameterName]] = None
    functions: Optional[Union[dict[Union[str, FunctionName], Union[dict, "Function"]], list[Union[dict, "Function"]]]] = empty_dict()
    stimulus: Optional[Union[dict, Stimulus]] = None
    modes: Optional[Union[dict[Union[str, NeuralMassModelName], Union[dict, "NeuralMassModel"]], list[Union[dict, "NeuralMassModel"]]]] = empty_dict()
    system_type: Optional[Union[str, "SystemType"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, DynamicsName):
            self.name = DynamicsName(self.name)

        if self.has_reference is not None and not isinstance(self.has_reference, str):
            self.has_reference = str(self.has_reference)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.iri is not None and not isinstance(self.iri, str):
            self.iri = str(self.iri)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.source is not None and not isinstance(self.source, str):
            self.source = str(self.source)

        if not isinstance(self.references, list):
            self.references = [self.references] if self.references is not None else []
        self.references = [v if isinstance(v, str) else str(v) for v in self.references]

        self._normalize_inlined_as_dict(slot_name="derived_parameters", slot_type=DerivedParameter, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="derived_variables", slot_type=DerivedVariable, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="coupling_terms", slot_type=Parameter, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="coupling_inputs", slot_type=CouplingInput, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="state_variables", slot_type=StateVariable, key_name="name", keyed=True)

        if self.modified is not None and not isinstance(self.modified, Bool):
            self.modified = Bool(self.modified)

        self._normalize_inlined_as_dict(slot_name="output_transforms", slot_type=DerivedVariable, key_name="name", keyed=True)

        if self.derived_from_model is not None and not isinstance(self.derived_from_model, NeuralMassModelName):
            self.derived_from_model = NeuralMassModelName(self.derived_from_model)

        if self.number_of_modes is not None and not isinstance(self.number_of_modes, int):
            self.number_of_modes = int(self.number_of_modes)

        if self.local_coupling_term is not None and not isinstance(self.local_coupling_term, ParameterName):
            self.local_coupling_term = ParameterName(self.local_coupling_term)

        self._normalize_inlined_as_dict(slot_name="functions", slot_type=Function, key_name="name", keyed=True)

        if self.stimulus is not None and not isinstance(self.stimulus, Stimulus):
            self.stimulus = Stimulus(**as_dict(self.stimulus))

        self._normalize_inlined_as_dict(slot_name="modes", slot_type=NeuralMassModel, key_name="name", keyed=True)

        if self.system_type is not None and not isinstance(self.system_type, SystemType):
            self.system_type = SystemType(self.system_type)

        if self.system_type is not None and not isinstance(self.system_type, str):
            self.system_type = str(self.system_type)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class NeuralMassModel(Dynamics):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["NeuralMassModel"]
    class_class_curie: ClassVar[str] = "tvbo:NeuralMassModel"
    class_name: ClassVar[str] = "NeuralMassModel"
    class_model_uri: ClassVar[URIRef] = TVBO.NeuralMassModel

    name: Union[str, NeuralMassModelName] = "Generic2dOscillator"

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, NeuralMassModelName):
            self.name = NeuralMassModelName(self.name)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class StateVariable(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["StateVariable"]
    class_class_curie: ClassVar[str] = "tvbo:StateVariable"
    class_name: ClassVar[str] = "StateVariable"
    class_model_uri: ClassVar[URIRef] = TVBO.StateVariable

    name: Union[str, StateVariableName] = None
    symbol: Optional[str] = None
    label: Optional[str] = None
    definition: Optional[str] = None
    domain: Optional[Union[dict, Range]] = None
    description: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    unit: Optional[str] = None
    variable_of_interest: Optional[Union[bool, Bool]] = True
    coupling_variable: Optional[Union[bool, Bool]] = False
    noise: Optional[Union[dict, "Noise"]] = None
    stimulation_variable: Optional[Union[bool, Bool]] = None
    boundaries: Optional[Union[dict, Range]] = None
    initial_value: Optional[float] = 0.1
    history: Optional[Union[dict, "TimeSeries"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, StateVariableName):
            self.name = StateVariableName(self.name)

        if self.symbol is not None and not isinstance(self.symbol, str):
            self.symbol = str(self.symbol)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.definition is not None and not isinstance(self.definition, str):
            self.definition = str(self.definition)

        if self.domain is not None and not isinstance(self.domain, Range):
            self.domain = Range(**as_dict(self.domain))

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        if self.variable_of_interest is not None and not isinstance(self.variable_of_interest, Bool):
            self.variable_of_interest = Bool(self.variable_of_interest)

        if self.coupling_variable is not None and not isinstance(self.coupling_variable, Bool):
            self.coupling_variable = Bool(self.coupling_variable)

        if self.noise is not None and not isinstance(self.noise, Noise):
            self.noise = Noise(**as_dict(self.noise))

        if self.stimulation_variable is not None and not isinstance(self.stimulation_variable, Bool):
            self.stimulation_variable = Bool(self.stimulation_variable)

        if self.boundaries is not None and not isinstance(self.boundaries, Range):
            self.boundaries = Range(**as_dict(self.boundaries))

        if self.initial_value is not None and not isinstance(self.initial_value, float):
            self.initial_value = float(self.initial_value)

        if self.history is not None and not isinstance(self.history, TimeSeries):
            self.history = TimeSeries(**as_dict(self.history))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Distribution(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Distribution"]
    class_class_curie: ClassVar[str] = "tvbo:Distribution"
    class_name: ClassVar[str] = "Distribution"
    class_model_uri: ClassVar[URIRef] = TVBO.Distribution

    name: Union[str, DistributionName] = None
    equation: Optional[Union[dict, Equation]] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    dependencies: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    correlation: Optional[Union[dict, Matrix]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, DistributionName):
            self.name = DistributionName(self.name)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="dependencies", slot_type=Parameter, key_name="name", keyed=True)

        if self.correlation is not None and not isinstance(self.correlation, Matrix):
            self.correlation = Matrix(**as_dict(self.correlation))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Parameter(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Parameter"]
    class_class_curie: ClassVar[str] = "tvbo:Parameter"
    class_name: ClassVar[str] = "Parameter"
    class_model_uri: ClassVar[URIRef] = TVBO.Parameter

    name: Union[str, ParameterName] = None
    label: Optional[str] = None
    symbol: Optional[str] = None
    definition: Optional[str] = None
    value: Optional[float] = None
    default: Optional[str] = None
    domain: Optional[Union[dict, Range]] = None
    reported_optimum: Optional[float] = None
    description: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    unit: Optional[str] = None
    comment: Optional[str] = None
    heterogeneous: Optional[Union[bool, Bool]] = None
    free: Optional[Union[bool, Bool]] = None
    shape: Optional[str] = None
    explored_values: Optional[Union[float, list[float]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, ParameterName):
            self.name = ParameterName(self.name)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.symbol is not None and not isinstance(self.symbol, str):
            self.symbol = str(self.symbol)

        if self.definition is not None and not isinstance(self.definition, str):
            self.definition = str(self.definition)

        if self.value is not None and not isinstance(self.value, float):
            self.value = float(self.value)

        if self.default is not None and not isinstance(self.default, str):
            self.default = str(self.default)

        if self.domain is not None and not isinstance(self.domain, Range):
            self.domain = Range(**as_dict(self.domain))

        if self.reported_optimum is not None and not isinstance(self.reported_optimum, float):
            self.reported_optimum = float(self.reported_optimum)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        if self.comment is not None and not isinstance(self.comment, str):
            self.comment = str(self.comment)

        if self.heterogeneous is not None and not isinstance(self.heterogeneous, Bool):
            self.heterogeneous = Bool(self.heterogeneous)

        if self.free is not None and not isinstance(self.free, Bool):
            self.free = Bool(self.free)

        if self.shape is not None and not isinstance(self.shape, str):
            self.shape = str(self.shape)

        if not isinstance(self.explored_values, list):
            self.explored_values = [self.explored_values] if self.explored_values is not None else []
        self.explored_values = [v if isinstance(v, float) else float(v) for v in self.explored_values]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class CouplingInput(YAMLRoot):
    """
    Specification of a coupling input channel for multi-coupling dynamics
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["CouplingInput"]
    class_class_curie: ClassVar[str] = "tvbo:CouplingInput"
    class_name: ClassVar[str] = "CouplingInput"
    class_model_uri: ClassVar[URIRef] = TVBO.CouplingInput

    name: Union[str, CouplingInputName] = None
    description: Optional[str] = None
    dimension: Optional[int] = 1

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, CouplingInputName):
            self.name = CouplingInputName(self.name)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dimension is not None and not isinstance(self.dimension, int):
            self.dimension = int(self.dimension)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Function(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Function"]
    class_class_curie: ClassVar[str] = "tvbo:Function"
    class_name: ClassVar[str] = "Function"
    class_model_uri: ClassVar[URIRef] = TVBO.Function

    name: Union[str, FunctionName] = None
    acronym: Optional[str] = None
    label: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    definition: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[Union[Union[str, SoftwareRequirementName], list[Union[str, SoftwareRequirementName]]]] = empty_list()
    iri: Optional[str] = None
    arguments: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    output: Optional[Union[dict, Equation]] = None
    source_code: Optional[str] = None
    callable: Optional[Union[dict, "Callable"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, FunctionName):
            self.name = FunctionName(self.name)

        if self.acronym is not None and not isinstance(self.acronym, str):
            self.acronym = str(self.acronym)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.definition is not None and not isinstance(self.definition, str):
            self.definition = str(self.definition)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if not isinstance(self.requirements, list):
            self.requirements = [self.requirements] if self.requirements is not None else []
        self.requirements = [v if isinstance(v, SoftwareRequirementName) else SoftwareRequirementName(v) for v in self.requirements]

        if self.iri is not None and not isinstance(self.iri, str):
            self.iri = str(self.iri)

        if self.definition is not None and not isinstance(self.definition, str):
            self.definition = str(self.definition)

        self._normalize_inlined_as_dict(slot_name="arguments", slot_type=Parameter, key_name="name", keyed=True)

        if self.output is not None and not isinstance(self.output, Equation):
            self.output = Equation(**as_dict(self.output))

        if self.source_code is not None and not isinstance(self.source_code, str):
            self.source_code = str(self.source_code)

        if self.callable is not None and not isinstance(self.callable, Callable):
            self.callable = Callable(**as_dict(self.callable))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Callable(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Callable"]
    class_class_curie: ClassVar[str] = "tvbo:Callable"
    class_name: ClassVar[str] = "Callable"
    class_model_uri: ClassVar[URIRef] = TVBO.Callable

    name: Union[str, CallableName] = None
    description: Optional[str] = None
    module: Optional[str] = None
    qualname: Optional[str] = None
    software: Optional[Union[dict, "SoftwareRequirement"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, CallableName):
            self.name = CallableName(self.name)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.module is not None and not isinstance(self.module, str):
            self.module = str(self.module)

        if self.qualname is not None and not isinstance(self.qualname, str):
            self.qualname = str(self.qualname)

        if self.software is not None and not isinstance(self.software, SoftwareRequirement):
            self.software = SoftwareRequirement(**as_dict(self.software))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Case(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Case"]
    class_class_curie: ClassVar[str] = "tvbo:Case"
    class_name: ClassVar[str] = "Case"
    class_model_uri: ClassVar[URIRef] = TVBO.Case

    condition: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.condition is not None and not isinstance(self.condition, str):
            self.condition = str(self.condition)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class DerivedParameter(Parameter):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["DerivedParameter"]
    class_class_curie: ClassVar[str] = "tvbo:DerivedParameter"
    class_name: ClassVar[str] = "DerivedParameter"
    class_model_uri: ClassVar[URIRef] = TVBO.DerivedParameter

    name: Union[str, DerivedParameterName] = None
    symbol: Optional[str] = None
    description: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    unit: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, DerivedParameterName):
            self.name = DerivedParameterName(self.name)

        if self.symbol is not None and not isinstance(self.symbol, str):
            self.symbol = str(self.symbol)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class DerivedVariable(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["DerivedVariable"]
    class_class_curie: ClassVar[str] = "tvbo:DerivedVariable"
    class_name: ClassVar[str] = "DerivedVariable"
    class_model_uri: ClassVar[URIRef] = TVBO.DerivedVariable

    name: Union[str, DerivedVariableName] = None
    symbol: Optional[str] = None
    description: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    unit: Optional[str] = None
    conditional: Optional[Union[bool, Bool]] = False
    cases: Optional[Union[Union[dict, Case], list[Union[dict, Case]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, DerivedVariableName):
            self.name = DerivedVariableName(self.name)

        if self.symbol is not None and not isinstance(self.symbol, str):
            self.symbol = str(self.symbol)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        if self.conditional is not None and not isinstance(self.conditional, Bool):
            self.conditional = Bool(self.conditional)

        if not isinstance(self.cases, list):
            self.cases = [self.cases] if self.cases is not None else []
        self.cases = [v if isinstance(v, Case) else Case(**as_dict(v)) for v in self.cases]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Noise(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Noise"]
    class_class_curie: ClassVar[str] = "tvbo:Noise"
    class_name: ClassVar[str] = "Noise"
    class_model_uri: ClassVar[URIRef] = TVBO.Noise

    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    equation: Optional[Union[dict, Equation]] = None
    noise_type: Optional[str] = "gaussian"
    correlated: Optional[Union[bool, Bool]] = False
    gaussian: Optional[Union[bool, Bool]] = False
    additive: Optional[Union[bool, Bool]] = True
    seed: Optional[int] = 42
    random_state: Optional[Union[dict, "RandomStream"]] = None
    intensity: Optional[Union[dict, Parameter]] = None
    function: Optional[Union[dict, Function]] = None
    pycode: Optional[str] = None
    targets: Optional[Union[dict[Union[str, StateVariableName], Union[dict, StateVariable]], list[Union[dict, StateVariable]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.noise_type is not None and not isinstance(self.noise_type, str):
            self.noise_type = str(self.noise_type)

        if self.correlated is not None and not isinstance(self.correlated, Bool):
            self.correlated = Bool(self.correlated)

        if self.gaussian is not None and not isinstance(self.gaussian, Bool):
            self.gaussian = Bool(self.gaussian)

        if self.additive is not None and not isinstance(self.additive, Bool):
            self.additive = Bool(self.additive)

        if self.seed is not None and not isinstance(self.seed, int):
            self.seed = int(self.seed)

        if self.random_state is not None and not isinstance(self.random_state, RandomStream):
            self.random_state = RandomStream(**as_dict(self.random_state))

        if self.intensity is not None and not isinstance(self.intensity, Parameter):
            self.intensity = Parameter(**as_dict(self.intensity))

        if self.function is not None and not isinstance(self.function, Function):
            self.function = Function(**as_dict(self.function))

        if self.pycode is not None and not isinstance(self.pycode, str):
            self.pycode = str(self.pycode)

        self._normalize_inlined_as_dict(slot_name="targets", slot_type=StateVariable, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class RandomStream(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["RandomStream"]
    class_class_curie: ClassVar[str] = "tvbo:RandomStream"
    class_name: ClassVar[str] = "RandomStream"
    class_model_uri: ClassVar[URIRef] = TVBO.RandomStream

    label: Optional[str] = None
    description: Optional[str] = None
    dataLocation: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class CostFunction(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["CostFunction"]
    class_class_curie: ClassVar[str] = "tvbo:CostFunction"
    class_name: ClassVar[str] = "CostFunction"
    class_model_uri: ClassVar[URIRef] = TVBO.CostFunction

    label: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class FittingTarget(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["FittingTarget"]
    class_class_curie: ClassVar[str] = "tvbo:FittingTarget"
    class_name: ClassVar[str] = "FittingTarget"
    class_model_uri: ClassVar[URIRef] = TVBO.FittingTarget

    label: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    symbol: Optional[str] = None
    definition: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.symbol is not None and not isinstance(self.symbol, str):
            self.symbol = str(self.symbol)

        if self.definition is not None and not isinstance(self.definition, str):
            self.definition = str(self.definition)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ModelFitting(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["ModelFitting"]
    class_class_curie: ClassVar[str] = "tvbo:ModelFitting"
    class_name: ClassVar[str] = "ModelFitting"
    class_model_uri: ClassVar[URIRef] = TVBO.ModelFitting

    label: Optional[str] = None
    description: Optional[str] = None
    targets: Optional[Union[Union[dict, FittingTarget], list[Union[dict, FittingTarget]]]] = empty_list()
    cost_function: Optional[Union[dict, CostFunction]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if not isinstance(self.targets, list):
            self.targets = [self.targets] if self.targets is not None else []
        self.targets = [v if isinstance(v, FittingTarget) else FittingTarget(**as_dict(v)) for v in self.targets]

        if self.cost_function is not None and not isinstance(self.cost_function, CostFunction):
            self.cost_function = CostFunction(**as_dict(self.cost_function))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Integrator(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Integrator"]
    class_class_curie: ClassVar[str] = "tvbo:Integrator"
    class_name: ClassVar[str] = "Integrator"
    class_model_uri: ClassVar[URIRef] = TVBO.Integrator

    time_scale: Optional[str] = "ms"
    unit: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    duration: Optional[float] = 1000
    method: Optional[str] = None
    step_size: Optional[float] = 0.01220703125
    steps: Optional[int] = None
    noise: Optional[Union[dict, Noise]] = None
    state_wise_sigma: Optional[Union[float, list[float]]] = empty_list()
    transient_time: Optional[float] = 0
    scipy_ode_base: Optional[Union[bool, Bool]] = False
    number_of_stages: Optional[int] = 1
    intermediate_expressions: Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, DerivedVariable]], list[Union[dict, DerivedVariable]]]] = empty_dict()
    update_expression: Optional[Union[dict, DerivedVariable]] = None
    delayed: Optional[Union[bool, Bool]] = True

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.time_scale is not None and not isinstance(self.time_scale, str):
            self.time_scale = str(self.time_scale)

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.duration is not None and not isinstance(self.duration, float):
            self.duration = float(self.duration)

        if self.method is not None and not isinstance(self.method, str):
            self.method = str(self.method)

        if self.step_size is not None and not isinstance(self.step_size, float):
            self.step_size = float(self.step_size)

        if self.steps is not None and not isinstance(self.steps, int):
            self.steps = int(self.steps)

        if self.noise is not None and not isinstance(self.noise, Noise):
            self.noise = Noise(**as_dict(self.noise))

        if not isinstance(self.state_wise_sigma, list):
            self.state_wise_sigma = [self.state_wise_sigma] if self.state_wise_sigma is not None else []
        self.state_wise_sigma = [v if isinstance(v, float) else float(v) for v in self.state_wise_sigma]

        if self.transient_time is not None and not isinstance(self.transient_time, float):
            self.transient_time = float(self.transient_time)

        if self.scipy_ode_base is not None and not isinstance(self.scipy_ode_base, Bool):
            self.scipy_ode_base = Bool(self.scipy_ode_base)

        if self.number_of_stages is not None and not isinstance(self.number_of_stages, int):
            self.number_of_stages = int(self.number_of_stages)

        self._normalize_inlined_as_dict(slot_name="intermediate_expressions", slot_type=DerivedVariable, key_name="name", keyed=True)

        if self.update_expression is not None and not isinstance(self.update_expression, DerivedVariable):
            self.update_expression = DerivedVariable(**as_dict(self.update_expression))

        if self.delayed is not None and not isinstance(self.delayed, Bool):
            self.delayed = Bool(self.delayed)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Monitor(ObservationModel):
    """
    Observation model for monitoring simulation output with optional processing pipeline
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Monitor"]
    class_class_curie: ClassVar[str] = "tvbo:Monitor"
    class_name: ClassVar[str] = "Monitor"
    class_model_uri: ClassVar[URIRef] = TVBO.Monitor

    name: Union[str, MonitorName] = None
    time_scale: Optional[str] = "ms"
    label: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    acronym: Optional[str] = None
    description: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    environment: Optional[Union[dict, "SoftwareEnvironment"]] = None
    period: Optional[float] = None
    imaging_modality: Optional[Union[str, "ImagingModality"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, MonitorName):
            self.name = MonitorName(self.name)

        if self.time_scale is not None and not isinstance(self.time_scale, str):
            self.time_scale = str(self.time_scale)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.acronym is not None and not isinstance(self.acronym, str):
            self.acronym = str(self.acronym)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.environment is not None and not isinstance(self.environment, SoftwareEnvironment):
            self.environment = SoftwareEnvironment(**as_dict(self.environment))

        if self.period is not None and not isinstance(self.period, float):
            self.period = float(self.period)

        if self.imaging_modality is not None and not isinstance(self.imaging_modality, ImagingModality):
            self.imaging_modality = ImagingModality(self.imaging_modality)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Coupling(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Coupling"]
    class_class_curie: ClassVar[str] = "tvbo:Coupling"
    class_name: ClassVar[str] = "Coupling"
    class_model_uri: ClassVar[URIRef] = TVBO.Coupling

    name: Union[str, CouplingName] = "Linear"
    label: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    coupling_function: Optional[Union[dict, Equation]] = None
    sparse: Optional[Union[bool, Bool]] = False
    pre_expression: Optional[Union[dict, Equation]] = None
    post_expression: Optional[Union[dict, Equation]] = None
    incoming_states: Optional[Union[dict, StateVariable]] = None
    local_states: Optional[Union[dict, StateVariable]] = None
    delayed: Optional[Union[bool, Bool]] = True
    inner_coupling: Optional[Union[dict, "Coupling"]] = None
    region_mapping: Optional[Union[dict, "RegionMapping"]] = None
    regional_connectivity: Optional[Union[dict, Network]] = None
    aggregation: Optional[str] = None
    distribution: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, CouplingName):
            self.name = CouplingName(self.name)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.coupling_function is not None and not isinstance(self.coupling_function, Equation):
            self.coupling_function = Equation(**as_dict(self.coupling_function))

        if self.sparse is not None and not isinstance(self.sparse, Bool):
            self.sparse = Bool(self.sparse)

        if self.pre_expression is not None and not isinstance(self.pre_expression, Equation):
            self.pre_expression = Equation(**as_dict(self.pre_expression))

        if self.post_expression is not None and not isinstance(self.post_expression, Equation):
            self.post_expression = Equation(**as_dict(self.post_expression))

        if self.incoming_states is not None and not isinstance(self.incoming_states, StateVariable):
            self.incoming_states = StateVariable(**as_dict(self.incoming_states))

        if self.local_states is not None and not isinstance(self.local_states, StateVariable):
            self.local_states = StateVariable(**as_dict(self.local_states))

        if self.delayed is not None and not isinstance(self.delayed, Bool):
            self.delayed = Bool(self.delayed)

        if self.inner_coupling is not None and not isinstance(self.inner_coupling, Coupling):
            self.inner_coupling = Coupling(**as_dict(self.inner_coupling))

        if self.region_mapping is not None and not isinstance(self.region_mapping, RegionMapping):
            self.region_mapping = RegionMapping(**as_dict(self.region_mapping))

        if self.regional_connectivity is not None and not isinstance(self.regional_connectivity, Network):
            self.regional_connectivity = Network(**as_dict(self.regional_connectivity))

        if self.aggregation is not None and not isinstance(self.aggregation, str):
            self.aggregation = str(self.aggregation)

        if self.distribution is not None and not isinstance(self.distribution, str):
            self.distribution = str(self.distribution)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class RegionMapping(YAMLRoot):
    """
    Maps vertices to parent regions for hierarchical/aggregated coupling
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["RegionMapping"]
    class_class_curie: ClassVar[str] = "tvbo:RegionMapping"
    class_name: ClassVar[str] = "RegionMapping"
    class_model_uri: ClassVar[URIRef] = TVBO.RegionMapping

    label: Optional[str] = None
    description: Optional[str] = None
    dataLocation: Optional[str] = None
    vertex_to_region: Optional[Union[int, list[int]]] = empty_list()
    n_vertices: Optional[int] = None
    n_regions: Optional[int] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        if not isinstance(self.vertex_to_region, list):
            self.vertex_to_region = [self.vertex_to_region] if self.vertex_to_region is not None else []
        self.vertex_to_region = [v if isinstance(v, int) else int(v) for v in self.vertex_to_region]

        if self.n_vertices is not None and not isinstance(self.n_vertices, int):
            self.n_vertices = int(self.n_vertices)

        if self.n_regions is not None and not isinstance(self.n_regions, int):
            self.n_regions = int(self.n_regions)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Sample(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Sample"]
    class_class_curie: ClassVar[str] = "tvbo:Sample"
    class_name: ClassVar[str] = "Sample"
    class_model_uri: ClassVar[URIRef] = TVBO.Sample

    groups: Optional[Union[str, list[str]]] = empty_list()
    size: Optional[int] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if not isinstance(self.groups, list):
            self.groups = [self.groups] if self.groups is not None else []
        self.groups = [v if isinstance(v, str) else str(v) for v in self.groups]

        if self.size is not None and not isinstance(self.size, int):
            self.size = int(self.size)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class SimulationExperiment(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Simulation"]
    class_class_curie: ClassVar[str] = "tvbo:Simulation"
    class_name: ClassVar[str] = "SimulationExperiment"
    class_model_uri: ClassVar[URIRef] = TVBO.SimulationExperiment

    id: Union[int, SimulationExperimentId] = None
    model: Optional[Union[str, DynamicsName]] = None
    description: Optional[str] = None
    additional_equations: Optional[Union[Union[dict, Equation], list[Union[dict, Equation]]]] = empty_list()
    label: Optional[str] = None
    local_dynamics: Optional[Union[dict, Dynamics]] = None
    dynamics: Optional[Union[dict[Union[str, DynamicsName], Union[dict, Dynamics]], list[Union[dict, Dynamics]]]] = empty_dict()
    integration: Optional[Union[dict, Integrator]] = None
    connectivity: Optional[Union[dict, Network]] = None
    network: Optional[Union[dict, Network]] = None
    coupling: Optional[Union[dict, Coupling]] = None
    monitors: Optional[Union[dict[Union[str, MonitorName], Union[dict, Monitor]], list[Union[dict, Monitor]]]] = empty_dict()
    stimulation: Optional[Union[dict, Stimulus]] = None
    field_dynamics: Optional[Union[dict, "PDE"]] = None
    modelfitting: Optional[Union[Union[dict, ModelFitting], list[Union[dict, ModelFitting]]]] = empty_list()
    environment: Optional[Union[dict, "SoftwareEnvironment"]] = None
    software: Optional[Union[dict, "SoftwareRequirement"]] = None
    references: Optional[Union[str, list[str]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, SimulationExperimentId):
            self.id = SimulationExperimentId(self.id)

        if self.model is not None and not isinstance(self.model, DynamicsName):
            self.model = DynamicsName(self.model)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if not isinstance(self.additional_equations, list):
            self.additional_equations = [self.additional_equations] if self.additional_equations is not None else []
        self.additional_equations = [v if isinstance(v, Equation) else Equation(**as_dict(v)) for v in self.additional_equations]

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.local_dynamics is not None and not isinstance(self.local_dynamics, Dynamics):
            self.local_dynamics = Dynamics(**as_dict(self.local_dynamics))

        self._normalize_inlined_as_list(slot_name="dynamics", slot_type=Dynamics, key_name="name", keyed=True)

        if self.integration is not None and not isinstance(self.integration, Integrator):
            self.integration = Integrator(**as_dict(self.integration))

        if self.connectivity is not None and not isinstance(self.connectivity, Network):
            self.connectivity = Network(**as_dict(self.connectivity))

        if self.network is not None and not isinstance(self.network, Network):
            self.network = Network(**as_dict(self.network))

        if self.coupling is not None and not isinstance(self.coupling, Coupling):
            self.coupling = Coupling(**as_dict(self.coupling))

        self._normalize_inlined_as_dict(slot_name="monitors", slot_type=Monitor, key_name="name", keyed=True)

        if self.stimulation is not None and not isinstance(self.stimulation, Stimulus):
            self.stimulation = Stimulus(**as_dict(self.stimulation))

        if self.field_dynamics is not None and not isinstance(self.field_dynamics, PDE):
            self.field_dynamics = PDE(**as_dict(self.field_dynamics))

        if not isinstance(self.modelfitting, list):
            self.modelfitting = [self.modelfitting] if self.modelfitting is not None else []
        self.modelfitting = [v if isinstance(v, ModelFitting) else ModelFitting(**as_dict(v)) for v in self.modelfitting]

        if self.environment is not None and not isinstance(self.environment, SoftwareEnvironment):
            self.environment = SoftwareEnvironment(**as_dict(self.environment))

        if self.software is not None and not isinstance(self.software, SoftwareRequirement):
            self.software = SoftwareRequirement(**as_dict(self.software))

        if not isinstance(self.references, list):
            self.references = [self.references] if self.references is not None else []
        self.references = [v if isinstance(v, str) else str(v) for v in self.references]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class SimulationStudy(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["SimulationStudy"]
    class_class_curie: ClassVar[str] = "tvbo:SimulationStudy"
    class_name: ClassVar[str] = "SimulationStudy"
    class_model_uri: ClassVar[URIRef] = TVBO.SimulationStudy

    label: Optional[str] = None
    derived_from: Optional[str] = None
    model: Optional[Union[str, DynamicsName]] = None
    description: Optional[str] = None
    key: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    sample: Optional[Union[dict, Sample]] = None
    simulation_experiments: Optional[Union[dict[Union[int, SimulationExperimentId], Union[dict, SimulationExperiment]], list[Union[dict, SimulationExperiment]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.derived_from is not None and not isinstance(self.derived_from, str):
            self.derived_from = str(self.derived_from)

        if self.model is not None and not isinstance(self.model, DynamicsName):
            self.model = DynamicsName(self.model)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.key is not None and not isinstance(self.key, str):
            self.key = str(self.key)

        if self.title is not None and not isinstance(self.title, str):
            self.title = str(self.title)

        if self.year is not None and not isinstance(self.year, int):
            self.year = int(self.year)

        if self.doi is not None and not isinstance(self.doi, str):
            self.doi = str(self.doi)

        if self.sample is not None and not isinstance(self.sample, Sample):
            self.sample = Sample(**as_dict(self.sample))

        self._normalize_inlined_as_list(slot_name="simulation_experiments", slot_type=SimulationExperiment, key_name="id", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class TimeSeries(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["TimeSeries"]
    class_class_curie: ClassVar[str] = "tvbo:TimeSeries"
    class_name: ClassVar[str] = "TimeSeries"
    class_model_uri: ClassVar[URIRef] = TVBO.TimeSeries

    label: Optional[str] = None
    description: Optional[str] = None
    dataLocation: Optional[str] = None
    data: Optional[Union[dict, Matrix]] = None
    time: Optional[Union[dict, Matrix]] = None
    sampling_rate: Optional[float] = None
    unit: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        if self.data is not None and not isinstance(self.data, Matrix):
            self.data = Matrix(**as_dict(self.data))

        if self.time is not None and not isinstance(self.time, Matrix):
            self.time = Matrix(**as_dict(self.time))

        if self.sampling_rate is not None and not isinstance(self.sampling_rate, float):
            self.sampling_rate = float(self.sampling_rate)

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class SoftwareEnvironment(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["SoftwareEnvironment"]
    class_class_curie: ClassVar[str] = "tvbo:SoftwareEnvironment"
    class_name: ClassVar[str] = "SoftwareEnvironment"
    class_model_uri: ClassVar[URIRef] = TVBO.SoftwareEnvironment

    label: Optional[str] = None
    description: Optional[str] = None
    dataLocation: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    platform: Optional[str] = None
    environment_type: Optional[Union[str, "EnvironmentType"]] = None
    container_image: Optional[str] = None
    build_hash: Optional[str] = None
    requirements: Optional[Union[dict[Union[str, SoftwareRequirementName], Union[dict, "SoftwareRequirement"]], list[Union[dict, "SoftwareRequirement"]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        if self.name is not None and not isinstance(self.name, str):
            self.name = str(self.name)

        if self.version is not None and not isinstance(self.version, str):
            self.version = str(self.version)

        if self.platform is not None and not isinstance(self.platform, str):
            self.platform = str(self.platform)

        if self.environment_type is not None and not isinstance(self.environment_type, EnvironmentType):
            self.environment_type = EnvironmentType(self.environment_type)

        if self.container_image is not None and not isinstance(self.container_image, str):
            self.container_image = str(self.container_image)

        if self.build_hash is not None and not isinstance(self.build_hash, str):
            self.build_hash = str(self.build_hash)

        self._normalize_inlined_as_dict(slot_name="requirements", slot_type=SoftwareRequirement, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class SoftwareRequirement(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["SoftwareRequirement"]
    class_class_curie: ClassVar[str] = "tvbo:SoftwareRequirement"
    class_name: ClassVar[str] = "SoftwareRequirement"
    class_model_uri: ClassVar[URIRef] = TVBO.SoftwareRequirement

    name: Union[str, SoftwareRequirementName] = None
    package: Union[str, SoftwarePackageName] = None
    description: Optional[str] = None
    dataLocation: Optional[str] = None
    version_spec: Optional[str] = None
    role: Optional[Union[str, "RequirementRole"]] = 'runtime'
    optional: Optional[Union[bool, Bool]] = False
    hash: Optional[str] = None
    source_url: Optional[str] = None
    url: Optional[str] = None
    license: Optional[str] = None
    modules: Optional[Union[str, list[str]]] = empty_list()
    version: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, SoftwareRequirementName):
            self.name = SoftwareRequirementName(self.name)

        if self._is_empty(self.package):
            self.MissingRequiredField("package")
        if not isinstance(self.package, SoftwarePackageName):
            self.package = SoftwarePackageName(self.package)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        if self.version_spec is not None and not isinstance(self.version_spec, str):
            self.version_spec = str(self.version_spec)

        if self.role is not None and not isinstance(self.role, RequirementRole):
            self.role = getattr(RequirementRole, self.role)

        if self.optional is not None and not isinstance(self.optional, Bool):
            self.optional = Bool(self.optional)

        if self.hash is not None and not isinstance(self.hash, str):
            self.hash = str(self.hash)

        if self.source_url is not None and not isinstance(self.source_url, str):
            self.source_url = str(self.source_url)

        if self.url is not None and not isinstance(self.url, str):
            self.url = str(self.url)

        if self.license is not None and not isinstance(self.license, str):
            self.license = str(self.license)

        if not isinstance(self.modules, list):
            self.modules = [self.modules] if self.modules is not None else []
        self.modules = [v if isinstance(v, str) else str(v) for v in self.modules]

        if self.version is not None and not isinstance(self.version, str):
            self.version = str(self.version)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class SoftwarePackage(YAMLRoot):
    """
    Identity information about a software package independent of a specific version requirement.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["SoftwarePackage"]
    class_class_curie: ClassVar[str] = "tvbo:SoftwarePackage"
    class_name: ClassVar[str] = "SoftwarePackage"
    class_model_uri: ClassVar[URIRef] = TVBO.SoftwarePackage

    name: Union[str, SoftwarePackageName] = None
    description: Optional[str] = None
    homepage: Optional[str] = None
    license: Optional[str] = None
    repository: Optional[str] = None
    doi: Optional[str] = None
    ecosystem: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, SoftwarePackageName):
            self.name = SoftwarePackageName(self.name)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.homepage is not None and not isinstance(self.homepage, str):
            self.homepage = str(self.homepage)

        if self.license is not None and not isinstance(self.license, str):
            self.license = str(self.license)

        if self.repository is not None and not isinstance(self.repository, str):
            self.repository = str(self.repository)

        if self.doi is not None and not isinstance(self.doi, str):
            self.doi = str(self.doi)

        if self.ecosystem is not None and not isinstance(self.ecosystem, str):
            self.ecosystem = str(self.ecosystem)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class NDArray(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["NDArray"]
    class_class_curie: ClassVar[str] = "tvbo:NDArray"
    class_name: ClassVar[str] = "NDArray"
    class_model_uri: ClassVar[URIRef] = TVBO.NDArray

    label: Optional[str] = None
    description: Optional[str] = None
    shape: Optional[Union[int, list[int]]] = empty_list()
    dtype: Optional[str] = None
    dataLocation: Optional[str] = None
    unit: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if not isinstance(self.shape, list):
            self.shape = [self.shape] if self.shape is not None else []
        self.shape = [v if isinstance(v, int) else int(v) for v in self.shape]

        if self.dtype is not None and not isinstance(self.dtype, str):
            self.dtype = str(self.dtype)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class SpatialDomain(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["SpatialDomain"]
    class_class_curie: ClassVar[str] = "tvbo:SpatialDomain"
    class_name: ClassVar[str] = "SpatialDomain"
    class_model_uri: ClassVar[URIRef] = TVBO.SpatialDomain

    label: Optional[str] = None
    description: Optional[str] = None
    coordinate_space: Optional[Union[str, CommonCoordinateSpaceName]] = None
    region: Optional[str] = None
    geometry: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.coordinate_space is not None and not isinstance(self.coordinate_space, CommonCoordinateSpaceName):
            self.coordinate_space = CommonCoordinateSpaceName(self.coordinate_space)

        if self.region is not None and not isinstance(self.region, str):
            self.region = str(self.region)

        if self.geometry is not None and not isinstance(self.geometry, str):
            self.geometry = str(self.geometry)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Mesh(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Mesh"]
    class_class_curie: ClassVar[str] = "tvbo:Mesh"
    class_name: ClassVar[str] = "Mesh"
    class_model_uri: ClassVar[URIRef] = TVBO.Mesh

    label: Optional[str] = None
    description: Optional[str] = None
    dataLocation: Optional[str] = None
    element_type: Optional[Union[str, "ElementType"]] = None
    coordinates: Optional[Union[Union[dict, "Coordinate"], list[Union[dict, "Coordinate"]]]] = empty_list()
    elements: Optional[str] = None
    coordinate_space: Optional[Union[str, CommonCoordinateSpaceName]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        if self.element_type is not None and not isinstance(self.element_type, ElementType):
            self.element_type = ElementType(self.element_type)

        if not isinstance(self.coordinates, list):
            self.coordinates = [self.coordinates] if self.coordinates is not None else []
        self.coordinates = [v if isinstance(v, Coordinate) else Coordinate(**as_dict(v)) for v in self.coordinates]

        if self.elements is not None and not isinstance(self.elements, str):
            self.elements = str(self.elements)

        if self.coordinate_space is not None and not isinstance(self.coordinate_space, CommonCoordinateSpaceName):
            self.coordinate_space = CommonCoordinateSpaceName(self.coordinate_space)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class SpatialField(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["SpatialField"]
    class_class_curie: ClassVar[str] = "tvbo:SpatialField"
    class_name: ClassVar[str] = "SpatialField"
    class_model_uri: ClassVar[URIRef] = TVBO.SpatialField

    label: Optional[str] = None
    description: Optional[str] = None
    quantity_kind: Optional[str] = None
    unit: Optional[str] = None
    mesh: Optional[Union[dict, Mesh]] = None
    values: Optional[Union[dict, NDArray]] = None
    time_dependent: Optional[Union[bool, Bool]] = False
    initial_value: Optional[float] = 0.1
    initial_expression: Optional[Union[dict, Equation]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.quantity_kind is not None and not isinstance(self.quantity_kind, str):
            self.quantity_kind = str(self.quantity_kind)

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        if self.mesh is not None and not isinstance(self.mesh, Mesh):
            self.mesh = Mesh(**as_dict(self.mesh))

        if self.values is not None and not isinstance(self.values, NDArray):
            self.values = NDArray(**as_dict(self.values))

        if self.time_dependent is not None and not isinstance(self.time_dependent, Bool):
            self.time_dependent = Bool(self.time_dependent)

        if self.initial_value is not None and not isinstance(self.initial_value, float):
            self.initial_value = float(self.initial_value)

        if self.initial_expression is not None and not isinstance(self.initial_expression, Equation):
            self.initial_expression = Equation(**as_dict(self.initial_expression))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class FieldStateVariable(StateVariable):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["FieldStateVariable"]
    class_class_curie: ClassVar[str] = "tvbo:FieldStateVariable"
    class_name: ClassVar[str] = "FieldStateVariable"
    class_model_uri: ClassVar[URIRef] = TVBO.FieldStateVariable

    name: Union[str, FieldStateVariableName] = None
    label: Optional[str] = None
    description: Optional[str] = None
    mesh: Optional[Union[dict, Mesh]] = None
    boundary_conditions: Optional[Union[Union[dict, "BoundaryCondition"], list[Union[dict, "BoundaryCondition"]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, FieldStateVariableName):
            self.name = FieldStateVariableName(self.name)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.mesh is not None and not isinstance(self.mesh, Mesh):
            self.mesh = Mesh(**as_dict(self.mesh))

        if not isinstance(self.boundary_conditions, list):
            self.boundary_conditions = [self.boundary_conditions] if self.boundary_conditions is not None else []
        self.boundary_conditions = [v if isinstance(v, BoundaryCondition) else BoundaryCondition(**as_dict(v)) for v in self.boundary_conditions]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class DifferentialOperator(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["DifferentialOperator"]
    class_class_curie: ClassVar[str] = "tvbo:DifferentialOperator"
    class_name: ClassVar[str] = "DifferentialOperator"
    class_model_uri: ClassVar[URIRef] = TVBO.DifferentialOperator

    label: Optional[str] = None
    definition: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    operator_type: Optional[Union[str, "OperatorType"]] = None
    coefficient: Optional[Union[str, ParameterName]] = None
    tensor_coefficient: Optional[Union[str, ParameterName]] = None
    expression: Optional[Union[dict, Equation]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.definition is not None and not isinstance(self.definition, str):
            self.definition = str(self.definition)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.operator_type is not None and not isinstance(self.operator_type, OperatorType):
            self.operator_type = OperatorType(self.operator_type)

        if self.coefficient is not None and not isinstance(self.coefficient, ParameterName):
            self.coefficient = ParameterName(self.coefficient)

        if self.tensor_coefficient is not None and not isinstance(self.tensor_coefficient, ParameterName):
            self.tensor_coefficient = ParameterName(self.tensor_coefficient)

        if self.expression is not None and not isinstance(self.expression, Equation):
            self.expression = Equation(**as_dict(self.expression))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class BoundaryCondition(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["BoundaryCondition"]
    class_class_curie: ClassVar[str] = "tvbo:BoundaryCondition"
    class_name: ClassVar[str] = "BoundaryCondition"
    class_model_uri: ClassVar[URIRef] = TVBO.BoundaryCondition

    label: Optional[str] = None
    description: Optional[str] = None
    bc_type: Optional[Union[str, "BoundaryConditionType"]] = None
    on_region: Optional[str] = None
    value: Optional[Union[dict, Equation]] = None
    time_dependent: Optional[Union[bool, Bool]] = False

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.bc_type is not None and not isinstance(self.bc_type, BoundaryConditionType):
            self.bc_type = BoundaryConditionType(self.bc_type)

        if self.on_region is not None and not isinstance(self.on_region, str):
            self.on_region = str(self.on_region)

        if self.value is not None and not isinstance(self.value, Equation):
            self.value = Equation(**as_dict(self.value))

        if self.time_dependent is not None and not isinstance(self.time_dependent, Bool):
            self.time_dependent = Bool(self.time_dependent)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class PDESolver(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["PDESolver"]
    class_class_curie: ClassVar[str] = "tvbo:PDESolver"
    class_name: ClassVar[str] = "PDESolver"
    class_model_uri: ClassVar[URIRef] = TVBO.PDESolver

    label: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[Union[Union[str, SoftwareRequirementName], list[Union[str, SoftwareRequirementName]]]] = empty_list()
    environment: Optional[Union[dict, SoftwareEnvironment]] = None
    discretization: Optional[Union[str, "DiscretizationMethod"]] = None
    time_integrator: Optional[str] = None
    step_size: Optional[float] = None
    tolerances: Optional[str] = None
    preconditioner: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if not isinstance(self.requirements, list):
            self.requirements = [self.requirements] if self.requirements is not None else []
        self.requirements = [v if isinstance(v, SoftwareRequirementName) else SoftwareRequirementName(v) for v in self.requirements]

        if self.environment is not None and not isinstance(self.environment, SoftwareEnvironment):
            self.environment = SoftwareEnvironment(**as_dict(self.environment))

        if self.discretization is not None and not isinstance(self.discretization, DiscretizationMethod):
            self.discretization = DiscretizationMethod(self.discretization)

        if self.time_integrator is not None and not isinstance(self.time_integrator, str):
            self.time_integrator = str(self.time_integrator)

        if self.step_size is not None and not isinstance(self.step_size, float):
            self.step_size = float(self.step_size)

        if self.tolerances is not None and not isinstance(self.tolerances, str):
            self.tolerances = str(self.tolerances)

        if self.preconditioner is not None and not isinstance(self.preconditioner, str):
            self.preconditioner = str(self.preconditioner)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class PDE(YAMLRoot):
    """
    Partial differential equation problem definition.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["PDE"]
    class_class_curie: ClassVar[str] = "tvbo:PDE"
    class_name: ClassVar[str] = "PDE"
    class_model_uri: ClassVar[URIRef] = TVBO.PDE

    label: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    domain: Optional[Union[dict, SpatialDomain]] = None
    mesh: Optional[Union[dict, Mesh]] = None
    state_variables: Optional[Union[Union[str, FieldStateVariableName], list[Union[str, FieldStateVariableName]]]] = empty_list()
    field: Optional[Union[dict, SpatialField]] = None
    operators: Optional[Union[Union[dict, DifferentialOperator], list[Union[dict, DifferentialOperator]]]] = empty_list()
    sources: Optional[Union[Union[dict, Equation], list[Union[dict, Equation]]]] = empty_list()
    boundary_conditions: Optional[Union[Union[dict, BoundaryCondition], list[Union[dict, BoundaryCondition]]]] = empty_list()
    solver: Optional[Union[dict, PDESolver]] = None
    derived_parameters: Optional[Union[Union[str, DerivedParameterName], list[Union[str, DerivedParameterName]]]] = empty_list()
    derived_variables: Optional[Union[Union[str, DerivedVariableName], list[Union[str, DerivedVariableName]]]] = empty_list()
    functions: Optional[Union[Union[str, FunctionName], list[Union[str, FunctionName]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.domain is not None and not isinstance(self.domain, SpatialDomain):
            self.domain = SpatialDomain(**as_dict(self.domain))

        if self.mesh is not None and not isinstance(self.mesh, Mesh):
            self.mesh = Mesh(**as_dict(self.mesh))

        if not isinstance(self.state_variables, list):
            self.state_variables = [self.state_variables] if self.state_variables is not None else []
        self.state_variables = [v if isinstance(v, FieldStateVariableName) else FieldStateVariableName(v) for v in self.state_variables]

        if self.field is not None and not isinstance(self.field, SpatialField):
            self.field = SpatialField(**as_dict(self.field))

        if not isinstance(self.operators, list):
            self.operators = [self.operators] if self.operators is not None else []
        self.operators = [v if isinstance(v, DifferentialOperator) else DifferentialOperator(**as_dict(v)) for v in self.operators]

        if not isinstance(self.sources, list):
            self.sources = [self.sources] if self.sources is not None else []
        self.sources = [v if isinstance(v, Equation) else Equation(**as_dict(v)) for v in self.sources]

        if not isinstance(self.boundary_conditions, list):
            self.boundary_conditions = [self.boundary_conditions] if self.boundary_conditions is not None else []
        self.boundary_conditions = [v if isinstance(v, BoundaryCondition) else BoundaryCondition(**as_dict(v)) for v in self.boundary_conditions]

        if self.solver is not None and not isinstance(self.solver, PDESolver):
            self.solver = PDESolver(**as_dict(self.solver))

        if not isinstance(self.derived_parameters, list):
            self.derived_parameters = [self.derived_parameters] if self.derived_parameters is not None else []
        self.derived_parameters = [v if isinstance(v, DerivedParameterName) else DerivedParameterName(v) for v in self.derived_parameters]

        if not isinstance(self.derived_variables, list):
            self.derived_variables = [self.derived_variables] if self.derived_variables is not None else []
        self.derived_variables = [v if isinstance(v, DerivedVariableName) else DerivedVariableName(v) for v in self.derived_variables]

        if not isinstance(self.functions, list):
            self.functions = [self.functions] if self.functions is not None else []
        self.functions = [v if isinstance(v, FunctionName) else FunctionName(v) for v in self.functions]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Coordinate(YAMLRoot):
    """
    A 3D coordinate with X, Y, Z values.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ATOM["Coordinate"]
    class_class_curie: ClassVar[str] = "atom:Coordinate"
    class_name: ClassVar[str] = "Coordinate"
    class_model_uri: ClassVar[URIRef] = TVBO.Coordinate

    coordinateSpace: Optional[Union[str, CommonCoordinateSpaceName]] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.coordinateSpace is not None and not isinstance(self.coordinateSpace, CommonCoordinateSpaceName):
            self.coordinateSpace = CommonCoordinateSpaceName(self.coordinateSpace)

        if self.x is not None and not isinstance(self.x, float):
            self.x = float(self.x)

        if self.y is not None and not isinstance(self.y, float):
            self.y = float(self.y)

        if self.z is not None and not isinstance(self.z, float):
            self.z = float(self.z)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class BrainAtlas(YAMLRoot):
    """
    A schema for representing a version of a brain atlas.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ATOM["atlas/Atlas"]
    class_class_curie: ClassVar[str] = "atom:atlas/Atlas"
    class_name: ClassVar[str] = "BrainAtlas"
    class_model_uri: ClassVar[URIRef] = TVBO.BrainAtlas

    name: Union[str, BrainAtlasName] = None
    coordinateSpace: Optional[Union[str, CommonCoordinateSpaceName]] = None
    abbreviation: Optional[str] = None
    author: Optional[Union[str, list[str]]] = empty_list()
    isVersionOf: Optional[str] = None
    versionIdentifier: Optional[str] = None
    terminology: Optional[Union[dict, "ParcellationTerminology"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, BrainAtlasName):
            self.name = BrainAtlasName(self.name)

        if self.coordinateSpace is not None and not isinstance(self.coordinateSpace, CommonCoordinateSpaceName):
            self.coordinateSpace = CommonCoordinateSpaceName(self.coordinateSpace)

        if self.abbreviation is not None and not isinstance(self.abbreviation, str):
            self.abbreviation = str(self.abbreviation)

        if not isinstance(self.author, list):
            self.author = [self.author] if self.author is not None else []
        self.author = [v if isinstance(v, str) else str(v) for v in self.author]

        if self.isVersionOf is not None and not isinstance(self.isVersionOf, str):
            self.isVersionOf = str(self.isVersionOf)

        if self.versionIdentifier is not None and not isinstance(self.versionIdentifier, str):
            self.versionIdentifier = str(self.versionIdentifier)

        if self.terminology is not None and not isinstance(self.terminology, ParcellationTerminology):
            self.terminology = ParcellationTerminology(**as_dict(self.terminology))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class CommonCoordinateSpace(YAMLRoot):
    """
    A schema for representing a version of a common coordinate space.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ATOM["atlas/Transformation"]
    class_class_curie: ClassVar[str] = "atom:atlas/Transformation"
    class_name: ClassVar[str] = "CommonCoordinateSpace"
    class_model_uri: ClassVar[URIRef] = TVBO.CommonCoordinateSpace

    name: Union[str, CommonCoordinateSpaceName] = None
    abbreviation: Optional[str] = None
    unit: Optional[str] = None
    license: Optional[str] = None
    anatomicalAxesOrientation: Optional[str] = None
    axesOrigin: Optional[str] = None
    nativeUnit: Optional[str] = None
    defaultImage: Optional[Union[str, list[str]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, CommonCoordinateSpaceName):
            self.name = CommonCoordinateSpaceName(self.name)

        if self.abbreviation is not None and not isinstance(self.abbreviation, str):
            self.abbreviation = str(self.abbreviation)

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        if self.license is not None and not isinstance(self.license, str):
            self.license = str(self.license)

        if self.anatomicalAxesOrientation is not None and not isinstance(self.anatomicalAxesOrientation, str):
            self.anatomicalAxesOrientation = str(self.anatomicalAxesOrientation)

        if self.axesOrigin is not None and not isinstance(self.axesOrigin, str):
            self.axesOrigin = str(self.axesOrigin)

        if self.nativeUnit is not None and not isinstance(self.nativeUnit, str):
            self.nativeUnit = str(self.nativeUnit)

        if not isinstance(self.defaultImage, list):
            self.defaultImage = [self.defaultImage] if self.defaultImage is not None else []
        self.defaultImage = [v if isinstance(v, str) else str(v) for v in self.defaultImage]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ParcellationEntity(YAMLRoot):
    """
    A schema for representing a parcellation entity, which is an anatomical location or study target.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ATOM["atlas/Region"]
    class_class_curie: ClassVar[str] = "atom:atlas/Region"
    class_name: ClassVar[str] = "ParcellationEntity"
    class_model_uri: ClassVar[URIRef] = TVBO.ParcellationEntity

    name: Union[str, ParcellationEntityName] = None
    abbreviation: Optional[str] = None
    alternateName: Optional[Union[str, list[str]]] = empty_list()
    lookupLabel: Optional[int] = None
    hasParent: Optional[Union[Union[str, ParcellationEntityName], list[Union[str, ParcellationEntityName]]]] = empty_list()
    ontologyIdentifier: Optional[Union[str, list[str]]] = empty_list()
    versionIdentifier: Optional[str] = None
    relatedUBERONTerm: Optional[str] = None
    originalLookupLabel: Optional[int] = None
    hemisphere: Optional[Union[str, "Hemisphere"]] = None
    center: Optional[Union[dict, Coordinate]] = None
    color: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, ParcellationEntityName):
            self.name = ParcellationEntityName(self.name)

        if self.abbreviation is not None and not isinstance(self.abbreviation, str):
            self.abbreviation = str(self.abbreviation)

        if not isinstance(self.alternateName, list):
            self.alternateName = [self.alternateName] if self.alternateName is not None else []
        self.alternateName = [v if isinstance(v, str) else str(v) for v in self.alternateName]

        if self.lookupLabel is not None and not isinstance(self.lookupLabel, int):
            self.lookupLabel = int(self.lookupLabel)

        if not isinstance(self.hasParent, list):
            self.hasParent = [self.hasParent] if self.hasParent is not None else []
        self.hasParent = [v if isinstance(v, ParcellationEntityName) else ParcellationEntityName(v) for v in self.hasParent]

        if not isinstance(self.ontologyIdentifier, list):
            self.ontologyIdentifier = [self.ontologyIdentifier] if self.ontologyIdentifier is not None else []
        self.ontologyIdentifier = [v if isinstance(v, str) else str(v) for v in self.ontologyIdentifier]

        if self.versionIdentifier is not None and not isinstance(self.versionIdentifier, str):
            self.versionIdentifier = str(self.versionIdentifier)

        if self.relatedUBERONTerm is not None and not isinstance(self.relatedUBERONTerm, str):
            self.relatedUBERONTerm = str(self.relatedUBERONTerm)

        if self.originalLookupLabel is not None and not isinstance(self.originalLookupLabel, int):
            self.originalLookupLabel = int(self.originalLookupLabel)

        if self.hemisphere is not None and not isinstance(self.hemisphere, Hemisphere):
            self.hemisphere = Hemisphere(self.hemisphere)

        if self.center is not None and not isinstance(self.center, Coordinate):
            self.center = Coordinate(**as_dict(self.center))

        if self.color is not None and not isinstance(self.color, str):
            self.color = str(self.color)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ParcellationTerminology(YAMLRoot):
    """
    A schema for representing a parcellation terminology, which consists of parcellation entities.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ATOM["parcellationTerminology"]
    class_class_curie: ClassVar[str] = "atom:parcellationTerminology"
    class_name: ClassVar[str] = "ParcellationTerminology"
    class_model_uri: ClassVar[URIRef] = TVBO.ParcellationTerminology

    label: Optional[str] = None
    dataLocation: Optional[str] = None
    ontologyIdentifier: Optional[Union[str, list[str]]] = empty_list()
    versionIdentifier: Optional[str] = None
    entities: Optional[Union[Union[str, ParcellationEntityName], list[Union[str, ParcellationEntityName]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        if not isinstance(self.ontologyIdentifier, list):
            self.ontologyIdentifier = [self.ontologyIdentifier] if self.ontologyIdentifier is not None else []
        self.ontologyIdentifier = [v if isinstance(v, str) else str(v) for v in self.ontologyIdentifier]

        if self.versionIdentifier is not None and not isinstance(self.versionIdentifier, str):
            self.versionIdentifier = str(self.versionIdentifier)

        if not isinstance(self.entities, list):
            self.entities = [self.entities] if self.entities is not None else []
        self.entities = [v if isinstance(v, ParcellationEntityName) else ParcellationEntityName(v) for v in self.entities]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Dataset(YAMLRoot):
    """
    Collection of data related to a specific DBS study.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO_DBS["Dataset"]
    class_class_curie: ClassVar[str] = "tvbo_dbs:Dataset"
    class_name: ClassVar[str] = "Dataset"
    class_model_uri: ClassVar[URIRef] = TVBO.Dataset

    label: Optional[str] = None
    dataset_id: Optional[str] = None
    subjects: Optional[Union[dict[Union[str, SubjectSubjectId], Union[dict, "Subject"]], list[Union[dict, "Subject"]]]] = empty_dict()
    clinical_scores: Optional[Union[Union[dict, "ClinicalScore"], list[Union[dict, "ClinicalScore"]]]] = empty_list()
    coordinate_space: Optional[Union[str, CommonCoordinateSpaceName]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.dataset_id is not None and not isinstance(self.dataset_id, str):
            self.dataset_id = str(self.dataset_id)

        self._normalize_inlined_as_dict(slot_name="subjects", slot_type=Subject, key_name="subject_id", keyed=True)

        if not isinstance(self.clinical_scores, list):
            self.clinical_scores = [self.clinical_scores] if self.clinical_scores is not None else []
        self.clinical_scores = [v if isinstance(v, ClinicalScore) else ClinicalScore(**as_dict(v)) for v in self.clinical_scores]

        if self.coordinate_space is not None and not isinstance(self.coordinate_space, CommonCoordinateSpaceName):
            self.coordinate_space = CommonCoordinateSpaceName(self.coordinate_space)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Subject(YAMLRoot):
    """
    Human or animal subject receiving DBS.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO_DBS["Subject"]
    class_class_curie: ClassVar[str] = "tvbo_dbs:Subject"
    class_name: ClassVar[str] = "Subject"
    class_model_uri: ClassVar[URIRef] = TVBO.Subject

    subject_id: Union[str, SubjectSubjectId] = None
    age: Optional[float] = None
    sex: Optional[str] = None
    diagnosis: Optional[str] = None
    handedness: Optional[str] = None
    protocols: Optional[Union[Union[str, DBSProtocolName], list[Union[str, DBSProtocolName]]]] = empty_list()
    coordinate_space: Optional[Union[str, CommonCoordinateSpaceName]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.subject_id):
            self.MissingRequiredField("subject_id")
        if not isinstance(self.subject_id, SubjectSubjectId):
            self.subject_id = SubjectSubjectId(self.subject_id)

        if self.age is not None and not isinstance(self.age, float):
            self.age = float(self.age)

        if self.sex is not None and not isinstance(self.sex, str):
            self.sex = str(self.sex)

        if self.diagnosis is not None and not isinstance(self.diagnosis, str):
            self.diagnosis = str(self.diagnosis)

        if self.handedness is not None and not isinstance(self.handedness, str):
            self.handedness = str(self.handedness)

        if not isinstance(self.protocols, list):
            self.protocols = [self.protocols] if self.protocols is not None else []
        self.protocols = [v if isinstance(v, DBSProtocolName) else DBSProtocolName(v) for v in self.protocols]

        if self.coordinate_space is not None and not isinstance(self.coordinate_space, CommonCoordinateSpaceName):
            self.coordinate_space = CommonCoordinateSpaceName(self.coordinate_space)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Electrode(YAMLRoot):
    """
    Implanted DBS electrode and contact geometry.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO_DBS["Electrode"]
    class_class_curie: ClassVar[str] = "tvbo_dbs:Electrode"
    class_name: ClassVar[str] = "Electrode"
    class_model_uri: ClassVar[URIRef] = TVBO.Electrode

    electrode_id: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    hemisphere: Optional[str] = "left"
    contacts: Optional[Union[Union[dict, "Contact"], list[Union[dict, "Contact"]]]] = empty_list()
    head: Optional[Union[dict, Coordinate]] = None
    tail: Optional[Union[dict, Coordinate]] = None
    trajectory: Optional[Union[Union[dict, Coordinate], list[Union[dict, Coordinate]]]] = empty_list()
    target_structure: Optional[Union[str, ParcellationEntityName]] = None
    coordinate_space: Optional[Union[str, CommonCoordinateSpaceName]] = None
    recon_path: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.electrode_id is not None and not isinstance(self.electrode_id, str):
            self.electrode_id = str(self.electrode_id)

        if self.manufacturer is not None and not isinstance(self.manufacturer, str):
            self.manufacturer = str(self.manufacturer)

        if self.model is not None and not isinstance(self.model, str):
            self.model = str(self.model)

        if self.hemisphere is not None and not isinstance(self.hemisphere, str):
            self.hemisphere = str(self.hemisphere)

        if not isinstance(self.contacts, list):
            self.contacts = [self.contacts] if self.contacts is not None else []
        self.contacts = [v if isinstance(v, Contact) else Contact(**as_dict(v)) for v in self.contacts]

        if self.head is not None and not isinstance(self.head, Coordinate):
            self.head = Coordinate(**as_dict(self.head))

        if self.tail is not None and not isinstance(self.tail, Coordinate):
            self.tail = Coordinate(**as_dict(self.tail))

        if not isinstance(self.trajectory, list):
            self.trajectory = [self.trajectory] if self.trajectory is not None else []
        self.trajectory = [v if isinstance(v, Coordinate) else Coordinate(**as_dict(v)) for v in self.trajectory]

        if self.target_structure is not None and not isinstance(self.target_structure, ParcellationEntityName):
            self.target_structure = ParcellationEntityName(self.target_structure)

        if self.coordinate_space is not None and not isinstance(self.coordinate_space, CommonCoordinateSpaceName):
            self.coordinate_space = CommonCoordinateSpaceName(self.coordinate_space)

        if self.recon_path is not None and not isinstance(self.recon_path, str):
            self.recon_path = str(self.recon_path)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Contact(YAMLRoot):
    """
    Individual contact on a DBS electrode.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO_DBS["Contact"]
    class_class_curie: ClassVar[str] = "tvbo_dbs:Contact"
    class_name: ClassVar[str] = "Contact"
    class_model_uri: ClassVar[URIRef] = TVBO.Contact

    contact_id: Optional[int] = None
    coordinate: Optional[Union[dict, Coordinate]] = None
    label: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.contact_id is not None and not isinstance(self.contact_id, int):
            self.contact_id = int(self.contact_id)

        if self.coordinate is not None and not isinstance(self.coordinate, Coordinate):
            self.coordinate = Coordinate(**as_dict(self.coordinate))

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class StimulationSetting(YAMLRoot):
    """
    DBS parameters for a specific session.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO_DBS["StimulationSetting"]
    class_class_curie: ClassVar[str] = "tvbo_dbs:StimulationSetting"
    class_name: ClassVar[str] = "StimulationSetting"
    class_model_uri: ClassVar[URIRef] = TVBO.StimulationSetting

    electrode_reference: Optional[Union[dict, Electrode]] = None
    amplitude: Optional[Union[str, ParameterName]] = None
    frequency: Optional[Union[str, ParameterName]] = None
    pulse_width: Optional[Union[str, ParameterName]] = None
    mode: Optional[str] = None
    active_contacts: Optional[Union[int, list[int]]] = empty_list()
    efield: Optional[Union[dict, "EField"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.electrode_reference is not None and not isinstance(self.electrode_reference, Electrode):
            self.electrode_reference = Electrode(**as_dict(self.electrode_reference))

        if self.amplitude is not None and not isinstance(self.amplitude, ParameterName):
            self.amplitude = ParameterName(self.amplitude)

        if self.frequency is not None and not isinstance(self.frequency, ParameterName):
            self.frequency = ParameterName(self.frequency)

        if self.pulse_width is not None and not isinstance(self.pulse_width, ParameterName):
            self.pulse_width = ParameterName(self.pulse_width)

        if self.mode is not None and not isinstance(self.mode, str):
            self.mode = str(self.mode)

        if not isinstance(self.active_contacts, list):
            self.active_contacts = [self.active_contacts] if self.active_contacts is not None else []
        self.active_contacts = [v if isinstance(v, int) else int(v) for v in self.active_contacts]

        if self.efield is not None and not isinstance(self.efield, EField):
            self.efield = EField(**as_dict(self.efield))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class DBSProtocol(YAMLRoot):
    """
    A protocol describing DBS therapy, potentially bilateral or multi-lead.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO_DBS["DBSProtocol"]
    class_class_curie: ClassVar[str] = "tvbo_dbs:DBSProtocol"
    class_name: ClassVar[str] = "DBSProtocol"
    class_model_uri: ClassVar[URIRef] = TVBO.DBSProtocol

    name: Union[str, DBSProtocolName] = None
    electrodes: Optional[Union[Union[dict, Electrode], list[Union[dict, Electrode]]]] = empty_list()
    settings: Optional[Union[Union[dict, StimulationSetting], list[Union[dict, StimulationSetting]]]] = empty_list()
    timing_info: Optional[str] = None
    notes: Optional[str] = None
    clinical_improvement: Optional[Union[Union[dict, "ClinicalImprovement"], list[Union[dict, "ClinicalImprovement"]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, DBSProtocolName):
            self.name = DBSProtocolName(self.name)

        if not isinstance(self.electrodes, list):
            self.electrodes = [self.electrodes] if self.electrodes is not None else []
        self.electrodes = [v if isinstance(v, Electrode) else Electrode(**as_dict(v)) for v in self.electrodes]

        if not isinstance(self.settings, list):
            self.settings = [self.settings] if self.settings is not None else []
        self.settings = [v if isinstance(v, StimulationSetting) else StimulationSetting(**as_dict(v)) for v in self.settings]

        if self.timing_info is not None and not isinstance(self.timing_info, str):
            self.timing_info = str(self.timing_info)

        if self.notes is not None and not isinstance(self.notes, str):
            self.notes = str(self.notes)

        if not isinstance(self.clinical_improvement, list):
            self.clinical_improvement = [self.clinical_improvement] if self.clinical_improvement is not None else []
        self.clinical_improvement = [v if isinstance(v, ClinicalImprovement) else ClinicalImprovement(**as_dict(v)) for v in self.clinical_improvement]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ClinicalScale(YAMLRoot):
    """
    A clinical assessment inventory or structured scale composed of multiple scores or items.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO_DBS["ClinicalScale"]
    class_class_curie: ClassVar[str] = "tvbo_dbs:ClinicalScale"
    class_name: ClassVar[str] = "ClinicalScale"
    class_model_uri: ClassVar[URIRef] = TVBO.ClinicalScale

    acronym: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    domain: Optional[str] = None
    reference: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.acronym is not None and not isinstance(self.acronym, str):
            self.acronym = str(self.acronym)

        if self.name is not None and not isinstance(self.name, str):
            self.name = str(self.name)

        if self.acronym is not None and not isinstance(self.acronym, str):
            self.acronym = str(self.acronym)

        if self.version is not None and not isinstance(self.version, str):
            self.version = str(self.version)

        if self.domain is not None and not isinstance(self.domain, str):
            self.domain = str(self.domain)

        if self.reference is not None and not isinstance(self.reference, str):
            self.reference = str(self.reference)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ClinicalScore(YAMLRoot):
    """
    Metadata about a clinical score or scale.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO_DBS["ClinicalScore"]
    class_class_curie: ClassVar[str] = "tvbo_dbs:ClinicalScore"
    class_name: ClassVar[str] = "ClinicalScore"
    class_model_uri: ClassVar[URIRef] = TVBO.ClinicalScore

    acronym: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    reference: Optional[str] = None
    scale: Optional[Union[dict, ClinicalScale]] = None
    parent_score: Optional[Union[dict, "ClinicalScore"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.acronym is not None and not isinstance(self.acronym, str):
            self.acronym = str(self.acronym)

        if self.name is not None and not isinstance(self.name, str):
            self.name = str(self.name)

        if self.acronym is not None and not isinstance(self.acronym, str):
            self.acronym = str(self.acronym)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.domain is not None and not isinstance(self.domain, str):
            self.domain = str(self.domain)

        if self.reference is not None and not isinstance(self.reference, str):
            self.reference = str(self.reference)

        if self.scale is not None and not isinstance(self.scale, ClinicalScale):
            self.scale = ClinicalScale(**as_dict(self.scale))

        if self.parent_score is not None and not isinstance(self.parent_score, ClinicalScore):
            self.parent_score = ClinicalScore(**as_dict(self.parent_score))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ClinicalImprovement(YAMLRoot):
    """
    Relative improvement on a defined clinical score.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO_DBS["ClinicalImprovement"]
    class_class_curie: ClassVar[str] = "tvbo_dbs:ClinicalImprovement"
    class_name: ClassVar[str] = "ClinicalImprovement"
    class_model_uri: ClassVar[URIRef] = TVBO.ClinicalImprovement

    score: Optional[Union[dict, ClinicalScore]] = None
    baseline_value: Optional[float] = None
    absolute_value: Optional[float] = None
    percent_change: Optional[float] = None
    time_post_surgery: Optional[float] = None
    evaluator: Optional[str] = None
    timepoint: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.score is not None and not isinstance(self.score, ClinicalScore):
            self.score = ClinicalScore(**as_dict(self.score))

        if self.baseline_value is not None and not isinstance(self.baseline_value, float):
            self.baseline_value = float(self.baseline_value)

        if self.absolute_value is not None and not isinstance(self.absolute_value, float):
            self.absolute_value = float(self.absolute_value)

        if self.percent_change is not None and not isinstance(self.percent_change, float):
            self.percent_change = float(self.percent_change)

        if self.time_post_surgery is not None and not isinstance(self.time_post_surgery, float):
            self.time_post_surgery = float(self.time_post_surgery)

        if self.evaluator is not None and not isinstance(self.evaluator, str):
            self.evaluator = str(self.evaluator)

        if self.timepoint is not None and not isinstance(self.timepoint, str):
            self.timepoint = str(self.timepoint)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class EField(YAMLRoot):
    """
    Simulated electric field from DBS modeling.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO_DBS["EField"]
    class_class_curie: ClassVar[str] = "tvbo_dbs:EField"
    class_name: ClassVar[str] = "EField"
    class_model_uri: ClassVar[URIRef] = TVBO.EField

    volume_data: Optional[str] = None
    coordinate_space: Optional[Union[str, CommonCoordinateSpaceName]] = None
    threshold_applied: Optional[float] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.volume_data is not None and not isinstance(self.volume_data, str):
            self.volume_data = str(self.volume_data)

        if self.coordinate_space is not None and not isinstance(self.coordinate_space, CommonCoordinateSpaceName):
            self.coordinate_space = CommonCoordinateSpaceName(self.coordinate_space)

        if self.threshold_applied is not None and not isinstance(self.threshold_applied, float):
            self.threshold_applied = float(self.threshold_applied)

        super().__post_init__(**kwargs)


# Enumerations
class ImagingModality(EnumDefinitionImpl):

    BOLD = PermissibleValue(
        text="BOLD",
        description="Blood Oxygen Level Dependent signal.")
    EEG = PermissibleValue(
        text="EEG",
        description="Electroencephalography.")
    MEG = PermissibleValue(
        text="MEG",
        description="Magnetoencephalography.")
    SEEG = PermissibleValue(
        text="SEEG",
        description="Stereoelectroencephalography.")
    IEEG = PermissibleValue(
        text="IEEG",
        description="Intracranial Electroencephalography.")

    _defn = EnumDefinition(
        name="ImagingModality",
    )

class OperationType(EnumDefinitionImpl):

    select = PermissibleValue(text="select")
    temporal_average = PermissibleValue(text="temporal_average")
    subsample = PermissibleValue(text="subsample")
    projection = PermissibleValue(text="projection")
    reference_subtract = PermissibleValue(text="reference_subtract")
    convolution = PermissibleValue(text="convolution")
    node_coupling = PermissibleValue(text="node_coupling")
    custom_transform = PermissibleValue(text="custom_transform")

    _defn = EnumDefinition(
        name="OperationType",
    )

class SystemType(EnumDefinitionImpl):

    continuous = PermissibleValue(
        text="continuous",
        description="Continuous-time dynamics (e.g., ODE/SDE).")
    discrete = PermissibleValue(
        text="discrete",
        description="Discrete-time dynamics (e.g., maps, iterated updates).")

    _defn = EnumDefinition(
        name="SystemType",
    )

class BoundaryConditionType(EnumDefinitionImpl):

    Dirichlet = PermissibleValue(text="Dirichlet")
    Neumann = PermissibleValue(text="Neumann")
    Robin = PermissibleValue(text="Robin")
    Periodic = PermissibleValue(text="Periodic")

    _defn = EnumDefinition(
        name="BoundaryConditionType",
    )

class DiscretizationMethod(EnumDefinitionImpl):

    FDM = PermissibleValue(
        text="FDM",
        description="Finite Difference Method")
    FEM = PermissibleValue(
        text="FEM",
        description="Finite Element Method")
    FVM = PermissibleValue(
        text="FVM",
        description="Finite Volume Method")
    Spectral = PermissibleValue(text="Spectral")

    _defn = EnumDefinition(
        name="DiscretizationMethod",
    )

class ElementType(EnumDefinitionImpl):

    triangle = PermissibleValue(text="triangle")
    quad = PermissibleValue(text="quad")
    tetrahedron = PermissibleValue(text="tetrahedron")
    hexahedron = PermissibleValue(text="hexahedron")

    _defn = EnumDefinition(
        name="ElementType",
    )

class OperatorType(EnumDefinitionImpl):

    gradient = PermissibleValue(text="gradient")
    divergence = PermissibleValue(text="divergence")
    laplacian = PermissibleValue(text="laplacian")
    curl = PermissibleValue(text="curl")

    _defn = EnumDefinition(
        name="OperatorType",
    )

class NoiseType(EnumDefinitionImpl):

    gaussian = PermissibleValue(text="gaussian")
    white = PermissibleValue(text="white")
    brown = PermissibleValue(text="brown")
    pink = PermissibleValue(text="pink")

    _defn = EnumDefinition(
        name="NoiseType",
    )

class RequirementRole(EnumDefinitionImpl):

    engine = PermissibleValue(
        text="engine",
        description="Primary simulation/processing engine")
    runtime = PermissibleValue(
        text="runtime",
        description="General runtime dependency")
    analysis = PermissibleValue(
        text="analysis",
        description="Post-processing / analysis tool")
    dev = PermissibleValue(
        text="dev",
        description="Development / build dependency")
    optional = PermissibleValue(
        text="optional",
        description="Optional or extra feature dependency")

    _defn = EnumDefinition(
        name="RequirementRole",
    )

class EnvironmentType(EnumDefinitionImpl):

    conda = PermissibleValue(text="conda")
    venv = PermissibleValue(text="venv")
    docker = PermissibleValue(text="docker")
    singularity = PermissibleValue(text="singularity")
    system = PermissibleValue(text="system")
    other = PermissibleValue(text="other")

    _defn = EnumDefinition(
        name="EnvironmentType",
    )

class SpecimenEnum(EnumDefinitionImpl):
    """
    A set of permissible types for specimens used in brain atlas creation.
    """
    Subject = PermissibleValue(text="Subject")
    SubjectGroup = PermissibleValue(text="SubjectGroup")
    TissueSample = PermissibleValue(text="TissueSample")
    TissueSampleCollection = PermissibleValue(text="TissueSampleCollection")

    _defn = EnumDefinition(
        name="SpecimenEnum",
        description="A set of permissible types for specimens used in brain atlas creation.",
    )

class Hemisphere(EnumDefinitionImpl):

    left = PermissibleValue(text="left")
    right = PermissibleValue(text="right")
    both = PermissibleValue(text="both")

    _defn = EnumDefinition(
        name="Hemisphere",
    )

# Slots
class slots:
    pass

slots.name = Slot(uri=TVBO.name, name="name", curie=TVBO.curie('name'),
                   model_uri=TVBO.name, domain=None, range=URIRef)

slots.time_scale = Slot(uri=TVBO.time_scale, name="time_scale", curie=TVBO.curie('time_scale'),
                   model_uri=TVBO.time_scale, domain=None, range=Optional[str])

slots.environment = Slot(uri=TVBO.environment, name="environment", curie=TVBO.curie('environment'),
                   model_uri=TVBO.environment, domain=None, range=Optional[Union[dict, SoftwareEnvironment]])

slots.requirements = Slot(uri=TVBO.requirements, name="requirements", curie=TVBO.curie('requirements'),
                   model_uri=TVBO.requirements, domain=None, range=Optional[Union[Union[str, SoftwareRequirementName], list[Union[str, SoftwareRequirementName]]]])

slots.duration = Slot(uri=TVBO.duration, name="duration", curie=TVBO.curie('duration'),
                   model_uri=TVBO.duration, domain=None, range=Optional[float])

slots.model = Slot(uri=TVBO.model, name="model", curie=TVBO.curie('model'),
                   model_uri=TVBO.model, domain=None, range=Optional[Union[str, DynamicsName]])

slots.has_reference = Slot(uri=TVBO.has_reference, name="has_reference", curie=TVBO.curie('has_reference'),
                   model_uri=TVBO.has_reference, domain=None, range=Optional[str])

slots.references = Slot(uri=TVBO.references, name="references", curie=TVBO.curie('references'),
                   model_uri=TVBO.references, domain=None, range=Optional[Union[str, list[str]]])

slots.label = Slot(uri=TVBO.label, name="label", curie=TVBO.curie('label'),
                   model_uri=TVBO.label, domain=None, range=Optional[str])

slots.acronym = Slot(uri=TVBO.acronym, name="acronym", curie=TVBO.curie('acronym'),
                   model_uri=TVBO.acronym, domain=None, range=Optional[str])

slots.symbol = Slot(uri=TVBO.symbol, name="symbol", curie=TVBO.curie('symbol'),
                   model_uri=TVBO.symbol, domain=None, range=Optional[str])

slots.domain = Slot(uri=TVBO.domain, name="domain", curie=TVBO.curie('domain'),
                   model_uri=TVBO.domain, domain=None, range=Optional[Union[dict, Range]])

slots.iri = Slot(uri=TVBO.iri, name="iri", curie=TVBO.curie('iri'),
                   model_uri=TVBO.iri, domain=None, range=Optional[str])

slots.value = Slot(uri=TVBO.value, name="value", curie=TVBO.curie('value'),
                   model_uri=TVBO.value, domain=None, range=Optional[float])

slots.file = Slot(uri=TVBO.file, name="file", curie=TVBO.curie('file'),
                   model_uri=TVBO.file, domain=None, range=Optional[str])

slots.reported_optimum = Slot(uri=TVBO.reported_optimum, name="reported_optimum", curie=TVBO.curie('reported_optimum'),
                   model_uri=TVBO.reported_optimum, domain=None, range=Optional[float])

slots.default = Slot(uri=TVBO.default, name="default", curie=TVBO.curie('default'),
                   model_uri=TVBO.default, domain=None, range=Optional[str])

slots.description = Slot(uri=TVBO.description, name="description", curie=TVBO.curie('description'),
                   model_uri=TVBO.description, domain=None, range=Optional[str])

slots.definition = Slot(uri=TVBO.definition, name="definition", curie=TVBO.curie('definition'),
                   model_uri=TVBO.definition, domain=None, range=Optional[str])

slots.parameters = Slot(uri=TVBO.parameters, name="parameters", curie=TVBO.curie('parameters'),
                   model_uri=TVBO.parameters, domain=None, range=Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]])

slots.equation = Slot(uri=TVBO.Equation, name="equation", curie=TVBO.curie('Equation'),
                   model_uri=TVBO.equation, domain=None, range=Optional[Union[dict, Equation]])

slots.unit = Slot(uri=TVBO.unit, name="unit", curie=TVBO.curie('unit'),
                   model_uri=TVBO.unit, domain=None, range=Optional[str])

slots.derived_from = Slot(uri=TVBO.derived_from, name="derived_from", curie=TVBO.curie('derived_from'),
                   model_uri=TVBO.derived_from, domain=None, range=Optional[str])

slots.source = Slot(uri=TVBO.source, name="source", curie=TVBO.curie('source'),
                   model_uri=TVBO.source, domain=None, range=Optional[str])

slots.abbreviation = Slot(uri=ATOM.abbreviation, name="abbreviation", curie=ATOM.curie('abbreviation'),
                   model_uri=TVBO.abbreviation, domain=None, range=Optional[str])

slots.alternateName = Slot(uri=ATOM['atlas/hasName'], name="alternateName", curie=ATOM.curie('atlas/hasName'),
                   model_uri=TVBO.alternateName, domain=None, range=Optional[Union[str, list[str]]])

slots.author = Slot(uri=ATOM.author, name="author", curie=ATOM.curie('author'),
                   model_uri=TVBO.author, domain=None, range=Optional[Union[str, list[str]]])

slots.digitalIdentifier = Slot(uri=ATOM.digitalIdentifier, name="digitalIdentifier", curie=ATOM.curie('digitalIdentifier'),
                   model_uri=TVBO.digitalIdentifier, domain=None, range=Optional[Union[str, list[str]]])

slots.hasParent = Slot(uri=ATOM['atlas/hasParent'], name="hasParent", curie=ATOM.curie('atlas/hasParent'),
                   model_uri=TVBO.hasParent, domain=None, range=Optional[Union[Union[str, ParcellationEntityName], list[Union[str, ParcellationEntityName]]]])

slots.isVersionOf = Slot(uri=ATOM.isVersionOf, name="isVersionOf", curie=ATOM.curie('isVersionOf'),
                   model_uri=TVBO.isVersionOf, domain=None, range=Optional[str])

slots.license = Slot(uri=ATOM.license, name="license", curie=ATOM.curie('license'),
                   model_uri=TVBO.license, domain=None, range=Optional[str])

slots.lookupLabel = Slot(uri=ATOM['atlas/lookupLabel'], name="lookupLabel", curie=ATOM.curie('atlas/lookupLabel'),
                   model_uri=TVBO.lookupLabel, domain=None, range=Optional[int])

slots.ontologyIdentifier = Slot(uri=ATOM['atlas/hasIlxId'], name="ontologyIdentifier", curie=ATOM.curie('atlas/hasIlxId'),
                   model_uri=TVBO.ontologyIdentifier, domain=None, range=Optional[Union[str, list[str]]])

slots.versionIdentifier = Slot(uri=ATOM.versionIdentifier, name="versionIdentifier", curie=ATOM.curie('versionIdentifier'),
                   model_uri=TVBO.versionIdentifier, domain=None, range=Optional[str])

slots.dataLocation = Slot(uri=ATOM.dataLocation, name="dataLocation", curie=ATOM.curie('dataLocation'),
                   model_uri=TVBO.dataLocation, domain=None, range=Optional[str])

slots.coordinateSpace = Slot(uri=ATOM.coordinateSpace, name="coordinateSpace", curie=ATOM.curie('coordinateSpace'),
                   model_uri=TVBO.coordinateSpace, domain=None, range=Optional[Union[str, CommonCoordinateSpaceName]])

slots.subject_id = Slot(uri=TVBO_DBS.subject_id, name="subject_id", curie=TVBO_DBS.curie('subject_id'),
                   model_uri=TVBO.subject_id, domain=None, range=URIRef)

slots.id = Slot(uri=TVBO_DBS.id, name="id", curie=TVBO_DBS.curie('id'),
                   model_uri=TVBO.id, domain=None, range=Optional[int])

slots.range__lo = Slot(uri=TVBO.lo, name="range__lo", curie=TVBO.curie('lo'),
                   model_uri=TVBO.range__lo, domain=None, range=Optional[float])

slots.range__hi = Slot(uri=TVBO.hi, name="range__hi", curie=TVBO.curie('hi'),
                   model_uri=TVBO.range__hi, domain=None, range=Optional[float])

slots.range__step = Slot(uri=TVBO.step, name="range__step", curie=TVBO.curie('step'),
                   model_uri=TVBO.range__step, domain=None, range=Optional[float])

slots.equation__lefthandside = Slot(uri=TVBO.lhs, name="equation__lefthandside", curie=TVBO.curie('lhs'),
                   model_uri=TVBO.equation__lefthandside, domain=None, range=Optional[str])

slots.equation__righthandside = Slot(uri=TVBO.rhs, name="equation__righthandside", curie=TVBO.curie('rhs'),
                   model_uri=TVBO.equation__righthandside, domain=None, range=Optional[str])

slots.equation__conditionals = Slot(uri=TVBO.conditionals, name="equation__conditionals", curie=TVBO.curie('conditionals'),
                   model_uri=TVBO.equation__conditionals, domain=None, range=Optional[Union[Union[dict, ConditionalBlock], list[Union[dict, ConditionalBlock]]]])

slots.equation__engine = Slot(uri=TVBO.engine, name="equation__engine", curie=TVBO.curie('engine'),
                   model_uri=TVBO.equation__engine, domain=None, range=Optional[Union[dict, SoftwareRequirement]])

slots.equation__pycode = Slot(uri=TVBO.pycode, name="equation__pycode", curie=TVBO.curie('pycode'),
                   model_uri=TVBO.equation__pycode, domain=None, range=Optional[str])

slots.equation__latex = Slot(uri=TVBO.latex, name="equation__latex", curie=TVBO.curie('latex'),
                   model_uri=TVBO.equation__latex, domain=None, range=Optional[Union[bool, Bool]])

slots.conditionalBlock__condition = Slot(uri=TVBO.condition, name="conditionalBlock__condition", curie=TVBO.curie('condition'),
                   model_uri=TVBO.conditionalBlock__condition, domain=None, range=Optional[str])

slots.conditionalBlock__expression = Slot(uri=TVBO.expression, name="conditionalBlock__expression", curie=TVBO.curie('expression'),
                   model_uri=TVBO.conditionalBlock__expression, domain=None, range=Optional[str])

slots.stimulus__regions = Slot(uri=TVBO.regions, name="stimulus__regions", curie=TVBO.curie('regions'),
                   model_uri=TVBO.stimulus__regions, domain=None, range=Optional[Union[int, list[int]]])

slots.stimulus__weighting = Slot(uri=TVBO.weighting, name="stimulus__weighting", curie=TVBO.curie('weighting'),
                   model_uri=TVBO.stimulus__weighting, domain=None, range=Optional[Union[float, list[float]]])

slots.temporalApplicableEquation__time_dependent = Slot(uri=TVBO.time_dependent, name="temporalApplicableEquation__time_dependent", curie=TVBO.curie('time_dependent'),
                   model_uri=TVBO.temporalApplicableEquation__time_dependent, domain=None, range=Optional[Union[bool, Bool]])

slots.parcellation__region_labels = Slot(uri=TVBO.region_labels, name="parcellation__region_labels", curie=TVBO.curie('region_labels'),
                   model_uri=TVBO.parcellation__region_labels, domain=None, range=Optional[Union[str, list[str]]])

slots.parcellation__center_coordinates = Slot(uri=TVBO.center_coordinates, name="parcellation__center_coordinates", curie=TVBO.curie('center_coordinates'),
                   model_uri=TVBO.parcellation__center_coordinates, domain=None, range=Optional[Union[float, list[float]]])

slots.parcellation__data_source = Slot(uri=TVBO.data_source, name="parcellation__data_source", curie=TVBO.curie('data_source'),
                   model_uri=TVBO.parcellation__data_source, domain=None, range=Optional[str])

slots.parcellation__atlas = Slot(uri=TVBO.atlas, name="parcellation__atlas", curie=TVBO.curie('atlas'),
                   model_uri=TVBO.parcellation__atlas, domain=None, range=Union[dict, BrainAtlas])

slots.matrix__x = Slot(uri=TVBO.x, name="matrix__x", curie=TVBO.curie('x'),
                   model_uri=TVBO.matrix__x, domain=None, range=Optional[Union[dict, BrainRegionSeries]])

slots.matrix__y = Slot(uri=TVBO.y, name="matrix__y", curie=TVBO.curie('y'),
                   model_uri=TVBO.matrix__y, domain=None, range=Optional[Union[dict, BrainRegionSeries]])

slots.matrix__values = Slot(uri=TVBO.values, name="matrix__values", curie=TVBO.curie('values'),
                   model_uri=TVBO.matrix__values, domain=None, range=Optional[Union[float, list[float]]])

slots.brainRegionSeries__values = Slot(uri=TVBO.values, name="brainRegionSeries__values", curie=TVBO.curie('values'),
                   model_uri=TVBO.brainRegionSeries__values, domain=None, range=Optional[Union[str, list[str]]])

slots.network__nodes = Slot(uri=TVBO.nodes, name="network__nodes", curie=TVBO.curie('nodes'),
                   model_uri=TVBO.network__nodes, domain=None, range=Optional[Union[Union[dict, Node], list[Union[dict, Node]]]])

slots.network__edges = Slot(uri=TVBO.edges, name="network__edges", curie=TVBO.curie('edges'),
                   model_uri=TVBO.network__edges, domain=None, range=Optional[Union[Union[dict, Edge], list[Union[dict, Edge]]]])

slots.network__coupling_library = Slot(uri=TVBO.coupling_library, name="network__coupling_library", curie=TVBO.curie('coupling_library'),
                   model_uri=TVBO.network__coupling_library, domain=None, range=Optional[Union[dict[Union[str, CouplingName], Union[dict, Coupling]], list[Union[dict, Coupling]]]])

slots.network__number_of_regions = Slot(uri=TVBO.number_of_regions, name="network__number_of_regions", curie=TVBO.curie('number_of_regions'),
                   model_uri=TVBO.network__number_of_regions, domain=None, range=Optional[int])

slots.network__number_of_nodes = Slot(uri=TVBO.number_of_nodes, name="network__number_of_nodes", curie=TVBO.curie('number_of_nodes'),
                   model_uri=TVBO.network__number_of_nodes, domain=None, range=Optional[int])

slots.network__parcellation = Slot(uri=TVBO.parcellation, name="network__parcellation", curie=TVBO.curie('parcellation'),
                   model_uri=TVBO.network__parcellation, domain=None, range=Optional[Union[dict, Parcellation]])

slots.network__tractogram = Slot(uri=TVBO.tractogram, name="network__tractogram", curie=TVBO.curie('tractogram'),
                   model_uri=TVBO.network__tractogram, domain=None, range=Optional[str])

slots.network__weights = Slot(uri=TVBO.weights, name="network__weights", curie=TVBO.curie('weights'),
                   model_uri=TVBO.network__weights, domain=None, range=Optional[Union[dict, Matrix]])

slots.network__lengths = Slot(uri=TVBO.lengths, name="network__lengths", curie=TVBO.curie('lengths'),
                   model_uri=TVBO.network__lengths, domain=None, range=Optional[Union[dict, Matrix]])

slots.network__normalization = Slot(uri=TVBO.normalization, name="network__normalization", curie=TVBO.curie('normalization'),
                   model_uri=TVBO.network__normalization, domain=None, range=Optional[Union[dict, Equation]])

slots.network__node_labels = Slot(uri=TVBO.node_labels, name="network__node_labels", curie=TVBO.curie('node_labels'),
                   model_uri=TVBO.network__node_labels, domain=None, range=Optional[Union[str, list[str]]])

slots.network__global_coupling_strength = Slot(uri=TVBO.global_coupling_strength, name="network__global_coupling_strength", curie=TVBO.curie('global_coupling_strength'),
                   model_uri=TVBO.network__global_coupling_strength, domain=None, range=Optional[Union[dict, Parameter]])

slots.network__conduction_speed = Slot(uri=TVBO.conduction_speed, name="network__conduction_speed", curie=TVBO.curie('conduction_speed'),
                   model_uri=TVBO.network__conduction_speed, domain=None, range=Optional[Union[dict, Parameter]])

slots.node__id = Slot(uri=TVBO.id, name="node__id", curie=TVBO.curie('id'),
                   model_uri=TVBO.node__id, domain=None, range=int)

slots.node__dynamics = Slot(uri=TVBO.dynamics, name="node__dynamics", curie=TVBO.curie('dynamics'),
                   model_uri=TVBO.node__dynamics, domain=None, range=Optional[Union[str, DynamicsName]])

slots.node__position = Slot(uri=TVBO.position, name="node__position", curie=TVBO.curie('position'),
                   model_uri=TVBO.node__position, domain=None, range=Optional[Union[dict, Coordinate]])

slots.node__region = Slot(uri=TVBO.region, name="node__region", curie=TVBO.curie('region'),
                   model_uri=TVBO.node__region, domain=None, range=Optional[str])

slots.node__parameters = Slot(uri=TVBO.parameters, name="node__parameters", curie=TVBO.curie('parameters'),
                   model_uri=TVBO.node__parameters, domain=None, range=Optional[Union[Union[str, ParameterName], list[Union[str, ParameterName]]]])

slots.node__initial_state = Slot(uri=TVBO.initial_state, name="node__initial_state", curie=TVBO.curie('initial_state'),
                   model_uri=TVBO.node__initial_state, domain=None, range=Optional[Union[float, list[float]]])

slots.edge__source = Slot(uri=TVBO.source, name="edge__source", curie=TVBO.curie('source'),
                   model_uri=TVBO.edge__source, domain=None, range=int)

slots.edge__target = Slot(uri=TVBO.target, name="edge__target", curie=TVBO.curie('target'),
                   model_uri=TVBO.edge__target, domain=None, range=int)

slots.edge__coupling = Slot(uri=TVBO.coupling, name="edge__coupling", curie=TVBO.curie('coupling'),
                   model_uri=TVBO.edge__coupling, domain=None, range=Union[dict, Coupling])

slots.edge__weight = Slot(uri=TVBO.weight, name="edge__weight", curie=TVBO.curie('weight'),
                   model_uri=TVBO.edge__weight, domain=None, range=Optional[float])

slots.edge__delay = Slot(uri=TVBO.delay, name="edge__delay", curie=TVBO.curie('delay'),
                   model_uri=TVBO.edge__delay, domain=None, range=Optional[float])

slots.edge__distance = Slot(uri=TVBO.distance, name="edge__distance", curie=TVBO.curie('distance'),
                   model_uri=TVBO.edge__distance, domain=None, range=Optional[float])

slots.edge__tract_properties = Slot(uri=TVBO.tract_properties, name="edge__tract_properties", curie=TVBO.curie('tract_properties'),
                   model_uri=TVBO.edge__tract_properties, domain=None, range=Optional[str])

slots.observationModel__transformation = Slot(uri=TVBO.transformation, name="observationModel__transformation", curie=TVBO.curie('transformation'),
                   model_uri=TVBO.observationModel__transformation, domain=None, range=Optional[Union[dict, Function]])

slots.observationModel__pipeline = Slot(uri=TVBO.pipeline, name="observationModel__pipeline", curie=TVBO.curie('pipeline'),
                   model_uri=TVBO.observationModel__pipeline, domain=None, range=Optional[Union[Union[dict, ProcessingStep], list[Union[dict, ProcessingStep]]]])

slots.observationModel__data_injections = Slot(uri=TVBO.data_injections, name="observationModel__data_injections", curie=TVBO.curie('data_injections'),
                   model_uri=TVBO.observationModel__data_injections, domain=None, range=Optional[Union[Union[dict, DataInjection], list[Union[dict, DataInjection]]]])

slots.observationModel__argument_mappings = Slot(uri=TVBO.argument_mappings, name="observationModel__argument_mappings", curie=TVBO.curie('argument_mappings'),
                   model_uri=TVBO.observationModel__argument_mappings, domain=None, range=Optional[Union[Union[dict, ArgumentMapping], list[Union[dict, ArgumentMapping]]]])

slots.observationModel__derivatives = Slot(uri=TVBO.derivatives, name="observationModel__derivatives", curie=TVBO.curie('derivatives'),
                   model_uri=TVBO.observationModel__derivatives, domain=None, range=Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, DerivedVariable]], list[Union[dict, DerivedVariable]]]])

slots.processingStep__order = Slot(uri=TVBO.order, name="processingStep__order", curie=TVBO.curie('order'),
                   model_uri=TVBO.processingStep__order, domain=None, range=Optional[int])

slots.processingStep__function = Slot(uri=TVBO.transformation, name="processingStep__function", curie=TVBO.curie('transformation'),
                   model_uri=TVBO.processingStep__function, domain=None, range=Union[dict, Function])

slots.processingStep__operation_type = Slot(uri=TVBO.type, name="processingStep__operation_type", curie=TVBO.curie('type'),
                   model_uri=TVBO.processingStep__operation_type, domain=None, range=Optional[Union[str, "OperationType"]])

slots.processingStep__input_mapping = Slot(uri=TVBO.input_mapping, name="processingStep__input_mapping", curie=TVBO.curie('input_mapping'),
                   model_uri=TVBO.processingStep__input_mapping, domain=None, range=Optional[Union[Union[dict, ArgumentMapping], list[Union[dict, ArgumentMapping]]]])

slots.processingStep__output_alias = Slot(uri=TVBO.output_alias, name="processingStep__output_alias", curie=TVBO.curie('output_alias'),
                   model_uri=TVBO.processingStep__output_alias, domain=None, range=Optional[str])

slots.processingStep__apply_on_dimension = Slot(uri=TVBO.apply_on_dimension, name="processingStep__apply_on_dimension", curie=TVBO.curie('apply_on_dimension'),
                   model_uri=TVBO.processingStep__apply_on_dimension, domain=None, range=Optional[str])

slots.processingStep__ensure_shape = Slot(uri=TVBO.ensure_shape, name="processingStep__ensure_shape", curie=TVBO.curie('ensure_shape'),
                   model_uri=TVBO.processingStep__ensure_shape, domain=None, range=Optional[str])

slots.processingStep__variables_of_interest = Slot(uri=TVBO.variables_of_interest, name="processingStep__variables_of_interest", curie=TVBO.curie('variables_of_interest'),
                   model_uri=TVBO.processingStep__variables_of_interest, domain=None, range=Optional[Union[dict[Union[str, StateVariableName], Union[dict, StateVariable]], list[Union[dict, StateVariable]]]])

slots.dataInjection__name = Slot(uri=TVBO.name, name="dataInjection__name", curie=TVBO.curie('name'),
                   model_uri=TVBO.dataInjection__name, domain=None, range=str)

slots.dataInjection__data_source = Slot(uri=TVBO.data_source, name="dataInjection__data_source", curie=TVBO.curie('data_source'),
                   model_uri=TVBO.dataInjection__data_source, domain=None, range=Optional[str])

slots.dataInjection__values = Slot(uri=TVBO.values, name="dataInjection__values", curie=TVBO.curie('values'),
                   model_uri=TVBO.dataInjection__values, domain=None, range=Optional[Union[float, list[float]]])

slots.dataInjection__shape = Slot(uri=TVBO.shape, name="dataInjection__shape", curie=TVBO.curie('shape'),
                   model_uri=TVBO.dataInjection__shape, domain=None, range=Optional[Union[int, list[int]]])

slots.dataInjection__generation_function = Slot(uri=TVBO.generation_function, name="dataInjection__generation_function", curie=TVBO.curie('generation_function'),
                   model_uri=TVBO.dataInjection__generation_function, domain=None, range=Optional[Union[dict, Function]])

slots.argumentMapping__function_argument = Slot(uri=TVBO.function_argument, name="argumentMapping__function_argument", curie=TVBO.curie('function_argument'),
                   model_uri=TVBO.argumentMapping__function_argument, domain=None, range=str)

slots.argumentMapping__source = Slot(uri=TVBO.source, name="argumentMapping__source", curie=TVBO.curie('source'),
                   model_uri=TVBO.argumentMapping__source, domain=None, range=str)

slots.argumentMapping__constant_value = Slot(uri=TVBO.constant_value, name="argumentMapping__constant_value", curie=TVBO.curie('constant_value'),
                   model_uri=TVBO.argumentMapping__constant_value, domain=None, range=Optional[str])

slots.downsamplingModel__period = Slot(uri=TVBO.period, name="downsamplingModel__period", curie=TVBO.curie('period'),
                   model_uri=TVBO.downsamplingModel__period, domain=None, range=Optional[float])

slots.dynamics__derived_parameters = Slot(uri=TVBO.derived_parameters, name="dynamics__derived_parameters", curie=TVBO.curie('derived_parameters'),
                   model_uri=TVBO.dynamics__derived_parameters, domain=None, range=Optional[Union[dict[Union[str, DerivedParameterName], Union[dict, DerivedParameter]], list[Union[dict, DerivedParameter]]]])

slots.dynamics__derived_variables = Slot(uri=TVBO.derived_variables, name="dynamics__derived_variables", curie=TVBO.curie('derived_variables'),
                   model_uri=TVBO.dynamics__derived_variables, domain=None, range=Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, DerivedVariable]], list[Union[dict, DerivedVariable]]]])

slots.dynamics__coupling_terms = Slot(uri=TVBO.coupling_terms, name="dynamics__coupling_terms", curie=TVBO.curie('coupling_terms'),
                   model_uri=TVBO.dynamics__coupling_terms, domain=None, range=Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]])

slots.dynamics__coupling_inputs = Slot(uri=TVBO.coupling_inputs, name="dynamics__coupling_inputs", curie=TVBO.curie('coupling_inputs'),
                   model_uri=TVBO.dynamics__coupling_inputs, domain=None, range=Optional[Union[dict[Union[str, CouplingInputName], Union[dict, CouplingInput]], list[Union[dict, CouplingInput]]]])

slots.dynamics__state_variables = Slot(uri=TVBO.state_variables, name="dynamics__state_variables", curie=TVBO.curie('state_variables'),
                   model_uri=TVBO.dynamics__state_variables, domain=None, range=Optional[Union[dict[Union[str, StateVariableName], Union[dict, StateVariable]], list[Union[dict, StateVariable]]]])

slots.dynamics__modified = Slot(uri=TVBO.modified, name="dynamics__modified", curie=TVBO.curie('modified'),
                   model_uri=TVBO.dynamics__modified, domain=None, range=Optional[Union[bool, Bool]])

slots.dynamics__output_transforms = Slot(uri=TVBO.output_transforms, name="dynamics__output_transforms", curie=TVBO.curie('output_transforms'),
                   model_uri=TVBO.dynamics__output_transforms, domain=None, range=Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, DerivedVariable]], list[Union[dict, DerivedVariable]]]])

slots.dynamics__derived_from_model = Slot(uri=TVBO.derived_from_model, name="dynamics__derived_from_model", curie=TVBO.curie('derived_from_model'),
                   model_uri=TVBO.dynamics__derived_from_model, domain=None, range=Optional[Union[str, NeuralMassModelName]])

slots.dynamics__number_of_modes = Slot(uri=TVBO.number_of_modes, name="dynamics__number_of_modes", curie=TVBO.curie('number_of_modes'),
                   model_uri=TVBO.dynamics__number_of_modes, domain=None, range=Optional[int])

slots.dynamics__local_coupling_term = Slot(uri=TVBO.local_coupling_term, name="dynamics__local_coupling_term", curie=TVBO.curie('local_coupling_term'),
                   model_uri=TVBO.dynamics__local_coupling_term, domain=None, range=Optional[Union[str, ParameterName]])

slots.dynamics__functions = Slot(uri=TVBO.functions, name="dynamics__functions", curie=TVBO.curie('functions'),
                   model_uri=TVBO.dynamics__functions, domain=None, range=Optional[Union[dict[Union[str, FunctionName], Union[dict, Function]], list[Union[dict, Function]]]])

slots.dynamics__stimulus = Slot(uri=TVBO.stimulus, name="dynamics__stimulus", curie=TVBO.curie('stimulus'),
                   model_uri=TVBO.dynamics__stimulus, domain=None, range=Optional[Union[dict, Stimulus]])

slots.dynamics__modes = Slot(uri=TVBO.modes, name="dynamics__modes", curie=TVBO.curie('modes'),
                   model_uri=TVBO.dynamics__modes, domain=None, range=Optional[Union[dict[Union[str, NeuralMassModelName], Union[dict, NeuralMassModel]], list[Union[dict, NeuralMassModel]]]])

slots.dynamics__system_type = Slot(uri=TVBO.system_type, name="dynamics__system_type", curie=TVBO.curie('system_type'),
                   model_uri=TVBO.dynamics__system_type, domain=None, range=Optional[Union[str, "SystemType"]])

slots.stateVariable__variable_of_interest = Slot(uri=TVBO.variable_of_interest, name="stateVariable__variable_of_interest", curie=TVBO.curie('variable_of_interest'),
                   model_uri=TVBO.stateVariable__variable_of_interest, domain=None, range=Optional[Union[bool, Bool]])

slots.stateVariable__coupling_variable = Slot(uri=TVBO.coupling_variable, name="stateVariable__coupling_variable", curie=TVBO.curie('coupling_variable'),
                   model_uri=TVBO.stateVariable__coupling_variable, domain=None, range=Optional[Union[bool, Bool]])

slots.stateVariable__noise = Slot(uri=TVBO.noise, name="stateVariable__noise", curie=TVBO.curie('noise'),
                   model_uri=TVBO.stateVariable__noise, domain=None, range=Optional[Union[dict, Noise]])

slots.stateVariable__stimulation_variable = Slot(uri=TVBO.stimulation_variable, name="stateVariable__stimulation_variable", curie=TVBO.curie('stimulation_variable'),
                   model_uri=TVBO.stateVariable__stimulation_variable, domain=None, range=Optional[Union[bool, Bool]])

slots.stateVariable__boundaries = Slot(uri=TVBO.boundaries, name="stateVariable__boundaries", curie=TVBO.curie('boundaries'),
                   model_uri=TVBO.stateVariable__boundaries, domain=None, range=Optional[Union[dict, Range]])

slots.stateVariable__initial_value = Slot(uri=TVBO.initial_value, name="stateVariable__initial_value", curie=TVBO.curie('initial_value'),
                   model_uri=TVBO.stateVariable__initial_value, domain=None, range=Optional[float])

slots.stateVariable__history = Slot(uri=TVBO.history, name="stateVariable__history", curie=TVBO.curie('history'),
                   model_uri=TVBO.stateVariable__history, domain=None, range=Optional[Union[dict, TimeSeries]])

slots.distribution__dependencies = Slot(uri=TVBO.dependencies, name="distribution__dependencies", curie=TVBO.curie('dependencies'),
                   model_uri=TVBO.distribution__dependencies, domain=None, range=Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]])

slots.distribution__correlation = Slot(uri=TVBO.correlation, name="distribution__correlation", curie=TVBO.curie('correlation'),
                   model_uri=TVBO.distribution__correlation, domain=None, range=Optional[Union[dict, Matrix]])

slots.parameter__comment = Slot(uri=TVBO.comment, name="parameter__comment", curie=TVBO.curie('comment'),
                   model_uri=TVBO.parameter__comment, domain=None, range=Optional[str])

slots.parameter__heterogeneous = Slot(uri=TVBO.heterogeneous, name="parameter__heterogeneous", curie=TVBO.curie('heterogeneous'),
                   model_uri=TVBO.parameter__heterogeneous, domain=None, range=Optional[Union[bool, Bool]])

slots.parameter__free = Slot(uri=TVBO.free, name="parameter__free", curie=TVBO.curie('free'),
                   model_uri=TVBO.parameter__free, domain=None, range=Optional[Union[bool, Bool]])

slots.parameter__shape = Slot(uri=TVBO.shape, name="parameter__shape", curie=TVBO.curie('shape'),
                   model_uri=TVBO.parameter__shape, domain=None, range=Optional[str])

slots.parameter__explored_values = Slot(uri=TVBO.explored_values, name="parameter__explored_values", curie=TVBO.curie('explored_values'),
                   model_uri=TVBO.parameter__explored_values, domain=None, range=Optional[Union[float, list[float]]])

slots.couplingInput__dimension = Slot(uri=TVBO.dimension, name="couplingInput__dimension", curie=TVBO.curie('dimension'),
                   model_uri=TVBO.couplingInput__dimension, domain=None, range=Optional[int])

slots.function__iri = Slot(uri=TVBO.iri, name="function__iri", curie=TVBO.curie('iri'),
                   model_uri=TVBO.function__iri, domain=None, range=Optional[str])

slots.function__definition = Slot(uri=TVBO.definition, name="function__definition", curie=TVBO.curie('definition'),
                   model_uri=TVBO.function__definition, domain=None, range=Optional[str])

slots.function__arguments = Slot(uri=TVBO.arguments, name="function__arguments", curie=TVBO.curie('arguments'),
                   model_uri=TVBO.function__arguments, domain=None, range=Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]])

slots.function__output = Slot(uri=TVBO.output, name="function__output", curie=TVBO.curie('output'),
                   model_uri=TVBO.function__output, domain=None, range=Optional[Union[dict, Equation]])

slots.function__source_code = Slot(uri=TVBO.source_code, name="function__source_code", curie=TVBO.curie('source_code'),
                   model_uri=TVBO.function__source_code, domain=None, range=Optional[str])

slots.function__callable = Slot(uri=TVBO.callable, name="function__callable", curie=TVBO.curie('callable'),
                   model_uri=TVBO.function__callable, domain=None, range=Optional[Union[dict, Callable]])

slots.callable__module = Slot(uri=TVBO.module, name="callable__module", curie=TVBO.curie('module'),
                   model_uri=TVBO.callable__module, domain=None, range=Optional[str])

slots.callable__qualname = Slot(uri=TVBO.qualname, name="callable__qualname", curie=TVBO.curie('qualname'),
                   model_uri=TVBO.callable__qualname, domain=None, range=Optional[str])

slots.callable__software = Slot(uri=TVBO.software, name="callable__software", curie=TVBO.curie('software'),
                   model_uri=TVBO.callable__software, domain=None, range=Optional[Union[dict, SoftwareRequirement]])

slots.case__condition = Slot(uri=TVBO.condition, name="case__condition", curie=TVBO.curie('condition'),
                   model_uri=TVBO.case__condition, domain=None, range=Optional[str])

slots.case__equation = Slot(uri=TVBO.equation, name="case__equation", curie=TVBO.curie('equation'),
                   model_uri=TVBO.case__equation, domain=None, range=Optional[Union[dict, Equation]])

slots.derivedVariable__conditional = Slot(uri=TVBO.conditional, name="derivedVariable__conditional", curie=TVBO.curie('conditional'),
                   model_uri=TVBO.derivedVariable__conditional, domain=None, range=Optional[Union[bool, Bool]])

slots.derivedVariable__cases = Slot(uri=TVBO.cases, name="derivedVariable__cases", curie=TVBO.curie('cases'),
                   model_uri=TVBO.derivedVariable__cases, domain=None, range=Optional[Union[Union[dict, Case], list[Union[dict, Case]]]])

slots.noise__noise_type = Slot(uri=TVBO.noise_type, name="noise__noise_type", curie=TVBO.curie('noise_type'),
                   model_uri=TVBO.noise__noise_type, domain=None, range=Optional[str])

slots.noise__correlated = Slot(uri=TVBO.correlated, name="noise__correlated", curie=TVBO.curie('correlated'),
                   model_uri=TVBO.noise__correlated, domain=None, range=Optional[Union[bool, Bool]])

slots.noise__gaussian = Slot(uri=TVBO.gaussian, name="noise__gaussian", curie=TVBO.curie('gaussian'),
                   model_uri=TVBO.noise__gaussian, domain=None, range=Optional[Union[bool, Bool]])

slots.noise__additive = Slot(uri=TVBO.additive, name="noise__additive", curie=TVBO.curie('additive'),
                   model_uri=TVBO.noise__additive, domain=None, range=Optional[Union[bool, Bool]])

slots.noise__seed = Slot(uri=TVBO.seed, name="noise__seed", curie=TVBO.curie('seed'),
                   model_uri=TVBO.noise__seed, domain=None, range=Optional[int])

slots.noise__random_state = Slot(uri=TVBO.random_state, name="noise__random_state", curie=TVBO.curie('random_state'),
                   model_uri=TVBO.noise__random_state, domain=None, range=Optional[Union[dict, RandomStream]])

slots.noise__intensity = Slot(uri=TVBO.intensity, name="noise__intensity", curie=TVBO.curie('intensity'),
                   model_uri=TVBO.noise__intensity, domain=None, range=Optional[Union[dict, Parameter]])

slots.noise__function = Slot(uri=TVBO.function, name="noise__function", curie=TVBO.curie('function'),
                   model_uri=TVBO.noise__function, domain=None, range=Optional[Union[dict, Function]])

slots.noise__pycode = Slot(uri=TVBO.pycode, name="noise__pycode", curie=TVBO.curie('pycode'),
                   model_uri=TVBO.noise__pycode, domain=None, range=Optional[str])

slots.noise__targets = Slot(uri=TVBO.targets, name="noise__targets", curie=TVBO.curie('targets'),
                   model_uri=TVBO.noise__targets, domain=None, range=Optional[Union[dict[Union[str, StateVariableName], Union[dict, StateVariable]], list[Union[dict, StateVariable]]]])

slots.modelFitting__targets = Slot(uri=TVBO.targets, name="modelFitting__targets", curie=TVBO.curie('targets'),
                   model_uri=TVBO.modelFitting__targets, domain=None, range=Optional[Union[Union[dict, FittingTarget], list[Union[dict, FittingTarget]]]])

slots.modelFitting__cost_function = Slot(uri=TVBO.cost_function, name="modelFitting__cost_function", curie=TVBO.curie('cost_function'),
                   model_uri=TVBO.modelFitting__cost_function, domain=None, range=Optional[Union[dict, CostFunction]])

slots.integrator__method = Slot(uri=TVBO.method, name="integrator__method", curie=TVBO.curie('method'),
                   model_uri=TVBO.integrator__method, domain=None, range=Optional[str])

slots.integrator__step_size = Slot(uri=TVBO.step_size, name="integrator__step_size", curie=TVBO.curie('step_size'),
                   model_uri=TVBO.integrator__step_size, domain=None, range=Optional[float])

slots.integrator__steps = Slot(uri=TVBO.steps, name="integrator__steps", curie=TVBO.curie('steps'),
                   model_uri=TVBO.integrator__steps, domain=None, range=Optional[int])

slots.integrator__noise = Slot(uri=TVBO.noise, name="integrator__noise", curie=TVBO.curie('noise'),
                   model_uri=TVBO.integrator__noise, domain=None, range=Optional[Union[dict, Noise]])

slots.integrator__state_wise_sigma = Slot(uri=TVBO.state_wise_sigma, name="integrator__state_wise_sigma", curie=TVBO.curie('state_wise_sigma'),
                   model_uri=TVBO.integrator__state_wise_sigma, domain=None, range=Optional[Union[float, list[float]]])

slots.integrator__transient_time = Slot(uri=TVBO.transient_time, name="integrator__transient_time", curie=TVBO.curie('transient_time'),
                   model_uri=TVBO.integrator__transient_time, domain=None, range=Optional[float])

slots.integrator__scipy_ode_base = Slot(uri=TVBO.scipy_ode_base, name="integrator__scipy_ode_base", curie=TVBO.curie('scipy_ode_base'),
                   model_uri=TVBO.integrator__scipy_ode_base, domain=None, range=Optional[Union[bool, Bool]])

slots.integrator__number_of_stages = Slot(uri=TVBO.number_of_stages, name="integrator__number_of_stages", curie=TVBO.curie('number_of_stages'),
                   model_uri=TVBO.integrator__number_of_stages, domain=None, range=Optional[int])

slots.integrator__intermediate_expressions = Slot(uri=TVBO.intermediate_expressions, name="integrator__intermediate_expressions", curie=TVBO.curie('intermediate_expressions'),
                   model_uri=TVBO.integrator__intermediate_expressions, domain=None, range=Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, DerivedVariable]], list[Union[dict, DerivedVariable]]]])

slots.integrator__update_expression = Slot(uri=TVBO.update_expression, name="integrator__update_expression", curie=TVBO.curie('update_expression'),
                   model_uri=TVBO.integrator__update_expression, domain=None, range=Optional[Union[dict, DerivedVariable]])

slots.integrator__delayed = Slot(uri=TVBO.delayed, name="integrator__delayed", curie=TVBO.curie('delayed'),
                   model_uri=TVBO.integrator__delayed, domain=None, range=Optional[Union[bool, Bool]])

slots.monitor__period = Slot(uri=TVBO.period, name="monitor__period", curie=TVBO.curie('period'),
                   model_uri=TVBO.monitor__period, domain=None, range=Optional[float])

slots.monitor__imaging_modality = Slot(uri=TVBO.imaging_modality, name="monitor__imaging_modality", curie=TVBO.curie('imaging_modality'),
                   model_uri=TVBO.monitor__imaging_modality, domain=None, range=Optional[Union[str, "ImagingModality"]])

slots.coupling__coupling_function = Slot(uri=TVBO.coupling_function, name="coupling__coupling_function", curie=TVBO.curie('coupling_function'),
                   model_uri=TVBO.coupling__coupling_function, domain=None, range=Optional[Union[dict, Equation]])

slots.coupling__sparse = Slot(uri=TVBO.sparse, name="coupling__sparse", curie=TVBO.curie('sparse'),
                   model_uri=TVBO.coupling__sparse, domain=None, range=Optional[Union[bool, Bool]])

slots.coupling__pre_expression = Slot(uri=TVBO.pre_expression, name="coupling__pre_expression", curie=TVBO.curie('pre_expression'),
                   model_uri=TVBO.coupling__pre_expression, domain=None, range=Optional[Union[dict, Equation]])

slots.coupling__post_expression = Slot(uri=TVBO.post_expression, name="coupling__post_expression", curie=TVBO.curie('post_expression'),
                   model_uri=TVBO.coupling__post_expression, domain=None, range=Optional[Union[dict, Equation]])

slots.coupling__incoming_states = Slot(uri=TVBO.incoming_states, name="coupling__incoming_states", curie=TVBO.curie('incoming_states'),
                   model_uri=TVBO.coupling__incoming_states, domain=None, range=Optional[Union[dict, StateVariable]])

slots.coupling__local_states = Slot(uri=TVBO.local_states, name="coupling__local_states", curie=TVBO.curie('local_states'),
                   model_uri=TVBO.coupling__local_states, domain=None, range=Optional[Union[dict, StateVariable]])

slots.coupling__delayed = Slot(uri=TVBO.delayed, name="coupling__delayed", curie=TVBO.curie('delayed'),
                   model_uri=TVBO.coupling__delayed, domain=None, range=Optional[Union[bool, Bool]])

slots.coupling__inner_coupling = Slot(uri=TVBO.inner_coupling, name="coupling__inner_coupling", curie=TVBO.curie('inner_coupling'),
                   model_uri=TVBO.coupling__inner_coupling, domain=None, range=Optional[Union[dict, Coupling]])

slots.coupling__region_mapping = Slot(uri=TVBO.region_mapping, name="coupling__region_mapping", curie=TVBO.curie('region_mapping'),
                   model_uri=TVBO.coupling__region_mapping, domain=None, range=Optional[Union[dict, RegionMapping]])

slots.coupling__regional_connectivity = Slot(uri=TVBO.regional_connectivity, name="coupling__regional_connectivity", curie=TVBO.curie('regional_connectivity'),
                   model_uri=TVBO.coupling__regional_connectivity, domain=None, range=Optional[Union[dict, Network]])

slots.coupling__aggregation = Slot(uri=TVBO.aggregation, name="coupling__aggregation", curie=TVBO.curie('aggregation'),
                   model_uri=TVBO.coupling__aggregation, domain=None, range=Optional[str])

slots.coupling__distribution = Slot(uri=TVBO.distribution, name="coupling__distribution", curie=TVBO.curie('distribution'),
                   model_uri=TVBO.coupling__distribution, domain=None, range=Optional[str])

slots.regionMapping__vertex_to_region = Slot(uri=TVBO.vertex_to_region, name="regionMapping__vertex_to_region", curie=TVBO.curie('vertex_to_region'),
                   model_uri=TVBO.regionMapping__vertex_to_region, domain=None, range=Optional[Union[int, list[int]]])

slots.regionMapping__n_vertices = Slot(uri=TVBO.n_vertices, name="regionMapping__n_vertices", curie=TVBO.curie('n_vertices'),
                   model_uri=TVBO.regionMapping__n_vertices, domain=None, range=Optional[int])

slots.regionMapping__n_regions = Slot(uri=TVBO.n_regions, name="regionMapping__n_regions", curie=TVBO.curie('n_regions'),
                   model_uri=TVBO.regionMapping__n_regions, domain=None, range=Optional[int])

slots.sample__groups = Slot(uri=TVBO.groups, name="sample__groups", curie=TVBO.curie('groups'),
                   model_uri=TVBO.sample__groups, domain=None, range=Optional[Union[str, list[str]]])

slots.sample__size = Slot(uri=TVBO.size, name="sample__size", curie=TVBO.curie('size'),
                   model_uri=TVBO.sample__size, domain=None, range=Optional[int])

slots.simulationExperiment__id = Slot(uri=TVBO.id, name="simulationExperiment__id", curie=TVBO.curie('id'),
                   model_uri=TVBO.simulationExperiment__id, domain=None, range=URIRef)

slots.simulationExperiment__description = Slot(uri=TVBO.description, name="simulationExperiment__description", curie=TVBO.curie('description'),
                   model_uri=TVBO.simulationExperiment__description, domain=None, range=Optional[str])

slots.simulationExperiment__additional_equations = Slot(uri=TVBO.additional_equations, name="simulationExperiment__additional_equations", curie=TVBO.curie('additional_equations'),
                   model_uri=TVBO.simulationExperiment__additional_equations, domain=None, range=Optional[Union[Union[dict, Equation], list[Union[dict, Equation]]]])

slots.simulationExperiment__label = Slot(uri=TVBO.label, name="simulationExperiment__label", curie=TVBO.curie('label'),
                   model_uri=TVBO.simulationExperiment__label, domain=None, range=Optional[str])

slots.simulationExperiment__local_dynamics = Slot(uri=TVBO.local_dynamics, name="simulationExperiment__local_dynamics", curie=TVBO.curie('local_dynamics'),
                   model_uri=TVBO.simulationExperiment__local_dynamics, domain=None, range=Optional[Union[dict, Dynamics]])

slots.simulationExperiment__dynamics = Slot(uri=TVBO.dynamics, name="simulationExperiment__dynamics", curie=TVBO.curie('dynamics'),
                   model_uri=TVBO.simulationExperiment__dynamics, domain=None, range=Optional[Union[dict[Union[str, DynamicsName], Union[dict, Dynamics]], list[Union[dict, Dynamics]]]])

slots.simulationExperiment__integration = Slot(uri=TVBO.integration, name="simulationExperiment__integration", curie=TVBO.curie('integration'),
                   model_uri=TVBO.simulationExperiment__integration, domain=None, range=Optional[Union[dict, Integrator]])

slots.simulationExperiment__connectivity = Slot(uri=TVBO.connectivity, name="simulationExperiment__connectivity", curie=TVBO.curie('connectivity'),
                   model_uri=TVBO.simulationExperiment__connectivity, domain=None, range=Optional[Union[dict, Network]])

slots.simulationExperiment__network = Slot(uri=TVBO.network, name="simulationExperiment__network", curie=TVBO.curie('network'),
                   model_uri=TVBO.simulationExperiment__network, domain=None, range=Optional[Union[dict, Network]])

slots.simulationExperiment__coupling = Slot(uri=TVBO.coupling, name="simulationExperiment__coupling", curie=TVBO.curie('coupling'),
                   model_uri=TVBO.simulationExperiment__coupling, domain=None, range=Optional[Union[dict, Coupling]])

slots.simulationExperiment__monitors = Slot(uri=TVBO.monitors, name="simulationExperiment__monitors", curie=TVBO.curie('monitors'),
                   model_uri=TVBO.simulationExperiment__monitors, domain=None, range=Optional[Union[dict[Union[str, MonitorName], Union[dict, Monitor]], list[Union[dict, Monitor]]]])

slots.simulationExperiment__stimulation = Slot(uri=TVBO.stimulation, name="simulationExperiment__stimulation", curie=TVBO.curie('stimulation'),
                   model_uri=TVBO.simulationExperiment__stimulation, domain=None, range=Optional[Union[dict, Stimulus]])

slots.simulationExperiment__field_dynamics = Slot(uri=TVBO.field_dynamics, name="simulationExperiment__field_dynamics", curie=TVBO.curie('field_dynamics'),
                   model_uri=TVBO.simulationExperiment__field_dynamics, domain=None, range=Optional[Union[dict, PDE]])

slots.simulationExperiment__modelfitting = Slot(uri=TVBO.modelfitting, name="simulationExperiment__modelfitting", curie=TVBO.curie('modelfitting'),
                   model_uri=TVBO.simulationExperiment__modelfitting, domain=None, range=Optional[Union[Union[dict, ModelFitting], list[Union[dict, ModelFitting]]]])

slots.simulationExperiment__environment = Slot(uri=TVBO.environment, name="simulationExperiment__environment", curie=TVBO.curie('environment'),
                   model_uri=TVBO.simulationExperiment__environment, domain=None, range=Optional[Union[dict, SoftwareEnvironment]])

slots.simulationExperiment__software = Slot(uri=TVBO.software, name="simulationExperiment__software", curie=TVBO.curie('software'),
                   model_uri=TVBO.simulationExperiment__software, domain=None, range=Optional[Union[dict, SoftwareRequirement]])

slots.simulationExperiment__references = Slot(uri=TVBO.references, name="simulationExperiment__references", curie=TVBO.curie('references'),
                   model_uri=TVBO.simulationExperiment__references, domain=None, range=Optional[Union[str, list[str]]])

slots.simulationStudy__key = Slot(uri=TVBO.key, name="simulationStudy__key", curie=TVBO.curie('key'),
                   model_uri=TVBO.simulationStudy__key, domain=None, range=Optional[str])

slots.simulationStudy__title = Slot(uri=TVBO.title, name="simulationStudy__title", curie=TVBO.curie('title'),
                   model_uri=TVBO.simulationStudy__title, domain=None, range=Optional[str])

slots.simulationStudy__year = Slot(uri=TVBO.year, name="simulationStudy__year", curie=TVBO.curie('year'),
                   model_uri=TVBO.simulationStudy__year, domain=None, range=Optional[int])

slots.simulationStudy__doi = Slot(uri=TVBO.doi, name="simulationStudy__doi", curie=TVBO.curie('doi'),
                   model_uri=TVBO.simulationStudy__doi, domain=None, range=Optional[str])

slots.simulationStudy__sample = Slot(uri=TVBO.sample, name="simulationStudy__sample", curie=TVBO.curie('sample'),
                   model_uri=TVBO.simulationStudy__sample, domain=None, range=Optional[Union[dict, Sample]])

slots.simulationStudy__simulation_experiments = Slot(uri=TVBO.simulation_experiments, name="simulationStudy__simulation_experiments", curie=TVBO.curie('simulation_experiments'),
                   model_uri=TVBO.simulationStudy__simulation_experiments, domain=None, range=Optional[Union[dict[Union[int, SimulationExperimentId], Union[dict, SimulationExperiment]], list[Union[dict, SimulationExperiment]]]])

slots.timeSeries__data = Slot(uri=TVBO.data, name="timeSeries__data", curie=TVBO.curie('data'),
                   model_uri=TVBO.timeSeries__data, domain=None, range=Optional[Union[dict, Matrix]])

slots.timeSeries__time = Slot(uri=TVBO.time, name="timeSeries__time", curie=TVBO.curie('time'),
                   model_uri=TVBO.timeSeries__time, domain=None, range=Optional[Union[dict, Matrix]])

slots.timeSeries__sampling_rate = Slot(uri=TVBO.sampling_rate, name="timeSeries__sampling_rate", curie=TVBO.curie('sampling_rate'),
                   model_uri=TVBO.timeSeries__sampling_rate, domain=None, range=Optional[float])

slots.timeSeries__unit = Slot(uri=TVBO.unit, name="timeSeries__unit", curie=TVBO.curie('unit'),
                   model_uri=TVBO.timeSeries__unit, domain=None, range=Optional[str])

slots.softwareEnvironment__name = Slot(uri=TVBO.name, name="softwareEnvironment__name", curie=TVBO.curie('name'),
                   model_uri=TVBO.softwareEnvironment__name, domain=None, range=Optional[str])

slots.softwareEnvironment__version = Slot(uri=TVBO.version, name="softwareEnvironment__version", curie=TVBO.curie('version'),
                   model_uri=TVBO.softwareEnvironment__version, domain=None, range=Optional[str])

slots.softwareEnvironment__platform = Slot(uri=TVBO.platform, name="softwareEnvironment__platform", curie=TVBO.curie('platform'),
                   model_uri=TVBO.softwareEnvironment__platform, domain=None, range=Optional[str])

slots.softwareEnvironment__environment_type = Slot(uri=TVBO.environment_type, name="softwareEnvironment__environment_type", curie=TVBO.curie('environment_type'),
                   model_uri=TVBO.softwareEnvironment__environment_type, domain=None, range=Optional[Union[str, "EnvironmentType"]])

slots.softwareEnvironment__container_image = Slot(uri=TVBO.container_image, name="softwareEnvironment__container_image", curie=TVBO.curie('container_image'),
                   model_uri=TVBO.softwareEnvironment__container_image, domain=None, range=Optional[str])

slots.softwareEnvironment__build_hash = Slot(uri=TVBO.build_hash, name="softwareEnvironment__build_hash", curie=TVBO.curie('build_hash'),
                   model_uri=TVBO.softwareEnvironment__build_hash, domain=None, range=Optional[str])

slots.softwareEnvironment__requirements = Slot(uri=TVBO.requirements, name="softwareEnvironment__requirements", curie=TVBO.curie('requirements'),
                   model_uri=TVBO.softwareEnvironment__requirements, domain=None, range=Optional[Union[dict[Union[str, SoftwareRequirementName], Union[dict, SoftwareRequirement]], list[Union[dict, SoftwareRequirement]]]])

slots.softwareRequirement__package = Slot(uri=TVBO.package, name="softwareRequirement__package", curie=TVBO.curie('package'),
                   model_uri=TVBO.softwareRequirement__package, domain=None, range=Union[str, SoftwarePackageName])

slots.softwareRequirement__version_spec = Slot(uri=TVBO.version_spec, name="softwareRequirement__version_spec", curie=TVBO.curie('version_spec'),
                   model_uri=TVBO.softwareRequirement__version_spec, domain=None, range=Optional[str])

slots.softwareRequirement__role = Slot(uri=TVBO.role, name="softwareRequirement__role", curie=TVBO.curie('role'),
                   model_uri=TVBO.softwareRequirement__role, domain=None, range=Optional[Union[str, "RequirementRole"]])

slots.softwareRequirement__optional = Slot(uri=TVBO.optional, name="softwareRequirement__optional", curie=TVBO.curie('optional'),
                   model_uri=TVBO.softwareRequirement__optional, domain=None, range=Optional[Union[bool, Bool]])

slots.softwareRequirement__hash = Slot(uri=TVBO.hash, name="softwareRequirement__hash", curie=TVBO.curie('hash'),
                   model_uri=TVBO.softwareRequirement__hash, domain=None, range=Optional[str])

slots.softwareRequirement__source_url = Slot(uri=TVBO.source_url, name="softwareRequirement__source_url", curie=TVBO.curie('source_url'),
                   model_uri=TVBO.softwareRequirement__source_url, domain=None, range=Optional[str])

slots.softwareRequirement__url = Slot(uri=TVBO.url, name="softwareRequirement__url", curie=TVBO.curie('url'),
                   model_uri=TVBO.softwareRequirement__url, domain=None, range=Optional[str])

slots.softwareRequirement__license = Slot(uri=TVBO.license, name="softwareRequirement__license", curie=TVBO.curie('license'),
                   model_uri=TVBO.softwareRequirement__license, domain=None, range=Optional[str])

slots.softwareRequirement__modules = Slot(uri=TVBO.modules, name="softwareRequirement__modules", curie=TVBO.curie('modules'),
                   model_uri=TVBO.softwareRequirement__modules, domain=None, range=Optional[Union[str, list[str]]])

slots.softwareRequirement__version = Slot(uri=TVBO.version, name="softwareRequirement__version", curie=TVBO.curie('version'),
                   model_uri=TVBO.softwareRequirement__version, domain=None, range=Optional[str])

slots.softwarePackage__homepage = Slot(uri=TVBO.homepage, name="softwarePackage__homepage", curie=TVBO.curie('homepage'),
                   model_uri=TVBO.softwarePackage__homepage, domain=None, range=Optional[str])

slots.softwarePackage__license = Slot(uri=TVBO.license, name="softwarePackage__license", curie=TVBO.curie('license'),
                   model_uri=TVBO.softwarePackage__license, domain=None, range=Optional[str])

slots.softwarePackage__repository = Slot(uri=TVBO.repository, name="softwarePackage__repository", curie=TVBO.curie('repository'),
                   model_uri=TVBO.softwarePackage__repository, domain=None, range=Optional[str])

slots.softwarePackage__doi = Slot(uri=TVBO.doi, name="softwarePackage__doi", curie=TVBO.curie('doi'),
                   model_uri=TVBO.softwarePackage__doi, domain=None, range=Optional[str])

slots.softwarePackage__ecosystem = Slot(uri=TVBO.ecosystem, name="softwarePackage__ecosystem", curie=TVBO.curie('ecosystem'),
                   model_uri=TVBO.softwarePackage__ecosystem, domain=None, range=Optional[str])

slots.nDArray__shape = Slot(uri=TVBO.shape, name="nDArray__shape", curie=TVBO.curie('shape'),
                   model_uri=TVBO.nDArray__shape, domain=None, range=Optional[Union[int, list[int]]])

slots.nDArray__dtype = Slot(uri=TVBO.dtype, name="nDArray__dtype", curie=TVBO.curie('dtype'),
                   model_uri=TVBO.nDArray__dtype, domain=None, range=Optional[str])

slots.nDArray__dataLocation = Slot(uri=TVBO.dataLocation, name="nDArray__dataLocation", curie=TVBO.curie('dataLocation'),
                   model_uri=TVBO.nDArray__dataLocation, domain=None, range=Optional[str])

slots.nDArray__unit = Slot(uri=TVBO.unit, name="nDArray__unit", curie=TVBO.curie('unit'),
                   model_uri=TVBO.nDArray__unit, domain=None, range=Optional[str])

slots.spatialDomain__coordinate_space = Slot(uri=TVBO.coordinate_space, name="spatialDomain__coordinate_space", curie=TVBO.curie('coordinate_space'),
                   model_uri=TVBO.spatialDomain__coordinate_space, domain=None, range=Optional[Union[str, CommonCoordinateSpaceName]])

slots.spatialDomain__region = Slot(uri=TVBO.region, name="spatialDomain__region", curie=TVBO.curie('region'),
                   model_uri=TVBO.spatialDomain__region, domain=None, range=Optional[str])

slots.spatialDomain__geometry = Slot(uri=TVBO.geometry, name="spatialDomain__geometry", curie=TVBO.curie('geometry'),
                   model_uri=TVBO.spatialDomain__geometry, domain=None, range=Optional[str])

slots.mesh__element_type = Slot(uri=TVBO.element_type, name="mesh__element_type", curie=TVBO.curie('element_type'),
                   model_uri=TVBO.mesh__element_type, domain=None, range=Optional[Union[str, "ElementType"]])

slots.mesh__coordinates = Slot(uri=TVBO.coordinates, name="mesh__coordinates", curie=TVBO.curie('coordinates'),
                   model_uri=TVBO.mesh__coordinates, domain=None, range=Optional[Union[Union[dict, Coordinate], list[Union[dict, Coordinate]]]])

slots.mesh__elements = Slot(uri=TVBO.elements, name="mesh__elements", curie=TVBO.curie('elements'),
                   model_uri=TVBO.mesh__elements, domain=None, range=Optional[str])

slots.mesh__coordinate_space = Slot(uri=TVBO.coordinate_space, name="mesh__coordinate_space", curie=TVBO.curie('coordinate_space'),
                   model_uri=TVBO.mesh__coordinate_space, domain=None, range=Optional[Union[str, CommonCoordinateSpaceName]])

slots.spatialField__quantity_kind = Slot(uri=TVBO.quantity_kind, name="spatialField__quantity_kind", curie=TVBO.curie('quantity_kind'),
                   model_uri=TVBO.spatialField__quantity_kind, domain=None, range=Optional[str])

slots.spatialField__unit = Slot(uri=TVBO.unit, name="spatialField__unit", curie=TVBO.curie('unit'),
                   model_uri=TVBO.spatialField__unit, domain=None, range=Optional[str])

slots.spatialField__mesh = Slot(uri=TVBO.mesh, name="spatialField__mesh", curie=TVBO.curie('mesh'),
                   model_uri=TVBO.spatialField__mesh, domain=None, range=Optional[Union[dict, Mesh]])

slots.spatialField__values = Slot(uri=TVBO.values, name="spatialField__values", curie=TVBO.curie('values'),
                   model_uri=TVBO.spatialField__values, domain=None, range=Optional[Union[dict, NDArray]])

slots.spatialField__time_dependent = Slot(uri=TVBO.time_dependent, name="spatialField__time_dependent", curie=TVBO.curie('time_dependent'),
                   model_uri=TVBO.spatialField__time_dependent, domain=None, range=Optional[Union[bool, Bool]])

slots.spatialField__initial_value = Slot(uri=TVBO.initial_value, name="spatialField__initial_value", curie=TVBO.curie('initial_value'),
                   model_uri=TVBO.spatialField__initial_value, domain=None, range=Optional[float])

slots.spatialField__initial_expression = Slot(uri=TVBO.initial_expression, name="spatialField__initial_expression", curie=TVBO.curie('initial_expression'),
                   model_uri=TVBO.spatialField__initial_expression, domain=None, range=Optional[Union[dict, Equation]])

slots.fieldStateVariable__mesh = Slot(uri=TVBO.mesh, name="fieldStateVariable__mesh", curie=TVBO.curie('mesh'),
                   model_uri=TVBO.fieldStateVariable__mesh, domain=None, range=Optional[Union[dict, Mesh]])

slots.fieldStateVariable__boundary_conditions = Slot(uri=TVBO.boundary_conditions, name="fieldStateVariable__boundary_conditions", curie=TVBO.curie('boundary_conditions'),
                   model_uri=TVBO.fieldStateVariable__boundary_conditions, domain=None, range=Optional[Union[Union[dict, BoundaryCondition], list[Union[dict, BoundaryCondition]]]])

slots.differentialOperator__operator_type = Slot(uri=TVBO.operator_type, name="differentialOperator__operator_type", curie=TVBO.curie('operator_type'),
                   model_uri=TVBO.differentialOperator__operator_type, domain=None, range=Optional[Union[str, "OperatorType"]])

slots.differentialOperator__coefficient = Slot(uri=TVBO.coefficient, name="differentialOperator__coefficient", curie=TVBO.curie('coefficient'),
                   model_uri=TVBO.differentialOperator__coefficient, domain=None, range=Optional[Union[str, ParameterName]])

slots.differentialOperator__tensor_coefficient = Slot(uri=TVBO.tensor_coefficient, name="differentialOperator__tensor_coefficient", curie=TVBO.curie('tensor_coefficient'),
                   model_uri=TVBO.differentialOperator__tensor_coefficient, domain=None, range=Optional[Union[str, ParameterName]])

slots.differentialOperator__expression = Slot(uri=TVBO.expression, name="differentialOperator__expression", curie=TVBO.curie('expression'),
                   model_uri=TVBO.differentialOperator__expression, domain=None, range=Optional[Union[dict, Equation]])

slots.boundaryCondition__bc_type = Slot(uri=TVBO.bc_type, name="boundaryCondition__bc_type", curie=TVBO.curie('bc_type'),
                   model_uri=TVBO.boundaryCondition__bc_type, domain=None, range=Optional[Union[str, "BoundaryConditionType"]])

slots.boundaryCondition__on_region = Slot(uri=TVBO.on_region, name="boundaryCondition__on_region", curie=TVBO.curie('on_region'),
                   model_uri=TVBO.boundaryCondition__on_region, domain=None, range=Optional[str])

slots.boundaryCondition__value = Slot(uri=TVBO.value, name="boundaryCondition__value", curie=TVBO.curie('value'),
                   model_uri=TVBO.boundaryCondition__value, domain=None, range=Optional[Union[dict, Equation]])

slots.boundaryCondition__time_dependent = Slot(uri=TVBO.time_dependent, name="boundaryCondition__time_dependent", curie=TVBO.curie('time_dependent'),
                   model_uri=TVBO.boundaryCondition__time_dependent, domain=None, range=Optional[Union[bool, Bool]])

slots.pDESolver__discretization = Slot(uri=TVBO.discretization, name="pDESolver__discretization", curie=TVBO.curie('discretization'),
                   model_uri=TVBO.pDESolver__discretization, domain=None, range=Optional[Union[str, "DiscretizationMethod"]])

slots.pDESolver__time_integrator = Slot(uri=TVBO.time_integrator, name="pDESolver__time_integrator", curie=TVBO.curie('time_integrator'),
                   model_uri=TVBO.pDESolver__time_integrator, domain=None, range=Optional[str])

slots.pDESolver__dt = Slot(uri=TVBO.step_size, name="pDESolver__dt", curie=TVBO.curie('step_size'),
                   model_uri=TVBO.pDESolver__dt, domain=None, range=Optional[float])

slots.pDESolver__tolerances = Slot(uri=TVBO.tolerances, name="pDESolver__tolerances", curie=TVBO.curie('tolerances'),
                   model_uri=TVBO.pDESolver__tolerances, domain=None, range=Optional[str])

slots.pDESolver__preconditioner = Slot(uri=TVBO.preconditioner, name="pDESolver__preconditioner", curie=TVBO.curie('preconditioner'),
                   model_uri=TVBO.pDESolver__preconditioner, domain=None, range=Optional[str])

slots.pDE__domain = Slot(uri=TVBO.domain, name="pDE__domain", curie=TVBO.curie('domain'),
                   model_uri=TVBO.pDE__domain, domain=None, range=Optional[Union[dict, SpatialDomain]])

slots.pDE__mesh = Slot(uri=TVBO.mesh, name="pDE__mesh", curie=TVBO.curie('mesh'),
                   model_uri=TVBO.pDE__mesh, domain=None, range=Optional[Union[dict, Mesh]])

slots.pDE__state_variables = Slot(uri=TVBO.state_variables, name="pDE__state_variables", curie=TVBO.curie('state_variables'),
                   model_uri=TVBO.pDE__state_variables, domain=None, range=Optional[Union[Union[str, FieldStateVariableName], list[Union[str, FieldStateVariableName]]]])

slots.pDE__field = Slot(uri=TVBO.field, name="pDE__field", curie=TVBO.curie('field'),
                   model_uri=TVBO.pDE__field, domain=None, range=Optional[Union[dict, SpatialField]])

slots.pDE__operators = Slot(uri=TVBO.operators, name="pDE__operators", curie=TVBO.curie('operators'),
                   model_uri=TVBO.pDE__operators, domain=None, range=Optional[Union[Union[dict, DifferentialOperator], list[Union[dict, DifferentialOperator]]]])

slots.pDE__sources = Slot(uri=TVBO.sources, name="pDE__sources", curie=TVBO.curie('sources'),
                   model_uri=TVBO.pDE__sources, domain=None, range=Optional[Union[Union[dict, Equation], list[Union[dict, Equation]]]])

slots.pDE__boundary_conditions = Slot(uri=TVBO.boundary_conditions, name="pDE__boundary_conditions", curie=TVBO.curie('boundary_conditions'),
                   model_uri=TVBO.pDE__boundary_conditions, domain=None, range=Optional[Union[Union[dict, BoundaryCondition], list[Union[dict, BoundaryCondition]]]])

slots.pDE__solver = Slot(uri=TVBO.solver, name="pDE__solver", curie=TVBO.curie('solver'),
                   model_uri=TVBO.pDE__solver, domain=None, range=Optional[Union[dict, PDESolver]])

slots.pDE__derived_parameters = Slot(uri=TVBO.derived_parameters, name="pDE__derived_parameters", curie=TVBO.curie('derived_parameters'),
                   model_uri=TVBO.pDE__derived_parameters, domain=None, range=Optional[Union[Union[str, DerivedParameterName], list[Union[str, DerivedParameterName]]]])

slots.pDE__derived_variables = Slot(uri=TVBO.derived_variables, name="pDE__derived_variables", curie=TVBO.curie('derived_variables'),
                   model_uri=TVBO.pDE__derived_variables, domain=None, range=Optional[Union[Union[str, DerivedVariableName], list[Union[str, DerivedVariableName]]]])

slots.pDE__functions = Slot(uri=TVBO.functions, name="pDE__functions", curie=TVBO.curie('functions'),
                   model_uri=TVBO.pDE__functions, domain=None, range=Optional[Union[Union[str, FunctionName], list[Union[str, FunctionName]]]])

slots.coordinate__x = Slot(uri=ATOM.x, name="coordinate__x", curie=ATOM.curie('x'),
                   model_uri=TVBO.coordinate__x, domain=None, range=Optional[float])

slots.coordinate__y = Slot(uri=ATOM.y, name="coordinate__y", curie=ATOM.curie('y'),
                   model_uri=TVBO.coordinate__y, domain=None, range=Optional[float])

slots.coordinate__z = Slot(uri=ATOM.z, name="coordinate__z", curie=ATOM.curie('z'),
                   model_uri=TVBO.coordinate__z, domain=None, range=Optional[float])

slots.brainAtlas__terminology = Slot(uri=ATOM.terminology, name="brainAtlas__terminology", curie=ATOM.curie('terminology'),
                   model_uri=TVBO.brainAtlas__terminology, domain=None, range=Optional[Union[dict, ParcellationTerminology]])

slots.commonCoordinateSpace__anatomicalAxesOrientation = Slot(uri=ATOM.anatomicalAxesOrientation, name="commonCoordinateSpace__anatomicalAxesOrientation", curie=ATOM.curie('anatomicalAxesOrientation'),
                   model_uri=TVBO.commonCoordinateSpace__anatomicalAxesOrientation, domain=None, range=Optional[str])

slots.commonCoordinateSpace__axesOrigin = Slot(uri=ATOM.axesOrigin, name="commonCoordinateSpace__axesOrigin", curie=ATOM.curie('axesOrigin'),
                   model_uri=TVBO.commonCoordinateSpace__axesOrigin, domain=None, range=Optional[str])

slots.commonCoordinateSpace__nativeUnit = Slot(uri=ATOM.nativeUnit, name="commonCoordinateSpace__nativeUnit", curie=ATOM.curie('nativeUnit'),
                   model_uri=TVBO.commonCoordinateSpace__nativeUnit, domain=None, range=Optional[str])

slots.commonCoordinateSpace__defaultImage = Slot(uri=ATOM.defaultImage, name="commonCoordinateSpace__defaultImage", curie=ATOM.curie('defaultImage'),
                   model_uri=TVBO.commonCoordinateSpace__defaultImage, domain=None, range=Optional[Union[str, list[str]]])

slots.parcellationEntity__relatedUBERONTerm = Slot(uri=ATOM.relatedUBERONTerm, name="parcellationEntity__relatedUBERONTerm", curie=ATOM.curie('relatedUBERONTerm'),
                   model_uri=TVBO.parcellationEntity__relatedUBERONTerm, domain=None, range=Optional[str])

slots.parcellationEntity__originalLookupLabel = Slot(uri=ATOM.originalLookupLabel, name="parcellationEntity__originalLookupLabel", curie=ATOM.curie('originalLookupLabel'),
                   model_uri=TVBO.parcellationEntity__originalLookupLabel, domain=None, range=Optional[int])

slots.parcellationEntity__hemisphere = Slot(uri=ATOM.hemisphere, name="parcellationEntity__hemisphere", curie=ATOM.curie('hemisphere'),
                   model_uri=TVBO.parcellationEntity__hemisphere, domain=None, range=Optional[Union[str, "Hemisphere"]])

slots.parcellationEntity__center = Slot(uri=ATOM.center, name="parcellationEntity__center", curie=ATOM.curie('center'),
                   model_uri=TVBO.parcellationEntity__center, domain=None, range=Optional[Union[dict, Coordinate]])

slots.parcellationEntity__color = Slot(uri=ATOM.color, name="parcellationEntity__color", curie=ATOM.curie('color'),
                   model_uri=TVBO.parcellationEntity__color, domain=None, range=Optional[str])

slots.parcellationTerminology__entities = Slot(uri=ATOM.entities, name="parcellationTerminology__entities", curie=ATOM.curie('entities'),
                   model_uri=TVBO.parcellationTerminology__entities, domain=None, range=Optional[Union[Union[str, ParcellationEntityName], list[Union[str, ParcellationEntityName]]]])

slots.dataset__dataset_id = Slot(uri=TVBO_DBS.dataset_id, name="dataset__dataset_id", curie=TVBO_DBS.curie('dataset_id'),
                   model_uri=TVBO.dataset__dataset_id, domain=None, range=Optional[str])

slots.dataset__subjects = Slot(uri=TVBO_DBS.subjects, name="dataset__subjects", curie=TVBO_DBS.curie('subjects'),
                   model_uri=TVBO.dataset__subjects, domain=None, range=Optional[Union[dict[Union[str, SubjectSubjectId], Union[dict, Subject]], list[Union[dict, Subject]]]])

slots.dataset__clinical_scores = Slot(uri=TVBO_DBS.clinical_scores, name="dataset__clinical_scores", curie=TVBO_DBS.curie('clinical_scores'),
                   model_uri=TVBO.dataset__clinical_scores, domain=None, range=Optional[Union[Union[dict, ClinicalScore], list[Union[dict, ClinicalScore]]]])

slots.dataset__coordinate_space = Slot(uri=TVBO_DBS.coordinate_space, name="dataset__coordinate_space", curie=TVBO_DBS.curie('coordinate_space'),
                   model_uri=TVBO.dataset__coordinate_space, domain=None, range=Optional[Union[str, CommonCoordinateSpaceName]])

slots.subject__age = Slot(uri=TVBO_DBS.age, name="subject__age", curie=TVBO_DBS.curie('age'),
                   model_uri=TVBO.subject__age, domain=None, range=Optional[float])

slots.subject__sex = Slot(uri=TVBO_DBS.sex, name="subject__sex", curie=TVBO_DBS.curie('sex'),
                   model_uri=TVBO.subject__sex, domain=None, range=Optional[str])

slots.subject__diagnosis = Slot(uri=TVBO_DBS.diagnosis, name="subject__diagnosis", curie=TVBO_DBS.curie('diagnosis'),
                   model_uri=TVBO.subject__diagnosis, domain=None, range=Optional[str])

slots.subject__handedness = Slot(uri=TVBO_DBS.handedness, name="subject__handedness", curie=TVBO_DBS.curie('handedness'),
                   model_uri=TVBO.subject__handedness, domain=None, range=Optional[str])

slots.subject__protocols = Slot(uri=TVBO_DBS.protocols, name="subject__protocols", curie=TVBO_DBS.curie('protocols'),
                   model_uri=TVBO.subject__protocols, domain=None, range=Optional[Union[Union[str, DBSProtocolName], list[Union[str, DBSProtocolName]]]])

slots.subject__coordinate_space = Slot(uri=TVBO_DBS.coordinate_space, name="subject__coordinate_space", curie=TVBO_DBS.curie('coordinate_space'),
                   model_uri=TVBO.subject__coordinate_space, domain=None, range=Optional[Union[str, CommonCoordinateSpaceName]])

slots.electrode__electrode_id = Slot(uri=TVBO_DBS.electrode_id, name="electrode__electrode_id", curie=TVBO_DBS.curie('electrode_id'),
                   model_uri=TVBO.electrode__electrode_id, domain=None, range=Optional[str])

slots.electrode__manufacturer = Slot(uri=TVBO_DBS.manufacturer, name="electrode__manufacturer", curie=TVBO_DBS.curie('manufacturer'),
                   model_uri=TVBO.electrode__manufacturer, domain=None, range=Optional[str])

slots.electrode__model = Slot(uri=TVBO_DBS.model, name="electrode__model", curie=TVBO_DBS.curie('model'),
                   model_uri=TVBO.electrode__model, domain=None, range=Optional[str])

slots.electrode__hemisphere = Slot(uri=TVBO_DBS.hemisphere, name="electrode__hemisphere", curie=TVBO_DBS.curie('hemisphere'),
                   model_uri=TVBO.electrode__hemisphere, domain=None, range=Optional[str])

slots.electrode__contacts = Slot(uri=TVBO_DBS.contacts, name="electrode__contacts", curie=TVBO_DBS.curie('contacts'),
                   model_uri=TVBO.electrode__contacts, domain=None, range=Optional[Union[Union[dict, Contact], list[Union[dict, Contact]]]])

slots.electrode__head = Slot(uri=TVBO_DBS.head, name="electrode__head", curie=TVBO_DBS.curie('head'),
                   model_uri=TVBO.electrode__head, domain=None, range=Optional[Union[dict, Coordinate]])

slots.electrode__tail = Slot(uri=TVBO_DBS.tail, name="electrode__tail", curie=TVBO_DBS.curie('tail'),
                   model_uri=TVBO.electrode__tail, domain=None, range=Optional[Union[dict, Coordinate]])

slots.electrode__trajectory = Slot(uri=TVBO_DBS.trajectory, name="electrode__trajectory", curie=TVBO_DBS.curie('trajectory'),
                   model_uri=TVBO.electrode__trajectory, domain=None, range=Optional[Union[Union[dict, Coordinate], list[Union[dict, Coordinate]]]])

slots.electrode__target_structure = Slot(uri=TVBO_DBS.target_structure, name="electrode__target_structure", curie=TVBO_DBS.curie('target_structure'),
                   model_uri=TVBO.electrode__target_structure, domain=None, range=Optional[Union[str, ParcellationEntityName]])

slots.electrode__coordinate_space = Slot(uri=TVBO_DBS.coordinate_space, name="electrode__coordinate_space", curie=TVBO_DBS.curie('coordinate_space'),
                   model_uri=TVBO.electrode__coordinate_space, domain=None, range=Optional[Union[str, CommonCoordinateSpaceName]])

slots.electrode__recon_path = Slot(uri=TVBO_DBS.recon_path, name="electrode__recon_path", curie=TVBO_DBS.curie('recon_path'),
                   model_uri=TVBO.electrode__recon_path, domain=None, range=Optional[str])

slots.contact__contact_id = Slot(uri=TVBO_DBS.contact_id, name="contact__contact_id", curie=TVBO_DBS.curie('contact_id'),
                   model_uri=TVBO.contact__contact_id, domain=None, range=Optional[int])

slots.contact__coordinate = Slot(uri=TVBO_DBS.coordinate, name="contact__coordinate", curie=TVBO_DBS.curie('coordinate'),
                   model_uri=TVBO.contact__coordinate, domain=None, range=Optional[Union[dict, Coordinate]])

slots.contact__label = Slot(uri=TVBO_DBS.label, name="contact__label", curie=TVBO_DBS.curie('label'),
                   model_uri=TVBO.contact__label, domain=None, range=Optional[str])

slots.stimulationSetting__electrode_reference = Slot(uri=TVBO_DBS.electrode_reference, name="stimulationSetting__electrode_reference", curie=TVBO_DBS.curie('electrode_reference'),
                   model_uri=TVBO.stimulationSetting__electrode_reference, domain=None, range=Optional[Union[dict, Electrode]])

slots.stimulationSetting__amplitude = Slot(uri=TVBO_DBS.amplitude, name="stimulationSetting__amplitude", curie=TVBO_DBS.curie('amplitude'),
                   model_uri=TVBO.stimulationSetting__amplitude, domain=None, range=Optional[Union[str, ParameterName]])

slots.stimulationSetting__frequency = Slot(uri=TVBO_DBS.frequency, name="stimulationSetting__frequency", curie=TVBO_DBS.curie('frequency'),
                   model_uri=TVBO.stimulationSetting__frequency, domain=None, range=Optional[Union[str, ParameterName]])

slots.stimulationSetting__pulse_width = Slot(uri=TVBO_DBS.pulse_width, name="stimulationSetting__pulse_width", curie=TVBO_DBS.curie('pulse_width'),
                   model_uri=TVBO.stimulationSetting__pulse_width, domain=None, range=Optional[Union[str, ParameterName]])

slots.stimulationSetting__mode = Slot(uri=TVBO_DBS.mode, name="stimulationSetting__mode", curie=TVBO_DBS.curie('mode'),
                   model_uri=TVBO.stimulationSetting__mode, domain=None, range=Optional[str])

slots.stimulationSetting__active_contacts = Slot(uri=TVBO_DBS.active_contacts, name="stimulationSetting__active_contacts", curie=TVBO_DBS.curie('active_contacts'),
                   model_uri=TVBO.stimulationSetting__active_contacts, domain=None, range=Optional[Union[int, list[int]]])

slots.stimulationSetting__efield = Slot(uri=TVBO_DBS.efield, name="stimulationSetting__efield", curie=TVBO_DBS.curie('efield'),
                   model_uri=TVBO.stimulationSetting__efield, domain=None, range=Optional[Union[dict, EField]])

slots.dBSProtocol__electrodes = Slot(uri=TVBO_DBS.electrodes, name="dBSProtocol__electrodes", curie=TVBO_DBS.curie('electrodes'),
                   model_uri=TVBO.dBSProtocol__electrodes, domain=None, range=Optional[Union[Union[dict, Electrode], list[Union[dict, Electrode]]]])

slots.dBSProtocol__settings = Slot(uri=TVBO_DBS.settings, name="dBSProtocol__settings", curie=TVBO_DBS.curie('settings'),
                   model_uri=TVBO.dBSProtocol__settings, domain=None, range=Optional[Union[Union[dict, StimulationSetting], list[Union[dict, StimulationSetting]]]])

slots.dBSProtocol__timing_info = Slot(uri=TVBO_DBS.timing_info, name="dBSProtocol__timing_info", curie=TVBO_DBS.curie('timing_info'),
                   model_uri=TVBO.dBSProtocol__timing_info, domain=None, range=Optional[str])

slots.dBSProtocol__notes = Slot(uri=TVBO_DBS.notes, name="dBSProtocol__notes", curie=TVBO_DBS.curie('notes'),
                   model_uri=TVBO.dBSProtocol__notes, domain=None, range=Optional[str])

slots.dBSProtocol__clinical_improvement = Slot(uri=TVBO_DBS.clinical_improvement, name="dBSProtocol__clinical_improvement", curie=TVBO_DBS.curie('clinical_improvement'),
                   model_uri=TVBO.dBSProtocol__clinical_improvement, domain=None, range=Optional[Union[Union[dict, ClinicalImprovement], list[Union[dict, ClinicalImprovement]]]])

slots.clinicalScale__name = Slot(uri=TVBO_DBS.name, name="clinicalScale__name", curie=TVBO_DBS.curie('name'),
                   model_uri=TVBO.clinicalScale__name, domain=None, range=Optional[str])

slots.clinicalScale__acronym = Slot(uri=TVBO_DBS.acronym, name="clinicalScale__acronym", curie=TVBO_DBS.curie('acronym'),
                   model_uri=TVBO.clinicalScale__acronym, domain=None, range=Optional[str])

slots.clinicalScale__version = Slot(uri=TVBO_DBS.version, name="clinicalScale__version", curie=TVBO_DBS.curie('version'),
                   model_uri=TVBO.clinicalScale__version, domain=None, range=Optional[str])

slots.clinicalScale__domain = Slot(uri=TVBO_DBS.domain, name="clinicalScale__domain", curie=TVBO_DBS.curie('domain'),
                   model_uri=TVBO.clinicalScale__domain, domain=None, range=Optional[str])

slots.clinicalScale__reference = Slot(uri=TVBO_DBS.reference, name="clinicalScale__reference", curie=TVBO_DBS.curie('reference'),
                   model_uri=TVBO.clinicalScale__reference, domain=None, range=Optional[str])

slots.clinicalScore__name = Slot(uri=TVBO_DBS.name, name="clinicalScore__name", curie=TVBO_DBS.curie('name'),
                   model_uri=TVBO.clinicalScore__name, domain=None, range=Optional[str])

slots.clinicalScore__acronym = Slot(uri=TVBO_DBS.acronym, name="clinicalScore__acronym", curie=TVBO_DBS.curie('acronym'),
                   model_uri=TVBO.clinicalScore__acronym, domain=None, range=Optional[str])

slots.clinicalScore__description = Slot(uri=TVBO_DBS.description, name="clinicalScore__description", curie=TVBO_DBS.curie('description'),
                   model_uri=TVBO.clinicalScore__description, domain=None, range=Optional[str])

slots.clinicalScore__domain = Slot(uri=TVBO_DBS.domain, name="clinicalScore__domain", curie=TVBO_DBS.curie('domain'),
                   model_uri=TVBO.clinicalScore__domain, domain=None, range=Optional[str])

slots.clinicalScore__reference = Slot(uri=TVBO_DBS.reference, name="clinicalScore__reference", curie=TVBO_DBS.curie('reference'),
                   model_uri=TVBO.clinicalScore__reference, domain=None, range=Optional[str])

slots.clinicalScore__scale = Slot(uri=TVBO_DBS.scale, name="clinicalScore__scale", curie=TVBO_DBS.curie('scale'),
                   model_uri=TVBO.clinicalScore__scale, domain=None, range=Optional[Union[dict, ClinicalScale]])

slots.clinicalScore__parent_score = Slot(uri=TVBO_DBS.parent_score, name="clinicalScore__parent_score", curie=TVBO_DBS.curie('parent_score'),
                   model_uri=TVBO.clinicalScore__parent_score, domain=None, range=Optional[Union[dict, ClinicalScore]])

slots.clinicalImprovement__score = Slot(uri=TVBO_DBS.score, name="clinicalImprovement__score", curie=TVBO_DBS.curie('score'),
                   model_uri=TVBO.clinicalImprovement__score, domain=None, range=Optional[Union[dict, ClinicalScore]])

slots.clinicalImprovement__baseline_value = Slot(uri=TVBO_DBS.baseline_value, name="clinicalImprovement__baseline_value", curie=TVBO_DBS.curie('baseline_value'),
                   model_uri=TVBO.clinicalImprovement__baseline_value, domain=None, range=Optional[float])

slots.clinicalImprovement__absolute_value = Slot(uri=TVBO_DBS.absolute_value, name="clinicalImprovement__absolute_value", curie=TVBO_DBS.curie('absolute_value'),
                   model_uri=TVBO.clinicalImprovement__absolute_value, domain=None, range=Optional[float])

slots.clinicalImprovement__percent_change = Slot(uri=TVBO_DBS.percent_change, name="clinicalImprovement__percent_change", curie=TVBO_DBS.curie('percent_change'),
                   model_uri=TVBO.clinicalImprovement__percent_change, domain=None, range=Optional[float])

slots.clinicalImprovement__time_post_surgery = Slot(uri=TVBO_DBS.time_post_surgery, name="clinicalImprovement__time_post_surgery", curie=TVBO_DBS.curie('time_post_surgery'),
                   model_uri=TVBO.clinicalImprovement__time_post_surgery, domain=None, range=Optional[float])

slots.clinicalImprovement__evaluator = Slot(uri=TVBO_DBS.evaluator, name="clinicalImprovement__evaluator", curie=TVBO_DBS.curie('evaluator'),
                   model_uri=TVBO.clinicalImprovement__evaluator, domain=None, range=Optional[str])

slots.clinicalImprovement__timepoint = Slot(uri=TVBO_DBS.timepoint, name="clinicalImprovement__timepoint", curie=TVBO_DBS.curie('timepoint'),
                   model_uri=TVBO.clinicalImprovement__timepoint, domain=None, range=Optional[str])

slots.eField__volume_data = Slot(uri=TVBO_DBS.volume_data, name="eField__volume_data", curie=TVBO_DBS.curie('volume_data'),
                   model_uri=TVBO.eField__volume_data, domain=None, range=Optional[str])

slots.eField__coordinate_space = Slot(uri=TVBO_DBS.coordinate_space, name="eField__coordinate_space", curie=TVBO_DBS.curie('coordinate_space'),
                   model_uri=TVBO.eField__coordinate_space, domain=None, range=Optional[Union[str, CommonCoordinateSpaceName]])

slots.eField__threshold_applied = Slot(uri=TVBO_DBS.threshold_applied, name="eField__threshold_applied", curie=TVBO_DBS.curie('threshold_applied'),
                   model_uri=TVBO.eField__threshold_applied, domain=None, range=Optional[float])

slots.system_type = Slot(uri=TVBO.system_type, name="system_type", curie=TVBO.curie('system_type'),
                   model_uri=TVBO.system_type, domain=None, range=Optional[str])

slots.Dynamics_name = Slot(uri=TVBO.name, name="Dynamics_name", curie=TVBO.curie('name'),
                   model_uri=TVBO.Dynamics_name, domain=Dynamics, range=Union[str, DynamicsName])

slots.Dynamics_system_type = Slot(uri=TVBO.system_type, name="Dynamics_system_type", curie=TVBO.curie('system_type'),
                   model_uri=TVBO.Dynamics_system_type, domain=Dynamics, range=Optional[str])

slots.Coupling_name = Slot(uri=TVBO.name, name="Coupling_name", curie=TVBO.curie('name'),
                   model_uri=TVBO.Coupling_name, domain=Coupling, range=Union[str, CouplingName])

