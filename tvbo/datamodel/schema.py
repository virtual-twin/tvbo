# Auto generated from tvbo_datamodel.yaml by pythongen.py version: 0.0.1
# Generation date: 2026-03-31T11:06:00
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

from linkml_runtime.linkml_model.types import Boolean, Datetime, Float, Integer, String
from linkml_runtime.utils.metamodelcore import Bool, XSDDateTime

metamodel_version = "1.7.0"
version = None

# Namespaces
UO = CurieNamespace('UO', 'http://purl.obolibrary.org/obo/UO_')
ATOM = CurieNamespace('atom', 'http://uri.interlex.org/tgbugs/uris/readable/')
LINKML = CurieNamespace('linkml', 'https://w3id.org/linkml/')
PROV = CurieNamespace('prov', 'http://www.w3.org/ns/prov#')
QUDT = CurieNamespace('qudt', 'http://qudt.org/vocab/unit/')
RDFS = CurieNamespace('rdfs', 'http://www.w3.org/2000/01/rdf-schema#')
SCHEMA = CurieNamespace('schema', 'http://schema.org/')
TVBO = CurieNamespace('tvbo', 'http://www.thevirtualbrain.org/tvb-o/')
TVBO_DBS = CurieNamespace('tvbo_dbs', 'http://www.thevirtualbrain.org/tvb-o/dbs/')
DEFAULT_ = TVBO


# Types

# Class references
class EventName(extended_str):
    pass


class TractogramName(extended_str):
    pass


class GraphGeneratorName(extended_str):
    pass


class FileName(extended_str):
    pass


class StateValueName(extended_str):
    pass


class ObservationName(extended_str):
    pass


class DerivedObservationName(ObservationName):
    pass


class DynamicsName(extended_str):
    pass


class StateVariableName(extended_str):
    pass


class DistributionName(extended_str):
    pass


class ParameterName(extended_str):
    pass


class CouplingInputName(extended_str):
    pass


class ArgumentName(extended_str):
    pass


class FunctionName(extended_str):
    pass


class LossFunctionName(FunctionName):
    pass


class CallableName(extended_str):
    pass


class ClassReferenceName(CallableName):
    pass


class DerivedParameterName(ParameterName):
    pass


class DerivedVariableName(extended_str):
    pass


class DataSourceName(extended_str):
    pass


class OptimizationStageName(extended_str):
    pass


class OptimizationName(OptimizationStageName):
    pass


class ExplorationName(extended_str):
    pass


class UpdateRuleName(extended_str):
    pass


class AlgorithmName(extended_str):
    pass


class OptionName(extended_str):
    pass


class BranchSwitchName(extended_str):
    pass


class ContinuationName(extended_str):
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
    """
    Specifies a range for array generation, parameter bounds, or grid exploration.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Range"]
    class_class_curie: ClassVar[str] = "tvbo:Range"
    class_name: ClassVar[str] = "Range"
    class_model_uri: ClassVar[URIRef] = TVBO.Range

    lo: Optional[str] = "0"
    hi: Optional[str] = None
    step: Optional[str] = None
    n: Optional[int] = None
    log_scale: Optional[Union[bool, Bool]] = False
    explored_values: Optional[Union[float, list[float]]] = empty_list()
    element: Optional[int] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.lo is not None and not isinstance(self.lo, str):
            self.lo = str(self.lo)

        if self.hi is not None and not isinstance(self.hi, str):
            self.hi = str(self.hi)

        if self.step is not None and not isinstance(self.step, str):
            self.step = str(self.step)

        if self.n is not None and not isinstance(self.n, int):
            self.n = int(self.n)

        if self.log_scale is not None and not isinstance(self.log_scale, Bool):
            self.log_scale = Bool(self.log_scale)

        if not isinstance(self.explored_values, list):
            self.explored_values = [self.explored_values] if self.explored_values is not None else []
        self.explored_values = [v if isinstance(v, float) else float(v) for v in self.explored_values]

        if self.element is not None and not isinstance(self.element, int):
            self.element = int(self.element)

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
    description: Optional[str] = None
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

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

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
class Event(YAMLRoot):
    """
    A discrete or continuous event that modifies the system during simulation. Generalizes Stimulus: can represent
    external inputs (stimulus type), threshold-triggered state changes (continuous/discrete type), or time-scheduled
    interventions (preset_time type). Attaches to components (nodes/edges) or to the experiment level.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Event"]
    class_class_curie: ClassVar[str] = "tvbo:Event"
    class_name: ClassVar[str] = "Event"
    class_model_uri: ClassVar[URIRef] = TVBO.Event

    name: Union[str, EventName] = None
    label: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    event_type: Optional[Union[str, "EventType"]] = 'stimulus'
    condition: Optional[Union[dict, Equation]] = None
    condition_states: Optional[Union[str, list[str]]] = empty_list()
    condition_parameters: Optional[Union[str, list[str]]] = empty_list()
    affect: Optional[Union[dict, Equation]] = None
    affect_states: Optional[Union[str, list[str]]] = empty_list()
    affect_parameters: Optional[Union[str, list[str]]] = empty_list()
    affect_negative: Optional[Union[dict, Equation]] = None
    trigger_times: Optional[Union[float, list[float]]] = empty_list()
    target_component: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    regions: Optional[Union[int, list[int]]] = empty_list()
    weighting: Optional[Union[float, list[float]]] = empty_list()
    duration: Optional[float] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, EventName):
            self.name = EventName(self.name)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.event_type is not None and not isinstance(self.event_type, EventType):
            self.event_type = getattr(EventType, self.event_type)

        if self.condition is not None and not isinstance(self.condition, Equation):
            self.condition = Equation(**as_dict(self.condition))

        if not isinstance(self.condition_states, list):
            self.condition_states = [self.condition_states] if self.condition_states is not None else []
        self.condition_states = [v if isinstance(v, str) else str(v) for v in self.condition_states]

        if not isinstance(self.condition_parameters, list):
            self.condition_parameters = [self.condition_parameters] if self.condition_parameters is not None else []
        self.condition_parameters = [v if isinstance(v, str) else str(v) for v in self.condition_parameters]

        if self.affect is not None and not isinstance(self.affect, Equation):
            self.affect = Equation(**as_dict(self.affect))

        if not isinstance(self.affect_states, list):
            self.affect_states = [self.affect_states] if self.affect_states is not None else []
        self.affect_states = [v if isinstance(v, str) else str(v) for v in self.affect_states]

        if not isinstance(self.affect_parameters, list):
            self.affect_parameters = [self.affect_parameters] if self.affect_parameters is not None else []
        self.affect_parameters = [v if isinstance(v, str) else str(v) for v in self.affect_parameters]

        if self.affect_negative is not None and not isinstance(self.affect_negative, Equation):
            self.affect_negative = Equation(**as_dict(self.affect_negative))

        if not isinstance(self.trigger_times, list):
            self.trigger_times = [self.trigger_times] if self.trigger_times is not None else []
        self.trigger_times = [v if isinstance(v, float) else float(v) for v in self.trigger_times]

        if self.target_component is not None and not isinstance(self.target_component, str):
            self.target_component = str(self.target_component)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if not isinstance(self.regions, list):
            self.regions = [self.regions] if self.regions is not None else []
        self.regions = [v if isinstance(v, int) else int(v) for v in self.regions]

        if not isinstance(self.weighting, list):
            self.weighting = [self.weighting] if self.weighting is not None else []
        self.weighting = [v if isinstance(v, float) else float(v) for v in self.weighting]

        if self.duration is not None and not isinstance(self.duration, float):
            self.duration = float(self.duration)

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
    data_source: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.atlas):
            self.MissingRequiredField("atlas")
        if not isinstance(self.atlas, BrainAtlas):
            self.atlas = BrainAtlas(**as_dict(self.atlas))

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.data_source is not None and not isinstance(self.data_source, str):
            self.data_source = str(self.data_source)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Tractogram(YAMLRoot):
    """
    Reference to tractography/diffusion MRI data used to derive structural connectivity
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Tractogram"]
    class_class_curie: ClassVar[str] = "tvbo:Tractogram"
    class_name: ClassVar[str] = "Tractogram"
    class_model_uri: ClassVar[URIRef] = TVBO.Tractogram

    name: Union[str, TractogramName] = None
    label: Optional[str] = None
    description: Optional[str] = None
    data_source: Optional[str] = None
    number_of_subjects: Optional[int] = None
    acquisition: Optional[str] = None
    processing_pipeline: Optional[str] = None
    reference: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, TractogramName):
            self.name = TractogramName(self.name)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.data_source is not None and not isinstance(self.data_source, str):
            self.data_source = str(self.data_source)

        if self.number_of_subjects is not None and not isinstance(self.number_of_subjects, int):
            self.number_of_subjects = int(self.number_of_subjects)

        if self.acquisition is not None and not isinstance(self.acquisition, str):
            self.acquisition = str(self.acquisition)

        if self.processing_pipeline is not None and not isinstance(self.processing_pipeline, str):
            self.processing_pipeline = str(self.processing_pipeline)

        if self.reference is not None and not isinstance(self.reference, str):
            self.reference = str(self.reference)

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
    format: Optional[Union[str, "SparseFormat"]] = None
    shape: Optional[Union[int, list[int]]] = empty_list()
    dtype: Optional[str] = "float32"

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

        if self.format is not None and not isinstance(self.format, SparseFormat):
            self.format = SparseFormat(self.format)

        if not isinstance(self.shape, list):
            self.shape = [self.shape] if self.shape is not None else []
        self.shape = [v if isinstance(v, int) else int(v) for v in self.shape]

        if self.dtype is not None and not isinstance(self.dtype, str):
            self.dtype = str(self.dtype)

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
class Provenance(YAMLRoot):
    """
    W3C PROV-O aligned provenance. Reusable on any entity (Network, TimeSeries, Dynamics, etc.).
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = PROV["Entity"]
    class_class_curie: ClassVar[str] = "prov:Entity"
    class_name: ClassVar[str] = "Provenance"
    class_model_uri: ClassVar[URIRef] = TVBO.Provenance

    derived_from: Optional[str] = None
    references: Optional[Union[str, list[str]]] = empty_list()
    date_created: Optional[str] = None
    license: Optional[str] = None
    generated_by: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.derived_from is not None and not isinstance(self.derived_from, str):
            self.derived_from = str(self.derived_from)

        if not isinstance(self.references, list):
            self.references = [self.references] if self.references is not None else []
        self.references = [v if isinstance(v, str) else str(v) for v in self.references]

        if self.date_created is not None and not isinstance(self.date_created, str):
            self.date_created = str(self.date_created)

        if self.license is not None and not isinstance(self.license, str):
            self.license = str(self.license)

        if self.generated_by is not None and not isinstance(self.generated_by, str):
            self.generated_by = str(self.generated_by)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class BidsEntities(YAMLRoot):
    """
    BIDS filename entities (BEP017-aligned) for provenance and data discovery. Reusable on Network, BrainAtlas,
    Tractogram, or any dataset with BIDS-conformant naming.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["BidsEntities"]
    class_class_curie: ClassVar[str] = "tvbo:BidsEntities"
    class_name: ClassVar[str] = "BidsEntities"
    class_model_uri: ClassVar[URIRef] = TVBO.BidsEntities

    template: Optional[str] = None
    cohort: Optional[str] = None
    reconstruction: Optional[str] = None
    segmentation: Optional[str] = None
    scale: Optional[str] = None
    atlas: Optional[str] = None
    acquisition: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.template is not None and not isinstance(self.template, str):
            self.template = str(self.template)

        if self.cohort is not None and not isinstance(self.cohort, str):
            self.cohort = str(self.cohort)

        if self.reconstruction is not None and not isinstance(self.reconstruction, str):
            self.reconstruction = str(self.reconstruction)

        if self.segmentation is not None and not isinstance(self.segmentation, str):
            self.segmentation = str(self.segmentation)

        if self.scale is not None and not isinstance(self.scale, str):
            self.scale = str(self.scale)

        if self.atlas is not None and not isinstance(self.atlas, str):
            self.atlas = str(self.atlas)

        if self.acquisition is not None and not isinstance(self.acquisition, str):
            self.acquisition = str(self.acquisition)

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
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    nodes: Optional[Union[Union[dict, "Node"], list[Union[dict, "Node"]]]] = empty_list()
    edges: Optional[Union[Union[dict, "Edge"], list[Union[dict, "Edge"]]]] = empty_list()
    coupling: Optional[Union[dict[Union[str, CouplingName], Union[dict, "Coupling"]], list[Union[dict, "Coupling"]]]] = empty_dict()
    dynamics: Optional[Union[dict[Union[str, DynamicsName], Union[dict, "Dynamics"]], list[Union[dict, "Dynamics"]]]] = empty_dict()
    number_of_nodes: Optional[int] = 1
    coordinate_space: Optional[Union[dict, "CommonCoordinateSpace"]] = None
    parcellation: Optional[Union[dict, Parcellation]] = None
    tractogram: Optional[Union[dict, Tractogram]] = None
    transforms: Optional[Union[dict[Union[str, FunctionName], Union[dict, "Function"]], list[Union[dict, "Function"]]]] = empty_dict()
    data_file: Optional[str] = None
    descriptor: Optional[str] = None
    bids_dir: Optional[str] = None
    bids: Optional[Union[dict, BidsEntities]] = None
    structural_measures: Optional[Union[str, list[str]]] = empty_list()
    observational_measures: Optional[Union[str, list[str]]] = empty_list()
    provenance: Optional[Union[dict, Provenance]] = None
    parent_network: Optional[str] = None
    node_mapping: Optional[str] = None
    distance_unit: Optional[Union[str, "UnitEnum"]] = 'mm'
    time_unit: Optional[Union[str, "UnitEnum"]] = 'ms'
    edge_matrix_files: Optional[Union[Union[str, FileName], list[Union[str, FileName]]]] = empty_list()
    graph_generator: Optional[Union[dict, "GraphGenerator"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        self._normalize_inlined_as_list(slot_name="nodes", slot_type=Node, key_name="id", keyed=False)

        if not isinstance(self.edges, list):
            self.edges = [self.edges] if self.edges is not None else []
        self.edges = [v if isinstance(v, Edge) else Edge(**as_dict(v)) for v in self.edges]

        self._normalize_inlined_as_dict(slot_name="coupling", slot_type=Coupling, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="dynamics", slot_type=Dynamics, key_name="name", keyed=True)

        if self.number_of_nodes is not None and not isinstance(self.number_of_nodes, int):
            self.number_of_nodes = int(self.number_of_nodes)

        if self.coordinate_space is not None and not isinstance(self.coordinate_space, CommonCoordinateSpace):
            self.coordinate_space = CommonCoordinateSpace(**as_dict(self.coordinate_space))

        if self.parcellation is not None and not isinstance(self.parcellation, Parcellation):
            self.parcellation = Parcellation(**as_dict(self.parcellation))

        if self.tractogram is not None and not isinstance(self.tractogram, Tractogram):
            self.tractogram = Tractogram(**as_dict(self.tractogram))

        self._normalize_inlined_as_list(slot_name="transforms", slot_type=Function, key_name="name", keyed=True)

        if self.data_file is not None and not isinstance(self.data_file, str):
            self.data_file = str(self.data_file)

        if self.descriptor is not None and not isinstance(self.descriptor, str):
            self.descriptor = str(self.descriptor)

        if self.bids_dir is not None and not isinstance(self.bids_dir, str):
            self.bids_dir = str(self.bids_dir)

        if self.bids is not None and not isinstance(self.bids, BidsEntities):
            self.bids = BidsEntities(**as_dict(self.bids))

        if not isinstance(self.structural_measures, list):
            self.structural_measures = [self.structural_measures] if self.structural_measures is not None else []
        self.structural_measures = [v if isinstance(v, str) else str(v) for v in self.structural_measures]

        if not isinstance(self.observational_measures, list):
            self.observational_measures = [self.observational_measures] if self.observational_measures is not None else []
        self.observational_measures = [v if isinstance(v, str) else str(v) for v in self.observational_measures]

        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            self.provenance = Provenance(**as_dict(self.provenance))

        if self.parent_network is not None and not isinstance(self.parent_network, str):
            self.parent_network = str(self.parent_network)

        if self.node_mapping is not None and not isinstance(self.node_mapping, str):
            self.node_mapping = str(self.node_mapping)

        if self.distance_unit is not None and not isinstance(self.distance_unit, UnitEnum):
            self.distance_unit = getattr(UnitEnum, self.distance_unit)

        if self.time_unit is not None and not isinstance(self.time_unit, UnitEnum):
            self.time_unit = getattr(UnitEnum, self.time_unit)

        if not isinstance(self.edge_matrix_files, list):
            self.edge_matrix_files = [self.edge_matrix_files] if self.edge_matrix_files is not None else []
        self.edge_matrix_files = [v if isinstance(v, FileName) else FileName(v) for v in self.edge_matrix_files]

        if self.graph_generator is not None and not isinstance(self.graph_generator, GraphGenerator):
            self.graph_generator = GraphGenerator(**as_dict(self.graph_generator))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class GraphGenerator(YAMLRoot):
    """
    Backend-agnostic graph generator specification. Captures the mathematical family (type) and its parameters so that
    each backend can emit the correct constructor call (Graphs.jl, NetworkX, etc.). The number of nodes is always
    taken from Network.number_of_nodes.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["GraphGenerator"]
    class_class_curie: ClassVar[str] = "tvbo:GraphGenerator"
    class_name: ClassVar[str] = "GraphGenerator"
    class_model_uri: ClassVar[URIRef] = TVBO.GraphGenerator

    name: Union[str, GraphGeneratorName] = None
    type: str = None
    description: Optional[str] = None
    seed: Optional[int] = None
    directed: Optional[Union[bool, Bool]] = False
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, GraphGeneratorName):
            self.name = GraphGeneratorName(self.name)

        if self._is_empty(self.type):
            self.MissingRequiredField("type")
        if not isinstance(self.type, str):
            self.type = str(self.type)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.seed is not None and not isinstance(self.seed, int):
            self.seed = int(self.seed)

        if self.directed is not None and not isinstance(self.directed, Bool):
            self.directed = Bool(self.directed)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class File(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["File"]
    class_class_curie: ClassVar[str] = "tvbo:File"
    class_name: ClassVar[str] = "File"
    class_model_uri: ClassVar[URIRef] = TVBO.File

    name: Union[str, FileName] = None
    description: Optional[str] = None
    type: Optional[str] = None
    path: Optional[str] = None
    extension: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, FileName):
            self.name = FileName(self.name)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.type is not None and not isinstance(self.type, str):
            self.type = str(self.type)

        if self.path is not None and not isinstance(self.path, str):
            self.path = str(self.path)

        if self.extension is not None and not isinstance(self.extension, str):
            self.extension = str(self.extension)

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
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    dynamics: Optional[Union[str, DynamicsName]] = None
    position: Optional[Union[dict, "Coordinate"]] = None
    region: Optional[str] = None
    state: Optional[Union[dict[Union[str, StateValueName], Union[dict, "StateValue"]], list[Union[dict, "StateValue"]]]] = empty_dict()
    events: Optional[Union[dict[Union[str, EventName], Union[dict, Event]], list[Union[dict, Event]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, int):
            self.id = int(self.id)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.dynamics is not None and not isinstance(self.dynamics, DynamicsName):
            self.dynamics = DynamicsName(self.dynamics)

        if self.position is not None and not isinstance(self.position, Coordinate):
            self.position = Coordinate(**as_dict(self.position))

        if self.region is not None and not isinstance(self.region, str):
            self.region = str(self.region)

        self._normalize_inlined_as_dict(slot_name="state", slot_type=StateValue, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="events", slot_type=Event, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class StateValue(YAMLRoot):
    """
    A named state variable value for per-node initialization.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["StateValue"]
    class_class_curie: ClassVar[str] = "tvbo:StateValue"
    class_name: ClassVar[str] = "StateValue"
    class_model_uri: ClassVar[URIRef] = TVBO.StateValue

    name: Union[str, StateValueName] = None
    value: Optional[float] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, StateValueName):
            self.name = StateValueName(self.name)

        if self.value is not None and not isinstance(self.value, float):
            self.value = float(self.value)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Edge(YAMLRoot):
    """
    An edge in a network. Two modes: explicit (source+target set, scalar parameters in YAML) or template (no
    source/target, N×N matrix measure in HDF5). Both coexist in the same edges list.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Edge"]
    class_class_curie: ClassVar[str] = "tvbo:Edge"
    class_name: ClassVar[str] = "Edge"
    class_model_uri: ClassVar[URIRef] = TVBO.Edge

    label: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    source: Optional[int] = None
    target: Optional[int] = None
    unit: Optional[str] = None
    format: Optional[Union[str, "SparseFormat"]] = None
    weighted: Optional[Union[bool, Bool]] = True
    valid_diagonal: Optional[Union[bool, Bool]] = False
    non_negative: Optional[Union[bool, Bool]] = True
    source_var: Optional[str] = None
    target_var: Optional[str] = None
    coupling: Optional[Union[str, CouplingName]] = None
    directed: Optional[Union[bool, Bool]] = False
    target_network: Optional[str] = None
    dimension_labels: Optional[Union[str, list[str]]] = empty_list()
    dynamics: Optional[Union[str, DynamicsName]] = None
    events: Optional[Union[dict[Union[str, EventName], Union[dict, Event]], list[Union[dict, Event]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.source is not None and not isinstance(self.source, int):
            self.source = int(self.source)

        if self.target is not None and not isinstance(self.target, int):
            self.target = int(self.target)

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        if self.format is not None and not isinstance(self.format, SparseFormat):
            self.format = SparseFormat(self.format)

        if self.weighted is not None and not isinstance(self.weighted, Bool):
            self.weighted = Bool(self.weighted)

        if self.valid_diagonal is not None and not isinstance(self.valid_diagonal, Bool):
            self.valid_diagonal = Bool(self.valid_diagonal)

        if self.non_negative is not None and not isinstance(self.non_negative, Bool):
            self.non_negative = Bool(self.non_negative)

        if self.source_var is not None and not isinstance(self.source_var, str):
            self.source_var = str(self.source_var)

        if self.target_var is not None and not isinstance(self.target_var, str):
            self.target_var = str(self.target_var)

        if self.coupling is not None and not isinstance(self.coupling, CouplingName):
            self.coupling = CouplingName(self.coupling)

        if self.directed is not None and not isinstance(self.directed, Bool):
            self.directed = Bool(self.directed)

        if self.target_network is not None and not isinstance(self.target_network, str):
            self.target_network = str(self.target_network)

        if not isinstance(self.dimension_labels, list):
            self.dimension_labels = [self.dimension_labels] if self.dimension_labels is not None else []
        self.dimension_labels = [v if isinstance(v, str) else str(v) for v in self.dimension_labels]

        if self.dynamics is not None and not isinstance(self.dynamics, DynamicsName):
            self.dynamics = DynamicsName(self.dynamics)

        self._normalize_inlined_as_dict(slot_name="events", slot_type=Event, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Observation(YAMLRoot):
    """
    Unified class for all observation/measurement specifications. Covers monitors (BOLD, EEG), tuning observables, and
    derived quantities. Pipeline is a sequence of Functions with input -> output flow.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Observation"]
    class_class_curie: ClassVar[str] = "tvbo:Observation"
    class_name: ClassVar[str] = "Observation"
    class_model_uri: ClassVar[URIRef] = TVBO.Observation

    name: Union[str, ObservationName] = None
    acronym: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    environment: Optional[Union[dict, "SoftwareEnvironment"]] = None
    time_scale: Optional[Union[str, "UnitEnum"]] = 'ms'
    source: Optional[Union[str, StateVariableName]] = None
    period: Optional[float] = None
    downsample_period: Optional[float] = None
    voi: Optional[int] = None
    imaging_modality: Optional[Union[str, "ImagingModality"]] = None
    warmup_source: Optional[str] = None
    data_source: Optional[Union[dict, "DataSource"]] = None
    skip_t: Optional[int] = None
    tail_samples: Optional[int] = None
    aggregation: Optional[Union[str, "AggregationType"]] = None
    window_size: Optional[int] = None
    pipeline: Optional[Union[Union[dict, "FunctionCall"], list[Union[dict, "FunctionCall"]]]] = empty_list()
    class_reference: Optional[Union[dict, "ClassReference"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, ObservationName):
            self.name = ObservationName(self.name)

        if self.acronym is not None and not isinstance(self.acronym, str):
            self.acronym = str(self.acronym)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.environment is not None and not isinstance(self.environment, SoftwareEnvironment):
            self.environment = SoftwareEnvironment(**as_dict(self.environment))

        if self.time_scale is not None and not isinstance(self.time_scale, UnitEnum):
            self.time_scale = getattr(UnitEnum, self.time_scale)

        if self.source is not None and not isinstance(self.source, StateVariableName):
            self.source = StateVariableName(self.source)

        if self.period is not None and not isinstance(self.period, float):
            self.period = float(self.period)

        if self.downsample_period is not None and not isinstance(self.downsample_period, float):
            self.downsample_period = float(self.downsample_period)

        if self.voi is not None and not isinstance(self.voi, int):
            self.voi = int(self.voi)

        if self.imaging_modality is not None and not isinstance(self.imaging_modality, ImagingModality):
            self.imaging_modality = ImagingModality(self.imaging_modality)

        if self.warmup_source is not None and not isinstance(self.warmup_source, str):
            self.warmup_source = str(self.warmup_source)

        if self.data_source is not None and not isinstance(self.data_source, DataSource):
            self.data_source = DataSource(**as_dict(self.data_source))

        if self.skip_t is not None and not isinstance(self.skip_t, int):
            self.skip_t = int(self.skip_t)

        if self.tail_samples is not None and not isinstance(self.tail_samples, int):
            self.tail_samples = int(self.tail_samples)

        if self.aggregation is not None and not isinstance(self.aggregation, AggregationType):
            self.aggregation = AggregationType(self.aggregation)

        if self.window_size is not None and not isinstance(self.window_size, int):
            self.window_size = int(self.window_size)

        if not isinstance(self.pipeline, list):
            self.pipeline = [self.pipeline] if self.pipeline is not None else []
        self.pipeline = [v if isinstance(v, FunctionCall) else FunctionCall(**as_dict(v)) for v in self.pipeline]

        if self.class_reference is not None and not isinstance(self.class_reference, ClassReference):
            self.class_reference = ClassReference(**as_dict(self.class_reference))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class DerivedObservation(Observation):
    """
    Observation derived from one or more other observations. Examples: - fc (from bold) - single source transformation
    - fc_corr (from fc and fc_target) - multi-source comparison Unlike regular Observations, these don't generate
    monitor classes but are computed from existing observation values.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["DerivedObservation"]
    class_class_curie: ClassVar[str] = "tvbo:DerivedObservation"
    class_name: ClassVar[str] = "DerivedObservation"
    class_model_uri: ClassVar[URIRef] = TVBO.DerivedObservation

    name: Union[str, DerivedObservationName] = None
    source_observations: Union[Union[str, ObservationName], list[Union[str, ObservationName]]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, DerivedObservationName):
            self.name = DerivedObservationName(self.name)

        if self._is_empty(self.source_observations):
            self.MissingRequiredField("source_observations")
        if not isinstance(self.source_observations, list):
            self.source_observations = [self.source_observations] if self.source_observations is not None else []
        self.source_observations = [v if isinstance(v, ObservationName) else ObservationName(v) for v in self.source_observations]

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
    output: Optional[Union[str, list[str]]] = empty_list()
    derived_from_model: Optional[Union[str, DynamicsName]] = None
    number_of_modes: Optional[int] = 1
    local_coupling_term: Optional[Union[str, ParameterName]] = None
    functions: Optional[Union[dict[Union[str, FunctionName], Union[dict, "Function"]], list[Union[dict, "Function"]]]] = empty_dict()
    stimulus: Optional[Union[dict, Stimulus]] = None
    modes: Optional[Union[dict[Union[str, DynamicsName], Union[dict, "Dynamics"]], list[Union[dict, "Dynamics"]]]] = empty_dict()
    model_type: Optional[Union[str, "ModelType"]] = None
    system_type: Optional[Union[str, "SystemType"]] = None
    autonomous: Optional[Union[bool, Bool]] = True
    observed: Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, "DerivedVariable"]], list[Union[dict, "DerivedVariable"]]]] = empty_dict()
    events: Optional[Union[dict[Union[str, EventName], Union[dict, Event]], list[Union[dict, Event]]]] = empty_dict()

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

        if not isinstance(self.output, list):
            self.output = [self.output] if self.output is not None else []
        self.output = [v if isinstance(v, str) else str(v) for v in self.output]

        if self.derived_from_model is not None and not isinstance(self.derived_from_model, DynamicsName):
            self.derived_from_model = DynamicsName(self.derived_from_model)

        if self.number_of_modes is not None and not isinstance(self.number_of_modes, int):
            self.number_of_modes = int(self.number_of_modes)

        if self.local_coupling_term is not None and not isinstance(self.local_coupling_term, ParameterName):
            self.local_coupling_term = ParameterName(self.local_coupling_term)

        self._normalize_inlined_as_dict(slot_name="functions", slot_type=Function, key_name="name", keyed=True)

        if self.stimulus is not None and not isinstance(self.stimulus, Stimulus):
            self.stimulus = Stimulus(**as_dict(self.stimulus))

        self._normalize_inlined_as_dict(slot_name="modes", slot_type=Dynamics, key_name="name", keyed=True)

        if self.model_type is not None and not isinstance(self.model_type, ModelType):
            self.model_type = ModelType(self.model_type)

        if self.system_type is not None and not isinstance(self.system_type, SystemType):
            self.system_type = SystemType(self.system_type)

        if self.autonomous is not None and not isinstance(self.autonomous, Bool):
            self.autonomous = Bool(self.autonomous)

        self._normalize_inlined_as_dict(slot_name="observed", slot_type=DerivedVariable, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="events", slot_type=Event, key_name="name", keyed=True)

        if self.system_type is not None and not isinstance(self.system_type, str):
            self.system_type = str(self.system_type)

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
    unit: Optional[Union[str, "UnitEnum"]] = None
    variable_of_interest: Optional[Union[bool, Bool]] = True
    coupling_variable: Optional[Union[bool, Bool]] = False
    equation_type: Optional[str] = "differential"
    equation_order: Optional[int] = 1
    noise: Optional[Union[dict, "Noise"]] = None
    stimulation_variable: Optional[Union[bool, Bool]] = None
    boundaries: Optional[Union[dict, Range]] = None
    initial_value: Optional[float] = 0.1
    derivative_initial_value: Optional[float] = None
    distribution: Optional[Union[dict, "Distribution"]] = None
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

        if self.unit is not None and not isinstance(self.unit, UnitEnum):
            self.unit = UnitEnum(self.unit)

        if self.variable_of_interest is not None and not isinstance(self.variable_of_interest, Bool):
            self.variable_of_interest = Bool(self.variable_of_interest)

        if self.coupling_variable is not None and not isinstance(self.coupling_variable, Bool):
            self.coupling_variable = Bool(self.coupling_variable)

        if self.equation_type is not None and not isinstance(self.equation_type, str):
            self.equation_type = str(self.equation_type)

        if self.equation_order is not None and not isinstance(self.equation_order, int):
            self.equation_order = int(self.equation_order)

        if self.noise is not None and not isinstance(self.noise, Noise):
            self.noise = Noise(**as_dict(self.noise))

        if self.stimulation_variable is not None and not isinstance(self.stimulation_variable, Bool):
            self.stimulation_variable = Bool(self.stimulation_variable)

        if self.boundaries is not None and not isinstance(self.boundaries, Range):
            self.boundaries = Range(**as_dict(self.boundaries))

        if self.initial_value is not None and not isinstance(self.initial_value, float):
            self.initial_value = float(self.initial_value)

        if self.derivative_initial_value is not None and not isinstance(self.derivative_initial_value, float):
            self.derivative_initial_value = float(self.derivative_initial_value)

        if self.distribution is not None and not isinstance(self.distribution, Distribution):
            self.distribution = Distribution(**as_dict(self.distribution))

        if self.history is not None and not isinstance(self.history, TimeSeries):
            self.history = TimeSeries(**as_dict(self.history))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Distribution(YAMLRoot):
    """
    A probability distribution for sampling parameters or initial conditions. Standard distributions (Uniform,
    Gaussian) are specified by name and domain/parameters. Custom distributions use a Function for the PDF/sampling
    rule. Default name is Uniform when only domain is given.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Distribution"]
    class_class_curie: ClassVar[str] = "tvbo:Distribution"
    class_name: ClassVar[str] = "Distribution"
    class_model_uri: ClassVar[URIRef] = TVBO.Distribution

    name: Union[str, DistributionName] = "Uniform"
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, "Parameter"]], list[Union[dict, "Parameter"]]]] = empty_dict()
    domain: Optional[Union[dict, Range]] = None
    function: Optional[Union[dict, "Function"]] = None
    seed: Optional[int] = None
    axis: Optional[Union[str, "SamplingAxis"]] = 'space'
    correlation: Optional[Union[dict, Matrix]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, DistributionName):
            self.name = DistributionName(self.name)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.domain is not None and not isinstance(self.domain, Range):
            self.domain = Range(**as_dict(self.domain))

        if self.function is not None and not isinstance(self.function, Function):
            self.function = Function(**as_dict(self.function))

        if self.seed is not None and not isinstance(self.seed, int):
            self.seed = int(self.seed)

        if self.axis is not None and not isinstance(self.axis, SamplingAxis):
            self.axis = getattr(SamplingAxis, self.axis)

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
    unit: Optional[Union[str, "UnitEnum"]] = None
    dataset_path: Optional[str] = None
    comment: Optional[str] = None
    heterogeneous: Optional[Union[bool, Bool]] = None
    distribution: Optional[Union[dict, Distribution]] = None
    free: Optional[Union[bool, Bool]] = None
    shape: Optional[str] = None
    explored_values: Optional[Union[float, list[float]]] = empty_list()
    element_domains: Optional[Union[Union[dict, Range], list[Union[dict, Range]]]] = empty_list()

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

        if self.unit is not None and not isinstance(self.unit, UnitEnum):
            self.unit = UnitEnum(self.unit)

        if self.dataset_path is not None and not isinstance(self.dataset_path, str):
            self.dataset_path = str(self.dataset_path)

        if self.comment is not None and not isinstance(self.comment, str):
            self.comment = str(self.comment)

        if self.heterogeneous is not None and not isinstance(self.heterogeneous, Bool):
            self.heterogeneous = Bool(self.heterogeneous)

        if self.distribution is not None and not isinstance(self.distribution, Distribution):
            self.distribution = Distribution(**as_dict(self.distribution))

        if self.free is not None and not isinstance(self.free, Bool):
            self.free = Bool(self.free)

        if self.shape is not None and not isinstance(self.shape, str):
            self.shape = str(self.shape)

        if not isinstance(self.explored_values, list):
            self.explored_values = [self.explored_values] if self.explored_values is not None else []
        self.explored_values = [v if isinstance(v, float) else float(v) for v in self.explored_values]

        if not isinstance(self.element_domains, list):
            self.element_domains = [self.element_domains] if self.element_domains is not None else []
        self.element_domains = [v if isinstance(v, Range) else Range(**as_dict(v)) for v in self.element_domains]

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
    source: Optional[str] = None
    dimension: Optional[int] = 1
    keys: Optional[Union[str, list[str]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, CouplingInputName):
            self.name = CouplingInputName(self.name)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.source is not None and not isinstance(self.source, str):
            self.source = str(self.source)

        if self.dimension is not None and not isinstance(self.dimension, int):
            self.dimension = int(self.dimension)

        if not isinstance(self.keys, list):
            self.keys = [self.keys] if self.keys is not None else []
        self.keys = [v if isinstance(v, str) else str(v) for v in self.keys]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Argument(YAMLRoot):
    """
    A function argument with explicit value specification. Value can be: literal (number/string), reference to input
    (input.key), or cross-observation reference (observation_name.output_key).
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Argument"]
    class_class_curie: ClassVar[str] = "tvbo:Argument"
    class_name: ClassVar[str] = "Argument"
    class_model_uri: ClassVar[URIRef] = TVBO.Argument

    name: Union[str, ArgumentName] = None
    description: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, ArgumentName):
            self.name = ArgumentName(self.name)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.value is not None and not isinstance(self.value, str):
            self.value = str(self.value)

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Function(YAMLRoot):
    """
    A function with explicit input -> transformation -> output flow. Can be equation-based (symbolic) or
    software-based (callable). In a pipeline, functions are chained: output of one becomes input of next.
    """
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
    input: Optional[Union[str, FunctionName]] = None
    output: Optional[str] = None
    iri: Optional[str] = None
    arguments: Optional[Union[dict[Union[str, ArgumentName], Union[dict, Argument]], list[Union[dict, Argument]]]] = empty_dict()
    output_equation: Optional[Union[dict, Equation]] = None
    source_code: Optional[str] = None
    callable: Optional[Union[dict, "Callable"]] = None
    apply_on_dimension: Optional[Union[str, "DimensionType"]] = None
    aggregate: Optional[Union[dict, "Aggregation"]] = None
    time_range: Optional[Union[dict, Range]] = None

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

        if self.input is not None and not isinstance(self.input, FunctionName):
            self.input = FunctionName(self.input)

        if self.output is not None and not isinstance(self.output, str):
            self.output = str(self.output)

        if self.iri is not None and not isinstance(self.iri, str):
            self.iri = str(self.iri)

        self._normalize_inlined_as_list(slot_name="arguments", slot_type=Argument, key_name="name", keyed=True)

        if self.output_equation is not None and not isinstance(self.output_equation, Equation):
            self.output_equation = Equation(**as_dict(self.output_equation))

        if self.source_code is not None and not isinstance(self.source_code, str):
            self.source_code = str(self.source_code)

        if self.callable is not None and not isinstance(self.callable, Callable):
            self.callable = Callable(**as_dict(self.callable))

        if self.apply_on_dimension is not None and not isinstance(self.apply_on_dimension, DimensionType):
            self.apply_on_dimension = DimensionType(self.apply_on_dimension)

        if self.aggregate is not None and not isinstance(self.aggregate, Aggregation):
            self.aggregate = Aggregation(**as_dict(self.aggregate))

        if self.time_range is not None and not isinstance(self.time_range, Range):
            self.time_range = Range(**as_dict(self.time_range))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Aggregation(YAMLRoot):
    """
    Specifies how to aggregate values across a dimension. Used for loss functions to define per-element loss with
    reduction.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Aggregation"]
    class_class_curie: ClassVar[str] = "tvbo:Aggregation"
    class_name: ClassVar[str] = "Aggregation"
    class_model_uri: ClassVar[URIRef] = TVBO.Aggregation

    over: Optional[Union[str, "DimensionType"]] = None
    type: Optional[Union[str, "ReductionType"]] = 'mean'

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.over is not None and not isinstance(self.over, DimensionType):
            self.over = DimensionType(self.over)

        if self.type is not None and not isinstance(self.type, ReductionType):
            self.type = getattr(ReductionType, self.type)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class LossFunction(Function):
    """
    A loss function for optimization with optional aggregation. Extends Function with aggregation specification for
    per-element losses.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["LossFunction"]
    class_class_curie: ClassVar[str] = "tvbo:LossFunction"
    class_name: ClassVar[str] = "LossFunction"
    class_model_uri: ClassVar[URIRef] = TVBO.LossFunction

    name: Union[str, LossFunctionName] = None
    aggregate: Optional[Union[dict, Aggregation]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, LossFunctionName):
            self.name = LossFunctionName(self.name)

        if self.aggregate is not None and not isinstance(self.aggregate, Aggregation):
            self.aggregate = Aggregation(**as_dict(self.aggregate))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class FunctionCall(YAMLRoot):
    """
    Invocation of a function in a pipeline. Can reference a defined Function by name, OR inline a callable directly
    for external library functions, OR inline an equation, OR use class_call for class instantiation. Mirrors Function
    attributes so pipeline steps can be self-contained.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["FunctionCall"]
    class_class_curie: ClassVar[str] = "tvbo:FunctionCall"
    class_name: ClassVar[str] = "FunctionCall"
    class_model_uri: ClassVar[URIRef] = TVBO.FunctionCall

    acronym: Optional[str] = None
    label: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    description: Optional[str] = None
    name: Optional[str] = None
    function: Optional[Union[str, FunctionName]] = None
    callable: Optional[Union[dict, "Callable"]] = None
    class_call: Optional[Union[dict, "ClassReference"]] = None
    input: Optional[str] = None
    output: Optional[str] = None
    apply_on_dimension: Optional[Union[str, "DimensionType"]] = None
    aggregate: Optional[Union[dict, Aggregation]] = None
    arguments: Optional[Union[dict[Union[str, ArgumentName], Union[dict, Argument]], list[Union[dict, Argument]]]] = empty_dict()
    time_range: Optional[Union[dict, Range]] = None
    source_code: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.acronym is not None and not isinstance(self.acronym, str):
            self.acronym = str(self.acronym)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.name is not None and not isinstance(self.name, str):
            self.name = str(self.name)

        if self.function is not None and not isinstance(self.function, FunctionName):
            self.function = FunctionName(self.function)

        if self.callable is not None and not isinstance(self.callable, Callable):
            self.callable = Callable(**as_dict(self.callable))

        if self.class_call is not None and not isinstance(self.class_call, ClassReference):
            self.class_call = ClassReference(**as_dict(self.class_call))

        if self.input is not None and not isinstance(self.input, str):
            self.input = str(self.input)

        if self.output is not None and not isinstance(self.output, str):
            self.output = str(self.output)

        if self.apply_on_dimension is not None and not isinstance(self.apply_on_dimension, DimensionType):
            self.apply_on_dimension = DimensionType(self.apply_on_dimension)

        if self.aggregate is not None and not isinstance(self.aggregate, Aggregation):
            self.aggregate = Aggregation(**as_dict(self.aggregate))

        self._normalize_inlined_as_list(slot_name="arguments", slot_type=Argument, key_name="name", keyed=True)

        if self.time_range is not None and not isinstance(self.time_range, Range):
            self.time_range = Range(**as_dict(self.time_range))

        if self.source_code is not None and not isinstance(self.source_code, str):
            self.source_code = str(self.source_code)

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

        if self.software is not None and not isinstance(self.software, SoftwareRequirement):
            self.software = SoftwareRequirement(**as_dict(self.software))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ClassReference(Callable):
    """
    Reference to a class that can be instantiated and called. Used for external library classes (e.g., tvboptim.Bold,
    custom monitors). The class is instantiated with constructor_args, then called with call_args. Generalizable
    pattern: works for tvboptim, TVB, or any Python class.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["ClassReference"]
    class_class_curie: ClassVar[str] = "tvbo:ClassReference"
    class_name: ClassVar[str] = "ClassReference"
    class_model_uri: ClassVar[URIRef] = TVBO.ClassReference

    name: Union[str, ClassReferenceName] = None
    constructor_args: Optional[Union[dict[Union[str, ArgumentName], Union[dict, Argument]], list[Union[dict, Argument]]]] = empty_dict()
    call_args: Optional[Union[dict[Union[str, ArgumentName], Union[dict, Argument]], list[Union[dict, Argument]]]] = empty_dict()
    warmup_source: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, ClassReferenceName):
            self.name = ClassReferenceName(self.name)

        self._normalize_inlined_as_list(slot_name="constructor_args", slot_type=Argument, key_name="name", keyed=True)

        self._normalize_inlined_as_list(slot_name="call_args", slot_type=Argument, key_name="name", keyed=True)

        if self.warmup_source is not None and not isinstance(self.warmup_source, str):
            self.warmup_source = str(self.warmup_source)

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
    unit: Optional[Union[str, "UnitEnum"]] = None

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

        if self.unit is not None and not isinstance(self.unit, UnitEnum):
            self.unit = UnitEnum(self.unit)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class DerivedVariable(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["DerivedVariable"]
    class_class_curie: ClassVar[str] = "tvbo:DerivedVariable"
    class_name: ClassVar[str] = "DerivedVariable"
    class_model_uri: ClassVar[URIRef] = TVBO.DerivedVariable

    name: Union[str, DerivedVariableName] = None
    label: Optional[str] = None
    symbol: Optional[str] = None
    description: Optional[str] = None
    equation: Optional[Union[dict, Equation]] = None
    unit: Optional[Union[str, "UnitEnum"]] = None
    conditional: Optional[Union[bool, Bool]] = False
    cases: Optional[Union[Union[dict, Case], list[Union[dict, Case]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, DerivedVariableName):
            self.name = DerivedVariableName(self.name)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.symbol is not None and not isinstance(self.symbol, str):
            self.symbol = str(self.symbol)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.equation is not None and not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.unit is not None and not isinstance(self.unit, UnitEnum):
            self.unit = UnitEnum(self.unit)

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
class DataSource(YAMLRoot):
    """
    Specification for loading external/empirical data.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["DataSource"]
    class_class_curie: ClassVar[str] = "tvbo:DataSource"
    class_name: ClassVar[str] = "DataSource"
    class_model_uri: ClassVar[URIRef] = TVBO.DataSource

    name: Union[str, DataSourceName] = None
    label: Optional[str] = None
    description: Optional[str] = None
    path: Optional[str] = None
    loader: Optional[Union[dict, Callable]] = None
    format: Optional[str] = None
    key: Optional[str] = None
    preprocessing: Optional[Union[dict, Function]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, DataSourceName):
            self.name = DataSourceName(self.name)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.path is not None and not isinstance(self.path, str):
            self.path = str(self.path)

        if self.loader is not None and not isinstance(self.loader, Callable):
            self.loader = Callable(**as_dict(self.loader))

        if self.format is not None and not isinstance(self.format, str):
            self.format = str(self.format)

        if self.key is not None and not isinstance(self.key, str):
            self.key = str(self.key)

        if self.preprocessing is not None and not isinstance(self.preprocessing, Function):
            self.preprocessing = Function(**as_dict(self.preprocessing))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class OptimizationStage(YAMLRoot):
    """
    A single stage in a multi-stage optimization workflow. Stages run sequentially, with each stage potentially using
    different parameters, shapes, learning rates, and algorithms.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["OptimizationStage"]
    class_class_curie: ClassVar[str] = "tvbo:OptimizationStage"
    class_name: ClassVar[str] = "OptimizationStage"
    class_model_uri: ClassVar[URIRef] = TVBO.OptimizationStage

    name: Union[str, OptimizationStageName] = None
    label: Optional[str] = None
    description: Optional[str] = None
    free_parameters: Optional[Union[Union[str, ParameterName], list[Union[str, ParameterName]]]] = empty_list()
    algorithm: Optional[str] = "adam"
    learning_rate: Optional[float] = 0.001
    max_iterations: Optional[int] = 100
    hyperparameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    freeze_parameters: Optional[Union[Union[str, ParameterName], list[Union[str, ParameterName]]]] = empty_list()
    warmup_from: Optional[Union[str, OptimizationStageName]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, OptimizationStageName):
            self.name = OptimizationStageName(self.name)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if not isinstance(self.free_parameters, list):
            self.free_parameters = [self.free_parameters] if self.free_parameters is not None else []
        self.free_parameters = [v if isinstance(v, ParameterName) else ParameterName(v) for v in self.free_parameters]

        if self.algorithm is not None and not isinstance(self.algorithm, str):
            self.algorithm = str(self.algorithm)

        if self.learning_rate is not None and not isinstance(self.learning_rate, float):
            self.learning_rate = float(self.learning_rate)

        if self.max_iterations is not None and not isinstance(self.max_iterations, int):
            self.max_iterations = int(self.max_iterations)

        self._normalize_inlined_as_list(slot_name="hyperparameters", slot_type=Parameter, key_name="name", keyed=True)

        if not isinstance(self.freeze_parameters, list):
            self.freeze_parameters = [self.freeze_parameters] if self.freeze_parameters is not None else []
        self.freeze_parameters = [v if isinstance(v, ParameterName) else ParameterName(v) for v in self.freeze_parameters]

        if self.warmup_from is not None and not isinstance(self.warmup_from, OptimizationStageName):
            self.warmup_from = OptimizationStageName(self.warmup_from)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Optimization(OptimizationStage):
    """
    Configuration for parameter optimization. Inherits single-stage fields from OptimizationStage. For multi-stage
    workflows, use 'stages' (ignores inherited single-stage fields). Loss equation references observations directly by
    name.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Optimization"]
    class_class_curie: ClassVar[str] = "tvbo:Optimization"
    class_name: ClassVar[str] = "Optimization"
    class_model_uri: ClassVar[URIRef] = TVBO.Optimization

    name: Union[str, OptimizationName] = None
    execution: Optional[Union[dict, "ExecutionConfig"]] = None
    integration: Optional[Union[dict, "Integrator"]] = None
    loss: Optional[Union[dict, FunctionCall]] = None
    stages: Optional[Union[dict[Union[str, OptimizationStageName], Union[dict, OptimizationStage]], list[Union[dict, OptimizationStage]]]] = empty_dict()
    depends_on: Optional[Union[str, AlgorithmName]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, OptimizationName):
            self.name = OptimizationName(self.name)

        if self.execution is not None and not isinstance(self.execution, ExecutionConfig):
            self.execution = ExecutionConfig(**as_dict(self.execution))

        if self.integration is not None and not isinstance(self.integration, Integrator):
            self.integration = Integrator(**as_dict(self.integration))

        if self.loss is not None and not isinstance(self.loss, FunctionCall):
            self.loss = FunctionCall(**as_dict(self.loss))

        self._normalize_inlined_as_list(slot_name="stages", slot_type=OptimizationStage, key_name="name", keyed=True)

        if self.depends_on is not None and not isinstance(self.depends_on, AlgorithmName):
            self.depends_on = AlgorithmName(self.depends_on)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Exploration(YAMLRoot):
    """
    Parameter space exploration (grid search, sweep).
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Exploration"]
    class_class_curie: ClassVar[str] = "tvbo:Exploration"
    class_name: ClassVar[str] = "Exploration"
    class_model_uri: ClassVar[URIRef] = TVBO.Exploration

    name: Union[str, ExplorationName] = None
    label: Optional[str] = None
    description: Optional[str] = None
    execution: Optional[Union[dict, "ExecutionConfig"]] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    mode: Optional[str] = "product"
    observable: Optional[Union[dict, FunctionCall]] = None
    n_parallel: Optional[int] = 1
    n_trials: Optional[int] = 1
    average: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, ExplorationName):
            self.name = ExplorationName(self.name)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.execution is not None and not isinstance(self.execution, ExecutionConfig):
            self.execution = ExecutionConfig(**as_dict(self.execution))

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.mode is not None and not isinstance(self.mode, str):
            self.mode = str(self.mode)

        if self.observable is not None and not isinstance(self.observable, FunctionCall):
            self.observable = FunctionCall(**as_dict(self.observable))

        if self.n_parallel is not None and not isinstance(self.n_parallel, int):
            self.n_parallel = int(self.n_parallel)

        if self.n_trials is not None and not isinstance(self.n_trials, int):
            self.n_trials = int(self.n_trials)

        if self.average is not None and not isinstance(self.average, str):
            self.average = str(self.average)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class UpdateRule(YAMLRoot):
    """
    Defines how a parameter is updated based on observables. Represents iterative learning rules like FIC or EIB
    updates. Functions from experiment.functions are available in the equation.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["UpdateRule"]
    class_class_curie: ClassVar[str] = "tvbo:UpdateRule"
    class_name: ClassVar[str] = "UpdateRule"
    class_model_uri: ClassVar[URIRef] = TVBO.UpdateRule

    name: Union[str, UpdateRuleName] = None
    target_parameter: Union[dict, Parameter] = None
    equation: Union[dict, Equation] = None
    description: Optional[str] = None
    bounds: Optional[Union[dict, Range]] = None
    warmup: Optional[Union[bool, Bool]] = None
    requires: Optional[Union[Union[str, ObservationName], list[Union[str, ObservationName]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, UpdateRuleName):
            self.name = UpdateRuleName(self.name)

        if self._is_empty(self.target_parameter):
            self.MissingRequiredField("target_parameter")
        if not isinstance(self.target_parameter, Parameter):
            self.target_parameter = Parameter(**as_dict(self.target_parameter))

        if self._is_empty(self.equation):
            self.MissingRequiredField("equation")
        if not isinstance(self.equation, Equation):
            self.equation = Equation(**as_dict(self.equation))

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.bounds is not None and not isinstance(self.bounds, Range):
            self.bounds = Range(**as_dict(self.bounds))

        if self.warmup is not None and not isinstance(self.warmup, Bool):
            self.warmup = Bool(self.warmup)

        if not isinstance(self.requires, list):
            self.requires = [self.requires] if self.requires is not None else []
        self.requires = [v if isinstance(v, ObservationName) else ObservationName(v) for v in self.requires]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class AlgorithmInclude(YAMLRoot):
    """
    Reference to an included algorithm with optional argument overrides. Allows combining algorithms with different
    hyperparameter values.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["AlgorithmInclude"]
    class_class_curie: ClassVar[str] = "tvbo:AlgorithmInclude"
    class_name: ClassVar[str] = "AlgorithmInclude"
    class_model_uri: ClassVar[URIRef] = TVBO.AlgorithmInclude

    algorithm: Union[str, AlgorithmName] = None
    arguments: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.algorithm):
            self.MissingRequiredField("algorithm")
        if not isinstance(self.algorithm, AlgorithmName):
            self.algorithm = AlgorithmName(self.algorithm)

        self._normalize_inlined_as_dict(slot_name="arguments", slot_type=Parameter, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class TuningObjective(YAMLRoot):
    """
    Defines what the tuning algorithm optimizes for. Can be an activity target (FIC) or a connectivity target (EIB).
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["TuningObjective"]
    class_class_curie: ClassVar[str] = "tvbo:TuningObjective"
    class_name: ClassVar[str] = "TuningObjective"
    class_model_uri: ClassVar[URIRef] = TVBO.TuningObjective

    label: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    target_variable: Optional[Union[str, StateVariableName]] = None
    target_value: Optional[float] = None
    target_data: Optional[Union[str, ObservationName]] = None
    metric: Optional[Union[dict, Equation]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.type is not None and not isinstance(self.type, str):
            self.type = str(self.type)

        if self.target_variable is not None and not isinstance(self.target_variable, StateVariableName):
            self.target_variable = StateVariableName(self.target_variable)

        if self.target_value is not None and not isinstance(self.target_value, float):
            self.target_value = float(self.target_value)

        if self.target_data is not None and not isinstance(self.target_data, ObservationName):
            self.target_data = ObservationName(self.target_data)

        if self.metric is not None and not isinstance(self.metric, Equation):
            self.metric = Equation(**as_dict(self.metric))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Algorithm(YAMLRoot):
    """
    A complete specification of an iterative parameter tuning algorithm. Combines update rules, objectives,
    observations, and hyperparameters.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Algorithm"]
    class_class_curie: ClassVar[str] = "tvbo:Algorithm"
    class_name: ClassVar[str] = "Algorithm"
    class_model_uri: ClassVar[URIRef] = TVBO.Algorithm

    name: Union[str, AlgorithmName] = None
    description: Optional[str] = None
    execution: Optional[Union[dict, "ExecutionConfig"]] = None
    type: Optional[str] = None
    includes: Optional[Union[Union[dict, AlgorithmInclude], list[Union[dict, AlgorithmInclude]]]] = empty_list()
    objective: Optional[Union[dict, TuningObjective]] = None
    observations: Optional[Union[Union[str, ObservationName], list[Union[str, ObservationName]]]] = empty_list()
    update_rules: Optional[Union[dict[Union[str, UpdateRuleName], Union[dict, UpdateRule]], list[Union[dict, UpdateRule]]]] = empty_dict()
    hyperparameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    learning_rate: Optional[float] = None
    learning_rate_warmup: Optional[Union[bool, Bool]] = False
    n_iterations: Optional[int] = None
    learning_rate_schedule: Optional[str] = None
    simulation_period: Optional[float] = None
    apply_every: Optional[int] = 1
    functions: Optional[Union[Union[dict, FunctionCall], list[Union[dict, FunctionCall]]]] = empty_list()
    depends_on: Optional[Union[Union[str, AlgorithmName], list[Union[str, AlgorithmName]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, AlgorithmName):
            self.name = AlgorithmName(self.name)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.execution is not None and not isinstance(self.execution, ExecutionConfig):
            self.execution = ExecutionConfig(**as_dict(self.execution))

        if self.type is not None and not isinstance(self.type, str):
            self.type = str(self.type)

        if not isinstance(self.includes, list):
            self.includes = [self.includes] if self.includes is not None else []
        self.includes = [v if isinstance(v, AlgorithmInclude) else AlgorithmInclude(**as_dict(v)) for v in self.includes]

        if self.objective is not None and not isinstance(self.objective, TuningObjective):
            self.objective = TuningObjective(**as_dict(self.objective))

        if not isinstance(self.observations, list):
            self.observations = [self.observations] if self.observations is not None else []
        self.observations = [v if isinstance(v, ObservationName) else ObservationName(v) for v in self.observations]

        self._normalize_inlined_as_dict(slot_name="update_rules", slot_type=UpdateRule, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="hyperparameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.learning_rate is not None and not isinstance(self.learning_rate, float):
            self.learning_rate = float(self.learning_rate)

        if self.learning_rate_warmup is not None and not isinstance(self.learning_rate_warmup, Bool):
            self.learning_rate_warmup = Bool(self.learning_rate_warmup)

        if self.n_iterations is not None and not isinstance(self.n_iterations, int):
            self.n_iterations = int(self.n_iterations)

        if self.learning_rate_schedule is not None and not isinstance(self.learning_rate_schedule, str):
            self.learning_rate_schedule = str(self.learning_rate_schedule)

        if self.simulation_period is not None and not isinstance(self.simulation_period, float):
            self.simulation_period = float(self.simulation_period)

        if self.apply_every is not None and not isinstance(self.apply_every, int):
            self.apply_every = int(self.apply_every)

        if not isinstance(self.functions, list):
            self.functions = [self.functions] if self.functions is not None else []
        self.functions = [v if isinstance(v, FunctionCall) else FunctionCall(**as_dict(v)) for v in self.functions]

        if not isinstance(self.depends_on, list):
            self.depends_on = [self.depends_on] if self.depends_on is not None else []
        self.depends_on = [v if isinstance(v, AlgorithmName) else AlgorithmName(v) for v in self.depends_on]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Option(YAMLRoot):
    """
    A toolkit-specific key-value option (string name + string value). Used for backend settings that are not universal
    numeric parameters (e.g., solver name, tangent method, jacobian type).
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Option"]
    class_class_curie: ClassVar[str] = "tvbo:Option"
    class_name: ClassVar[str] = "Option"
    class_model_uri: ClassVar[URIRef] = TVBO.Option

    name: Union[str, OptionName] = None
    value: str = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, OptionName):
            self.name = OptionName(self.name)

        if self._is_empty(self.value):
            self.MissingRequiredField("value")
        if not isinstance(self.value, str):
            self.value = str(self.value)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Discretization(YAMLRoot):
    """
    Discretization method for boundary value problems in continuation (periodic orbits, connecting orbits,
    quasi-periodic tori). Specifies the method; method-specific numerics go in parameters.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Discretization"]
    class_class_curie: ClassVar[str] = "tvbo:Discretization"
    class_name: ClassVar[str] = "Discretization"
    class_model_uri: ClassVar[URIRef] = TVBO.Discretization

    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    method: Optional[Union[str, "NumericalDiscretizationMethod"]] = 'collocation'
    ode_solver: Optional[Union[dict, "Solver"]] = None
    linear_solver: Optional[Union[dict, "Solver"]] = None
    mesh_intervals: Optional[int] = 50
    degree: Optional[int] = 4
    n_sections: Optional[int] = 3
    options: Optional[Union[dict[Union[str, OptionName], Union[dict, Option]], list[Union[dict, Option]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.method is not None and not isinstance(self.method, NumericalDiscretizationMethod):
            self.method = getattr(NumericalDiscretizationMethod, self.method)

        if self.ode_solver is not None and not isinstance(self.ode_solver, Solver):
            self.ode_solver = Solver(**as_dict(self.ode_solver))

        if self.linear_solver is not None and not isinstance(self.linear_solver, Solver):
            self.linear_solver = Solver(**as_dict(self.linear_solver))

        if self.mesh_intervals is not None and not isinstance(self.mesh_intervals, int):
            self.mesh_intervals = int(self.mesh_intervals)

        if self.degree is not None and not isinstance(self.degree, int):
            self.degree = int(self.degree)

        if self.n_sections is not None and not isinstance(self.n_sections, int):
            self.n_sections = int(self.n_sections)

        self._normalize_inlined_as_dict(slot_name="options", slot_type=Option, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class InitialState(YAMLRoot):
    """
    How to obtain the starting equilibrium or periodic orbit for continuation. Most robust: time-integrate to steady
    state.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["InitialState"]
    class_class_curie: ClassVar[str] = "tvbo:InitialState"
    class_name: ClassVar[str] = "InitialState"
    class_model_uri: ClassVar[URIRef] = TVBO.InitialState

    method: Optional[Union[str, "InitialStateMethod"]] = 'time_integration'
    duration: Optional[float] = 2000.0
    abs_tol: Optional[float] = 1e-10
    rel_tol: Optional[float] = 1e-10
    solver: Optional[Union[dict, "Solver"]] = None
    source_branch: Optional[str] = None
    source_point: Optional[str] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.method is not None and not isinstance(self.method, InitialStateMethod):
            self.method = getattr(InitialStateMethod, self.method)

        if self.duration is not None and not isinstance(self.duration, float):
            self.duration = float(self.duration)

        if self.abs_tol is not None and not isinstance(self.abs_tol, float):
            self.abs_tol = float(self.abs_tol)

        if self.rel_tol is not None and not isinstance(self.rel_tol, float):
            self.rel_tol = float(self.rel_tol)

        if self.solver is not None and not isinstance(self.solver, Solver):
            self.solver = Solver(**as_dict(self.solver))

        if self.source_branch is not None and not isinstance(self.source_branch, str):
            self.source_branch = str(self.source_branch)

        if self.source_point is not None and not isinstance(self.source_point, str):
            self.source_point = str(self.source_point)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class BranchSwitch(YAMLRoot):
    """
    Specification for switching from a detected bifurcation point to a new branch (periodic orbits from Hopf, fold
    continuation, etc.). Each BranchSwitch says: "from which special point on the parent branch, continue what kind of
    object, with what settings." Override parent solver settings via the inline continuation field — only explicitly
    set attributes take effect; everything else is inherited from the parent Continuation.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["BranchSwitch"]
    class_class_curie: ClassVar[str] = "tvbo:BranchSwitch"
    class_name: ClassVar[str] = "BranchSwitch"
    class_model_uri: ClassVar[URIRef] = TVBO.BranchSwitch

    name: Union[str, BranchSwitchName] = None
    description: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    source_point: Optional[str] = None
    delta_p: Optional[float] = None
    continuation: Optional[Union[dict, "Continuation"]] = None
    discretization: Optional[Union[dict, Discretization]] = None
    bothside: Optional[Union[bool, Bool]] = None
    options: Optional[Union[dict[Union[str, OptionName], Union[dict, Option]], list[Union[dict, Option]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, BranchSwitchName):
            self.name = BranchSwitchName(self.name)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.source_point is not None and not isinstance(self.source_point, str):
            self.source_point = str(self.source_point)

        if self.delta_p is not None and not isinstance(self.delta_p, float):
            self.delta_p = float(self.delta_p)

        if self.continuation is not None and not isinstance(self.continuation, Continuation):
            self.continuation = Continuation(**as_dict(self.continuation))

        if self.discretization is not None and not isinstance(self.discretization, Discretization):
            self.discretization = Discretization(**as_dict(self.discretization))

        if self.bothside is not None and not isinstance(self.bothside, Bool):
            self.bothside = Bool(self.bothside)

        self._normalize_inlined_as_dict(slot_name="options", slot_type=Option, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Continuation(YAMLRoot):
    """
    Complete specification of a numerical continuation / bifurcation analysis. All universal solver settings live
    directly here. Toolkit-specific string options go in the options slot. When used inside a BranchSwitch, only
    explicitly set attributes override the parent's values.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Continuation"]
    class_class_curie: ClassVar[str] = "tvbo:Continuation"
    class_name: ClassVar[str] = "Continuation"
    class_model_uri: ClassVar[URIRef] = TVBO.Continuation

    name: Union[str, ContinuationName] = None
    label: Optional[str] = None
    description: Optional[str] = None
    dynamics: Optional[Union[str, DynamicsName]] = None
    free_parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    ds: Optional[float] = None
    ds_min: Optional[float] = None
    ds_max: Optional[float] = None
    max_steps: Optional[int] = None
    newton_tol: Optional[float] = None
    newton_max_iterations: Optional[int] = None
    nev: Optional[int] = None
    tol_stability: Optional[float] = None
    detect_bifurcation: Optional[int] = None
    detect_fold: Optional[Union[bool, Bool]] = None
    n_inversion: Optional[int] = None
    max_bisection_steps: Optional[int] = None
    algorithm: Optional[Union[str, "ContinuationAlgorithm"]] = 'PALC'
    initial_state: Optional[Union[dict, InitialState]] = None
    branches: Optional[Union[dict[Union[str, BranchSwitchName], Union[dict, BranchSwitch]], list[Union[dict, BranchSwitch]]]] = empty_dict()
    bothside: Optional[Union[bool, Bool]] = None
    execution: Optional[Union[dict, "ExecutionConfig"]] = None
    software: Optional[Union[dict, "SoftwareRequirement"]] = None
    options: Optional[Union[dict[Union[str, OptionName], Union[dict, Option]], list[Union[dict, Option]]]] = empty_dict()

    def __post_init__(self, *_: str, **kwargs: Any):
        if self._is_empty(self.name):
            self.MissingRequiredField("name")
        if not isinstance(self.name, ContinuationName):
            self.name = ContinuationName(self.name)

        if self.label is not None and not isinstance(self.label, str):
            self.label = str(self.label)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dynamics is not None and not isinstance(self.dynamics, DynamicsName):
            self.dynamics = DynamicsName(self.dynamics)

        self._normalize_inlined_as_dict(slot_name="free_parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.ds is not None and not isinstance(self.ds, float):
            self.ds = float(self.ds)

        if self.ds_min is not None and not isinstance(self.ds_min, float):
            self.ds_min = float(self.ds_min)

        if self.ds_max is not None and not isinstance(self.ds_max, float):
            self.ds_max = float(self.ds_max)

        if self.max_steps is not None and not isinstance(self.max_steps, int):
            self.max_steps = int(self.max_steps)

        if self.newton_tol is not None and not isinstance(self.newton_tol, float):
            self.newton_tol = float(self.newton_tol)

        if self.newton_max_iterations is not None and not isinstance(self.newton_max_iterations, int):
            self.newton_max_iterations = int(self.newton_max_iterations)

        if self.nev is not None and not isinstance(self.nev, int):
            self.nev = int(self.nev)

        if self.tol_stability is not None and not isinstance(self.tol_stability, float):
            self.tol_stability = float(self.tol_stability)

        if self.detect_bifurcation is not None and not isinstance(self.detect_bifurcation, int):
            self.detect_bifurcation = int(self.detect_bifurcation)

        if self.detect_fold is not None and not isinstance(self.detect_fold, Bool):
            self.detect_fold = Bool(self.detect_fold)

        if self.n_inversion is not None and not isinstance(self.n_inversion, int):
            self.n_inversion = int(self.n_inversion)

        if self.max_bisection_steps is not None and not isinstance(self.max_bisection_steps, int):
            self.max_bisection_steps = int(self.max_bisection_steps)

        if self.algorithm is not None and not isinstance(self.algorithm, ContinuationAlgorithm):
            self.algorithm = getattr(ContinuationAlgorithm, self.algorithm)

        if self.initial_state is not None and not isinstance(self.initial_state, InitialState):
            self.initial_state = InitialState(**as_dict(self.initial_state))

        self._normalize_inlined_as_dict(slot_name="branches", slot_type=BranchSwitch, key_name="name", keyed=True)

        if self.bothside is not None and not isinstance(self.bothside, Bool):
            self.bothside = Bool(self.bothside)

        if self.execution is not None and not isinstance(self.execution, ExecutionConfig):
            self.execution = ExecutionConfig(**as_dict(self.execution))

        if self.software is not None and not isinstance(self.software, SoftwareRequirement):
            self.software = SoftwareRequirement(**as_dict(self.software))

        self._normalize_inlined_as_dict(slot_name="options", slot_type=Option, key_name="name", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Solver(YAMLRoot):
    """
    Lightweight specification of a numerical ODE solver / integrator. Covers adaptive solvers (Vern9, Rodas5, Tsit5,
    etc.) used in shooting methods, initial-state integration, and other contexts where only the algorithm and
    tolerances matter.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Solver"]
    class_class_curie: ClassVar[str] = "tvbo:Solver"
    class_name: ClassVar[str] = "Solver"
    class_model_uri: ClassVar[URIRef] = TVBO.Solver

    method: Optional[str] = "Tsit5"
    abs_tol: Optional[float] = 1e-10
    rel_tol: Optional[float] = 1e-10

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.method is not None and not isinstance(self.method, str):
            self.method = str(self.method)

        if self.abs_tol is not None and not isinstance(self.abs_tol, float):
            self.abs_tol = float(self.abs_tol)

        if self.rel_tol is not None and not isinstance(self.rel_tol, float):
            self.rel_tol = float(self.rel_tol)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Integrator(Solver):
    """
    Fixed-step or adaptive ODE integrator with TVB-specific extensions (noise, transient time, etc.). Inherits
    abs_tol, rel_tol from Solver. Overrides method default to 'euler'.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Integrator"]
    class_class_curie: ClassVar[str] = "tvbo:Integrator"
    class_name: ClassVar[str] = "Integrator"
    class_model_uri: ClassVar[URIRef] = TVBO.Integrator

    time_scale: Optional[Union[str, "UnitEnum"]] = 'ms'
    unit: Optional[Union[str, "UnitEnum"]] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    duration: Optional[float] = 1000
    description: Optional[str] = None
    method: Optional[str] = "euler"
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
        if self.time_scale is not None and not isinstance(self.time_scale, UnitEnum):
            self.time_scale = getattr(UnitEnum, self.time_scale)

        if self.unit is not None and not isinstance(self.unit, UnitEnum):
            self.unit = UnitEnum(self.unit)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.duration is not None and not isinstance(self.duration, float):
            self.duration = float(self.duration)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

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
class Coupling(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["Coupling"]
    class_class_curie: ClassVar[str] = "tvbo:Coupling"
    class_name: ClassVar[str] = "Coupling"
    class_model_uri: ClassVar[URIRef] = TVBO.Coupling

    name: Union[str, CouplingName] = "Linear"
    label: Optional[str] = None
    iri: Optional[str] = None
    parameters: Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]] = empty_dict()
    description: Optional[str] = None
    coupling_function: Optional[Union[dict, Equation]] = None
    sparse: Optional[Union[bool, Bool]] = False
    pre_expression: Optional[Union[dict, Equation]] = None
    post_expression: Optional[Union[dict, Equation]] = None
    incoming_states: Optional[Union[Union[str, StateVariableName], list[Union[str, StateVariableName]]]] = empty_list()
    local_states: Optional[Union[Union[str, StateVariableName], list[Union[str, StateVariableName]]]] = empty_list()
    delayed: Optional[Union[bool, Bool]] = True
    symmetry: Optional[str] = "directed"
    outsym: Optional[Union[str, list[str]]] = empty_list()
    observed: Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, DerivedVariable]], list[Union[dict, DerivedVariable]]]] = empty_dict()
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

        if self.iri is not None and not isinstance(self.iri, str):
            self.iri = str(self.iri)

        self._normalize_inlined_as_dict(slot_name="parameters", slot_type=Parameter, key_name="name", keyed=True)

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.coupling_function is not None and not isinstance(self.coupling_function, Equation):
            self.coupling_function = Equation(**as_dict(self.coupling_function))

        if self.sparse is not None and not isinstance(self.sparse, Bool):
            self.sparse = Bool(self.sparse)

        if self.pre_expression is not None and not isinstance(self.pre_expression, Equation):
            self.pre_expression = Equation(**as_dict(self.pre_expression))

        if self.post_expression is not None and not isinstance(self.post_expression, Equation):
            self.post_expression = Equation(**as_dict(self.post_expression))

        if not isinstance(self.incoming_states, list):
            self.incoming_states = [self.incoming_states] if self.incoming_states is not None else []
        self.incoming_states = [v if isinstance(v, StateVariableName) else StateVariableName(v) for v in self.incoming_states]

        if not isinstance(self.local_states, list):
            self.local_states = [self.local_states] if self.local_states is not None else []
        self.local_states = [v if isinstance(v, StateVariableName) else StateVariableName(v) for v in self.local_states]

        if self.delayed is not None and not isinstance(self.delayed, Bool):
            self.delayed = Bool(self.delayed)

        if self.symmetry is not None and not isinstance(self.symmetry, str):
            self.symmetry = str(self.symmetry)

        if not isinstance(self.outsym, list):
            self.outsym = [self.outsym] if self.outsym is not None else []
        self.outsym = [v if isinstance(v, str) else str(v) for v in self.outsym]

        self._normalize_inlined_as_dict(slot_name="observed", slot_type=DerivedVariable, key_name="name", keyed=True)

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
class ExecutionConfig(YAMLRoot):
    """
    Configuration for computational execution (parallelization, precision, hardware).
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = TVBO["ExecutionConfig"]
    class_class_curie: ClassVar[str] = "tvbo:ExecutionConfig"
    class_name: ClassVar[str] = "ExecutionConfig"
    class_model_uri: ClassVar[URIRef] = TVBO.ExecutionConfig

    n_workers: Optional[int] = 1
    n_threads: Optional[int] = -1
    precision: Optional[str] = "float64"
    accelerator: Optional[str] = "cpu"
    batch_size: Optional[int] = None
    random_seed: Optional[int] = 42
    find_fixpoint: Optional[Union[bool, Bool]] = False

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.n_workers is not None and not isinstance(self.n_workers, int):
            self.n_workers = int(self.n_workers)

        if self.n_threads is not None and not isinstance(self.n_threads, int):
            self.n_threads = int(self.n_threads)

        if self.precision is not None and not isinstance(self.precision, str):
            self.precision = str(self.precision)

        if self.accelerator is not None and not isinstance(self.accelerator, str):
            self.accelerator = str(self.accelerator)

        if self.batch_size is not None and not isinstance(self.batch_size, int):
            self.batch_size = int(self.batch_size)

        if self.random_seed is not None and not isinstance(self.random_seed, int):
            self.random_seed = int(self.random_seed)

        if self.find_fixpoint is not None and not isinstance(self.find_fixpoint, Bool):
            self.find_fixpoint = Bool(self.find_fixpoint)

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
    dynamics: Optional[Union[dict, Dynamics]] = None
    integration: Optional[Union[dict, Integrator]] = None
    connectivity: Optional[Union[dict, Network]] = None
    network: Optional[Union[dict, Network]] = None
    coupling: Optional[Union[dict, Coupling]] = None
    observations: Optional[Union[dict[Union[str, ObservationName], Union[dict, Observation]], list[Union[dict, Observation]]]] = empty_dict()
    derived_observations: Optional[Union[dict[Union[str, DerivedObservationName], Union[dict, DerivedObservation]], list[Union[dict, DerivedObservation]]]] = empty_dict()
    functions: Optional[Union[dict[Union[str, FunctionName], Union[dict, Function]], list[Union[dict, Function]]]] = empty_dict()
    stimulation: Optional[Union[dict, Stimulus]] = None
    events: Optional[Union[dict[Union[str, EventName], Union[dict, Event]], list[Union[dict, Event]]]] = empty_dict()
    field_dynamics: Optional[Union[dict, "PDE"]] = None
    optimizations: Optional[Union[dict[Union[str, OptimizationName], Union[dict, Optimization]], list[Union[dict, Optimization]]]] = empty_dict()
    explorations: Optional[Union[dict[Union[str, ExplorationName], Union[dict, Exploration]], list[Union[dict, Exploration]]]] = empty_dict()
    algorithms: Optional[Union[dict[Union[str, AlgorithmName], Union[dict, Algorithm]], list[Union[dict, Algorithm]]]] = empty_dict()
    continuations: Optional[Union[dict[Union[str, ContinuationName], Union[dict, Continuation]], list[Union[dict, Continuation]]]] = empty_dict()
    environment: Optional[Union[dict, "SoftwareEnvironment"]] = None
    execution: Optional[Union[dict, ExecutionConfig]] = None
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

        if self.dynamics is not None and not isinstance(self.dynamics, Dynamics):
            self.dynamics = Dynamics(**as_dict(self.dynamics))

        if self.integration is not None and not isinstance(self.integration, Integrator):
            self.integration = Integrator(**as_dict(self.integration))

        if self.connectivity is not None and not isinstance(self.connectivity, Network):
            self.connectivity = Network(**as_dict(self.connectivity))

        if self.network is not None and not isinstance(self.network, Network):
            self.network = Network(**as_dict(self.network))

        if self.coupling is not None and not isinstance(self.coupling, Coupling):
            self.coupling = Coupling(**as_dict(self.coupling))

        self._normalize_inlined_as_dict(slot_name="observations", slot_type=Observation, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="derived_observations", slot_type=DerivedObservation, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="functions", slot_type=Function, key_name="name", keyed=True)

        if self.stimulation is not None and not isinstance(self.stimulation, Stimulus):
            self.stimulation = Stimulus(**as_dict(self.stimulation))

        self._normalize_inlined_as_dict(slot_name="events", slot_type=Event, key_name="name", keyed=True)

        if self.field_dynamics is not None and not isinstance(self.field_dynamics, PDE):
            self.field_dynamics = PDE(**as_dict(self.field_dynamics))

        self._normalize_inlined_as_dict(slot_name="optimizations", slot_type=Optimization, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="explorations", slot_type=Exploration, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="algorithms", slot_type=Algorithm, key_name="name", keyed=True)

        self._normalize_inlined_as_dict(slot_name="continuations", slot_type=Continuation, key_name="name", keyed=True)

        if self.environment is not None and not isinstance(self.environment, SoftwareEnvironment):
            self.environment = SoftwareEnvironment(**as_dict(self.environment))

        if self.execution is not None and not isinstance(self.execution, ExecutionConfig):
            self.execution = ExecutionConfig(**as_dict(self.execution))

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
    experiments: Optional[Union[dict[Union[int, SimulationExperimentId], Union[dict, SimulationExperiment]], list[Union[dict, SimulationExperiment]]]] = empty_dict()

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

        self._normalize_inlined_as_list(slot_name="experiments", slot_type=SimulationExperiment, key_name="id", keyed=True)

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class TimeSeries(YAMLRoot):
    """
    Time series data from simulations or measurements. Supports BIDS-compatible export for computational modeling
    (BEP034).
    """
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
    sampling_period: Optional[float] = None
    sampling_period_unit: Optional[str] = "ms"
    unit: Optional[str] = None
    labels_ordering: Optional[Union[str, list[str]]] = empty_list()
    labels_dimensions: Optional[str] = None
    source_experiment: Optional[Union[int, SimulationExperimentId]] = None
    generated_at: Optional[Union[str, XSDDateTime]] = None
    software_environment: Optional[Union[dict, "SoftwareEnvironment"]] = None
    task_name: Optional[str] = None
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[int] = None
    modality: Optional[Union[str, "ImagingModality"]] = None
    model_equation_ref: Optional[str] = None
    model_param_ref: Optional[str] = None
    connectivity_ref: Optional[str] = None

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

        if self.sampling_period is not None and not isinstance(self.sampling_period, float):
            self.sampling_period = float(self.sampling_period)

        if self.sampling_period_unit is not None and not isinstance(self.sampling_period_unit, str):
            self.sampling_period_unit = str(self.sampling_period_unit)

        if self.unit is not None and not isinstance(self.unit, str):
            self.unit = str(self.unit)

        if not isinstance(self.labels_ordering, list):
            self.labels_ordering = [self.labels_ordering] if self.labels_ordering is not None else []
        self.labels_ordering = [v if isinstance(v, str) else str(v) for v in self.labels_ordering]

        if self.labels_dimensions is not None and not isinstance(self.labels_dimensions, str):
            self.labels_dimensions = str(self.labels_dimensions)

        if self.source_experiment is not None and not isinstance(self.source_experiment, SimulationExperimentId):
            self.source_experiment = SimulationExperimentId(self.source_experiment)

        if self.generated_at is not None and not isinstance(self.generated_at, XSDDateTime):
            self.generated_at = XSDDateTime(self.generated_at)

        if self.software_environment is not None and not isinstance(self.software_environment, SoftwareEnvironment):
            self.software_environment = SoftwareEnvironment(**as_dict(self.software_environment))

        if self.task_name is not None and not isinstance(self.task_name, str):
            self.task_name = str(self.task_name)

        if self.subject_id is not None and not isinstance(self.subject_id, str):
            self.subject_id = str(self.subject_id)

        if self.session_id is not None and not isinstance(self.session_id, str):
            self.session_id = str(self.session_id)

        if self.run_id is not None and not isinstance(self.run_id, int):
            self.run_id = int(self.run_id)

        if self.modality is not None and not isinstance(self.modality, ImagingModality):
            self.modality = ImagingModality(self.modality)

        if self.model_equation_ref is not None and not isinstance(self.model_equation_ref, str):
            self.model_equation_ref = str(self.model_equation_ref)

        if self.model_param_ref is not None and not isinstance(self.model_param_ref, str):
            self.model_param_ref = str(self.model_param_ref)

        if self.connectivity_ref is not None and not isinstance(self.connectivity_ref, str):
            self.connectivity_ref = str(self.connectivity_ref)

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
    description: Optional[str] = None
    dataLocation: Optional[str] = None
    package: Optional[Union[str, SoftwarePackageName]] = None
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

        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)

        if self.dataLocation is not None and not isinstance(self.dataLocation, str):
            self.dataLocation = str(self.dataLocation)

        if self.package is not None and not isinstance(self.package, SoftwarePackageName):
            self.package = SoftwarePackageName(self.package)

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
    mesh_file: Optional[str] = None
    mesh_format: Optional[str] = None
    number_of_vertices: Optional[int] = None
    number_of_elements: Optional[int] = None

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

        if self.mesh_file is not None and not isinstance(self.mesh_file, str):
            self.mesh_file = str(self.mesh_file)

        if self.mesh_format is not None and not isinstance(self.mesh_format, str):
            self.mesh_format = str(self.mesh_format)

        if self.number_of_vertices is not None and not isinstance(self.number_of_vertices, int):
            self.number_of_vertices = int(self.number_of_vertices)

        if self.number_of_elements is not None and not isinstance(self.number_of_elements, int):
            self.number_of_elements = int(self.number_of_elements)

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
    dt: Optional[float] = None
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

        if self.dt is not None and not isinstance(self.dt, float):
            self.dt = float(self.dt)

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
    state_variables: Optional[Union[dict[Union[str, FieldStateVariableName], Union[dict, FieldStateVariable]], list[Union[dict, FieldStateVariable]]]] = empty_dict()
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

        self._normalize_inlined_as_list(slot_name="state_variables", slot_type=FieldStateVariable, key_name="name", keyed=True)

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
    unit: Optional[Union[str, "UnitEnum"]] = None
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

        if self.unit is not None and not isinstance(self.unit, UnitEnum):
            self.unit = UnitEnum(self.unit)

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
    entities: Optional[Union[dict[Union[str, ParcellationEntityName], Union[dict, ParcellationEntity]], list[Union[dict, ParcellationEntity]]]] = empty_dict()

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

        self._normalize_inlined_as_dict(slot_name="entities", slot_type=ParcellationEntity, key_name="name", keyed=True)

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
class UnitEnum(EnumDefinitionImpl):
    """
    Physical units of measurement for model parameters, state variables, and integration settings. Uses conventional
    abbreviations as values, mapped to the QUDT ontology (http://qudt.org/vocab/unit/) with UO cross-references where
    available.
    """
    s = PermissibleValue(
        text="s",
        description="Second",
        meaning=QUDT["SEC"])
    ms = PermissibleValue(
        text="ms",
        description="Millisecond",
        meaning=QUDT["MilliSEC"])
    us = PermissibleValue(
        text="us",
        description="Microsecond",
        meaning=QUDT["MicroSEC"])
    per_s = PermissibleValue(
        text="per_s",
        description="Per second (s⁻¹)",
        meaning=QUDT["PER-SEC"])
    per_ms = PermissibleValue(
        text="per_ms",
        description="Per millisecond (ms⁻¹)",
        meaning=QUDT["PER-MilliSEC"])
    Hz = PermissibleValue(
        text="Hz",
        description="Hertz (s⁻¹)",
        meaning=QUDT["HZ"])
    kHz = PermissibleValue(
        text="kHz",
        description="Kilohertz",
        meaning=QUDT["KiloHZ"])
    V = PermissibleValue(
        text="V",
        description="Volt",
        meaning=QUDT["V"])
    mV = PermissibleValue(
        text="mV",
        description="Millivolt",
        meaning=QUDT["MilliV"])
    per_mV = PermissibleValue(
        text="per_mV",
        description="Reciprocal millivolt (mV⁻¹)",
        meaning=QUDT["PER-MilliV"])
    mV_per_ms = PermissibleValue(
        text="mV_per_ms",
        description="Millivolt per millisecond",
        meaning=QUDT["MilliV-PER-MilliSEC"])
    mV_per_s = PermissibleValue(
        text="mV_per_s",
        description="Millivolt per second",
        meaning=QUDT["MilliV-PER-SEC"])
    A = PermissibleValue(
        text="A",
        description="Ampere",
        meaning=QUDT["A"])
    nA = PermissibleValue(
        text="nA",
        description="Nanoampere",
        meaning=QUDT["NanoA"])
    pA = PermissibleValue(
        text="pA",
        description="Picoampere",
        meaning=QUDT["PicoA"])
    pF = PermissibleValue(
        text="pF",
        description="Picofarad",
        meaning=QUDT["PicoFARAD"])
    nF = PermissibleValue(
        text="nF",
        description="Nanofarad",
        meaning=QUDT["NanoFARAD"])
    nS = PermissibleValue(
        text="nS",
        description="Nanosiemens",
        meaning=QUDT["NanoS"])
    uS = PermissibleValue(
        text="uS",
        description="Microsiemens",
        meaning=QUDT["MicroS"])
    per_nC = PermissibleValue(
        text="per_nC",
        description="Reciprocal nanocoulomb (nC⁻¹)",
        meaning=QUDT["PER-NanoC"])
    per_pC = PermissibleValue(
        text="per_pC",
        description="Reciprocal picocoulomb (pC⁻¹)",
        meaning=QUDT["PER-PicoC"])
    mol_per_m3 = PermissibleValue(
        text="mol_per_m3",
        description="Mole per cubic metre (mol/m³)",
        meaning=QUDT["MOL-PER-M3"])
    mmol_per_m3 = PermissibleValue(
        text="mmol_per_m3",
        description="Millimole per cubic metre (mmol/m³ ≈ mM)",
        meaning=QUDT["MilliMOL-PER-M3"])
    um3 = PermissibleValue(
        text="um3",
        description="Cubic micrometre (µm³)",
        meaning=QUDT["MicroM3"])
    m = PermissibleValue(
        text="m",
        description="Metre",
        meaning=QUDT["M"])
    mm = PermissibleValue(
        text="mm",
        description="Millimetre",
        meaning=QUDT["MilliM"])
    cm = PermissibleValue(
        text="cm",
        description="Centimetre",
        meaning=QUDT["CentiM"])
    m_per_s = PermissibleValue(
        text="m_per_s",
        description="Metre per second",
        meaning=QUDT["M-PER-SEC"])
    mm_per_ms = PermissibleValue(
        text="mm_per_ms",
        description="Millimetre per millisecond (= m/s)",
        meaning=QUDT["MilliM-PER-MilliSEC"])
    Hz_per_nA = PermissibleValue(
        text="Hz_per_nA",
        description="Hertz per nanoampere (neural gain)")
    S_per_m = PermissibleValue(
        text="S_per_m",
        description="Siemens per metre (conductivity)",
        meaning=QUDT["S-PER-M"])
    H_per_m = PermissibleValue(
        text="H_per_m",
        description="Henry per metre (permeability)",
        meaning=QUDT["H-PER-M"])
    rad_per_ms = PermissibleValue(
        text="rad_per_ms",
        description="Radian per millisecond")
    dimensionless = PermissibleValue(
        text="dimensionless",
        description="Dimensionless (unitless)",
        meaning=QUDT["UNITLESS"])
    percent = PermissibleValue(
        text="percent",
        description="Percent (%)",
        meaning=QUDT["PERCENT"])
    arbitrary_unit = PermissibleValue(
        text="arbitrary_unit",
        description="Arbitrary units (a.u.)")
    kg = PermissibleValue(
        text="kg",
        description="Kilogram",
        meaning=QUDT["KiloGM"])
    kg_per_s = PermissibleValue(
        text="kg_per_s",
        description="Kilogram per second")
    m_per_s2 = PermissibleValue(
        text="m_per_s2",
        description="Metre per second squared (acceleration)",
        meaning=QUDT["M-PER-SEC2"])
    N_per_m = PermissibleValue(
        text="N_per_m",
        description="Newton per metre (spring constant)",
        meaning=QUDT["N-PER-M"])
    rad = PermissibleValue(
        text="rad",
        description="Radian",
        meaning=QUDT["RAD"])
    rad_per_s = PermissibleValue(
        text="rad_per_s",
        description="Radian per second (angular velocity)",
        meaning=QUDT["RAD-PER-SEC"])
    s2 = PermissibleValue(
        text="s2",
        description="Second squared (inertia constant)",
        meaning=QUDT["SEC2"])
    per_unit = PermissibleValue(
        text="per_unit",
        description="Per-unit (dimensionless power-systems convention)")

    _defn = EnumDefinition(
        name="UnitEnum",
        description="""Physical units of measurement for model parameters, state variables, and integration settings. Uses conventional abbreviations as values, mapped to the QUDT ontology (http://qudt.org/vocab/unit/) with UO cross-references where available.""",
    )

class PhysicalDimension(EnumDefinitionImpl):
    """
    Physical dimension categories for LEMS and dimensional analysis. Each dimension decomposes into SI base dimensions
    (M, L, T, I, K, N).
    """
    none = PermissibleValue(
        text="none",
        description="Dimensionless")
    time = PermissibleValue(
        text="time",
        description="Time [T]")
    per_time = PermissibleValue(
        text="per_time",
        description="Inverse time [T⁻¹]")
    voltage = PermissibleValue(
        text="voltage",
        description="Voltage [M L² T⁻³ I⁻¹]")
    current = PermissibleValue(
        text="current",
        description="Electric current [I]")
    capacitance = PermissibleValue(
        text="capacitance",
        description="Capacitance [M⁻¹ L⁻² T⁴ I²]")
    conductance = PermissibleValue(
        text="conductance",
        description="Conductance [M⁻¹ L⁻² T³ I²]")
    resistance = PermissibleValue(
        text="resistance",
        description="Resistance [M L² T⁻³ I⁻²]")
    charge = PermissibleValue(
        text="charge",
        description="Electric charge [T I]")
    concentration = PermissibleValue(
        text="concentration",
        description="Concentration [L⁻³ N]")
    substance = PermissibleValue(
        text="substance",
        description="Amount of substance [N]")
    length = PermissibleValue(
        text="length",
        description="Length [L]")
    volume = PermissibleValue(
        text="volume",
        description="Volume [L³]")
    temperature = PermissibleValue(
        text="temperature",
        description="Temperature [K]")

    _defn = EnumDefinition(
        name="PhysicalDimension",
        description="""Physical dimension categories for LEMS and dimensional analysis. Each dimension decomposes into SI base dimensions (M, L, T, I, K, N).""",
    )

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

class ModelType(EnumDefinitionImpl):
    """
    Coarse classification of a Dynamics model by its mathematical/biological origin. Used for filtering and display in
    list_db().
    """
    mean_field = PermissibleValue(
        text="mean_field",
        description="""Mathematically derived mean-field models obtained by exact reduction of spiking networks (Ott-Antonsen ansatz, Lorentzian heterogeneity, etc.). Examples: MontbrioPazoRoxin, CoombesByrne, ReducedWongWang, ZerlautAdaptationFirstOrder.""")
    neural_mass = PermissibleValue(
        text="neural_mass",
        description="""Phenomenological population-rate / neural-mass models that describe synaptic and firing-rate dynamics without an explicit derivation from single-neuron statistics. Examples: JansenRit, WilsonCowan, LarterBreakspear, TsodyksMarkram.""")
    phase_oscillator = PermissibleValue(
        text="phase_oscillator",
        description="Phase-reduced or Kuramoto-type oscillator models. Examples: Kuramoto, SupHopf.")
    phenomenological = PermissibleValue(
        text="phenomenological",
        description="""Empirical / phenomenological models that capture macroscopic dynamics without direct biophysical derivation. Examples: Epileptor2D, Epileptor5D.""")
    spiking = PermissibleValue(
        text="spiking",
        description="""Single-neuron or conductance-based spiking models (HH, AdEx, LIF, Izhikevich, etc.). These can be used as nodes in a network alongside mean-field models.""")
    generic = PermissibleValue(
        text="generic",
        description="""Generic / normal-form dynamical systems not specific to neural modelling (e.g. Generic2dOscillator, GenericLinear).""")
    field = PermissibleValue(
        text="field",
        description="""Spatially distributed neural-field models described by integro- differential or PDE formulations.""")

    _defn = EnumDefinition(
        name="ModelType",
        description="""Coarse classification of a Dynamics model by its mathematical/biological origin. Used for filtering and display in list_db().""",
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

class SamplingAxis(EnumDefinitionImpl):
    """
    Dimension along which a distribution is sampled.
    """
    space = PermissibleValue(
        text="space",
        description="Sample once per node (heterogeneous parameter or spatially varying IC).")
    time = PermissibleValue(
        text="time",
        description="Resample every integration timestep (stochastic time-varying input).")

    _defn = EnumDefinition(
        name="SamplingAxis",
        description="Dimension along which a distribution is sampled.",
    )

class NoiseType(EnumDefinitionImpl):

    gaussian = PermissibleValue(text="gaussian")
    white = PermissibleValue(text="white")
    brown = PermissibleValue(text="brown")
    pink = PermissibleValue(text="pink")

    _defn = EnumDefinition(
        name="NoiseType",
    )

class AggregationType(EnumDefinitionImpl):
    """
    How to aggregate time series data
    """
    mean = PermissibleValue(
        text="mean",
        description="Average over time")
    last = PermissibleValue(
        text="last",
        description="Last value in window")
    first = PermissibleValue(
        text="first",
        description="First value in window")
    window = PermissibleValue(
        text="window",
        description="Sliding window aggregation")
    none = PermissibleValue(
        text="none",
        description="No aggregation")

    _defn = EnumDefinition(
        name="AggregationType",
        description="How to aggregate time series data",
    )

class EventType(EnumDefinitionImpl):
    """
    Type of event triggering mechanism.
    """
    continuous = PermissibleValue(
        text="continuous",
        description="""Triggered when condition function crosses zero (root-finding). Maps to ContinuousCallback / ContinuousComponentCallback.""")
    discrete = PermissibleValue(
        text="discrete",
        description="""Triggered when condition function returns true (checked at each step). Maps to DiscreteCallback / DiscreteComponentCallback.""")
    preset_time = PermissibleValue(
        text="preset_time",
        description="""Triggered at predetermined time points. Maps to PresetTimeCallback / PresetTimeComponentCallback.""")
    stimulus = PermissibleValue(
        text="stimulus",
        description="Continuous time-dependent input signal (e.g., external current). Legacy Stimulus behavior.")

    _defn = EnumDefinition(
        name="EventType",
        description="Type of event triggering mechanism.",
    )

class StandardGraphType(EnumDefinitionImpl):
    """
    Well-known graph generator families with automatic backend mapping. The type field on GraphGenerator is a free
    string; this enum lists common types that get automatic code generation for Julia (Graphs.jl) and Python
    (NetworkX).
    """
    BarabasiAlbert = PermissibleValue(
        text="BarabasiAlbert",
        description="Barabasi-Albert preferential attachment (params: k)")
    WattsStrogatz = PermissibleValue(
        text="WattsStrogatz",
        description="Watts-Strogatz small-world (params: k, p)")
    ErdosRenyi = PermissibleValue(
        text="ErdosRenyi",
        description="Erdos-Renyi random graph (params: p)")
    Complete = PermissibleValue(
        text="Complete",
        description="Complete graph (all-to-all)")
    Cycle = PermissibleValue(
        text="Cycle",
        description="Cycle graph (ring)")
    Star = PermissibleValue(
        text="Star",
        description="Star graph")
    RandomRegular = PermissibleValue(
        text="RandomRegular",
        description="Random regular graph (params: k)")
    Grid = PermissibleValue(
        text="Grid",
        description="Grid/lattice graph (params: dims)")

    _defn = EnumDefinition(
        name="StandardGraphType",
        description="""Well-known graph generator families with automatic backend mapping. The type field on GraphGenerator is a free string; this enum lists common types that get automatic code generation for Julia (Graphs.jl) and Python (NetworkX).""",
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

    _defn = EnumDefinition(
        name="EnvironmentType",
    )

class DimensionType(EnumDefinitionImpl):
    """
    Dimensions along which operations can be applied
    """
    time = PermissibleValue(
        text="time",
        description="Temporal dimension")
    state = PermissibleValue(
        text="state",
        description="State variable dimension")
    node = PermissibleValue(
        text="node",
        description="Network node dimension (general graph term)")
    region = PermissibleValue(
        text="region",
        description="Spatial/regional dimension (alias for node in brain networks)")
    mode = PermissibleValue(
        text="mode",
        description="Mode dimension (e.g., coupling modes)")
    sample = PermissibleValue(
        text="sample",
        description="Sample/trial/realization dimension")
    batch = PermissibleValue(
        text="batch",
        description="Batch dimension (for parallel processing)")
    frequency = PermissibleValue(
        text="frequency",
        description="Frequency dimension (spectral analysis)")

    _defn = EnumDefinition(
        name="DimensionType",
        description="Dimensions along which operations can be applied",
    )

class ReductionType(EnumDefinitionImpl):
    """
    Operations for reducing/aggregating values across dimensions
    """
    mean = PermissibleValue(
        text="mean",
        description="Arithmetic mean")
    sum = PermissibleValue(
        text="sum",
        description="Sum of values")
    max = PermissibleValue(
        text="max",
        description="Maximum value")
    min = PermissibleValue(
        text="min",
        description="Minimum value")
    none = PermissibleValue(
        text="none",
        description="No reduction (return per-element values)")

    _defn = EnumDefinition(
        name="ReductionType",
        description="Operations for reducing/aggregating values across dimensions",
    )

class ContinuationAlgorithm(EnumDefinitionImpl):
    """
    Predictor-corrector algorithm for numerical continuation.
    """
    PALC = PermissibleValue(
        text="PALC",
        description="Pseudo-arclength continuation (default). Uses weighted dot product constraint.")
    MoorePenrose = PermissibleValue(
        text="MoorePenrose",
        description="Moore-Penrose continuation.")
    Natural = PermissibleValue(
        text="Natural",
        description="Natural parameter continuation. Simple parameter stepping, no arc-length constraint.")

    _defn = EnumDefinition(
        name="ContinuationAlgorithm",
        description="Predictor-corrector algorithm for numerical continuation.",
    )

class NumericalDiscretizationMethod(EnumDefinitionImpl):
    """
    Numerical discretization method for boundary value problems (periodic orbits, connecting orbits, quasi-periodic
    tori).
    """
    collocation = PermissibleValue(
        text="collocation",
        description="Orthogonal collocation at Gauss points.")
    trapezoid = PermissibleValue(
        text="trapezoid",
        description="Trapezoidal rule discretization.")
    shooting = PermissibleValue(
        text="shooting",
        description="Standard multiple shooting.")
    poincare = PermissibleValue(
        text="poincare",
        description="Poincaré shooting.")

    _defn = EnumDefinition(
        name="NumericalDiscretizationMethod",
        description="""Numerical discretization method for boundary value problems (periodic orbits, connecting orbits, quasi-periodic tori).""",
    )

class InitialStateMethod(EnumDefinitionImpl):
    """
    Strategy for obtaining the starting equilibrium or periodic orbit.
    """
    time_integration = PermissibleValue(
        text="time_integration",
        description="Integrate the ODE forward until convergence (robust, default).")
    newton = PermissibleValue(
        text="newton",
        description="Use Newton's method to find the nearest fixed point.")
    given = PermissibleValue(
        text="given",
        description="Use the model's default initial values directly.")
    from_branch = PermissibleValue(
        text="from_branch",
        description="Start from a point on a previously computed branch.")

    _defn = EnumDefinition(
        name="InitialStateMethod",
        description="Strategy for obtaining the starting equilibrium or periodic orbit.",
    )

class SparseFormat(EnumDefinitionImpl):

    dense = PermissibleValue(
        text="dense",
        description="Dense N×N array with gzip compression")
    csr = PermissibleValue(
        text="csr",
        description="Compressed Sparse Row (data, indices, indptr)")
    coo = PermissibleValue(
        text="coo",
        description="Coordinate list (data, row, col)")

    _defn = EnumDefinition(
        name="SparseFormat",
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
                   model_uri=TVBO.time_scale, domain=None, range=Optional[Union[str, "UnitEnum"]])

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
                   model_uri=TVBO.unit, domain=None, range=Optional[Union[str, "UnitEnum"]])

slots.derived_from = Slot(uri=TVBO.derived_from, name="derived_from", curie=TVBO.curie('derived_from'),
                   model_uri=TVBO.derived_from, domain=None, range=Optional[str])

slots.source = Slot(uri=TVBO.source, name="source", curie=TVBO.curie('source'),
                   model_uri=TVBO.source, domain=None, range=Optional[str])

slots.dataset_path = Slot(uri=TVBO.dataset_path, name="dataset_path", curie=TVBO.curie('dataset_path'),
                   model_uri=TVBO.dataset_path, domain=None, range=Optional[str])

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
                   model_uri=TVBO.range__lo, domain=None, range=Optional[str])

slots.range__hi = Slot(uri=TVBO.hi, name="range__hi", curie=TVBO.curie('hi'),
                   model_uri=TVBO.range__hi, domain=None, range=Optional[str])

slots.range__step = Slot(uri=TVBO.step, name="range__step", curie=TVBO.curie('step'),
                   model_uri=TVBO.range__step, domain=None, range=Optional[str])

slots.range__n = Slot(uri=TVBO.n, name="range__n", curie=TVBO.curie('n'),
                   model_uri=TVBO.range__n, domain=None, range=Optional[int])

slots.range__log_scale = Slot(uri=TVBO.log_scale, name="range__log_scale", curie=TVBO.curie('log_scale'),
                   model_uri=TVBO.range__log_scale, domain=None, range=Optional[Union[bool, Bool]])

slots.range__explored_values = Slot(uri=TVBO.explored_values, name="range__explored_values", curie=TVBO.curie('explored_values'),
                   model_uri=TVBO.range__explored_values, domain=None, range=Optional[Union[float, list[float]]])

slots.range__element = Slot(uri=TVBO.element, name="range__element", curie=TVBO.curie('element'),
                   model_uri=TVBO.range__element, domain=None, range=Optional[int])

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

slots.event__event_type = Slot(uri=TVBO.event_type, name="event__event_type", curie=TVBO.curie('event_type'),
                   model_uri=TVBO.event__event_type, domain=None, range=Optional[Union[str, "EventType"]])

slots.event__condition = Slot(uri=TVBO.condition, name="event__condition", curie=TVBO.curie('condition'),
                   model_uri=TVBO.event__condition, domain=None, range=Optional[Union[dict, Equation]])

slots.event__condition_states = Slot(uri=TVBO.condition_states, name="event__condition_states", curie=TVBO.curie('condition_states'),
                   model_uri=TVBO.event__condition_states, domain=None, range=Optional[Union[str, list[str]]])

slots.event__condition_parameters = Slot(uri=TVBO.condition_parameters, name="event__condition_parameters", curie=TVBO.curie('condition_parameters'),
                   model_uri=TVBO.event__condition_parameters, domain=None, range=Optional[Union[str, list[str]]])

slots.event__affect = Slot(uri=TVBO.affect, name="event__affect", curie=TVBO.curie('affect'),
                   model_uri=TVBO.event__affect, domain=None, range=Optional[Union[dict, Equation]])

slots.event__affect_states = Slot(uri=TVBO.affect_states, name="event__affect_states", curie=TVBO.curie('affect_states'),
                   model_uri=TVBO.event__affect_states, domain=None, range=Optional[Union[str, list[str]]])

slots.event__affect_parameters = Slot(uri=TVBO.affect_parameters, name="event__affect_parameters", curie=TVBO.curie('affect_parameters'),
                   model_uri=TVBO.event__affect_parameters, domain=None, range=Optional[Union[str, list[str]]])

slots.event__affect_negative = Slot(uri=TVBO.affect_negative, name="event__affect_negative", curie=TVBO.curie('affect_negative'),
                   model_uri=TVBO.event__affect_negative, domain=None, range=Optional[Union[dict, Equation]])

slots.event__trigger_times = Slot(uri=TVBO.trigger_times, name="event__trigger_times", curie=TVBO.curie('trigger_times'),
                   model_uri=TVBO.event__trigger_times, domain=None, range=Optional[Union[float, list[float]]])

slots.event__target_component = Slot(uri=TVBO.target_component, name="event__target_component", curie=TVBO.curie('target_component'),
                   model_uri=TVBO.event__target_component, domain=None, range=Optional[str])

slots.event__equation = Slot(uri=TVBO.equation, name="event__equation", curie=TVBO.curie('equation'),
                   model_uri=TVBO.event__equation, domain=None, range=Optional[Union[dict, Equation]])

slots.event__regions = Slot(uri=TVBO.regions, name="event__regions", curie=TVBO.curie('regions'),
                   model_uri=TVBO.event__regions, domain=None, range=Optional[Union[int, list[int]]])

slots.event__weighting = Slot(uri=TVBO.weighting, name="event__weighting", curie=TVBO.curie('weighting'),
                   model_uri=TVBO.event__weighting, domain=None, range=Optional[Union[float, list[float]]])

slots.event__duration = Slot(uri=TVBO.duration, name="event__duration", curie=TVBO.curie('duration'),
                   model_uri=TVBO.event__duration, domain=None, range=Optional[float])

slots.temporalApplicableEquation__time_dependent = Slot(uri=TVBO.time_dependent, name="temporalApplicableEquation__time_dependent", curie=TVBO.curie('time_dependent'),
                   model_uri=TVBO.temporalApplicableEquation__time_dependent, domain=None, range=Optional[Union[bool, Bool]])

slots.parcellation__data_source = Slot(uri=TVBO.data_source, name="parcellation__data_source", curie=TVBO.curie('data_source'),
                   model_uri=TVBO.parcellation__data_source, domain=None, range=Optional[str])

slots.parcellation__atlas = Slot(uri=TVBO.atlas, name="parcellation__atlas", curie=TVBO.curie('atlas'),
                   model_uri=TVBO.parcellation__atlas, domain=None, range=Union[dict, BrainAtlas])

slots.tractogram__data_source = Slot(uri=TVBO.data_source, name="tractogram__data_source", curie=TVBO.curie('data_source'),
                   model_uri=TVBO.tractogram__data_source, domain=None, range=Optional[str])

slots.tractogram__number_of_subjects = Slot(uri=TVBO.number_of_subjects, name="tractogram__number_of_subjects", curie=TVBO.curie('number_of_subjects'),
                   model_uri=TVBO.tractogram__number_of_subjects, domain=None, range=Optional[int])

slots.tractogram__acquisition = Slot(uri=TVBO.acquisition, name="tractogram__acquisition", curie=TVBO.curie('acquisition'),
                   model_uri=TVBO.tractogram__acquisition, domain=None, range=Optional[str])

slots.tractogram__processing_pipeline = Slot(uri=TVBO.processing_pipeline, name="tractogram__processing_pipeline", curie=TVBO.curie('processing_pipeline'),
                   model_uri=TVBO.tractogram__processing_pipeline, domain=None, range=Optional[str])

slots.tractogram__reference = Slot(uri=TVBO.reference, name="tractogram__reference", curie=TVBO.curie('reference'),
                   model_uri=TVBO.tractogram__reference, domain=None, range=Optional[str])

slots.matrix__x = Slot(uri=TVBO.x, name="matrix__x", curie=TVBO.curie('x'),
                   model_uri=TVBO.matrix__x, domain=None, range=Optional[Union[dict, BrainRegionSeries]])

slots.matrix__y = Slot(uri=TVBO.y, name="matrix__y", curie=TVBO.curie('y'),
                   model_uri=TVBO.matrix__y, domain=None, range=Optional[Union[dict, BrainRegionSeries]])

slots.matrix__values = Slot(uri=TVBO.values, name="matrix__values", curie=TVBO.curie('values'),
                   model_uri=TVBO.matrix__values, domain=None, range=Optional[Union[float, list[float]]])

slots.matrix__format = Slot(uri=TVBO.format, name="matrix__format", curie=TVBO.curie('format'),
                   model_uri=TVBO.matrix__format, domain=None, range=Optional[Union[str, "SparseFormat"]])

slots.matrix__shape = Slot(uri=TVBO.shape, name="matrix__shape", curie=TVBO.curie('shape'),
                   model_uri=TVBO.matrix__shape, domain=None, range=Optional[Union[int, list[int]]])

slots.matrix__dtype = Slot(uri=TVBO.dtype, name="matrix__dtype", curie=TVBO.curie('dtype'),
                   model_uri=TVBO.matrix__dtype, domain=None, range=Optional[str])

slots.brainRegionSeries__values = Slot(uri=TVBO.values, name="brainRegionSeries__values", curie=TVBO.curie('values'),
                   model_uri=TVBO.brainRegionSeries__values, domain=None, range=Optional[Union[str, list[str]]])

slots.provenance__date_created = Slot(uri=TVBO.date_created, name="provenance__date_created", curie=TVBO.curie('date_created'),
                   model_uri=TVBO.provenance__date_created, domain=None, range=Optional[str])

slots.provenance__license = Slot(uri=TVBO.license, name="provenance__license", curie=TVBO.curie('license'),
                   model_uri=TVBO.provenance__license, domain=None, range=Optional[str])

slots.provenance__generated_by = Slot(uri=TVBO.generated_by, name="provenance__generated_by", curie=TVBO.curie('generated_by'),
                   model_uri=TVBO.provenance__generated_by, domain=None, range=Optional[str])

slots.bidsEntities__template = Slot(uri=TVBO.template, name="bidsEntities__template", curie=TVBO.curie('template'),
                   model_uri=TVBO.bidsEntities__template, domain=None, range=Optional[str])

slots.bidsEntities__cohort = Slot(uri=TVBO.cohort, name="bidsEntities__cohort", curie=TVBO.curie('cohort'),
                   model_uri=TVBO.bidsEntities__cohort, domain=None, range=Optional[str])

slots.bidsEntities__reconstruction = Slot(uri=TVBO.reconstruction, name="bidsEntities__reconstruction", curie=TVBO.curie('reconstruction'),
                   model_uri=TVBO.bidsEntities__reconstruction, domain=None, range=Optional[str])

slots.bidsEntities__segmentation = Slot(uri=TVBO.segmentation, name="bidsEntities__segmentation", curie=TVBO.curie('segmentation'),
                   model_uri=TVBO.bidsEntities__segmentation, domain=None, range=Optional[str])

slots.bidsEntities__scale = Slot(uri=TVBO.scale, name="bidsEntities__scale", curie=TVBO.curie('scale'),
                   model_uri=TVBO.bidsEntities__scale, domain=None, range=Optional[str])

slots.bidsEntities__atlas = Slot(uri=TVBO.atlas, name="bidsEntities__atlas", curie=TVBO.curie('atlas'),
                   model_uri=TVBO.bidsEntities__atlas, domain=None, range=Optional[str])

slots.bidsEntities__acquisition = Slot(uri=TVBO.acquisition, name="bidsEntities__acquisition", curie=TVBO.curie('acquisition'),
                   model_uri=TVBO.bidsEntities__acquisition, domain=None, range=Optional[str])

slots.network__nodes = Slot(uri=TVBO.nodes, name="network__nodes", curie=TVBO.curie('nodes'),
                   model_uri=TVBO.network__nodes, domain=None, range=Optional[Union[Union[dict, Node], list[Union[dict, Node]]]])

slots.network__edges = Slot(uri=TVBO.edges, name="network__edges", curie=TVBO.curie('edges'),
                   model_uri=TVBO.network__edges, domain=None, range=Optional[Union[Union[dict, Edge], list[Union[dict, Edge]]]])

slots.network__coupling = Slot(uri=TVBO.coupling, name="network__coupling", curie=TVBO.curie('coupling'),
                   model_uri=TVBO.network__coupling, domain=None, range=Optional[Union[dict[Union[str, CouplingName], Union[dict, Coupling]], list[Union[dict, Coupling]]]])

slots.network__dynamics = Slot(uri=TVBO.dynamics, name="network__dynamics", curie=TVBO.curie('dynamics'),
                   model_uri=TVBO.network__dynamics, domain=None, range=Optional[Union[dict[Union[str, DynamicsName], Union[dict, Dynamics]], list[Union[dict, Dynamics]]]])

slots.network__number_of_nodes = Slot(uri=TVBO.number_of_nodes, name="network__number_of_nodes", curie=TVBO.curie('number_of_nodes'),
                   model_uri=TVBO.network__number_of_nodes, domain=None, range=Optional[int])

slots.network__coordinate_space = Slot(uri=TVBO.coordinate_space, name="network__coordinate_space", curie=TVBO.curie('coordinate_space'),
                   model_uri=TVBO.network__coordinate_space, domain=None, range=Optional[Union[dict, CommonCoordinateSpace]])

slots.network__parcellation = Slot(uri=TVBO.parcellation, name="network__parcellation", curie=TVBO.curie('parcellation'),
                   model_uri=TVBO.network__parcellation, domain=None, range=Optional[Union[dict, Parcellation]])

slots.network__tractogram = Slot(uri=TVBO.tractogram, name="network__tractogram", curie=TVBO.curie('tractogram'),
                   model_uri=TVBO.network__tractogram, domain=None, range=Optional[Union[dict, Tractogram]])

slots.network__transforms = Slot(uri=TVBO.transforms, name="network__transforms", curie=TVBO.curie('transforms'),
                   model_uri=TVBO.network__transforms, domain=None, range=Optional[Union[dict[Union[str, FunctionName], Union[dict, Function]], list[Union[dict, Function]]]])

slots.network__data_file = Slot(uri=TVBO.data_file, name="network__data_file", curie=TVBO.curie('data_file'),
                   model_uri=TVBO.network__data_file, domain=None, range=Optional[str])

slots.network__descriptor = Slot(uri=TVBO.descriptor, name="network__descriptor", curie=TVBO.curie('descriptor'),
                   model_uri=TVBO.network__descriptor, domain=None, range=Optional[str])

slots.network__bids_dir = Slot(uri=TVBO.bids_dir, name="network__bids_dir", curie=TVBO.curie('bids_dir'),
                   model_uri=TVBO.network__bids_dir, domain=None, range=Optional[str])

slots.network__bids = Slot(uri=TVBO.bids, name="network__bids", curie=TVBO.curie('bids'),
                   model_uri=TVBO.network__bids, domain=None, range=Optional[Union[dict, BidsEntities]])

slots.network__structural_measures = Slot(uri=TVBO.structural_measures, name="network__structural_measures", curie=TVBO.curie('structural_measures'),
                   model_uri=TVBO.network__structural_measures, domain=None, range=Optional[Union[str, list[str]]])

slots.network__observational_measures = Slot(uri=TVBO.observational_measures, name="network__observational_measures", curie=TVBO.curie('observational_measures'),
                   model_uri=TVBO.network__observational_measures, domain=None, range=Optional[Union[str, list[str]]])

slots.network__provenance = Slot(uri=TVBO.provenance, name="network__provenance", curie=TVBO.curie('provenance'),
                   model_uri=TVBO.network__provenance, domain=None, range=Optional[Union[dict, Provenance]])

slots.network__parent_network = Slot(uri=TVBO.parent_network, name="network__parent_network", curie=TVBO.curie('parent_network'),
                   model_uri=TVBO.network__parent_network, domain=None, range=Optional[str])

slots.network__node_mapping = Slot(uri=TVBO.node_mapping, name="network__node_mapping", curie=TVBO.curie('node_mapping'),
                   model_uri=TVBO.network__node_mapping, domain=None, range=Optional[str])

slots.network__distance_unit = Slot(uri=TVBO.distance_unit, name="network__distance_unit", curie=TVBO.curie('distance_unit'),
                   model_uri=TVBO.network__distance_unit, domain=None, range=Optional[Union[str, "UnitEnum"]])

slots.network__time_unit = Slot(uri=TVBO.time_unit, name="network__time_unit", curie=TVBO.curie('time_unit'),
                   model_uri=TVBO.network__time_unit, domain=None, range=Optional[Union[str, "UnitEnum"]])

slots.network__edge_matrix_files = Slot(uri=TVBO.edge_matrix_files, name="network__edge_matrix_files", curie=TVBO.curie('edge_matrix_files'),
                   model_uri=TVBO.network__edge_matrix_files, domain=None, range=Optional[Union[Union[str, FileName], list[Union[str, FileName]]]])

slots.network__graph_generator = Slot(uri=TVBO.graph_generator, name="network__graph_generator", curie=TVBO.curie('graph_generator'),
                   model_uri=TVBO.network__graph_generator, domain=None, range=Optional[Union[dict, GraphGenerator]])

slots.graphGenerator__type = Slot(uri=TVBO.type, name="graphGenerator__type", curie=TVBO.curie('type'),
                   model_uri=TVBO.graphGenerator__type, domain=None, range=str)

slots.graphGenerator__seed = Slot(uri=TVBO.seed, name="graphGenerator__seed", curie=TVBO.curie('seed'),
                   model_uri=TVBO.graphGenerator__seed, domain=None, range=Optional[int])

slots.graphGenerator__directed = Slot(uri=TVBO.directed, name="graphGenerator__directed", curie=TVBO.curie('directed'),
                   model_uri=TVBO.graphGenerator__directed, domain=None, range=Optional[Union[bool, Bool]])

slots.graphGenerator__parameters = Slot(uri=TVBO.parameters, name="graphGenerator__parameters", curie=TVBO.curie('parameters'),
                   model_uri=TVBO.graphGenerator__parameters, domain=None, range=Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]])

slots.file__type = Slot(uri=TVBO.type, name="file__type", curie=TVBO.curie('type'),
                   model_uri=TVBO.file__type, domain=None, range=Optional[str])

slots.file__path = Slot(uri=TVBO.path, name="file__path", curie=TVBO.curie('path'),
                   model_uri=TVBO.file__path, domain=None, range=Optional[str])

slots.file__extension = Slot(uri=TVBO.extension, name="file__extension", curie=TVBO.curie('extension'),
                   model_uri=TVBO.file__extension, domain=None, range=Optional[str])

slots.node__id = Slot(uri=TVBO.id, name="node__id", curie=TVBO.curie('id'),
                   model_uri=TVBO.node__id, domain=None, range=int)

slots.node__dynamics = Slot(uri=TVBO.dynamics, name="node__dynamics", curie=TVBO.curie('dynamics'),
                   model_uri=TVBO.node__dynamics, domain=None, range=Optional[Union[str, DynamicsName]])

slots.node__position = Slot(uri=TVBO.position, name="node__position", curie=TVBO.curie('position'),
                   model_uri=TVBO.node__position, domain=None, range=Optional[Union[dict, Coordinate]])

slots.node__region = Slot(uri=TVBO.region, name="node__region", curie=TVBO.curie('region'),
                   model_uri=TVBO.node__region, domain=None, range=Optional[str])

slots.node__state = Slot(uri=TVBO.state, name="node__state", curie=TVBO.curie('state'),
                   model_uri=TVBO.node__state, domain=None, range=Optional[Union[dict[Union[str, StateValueName], Union[dict, StateValue]], list[Union[dict, StateValue]]]])

slots.node__events = Slot(uri=TVBO.events, name="node__events", curie=TVBO.curie('events'),
                   model_uri=TVBO.node__events, domain=None, range=Optional[Union[dict[Union[str, EventName], Union[dict, Event]], list[Union[dict, Event]]]])

slots.edge__source = Slot(uri=TVBO.source, name="edge__source", curie=TVBO.curie('source'),
                   model_uri=TVBO.edge__source, domain=None, range=Optional[int])

slots.edge__target = Slot(uri=TVBO.target, name="edge__target", curie=TVBO.curie('target'),
                   model_uri=TVBO.edge__target, domain=None, range=Optional[int])

slots.edge__unit = Slot(uri=TVBO.unit, name="edge__unit", curie=TVBO.curie('unit'),
                   model_uri=TVBO.edge__unit, domain=None, range=Optional[str])

slots.edge__format = Slot(uri=TVBO.format, name="edge__format", curie=TVBO.curie('format'),
                   model_uri=TVBO.edge__format, domain=None, range=Optional[Union[str, "SparseFormat"]])

slots.edge__weighted = Slot(uri=TVBO.weighted, name="edge__weighted", curie=TVBO.curie('weighted'),
                   model_uri=TVBO.edge__weighted, domain=None, range=Optional[Union[bool, Bool]])

slots.edge__valid_diagonal = Slot(uri=TVBO.valid_diagonal, name="edge__valid_diagonal", curie=TVBO.curie('valid_diagonal'),
                   model_uri=TVBO.edge__valid_diagonal, domain=None, range=Optional[Union[bool, Bool]])

slots.edge__non_negative = Slot(uri=TVBO.non_negative, name="edge__non_negative", curie=TVBO.curie('non_negative'),
                   model_uri=TVBO.edge__non_negative, domain=None, range=Optional[Union[bool, Bool]])

slots.edge__source_var = Slot(uri=TVBO.source_var, name="edge__source_var", curie=TVBO.curie('source_var'),
                   model_uri=TVBO.edge__source_var, domain=None, range=Optional[str])

slots.edge__target_var = Slot(uri=TVBO.target_var, name="edge__target_var", curie=TVBO.curie('target_var'),
                   model_uri=TVBO.edge__target_var, domain=None, range=Optional[str])

slots.edge__coupling = Slot(uri=TVBO.coupling, name="edge__coupling", curie=TVBO.curie('coupling'),
                   model_uri=TVBO.edge__coupling, domain=None, range=Optional[Union[str, CouplingName]])

slots.edge__directed = Slot(uri=TVBO.directed, name="edge__directed", curie=TVBO.curie('directed'),
                   model_uri=TVBO.edge__directed, domain=None, range=Optional[Union[bool, Bool]])

slots.edge__target_network = Slot(uri=TVBO.target_network, name="edge__target_network", curie=TVBO.curie('target_network'),
                   model_uri=TVBO.edge__target_network, domain=None, range=Optional[str])

slots.edge__dimension_labels = Slot(uri=TVBO.dimension_labels, name="edge__dimension_labels", curie=TVBO.curie('dimension_labels'),
                   model_uri=TVBO.edge__dimension_labels, domain=None, range=Optional[Union[str, list[str]]])

slots.edge__dynamics = Slot(uri=TVBO.dynamics, name="edge__dynamics", curie=TVBO.curie('dynamics'),
                   model_uri=TVBO.edge__dynamics, domain=None, range=Optional[Union[str, DynamicsName]])

slots.edge__events = Slot(uri=TVBO.events, name="edge__events", curie=TVBO.curie('events'),
                   model_uri=TVBO.edge__events, domain=None, range=Optional[Union[dict[Union[str, EventName], Union[dict, Event]], list[Union[dict, Event]]]])

slots.observation__source = Slot(uri=TVBO.source, name="observation__source", curie=TVBO.curie('source'),
                   model_uri=TVBO.observation__source, domain=None, range=Optional[Union[str, StateVariableName]])

slots.observation__period = Slot(uri=TVBO.period, name="observation__period", curie=TVBO.curie('period'),
                   model_uri=TVBO.observation__period, domain=None, range=Optional[float])

slots.observation__downsample_period = Slot(uri=TVBO.downsample_period, name="observation__downsample_period", curie=TVBO.curie('downsample_period'),
                   model_uri=TVBO.observation__downsample_period, domain=None, range=Optional[float])

slots.observation__voi = Slot(uri=TVBO.voi, name="observation__voi", curie=TVBO.curie('voi'),
                   model_uri=TVBO.observation__voi, domain=None, range=Optional[int])

slots.observation__imaging_modality = Slot(uri=TVBO.imaging_modality, name="observation__imaging_modality", curie=TVBO.curie('imaging_modality'),
                   model_uri=TVBO.observation__imaging_modality, domain=None, range=Optional[Union[str, "ImagingModality"]])

slots.observation__warmup_source = Slot(uri=TVBO.warmup_source, name="observation__warmup_source", curie=TVBO.curie('warmup_source'),
                   model_uri=TVBO.observation__warmup_source, domain=None, range=Optional[str])

slots.observation__data_source = Slot(uri=TVBO.data_source, name="observation__data_source", curie=TVBO.curie('data_source'),
                   model_uri=TVBO.observation__data_source, domain=None, range=Optional[Union[dict, DataSource]])

slots.observation__skip_t = Slot(uri=TVBO.skip_t, name="observation__skip_t", curie=TVBO.curie('skip_t'),
                   model_uri=TVBO.observation__skip_t, domain=None, range=Optional[int])

slots.observation__tail_samples = Slot(uri=TVBO.tail_samples, name="observation__tail_samples", curie=TVBO.curie('tail_samples'),
                   model_uri=TVBO.observation__tail_samples, domain=None, range=Optional[int])

slots.observation__aggregation = Slot(uri=TVBO.aggregation, name="observation__aggregation", curie=TVBO.curie('aggregation'),
                   model_uri=TVBO.observation__aggregation, domain=None, range=Optional[Union[str, "AggregationType"]])

slots.observation__window_size = Slot(uri=TVBO.window_size, name="observation__window_size", curie=TVBO.curie('window_size'),
                   model_uri=TVBO.observation__window_size, domain=None, range=Optional[int])

slots.observation__pipeline = Slot(uri=TVBO.pipeline, name="observation__pipeline", curie=TVBO.curie('pipeline'),
                   model_uri=TVBO.observation__pipeline, domain=None, range=Optional[Union[Union[dict, FunctionCall], list[Union[dict, FunctionCall]]]])

slots.observation__class_reference = Slot(uri=TVBO.class_reference, name="observation__class_reference", curie=TVBO.curie('class_reference'),
                   model_uri=TVBO.observation__class_reference, domain=None, range=Optional[Union[dict, ClassReference]])

slots.derivedObservation__source_observations = Slot(uri=TVBO.source_observations, name="derivedObservation__source_observations", curie=TVBO.curie('source_observations'),
                   model_uri=TVBO.derivedObservation__source_observations, domain=None, range=Union[Union[str, ObservationName], list[Union[str, ObservationName]]])

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

slots.dynamics__output = Slot(uri=TVBO.output, name="dynamics__output", curie=TVBO.curie('output'),
                   model_uri=TVBO.dynamics__output, domain=None, range=Optional[Union[str, list[str]]])

slots.dynamics__derived_from_model = Slot(uri=TVBO.derived_from_model, name="dynamics__derived_from_model", curie=TVBO.curie('derived_from_model'),
                   model_uri=TVBO.dynamics__derived_from_model, domain=None, range=Optional[Union[str, DynamicsName]])

slots.dynamics__number_of_modes = Slot(uri=TVBO.number_of_modes, name="dynamics__number_of_modes", curie=TVBO.curie('number_of_modes'),
                   model_uri=TVBO.dynamics__number_of_modes, domain=None, range=Optional[int])

slots.dynamics__local_coupling_term = Slot(uri=TVBO.local_coupling_term, name="dynamics__local_coupling_term", curie=TVBO.curie('local_coupling_term'),
                   model_uri=TVBO.dynamics__local_coupling_term, domain=None, range=Optional[Union[str, ParameterName]])

slots.dynamics__functions = Slot(uri=TVBO.functions, name="dynamics__functions", curie=TVBO.curie('functions'),
                   model_uri=TVBO.dynamics__functions, domain=None, range=Optional[Union[dict[Union[str, FunctionName], Union[dict, Function]], list[Union[dict, Function]]]])

slots.dynamics__stimulus = Slot(uri=TVBO.stimulus, name="dynamics__stimulus", curie=TVBO.curie('stimulus'),
                   model_uri=TVBO.dynamics__stimulus, domain=None, range=Optional[Union[dict, Stimulus]])

slots.dynamics__modes = Slot(uri=TVBO.modes, name="dynamics__modes", curie=TVBO.curie('modes'),
                   model_uri=TVBO.dynamics__modes, domain=None, range=Optional[Union[dict[Union[str, DynamicsName], Union[dict, Dynamics]], list[Union[dict, Dynamics]]]])

slots.dynamics__model_type = Slot(uri=TVBO.model_type, name="dynamics__model_type", curie=TVBO.curie('model_type'),
                   model_uri=TVBO.dynamics__model_type, domain=None, range=Optional[Union[str, "ModelType"]])

slots.dynamics__system_type = Slot(uri=TVBO.system_type, name="dynamics__system_type", curie=TVBO.curie('system_type'),
                   model_uri=TVBO.dynamics__system_type, domain=None, range=Optional[Union[str, "SystemType"]])

slots.dynamics__autonomous = Slot(uri=TVBO.autonomous, name="dynamics__autonomous", curie=TVBO.curie('autonomous'),
                   model_uri=TVBO.dynamics__autonomous, domain=None, range=Optional[Union[bool, Bool]])

slots.dynamics__observed = Slot(uri=TVBO.observed, name="dynamics__observed", curie=TVBO.curie('observed'),
                   model_uri=TVBO.dynamics__observed, domain=None, range=Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, DerivedVariable]], list[Union[dict, DerivedVariable]]]])

slots.dynamics__events = Slot(uri=TVBO.events, name="dynamics__events", curie=TVBO.curie('events'),
                   model_uri=TVBO.dynamics__events, domain=None, range=Optional[Union[dict[Union[str, EventName], Union[dict, Event]], list[Union[dict, Event]]]])

slots.stateVariable__variable_of_interest = Slot(uri=TVBO.variable_of_interest, name="stateVariable__variable_of_interest", curie=TVBO.curie('variable_of_interest'),
                   model_uri=TVBO.stateVariable__variable_of_interest, domain=None, range=Optional[Union[bool, Bool]])

slots.stateVariable__coupling_variable = Slot(uri=TVBO.coupling_variable, name="stateVariable__coupling_variable", curie=TVBO.curie('coupling_variable'),
                   model_uri=TVBO.stateVariable__coupling_variable, domain=None, range=Optional[Union[bool, Bool]])

slots.stateVariable__equation_type = Slot(uri=TVBO.equation_type, name="stateVariable__equation_type", curie=TVBO.curie('equation_type'),
                   model_uri=TVBO.stateVariable__equation_type, domain=None, range=Optional[str])

slots.stateVariable__equation_order = Slot(uri=TVBO.equation_order, name="stateVariable__equation_order", curie=TVBO.curie('equation_order'),
                   model_uri=TVBO.stateVariable__equation_order, domain=None, range=Optional[int])

slots.stateVariable__noise = Slot(uri=TVBO.noise, name="stateVariable__noise", curie=TVBO.curie('noise'),
                   model_uri=TVBO.stateVariable__noise, domain=None, range=Optional[Union[dict, Noise]])

slots.stateVariable__stimulation_variable = Slot(uri=TVBO.stimulation_variable, name="stateVariable__stimulation_variable", curie=TVBO.curie('stimulation_variable'),
                   model_uri=TVBO.stateVariable__stimulation_variable, domain=None, range=Optional[Union[bool, Bool]])

slots.stateVariable__boundaries = Slot(uri=TVBO.boundaries, name="stateVariable__boundaries", curie=TVBO.curie('boundaries'),
                   model_uri=TVBO.stateVariable__boundaries, domain=None, range=Optional[Union[dict, Range]])

slots.stateVariable__initial_value = Slot(uri=TVBO.initial_value, name="stateVariable__initial_value", curie=TVBO.curie('initial_value'),
                   model_uri=TVBO.stateVariable__initial_value, domain=None, range=Optional[float])

slots.stateVariable__derivative_initial_value = Slot(uri=TVBO.derivative_initial_value, name="stateVariable__derivative_initial_value", curie=TVBO.curie('derivative_initial_value'),
                   model_uri=TVBO.stateVariable__derivative_initial_value, domain=None, range=Optional[float])

slots.stateVariable__distribution = Slot(uri=TVBO.distribution, name="stateVariable__distribution", curie=TVBO.curie('distribution'),
                   model_uri=TVBO.stateVariable__distribution, domain=None, range=Optional[Union[dict, Distribution]])

slots.stateVariable__history = Slot(uri=TVBO.history, name="stateVariable__history", curie=TVBO.curie('history'),
                   model_uri=TVBO.stateVariable__history, domain=None, range=Optional[Union[dict, TimeSeries]])

slots.distribution__domain = Slot(uri=TVBO.domain, name="distribution__domain", curie=TVBO.curie('domain'),
                   model_uri=TVBO.distribution__domain, domain=None, range=Optional[Union[dict, Range]])

slots.distribution__function = Slot(uri=TVBO.function, name="distribution__function", curie=TVBO.curie('function'),
                   model_uri=TVBO.distribution__function, domain=None, range=Optional[Union[dict, Function]])

slots.distribution__seed = Slot(uri=TVBO.seed, name="distribution__seed", curie=TVBO.curie('seed'),
                   model_uri=TVBO.distribution__seed, domain=None, range=Optional[int])

slots.distribution__axis = Slot(uri=TVBO.axis, name="distribution__axis", curie=TVBO.curie('axis'),
                   model_uri=TVBO.distribution__axis, domain=None, range=Optional[Union[str, "SamplingAxis"]])

slots.distribution__correlation = Slot(uri=TVBO.correlation, name="distribution__correlation", curie=TVBO.curie('correlation'),
                   model_uri=TVBO.distribution__correlation, domain=None, range=Optional[Union[dict, Matrix]])

slots.parameter__comment = Slot(uri=TVBO.comment, name="parameter__comment", curie=TVBO.curie('comment'),
                   model_uri=TVBO.parameter__comment, domain=None, range=Optional[str])

slots.parameter__heterogeneous = Slot(uri=TVBO.heterogeneous, name="parameter__heterogeneous", curie=TVBO.curie('heterogeneous'),
                   model_uri=TVBO.parameter__heterogeneous, domain=None, range=Optional[Union[bool, Bool]])

slots.parameter__distribution = Slot(uri=TVBO.distribution, name="parameter__distribution", curie=TVBO.curie('distribution'),
                   model_uri=TVBO.parameter__distribution, domain=None, range=Optional[Union[dict, Distribution]])

slots.parameter__free = Slot(uri=TVBO.free, name="parameter__free", curie=TVBO.curie('free'),
                   model_uri=TVBO.parameter__free, domain=None, range=Optional[Union[bool, Bool]])

slots.parameter__shape = Slot(uri=TVBO.shape, name="parameter__shape", curie=TVBO.curie('shape'),
                   model_uri=TVBO.parameter__shape, domain=None, range=Optional[str])

slots.parameter__explored_values = Slot(uri=TVBO.explored_values, name="parameter__explored_values", curie=TVBO.curie('explored_values'),
                   model_uri=TVBO.parameter__explored_values, domain=None, range=Optional[Union[float, list[float]]])

slots.parameter__element_domains = Slot(uri=TVBO.element_domains, name="parameter__element_domains", curie=TVBO.curie('element_domains'),
                   model_uri=TVBO.parameter__element_domains, domain=None, range=Optional[Union[Union[dict, Range], list[Union[dict, Range]]]])

slots.couplingInput__source = Slot(uri=TVBO.source, name="couplingInput__source", curie=TVBO.curie('source'),
                   model_uri=TVBO.couplingInput__source, domain=None, range=Optional[str])

slots.couplingInput__dimension = Slot(uri=TVBO.dimension, name="couplingInput__dimension", curie=TVBO.curie('dimension'),
                   model_uri=TVBO.couplingInput__dimension, domain=None, range=Optional[int])

slots.couplingInput__keys = Slot(uri=TVBO.keys, name="couplingInput__keys", curie=TVBO.curie('keys'),
                   model_uri=TVBO.couplingInput__keys, domain=None, range=Optional[Union[str, list[str]]])

slots.argument__value = Slot(uri=TVBO.value, name="argument__value", curie=TVBO.curie('value'),
                   model_uri=TVBO.argument__value, domain=None, range=Optional[str])

slots.argument__unit = Slot(uri=TVBO.unit, name="argument__unit", curie=TVBO.curie('unit'),
                   model_uri=TVBO.argument__unit, domain=None, range=Optional[str])

slots.function__input = Slot(uri=TVBO.input, name="function__input", curie=TVBO.curie('input'),
                   model_uri=TVBO.function__input, domain=None, range=Optional[Union[str, FunctionName]])

slots.function__output = Slot(uri=TVBO.output, name="function__output", curie=TVBO.curie('output'),
                   model_uri=TVBO.function__output, domain=None, range=Optional[str])

slots.function__iri = Slot(uri=TVBO.iri, name="function__iri", curie=TVBO.curie('iri'),
                   model_uri=TVBO.function__iri, domain=None, range=Optional[str])

slots.function__arguments = Slot(uri=TVBO.arguments, name="function__arguments", curie=TVBO.curie('arguments'),
                   model_uri=TVBO.function__arguments, domain=None, range=Optional[Union[dict[Union[str, ArgumentName], Union[dict, Argument]], list[Union[dict, Argument]]]])

slots.function__output_equation = Slot(uri=TVBO.output_equation, name="function__output_equation", curie=TVBO.curie('output_equation'),
                   model_uri=TVBO.function__output_equation, domain=None, range=Optional[Union[dict, Equation]])

slots.function__source_code = Slot(uri=TVBO.source_code, name="function__source_code", curie=TVBO.curie('source_code'),
                   model_uri=TVBO.function__source_code, domain=None, range=Optional[str])

slots.function__callable = Slot(uri=TVBO.callable, name="function__callable", curie=TVBO.curie('callable'),
                   model_uri=TVBO.function__callable, domain=None, range=Optional[Union[dict, Callable]])

slots.function__apply_on_dimension = Slot(uri=TVBO.apply_on_dimension, name="function__apply_on_dimension", curie=TVBO.curie('apply_on_dimension'),
                   model_uri=TVBO.function__apply_on_dimension, domain=None, range=Optional[Union[str, "DimensionType"]])

slots.function__aggregate = Slot(uri=TVBO.aggregate, name="function__aggregate", curie=TVBO.curie('aggregate'),
                   model_uri=TVBO.function__aggregate, domain=None, range=Optional[Union[dict, Aggregation]])

slots.function__time_range = Slot(uri=TVBO.time_range, name="function__time_range", curie=TVBO.curie('time_range'),
                   model_uri=TVBO.function__time_range, domain=None, range=Optional[Union[dict, Range]])

slots.aggregation__over = Slot(uri=TVBO.over, name="aggregation__over", curie=TVBO.curie('over'),
                   model_uri=TVBO.aggregation__over, domain=None, range=Optional[Union[str, "DimensionType"]])

slots.aggregation__type = Slot(uri=TVBO.type, name="aggregation__type", curie=TVBO.curie('type'),
                   model_uri=TVBO.aggregation__type, domain=None, range=Optional[Union[str, "ReductionType"]])

slots.lossFunction__aggregate = Slot(uri=TVBO.aggregate, name="lossFunction__aggregate", curie=TVBO.curie('aggregate'),
                   model_uri=TVBO.lossFunction__aggregate, domain=None, range=Optional[Union[dict, Aggregation]])

slots.functionCall__name = Slot(uri=TVBO.name, name="functionCall__name", curie=TVBO.curie('name'),
                   model_uri=TVBO.functionCall__name, domain=None, range=Optional[str])

slots.functionCall__function = Slot(uri=TVBO.function, name="functionCall__function", curie=TVBO.curie('function'),
                   model_uri=TVBO.functionCall__function, domain=None, range=Optional[Union[str, FunctionName]])

slots.functionCall__callable = Slot(uri=TVBO.callable, name="functionCall__callable", curie=TVBO.curie('callable'),
                   model_uri=TVBO.functionCall__callable, domain=None, range=Optional[Union[dict, Callable]])

slots.functionCall__class_call = Slot(uri=TVBO.class_call, name="functionCall__class_call", curie=TVBO.curie('class_call'),
                   model_uri=TVBO.functionCall__class_call, domain=None, range=Optional[Union[dict, ClassReference]])

slots.functionCall__input = Slot(uri=TVBO.input, name="functionCall__input", curie=TVBO.curie('input'),
                   model_uri=TVBO.functionCall__input, domain=None, range=Optional[str])

slots.functionCall__output = Slot(uri=TVBO.output, name="functionCall__output", curie=TVBO.curie('output'),
                   model_uri=TVBO.functionCall__output, domain=None, range=Optional[str])

slots.functionCall__apply_on_dimension = Slot(uri=TVBO.apply_on_dimension, name="functionCall__apply_on_dimension", curie=TVBO.curie('apply_on_dimension'),
                   model_uri=TVBO.functionCall__apply_on_dimension, domain=None, range=Optional[Union[str, "DimensionType"]])

slots.functionCall__aggregate = Slot(uri=TVBO.aggregate, name="functionCall__aggregate", curie=TVBO.curie('aggregate'),
                   model_uri=TVBO.functionCall__aggregate, domain=None, range=Optional[Union[dict, Aggregation]])

slots.functionCall__arguments = Slot(uri=TVBO.arguments, name="functionCall__arguments", curie=TVBO.curie('arguments'),
                   model_uri=TVBO.functionCall__arguments, domain=None, range=Optional[Union[dict[Union[str, ArgumentName], Union[dict, Argument]], list[Union[dict, Argument]]]])

slots.functionCall__time_range = Slot(uri=TVBO.time_range, name="functionCall__time_range", curie=TVBO.curie('time_range'),
                   model_uri=TVBO.functionCall__time_range, domain=None, range=Optional[Union[dict, Range]])

slots.functionCall__source_code = Slot(uri=TVBO.source_code, name="functionCall__source_code", curie=TVBO.curie('source_code'),
                   model_uri=TVBO.functionCall__source_code, domain=None, range=Optional[str])

slots.callable__module = Slot(uri=TVBO.module, name="callable__module", curie=TVBO.curie('module'),
                   model_uri=TVBO.callable__module, domain=None, range=Optional[str])

slots.callable__software = Slot(uri=TVBO.software, name="callable__software", curie=TVBO.curie('software'),
                   model_uri=TVBO.callable__software, domain=None, range=Optional[Union[dict, SoftwareRequirement]])

slots.classReference__constructor_args = Slot(uri=TVBO.constructor_args, name="classReference__constructor_args", curie=TVBO.curie('constructor_args'),
                   model_uri=TVBO.classReference__constructor_args, domain=None, range=Optional[Union[dict[Union[str, ArgumentName], Union[dict, Argument]], list[Union[dict, Argument]]]])

slots.classReference__call_args = Slot(uri=TVBO.call_args, name="classReference__call_args", curie=TVBO.curie('call_args'),
                   model_uri=TVBO.classReference__call_args, domain=None, range=Optional[Union[dict[Union[str, ArgumentName], Union[dict, Argument]], list[Union[dict, Argument]]]])

slots.classReference__warmup_source = Slot(uri=TVBO.warmup_source, name="classReference__warmup_source", curie=TVBO.curie('warmup_source'),
                   model_uri=TVBO.classReference__warmup_source, domain=None, range=Optional[str])

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

slots.dataSource__path = Slot(uri=TVBO.path, name="dataSource__path", curie=TVBO.curie('path'),
                   model_uri=TVBO.dataSource__path, domain=None, range=Optional[str])

slots.dataSource__loader = Slot(uri=TVBO.loader, name="dataSource__loader", curie=TVBO.curie('loader'),
                   model_uri=TVBO.dataSource__loader, domain=None, range=Optional[Union[dict, Callable]])

slots.dataSource__format = Slot(uri=TVBO.format, name="dataSource__format", curie=TVBO.curie('format'),
                   model_uri=TVBO.dataSource__format, domain=None, range=Optional[str])

slots.dataSource__key = Slot(uri=TVBO.key, name="dataSource__key", curie=TVBO.curie('key'),
                   model_uri=TVBO.dataSource__key, domain=None, range=Optional[str])

slots.dataSource__preprocessing = Slot(uri=TVBO.preprocessing, name="dataSource__preprocessing", curie=TVBO.curie('preprocessing'),
                   model_uri=TVBO.dataSource__preprocessing, domain=None, range=Optional[Union[dict, Function]])

slots.optimizationStage__free_parameters = Slot(uri=TVBO.free_parameters, name="optimizationStage__free_parameters", curie=TVBO.curie('free_parameters'),
                   model_uri=TVBO.optimizationStage__free_parameters, domain=None, range=Optional[Union[Union[str, ParameterName], list[Union[str, ParameterName]]]])

slots.optimizationStage__algorithm = Slot(uri=TVBO.algorithm, name="optimizationStage__algorithm", curie=TVBO.curie('algorithm'),
                   model_uri=TVBO.optimizationStage__algorithm, domain=None, range=Optional[str])

slots.optimizationStage__learning_rate = Slot(uri=TVBO.learning_rate, name="optimizationStage__learning_rate", curie=TVBO.curie('learning_rate'),
                   model_uri=TVBO.optimizationStage__learning_rate, domain=None, range=Optional[float])

slots.optimizationStage__max_iterations = Slot(uri=TVBO.max_iterations, name="optimizationStage__max_iterations", curie=TVBO.curie('max_iterations'),
                   model_uri=TVBO.optimizationStage__max_iterations, domain=None, range=Optional[int])

slots.optimizationStage__hyperparameters = Slot(uri=TVBO.hyperparameters, name="optimizationStage__hyperparameters", curie=TVBO.curie('hyperparameters'),
                   model_uri=TVBO.optimizationStage__hyperparameters, domain=None, range=Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]])

slots.optimizationStage__freeze_parameters = Slot(uri=TVBO.freeze_parameters, name="optimizationStage__freeze_parameters", curie=TVBO.curie('freeze_parameters'),
                   model_uri=TVBO.optimizationStage__freeze_parameters, domain=None, range=Optional[Union[Union[str, ParameterName], list[Union[str, ParameterName]]]])

slots.optimizationStage__warmup_from = Slot(uri=TVBO.warmup_from, name="optimizationStage__warmup_from", curie=TVBO.curie('warmup_from'),
                   model_uri=TVBO.optimizationStage__warmup_from, domain=None, range=Optional[Union[str, OptimizationStageName]])

slots.optimization__execution = Slot(uri=TVBO.execution, name="optimization__execution", curie=TVBO.curie('execution'),
                   model_uri=TVBO.optimization__execution, domain=None, range=Optional[Union[dict, ExecutionConfig]])

slots.optimization__integration = Slot(uri=TVBO.integration, name="optimization__integration", curie=TVBO.curie('integration'),
                   model_uri=TVBO.optimization__integration, domain=None, range=Optional[Union[dict, Integrator]])

slots.optimization__loss = Slot(uri=TVBO.loss, name="optimization__loss", curie=TVBO.curie('loss'),
                   model_uri=TVBO.optimization__loss, domain=None, range=Optional[Union[dict, FunctionCall]])

slots.optimization__stages = Slot(uri=TVBO.stages, name="optimization__stages", curie=TVBO.curie('stages'),
                   model_uri=TVBO.optimization__stages, domain=None, range=Optional[Union[dict[Union[str, OptimizationStageName], Union[dict, OptimizationStage]], list[Union[dict, OptimizationStage]]]])

slots.optimization__depends_on = Slot(uri=TVBO.depends_on, name="optimization__depends_on", curie=TVBO.curie('depends_on'),
                   model_uri=TVBO.optimization__depends_on, domain=None, range=Optional[Union[str, AlgorithmName]])

slots.exploration__execution = Slot(uri=TVBO.execution, name="exploration__execution", curie=TVBO.curie('execution'),
                   model_uri=TVBO.exploration__execution, domain=None, range=Optional[Union[dict, ExecutionConfig]])

slots.exploration__parameters = Slot(uri=TVBO.parameters, name="exploration__parameters", curie=TVBO.curie('parameters'),
                   model_uri=TVBO.exploration__parameters, domain=None, range=Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]])

slots.exploration__mode = Slot(uri=TVBO.mode, name="exploration__mode", curie=TVBO.curie('mode'),
                   model_uri=TVBO.exploration__mode, domain=None, range=Optional[str])

slots.exploration__observable = Slot(uri=TVBO.observable, name="exploration__observable", curie=TVBO.curie('observable'),
                   model_uri=TVBO.exploration__observable, domain=None, range=Optional[Union[dict, FunctionCall]])

slots.exploration__n_parallel = Slot(uri=TVBO.n_parallel, name="exploration__n_parallel", curie=TVBO.curie('n_parallel'),
                   model_uri=TVBO.exploration__n_parallel, domain=None, range=Optional[int])

slots.exploration__n_trials = Slot(uri=TVBO.n_trials, name="exploration__n_trials", curie=TVBO.curie('n_trials'),
                   model_uri=TVBO.exploration__n_trials, domain=None, range=Optional[int])

slots.exploration__average = Slot(uri=TVBO.average, name="exploration__average", curie=TVBO.curie('average'),
                   model_uri=TVBO.exploration__average, domain=None, range=Optional[str])

slots.updateRule__target_parameter = Slot(uri=TVBO.target_parameter, name="updateRule__target_parameter", curie=TVBO.curie('target_parameter'),
                   model_uri=TVBO.updateRule__target_parameter, domain=None, range=Union[dict, Parameter])

slots.updateRule__equation = Slot(uri=TVBO.equation, name="updateRule__equation", curie=TVBO.curie('equation'),
                   model_uri=TVBO.updateRule__equation, domain=None, range=Union[dict, Equation])

slots.updateRule__bounds = Slot(uri=TVBO.bounds, name="updateRule__bounds", curie=TVBO.curie('bounds'),
                   model_uri=TVBO.updateRule__bounds, domain=None, range=Optional[Union[dict, Range]])

slots.updateRule__warmup = Slot(uri=TVBO.warmup, name="updateRule__warmup", curie=TVBO.curie('warmup'),
                   model_uri=TVBO.updateRule__warmup, domain=None, range=Optional[Union[bool, Bool]])

slots.updateRule__requires = Slot(uri=TVBO.requires, name="updateRule__requires", curie=TVBO.curie('requires'),
                   model_uri=TVBO.updateRule__requires, domain=None, range=Optional[Union[Union[str, ObservationName], list[Union[str, ObservationName]]]])

slots.algorithmInclude__algorithm = Slot(uri=TVBO.algorithm, name="algorithmInclude__algorithm", curie=TVBO.curie('algorithm'),
                   model_uri=TVBO.algorithmInclude__algorithm, domain=None, range=Union[str, AlgorithmName])

slots.algorithmInclude__arguments = Slot(uri=TVBO.arguments, name="algorithmInclude__arguments", curie=TVBO.curie('arguments'),
                   model_uri=TVBO.algorithmInclude__arguments, domain=None, range=Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]])

slots.tuningObjective__type = Slot(uri=TVBO.type, name="tuningObjective__type", curie=TVBO.curie('type'),
                   model_uri=TVBO.tuningObjective__type, domain=None, range=Optional[str])

slots.tuningObjective__target_variable = Slot(uri=TVBO.target_variable, name="tuningObjective__target_variable", curie=TVBO.curie('target_variable'),
                   model_uri=TVBO.tuningObjective__target_variable, domain=None, range=Optional[Union[str, StateVariableName]])

slots.tuningObjective__target_value = Slot(uri=TVBO.target_value, name="tuningObjective__target_value", curie=TVBO.curie('target_value'),
                   model_uri=TVBO.tuningObjective__target_value, domain=None, range=Optional[float])

slots.tuningObjective__target_data = Slot(uri=TVBO.target_data, name="tuningObjective__target_data", curie=TVBO.curie('target_data'),
                   model_uri=TVBO.tuningObjective__target_data, domain=None, range=Optional[Union[str, ObservationName]])

slots.tuningObjective__metric = Slot(uri=TVBO.metric, name="tuningObjective__metric", curie=TVBO.curie('metric'),
                   model_uri=TVBO.tuningObjective__metric, domain=None, range=Optional[Union[dict, Equation]])

slots.algorithm__execution = Slot(uri=TVBO.execution, name="algorithm__execution", curie=TVBO.curie('execution'),
                   model_uri=TVBO.algorithm__execution, domain=None, range=Optional[Union[dict, ExecutionConfig]])

slots.algorithm__type = Slot(uri=TVBO.type, name="algorithm__type", curie=TVBO.curie('type'),
                   model_uri=TVBO.algorithm__type, domain=None, range=Optional[str])

slots.algorithm__includes = Slot(uri=TVBO.includes, name="algorithm__includes", curie=TVBO.curie('includes'),
                   model_uri=TVBO.algorithm__includes, domain=None, range=Optional[Union[Union[dict, AlgorithmInclude], list[Union[dict, AlgorithmInclude]]]])

slots.algorithm__objective = Slot(uri=TVBO.objective, name="algorithm__objective", curie=TVBO.curie('objective'),
                   model_uri=TVBO.algorithm__objective, domain=None, range=Optional[Union[dict, TuningObjective]])

slots.algorithm__observations = Slot(uri=TVBO.observations, name="algorithm__observations", curie=TVBO.curie('observations'),
                   model_uri=TVBO.algorithm__observations, domain=None, range=Optional[Union[Union[str, ObservationName], list[Union[str, ObservationName]]]])

slots.algorithm__update_rules = Slot(uri=TVBO.update_rules, name="algorithm__update_rules", curie=TVBO.curie('update_rules'),
                   model_uri=TVBO.algorithm__update_rules, domain=None, range=Optional[Union[dict[Union[str, UpdateRuleName], Union[dict, UpdateRule]], list[Union[dict, UpdateRule]]]])

slots.algorithm__hyperparameters = Slot(uri=TVBO.hyperparameters, name="algorithm__hyperparameters", curie=TVBO.curie('hyperparameters'),
                   model_uri=TVBO.algorithm__hyperparameters, domain=None, range=Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]])

slots.algorithm__learning_rate = Slot(uri=TVBO.learning_rate, name="algorithm__learning_rate", curie=TVBO.curie('learning_rate'),
                   model_uri=TVBO.algorithm__learning_rate, domain=None, range=Optional[float])

slots.algorithm__learning_rate_warmup = Slot(uri=TVBO.learning_rate_warmup, name="algorithm__learning_rate_warmup", curie=TVBO.curie('learning_rate_warmup'),
                   model_uri=TVBO.algorithm__learning_rate_warmup, domain=None, range=Optional[Union[bool, Bool]])

slots.algorithm__n_iterations = Slot(uri=TVBO.n_iterations, name="algorithm__n_iterations", curie=TVBO.curie('n_iterations'),
                   model_uri=TVBO.algorithm__n_iterations, domain=None, range=Optional[int])

slots.algorithm__learning_rate_schedule = Slot(uri=TVBO.learning_rate_schedule, name="algorithm__learning_rate_schedule", curie=TVBO.curie('learning_rate_schedule'),
                   model_uri=TVBO.algorithm__learning_rate_schedule, domain=None, range=Optional[str])

slots.algorithm__simulation_period = Slot(uri=TVBO.simulation_period, name="algorithm__simulation_period", curie=TVBO.curie('simulation_period'),
                   model_uri=TVBO.algorithm__simulation_period, domain=None, range=Optional[float])

slots.algorithm__apply_every = Slot(uri=TVBO.apply_every, name="algorithm__apply_every", curie=TVBO.curie('apply_every'),
                   model_uri=TVBO.algorithm__apply_every, domain=None, range=Optional[int])

slots.algorithm__functions = Slot(uri=TVBO.functions, name="algorithm__functions", curie=TVBO.curie('functions'),
                   model_uri=TVBO.algorithm__functions, domain=None, range=Optional[Union[Union[dict, FunctionCall], list[Union[dict, FunctionCall]]]])

slots.algorithm__depends_on = Slot(uri=TVBO.depends_on, name="algorithm__depends_on", curie=TVBO.curie('depends_on'),
                   model_uri=TVBO.algorithm__depends_on, domain=None, range=Optional[Union[Union[str, AlgorithmName], list[Union[str, AlgorithmName]]]])

slots.option__name = Slot(uri=TVBO.name, name="option__name", curie=TVBO.curie('name'),
                   model_uri=TVBO.option__name, domain=None, range=URIRef)

slots.option__value = Slot(uri=TVBO.value, name="option__value", curie=TVBO.curie('value'),
                   model_uri=TVBO.option__value, domain=None, range=str)

slots.discretization__method = Slot(uri=TVBO.method, name="discretization__method", curie=TVBO.curie('method'),
                   model_uri=TVBO.discretization__method, domain=None, range=Optional[Union[str, "NumericalDiscretizationMethod"]])

slots.discretization__ode_solver = Slot(uri=TVBO.ode_solver, name="discretization__ode_solver", curie=TVBO.curie('ode_solver'),
                   model_uri=TVBO.discretization__ode_solver, domain=None, range=Optional[Union[dict, Solver]])

slots.discretization__linear_solver = Slot(uri=TVBO.linear_solver, name="discretization__linear_solver", curie=TVBO.curie('linear_solver'),
                   model_uri=TVBO.discretization__linear_solver, domain=None, range=Optional[Union[dict, Solver]])

slots.discretization__mesh_intervals = Slot(uri=TVBO.mesh_intervals, name="discretization__mesh_intervals", curie=TVBO.curie('mesh_intervals'),
                   model_uri=TVBO.discretization__mesh_intervals, domain=None, range=Optional[int])

slots.discretization__degree = Slot(uri=TVBO.degree, name="discretization__degree", curie=TVBO.curie('degree'),
                   model_uri=TVBO.discretization__degree, domain=None, range=Optional[int])

slots.discretization__n_sections = Slot(uri=TVBO.n_sections, name="discretization__n_sections", curie=TVBO.curie('n_sections'),
                   model_uri=TVBO.discretization__n_sections, domain=None, range=Optional[int])

slots.discretization__options = Slot(uri=TVBO.options, name="discretization__options", curie=TVBO.curie('options'),
                   model_uri=TVBO.discretization__options, domain=None, range=Optional[Union[dict[Union[str, OptionName], Union[dict, Option]], list[Union[dict, Option]]]])

slots.initialState__method = Slot(uri=TVBO.method, name="initialState__method", curie=TVBO.curie('method'),
                   model_uri=TVBO.initialState__method, domain=None, range=Optional[Union[str, "InitialStateMethod"]])

slots.initialState__duration = Slot(uri=TVBO.duration, name="initialState__duration", curie=TVBO.curie('duration'),
                   model_uri=TVBO.initialState__duration, domain=None, range=Optional[float])

slots.initialState__abs_tol = Slot(uri=TVBO.abs_tol, name="initialState__abs_tol", curie=TVBO.curie('abs_tol'),
                   model_uri=TVBO.initialState__abs_tol, domain=None, range=Optional[float])

slots.initialState__rel_tol = Slot(uri=TVBO.rel_tol, name="initialState__rel_tol", curie=TVBO.curie('rel_tol'),
                   model_uri=TVBO.initialState__rel_tol, domain=None, range=Optional[float])

slots.initialState__solver = Slot(uri=TVBO.solver, name="initialState__solver", curie=TVBO.curie('solver'),
                   model_uri=TVBO.initialState__solver, domain=None, range=Optional[Union[dict, Solver]])

slots.initialState__source_branch = Slot(uri=TVBO.source_branch, name="initialState__source_branch", curie=TVBO.curie('source_branch'),
                   model_uri=TVBO.initialState__source_branch, domain=None, range=Optional[str])

slots.initialState__source_point = Slot(uri=TVBO.source_point, name="initialState__source_point", curie=TVBO.curie('source_point'),
                   model_uri=TVBO.initialState__source_point, domain=None, range=Optional[str])

slots.branchSwitch__source_point = Slot(uri=TVBO.source_point, name="branchSwitch__source_point", curie=TVBO.curie('source_point'),
                   model_uri=TVBO.branchSwitch__source_point, domain=None, range=Optional[str])

slots.branchSwitch__delta_p = Slot(uri=TVBO.delta_p, name="branchSwitch__delta_p", curie=TVBO.curie('delta_p'),
                   model_uri=TVBO.branchSwitch__delta_p, domain=None, range=Optional[float])

slots.branchSwitch__continuation = Slot(uri=TVBO.continuation, name="branchSwitch__continuation", curie=TVBO.curie('continuation'),
                   model_uri=TVBO.branchSwitch__continuation, domain=None, range=Optional[Union[dict, Continuation]])

slots.branchSwitch__discretization = Slot(uri=TVBO.discretization, name="branchSwitch__discretization", curie=TVBO.curie('discretization'),
                   model_uri=TVBO.branchSwitch__discretization, domain=None, range=Optional[Union[dict, Discretization]])

slots.branchSwitch__bothside = Slot(uri=TVBO.bothside, name="branchSwitch__bothside", curie=TVBO.curie('bothside'),
                   model_uri=TVBO.branchSwitch__bothside, domain=None, range=Optional[Union[bool, Bool]])

slots.branchSwitch__options = Slot(uri=TVBO.options, name="branchSwitch__options", curie=TVBO.curie('options'),
                   model_uri=TVBO.branchSwitch__options, domain=None, range=Optional[Union[dict[Union[str, OptionName], Union[dict, Option]], list[Union[dict, Option]]]])

slots.continuation__dynamics = Slot(uri=TVBO.dynamics, name="continuation__dynamics", curie=TVBO.curie('dynamics'),
                   model_uri=TVBO.continuation__dynamics, domain=None, range=Optional[Union[str, DynamicsName]])

slots.continuation__free_parameters = Slot(uri=TVBO.free_parameters, name="continuation__free_parameters", curie=TVBO.curie('free_parameters'),
                   model_uri=TVBO.continuation__free_parameters, domain=None, range=Optional[Union[dict[Union[str, ParameterName], Union[dict, Parameter]], list[Union[dict, Parameter]]]])

slots.continuation__ds = Slot(uri=TVBO.ds, name="continuation__ds", curie=TVBO.curie('ds'),
                   model_uri=TVBO.continuation__ds, domain=None, range=Optional[float])

slots.continuation__ds_min = Slot(uri=TVBO.ds_min, name="continuation__ds_min", curie=TVBO.curie('ds_min'),
                   model_uri=TVBO.continuation__ds_min, domain=None, range=Optional[float])

slots.continuation__ds_max = Slot(uri=TVBO.ds_max, name="continuation__ds_max", curie=TVBO.curie('ds_max'),
                   model_uri=TVBO.continuation__ds_max, domain=None, range=Optional[float])

slots.continuation__max_steps = Slot(uri=TVBO.max_steps, name="continuation__max_steps", curie=TVBO.curie('max_steps'),
                   model_uri=TVBO.continuation__max_steps, domain=None, range=Optional[int])

slots.continuation__newton_tol = Slot(uri=TVBO.newton_tol, name="continuation__newton_tol", curie=TVBO.curie('newton_tol'),
                   model_uri=TVBO.continuation__newton_tol, domain=None, range=Optional[float])

slots.continuation__newton_max_iterations = Slot(uri=TVBO.newton_max_iterations, name="continuation__newton_max_iterations", curie=TVBO.curie('newton_max_iterations'),
                   model_uri=TVBO.continuation__newton_max_iterations, domain=None, range=Optional[int])

slots.continuation__nev = Slot(uri=TVBO.nev, name="continuation__nev", curie=TVBO.curie('nev'),
                   model_uri=TVBO.continuation__nev, domain=None, range=Optional[int])

slots.continuation__tol_stability = Slot(uri=TVBO.tol_stability, name="continuation__tol_stability", curie=TVBO.curie('tol_stability'),
                   model_uri=TVBO.continuation__tol_stability, domain=None, range=Optional[float])

slots.continuation__detect_bifurcation = Slot(uri=TVBO.detect_bifurcation, name="continuation__detect_bifurcation", curie=TVBO.curie('detect_bifurcation'),
                   model_uri=TVBO.continuation__detect_bifurcation, domain=None, range=Optional[int])

slots.continuation__detect_fold = Slot(uri=TVBO.detect_fold, name="continuation__detect_fold", curie=TVBO.curie('detect_fold'),
                   model_uri=TVBO.continuation__detect_fold, domain=None, range=Optional[Union[bool, Bool]])

slots.continuation__n_inversion = Slot(uri=TVBO.n_inversion, name="continuation__n_inversion", curie=TVBO.curie('n_inversion'),
                   model_uri=TVBO.continuation__n_inversion, domain=None, range=Optional[int])

slots.continuation__max_bisection_steps = Slot(uri=TVBO.max_bisection_steps, name="continuation__max_bisection_steps", curie=TVBO.curie('max_bisection_steps'),
                   model_uri=TVBO.continuation__max_bisection_steps, domain=None, range=Optional[int])

slots.continuation__algorithm = Slot(uri=TVBO.algorithm, name="continuation__algorithm", curie=TVBO.curie('algorithm'),
                   model_uri=TVBO.continuation__algorithm, domain=None, range=Optional[Union[str, "ContinuationAlgorithm"]])

slots.continuation__initial_state = Slot(uri=TVBO.initial_state, name="continuation__initial_state", curie=TVBO.curie('initial_state'),
                   model_uri=TVBO.continuation__initial_state, domain=None, range=Optional[Union[dict, InitialState]])

slots.continuation__branches = Slot(uri=TVBO.branches, name="continuation__branches", curie=TVBO.curie('branches'),
                   model_uri=TVBO.continuation__branches, domain=None, range=Optional[Union[dict[Union[str, BranchSwitchName], Union[dict, BranchSwitch]], list[Union[dict, BranchSwitch]]]])

slots.continuation__bothside = Slot(uri=TVBO.bothside, name="continuation__bothside", curie=TVBO.curie('bothside'),
                   model_uri=TVBO.continuation__bothside, domain=None, range=Optional[Union[bool, Bool]])

slots.continuation__execution = Slot(uri=TVBO.execution, name="continuation__execution", curie=TVBO.curie('execution'),
                   model_uri=TVBO.continuation__execution, domain=None, range=Optional[Union[dict, ExecutionConfig]])

slots.continuation__software = Slot(uri=TVBO.software, name="continuation__software", curie=TVBO.curie('software'),
                   model_uri=TVBO.continuation__software, domain=None, range=Optional[Union[dict, SoftwareRequirement]])

slots.continuation__options = Slot(uri=TVBO.options, name="continuation__options", curie=TVBO.curie('options'),
                   model_uri=TVBO.continuation__options, domain=None, range=Optional[Union[dict[Union[str, OptionName], Union[dict, Option]], list[Union[dict, Option]]]])

slots.solver__method = Slot(uri=TVBO.method, name="solver__method", curie=TVBO.curie('method'),
                   model_uri=TVBO.solver__method, domain=None, range=Optional[str])

slots.solver__abs_tol = Slot(uri=TVBO.abs_tol, name="solver__abs_tol", curie=TVBO.curie('abs_tol'),
                   model_uri=TVBO.solver__abs_tol, domain=None, range=Optional[float])

slots.solver__rel_tol = Slot(uri=TVBO.rel_tol, name="solver__rel_tol", curie=TVBO.curie('rel_tol'),
                   model_uri=TVBO.solver__rel_tol, domain=None, range=Optional[float])

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

slots.coupling__coupling_function = Slot(uri=TVBO.coupling_function, name="coupling__coupling_function", curie=TVBO.curie('coupling_function'),
                   model_uri=TVBO.coupling__coupling_function, domain=None, range=Optional[Union[dict, Equation]])

slots.coupling__sparse = Slot(uri=TVBO.sparse, name="coupling__sparse", curie=TVBO.curie('sparse'),
                   model_uri=TVBO.coupling__sparse, domain=None, range=Optional[Union[bool, Bool]])

slots.coupling__pre_expression = Slot(uri=TVBO.pre_expression, name="coupling__pre_expression", curie=TVBO.curie('pre_expression'),
                   model_uri=TVBO.coupling__pre_expression, domain=None, range=Optional[Union[dict, Equation]])

slots.coupling__post_expression = Slot(uri=TVBO.post_expression, name="coupling__post_expression", curie=TVBO.curie('post_expression'),
                   model_uri=TVBO.coupling__post_expression, domain=None, range=Optional[Union[dict, Equation]])

slots.coupling__incoming_states = Slot(uri=TVBO.incoming_states, name="coupling__incoming_states", curie=TVBO.curie('incoming_states'),
                   model_uri=TVBO.coupling__incoming_states, domain=None, range=Optional[Union[Union[str, StateVariableName], list[Union[str, StateVariableName]]]])

slots.coupling__local_states = Slot(uri=TVBO.local_states, name="coupling__local_states", curie=TVBO.curie('local_states'),
                   model_uri=TVBO.coupling__local_states, domain=None, range=Optional[Union[Union[str, StateVariableName], list[Union[str, StateVariableName]]]])

slots.coupling__delayed = Slot(uri=TVBO.delayed, name="coupling__delayed", curie=TVBO.curie('delayed'),
                   model_uri=TVBO.coupling__delayed, domain=None, range=Optional[Union[bool, Bool]])

slots.coupling__symmetry = Slot(uri=TVBO.symmetry, name="coupling__symmetry", curie=TVBO.curie('symmetry'),
                   model_uri=TVBO.coupling__symmetry, domain=None, range=Optional[str])

slots.coupling__outsym = Slot(uri=TVBO.outsym, name="coupling__outsym", curie=TVBO.curie('outsym'),
                   model_uri=TVBO.coupling__outsym, domain=None, range=Optional[Union[str, list[str]]])

slots.coupling__observed = Slot(uri=TVBO.observed, name="coupling__observed", curie=TVBO.curie('observed'),
                   model_uri=TVBO.coupling__observed, domain=None, range=Optional[Union[dict[Union[str, DerivedVariableName], Union[dict, DerivedVariable]], list[Union[dict, DerivedVariable]]]])

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

slots.executionConfig__n_workers = Slot(uri=TVBO.n_workers, name="executionConfig__n_workers", curie=TVBO.curie('n_workers'),
                   model_uri=TVBO.executionConfig__n_workers, domain=None, range=Optional[int])

slots.executionConfig__n_threads = Slot(uri=TVBO.n_threads, name="executionConfig__n_threads", curie=TVBO.curie('n_threads'),
                   model_uri=TVBO.executionConfig__n_threads, domain=None, range=Optional[int])

slots.executionConfig__precision = Slot(uri=TVBO.precision, name="executionConfig__precision", curie=TVBO.curie('precision'),
                   model_uri=TVBO.executionConfig__precision, domain=None, range=Optional[str])

slots.executionConfig__accelerator = Slot(uri=TVBO.accelerator, name="executionConfig__accelerator", curie=TVBO.curie('accelerator'),
                   model_uri=TVBO.executionConfig__accelerator, domain=None, range=Optional[str])

slots.executionConfig__batch_size = Slot(uri=TVBO.batch_size, name="executionConfig__batch_size", curie=TVBO.curie('batch_size'),
                   model_uri=TVBO.executionConfig__batch_size, domain=None, range=Optional[int])

slots.executionConfig__random_seed = Slot(uri=TVBO.random_seed, name="executionConfig__random_seed", curie=TVBO.curie('random_seed'),
                   model_uri=TVBO.executionConfig__random_seed, domain=None, range=Optional[int])

slots.executionConfig__find_fixpoint = Slot(uri=TVBO.find_fixpoint, name="executionConfig__find_fixpoint", curie=TVBO.curie('find_fixpoint'),
                   model_uri=TVBO.executionConfig__find_fixpoint, domain=None, range=Optional[Union[bool, Bool]])

slots.simulationExperiment__id = Slot(uri=TVBO.id, name="simulationExperiment__id", curie=TVBO.curie('id'),
                   model_uri=TVBO.simulationExperiment__id, domain=None, range=URIRef)

slots.simulationExperiment__description = Slot(uri=TVBO.description, name="simulationExperiment__description", curie=TVBO.curie('description'),
                   model_uri=TVBO.simulationExperiment__description, domain=None, range=Optional[str])

slots.simulationExperiment__additional_equations = Slot(uri=TVBO.additional_equations, name="simulationExperiment__additional_equations", curie=TVBO.curie('additional_equations'),
                   model_uri=TVBO.simulationExperiment__additional_equations, domain=None, range=Optional[Union[Union[dict, Equation], list[Union[dict, Equation]]]])

slots.simulationExperiment__label = Slot(uri=TVBO.label, name="simulationExperiment__label", curie=TVBO.curie('label'),
                   model_uri=TVBO.simulationExperiment__label, domain=None, range=Optional[str])

slots.simulationExperiment__dynamics = Slot(uri=TVBO.dynamics, name="simulationExperiment__dynamics", curie=TVBO.curie('dynamics'),
                   model_uri=TVBO.simulationExperiment__dynamics, domain=None, range=Optional[Union[dict, Dynamics]])

slots.simulationExperiment__integration = Slot(uri=TVBO.integration, name="simulationExperiment__integration", curie=TVBO.curie('integration'),
                   model_uri=TVBO.simulationExperiment__integration, domain=None, range=Optional[Union[dict, Integrator]])

slots.simulationExperiment__connectivity = Slot(uri=TVBO.connectivity, name="simulationExperiment__connectivity", curie=TVBO.curie('connectivity'),
                   model_uri=TVBO.simulationExperiment__connectivity, domain=None, range=Optional[Union[dict, Network]])

slots.simulationExperiment__network = Slot(uri=TVBO.network, name="simulationExperiment__network", curie=TVBO.curie('network'),
                   model_uri=TVBO.simulationExperiment__network, domain=None, range=Optional[Union[dict, Network]])

slots.simulationExperiment__coupling = Slot(uri=TVBO.coupling, name="simulationExperiment__coupling", curie=TVBO.curie('coupling'),
                   model_uri=TVBO.simulationExperiment__coupling, domain=None, range=Optional[Union[dict, Coupling]])

slots.simulationExperiment__observations = Slot(uri=TVBO.observations, name="simulationExperiment__observations", curie=TVBO.curie('observations'),
                   model_uri=TVBO.simulationExperiment__observations, domain=None, range=Optional[Union[dict[Union[str, ObservationName], Union[dict, Observation]], list[Union[dict, Observation]]]])

slots.simulationExperiment__derived_observations = Slot(uri=TVBO.derived_observations, name="simulationExperiment__derived_observations", curie=TVBO.curie('derived_observations'),
                   model_uri=TVBO.simulationExperiment__derived_observations, domain=None, range=Optional[Union[dict[Union[str, DerivedObservationName], Union[dict, DerivedObservation]], list[Union[dict, DerivedObservation]]]])

slots.simulationExperiment__functions = Slot(uri=TVBO.functions, name="simulationExperiment__functions", curie=TVBO.curie('functions'),
                   model_uri=TVBO.simulationExperiment__functions, domain=None, range=Optional[Union[dict[Union[str, FunctionName], Union[dict, Function]], list[Union[dict, Function]]]])

slots.simulationExperiment__stimulation = Slot(uri=TVBO.stimulation, name="simulationExperiment__stimulation", curie=TVBO.curie('stimulation'),
                   model_uri=TVBO.simulationExperiment__stimulation, domain=None, range=Optional[Union[dict, Stimulus]])

slots.simulationExperiment__events = Slot(uri=TVBO.events, name="simulationExperiment__events", curie=TVBO.curie('events'),
                   model_uri=TVBO.simulationExperiment__events, domain=None, range=Optional[Union[dict[Union[str, EventName], Union[dict, Event]], list[Union[dict, Event]]]])

slots.simulationExperiment__field_dynamics = Slot(uri=TVBO.field_dynamics, name="simulationExperiment__field_dynamics", curie=TVBO.curie('field_dynamics'),
                   model_uri=TVBO.simulationExperiment__field_dynamics, domain=None, range=Optional[Union[dict, PDE]])

slots.simulationExperiment__optimizations = Slot(uri=TVBO.optimizations, name="simulationExperiment__optimizations", curie=TVBO.curie('optimizations'),
                   model_uri=TVBO.simulationExperiment__optimizations, domain=None, range=Optional[Union[dict[Union[str, OptimizationName], Union[dict, Optimization]], list[Union[dict, Optimization]]]])

slots.simulationExperiment__explorations = Slot(uri=TVBO.explorations, name="simulationExperiment__explorations", curie=TVBO.curie('explorations'),
                   model_uri=TVBO.simulationExperiment__explorations, domain=None, range=Optional[Union[dict[Union[str, ExplorationName], Union[dict, Exploration]], list[Union[dict, Exploration]]]])

slots.simulationExperiment__algorithms = Slot(uri=TVBO.algorithms, name="simulationExperiment__algorithms", curie=TVBO.curie('algorithms'),
                   model_uri=TVBO.simulationExperiment__algorithms, domain=None, range=Optional[Union[dict[Union[str, AlgorithmName], Union[dict, Algorithm]], list[Union[dict, Algorithm]]]])

slots.simulationExperiment__continuations = Slot(uri=TVBO.continuations, name="simulationExperiment__continuations", curie=TVBO.curie('continuations'),
                   model_uri=TVBO.simulationExperiment__continuations, domain=None, range=Optional[Union[dict[Union[str, ContinuationName], Union[dict, Continuation]], list[Union[dict, Continuation]]]])

slots.simulationExperiment__environment = Slot(uri=TVBO.environment, name="simulationExperiment__environment", curie=TVBO.curie('environment'),
                   model_uri=TVBO.simulationExperiment__environment, domain=None, range=Optional[Union[dict, SoftwareEnvironment]])

slots.simulationExperiment__execution = Slot(uri=TVBO.execution, name="simulationExperiment__execution", curie=TVBO.curie('execution'),
                   model_uri=TVBO.simulationExperiment__execution, domain=None, range=Optional[Union[dict, ExecutionConfig]])

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

slots.simulationStudy__simulation_experiments = Slot(uri=TVBO.experiments, name="simulationStudy__simulation_experiments", curie=TVBO.curie('experiments'),
                   model_uri=TVBO.simulationStudy__simulation_experiments, domain=None, range=Optional[Union[dict[Union[int, SimulationExperimentId], Union[dict, SimulationExperiment]], list[Union[dict, SimulationExperiment]]]])

slots.timeSeries__data = Slot(uri=TVBO.data, name="timeSeries__data", curie=TVBO.curie('data'),
                   model_uri=TVBO.timeSeries__data, domain=None, range=Optional[Union[dict, Matrix]])

slots.timeSeries__time = Slot(uri=TVBO.time, name="timeSeries__time", curie=TVBO.curie('time'),
                   model_uri=TVBO.timeSeries__time, domain=None, range=Optional[Union[dict, Matrix]])

slots.timeSeries__sampling_rate = Slot(uri=TVBO.sampling_rate, name="timeSeries__sampling_rate", curie=TVBO.curie('sampling_rate'),
                   model_uri=TVBO.timeSeries__sampling_rate, domain=None, range=Optional[float])

slots.timeSeries__sampling_period = Slot(uri=TVBO.sampling_period, name="timeSeries__sampling_period", curie=TVBO.curie('sampling_period'),
                   model_uri=TVBO.timeSeries__sampling_period, domain=None, range=Optional[float])

slots.timeSeries__sampling_period_unit = Slot(uri=TVBO.sampling_period_unit, name="timeSeries__sampling_period_unit", curie=TVBO.curie('sampling_period_unit'),
                   model_uri=TVBO.timeSeries__sampling_period_unit, domain=None, range=Optional[str])

slots.timeSeries__unit = Slot(uri=TVBO.unit, name="timeSeries__unit", curie=TVBO.curie('unit'),
                   model_uri=TVBO.timeSeries__unit, domain=None, range=Optional[str])

slots.timeSeries__labels_ordering = Slot(uri=TVBO.labels_ordering, name="timeSeries__labels_ordering", curie=TVBO.curie('labels_ordering'),
                   model_uri=TVBO.timeSeries__labels_ordering, domain=None, range=Optional[Union[str, list[str]]])

slots.timeSeries__labels_dimensions = Slot(uri=TVBO.labels_dimensions, name="timeSeries__labels_dimensions", curie=TVBO.curie('labels_dimensions'),
                   model_uri=TVBO.timeSeries__labels_dimensions, domain=None, range=Optional[str])

slots.timeSeries__source_experiment = Slot(uri=TVBO.source_experiment, name="timeSeries__source_experiment", curie=TVBO.curie('source_experiment'),
                   model_uri=TVBO.timeSeries__source_experiment, domain=None, range=Optional[Union[int, SimulationExperimentId]])

slots.timeSeries__generated_at = Slot(uri=TVBO.generated_at, name="timeSeries__generated_at", curie=TVBO.curie('generated_at'),
                   model_uri=TVBO.timeSeries__generated_at, domain=None, range=Optional[Union[str, XSDDateTime]])

slots.timeSeries__software_environment = Slot(uri=TVBO.software_environment, name="timeSeries__software_environment", curie=TVBO.curie('software_environment'),
                   model_uri=TVBO.timeSeries__software_environment, domain=None, range=Optional[Union[dict, SoftwareEnvironment]])

slots.timeSeries__task_name = Slot(uri=TVBO.task_name, name="timeSeries__task_name", curie=TVBO.curie('task_name'),
                   model_uri=TVBO.timeSeries__task_name, domain=None, range=Optional[str])

slots.timeSeries__subject_id = Slot(uri=TVBO.subject_id, name="timeSeries__subject_id", curie=TVBO.curie('subject_id'),
                   model_uri=TVBO.timeSeries__subject_id, domain=None, range=Optional[str])

slots.timeSeries__session_id = Slot(uri=TVBO.session_id, name="timeSeries__session_id", curie=TVBO.curie('session_id'),
                   model_uri=TVBO.timeSeries__session_id, domain=None, range=Optional[str])

slots.timeSeries__run_id = Slot(uri=TVBO.run_id, name="timeSeries__run_id", curie=TVBO.curie('run_id'),
                   model_uri=TVBO.timeSeries__run_id, domain=None, range=Optional[int])

slots.timeSeries__modality = Slot(uri=TVBO.modality, name="timeSeries__modality", curie=TVBO.curie('modality'),
                   model_uri=TVBO.timeSeries__modality, domain=None, range=Optional[Union[str, "ImagingModality"]])

slots.timeSeries__model_equation_ref = Slot(uri=TVBO.model_equation_ref, name="timeSeries__model_equation_ref", curie=TVBO.curie('model_equation_ref'),
                   model_uri=TVBO.timeSeries__model_equation_ref, domain=None, range=Optional[str])

slots.timeSeries__model_param_ref = Slot(uri=TVBO.model_param_ref, name="timeSeries__model_param_ref", curie=TVBO.curie('model_param_ref'),
                   model_uri=TVBO.timeSeries__model_param_ref, domain=None, range=Optional[str])

slots.timeSeries__connectivity_ref = Slot(uri=TVBO.connectivity_ref, name="timeSeries__connectivity_ref", curie=TVBO.curie('connectivity_ref'),
                   model_uri=TVBO.timeSeries__connectivity_ref, domain=None, range=Optional[str])

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
                   model_uri=TVBO.softwareRequirement__package, domain=None, range=Optional[Union[str, SoftwarePackageName]])

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

slots.mesh__mesh_file = Slot(uri=TVBO.mesh_file, name="mesh__mesh_file", curie=TVBO.curie('mesh_file'),
                   model_uri=TVBO.mesh__mesh_file, domain=None, range=Optional[str])

slots.mesh__mesh_format = Slot(uri=TVBO.mesh_format, name="mesh__mesh_format", curie=TVBO.curie('mesh_format'),
                   model_uri=TVBO.mesh__mesh_format, domain=None, range=Optional[str])

slots.mesh__number_of_vertices = Slot(uri=TVBO.number_of_vertices, name="mesh__number_of_vertices", curie=TVBO.curie('number_of_vertices'),
                   model_uri=TVBO.mesh__number_of_vertices, domain=None, range=Optional[int])

slots.mesh__number_of_elements = Slot(uri=TVBO.number_of_elements, name="mesh__number_of_elements", curie=TVBO.curie('number_of_elements'),
                   model_uri=TVBO.mesh__number_of_elements, domain=None, range=Optional[int])

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

slots.pDESolver__dt = Slot(uri=TVBO.dt, name="pDESolver__dt", curie=TVBO.curie('dt'),
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
                   model_uri=TVBO.pDE__state_variables, domain=None, range=Optional[Union[dict[Union[str, FieldStateVariableName], Union[dict, FieldStateVariable]], list[Union[dict, FieldStateVariable]]]])

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
                   model_uri=TVBO.parcellationTerminology__entities, domain=None, range=Optional[Union[dict[Union[str, ParcellationEntityName], Union[dict, ParcellationEntity]], list[Union[dict, ParcellationEntity]]]])

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

slots.Distribution_name = Slot(uri=TVBO.name, name="Distribution_name", curie=TVBO.curie('name'),
                   model_uri=TVBO.Distribution_name, domain=Distribution, range=Union[str, DistributionName])

slots.Coupling_name = Slot(uri=TVBO.name, name="Coupling_name", curie=TVBO.curie('name'),
                   model_uri=TVBO.Coupling_name, domain=Coupling, range=Union[str, CouplingName])

