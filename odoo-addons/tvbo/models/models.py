# -*- coding: utf-8 -*-
# Generated from TVBO schemas
from odoo import models, fields, api


class Range(models.Model):
    _name = 'tvbo.range'
    _description = 'Range'


    lo = fields.Float()
    hi = fields.Float()
    step = fields.Float()


class Equation(models.Model):
    _name = 'tvbo.equation'
    _description = 'Equation'

    _rec_name = 'label'

    label = fields.Char(index=True)
    lefthandside = fields.Char()
    righthandside = fields.Char()
    conditionals = fields.One2many(comodel_name='tvbo.conditional_block', string='Conditional logic for piecewise equations.')
    engine = fields.Many2one(comodel_name='tvbo.software_requirement', string="Primary engine (must appear in environment.requirements; migration target replacing deprecated 'software').")
    pycode = fields.Char(string='Python code for the equation.')
    latex = fields.Boolean()


class ConditionalBlock(models.Model):
    _name = 'tvbo.conditional_block'
    _description = 'A single condition and its corresponding equation segment.'


    condition = fields.Char(string='The condition for this block (e.g., t > onset).')
    expression = fields.Char(string='The equation to apply when the condition is met.')


class Stimulus(models.Model):
    _name = 'tvbo.stimulus'
    _description = 'Stimulus'

    _rec_name = 'label'

    description = fields.Text()
    label = fields.Char(index=True)
    regions = fields.Text()
    weighting = fields.Text()


class TemporalApplicableEquation(models.Model):
    _name = 'tvbo.temporal_applicable_equation'
    _description = 'TemporalApplicableEquation'


    time_dependent = fields.Boolean()


class Parcellation(models.Model):
    _name = 'tvbo.parcellation'
    _description = 'Parcellation'

    _rec_name = 'label'

    label = fields.Char(index=True)
    region_labels = fields.Text()
    center_coordinates = fields.Text()
    data_source = fields.Char()
    atlas = fields.Many2one(comodel_name='tvbo.brain_atlas', required=True)


class Matrix(models.Model):
    _name = 'tvbo.matrix'
    _description = 'Adjacency matrix of a network.'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    x = fields.Many2one(comodel_name='tvbo.brain_region_series')
    y = fields.Many2one(comodel_name='tvbo.brain_region_series')
    values = fields.Text()


class BrainRegionSeries(models.Model):
    _name = 'tvbo.brain_region_series'
    _description = 'A series whose values represent latitude'


    values = fields.Text()


class Connectome(models.Model):
    _name = 'tvbo.connectome'
    _description = 'Connectome'


    number_of_regions = fields.Integer(default=1)
    number_of_nodes = fields.Integer(default=1)
    parcellation = fields.Many2one(comodel_name='tvbo.parcellation')
    tractogram = fields.Char()
    weights = fields.Many2one(comodel_name='tvbo.matrix')
    lengths = fields.Many2one(comodel_name='tvbo.matrix')
    normalization = fields.Many2one(comodel_name='tvbo.equation')
    conduction_speed = fields.Many2one(comodel_name='tvbo.parameter')
    node_labels = fields.Text()


class Network(models.Model):
    _name = 'tvbo.network'
    _description = 'Complete network specification combining dynamics, graph topology, and coupling configurations'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    dynamics = fields.Many2one(comodel_name='tvbo.dynamics', string='Homogeneous dynamics: single Dynamics model applied to all nodes')
    node_dynamics = fields.One2many(comodel_name='tvbo.dynamics', string='Heterogeneous dynamics: list of Dynamics models, one per node or mapped by node_dynamics_mapping')
    node_dynamics_mapping = fields.Text(string='Maps each node to a Dynamics model index in node_dynamics list (if heterogeneous)')
    graph = fields.Many2one(comodel_name='tvbo.connectome', string='Network topology with weights, delays, and connectivity structure', required=True)
    couplings = fields.One2many(comodel_name='tvbo.coupling', string='Named coupling configurations matching dynamics.coupling_inputs (e.g., instant, delayed)')


class ObservationModel(models.Model):
    _name = 'tvbo.observation_model'
    _description = 'ObservationModel'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    acronym = fields.Char()
    description = fields.Text()
    transformation = fields.Many2one(comodel_name='tvbo.function')
    pipeline = fields.One2many(comodel_name='tvbo.processing_step', string='Ordered sequence of processing functions')
    data_injections = fields.One2many(comodel_name='tvbo.data_injection', string='External data added to the pipeline (e.g., timepoints, kernels)')
    argument_mappings = fields.One2many(comodel_name='tvbo.argument_mapping', string='How inputs/outputs connect between pipeline steps')
    derivatives = fields.One2many(comodel_name='tvbo.derived_variable', string='Side computations (e.g., functional connectivity)')


class ProcessingStep(models.Model):
    _name = 'tvbo.processing_step'
    _description = 'A single processing step in an observation model pipeline or standalone operation'


    order = fields.Integer(string='Execution order in the pipeline (optional for standalone operations)')
    function = fields.Many2one(comodel_name='tvbo.function', string='Function or transformation to apply', required=True)
    operation_type = fields.Many2one(comodel_name='tvbo.operation_type', string='Kind of operation to perform (e.g., subsample, projection, convolution).')
    input_mapping = fields.One2many(comodel_name='tvbo.argument_mapping', string='Maps function arguments to pipeline data/outputs')
    output_alias = fields.Char(string="Optional name for this step's output (default: function name)")
    apply_on_dimension = fields.Char(string="Which dimension to apply function on (e.g., 'time', 'space')")
    ensure_shape = fields.Char(string="Ensure output has specific dimensionality (e.g., '4d')")
    variables_of_interest = fields.One2many(comodel_name='tvbo.state_variable', string='Optional per-step variable selection')


class DataInjection(models.Model):
    _name = 'tvbo.data_injection'
    _description = 'External data injected into the observation pipeline'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    data_source = fields.Char(string='Source of the data (file, array, generated)')
    values = fields.Text(string='Actual data values (for small arrays)')
    shape = fields.Text(string='Shape of the injected data')
    generation_function = fields.Many2one(comodel_name='tvbo.function', string='Function to generate the data (e.g., np.arange)')


class ArgumentMapping(models.Model):
    _name = 'tvbo.argument_mapping'
    _description = 'Maps function arguments to pipeline inputs/outputs'


    function_argument = fields.Char(string='Name of the function parameter', required=True)
    source = fields.Char(string="Where the data comes from (e.g., 'Input', 'subsample', 'HRF')", required=True)
    constant_value = fields.Char(string='Use a constant value instead of pipeline data')


class DownsamplingModel(models.Model):
    _name = 'tvbo.downsampling_model'
    _description = 'DownsamplingModel'


    period = fields.Float(default=0.9765625)


class Dynamics(models.Model):
    _name = 'tvbo.dynamics'
    _description = 'Dynamics'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    label = fields.Char(index=True)
    description = fields.Text()
    derived_parameters = fields.One2many(comodel_name='tvbo.derived_parameter')
    derived_variables = fields.One2many(comodel_name='tvbo.derived_variable')
    coupling_terms = fields.One2many(comodel_name='tvbo.parameter')
    coupling_inputs = fields.One2many(comodel_name='tvbo.coupling_input')
    state_variables = fields.One2many(comodel_name='tvbo.state_variable')
    modified = fields.Boolean()
    output_transforms = fields.One2many(comodel_name='tvbo.derived_variable')
    derived_from_model = fields.Many2one(comodel_name='tvbo.neural_mass_model')
    number_of_modes = fields.Integer(default=1)
    local_coupling_term = fields.Many2one(comodel_name='tvbo.parameter')
    functions = fields.One2many(comodel_name='tvbo.function')
    stimulus = fields.Many2one(comodel_name='tvbo.stimulus')
    modes = fields.One2many(comodel_name='tvbo.neural_mass_model')
    system_type = fields.Many2one(comodel_name='tvbo.system_type')


class NeuralMassModel(models.Model):
    _name = 'tvbo.neural_mass_model'
    _description = 'NeuralMassModel'




class StateVariable(models.Model):
    _name = 'tvbo.state_variable'
    _description = 'StateVariable'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    label = fields.Char(index=True)
    description = fields.Text()
    variable_of_interest = fields.Boolean()
    coupling_variable = fields.Boolean()
    noise = fields.Many2one(comodel_name='tvbo.noise')
    stimulation_variable = fields.Boolean()
    boundaries = fields.Many2one(comodel_name='tvbo.range')
    initial_value = fields.Float(default=0.1)
    history = fields.Many2one(comodel_name='tvbo.time_series')


class Distribution(models.Model):
    _name = 'tvbo.distribution'
    _description = 'Distribution'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    dependencies = fields.One2many(comodel_name='tvbo.parameter')
    correlation = fields.Many2one(comodel_name='tvbo.matrix')


class Parameter(models.Model):
    _name = 'tvbo.parameter'
    _description = 'Parameter'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    label = fields.Char(index=True)
    description = fields.Text()
    comment = fields.Char()
    heterogeneous = fields.Boolean()
    free = fields.Boolean()
    shape = fields.Char()
    explored_values = fields.Text()


class CouplingInput(models.Model):
    _name = 'tvbo.coupling_input'
    _description = 'Specification of a coupling input channel for multi-coupling dynamics'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    description = fields.Text()
    dimension = fields.Integer(string='Dimensionality of the coupling input (number of coupled values)', default=1)


class Function(models.Model):
    _name = 'tvbo.function'
    _description = 'Function'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    acronym = fields.Char()
    label = fields.Char(index=True)
    description = fields.Text()
    iri = fields.Char()
    arguments = fields.One2many(comodel_name='tvbo.parameter')
    output = fields.Many2one(comodel_name='tvbo.equation')
    source_code = fields.Char()
    callable = fields.Many2one(comodel_name='tvbo.callable')


class Callable(models.Model):
    _name = 'tvbo.callable'
    _description = 'Callable'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    description = fields.Text()
    module = fields.Char()
    qualname = fields.Char()
    software = fields.Many2one(comodel_name='tvbo.software_requirement')


class Case(models.Model):
    _name = 'tvbo.case'
    _description = 'Case'


    condition = fields.Char()
    equation = fields.Many2one(comodel_name='tvbo.equation')


class DerivedParameter(models.Model):
    _name = 'tvbo.derived_parameter'
    _description = 'DerivedParameter'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    description = fields.Text()


class DerivedVariable(models.Model):
    _name = 'tvbo.derived_variable'
    _description = 'DerivedVariable'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    description = fields.Text()
    conditional = fields.Boolean()
    cases = fields.One2many(comodel_name='tvbo.case')


class Noise(models.Model):
    _name = 'tvbo.noise'
    _description = 'Noise'


    noise_type = fields.Char()
    correlated = fields.Boolean()
    gaussian = fields.Boolean(string='Indicates whether the noise is Gaussian')
    additive = fields.Boolean(string='Indicates whether the noise is additive')
    seed = fields.Integer(default=42)
    random_state = fields.Many2one(comodel_name='tvbo.random_stream')
    intensity = fields.Many2one(comodel_name='tvbo.parameter', string='Optional scalar or vector intensity parameter for noise.')
    function = fields.Many2one(comodel_name='tvbo.function', string='Optional functional form of the noise (callable specification).')
    pycode = fields.Char(string='Inline Python code representation of the noise process.')
    targets = fields.One2many(comodel_name='tvbo.state_variable', string='State variables this noise applies to; if omitted, applies globally.')


class RandomStream(models.Model):
    _name = 'tvbo.random_stream'
    _description = 'RandomStream'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()


class CostFunction(models.Model):
    _name = 'tvbo.cost_function'
    _description = 'CostFunction'

    _rec_name = 'label'

    label = fields.Char(index=True)


class FittingTarget(models.Model):
    _name = 'tvbo.fitting_target'
    _description = 'FittingTarget'

    _rec_name = 'label'

    label = fields.Char(index=True)


class ModelFitting(models.Model):
    _name = 'tvbo.model_fitting'
    _description = 'ModelFitting'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    targets = fields.One2many(comodel_name='tvbo.fitting_target')
    cost_function = fields.Many2one(comodel_name='tvbo.cost_function')


class Integrator(models.Model):
    _name = 'tvbo.integrator'
    _description = 'Integrator'


    method = fields.Char()
    step_size = fields.Float(default=0.01220703125)
    steps = fields.Integer()
    noise = fields.Many2one(comodel_name='tvbo.noise')
    state_wise_sigma = fields.Text()
    transient_time = fields.Float(default=0.0)
    scipy_ode_base = fields.Boolean()
    number_of_stages = fields.Integer(default=1)
    intermediate_expressions = fields.One2many(comodel_name='tvbo.derived_variable')
    update_expression = fields.Many2one(comodel_name='tvbo.derived_variable')
    delayed = fields.Boolean()


class Monitor(models.Model):
    _name = 'tvbo.monitor'
    _description = 'Observation model for monitoring simulation output with optional processing pipeline'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    label = fields.Char(index=True)
    acronym = fields.Char()
    description = fields.Text()
    period = fields.Float(string='Sampling period for the monitor')
    imaging_modality = fields.Many2one(comodel_name='tvbo.imaging_modality', string='Type of imaging modality (BOLD, EEG, MEG, etc.)')


class Coupling(models.Model):
    _name = 'tvbo.coupling'
    _description = 'Coupling'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    label = fields.Char(index=True)
    coupling_function = fields.Many2one(comodel_name='tvbo.equation', string='Mathematical function defining the coupling')
    sparse = fields.Boolean(string='Whether the coupling uses sparse representations')
    pre_expression = fields.Many2one(comodel_name='tvbo.equation', string='Pre-processing expression applied before coupling')
    post_expression = fields.Many2one(comodel_name='tvbo.equation', string='Post-processing expression applied after coupling')
    incoming_states = fields.Many2one(comodel_name='tvbo.state_variable', string='State variables from connected nodes (source)')
    local_states = fields.Many2one(comodel_name='tvbo.state_variable', string='State variables from local node (target)')
    delayed = fields.Boolean(string='Whether coupling includes transmission delays')
    inner_coupling = fields.Many2one(comodel_name='tvbo.coupling', string='For hierarchical coupling: inner coupling applied at regional level')
    region_mapping = fields.Many2one(comodel_name='tvbo.region_mapping', string='For hierarchical coupling: vertex-to-region mapping for aggregation')
    regional_connectivity = fields.Many2one(comodel_name='tvbo.connectome', string='For hierarchical coupling: region-to-region connectivity with weights and delays')
    aggregation = fields.Char(string="For hierarchical coupling: aggregation method ('sum', 'mean', 'max') or custom Function")
    distribution = fields.Char(string="For hierarchical coupling: distribution method ('broadcast', 'weighted') or custom Function")


class RegionMapping(models.Model):
    _name = 'tvbo.region_mapping'
    _description = 'Maps vertices to parent regions for hierarchical/aggregated coupling'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    vertex_to_region = fields.Text(string='Array mapping each vertex index to its parent region index. Can use dataLocation instead for large arrays.')
    n_vertices = fields.Integer(string='Total number of vertices')
    n_regions = fields.Integer(string='Total number of regions')


class Sample(models.Model):
    _name = 'tvbo.sample'
    _description = 'Sample'


    groups = fields.Text()
    size = fields.Integer()


class SimulationExperiment(models.Model):
    _name = 'tvbo.simulation_experiment'
    _description = 'SimulationExperiment'

    _rec_name = 'label'

    id = fields.Integer()
    description = fields.Text()
    additional_equations = fields.One2many(comodel_name='tvbo.equation')
    label = fields.Char(index=True)
    local_dynamics = fields.Many2one(comodel_name='tvbo.dynamics')
    dynamics = fields.Text()
    integration = fields.Many2one(comodel_name='tvbo.integrator')
    connectivity = fields.Many2one(comodel_name='tvbo.connectome')
    network = fields.Many2one(comodel_name='tvbo.connectome')
    coupling = fields.Many2one(comodel_name='tvbo.coupling')
    monitors = fields.One2many(comodel_name='tvbo.monitor')
    stimulation = fields.Many2one(comodel_name='tvbo.stimulus')
    field_dynamics = fields.Many2one(comodel_name='tvbo.pde')
    modelfitting = fields.One2many(comodel_name='tvbo.model_fitting')
    environment = fields.Many2one(comodel_name='tvbo.software_environment', string='Execution environment (collection of requirements).')
    software = fields.Many2one(comodel_name='tvbo.software_requirement', string="(Deprecated) Single software requirement; prefer 'environment' with aggregated requirements.")
    references = fields.Text()


class SimulationStudy(models.Model):
    _name = 'tvbo.simulation_study'
    _description = 'SimulationStudy'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    key = fields.Char()
    title = fields.Char()
    year = fields.Integer()
    doi = fields.Char()
    sample = fields.Many2one(comodel_name='tvbo.sample')
    simulation_experiments = fields.One2many(comodel_name='tvbo.simulation_experiment')


class TimeSeries(models.Model):
    _name = 'tvbo.time_series'
    _description = 'TimeSeries'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    data = fields.Many2one(comodel_name='tvbo.matrix')
    time = fields.Many2one(comodel_name='tvbo.matrix')
    sampling_rate = fields.Float()
    unit = fields.Char()


class SoftwareEnvironment(models.Model):
    _name = 'tvbo.software_environment'
    _description = 'SoftwareEnvironment'

    _rec_name = 'name'

    label = fields.Char(index=True)
    description = fields.Text()
    name = fields.Char(required=True, index=True, string="Human-readable environment label/name (deprecated alias was 'software').")
    version = fields.Char(string='Optional version tag for the environment definition (not a package version).')
    platform = fields.Char(string='OS / architecture description (e.g., linux-64).')
    environment_type = fields.Many2one(comodel_name='tvbo.environment_type', string='Category: conda, venv, docker, etc.')
    container_image = fields.Char(string='Container image reference (e.g., ghcr.io/org/img:tag@sha256:...).')
    build_hash = fields.Char(string='Deterministic hash/fingerprint of the resolved dependency set.')
    requirements = fields.One2many(comodel_name='tvbo.software_requirement', string='Constituent software/module requirements that define this environment.')


class SoftwareRequirement(models.Model):
    _name = 'tvbo.software_requirement'
    _description = 'SoftwareRequirement'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True, string="Human-readable environment label/name (deprecated alias was 'software').")
    description = fields.Text()
    package = fields.Many2one(comodel_name='tvbo.software_package', string='Reference to the software package identity.', required=True)
    version_spec = fields.Char(string="Version or constraint specifier (e.g., '==2.7.3', '>=1.2,<2').")
    role = fields.Many2one(comodel_name='tvbo.requirement_role')
    optional = fields.Boolean()
    hash = fields.Char(string='Build or artifact hash for exact reproducibility (wheel, sdist, image layer).')
    source_url = fields.Char(string='Canonical source or repository URL.')
    url = fields.Char(string='(Deprecated) Use source_url.')
    license = fields.Char()
    modules = fields.Text(string='(Deprecated) Former ad-hoc list; use environment.requirements list instead.')
    version = fields.Char(string='(Deprecated) Use version_spec.')


class SoftwarePackage(models.Model):
    _name = 'tvbo.software_package'
    _description = 'Identity information about a software package independent of a specific version requirement.'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True, string="Human-readable environment label/name (deprecated alias was 'software').")
    description = fields.Text()
    homepage = fields.Char()
    license = fields.Char()
    repository = fields.Char()
    doi = fields.Char()
    ecosystem = fields.Char(string='Package ecosystem or index (e.g., pypi, conda-forge).')


class NDArray(models.Model):
    _name = 'tvbo.nd_array'
    _description = 'NDArray'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    shape = fields.Text()
    dtype = fields.Char()
    dataLocation = fields.Char()
    unit = fields.Char()


class SpatialDomain(models.Model):
    _name = 'tvbo.spatial_domain'
    _description = 'SpatialDomain'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    coordinate_space = fields.Many2one(comodel_name='tvbo.common_coordinate_space')
    region = fields.Char(string='Optional named region/ROI in the atlas/parcellation.')
    geometry = fields.Char(string='Optional file for geometry/ROI mask (e.g., NIfTI, GIfTI).')


class Mesh(models.Model):
    _name = 'tvbo.mesh'
    _description = 'Mesh'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    element_type = fields.Many2one(comodel_name='tvbo.element_type')
    coordinates = fields.One2many(comodel_name='tvbo.coordinate', string='Node coordinates (x,y,z) in the given coordinate space.')
    elements = fields.Char(string='Connectivity (indices) or file reference to topology.')
    coordinate_space = fields.Many2one(comodel_name='tvbo.common_coordinate_space')


class SpatialField(models.Model):
    _name = 'tvbo.spatial_field'
    _description = 'SpatialField'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    quantity_kind = fields.Char(string='Scalar, vector, or tensor.')
    unit = fields.Char()
    mesh = fields.Many2one(comodel_name='tvbo.mesh')
    values = fields.Many2one(comodel_name='tvbo.nd_array')
    time_dependent = fields.Boolean()
    initial_value = fields.Float(string='Constant initial value for the field.', default=0.1)
    initial_expression = fields.Many2one(comodel_name='tvbo.equation', string='Analytic initial condition for the field.')


class FieldStateVariable(models.Model):
    _name = 'tvbo.field_state_variable'
    _description = 'FieldStateVariable'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    mesh = fields.Many2one(comodel_name='tvbo.mesh')
    boundary_conditions = fields.One2many(comodel_name='tvbo.boundary_condition')


class DifferentialOperator(models.Model):
    _name = 'tvbo.differential_operator'
    _description = 'DifferentialOperator'

    _rec_name = 'label'

    label = fields.Char(index=True)
    operator_type = fields.Many2one(comodel_name='tvbo.operator_type')
    coefficient = fields.Many2one(comodel_name='tvbo.parameter')
    tensor_coefficient = fields.Many2one(comodel_name='tvbo.parameter', string='Optional anisotropic tensor (e.g., diffusion).')
    expression = fields.Many2one(comodel_name='tvbo.equation', string="Symbolic form (e.g., '-div(D * grad(u))').")


class BoundaryCondition(models.Model):
    _name = 'tvbo.boundary_condition'
    _description = 'BoundaryCondition'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    bc_type = fields.Many2one(comodel_name='tvbo.boundary_condition_type')
    on_region = fields.Char(string='Mesh/atlas subset where BC applies.')
    value = fields.Many2one(comodel_name='tvbo.equation', string='Constant, parameter, or equation.')
    time_dependent = fields.Boolean()


class PDESolver(models.Model):
    _name = 'tvbo.pde_solver'
    _description = 'PDESolver'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    discretization = fields.Many2one(comodel_name='tvbo.discretization_method')
    time_integrator = fields.Char(string='e.g., implicit Euler, Crank–Nicolson.')
    dt = fields.Float(string='Time step (s).')
    tolerances = fields.Char(string='Abs/rel tolerances.')
    preconditioner = fields.Char()


class PDE(models.Model):
    _name = 'tvbo.pde'
    _description = 'Partial differential equation problem definition.'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    domain = fields.Many2one(comodel_name='tvbo.spatial_domain')
    mesh = fields.Many2one(comodel_name='tvbo.mesh', string='Shared mesh for all field state variables in this PDE.')
    state_variables = fields.One2many(comodel_name='tvbo.field_state_variable')
    field = fields.Many2one(comodel_name='tvbo.spatial_field', string='Primary field being solved for (deprecated; use state_variables).')
    operators = fields.One2many(comodel_name='tvbo.differential_operator')
    sources = fields.One2many(comodel_name='tvbo.equation')
    boundary_conditions = fields.One2many(comodel_name='tvbo.boundary_condition')
    solver = fields.Many2one(comodel_name='tvbo.pde_solver')
    derived_parameters = fields.One2many(comodel_name='tvbo.derived_parameter')
    derived_variables = fields.One2many(comodel_name='tvbo.derived_variable')
    functions = fields.One2many(comodel_name='tvbo.function')


class Dataset(models.Model):
    _name = 'tvbo.dataset'
    _description = 'Collection of data related to a specific DBS study.'

    _rec_name = 'label'

    label = fields.Char(index=True)
    dataset_id = fields.Char()
    subjects = fields.One2many(comodel_name='tvbo.subject')
    clinical_scores = fields.One2many(comodel_name='tvbo.clinical_score')
    coordinate_space = fields.Many2one(comodel_name='tvbo.common_coordinate_space')


class Subject(models.Model):
    _name = 'tvbo.subject'
    _description = 'Human or animal subject receiving DBS.'


    age = fields.Float()
    sex = fields.Char()
    diagnosis = fields.Char()
    handedness = fields.Char()
    protocols = fields.One2many(comodel_name='tvbo.dbs_protocol', string='All DBS protocols assigned to this subject.')
    coordinate_space = fields.Many2one(comodel_name='tvbo.common_coordinate_space', string="Coordinate space used for this subject's data")


class Electrode(models.Model):
    _name = 'tvbo.electrode'
    _description = 'Implanted DBS electrode and contact geometry.'


    electrode_id = fields.Char(string='Unique identifier for this electrode')
    manufacturer = fields.Char()
    model = fields.Char()
    hemisphere = fields.Char(string='Hemisphere of electrode (left/right)')
    contacts = fields.One2many(comodel_name='tvbo.contact', string='List of physical contacts along the electrode')
    head = fields.Many2one(comodel_name='tvbo.coordinate')
    tail = fields.Many2one(comodel_name='tvbo.coordinate')
    trajectory = fields.One2many(comodel_name='tvbo.coordinate', string='The planned trajectory for electrode implantation')
    target_structure = fields.Many2one(comodel_name='tvbo.parcellation_entity', string='Anatomical target structure from a brain atlas')
    coordinate_space = fields.Many2one(comodel_name='tvbo.common_coordinate_space', string='Coordinate space used for implantation planning')
    recon_path = fields.Char()


class Contact(models.Model):
    _name = 'tvbo.contact'
    _description = 'Individual contact on a DBS electrode.'

    _rec_name = 'label'

    contact_id = fields.Integer(string='Identifier (e.g., 0, 1, 2)')
    coordinate = fields.Many2one(comodel_name='tvbo.coordinate', string='3D coordinate of the contact center in the defined coordinate space')
    label = fields.Char(index=True, string='Optional human-readable label (e.g., "1a")')


class StimulationSetting(models.Model):
    _name = 'tvbo.stimulation_setting'
    _description = 'DBS parameters for a specific session.'


    electrode_reference = fields.Many2one(comodel_name='tvbo.electrode')
    amplitude = fields.Many2one(comodel_name='tvbo.parameter')
    frequency = fields.Many2one(comodel_name='tvbo.parameter')
    pulse_width = fields.Many2one(comodel_name='tvbo.parameter')
    mode = fields.Char()
    active_contacts = fields.Text()
    efield = fields.Many2one(comodel_name='tvbo.e_field', string='Metadata about the E-field result for this setting')


class DBSProtocol(models.Model):
    _name = 'tvbo.dbs_protocol'
    _description = 'A protocol describing DBS therapy, potentially bilateral or multi-lead.'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True, string="Human-readable environment label/name (deprecated alias was 'software').")
    electrodes = fields.One2many(comodel_name='tvbo.electrode')
    settings = fields.One2many(comodel_name='tvbo.stimulation_setting')
    timing_info = fields.Char()
    notes = fields.Char()
    clinical_improvement = fields.One2many(comodel_name='tvbo.clinical_improvement', string='Observed improvement relative to baseline based on a defined score.')


class ClinicalScale(models.Model):
    _name = 'tvbo.clinical_scale'
    _description = 'A clinical assessment inventory or structured scale composed of multiple scores or items.'

    _rec_name = 'name'

    acronym = fields.Char()
    name = fields.Char(required=True, index=True, string='Full name of the scale (e.g., Unified Parkinson’s Disease Rating Scale)')
    version = fields.Char(string='Version of the instrument (e.g., 3.0)')
    domain = fields.Char(string='Overall clinical domain (e.g., motor, cognition)')
    reference = fields.Char(string='DOI, PMID or persistent identifier')


class ClinicalScore(models.Model):
    _name = 'tvbo.clinical_score'
    _description = 'Metadata about a clinical score or scale.'

    _rec_name = 'name'

    acronym = fields.Char()
    name = fields.Char(required=True, index=True, string="Full name of the score (e.g., Unified Parkinson's Disease Rating Scale - Part III)")
    description = fields.Text()
    domain = fields.Char(string='Domain assessed (e.g. motor, mood, pain)')
    reference = fields.Char(string='PubMed ID, DOI, or other reference to the score definition')
    scale = fields.Many2one(comodel_name='tvbo.clinical_scale', string='The scale this score belongs to, if applicable')
    parent_score = fields.Many2one(comodel_name='tvbo.clinical_score', string='If this score is a subscore of a broader composite')


class ClinicalImprovement(models.Model):
    _name = 'tvbo.clinical_improvement'
    _description = 'Relative improvement on a defined clinical score.'


    score = fields.Many2one(comodel_name='tvbo.clinical_score')
    baseline_value = fields.Float(string='Preoperative baseline value of the score')
    absolute_value = fields.Float(string='Absolute value of the score at the time of assessment')
    percent_change = fields.Float(string='Percent change compared to preoperative baseline (positive = improvement)')
    time_post_surgery = fields.Float(string='Timepoint of assessment in days or months after implantation')
    evaluator = fields.Char(string='Who performed the rating (e.g., rater initials, clinician ID, or system)')
    timepoint = fields.Char(string='Timepoint of assessment (e.g., "1 month post-op", "6 months post-op")')


class EField(models.Model):
    _name = 'tvbo.e_field'
    _description = 'Simulated electric field from DBS modeling.'


    volume_data = fields.Char(string='Reference to raw or thresholded volume')
    coordinate_space = fields.Many2one(comodel_name='tvbo.common_coordinate_space', string='Reference to a common coordinate space (e.g. MNI152)')
    threshold_applied = fields.Float(string='Threshold value applied to the E-field simulation')


class Coordinate(models.Model):
    _name = 'tvbo.coordinate'
    _description = 'A 3D coordinate with X, Y, Z values.'


    x = fields.Float(string='X coordinate')
    y = fields.Float(string='Y coordinate')
    z = fields.Float(string='Z coordinate')


class BrainAtlas(models.Model):
    _name = 'tvbo.brain_atlas'
    _description = 'A schema for representing a version of a brain atlas.'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True, string="Full name of the score (e.g., Unified Parkinson's Disease Rating Scale - Part III)")


class CommonCoordinateSpace(models.Model):
    _name = 'tvbo.common_coordinate_space'
    _description = 'A schema for representing a version of a common coordinate space.'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True, string="Full name of the score (e.g., Unified Parkinson's Disease Rating Scale - Part III)")
    anatomicalAxesOrientation = fields.Char(string='Add the axes orientation in standard anatomical terms (XYZ).')
    axesOrigin = fields.Char(string='Enter the origin (central point where all axes intersect).')
    nativeUnit = fields.Char(string='Add the native unit that is used for this common coordinate space version.')
    defaultImage = fields.Text(string='Add all image files used as visual representation of this common coordinate space version.')


class ParcellationEntity(models.Model):
    _name = 'tvbo.parcellation_entity'
    _description = 'A schema for representing a parcellation entity, which is an anatomical location or study target.'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True, string="Full name of the score (e.g., Unified Parkinson's Disease Rating Scale - Part III)")


class ParcellationTerminology(models.Model):
    _name = 'tvbo.parcellation_terminology'
    _description = 'A schema for representing a parcellation terminology, which consists of parcellation entities.'

    _rec_name = 'label'

    label = fields.Char(index=True, string='Optional human-readable label (e.g., "1a")')

