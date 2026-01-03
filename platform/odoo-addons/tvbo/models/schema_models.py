# -*- coding: utf-8 -*-
# Auto-generated from TVBO schemas - DO NOT EDIT MANUALLY
# Re-run scripts/generate_odoo_models.py to update
from odoo import models, fields, api


class ImagingModality(models.Model):
    _name = 'tvbo.imaging_modality'
    _description = 'ImagingModality'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class OperationType(models.Model):
    _name = 'tvbo.operation_type'
    _description = 'OperationType'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class SystemType(models.Model):
    _name = 'tvbo.system_type'
    _description = 'SystemType'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class BoundaryConditionType(models.Model):
    _name = 'tvbo.boundary_condition_type'
    _description = 'BoundaryConditionType'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class DiscretizationMethod(models.Model):
    _name = 'tvbo.discretization_method'
    _description = 'DiscretizationMethod'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class ElementType(models.Model):
    _name = 'tvbo.element_type'
    _description = 'ElementType'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class OperatorType(models.Model):
    _name = 'tvbo.operator_type'
    _description = 'OperatorType'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class NoiseType(models.Model):
    _name = 'tvbo.noise_type'
    _description = 'NoiseType'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class RequirementRole(models.Model):
    _name = 'tvbo.requirement_role'
    _description = 'RequirementRole'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class EnvironmentType(models.Model):
    _name = 'tvbo.environment_type'
    _description = 'EnvironmentType'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class SpecimenEnum(models.Model):
    _name = 'tvbo.specimen_enum'
    _description = 'A set of permissible types for specimens used in brain atlas creation.'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


class Hemisphere(models.Model):
    _name = 'tvbo.hemisphere'
    _description = 'Hemisphere'
    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    technical_name = fields.Char(required=True, index=True)
    description = fields.Text()


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
    definition = fields.Char()
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_equation_parameters_rel')
    lefthandside = fields.Char()
    righthandside = fields.Char()
    conditionals = fields.Many2many(comodel_name='tvbo.conditional_block', relation='tvbo_equation_conditionals_rel', string='Conditional logic for piecewise equations.')
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

    equation = fields.Many2one(comodel_name='tvbo.equation')
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_stimulus_parameters_rel')
    description = fields.Text()
    dataLocation = fields.Char(string='Add the location of the data file containing the parcellation terminology.')
    duration = fields.Float(default=1000.0)
    label = fields.Char(index=True)
    regions = fields.Text()
    weighting = fields.Text()


class TemporalApplicableEquation(models.Model):
    _name = 'tvbo.temporal_applicable_equation'
    _description = 'TemporalApplicableEquation'

    _inherits = {'tvbo.equation': 'equation_id'}

    equation_id = fields.Many2one('tvbo.equation', required=True, ondelete='cascade')

    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_temporal_applicable_equation_parameters_rel')
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


class Tractogram(models.Model):
    _name = 'tvbo.tractogram'
    _description = 'Reference to tractography/diffusion MRI data used to derive structural connectivity'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    label = fields.Char(index=True)
    description = fields.Text()
    data_source = fields.Char(string='Path or URI to the tractography data file')
    number_of_subjects = fields.Integer(string='Number of subjects in the tractography dataset')
    acquisition = fields.Char(string='Acquisition protocol or scanner information')
    processing_pipeline = fields.Char(string='Processing pipeline used to generate the tractography')
    reference = fields.Char(string='Publication or DOI reference for this tractography dataset')


class Matrix(models.Model):
    _name = 'tvbo.matrix'
    _description = 'Adjacency matrix of a network.'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    dataLocation = fields.Char(string='Add the location of the data file containing the parcellation terminology.')
    x = fields.Many2one(comodel_name='tvbo.brain_region_series')
    y = fields.Many2one(comodel_name='tvbo.brain_region_series')
    values = fields.Text()


class BrainRegionSeries(models.Model):
    _name = 'tvbo.brain_region_series'
    _description = 'A series whose values represent latitude'


    values = fields.Text()


class Network(models.Model):
    _name = 'tvbo.network'
    _description = 'Network specification with nodes, edges, and reusable coupling configurations. Supports both explicit node/edge representation and matrix-based connectivity (Connectome compatibility).'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    nodes = fields.Many2many(comodel_name='tvbo.node', relation='tvbo_network_nodes_rel', string='List of nodes with individual dynamics (optional, for heterogeneous networks)')
    edges = fields.Many2many(comodel_name='tvbo.edge', relation='tvbo_network_edges_rel', string='List of directed edges with coupling references (optional, for explicit edge definition)')
    coupling = fields.Many2many(comodel_name='tvbo.coupling', relation='tvbo_network_coupling_library_rel', string="Reusable coupling configurations referenced by edges (e.g., 'instant', 'delayed', 'inhibitory')")
    number_of_regions = fields.Integer(string='Number of regions (backward compatibility)', default=1)
    number_of_nodes = fields.Integer(string='Number of nodes in the network', default=1)
    parcellation = fields.Many2one(comodel_name='tvbo.parcellation', string='Brain parcellation/atlas reference')
    tractogram = fields.Char(string='Reference to tractography data')
    weights = fields.Many2one(comodel_name='tvbo.matrix', string='Adjacency/weight matrix (backward compatibility with Connectome)')
    lengths = fields.Many2one(comodel_name='tvbo.matrix', string='Distance/length matrix (backward compatibility with Connectome)')
    normalization = fields.Many2one(comodel_name='tvbo.equation', string='Normalization equation for connectivity weights')
    node_labels = fields.Text(string='Labels for nodes/regions')
    global_coupling_strength = fields.Many2one(comodel_name='tvbo.parameter', string='Global scaling factor for all coupling weights')
    conduction_speed = fields.Many2one(comodel_name='tvbo.parameter', string='Conduction speed for computing delays from distances')


class Node(models.Model):
    _name = 'tvbo.node'
    _description = 'A node in a network with its own dynamics and properties'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    record_id = fields.Integer(string='Unique node identifier', required=True)
    dynamics = fields.Many2one(comodel_name='tvbo.dynamics', string="Dynamics model governing this node's behavior. Can be a reference (by name) or inline definition. If not provided, uses experiment's local_dynamics.")
    position = fields.Many2one(comodel_name='tvbo.coordinate', string='Spatial coordinates (x, y, z) of the node')
    region = fields.Char(string='Brain region or anatomical label')
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_node_parameters_rel', string='Node-specific parameter overrides')
    initial_state = fields.Text(string='Initial values for state variables')


class Edge(models.Model):
    _name = 'tvbo.edge'
    _description = 'A directed edge in a network with coupling and connectivity properties'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    source = fields.Integer(string='Source node ID', required=True)
    target = fields.Integer(string='Target node ID', required=True)
    coupling = fields.Many2one(comodel_name='tvbo.coupling', string='Coupling function applied on this edge (can be shared across edges)', required=True)
    weight = fields.Float(string='Connection strength/weight', default=1.0)
    delay = fields.Float(string='Transmission delay (ms or specified time unit)', default=0.0)
    distance = fields.Float(string='Physical distance between nodes (e.g., tract length in mm)')
    tract_properties = fields.Char(string='Additional tract metadata or reference to tractography data')


class ObservationModel(models.Model):
    _name = 'tvbo.observation_model'
    _description = 'ObservationModel'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    acronym = fields.Char()
    description = fields.Text()
    equation = fields.Many2one(comodel_name='tvbo.equation')
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_observation_model_parameters_rel')
    environment = fields.Many2one(comodel_name='tvbo.software_environment')
    transformation = fields.Many2one(comodel_name='tvbo.function')
    pipeline = fields.Many2many(comodel_name='tvbo.processing_step', relation='tvbo_observation_model_pipeline_rel', string='Ordered sequence of processing functions')
    data_injections = fields.Many2many(comodel_name='tvbo.data_injection', relation='tvbo_observation_model_data_injections_rel', string='External data added to the pipeline (e.g., timepoints, kernels)')
    argument_mappings = fields.Many2many(comodel_name='tvbo.argument_mapping', relation='tvbo_observation_model_argument_mappings_rel', string='How inputs/outputs connect between pipeline steps')
    derivatives = fields.Many2many(comodel_name='tvbo.derived_variable', relation='tvbo_observation_model_derivatives_rel', string='Side computations (e.g., functional connectivity)')


class ProcessingStep(models.Model):
    _name = 'tvbo.processing_step'
    _description = 'A single processing step in an observation model pipeline or standalone operation'


    order = fields.Integer(string='Execution order in the pipeline (optional for standalone operations)')
    function = fields.Many2one(comodel_name='tvbo.function', string='Function or transformation to apply', required=True)
    operation_type = fields.Many2one(comodel_name='tvbo.operation_type', string='Kind of operation to perform (e.g., subsample, projection, convolution).')
    input_mapping = fields.Many2many(comodel_name='tvbo.argument_mapping', relation='tvbo_processing_step_input_mapping_rel', string='Maps function arguments to pipeline data/outputs')
    output_alias = fields.Char(string="Optional name for this step's output (default: function name)")
    apply_on_dimension = fields.Char(string="Which dimension to apply function on (e.g., 'time', 'space')")
    ensure_shape = fields.Char(string="Ensure output has specific dimensionality (e.g., '4d')")
    variables_of_interest = fields.Many2many(comodel_name='tvbo.state_variable', relation='tvbo_processing_step_variables_of_interest_rel', string='Optional per-step variable selection')


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

    _inherits = {'tvbo.observation_model': 'observation_model_id'}

    observation_model_id = fields.Many2one('tvbo.observation_model', required=True, ondelete='cascade')

    period = fields.Float(default=0.9765625)


class Dynamics(models.Model):
    _name = 'tvbo.dynamics'
    _description = 'Dynamics'

    _rec_name = 'name'

    has_reference = fields.Char()
    name = fields.Char(required=True, index=True)
    label = fields.Char(index=True)
    iri = fields.Char()
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_dynamics_parameters_rel')
    description = fields.Text()
    source = fields.Char()
    references = fields.Text()
    derived_parameters = fields.Many2many(comodel_name='tvbo.derived_parameter', relation='tvbo_dynamics_derived_parameters_rel')
    derived_variables = fields.Many2many(comodel_name='tvbo.derived_variable', relation='tvbo_dynamics_derived_variables_rel')
    coupling_terms = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_dynamics_coupling_terms_rel')
    coupling_inputs = fields.Many2many(comodel_name='tvbo.coupling_input', relation='tvbo_dynamics_coupling_inputs_rel')
    state_variables = fields.Many2many(comodel_name='tvbo.state_variable', relation='tvbo_dynamics_state_variables_rel')
    is_modified = fields.Boolean(string="MODIFIED (renamed from 'modified')")
    output = fields.Many2many(comodel_name='tvbo.derived_variable', relation='tvbo_dynamics_output_transforms_rel')
    derived_from_model = fields.Many2one(comodel_name='tvbo.neural_mass_model')
    number_of_modes = fields.Integer(default=1)
    local_coupling_term = fields.Many2one(comodel_name='tvbo.parameter')
    functions = fields.Many2many(comodel_name='tvbo.function', relation='tvbo_dynamics_functions_rel')
    stimulus = fields.Many2one(comodel_name='tvbo.stimulus')
    modes = fields.Many2many(comodel_name='tvbo.neural_mass_model', relation='tvbo_dynamics_modes_rel')
    system_type = fields.Many2one(comodel_name='tvbo.system_type')


class NeuralMassModel(models.Model):
    _name = 'tvbo.neural_mass_model'
    _description = 'NeuralMassModel'

    _inherits = {'tvbo.dynamics': 'dynamics_id'}

    dynamics_id = fields.Many2one('tvbo.dynamics', required=True, ondelete='cascade')



class StateVariable(models.Model):
    _name = 'tvbo.state_variable'
    _description = 'StateVariable'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    symbol = fields.Char()
    label = fields.Char(index=True)
    definition = fields.Char()
    domain = fields.Many2one(comodel_name='tvbo.range')
    description = fields.Text()
    equation = fields.Many2one(comodel_name='tvbo.equation')
    unit = fields.Char()
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
    equation = fields.Many2one(comodel_name='tvbo.equation')
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_distribution_parameters_rel')
    dependencies = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_distribution_dependencies_rel')
    correlation = fields.Many2one(comodel_name='tvbo.matrix')


class Parameter(models.Model):
    _name = 'tvbo.parameter'
    _description = 'Parameter'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    label = fields.Char(index=True)
    symbol = fields.Char()
    definition = fields.Char()
    value = fields.Float()
    default = fields.Char()
    domain = fields.Many2one(comodel_name='tvbo.range')
    reported_optimum = fields.Float()
    description = fields.Text()
    equation = fields.Many2one(comodel_name='tvbo.equation')
    unit = fields.Char()
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
    equation = fields.Many2one(comodel_name='tvbo.equation')
    definition = fields.Char()
    description = fields.Text()
    requirements = fields.Many2many(comodel_name='tvbo.software_requirement', relation='tvbo_function_requirements_rel')
    iri = fields.Char()
    arguments = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_function_arguments_rel')
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

    _inherits = {'tvbo.parameter': 'parameter_id'}

    parameter_id = fields.Many2one('tvbo.parameter', required=True, ondelete='cascade')

    name = fields.Char(required=True, index=True)
    symbol = fields.Char()
    description = fields.Text()
    equation = fields.Many2one(comodel_name='tvbo.equation')
    unit = fields.Char()


class DerivedVariable(models.Model):
    _name = 'tvbo.derived_variable'
    _description = 'DerivedVariable'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    symbol = fields.Char()
    description = fields.Text()
    equation = fields.Many2one(comodel_name='tvbo.equation')
    unit = fields.Char()
    conditional = fields.Boolean()
    cases = fields.Many2many(comodel_name='tvbo.case', relation='tvbo_derived_variable_cases_rel')


class Noise(models.Model):
    _name = 'tvbo.noise'
    _description = 'Noise'


    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_noise_parameters_rel')
    equation = fields.Many2one(comodel_name='tvbo.equation')
    noise_type = fields.Char()
    correlated = fields.Boolean()
    gaussian = fields.Boolean(string='Indicates whether the noise is Gaussian')
    additive = fields.Boolean(string='Indicates whether the noise is additive')
    seed = fields.Integer(default=42)
    random_state = fields.Many2one(comodel_name='tvbo.random_stream')
    intensity = fields.Many2one(comodel_name='tvbo.parameter', string='Optional scalar or vector intensity parameter for noise.')
    function = fields.Many2one(comodel_name='tvbo.function', string='Optional functional form of the noise (callable specification).')
    pycode = fields.Char(string='Inline Python code representation of the noise process.')
    targets = fields.Many2many(comodel_name='tvbo.state_variable', relation='tvbo_noise_targets_rel', string='State variables this noise applies to; if omitted, applies globally.')


class RandomStream(models.Model):
    _name = 'tvbo.random_stream'
    _description = 'RandomStream'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    dataLocation = fields.Char(string='Add the location of the data file containing the parcellation terminology.')


class CostFunction(models.Model):
    _name = 'tvbo.cost_function'
    _description = 'CostFunction'

    _rec_name = 'label'

    label = fields.Char(index=True)
    equation = fields.Many2one(comodel_name='tvbo.equation')
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_cost_function_parameters_rel')


class FittingTarget(models.Model):
    _name = 'tvbo.fitting_target'
    _description = 'FittingTarget'

    _rec_name = 'label'

    label = fields.Char(index=True)
    equation = fields.Many2one(comodel_name='tvbo.equation')
    symbol = fields.Char()
    definition = fields.Char()
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_fitting_target_parameters_rel')


class ModelFitting(models.Model):
    _name = 'tvbo.model_fitting'
    _description = 'ModelFitting'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    targets = fields.Many2many(comodel_name='tvbo.fitting_target', relation='tvbo_model_fitting_targets_rel')
    cost_function = fields.Many2one(comodel_name='tvbo.cost_function')


class Integrator(models.Model):
    _name = 'tvbo.integrator'
    _description = 'Integrator'


    time_scale = fields.Char()
    unit = fields.Char()
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_integrator_parameters_rel')
    duration = fields.Float(default=1000.0)
    method = fields.Char()
    step_size = fields.Float(default=0.01220703125)
    steps = fields.Integer()
    noise = fields.Many2one(comodel_name='tvbo.noise')
    state_wise_sigma = fields.Text()
    transient_time = fields.Float(default=0.0)
    scipy_ode_base = fields.Boolean()
    number_of_stages = fields.Integer(default=1)
    intermediate_expressions = fields.Many2many(comodel_name='tvbo.derived_variable', relation='tvbo_integrator_intermediate_expressions_rel')
    update_expression = fields.Many2one(comodel_name='tvbo.derived_variable')
    delayed = fields.Boolean()


class Monitor(models.Model):
    _name = 'tvbo.monitor'
    _description = 'Observation model for monitoring simulation output with optional processing pipeline'

    _inherits = {'tvbo.observation_model': 'observation_model_id'}

    observation_model_id = fields.Many2one('tvbo.observation_model', required=True, ondelete='cascade')

    time_scale = fields.Char()
    name = fields.Char(required=True, index=True)
    label = fields.Char(index=True)
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_monitor_parameters_rel')
    acronym = fields.Char()
    description = fields.Text()
    equation = fields.Many2one(comodel_name='tvbo.equation')
    environment = fields.Many2one(comodel_name='tvbo.software_environment')
    period = fields.Float(string='Sampling period for the monitor')
    imaging_modality = fields.Many2one(comodel_name='tvbo.imaging_modality', string='Type of imaging modality (BOLD, EEG, MEG, etc.)')


class Coupling(models.Model):
    _name = 'tvbo.coupling'
    _description = 'Coupling'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True)
    label = fields.Char(index=True)
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_coupling_parameters_rel')
    coupling_function = fields.Many2one(comodel_name='tvbo.equation', string='Mathematical function defining the coupling')
    sparse = fields.Boolean(string='Whether the coupling uses sparse representations')
    pre_expression = fields.Many2one(comodel_name='tvbo.equation', string='Pre-processing expression applied before coupling')
    post_expression = fields.Many2one(comodel_name='tvbo.equation', string='Post-processing expression applied after coupling')
    incoming_states = fields.Many2one(comodel_name='tvbo.state_variable', string='State variables from connected nodes (source)')
    local_states = fields.Many2one(comodel_name='tvbo.state_variable', string='State variables from local node (target)')
    delayed = fields.Boolean(string='Whether coupling includes transmission delays')
    inner_coupling = fields.Many2one(comodel_name='tvbo.coupling', string='For hierarchical coupling: inner coupling applied at regional level')
    region_mapping = fields.Many2one(comodel_name='tvbo.region_mapping', string='For hierarchical coupling: vertex-to-region mapping for aggregation')
    regional_connectivity = fields.Many2one(comodel_name='tvbo.network', string='For hierarchical coupling: region-to-region connectivity with weights and delays')
    aggregation = fields.Char(string="For hierarchical coupling: aggregation method ('sum', 'mean', 'max') or custom Function")
    distribution = fields.Char(string="For hierarchical coupling: distribution method ('broadcast', 'weighted') or custom Function")


class RegionMapping(models.Model):
    _name = 'tvbo.region_mapping'
    _description = 'Maps vertices to parent regions for hierarchical/aggregated coupling'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    dataLocation = fields.Char(string='Add the location of the data file containing the parcellation terminology.')
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

    model = fields.Many2one(comodel_name='tvbo.dynamics')
    record_id = fields.Integer(string="ID (renamed from 'id')")
    description = fields.Text()
    additional_equations = fields.Many2many(comodel_name='tvbo.equation', relation='tvbo_simulation_experiment_additional_equations_rel')
    label = fields.Char(index=True)
    local_dynamics = fields.Many2one(comodel_name='tvbo.dynamics', string='Default dynamics model for all nodes (used when node.dynamics not specified or as fallback)')
    dynamics = fields.Many2many(comodel_name='tvbo.dynamics', relation='tvbo_simulation_experiment_dynamics_rel', string='Dictionary of dynamics models keyed by name. Nodes reference these by name.')
    integration = fields.Many2one(comodel_name='tvbo.integrator')
    connectivity = fields.Many2one(comodel_name='tvbo.network')
    network = fields.Many2one(comodel_name='tvbo.network')
    coupling = fields.Many2one(comodel_name='tvbo.coupling')
    monitors = fields.Many2many(comodel_name='tvbo.monitor', relation='tvbo_simulation_experiment_monitors_rel')
    stimulation = fields.Many2one(comodel_name='tvbo.stimulus')
    field_dynamics = fields.Many2one(comodel_name='tvbo.pde')
    modelfitting = fields.Many2many(comodel_name='tvbo.model_fitting', relation='tvbo_simulation_experiment_modelfitting_rel')
    environment = fields.Many2one(comodel_name='tvbo.software_environment', string='Execution environment (collection of requirements).')
    software = fields.Many2one(comodel_name='tvbo.software_requirement', string="(Deprecated) Single software requirement; prefer 'environment' with aggregated requirements.")
    references = fields.Text()


class SimulationStudy(models.Model):
    _name = 'tvbo.simulation_study'
    _description = 'SimulationStudy'

    _rec_name = 'label'

    label = fields.Char(index=True)
    model = fields.Many2one(comodel_name='tvbo.dynamics')
    description = fields.Text()
    key = fields.Char()
    title = fields.Char()
    year = fields.Integer()
    doi = fields.Char()
    sample = fields.Many2one(comodel_name='tvbo.sample')
    simulation_experiments = fields.Many2many(comodel_name='tvbo.simulation_experiment', relation='tvbo_simulation_study_simulation_experiments_rel')


class TimeSeries(models.Model):
    _name = 'tvbo.time_series'
    _description = 'TimeSeries'

    _rec_name = 'label'

    label = fields.Char(index=True)
    description = fields.Text()
    dataLocation = fields.Char(string='Add the location of the data file containing the parcellation terminology.')
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
    dataLocation = fields.Char(string='Add the location of the data file containing the parcellation terminology.')
    name = fields.Char(required=True, index=True, string="Human-readable environment label/name (deprecated alias was 'software').")
    version = fields.Char(string='Optional version tag for the environment definition (not a package version).')
    platform = fields.Char(string='OS / architecture description (e.g., linux-64).')
    environment_type = fields.Many2one(comodel_name='tvbo.environment_type', string='Category: conda, venv, docker, etc.')
    container_image = fields.Char(string='Container image reference (e.g., ghcr.io/org/img:tag@sha256:...).')
    build_hash = fields.Char(string='Deterministic hash/fingerprint of the resolved dependency set.')
    requirements = fields.Many2many(comodel_name='tvbo.software_requirement', relation='tvbo_software_environment_requirements_rel', string='Constituent software/module requirements that define this environment.')


class SoftwareRequirement(models.Model):
    _name = 'tvbo.software_requirement'
    _description = 'SoftwareRequirement'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True, string="Human-readable environment label/name (deprecated alias was 'software').")
    description = fields.Text()
    dataLocation = fields.Char(string='Add the location of the data file containing the parcellation terminology.')
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
    dataLocation = fields.Char(string='Add the location of the data file containing the parcellation terminology.')
    element_type = fields.Many2one(comodel_name='tvbo.element_type')
    coordinates = fields.Many2many(comodel_name='tvbo.coordinate', relation='tvbo_mesh_coordinates_rel', string='Node coordinates (x,y,z) in the given coordinate space.')
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

    _inherits = {'tvbo.state_variable': 'state_variable_id'}

    state_variable_id = fields.Many2one('tvbo.state_variable', required=True, ondelete='cascade')

    label = fields.Char(index=True)
    description = fields.Text()
    mesh = fields.Many2one(comodel_name='tvbo.mesh')
    boundary_conditions = fields.Many2many(comodel_name='tvbo.boundary_condition', relation='tvbo_field_state_variable_boundary_conditions_rel')


class DifferentialOperator(models.Model):
    _name = 'tvbo.differential_operator'
    _description = 'DifferentialOperator'

    _rec_name = 'label'

    label = fields.Char(index=True)
    definition = fields.Char()
    equation = fields.Many2one(comodel_name='tvbo.equation')
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
    requirements = fields.Many2many(comodel_name='tvbo.software_requirement', relation='tvbo_pde_solver_requirements_rel')
    environment = fields.Many2one(comodel_name='tvbo.software_environment')
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
    parameters = fields.Many2many(comodel_name='tvbo.parameter', relation='tvbo_pde_parameters_rel')
    domain = fields.Many2one(comodel_name='tvbo.spatial_domain')
    mesh = fields.Many2one(comodel_name='tvbo.mesh', string='Shared mesh for all field state variables in this PDE.')
    state_variables = fields.Many2many(comodel_name='tvbo.field_state_variable', relation='tvbo_pde_state_variables_rel')
    field = fields.Many2one(comodel_name='tvbo.spatial_field', string='Primary field being solved for (deprecated; use state_variables).')
    operators = fields.Many2many(comodel_name='tvbo.differential_operator', relation='tvbo_pde_operators_rel')
    sources = fields.Many2many(comodel_name='tvbo.equation', relation='tvbo_pde_sources_rel')
    boundary_conditions = fields.Many2many(comodel_name='tvbo.boundary_condition', relation='tvbo_pde_boundary_conditions_rel')
    solver = fields.Many2one(comodel_name='tvbo.pde_solver')
    derived_parameters = fields.Many2many(comodel_name='tvbo.derived_parameter', relation='tvbo_pde_derived_parameters_rel')
    derived_variables = fields.Many2many(comodel_name='tvbo.derived_variable', relation='tvbo_pde_derived_variables_rel')
    functions = fields.Many2many(comodel_name='tvbo.function', relation='tvbo_pde_functions_rel')


class Dataset(models.Model):
    _name = 'tvbo.dataset'
    _description = 'Collection of data related to a specific DBS study.'

    _rec_name = 'label'

    label = fields.Char(index=True)
    dataset_id = fields.Char()
    subjects = fields.Many2many(comodel_name='tvbo.subject', relation='tvbo_dataset_subjects_rel')
    clinical_scores = fields.Many2many(comodel_name='tvbo.clinical_score', relation='tvbo_dataset_clinical_scores_rel')
    coordinate_space = fields.Many2one(comodel_name='tvbo.common_coordinate_space')


class Subject(models.Model):
    _name = 'tvbo.subject'
    _description = 'Human or animal subject receiving DBS.'


    subject_id = fields.Char(string='Unique identifier for a subject within a dataset.', required=True)
    age = fields.Float()
    sex = fields.Char()
    diagnosis = fields.Char()
    handedness = fields.Char()
    protocols = fields.Many2many(comodel_name='tvbo.dbs_protocol', relation='tvbo_subject_protocols_rel', string='All DBS protocols assigned to this subject.')
    coordinate_space = fields.Many2one(comodel_name='tvbo.common_coordinate_space', string="Coordinate space used for this subject's data")


class Electrode(models.Model):
    _name = 'tvbo.electrode'
    _description = 'Implanted DBS electrode and contact geometry.'


    electrode_id = fields.Char(string='Unique identifier for this electrode')
    manufacturer = fields.Char()
    model = fields.Char()
    hemisphere = fields.Char(string='Hemisphere of electrode (left/right)')
    contacts = fields.Many2many(comodel_name='tvbo.contact', relation='tvbo_electrode_contacts_rel', string='List of physical contacts along the electrode')
    head = fields.Many2one(comodel_name='tvbo.coordinate')
    tail = fields.Many2one(comodel_name='tvbo.coordinate')
    trajectory = fields.Many2many(comodel_name='tvbo.coordinate', relation='tvbo_electrode_trajectory_rel', string='The planned trajectory for electrode implantation')
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
    electrodes = fields.Many2many(comodel_name='tvbo.electrode', relation='tvbo_dbs_protocol_electrodes_rel')
    settings = fields.Many2many(comodel_name='tvbo.stimulation_setting', relation='tvbo_dbs_protocol_settings_rel')
    timing_info = fields.Char()
    notes = fields.Char()
    clinical_improvement = fields.Many2many(comodel_name='tvbo.clinical_improvement', relation='tvbo_dbs_protocol_clinical_improvement_rel', string='Observed improvement relative to baseline based on a defined score.')


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


    coordinateSpace = fields.Many2one(comodel_name='tvbo.common_coordinate_space', string='Add the common coordinate space used for this brain atlas version.')
    x = fields.Float(string='X coordinate')
    y = fields.Float(string='Y coordinate')
    z = fields.Float(string='Z coordinate')


class BrainAtlas(models.Model):
    _name = 'tvbo.brain_atlas'
    _description = 'A schema for representing a version of a brain atlas.'

    _rec_name = 'name'

    coordinateSpace = fields.Many2one(comodel_name='tvbo.common_coordinate_space', string='Add the common coordinate space used for this brain atlas version.')
    name = fields.Char(required=True, index=True, string="Full name of the score (e.g., Unified Parkinson's Disease Rating Scale - Part III)")
    abbreviation = fields.Char(string='Slot for the abbreviation of a resource.')
    author = fields.Text()
    isVersionOf = fields.Char(string='Linked type for the version of a brain atlas or coordinate space.')
    versionIdentifier = fields.Char(string='Enter the version identifier of this brain atlas or coordinate space version.')


class CommonCoordinateSpace(models.Model):
    _name = 'tvbo.common_coordinate_space'
    _description = 'A schema for representing a version of a common coordinate space.'

    _rec_name = 'name'

    name = fields.Char(required=True, index=True, string="Full name of the score (e.g., Unified Parkinson's Disease Rating Scale - Part III)")
    abbreviation = fields.Char(string='Slot for the abbreviation of a resource.')
    unit = fields.Char()
    license = fields.Char(string='Linked type for the license of the brain atlas or coordinate space version.')
    anatomicalAxesOrientation = fields.Char(string='Add the axes orientation in standard anatomical terms (XYZ).')
    axesOrigin = fields.Char(string='Enter the origin (central point where all axes intersect).')
    nativeUnit = fields.Char(string='Add the native unit that is used for this common coordinate space version.')
    defaultImage = fields.Text(string='Add all image files used as visual representation of this common coordinate space version.')


class ParcellationEntity(models.Model):
    _name = 'tvbo.parcellation_entity'
    _description = 'A schema for representing a parcellation entity, which is an anatomical location or study target.'

    _rec_name = 'name'

    abbreviation = fields.Char(string='Slot for the abbreviation of a resource.')
    alternateName = fields.Text(string='Enter any alternate names, including abbreviations, for this entity.')
    lookupLabel = fields.Integer(string='Enter the label used for looking up this entity in the parcellation terminology.')
    hasParent = fields.Many2many(comodel_name='tvbo.parcellation_entity', relation='tvbo_parcellation_entity_hasParent_rel', column1='parcellation_entity_id', column2='hasParent_id', string='Add all anatomical parent structures for this entity as defined within the corresponding brain atlas.')
    name = fields.Char(required=True, index=True, string="Full name of the score (e.g., Unified Parkinson's Disease Rating Scale - Part III)")
    ontologyIdentifier = fields.Text(string='Enter the internationalized resource identifier (IRI) to the related ontological terms.')
    versionIdentifier = fields.Char(string='Enter the version identifier of this brain atlas or coordinate space version.')


class ParcellationTerminology(models.Model):
    _name = 'tvbo.parcellation_terminology'
    _description = 'A schema for representing a parcellation terminology, which consists of parcellation entities.'

    _rec_name = 'label'

    label = fields.Char(index=True, string='Optional human-readable label (e.g., "1a")')
    dataLocation = fields.Char(string='Add the location of the data file containing the parcellation terminology.')
    ontologyIdentifier = fields.Text(string='Enter the internationalized resource identifier (IRI) to the related ontological terms.')
    versionIdentifier = fields.Char(string='Enter the version identifier of this brain atlas or coordinate space version.')

