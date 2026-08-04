<%!
import numpy as np
%>
<%include file="/tvbo-tvb-imports.py.mako" />
<%include file="/tvbo-tvb-model.py.mako" />
# Stable alias for the model class, bound before any later class can shadow it
# at module scope — a coupling may share the model's name (e.g. TVB has both a
# ``Linear`` model and a ``Linear`` coupling), and the later ``class Linear(Coupling)``
# would otherwise rebind the name used to construct the model.
_TVBO_MODEL_CLS = ${context['experiment'].dynamics.name}
<%include file="/tvbo-tvb-coupling.py.mako" />
<%include file="/tvbo-tvb-integration.py.mako" />
<%include file="/tvbo-tvb-noise.py.mako" />
<%include file="/tvbo-tvb-observation.py.mako" />
<%
experiment = context['experiment']

from tvbo.utils import initial_value as _initial_value

_n_reg = getattr(experiment.network, 'number_of_nodes', None) or experiment.network.number_of_regions
initial_conditions = np.array([
    np.full((_n_reg,), _initial_value(v)) if np.isscalar(v.initial_value) or v.initial_value is None
    else v.initial_value
    for k, v in experiment.dynamics.state_variables.items()
]).reshape(
    1,
    len(experiment.dynamics.state_variables),
    (getattr(experiment.network, 'number_of_nodes', None) or experiment.network.number_of_regions),
    1
)
%>
######### Stimulus #########
%if experiment.stimulation:
    <%include file="/tvbo-tvb-stimulus_equation.py.mako" />
%endif
######### Simulation #########
def define_simulation(connectivity, simulation_length=${experiment.integration.duration}, initial_conditions=None, model_kwargs={}, coupling_kwargs={}, integration_kwargs={'dt':${experiment.integration.step_size}}, stimulus_kwargs={}):
%if experiment.stimulation:
    from tvb.datatypes.patterns import StimuliRegion
    %if experiment.stimulation.weighting:
    weight = np.array(${experiment.stimulation.weighting})
    %else:
    if 'weight' not in stimulus_kwargs:
        weight = np.zeros(connectivity.number_of_regions)
        weight[${experiment.stimulation.regions}] = 1.0
    %endif
%endif
    simulator = Simulator(
        model=_TVBO_MODEL_CLS(**model_kwargs),
        connectivity=connectivity,
        coupling=${'%s(**coupling_kwargs)' % experiment.coupling.name if experiment.coupling else 'Linear(**coupling_kwargs)'},
        conduction_speed=${experiment.network.conduction_speed.value},
        integrator=${experiment.integration.method + ('Stochastic' if (experiment.integration.noise or np.any(np.asarray(context['experiment'].noise_sigma_array)>0)) else '')}(${'noise=noise,' if (experiment.integration.noise or np.any(np.asarray(context['experiment'].noise_sigma_array)>0)) else ''}**integration_kwargs),
        monitors=monitors,
        simulation_length=simulation_length,
        initial_conditions=initial_conditions,
        %if experiment.stimulation:
        stimulus=StimuliRegion(
                    temporal=${experiment.stimulation.label+'Equation'}(),
                    connectivity=connectivity,
                    weight=weight,
                    **stimulus_kwargs
                ),
        %endif
        # random_state=self.random_state,
    )
    simulator.configure()
    return simulator

######### Connectivity #########
def load_connectivity(weights_path, lengths_path):
    """
    Load the weights and lengths from the provided CSV files and create the Connectivity object.

    Args:
        weights_path (str): Path to the CSV file containing the weights matrix.
        lengths_path (str): Path to the CSV file containing the tract lengths matrix.

    Returns:
        Connectivity: Configured TVB Connectivity object.
    """
    weights = pd.read_csv(weights_path, header=None).values
    lengths = pd.read_csv(lengths_path, header=None).values

    SC = Connectivity(
        weights=weights,
        tract_lengths=lengths,
    )
    SC.set_centres(np.repeat((0, 0, 0), weights.shape[0]), weights.shape[0])
    SC.create_region_labels()
    SC.configure()

    return SC

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
        description="Run TVB simulator with custom weights and lengths."
    )

    parser.add_argument(
        "--fweights",
        type=str,
        ${f"default={fweights}" if fweights else "required=True"},
        help="Path to the CSV file containing the weights matrix.",
    )
    parser.add_argument(
        "--flengths",
        type=str,
        ${f"default={flengths}" if flengths else "required=True"},
        help="Path to the CSV file containing the tract lengths matrix.",
    )
    parser.add_argument(
        "--simulation_length",
        type=float,
        default=1000,
        help="Simulation length for the TVB simulator. Default is 1000.",
    )
    args = parser.parse_args()

    # Run the simulation
    sc = load_connectivity(args.fweights, args.flengths)
    sim = define_simulation(sc, simulation_length=args.simulation_length)
    sim.configure()
    raw = sim.run()

    # Wrap in ExperimentResult for consistent access and export
    from tvbo.data.types import ExperimentResult
    results = ExperimentResult.from_tvb(sim, raw)
    print(results)

    # Export BIDS-compatible output
    from pathlib import Path
    results.export(Path(__file__).parent / "derivatives")
    print(f"BIDS output: {Path(__file__).parent / 'derivatives'}")
