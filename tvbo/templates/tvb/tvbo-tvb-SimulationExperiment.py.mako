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

_tvb_coupling = context['coupling']
_float64 = str(getattr(experiment.execution, 'precision', None) or 'float64') == 'float64'

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
%if _float64:
######### Float64 history #########
from tvb.simulator.descriptors import NDArray


def _at_float64(tvb_array):
    """TVB's history array *tvb_array* declared at float64, with its shape and mutability."""
    return NDArray(tvb_array.shape, "d", tvb_array.read_only)


class Float64SparseHistory(SparseHistory):
    """TVB's sparse delay history at the declared float64.

    TVB declares the delay buffer, the weights its coupling reads (`es_weights`, `nnz_weights`), the weights and delays they are cut from and the delayed and current states as float32 `NDArray`s, which allocate once and cast every assignment, so a coupling would read its sources and weights rounded to float32. Each of those arrays is declared here at float64; the integer index arrays stay TVB's.
    """

    weights = _at_float64(SparseHistory.weights)
    delays = _at_float64(SparseHistory.delays)
    es_weights = _at_float64(SparseHistory.es_weights)
    buffer = _at_float64(SparseHistory.buffer)
    current_state = _at_float64(SparseHistory.current_state)
    delayed_state = _at_float64(SparseHistory.delayed_state)
    nnz_weights = _at_float64(SparseHistory.nnz_weights)


class Float64Simulator(Simulator):
    """TVB's `Simulator`, its delay history a `Float64SparseHistory` on every configure."""

    def _configure_history(self, initial_conditions=None):
        self.history = Float64SparseHistory.from_simulator(self, initial_conditions)


%endif
######### Simulation #########
def define_simulation(connectivity, simulation_length=${settle['total_duration']}, initial_conditions=None, model_kwargs={}, coupling_kwargs={}, integration_kwargs={'dt':${settle['dt']}}, stimulus_kwargs={}):
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
    simulator = ${'Float64Simulator' if _float64 else 'Simulator'}(
        model=_TVBO_MODEL_CLS(**model_kwargs),
        connectivity=connectivity,
        coupling=${'%s(**coupling_kwargs)' % _tvb_coupling.name if _tvb_coupling else 'Linear(**coupling_kwargs)'},
        conduction_speed=${experiment.network.conduction_speed.value},
        integrator=${experiment.integration.method + ('Stochastic' if (experiment.integration.noise or np.any(np.asarray(context['experiment'].noise_sigma_array)>0)) else '')}(${'noise=noise,' if (experiment.integration.noise or np.any(np.asarray(context['experiment'].noise_sigma_array)>0)) else ''}**integration_kwargs),
        monitors=monitors,
        simulation_length=simulation_length,
        initial_conditions=initial_conditions,
        %if experiment.stimulation:
        stimulus=StimuliRegion(
                    temporal=${experiment.stimulation.identifier + 'Equation'}(),
                    connectivity=connectivity,
                    weight=weight,
                    **stimulus_kwargs
                ),
        %endif
        # random_state=self.random_state,
    )
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
        default=${settle['total_duration']},
        help="Total integrated length: the measured duration plus any declared settle.",
    )
    args = parser.parse_args()

    # Run the simulation
    sc = load_connectivity(args.fweights, args.flengths)
    sim = define_simulation(sc, simulation_length=args.simulation_length)
    sim.configure()
    start_step = sim.current_step
    raw = sim.run()

    # Wrap in ExperimentResult for consistent access and export
    from tvbo.data.types import ExperimentResult
    results = ExperimentResult.from_tvb(sim, raw, transient_time=${settle['transient_time']}, start_step=start_step)
    print(results)

    # Export BIDS-compatible output
    from pathlib import Path
    results.export(Path(__file__).parent / "derivatives")
    print(f"BIDS output: {Path(__file__).parent / 'derivatives'}")
