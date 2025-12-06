from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field
import numpy as np

from tvbo.api.ontology_api import OntologyAPI

app = FastAPI(
    title="TVBO API",
    description="The Virtual Brain Ontology API for simulation experiments",
    version="0.1.0"
)
api = OntologyAPI()


# ============================================
# Request/Response Models (API-specific only)
# ============================================

class RunExperimentRequest(BaseModel):
    """Request payload for running a simulation."""
    experiment: dict = Field(..., description="Experiment configuration dictionary")
    duration: Optional[float] = Field(1000.0, description="Simulation duration in ms")
    step_size: Optional[float] = Field(0.1, description="Integration step size in ms")
    backend: Optional[str] = Field("jax", description="Simulation backend: 'jax' or 'tvb'")


class RunExperimentResponse(BaseModel):
    """Response payload with simulation results."""
    success: bool
    data: Optional[List] = None  # [time, state_vars, regions, modes]
    time: Optional[List[float]] = None
    state_variables: Optional[List[str]] = None
    region_labels: Optional[List[str]] = None
    sample_period: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None


# Legacy model for backwards compatibility
class SimulationMetadata(BaseModel):
    model: dict
    connectivity: Optional[dict] = None
    coupling: Optional[dict] = None
    integration: Optional[dict] = None


# ============================================
# API Endpoints
# ============================================

@app.get("/")
def root():
    return {"message": "TVBO API", "version": "0.1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/search")
def search(term: str = Query(..., description="Ontology term to search for")):
    return api.search_by_term(term)


@app.get("/query")
def query_nodes(query_str: str = Query(..., description="Term to query in ontology")):
    return api.query_nodes(query_str)


@app.get("/children/{node_id}")
def get_child_connections(node_id: int = Path(..., description="Node ID")):
    return api.get_child_connections(node_id)


@app.get("/parents/{node_id}")
def get_parent_connections(node_id: int = Path(..., description="Node ID")):
    return api.get_parent_connections(node_id)


@app.post("/experiment/configure")
def configure_experiment(metadata: SimulationMetadata = Body(...)):
    api.configure_simulation_experiment(metadata.dict())
    return {"message": "Experiment configured successfully"}


@app.post("/experiment/run", response_model=RunExperimentResponse)
def run_experiment(request: RunExperimentRequest = Body(...)):
    """
    Run a simulation experiment and return the results.

    The experiment configuration is converted to a SimulationExperiment
    object using tvbo classes, then executed using the specified backend.
    """
    try:
        # Import tvbo runtime classes
        from tvbo.export.experiment import SimulationExperiment
        from tvbo.knowledge.simulation.localdynamics import Dynamics
        from tvbo.knowledge.simulation.integration import Integrator
        from tvbo.knowledge.simulation.network import Coupling
        from tvbo.data.tvbo_data.connectomes import Connectome

        exp_data = request.experiment
        duration = request.duration or 1000.0
        step_size = request.step_size or 0.1
        backend = request.backend or "jax"

        # Build local dynamics from configuration
        local_dynamics_data = exp_data.get('local_dynamics') or exp_data.get('model') or {}
        local_dynamics = None
        if local_dynamics_data:
            model_name = local_dynamics_data.get('name', 'Generic2dOscillator')
            try:
                local_dynamics = Dynamics.from_database(model_name)
                # Apply parameter overrides
                for param in local_dynamics_data.get('parameters', []):
                    pname = param.get('name')
                    pval = param.get('value')
                    if pname and pval is not None and hasattr(local_dynamics, 'parameters'):
                        if pname in local_dynamics.parameters:
                            local_dynamics.parameters[pname].value = float(pval)
            except Exception:
                local_dynamics = Dynamics.from_database('Generic2dOscillator')
        else:
            local_dynamics = Dynamics.from_database('Generic2dOscillator')

        # Build network/connectivity
        network_data = exp_data.get('network') or {}
        network = None
        if network_data:
            nodes = network_data.get('nodes', [])
            edges = network_data.get('edges', [])
            n_regions = network_data.get('number_of_regions') or len(nodes) or 2

            if nodes and edges:
                # Build from nodes/edges
                weights = np.zeros((n_regions, n_regions))
                delays = np.zeros((n_regions, n_regions))
                for edge in edges:
                    src = edge.get('source', 0)
                    tgt = edge.get('target', 1)
                    w = edge.get('weight', 1.0)
                    d = edge.get('delay', 0.0)
                    if 0 <= src < n_regions and 0 <= tgt < n_regions:
                        weights[src, tgt] = w
                        delays[src, tgt] = d

                labels = [n.get('label', f'Region_{i}') for i, n in enumerate(nodes)]

                network = Connectome(
                    weights=weights,
                    tract_lengths=delays,
                    region_labels=labels,
                    number_of_regions=n_regions,
                )
            elif network_data.get('weights_matrix'):
                weights = np.array(network_data['weights_matrix'])
                lengths = np.array(network_data.get('lengths_matrix') or np.zeros_like(weights))
                network = Connectome(
                    weights=weights,
                    tract_lengths=lengths,
                    number_of_regions=weights.shape[0],
                )
            else:
                network = Connectome(number_of_regions=n_regions)
        else:
            network = Connectome(number_of_regions=2)

        # Build integration
        integration_data = exp_data.get('integration') or {}
        integration = Integrator(
            method=integration_data.get('method', 'Heun'),
            step_size=step_size,
            duration=duration,
        )

        # Build coupling
        coupling_data = exp_data.get('coupling') or {}
        coupling = Coupling(
            name=coupling_data.get('name', 'Linear'),
        )
        if coupling_data.get('global_coupling') is not None:
            coupling.global_coupling = float(coupling_data['global_coupling'])

        # Create and run the experiment
        experiment = SimulationExperiment(
            label=exp_data.get('label', 'WebExperiment'),
            local_dynamics=local_dynamics,
            network=network,
            integration=integration,
            coupling=coupling,
        )

        ts = experiment.run(format=backend, duration=duration)

        # Extract results
        time_arr = ts.time.tolist() if hasattr(ts.time, 'tolist') else list(ts.time)
        data_arr = ts.data.tolist() if hasattr(ts.data, 'tolist') else ts.data

        state_vars = list(local_dynamics.state_variables.keys()) if hasattr(local_dynamics, 'state_variables') else ['V']

        region_labels = []
        if hasattr(network, 'region_labels') and network.region_labels is not None:
            region_labels = list(network.region_labels)
        else:
            n_reg = network.number_of_regions if hasattr(network, 'number_of_regions') else ts.data.shape[2]
            region_labels = [f'Region_{i}' for i in range(n_reg)]

        return RunExperimentResponse(
            success=True,
            data=data_arr,
            time=time_arr,
            state_variables=state_vars,
            region_labels=region_labels,
            sample_period=step_size,
            message="Simulation completed successfully",
        )

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return RunExperimentResponse(
            success=False,
            error=error_msg,
        )
