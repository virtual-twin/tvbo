from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

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

    The experiment dict should match the YAML schema and is passed
    directly to SimulationExperiment for initialization.
    """
    try:
        from tvbo.export.experiment import SimulationExperiment

        exp_data = request.experiment
        duration = request.duration or 1000.0
        step_size = request.step_size or 0.1
        backend = request.backend or "jax"

        # Pass schema dict directly to SimulationExperiment
        experiment = SimulationExperiment(**exp_data)

        # Run simulation
        ts = experiment.run(format=backend, duration=duration)

        # Extract results
        time_arr = ts.time.tolist() if hasattr(ts.time, 'tolist') else list(ts.time)
        data_arr = ts.data.tolist() if hasattr(ts.data, 'tolist') else ts.data

        state_vars = list(experiment.local_dynamics.state_variables.keys()) if experiment.local_dynamics and experiment.local_dynamics.state_variables else ['V']

        region_labels = []
        if hasattr(experiment.network, 'region_labels') and experiment.network.region_labels is not None:
            region_labels = list(experiment.network.region_labels)
        else:
            n_reg = experiment.network.number_of_regions if hasattr(experiment.network, 'number_of_regions') else ts.data.shape[2]
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
