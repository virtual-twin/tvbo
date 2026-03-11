from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from tvbo.api.ontology_api import OntologyAPI
from tvbo.api.network_api import router as network_router
from tvbo.api.dynamics_api import router as dynamics_router
from tvbo.api.experiment_api import router as experiment_router

app = FastAPI(
    title="TVBO API",
    description="The Virtual Brain Ontology API for simulation experiments",
    version="0.1.0"
)
api = OntologyAPI()

# Register sub-routers
app.include_router(network_router)
app.include_router(dynamics_router)
app.include_router(experiment_router)


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
    import logging
    logger = logging.getLogger(__name__)

    try:
        from tvbo.export.experiment import SimulationExperiment

        exp_data = request.experiment
        duration = request.duration or 1000.0
        step_size = request.step_size or 0.1
        backend = request.backend or "jax"

        logger.info("=" * 60)
        logger.info("EXPERIMENT RUN REQUEST")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration}, Step size: {step_size}, Backend: {backend}")
        logger.info(f"Input experiment data:\n{exp_data}")

        # Pass schema dict directly to SimulationExperiment
        experiment = SimulationExperiment(**exp_data)

        # Debug: Check parsed experiment via to_yaml()
        try:
            yaml_output = experiment.to_yaml()
            logger.info("=" * 60)
            logger.info("PARSED EXPERIMENT (to_yaml()):")
            logger.info("=" * 60)
            logger.info(f"\n{yaml_output}")
        except Exception as yaml_err:
            logger.warning(f"Could not generate to_yaml(): {yaml_err}")

        # Debug: Log key components
        logger.info(f"dynamics type: {type(experiment.dynamics)}")
        logger.info(f"dynamics: {experiment.dynamics}")
        if experiment.dynamics:
            logger.info(f"  - name: {experiment.dynamics.name}")
            logger.info(f"  - state_variables: {experiment.dynamics.state_variables}")
            logger.info(f"  - parameters: {experiment.dynamics.parameters}")
        logger.info(f"network type: {type(experiment.network)}")
        logger.info(f"network: {experiment.network}")
        if experiment.network:
            logger.info(f"  - number_of_nodes: {getattr(experiment.network, 'number_of_nodes', 'N/A')}")

        # Run simulation
        logger.info("Starting simulation run...")
        ts = experiment.run(format=backend, duration=duration)
        logger.info(f"Simulation complete. TimeSeries data shape: {ts.data.shape}")
        logger.info(f"TimeSeries time shape: {ts.time.shape}")

        # Extract results
        time_arr = ts.time.tolist() if hasattr(ts.time, 'tolist') else list(ts.time)
        data_arr = ts.data.tolist() if hasattr(ts.data, 'tolist') else ts.data

        logger.info(f"time_arr length: {len(time_arr)}")
        logger.info(f"data_arr type: {type(data_arr)}, length: {len(data_arr) if isinstance(data_arr, list) else 'N/A'}")

        state_vars = list(experiment.dynamics.state_variables.keys()) if experiment.dynamics and experiment.dynamics.state_variables else ['V']
        logger.info(f"state_variables: {state_vars}")

        region_labels = []
        if hasattr(experiment.network, 'region_labels') and experiment.network.region_labels is not None:
            region_labels = list(experiment.network.region_labels)
        else:
            n_reg = getattr(experiment.network, 'number_of_nodes', None) or ts.data.shape[2]
            region_labels = [f'Region_{i}' for i in range(n_reg)]
        logger.info(f"region_labels: {region_labels}")

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
        logger.error(f"EXPERIMENT RUN ERROR: {error_msg}")
        return RunExperimentResponse(
            success=False,
            error=error_msg,
        )
