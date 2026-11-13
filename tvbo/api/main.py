"""FastAPI application exposing the TVBO ontology and simulation experiment endpoints.

Defines the top-level `app`, wires in the network, dynamics, and experiment sub-routers, and provides ontology search/query endpoints plus routes to configure and run [`SimulationExperiment`](/api/classes/experiment.qmd) instances.
"""

from fastapi import Body, FastAPI, Path, Query
from pydantic import BaseModel, Field

from tvbo.api.dynamics_api import router as dynamics_router
from tvbo.api.experiment_api import router as experiment_router
from tvbo.api.network_api import router as network_router
from tvbo.api.ontology_api import OntologyAPI

app = FastAPI(title="TVBO API", description="The Virtual Brain Ontology API for simulation experiments", version="0.1.0")
api = OntologyAPI()

# Register sub-routers
app.include_router(network_router)
app.include_router(dynamics_router)
app.include_router(experiment_router)


# Request/Response Models (API-specific only)


class RunExperimentRequest(BaseModel):
    """Request payload for running a simulation."""

    experiment: dict = Field(..., description="Experiment configuration dictionary")
    duration: float | None = Field(None, description="Simulation duration in ms; the experiment's own when omitted")
    step_size: float | None = Field(None, description="Integration step size in ms; the experiment's own when omitted")
    backend: str | None = Field("jax", description="Simulation backend: 'jax' or 'tvb'")


class RunExperimentResponse(BaseModel):
    """Response payload with simulation results."""

    success: bool
    data: list | None = None  # [time, state_vars, regions, modes]
    time: list[float] | None = None
    state_variables: list[str] | None = None
    region_labels: list[str] | None = None
    sample_period: float | None = None
    message: str | None = None
    error: str | None = None


# Legacy model for backwards compatibility
class SimulationMetadata(BaseModel):
    """Legacy request payload describing a simulation configuration.

    Retained for backwards compatibility with clients that submit the model, connectivity, coupling, and integration blocks separately rather than as a single experiment dictionary.

    Args:
        model: Dynamics/model configuration block.
        connectivity: Optional network connectivity configuration.
        coupling: Optional coupling configuration.
        integration: Optional integration scheme configuration.
    """

    model: dict
    connectivity: dict | None = None
    coupling: dict | None = None
    integration: dict | None = None


# API Endpoints


@app.get("/")
def root():
    """Return the API name and version at the root endpoint.

    Returns:
        A mapping with the API `message` label and `version` string.
    """
    return {"message": "TVBO API", "version": "0.1.0"}


@app.get("/health")
def health():
    """Report service liveness for health checks.

    Returns:
        A mapping with a `status` of `"healthy"`.
    """
    return {"status": "healthy"}


@app.get("/search")
def search(term: str = Query(..., description="Ontology term to search for")):
    """Search the ontology for entries matching a term.

    Args:
        term: Ontology term to search for.

    Returns:
        The ontology search results for the given term.
    """
    return api.search_by_term(term)


@app.get("/query")
def query_nodes(query_str: str = Query(..., description="Term to query in ontology")):
    """Query the ontology for nodes matching a term.

    Args:
        query_str: Term to query in the ontology.

    Returns:
        The ontology nodes matching the query.
    """
    return api.query_nodes(query_str)


@app.get("/children/{node_id}")
def get_child_connections(node_id: int = Path(..., description="Node ID")):
    """Return the child connections of an ontology node.

    Args:
        node_id: Identifier of the node whose children are requested.

    Returns:
        The connections from the node to its child nodes.
    """
    return api.get_child_connections(node_id)


@app.get("/parents/{node_id}")
def get_parent_connections(node_id: int = Path(..., description="Node ID")):
    """Return the parent connections of an ontology node.

    Args:
        node_id: Identifier of the node whose parents are requested.

    Returns:
        The connections from the node to its parent nodes.
    """
    return api.get_parent_connections(node_id)


@app.post("/experiment/configure")
def configure_experiment(metadata: SimulationMetadata = Body(...)):
    """Configure a simulation experiment from legacy metadata.

    Passes the submitted metadata to the ontology API to set up a simulation experiment.

    Args:
        metadata: Legacy simulation configuration payload.

    Returns:
        A confirmation message that the experiment was configured.
    """
    api.configure_simulation_experiment(metadata.dict())
    return {"message": "Experiment configured successfully"}


@app.post("/experiment/run", response_model=RunExperimentResponse)
def run_experiment(request: RunExperimentRequest = Body(...)):
    """Run a simulation experiment and return its recorded state.

    The experiment dict follows the YAML schema; a duration or step size given in the request overrides the experiment's own. The state comes back as nested lists indexed time, state variable, node, mode, with the names of the variables and nodes and the sample period beside it.
    """
    import logging

    from tvbo.classes.experiment import SimulationExperiment

    logger = logging.getLogger(__name__)
    try:
        experiment = SimulationExperiment(**request.experiment)
        if request.step_size is not None:
            experiment.integration.step_size = request.step_size
        run_options = {} if request.duration is None else {"duration": request.duration}
        ts = experiment.run(request.backend or "jax", **run_options).integration.to_timeseries()
        return RunExperimentResponse(
            success=True,
            data=ts.data.tolist(),
            time=ts.time.tolist(),
            state_variables=[str(name) for name in ts.variables_labels],
            region_labels=[str(label) for label in ts.space_labels] or [f"Region_{i}" for i in range(ts.data.shape[2])],
            sample_period=float(ts.sample_period),
            message="Simulation completed successfully",
        )
    except Exception as e:
        logger.exception("experiment run failed")
        return RunExperimentResponse(success=False, error=str(e))
