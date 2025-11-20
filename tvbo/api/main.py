from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Path, Query
from pydantic import BaseModel

from tvbo.api.ontology_api import OntologyAPI

app = FastAPI()
api = OntologyAPI()

class SimulationMetadata(BaseModel):
    model: dict
    connectivity: Optional[dict] = None
    coupling: Optional[dict] = None
    integration: Optional[dict] = None


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
