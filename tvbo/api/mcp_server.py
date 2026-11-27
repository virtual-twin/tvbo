"""The curated database, the software network and the backend check, served to agents over the Model Context Protocol.

`tvbo mcp` runs the server on stdio, the transport through which a coding agent launches a local server; it needs the `mcp` package (`pip install tvbo[mcp]`). Every tool reads: none writes a file, runs a simulation or reaches the network. `explore` is the entry point and answers a query that matches nothing with an empty list, so an agent widens its query instead of handling an error.
"""

from __future__ import annotations

import functools
from urllib.parse import unquote

import yaml
from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from tvbo.data.registry import _IRI_SCOPES, _curie, database_dir, interface_iri, record_iri

INTERFACE = "SoftwareInterface"
"""The kind `explore` gives a framework name the generated code uses, recorded under a software entry's `interfaces:`."""

INSTRUCTIONS = """tvbo specifies brain network models declaratively and generates code for simulation backends (TVB, tvboptim, JAX, NumPy, Julia, NeuroML/LEMS, ...).

Start with `explore`. It searches every curated record (models, coupling functions, integrators, networks, atlases, observation models, experiments, studies, software tools, ...) and every framework name tvbo's generated code imports, calls or passes as a keyword, and returns each match's IRI. A query that matches nothing returns an empty list: widen it or drop the `kind` filter.

`describe` returns the full record behind an IRI. A software tool's record lists the framework names tvbo depends on, with the version each was checked against and whether that version deprecates it. `render` shows the code one backend format generates for a curated model, and `check_backend` reads that code back and says, per state variable, whether it states the model's equations."""

server = MCPServer(name="tvbo", instructions=INSTRUCTIONS)
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


@functools.cache
def _rows() -> tuple[dict, ...]:
    """One row per curated record and per recorded software interface: `iri`, `kind`, `name`, `description` and the `record` it was read from."""
    rows = []
    for folder, kind in _IRI_SCOPES.items():
        for path in sorted(database_dir(kind).rglob("*.yaml")):
            if path.stem.startswith("_"):
                continue
            record = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                continue
            iri = record_iri(folder, path)
            name = str(record.get("name") or path.stem)
            rows.append(
                {
                    "iri": iri,
                    "kind": kind,
                    "name": name,
                    "description": " ".join(str(record.get("description") or "").split()),
                    "record": record,
                }
            )
            for interface, recorded in (record.get("interfaces") or {}).items():
                versions = f", checked against {name} {recorded['checked_version']}" if recorded.get("checked_version") else ""
                deprecation = (
                    f"; deprecated, use {recorded.get('replaced_by') or 'its replacement'}"
                    if recorded.get("deprecated")
                    else ""
                )
                rows.append(
                    {
                        "iri": interface_iri(iri, interface),
                        "kind": INTERFACE,
                        "name": interface,
                        "description": f"{' and '.join(recorded.get('kind') or [])} provided by {name}{versions}{deprecation}",
                        "record": recorded,
                    }
                )
    return tuple(rows)


def _summary(row: dict) -> dict:
    return {key: row[key] for key in ("iri", "kind", "name", "description")}


@server.tool(annotations=READ_ONLY)
def explore(query: str, kind: str = "", limit: int = 20) -> list[dict]:
    """Search the curated records and the framework names tvbo's generated code uses.

    Args:
        query: Words that must all appear in a match's name, IRI or description, in any case and order, e.g. `jansen rit`, `delay coupling` or `incoming_states`. An empty query lists every row of *kind*.
        kind: Only rows of this kind: a database class such as `Dynamics`, `Coupling`, `Network`, `SimulationExperiment`, `SimulationStudy` or `SimulationTool`, or `SoftwareInterface` for framework names. Empty for every kind.
        limit: The most rows returned; matches by name come first.

    Returns:
        `{iri, kind, name, description}` per match, possibly none.
    """
    words = query.lower().split()
    matches = [
        row
        for row in _rows()
        if (not kind or row["kind"].lower() == kind.lower())
        and all(word in f"{row['name']} {row['iri']} {row['description']}".lower() for word in words)
    ]
    matches.sort(
        key=lambda row: (
            row["name"].lower() != query.lower(),
            not all(word in row["name"].lower() for word in words),
            row["name"],
        )
    )
    return [_summary(row) for row in matches[:limit]]


@server.tool(annotations=READ_ONLY)
def describe(iri: str) -> dict:
    """The full record behind an IRI `explore` returned.

    Args:
        iri: A `tvbo:<folder>/<name>` IRI, written out under `https://w3id.org/tvbo/` or not, or a software interface's `tvbo:software/<tool>#<name>`, percent-encoded or not.

    Returns:
        The row's `iri`, `kind`, `name` and `description`, and its `record` as YAML; `{found: false}` with a hint when no record has the IRI.
    """
    curie = unquote(_curie(iri))
    for row in _rows():
        if unquote(row["iri"]) == curie:
            return _summary(row) | {"record": yaml.safe_dump(row["record"], sort_keys=False, allow_unicode=True, width=10_000)}
    return {"found": False, "hint": f"No curated record has the IRI {iri!r}; `explore` returns the IRIs there are."}


@server.tool(annotations=READ_ONLY)
def render(model: str, format: str) -> str:
    """The code one backend format generates for a curated model.

    Args:
        model: A curated `Dynamics` by name, e.g. `JansenRit`.
        format: The code format: `jax`, `numpy`, `tvb`, `tvboptim`, `julia`, `neuroml`, or another format the model supports.
    """
    from tvbo.classes.dynamics import Dynamics

    return Dynamics.from_db(model).render_code(format)


@server.tool(annotations=READ_ONLY)
def check_backend(model: str, format: str) -> list[dict]:
    """Whether the code *format* generates for *model* states the model's equations, per state variable.

    The generated code is read back into SymPy and each state variable's derivative is compared with the specification, structurally and at random points. A verdict is `equal`; `differs`, with a point where the two disagree; or `unread`, naming a construct the reader does not interpret, which is a statement about the reader and not the backend.

    Args:
        model: A curated `Dynamics` by name, e.g. `JansenRit`.
        format: One of `jax`, `numpy`, `tvb`, `tvboptim`, `julia` or `neuroml`.
    """
    from tvbo.classes.dynamics import Dynamics

    return [
        {"state_variable": verdict.name, "status": verdict.status, "detail": verdict.detail}
        for verdict in Dynamics.from_db(model).check_backend(format)
    ]


def serve() -> None:
    """Run the server on stdio until the client closes it."""
    server.run("stdio")
