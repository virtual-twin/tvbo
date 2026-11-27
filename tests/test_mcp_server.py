"""The MCP server `tvbo mcp` runs: what its tools answer, and that a query matching nothing is an empty answer rather than an error."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp.server.mcpserver", reason="the MCP server needs tvbo[mcp]")

from tvbo.api import mcp_server  # noqa: E402


@pytest.mark.backend_core
def test_explore_matches_every_word_in_any_case_and_ranks_the_name_first():
    found = mcp_server.explore("jansen RIT")

    assert found[0] == {**found[0], "iri": "tvbo:models/JansenRit", "kind": "Dynamics"}
    assert {"iri", "kind", "name", "description"} == set(found[0])
    assert {row["kind"] for row in mcp_server.explore("jansen rit", kind="coupling")} == {"Coupling"}
    assert mcp_server.explore("no record says this zzq") == []


@pytest.mark.backend_core
def test_explore_reaches_a_framework_name_and_its_deprecation():
    (interface,) = mcp_server.explore("InstantaneousCoupling incoming_states", kind="SoftwareInterface")

    assert (
        interface["iri"]
        == "tvbo:software/TVB-Optim#tvboptim.experimental.network_dynamics.coupling.base.InstantaneousCoupling(incoming_states=)"
    )
    assert "deprecated, use source" in interface["description"]
    assert "deprecated: true" in mcp_server.describe(interface["iri"])["record"]
    encoded = interface["iri"].replace("(", "%28").replace("=", "%3D").replace(")", "%29")
    assert mcp_server.describe(encoded) == mcp_server.describe(interface["iri"])


@pytest.mark.backend_core
def test_describe_reads_either_spelling_of_an_iri_and_answers_an_unknown_one():
    curie = mcp_server.describe("tvbo:software/TVB-Optim")

    assert curie == mcp_server.describe("https://w3id.org/tvbo/software/TVB-Optim")
    assert curie["kind"] == "SimulationTool" and "interfaces:" in curie["record"]
    assert mcp_server.describe("tvbo:models/NoSuchModel")["found"] is False


@pytest.mark.backend_core
def test_check_backend_reports_a_verdict_per_state_variable():
    verdicts = mcp_server.check_backend("Kuramoto", "numpy")

    assert [verdict["status"] for verdict in verdicts] == ["equal"]
    assert mcp_server.render("Kuramoto", "numpy").count("def ") >= 1


@pytest.mark.backend_core
def test_every_tool_is_read_only_and_the_instructions_start_at_explore():
    tools = asyncio.run(mcp_server.server.list_tools())

    assert [tool.name for tool in tools] == ["explore", "describe", "render", "check_backend"]
    assert all(tool.annotations.read_only_hint for tool in tools)
    assert "Start with `explore`" in mcp_server.server.instructions
